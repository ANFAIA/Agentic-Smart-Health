"""El `Segmenter` que junta las dos medidas, probado por lo que cada una NO puede sola.

El modelo del CBCT sabe *qué* es diente y no sabe *cuál*; el escáner intraoral sabe cuál y
solo ve corona. Estos tests comprueban que ninguna de las dos se usa para lo que no sirve.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis_agents import (
    DEFAULT_CODES,
    GUM_CLASS,
    SegmentadorDental,
    absorbe_islas,
    afina_fronteras,
    quita_motas,
    rellena_etiquetas,
    rellena_huecos_interiores,
)
from analysis_agents.segmentation import SegmentationAgent


def _coronas() -> tuple[np.ndarray, np.ndarray]:
    """Dos coronas separadas 10 mm, con códigos FDI distintos."""
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 1.0, (200, 3))
    b = rng.normal([10, 0, 0], 1.0, (200, 3))
    return np.vstack([a, b]), np.array([36] * 200 + [37] * 200)


def test_devuelve_log_probabilidades_de_verdad():
    """El agente lo comprueba, y con razón: leer logits como log-probabilidades daría
    confianzas plausibles y falsas — el modo de fallo caro."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.9), coronas, etq)
    logp = seg(coronas[:50])

    assert np.allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-3)
    assert (logp <= 0).all()


def test_el_fdi_lo_pone_el_escaner_no_el_modelo():
    """La separación entre dientes viene del IOS. El modelo del CBCT es binario y no
    puede darla: los dientes se tocan en el punto de contacto interproximal."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.95), coronas, etq)
    logp = seg(np.array([[0.0, 0, 0], [10.0, 0, 0]]))

    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    assert int(np.argmax(logp[0])) == col[36]
    assert int(np.argmax(logp[1])) == col[37]


def test_un_punto_de_diente_sin_nombre_cuenta_como_encia():
    """Lo importante, y es una decisión, no un descuido.

    Si el modelo dice «diente» pero ninguna corona lo reclama, no se puede declarar
    diente: no habría a qué colgarle un hallazgo clínico. Declarar «diente sin saber cuál»
    sería inventar la mitad que falta — y el sitio donde pasa es el hueso alveolar que el
    modelo marca de más, justo lo que NO debe heredar el FDI del vecino.
    """
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.99), coronas, etq)
    lejos = np.array([[0.0, 0.0, 500.0]])  # muy fuera del radio de nombre

    logp = seg(lejos)
    assert int(np.argmax(logp[0])) == GUM_CLASS
    assert np.exp(logp[0, GUM_CLASS]) == pytest.approx(1.0)


def test_lejos_de_la_corona_decide_el_modelo():
    """El IOS dice cuál, pero a distancia de la corona no dice si hay diente: eso es del
    modelo del CBCT, que es lo único que ve por debajo del margen."""
    coronas, etq = _coronas()
    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    lejos = np.array([[0.0, 0.0, 6.0]])   # dentro del radio de nombre, fuera del de corona

    alto = SegmentadorDental(lambda p: np.full(len(p), 0.9), coronas, etq)(lejos)
    bajo = SegmentadorDental(lambda p: np.full(len(p), 0.1), coronas, etq)(lejos)

    assert int(np.argmax(alto[0])) == col[36]
    assert int(np.argmax(bajo[0])) == GUM_CLASS
    assert np.exp(alto[0, col[36]]) == pytest.approx(0.9, abs=1e-6)


def test_JUNTO_a_la_corona_manda_el_escaner():
    """El arreglo del hueco de la unión corona-raíz, y no es una excepción de comodidad.

    La corona y la raíz se veían como dos objetos separados. Medido, no era desajuste de
    registro —el 94 % de las coronas tiene material del CBCT a menos de 2 mm— sino que el
    modelo no llamaba diente a ese material: a 1 mm de una corona hay 5.215 gaussianas
    nombradas y 2.927 sin nombrar.

    Ahí la superficie la midió un escáner con exactitud de decenas de micras, y la opinión
    de un clasificador sobre un vóxel de 0,30 mm es evidencia más débil que esa medida.
    """
    coronas, etq = _coronas()
    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    pegado = np.array([[0.0, 0.0, 0.0]])   # en el centro de la nube de corona

    bajo = SegmentadorDental(lambda p: np.full(len(p), 0.05), coronas, etq)(pegado)
    assert int(np.argmax(bajo[0])) == col[36], "el escáner lo midió: es ese diente"


def test_el_escaner_solo_manda_donde_MIDIO():
    """No es una patente de corso: fuera del radio de corona el modelo recupera el mando,
    porque ahí el escáner no vio nada."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.05), coronas, etq,
                            radio_corona_mm=0.5)
    assert int(np.argmax(seg(np.array([[0.0, 0.0, 5.0]]))[0])) == GUM_CLASS


def test_sin_ningun_codigo_falla_en_vez_de_segmentar_a_ciegas():
    """Correr sin etiquetas produciría un `region_id` con un solo código para todo, que es
    exactamente lo que pasó al ejecutar la composición sin `--fdi`: 30.592 gaussianas bajo
    un único FDI y una «pieza» de 66 mm."""
    coronas, _ = _coronas()
    with pytest.raises(ValueError, match="código FDI"):
        SegmentadorDental(lambda p: np.ones(len(p)), coronas, np.zeros(len(coronas)))


def test_encaja_con_el_agente_que_lo_va_a_usar():
    """El contrato real: `SegmentationAgent` lo acepta como `Segmenter` y saca `detected`."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.9), coronas, etq)
    agente = SegmentationAgent(_AlmacenFalso({"centers": coronas}), segmenter=seg)

    assert isinstance(agente.segmenter(coronas[:10]), np.ndarray)


class _AlmacenFalso:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    def load(self, ref: str) -> dict[str, np.ndarray]:
        return self._arrays

    def put(self, **arrays: np.ndarray) -> str:
        return "sha256:" + "0" * 64


def test_probabilidad_1_no_produce_menos_infinito():
    """`SegmentationAgent` rechaza no finitos, y `log(1 - 1.0)` es `-inf`.

    No es teórico: el modelo devuelve 1,0 exacto para puntos de esmalte, y sin el recorte
    la etapa entera salía `FAILED` con «el segmentador devolvió valores no finitos».
    """
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.ones(len(p)), coronas, etq)
    logp = seg(coronas[:20])
    assert np.isfinite(logp).all()
    assert np.allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-3)


# --- la direccion de la raiz ------------------------------------------------- #
def _en_oclusion() -> tuple[np.ndarray, np.ndarray]:
    """Una corona superior y, a 1 mm por debajo, la inferior que la ocluye.

    Es la geometría real: en un CBCT dental el paciente muerde, así que las dos coronas se
    tocan. Solo la de arriba está etiquetada — el escaneo intraoral es de una arcada.
    """
    rng = np.random.default_rng(0)
    return rng.normal([0, 0, 0], [1.0, 1.0, 0.5], (200, 3)), np.array([36] * 200)


def test_sin_direccion_el_diente_de_enfrente_se_cuela(tmp_path):
    """El fallo que esto arregla, y se veía en el visor: los dientes cruzaban la encía por
    arriba y por abajo, con piezas de 44 y 47 mm donde una mide 20-25."""
    coronas, etq = _en_oclusion()
    seg = SegmentadorDental(lambda p: np.ones(len(p)), coronas, etq)
    # Un punto 12 mm por DEBAJO de la corona: es la raíz del diente que ocluye.
    logp = seg(np.array([[0.0, 0.0, -12.0]]))

    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    assert int(np.argmax(logp[0])) == col[36], "sin dirección, lo nombra igual"


def test_con_la_direccion_medida_ya_no(tmp_path):
    """Un diente se extiende desde su corona HACIA el hueso. Lo que queda por delante de
    la corona, hacia la boca, es el diente que la muerde."""
    coronas, etq = _en_oclusion()
    seg = SegmentadorDental(
        lambda p: np.ones(len(p)), coronas, etq, direccion_raiz=np.array([0.0, 0.0, 1.0])
    )
    abajo = seg(np.array([[0.0, 0.0, -12.0]]))   # el que ocluye
    arriba = seg(np.array([[0.0, 0.0, 12.0]]))   # su propia raíz

    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    assert int(np.argmax(abajo[0])) == GUM_CLASS, "el de enfrente ya no se cuela"
    assert int(np.argmax(arriba[0])) == col[36], "y la raíz propia sigue entrando"


def test_la_tolerancia_deja_pasar_el_espesor_de_la_corona(tmp_path):
    """No es un corte a cero: el escáner ve la superficie externa, no el esmalte entero, y
    el registro tiene su error (medido: p50 0,8-4 mm). Por eso hay 3 mm de holgura."""
    coronas, etq = _en_oclusion()
    seg = SegmentadorDental(
        lambda p: np.ones(len(p)), coronas, etq, direccion_raiz=np.array([0.0, 0.0, 1.0])
    )
    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    assert int(np.argmax(seg(np.array([[0.0, 0.0, -2.0]]))[0])) == col[36]


def test_la_direccion_no_hace_falta_normalizarla(tmp_path):
    """Quien la mide la pasa tal cual —`media(encía) - media(coronas)`— y son milímetros,
    no un unitario. Normalizarla aquí evita que el llamante tenga que acordarse."""
    coronas, etq = _en_oclusion()
    seg = SegmentadorDental(
        lambda p: np.ones(len(p)), coronas, etq, direccion_raiz=np.array([0.0, 0.0, 6.9])
    )
    assert int(np.argmax(seg(np.array([[0.0, 0.0, -12.0]]))[0])) == GUM_CLASS


# --- huecos del etiquetado del escaner --------------------------------------- #
def test_un_vertice_RODEADO_de_una_pieza_es_esa_pieza():
    """El segmentador del escáner deja vértices sueltos sin etiquetar dentro de una
    corona, y salen pintados como encía."""
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1.0, (300, 3))
    e = np.full(300, 36)
    e[[10, 50, 120]] = 0          # tres agujeros en medio de la pieza

    lleno = rellena_etiquetas(v, e)
    assert (lleno == 36).all(), "un agujero rodeado de 36 es 36"


def test_el_MARGEN_gingival_no_se_come():
    """⚠️ Lo que distingue este criterio de un relleno por radio.

    En el margen la encía toca la corona de verdad. Medido sobre un caso real: por radio
    de 0,5 mm entrarían 10.128 vértices; con el criterio de vecindario quedan 1.851. Esos
    8.000 de diferencia son margen, que es justo la frontera clínica que interesa.
    """
    rng = np.random.default_rng(1)
    corona = rng.normal([0, 0, 2.0], [1.0, 1.0, 0.4], (250, 3))
    encia = rng.normal([0, 0, -0.6], [1.2, 1.2, 0.4], (250, 3))
    v = np.vstack([corona, encia])
    e = np.concatenate([np.full(250, 36), np.zeros(250, dtype=int)])

    lleno = rellena_etiquetas(v, e)
    invadidos = int((lleno[250:] > 0).sum())
    assert invadidos < 25, f"se comió {invadidos} vértices de encía en el margen"


def test_sin_agujeros_no_toca_nada():
    """No es un filtro que reescriba por si acaso."""
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1.0, (200, 3))
    e = np.full(200, 36)
    assert (rellena_etiquetas(v, e) == e).all()


# --- el recorte apical ------------------------------------------------------- #
def test_sin_recorte_la_raiz_sigue_hasta_donde_el_modelo_diga():
    """El comportamiento por defecto no cambia: el recorte es opt-in porque convierte el
    ápice en supuesto, y esa es una decisión de quien monta el caso."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.ones(len(p)), coronas, etq,
                            direccion_raiz=np.array([0.0, 0.0, 1.0]))
    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    lejisimos = np.array([[0.0, 0.0, 14.0]])   # 14 mm hacia la raíz
    assert int(np.argmax(seg(lejisimos)[0])) == col[36]


def test_con_recorte_la_raiz_para_en_la_longitud_de_SU_TIPO():
    """Medido: el 16 salía con 34,3 mm cuando un molar superior entero mide ~20. Los 14 de
    más son seno maxilar y hueso alveolar, y el modelo no puede pararse ahí porque bajo la
    cresta ósea no hay frontera que segmentar."""
    coronas, etq = _coronas()          # las dos piezas son molares (36 y 37)
    seg = SegmentadorDental(lambda p: np.ones(len(p)), coronas, etq,
                            direccion_raiz=np.array([0.0, 0.0, 1.0]),
                            recorta_por_longitud=True)
    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}

    dentro = seg(np.array([[0.0, 0.0, 14.0]]))   # dentro de los 22 mm de un molar
    fuera = seg(np.array([[0.0, 0.0, 30.0]]))    # más allá

    assert int(np.argmax(dentro[0])) == col[36]
    assert int(np.argmax(fuera[0])) == GUM_CLASS


def test_el_canino_no_se_recorta_con_la_cota_del_incisivo():
    """Es el diente más largo de la boca. Una cota única le quitaría ápice de verdad."""
    from analysis_agents.dental import LONGITUD_MM

    assert LONGITUD_MM["canino"] > LONGITUD_MM["incisivo"]
    assert LONGITUD_MM["canino"] > LONGITUD_MM["molar"]


def test_el_eje_de_la_cota_es_el_GLOBAL_y_esta_medido_que_tiene_que_serlo():
    """Registro de un negativo, para que no se reintente sin datos nuevos.

    Un eje por pieza deberia cortar mejor. No se puede sacar de la corona: el escaner ve un
    casquete y sus direcciones principales las manda el contorno, no el eje del diente.
    Medido, tres estimadores distintos dan angulos de 17 a 90 grados contra el eje de la
    arcada y alturas de corona de 5 a 23 mm donde lo anatomico son 7-9. Puesto en
    produccion empeoro: mediana 26,7 mm contra 25,8, ocho piezas desbordadas contra seis.
    """
    coronas, etq = _coronas()
    global_ = np.array([0.0, 0.0, 1.0])
    seg = SegmentadorDental(lambda p: np.ones(len(p)), coronas, etq,
                            direccion_raiz=global_, recorta_por_longitud=True)

    for eje, _, _ in seg._cota_apical.values():
        assert np.allclose(eje, global_)


def test_la_cota_arranca_en_el_extremo_oclusal_de_CADA_pieza():
    """Dos piezas del mismo tipo a distinta altura tienen que recibir cotas distintas: la
    longitud se cuenta desde donde empieza el diente, no desde el origen del marco."""
    rng = np.random.default_rng(4)
    a = rng.normal([0, 0, 0], 1.0, (200, 3))
    b = rng.normal([10, 0, 6.0], 1.0, (200, 3))     # el 37, seis mm mas adentro
    seg = SegmentadorDental(
        lambda p: np.ones(len(p)), np.vstack([a, b]),
        np.array([36] * 200 + [37] * 200),
        direccion_raiz=np.array([0.0, 0.0, 1.0]), recorta_por_longitud=True,
    )

    (_, o36, c36), (_, o37, c37) = seg._cota_apical[36], seg._cota_apical[37]
    # Las cotas son relativas a su propio origen; en absoluto el 37 corta seis mm mas alla.
    assert (o37[2] + c37) - (o36[2] + c36) == pytest.approx(6.0, abs=1.0)


# --- islas de etiquetado ---------------------------------------------------- #
def _dos_dientes_y_un_parche() -> tuple[np.ndarray, np.ndarray]:
    """Dos piezas separadas y un parche pegado a la cara de una de ellas.

    Reproduce la forma del fallo real: 275 vértices a menos de un milímetro del vecino,
    contra piezas de miles. Aquí a escala, pero con las mismas proporciones.
    """
    rng = np.random.default_rng(0)
    a = rng.uniform(-2, 2, (900, 3))
    b = rng.uniform(-2, 2, (900, 3)) + np.array([10.0, 0.0, 0.0])
    # El parche: pegado a la cara +x de `b`, con su propio código.
    parche = rng.uniform(-0.4, 0.4, (55, 3)) + np.array([12.3, 0.0, 0.0])
    pos = np.concatenate([a, b, parche])
    etq = np.concatenate([
        np.full(900, 26, dtype=np.int64),
        np.full(900, 27, dtype=np.int64),
        np.full(55, 28, dtype=np.int64),
    ])
    return pos, etq


def test_una_isla_pegada_a_su_vecino_se_absorbe():
    """El fallo exacto: un parche con código propio convertía un caso de 14 dientes en
    uno de 15, y ese código era la SEMILLA con la que se nombraba el CBCT."""
    pos, etq = _dos_dientes_y_un_parche()
    nuevo, actas = absorbe_islas(pos, etq)

    assert actas == [(28, 27, 55)]
    assert sorted({int(x) for x in nuevo}) == [26, 27]


def test_la_absorcion_deja_ACTA_y_no_se_hace_en_silencio():
    """Un tercer molar parcialmente erupcionado se parece mucho a una isla: pequeño y
    con un solo vecino. Que un diente deje de existir es de las decisiones que un
    clínico tiene que poder mirar."""
    pos, etq = _dos_dientes_y_un_parche()
    _, actas = absorbe_islas(pos, etq)

    origen, destino, n = actas[0]
    assert (origen, destino) == (28, 27)
    assert n == int((etq == 28).sum()), "el acta tiene que decir cuántos vértices se movieron"


def test_un_diente_pequeno_de_verdad_NO_se_absorbe():
    """El 17 del caso real tiene el 100 % de sus vecinos ajenos en el 16 —está al final
    de la arcada— y es un diente. Lo que separa es el TAMAÑO, no el acuerdo."""
    pos, etq = _dos_dientes_y_un_parche()
    # El parche pasa a tener tamaño de pieza: mismo vecindario, otro tamaño.
    grande = np.random.default_rng(1).uniform(-2, 2, (700, 3)) + np.array([14.0, 0.0, 0.0])
    pos = np.concatenate([pos[:1800], grande])
    etq = np.concatenate([etq[:1800], np.full(700, 28, dtype=np.int64)])

    _, actas = absorbe_islas(pos, etq)
    assert actas == []


def test_una_isla_rodeada_de_ENCIA_no_se_absorbe():
    """Una pieza pequeña y suelta en la encía es un diente pequeño y suelto, no un trozo
    de su vecino: no hay a quién absorberla, y adivinar sería inventarse anatomía."""
    rng = np.random.default_rng(2)
    pos = np.concatenate([
        rng.uniform(-2, 2, (900, 3)),
        rng.uniform(-2, 2, (900, 3)) + np.array([10.0, 0.0, 0.0]),
        rng.uniform(-0.4, 0.4, (55, 3)) + np.array([30.0, 0.0, 0.0]),
    ])
    etq = np.concatenate([
        np.full(900, 26, dtype=np.int64), np.full(900, 27, dtype=np.int64),
        np.full(55, 28, dtype=np.int64),
    ])
    nuevo, actas = absorbe_islas(pos, etq)
    assert actas == []
    assert int((nuevo == 28).sum()) == 55, "la pieza suelta ha perdido su código"


# --- frontera entre dientes contiguos --------------------------------------- #
def _dos_contiguos_con_frontera_sucia() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dos piezas que se tocan, con la etiqueta salpicada a los dos lados del contacto.

    Es la forma del fallo real: el contacto interproximal es superficie continua en la
    malla, así que el modelo no tiene ahí ningún borde geométrico y la etiqueta sale
    difuminada. Medido sobre un caso clínico, el 7,0 % del área del escaneo estaba en
    caras con dos dientes distintos.
    """
    rng = np.random.default_rng(0)
    a = rng.uniform(-3, 0, (600, 3))
    b = rng.uniform(0, 3, (600, 3))
    pos = np.concatenate([a, b])
    limpio = np.concatenate([np.full(600, 26), np.full(600, 27)]).astype(np.int64)
    sucio = limpio.copy()
    # Salpica: cerca del contacto, uno de cada tres vértices lleva el código del vecino.
    cerca = np.flatnonzero(np.abs(pos[:, 0]) < 0.8)
    sucio[cerca[::3]] = np.where(limpio[cerca[::3]] == 26, 27, 26)
    return pos, sucio, limpio


def test_la_frontera_entre_vecinos_se_afila():
    """Sin esto, encender una pieza en el visor enciende también un trozo de la de al
    lado — que es exactamente como se descubrió."""
    pos, sucio, limpio = _dos_contiguos_con_frontera_sucia()
    antes = int((sucio != limpio).sum())

    afinado, reasignados = afina_fronteras(pos, sucio)

    despues = int((afinado != limpio).sum())
    assert despues < antes / 3, f"apenas ha afilado: {antes} → {despues}"
    assert reasignados > 0


def test_afinar_no_puede_crear_ni_borrar_una_pieza():
    """Vota entre los códigos que ya hay: es un filtro de frontera, no un clasificador.
    Si pudiera hacer desaparecer una pieza sería otra cosa y necesitaría otro gate."""
    pos, sucio, _ = _dos_contiguos_con_frontera_sucia()
    afinado, _ = afina_fronteras(pos, sucio)

    assert {int(x) for x in afinado} == {int(x) for x in sucio}


def test_el_margen_gingival_NO_se_toca():
    """La banda corona/encía es una frontera clínica de verdad —12,8 % del área en el
    caso medido— y es lo que un periodoncista viene a mirar. Difuminarla o moverla sería
    borrar el dato."""
    rng = np.random.default_rng(1)
    diente = rng.uniform(0, 3, (600, 3))
    encia = rng.uniform(-3, 0, (600, 3))
    pos = np.concatenate([diente, encia])
    etq = np.concatenate([np.full(600, 26), np.zeros(600)]).astype(np.int64)

    afinado, _ = afina_fronteras(pos, etq)
    assert int((afinado == 0).sum()) == 600, "la encía ha perdido vértices"
    assert int((afinado == 26).sum()) == 600, "el diente ha ganado vértices de encía"


def test_una_sola_pieza_no_se_toca():
    """Con un solo código no hay frontera entre vecinos que afilar."""
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 3, (300, 3))
    etq = np.full(300, 26, dtype=np.int64)

    afinado, n = afina_fronteras(pos, etq)
    assert n == 0
    assert np.array_equal(afinado, etq)


# --- motas de diente en la encía -------------------------------------------- #
def test_una_mota_de_diente_en_mitad_de_la_encia_se_quita():
    """El simétrico de `rellena_etiquetas`, que sólo iba en un sentido. Medido tras el
    pipeline entero quedaban 692 vértices de diente flotando en la encía que nadie
    tocaba, y en el visor eso son motas color hueso salpicadas sobre el rosa."""
    rng = np.random.default_rng(0)
    encia = rng.uniform(-4, 4, (900, 3))
    etq = np.zeros(900, dtype=np.int64)
    # Media docena de motas sueltas, cada una lejos de las demás.
    motas = np.array([0, 150, 300, 450, 600, 750])
    etq[motas] = 26

    limpio, quitados = quita_motas(encia, etq)
    assert quitados == len(motas)
    assert not (limpio > 0).any()


def test_el_margen_gingival_sobrevive_a_quitar_motas():
    """El umbral es el mismo que usa `rellena_etiquetas` justamente para esto: un vértice
    del margen tiene vecinos de las dos clases, no llega al 0,85 y no se toca. Lo que se
    quita está en mitad de la encía, lejos de cualquier corona."""
    rng = np.random.default_rng(1)
    diente = rng.uniform(0.05, 4, (700, 3))
    encia = rng.uniform(-4, -0.05, (700, 3))
    pos = np.concatenate([diente, encia])
    etq = np.concatenate([np.full(700, 26), np.zeros(700)]).astype(np.int64)

    limpio, quitados = quita_motas(pos, etq)
    assert quitados == 0, "se ha comido el borde del diente"
    assert int((limpio == 26).sum()) == 700


def test_rellenar_CONVERGE_y_no_avanza_sobre_la_encia():
    """Rellenar un hueco deja al descubierto el siguiente, asi que la funcion itera.

    Lo que hay que atar de una iteracion no es que rellene mas —sobre un hueco esferico
    aislado una sola pasada ya lo cierra— sino que **pare**. Medido sobre un caso clinico
    los huecos van 3.299 → 1.519 → 1.198 → 1.096 → 1.051 y los incrementos caen en
    progresion; si en vez de cerrar agujeros estuviera avanzando sobre la encia, el numero
    no convergeria y cada pasada se comeria un poco mas de margen.
    """
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 4, (4000, 3))
    etq = np.full(4000, 26, dtype=np.int64)
    etq[rng.choice(4000, 400, replace=False)] = 0

    estable = rellena_etiquetas(pos, etq)
    otra_vez = rellena_etiquetas(pos, estable)

    assert np.array_equal(estable, otra_vez), "no es punto fijo: seguiria creciendo"
    assert int((estable > 0).sum()) < len(etq), "se ha comido la encia entera"


# --- huecos en mitad de una corona ------------------------------------------- #
def test_un_hueco_en_mitad_de_la_corona_se_rellena():
    """Lo que más se ve en el visor, porque el color va por vértice y se interpola: un
    solo vértice mal puesto mancha los seis triángulos que lo tocan. `rellena_etiquetas`
    no los alcanza porque son mayores que el vecindario."""
    rng = np.random.default_rng(0)
    diente = rng.uniform(0, 6, (2500, 3))
    encia = rng.uniform(-6, -2, (1200, 3))
    pos = np.concatenate([diente, encia])
    etq = np.concatenate([np.full(2500, 26), np.zeros(1200)]).astype(np.int64)
    # Un hueco macizo en el centro del diente, lejísimos de la encía.
    hueco = np.linalg.norm(diente - 3.0, axis=1) < 1.0
    etq[: 2500][hueco] = 0

    relleno, n = rellena_huecos_interiores(pos, etq)
    assert n > 0
    assert int((relleno[:2500] == 26).sum()) > int((etq[:2500] == 26).sum())
    assert int((relleno[2500:] == 0).sum()) == 1200, "se ha comido encía de verdad"


def test_el_margen_no_entra_porque_esta_PEGADO_a_la_encia():
    """El criterio es la distancia a encía real, y eso es lo que lo hace seguro: un
    vértice del margen gingival está pegado a la encía —lo es por definición— así que
    nunca supera el umbral."""
    rng = np.random.default_rng(1)
    diente = rng.uniform(0.05, 4, (900, 3))
    encia = rng.uniform(-4, -0.05, (900, 3))
    pos = np.concatenate([diente, encia])
    etq = np.concatenate([np.full(900, 26), np.zeros(900)]).astype(np.int64)

    relleno, n = rellena_huecos_interiores(pos, etq)
    assert n == 0, "ha movido el margen"
    assert np.array_equal(relleno, etq)


def test_el_hueco_toma_el_codigo_de_la_pieza_QUE_LO_RODEA():
    """Si el hueco está dentro de la corona del 26, es del 26. Ponerle cualquier otro
    código sería inventar una pieza donde no la hay."""
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 4, (1200, 3))
    b = rng.uniform(20, 24, (1200, 3))
    # La encía tiene que ser MUCHO mayor que el hueco: el criterio es la conectividad y
    # el tope está en el 5 % de la componente mayor, así que una encía pequeña convierte
    # al hueco en una componente respetable y deja de serlo.
    lejos = rng.uniform(-30, -24, (4000, 3))
    pos = np.concatenate([a, b, lejos])
    etq = np.concatenate([np.full(1200, 26), np.full(1200, 27), np.zeros(4000)]).astype(np.int64)
    dentro = np.linalg.norm(b - 22.0, axis=1) < 0.9
    etq[1200:2400][dentro] = 0

    relleno, n = rellena_huecos_interiores(pos, etq)
    assert n > 0
    assert int((relleno[1200:2400] == 27).sum()) == 1200
    assert int((relleno[1200:2400] == 26).sum()) == 0
