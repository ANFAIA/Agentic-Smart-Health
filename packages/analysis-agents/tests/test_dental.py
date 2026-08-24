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
    rellena_etiquetas,
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
