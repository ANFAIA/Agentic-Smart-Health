"""Color por pieza desde una foto, sin pose. Ver `gaussian_engine.tono_foto`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gaussian_engine.tono_foto import (
    MINIMO_CORONAS_SONDA,
    NO_ES_TIRA,
    Alineamiento,
    _lab,
    _lineal,
    _sin_mucosa,
    _srgb,
    ajuste_de_iluminacion,
    alinea_con_el_arco,
    anchos_aparentes,
    banda_superior,
    costura_oclusal,
    rgb_de_lab,
    sondas_de_mucosa,
    tono_por_tercios,
)


def _dos_arcadas(alto: int = 60, ancho: int = 120, fila: int = 30) -> tuple:
    """Dos bandas de diente pegadas por una linea oscura, como una lateral."""
    mascara = np.zeros((alto, ancho), bool)
    mascara[8:fila, :] = True
    mascara[fila + 1:alto - 8, :] = True
    mascara[fila, :] = True  # la costura tambien es diente: por eso hay que partir
    lum = np.full((alto, ancho), 200.0)
    lum[fila, :] = 20.0
    return mascara, lum


# --- la costura oclusal ------------------------------------------------------ #
def test_la_costura_encuentra_la_linea_de_mordida() -> None:
    """Las dos arcadas de una lateral llegan pegadas en UNA sola componente.

    Sin partirlas no se puede hablar de piezas del maxilar: el watershed daria blobs que
    cruzan la mordida.
    """
    mascara, lum = _dos_arcadas()
    assert (costura_oclusal(mascara, lum) == 30).all()


def test_la_costura_sigue_una_mordida_inclinada() -> None:
    """La linea de mordida no es horizontal: baja hacia posterior y da escalones."""
    alto, ancho = 80, 120
    mascara = np.ones((alto, ancho), bool)
    lum = np.full((alto, ancho), 200.0)
    fila = (20 + np.arange(ancho) * 0.3).astype(int)
    lum[fila, np.arange(ancho)] = 10.0
    assert np.abs(costura_oclusal(mascara, lum) - fila).max() <= 1


def test_la_costura_no_se_escapa_por_fuera_de_la_mascara() -> None:
    """Fuera de la mascara el coste es prohibitivo aunque haya pixeles mas oscuros.

    El fondo de una foto intraoral es mas oscuro que la mordida; sin esa barrera el camino
    de coste minimo se iria por ahi y no partiria nada.
    """
    mascara, lum = _dos_arcadas()
    lum[0, :] = 0.0  # el fondo, aun mas oscuro que la costura
    assert (costura_oclusal(mascara, lum) == 30).all()


def test_la_banda_superior_se_retira_de_la_costura() -> None:
    """El pixel pegado a la mordida lleva sombra; se erosiona antes de medir."""
    mascara, lum = _dos_arcadas()
    costura = costura_oclusal(mascara, lum)
    assert banda_superior(mascara, costura, margen=0)[29].all()
    assert not banda_superior(mascara, costura, margen=6)[29].any()
    assert banda_superior(mascara, costura, margen=6)[20].all()


# --- alinear la tira con el arco --------------------------------------------- #
def test_alinear_declara_AMBIGUO_cuando_el_arco_es_un_espejo() -> None:
    """⚠️ **La huella de anchuras no puede decir de que lado de la boca es la foto.**

    La tabla anatomica del maxilar leida al derecho y al reves es la MISMA lista, asi que
    la mejor hipotesis y su imagen especular empatan siempre. No es un fallo del metodo:
    ninguna medida de anchuras distingue el 16 del 26. Lo que no vale es tragarse el empate
    y devolver un lado como si se supiera.
    """
    arco = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    tabla = np.array([9.0, 10.0, 6.5, 7.0, 7.5, 6.5, 8.5,
                      8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0])
    assert np.array_equal(tabla, tabla[::-1]), "el arco tiene que ser un espejo"

    al = alinea_con_el_arco(tabla[6:], arco, tabla)
    assert isinstance(al, Alineamiento)
    assert al.ambiguo
    assert al.margen == pytest.approx(0.0, abs=1e-9)


def test_alinear_encuentra_el_tramo_cuando_la_referencia_NO_es_simetrica() -> None:
    """Con una referencia asimetrica el empate desaparece y el tramo sale identificado."""
    arco = [1, 2, 3, 4, 5, 6, 7]
    ref = np.array([3.0, 9.0, 4.0, 4.2, 7.0, 5.0, 6.0])
    al = alinea_con_el_arco(ref[2:6], arco, ref)
    assert al.fdis == [3, 4, 5, 6]
    assert not al.ambiguo


def test_alinear_absorbe_el_escorzo_de_una_lateral() -> None:
    """⚠️ En una lateral el ancho aparente DECRECE hacia los molares mientras el real crece.

    Cada corona esta un poco mas lejos, asi que el ancho se multiplica por un factor casi
    fijo a cada paso: en log-razon es un sumando constante. Centrando las secuencias
    desaparece, y sin centrar el alineamiento se va al tramo equivocado.
    """
    arco = [1, 2, 3, 4, 5, 6, 7]
    ref = np.array([3.0, 9.0, 4.0, 4.2, 7.0, 5.0, 6.0])
    escorzo = 0.82 ** np.arange(4)
    al = alinea_con_el_arco(ref[2:6] * escorzo, arco, ref)
    assert al.fdis == [3, 4, 5, 6]


def test_alinear_no_inventa_con_menos_de_tres_coronas() -> None:
    """Con dos coronas hay una sola razon: cualquier tramo la iguala."""
    assert alinea_con_el_arco(np.array([5.0, 6.0]), [1, 2, 3], np.array([1.0, 2.0, 3.0])) is None


# --- el color por tercios ---------------------------------------------------- #
def _corona(alto: int = 90, ancho: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Una corona sintetica: cuello rojizo arriba, borde neutro abajo."""
    img = np.zeros((alto, ancho, 3), np.uint8)
    for y in range(alto):
        t = y / (alto - 1)
        img[y, :] = (210, int(150 + 40 * t), int(120 + 60 * t))
    return img, np.ones((alto, ancho), bool)


def test_el_primer_tercio_es_el_CERVICAL() -> None:
    """⚠️ El orden de los tercios no es un detalle: un color por tercio sin saber que tercio
    es no vale para nada.

    `hacia_cervical` apunta al cuello, asi que la franja de proyeccion ALTA es la cervical.
    Devolverlas en el orden natural de los cortes da la lista al reves — y entonces se lee
    que el borde incisal es el mas saturado, que contradice la anatomia.
    """
    img, m = _corona()
    tercios = tono_por_tercios(img, m, (0, -1))  # el cuello queda ARRIBA
    assert tercios is not None
    assert tercios.shape == (3, 3)
    # El cuello es mas rojo que el borde: `a*` tiene que DECRECER de cervical a incisal.
    assert tercios[0][1] > tercios[1][1] > tercios[2][1]


def test_los_tercios_se_invierten_si_se_invierte_la_direccion() -> None:
    """La direccion se pasa MEDIDA, no supuesta; el resultado la obedece."""
    img, m = _corona()
    arriba = tono_por_tercios(img, m, (0, -1))
    abajo = tono_por_tercios(img, m, (0, 1))
    assert np.allclose(arriba, abajo[::-1], atol=1e-6)


def test_el_brillo_especular_no_arrastra_el_color() -> None:
    """El esmalte mojado devuelve el flash directo: eso es la camara, no el diente."""
    img, m = _corona()
    limpio = tono_por_tercios(img, m, (0, -1))
    img[40:50, 10:20] = 255  # un brillo en el tercio medio
    con_brillo = tono_por_tercios(img, m, (0, -1))
    assert abs(con_brillo[1][0] - limpio[1][0]) < 2.0


def test_una_pieza_con_pocos_pixeles_no_declara_color() -> None:
    """Por debajo del minimo la mediana la decide el ruido: mejor no afirmar nada."""
    img, _ = _corona()
    m = np.zeros(img.shape[:2], bool)
    m[:5, :5] = True
    assert tono_por_tercios(img, m, (0, -1)) is None


def test_lab_y_rgb_van_y_vuelven() -> None:
    from gaussian_engine.tono_foto import _lab

    for color in ((210, 190, 160), (120, 60, 55), (255, 255, 255), (20, 20, 20)):
        ida = _lab(np.array([[color]], dtype=np.uint8))[0, 0]
        assert np.abs(rgb_de_lab(ida).astype(int) - np.array(color)).max() <= 1


# --- los anchos aparentes ----------------------------------------------------- #
def test_los_anchos_aparentes_siguen_el_orden_de_la_tira() -> None:
    etiquetas = np.zeros((10, 60), int)
    etiquetas[:, 2:12] = 1
    etiquetas[:, 20:26] = 2
    etiquetas[:, 30:50] = 3
    anchos = anchos_aparentes(etiquetas, [1, 2, 3])
    assert anchos[2] > anchos[0] > anchos[1]


# --- que foto sirve para medir tercios ---------------------------------------- #
def test_una_costura_que_no_es_una_mordida_se_rechaza() -> None:
    """⚠️ En una foto OCLUSAL no existe eje cervical-incisal: se ve la cara de mordida de
    frente, asi que partir la corona en tercios reparte bandas que no son nada.

    El camino de coste minimo existe siempre —siempre hay un camino—, asi que la costura
    no se puede dar por buena por el hecho de haberla encontrado. Lo que la distingue es
    que sea OSCURA: si no hay linea de mordida, el camino se ve obligado a pasar por diente
    iluminado. Medido: la oclusal de un caso real da 1,01 y las cuatro vistas con eje 0,76
    a 0,87.
    """
    from gaussian_engine.tono_foto import costura_es_real

    mascara, lum = _dos_arcadas()
    costura = costura_oclusal(mascara, lum)
    assert costura_es_real(mascara, lum, costura)

    sin_linea = np.full_like(lum, 200.0)
    assert not costura_es_real(mascara, sin_linea, costura_oclusal(mascara, sin_linea))


# --- juntar varias fotos ------------------------------------------------------ #
def _lateral_sintetica(destino: Path, anchos: list[int], corona: int = 66) -> Path:
    """Una lateral de mentira: encia roja, dos filas de dientes palidos y su mordida.

    Los dientes van separados por franjas de encia para que el watershed tenga cuellos por
    donde partir, y el cuello de cada corona es mas rojo que el borde para que los tercios
    tengan un gradiente que comprobar.
    """
    from PIL import Image

    alto, sep, margen = 200, 8, 12
    ancho = margen * 2 + sum(anchos) + sep * (len(anchos) - 1)
    img = np.zeros((alto, ancho, 3), np.uint8)
    img[:, :] = (200, 90, 90)          # encia
    # ⚠️ Las coronas SE TOCAN en el punto de contacto y por eso la arcada es una sola
    # componente. Separandolas del todo con encia, `_arco` —que se queda con la mayor— se
    # quedaba con UN diente y no habia nada que partir. El hueco va solo por arriba: abajo
    # queda el puente del contacto, que es el cuello por donde el watershed corta.
    for y in range(96 - corona, 96):
        t = (y - (96 - corona)) / max(corona - 1, 1)
        img[y, margen:ancho - margen] = (225, int(180 + 30 * t), int(150 + 45 * t))
    img[99:170, margen:ancho - margen] = (225, 205, 190)   # corona de abajo, plana
    x = margen
    for w in anchos[:-1]:
        x += w
        img[96 - corona:96 - corona // 3, x:x + sep] = (200, 90, 90)   # hueco interproximal
        x += sep
    # ⚠️ La mordida es una linea FINA. Dibujandola gruesa las dos arcadas quedan en
    # componentes distintas y `_arco` —que se queda con la mayor— descarta una entera. En
    # una foto real el surco es de unos pocos pixeles y el cierre morfologico de
    # `mascara_diente` lo puentea, asi que la arcada llega como una sola componente: que es
    # justo el motivo por el que hay que partirla por la costura.
    img[96:99, :] = (25, 20, 20)
    Image.fromarray(img).save(destino, quality=98)
    return destino


def test_sin_saber_el_lado_no_se_declara_ni_un_color(tmp_path) -> None:
    """⚠️ **Jugarse el 16 contra el 26 a cara o cruz no es un error pequeno.**

    El arco es un espejo exacto y una foto intraoral puede estar tomada con espejo, asi
    que ni la huella de anchuras ni la imagen dicen de que lado es una tira. Aceptar la
    hipotesis de mas peso seria inventar la mitad de la boca. Se declara y se pregunta.
    """
    from gaussian_engine.tono_foto import tonos_de_fotos

    arco = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    tabla = np.array([9.0, 10.0, 6.5, 7.0, 7.5, 6.5, 8.5,
                      8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0])
    foto = _lateral_sintetica(tmp_path / "lateral.jpg", [45, 50, 33, 35, 38])

    tonos, motivos = tonos_de_fotos([foto], arco, tabla, esperadas=5)
    assert tonos == []
    # ⚠️ Contenido, no recuento: contar motivos falla cada vez que el emisor declara algo
    # más, que es justo lo que se le pide. Misma lección que en el test del lado declarado.
    assert any("espejo" in m for m in motivos)
    assert any("NINGUNA de las" in m for m in motivos)
    assert "espejo" in motivos[0]
    # El nombre del fichero NO viaja en el motivo: lleva datos del paciente.
    assert "lateral.jpg" not in motivos[0]
    assert "sha256:" in motivos[0]


def test_con_el_lado_declarado_cada_pieza_recibe_su_color(tmp_path) -> None:
    """Un bit de una persona y el resto sigue siendo medido."""
    from gaussian_engine.tono_foto import tonos_de_fotos

    arco = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    tabla = np.array([9.0, 10.0, 6.5, 7.0, 7.5, 6.5, 8.5,
                      8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0])
    foto = _lateral_sintetica(tmp_path / "lateral.jpg", [45, 50, 33, 35, 38])

    tonos, motivos = tonos_de_fotos([foto], arco, tabla, lado_conocido={foto: 17},
                                    esperadas=5)
    assert [t.fdi for t in tonos] == [13, 14, 15, 16, 17]  # ordenado por FDI
    # ⚠️ Y queda UN motivo, que es el correcto: una lateral ve cinco coronas de catorce, y
    # las nueve que no ve tienen que salir nombradas. Este `assert` decia `motivos == []`
    # cuando lo unico que se declaraba era la ambiguedad de lado; que una pieza se quede
    # sin color medido no se declaraba en ninguna parte.
    # ⚠️ Se comprueba QUE se declara, no CUANTAS cosas se declaran. Este `assert` decia
    # `motivos == []`, luego `len(motivos) == 1`, y volvio a romperse al empezar a
    # declararse la correccion de iluminacion — sin que ninguna de las dos veces hubiera
    # nada mal. Un test que cuenta motivos falla cada vez que el emisor declara algo mas,
    # que es justo lo que se le pide que haga.
    assert any("sin color medido" in m for m in motivos)
    assert any("corregido de iluminacion" in m for m in motivos)
    assert not any("espejo" in m for m in motivos)
    for t in tonos:
        assert t.n_pixeles > 0
        assert len(t.foto_sha256) == 64
        # El cuello se pinto mas rojo que el borde: `a*` tiene que decrecer.
        assert t.lab[0][1] > t.lab[2][1], t.fdi


def test_las_piezas_que_ninguna_foto_ve_salen_NOMBRADAS(tmp_path) -> None:
    """⚠️ **No declarar el color y no tener color son cosas distintas.**

    Quien abre el contenedor ve trece coronas con su color y una sin campo `color`, y no
    puede distinguir «no se midio» de «se midio y salio asi». Lo que el campo gaussiano
    tiene para esa pieza es el degradado de respaldo, que no es color de nadie. El caso
    real es el FDI 17: ninguna de las seis fotos del paciente lo ve con su eje
    cuello-borde, y el `.uos` no lo decia.
    """
    from gaussian_engine.tono_foto import tonos_de_fotos

    arco = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    tabla = np.array([9.0, 10.0, 6.5, 7.0, 7.5, 6.5, 8.5,
                      8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0])
    foto = _lateral_sintetica(tmp_path / "lateral.jpg", [45, 50, 33, 35, 38])

    _tonos, motivos = tonos_de_fotos([foto], arco, tabla, lado_conocido={foto: 17},
                                     esperadas=5)
    motivo = next(m for m in motivos if "sin color medido" in m)
    for fdi in (11, 12, 21, 22, 23, 24, 25, 26, 27):
        assert f"{fdi}" in motivo, fdi
    # Y NO nombra las que si se midieron.
    assert "degradado de respaldo" in motivo


def test_sin_una_sola_pieza_medida_no_se_declara_cobertura(tmp_path) -> None:
    """El motivo de cobertura sobra cuando el de verdad es que no se midio NADA.

    Listar las catorce piezas como «sin color» cuando ninguna foto sirvio es ruido encima
    del motivo que importa, que ya esta declarado aparte.
    """
    from gaussian_engine.tono_foto import tonos_de_fotos

    arco = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    tabla = np.array([9.0, 10.0, 6.5, 7.0, 7.5, 6.5, 8.5,
                      8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0])
    foto = _lateral_sintetica(tmp_path / "lateral.jpg", [45, 50, 33, 35, 38])

    tonos, motivos = tonos_de_fotos([foto], arco, tabla, esperadas=5)
    assert tonos == []
    assert not any("sin color medido" in m for m in motivos)


def test_cada_pieza_se_queda_con_la_foto_donde_ocupa_MAS_pixeles(tmp_path) -> None:
    """⚠️ No la primera ni una media. Una lateral ve la cara vestibular de frente y la
    oclusal la ve de canto; promediarlas mezcla una medida buena con una estirada."""
    from gaussian_engine.tono_foto import tonos_de_fotos

    arco = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    tabla = np.array([9.0, 10.0, 6.5, 7.0, 7.5, 6.5, 8.5,
                      8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0])
    anchos = [45, 50, 33, 35, 38]
    # Mismos anchos: lo que cambia es cuanta corona se ve, o sea el area.
    pequena = _lateral_sintetica(tmp_path / "a.jpg", anchos, corona=26)
    grande = _lateral_sintetica(tmp_path / "b.jpg", anchos, corona=66)
    lado = {pequena: 17, grande: 17}

    solo_pequena = tonos_de_fotos([pequena], arco, tabla, lado_conocido=lado,
                                  esperadas=5)[0]
    juntas = tonos_de_fotos([pequena, grande], arco, tabla, lado_conocido=lado,
                            esperadas=5)[0]
    assert [t.fdi for t in juntas] == [t.fdi for t in solo_pequena]
    for chica, mezcla in zip(solo_pequena, juntas, strict=True):
        assert mezcla.n_pixeles > chica.n_pixeles


# --- pintar la malla con el color por pieza ----------------------------------- #
def test_pinta_malla_da_a_cada_vertice_el_color_de_SU_pieza() -> None:
    """El degradado va **dentro** de la corona, entre sus tres tercios.

    ⚠️ Es lo que sustituye a interpolar entre vertices vecinos, que es de donde salia el
    color raro del visor: un vertice sin medida heredaba el del vertice medido mas cercano,
    y en la cara vestibular ese vecino es una sombra interdental de la foto oclusal.
    """
    from gaussian_engine.tono_foto import TonoPieza, pinta_malla

    # Dos coronas a distinta altura y un poco de encia.
    z = np.linspace(0.0, 10.0, 11)
    pos = np.concatenate([
        np.column_stack([np.zeros(11), np.zeros(11), z]),
        np.column_stack([np.ones(11) * 5, np.zeros(11), z]),
        np.array([[2.0, 0.0, -1.0]]),
    ])
    etq = np.array([11] * 11 + [21] * 11 + [0])
    tonos = [
        TonoPieza(fdi=11, lab=np.array([[60.0, 20.0, 20.0], [70.0, 10.0, 15.0],
                                        [80.0, 0.0, 10.0]]),
                  n_pixeles=100, foto_sha256="a" * 64),
        TonoPieza(fdi=21, lab=np.array([[50.0, 5.0, 5.0]] * 3),
                  n_pixeles=100, foto_sha256="a" * 64),
    ]
    encia = np.array([40.0, 25.0, 15.0])
    rgb, medido = pinta_malla(pos, etq, tonos, encia, np.array([0.0, 0.0, 1.0]))

    assert medido.all()
    # El 21 es plano: todos sus vertices salen del mismo color.
    assert len({tuple(c) for c in rgb[etq == 21]} ) == 1
    # El 11 tiene degradado a lo largo del eje, y no salta de golpe.
    del_11 = rgb[etq == 11].astype(int)
    assert len({tuple(c) for c in del_11}) > 3
    # La encia lleva SU color medido, no el de un diente.
    assert tuple(rgb[etq == 0][0]) not in {tuple(c) for c in rgb[etq != 0]}


def test_pinta_malla_no_inventa_donde_no_hay_medida() -> None:
    """Una pieza sin tono y una encia sin medir se quedan SIN declarar.

    Heredar de un vecino cualquiera es exactamente el fallo que este camino evita.
    """
    from gaussian_engine.tono_foto import TonoPieza, pinta_malla

    pos = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 5, 6)])
    etq = np.array([11, 11, 11, 26, 26, 0])
    tonos = [TonoPieza(fdi=11, lab=np.full((3, 3), 50.0), n_pixeles=99,
                       foto_sha256="b" * 64)]
    _, medido = pinta_malla(pos, etq, tonos, None, np.array([0.0, 0.0, 1.0]))
    assert medido[:3].all()          # el 11, que tiene tono
    assert not medido[3:].any()      # el 26 sin tono y la encia sin medir


def test_pinta_malla_obedece_al_eje_oclusal() -> None:
    """El reparto de tercios sale del eje MEDIDO, no de un supuesto sobre `z`."""
    from gaussian_engine.tono_foto import TonoPieza, pinta_malla

    pos = np.column_stack([np.linspace(0, 5, 6), np.zeros(6), np.zeros(6)])
    etq = np.full(6, 11)
    tonos = [TonoPieza(fdi=11, lab=np.array([[30.0, 0.0, 0.0], [55.0, 0.0, 0.0],
                                             [80.0, 0.0, 0.0]]),
                       n_pixeles=99, foto_sha256="c" * 64)]
    por_x = pinta_malla(pos, etq, tonos, None, np.array([1.0, 0.0, 0.0]))[0]
    por_z = pinta_malla(pos, etq, tonos, None, np.array([0.0, 0.0, 1.0]))[0]
    assert por_x[0].mean() < por_x[-1].mean()      # degradado a lo largo de x
    assert len({tuple(c) for c in por_z}) == 1     # sobre z no hay extension: color plano


def test_color_de_encia_sale_de_lo_que_rodea_al_arco(tmp_path) -> None:
    """⚠️ La mucosa tambien se MIDE, y es UN valor.

    El marron embarrado sobre la encia vestibular no es un color de encia: es interpolacion
    tirando del vertice con color mas cercano, que ahi es una sombra interdental. Con un
    valor medido no hay nada que interpolar.

    Se probo a partirlo en paladar y vestibulo: medido, ΔE 2,9 — el umbral de lo que un ojo
    distingue. Ver el docstring de `color_de_encia`.
    """
    from gaussian_engine.tono_foto import _lab, color_de_encia

    foto = _lateral_sintetica(tmp_path / "lateral.jpg", [45, 50, 33, 35, 38])
    medida = color_de_encia([foto])
    assert medida is not None
    esperado = _lab(np.array([[(200, 90, 90)]], dtype=np.uint8))[0, 0]
    assert np.abs(medida - esperado).max() < 12.0
    diente = _lab(np.array([[(225, 195, 172)]], dtype=np.uint8))[0, 0]
    assert np.linalg.norm(medida - esperado) < np.linalg.norm(medida - diente)


def test_color_de_encia_ignora_una_foto_sin_eje_cervical_incisal(tmp_path) -> None:
    """Sin linea de mordida no se sabe que es mucosa vestibular y que es otra cosa."""
    from gaussian_engine.tono_foto import color_de_encia
    from PIL import Image

    plana = tmp_path / "sin_mordida.jpg"
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :] = (200, 90, 90)
    img[40:160, 40:160] = (225, 205, 190)
    Image.fromarray(img).save(plana, quality=98)
    assert color_de_encia([plana]) is None


def test_el_color_de_una_pieza_NO_baja_mas_de_su_altura_de_corona() -> None:
    """⚠️ **La etiqueta se pasa, y sin cota el color se pasa con ella.**

    La segmentación etiqueta como diente el 70,2 % de la malla cuando un experto etiqueta
    el 53,9 %: en el caso real hay 11.800 vértices con código FDI por debajo del arranque
    de las coronas. Pintándolos con el tono de su pieza, el paladar salía como un mosaico
    —cada parche del color del diente que se lo llevó— y el degradado de los tres tercios
    se repartía sobre una altura falsa.

    La cota no dice dónde está el margen gingival: dice hasta dónde se puede AFIRMAR color
    de corona. Lo de más abajo se pinta con encía, que también es medida.
    """
    import numpy as np
    from gaussian_engine.tono_foto import TonoPieza, pinta_malla

    eje = np.array([0.0, 0.0, 1.0])
    # Una columna de 40 mm etiquetada entera como el 11, cuya corona mide 10,5 de tabla.
    z = np.linspace(0.0, 40.0, 81)
    pos = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    etq = np.full(len(z), 11)
    tono = TonoPieza(fdi=11, lab=np.array([[55.0, 10.0, 20.0]] * 3),
                     n_pixeles=100, foto_sha256="a" * 64)
    encia = np.array([50.0, 25.0, 12.0])

    rgb, medido = pinta_malla(pos, etq, [tono], encia, eje)

    assert medido.all(), "todo queda medido: corona arriba, encía abajo"
    # 10,5 mm x 1,3 de margen = 13,65 mm por debajo del borde, o sea desde z = 26,35.
    es_corona = z >= 40.0 - 1.3 * 10.5
    color_encia = np.asarray(rgb[~es_corona])
    assert (color_encia == color_encia[0]).all(), "lo de debajo de la cota es todo encía"
    assert not np.array_equal(rgb[es_corona][0], color_encia[0])
    # Y lo importante: los 26 mm más apicales NO llevan el tono del 11.
    assert (~es_corona).sum() > len(z) // 2


def test_sin_encia_medida_la_cota_no_inventa_un_color() -> None:
    """Si no se midió encía, lo que cae bajo la cota se queda sin declarar, no gris."""
    import numpy as np
    from gaussian_engine.tono_foto import TonoPieza, pinta_malla

    z = np.linspace(0.0, 40.0, 81)
    pos = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    tono = TonoPieza(fdi=11, lab=np.array([[55.0, 10.0, 20.0]] * 3),
                     n_pixeles=100, foto_sha256="a" * 64)

    _rgb, medido = pinta_malla(pos, np.full(len(z), 11), [tono], None,
                               np.array([0.0, 0.0, 1.0]))
    assert not medido.all()
    assert medido[z >= 40.0 - 1.3 * 10.5].all()


# ───────────────────────── la caida del flash ──────────────────────────────
#
# Lo que se corrige aqui es la razon de que el mismo paciente saliera con el 21 blanco
# (`L*` 81,8) y el 27 marron (`L*` 59,0): 22,7 puntos de recorrido con correlacion −0,86
# contra la distancia a la linea media. No son dientes distintos, es el flash.


def _caso_con_flash(n: int = 8, gamma: float = 0.6) -> tuple[list, list, np.ndarray]:
    """`n` coronas IGUALES vistas con luz decreciente, y su encia como sonda.

    La encia se oscurece MAS deprisa que el diente (`gamma > 0`), que es lo medido en el
    caso real: 24,8 puntos de recorrido frente a 20,8 del diente. Esa es exactamente la
    condicion que hace que dividir por la sonda se pase de frenada.
    """
    diente = np.array([0.62, 0.44, 0.34])   # un esmalte, en luz lineal
    encia = np.array([0.55, 0.18, 0.13])    # mucosa
    luz = np.linspace(1.0, 0.35, n)
    dientes = [diente * s for s in luz]
    encias = [encia * s ** (1.0 + gamma) for s in luz]
    return dientes, encias, luz


def test_la_correccion_aplana_el_degradado_del_flash() -> None:
    """Ocho coronas identicas vistas con luz distinta tienen que salir identicas."""
    dientes, encias, _ = _caso_con_flash()
    ref, beta = ajuste_de_iluminacion(dientes, encias)

    antes = _lab(_srgb(np.stack(dientes)))[:, 0]
    corregidos = np.stack([d * (ref / e) ** beta for d, e in zip(dientes, encias, strict=True)])
    despues = _lab(_srgb(np.clip(corregidos, 0, 1)))[:, 0]

    assert antes.max() - antes.min() > 25.0     # el artefacto existe
    assert despues.max() - despues.min() < 2.0  # y se ha ido


def test_la_correccion_no_invierte_el_degradado() -> None:
    """⚠️ **El error que hay que no volver a cometer.**

    Dividir por la sonda —`beta = 1`— asume que la encia recibe la misma luz que el diente
    y nada mas. Es falso: la encia esta retraida y le entra sombra propia, asi que recorre
    MAS que el diente. Al dividir, el molar mas oscuro de la boca sale como el diente mas
    claro. Medido sobre el caso real: `L*` 58,6 se convertia en 83,4, por encima del 80,2
    del incisivo. `beta` sale del dato justamente para que eso no pueda pasar.
    """
    dientes, encias, luz = _caso_con_flash()
    ref, beta = ajuste_de_iluminacion(dientes, encias)
    assert np.all(beta < 1.0), "si beta llega a 1 esto es una division y se pasa"

    # ⚠️ En luz LINEAL y sin recortar. Dividir dispara el canal por encima de 1,0 y el
    # recorte a blanco lo dejaria todo constante: la inversion quedaria escondida detras
    # de una saturacion, y la correlacion saldria `NaN` en vez de negativa.
    ingenuo = np.stack([d * (ref / e) for d, e in zip(dientes, encias, strict=True)])[:, 1]
    medido = np.stack([d * (ref / e) ** beta
                       for d, e in zip(dientes, encias, strict=True)])[:, 1]

    bruto = np.stack(dientes)[:, 1]

    def recorrido(v: np.ndarray) -> float:
        return float(v.max() - v.min())

    # Dividir le da la vuelta: donde menos luz habia, mas claro sale.
    assert np.corrcoef(luz, ingenuo)[0, 1] < -0.9
    assert np.argmax(ingenuo) == np.argmin(bruto), \
        "el mas oscuro no acaba siendo el mas claro"
    # ⚠️ **Lo que NO se afirma:** que dividir amplie el artefacto. Se escribio ese `assert`
    # y es falso — la amplitud que queda depende de cuanto mas caiga la encia que el
    # diente, y con la caida de este sintetico sale menor. Lo que si es siempre cierto, y
    # es lo grave, es que cambia de signo: el orden de las piezas queda del reves.

    # La regresion no. Aqui NO se mira la correlacion: la correccion sale exacta sobre el
    # sintetico, `medido` queda constante salvo error de coma flotante, y correlacionar una
    # constante devuelve ruido —este mismo `assert` fallaba por 0,5175 midiendo eso—. Lo
    # que se afirma es el recorrido, que es lo que se queria arreglar.
    assert recorrido(medido) < 0.02 * recorrido(bruto)


def test_sin_coronas_suficientes_no_se_corrige_nada() -> None:
    """Con tres sondas la pendiente la decide el ruido: mejor no tocar y declararlo."""
    dientes, encias, _ = _caso_con_flash(n=MINIMO_CORONAS_SONDA - 1)
    assert ajuste_de_iluminacion(dientes, encias) is None


def test_la_pendiente_es_cero_si_la_sonda_no_explica_nada() -> None:
    """Encia constante y dientes distintos: no hay nada que descontar.

    Es el caso que protege de aplanar diferencias REALES entre piezas. Si la sonda no
    varia, `beta` no tiene sobre que regresar y la correccion no puede inventarse una.
    """
    encia = np.array([0.55, 0.18, 0.13])
    dientes = [np.array([0.62, 0.44, 0.34]) * f for f in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)]
    ref, beta = ajuste_de_iluminacion(dientes, [encia.copy() for _ in dientes])
    corregidos = np.stack([d * (ref / encia) ** beta for d in dientes])
    assert np.allclose(corregidos, np.stack(dientes), rtol=1e-9)


# ───────────────────────── la encia dentro de la muestra ───────────────────


def _parche(alto: int, ancho: int, diente: list[float], *, hasta: int,
            mucosa: list[float]) -> np.ndarray:
    """Una corona con las `hasta` primeras filas de mucosa, y con RUIDO.

    ⚠️ **El ruido no es decoracion del fixture.** Sin el, la imagen tiene dos valores
    exactos de `L*` y el percentil 90 del filtro de brillo cae justo sobre el del esmalte:
    `lab[:, 0] < percentile(...)` se lleva de golpe los 44 pixeles de diente y deja solo
    mucosa. El primer intento de este test fallaba por eso y no por el codigo.
    """
    rng = np.random.default_rng(7)
    rgb = np.empty((alto, ancho, 3), np.uint8)
    rgb[:, :] = rgb_de_lab(np.asarray(diente, dtype=float))
    rgb[:hasta, :] = rgb_de_lab(np.asarray(mucosa, dtype=float))
    ruido = rng.integers(-3, 4, size=rgb.shape)
    return np.clip(rgb.astype(int) + ruido, 0, 255).astype(np.uint8)


def test_se_rechaza_la_mucosa_que_se_come_el_tercio_cervical() -> None:
    """Un canino salio con `a*` 17,5 teniendo vecinos en 7,2 y 7,4. Eso era encia.

    ⚠️ **Y no basta con que la mascara lleve «algo» de mucosa: tiene que llevarla DONDE
    duele.** Una mediana sobre toda la corona aguanta un 30 % de encia casi sin moverse
    —el primer intento de este test pedia que se moviera y no se movio—, porque la mediana
    es robusta y para eso se eligio. Lo que la rompe es que la mascara se desborde por el
    CUELLO, que es por donde se desborda de verdad: entonces el tercio cervical es mucosa
    casi entera y el degradado por tercios, que es lo que se declara, sale invertido.
    """
    rgb = _parche(60, 40, [75.0, 8.0, 22.0], hasta=16, mucosa=[55.0, 30.0, 24.0])
    mascara = np.ones(rgb.shape[:2], bool)
    encia_lineal = _lineal(rgb_de_lab(np.array([55.0, 30.0, 24.0])))

    crudo = tono_por_tercios(rgb, mascara, (0.0, -1.0))
    limpio = tono_por_tercios(rgb, mascara, (0.0, -1.0), encia=encia_lineal)
    assert crudo is not None and limpio is not None

    # Sin rechazo el cervical se declara rosa; con rechazo, esmalte.
    assert crudo[0][1] > 20.0
    assert abs(limpio[0][1] - 8.0) < 1.5
    # Y el tercio incisal, que nunca tuvo mucosa, no se toca.
    assert abs(crudo[2][1] - limpio[2][1]) < 1.5


def test_no_se_rechaza_cuando_lo_que_sobra_es_media_corona() -> None:
    """Tirar media mascara no es limpiar un borde: es que esa mascara no es esa pieza.

    Ahi el rechazo no arregla nada —el resto tampoco es diente— y ademas taparia el
    problema. Se deja el dato crudo y que el gate lo vea. Ver `RECHAZO_MAXIMO`.
    """
    # ⚠️ 48 % de mucosa Y con la MISMA `L*` que el esmalte. Las dos cosas hacen falta: la
    # fraccion, para que la mediana siga siendo diente por poco y el rechazo quiera tirar
    # casi media mascara; y la claridad igual, para que el filtro de brillo se lleve el
    # mismo 10 % de las dos poblaciones y no cambie la proporcion por detras. Con `L*`
    # distintas el filtro recorta solo la cola clara del diente y la banda se escapa.
    rgb = _parche(100, 40, [75.0, 8.0, 22.0], hasta=48, mucosa=[75.0, 30.0, 24.0])
    mascara = np.ones(rgb.shape[:2], bool)
    encia_lineal = _lineal(rgb_de_lab(np.array([75.0, 30.0, 24.0])))

    crudo = tono_por_tercios(rgb, mascara, (0.0, -1.0))
    con_sonda = tono_por_tercios(rgb, mascara, (0.0, -1.0), encia=encia_lineal)
    assert crudo is not None and con_sonda is not None
    assert np.allclose(crudo, con_sonda)


def test_el_rechazo_separa_por_cromaticidad_y_no_por_claridad() -> None:
    """Una corona en sombra NO es encia, y un umbral en `L*` la tiraria.

    La iluminacion mueve sobre todo `L*` —es justo el artefacto que se acaba de corregir—,
    asi que separar por claridad confundiria sombra con mucosa. En `a*b*` la mucosa y el
    esmalte estan lejos y la luz apenas los mueve.
    """
    rng = np.random.default_rng(1)
    # El mismo diente, la mitad a media luz: `L*` 75 y 55, cromaticidad igual.
    lab = np.concatenate([
        np.array([75.0, 8.0, 22.0]) + rng.normal(0, 1.0, (500, 3)),
        np.array([55.0, 8.0, 22.0]) + rng.normal(0, 1.0, (500, 3)),
    ])
    queda = _sin_mucosa(lab, np.array([55.0, 30.0, 24.0]))
    assert queda.mean() > 0.98, "la parte en sombra se ha confundido con encia"


def _lineal_de_lab(lab: list[float]) -> np.ndarray:
    return _lineal(rgb_de_lab(np.asarray(lab, dtype=float)))


def test_una_sonda_que_no_es_mucosa_no_sirve_de_referencia() -> None:
    """Un separador de plástico por encima del cuello no es encía.

    Corregir una corona contra él la deja con un color inventado y con toda la pinta de
    estar medido, que es la peor combinación posible.
    """
    diente = _lineal_de_lab([75.0, 8.0, 22.0])
    mucosa = _lineal_de_lab([55.0, 29.0, 24.0])
    plastico = _lineal_de_lab([88.0, 1.0, 2.0])
    espejo = _lineal_de_lab([40.0, 0.0, -3.0])

    ok = sondas_de_mucosa([diente] * 3, [mucosa, plastico, espejo])
    assert ok.tolist() == [True, False, False]


def test_la_papila_es_mucosa_aunque_se_aparte_de_las_demas() -> None:
    """⚠️ **El criterio que se probó primero y era peor que no filtrar.**

    Rechazaba las sondas que se apartaban más de 3 MAD de las demás. Sobre el caso real
    tiró exactamente las dos buenas: la encía del 13 y la del 22 son papila, más saturadas
    que la encía adherida —`a*` 36,1 y 31,2 frente a 25-30—, no un separador. Esas dos
    piezas se quedaban con el artefacto del flash puesto Y con mucosa dentro de la muestra:
    el 13 volvía a declararse con `a*` 17,5, que es justo el diente «rosa» del que venía
    toda esta investigación.

    Lo que separa mucosa de no-mucosa no es apartarse de la media, es el signo.
    """
    diente = _lineal_de_lab([75.0, 8.0, 22.0])
    adherida = [_lineal_de_lab([55.0, a, 24.0]) for a in (25.0, 26.0, 27.0, 28.0, 29.0)]
    papila = _lineal_de_lab([48.0, 36.0, 26.0])

    ok = sondas_de_mucosa([diente] * 6, [*adherida, papila])
    assert ok.all(), "la papila se ha descartado y es encía"


def test_una_foto_declarada_como_no_valida_no_aporta_color(tmp_path) -> None:
    """⚠️ **«No sirve» tiene que poder decirse, y darle un lado sería peor.**

    En un caso real el gate pedía «de qué lado es» a dos fotos que no eran tiras
    vestibulares de esa arcada: una era de la arcada contraria y otra un primer plano de
    una sola pieza, cuyas ocho «coronas» detectadas eran sus cúspides. Contestarle un lado
    a esa segunda repartiría ocho códigos FDI entre las cúspides de un molar y plantaría
    ese color en el contenedor, sin que nada lo delatara.
    """
    from gaussian_engine.tono_foto import tonos_de_fotos

    arco = [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    tabla = np.array([9.0, 10.0, 6.5, 7.0, 7.5, 6.5, 8.5,
                      8.5, 6.5, 7.5, 7.0, 6.5, 10.0, 9.0])
    foto = _lateral_sintetica(tmp_path / "lateral.jpg", [45, 50, 33, 35, 38])

    tonos, motivos = tonos_de_fotos([foto], arco, tabla,
                                    lado_conocido={foto: NO_ES_TIRA}, esperadas=5)
    assert tonos == [], "una foto descartada no puede aportar ni un tono"
    assert any("NO tira vestibular" in m for m in motivos)
    # Y no se queda callado: un contenedor sin una sola corona con color tiene que decirlo.
    assert any("NINGUNA de las" in m for m in motivos)
