"""Color por pieza desde una foto, sin pose. Ver `gaussian_engine.tono_foto`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gaussian_engine.tono_foto import (
    Alineamiento,
    alinea_con_el_arco,
    anchos_aparentes,
    banda_superior,
    costura_oclusal,
    rgb_de_lab,
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
    assert len(motivos) == 1
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
    assert motivos == []
    assert [t.fdi for t in tonos] == [13, 14, 15, 16, 17]  # ordenado por FDI
    for t in tonos:
        assert t.n_pixeles > 0
        assert len(t.foto_sha256) == 64
        # El cuello se pinto mas rojo que el borde: `a*` tiene que decrecer.
        assert t.lab[0][1] > t.lab[2][1], t.fdi


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
