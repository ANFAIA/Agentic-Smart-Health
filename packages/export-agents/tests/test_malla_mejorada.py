"""La malla del escáner con el color que mide el campo. Ver `export_agents.malla_mejorada`.

Esto existe por una queja concreta y reproducible: el fichero emitido tenía «motas de
colores extrañas» sobre las coronas. La causa no era el color medido —que es bueno, ΔE
mediano 0,35 por pieza— sino cómo se leía del campo.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
from export_agents.malla_mejorada import (
    ALCANCE_SIGMAS,
    C0,
    NEUTRO,
    color_desde_gaussianas,
    escribe_3mf,
    escribe_ply,
    escribe_stl_viscam,
)


def _dc(rgb: tuple[float, float, float]) -> np.ndarray:
    """El `f_dc` que produce ese color: `color = C0 * f_dc + 0,5`."""
    return (np.asarray(rgb, float) - 0.5) / C0


def test_una_gaussiana_rara_y_cercana_no_pinta_una_mota() -> None:
    """⚠️ **El fallo que motivó `VECINOS_COLOR`.**

    El campo de apariencia tiene opacidad mediana 0,026: cada píxel del render es la suma
    de decenas de gaussianas, así que una con color extremo aporta su 2,6 % y no se ve.
    Muestreando la MÁS CERCANA, esa misma gaussiana sale al 100 % y aparece como una mota
    azul sobre una corona. Aquí la rara es además la más próxima, que es el caso peor.
    """
    # Una capa de gaussianas sobre la superficie, como el campo real: todas a distancia
    # parecida del vértice, ninguna dominante. La rara es la ÚNICA pegada de verdad.
    u = np.arange(-3, 4) * 0.08
    plano = np.stack(np.meshgrid(u, u, [0.06]), -1).reshape(-1, 3)
    centros = np.concatenate([np.array([[0.0, 0.0, 0.02]]), plano])
    f_dc = np.concatenate([_dc((0.0, 0.0, 1.0))[None],
                           np.tile(_dc((0.9, 0.85, 0.8)), (len(plano), 1))])
    opac = np.full(len(centros), 0.026)
    esc = np.full(len(centros), 0.2)
    vertice = np.zeros((1, 3))

    rgb, medido = color_desde_gaussianas(vertice, centros, f_dc, opac, esc)
    assert medido[0]

    # Lo que habría dado muestrear la MÁS CERCANA, que es lo que se hacía: azul puro.
    cercana = np.clip(f_dc[0] * C0 + 0.5, 0, 1) * 255
    assert cercana[2] > 240 and cercana[0] < 15

    # Lo que da la mezcla: esmalte, con la rara reducida a su peso real.
    assert rgb[0][0] > rgb[0][2], "el vértice ha salido más azul que rojo: es la mota"
    assert rgb[0][0] > 190


def test_lo_que_ninguna_gaussiana_alcanza_sale_neutro_y_marcado() -> None:
    """⚠️ **«No sé el color» tiene que distinguirse de «lo sé mal».**

    Es la superficie que ninguna foto ve. Rellenarla con marfil la haría pasar por dato.
    """
    sigma = 0.3
    centros = np.zeros((4, 3))
    f_dc = np.tile(_dc((0.9, 0.85, 0.8)), (4, 1))
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ALCANCE_SIGMAS * sigma * 3]])

    rgb, medido = color_desde_gaussianas(pos, centros, f_dc,
                                         np.full(4, 0.5), np.full(4, sigma))
    assert medido.tolist() == [True, False]
    assert np.array_equal(rgb[1], NEUTRO)


def test_una_gaussiana_grande_SI_cubre_un_punto_lejano() -> None:
    """⚠️ **El fallo que se veía en pantalla: manchas grises sobre los incisivos.**

    El alcance iba en milímetros, con un corte fijo de 1,0 mm. Sobre el caso real dejaba
    2.635 vértices en `NEUTRO` — el **100 %** de ellos sobre coronas etiquetadas y a 1,08 mm
    de mediana, o sea justo al otro lado del corte. En una cara vestibular, que es lisa, el
    optimizador usa gaussianas GRANDES: una con `sigma` 0,61 mm cubre de sobra un punto a
    1,08 mm, que son 1,81 sigmas. El corte no estaba mal calibrado, estaba mal planteado —
    un milímetro no significa lo mismo para cada gaussiana.
    """
    grande, pequena = 0.6, 0.05
    f_dc = np.tile(_dc((0.9, 0.85, 0.8)), (4, 1))
    punto = np.array([[0.0, 0.0, 1.1]])          # a 1,1 mm: fuera del viejo corte de 1 mm

    _, con_grande = color_desde_gaussianas(punto, np.zeros((4, 3)), f_dc,
                                           np.full(4, 0.5), np.full(4, grande))
    _, con_pequena = color_desde_gaussianas(punto, np.zeros((4, 3)), f_dc,
                                            np.full(4, 0.5), np.full(4, pequena))
    assert con_grande[0], "una gaussiana de sigma 0,6 mm sí cubre un punto a 1,1 mm"
    assert not con_pequena[0], "una de sigma 0,05 mm no, y eso sí es superficie sin color"


def test_el_ply_declara_donde_el_color_es_del_paciente(tmp_path) -> None:
    """La columna `medido` viaja y se lee sin abrir el `.uos`."""
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    caras = np.array([[0, 1, 2]], np.int32)
    rgb = np.array([[10, 20, 30], [40, 50, 60], [160, 160, 160]], np.uint8)
    medido = np.array([True, True, False])
    ruta = tmp_path / "arcada.ply"
    escribe_ply(ruta, pos, caras, rgb, np.array([11, 11, 0], np.int16),
                ["color medido del campo de apariencia"], medido)

    crudo = ruta.read_bytes()
    fin = crudo.index(b"end_header\n") + len(b"end_header\n")
    cab = crudo[:fin].decode("ascii")
    assert "property uchar medido" in cab
    assert "element face 1" in cab

    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                   ("b", "u1"), ("fdi", "<i2"), ("medido", "u1")])
    v = np.frombuffer(crudo[fin:fin + 3 * dt.itemsize], dt)
    assert v["medido"].tolist() == [1, 1, 0]
    assert v["fdi"].tolist() == [11, 11, 0]


def test_la_cabecera_del_ply_es_ascii_estricto(tmp_path) -> None:
    """⚠️ El formato PLY no define otra codificación, así que un carácter fuera de rango
    deja el fichero ilegible para quien sí la respeta. Mejor que reviente al escribir."""
    with pytest.raises(UnicodeEncodeError):
        escribe_ply(tmp_path / "x.ply", np.zeros((1, 3), np.float32),
                    np.zeros((0, 3), np.int32), np.zeros((1, 3), np.uint8),
                    np.zeros(1, np.int16), ["ΔE mediano 0,35"])


def test_el_3mf_lleva_color_por_VERTICE_y_no_por_triangulo(tmp_path) -> None:
    """⚠️ **Si el color fuera por triángulo, el degradado de la corona se perdería.**

    Un diente no es de un color: se oscurece hacia el cuello. Ese degradado es el dato
    clínico, y un color plano por cara lo tira.
    """
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    caras = np.array([[0, 1, 2]])
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    ruta = tmp_path / "arcada.3mf"
    escribe_3mf(ruta, pos, caras, rgb, "arcada con color medido")

    with zipfile.ZipFile(ruta) as z:
        assert set(z.namelist()) == {"[Content_Types].xml", "_rels/.rels",
                                     "3D/3dmodel.model"}
        modelo = z.read("3D/3dmodel.model").decode()
    assert 'unit="millimeter"' in modelo
    assert modelo.count("<m:color ") == 3
    # Tres índices DISTINTOS en el mismo triángulo: eso es lo que permite el degradado.
    triangulo = modelo[modelo.index("<triangle "):modelo.index("</triangles>")]
    indices = {triangulo.split(f'p{k}="')[1].split('"')[0] for k in (1, 2, 3)}
    assert len(indices) == 3


def test_el_stl_con_color_tiene_el_tamano_que_dice_el_formato(tmp_path) -> None:
    """80 bytes de cabecera + 4 de cuenta + 50 por triángulo, y el bit 15 a 1.

    ⚠️ Ese bit es lo único que distingue «este campo lleva color» de «este campo lleva
    basura», y aun así la mayoría de lectores lo ignoran. Ver el docstring de la función.
    """
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    caras = np.array([[0, 1, 2]])
    rgb = np.full((3, 3), 255, np.uint8)
    ruta = tmp_path / "arcada.stl"
    escribe_stl_viscam(ruta, pos, caras, rgb, "arcada con color medido (VisCAM, no estandar)")

    crudo = ruta.read_bytes()
    assert len(crudo) == 84 + 50
    assert crudo[:80].rstrip(b"\0").startswith(b"arcada con color medido")
    assert int.from_bytes(crudo[80:84], "little") == 1
    attr = int.from_bytes(crudo[84 + 48:84 + 50], "little")
    assert attr & (1 << 15)
    assert attr & 0x1F == 31  # blanco saturado en el canal rojo


def test_el_3mf_es_estructuralmente_valido(tmp_path) -> None:
    """XML bien formado, y ningún índice fuera de rango.

    ⚠️ **Esto se comprobó porque un fichero real «no se podía abrir».** Resultó no ser el
    fichero —MeshLab sencillamente no soporta 3MF— pero la única forma de saberlo era
    poder afirmar que el fichero estaba bien. Un índice de vértice o de color fuera de
    rango produce exactamente el mismo síntoma y sí sería culpa nuestra.
    """
    from xml.etree import ElementTree as ET

    rng = np.random.default_rng(3)
    pos = rng.normal(0, 5, (60, 3))
    caras = rng.integers(0, 60, (80, 3))
    caras = caras[[len(set(c)) == 3 for c in caras]]
    rgb = rng.integers(0, 256, (60, 3)).astype(np.uint8)
    ruta = tmp_path / "a.3mf"
    escribe_3mf(ruta, pos, caras, rgb, "prueba")

    with zipfile.ZipFile(ruta) as z:
        raiz = ET.fromstring(z.read("3D/3dmodel.model"))
    ns = {"c": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
          "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"}
    vert = raiz.findall(".//c:vertex", ns)
    tri = raiz.findall(".//c:triangle", ns)
    col = raiz.findall(".//m:color", ns)
    assert len(vert) == 60 and len(tri) == len(caras) and col
    for t in tri:
        assert max(int(t.get(k)) for k in ("v1", "v2", "v3")) < len(vert)
        assert max(int(t.get(k)) for k in ("p1", "p2", "p3")) < len(col)


def test_la_paleta_del_3mf_se_reduce_sin_cambiar_el_color_a_ojo(tmp_path) -> None:
    """⚠️ **Se cuantiza el 3MF y NO el PLY, y la diferencia importa.**

    El PLY es el fichero que se mira y del que se puede medir; el 3MF es el que se imprime,
    y ninguna impresora reproduce un ΔE de 1. Un caso real daba 10.627 colores distintos
    sobre 112.067 vértices, y hay lectores que acotan el tamaño de un `colorgroup`.
    """
    from xml.etree import ElementTree as ET

    from export_agents.malla_mejorada import PASO_PALETA_3MF

    rng = np.random.default_rng(4)
    rgb = rng.integers(100, 200, (300, 3)).astype(np.uint8)
    pos = rng.normal(0, 1, (300, 3))
    caras = np.arange(300).reshape(100, 3)
    ruta = tmp_path / "a.3mf"
    escribe_3mf(ruta, pos, caras, rgb, "prueba")

    with zipfile.ZipFile(ruta) as z:
        raiz = ET.fromstring(z.read("3D/3dmodel.model"))
    colores = raiz.findall(".//m:color",
                           {"m": "http://schemas.microsoft.com/3dmanufacturing/"
                                 "material/2015/02"})
    assert len(colores) < len(np.unique(rgb, axis=0))
    # Y cada componente sigue siendo múltiplo del paso: la reducción es la declarada.
    for c in colores:
        v = c.get("color").lstrip("#")
        assert all(int(v[i:i + 2], 16) % PASO_PALETA_3MF == 0 for i in (0, 2, 4))


def test_un_hueco_suelto_se_rellena_de_sus_vecinos() -> None:
    """⚠️ **Un vértice gris en medio de una corona medida se lee como un fallo del color.**

    Sobre el caso real eran 112 vértices de 112.067 que ninguna gaussiana llegaba a cubrir,
    y en pantalla eran una mota gris azulada sobre un incisivo. Se les da el color de sus
    vecinos medidos y `medido` **sigue a 0**: lo que se rellena es la visualización, no la
    afirmación.
    """
    from export_agents.malla_mejorada import rellena_huecos

    idx = np.arange(25).reshape(5, 5)
    caras = np.concatenate([
        np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
    ])
    rgb = np.tile(np.array([210, 180, 140], np.uint8), (25, 1))
    hueco = 12                                   # el del centro
    rgb[hueco] = NEUTRO
    medido = np.ones(25, bool)
    medido[hueco] = False

    salida = rellena_huecos(caras, rgb, medido, np.full(25, 11, np.int16))
    assert not np.array_equal(salida[hueco], NEUTRO)
    assert np.array_equal(salida[hueco], np.array([210, 180, 140], np.uint8))


def test_una_pieza_ENTERA_sin_color_no_se_rellena_de_las_vecinas() -> None:
    """⚠️ **La condición que impide inventarse un diente.**

    El FDI 17 no lo ve ninguna foto y el emisor lo declara. Rellenando sin mirar el código
    se llenaría desde las piezas de al lado y saldría con un color que no es de nadie, sin
    que nada en el fichero lo delatara. Al propagar sólo dentro de la misma pieza, sus
    vértices no tienen ni un vecino medido de su propio código y se quedan en gris.
    """
    from export_agents.malla_mejorada import rellena_huecos

    idx = np.arange(25).reshape(5, 5)
    caras = np.concatenate([
        np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
    ])
    fdi = np.where(np.arange(25) % 5 < 3, 11, 17).astype(np.int16)
    rgb = np.tile(np.array([210, 180, 140], np.uint8), (25, 1))
    medido = fdi == 11
    rgb[~medido] = NEUTRO

    salida = rellena_huecos(caras, rgb, medido, fdi)
    assert (salida[fdi == 17] == NEUTRO).all(), "el 17 se ha rellenado desde sus vecinas"
