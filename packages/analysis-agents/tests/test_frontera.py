"""La frontera de cada pieza. Ver `analysis_agents.frontera`.

Esto existe por lo que se ve en el visor: seleccionas una pieza y viene con medio diente
vecino. El contenedor decía la verdad sobre esa pieza y enseñaba otra cosa.
"""

from __future__ import annotations

import numpy as np
from analysis_agents.frontera import (
    EXCESO_ADMITIDO_MM,
    nucleos,
    reparte_por_geodesica,
)


def _rejilla(n: int = 24, m: int = 12, paso: float = 0.4):
    """Una lámina `n x m` triangulada, como un trozo de superficie."""
    idx = np.arange(n * m).reshape(n, m)
    x, y = np.meshgrid(np.arange(n) * paso, np.arange(m) * paso, indexing="ij")
    pos = np.stack([x.ravel(), y.ravel(), np.zeros(n * m)], 1)
    caras = np.concatenate([
        np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
    ])
    return pos, caras, idx


def test_una_isla_suelta_vuelve_a_su_pieza() -> None:
    """Un maxilar real trae decenas de islas por código, lejos de la pieza que las declara.

    ⚠️ **Lo que NO se afirma:** que la competencia arregle el desborde. Se escribió un test
    que lo daba por hecho y es falso — ver `test_un_desborde_UNIFORME…` y el aviso de la
    propia función. Lo que sí hace, y es comprobable, es devolver a su sitio un trozo que
    está más cerca de otra pieza que de la que lo reclama.
    """
    pos, caras, idx = _rejilla()
    verdad = np.where(np.arange(24 * 12) // 12 < 12, 11, 12).astype(np.int64)
    sucia = verdad.copy()
    sucia[idx[20, 3]] = 11                    # una isla del 11 en pleno territorio del 12

    limpia = reparte_por_geodesica(pos, caras, sucia)
    assert limpia[idx[20, 3]] == 12
    # La frontera entre las dos piezas se mueve como mucho un anillo: en una rejilla el
    # punto medio geodesico no cae exactamente sobre la fila de la verdad.
    assert (limpia == verdad).mean() > 0.95


def test_un_desborde_UNIFORME_no_lo_arregla_la_competencia_sola() -> None:
    """⚠️ **El límite del método, y hay que tenerlo escrito.**

    La competencia deja la frontera a medio camino ENTRE LOS NÚCLEOS, así que si la
    etiqueta invade a la vecina de forma pareja, el núcleo se desplaza con ella y el
    resultado no se mueve. Reconstruir la frontera diente-diente de Teeth3DS+ desde los
    núcleos da 0,995 de acuerdo — pero eso es partiendo de etiquetas de experto, donde los
    núcleos están bien. Sobre el caso real la competencia sola no cambia ni una pieza de
    veredicto (4/14) y sólo baja el |exceso| mediano de 2,60 a 2,38 mm.

    Lo que sí funciona es combinarla con el recorte al cuello: 4/14 -> 7/14.
    """
    pos, caras, idx = _rejilla()
    verdad = np.where(np.arange(24 * 12) // 12 < 12, 11, 12).astype(np.int64)
    sucia = verdad.copy()
    sucia[idx[12:18].ravel()] = 11            # seis columnas enteras, parejo

    limpia = reparte_por_geodesica(pos, caras, sucia)
    assert (limpia == verdad).mean() == (sucia == verdad).mean()


def test_el_nucleo_borra_las_islas_sueltas() -> None:
    """⚠️ **Un maxilar real trae decenas de islas por código**, y bastan para inflar un
    ancho medido hasta 41 mm. El núcleo las borra porque una isla es todo borde."""
    pos, caras, idx = _rejilla()
    etq = np.full(24 * 12, 11, np.int64)
    etq[idx[20:, :].ravel()] = 0
    etq[idx[22, 5]] = 12                      # una isla de un vértice
    nuc = nucleos(caras, etq)
    assert (nuc == 12).sum() == 0
    assert (nuc == 11).sum() > 0


def test_la_competencia_no_toca_la_encia() -> None:
    """Aquí se decide de qué DIENTE es cada vértice de diente, no dónde acaba el diente."""
    pos, caras, idx = _rejilla()
    etq = np.where(np.arange(24 * 12) // 12 < 12, 11, 12).astype(np.int64)
    etq[idx[:, 10:].ravel()] = 0
    salida = reparte_por_geodesica(pos, caras, etq)
    assert ((etq == 0) == (salida == 0)).all()


def test_el_umbral_de_ancho_sale_de_etiquetas_de_experto() -> None:
    """⚠️ **Contar coronas anchas no mide un defecto: las de experto también lo fallan.**

    El gate declaraba «11 de 14 coronas más anchas de lo admitido» — y las etiquetas de
    experto de Teeth3DS+ dan **86 de 111 (77 %)** con ese mismo recuento. Lo que separa es
    la magnitud: `|medido - tabla|` da p95 **1,92 mm** sobre 188 coronas de experto, y
    nuestro caso daba p50 2,38 y p90 7,35.
    """
    assert 1.5 <= EXCESO_ADMITIDO_MM <= 2.0
