"""Las medidas del banco de pruebas: que separen desborde de sobre-recorte.

El recorte al cuello subía el acuerdo global mientras estropeaba el caso clínico. Estas
pruebas fijan que las dos direcciones del error se ven por separado y con su signo.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis_agents.frontera import margen_gingival
from analysis_agents.margen_metricas import (
    agrega,
    area_por_vertice,
    compara,
    metricas_de_arcada,
)

PASO = 0.5


def _valle(tooth_desde: float, nx: int = 64):
    """Loseta con un pliegue en `y = 0`; es diente lo que queda en `y >= tooth_desde`."""
    ys = np.arange(-10.0, 10.0 + PASO, PASO)
    xs = np.arange(nx) * PASO
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X, Y, 0.4 * np.abs(Y)], axis=-1).reshape(-1, 3)
    ny = len(ys)
    tri = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = i * ny + j, i * ny + j + 1
            c, d = (i + 1) * ny + j, (i + 1) * ny + j + 1
            tri += [(a, c, b), (b, c, d)]
    etq = np.where(pos[:, 1] >= tooth_desde, 11, 0).astype(np.int64)
    return pos, np.asarray(tri, dtype=np.int64), etq


def test_etiquetas_identicas_no_tienen_error() -> None:
    pos, tri, experto = _valle(0.0)
    m = metricas_de_arcada(pos, tri, experto, experto)
    assert m["binary_agreement"] == 1.0 and m["fdi_agreement"] == 1.0
    assert m["false_tooth_rate"] == 0.0 and m["false_gum_rate"] == 0.0
    assert m["margin"]["symmetric_p50_mm"] == 0.0
    assert m["labelled_fraction"]["paired_diff_area"] == 0.0
    assert m["teeth"]["under_retained"] == 0


def test_el_desborde_es_falso_diente_con_signo_positivo() -> None:
    pos, tri, experto = _valle(0.0)
    _, _, pred = _valle(-3.0)                 # la etiqueta baja 3 mm sobre la encía
    m = metricas_de_arcada(pos, tri, pred, experto)
    assert m["false_tooth_rate"] > 0.0 and m["false_gum_rate"] == 0.0
    assert m["margin"]["signed_mean_mm"] > 0.0
    assert m["margin"]["symmetric_p50_mm"] == pytest.approx(3.0, abs=PASO)
    assert m["labelled_fraction"]["paired_diff_area"] > 0.0
    assert m["teeth"]["under_retained"] == 0   # desbordarse no se come la corona


def test_el_sobre_recorte_es_falsa_encia_con_signo_negativo() -> None:
    pos, tri, experto = _valle(0.0)
    _, _, pred = _valle(3.0)                  # el recorte se come 3 mm de corona
    m = metricas_de_arcada(pos, tri, pred, experto)
    assert m["false_gum_rate"] > 0.0 and m["false_tooth_rate"] == 0.0
    assert m["margin"]["signed_mean_mm"] < 0.0
    assert m["labelled_fraction"]["paired_diff_area"] < 0.0
    assert m["teeth"]["under_retained"] == 1
    assert m["per_tooth"]["11"]["retained"] < 0.85


def test_los_dos_errores_dan_el_mismo_acuerdo_global() -> None:
    """Por eso el acuerdo solo no decide: no distingue en qué dirección se falla."""
    pos, tri, experto = _valle(0.0)
    desborde = metricas_de_arcada(pos, tri, _valle(-3.0)[2], experto)
    recorte = metricas_de_arcada(pos, tri, _valle(3.0)[2], experto)
    assert desborde["binary_agreement"] == pytest.approx(recorte["binary_agreement"])


def test_el_contacto_entre_dientes_no_es_margen() -> None:
    pos, tri, etq = _valle(0.0)
    dos = etq.copy()
    dos[(etq > 0) & (pos[:, 0] > 16.0)] = 12   # dos piezas contiguas
    margen = margen_gingival(tri, dos)
    assert margen.sum() == margen_gingival(tri, etq).sum()
    assert not margen[(pos[:, 1] > 2.0)].any()


def test_longitudes_distintas_fallan() -> None:
    pos, tri, etq = _valle(0.0)
    with pytest.raises(ValueError):
        metricas_de_arcada(pos, tri, etq[:-1], etq)


def test_el_area_por_vertice_suma_la_de_la_malla() -> None:
    pos, tri, _ = _valle(0.0)
    a, b, c = (pos[tri[:, k]] for k in range(3))
    total = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert area_por_vertice(pos, tri).sum() == pytest.approx(total)


def test_agregar_no_promedia_lo_que_no_se_midio() -> None:
    pos, tri, experto = _valle(0.0)
    con_margen = metricas_de_arcada(pos, tri, experto, experto)
    sin_prediccion = metricas_de_arcada(pos, tri, np.zeros_like(experto), experto)
    assert sin_prediccion["margin"]["symmetric_p50_mm"] is None
    resumen = agrega([con_margen, sin_prediccion])
    assert resumen["arches"] == 2
    assert resumen["margin_p50_mm"]["n"] == 1
    assert resumen["binary_agreement"]["n"] == 2
    assert resumen["teeth_under_retained"] == 1


def test_la_comparacion_es_pareada() -> None:
    pos, tri, experto = _valle(0.0)
    base = [metricas_de_arcada(pos, tri, _valle(-3.0)[2], experto),
            metricas_de_arcada(pos, tri, _valle(-2.0)[2], experto)]
    candidato = [metricas_de_arcada(pos, tri, _valle(-1.0)[2], experto),
                 metricas_de_arcada(pos, tri, _valle(-2.0)[2], experto)]
    c = compara(candidato, base, "margin_p50_mm", ("margin", "symmetric_p50_mm"),
                menor_es_mejor=True)
    assert c["n"] == 2 and c["improved_fraction"] == 0.5
    assert c["mean_diff"] < 0.0 and c["worst_diff"] == 0.0


def _arcada(*piezas: int, n: int = 20) -> np.ndarray:
    """Etiquetas de experto: `n` vértices de encía y `n` por cada pieza, en orden."""
    return np.concatenate([np.zeros(n, dtype=np.int64)]
                          + [np.full(n, p, dtype=np.int64) for p in piezas])


def test_identidad_de_una_arcada_bien_numerada() -> None:
    from analysis_agents.margen_metricas import identidad_de_piezas

    experto = _arcada(13, 12, 11, 21)
    r = identidad_de_piezas(experto, experto)
    assert r["correct"] == 4 and r["perfect"]
    assert r["shifted"] == r["ghost"] == r["split"] == r["missing"] == 0


def test_un_desplazamiento_a_la_vecina_se_distingue_de_otro_error() -> None:
    """Cruzar la línea media también es ser vecina: 11 y 21 distan un puesto."""
    from analysis_agents.margen_metricas import identidad_de_piezas

    experto = _arcada(13, 12, 11, 21)
    pred = experto.copy()
    pred[experto == 11] = 21
    pred[experto == 21] = 22
    pred[experto == 13] = 15
    r = identidad_de_piezas(pred, experto)
    assert r["correct"] == 1 and r["shifted"] == 3 and r["neighbour_shift"] == 2
    assert r["per_label"]["21"] == {"status": "neighbour_shift", "expert": 11}
    assert r["per_label"]["15"]["status"] == "shifted"
    assert not r["perfect"]


def test_un_fantasma_sobre_la_encia_y_una_pieza_ausente() -> None:
    from analysis_agents.margen_metricas import identidad_de_piezas

    experto = _arcada(16, 15, n=20)
    pred = experto.copy()
    pred[:15] = 17            # 15 de los 20 vértices de encía: un 17 que no existe
    pred[experto == 15] = 0   # el 15 no aparece
    r = identidad_de_piezas(pred, experto)
    assert r["ghost"] == 1 and r["per_label"]["17"]["status"] == "ghost"
    assert r["missing"] == 1 and r["missing_teeth"] == [15]


def test_un_fragmento_con_otro_codigo_cuenta_como_partido() -> None:
    from analysis_agents.margen_metricas import identidad_de_piezas

    experto = _arcada(26, 25)
    pred = experto.copy()
    pred[np.flatnonzero(experto == 26)[:5]] = 27
    r = identidad_de_piezas(pred, experto)
    assert r["split"] == 1 and r["shifted"] == 1 and r["correct"] == 2


def test_posicion_en_arcada() -> None:
    from analysis_agents.margen_metricas import posicion_en_arcada

    assert posicion_en_arcada(18) == ("upper", 0)
    assert posicion_en_arcada(11) == ("upper", 7) and posicion_en_arcada(21) == ("upper", 8)
    assert posicion_en_arcada(38) == ("lower", 15) and posicion_en_arcada(41) == ("lower", 7)
