from __future__ import annotations

import numpy as np
import pytest
from analysis_agents.testigo import rescata_coronas


def _rejilla(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Malla plana de n×n vértices a 1 mm: cada vértice interior lleva 1 mm² de área."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    pos = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)]
    caras = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            caras += [(a, b, d), (a, d, c)]
    return pos, np.array(caras)


def _bloque(pos: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    x, y = pos[:, 0], pos[:, 1]
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)


POS, CARAS = _rejilla()
PIEZA = _bloque(POS, 5, 15, 5, 15)           # 100 vértices, ~100 mm²


def _testigo(codigo: int = 11) -> np.ndarray:
    t = np.zeros(len(POS), dtype=np.int64)
    t[PIEZA] = codigo
    return t


def test_una_pieza_conservada_no_se_toca() -> None:
    principal = _testigo(21)
    principal[np.flatnonzero(PIEZA)[:5]] = 0          # conserva el 95 %
    salida, registro = rescata_coronas(POS, CARAS, principal, _testigo())
    assert np.array_equal(salida, principal) and registro == []


def test_una_corona_perdida_entera_recibe_el_codigo_del_testigo() -> None:
    principal = np.zeros(len(POS), dtype=np.int64)
    salida, registro = rescata_coronas(POS, CARAS, principal, _testigo(11))
    assert (salida[PIEZA] == 11).all() and (salida[~PIEZA] == 0).all()
    assert registro[0]["assigned_code"] == 11 and registro[0]["retained_before"] == 0.0


def test_si_el_principal_conserva_un_trozo_se_completa_con_su_codigo() -> None:
    principal = np.zeros(len(POS), dtype=np.int64)
    principal[_bloque(POS, 5, 8, 5, 15)] = 21          # 30 % de la pieza, todo dentro
    salida, registro = rescata_coronas(POS, CARAS, principal, _testigo(11))
    assert (salida[PIEZA] == 21).all()
    assert registro[0]["assigned_code"] == 21


def test_el_codigo_de_un_vecino_que_vive_fuera_no_se_extiende() -> None:
    principal = np.zeros(len(POS), dtype=np.int64)
    vecino = _bloque(POS, 12, 25, 5, 15)               # 3 columnas dentro, 10 fuera
    principal[vecino] = 22
    salida, registro = rescata_coronas(POS, CARAS, principal, _testigo(11))
    assert registro[0]["assigned_code"] == 11
    assert (salida[vecino] == 22).all()                # el vecino no cambia
    assert (salida[PIEZA & ~vecino] == 11).all()


def test_nunca_cambia_un_vertice_que_el_principal_llama_diente() -> None:
    rng = np.random.default_rng(0)
    principal = np.where(rng.random(len(POS)) < 0.3, rng.integers(11, 18, len(POS)), 0)
    testigo = np.where(rng.random(len(POS)) < 0.5, 11, 0)
    salida, _ = rescata_coronas(POS, CARAS, principal, testigo)
    diente = principal > 0
    assert np.array_equal(salida[diente], principal[diente])
    assert ((salida > 0) >= (principal > 0)).all()


def test_una_pieza_pequena_del_testigo_no_se_importa() -> None:
    testigo = np.zeros(len(POS), dtype=np.int64)
    testigo[_bloque(POS, 5, 10, 5, 10)] = 18           # ~25 mm² < 39
    principal = np.zeros(len(POS), dtype=np.int64)
    salida, registro = rescata_coronas(POS, CARAS, principal, testigo)
    assert registro == [] and not salida.any()


def test_rechaza_etiquetas_que_no_indexan_la_malla() -> None:
    with pytest.raises(ValueError):
        rescata_coronas(POS, CARAS, np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64))
