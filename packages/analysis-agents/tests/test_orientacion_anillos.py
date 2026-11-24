"""§6.19: el sentido oclusal leído del borde, y devolver a la corona la fila del pliegue."""

from __future__ import annotations

import numpy as np
import pytest
from analysis_agents import orientacion
from analysis_agents.frontera import devuelve_anillos
from analysis_agents.orientacion import marco_orientado, vertices_de_borde
from fusion_agents.marco import marco_canonico


def _arcada(nu: int = 60, nv: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Un casquete en U: medio tubo sobre una parábola, abierto por abajo (lado apical)."""
    u = np.linspace(-1.0, 1.0, nu)
    v = np.linspace(0.0, np.pi, nv)
    U, W = np.meshgrid(u, v, indexing="ij")
    x, y = 25.0 * U, -18.0 * U ** 2
    radio = 4.0 + 1.5 * np.cos(np.pi * U) ** 2                # muelas más anchas que incisivos
    pos = np.stack([x + radio * np.cos(W) * 0.3, y + radio * np.cos(W),
                    radio * np.sin(W) * 1.6], -1).reshape(-1, 3)
    idx = np.arange(nu * nv).reshape(nu, nv)
    tri = np.concatenate([
        np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3)])
    return pos, tri


def test_el_borde_de_un_casquete_son_sus_cuatro_lados() -> None:
    pos, tri = _arcada(10, 5)
    assert len(vertices_de_borde(tri)) == 2 * 10 + 2 * 5 - 4


def test_bien_orientada_da_el_mismo_marco_que_marco_canonico() -> None:
    pos, tri = _arcada()
    c, R, razon, volteado = marco_orientado(pos, tri)
    c0, R0, razon0 = marco_canonico(pos)
    assert not volteado
    assert np.array_equal(c, c0) and np.array_equal(R, R0) and razon == razon0
    assert R[2] @ [0, 0, 1.0] > 0.9


def test_si_el_ajuste_elige_mal_el_borde_le_da_la_vuelta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Se fuerza el fallo de `marco_arcada` para ver que el borde lo corrige."""
    pos, tri = _arcada()
    original = orientacion.marco_arcada

    def al_reves(V: np.ndarray):  # type: ignore[no-untyped-def]
        centro, ejes, _, razon = original(V)
        ejes = ejes.copy()
        ejes[1] *= -1
        ejes[2] *= -1
        return centro, ejes, (V - centro) @ ejes.T, razon

    monkeypatch.setattr(orientacion, "marco_arcada", al_reves)
    _, R, _, volteado = marco_orientado(pos, tri)
    assert volteado
    assert R[2] @ [0, 0, 1.0] > 0.9
    assert np.linalg.det(R) > 0


def test_girada_media_vuelta_el_oclusal_sigue_hacia_las_coronas() -> None:
    pos, tri = _arcada()
    girada = pos * [1.0, -1.0, -1.0]                          # 180° alrededor de x, sin espejo
    _, R, _, _ = marco_orientado(girada, tri)
    assert R[2] @ [0, 0, -1.0] > 0.9


def _tira(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Una tira de `n` columnas por 2 filas: vértice i abajo, i + n arriba."""
    idx = np.arange(2 * n).reshape(2, n)
    tri = np.concatenate([np.stack([idx[0, :-1], idx[0, 1:], idx[1, :-1]], -1),
                          np.stack([idx[0, 1:], idx[1, 1:], idx[1, :-1]], -1)])
    return idx, tri


def test_devuelve_solo_la_fila_que_toca_la_corona() -> None:
    _, tri = _tira()
    antes = np.zeros(24, dtype=np.int64)
    antes[2:9] = 11                                           # una pieza en la fila de abajo
    despues = antes.copy()
    despues[2:5] = 0                                          # el corte cede tres vértices
    salida = devuelve_anillos(tri, antes, despues, anillos=1)
    assert salida[4] == 11                                    # el que toca la corona vuelve
    assert (salida[2:4] == 0).all()                           # los demás siguen cedidos
    assert (salida[[0, 1, *range(9, 24)]] == 0).all()         # la encía de antes no cambia


def test_mas_anillos_devuelven_mas_pero_nunca_lo_que_era_encia() -> None:
    _, tri = _tira()
    antes = np.zeros(24, dtype=np.int64)
    antes[2:9] = 11
    despues = antes.copy()
    despues[2:5] = 0
    salida = devuelve_anillos(tri, antes, despues, anillos=5)
    assert np.array_equal(salida, antes)


def test_rechaza_etiquetas_de_tamanos_distintos() -> None:
    _, tri = _tira()
    with pytest.raises(ValueError):
        devuelve_anillos(tri, np.zeros(24, dtype=np.int64), np.zeros(5, dtype=np.int64))
