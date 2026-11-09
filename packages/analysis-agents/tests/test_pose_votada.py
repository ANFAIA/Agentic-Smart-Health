"""Votar entre segmentaciones del mismo escaneo. Ver `analysis_agents.pose_votada`."""

from __future__ import annotations

import numpy as np
import pytest
from analysis_agents.pose_votada import confianza, vota


def test_la_confianza_solo_mira_los_dientes() -> None:
    """La encía es fácil en cualquier pose: contarla diluiría la diferencia."""
    logp = np.log(np.array([[0.99, 0.01], [0.6, 0.4], [0.6, 0.4]]))
    etiquetas = np.array([0, 11, 11])
    assert confianza(logp, etiquetas) == pytest.approx(np.log(0.6))
    assert confianza(logp, np.zeros(3)) == float("-inf")


def test_votan_solo_las_mas_confiadas() -> None:
    buena = np.array([0, 11, 12, 13])
    otra_buena = np.array([0, 11, 12, 13])
    mala = np.array([0, 21, 22, 23])
    segs = {"a": buena, "b": otra_buena, "c": mala, "d": mala, "e": mala}
    conf = {"a": -0.30, "b": -0.31, "c": -0.32, "d": -0.50, "e": -0.60}
    resultado, registro = vota(segs, conf)
    assert registro["chosen"] == ["a", "b", "c"]
    assert np.array_equal(resultado, buena)


def test_una_pose_mala_y_confiada_queda_en_minoria() -> None:
    """Es la razón de votar en lugar de quedarse con la más confiada."""
    mala = np.array([0, 21, 22])
    buena = np.array([0, 11, 12])
    resultado, registro = vota({"m": mala, "b1": buena, "b2": buena},
                               {"m": -0.29, "b1": -0.30, "b2": -0.31})
    assert registro["chosen"][0] == "m"
    assert np.array_equal(resultado, buena)
    assert registro["agreement_with_best"] == pytest.approx(1 / 3)


def test_sin_mayoria_decide_la_mas_confiada() -> None:
    resultado, _ = vota({"a": np.array([11]), "b": np.array([12]), "c": np.array([13])},
                        {"a": -0.3, "b": -0.4, "c": -0.5})
    assert resultado.tolist() == [11]


def test_sin_segmentaciones_es_un_error() -> None:
    with pytest.raises(ValueError):
        vota({}, {})
