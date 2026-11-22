"""El aumento de desborde del verificador v2 (§6.12): crece sobre encía, y solo hasta donde debe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("entrena_pieza",
                                               _RAIZ / "scripts" / "entrena_pieza.py")
assert _spec and _spec.loader
ep = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ep
_spec.loader.exec_module(ep)

from analysis_agents.identidad import distancia_con_signo  # noqa: E402


def _tira() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Una tira de 0 a 30 mm: la pieza es x ≤ 5, la encía x > 5 salvo un diente en 8-10."""
    x = np.arange(0, 30, 0.25)
    pos = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    encia = (x > 5) & ~((x >= 8) & (x <= 10))
    return pos, encia, np.flatnonzero(x <= 5)


def test_el_desborde_solo_crece_sobre_encia_y_hasta_el_maximo() -> None:
    pos, encia, pieza = _tira()
    idx, d = distancia_con_signo(pos, pieza, ep.RADIO_REGION)
    rng = np.random.default_rng(0)
    for _ in range(30):
        i2, d2 = ep.desborda(pos, idx, d, encia, rng)
        crecida = set(i2[d2 <= 0])
        assert set(pieza) <= crecida
        nuevas = np.array(sorted(crecida - set(pieza)), dtype=np.int64)
        assert encia[nuevas].all()  # nunca el diente vecino
        if len(nuevas):
            assert pos[nuevas, 0].max() - 5 <= ep.DESBORDE_MAXIMO_MM + 1e-9
        # el margen se recalcula alrededor de la pieza crecida
        fuera = i2[d2 > 0]
        assert len(fuera) == 0 or d2[d2 > 0].max() <= ep.RADIO_REGION + 1e-9

