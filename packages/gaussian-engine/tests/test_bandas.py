"""La siembra por bandas de HU: partición disjunta, σ normalizada por capa, origin común."""

from __future__ import annotations

import numpy as np
import pytest
from gaussian_engine.bandas import ORDEN, PARTICION_HU, siembra_por_banda


def _volumen_prueba() -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """Un corte de 6 vóxeles con HU conocidos en cada tramo, más aire."""
    vol = np.full((1, 1, 6), -1000.0, dtype=np.float32)
    vol[0, 0, 0] = 100.0    # aire/tejido blando, por debajo del umbral de paciente
    vol[0, 0, 1] = 500.0    # densidad-baja
    vol[0, 0, 2] = 1000.0   # densidad-media
    vol[0, 0, 3] = 1500.0   # densidad-alta
    vol[0, 0, 4] = 2500.0   # densidad-muy-alta
    vol[0, 0, 5] = -500.0   # aire
    return vol, (0.3, 0.3, 0.3), np.array([0.0])


def test_particion_disjunta_y_cubre_el_paciente():
    vol, spacing, z = _volumen_prueba()
    capas = siembra_por_banda(vol, spacing, z)
    assert [c.banda for c in capas] == ORDEN, "las 4 bandas tienen que salir, en orden"
    assert all(c.n_primitivas == 1 for c in capas)
    # La unión es exactamente los 4 vóxeles de paciente (500/1000/1500/2500), disjuntos.
    assert sorted(int(c.hu[0]) for c in capas) == [500, 1000, 1500, 2500]


def test_origin_compartido():
    vol, spacing, z = _volumen_prueba()
    capas = siembra_por_banda(vol, spacing, z)
    # Un único `origin` para todas: si cada capa se centrara por su cuenta, al sumar
    # en mundo quedarían desalineadas.
    assert len({tuple(c.arrays["origin"]) for c in capas}) == 1


def test_normalizacion_de_sigma_por_capa():
    vol, spacing, z = _volumen_prueba()
    capas = {c.banda: c for c in siembra_por_banda(vol, spacing, z)}
    muy_alta = capas["densidad-muy-alta"]
    (p_lo, p_hi), (n_lo, n_hi) = PARTICION_HU["densidad-muy-alta"]
    assert p_lo == 2000.0 and p_hi == np.inf
    assert muy_alta.arrays["density"][0] == pytest.approx((2500.0 - n_lo) / (n_hi - n_lo))
    assert tuple(muy_alta.arrays["hu_range"]) == (n_lo, n_hi)


def test_banda_vacia_se_salta():
    # Todo el paciente en un único tramo: el resto de bandas no existen y no reventan.
    vol = np.full((1, 1, 4), 800.0, dtype=np.float32)  # solo densidad-media
    vol[0, 0, 0] = -1000.0
    capas = siembra_por_banda(vol, (0.3, 0.3, 0.3), np.array([0.0]))
    assert [c.banda for c in capas] == ["densidad-media"]


def test_sin_paciente_es_un_error():
    vol = np.full((1, 1, 4), -1000.0, dtype=np.float32)
    with pytest.raises(ValueError, match="umbral"):
        siembra_por_banda(vol, (0.3, 0.3, 0.3), np.array([0.0]))
