"""La apariencia reutilizada se renumera con las etiquetas del escáner de esta ejecución."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "restaura_apariencia", _RAIZ / "scripts" / "restaura_apariencia.py")
assert _spec is not None and _spec.loader is not None
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def test_reetiqueta_sigue_a_las_etiquetas_nuevas_y_cuenta_los_cambios() -> None:
    rng = np.random.default_rng(0)
    # Dos coronas en mm, separadas 20 mm, y una gaussiana lejos de todo.
    a = rng.normal(scale=0.5, size=(40, 3))
    b = rng.normal(scale=0.5, size=(40, 3)) + [20.0, 0.0, 0.0]
    vertices = np.vstack([a, b])
    etiquetas = np.array([11] * 40 + [21] * 40)
    escala, offset = 0.05, np.array([1.0, 2.0, 3.0])
    centros_mm = np.vstack([a[:5], b[:5], [[10.0, 30.0, 0.0]]])
    params = {
        "means": (centros_mm - offset) * escala,
        "scan_scale": np.asarray(escala),
        "scan_offset": offset,
        # Numeración vieja: todo 12, como si otro modelo lo hubiera numerado.
        "region_id": np.full(11, 12, dtype=np.int16),
    }
    cambian, cruzan = ra.reetiqueta(params, vertices, etiquetas)

    assert params["region_id"].tolist() == [11] * 5 + [21] * 5 + [0]
    assert cambian == 11
    # La lejana pasa de diente (12) a encía (0): cruza la frontera.
    assert cruzan == 1


def test_cambiar_de_pieza_no_cruza_la_frontera_diente_encia() -> None:
    # El color se entrenó condicionado a «diente o encía»: pasar de 12 a 11 no lo
    # contradice, pasar de diente a encía sí. Sólo lo segundo hace rechazar la reutilización.
    rng = np.random.default_rng(1)
    vertices = rng.normal(scale=0.5, size=(40, 3))
    etiquetas = np.full(40, 11)
    escala, offset = 0.05, np.zeros(3)
    params = {
        "means": (vertices[:6] - offset) * escala,
        "scan_scale": np.asarray(escala),
        "scan_offset": offset,
        "region_id": np.full(6, 12, dtype=np.int16),
    }
    cambian, cruzan = ra.reetiqueta(params, vertices, etiquetas)

    assert cambian == 6
    assert cruzan == 0
