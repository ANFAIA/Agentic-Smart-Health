"""El banco de pruebas del margen: la partición no puede cambiar sin que se note.

Si la partición deja de ser la del entrenamiento, pacientes que el modelo ya vio entran en
la evaluación y todas las cifras suben sin que nada avise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_margen.py"
_spec = importlib.util.spec_from_file_location("benchmark_margen", _RUTA)
assert _spec and _spec.loader
bm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bm
_spec.loader.exec_module(bm)

IDS = [f"P{k:03d}" for k in range(300)]


def test_la_particion_es_la_del_entrenamiento() -> None:
    """El mismo barajado que la celda 5 del notebook: ordenados, semilla 42, primer 20 %."""
    esperado = list(IDS)
    np.random.default_rng(42).shuffle(esperado)
    p = bm.particion(IDS)
    assert p["dev"] + p["eval"] == esperado[:60]
    assert p["train"] == esperado[60:]


def test_los_conjuntos_no_se_pisan() -> None:
    p = bm.particion(IDS)
    assert (len(p["train"]), len(p["dev"]), len(p["eval"])) == (240, 20, 40)
    assert not set(p["dev"]) & set(p["eval"])
    assert not (set(p["dev"]) | set(p["eval"])) & set(p["train"])
    assert set(p["fuga"]) <= set(p["train"])


def test_la_particion_no_depende_del_orden_de_entrada_ya_ordenada() -> None:
    assert bm.particion(sorted(IDS)) == bm.particion(IDS)
    assert bm.huella(IDS) == bm.huella(list(IDS))
    assert bm.huella(IDS) != bm.huella(IDS[::-1])


def test_la_limpieza_conserva_forma_y_no_inventa_codigos() -> None:
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 20, (600, 3))
    etq = np.where(pos[:, 0] > 10, 11, 0).astype(np.int64)
    limpia = bm.limpieza_actual(pos, etq)
    assert limpia.shape == etq.shape
    assert set(np.unique(limpia)) <= {0, 11}


def test_el_informe_declara_su_alcance() -> None:
    resumen = {"binary_agreement": {"mean": 0.9}, "teeth_under_retained": 1,
               "teeth_expert": 14}
    texto = bm.informe_md({"split": "dev", "arches": 2, "patients": 1, "protocol": "1",
                           "summary": {"M0": resumen}})
    assert "No miden los casos clínicos propios" in texto
    assert "| M0 | modelo, sin tocar | 0.900 |" in texto and "1 de 14" in texto
