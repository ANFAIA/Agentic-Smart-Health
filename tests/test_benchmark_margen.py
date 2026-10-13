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


# --------------------------------------------------------------------------- #
# Los criterios de aceptación: el veredicto lo da el script, no quien lo lee.

def _fila(margen: float, falsa_encia: float, conservada: float, verdad: float = 1.0,
          base_conservada: float = 1.0) -> dict:
    def metodo(m: float, g: float, r: float, acuerdo: float) -> dict:
        return {"margin": {"symmetric_p50_mm": m}, "binary_agreement": acuerdo,
                "fdi_agreement": acuerdo, "false_gum_rate": g,
                "labelled_fraction": {"paired_diff_area": m / 20},
                "width_error_mm": {"p50": m, "p90": m + 1},
                "per_tooth": {"11": {"retained": r}}}
    return {"methods": {"M1": metodo(2.4, 0.008, base_conservada, 0.80),
                        "M5": metodo(margen, falsa_encia, conservada, 0.90),
                        "E2": {"binary_agreement": verdad}}}


def test_un_metodo_que_mejora_sin_romper_cumple() -> None:
    c = bm.criterios([_fila(0.5, 0.012, 0.95) for _ in range(10)])
    assert c["all_pass"] and c["A3_overtrim"]["newly_under_retained"] == 0


def test_romper_piezas_suspende_aunque_todo_lo_demas_mejore() -> None:
    filas = [_fila(0.5, 0.012, 0.95) for _ in range(9)] + [_fila(0.5, 0.012, 0.60)]
    c = bm.criterios(filas)
    assert not c["A3_overtrim"]["pass"] and not c["all_pass"]
    assert c["A1_margin"]["pass"] and c["A2_agreement"]["pass"]


def test_una_pieza_que_ya_venia_rota_no_cuenta_como_danada_por_el_metodo() -> None:
    c = bm.criterios([_fila(0.5, 0.012, 0.60, base_conservada=0.60) for _ in range(10)])
    assert c["A3_overtrim"]["newly_under_retained"] == 0 and c["A3_overtrim"]["pass"]


def test_subir_la_falsa_encia_mas_de_un_punto_suspende() -> None:
    c = bm.criterios([_fila(0.5, 0.030, 0.95) for _ in range(10)])
    assert not c["A3_overtrim"]["pass"]


def test_danar_la_verdad_de_campo_suspende() -> None:
    c = bm.criterios([_fila(0.5, 0.012, 0.95, verdad=0.97) for _ in range(10)])
    assert not c["A6_truth_untouched"]["pass"] and not c["all_pass"]


def test_sin_candidato_no_hay_veredicto() -> None:
    fila = _fila(0.5, 0.012, 0.95)
    del fila["methods"]["M5"]
    assert bm.criterios([fila]) == {} and bm.veredicto_md({}) == []
