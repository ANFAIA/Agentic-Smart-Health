"""El banco de separación: la regla de elección en `dev` no puede elegir lo que no debe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_separacion.py"
_spec = importlib.util.spec_from_file_location("benchmark_separacion", _RUTA)
assert _spec and _spec.loader
bs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bs
_spec.loader.exec_module(bs)


def _resumen(contacto: float, sangran: int, k3: bool, todos: bool) -> dict:
    criterios = {f"K{i}": {"pass": todos} for i in (1, 2, 4, 5, 6)}
    criterios["K1_contact"] = {"C": contacto, "pass": todos}
    criterios["K2_bleeding"] = {"C": sangran, "pass": todos}
    criterios["K3_numbering"] = {"pass": k3}
    criterios["all_pass"] = todos
    return {"criteria": criterios}


def test_la_resolucion_gana_por_contacto_solo_entre_las_que_no_numeran_peor() -> None:
    r = {"a": _resumen(0.40, 100, k3=False, todos=False),
         "b": _resumen(0.55, 120, k3=True, todos=False),
         "c": _resumen(0.50, 130, k3=True, todos=False)}
    assert bs.elige_candidato(r, "resolucion")["chosen"] == "c"


def test_el_empate_en_contacto_lo_rompe_el_sangrado() -> None:
    r = {"a": _resumen(0.50, 130, k3=True, todos=True),
         "b": _resumen(0.50, 120, k3=True, todos=True)}
    assert bs.elige_candidato(r, "final")["chosen"] == "b"


def test_sin_ningun_candidato_que_cumpla_todo_no_se_elige_nada() -> None:
    r = {"a": _resumen(0.30, 10, k3=True, todos=False)}
    e = bs.elige_candidato(r, "final")
    assert e["chosen"] is None and "K1-K6" in e["reason"]


def test_los_criterios_estan_fijados_como_se_preregistraron() -> None:
    assert (bs.MEJORA_CONTACTO, bs.ARCADAS_QUE_MEJORAN, bs.MEJORA_SANGRADO) == (0.75, 0.70, 0.75)
    assert (bs.SUBIDA_FALSO_DIENTE, bs.SUBIDA_FALSA_ENCIA, bs.PIEZAS_DANADAS) == (
        0.01, 0.005, 0.005)
    assert bs.RETROCESO_FDI == 0.005


def _aceptacion(contacto: float, correctas: int, nuevas: int) -> dict:
    return {"criteria": {
        "K1_contact": {"B": 0.86, "C": contacto},
        "K2_bleeding": {"B": 789, "C": 100},
        "K3_numbering": {"correct_B": 1365, "correct_C": correctas, "ghost_B": 106, "ghost_C": 95},
        "K4_pose": {"gap_B": 42, "gap_C": 14},
        "K5_margin": {"false_tooth_B": 0.36, "false_tooth_C": 0.25, "false_gum_B": 0.065,
                      "false_gum_C": 0.019, "newly_under_retained": nuevas},
        "K6_tooth_fdi": {"B": 0.738, "C": 0.888},
        "all_pass": nuevas <= 8}}


def test_la_lectura_de_6_15_solo_atribuye_lo_que_supera_el_ruido_de_semilla():
    t = bs.lee_6_15({"C2": _aceptacion(0.17, 1531, 10), "C2-s1": _aceptacion(0.19, 1520, 6),
                     "D": _aceptacion(0.40, 1525, 12)})["figures"]
    # Ruido de contacto 0,02: B → D (−0,46) y D → C2 (−0,23) lo superan.
    assert t["contact_p50_mm"]["data_exceeds_noise"]
    assert t["contact_p50_mm"]["resolution_exceeds_noise"]
    # Ruido de correctas 11: D → C2 (+6) no se distingue; B → D (+160) sí.
    assert not t["correct"]["resolution_exceeds_noise"]
    assert t["correct"]["data_exceeds_noise"]
    # La rotura de corona se cuenta contra B, que vale 0 por definición.
    assert t["newly_under_retained"]["B"] == 0.0 and t["newly_under_retained"]["data_effect"] == 12
