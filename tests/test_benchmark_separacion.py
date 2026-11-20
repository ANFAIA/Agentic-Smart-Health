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
