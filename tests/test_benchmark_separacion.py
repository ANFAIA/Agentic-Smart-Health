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


def test_r2_de_6_16_tolera_el_ruido_de_semilla_y_ni_un_paso_mas():
    ruido = {"contact_p50_mm": 0.01, "bleeding_teeth": 20, "ghosts": 13, "false_tooth": 0.011,
             "false_gum": 0.001, "correct": 6, "tooth_fdi": 0.001}
    c2 = _aceptacion(0.17, 1531, 10)
    dentro = _aceptacion(0.18, 1525, 2)                 # contacto +0,01 y −6 correctas: justo
    fuera = _aceptacion(0.17, 1524, 2)                  # −7 correctas: una de más
    r = bs.lee_6_16({"C2": c2, "D": c2, "CT": dentro}, ruido)
    assert r["R2"] and r["R1"]
    r = bs.lee_6_16({"C2": c2, "D": c2, "CT": fuera}, ruido)
    assert not r["R2"] and not r["R2_figures"]["correct"]["pass"]
    assert r["R2_figures"]["contact_p50_mm"]["pass"]


def test_el_cerrojo_de_reserva_2_es_comun_a_los_dos_benchmarks(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "SALIDA", tmp_path / "sep")
    monkeypatch.setattr(bs, "RESULTADO_RESERVA_2_IDENTIDAD", tmp_path / "id")
    (tmp_path / "sep").mkdir()
    (tmp_path / "id").mkdir()
    assert bs.reserva_2_gastada() is None
    (tmp_path / "id" / "resumen-reserva-2-v3.json").write_text("{}")
    assert bs.reserva_2_gastada() == tmp_path / "id" / "resumen-reserva-2-v3.json"


def _a(pasa_resto: bool = True, subida: float = 0.0, danadas: float = 0.0) -> dict:
    c = {k: {"pass": pasa_resto} for k in ("A1_margin", "A2_agreement", "A4_labelled_fraction",
                                           "A5_width", "A6_truth_untouched")}
    c["A3_overtrim"] = {"pass": True, "false_gum_rise": subida,
                        "newly_under_retained_fraction": danadas}
    return c


def test_la_holgura_de_6_17_es_la_mitad_de_a3_y_exige_el_resto():
    assert bs.con_holgura(_a(subida=0.005, danadas=0.005))
    assert not bs.con_holgura(_a(subida=0.0051))
    assert not bs.con_holgura(_a(danadas=0.0051))
    assert not bs.con_holgura(_a(pasa_resto=False))
    assert not bs.con_holgura({})


def test_6_17_elige_la_que_pasa_en_los_dos_con_menos_encia_como_diente():
    def salida(pasan: set[str], falso: dict[str, float]) -> dict:
        return {"variants": {v: {"slack_pass": v in pasan, "false_tooth": falso.get(v, 0.5),
                                 "newly_under_retained": 0} for v in bs.VARIANTES_MARGEN}}
    dev = salida({"M5", "M7", "M8"}, {})
    reserva = salida({"M5", "M7", "M9"}, {"M5": 0.20, "M7": 0.15, "M9": 0.10})
    e = bs.elige_margen(dev, reserva)
    assert e["passing"] == ["M5", "M7"] and e["chosen"] == "M7"
    assert bs.elige_margen(salida(set(), {}), reserva)["chosen"] is None


def test_el_cerrojo_de_reserva_2_tambien_lo_cierra_el_margen(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "SALIDA", tmp_path)
    monkeypatch.setattr(bs, "RESULTADO_RESERVA_2_IDENTIDAD", tmp_path / "nada")
    assert bs.reserva_2_gastada() is None
    (tmp_path / "margen-reserva-2.json").write_text("{}")
    assert bs.reserva_2_gastada() is not None


def test_la_desviacion_de_6_18_solo_acepta_m8_si_cumple_los_limites_completos():
    import pytest

    def salida(cumple: bool) -> dict:
        return {"variants": {v: {"criteria": {"all_pass": cumple}} for v in bs.VARIANTES_MARGEN}}
    e = bs.elige_por_desviacion(salida(True), salida(True), "M8")
    assert e["chosen"] == "M8" and e["deviation"] == "6.18"
    with pytest.raises(SystemExit):
        bs.elige_por_desviacion(salida(True), salida(True), "M7")
    with pytest.raises(SystemExit):
        bs.elige_por_desviacion(salida(True), salida(False), "M8")


def test_el_cerrojo_de_reserva_3_lo_cierra_el_primer_resultado_de_6_19(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "SALIDA", tmp_path)
    assert bs.reserva_3_gastada() is None
    (tmp_path / "seccion-6-19-reserva-2.json").write_text("{}")
    assert bs.reserva_3_gastada() is None
    (tmp_path / "seccion-6-19-reserva-3.json").write_text("{}")
    assert bs.reserva_3_gastada() == tmp_path / "seccion-6-19-reserva-3.json"


def test_la_puerta_de_6_19_exige_holgura_en_los_tres_gastados_con_el_mismo_c2(tmp_path,
                                                                          monkeypatch):
    import json

    import pytest

    monkeypatch.setattr(bs, "SALIDA", tmp_path)
    with pytest.raises(SystemExit):
        bs.puerta_6_19("abc")
    for s, ok in (("dev", True), ("reserva", True), ("reserva-2", False)):
        (tmp_path / f"seccion-6-19-{s}.json").write_text(json.dumps(
            {"checkpoints": {"C2": "abc"}, "M10": {"slack_pass": ok}}))
    assert bs.puerta_6_19("abc")["pass"] is False
    (tmp_path / "seccion-6-19-reserva-2.json").write_text(json.dumps(
        {"checkpoints": {"C2": "abc"}, "M10": {"slack_pass": True}}))
    assert bs.puerta_6_19("abc")["pass"] is True
    with pytest.raises(SystemExit):
        bs.puerta_6_19("otro")


def _fila_6_19(cos: float, volteado: bool, antes: dict, despues: dict, bien: tuple) -> dict:
    def modelo(ret: dict, correctas: int) -> dict:
        return {"voted": {"per_tooth": {t: {"retained": v} for t, v in ret.items()},
                          "identity": {"correct": correctas}}}
    return {"cos_canonical": -cos if volteado else cos, "cos_oriented": cos,
            "flipped": volteado,
            "models": {"C2": modelo(antes, bien[0]), "C2O": modelo(despues, bien[1])}}


def test_o_se_confirma_sin_perder_coronas_ni_numeracion_y_con_el_eje_bien():
    filas = [_fila_6_19(0.99, True, {"11": 0.1, "21": 0.2}, {"11": 1.0, "21": 0.99}, (0, 2)),
             _fila_6_19(0.98, False, {"31": 1.0}, {"31": 1.0}, (1, 1))]
    o = bs.criterios_6_19(filas)
    assert o["confirmed"] and o["benefit_measured"]
    assert o["flipped"] == 1 and o["inverted_before"] == 1
    assert "O4_crowns_vs_B" not in o                       # sin B no hay O4


def test_o_no_se_confirma_si_pierde_una_corona_o_deja_un_eje_al_reves():
    pierde = [_fila_6_19(0.99, True, {"11": 0.9}, {"11": 0.5}, (1, 1))]
    assert not bs.criterios_6_19(pierde)["O2_no_crown_lost"]["pass"]
    al_reves = [_fila_6_19(-0.99, False, {"11": 1.0}, {"11": 1.0}, (1, 1))]
    o = bs.criterios_6_19(al_reves)
    assert not o["O1_orientation"]["pass"] and not o["confirmed"]
    sin_vuelta = bs.criterios_6_19([_fila_6_19(0.99, False, {"11": 1.0}, {"11": 1.0}, (1, 1))])
    assert sin_vuelta["confirmed"] and not sin_vuelta["benefit_measured"]
