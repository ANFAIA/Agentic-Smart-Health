"""La incertidumbre declarada del registro por cortes (`scripts/incertidumbre_arranque.py`).

Todo sintético: ningún dato de paciente, y ninguna configuración de la batería del protocolo
se registra aquí (eso se hace una vez, desde el commit del pre-registro): de la batería solo se
construyen los campos (`campo_de`, sin escaneo), para comprobar su grupo de P2. El fantasma
de estos tests es una variante más gruesa de la losa de F1, inyectada con `monkeypatch`. Se
prueban las negativas de `main` antes de tocar ningún dato, el ciclo del `run-record.json`, que una
configuración que lanza se escribe y no se pierde, y que la preparación del script es la del
pipeline bit a bit.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _carga(nombre: str, ruta: Path):
    if nombre in sys.modules:
        return sys.modules[nombre]
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


ia = _carga("incertidumbre_arranque", RAIZ / "scripts" / "incertidumbre_arranque.py")
rn = ia.rn
fa = ia.fantasma_arcada()

HEAD = "c" * 40
# Una losa más gruesa que la de F1 (rejilla de 0,5 mm, cortes cada 0,5 con un hueco de 1,8):
# barata, dispersa en la banda y distinta de toda configuración de la batería.
LOSA_TEST = {**ia.LOSA, "paso": 0.5, "z_desde": -2.0, "z_hasta": 8.0, "paso_z": 0.5,
             "quita": (2.1, 3.9)}
CONFIG_TEST = {"id": "T1", "campo": "losa", "escaneo": {}, "plano": None, "gradiente": False}


# --------------------------------------------------------------------------- #
# La batería, sin construirla
# --------------------------------------------------------------------------- #
def test_la_bateria_es_la_del_protocolo_y_sus_argumentos_existen() -> None:
    ids = [c["id"] for c in ia.BATERIA]
    assert ids == [f"F{i}" for i in range(1, 13)]
    campo = inspect.signature(fa.campo_cbct)
    escaneo = inspect.signature(fa.escaneo_superior)
    for c in ia.BATERIA:
        escaneo.bind(**c["escaneo"])
        if c["campo"] == "losa":
            assert c["plano"] is None
            continue
        assert c["plano"] == 0.0
        argumentos = dict(c["campo"])
        argumentos.pop("inclinacion", None)
        campo.bind(**argumentos)
        # Todas son series con un hueco: lo que `por_cortes` acepta como dispersa.
        assert "paso_z" in c["campo"] and "quita" in c["campo"]
    # La batería entra en la huella: cambiarla cambia el hash de las constantes del script.
    assert ia.huella_constantes()["script_values"]["BATERIA"][2]["campo"]["raices"] == "realistas"


def test_el_grupo_de_p2_sale_de_la_bateria() -> None:
    assert all(isinstance(c["gradiente"], bool) for c in ia.BATERIA)
    assert ia.GRUPO_P2 == ("F4", "F5", "F6", "F7", "F8", "F10", "F12")
    assert ia.P2_MINIMO == 6                                     # 80 % de 7, hacia arriba
    assert [ia.minimo_cubiertas(n) for n in (1, 4, 5, 7, 10, 12)] == [1, 4, 4, 6, 8, 10]


def test_el_gradiente_declarado_es_el_de_los_campos() -> None:
    """§2.1: sin gradiente, ningún vóxel de la banda entre 1100 y 1900 HU, y los tres umbrales
    dibujan el mismo contorno. Solo se construyen los campos (`campo_de`): ni escaneo ni
    registro."""
    from fusion_agents.por_cortes import banda

    for c in ia.BATERIA:
        campo, hu, _ = ia.campo_de(c)
        if c["campo"] == "losa":
            # F1: todo el campo a un solo valor, así que no depende de dónde caiga el plano.
            assert not c["gradiente"] and set(np.unique(hu)) == {2200.0}
            continue
        n = ia.voxeles_entre_umbrales(campo, hu, banda("upper", c["plano"]))
        assert (n > 0) == c["gradiente"], (c["id"], n)


def test_los_giros_son_los_de_la_medida_3_bit_a_bit() -> None:
    from fusion_agents.por_cortes import GIRO_ARRANQUE_GRADOS, giros_de_arranque

    assert GIRO_ARRANQUE_GRADOS in rn.ANGULOS_K1_GRADOS
    esperados = [rn._gira(rn.eje_anatomico(d), GIRO_ARRANQUE_GRADOS) for d in rn.DIRECCIONES]
    for a, b in zip(giros_de_arranque(), esperados, strict=True):
        assert a.tobytes() == b.tobytes()


def test_incertidumbre_combina_las_dos_dispersiones() -> None:
    informe = {"threshold_spread_mm": {"median": 0.3, "max": 1.2, "vertical": 0.0},
               "start_spread_mm": {"median": 0.4, "max": 0.5, "vertical": 0.1}}
    u = ia.incertidumbre(informe)
    assert u["U_mm"] == pytest.approx(0.5) and u["U_max_mm"] == pytest.approx(1.3)
    assert u["U_vertical_mm"] == pytest.approx(0.1) and u["missing"] == []
    solo = ia.incertidumbre({"start_spread_mm": informe["start_spread_mm"]})
    assert solo["U_mm"] == pytest.approx(0.4) and solo["missing"] == ["threshold_spread_mm"]


def _fila(i: str, cubre: bool, gradiente: bool = True, p1: bool = True,
          t: tuple = (1.0, 3.0)) -> dict:
    return {"config": {"id": i, "gradiente": gradiente}, "registered": True, "covered": cubre,
            "covered_by_start_alone": cubre, "covered_max": True, "covered_vertical": False,
            "converged": True, "gradient_declaration_holds": True,
            "P1": {"pass": p1, "max_mm": 0.0 if p1 else 0.2},
            "seconds_without_starts": t[0], "seconds_with_starts": t[1],
            "error": {"median_mm": 0.3},
            "uncertainty": {"threshold_spread_mm": {"median": 0.0},
                            "start_spread_mm": {"median": 0.2}},
            "preparation_report": {"start_spread_mm": {
                "median": 0.1, "max": 0.2, "vertical": 0.0, "not_converged": 1}}}


def _lanzada(i: str, gradiente: bool = True) -> dict:
    return {"config": {"id": i, "gradiente": gradiente}, "registered": False,
            "error_type": "SerieDensa", "message": "sin huecos"}


def test_p2_se_lee_solo_en_las_configuraciones_con_gradiente() -> None:
    # 7 con gradiente (6 cubiertas) y 5 sin gradiente, ninguna cubierta: P2 pasa con 6 de 7.
    filas = ([_fila(f"G{i}", cubre=i <= 6) for i in range(1, 8)]
             + [_fila(f"S{i}", cubre=False, gradiente=False) for i in range(1, 6)])
    c = ia.criterios(filas)
    assert c["P2"]["pass"] and c["P2"]["n"] == 7 and c["P2"]["minimum_covered"] == 6
    assert c["P2"]["covered"] == 6 and c["P2"]["ids"] == [f"G{i}" for i in range(1, 8)]
    sin = c["no_gradient_descriptive"]
    assert sin["criterion"] is None and sin["n"] == 5 and sin["covered"] == 0
    assert sin["rows"][0] == {"id": "S1", "registered": True, "E_mm": 0.3,
                              "U_threshold_mm": 0.0, "U_start_mm": 0.2,
                              "covered_by_start_alone": False, "covered": False}
    # 5 de 7 no basta; las sin gradiente no cuentan aunque cubran.
    filas = ([_fila(f"G{i}", cubre=i <= 5) for i in range(1, 8)]
             + [_fila(f"S{i}", cubre=True, gradiente=False) for i in range(1, 6)])
    c = ia.criterios(filas)
    assert not c["P2"]["pass"] and c["P2"]["covered"] == 5
    assert c["phantom_part"]["decides_wording_only"] == {
        "P2": False, "threshold_line_must_say_precision": True}
    # Una con gradiente que lanza cuenta como no cubierta dentro de su grupo.
    filas = ([_fila(f"G{i}", cubre=True) for i in range(1, 7)] + [_lanzada("G7")]
             + [_lanzada("S1", gradiente=False)])
    c = ia.criterios(filas)
    assert c["P2"]["pass"] and c["P2"]["n"] == 7 and c["P2"]["covered"] == 6
    assert c["P2"]["not_registered_count_as_uncovered"] == ["G7"]
    assert c["no_gradient_descriptive"]["rows"] == [
        {"id": "S1", "registered": False, "covered": False}]
    assert [x["id"] for x in c["not_registered"]] == ["G7", "S1"] and c["P1"]["n"] == 6
    c = ia.criterios([*filas[:5], _lanzada("G6"), _lanzada("G7")])
    assert not c["P2"]["pass"] and c["P2"]["covered"] == 5


def test_criterios_del_fantasma_p1_p4_y_tiempos_descriptivos() -> None:
    filas = [_fila(f"F{i}", cubre=True) for i in range(1, 5)]
    c = ia.criterios(filas)
    assert c["P1"]["pass"] and c["P4"]["pass"] and "P3" not in c
    assert c["P4"]["not_converged_starts"] == 4
    assert c["phantom_part"]["decides_shipping"] == {"P1_phantom": True, "P4_phantom": True}
    # La razón de tiempos del fantasma se informa por la suma y por configuración, sin criterio.
    c = ia.criterios([_fila("F1", True, t=(1.0, 6.0)), _fila("F2", True)])
    r = c["phantom_time_ratio_descriptive"]
    assert r["criterion"] is None and r["factor"] == pytest.approx(4.5)
    assert r["per_config_factor"] == {"F1": 6.0, "F2": 3.0}
    assert "pass" not in r and c["phantom_part"]["decides_shipping"]["P1_phantom"]
    c = ia.criterios([_fila("F1", True), _fila("F2", True, p1=False)])
    assert not c["P1"]["pass"] and c["P1"]["failing"] == ["F2"]
    assert not c["phantom_part"]["decides_shipping"]["P1_phantom"]


def test_p3_es_el_tiempo_que_anade_la_medida_en_r1() -> None:
    p3 = ia.p3_de([20.0, 18.0], [70.0, 75.0])
    assert p3["pass"] and p3["extra_seconds"] == pytest.approx(52.0)
    assert p3["min_seconds_without"] == 18.0 and p3["min_seconds_with"] == 70.0
    assert p3["limit_s"] == ia.P3_LIMITE_S == 120.0
    assert ia.p3_de([18.0, 18.0], [138.0, 150.0])["pass"]           # justo en el límite
    assert not ia.p3_de([18.0, 19.0], [138.5, 140.0])["pass"]


def test_cronometra_calienta_y_sigue_el_orden_fijo(monkeypatch) -> None:
    # Un reloj falso que solo avanza lo que dura cada llamada: la base anatómica tarda 100 s,
    # el calentamiento 7 s, y las cuatro cronometradas 2, 5, 1,5 y 4,5 s. Si la base o el
    # calentamiento entraran en el cronómetro, los tiempos lo dirían.
    reloj = [0.0]
    monkeypatch.setattr(ia.time, "perf_counter", lambda: reloj[0])
    bases: list[int] = []

    def entrada(V, etq):
        bases.append(1)
        reloj[0] += 100.0
        return "ent"

    llamadas: list[bool] = []
    duraciones = iter([7.0, 2.0, 5.0, 1.5, 4.5])

    def prepara_desde(campo, hu, ent, *, mide_arranque, plano=None):
        assert ent == "ent" and plano is None
        llamadas.append(mide_arranque)
        reloj[0] += next(duraciones)
        return ("con" if mide_arranque else "sin", len(llamadas))

    monkeypatch.setattr(ia, "entrada", entrada)
    monkeypatch.setattr(ia, "prepara_desde", prepara_desde)
    datos = rn.Datos(campo={}, hu=np.zeros(0), V=np.zeros((0, 3)), etq=np.zeros(0),
                     T_ref=np.eye(4), entradas={})
    p3, sin = ia.cronometra(datos)
    assert bases == [1]                                            # la base, una vez
    assert llamadas == [False, False, True, False, True]          # calentamiento + ORDEN_P3
    assert p3["order"] == ["without", "with", "without", "with"]
    assert p3["seconds_without"] == [2.0, 1.5] and p3["seconds_with"] == [5.0, 4.5]
    assert p3["min_seconds_without"] == 1.5 and p3["min_seconds_with"] == 4.5
    assert p3["extra_seconds"] == pytest.approx(3.0) and p3["pass"]
    assert sin == ("sin", 4)                                       # el último sin la medida


def _partes(p1f=True, p4f=True, p2=True, r1=(True, True)) -> tuple[dict, dict | None]:
    fantasma = {"phantom_part": {
        "decides_shipping": {"P1_phantom": p1f, "P4_phantom": p4f},
        "decides_wording_only": {"P2": p2, "threshold_line_must_say_precision": not p2}}}
    if r1 is None:
        return fantasma, None
    return fantasma, {"r1_part": {"decides_shipping": {"P1_r1": r1[0], "P3_r1": r1[1]}}}


@pytest.mark.parametrize("partes, se_queda, fallan, sin_evaluar", [
    (dict(), True, [], []),
    (dict(p2=False), True, [], []),                                  # P2 no decide
    (dict(p1f=False), False, ["P1_phantom"], []),
    (dict(p4f=False), False, ["P4_phantom"], []),
    (dict(r1=(False, True)), False, ["P1_r1"], []),
    (dict(r1=(True, False)), False, ["P3_r1"], []),
    (dict(r1=None), False, [], ["P1_r1", "P3_r1"]),                  # r1 sin ejecutar o en V0
    (dict(r1=(None, True)), True, [], ["P1_r1"]),                    # se declara, no bloquea
    (dict(r1=(True, None)), False, [], ["P3_r1"]),
])
def test_la_regla_de_envio_junta_las_dos_partes(partes, se_queda, fallan,
                                               sin_evaluar) -> None:
    d = ia.decision_de_envio(*_partes(**partes))
    assert d["ships"] is se_queda and d["mide_arranque_default"] is se_queda
    assert d["failed"] == fallan and d["not_evaluated"] == sin_evaluar
    assert d["threshold_line_must_say_precision"] is (partes.get("p2", True) is False)
    assert any("P1 no se evaluó en R1" in x for x in d["declared"]) == ("P1_r1" in sin_evaluar)
    assert any("P3 no se evaluó" in x for x in d["declared"]) == ("P3_r1" in sin_evaluar)


# --------------------------------------------------------------------------- #
# Sobre un fantasma pequeño
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pequeno():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ia, "LOSA", LOSA_TEST)
        return ia.construye(CONFIG_TEST)


def test_prepara_es_la_del_pipeline_bit_a_bit(pequeno) -> None:
    f = pequeno
    b = rn.prepara_a(f.campo, f.hu, f.V, f.etq)
    a = ia.prepara(f.campo, f.hu, f.V, f.etq, mide_arranque=True)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    assert a[3] == b[3] and a[3]["sparse"] and "start_spread_mm" in a[3]
    assert np.array_equal(rn.pose_de(a[2](a[0], a[1])), rn.pose_de(b[2](b[0], b[1])))
    # Apagada (P1 y el denominador de P3): lo mismo que el pipeline salvo `start_spread_mm`.
    off = ia.prepara(f.campo, f.hu, f.V, f.etq, mide_arranque=False)
    assert ia.misma_pose(off, b)["pass"]
    assert off[3] == {k: v for k, v in b[3].items() if k != "start_spread_mm"}


def test_la_pose_es_la_misma_con_y_sin_arranques(pequeno) -> None:
    f = pequeno
    con = ia.prepara(f.campo, f.hu, f.V, f.etq, mide_arranque=True)
    sin = ia.prepara(f.campo, f.hu, f.V, f.etq, mide_arranque=False)
    p1 = ia.misma_pose(con, sin)
    assert p1["pass"] and p1["max_mm"] == 0.0 and all(p1["identical"].values())
    assert "start_spread_mm" not in sin[3]
    assert {"origin", "target", "registrar_type", "registrar.rot0",
            "registrar.extra"} <= set(p1["identical"])


def test_p1_ve_cualquier_campo_del_registrador(pequeno) -> None:
    from dataclasses import replace

    f = pequeno
    con = ia.prepara(f.campo, f.hu, f.V, f.etq, mide_arranque=False)
    otro = (con[0], con[1], replace(con[2], extra={"x": 1}), con[3])
    p1 = ia.misma_pose(con, otro)
    assert not p1["pass"] and not p1["identical"]["registrar.extra"]
    assert p1["max_mm"] == 0.0         # la pose es la misma; lo que falla es el registrador


# --------------------------------------------------------------------------- #
# main: las negativas, antes de tocar ningún dato
# --------------------------------------------------------------------------- #
@pytest.fixture
def disciplina(tmp_path, monkeypatch) -> Path:
    """Árbol limpio, protocolo versionado y sin cambios, entregas vacías en una copia
    principal falsa. Cada test rompe una sola cosa."""
    principal = tmp_path / "principal"
    (principal / rn.ANOTACIONES / "anotadora-1").mkdir(parents=True)
    monkeypatch.setattr(ia, "checkout_principal", lambda: principal)
    monkeypatch.setattr(ia, "_revision", lambda: {"git_head": HEAD, "git_dirty": False,
                                                  "git_diff_sha256": None,
                                                  "untracked_code": []})
    monkeypatch.setattr(ia, "_git_rastrea", lambda ruta: True)
    monkeypatch.setattr(ia, "_protocolo_sin_cambios", lambda: True)
    monkeypatch.setattr(ia, "_blob_del_protocolo", lambda: "b" * 40)
    monkeypatch.setattr(ia, "_entorno", lambda: {"python": "x"})
    return principal


def _referencia(raiz: Path, contenido: bytes = b"no es R1") -> Path:
    ref = raiz / "ref"
    (ref / "export").mkdir(parents=True)
    (ref / "export" / "caso.uos").write_bytes(contenido)
    (ref / "run-record.json").write_text(json.dumps(
        {"code": {"git_head": "57eb021" + "0" * 33, "git_dirty": False}}))
    return ref


def _nada_escrito(principal: Path) -> bool:
    return not (principal / ia.SALIDAS).exists()


@pytest.mark.parametrize("sub", [["fantasma"], ["r1", "--referencia", "/x/NOMBRE"]])
def test_main_se_niega_con_el_codigo_sucio(disciplina, monkeypatch, sub) -> None:
    monkeypatch.setattr(ia, "_revision", lambda: {"git_head": HEAD, "git_dirty": True})
    with pytest.raises(SystemExit, match="sucio"):
        ia.main(["x", *sub])
    assert _nada_escrito(disciplina)


@pytest.mark.parametrize("que, patron", [("sin_versionar", "no está en git"),
                                         ("con_cambios", "cambios sin confirmar")])
def test_main_se_niega_si_el_protocolo_no_esta_fijado(disciplina, monkeypatch, que,
                                                      patron) -> None:
    if que == "sin_versionar":
        monkeypatch.setattr(ia, "_git_rastrea", lambda ruta: False)
    else:
        monkeypatch.setattr(ia, "_protocolo_sin_cambios", lambda: False)
    with pytest.raises(SystemExit, match=patron):
        ia.main(["x", "fantasma"])
    assert _nada_escrito(disciplina)


def test_main_r1_se_niega_con_otro_contenedor(disciplina, tmp_path) -> None:
    with pytest.raises(SystemExit, match="SHA-256"):
        ia.main(["x", "r1", "--referencia", str(_referencia(tmp_path))])
    assert _nada_escrito(disciplina)


def test_main_r1_se_niega_si_la_medida_3_cambio_de_referencia(disciplina, tmp_path,
                                                             monkeypatch) -> None:
    monkeypatch.setitem(rn.CONTENEDOR_SHA256, "A", "0" * 64)
    with pytest.raises(SystemExit, match="ya no es R1"):
        ia.main(["x", "r1", "--referencia", str(_referencia(tmp_path))])
    assert _nada_escrito(disciplina)


def test_main_r1_se_niega_con_entregas_en_la_carpeta_fija(disciplina, tmp_path) -> None:
    (disciplina / rn.ANOTACIONES / "anotadora-1" / "puntos.mrk.json").write_text("{}")
    with pytest.raises(SystemExit, match="entrega"):
        ia.main(["x", "r1", "--referencia", str(_referencia(tmp_path))])
    assert _nada_escrito(disciplina)


def test_main_se_niega_si_la_salida_existe(disciplina) -> None:
    (disciplina / ia.SALIDAS / f"fantasma-{HEAD[:12]}").mkdir(parents=True)
    with pytest.raises(SystemExit, match="ya existe"):
        ia.main(["x", "fantasma"])


def test_main_se_niega_a_repetir_desde_otro_commit(disciplina) -> None:
    (disciplina / ia.SALIDAS / f"fantasma-{'e' * 12}").mkdir(parents=True)
    with pytest.raises(SystemExit, match="ya existe una ejecución de `fantasma`"):
        ia.main(["x", "fantasma"])
    assert not (disciplina / ia.SALIDAS / f"fantasma-{HEAD[:12]}").exists()
    # Otro subcomando no cuenta: r1 llega hasta su referencia y se niega por ella.
    with pytest.raises(SystemExit, match="SHA-256"):
        ia.main(["x", "r1", "--referencia", str(_referencia(disciplina.parent))])


@pytest.mark.parametrize("crudo", [["r1", "--ref", "/r"], ["r1", "--referencia=/r", "--sal=x"],
                                   ["fant"], ["fantasma", "--referencia", "/r"]])
def test_el_analizador_no_acepta_abreviaturas_ni_extras(crudo) -> None:
    with pytest.raises(SystemExit):
        ia.analizador().parse_args(crudo)


def test_el_argv_registrado_no_lleva_rutas() -> None:
    ap = ia.analizador()
    assert ia.argv_registrado(ap.parse_args(["r1", "--referencia=/home/p/NOMBRE/run-6"])) == [
        "scripts/incertidumbre_arranque.py", "r1", "--referencia", "<referencia>"]
    assert ia.argv_registrado(ap.parse_args(["fantasma"])) == [
        "scripts/incertidumbre_arranque.py", "fantasma"]


# --------------------------------------------------------------------------- #
# main: el ciclo del run-record
# --------------------------------------------------------------------------- #
def _registro(principal: Path, sub: str) -> dict:
    ruta = principal / ia.SALIDAS / f"{sub}-{HEAD[:12]}" / "run-record.json"
    return json.loads(ruta.read_text())


def test_fantasma_de_punta_a_punta_con_una_bateria_reducida(disciplina, monkeypatch) -> None:
    monkeypatch.setattr(ia, "LOSA", LOSA_TEST)
    real = ia.construye

    def construye(config):
        if config["id"] == "T2":
            raise ValueError("fantasma roto a propósito")
        if config["id"] == "T3":
            raise IndexError("fallo inesperado a propósito")
        return real(config)

    monkeypatch.setattr(ia, "construye", construye)
    monkeypatch.setattr(ia, "BATERIA", (CONFIG_TEST,
                                        {**CONFIG_TEST, "id": "T2", "gradiente": True},
                                        {**CONFIG_TEST, "id": "T3", "gradiente": True}))
    assert ia.main(["x", "fantasma"]) == 0
    salida = disciplina / ia.SALIDAS / f"fantasma-{HEAD[:12]}"
    rr = _registro(disciplina, "fantasma")
    assert rr["status"] == "ok" and rr["error_type"] is None
    assert rr["argv"] == ["scripts/incertidumbre_arranque.py", "fantasma"]
    assert rr["protocol"] == {"path": ia.PROTOCOLO, "git_blob": "b" * 40}
    assert rr["annotations_guard"] is None and rr["constants"]["sha256"]
    assert set(rr["outputs"]) == {"T1.json", "T2.json", "T3.json", "criterios.json"}
    for nombre, sha in rr["outputs"].items():
        assert hashlib.sha256((salida / nombre).read_bytes()).hexdigest() == sha
    t1 = json.loads((salida / "T1.json").read_text())
    assert t1["registered"] and t1["P1"]["pass"] and t1["P1"]["max_mm"] == 0.0
    a = t1["preparation_report"]["start_spread_mm"]
    assert a["n_starts"] == 8 and a["angle_deg"] == 5.0
    assert any("orientación de partida" in m for m in t1["gate_reasons"])
    # La losa de prueba es un solo valor: sin gradiente, y el umbral no mueve nada.
    assert t1["band_voxels_between_thresholds"] == 0 and t1["gradient_declaration_holds"]
    assert t1["preparation_report"]["threshold_spread_mm"]["median"] == 0.0
    t2 = json.loads((salida / "T2.json").read_text())
    assert not t2["registered"] and t2["error_type"] == "ValueError"
    assert "fantasma roto a propósito" in t2["traceback"]
    c = json.loads((salida / "criterios.json").read_text())
    assert c["P1"]["pass"] and c["P4"]["pass"] and "P3" not in c
    assert c["P2"]["n"] == 2 and c["P2"]["covered"] == 0 and not c["P2"]["pass"]
    assert c["no_gradient_descriptive"]["ids"] == ["T1"]
    assert c["gradient_declaration"] == {"holds": True, "mismatched": []}
    # Cualquier excepción, no solo ValueError, se escribe y cuenta como no cubierta.
    assert c["not_registered"] == [
        {"id": "T2", "error_type": "ValueError", "message": "fantasma roto a propósito"},
        {"id": "T3", "error_type": "IndexError", "message": "fallo inesperado a propósito"}]
    assert c["P2"]["not_registered_count_as_uncovered"] == ["T2", "T3"]
    # Una segunda ejecución desde el mismo commit no pisa la primera.
    with pytest.raises(SystemExit, match="ya existe"):
        ia.main(["x", "fantasma"])


def test_un_fallo_queda_en_el_run_record(disciplina, monkeypatch) -> None:
    def revienta(*_a, **_k):
        raise RuntimeError("fallo de prueba")

    monkeypatch.setattr(ia, "evalua_bateria", revienta)
    with pytest.raises(RuntimeError):
        ia.main(["x", "fantasma"])
    rr = _registro(disciplina, "fantasma")
    assert rr["status"] == "error" and rr["error_type"] == "RuntimeError"
    assert rr["finished_at"] is not None and rr["outputs"] == {}


@pytest.mark.parametrize("reproduce", [True, False])
def test_r1_de_punta_a_punta_sobre_el_fantasma(disciplina, tmp_path, monkeypatch, pequeno,
                                               reproduce: bool) -> None:
    f = pequeno
    corona, destino, reg, _ = rn.prepara_a(f.campo, f.hu, f.V, f.etq)
    T = rn.pose_de(reg(corona, destino))
    if not reproduce:
        T = rn._4x4(np.eye(3), np.array([0.0, 0.0, 0.05])) @ T
    datos = rn.Datos(campo=f.campo, hu=f.hu, V=f.V, etq=f.etq, T_ref=T,
                     entradas={"reference_uos_sha256": "d" * 64})
    monkeypatch.setattr(ia, "comprueba_r1", lambda ref: (ref / "export" / "caso.uos", {}))
    monkeypatch.setattr(rn, "carga_referencia", lambda *a: datos)
    codigo = ia.main(["x", "r1", "--referencia", str(_referencia(tmp_path))])
    rr = _registro(disciplina, "r1")
    salida = disciplina / ia.SALIDAS / f"r1-{HEAD[:12]}"
    assert rr["annotations_guard"] == {"path": rn.ANOTACIONES, "empty": True}
    assert rr["argv"] == ["scripts/incertidumbre_arranque.py", "r1", "--referencia",
                          "<referencia>"]
    if not reproduce:
        assert codigo == 2 and rr["status"] == "aborted" and rr["aborted"] == "V0"
        assert set(rr["outputs"]) == {"V0.json"}
        return
    assert codigo == 0 and rr["status"] == "ok"
    assert set(rr["outputs"]) == {"V0.json", "declaracion.json"}
    d = json.loads((salida / "declaracion.json").read_text())
    assert d["P1"]["pass"] and d["start_spread_mm"]["n_starts"] == 8
    assert d["P3"]["order"] == ["without", "with", "without", "with"]
    assert d["P3"]["extra_seconds"] == pytest.approx(
        d["P3"]["min_seconds_with"] - d["P3"]["min_seconds_without"])
    assert d["P3"]["pass"] and d["P3"]["limit_s"] == 120.0
    assert d["r1_part"]["decides_shipping"] == {"P1_r1": True, "P3_r1": True}
    assert d["U_mm"] == pytest.approx(np.hypot(d["threshold_spread_mm"]["median"]
                                               if d["threshold_spread_mm"] else 0.0,
                                               d["start_spread_mm"]["median"]))
