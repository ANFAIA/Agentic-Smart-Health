"""La navaja por diente y los arranques desplazados (`scripts/registro_navaja_arranques.py`).

Todo sintético: ningún dato de paciente. Se prueban las piezas que deciden el resultado —la
composición de transformaciones, los residuos, el movimiento, el evaluador de criterios con
números hechos a mano— y las guardas que hacen que la medida se ejecute una sola vez antes de
abrir las entregas. Sobre el fantasma de `fusion-agents` se ejecutan los dos casos de punta
a punta, y la equivarianza tiene que salir exacta: es lo que dice que las poses se componen
en el orden correcto.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

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


rn = _carga("registro_navaja_arranques", RAIZ / "scripts" / "registro_navaja_arranques.py")
fa = _carga("fantasma_arcada", RAIZ / "packages/fusion-agents/tests/fantasma_arcada.py")

from fusion_agents.registration import (  # noqa: E402
    RegistrationResult,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def _rigida(semilla: int) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    q = rng.normal(size=4)
    R = quaternion_to_matrix(tuple(q / np.linalg.norm(q)))
    return rn._4x4(R, rng.uniform(-50, 50, 3))


# --------------------------------------------------------------------------- #
# Transformaciones
# --------------------------------------------------------------------------- #
def test_inversa_y_composicion() -> None:
    a, b = _rigida(1), _rigida(2)
    X = np.random.default_rng(3).normal(size=(50, 3)) * 20
    np.testing.assert_allclose(rn.inversa(a) @ a, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(rn.aplica(rn.inversa(a), rn.aplica(a, X)), X, atol=1e-10)
    # `a @ b` aplica primero `b`.
    np.testing.assert_allclose(rn.aplica(a @ b, X), rn.aplica(a, rn.aplica(b, X)), atol=1e-10)
    # La conjugación de la equivarianza B: G·T·G⁻¹ sobre G·x es G·(T·x).
    np.testing.assert_allclose(rn.aplica(a @ b @ rn.inversa(a), rn.aplica(a, X)),
                               rn.aplica(a, rn.aplica(b, X)), atol=1e-10)
    # La de A: (T∘G⁻¹) sobre G·x es T·x.
    np.testing.assert_allclose(rn.aplica(b @ rn.inversa(a), rn.aplica(a, X)),
                               rn.aplica(b, X), atol=1e-10)


def test_pose_de_un_resultado_y_desplazamiento() -> None:
    m = _rigida(4)
    r = RegistrationResult(rotation=matrix_to_quaternion(m[:3, :3]),
                           translation=tuple(float(x) for x in m[:3, 3]),
                           rms_mm=0.0, iterations=0, converged=True)
    np.testing.assert_allclose(rn.pose_de(r), m, atol=1e-12)
    X = np.random.default_rng(5).normal(size=(20, 3))
    assert rn.desplazamiento(m, m, X).max() == 0.0
    t = rn._4x4(np.eye(3), [0.0, 3.0, 4.0])
    np.testing.assert_allclose(rn.desplazamiento(t, np.eye(4), X), 5.0)


def test_giro_en_torno_deja_quieto_el_centro() -> None:
    c = np.array([10.0, -20.0, 35.0])
    G = rn.giro_en_torno(rn._gira(np.array([1.0, 1.0, 0.0]), 5.0), c)
    np.testing.assert_allclose(rn.aplica(G, c[None])[0], c, atol=1e-12)
    assert abs(rn._angulo_4x4(G) - 5.0) < 1e-9


def test_eje_anatomico_va_a_los_ejes_lps() -> None:
    # derecha = −x, anterior = −y, superior = +z.
    np.testing.assert_array_equal(rn.eje_anatomico((1, 0, 0)), [-1.0, 0.0, 0.0])
    np.testing.assert_array_equal(rn.eje_anatomico((0, 1, 0)), [0.0, -1.0, 0.0])
    np.testing.assert_array_equal(rn.eje_anatomico((0, 0, 1)), [0.0, 0.0, 1.0])
    assert len(rn.DIRECCIONES) == 8


def test_movimientos_b_son_giros_propios_en_z_sin_traslacion_en_z() -> None:
    movs = rn.movimientos_b()
    assert len(movs) == 10
    for k, (_, G) in enumerate(movs):
        R, t = G[:3, :3], G[:3, 3]
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        # Ceros exactos: la z de cada punto no cambia ni en el último bit.
        assert list(R[2]) == [0.0, 0.0, 1.0] and list(R[:, 2]) == [0.0, 0.0, 1.0]
        assert t[2] == 0.0 and np.all(np.abs(t[:2]) <= 100.0)
        grados = np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360
        assert abs(grados - (36.0 * k + 10.0)) < 1e-9
        esperado = np.random.default_rng(1001 + k).uniform(-100, 100, 2)
        np.testing.assert_array_equal(t[:2], esperado)
    # Fijos: dos llamadas dan lo mismo.
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(movs, rn.movimientos_b(),
                                                           strict=True))


def test_mueve_campo_no_cambia_la_z() -> None:
    rng = np.random.default_rng(6)
    origen = np.array([12.5, -33.0, 41.0])
    campo = {"centers": rng.normal(size=(200, 3)).astype(np.float32) * 30, "origin": origen}
    _, G = rn.movimientos_b()[3]
    movido = rn.mueve_campo(campo, G)
    absolutas = campo["centers"].astype(np.float64) + origen
    assert movido["centers"].dtype == np.float64 and not movido["origin"].any()
    np.testing.assert_array_equal(movido["centers"][:, 2], absolutas[:, 2])
    np.testing.assert_allclose(movido["centers"], rn.aplica(G, absolutas), atol=1e-12)


# --------------------------------------------------------------------------- #
# Residuos y movimiento
# --------------------------------------------------------------------------- #
def test_residuo_de_contorno_es_la_mediana_sin_truncar() -> None:
    corona = np.zeros((1, 3))
    contorno = np.array([[0.2, 0, 0], [0.5, 0, 0], [3.0, 0, 0]])
    assert rn.residuo_contorno(np.eye(4), contorno, corona) == pytest.approx(0.5)
    # Una pose que lleva la corona a x=3: los mismos puntos a 2,8 · 2,5 · 0.
    t = rn._4x4(np.eye(3), [3.0, 0, 0])
    assert rn.residuo_contorno(t, contorno, corona) == pytest.approx(2.5)
    assert rn.residuo_contorno(np.eye(4), contorno[:0], corona) is None


def test_residuo_contra_esmalte() -> None:
    arbol = cKDTree(np.array([[0.0, 0, 0], [10.0, 0, 0]]))
    corona = np.array([[1.0, 0, 0], [2.0, 0, 0], [9.0, 0, 0]])
    assert rn.residuo_esmalte(np.eye(4), corona, arbol) == pytest.approx(1.0)
    assert rn.residuo_esmalte(np.eye(4), corona[:0], arbol) is None


def test_movimiento_de_una_traslacion() -> None:
    X = np.random.default_rng(7).normal(size=(30, 3))
    sin = rn._4x4(np.eye(3), [0.0, 0.3, -0.4])
    m = rn.movimiento(np.eye(4), sin, X)
    assert m["median_mm"] == pytest.approx(0.5) and m["max_mm"] == pytest.approx(0.5)
    assert m["mean_vertical_mm"] == pytest.approx(-0.4)
    assert rn.movimiento(np.eye(4), sin, X[:0])["median_mm"] is None


def test_etiqueta_del_contorno_con_la_pose_completa() -> None:
    corona = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    etq = np.array([11, 21])
    destino = np.array([[1.0, 0, 0], [9.0, 0, 0], [4.0, 0, 0]])
    np.testing.assert_array_equal(
        rn.etiqueta_del_contorno(np.eye(4), corona, etq, destino), [11, 21, 11])
    # Con la corona desplazada 3 mm, el punto de x=4 pasa a ser de la otra.
    t = rn._4x4(np.eye(3), [-3.0, 0, 0])
    np.testing.assert_array_equal(
        rn.etiqueta_del_contorno(t, corona, etq, destino), [11, 21, 21])


def test_vuelta_es_la_mediana_por_encima_de_5_mm() -> None:
    X = np.zeros((5, 3))
    assert rn.es_vuelta(rn._4x4(np.eye(3), [5.0, 0, 0]), np.eye(4), X) == (False, 5.0)
    vuelta, m = rn.es_vuelta(rn._4x4(np.eye(3), [5.001, 0, 0]), np.eye(4), X)
    assert vuelta and m == pytest.approx(5.001)


# --------------------------------------------------------------------------- #
# Criterios, con números hechos a mano
# --------------------------------------------------------------------------- #
def _par(x: int, con: float, sin: float, mov: float, valido: bool = True) -> dict:
    return {"tooth": x, "valid": valido, "residual_full_mm": con, "residual_without_mm": sin,
            "movement": {"median_mm": mov, "max_mm": mov * 2, "mean_vertical_mm": -mov}}


def test_n1_en_el_limite_pasa_y_por_encima_no() -> None:
    pares = [_par(11, 0.5, 1.0, 0.2), _par(12, 0.4, 0.8, 0.2), _par(13, 0.6, 1.2, 0.2)]
    n1 = rn.evalua_n1(rn.agrega_pares(pares, (11, 12, 13)))
    assert n1["pass"] is True and n1["median_without_mm"] == pytest.approx(1.0)
    assert n1["limit_mm"] == pytest.approx(1.0) and n1["ratio"] == pytest.approx(2.0)
    pares[0] = _par(11, 0.5, 1.0001, 0.2)
    pares[1] = _par(12, 0.4, 1.0001, 0.2)
    assert rn.evalua_n1(rn.agrega_pares(pares, (11, 12, 13)))["pass"] is False


def test_n1_excluye_los_dientes_sin_contorno() -> None:
    pares = [_par(11, 0.5, 0.6, 0.1), {**_par(12, 0.5, 9.0, 0.1), "residual_full_mm": None}]
    n1 = rn.evalua_n1(rn.agrega_pares(pares, (11, 12)))
    assert n1["pass"] is True and n1["teeth_used"] == [11] and n1["teeth_without_data"] == [12]
    assert rn.evalua_n1(rn.agrega_pares([], (11,)))["pass"] is None


def test_las_vueltas_no_entran_en_las_medianas_de_n1_ni_n2() -> None:
    pares = [_par(11, 0.5, 0.6, 0.2), _par(11, 0.5, 0.7, 0.4),
             _par(11, 0.5, 50.0, 60.0, valido=False),          # par con vuelta
             _par(12, 0.5, 0.5, 9.0, valido=False)]             # todos con vuelta
    por = rn.agrega_pares(pares, (11, 12))
    assert por[11]["n_pairs"] == 3 and por[11]["n_valid"] == 2
    assert por[11]["movement_median_mm"] == pytest.approx(0.3)
    assert por[11]["residual_without_mm"] == pytest.approx(0.65)
    assert por[11]["movement_max_mm"] == pytest.approx(0.8)
    assert por[12]["n_valid"] == 0 and por[12]["movement_median_mm"] is None
    n2 = rn.evalua_n2(por)
    assert n2["pass"] is True and n2["teeth_without_data"] == [12]
    assert rn.evalua_n1(por)["teeth_without_data"] == [12]


def test_n2_falla_con_un_diente_por_encima_de_epsilon() -> None:
    assert rn.N2_MAX_MM == 1.5
    por = rn.agrega_pares([_par(11, 0.5, 0.5, 1.5), _par(12, 0.5, 0.5, 1.51)], (11, 12))
    n2 = rn.evalua_n2(por)
    assert n2["pass"] is False and n2["teeth_over"] == [12]
    assert rn.evalua_n2(rn.agrega_pares([_par(11, 0.5, 0.5, 1.5)], (11,)))["pass"] is True


def test_k1_cuenta_una_vuelta_como_fallo_aunque_vuelva_cerca() -> None:
    filas = [{"with_criterion": True, "displacement_median_mm": 0.1, "flip": False,
              "converged": True},
             {"with_criterion": True, "displacement_median_mm": 0.0, "flip": True,
              "converged": True},
             {"with_criterion": False, "displacement_median_mm": 30.0, "flip": True,
              "converged": False}]
    k1 = rn.evalua_k1(filas)
    assert k1["pass"] is False and k1["n"] == 2 and k1["failures"] == 1
    assert k1["failures_by_flip"] == 1
    assert rn.evalua_k1(filas[:1])["pass"] is True
    assert rn.evalua_k1([{**filas[0], "displacement_median_mm": 0.1001}])["pass"] is False
    assert rn.evalua_k1(filas[2:])["pass"] is False          # sin filas con criterio


def test_k1_una_ejecucion_sin_converger_es_fallo_y_se_cuenta_aparte() -> None:
    filas = [{"with_criterion": True, "displacement_median_mm": 0.0, "flip": False,
              "converged": False},
             {"with_criterion": True, "displacement_median_mm": 0.05, "flip": False,
              "converged": True},
             {"with_criterion": True, "displacement_median_mm": 0.2, "flip": False,
              "converged": True}]
    k1 = rn.evalua_k1(filas)
    assert k1["pass"] is False and k1["failures"] == 2
    assert k1["failures_not_converged"] == 1 and k1["failures_by_distance"] == 1
    assert k1["failures_by_flip"] == 0
    assert rn.evalua_k1(filas[1:2])["pass"] is True
    # Sin criterio: los mismos recuentos, sin veredicto.
    resumen = rn.resume_sin_criterio(filas)
    assert "pass" not in resumen and resumen["failures_not_converged"] == 1


def test_e1() -> None:
    assert rn.evalua_e1([{"max_mm": 0.01}, {"max_mm": 1e-12}])["pass"] is True
    assert rn.evalua_e1([{"max_mm": 0.0100001}])["pass"] is False
    assert rn.evalua_e1([])["pass"] is False


# --------------------------------------------------------------------------- #
# Guardas
# --------------------------------------------------------------------------- #
def test_las_entregas_vacias_dejan_ejecutar(tmp_path: Path) -> None:
    (tmp_path / "anotadora-1" / "caso-a").mkdir(parents=True)
    (tmp_path / "anotadora-2").mkdir()
    rn.comprueba_anotaciones(tmp_path)


@pytest.mark.parametrize("que", ["fichero", "oculto", "enlace"])
def test_cualquier_fichero_en_las_entregas_lo_impide(tmp_path: Path, que: str) -> None:
    carpeta = tmp_path / "anotaciones" / "anotadora-1" / "caso-a"
    carpeta.mkdir(parents=True)
    if que == "fichero":
        (carpeta / "puntos.mrk.json").write_text("{}")
    elif que == "oculto":
        (carpeta / ".gitkeep").write_text("")
    else:
        (carpeta / "enlace").symlink_to(tmp_path)
    with pytest.raises(SystemExit, match="entrega"):
        rn.comprueba_anotaciones(tmp_path / "anotaciones")


def test_sin_carpeta_de_entregas_no_se_ejecuta(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no es una carpeta"):
        rn.comprueba_anotaciones(tmp_path / "no-existe")


def test_el_analizador_no_acepta_abreviaturas_ni_anotaciones() -> None:
    for argv in (["x", "--caso", "A", "--ref", "/home/p/NOMBRE/run-6"],
                 ["x", "--caso", "A", "--refer=/home/p/NOMBRE/run-6"],
                 ["x", "--caso", "A", "--referencia", "/r", "--sal", "/o"],
                 ["x", "--caso", "A", "--referencia", "/r", "--anotaciones", "/a"]):
        with pytest.raises(SystemExit) as exc:
            rn.main(argv)
        assert exc.value.code == 2               # el error de argparse, antes de nada


def test_no_se_pisa_una_salida_ni_una_etapa(tmp_path: Path) -> None:
    salida = rn.prepara_salida(tmp_path / "a" / "A")
    assert salida.is_dir()
    with pytest.raises(SystemExit, match="ya existe"):
        rn.prepara_salida(tmp_path / "a" / "A")
    rn.escribe_etapa(salida, "V0", {"x": np.float64(1.5), "m": np.eye(2)}, {"case": "A"})
    texto = json.loads((salida / "V0.json").read_text())
    assert texto["stage"] == "V0" and texto["x"] == 1.5 and texto["m"] == [[1, 0], [0, 1]]
    with pytest.raises(SystemExit, match="ya existe"):
        rn.escribe_etapa(salida, "V0", {}, {})


def test_el_argv_se_reconstruye_de_los_argumentos_interpretados() -> None:
    ap = rn.analizador()
    for crudo in (["--caso", "A", "--referencia", "/home/p/NOMBRE/run-6", "--salida", "/tmp/s"],
                  ["--salida=/tmp/NOMBRE", "--referencia=/home/p/NOMBRE/run-6", "--caso=A"]):
        registrado = rn.argv_registrado(ap.parse_args(crudo))
        assert registrado == ["scripts/registro_navaja_arranques.py", "--caso", "A",
                              "--referencia", "<referencia>", "--salida", "<salida>"]
        assert not any("NOMBRE" in a or "/" in a.replace("scripts/", "") for a in registrado)
    assert rn.argv_registrado(ap.parse_args(["--caso", "B", "--referencia", "/r"])) == [
        "scripts/registro_navaja_arranques.py", "--caso", "B", "--referencia", "<referencia>"]


def test_la_huella_es_de_valores_resueltos(monkeypatch) -> None:
    h = rn.huella_constantes()
    assert set(h["values"]) == {*rn.MODULOS_HUELLA, "precision_registro"}
    assert h["values"]["fusion_agents.por_cortes"]["UMBRAL_CONTORNO"] == 1400.0
    assert h["values"]["fusion_agents.preparacion"]["MUESTRA"] == 4000
    assert "_MINIMO_COSENO" in h["values"]["export_agents.anatomia"]
    assert set(h["values"]["precision_registro"]) == set(rn.CONSTANTES_PRECISION)
    # El valor, no el nombre de la constante de la que sale.
    assert h["script_values"]["UMBRAL_VUELTA_MM"] == 5.0
    assert h["script_values"]["N2_MAX_MM"] == 1.5
    assert "RAIZ" not in h["script_values"] and "UTC" not in h["script_values"]
    assert len(h["sha256"]) == 64 and len(h["script_sha256"]) == 64
    # Cambiar un valor cambia la huella; cambiar los de `movimientos()` también.
    monkeypatch.setattr(rn, "N1_FACTOR", 2.5)
    assert rn.huella_constantes()["script_sha256"] != h["script_sha256"]
    monkeypatch.setattr(rn.precision_registro, "TRASLACION_MAX_MM", 99.0)
    assert rn.huella_constantes()["sha256"] != h["sha256"]


def test_la_huella_no_acepta_lo_que_no_es_dato() -> None:
    assert rn.valores_resueltos({"A": (1, 2.5), "b": 3, "C": print, "D": {3, 1}}) == {
        "A": [1, 2.5], "D": [1, 3]}
    with pytest.raises(TypeError):
        rn._valor_json([object()])


# --------------------------------------------------------------------------- #
# main: las negativas, antes de tocar ningún dato
# --------------------------------------------------------------------------- #
HEAD = "a" * 40


@pytest.fixture
def disciplina(tmp_path, monkeypatch) -> Path:
    """Árbol limpio, protocolo versionado y sin cambios, entregas vacías en una copia
    principal falsa. Cada test rompe una sola cosa."""
    principal = tmp_path / "principal"
    (principal / rn.ANOTACIONES / "anotadora-1").mkdir(parents=True)
    monkeypatch.setattr(rn, "checkout_principal", lambda: principal)
    monkeypatch.setattr(rn, "_revision", lambda: {"git_head": HEAD, "git_dirty": False,
                                                  "git_diff_sha256": None,
                                                  "untracked_code": []})
    monkeypatch.setattr(rn, "_git_rastrea", lambda ruta: True)
    monkeypatch.setattr(rn, "_protocolo_sin_cambios", lambda: True)
    monkeypatch.setattr(rn, "_blob_del_protocolo", lambda: "b" * 40)
    monkeypatch.setattr(rn, "_entorno", lambda: {"python": "x"})
    return principal


def _referencia(raiz: Path, head: str = "57eb021" + "0" * 33, sucio: object = False,
                contenido: bytes = b"no es el contenedor") -> Path:
    ref = raiz / "ref"
    (ref / "export").mkdir(parents=True)
    (ref / "export" / "caso.uos").write_bytes(contenido)
    (ref / "run-record.json").write_text(json.dumps(
        {"code": {"git_head": head, "git_dirty": sucio}}))
    return ref


def _nada_escrito(principal: Path) -> bool:
    return not (principal / rn.SALIDAS).exists()


def test_main_se_niega_con_el_codigo_sucio(disciplina, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rn, "_revision", lambda: {"git_head": HEAD, "git_dirty": True})
    with pytest.raises(SystemExit, match="sucio"):
        rn.main(["x", "--caso", "A", "--referencia", str(_referencia(tmp_path))])
    assert _nada_escrito(disciplina)


@pytest.mark.parametrize("que", ["sin_versionar", "con_cambios"])
def test_main_se_niega_si_el_protocolo_no_esta_fijado(disciplina, tmp_path, monkeypatch,
                                                      que: str) -> None:
    if que == "sin_versionar":
        monkeypatch.setattr(rn, "_git_rastrea", lambda ruta: False)
        patron = "no está en git"
    else:
        monkeypatch.setattr(rn, "_protocolo_sin_cambios", lambda: False)
        patron = "cambios sin confirmar"
    with pytest.raises(SystemExit, match=patron):
        rn.main(["x", "--caso", "B", "--referencia", str(_referencia(tmp_path))])
    assert _nada_escrito(disciplina)


def test_main_se_niega_con_entregas_en_la_carpeta_fija(disciplina, tmp_path) -> None:
    (disciplina / rn.ANOTACIONES / "anotadora-1" / "puntos.mrk.json").write_text("{}")
    with pytest.raises(SystemExit, match="entrega"):
        rn.main(["x", "--caso", "A", "--referencia", str(_referencia(tmp_path))])
    assert _nada_escrito(disciplina)
    # En el caso B la carpeta no se mira: lo que lo para es la referencia, no las entregas.
    with pytest.raises(SystemExit, match="SHA-256"):
        rn.main(["x", "--caso", "B", "--referencia",
                 str(_referencia(tmp_path / "b", head="eef9d27" + "0" * 33))])


@pytest.mark.parametrize("por_defecto", [True, False])
def test_main_se_niega_si_la_salida_existe(disciplina, tmp_path, por_defecto: bool) -> None:
    if por_defecto:
        (disciplina / rn.SALIDAS / f"A-{HEAD[:12]}").mkdir(parents=True)
        extra: list[str] = []
    else:
        (tmp_path / f"A-{HEAD[:12]}").mkdir()
        extra = ["--salida", str(tmp_path / f"A-{HEAD[:12]}")]
    with pytest.raises(SystemExit, match="ya existe"):
        rn.main(["x", "--caso", "A", "--referencia", str(_referencia(tmp_path)), *extra])


def test_main_se_niega_con_otro_contenedor(disciplina, tmp_path) -> None:
    with pytest.raises(SystemExit, match="SHA-256"):
        rn.main(["x", "--caso", "A", "--referencia", str(_referencia(tmp_path))])
    assert _nada_escrito(disciplina)


@pytest.mark.parametrize("head, sucio, patron", [
    ("eef9d27" + "0" * 33, False, "commit"),        # el commit del otro caso
    ("57eb02", False, "commit"),                    # un prefijo más corto no vale
    ("57eb021" + "0" * 33, True, "git_dirty"),
    ("57eb021" + "0" * 33, None, "git_dirty"),     # sin declarar no es limpio
])
def test_main_se_niega_con_otra_ejecucion_de_referencia(disciplina, tmp_path, monkeypatch,
                                                        head, sucio, patron) -> None:
    contenido = b"contenedor de mentira"
    monkeypatch.setitem(rn.CONTENEDOR_SHA256, "A", hashlib.sha256(contenido).hexdigest())
    with pytest.raises(SystemExit, match=patron):
        rn.main(["x", "--caso", "A", "--referencia",
                 str(_referencia(tmp_path, head=head, sucio=sucio, contenido=contenido))])
    assert _nada_escrito(disciplina)


def test_comprueba_referencia_guarda_su_run_record(tmp_path, monkeypatch) -> None:
    contenido = b"contenedor de mentira"
    monkeypatch.setitem(rn.CONTENEDOR_SHA256, "B", hashlib.sha256(contenido).hexdigest())
    ref = _referencia(tmp_path, head="eef9d27" + "1" * 33, contenido=contenido)
    contenedor, entradas = rn.comprueba_referencia(ref, "B")
    assert contenedor.name == "caso.uos"
    assert entradas["reference_git_head"] == "eef9d27" + "1" * 33
    assert entradas["reference_run_record_sha256"] == hashlib.sha256(
        (ref / "run-record.json").read_bytes()).hexdigest()
    assert entradas["reference_uos_sha256"] == rn.CONTENEDOR_SHA256["B"]


def test_los_dientes_distintos_se_declaran_y_no_detienen() -> None:
    etq = np.array([0, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 18, 0])
    d = rn.declara_dientes("A", etq)
    assert d["match_expected"] is False and 18 in d["found"] and 17 not in d["found"]
    assert d["expected"] == list(rn.DIENTES_ESPERADOS["A"])
    assert rn.dientes_de(etq) == tuple(d["found"])
    assert rn.declara_dientes("B", etq[etq != 18])["match_expected"] is True


# --------------------------------------------------------------------------- #
# Entradas
# --------------------------------------------------------------------------- #
def test_elige_el_campo_semilla_y_la_superficie_de_las_etiquetas(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    c = rng.normal(size=(30, 3)).astype(np.float32)
    ident = np.tile(np.array([1, 0, 0, 0], np.float32), (30, 1))
    comun = {"density": np.ones(30), "origin": np.zeros(3), "hu_range": np.array([0., 1.])}
    np.savez(tmp_path / "semilla.npz", centers=c, rotations=ident, **comun)
    np.savez(tmp_path / "segmentado.npz", centers=c, rotations=ident,
             region_id=np.zeros(30, np.int16), **comun)
    girado = ident.copy()
    girado[0] = [0, 1, 0, 0]
    np.savez(tmp_path / "ajustado.npz", centers=c[:10], rotations=girado[:10],
             density=np.ones(10), origin=np.zeros(3), hu_range=np.array([0., 1.]))
    V = rng.normal(size=(12, 3))
    np.savez(tmp_path / "escaner.npz", positions=V, faces=np.zeros((1, 3), np.int64))
    np.savez(tmp_path / "otro.npz", positions=V[::-1].copy(), faces=np.zeros((1, 3), np.int64))
    npzs = sorted(tmp_path.glob("*.npz"))
    assert rn.elige_campo(npzs) == (tmp_path / "semilla.npz", "only_seed_field")
    elegido, como = rn.elige_superficie(npzs, 12, rn._sha_posiciones(V))
    assert elegido == tmp_path / "escaner.npz" and como == "source_positions_sha256"
    with pytest.raises(SystemExit):           # sin hash, dos superficies del mismo tamaño
        rn.elige_superficie(npzs, 12, None)
    # Dos semillas: vale la que comparte `centers` con el campo segmentado.
    np.savez(tmp_path / "semilla2.npz", centers=c + 1, rotations=ident, **comun)
    npzs = sorted(tmp_path.glob("*.npz"))
    assert rn.elige_campo(npzs) == (tmp_path / "semilla.npz", "centers_of_segmented_field")


def test_lee_el_contenedor_de_referencia(tmp_path: Path) -> None:
    T = _rigida(9)
    etq = np.array([0, 11, 11, 21, 0], dtype=np.int16)
    manifiesto = {"registrations": [
        {"id": "reg.ct_centered_to_ios", "transform_4x4_row_major": np.eye(4).ravel().tolist()},
        {"id": "reg.ct_to_ios", "rms_error_mm": 0.5,
         "transform_4x4_row_major": np.linalg.inv(T).ravel().tolist()}]}
    ruta = tmp_path / "ref.uos"
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("manifest.json", json.dumps(manifiesto))
        z.writestr("derived/seg_teeth.bin", etq.astype("<i2").tobytes())
        z.writestr("derived/seg_teeth.meta.json", json.dumps({"encoding": {}}))
    T_ref, leidas, meta, hashes = rn.lee_contenedor(ruta)
    np.testing.assert_allclose(T_ref, T, atol=1e-12)
    np.testing.assert_array_equal(leidas, etq)
    assert leidas.dtype == np.int64 and meta == {"encoding": {}}
    assert hashes["reference_rms_mm"] == 0.5 and len(hashes["seg_teeth_sha256"]) == 64


# --------------------------------------------------------------------------- #
# De punta a punta sobre el fantasma
# --------------------------------------------------------------------------- #
ROT = fa.rotacion(np.array([0.3, -1.0, 0.4]), 70)
TRAS = np.array([12.0, -40.0, 33.0])


def _rejilla(paso: float) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(-32, 32, paso), np.arange(-16, 20, paso), indexing="ij")
    return np.c_[gx.ravel(), gy.ravel()]


def _coronas_en(xy: np.ndarray, z: float) -> np.ndarray:
    sup, fs = fa.centros_coronas("upper")
    inf, fi = fa.centros_coronas("lower")
    p = np.c_[xy, np.full(len(xy), z)]
    return p[fa._dentro(p, sup, fs, True) | fa._dentro(p, inf, fi, False)]


@pytest.fixture(scope="module")
def caso_a_sintetico() -> rn.Datos:
    """Serie dispersa (cortes cada 0,45 mm con un hueco de 2,25 mm en la banda), y una losa
    de esmalte lejos de la arcada a z = 0,15 para que el modo del esmalte caiga en el plano
    oclusal del fantasma: así `prepara_por_cortes` se llama sin `plano`, como en la medida."""
    xy = _rejilla(0.4)
    zs = [z for z in np.round(np.arange(-3.0, 9.0, 0.45), 3) if not 2.2 < z < 3.9]
    lx, ly = np.meshgrid(np.arange(-30, 30, 0.4), np.arange(30, 46, 0.4), indexing="ij")
    losa = np.c_[lx.ravel(), ly.ravel(), np.full(lx.size, 0.15)]
    c = np.vstack([*(_coronas_en(xy, z) for z in zs), losa])
    origen = np.array([3.0, -2.0, 1.5])
    Vw, _, etq = fa.escaneo_superior()
    return rn.Datos(campo={"centers": (c - origen).astype(np.float32), "origin": origen},
                    hu=np.full(len(c), 2200.0), V=fa.mueve(Vw, ROT, TRAS),
                    etq=etq.astype(np.int64), T_ref=np.eye(4), entradas={})


def test_caso_a_sintetico_se_registra_por_cortes(caso_a_sintetico) -> None:
    d = caso_a_sintetico
    corona, destino, reg, informe = rn.prepara_a(d.campo, d.hu, d.V, d.etq)
    assert informe["sparse"] and abs(informe["occlusal_plane_mm"] - 0.15) < 0.1
    T0 = rn.pose_de(reg(corona, destino))
    verdad = rn._4x4(ROT.T, -ROT.T @ TRAS)
    assert np.median(rn.desplazamiento(T0, verdad, corona)) < 1.5
    # Determinista: la segunda vez sale lo mismo bit a bit (lo que V0 da por hecho).
    assert np.array_equal(rn.pose_de(reg(corona, destino)), T0)


def test_navaja_a_y_arranques_a_sobre_el_fantasma(caso_a_sintetico) -> None:
    d = caso_a_sintetico
    corona, destino, reg, _ = rn.prepara_a(d.campo, d.hu, d.V, d.etq)
    T0 = rn.pose_de(reg(corona, destino))
    etq_c = d.etq[d.etq > 0]
    nav = rn.navaja_a(reg, corona, etq_c, destino, T0, (11, 17))
    for fila in nav["teeth"]:
        assert fila["n_contour_points"] > 50 and fila["residual_full_mm"] is not None
        for v in ("a", "b"):
            assert fila[v]["movement"]["median_mm"] < 1.0
            assert fila[v]["movement"]["n"] == fila["n_crown_vertices"]
    assert set(nav["per_tooth"]["b"]) == {11, 17}
    # Sin perturbar, el registrador vuelve exactamente a la pose de referencia.
    arr = rn.arranques_a(reg, corona, destino, T0, angulos=(2.0,),
                         direcciones=rn.DIRECCIONES[:1], alturas=(0.0,))
    assert arr[1]["kind"] == "height" and arr[1]["displacement_median_mm"] == 0.0
    assert arr[0]["with_criterion"] and not arr[1]["with_criterion"]


def test_equivarianza_a_exacta_sobre_el_fantasma(caso_a_sintetico) -> None:
    d = caso_a_sintetico
    corona, destino, reg, _ = rn.prepara_a(d.campo, d.hu, d.V, d.etq)
    T0 = rn.pose_de(reg(corona, destino))
    filas = rn.equivariancia_a(d.campo, d.hu, d.V, d.etq, T0, rn.movimientos()[:3])
    assert rn.evalua_e1(filas)["pass"]
    assert max(f["max_mm"] for f in filas) < 1e-9
    # Y una expectativa mal compuesta (G⁻¹∘T) se nota: no es una prueba que pase siempre.
    G = rn._4x4(*rn.movimientos()[0][1:])
    corona2 = rn.aplica(G, corona)
    assert rn.desplazamiento(filas[0]["pose"], rn.inversa(G) @ T0, corona2).max() > 1.0


def test_ejecuta_a_escribe_cada_etapa_y_los_criterios(caso_a_sintetico, tmp_path) -> None:
    d = caso_a_sintetico
    corona, destino, reg, _ = rn.prepara_a(d.campo, d.hu, d.V, d.etq)
    datos = rn.Datos(**{**d.__dict__, "T_ref": rn.pose_de(reg(corona, destino))})
    salida = rn.prepara_salida(tmp_path / "A")
    hecho = rn.ejecuta_a(datos, salida, {"case": "A"})
    assert hecho["stages"] == ["V0", "equivarianza", "navaja", "arranques", "criterios"]
    criterios = json.loads((salida / "criterios.json").read_text())
    assert criterios["E1"]["pass"] is True
    assert criterios["K1"]["n"] == 20 and criterios["E1"]["n"] == 10
    assert set(criterios["N1"]["teeth_used"]) == set(rn.dientes_de(d.etq))
    arr = json.loads((salida / "arranques.json").read_text())["runs"]
    assert len(arr) == 30


def test_v0_que_no_reproduce_detiene_sin_medir(caso_a_sintetico, tmp_path) -> None:
    d = caso_a_sintetico
    datos = rn.Datos(**{**d.__dict__, "T_ref": np.eye(4)})
    salida = rn.prepara_salida(tmp_path / "A")
    assert rn.ejecuta_a(datos, salida, {}) == {"aborted": "V0", "stages": ["V0"]}
    assert [p.name for p in salida.iterdir()] == ["V0.json"]
    assert json.loads((salida / "V0.json").read_text())["pass"] is False


@pytest.fixture(scope="module")
def caso_b_sintetico() -> rn.Datos:
    """Serie densa, las dos arcadas, y el escaneo ya en el marco del CBCT (el ICP clásico
    arranca en la identidad)."""
    xy = _rejilla(0.6)
    c = np.vstack([_coronas_en(xy, z) for z in np.round(np.arange(-12.0, 12.0, 0.6), 3)])
    origen = np.array([3.0, -2.0, 1.5])
    Vw, _, etq = fa.escaneo_superior()
    return rn.Datos(campo={"centers": (c - origen).astype(np.float32), "origin": origen},
                    hu=np.where(np.abs(c[:, 2]) < 4, 2200.0, 1500.0), V=Vw,
                    etq=etq.astype(np.int64), T_ref=np.eye(4), entradas={})


def test_caso_b_registro_y_equivarianza_exacta(caso_b_sintetico) -> None:
    d = caso_b_sintetico
    r0 = rn.registra_b(d.campo, d.hu, d.V, 0)
    assert r0["hu_target"] is not None
    assert np.array_equal(rn.registra_b(d.campo, d.hu, d.V, 0)["pose"], r0["pose"])
    filas = rn.equivariancia_b(d.campo, d.hu, d.V, d.etq, r0["pose"], rn.movimientos_b()[:2])
    assert rn.evalua_e1(filas)["pass"] and max(f["max_mm"] for f in filas) < 1e-9


def test_caso_b_arranque_en_caliente_devuelve_la_pose_efectiva(caso_b_sintetico) -> None:
    d = caso_b_sintetico
    T = rn.registra_b(d.campo, d.hu, d.V, 0)["pose"]
    caliente = rn.registra_b(d.campo, d.hu, d.V, 0, previa=T)
    coronas = d.V[d.etq > 0]
    # Desde la propia solución, el ICP afina poco: la pose efectiva (W·T) está cerca de T.
    assert np.median(rn.desplazamiento(caliente["pose"], T, coronas)) < 0.5


def test_ejecuta_b_reducido_escribe_cada_etapa(caso_b_sintetico, tmp_path,
                                               monkeypatch) -> None:
    d = caso_b_sintetico
    datos = rn.Datos(**{**d.__dict__,
                        "T_ref": rn.registra_b(d.campo, d.hu, d.V, 0)["pose"]})
    monkeypatch.setattr(rn, "dientes_de", lambda etq: (11, 21))
    monkeypatch.setattr(rn, "navaja_b", functools.partial(rn.navaja_b, semillas=(0, 1)))
    monkeypatch.setattr(rn, "arranques_b", functools.partial(rn.arranques_b, semillas=(0, 1)))
    monkeypatch.setattr(rn, "perturbaciones_b", functools.partial(
        rn.perturbaciones_b, angulos=(2.0,), direcciones=rn.DIRECCIONES[:1],
        traslaciones=(1.0,), ejes=rn.EJES_TRASLACION[:1]))
    movs = rn.movimientos_b()[:2]
    monkeypatch.setattr(rn, "movimientos_b", lambda: movs)
    salida = rn.prepara_salida(tmp_path / "B")
    hecho = rn.ejecuta_b(datos, salida, {"case": "B"})
    assert hecho["stages"] == ["V0", "equivarianza", "suelo", "navaja", "arranques",
                               "criterios"]
    navaja = json.loads((salida / "navaja.json").read_text())
    # La ejecución completa con la semilla del pipeline es la de V0, no otra.
    v0 = json.loads((salida / "V0.json").read_text())
    completa0 = next(r for r in navaja["runs"]["scratch"]
                     if r["seed"] == 0 and r["without"] is None)
    assert completa0["pose"] == v0["run"]["pose"]
    assert len(navaja["pairs"]["scratch"]) == 4 and len(navaja["pairs"]["warm"]) == 4
    suelo = json.loads((salida / "suelo.json").read_text())
    assert suelo["seeds"] == [1]
    assert set(suelo["per_tooth_values_mm"]) == {"11", "21"}
    criterios = json.loads((salida / "criterios.json").read_text())
    assert criterios["E1"]["pass"] is True and criterios["K1"]["n"] == 4
    assert criterios["readable"]["readable"] is True
    # La navaja de B no tiene criterio (§3.2): solo la lectura contra el suelo.
    for k in ("N1", "N2"):
        assert criterios[k]["pass"] is None and criterios[k]["verdict"] == "sin criterio"
    lectura = criterios["descriptive"]["seed_floor_reading"]["scratch"]
    # Dos semillas: menos de 10 pares válidos, así que todo queda indeterminado.
    assert {v["state"] for v in lectura.values()} == {"indeterminado"}
    assert set(criterios["descriptive"]["flip_rates"]["scratch"]) == {"full", "without"}
    # El residuo usa el esmalte del lado maxilar del plano del pipeline.
    esmalte = navaja["residual_enamel"]
    assert esmalte["split_plane_mm"] == v0["run"]["occlusal_plane_mm"]


def test_e1_que_falla_detiene_antes_de_la_navaja(caso_a_sintetico, disciplina, tmp_path,
                                                 monkeypatch) -> None:
    """§4: E1 va antes que todo; si falla, solo quedan V0, equivarianza y el run-record."""
    d = caso_a_sintetico
    corona, destino, reg, _ = rn.prepara_a(d.campo, d.hu, d.V, d.etq)
    datos = rn.Datos(**{**d.__dict__, "T_ref": rn.pose_de(reg(corona, destino)),
                        "entradas": {"reference_uos_sha256": "c" * 64}})
    movs = rn.movimientos()[:2]
    monkeypatch.setattr(rn, "movimientos", lambda: movs)
    monkeypatch.setattr(rn, "E1_MAX_MM", -1.0)          # E1 no puede cumplirse
    monkeypatch.setattr(rn, "comprueba_referencia",
                        lambda ref, caso: (Path("caso.uos"), {}))
    monkeypatch.setattr(rn, "carga_referencia", lambda *a: datos)

    def no_se_llama(*a, **k):
        raise AssertionError("la navaja no se calcula si E1 falla")

    monkeypatch.setattr(rn, "navaja_a", no_se_llama)
    monkeypatch.setattr(rn, "arranques_a", no_se_llama)
    referencia = tmp_path / "datos-de-NOMBRE" / "run-6"
    assert rn.main(["x", "--caso", "A", "--referencia", str(referencia)]) == 2
    salida = disciplina / rn.SALIDAS / f"A-{HEAD[:12]}"
    assert sorted(p.name for p in salida.iterdir()) == [
        "V0.json", "equivarianza.json", "run-record.json"]
    texto = (salida / "run-record.json").read_text()
    registro = json.loads(texto)
    assert registro["status"] == "aborted" and registro["aborted"] == "E1"
    assert registro["stages"] == ["V0", "equivarianza"]
    assert registro["finished_at"] is not None
    assert set(registro["outputs"]) == {"V0.json", "equivarianza.json"}
    assert registro["argv"] == ["scripts/registro_navaja_arranques.py", "--caso", "A",
                                "--referencia", "<referencia>"]
    assert registro["output"] == f"{rn.SALIDAS}/A-{HEAD[:12]}"
    assert registro["annotations_guard"] == {"path": rn.ANOTACIONES, "empty": True}
    assert registro["protocol"] == {"path": rn.PROTOCOLO, "git_blob": "b" * 40}
    assert registro["agents"] == {"geometric-fusion-agent": rn.GeometricFusionAgent.version}
    assert registro["seeds"] == [] and registro["teeth"]["found"]
    # Ninguna ruta llega al registro: ni la de la referencia ni la de la copia.
    assert "NOMBRE" not in texto and str(tmp_path) not in texto
    assert "anotadora" not in texto


def test_el_run_record_se_escribe_al_empezar_y_al_fallar(caso_a_sintetico, disciplina,
                                                         tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rn, "comprueba_referencia", lambda ref, caso: (Path("caso.uos"), {}))
    monkeypatch.setattr(rn, "carga_referencia", lambda *a: caso_a_sintetico)
    salida = disciplina / rn.SALIDAS / f"A-{HEAD[:12]}"
    vistos = []

    def ejecuta(datos, sal, meta):
        vistos.append(json.loads((sal / "run-record.json").read_text()))
        raise RuntimeError("algo falla a mitad")

    monkeypatch.setattr(rn, "ejecuta_a", ejecuta)
    with pytest.raises(RuntimeError):
        rn.main(["x", "--caso", "A", "--referencia", str(tmp_path / "ref")])
    assert vistos[0]["status"] == "running" and vistos[0]["finished_at"] is None
    assert vistos[0]["started_at"]
    final = json.loads((salida / "run-record.json").read_text())
    assert final["status"] == "error" and final["error_type"] == "RuntimeError"
    assert final["finished_at"] is not None
    assert not (salida / "run-record.json.tmp").exists()
    # Y el intento queda en disco: el siguiente desde el mismo commit se niega.
    with pytest.raises(SystemExit, match="ya existe"):
        rn.main(["x", "--caso", "A", "--referencia", str(tmp_path / "ref")])


# --------------------------------------------------------------------------- #
# Caso A: dientes sin contorno
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class _RegistradorQuieto:
    """Un registrador que devuelve siempre la identidad: basta para probar la contabilidad."""

    max_iter: int = 10

    def __call__(self, origen, destino):
        return RegistrationResult(rotation=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
                                  rms_mm=0.0, iterations=0, converged=True)


def test_un_diente_sin_contorno_sale_de_n1_y_queda_en_n2() -> None:
    corona = np.array([[0.0, 0, 0], [0.5, 0, 0], [50.0, 0, 0], [50.5, 0, 0]])
    etq_c = np.array([11, 11, 12, 12])
    destino = np.array([[0.1, 0.2, 0], [0.4, -0.2, 0], [0.2, 0.0, 0.3]])  # todo cerca del 11
    nav = rn.navaja_a(_RegistradorQuieto(), corona, etq_c, destino, np.eye(4), (11, 12))
    assert nav["teeth_without_contour"] == [12]
    por = nav["per_tooth"]["b"]
    assert por[12]["residual_full_mm"] is None and por[12]["movement_median_mm"] == 0.0
    n1, n2 = rn.evalua_n1(por), rn.evalua_n2(por)
    assert n1["teeth_used"] == [11] and n1["teeth_without_data"] == [12]
    assert n2["teeth_without_data"] == [] and n2["pass"] is True


# --------------------------------------------------------------------------- #
# Caso B: suelo de semilla, indeterminados y legibilidad, con números hechos a mano
# --------------------------------------------------------------------------- #
def test_lee_suelo_supera_no_supera_e_indeterminado() -> None:
    suelo = {x: [round(0.1 * k, 10) for k in range(11)] for x in (11, 12, 13)}  # p95 = 0.95
    suelo[14] = []
    por = {11: {"n_valid": 12, "movement_median_mm": 0.96},
           12: {"n_valid": 10, "movement_median_mm": 0.95},     # igual al p95: no supera
           13: {"n_valid": 9, "movement_median_mm": 5.0},       # < 10 pares válidos
           14: {"n_valid": 20, "movement_median_mm": 3.0},      # sin suelo
           15: {"n_valid": 20, "movement_median_mm": None}}
    lectura = rn.lee_suelo(por, suelo)
    assert lectura[11]["floor_p95_mm"] == pytest.approx(0.95)
    assert [lectura[x]["state"] for x in (11, 12, 13, 14, 15)] == [
        "supera", "no supera", "indeterminado", "indeterminado", "indeterminado"]
    assert lectura[11]["floor_n_seeds"] == 11


def test_suelo_b_por_diente_sin_la_semilla_0_ni_las_vueltas() -> None:
    V = np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [11.0, 0, 0], [5.0, 5, 0]])
    etq = np.array([11, 11, 21, 21, 0])

    def corrida(s, dx, vuelta=False):
        return {"seed": s, "pose": rn._4x4(np.eye(3), [dx, 0, 0]), "flip": vuelta}

    completos = [corrida(0, 9.0), corrida(1, 0.1), corrida(2, 0.3), corrida(3, 7.0, True),
                 corrida(4, 0.2)]
    suelo = rn.suelo_b(completos, np.eye(4), V, etq, (11, 21))
    assert suelo["seeds"] == [1, 2, 3, 4] and suelo["flips"] == [3]
    assert suelo["per_tooth_values_mm"][11] == pytest.approx([0.1, 0.3, 0.2])
    assert suelo["per_tooth_p95_mm"][21] == pytest.approx(np.percentile([0.1, 0.3, 0.2], 95))


def test_mas_de_la_mitad_de_vueltas_deja_el_caso_b_ilegible() -> None:
    def filas(k):
        return [{"flip": i < k} for i in range(20)]

    assert rn.legibilidad_b(filas(10)) == {"readable": True, "n_full_scratch": 20,
                                           "flips_full_scratch": 10}
    assert rn.legibilidad_b(filas(11))["readable"] is False
    assert rn.legibilidad_b([])["readable"] is False
    assert rn.tasa(filas(5)) == {"n": 20, "flips": 5, "rate": 0.25}


def test_k1_de_b_una_referencia_vuelta_hace_fallar_sus_arranques(caso_b_sintetico) -> None:
    d = caso_b_sintetico
    T0 = rn.registra_b(d.campo, d.hu, d.V, 0)["pose"]
    coronas = d.V[d.etq > 0]
    ref = rn._marca(rn.registra_b(d.campo, d.hu, d.V, 0, previa=T0), T0, coronas)
    ref_vuelta = {**ref, "flip": True}
    pert = rn.perturbaciones_b(rn.aplica(T0, coronas).mean(axis=0), angulos=(2.0,),
                               direcciones=rn.DIRECCIONES[:1], traslaciones=(), ejes=())
    filas = rn.arranques_b(d.campo, d.hu, d.V, d.etq, T0, {0: ref_vuelta}, pert,
                           semillas=(0,))
    assert filas[0]["reference_flip"] and not filas[0]["run_flip"] and filas[0]["flip"]
    k1 = rn.evalua_k1(filas)
    assert k1["pass"] is False and k1["failures_by_flip"] == 1


def test_main_se_niega_si_la_salida_no_lleva_caso_y_commit(disciplina, tmp_path) -> None:
    with pytest.raises(SystemExit, match="tiene que llamarse"):
        rn.main(["x", "--caso", "A", "--referencia", str(_referencia(tmp_path)),
                 "--salida", str(tmp_path / "otra")])
    assert not (tmp_path / "otra").exists()


def test_un_fallo_al_cargar_los_datos_queda_en_disco(disciplina, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rn, "comprueba_referencia", lambda ref, caso: (Path("caso.uos"), {}))

    def falla(*a):
        raise SystemExit("artefacto ambiguo")

    monkeypatch.setattr(rn, "carga_referencia", falla)
    with pytest.raises(SystemExit, match="ambiguo"):
        rn.main(["x", "--caso", "A", "--referencia", str(tmp_path / "ref")])
    final = json.loads((disciplina / rn.SALIDAS / f"A-{HEAD[:12]}" / "run-record.json")
                       .read_text())
    assert final["status"] == "error" and final["error_type"] == "SystemExit"
    assert final["inputs"] is None and final["finished_at"] is not None
