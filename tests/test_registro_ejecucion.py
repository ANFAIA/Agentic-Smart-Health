"""Una ejecución del complete case debe poder atribuirse a un código concreto.

Las seis ejecuciones del 7 de octubre se registraron con código ajeno al repositorio y un
hash de diff imposible de recalcular. Estas pruebas fijan lo que el ejecutor registra ahora
y que la comparación entre ejecuciones exige el mismo código en ambas.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _carga(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


cc = _carga("caso_completo")
vc = _carga("verifica_complete_case")


def _repo(tmp_path: Path) -> Path:
    raiz = tmp_path / "repo"
    (raiz / "scripts").mkdir(parents=True)
    (raiz / "scripts" / "a.py").write_text("x = 1\n")
    for orden in (["init", "-q"], ["add", "."],
                  ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"]):
        subprocess.run(["git", *orden], cwd=raiz, check=True)
    return raiz


def test_arbol_limpio_no_declara_diff(tmp_path: Path) -> None:
    raiz = _repo(tmp_path)
    rev = cc.revision_de_codigo(raiz)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert rev == {"git_head": head, "git_dirty": False, "git_diff_sha256": None,
                   "untracked_code": []}


def test_cambio_y_fichero_nuevo_ensucian_el_arbol(tmp_path: Path) -> None:
    raiz = _repo(tmp_path)
    (raiz / "scripts" / "a.py").write_text("x = 2\n")
    (raiz / "scripts" / "nuevo.py").write_text("y = 1\n")
    rev = cc.revision_de_codigo(raiz)
    assert rev["git_dirty"] is True
    assert len(rev["git_diff_sha256"]) == 64
    assert rev["untracked_code"] == ["scripts/nuevo.py"]


def test_cambios_fuera_del_codigo_no_cuentan(tmp_path: Path) -> None:
    raiz = _repo(tmp_path)
    (raiz / "notas.txt").write_text("borrador\n")
    assert cc.revision_de_codigo(raiz)["git_dirty"] is False


def test_sin_git_se_declara_desconocido(tmp_path: Path) -> None:
    rev = cc.revision_de_codigo(tmp_path)
    assert rev["git_head"] is None and rev["git_dirty"] is None


def test_el_registro_lleva_lo_que_identifica_la_ejecucion(tmp_path: Path) -> None:
    registro = cc.registro_ejecucion(["caso_completo.py", "--semilla", "7"], 7)
    assert registro["seed"] == 7
    assert registro["argv"][-1] == "7"
    assert registro["agents"] == {"cbct-agent": "0.2.0", "uos-export-agent": "0.18.0"}
    assert registro["uos_format"] == "0.3"
    assert {"python", "numpy", "scipy", "torch"} <= registro["environment"].keys()
    assert set(registro["code"]) == {"git_head", "git_dirty", "git_diff_sha256",
                                     "untracked_code"}
    ruta = cc.escribe_registro(registro, tmp_path / "salida")
    assert json.loads(ruta.read_text()) == registro


def _ejecucion(tmp_path: Path, nombre: str, code: dict | None) -> Path:
    uos = tmp_path / nombre / "export" / "caso.uos"
    uos.parent.mkdir(parents=True)
    if code is not None:
        (tmp_path / nombre / "run-record.json").write_text(
            json.dumps({"seed": 0, "code": code}))
    return uos


_LIMPIO = {"git_head": "abc", "git_dirty": False, "git_diff_sha256": None,
           "untracked_code": []}


def test_misma_revision_coincide(tmp_path: Path) -> None:
    a = _ejecucion(tmp_path, "a", _LIMPIO)
    b = _ejecucion(tmp_path, "b", dict(_LIMPIO))
    assert vc.revision_comparada(a, b)["same_code_revision"] is True


def test_diff_distinto_no_coincide(tmp_path: Path) -> None:
    a = _ejecucion(tmp_path, "a", {**_LIMPIO, "git_dirty": True, "git_diff_sha256": "1"})
    b = _ejecucion(tmp_path, "b", {**_LIMPIO, "git_dirty": True, "git_diff_sha256": "2"})
    assert vc.revision_comparada(a, b)["same_code_revision"] is False


def test_ejecucion_antigua_sin_registro_no_falla(tmp_path: Path) -> None:
    a = _ejecucion(tmp_path, "a", _LIMPIO)
    b = _ejecucion(tmp_path, "b", None)
    assert vc.revision_comparada(a, b) == {"code_revision": "unrecorded",
                                           "same_code_revision": None}
