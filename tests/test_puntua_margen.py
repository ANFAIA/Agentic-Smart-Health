"""El script de puntuación lee candidatos y declara el alcance de lo que mide."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "puntua_margen.py"
_spec = importlib.util.spec_from_file_location("puntua_margen", _RUTA)
assert _spec and _spec.loader
pm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pm
_spec.loader.exec_module(pm)

CENTRO = np.array([5.0, 5.0, 0.0])


def _lamina(n: int = 41, paso: float = 0.25):
    idx = np.arange(n * n).reshape(n, n)
    x, y = np.meshgrid(np.arange(n) * paso, np.arange(n) * paso, indexing="ij")
    pos = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], 1)
    caras = np.concatenate([
        np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
    ])
    return pos, caras


def _circulo(radio: float) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    return CENTRO + radio * np.stack([np.cos(a), np.sin(a), np.zeros(24)], 1)


def _disco(pos: np.ndarray, radio: float) -> np.ndarray:
    return np.where(np.linalg.norm(pos - CENTRO, axis=1) <= radio, 16, 0)


def test_lee_un_candidato_npz(tmp_path: Path) -> None:
    pos, tri = _lamina()
    np.savez(tmp_path / "e.npz", V=pos, F=tri, despues=_disco(pos, 2.5))
    nombre, p, t, etq = pm.lee_candidato(f"pliegue={tmp_path / 'e.npz'}:despues")
    assert nombre == "pliegue" and p.shape == pos.shape and (etq > 0).any()
    for malo in ("sin_igual", f"x={tmp_path / 'e.npz'}", "x=fichero.stl:clave"):
        with pytest.raises(SystemExit):
            pm.lee_candidato(malo)


def test_el_que_desborda_y_el_que_recorta_no_se_confunden() -> None:
    pos, tri = _lamina()
    r = pm.puntua(pos, tri, {"ancho": _disco(pos, 3.5), "corto": _disco(pos, 1.5)},
                  {"obs1": {16: _circulo(2.5)}})
    ancho, corto = r["scores"]["obs1"]["ancho"], r["scores"]["obs1"]["corto"]
    assert ancho["labels"]["false_tooth_rate"] > 0 and ancho["labels"]["false_gum_rate"] == 0
    assert corto["labels"]["false_gum_rate"] > 0.5
    assert corto["labels"]["teeth"]["under_retained"] == 1
    assert ancho["curves"]["per_tooth"]["16"]["signed_mean_mm"] > 0
    assert corto["curves"]["per_tooth"]["16"]["signed_mean_mm"] < 0
    json.dumps(r)


def test_el_informe_declara_alcance_y_anotacion_unica() -> None:
    pos, tri = _lamina()
    r = pm.puntua(pos, tri, {"actual": _disco(pos, 3.5)}, {"obs1": {16: _circulo(2.5)}})
    texto = pm.informe_md(r)
    assert "No es validación clínica" in texto
    assert "Una sola anotación" in texto
    assert "| 16 |" in texto


def test_con_dos_anotaciones_se_informa_el_suelo() -> None:
    pos, tri = _lamina()
    r = pm.puntua(pos, tri, {"actual": _disco(pos, 2.5)},
                  {"obs1": {16: _circulo(2.5)}, "obs2": {16: _circulo(2.7)}})
    par = r["between_annotations"]["obs1 / obs2"]
    assert par["overall"]["p50_mm"] == pytest.approx(0.2, abs=0.05)
    assert "no se distingue de correcto" in pm.informe_md(r)
