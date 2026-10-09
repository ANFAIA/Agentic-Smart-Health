"""TRE con puntos externos: medir sobre lo que el registro no usó, sin adivinar convenios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fusion_agents.registration import quaternion_to_matrix
from fusion_agents.tre import (
    a_coordenadas_stl,
    consenso,
    lee_puntos,
    region,
    tre_de_registro,
    tre_por_puntos,
)

ETIQUETAS = ["16-mv", "26-mv", "13-c", "23-c", "11-i", "21-i", "14-v", "24-v"]


def _escena(ruido_mm: float = 0.0, semilla: int = 0):
    rng = np.random.default_rng(semilla)
    cbct = {k: rng.uniform(-30, 30, 3) for k in ETIQUETAS}
    q = np.array([0.9, 0.1, -0.3, 0.2])
    rot = quaternion_to_matrix(tuple(q / np.linalg.norm(q)))
    t = np.array([5.0, -12.0, 40.0])
    ios = {k: rot @ v + t + rng.normal(0, ruido_mm, 3) for k, v in cbct.items()}
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, t
    return cbct, ios, m


def test_registro_exacto_da_tre_nulo() -> None:
    cbct, ios, m = _escena()
    r = tre_de_registro(cbct, ios, m)
    assert r["n"] == len(ETIQUETAS) and r["max_mm"] < 1e-9


def test_registro_desplazado_se_mide_entero() -> None:
    cbct, ios, m = _escena()
    m[:3, 3] += [0.0, 2.0, 0.0]
    r = tre_de_registro(cbct, ios, m)
    assert r["rms_mm"] == pytest.approx(2.0) and r["max_mm"] == pytest.approx(2.0)


def test_ruido_conocido_se_recupera_por_puntos() -> None:
    cbct, ios, _ = _escena(ruido_mm=0.3)
    r = tre_por_puntos(cbct, ios, ajuste=ETIQUETAS[:5])
    # Tres componentes de 0,3 mm dan unos 0,52 mm de error por punto.
    assert 0.2 < r["evaluation"]["rms_mm"] < 1.0
    assert r["evaluation"]["n"] == 3 and r["leave_one_out"]["n"] == len(ETIQUETAS)
    assert r["fiducial_residual"]["rms_mm"] < r["leave_one_out"]["rms_mm"]


def test_pocos_puntos_de_ajuste_falla() -> None:
    cbct, ios, _ = _escena()
    with pytest.raises(ValueError, match="al menos 3"):
        tre_por_puntos(cbct, ios, ajuste=ETIQUETAS[:2])


def test_etiquetas_sin_pareja_se_declaran() -> None:
    cbct, ios, m = _escena()
    cbct["17-mv"] = np.zeros(3)
    del ios["11-i"]
    r = tre_de_registro(cbct, ios, m)
    assert r["unpaired"] == {"only_cbct": ["11-i", "17-mv"], "only_ios": []}
    assert r["n"] == len(ETIQUETAS) - 1


def test_regiones_por_fdi() -> None:
    assert region("16-mv") == "Q1-posterior"
    assert region("21-i") == "Q2-anterior"
    assert region("espina-nasal") == "sin_region"
    assert region("19-x") == "sin_region"


def _mrk(ruta: Path, puntos: dict[str, list[float]], sistema: str | None) -> Path:
    marca: dict = {"controlPoints": [{"label": k, "position": v} for k, v in puntos.items()]}
    if sistema:
        marca["coordinateSystem"] = sistema
    ruta.write_text(json.dumps({"markups": [marca]}))
    return ruta


def test_ras_y_lps_dan_el_mismo_punto(tmp_path: Path) -> None:
    lps = lee_puntos(_mrk(tmp_path / "a.mrk.json", {"11-i": [1.0, 2.0, 3.0]}, "LPS"))
    ras = lee_puntos(_mrk(tmp_path / "b.mrk.json", {"11-i": [-1.0, -2.0, 3.0]}, "RAS"))
    assert np.allclose(lps["11-i"], ras["11-i"])


def test_sistema_ausente_o_contradictorio_falla(tmp_path: Path) -> None:
    sin = _mrk(tmp_path / "a.mrk.json", {"11-i": [1.0, 2.0, 3.0]}, None)
    with pytest.raises(ValueError, match="no declara"):
        lee_puntos(sin)
    assert np.allclose(lee_puntos(sin, "LPS")["11-i"], [1, 2, 3])
    con = _mrk(tmp_path / "b.mrk.json", {"11-i": [1.0, 2.0, 3.0]}, "LPS")
    with pytest.raises(ValueError, match="declara LPS"):
        lee_puntos(con, "RAS")


def test_csv_exige_sistema(tmp_path: Path) -> None:
    ruta = tmp_path / "p.csv"
    ruta.write_text("etiqueta,x,y,z\n11-i,1,2,3\n")
    with pytest.raises(ValueError, match="CSV"):
        lee_puntos(ruta)
    assert np.allclose(lee_puntos(ruta, "ras")["11-i"], [-1, -2, 3])


def test_stl_cargado_como_ras_invierte_xy() -> None:
    p = {"11-i": np.array([1.0, 2.0, 3.0])}
    assert np.allclose(a_coordenadas_stl(p, "LPS")["11-i"], [1, 2, 3])
    assert np.allclose(a_coordenadas_stl(p, "RAS")["11-i"], [-1, -2, 3])


def test_consenso_entre_observadores() -> None:
    a = {"11-i": np.array([0.0, 0.0, 0.0]), "21-i": np.array([10.0, 0.0, 0.0])}
    b = {"11-i": np.array([0.4, 0.0, 0.0]), "13-c": np.array([5.0, 5.0, 0.0])}
    media, disp = consenso([a, b])
    assert list(media) == ["11-i"] and np.allclose(media["11-i"], [0.2, 0, 0])
    assert disp["not_marked_by_all"] == ["13-c", "21-i"]
    assert disp["max_pairwise_mm"] == pytest.approx(0.4)
