"""Líneas marcadas a mano contra nuestro margen. Ver `analysis_agents.margen_anotado`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from analysis_agents.margen_anotado import (
    acuerdo_entre_anotaciones,
    comprueba_sobre_malla,
    distancia_a_las_curvas,
    etiquetas_de_curvas,
    fdi_de_fichero,
    lee_anotacion,
    lee_curva,
)

PASO = 0.25
CENTRO = np.array([5.0, 5.0, 0.0])


def _lamina(n: int = 41):
    """Una lámina de 10 × 10 mm triangulada, con el origen en una esquina."""
    idx = np.arange(n * n).reshape(n, n)
    x, y = np.meshgrid(np.arange(n) * PASO, np.arange(n) * PASO, indexing="ij")
    pos = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], 1)
    caras = np.concatenate([
        np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
    ])
    return pos, caras


def _circulo(radio: float, centro: np.ndarray = CENTRO, puntos: int = 24) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, puntos, endpoint=False)
    return centro + radio * np.stack([np.cos(a), np.sin(a), np.zeros(puntos)], 1)


def _disco(pos: np.ndarray, radio: float, fdi: int = 16) -> np.ndarray:
    return np.where(np.linalg.norm(pos - CENTRO, axis=1) <= radio, fdi, 0)


def _escribe(ruta: Path, puntos: np.ndarray, *, tipo: str = "ClosedCurve",
             sistema: str = "LPS") -> Path:
    ruta.write_text(json.dumps({"markups": [{
        "type": tipo, "coordinateSystem": sistema,
        "controlPoints": [{"label": f"C-{k}", "position": p.tolist(),
                           "positionStatus": "defined"} for k, p in enumerate(puntos)]}]}))
    return ruta


def test_la_linea_encierra_la_region_que_rodea() -> None:
    pos, tri = _lamina()
    etq, registros = etiquetas_de_curvas(pos, tri, {16: _circulo(2.5)})
    d = np.linalg.norm(pos - CENTRO, axis=1)
    assert registros["per_tooth"]["16"]["usable"]
    assert (etq[d < 2.5 - 2 * PASO] == 16).all()
    assert (etq[d > 2.5 + 2 * PASO] == 0).all()
    assert registros["disputed_vertices"] == 0


@pytest.mark.parametrize(("radio", "signo"), [(3.5, 1), (1.5, -1)])
def test_la_distancia_lleva_el_signo_del_error(radio: float, signo: int) -> None:
    """Un margen 1 mm por fuera de la línea es desborde (+); 1 mm por dentro, recorte (−)."""
    pos, tri = _lamina()
    curvas = {16: _circulo(2.5)}
    anotado, _ = etiquetas_de_curvas(pos, tri, curvas)
    r = distancia_a_las_curvas(pos, tri, _disco(pos, radio), curvas, anotado)
    pieza = r["per_tooth"]["16"]
    assert pieza["pred_to_curve"]["p50_mm"] == pytest.approx(1.0, abs=0.3)
    assert pieza["curve_to_pred"]["p50_mm"] == pytest.approx(1.0, abs=0.3)
    assert np.sign(pieza["signed_mean_mm"]) == signo
    assert r["only_annotated"] == [] and r["only_predicted"] == []


def test_un_margen_sobre_la_linea_dista_menos_que_una_arista() -> None:
    pos, tri = _lamina()
    curvas = {16: _circulo(2.5)}
    anotado, _ = etiquetas_de_curvas(pos, tri, curvas)
    r = distancia_a_las_curvas(pos, tri, anotado, curvas, anotado)
    assert r["pred_to_curve"]["p95_mm"] < 2 * PASO


def test_una_linea_que_no_encierra_nada_se_declara() -> None:
    """Ida y vuelta por el mismo camino: es un lazo, pero dentro no queda superficie."""
    pos, tri = _lamina()
    recta = np.array([[2.0, 5.0, 0], [4.0, 5.0, 0], [6.0, 5.0, 0], [4.0, 5.0, 0]])
    etq, registros = etiquetas_de_curvas(pos, tri, {11: recta})
    assert not registros["per_tooth"]["11"]["usable"]
    assert registros["per_tooth"]["11"]["reason"]
    assert not etq.any()


def test_una_pieza_sin_region_se_mide_sin_signo() -> None:
    pos, tri = _lamina()
    curvas = {16: _circulo(2.5)}
    anotado = np.zeros(len(pos), dtype=np.int64)
    r = distancia_a_las_curvas(pos, tri, _disco(pos, 3.5), curvas, anotado, con_region=set())
    assert r["per_tooth"]["16"]["signed_mean_mm"] is None
    assert r["per_tooth"]["16"]["pred_to_curve"]["p50_mm"] == pytest.approx(1.0, abs=0.3)


def test_el_contacto_entre_dientes_no_cuenta_como_margen() -> None:
    """Dos piezas pegadas: en la cara común no hay encía, y la línea pasa por ahí.

    Sin excluir ese tramo, su centro quedaría a 1,5 mm del margen gingival más cercano y
    un etiquetado perfecto parecería malo.
    """
    pos, tri = _lamina()

    def cuadro(x0: float, x1: float) -> np.ndarray:
        lado = np.linspace(0, 1, 7)[:-1]
        esquinas = np.array([[x0, 3.5, 0], [x1, 3.5, 0], [x1, 6.5, 0], [x0, 6.5, 0]])
        return np.concatenate([a + (b - a) * lado[:, None]
                               for a, b in zip(esquinas, np.roll(esquinas, -1, 0),
                                               strict=True)])

    curvas = {16: cuadro(2.0, 5.0), 17: cuadro(5.0, 8.0)}
    anotado, registros = etiquetas_de_curvas(pos, tri, curvas)
    assert all(r["usable"] for r in registros["per_tooth"].values())
    r = distancia_a_las_curvas(pos, tri, anotado, curvas, anotado)
    assert r["curve_to_pred"]["max_mm"] < 1.0


def test_dos_anotaciones_distan_lo_que_distan_sus_lineas() -> None:
    r = acuerdo_entre_anotaciones({16: _circulo(2.5, puntos=60), 11: _circulo(1.0)},
                                  {16: _circulo(2.8, puntos=60), 21: _circulo(1.0)})
    assert r["per_tooth"]["16"]["p50_mm"] == pytest.approx(0.3, abs=0.05)
    assert r["only_first"] == [11] and r["only_second"] == [21]


def test_el_fichero_ras_se_lee_en_lps(tmp_path: Path) -> None:
    puntos = _circulo(2.5)
    lps = lee_curva(_escribe(tmp_path / "16.mrk.json", puntos))
    ras = lee_curva(_escribe(tmp_path / "17.mrk.json", puntos * [-1, -1, 1], sistema="RAS"))
    assert np.allclose(lps, puntos) and np.allclose(ras, puntos)


def test_una_curva_abierta_o_unos_puntos_sueltos_no_valen(tmp_path: Path) -> None:
    for tipo in ("Curve", "Fiducial"):
        with pytest.raises(ValueError, match="curva cerrada"):
            lee_curva(_escribe(tmp_path / "16.mrk.json", _circulo(2.5), tipo=tipo))
    with pytest.raises(ValueError, match="no cierran"):
        lee_curva(_escribe(tmp_path / "16.mrk.json", _circulo(2.5, puntos=3)))


@pytest.mark.parametrize(("nombre", "fdi"), [("16.mrk.json", 16), ("diente_27.mrk.json", 27),
                                             ("obs1-11-v2.mrk.json", 11)])
def test_el_nombre_del_fichero_dice_la_pieza(nombre: str, fdi: int) -> None:
    assert fdi_de_fichero(Path(nombre)) == fdi


@pytest.mark.parametrize("nombre", ["CC.mrk.json", "16-17.mrk.json", "19.mrk.json",
                                    "116.mrk.json"])
def test_un_nombre_sin_una_sola_pieza_no_se_adivina(nombre: str) -> None:
    with pytest.raises(ValueError, match="FDI"):
        fdi_de_fichero(Path(nombre))


def test_el_stl_cargado_como_ras_invierte_x_e_y(tmp_path: Path) -> None:
    _escribe(tmp_path / "16.mrk.json", _circulo(2.5))
    como_lps = lee_anotacion(tmp_path, "lps")[16]
    como_ras = lee_anotacion(tmp_path, "ras")[16]
    assert np.allclose(como_ras, como_lps * [-1, -1, 1])


def test_el_convenio_equivocado_se_rechaza_y_se_dice_cual_es() -> None:
    pos, _ = _lamina()
    assert comprueba_sobre_malla(pos, {16: _circulo(2.5)})["control_to_mesh_p50_mm"] < PASO
    with pytest.raises(ValueError, match="convenio contrario .* sí caen"):
        comprueba_sobre_malla(pos, {16: _circulo(2.5) * [-1, -1, 1]})
    with pytest.raises(ValueError, match="otro escaneo"):
        comprueba_sobre_malla(pos, {16: _circulo(2.5) + [0, 0, 50]})
