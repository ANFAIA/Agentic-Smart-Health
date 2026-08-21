"""El ajuste tiene que recuperar geometría que se conoce, no sólo bajar una pérdida."""

from __future__ import annotations

import numpy as np
import pytest
from gaussian_engine import Ajuste, ajusta, evalua, siembra_por_rejilla

torch = pytest.importorskip("torch", reason="el ajuste necesita torch (extra `gpu`)")


def _marco(eje: np.ndarray) -> np.ndarray:
    """Base ortonormal cuya primera columna es `eje`."""
    u = eje / np.linalg.norm(eje)
    otro = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(u, otro)
    v /= np.linalg.norm(v)
    return np.column_stack([u, v, np.cross(u, v)])


def _elipsoide(
    n: int, sigma: tuple[float, float, float], eje: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Puntos que cubren un elipsoide conocido, y la densidad exacta en cada uno.

    ⚠️ Los puntos se reparten UNIFORMES en la caja que lo contiene, no muestreados de la
    propia gaussiana. La diferencia importa para lo que se quiere probar: si sólo hubiera
    puntos donde la densidad es alta, la periferia no restringiría nada y muchas formas
    ajustarían igual de bien. Cubriendo la caja, el elipsoide correcto es el único que
    explica a la vez el centro y los bordes.
    """
    rng = np.random.default_rng(0)
    s = np.asarray(sigma, dtype=np.float64)
    local = rng.uniform(-3.0 * s, 3.0 * s, (n, 3))
    return local @ _marco(eje).T, np.exp(-0.5 * ((local / s) ** 2).sum(axis=1))


# --- la siembra -------------------------------------------------------------- #
def test_la_siembra_es_reproducible_y_acierta_el_tamano():
    """Dos ejecuciones sobre el mismo caso tienen que dar el mismo campo: si la siembra
    fuera aleatoria, el gemelo cambiaría entre ejecuciones sin que cambiara el dato."""
    rng = np.random.default_rng(1)
    c = rng.uniform(-10, 10, (20_000, 3))
    d = rng.uniform(0, 1, 20_000)

    a, _, _ = siembra_por_rejilla(c, d, 500)
    b, _, _ = siembra_por_rejilla(c, d, 500)

    assert np.array_equal(a, b)
    assert 400 < len(a) < 620, f"{len(a)} gaussianas para un objetivo de 500"


def test_la_siembra_no_puede_pedir_mas_gaussianas_que_semillas():
    c = np.random.default_rng(2).uniform(-1, 1, (50, 3))
    medias, _, _ = siembra_por_rejilla(c, np.ones(50), 10_000)
    assert len(medias) <= 50


def test_un_campo_vacio_se_declara_en_vez_de_devolver_nada():
    with pytest.raises(ValueError, match="vacío"):
        siembra_por_rejilla(np.zeros((0, 3)), np.zeros(0), 10)


# --- el ajuste --------------------------------------------------------------- #
def test_recupera_la_anisotropia_de_una_gaussiana_que_se_conoce():
    """La prueba de fuego: si la nube ES un elipsoide 4:1, el ajuste tiene que encontrar
    esa proporción. Es exactamente lo que la semilla del CBCT no puede hacer, porque
    escribe la misma escala isótropa en todas."""
    eje = np.array([1.0, 1.0, 0.0])
    puntos, densidad = _elipsoide(6_000, (2.0, 0.5, 0.5), eje)

    r = ajusta(puntos, densidad, n_objetivo=1, iteraciones=2_000, tasa=0.05,
               dispositivo="cpu")

    s = np.sort(r.scales[0])[::-1]
    assert s[0] == pytest.approx(2.0, abs=0.3), f"escalas {s}"
    assert s[1] == pytest.approx(0.5, abs=0.2), f"escalas {s}"
    # Y el semieje mayor tiene que apuntar donde apunta el elipsoide de verdad.
    assert abs(float(_eje_mayor(r) @ (eje / np.linalg.norm(eje)))) > 0.95


def _eje_mayor(r: Ajuste) -> np.ndarray:
    """Dirección del semieje mayor de la primera gaussiana, en el mundo."""
    w, x, y, z = r.rotations[0]
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rot[:, int(np.argmax(r.scales[0]))]


def test_ajustar_reduce_el_error_frente_a_la_semilla_isotropa():
    """El campo semilla es el punto de partida honesto; el ajustado tiene que ser mejor
    reconstruyendo la densidad, o no vale la pena tenerlo."""
    puntos, densidad = _elipsoide(3_000, (2.0, 0.4, 0.4), np.array([0.0, 0.0, 1.0]))

    r = ajusta(puntos, densidad, n_objetivo=1, iteraciones=2_000, tasa=0.05,
               dispositivo="cpu")

    medias, amplitudes, paso = siembra_por_rejilla(puntos, densidad, 1)
    isotropa = evalua(
        puntos, medias, np.full((len(medias), 3), paso * 0.5),
        np.tile([1.0, 0.0, 0.0, 0.0], (len(medias), 1)), amplitudes,
    )
    rmse_isotropa = float(np.sqrt(((isotropa - densidad) ** 2).mean()))

    assert r.rmse < rmse_isotropa


def test_el_error_se_declara_en_HU_no_solo_normalizado():
    """`rmse` en [0,1] no le dice nada a nadie. El rango de HU viaja en el artefacto justo
    para poder deshacer la normalización."""
    puntos, d = _elipsoide(1_000, (1.0, 1.0, 1.0), np.array([0.0, 0.0, 1.0]))

    r = ajusta(puntos, d, n_objetivo=1, iteraciones=50, hu_range=(500.0, 2000.0),
               dispositivo="cpu")

    assert r.rmse_hu == pytest.approx(r.rmse * 1500.0)


def test_los_parametros_salen_siempre_validos():
    """Cuaternión unitario, sigma dentro de los límites físicos y amplitud positiva. Son
    invariantes de la representación, no preferencias: un cuaternión sin normalizar no es
    ninguna rotación, y una sigma negativa no es ninguna elipse."""
    puntos, d = _elipsoide(2_000, (1.5, 0.5, 0.8), np.array([1.0, 0.0, 1.0]))

    r = ajusta(puntos, d, n_objetivo=8, iteraciones=200, dispositivo="cpu")

    assert np.allclose(np.linalg.norm(r.rotations, axis=1), 1.0, atol=1e-5)
    assert (r.scales >= 0.02 - 1e-9).all() and (r.scales <= 3.0 + 1e-9).all()
    assert (r.density > 0).all()
    assert np.isfinite(r.centers).all()


def test_centros_y_densidades_descuadrados_se_declaran():
    with pytest.raises(ValueError, match="cada semilla"):
        ajusta(np.zeros((10, 3)), np.zeros(9), n_objetivo=2, dispositivo="cpu")


def test_el_artefacto_sale_con_las_claves_del_campo_semilla():
    """Tiene que caber en el mismo almacén y el mismo exportador sin casos especiales."""
    puntos, d = _elipsoide(500, (1.0, 1.0, 1.0), np.array([0.0, 0.0, 1.0]))
    r = ajusta(puntos, d, n_objetivo=4, iteraciones=20, dispositivo="cpu")

    art = r.como_artefacto()
    assert set(art) == {"centers", "scales", "rotations", "density"}
    assert all(v.dtype == np.float32 for v in art.values())
    assert art["rotations"].shape == (len(art["centers"]), 4)
