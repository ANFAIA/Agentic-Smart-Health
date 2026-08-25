"""El ajuste tiene que recuperar geometría que se conoce, no sólo bajar una pérdida."""

from __future__ import annotations

import numpy as np
import pytest
from gaussian_engine import (
    Ajuste,
    ajusta,
    ajusta_por_region,
    evalua,
    siembra_por_rejilla,
)

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


# --- el ajuste por region ---------------------------------------------------- #
def _ejes_mayores(r: Ajuste) -> np.ndarray:
    """Direccion del semieje mayor de cada gaussiana, en el mundo."""
    w, x, y, z = r.rotations.T
    rot = np.empty((len(r.rotations), 3, 3))
    rot[:, 0] = np.column_stack([1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)])
    rot[:, 1] = np.column_stack([2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)])
    rot[:, 2] = np.column_stack([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)])
    return rot[np.arange(len(rot)), :, np.argmax(r.scales, axis=1)]


def _dos_regiones() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dos elipsoides que se tocan, con los ejes PERPENDICULARES entre si.

    Es la geometria de una raiz contra el hueso alveolar que la rodea, y las orientaciones
    cruzadas son lo que hace la prueba concluyente: dos regiones con la misma forma no
    pueden demostrar que el ajuste no las mezclo, porque mezclarlas daria el mismo
    resultado. Con ejes perpendiculares, un elipsoide que hubiera visto puntos de las dos
    saldria orientado a medio camino, y se nota.

    ⚠️ **Se tocan, no se atraviesan.** El desplazamiento deja las dos cajas compartiendo
    frontera y ni un solo punto. Interpenetradas, "de quien es esta gaussiana" no tendria
    respuesta correcta —las dos nubes ocuparian el mismo volumen— y tampoco seria el caso
    real: dos tejidos se tocan, no se ocupan.
    """
    eje_a, eje_b = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    a, da = _elipsoide(2_000, (2.0, 0.5, 0.5), eje_a)      # largo en z
    b, db = _elipsoide(2_000, (2.0, 0.5, 0.5), eje_b)      # largo en x
    b = b + np.array([0.0, 0.0, 7.5])                      # apilados, sin solapar
    return (np.vstack([a, b]), np.concatenate([da, db]),
            np.array([11] * 2_000 + [0] * 2_000))


def test_cada_gaussiana_SOLO_VIO_los_puntos_de_su_region():
    """La razon de ser de todo esto. Ajustando global habria que heredar el `region_id` del
    vecino mas cercano, y una etiqueta heredada se ve identica a una medida: el visor
    pintaria una raiz con el color de su diente sin que nadie lo haya comprobado.

    Se comprueba por la ORIENTACION, y con UNA gaussiana por region. Las dos regiones son
    elipsoides perpendiculares, asi que la unica gaussiana de cada una tiene que salir
    alineada con el eje de SU region; una que hubiera visto puntos de las dos saldria a
    medio camino.

    ⚠️ Con muchas gaussianas por region esta prueba no diria nada, y conviene saber por
    que: medido, con cien por region la anisotropia de cada una baja a 1,27 y su eje mayor
    es ruido. No es un fallo — la forma alargada de la region la representan muchas
    gaussianas EN FILA, no cada una siendo alargada. La orientacion individual solo
    significa algo cuando una sola gaussiana carga con la region entera.
    """
    pts, d, reg = _dos_regiones()

    r = ajusta_por_region(pts, d, reg, compresion=2_000.0, compresion_region=2_000.0,
                          minimo=1, iteraciones=1_500, tasa=0.05, dispositivo="cpu")

    assert r.region_id is not None
    assert set(np.unique(r.region_id).tolist()) == {0, 11}

    ejes = _ejes_mayores(r)
    for codigo, esperado in ((11, np.array([0.0, 0.0, 1.0])), (0, np.array([1.0, 0.0, 0.0]))):
        m = r.region_id == codigo
        assert m.sum() == 1, f"region {codigo}: {m.sum()} gaussianas, se esperaba una"
        escalas = np.sort(r.scales[m][0])[::-1]
        assert escalas[0] / escalas[1] > 2.0, f"region {codigo}: escalas {escalas}"
        assert abs(float(ejes[m][0] @ esperado)) > 0.9, (
            f"region {codigo}: coseno {abs(float(ejes[m][0] @ esperado)):.2f} con su eje"
        )


def test_las_gaussianas_que_se_salen_de_su_region_se_salen_EN_LA_FRONTERA():
    """El comportamiento real, escrito para que no sorprenda a nadie.

    El optimizador mueve el centro libremente, asi que una gaussiana de borde puede acabar
    unas decimas al otro lado de la frontera. No es un error de etiqueta —se ajusto solo
    con puntos de su region— sino que un elipsoide que representa el borde tiene su centro
    cerca del borde. Lo que NO puede pasar es que aparezca una en mitad de la otra region:
    eso si seria una etiqueta mal puesta.
    """
    from scipy.spatial import cKDTree

    pts, d, reg = _dos_regiones()
    frontera = pts[reg == 0][:, 2].min()      # las dos cajas se tocan en este plano

    r = ajusta_por_region(pts, d, reg, compresion=20.0, minimo=8, iteraciones=400,
                          dispositivo="cpu")

    assert r.region_id is not None
    for codigo in (0, 11):
        propias = r.centers[r.region_id == codigo]
        d_propia, _ = cKDTree(pts[reg == codigo]).query(propias)
        d_ajena, _ = cKDTree(pts[reg != codigo]).query(propias)
        invasoras = propias[d_propia > d_ajena]
        if not len(invasoras):
            continue
        assert np.abs(invasoras[:, 2] - frontera).max() < 1.0, (
            f"region {codigo}: hay una gaussiana a "
            f"{np.abs(invasoras[:, 2] - frontera).max():.2f} mm de la frontera, "
            "eso no es efecto de borde"
        )


def test_la_etiqueta_sobrevive_al_artefacto():
    """Sin `region_id` en el .npz el visor no puede seleccionar por pieza."""
    pts, d, reg = _dos_regiones()
    r = ajusta_por_region(pts, d, reg, compresion=20.0, minimo=8, iteraciones=50,
                          dispositivo="cpu")

    art = r.como_artefacto()
    assert art["region_id"].dtype == np.int16
    assert len(art["region_id"]) == len(art["centers"])


def test_ninguna_region_se_queda_sin_representacion():
    """Una pieza representada por cuatro elipsoides deja de poder seleccionarse."""
    pts, d, reg = _dos_regiones()
    r = ajusta_por_region(pts, d, reg, compresion=500.0, minimo=64, iteraciones=20,
                          dispositivo="cpu")

    assert r.region_id is not None
    for codigo in (0, 11):
        assert (r.region_id == codigo).sum() >= 64


def test_el_error_se_mide_sobre_la_UNION_no_region_a_region():
    """Las regiones se ajustan por separado pero se renderizan juntas: en la frontera un
    punto recibe tambien las gaussianas vecinas. Medir region a region esconderia ese
    exceso, asi que el `rmse` declarado es el de la union."""
    pts, d, reg = _dos_regiones()
    r = ajusta_por_region(pts, d, reg, compresion=20.0, minimo=8, iteraciones=200,
                          dispositivo="cpu")

    union = evalua(pts, r.centers, r.scales, r.rotations, r.density)
    assert r.rmse == pytest.approx(float(np.sqrt(((union - d) ** 2).mean())), rel=1e-6)
    # Y el desglose esta, para poder ver DONDE duele.
    assert set(r.rmse_hu_por_region) == {0, 11}


def test_comprimir_por_debajo_de_uno_se_declara():
    """Pediria mas gaussianas que semillas, que no es ajustar sino duplicar."""
    pts, d, reg = _dos_regiones()
    with pytest.raises(ValueError, match="duplicar"):
        ajusta_por_region(pts, d, reg, compresion=0.5, dispositivo="cpu")


def test_arrays_descuadrados_se_declaran():
    with pytest.raises(ValueError, match="punto a punto"):
        ajusta_por_region(np.zeros((10, 3)), np.zeros(10), np.zeros(9), dispositivo="cpu")


def test_el_fondo_se_comprime_y_las_piezas_con_nombre_NO():
    """El reparto asimetrico, que es la politica y no un detalle de ajuste.

    Comprimir todo por igual costaba la resolucion justo donde importa: medido sobre el
    caso real, con 13 uniforme las raices bajaban a 7.902 gaussianas y su espaciado subia a
    0,883 mm — una raiz mide ~4 mm de ancho, o sea cinco gaussianas de lado a lado. El
    espaciado va con n^(-1/3), asi que comprimir y resolver son la misma moneda. En el
    fondo se gasta; en los dientes no.
    """
    pts, d, reg = _dos_regiones()

    r = ajusta_por_region(pts, d, reg, compresion=20.0, compresion_region=1.0,
                          minimo=1, iteraciones=20, dispositivo="cpu")

    assert r.region_id is not None
    con_nombre = int((r.region_id == 11).sum())
    fondo = int((r.region_id == 0).sum())
    assert con_nombre > 10 * fondo, f"con nombre {con_nombre}, fondo {fondo}"
    # Sin comprimir, la region con nombre conserva del orden de sus semillas.
    assert con_nombre > 0.9 * int((reg == 11).sum())


def test_comprimir_las_piezas_por_debajo_de_uno_se_declara():
    pts, d, reg = _dos_regiones()
    with pytest.raises(ValueError, match="compresion_region"):
        ajusta_por_region(pts, d, reg, compresion_region=0.5, dispositivo="cpu")

