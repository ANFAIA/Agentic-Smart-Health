"""Registro por cortes: pose inicial anatómica y solo la banda por encima de la mordida."""

from __future__ import annotations

import numpy as np
import pytest
from fusion_agents.por_cortes import (
    EJES_LPS,
    RegistroPorCortes,
    SerieDensa,
    banda,
    contornos,
    describe_cortes,
    motivos,
    pose_inicial,
    prepara_por_cortes,
)
from fusion_agents.registration import quaternion_to_matrix

# Semiejes de cada corona y altura de su centro sobre el plano oclusal: la cúspide queda
# en −0,5 y la parte más ancha por encima de la banda, como en una corona real, que se
# estrecha hacia la cúspide. Con el ecuador dentro de la banda las paredes serían
# verticales y la altura no se podría medir.
SEMI = np.array([2.4, 2.4, 6.5])
CENTRO_S = 6.0


def _rotacion(eje: np.ndarray, grados: float) -> np.ndarray:
    eje = eje / np.linalg.norm(eje)
    k = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]])
    a = np.radians(grados)
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def _centros(arcada: str = "upper") -> np.ndarray:
    """Centros de 10 coronas en LPS, sobre una parábola: incisivos delante (−y)."""
    t = np.linspace(-1.2, 1.2, 10)
    derecha, anterior = 22 * np.sin(t), 20 * np.cos(t) - 10
    s = CENTRO_S if arcada == "upper" else -CENTRO_S
    return np.c_[-derecha, -anterior, np.full(10, s)]


def _superficie(arcada: str = "upper", n: int = 1500) -> np.ndarray:
    """Coronas del escáner en el mundo: elipsoides sin el casquete de la encía."""
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
    u = np.c_[np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    puntos = np.vstack([c + u * SEMI for c in _centros(arcada)])
    # La encía tapa la mitad de la corona que mira a la raíz.
    hacia_raiz = puntos[:, 2] - np.repeat(_centros(arcada)[:, 2], n)
    return puntos[hacia_raiz < 2.5] if arcada == "upper" else puntos[hacia_raiz > -2.5]


def _dentro(p: np.ndarray, arcada: str) -> np.ndarray:
    d = (p[:, None, :] - _centros(arcada)[None]) / SEMI
    return ((d**2).sum(-1) <= 1).any(1)


def _campo(alturas: list[float], paso: float = 0.3) -> dict[str, np.ndarray]:
    """Campo como el del `cbct-agent`: solo tejido duro, una rejilla por corte."""
    xs = np.arange(-30, 30, paso)
    gx, gy = np.meshgrid(xs, np.arange(-20, 20, paso))
    capas = []
    for z in alturas:
        p = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z)]
        capas.append(p[_dentro(p, "upper") | _dentro(p, "lower")])
    # Un hueso denso lejos de la arcada, a la altura de las coronas: no debe entrar.
    hueso = np.c_[np.full(400, 0.0) + np.repeat(np.arange(20) * paso, 20),
                  np.full(400, 25.0) + np.tile(np.arange(20) * paso, 20),
                  np.full(400, alturas[len(alturas) // 2])]
    centros = np.vstack([*capas, hueso])
    hu = np.r_[np.full(len(centros) - len(hueso), 2000.0), np.full(len(hueso), 1500.0)]
    origen = np.array([5.0, -3.0, 1.0])
    return {"centers": centros - origen, "origin": origen}, hu


# Cortes en parejas con huecos de 1,5-3,7 mm entre ellas, como la serie dispersa real.
DISPERSA = [-4.0, -3.7, -2.0, -1.7, 0.0, 0.3, 2.0, 2.3, 4.1, 4.4, 8.0, 8.3]


def _escaner(rot_real: np.ndarray, trans_real: np.ndarray) -> np.ndarray:
    """Las coronas en el marco del escáner: mundo = rot_real · escáner + trans_real."""
    return (_superficie() - trans_real) @ rot_real


def test_pose_inicial_lleva_los_ejes_del_escaner_a_los_del_dicom() -> None:
    rot = _rotacion(np.array([0.3, -1.0, 0.4]), 70)
    ejes_escaner = EJES_LPS @ rot  # cada eje del paciente, visto desde el escáner
    np.testing.assert_allclose(pose_inicial(ejes_escaner), rot, atol=1e-12)


def test_un_escaner_en_espejo_no_se_registra() -> None:
    espejo = EJES_LPS * np.array([[-1.0], [1.0], [1.0]])
    with pytest.raises(ValueError, match="espejo"):
        pose_inicial(espejo)


def test_la_banda_va_hacia_la_raiz_de_la_arcada_escaneada() -> None:
    lo, hi = banda("upper", 30.0)
    assert 30.0 < lo < hi
    lo, hi = banda("lower", 30.0)
    assert lo < hi < 30.0
    with pytest.raises(ValueError):
        banda("ambas", 30.0)


def test_una_serie_con_huecos_se_declara_dispersa_y_una_continua_no() -> None:
    assert describe_cortes(np.array(DISPERSA), 1.5, 6.0)["sparse"]
    continua = describe_cortes(np.arange(-4, 10, 0.3), 1.5, 6.0)
    assert not continua["sparse"] and continua["slices_in_band"] == 15


def test_el_contorno_es_el_borde_del_diente_no_su_interior() -> None:
    campo, hu = _campo([2.0])
    mundo = campo["centers"] + campo["origin"]
    borde = contornos(mundo, hu, 1400.0, 1.0, 3.0)
    # Cada punto de borde está a menos de un paso de rejilla de la elipse del corte.
    radio = SEMI[0] * np.sqrt(1 - ((2.0 - CENTRO_S) / SEMI[2]) ** 2)
    d = np.linalg.norm(borde[:, None, :2] - _centros()[None, :, :2], axis=-1).min(1)
    dientes = borde[:, 1] < 20  # el hueso lejano también tiene borde; se filtra después
    assert dientes.sum() > 100
    # Llevado medio paso hacia fuera, el borde cae sobre la elipse, no medio paso dentro.
    assert abs(float(np.median(d[dientes] - radio))) < 0.08
    assert np.all(np.abs(d[dientes] - radio) < 0.35)


def test_recupera_la_pose_desde_ejes_algo_torcidos() -> None:
    """Los ejes medidos en el escáner nunca son exactos: 5° de error se corrigen."""
    rot_real = _rotacion(np.array([1.0, 2.0, -0.5]), 110)
    trans_real = np.array([12.0, -40.0, 33.0])
    escaner = _escaner(rot_real, trans_real)

    # Contornos densos: la elipse exacta en cada corte de la banda.
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    anillos = []
    for z in (2.0, 2.3, 4.1, 4.4):
        r = SEMI[:2] * np.sqrt(1 - ((z - CENTRO_S) / SEMI[2]) ** 2)
        for c in _centros():
            anillos.append(
                np.c_[c[0] + r[0] * np.cos(t), c[1] + r[1] * np.sin(t), np.full(len(t), z)]
            )
    contorno = np.vstack(anillos)

    torcido = _rotacion(np.array([0.2, 1.0, 0.7]), 5) @ rot_real
    registro = RegistroPorCortes(rot0=torcido, plano=0.0, arcada="upper")
    r = registro(escaner, contorno)

    movido = escaner @ quaternion_to_matrix(r.rotation).T + np.asarray(r.translation)
    error = movido - _superficie()
    # En planta es casi exacto; en altura, peor, porque con cortes horizontales y paredes
    # casi verticales subir la corona apenas cambia su sección. Es la incertidumbre que
    # el gate declara, y por eso se comprueba aparte.
    assert np.median(np.linalg.norm(error[:, :2], axis=1)) < 0.1
    assert abs(float(error[:, 2].mean())) < 0.3
    assert r.overlap_fraction is not None and r.overlap_fraction > 0.95


def test_prepara_registra_una_serie_dispersa_y_declara_lo_que_no_sabe() -> None:
    campo, hu = _campo(DISPERSA)
    rot_real = _rotacion(np.array([0.0, 1.0, 1.0]), 40)
    trans_real = np.array([3.0, 2.0, -5.0])
    escaner = _escaner(rot_real, trans_real)

    origen, destino, registrador, informe = prepara_por_cortes(
        campo, hu, escaner, EJES_LPS @ rot_real, "upper", plano=0.0
    )
    assert informe["sparse"] and informe["slices_in_band"] == 4
    # El hueso denso lejos de la arcada no entra en el objetivo.
    assert not np.any(destino[:, 1] > 20)
    assert "threshold_spread_mm" in informe

    r = registrador(origen, destino)
    movido = origen @ quaternion_to_matrix(r.rotation).T + np.asarray(r.translation)
    error = movido - _superficie()
    assert np.median(np.linalg.norm(error[:, :2], axis=1)) < 0.15
    # ⚠️ Con un campo en rejilla la altura queda sesgada hacia la raíz, ~1,3 pasos: 0,37
    # mm con 0,3 mm de rejilla, 0,2 con 0,15. Es el mismo sentido que el desfase residual
    # del caso real y la razón de que el gate declare la altura como lo peor medido.
    assert 0.0 < float(error[:, 2].mean()) < 0.5

    texto = " ".join(motivos(informe))
    assert "dispersa" in texto and "umbral" in texto and "no es un TRE" in texto


def test_una_serie_continua_se_deja_al_camino_de_siempre() -> None:
    """Medido en `histora`: con la serie completa el camino de siempre registra mejor."""
    campo, hu = _campo(list(np.arange(-4, 9, 0.3)))
    with pytest.raises(SerieDensa):
        prepara_por_cortes(campo, hu, _superficie(), EJES_LPS, "upper", plano=0.0)
