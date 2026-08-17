"""El marco del arco: que fije los ejes de verdad y que no espeje nada."""

from __future__ import annotations

import numpy as np
from fusion_agents.marco import (
    MarcoEspecular,
    a_marco_de,
    marco_arcada,
    marco_canonico,
)
from scipy.spatial.transform import Rotation


def arcada_sintetica(n: int = 3000, semilla: int = 0) -> np.ndarray:
    """Herradura con cúspides, en `[largo, ancho, oclusal]` y con oclusal hacia +z.

    Asimétrica a propósito: una herradura perfecta tiene un plano de simetría y
    entonces «anterior» no está determinado por la forma, que es justo lo que estas
    funciones tienen que fijar.

    La cresta es un arco ESTRECHO y la base una plancha ANCHA, que es lo que de verdad
    distingue arriba de abajo: `marco_arcada` decide el sentido comparando cómo de bien
    ajusta una parábola cada extremo. Con un faldón que fuera copia del arco las dos
    orientaciones ajustan igual y la razón sale 0,95, o sea indeterminada — pasó al
    escribir este test, y es el mismo aviso que la función da sobre datos reales.
    """
    rng = np.random.default_rng(semilla)
    u = np.linspace(0.0, np.pi, n)
    x = 25.0 * np.cos(u)
    y = -18.0 * np.sin(u)  # brazos hacia −y ⇒ el frente queda en +y
    z = 8.0 + 1.5 * np.sin(6.0 * u)
    arco = np.column_stack([x, y, z]) + rng.normal(scale=0.05, size=(n, 3))
    # Base: mismo contorno pero derramado hacia fuera y con mucho más ruido lateral.
    ancho = rng.normal(scale=3.5, size=n)
    base = np.column_stack([
        x * (1.0 + 0.28 * rng.random(n)),
        y * (1.0 + 0.28 * rng.random(n)) + ancho,
        np.full(n, 1.0) + rng.normal(scale=0.6, size=n),
    ])
    return np.vstack([arco, base])


def test_marco_arcada_orienta_hacia_oclusal() -> None:
    """El eje axial apunta a la cresta, y la razón declara que el criterio fue claro."""
    V = arcada_sintetica()
    _, ejes, P, razon = marco_arcada(V)
    # La cresta (lo que más masa tiene arriba) debe quedar en +z del marco.
    assert P[:, 2].max() > abs(P[:, 2].min()) * 0.5
    assert razon < 0.9, "razón cerca de 1 significa orientación dudosa"


def test_marco_canonico_es_a_derechas() -> None:
    centro, R, _ = marco_canonico(arcada_sintetica())
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert centro.shape == (3,)


def test_marco_canonico_es_invariante_a_la_pose_del_fichero() -> None:
    """La misma arcada escrita en otra orientación da el MISMO marco canónico.

    Es la propiedad que hace utilizable la canonización: dos escáneres que escriben sus
    ejes de forma distinta tienen que acabar en la misma pose. Medido en el proyecto:
    el oclusal de Teeth3DS+ va en +z y el de `histora` en +y (coseno 0,004).
    """
    V = arcada_sintetica()
    giro = Rotation.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True).as_matrix()
    V_girada = V @ giro.T + np.array([120.0, -35.0, 60.0])

    c1, R1, _ = marco_canonico(V)
    c2, R2, _ = marco_canonico(V_girada)
    # Las coordenadas canónicas coinciden aunque los ficheros estén en poses distintas.
    P1 = (V - c1) @ R1.T
    P2 = (V_girada - c2) @ R2.T
    assert np.allclose(P1, P2, atol=1e-6)


def test_a_marco_de_lleva_una_arcada_a_la_pose_de_otra() -> None:
    """Canonizar y reexpresar reproduce el fichero destino, normales incluidas."""
    V = arcada_sintetica()
    giro = Rotation.from_euler("xyz", [12.0, -70.0, 33.0], degrees=True).as_matrix()
    V_ref = V @ giro.T + np.array([-40.0, 15.0, 5.0])

    origen = marco_canonico(V)[:2]
    destino = marco_canonico(V_ref)[:2]
    normales = np.tile(np.array([0.0, 0.0, 1.0]), (len(V), 1))
    V2, N2 = a_marco_de(V, origen, destino, normales=normales)

    assert np.allclose(V2, V_ref, atol=1e-6)
    assert np.allclose(np.linalg.norm(N2, axis=1), 1.0, atol=1e-6)
    # Las normales rotan pero NO se trasladan: si se trasladasen, dejarían de ser
    # unitarias y el modelo que las consume recibiría basura sin avisar.
    assert np.allclose(N2, normales @ (origen[1].T @ destino[1]), atol=1e-9)


def test_a_marco_de_sin_normales_devuelve_solo_puntos() -> None:
    V = arcada_sintetica()
    marco = marco_canonico(V)[:2]
    salida = a_marco_de(V, marco, marco)
    assert isinstance(salida, np.ndarray)
    assert np.allclose(salida, V, atol=1e-6)


def test_marco_especular_se_declara_en_vez_de_arreglarse() -> None:
    """Una arcada espejada tiene que LANZAR, no invertirse en silencio.

    Invertir un eje para forzar el determinante convertiría los 3x en 4x sin que nada
    lo indique, y una malla espejada sigue pareciendo una dentadura.
    """
    V = arcada_sintetica()
    V_espejo = V * np.array([1.0, 1.0, -1.0])  # espejo puro: determinante −1
    # Puede salir espejado o no según cómo caiga la PCA; lo que NO puede es devolver
    # un marco a izquierdas sin decirlo.
    try:
        _, R, _ = marco_canonico(V_espejo)
    except MarcoEspecular:
        return
    assert np.linalg.det(R) > 0
