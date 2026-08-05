"""Transferencia del color malla → gaussianas (ADR 004 §2.8)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from core_schemas import RigidTransform
from fusion_agents import GeometricFusionAgent, transfer_surface_color

ROJO, VERDE = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]


def test_la_gaussiana_dentro_de_la_banda_toma_el_color_del_vertice():
    colors, has = transfer_surface_color(
        np.array([[0.0, 0.0, 0.2]]), np.array([[0.0, 0.0, 0.0]]), np.array([ROJO]), epsilon_mm=0.5
    )
    assert has.tolist() == [True]
    assert colors[0].tolist() == ROJO


def test_fuera_de_la_banda_no_hay_color():
    """El criterio del ADR 001: fuera de ε la gaussiana no es superficie."""
    colors, has = transfer_surface_color(
        np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, 0.0]]), np.array([ROJO]), epsilon_mm=0.5
    )
    assert has.tolist() == [False]
    assert colors[0].tolist() == [0.0, 0.0, 0.0]  # cero = ausencia, NO negro


def test_cada_gaussiana_toma_el_vertice_mas_cercano():
    malla = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    colores = np.array([ROJO, VERDE])
    gauss = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    colors, has = transfer_surface_color(gauss, malla, colores, epsilon_mm=0.5)
    assert has.all()
    assert colors[0].tolist() == ROJO
    assert colors[1].tolist() == VERDE


def test_malla_pelada_no_es_un_fallo():
    """STL sin color o gris placeholder de Teeth3DS+: `None` es respuesta válida."""
    colors, has = transfer_surface_color(
        np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]), None, epsilon_mm=0.5
    )
    assert not has.any()
    assert colors.shape == (1, 3)


def test_aplica_la_transformacion_del_registro():
    """Sin transformar, la malla estaría lejos y no habría color."""
    malla = np.array([[0.0, 0.0, 0.0]])
    gauss = np.array([[10.0, 0.0, 0.0]])
    assert not transfer_surface_color(gauss, malla, np.array([ROJO]), epsilon_mm=0.5)[1].any()

    tr = RigidTransform(rotation=(1.0, 0.0, 0.0, 0.0), translation=(10.0, 0.0, 0.0))
    colors, has = transfer_surface_color(
        gauss, malla, np.array([ROJO]), epsilon_mm=0.5, transform=tr
    )
    assert has.tolist() == [True]
    assert colors[0].tolist() == ROJO


def test_la_transformacion_con_rotacion_tambien_se_aplica():
    ang = math.radians(90)
    tr = RigidTransform(
        rotation=(math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)), translation=(0.0, 0.0, 0.0)
    )
    # (1,0,0) rotado 90° en Z -> (0,1,0)
    colors, has = transfer_surface_color(
        np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]),
        np.array([VERDE]), epsilon_mm=0.1, transform=tr,
    )
    assert has.tolist() == [True]
    assert colors[0].tolist() == VERDE


@pytest.mark.parametrize(
    ("gauss", "malla", "colores", "eps"),
    [
        (np.zeros((2, 2)), np.zeros((1, 3)), np.zeros((1, 3)), 0.5),  # gaussianas no (N,3)
        (np.zeros((1, 3)), np.zeros((2, 2)), np.zeros((1, 3)), 0.5),  # malla no (M,3)
        (np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((1, 3)), 0.5),  # colores descuadran
        (np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), 0.0),  # banda inválida
    ],
)
def test_rechaza_entradas_invalidas(gauss, malla, colores, eps):
    with pytest.raises(ValueError):
        transfer_surface_color(gauss, malla, colores, epsilon_mm=eps)


def test_nubes_vacias_no_revientan():
    colors, has = transfer_surface_color(
        np.zeros((0, 3)), np.zeros((0, 3)), None, epsilon_mm=0.5
    )
    assert colors.shape == (0, 3) and has.shape == (0,)


def test_el_agente_usa_su_propia_banda():
    """`transfer_color` hereda el ε del agente, no uno suelto."""
    estrecho = GeometricFusionAgent(epsilon_mm=0.1)
    ancho = GeometricFusionAgent(epsilon_mm=5.0)
    gauss, malla, col = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0]]), np.array([ROJO])

    assert not estrecho.transfer_color(gauss, malla, col)[1].any()
    assert ancho.transfer_color(gauss, malla, col)[1].all()
