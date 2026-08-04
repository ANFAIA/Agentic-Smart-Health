"""Transferencia del color de la malla a las gaussianas (ADR 004 §2.8).

La fuente es el **color por vértice de la malla intraoral**, no las fotos: cada
gaussiana dentro de la **banda ε** de la superficie toma el color de su vértice más
cercano. Es lo que describe el ADR 001 — *«None si la gaussiana no cae en la banda ε
de la superficie»*.

**Por qué no las fotos.** El notebook 07 lo midió: proyectar la foto oclusal sobre
las coronas deja el ICP estancado en **IoU ≈ 0,55** porque el error residual es
**no-rígido** (perspectiva de foto intraoral sin calibrar, con retractor). Foto↔malla
y CBCT↔malla no son el mismo problema, así que no viven en el mismo agente.
"""

from __future__ import annotations

import numpy as np
from core_schemas import RigidTransform
from scipy.spatial import cKDTree

from fusion_agents.registration import apply, quaternion_to_matrix


def transfer_surface_color(
    gaussians: np.ndarray,
    mesh_points: np.ndarray,
    mesh_colors: np.ndarray | None,
    *,
    epsilon_mm: float,
    transform: RigidTransform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Color por gaussiana desde la malla, y máscara de a cuáles les tocó.

    Args:
        gaussians: `(N, 3)` centros de las gaussianas, en el marco del campo.
        mesh_points: `(M, 3)` vértices de la malla.
        mesh_colors: `(M, 3)` RGB en [0, 1], o `None` si la malla viene **pelada**
            (STL) o su color es un *placeholder* — el gris plano de Teeth3DS+, que el
            `mesh-agent` ya trata como ausencia. `None` es una respuesta válida, no
            un fallo.
        epsilon_mm: banda ε. Fuera de ella la gaussiana **no** es superficie y se
            queda sin color: es el criterio del ADR 001, no un ajuste.
        transform: si se da, lleva `mesh_points` al marco del campo antes de buscar.
            Es la transformación que devolvió el registro.

    Returns:
        `(colors, has_color)` — `colors` es `(N, 3)`; las filas con `has_color`
        a `False` quedan a cero y **no deben interpretarse como negro**: son ausencia.
    """
    gaussians = np.asarray(gaussians, dtype=float)
    mesh_points = np.asarray(mesh_points, dtype=float)
    if gaussians.ndim != 2 or gaussians.shape[1] != 3:
        raise ValueError(f"`gaussians` debe ser (N, 3), recibido {gaussians.shape}")
    if mesh_points.ndim != 2 or mesh_points.shape[1] != 3:
        raise ValueError(f"`mesh_points` debe ser (M, 3), recibido {mesh_points.shape}")
    if epsilon_mm <= 0:
        raise ValueError("epsilon_mm debe ser > 0: es la banda que define 'ser superficie'")

    sin_color = (np.zeros((len(gaussians), 3)), np.zeros(len(gaussians), dtype=bool))
    if mesh_colors is None or len(mesh_points) == 0 or len(gaussians) == 0:
        return sin_color

    mesh_colors = np.asarray(mesh_colors, dtype=float)
    if mesh_colors.shape != mesh_points.shape:
        raise ValueError(
            f"`mesh_colors` {mesh_colors.shape} no casa con `mesh_points` {mesh_points.shape}"
        )

    if transform is not None:
        mesh_points = apply(
            quaternion_to_matrix(transform.rotation), np.asarray(transform.translation), mesh_points
        )

    dist, idx = cKDTree(mesh_points).query(gaussians)
    has_color = dist <= epsilon_mm

    colors = np.zeros((len(gaussians), 3))
    colors[has_color] = mesh_colors[idx[has_color]]
    return colors, has_color
