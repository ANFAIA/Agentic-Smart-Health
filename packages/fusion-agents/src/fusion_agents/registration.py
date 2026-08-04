"""Registro rígido de dos nubes de puntos (ADR 004 §2.6).

El algoritmo vive **detrás de un `Protocol`** y no dentro del agente, por tres
motivos: se puede sustituir sin tocar el contrato, se puede testear el agente con
un registrador trivial, y permite enchufar una etapa gruesa distinta cuando el dato
real diga que hace falta.

**Qué hay implementado y qué no.** Aquí está el **ICP multiescala** (la etapa fina),
en numpy + scipy. La etapa **gruesa** del ADR —RANSAC sobre descriptores FPFH— *no*
está: es la que da una pose inicial cuando las dos nubes están muy desalineadas, y
depende de una librería pesada (Open3D). El `Protocol` deja el hueco preparado.

Consecuencia práctica, dicha sin rodeos: **el ICP de aquí converge solo si la pose
inicial ya está razonablemente cerca**. Con mallas derivadas del propio volumen eso
se cumple por construcción; con un intraoral y un CBCT capturados por separado,
habrá que medirlo antes de dar el registro por bueno.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from core_schemas import RigidTransform
from scipy.spatial import cKDTree

# Banda de tolerancia del ADR 004 §2.6. No es la métrica de 0.1 mm del brief: esa
# mide reversibilidad de UNA malla; esta, alineamiento entre DOS modalidades.
DEFAULT_EPSILON_MM = 0.5


@dataclass(frozen=True)
class RegistrationResult:
    """Lo que devuelve un registrador: la transformación y cómo de bien encajó."""

    rotation: tuple[float, float, float, float]  # cuaternión (w, x, y, z)
    translation: tuple[float, float, float]  # mm
    rms_mm: float
    iterations: int
    converged: bool

    def to_rigid_transform(self) -> RigidTransform:
        """Puente al contrato: lo que se guarda en `Provenance.transform`."""
        return RigidTransform(
            rotation=self.rotation, translation=self.translation, rms_mm=self.rms_mm
        )


class Registrar(Protocol):
    """Superficie mínima de un algoritmo de registro.

    `source` y `target` son `(N, 3)` en milímetros. Devuelve la transformación que
    lleva `source` sobre `target`.
    """

    def __call__(self, source: np.ndarray, target: np.ndarray) -> RegistrationResult: ...


# --------------------------------------------------------------------------- #
# Piezas
# --------------------------------------------------------------------------- #
def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transformación rígida óptima entre dos nubes **ya emparejadas** punto a punto.

    Solución cerrada por SVD (Kabsch/Umeyama). La corrección con el determinante
    evita que la SVD devuelva una **reflexión** en vez de una rotación: sería una
    solución de menor error que no corresponde a ningún movimiento físico.
    """
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    h = (source - cs).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, ct - rot @ cs


def matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Matriz de rotación 3×3 → cuaternión (w, x, y, z).

    Método de Shepperd: elige la rama según la traza para no dividir por algo
    cercano a cero, que es donde la conversión ingenua pierde precisión.
    """
    tr = float(np.trace(rot))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (
            0.25 * s,
            (rot[2, 1] - rot[1, 2]) / s,
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[1, 0] - rot[0, 1]) / s,
        )
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        q = (
            (rot[2, 1] - rot[1, 2]) / s,
            0.25 * s,
            (rot[0, 1] + rot[1, 0]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
        )
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        q = (
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[0, 1] + rot[1, 0]) / s,
            0.25 * s,
            (rot[1, 2] + rot[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        q = (
            (rot[1, 0] - rot[0, 1]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
            (rot[1, 2] + rot[2, 1]) / s,
            0.25 * s,
        )
    norma = math.sqrt(sum(c * c for c in q))
    return (q[0] / norma, q[1] / norma, q[2] / norma, q[3] / norma)


def apply(rot: np.ndarray, trans: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ rot.T + trans


# --------------------------------------------------------------------------- #
# ICP multiescala
# --------------------------------------------------------------------------- #
def icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    scales: tuple[float, ...] = (0.1, 0.35, 1.0),
    max_iter: int = 60,
    tol_mm: float = 1e-6,
    seed: int = 0,
) -> RegistrationResult:
    """Iterative Closest Point multiescala.

    En cada escala se usa una fracción creciente de `source`: las primeras pasadas
    son baratas y sacan al registro del grueso del desalineamiento, las últimas
    afinan con todos los puntos. Sale más rápido y cae en menos mínimos locales que
    iterar siempre con la nube completa.

    El submuestreo va con semilla fija: dos ejecuciones sobre el mismo dato tienen
    que dar la misma transformación, o la confianza que se deriva del residuo no
    significaría nada.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"`source` debe ser (N, 3), recibido {source.shape}")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"`target` debe ser (N, 3), recibido {target.shape}")
    if len(source) < 3 or len(target) < 3:
        raise ValueError("hacen falta al menos 3 puntos en cada nube para fijar una pose")

    rng = np.random.default_rng(seed)
    arbol = cKDTree(target)
    rot, trans = np.eye(3), np.zeros(3)
    iteraciones, convergio = 0, False

    for escala in scales:
        n = max(3, int(len(source) * escala))
        sub = source if n >= len(source) else source[rng.choice(len(source), n, replace=False)]
        previo = math.inf

        for _ in range(max_iter):
            iteraciones += 1
            movido = apply(rot, trans, sub)
            dist, idx = arbol.query(movido)
            d_rot, d_trans = kabsch(movido, target[idx])
            rot, trans = d_rot @ rot, d_rot @ trans + d_trans

            rms = float(np.sqrt(np.mean(dist**2)))
            if abs(previo - rms) < tol_mm:
                convergio = True
                break
            previo = rms

    # El residuo se reporta sobre la nube COMPLETA, no sobre el submuestreo: es el
    # número del que sale la confianza, y tiene que ser honesto.
    dist_final, _ = arbol.query(apply(rot, trans, source))
    return RegistrationResult(
        rotation=matrix_to_quaternion(rot),
        translation=(float(trans[0]), float(trans[1]), float(trans[2])),
        rms_mm=float(np.sqrt(np.mean(dist_final**2))),
        iterations=iteraciones,
        converged=convergio,
    )
