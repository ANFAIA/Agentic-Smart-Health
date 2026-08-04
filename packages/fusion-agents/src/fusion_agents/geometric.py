"""`geometric-fusion-agent`: registra la malla intraoral contra el CBCT (ADR 004).

Primera etapa de fusión, la que va **antes** de la segmentación. No usa FDI: solo
alinea dos medidas del mismo objeto físico y deja constancia **invertible** de la
transformación que aplicó.

También **transfiere el color** de la malla a las gaussianas (§2.8): cada gaussiana
dentro de la banda ε toma el de su vértice más cercano. La fuente es la malla, no las
fotos — el error foto↔malla es **no-rígido** y por tanto otro problema.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot

from fusion_agents.base import BaseFusionAgent, FusionOutput
from fusion_agents.color import transfer_surface_color
from fusion_agents.registration import DEFAULT_EPSILON_MM, Registrar, icp


class GeometricFusionAgent(BaseFusionAgent):
    """Alinea `source` sobre `target` y guarda la transformación en la procedencia.

    **La confianza sale del residuo**, no de una intuición (ADR 004 §2.3):

        confianza = clamp(1 − rms / ε, 0, 1)

    Con `rms = 0` da 1.0 y con `rms = ε` da 0.0, así que el gate por defecto (0.7)
    equivale a exigir `rms ≤ 0.3·ε`. Un registro que se pasa de la banda no falla:
    entrega con confianza baja y pide revisión humana, que es la diferencia entre un
    fallo declarado y uno silencioso.

    **El algoritmo es sustituible.** `registrar` cumple el `Protocol` del módulo
    `registration`; por defecto es el ICP multiescala. Cuando el dato real diga que
    hace falta una etapa gruesa (RANSAC-FPFH), se enchufa aquí sin tocar el agente.
    """

    name = "geometric-fusion-agent"
    version = "0.1.0"

    def __init__(
        self,
        *,
        epsilon_mm: float = DEFAULT_EPSILON_MM,
        registrar: Registrar = icp,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if epsilon_mm <= 0:
            raise ValueError("epsilon_mm debe ser > 0: es el divisor de la confianza")
        self.epsilon_mm = epsilon_mm
        self.registrar = registrar

    def _fuse(  # type: ignore[override]
        self, snapshot: TwinSnapshot, *, source: np.ndarray, target: np.ndarray
    ) -> FusionOutput:
        if len(source) == 0 or len(target) == 0:
            # Falta una de las dos modalidades: no hay nada que registrar. Es
            # ausencia de entrada, no un fallo del algoritmo.
            return self._outcome(
                ModalityStatus.MISSING,
                detail="Hacen falta las dos nubes (malla y CBCT) para registrar.",
            )

        resultado = self.registrar(source, target)
        confianza = min(max(1.0 - resultado.rms_mm / self.epsilon_mm, 0.0), 1.0)

        motivos: list[str] = []
        if confianza < self.hitl_threshold:
            motivos.append(
                f"registro con rms {resultado.rms_mm:.3f} mm sobre una banda de "
                f"{self.epsilon_mm:.3f} mm → confianza {confianza:.2f}, bajo el umbral "
                f"{self.hitl_threshold:.2f}"
            )
        if not resultado.converged:
            motivos.append(
                f"el registro agotó las iteraciones ({resultado.iterations}) sin converger"
            )

        prov = snapshot.provenance.model_copy(
            update={
                "agent": self.qualified,
                "confidence": confianza,
                "transform": resultado.to_rigid_transform(),
            }
        )
        return self._outcome(
            ModalityStatus.OK,
            snapshot=snapshot.model_copy(update={"provenance": prov}),
            hitl_reasons=motivos,
            detail=(
                f"rms {resultado.rms_mm:.3f} mm en {resultado.iterations} iteraciones "
                f"(converge={resultado.converged})."
            ),
        )

    def transfer_color(
        self,
        gaussians: np.ndarray,
        mesh_points: np.ndarray,
        mesh_colors: np.ndarray | None,
        *,
        transform: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Color por gaussiana desde la malla, con la banda ε de este agente (§2.8).

        Devuelve `(colors, has_color)` y **no persiste nada**: `color_superficie` vive
        en `GaussianPrimitive`, y el `TwinSnapshot` solo guarda una referencia por hash
        al campo. Materializarlo es de quien sea dueño del `ArtifactStore` — mantener
        esa frontera evita que un agente de fusión acabe reescribiendo blobs pesados.
        """
        return transfer_surface_color(
            gaussians, mesh_points, mesh_colors, epsilon_mm=self.epsilon_mm, transform=transform
        )
