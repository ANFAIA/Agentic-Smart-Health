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

# Por debajo de esta fracción emparejada el residuo deja de informar. Medido: un
# escaneo intraoral contra el esmalte de un CBCT solapa el 27 %, y una malla derivada
# del propio volumen ~100 %; 20 % pasa los dos casos reales y corta el degenerado.
# Como ε, es un número por par de modalidades: aquí está el valor conservador.
DEFAULT_MIN_OVERLAP = 0.20


class GeometricFusionAgent(BaseFusionAgent):
    """Alinea `source` sobre `target` y guarda la transformación en la procedencia.

    **La confianza sale del residuo**, no de una intuición (ADR 004 §2.3):

        confianza = clamp(1 − rms_efectivo / ε, 0, 1)

    Con `rms = 0` da 1.0 y con `rms = ε` da 0.0, así que el gate por defecto (0.7)
    equivale a exigir `rms ≤ 0.3·ε`. Un registro que se pasa de la banda no falla:
    entrega con confianza baja y pide revisión humana, que es la diferencia entre un
    fallo declarado y uno silencioso.

    **El residuo es el de la población solapada**, no el de la nube completa. Medido
    sobre el paciente de `histora` (`scripts/registro_ios_cbct.py`): el mismo registro
    da 4,98 mm sobre la nube entera y **0,452 mm** sobre los puntos que sí tienen
    contrapartida. El primero no describe el registro: describe qué fracción del
    escaneo intraoral es paladar, que no tiene esmalte contra el que emparejarse.

    **Y ε hay que darlo por par de modalidades.** El valor por defecto (0,5 mm) es el
    del caso fácil —malla derivada del propio volumen—. Para intraoral↔CBCT hay que
    pasar `EPSILON_IOS_CBCT_MM`: con 0,5 el gate exigiría `rms ≤ 0,15 mm`, por debajo
    del suelo físico de un CBCT de vóxel 0,30 mm, así que **ningún registro podría
    pasarlo nunca**.

    **Un solapamiento ridículo invalida el registro por bajo que salga el rms**: con
    cuatro puntos emparejados siempre hay alguna pose que los acerca. Por debajo de
    `min_overlap` la confianza se pone a 0 y se dice por qué, en vez de premiar un
    resultado que solo parece bueno porque casi no se midió.

    **El algoritmo es sustituible.** `registrar` cumple el `Protocol` del módulo
    `registration`; por defecto es el ICP multiescala. Cuando el dato real diga que
    hace falta una etapa gruesa (RANSAC-FPFH), se enchufa aquí sin tocar el agente.
    """

    name = "geometric-fusion-agent"
    version = "0.2.0"

    def __init__(
        self,
        *,
        epsilon_mm: float = DEFAULT_EPSILON_MM,
        registrar: Registrar = icp,
        min_overlap: float = DEFAULT_MIN_OVERLAP,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if epsilon_mm <= 0:
            raise ValueError("epsilon_mm debe ser > 0: es el divisor de la confianza")
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_overlap es una fracción: tiene que estar en [0, 1]")
        self.epsilon_mm = epsilon_mm
        self.registrar = registrar
        self.min_overlap = min_overlap

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
        solape = resultado.overlap_fraction
        escaso = solape is not None and solape < self.min_overlap
        confianza = (
            0.0
            if escaso
            else min(max(1.0 - resultado.rms_efectivo_mm / self.epsilon_mm, 0.0), 1.0)
        )

        motivos: list[str] = []
        if escaso:
            motivos.append(
                f"solo el {solape:.1%} de la nube tiene contrapartida (mínimo "
                f"{self.min_overlap:.0%}): con tan pocos puntos emparejados el residuo "
                f"de {resultado.rms_efectivo_mm:.3f} mm no dice si el registro es bueno"
            )
        elif confianza < self.hitl_threshold:
            motivos.append(
                f"registro con rms {resultado.rms_efectivo_mm:.3f} mm sobre una banda de "
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
                f"rms {resultado.rms_efectivo_mm:.3f} mm sobre "
                f"{'toda la nube' if solape is None else f'el {solape:.1%} solapado'} "
                f"({resultado.rms_mm:.3f} mm sobre la nube completa) en "
                f"{resultado.iterations} iteraciones (converge={resultado.converged})."
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
