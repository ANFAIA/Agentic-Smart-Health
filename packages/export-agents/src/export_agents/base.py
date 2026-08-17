"""Contrato común de los agentes de **exportación** (fase 6 del pipeline).

Un agente de exportación **materializa** un `TwinSnapshot`: lo convierte en un
fichero que consume alguien de fuera del sistema (un visor, un laboratorio, el
propio clínico). Es la única familia que **escribe ficheros de salida**, igual que
la ingesta es la única que **lee ficheros de entrada**.

**Por qué una cuarta base.** Es la misma respuesta que dio `BaseAnalysisAgent`: la
*forma* del contrato se repite en las cuatro familias —nunca lanzar, devolver
estado y motivos de revisión, cuarentena sin dato clínico— porque es el idioma del
repositorio; el *contrato* en sí no se comparte:

- `BaseIngestionAgent` toma una ruta de **entrada**; aquí la ruta es de **salida**.
- `BaseFusionAgent` y `BaseAnalysisAgent` devuelven un `TwinSnapshot` enriquecido.
  Un exportador **no enriquece nada**: devuelve un fichero y la medida de cuánto se
  parece a lo que entró. Meter un `path` en `FusionOutput` (que declara
  `extra="forbid"`) obligaría a ensanchar el contrato de la fusión con un campo que
  la fusión no puede rellenar nunca.

**Lo que no hace ningún exportador.** No modifica el twin y no toca el almacén más
que para leer. Exportar es una operación de **solo lectura sobre el gemelo**: si un
export pudiera reescribir el artefacto que exporta, la copia fiel de la superficie
—el guardarraíl de reversibilidad del `mesh-agent`— dejaría de ser fiel al primer
round-trip.
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot
from pydantic import BaseModel, ConfigDict, Field

# Mismo umbral que la ingesta, la fusión y el análisis (`DEFAULT_HITL_THRESHOLD`).
# Se replica como constante propia por la misma razón que allí: la dirección de la
# dependencia es apps → packages, nunca al revés.
DEFAULT_HITL_THRESHOLD = 0.7

# Métrica del brief: «fidelidad de reconstrucción STL desde el Digital Twin, error
# de malla < 0,1 mm». Vive aquí porque es **el criterio de aceptación de esta
# familia**: es exactamente lo que mide exportar y volver a comparar. No confundir
# con ε, la banda de tolerancia del registro entre modalidades (ADR 004 §2.3): esa
# mide alineamiento entre dos medidas distintas, ésta mide reversibilidad de una.
REVERSIBILITY_BUDGET_MM = 0.1

# Presupuesto del **canal de imagen**, el análogo de los 0,1 mm para lo que no es
# geometría. Los dos umbrales son los de «visualmente sin pérdida» de la literatura de
# compresión, y aquí se pueden exigir de sobra porque el ciclo del twin a su render **no
# debería perder nada**: el campo se escribe con posiciones en `double` y el render es
# determinista. Un valor por debajo no significa «se ve un poco peor», significa que hay
# un bug — un stride mal calculado, un eje intercambiado, una vista no reproducible.
RENDER_PSNR_BUDGET_DB = 40.0
RENDER_SSIM_BUDGET = 0.99


@runtime_checkable
class SurfaceStore(Protocol):
    """Lo único que un exportador necesita del almacén de artefactos.

    `Protocol` en vez de importar `ArtifactStore`, igual que en el análisis: el
    almacén es un *seam* previsto para sustituirse cuando exista `3dgs-engine` de
    verdad. Y solo declara `load`: **un exportador no escribe en el almacén**, y un
    `Protocol` que no menciona `put` hace que eso no sea una promesa en prosa sino
    algo que el tipo no permite expresar.
    """

    def load(self, ref: str) -> dict[str, np.ndarray]: ...


class ExportOutput(BaseModel):
    """Lo que un agente de exportación devuelve. **Siempre**, también al fallar."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    status: ModalityStatus
    detail: str | None = Field(default=None, description="Motivo legible si status != ok.")

    path: Path | None = Field(
        default=None, description="Fichero escrito. `None` si no se llegó a escribir nada."
    )
    format: str | None = Field(default=None, description="Formato materializado, p. ej. 'stl'.")
    frame: str | None = Field(
        default=None,
        description="Sistema de referencia en el que se escribió la geometría: 'source' "
        "(el del escáner, tal como entró) o 'twin' (el del CBCT, tras aplicar la "
        "transformación de la fusión geométrica).",
    )
    n_vertices: int | None = Field(default=None, ge=0)
    n_faces: int | None = Field(default=None, ge=0)

    max_deviation_mm: float | None = Field(
        default=None,
        ge=0.0,
        description="Desviación máxima **medida** entre la geometría del twin y la que "
        "quedó escrita en el fichero, releyéndolo. Es la métrica de reversibilidad del "
        "brief (< 0,1 mm). `None` si no se verificó.",
    )
    mean_deviation_mm: float | None = Field(
        default=None,
        ge=0.0,
        description="Desviación MEDIA sobre los mismos puntos que `max_deviation_mm`. Con "
        "la correspondencia conocida —y en un round-trip lo es, vértice a vértice— esto "
        "**es** la distancia de Chamfer. Se calcula así a propósito: buscar el vecino más "
        "próximo en vez de usar la correspondencia daría un número más bonito y esconderría "
        "una permutación de vértices, emparejando cada punto con otro distinto que cae al "
        "lado. Un máximo dice si algo se rompió; una media, cuánto se degradó en conjunto.",
    )

    paths: list[Path] = Field(
        default_factory=list,
        description="Ficheros escritos cuando la exportación produce **varios** (un render "
        "multivista). `path` queda entonces en el directorio que los contiene. Vacío en los "
        "exportadores de un solo fichero, que usan `path`.",
    )
    psnr_db: float | None = Field(
        default=None,
        description="PSNR entre el render del campo del twin y el del fichero exportado, "
        "releído. `inf` si son idénticos, que es lo esperado y no un error.",
    )
    ssim: float | None = Field(
        default=None,
        le=1.0,
        description="SSIM entre los mismos dos renders. Acompaña al PSNR porque miden cosas "
        "distintas: el PSNR promedia el error por píxel y se deja engañar por un "
        "desplazamiento global, el SSIM compara estructura local.",
    )

    hitl_reasons: list[str] = Field(
        default_factory=list,
        description="Motivos por los que el fichero no debería entregarse sin revisión "
        "humana. Vacío = no hace falta.",
    )
    latency_s: float = Field(default=0.0, ge=0.0)
    quarantine_ref: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ModalityStatus.OK

    @property
    def hitl_required(self) -> bool:
        return bool(self.hitl_reasons)

    @property
    def within_budget(self) -> bool:
        """¿Cumple el error **geométrico** del brief? `False` si no se midió.

        Sin medida no hay cumplimiento: un exportador que no verifica no puede
        afirmar que reconstruye por debajo de 0,1 mm.
        """
        return (
            self.max_deviation_mm is not None and self.max_deviation_mm <= REVERSIBILITY_BUDGET_MM
        )

    @property
    def image_within_budget(self) -> bool:
        """¿Cumple el presupuesto del canal de **imagen**? `False` si no se midió.

        Propiedad aparte y no una ampliación de `within_budget` porque son canales
        distintos: un exportador de malla no produce imágenes y un render no tiene
        milímetros. Colapsarlos en un booleano obligaría a uno de los dos a declarar
        `True` sobre algo que no ha medido.
        """
        return (
            self.psnr_db is not None
            and self.ssim is not None
            and self.psnr_db >= RENDER_PSNR_BUDGET_DB
            and self.ssim >= RENDER_SSIM_BUDGET
        )


@runtime_checkable
class ExportAgentProtocol(Protocol):
    """Superficie mínima que el orquestador conoce de un agente de exportación."""

    name: str
    version: str

    def export(
        self, snapshot: TwinSnapshot, destination: str | Path, **inputs: Any
    ) -> ExportOutput: ...


class BaseExportAgent(ABC):
    """Esqueleto *fail-loud* común. Las subclases solo implementan `_export`."""

    name: ClassVar[str]
    version: ClassVar[str]

    def __init__(
        self,
        *,
        hitl_threshold: float = DEFAULT_HITL_THRESHOLD,
        quarantine_dir: str | Path | None = None,
    ) -> None:
        self.hitl_threshold = hitl_threshold
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else None

    # --- API pública ----------------------------------------------------- #
    def export(
        self, snapshot: TwinSnapshot, destination: str | Path, **inputs: Any
    ) -> ExportOutput:
        """Materializa el snapshot en `destination`. **Nunca lanza.**"""
        started = time.perf_counter()
        try:
            outcome = self._export(snapshot, Path(destination), **inputs)
        except Exception as exc:  # el fallo es un dato, no una excepción hacia arriba
            return self._outcome(
                ModalityStatus.FAILED,
                detail=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - started,
                quarantine_ref=self._quarantine(snapshot, exc),
            )
        outcome.latency_s = time.perf_counter() - started
        return outcome

    # --- a implementar por cada agente ----------------------------------- #
    @abstractmethod
    def _export(self, snapshot: TwinSnapshot, destination: Path, **inputs: Any) -> ExportOutput:
        """Escribe el fichero. Puede lanzar: `export` lo captura."""

    # --- utilidades para las subclases ----------------------------------- #
    @property
    def qualified(self) -> str:
        return f"{self.name}@{self.version}"

    def _outcome(self, status: ModalityStatus, **kwargs: Any) -> ExportOutput:
        return ExportOutput(agent=self.qualified, status=status, **kwargs)

    def _partial_reasons(self, snapshot: TwinSnapshot) -> list[str]:
        """Motivos que salen del propio snapshot, antes de mirar la geometría.

        Es el sitio donde se cumple el invariante del ADR 001: «un snapshot parcial
        debe declararse como parcial, **no llegar callado a exportación**». El
        exportador es literalmente el último punto en el que eso se puede decir, así
        que lo dice aquí y no en cada formato.

        Se declara lo que **falló**, no lo que faltó: que una adquisición no traiga
        fotos es normal y no dice nada del fichero exportado; que una modalidad se
        intentara y reventara sí — el twin del que sale este fichero está incompleto
        por un error, y quien lo reciba tiene que saberlo.
        """
        motivos = [
            f"el snapshot es parcial: la modalidad `{i.modality.value}` falló en la "
            f"ingesta ({i.detail or 'sin detalle'})"
            for i in snapshot.ingestion
            if i.status is ModalityStatus.FAILED
        ]
        confianza = snapshot.provenance.confidence
        if confianza < self.hitl_threshold:
            motivos.append(
                f"el snapshot se ensambló con confianza {confianza:.2f} "
                f"(< {self.hitl_threshold:.2f}): revisar antes de entregar el fichero."
            )
        return motivos

    def _quarantine(self, snapshot: TwinSnapshot, exc: BaseException) -> str | None:
        """Registra el caso fallido. Guarda el `acquisition_id`, nunca el contenido.

        Igual que en las otras tres familias: volcar el snapshot duplicaría dato de
        paciente fuera del almacenamiento autorizado (soberanía del dato, AGENTS.md).
        """
        if self.quarantine_dir is None:
            return None
        try:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            record = self.quarantine_dir / f"{self.name}-{stamp}.json"
            record.write_text(
                json.dumps(
                    {
                        "agent": self.qualified,
                        "acquisition_id": snapshot.acquisition_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exception(exc),
                        "quarantined_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return str(record)
        except OSError:
            # La cuarentena es diagnóstico: no puede convertirse ella misma en el
            # fallo que tumba la exportación.
            return None
