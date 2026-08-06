"""Contrato común de los agentes de **análisis**.

Un agente de análisis consume un `TwinSnapshot` y lo **enriquece** con algo que no
estaba en el dato crudo: etiquetas anatómicas, hallazgos, métricas. No toca
ficheros (eso es ingesta) y no alinea modalidades entre sí (eso es fusión).

**Por qué una tercera base y no reutilizar una de las dos que ya hay.** La forma
del contrato es la misma en las tres familias —nunca lanzar, devolver estado y
confianza, cuarentena sin dato clínico— porque es el idioma del repositorio, y eso
sí se repite. Lo que no se comparte es el *contrato* en sí:

- `BaseIngestionAgent` asume una **ruta en disco** como entrada; aquí no hay ruta.
- `BaseFusionAgent` tiene la entrada correcta, pero heredar de él pondría el
  análisis **río abajo de la fusión** en el grafo de dependencias, y en el pipeline
  la segmentación corre **entre** las dos etapas de fusión (la semántica necesita el
  `region_id` que produce esta). Una flecha que contradice el orden real del
  pipeline es justo el tipo de mentira estructural que este repositorio evita.
- Y la salida es genuinamente distinta: un agente de análisis devuelve además
  **lo que ha inferido** (para la segmentación, el mapa `FDI → confianza`), que en
  `FusionOutput` no tiene sitio (`extra="forbid"`).

El precio es un esqueleto de ~40 líneas repetido por tercera vez. Lo que se compra
es que cada familia pueda mover su contrato sin arrastrar a las otras dos.
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from core_schemas import ModalityStatus, TwinSnapshot
from pydantic import BaseModel, ConfigDict, Field

# Mismo umbral que el orquestador y que la fusión (`DEFAULT_HITL_THRESHOLD`). Se
# replica como constante propia por la misma razón que allí: la dirección de la
# dependencia es apps → packages, nunca al revés.
DEFAULT_HITL_THRESHOLD = 0.7


class AnalysisOutput(BaseModel):
    """Lo que un agente de análisis devuelve. **Siempre**, también al fallar."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    status: ModalityStatus
    detail: str | None = Field(default=None, description="Motivo legible si status != ok.")

    snapshot: TwinSnapshot | None = Field(
        default=None,
        description="Snapshot enriquecido. Nuevo, nunca el de entrada mutado (ADR 004 §2.5).",
    )
    hitl_reasons: list[str] = Field(
        default_factory=list,
        description="Motivos por los que hace falta revisión humana. Vacío = no hace falta.",
    )
    latency_s: float = Field(default=0.0, ge=0.0)
    quarantine_ref: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is ModalityStatus.OK

    @property
    def hitl_required(self) -> bool:
        return bool(self.hitl_reasons)


@runtime_checkable
class AnalysisAgent(Protocol):
    """Superficie mínima que el orquestador conoce de un agente de análisis."""

    name: str
    version: str

    def analyze(self, snapshot: TwinSnapshot, **inputs: Any) -> AnalysisOutput: ...


OutputT = TypeVar("OutputT", bound=AnalysisOutput)


class BaseAnalysisAgent(ABC, Generic[OutputT]):
    """Esqueleto *fail-loud* común. Las subclases solo implementan `_analyze`.

    Genérico en su salida porque cada análisis devuelve **además lo que ha
    inferido** (la segmentación, el mapa `FDI → confianza`). Sin esto, el camino de
    fallo del envoltorio devolvería un `AnalysisOutput` pelado y quien llama tendría
    que preguntar si el agente falló antes de leer su propio campo: un fallo
    cambiaría la *forma* de la respuesta, no solo su contenido.
    """

    name: ClassVar[str]
    version: ClassVar[str]
    # Sin `ClassVar`: una variable de tipo no puede vivir dentro de uno. Cada
    # subclase declara con qué modelo construye sus resultados.
    output_model: type[OutputT]

    def __init__(
        self,
        *,
        hitl_threshold: float = DEFAULT_HITL_THRESHOLD,
        quarantine_dir: str | Path | None = None,
    ) -> None:
        self.hitl_threshold = hitl_threshold
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else None

    # --- API pública ----------------------------------------------------- #
    def analyze(self, snapshot: TwinSnapshot, **inputs: Any) -> OutputT:
        """Analiza y devuelve el resultado. **Nunca lanza.**"""
        started = time.perf_counter()
        try:
            outcome = self._analyze(snapshot, **inputs)
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
    def _analyze(self, snapshot: TwinSnapshot, **inputs: Any) -> OutputT:
        """Hace el análisis. Puede lanzar: `analyze` lo captura."""

    # --- utilidades para las subclases ----------------------------------- #
    @property
    def qualified(self) -> str:
        return f"{self.name}@{self.version}"

    def _outcome(self, status: ModalityStatus, **kwargs: Any) -> OutputT:
        return self.output_model(agent=self.qualified, status=status, **kwargs)

    def _quarantine(self, snapshot: TwinSnapshot, exc: BaseException) -> str | None:
        """Registra el caso fallido. Guarda el `acquisition_id`, nunca el contenido.

        Igual que en ingesta y fusión: volcar el snapshot duplicaría dato de paciente
        fuera del almacenamiento autorizado (soberanía del dato, AGENTS.md).
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
            # fallo que tumba el análisis.
            return None
