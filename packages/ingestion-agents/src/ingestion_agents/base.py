"""Contrato común de los agentes de ingesta.

Regla de diseño del pipeline (ADR — `docs/architecture/multi-agent-pipeline.md` §2):
**1 modalidad = 1 soporte = 1 agente**. Cada agente traduce *su* modalidad al
contrato de `core-schemas` y declara su `Provenance`. Los agentes de ingesta son
los únicos que tocan ficheros crudos.

Lo que este módulo aporta es la parte que **todos** comparten, para que ningún
agente la reimplemente:

1. `IngestionOutput` — lo que devuelve un agente, **siempre**, también al
   fallar. No hay canal de excepción hacia el orquestador.
2. `BaseIngestionAgent.ingest()` — envoltorio *fail-loud*: mide latencia, captura
   cualquier excepción, la convierte en `status=failed` con motivo, y deja el
   caso en **cuarentena** para inspección posterior.

**Por qué un agente de ingesta nunca lanza.** El brief exige >95% de fiabilidad
de la ingesta sobre un dataset de validación. Si un DICOM corrupto tumbara el
proceso, un fallo de una modalidad se llevaría por delante las otras dos. Aquí un
fallo es un *dato* (`ModalityStatus.FAILED` + `detail`) que viaja dentro del
`TwinSnapshot`: el snapshot parcial se declara parcial en vez de llegar callado a
exportación.
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from core_schemas import (
    Modality,
    ModalityIngestion,
    ModalityStatus,
    Provenance,
    RegionalObservation,
    Support,
)
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Resultado de un agente de ingesta
# --------------------------------------------------------------------------- #
class IngestionOutput(BaseModel):
    """Lo que un agente de ingesta devuelve al orquestador.

    Es un supraconjunto de `ModalityIngestion`: además del estado lleva la carga
    útil (referencia al artefacto pesado y/o observaciones regionales) y la
    métrica de latencia que el brief exige medir (<60 s para las tres modalidades).
    """

    model_config = ConfigDict(extra="forbid")

    # --- identidad ------------------------------------------------------- #
    agent: str
    modality: Modality
    support: Support = Field(
        description="Soporte geométrico del atributo que produce este agente "
        "(volumétrico / superficial / regional). Es la mitad de la regla "
        "'1 modalidad = 1 soporte = 1 agente'."
    )

    # --- resultado ------------------------------------------------------- #
    status: ModalityStatus
    detail: str | None = Field(
        default=None, description="Motivo legible si status != ok."
    )

    # --- carga útil (según soporte) -------------------------------------- #
    artifact_ref: str | None = Field(
        default=None,
        description="Referencia content-addressed (`sha256:…`) al blob pesado en "
        "el ArtifactStore. Volumétrico → campo gaussiano; superficial → malla.",
    )
    n_primitives: int | None = Field(default=None, ge=0)
    regional: list[RegionalObservation] = Field(
        default_factory=list,
        description="Observaciones por región FDI (soporte REGIONAL, p. ej. pH).",
    )

    # --- trazabilidad y métrica ------------------------------------------ #
    provenance: Provenance | None = Field(
        default=None, description="None solo si la modalidad no se aportó (missing)."
    )
    latency_s: float = Field(default=0.0, ge=0.0)
    quarantine_ref: str | None = Field(
        default=None, description="Ruta del registro de cuarentena si status == failed."
    )

    @property
    def ok(self) -> bool:
        return self.status is ModalityStatus.OK

    @property
    def ingestion(self) -> ModalityIngestion:
        """Proyección al modelo del contrato que viaja dentro del `TwinSnapshot`."""
        return ModalityIngestion(
            modality=self.modality, status=self.status, detail=self.detail
        )


# --------------------------------------------------------------------------- #
# Interfaz (lo que el orquestador necesita saber de un agente)
# --------------------------------------------------------------------------- #
@runtime_checkable
class IngestionAgent(Protocol):
    """Superficie mínima que el orquestador conoce de un agente de ingesta.

    Es un `Protocol` a propósito: el orquestador depende de esta forma, no de
    `BaseIngestionAgent`. Así la elección de framework de agentes (LangGraph,
    CrewAI, MCP, Python plano) queda **fuera** de los agentes — se decide en la
    capa de orquestación sin tocar una línea de ingesta.
    """

    name: str
    version: str
    modality: Modality
    support: Support

    def ingest(self, source: str | Path) -> IngestionOutput: ...


# --------------------------------------------------------------------------- #
# Base compartida
# --------------------------------------------------------------------------- #
class BaseIngestionAgent(ABC):
    """Esqueleto *fail-loud* común. Las subclases solo implementan `_ingest`."""

    name: ClassVar[str]
    version: ClassVar[str]
    modality: ClassVar[Modality]
    support: ClassVar[Support]

    def __init__(self, *, quarantine_dir: str | Path | None = None) -> None:
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else None

    # --- API pública ----------------------------------------------------- #
    def ingest(self, source: str | Path) -> IngestionOutput:
        """Ingiere `source` y devuelve el resultado. **Nunca lanza.**"""
        started = time.perf_counter()
        path = Path(source)

        if not path.exists():
            return self._outcome(
                ModalityStatus.MISSING,
                detail=f"No se aportó el fichero de esta modalidad: {path}",
                latency_s=time.perf_counter() - started,
            )

        try:
            outcome = self._ingest(path)
        except Exception as exc:  # fail-loud, no fail-fast: el fallo es un dato
            return self._outcome(
                ModalityStatus.FAILED,
                detail=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - started,
                quarantine_ref=self._quarantine(path, exc),
            )

        outcome.latency_s = time.perf_counter() - started
        return outcome

    # --- a implementar por cada agente ----------------------------------- #
    @abstractmethod
    def _ingest(self, source: Path) -> IngestionOutput:
        """Traduce el fichero crudo al contrato. Puede lanzar: `ingest` lo captura."""

    # --- utilidades para las subclases ----------------------------------- #
    def _provenance(self, source: Path, confidence: float = 1.0) -> Provenance:
        return Provenance(
            source_file=str(source),
            modality=self.modality,
            agent=f"{self.name}@{self.version}",
            confidence=confidence,
        )

    def _outcome(self, status: ModalityStatus, **kwargs: object) -> IngestionOutput:
        return IngestionOutput(
            agent=f"{self.name}@{self.version}",
            modality=self.modality,
            support=self.support,
            status=status,
            **kwargs,  # type: ignore[arg-type]
        )

    def _success(
        self, source: Path, *, confidence: float = 1.0, **kwargs: object
    ) -> IngestionOutput:
        return self._outcome(
            ModalityStatus.OK, provenance=self._provenance(source, confidence), **kwargs
        )

    # --- cuarentena ------------------------------------------------------ #
    def _quarantine(self, source: Path, exc: BaseException) -> str | None:
        """Registra el caso fallido para inspección; no copia el dato clínico.

        Se guarda la **ruta** del fichero y el traceback, nunca su contenido:
        mover un DICOM a un directorio de cuarentena duplicaría dato de paciente
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
                        "agent": f"{self.name}@{self.version}",
                        "modality": self.modality.value,
                        "source_file": str(source),
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
            # La cuarentena es diagnóstico, no puede convertirse ella misma en
            # el fallo que tumba la ingesta.
            return None
