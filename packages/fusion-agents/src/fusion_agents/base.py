"""Contrato común de los agentes de fusión (ADR 004).

La fusión se parece a la ingesta en el **contrato** —no lanza, devuelve estado y
confianza, manda a cuarentena lo que no puede procesar— y se diferencia en la
**entrada**: no toca ficheros crudos, sino `TwinSnapshot` ya ingeridos. Por eso
tiene su propia base en vez de heredar de `BaseIngestionAgent`, que asume una ruta
en disco.

Dos cosas que **no** se reinventan aquí, a propósito:

- **`ModalityStatus`** como estado. Los tres desenlaces son los mismos (se hizo /
  no había qué hacer / falló) y el contrato *fail-loud* del repo ya está construido
  sobre ese vocabulario. Un enum paralelo con la misma semántica fragmentaría el
  idioma común por un matiz de nombre.
- **`hitl_reasons`** como forma de pedir revisión humana, igual que
  `PipelineResult` en el orquestador: una lista vacía significa que no hace falta.
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from core_schemas import ModalityStatus, TwinSnapshot
from pydantic import BaseModel, ConfigDict, Field

# Mismo umbral que el orquestador (`DEFAULT_HITL_THRESHOLD`). Se replica aquí como
# constante propia para que el paquete no dependa de una `app`: la dirección de la
# dependencia es apps → packages, nunca al revés.
DEFAULT_HITL_THRESHOLD = 0.7


class FusionOutput(BaseModel):
    """Lo que un agente de fusión devuelve al orquestador. **Siempre**, también al fallar."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    status: ModalityStatus
    detail: str | None = Field(default=None, description="Motivo legible si status != ok.")

    snapshot: TwinSnapshot | None = Field(
        default=None,
        description="Snapshot fusionado. Nuevo, nunca el de entrada mutado (ADR 004 §2.5).",
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
class FusionAgent(Protocol):
    """Superficie mínima que el orquestador conoce de un agente de fusión.

    `Protocol` por el mismo motivo que en ingesta: el orquestador depende de esta
    forma y no de la clase base, así la elección de framework de agentes queda
    fuera de los agentes.
    """

    name: str
    version: str

    def fuse(self, snapshot: TwinSnapshot, **inputs: Any) -> FusionOutput: ...


class BaseFusionAgent(ABC):
    """Esqueleto *fail-loud* común. Las subclases solo implementan `_fuse`."""

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
    def fuse(self, snapshot: TwinSnapshot, **inputs: Any) -> FusionOutput:
        """Fusiona y devuelve el resultado. **Nunca lanza.**"""
        started = time.perf_counter()
        try:
            outcome = self._fuse(snapshot, **inputs)
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
    def _fuse(self, snapshot: TwinSnapshot, **inputs: Any) -> FusionOutput:
        """Hace la fusión. Puede lanzar: `fuse` lo captura."""

    # --- utilidades para las subclases ----------------------------------- #
    @property
    def qualified(self) -> str:
        return f"{self.name}@{self.version}"

    def _outcome(self, status: ModalityStatus, **kwargs: Any) -> FusionOutput:
        return FusionOutput(agent=self.qualified, status=status, **kwargs)

    def _quarantine(self, snapshot: TwinSnapshot, exc: BaseException) -> str | None:
        """Registra el caso fallido. Guarda el `acquisition_id`, nunca el contenido.

        Igual que en ingesta: volcar el snapshot duplicaría dato de paciente fuera
        del almacenamiento autorizado (soberanía del dato, AGENTS.md).
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
            # fallo que tumba la fusión.
            return None
