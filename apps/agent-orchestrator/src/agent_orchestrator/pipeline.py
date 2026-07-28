"""Orquestación de la fase de **ingesta**: los tres agentes → un `TwinSnapshot`.

Este módulo es la Fase 1 del pipeline (`docs/architecture/multi-agent-pipeline.md`
§6): dispara los agentes de ingesta, recoge sus `IngestionOutput` y **ensambla el
contrato**. No hace fusión, ni segmentación, ni análisis.

**Sobre el framework de agentes.** El brief pide elegir y justificar uno
(LangGraph / CrewAI / MCP / …). Aquí la elección es deliberadamente **ninguno
todavía**, y el motivo es medible: en esta fase no hay nada que un framework de
agentes aporte. Sus ventajas —enrutado condicional, estado compartido, bucles de
replanificación, herramientas por agente— presuponen un grafo con decisiones.
La ingesta no tiene decisiones: son tres tareas **independientes** con esquema
fijo de entrada y salida. Meter un framework aquí añadiría una dependencia y una
capa de indirección a cambio de cero enrutado.

La decisión se toma donde sí empieza a haber grafo (fusión ↔ segmentación ↔
análisis, con gates de human-in-the-loop y reintentos), y se puede tomar **sin
tocar los agentes**: dependen del `Protocol` `IngestionAgent`, no de esta clase.

**Concurrencia.** Las tres modalidades son independientes, así que se ingieren en
paralelo con hilos: el trabajo real está en I/O de disco y en numpy, que sueltan
el GIL. Es lo que da margen al presupuesto de <60 s del brief.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core_schemas import (
    Modality,
    ModalityStatus,
    PatientDigitalTwin,
    Provenance,
    TwinSnapshot,
)
from ingestion_agents import (
    ArtifactStore,
    CBCTAgent,
    IngestionOutput,
    MeshAgent,
    ReportAgent,
    pseudonymize,
)
from ingestion_agents.base import BaseIngestionAgent

# Por debajo de esta confianza una modalidad no se da por buena sola: pasa por
# revisión humana. Es el gate de human-in-the-loop en su forma más simple —
# un umbral explícito y auditable, no un juicio del agente sobre sí mismo.
DEFAULT_HITL_THRESHOLD = 0.7

# Presupuesto de latencia de ingesta acordado con el partner (brief, Semana 3-4).
LATENCY_BUDGET_S = 60.0


@dataclass
class CaseInput:
    """Los ficheros crudos de una adquisición. Cualquier modalidad puede faltar."""

    acquisition_id: str
    patient_id: str = "SYNTH-0001"
    mesh: Path | None = None
    cbct: Path | None = None
    report: Path | None = None
    timestamp: datetime | None = None

    @classmethod
    def from_case_dir(cls, root: Path, **kwargs: object) -> CaseInput:
        """Descubre las modalidades en el layout que produce `synthetic.write_case`."""
        root = Path(root)
        objs = sorted(root.glob("*.obj"))
        reports = sorted(root.glob("*.txt")) + sorted(root.glob("*.pdf"))
        cbct = root / "cbct"
        return cls(
            acquisition_id=kwargs.pop("acquisition_id", root.name),  # type: ignore[arg-type]
            mesh=objs[0] if objs else None,
            cbct=cbct if cbct.is_dir() else None,
            report=reports[0] if reports else None,
            **kwargs,  # type: ignore[arg-type]
        )


@dataclass
class PipelineResult:
    """Resultado completo de la ingesta: el contrato + todo lo que hizo falta para juzgarlo."""

    snapshot: TwinSnapshot | None
    outcomes: list[IngestionOutput] = field(default_factory=list)
    hitl_reasons: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    patient_pseudonym: str = ""

    @property
    def hitl_required(self) -> bool:
        """¿Necesita revisión humana antes de persistirse?"""
        return bool(self.hitl_reasons)

    @property
    def within_budget(self) -> bool:
        return self.latency_s <= LATENCY_BUDGET_S

    def outcome(self, modality: Modality) -> IngestionOutput | None:
        return next((o for o in self.outcomes if o.modality is modality), None)


class IngestionPipeline:
    """Coordina los agentes de ingesta y ensambla el `TwinSnapshot`."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        quarantine_dir: str | Path | None = None,
        hitl_threshold: float = DEFAULT_HITL_THRESHOLD,
        report_backend: str = "rules",
        parallel: bool = True,
    ) -> None:
        self.store = store
        self.hitl_threshold = hitl_threshold
        self.parallel = parallel
        self.agents: dict[Modality, BaseIngestionAgent] = {
            Modality.MESH: MeshAgent(store, quarantine_dir=quarantine_dir),
            Modality.CBCT: CBCTAgent(store, quarantine_dir=quarantine_dir),
            Modality.REPORT: ReportAgent(
                backend=report_backend, quarantine_dir=quarantine_dir
            ),
        }

    # --- ejecución ------------------------------------------------------- #
    def run(self, case: CaseInput) -> PipelineResult:
        started = time.perf_counter()
        sources = {
            Modality.MESH: case.mesh,
            Modality.CBCT: case.cbct,
            Modality.REPORT: case.report,
        }

        # Una modalidad no aportada no se le pasa al agente: se declara `missing`
        # directamente. Así `missing` (no había fichero) y `failed` (lo había y no
        # se pudo leer) no se confunden nunca.
        pending = {m: p for m, p in sources.items() if p is not None}
        outcomes: dict[Modality, IngestionOutput] = {
            m: self._missing(m) for m, p in sources.items() if p is None
        }

        if self.parallel and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = {
                    modality: pool.submit(self.agents[modality].ingest, path)
                    for modality, path in pending.items()
                }
                outcomes.update({m: f.result() for m, f in futures.items()})
        else:
            outcomes.update(
                {m: self.agents[m].ingest(p) for m, p in pending.items()}
            )

        ordered = [outcomes[m] for m in (Modality.CBCT, Modality.MESH, Modality.REPORT)]
        pseudonym = pseudonymize(case.patient_id)
        snapshot = self._assemble(case, ordered, pseudonym)

        return PipelineResult(
            snapshot=snapshot,
            outcomes=ordered,
            hitl_reasons=self._hitl_reasons(ordered, snapshot),
            latency_s=time.perf_counter() - started,
            patient_pseudonym=pseudonym,
        )

    def run_into_twin(
        self, case: CaseInput, twin: PatientDigitalTwin | None = None
    ) -> tuple[PatientDigitalTwin, PipelineResult]:
        """Ingiere y añade el snapshot a la serie temporal del paciente."""
        result = self.run(case)
        if twin is None:
            twin = PatientDigitalTwin(patient_id=result.patient_pseudonym)
        if twin.patient_id != result.patient_pseudonym:
            raise ValueError(
                "El snapshot es de otro paciente: "
                f"{result.patient_pseudonym} != {twin.patient_id}"
            )
        if result.snapshot is not None:
            twin.snapshots.append(result.snapshot)
        return twin, result

    # --- ensamblado del contrato ----------------------------------------- #
    def _assemble(
        self, case: CaseInput, outcomes: list[IngestionOutput], pseudonym: str
    ) -> TwinSnapshot | None:
        cbct = next(o for o in outcomes if o.modality is Modality.CBCT)

        # Sin campo gaussiano no hay `TwinSnapshot`: `gaussian_field_ref` es
        # obligatorio en el contrato porque un twin *es* el campo más sus
        # metadatos. Antes que degradar el contrato a "referencia opcional", la
        # ingesta devuelve `snapshot=None` con los outcomes intactos, y quien
        # llama decide. Ver la nota de diseño en el ADR 004.
        if not cbct.ok or cbct.artifact_ref is None:
            return None

        for outcome in outcomes:
            if outcome.artifact_ref and not self.store.exists(outcome.artifact_ref):
                raise RuntimeError(
                    f"Referencia colgante de {outcome.agent}: {outcome.artifact_ref}"
                )

        mesh = next(o for o in outcomes if o.modality is Modality.MESH)
        report = next(o for o in outcomes if o.modality is Modality.REPORT)

        return TwinSnapshot(
            acquisition_id=case.acquisition_id,
            timestamp=case.timestamp or datetime.now(UTC),
            modalities=[o.modality for o in outcomes if o.ok],
            ingestion=[o.ingestion for o in outcomes],
            gaussian_field_ref=cbct.artifact_ref,
            surface_ref=mesh.artifact_ref if mesh.ok else None,
            n_primitives=cbct.n_primitives,
            regional=list(report.regional),
            provenance=Provenance(
                source_file=str(case.cbct),
                modality=Modality.CBCT,
                agent="agent-orchestrator@0.1.0",
                confidence=min(
                    (o.provenance.confidence for o in outcomes if o.provenance), default=0.0
                ),
            ),
        )

    # --- gate de human-in-the-loop --------------------------------------- #
    def _hitl_reasons(
        self, outcomes: list[IngestionOutput], snapshot: TwinSnapshot | None
    ) -> list[str]:
        """Motivos por los que este snapshot no debe persistirse sin revisión humana.

        El agente **no** decide: solo reporta estado y confianza. La decisión vive
        aquí, en una regla explícita y auditable, no dentro del agente.
        """
        reasons: list[str] = []
        if snapshot is None:
            reasons.append(
                "Sin campo gaussiano (CBCT ausente o fallido): no se pudo formar un TwinSnapshot."
            )
        for outcome in outcomes:
            if outcome.status is ModalityStatus.FAILED:
                reasons.append(f"{outcome.agent} falló: {outcome.detail}")
            elif outcome.ok and outcome.provenance is not None:
                confidence = outcome.provenance.confidence
                if confidence < self.hitl_threshold:
                    reasons.append(
                        f"{outcome.agent} ingirió con confianza {confidence:.2f} "
                        f"(< {self.hitl_threshold:.2f}): revisar antes de persistir."
                    )
        return reasons

    def _missing(self, modality: Modality) -> IngestionOutput:
        agent = self.agents[modality]
        return IngestionOutput(
            agent=f"{agent.name}@{agent.version}",
            modality=modality,
            support=agent.support,
            status=ModalityStatus.MISSING,
            detail="No se aportó esta modalidad en la adquisición.",
        )
