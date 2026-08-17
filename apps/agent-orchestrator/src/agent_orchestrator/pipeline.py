"""Orquestación de la fase de **ingesta**: los agentes → un `TwinSnapshot`.

Este módulo es la Fase 1 del pipeline (`docs/architecture/multi-agent-pipeline.md`
§6): dispara los agentes de ingesta, recoge sus `IngestionOutput` y **ensambla el
contrato**. `fuse()` encadena después las etapas que enriquecen ese snapshot
—fusión geométrica → segmentación → fusión semántica—, cada una solo si se le da
su entrada. El análisis clínico (patología) sigue fuera.

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
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis_agents import AnalysisOutput, SegmentationAgent, Segmenter
from core_schemas import (
    Modality,
    ModalityStatus,
    PatientDigitalTwin,
    Provenance,
    TwinSnapshot,
)
from export_agents import (
    ExportAgent,
    ExportOutput,
    FieldExportAgent,
    RenderExportAgent,
)
from fusion_agents import (
    FusionOutput,
    GeometricFusionAgent,
    SemanticFusionAgent,
    insert_snapshot,
)
from ingestion_agents import (
    ArtifactStore,
    CBCTAgent,
    ImageAgent,
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
    # Varias fotos por adquisición (Bite2Text trae 5): 0..N, no exactamente una.
    images: list[Path] = field(default_factory=list)
    timestamp: datetime | None = None

    @classmethod
    def from_case_dir(cls, root: Path, **kwargs: object) -> CaseInput:
        """Descubre las modalidades en el layout que produce `synthetic.write_case`
        (y en el de Bite2Text: fotos en `intraoral-photo/`)."""
        root = Path(root)
        objs = sorted(root.glob("*.obj"))
        reports = sorted(root.glob("*.txt")) + sorted(root.glob("*.pdf"))
        cbct = root / "cbct"
        # JPG/PNG en la raíz o en un subdir de fotos. HEIC se omite: necesita el
        # extra `heic`, así que no se descubre por defecto.
        photos: list[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "intraoral-photo/*"):
            photos += [p for p in sorted(root.glob(pattern)) if p.suffix.lower() in
                       {".jpg", ".jpeg", ".png"}]
        return cls(
            acquisition_id=kwargs.pop("acquisition_id", root.name),  # type: ignore[arg-type]
            mesh=objs[0] if objs else None,
            cbct=cbct if cbct.is_dir() else None,
            report=reports[0] if reports else None,
            images=photos,
            **kwargs,  # type: ignore[arg-type]
        )


@dataclass
class PipelineResult:
    """Resultado completo de la ingesta: el contrato + todo lo que hizo falta para juzgarlo."""

    snapshot: TwinSnapshot | None
    outcomes: list[IngestionOutput] = field(default_factory=list)
    fusion: list[FusionOutput] = field(default_factory=list)
    # La segmentación no es fusión: corre entre las dos etapas y devuelve además lo
    # que ha inferido (el mapa FDI → confianza). Lista aparte para que `fusion` siga
    # significando exactamente lo que dice.
    analysis: list[AnalysisOutput] = field(default_factory=list)
    # La exportación tampoco es fusión ni análisis: no enriquece el snapshot, lo
    # materializa. Lista aparte por lo mismo, y porque su salida —ficheros y errores
    # medidos— no cabe en un `FusionOutput`, que declara `extra="forbid"`.
    exports: list[ExportOutput] = field(default_factory=list)
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

    @property
    def image_outcomes(self) -> list[IngestionOutput]:
        """Todas las fotos ingeridas (la imagen es 0..N, no una)."""
        return [o for o in self.outcomes if o.modality is Modality.IMAGE]

    def export(self, agente: str) -> ExportOutput | None:
        """La salida de un exportador por nombre (`export-agent`, `field-export-agent`…)."""
        return next((e for e in self.exports if e.agent.split("@")[0] == agente), None)

    @property
    def reversible(self) -> bool:
        """¿El recorrido entrada → twin → fichero cumple los presupuestos del brief?

        Exige que **todos** los exportadores que corrieron con éxito estén dentro de su
        presupuesto, cada uno en el canal que le toca: milímetros para lo que es geometría,
        PSNR/SSIM para lo que es imagen. Un exportador que no verificó devuelve `False` en
        su propiedad, así que aquí también: sin medida no hay reversibilidad demostrada.

        Un canal en `FAILED` deja el recorrido en rojo aunque los demás cumplan: es un
        fichero que se pidió y no se pudo escribir. Uno en `MISSING`, no — que una
        adquisición no traiga malla es normal y no dice nada del recorrido.

        `False` si no se exportó nada — no se puede afirmar que un recorrido es reversible
        sin haberlo recorrido.
        """
        if any(e.status is ModalityStatus.FAILED for e in self.exports):
            return False
        hechos = [e for e in self.exports if e.ok]
        if not hechos:
            return False
        return all(
            e.image_within_budget if e.format == "png" else e.within_budget for e in hechos
        )


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
        segmenter: Segmenter | None = None,
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
            Modality.IMAGE: ImageAgent(store, quarantine_dir=quarantine_dir),
        }
        # La fusión no es una modalidad: son etapas posteriores que enriquecen el
        # snapshot ya ensamblado (ADR 004). Por eso viven aparte del dict de ingesta.
        self.geometric_fusion = GeometricFusionAgent(
            hitl_threshold=hitl_threshold, quarantine_dir=quarantine_dir
        )
        self.semantic_fusion = SemanticFusionAgent(
            hitl_threshold=hitl_threshold, quarantine_dir=quarantine_dir
        )
        # La segmentación necesita un modelo, y no hay uno por defecto a propósito
        # (ver `SegmentationAgent`): sin `segmenter` la etapa sencillamente no corre
        # y el ancla FDI se pasa a mano, como hasta ahora.
        self.segmentation = (
            None
            if segmenter is None
            else SegmentationAgent(
                store,
                segmenter=segmenter,
                hitl_threshold=hitl_threshold,
                quarantine_dir=quarantine_dir,
            )
        )

    # --- ejecución ------------------------------------------------------- #
    def run(self, case: CaseInput) -> PipelineResult:
        started = time.perf_counter()
        single = {
            Modality.MESH: case.mesh,
            Modality.CBCT: case.cbct,
            Modality.REPORT: case.report,
        }

        # Una tarea por modalidad *single* aportada + una por CADA foto: la imagen
        # es 0..N, no exactamente una, así que cada foto es su propia ingesta.
        tasks: list[tuple[Modality, Path]] = [
            (m, p) for m, p in single.items() if p is not None
        ]
        tasks += [(Modality.IMAGE, p) for p in case.images]

        if self.parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                futures = [(m, pool.submit(self.agents[m].ingest, p)) for m, p in tasks]
                done = [(m, f.result()) for m, f in futures]
        else:
            done = [(m, self.agents[m].ingest(p)) for m, p in tasks]

        # Las single no aportadas se declaran MISSING (así `missing` = no había
        # fichero, y `failed` = lo había y no se pudo leer, nunca se confunden).
        by_mod = {m: o for m, o in done if m is not Modality.IMAGE}
        for m in (Modality.MESH, Modality.CBCT, Modality.REPORT):
            by_mod.setdefault(m, self._missing(m))
        image_outcomes = [o for m, o in done if m is Modality.IMAGE]

        ordered = [
            by_mod[Modality.CBCT], by_mod[Modality.MESH], by_mod[Modality.REPORT],
            *image_outcomes,
        ]
        pseudonym = pseudonymize(case.patient_id)
        snapshot = self._assemble(case, ordered, pseudonym)

        return PipelineResult(
            snapshot=snapshot,
            outcomes=ordered,
            hitl_reasons=self._hitl_reasons(ordered, snapshot),
            latency_s=time.perf_counter() - started,
            patient_pseudonym=pseudonym,
        )

    # --- fusión (ADR 004) ------------------------------------------------ #
    def fuse(
        self,
        result: PipelineResult,
        *,
        registration: tuple[Any, Any] | None = None,
        detected: Mapping[str, float] | None = None,
    ) -> PipelineResult:
        """Aplica las etapas de fusión sobre un resultado de ingesta.

        Cada etapa corre **solo si se le da su entrada**, y en el orden del pipeline:
        geométrica (registro malla↔CBCT) → **segmentación** (puebla `region_id`) →
        semántica (anclaje al FDI).

        `detected` sigue siendo un parámetro y **manda sobre la segmentación**: es
        por donde entran las etiquetas revisadas por un clínico. Human-in-the-loop
        sin esta puerta sería un adorno — si la corrección humana no pudiera
        sustituir a la del modelo, no habría nada que revisar.

        Devuelve un `PipelineResult` **nuevo**: ni muta el de entrada ni el snapshot.

        Si una etapa falla, se registra su salida, se añade el motivo a
        `hitl_reasons` y **se conserva el último snapshot bueno**. Estas etapas
        enriquecen; que fallen no debe destruir lo que la ingesta sí consiguió.
        """
        snapshot, salidas, analisis, motivos = result.snapshot, [], [], []

        if snapshot is not None and registration is not None:
            origen, destino = registration
            out = self.geometric_fusion.fuse(snapshot, source=origen, target=destino)
            salidas.append(out)
            motivos += self._stage_reasons(out)
            snapshot = out.snapshot or snapshot

        if snapshot is not None and detected is None and self.segmentation is not None:
            seg = self.segmentation.analyze(snapshot)
            analisis.append(seg)
            motivos += self._stage_reasons(seg)
            snapshot = seg.snapshot or snapshot
            # Solo se ancla contra lo que la segmentación encontró de verdad: si
            # falló, `detected` sigue vacío y la fusión semántica se declarará
            # MISSING en vez de anclar el informe contra un ancla que no existe.
            detected = seg.detected or None

        if snapshot is not None and detected is not None:
            out = self.semantic_fusion.fuse(snapshot, detected=detected)
            salidas.append(out)
            motivos += self._stage_reasons(out)
            snapshot = out.snapshot or snapshot

        return replace(
            result,
            snapshot=snapshot,
            fusion=[*result.fusion, *salidas],
            analysis=[*result.analysis, *analisis],
            hitl_reasons=[*result.hitl_reasons, *motivos],
        )

    def exportar(
        self,
        result: PipelineResult,
        destino: str | Path,
        *,
        marco_malla: str = "source",
        render: bool = True,
    ) -> PipelineResult:
        """Materializa el snapshot en `destino`: STL + PLY + render, con su error medido.

        Es la **fase 6**, y va en un método aparte —no dentro de `run`— por una razón de
        contrato: exportar **escribe ficheros**. `run` y `fuse` son puras respecto al disco
        salvo por el almacén de artefactos, y meter escritura de salida ahí obligaría a
        todos sus llamantes a pasar una ruta que casi ninguno quiere.

        Los tres canales corren **independientes**: que no haya malla no impide exportar el
        campo, y que falle el render no borra el STL ya escrito. Cada uno declara su propio
        estado, y sus motivos de revisión se acumulan en el resultado como los de la fusión.

        `marco_malla` elige el sistema del STL (`"source"` como entró, `"twin"` aplicando la
        transformación de la fusión geométrica). El PLY se escribe siempre en el marco del
        twin: pedir `"cbct"` exigiría `origin` en el artefacto, y un snapshot ingerido con
        una versión antigua del `cbct-agent` no lo tiene — que falle la exportación entera
        por eso sería desproporcionado. Quien lo necesite llama al agente directamente.

        Devuelve un `PipelineResult` **nuevo**: ni muta el de entrada ni el snapshot.
        """
        if result.snapshot is None:
            return replace(
                result,
                hitl_reasons=[
                    *result.hitl_reasons,
                    "no hay snapshot que exportar: la ingesta no llegó a ensamblar el twin",
                ],
            )

        destino = Path(destino)
        snapshot = result.snapshot
        salidas: list[ExportOutput] = [
            ExportAgent(self.store, frame=marco_malla).export(  # type: ignore[arg-type]
                snapshot, destino / f"{snapshot.acquisition_id}.stl"
            ),
            FieldExportAgent(self.store).export(
                snapshot, destino / f"{snapshot.acquisition_id}.ply"
            ),
        ]
        if render:
            # El render se verifica contra el PLY que se acaba de escribir: así la métrica
            # de imagen mide el ciclo completo twin → fichero → render y no el render
            # contra sí mismo. Si el PLY no llegó a escribirse, se exporta sin verificar.
            ply = salidas[1].path if salidas[1].ok else None
            salidas.append(
                RenderExportAgent(self.store).export(
                    snapshot, destino / "render", ply=ply
                )
            )

        motivos = [m for s in salidas for m in self._export_reasons(s)]
        return replace(
            result,
            exports=[*result.exports, *salidas],
            hitl_reasons=[*result.hitl_reasons, *motivos],
        )

    def _stage_reasons(self, out: FusionOutput | AnalysisOutput) -> list[str]:
        """Los motivos del agente, más el de que la etapa no llegara a hacerse.

        Igual que en ingesta: el agente reporta, la decisión de parar vive aquí.
        """
        if out.status is ModalityStatus.FAILED:
            return [f"{out.agent} falló: {out.detail}"]
        if out.status is ModalityStatus.MISSING:
            return [f"{out.agent} no pudo fusionar: {out.detail}"]
        return list(out.hitl_reasons)

    def _export_reasons(self, out: ExportOutput) -> list[str]:
        """Como `_stage_reasons`, pero `MISSING` aquí no es un problema.

        Que un snapshot no traiga malla es normal —una adquisición puede ser solo CBCT— y no
        dice nada malo del fichero que sí se escribió. Un fallo sí, y se declara.
        """
        if out.status is ModalityStatus.FAILED:
            return [f"{out.agent} falló al exportar: {out.detail}"]
        return list(out.hitl_reasons)

    def run_into_twin(
        self,
        case: CaseInput,
        twin: PatientDigitalTwin | None = None,
        *,
        registration: tuple[Any, Any] | None = None,
        detected: Mapping[str, float] | None = None,
    ) -> tuple[PatientDigitalTwin, PipelineResult]:
        """Ingiere, fusiona/segmenta si hay con qué, y encaja el snapshot en la serie."""
        result = self.run(case)
        if registration is not None or detected is not None or self.segmentation is not None:
            result = self.fuse(result, registration=registration, detected=detected)
        if twin is None:
            twin = PatientDigitalTwin(patient_id=result.patient_pseudonym)
        if twin.patient_id != result.patient_pseudonym:
            raise ValueError(
                "El snapshot es de otro paciente: "
                f"{result.patient_pseudonym} != {twin.patient_id}"
            )
        if result.snapshot is not None:
            # `insert_snapshot` y no `append`: la identidad de visita es el
            # `acquisition_id`, así que reprocesar la misma adquisición REEMPLAZA.
            # Con `append`, cada reejecución inflaba el historial clínico con una
            # visita que nunca ocurrió (ADR 004 §2.5 · issue #33).
            twin = insert_snapshot(twin, result.snapshot)
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
        # Todas las fotos que se ingirieron bien → lista de referencias (apariencia
        # pre-fusión). La comprobación de referencia colgante de arriba ya las validó.
        image_refs = [
            o.artifact_ref
            for o in outcomes
            if o.modality is Modality.IMAGE and o.ok and o.artifact_ref
        ]

        return TwinSnapshot(
            acquisition_id=case.acquisition_id,
            timestamp=case.timestamp or datetime.now(UTC),
            # dedup: con N fotos, IMAGE aparecería N veces en la lista de outcomes.
            modalities=list(dict.fromkeys(o.modality for o in outcomes if o.ok)),
            ingestion=[o.ingestion for o in outcomes],
            gaussian_field_ref=cbct.artifact_ref,
            surface_ref=mesh.artifact_ref if mesh.ok else None,
            image_refs=image_refs,
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
