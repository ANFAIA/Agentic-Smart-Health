"""Orquestador de agentes — Agentic Smart Health.

Hoy cubre la **Fase 1 (ingesta)**: dispara los agentes de `ingestion-agents`,
aplica el gate de human-in-the-loop y ensambla el `TwinSnapshot` del contrato.
Las fases de fusión, segmentación, análisis y export se irán añadiendo aquí.
"""

from agent_orchestrator.pipeline import (
    DEFAULT_HITL_THRESHOLD,
    LATENCY_BUDGET_S,
    CaseInput,
    IngestionPipeline,
    PipelineResult,
)

__all__ = [
    "DEFAULT_HITL_THRESHOLD",
    "LATENCY_BUDGET_S",
    "CaseInput",
    "IngestionPipeline",
    "PipelineResult",
]
