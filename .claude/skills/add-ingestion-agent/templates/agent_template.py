"""Plantilla de agente de ingesta — cópiala a apps/<modalidad>-agent/ y rellena.

Reemplaza <MODALIDAD>, <modalidad>, y la lógica de parseo. Sustituye TODOs.
Un agente de ingesta traduce UN fichero crudo a un fragmento del contrato,
declarando su Provenance y el resultado de ingesta (fail-loud).

Ver la skill `add-ingestion-agent` para las reglas duras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core_schemas import (
    Modality,
    ModalityIngestion,
    ModalityStatus,
    Provenance,
)

AGENT_NAME = "<modalidad>-agent"
MODALITY = Modality.<MODALIDAD>  # p. ej. Modality.MESH / CBCT / REPORT


# NOTA: `IngestionOutput` es la interfaz común (issue #1). Mientras no viva en
# `packages/core-schemas`, se declara aquí; MUÉVELA allí en cuanto exista.
@dataclass
class IngestionOutput:
    """Resultado de ingerir una modalidad: fragmento de contrato + estado."""

    ingestion: ModalityIngestion            # ok/missing/failed (SIEMPRE presente)
    provenance: Provenance | None = None    # None si failed/missing
    gaussian_field_ref: str | None = None   # para modalidades con campo (mesh/cbct)
    regional: list | None = None            # RegionalObservation[] (report)
    # ... añade lo que produzca tu modalidad; NO embebas el campo gaussiano.


def ingest(source_file: Path) -> IngestionOutput:
    """Traduce `source_file` (crudo) a un fragmento del contrato.

    NO lanza al llamador: cualquier fallo se devuelve como status=FAILED.
    """
    if not source_file.is_file():
        return IngestionOutput(
            ingestion=ModalityIngestion(
                modality=MODALITY,
                status=ModalityStatus.MISSING,
                detail=f"no existe: {source_file}",
            )
        )

    try:
        # TODO: parsea el fichero crudo (pydicom / VTK / pypdf según modalidad).
        #   - mesh: preserva la superficie sin pérdida (guardarraíl ADR 004) +
        #           deriva el campo gaussiano a 3dgs-engine -> gaussian_field_ref.
        #   - cbct: reconstruye el campo σ (envuelve RGS) -> gaussian_field_ref.
        #   - report: extrae pH por FDI -> list[RegionalObservation].
        raise NotImplementedError("implementa el parseo de la modalidad")
    except Exception as exc:  # noqa: BLE001 — fail-loud: se devuelve, no se propaga
        return IngestionOutput(
            ingestion=ModalityIngestion(
                modality=MODALITY, status=ModalityStatus.FAILED, detail=str(exc)
            )
        )

    provenance = Provenance(
        source_file=str(source_file),
        modality=MODALITY,
        agent=AGENT_NAME,
        confidence=1.0,  # TODO: ajusta si el parseo tiene incertidumbre real
    )
    return IngestionOutput(
        ingestion=ModalityIngestion(modality=MODALITY, status=ModalityStatus.OK),
        provenance=provenance,
        # gaussian_field_ref=... / regional=...   según la modalidad
    )
