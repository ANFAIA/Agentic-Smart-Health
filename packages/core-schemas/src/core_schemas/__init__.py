"""Esquemas Pydantic compartidos del Digital Twin dental (contrato de datos).

Punto único de importación para el resto del monorepo. Los agentes se comunican
exclusivamente a través de estos modelos (ver AGENTS.md y ADR 001).
"""

from core_schemas.models import (
    SCHEMA_VERSION,
    ClinicalAttributes,
    Color,
    FDICode,
    GaussianPrimitive,
    Modality,
    ModalityIngestion,
    ModalityStatus,
    PatientDigitalTwin,
    Provenance,
    RegionalObservation,
    Support,
    TwinSnapshot,
)

__all__ = [
    "SCHEMA_VERSION",
    "ClinicalAttributes",
    "Color",
    "FDICode",
    "GaussianPrimitive",
    "Modality",
    "ModalityIngestion",
    "ModalityStatus",
    "PatientDigitalTwin",
    "Provenance",
    "RegionalObservation",
    "Support",
    "TwinSnapshot",
]
