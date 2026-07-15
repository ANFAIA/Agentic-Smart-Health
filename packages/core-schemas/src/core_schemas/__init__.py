"""Esquemas Pydantic compartidos del Digital Twin dental (contrato de datos).

Punto único de importación para el resto del monorepo. Los agentes se comunican
exclusivamente a través de estos modelos (ver AGENTS.md y ADR 001).
"""

from core_schemas.models import (
    ClinicalAttributes,
    Color,
    FDICode,
    GaussianPrimitive,
    Modality,
    PatientDigitalTwin,
    Provenance,
    RegionalObservation,
    Support,
    TwinSnapshot,
)

__all__ = [
    "ClinicalAttributes",
    "Color",
    "FDICode",
    "GaussianPrimitive",
    "Modality",
    "PatientDigitalTwin",
    "Provenance",
    "RegionalObservation",
    "Support",
    "TwinSnapshot",
]
