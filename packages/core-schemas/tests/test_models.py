"""Tests del contrato de datos (`core-schemas`).

Cubren las garantías que ahora tienen lógica clínica, no solo tipos:
  · versión de contrato serializada  (schema_version)
  · manejo explícito de fallos de ingesta  (ModalityIngestion / status)
  · retrocompatibilidad con la forma que usan los notebooks
  · validación de formato ISO-FDI
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from core_schemas import (
    SCHEMA_VERSION,
    ClinicalAttributes,
    Modality,
    ModalityIngestion,
    ModalityStatus,
    Provenance,
    RegionalObservation,
    TwinSnapshot,
)
from pydantic import ValidationError


def _prov(**kw) -> Provenance:
    base = dict(source_file="scan.obj", modality=Modality.MESH, agent="mesh-agent")
    base.update(kw)
    return Provenance(**base)


def _snap(**kw) -> TwinSnapshot:
    base = dict(
        acquisition_id="A1",
        timestamp=datetime.now(timezone.utc),
        gaussian_field_ref="sha256:abc",
        provenance=_prov(),
    )
    base.update(kw)
    return TwinSnapshot(**base)


# --- #8 schema_version ------------------------------------------------------ #
def test_schema_version_default_y_en_dump():
    snap = _snap()
    assert snap.schema_version == SCHEMA_VERSION
    assert f'"schema_version":"{SCHEMA_VERSION}"' in snap.model_dump_json()


def test_round_trip_json_identico():
    snap = _snap(
        ingestion=[ModalityIngestion(modality=Modality.MESH, status=ModalityStatus.OK)]
    )
    assert TwinSnapshot.model_validate_json(snap.model_dump_json()) == snap


# --- #4 manejo explícito de fallos de ingesta ------------------------------- #
def test_snapshot_parcial_se_declara():
    snap = _snap(
        modalities=[Modality.MESH],
        ingestion=[
            ModalityIngestion(modality=Modality.MESH, status=ModalityStatus.OK),
            ModalityIngestion(
                modality=Modality.CBCT, status=ModalityStatus.FAILED, detail="DICOM corrupto"
            ),
            ModalityIngestion(modality=Modality.REPORT, status=ModalityStatus.MISSING),
        ],
    )
    estados = {r.modality: r.status for r in snap.ingestion}
    assert estados[Modality.CBCT] is ModalityStatus.FAILED
    assert estados[Modality.REPORT] is ModalityStatus.MISSING


def test_retrocompatibilidad_forma_notebook():
    """`TwinSnapshot(modalities=[...])` sin campos nuevos → defaults, sin romper."""
    snap = _snap(modalities=[Modality.MESH])
    assert snap.ingestion == []
    assert snap.schema_version == SCHEMA_VERSION


# --- validación de formato ISO-FDI (regresión) ------------------------------ #
@pytest.mark.parametrize("code", ["16", "48", "51", "85"])
def test_fdi_valido(code):
    obs = RegionalObservation(
        region_id=code,
        attributes=ClinicalAttributes(ph=7.0),
        timestamp=datetime.now(timezone.utc),
        provenance=_prov(modality=Modality.REPORT, agent="report-agent"),
    )
    assert obs.region_id == code


@pytest.mark.parametrize("code", ["00", "49", "86", "9", "160"])
def test_fdi_invalido_se_rechaza(code):
    with pytest.raises(ValidationError):
        RegionalObservation(
            region_id=code,
            attributes=ClinicalAttributes(ph=7.0),
            timestamp=datetime.now(timezone.utc),
            provenance=_prov(modality=Modality.REPORT, agent="report-agent"),
        )
