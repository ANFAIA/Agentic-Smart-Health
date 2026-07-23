"""El contrato *fail-loud*: un agente de ingesta nunca lanza, siempre declara."""

from __future__ import annotations

import json
from pathlib import Path

from core_schemas import Modality, ModalityStatus, Support
from ingestion_agents.base import BaseIngestionAgent, IngestionAgent, IngestionOutput


class _ExplotaAgent(BaseIngestionAgent):
    """Agente de prueba que siempre revienta: modela el DICOM corrupto."""

    name = "explota-agent"
    version = "9.9.9"
    modality = Modality.CBCT
    support = Support.VOLUMETRIC

    def _ingest(self, source: Path) -> IngestionOutput:
        raise ValueError("DICOM corrupto")


class _OkAgent(BaseIngestionAgent):
    name = "ok-agent"
    version = "0.1.0"
    modality = Modality.MESH
    support = Support.SURFACE

    def _ingest(self, source: Path) -> IngestionOutput:
        return self._success(source, n_primitives=7)


def test_una_excepcion_no_escapa_al_orquestador(tmp_path: Path) -> None:
    """Es lo que impide que un fichero corrupto se lleve por delante las otras dos modalidades."""
    src = tmp_path / "roto.dcm"
    src.write_bytes(b"basura")

    outcome = _ExplotaAgent().ingest(src)

    assert outcome.status is ModalityStatus.FAILED
    assert "DICOM corrupto" in (outcome.detail or "")
    assert outcome.ok is False


def test_fichero_ausente_es_missing_no_failed(tmp_path: Path) -> None:
    """`missing` (no se aportó) y `failed` (se aportó y falló) no se confunden."""
    outcome = _ExplotaAgent().ingest(tmp_path / "no-existe.dcm")
    assert outcome.status is ModalityStatus.MISSING
    assert outcome.provenance is None


def test_cuarentena_registra_el_caso_sin_copiar_el_dato(tmp_path: Path) -> None:
    """Soberanía del dato: se guarda la ruta y el traceback, nunca el contenido clínico."""
    src = tmp_path / "paciente.dcm"
    src.write_bytes(b"contenido-clinico-sensible")
    quarantine = tmp_path / "quarantine"

    outcome = _ExplotaAgent(quarantine_dir=quarantine).ingest(src)

    assert outcome.quarantine_ref is not None
    record = json.loads(Path(outcome.quarantine_ref).read_text(encoding="utf-8"))
    assert record["source_file"] == str(src)
    assert "DICOM corrupto" in record["error"]
    assert "contenido-clinico-sensible" not in json.dumps(record)


def test_sin_directorio_de_cuarentena_no_falla(tmp_path: Path) -> None:
    """La cuarentena es diagnóstico opcional; su ausencia no puede tumbar la ingesta."""
    src = tmp_path / "roto.dcm"
    src.write_bytes(b"basura")
    assert _ExplotaAgent().ingest(src).quarantine_ref is None


def test_exito_lleva_provenance_y_latencia(tmp_path: Path) -> None:
    src = tmp_path / "malla.obj"
    src.write_text("v 0 0 0\n")

    outcome = _OkAgent().ingest(src)

    assert outcome.ok
    assert outcome.provenance is not None
    assert outcome.provenance.agent == "ok-agent@0.1.0"
    assert outcome.provenance.source_file == str(src)
    assert outcome.latency_s >= 0.0


def test_proyeccion_al_contrato(tmp_path: Path) -> None:
    src = tmp_path / "malla.obj"
    src.write_text("v 0 0 0\n")
    ingestion = _OkAgent().ingest(src).ingestion
    assert ingestion.modality is Modality.MESH
    assert ingestion.status is ModalityStatus.OK


def test_los_agentes_cumplen_el_protocolo() -> None:
    """El orquestador depende del `Protocol`, no de la clase base."""
    assert isinstance(_OkAgent(), IngestionAgent)
