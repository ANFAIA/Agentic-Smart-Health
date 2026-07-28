"""El pipeline de ingesta: ensamblado del contrato, gate humano y fail-loud."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from agent_orchestrator import LATENCY_BUDGET_S, CaseInput, IngestionPipeline
from core_schemas import Modality, ModalityStatus, PatientDigitalTwin, TwinSnapshot
from ingestion_agents import ArtifactStore, synthetic


@pytest.fixture(scope="session")
def case_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("caso") / "acq-001"
    synthetic.write_case(root, spacing=1.2)
    return root


@pytest.fixture
def pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        ArtifactStore(tmp_path / "artifacts"), quarantine_dir=tmp_path / "quarantine"
    )


# --- caso completo --------------------------------------------------------- #
def test_las_tres_modalidades_producen_un_snapshot(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    result = pipeline.run(CaseInput.from_case_dir(case_dir))

    assert isinstance(result.snapshot, TwinSnapshot)
    assert {o.modality for o in result.outcomes} == {
        Modality.CBCT,
        Modality.MESH,
        Modality.REPORT,
    }
    assert all(o.ok for o in result.outcomes)


def test_el_snapshot_referencia_ambos_soportes(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """Volumétrico (CBCT) y superficial (malla) van por referencias distintas."""
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    assert snapshot is not None
    assert snapshot.gaussian_field_ref.startswith("sha256:")
    assert snapshot.surface_ref is not None
    assert snapshot.gaussian_field_ref != snapshot.surface_ref


def test_las_referencias_resuelven_en_el_almacen(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    assert snapshot is not None
    assert pipeline.store.exists(snapshot.gaussian_field_ref)
    assert pipeline.store.exists(snapshot.surface_ref or "")


def test_el_log_de_ingesta_cubre_las_tres_modalidades(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """Fail-loud: el snapshot declara qué pasó con cada modalidad, incluidas las ausentes."""
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    assert snapshot is not None
    assert len(snapshot.ingestion) == 3
    assert all(entry.status is ModalityStatus.OK for entry in snapshot.ingestion)


def test_los_hallazgos_regionales_llegan_al_contrato(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    assert snapshot is not None
    assert {obs.region_id for obs in snapshot.regional} == {"11", "16", "21", "26"}


def test_el_snapshot_serializa_a_json(pipeline: IngestionPipeline, case_dir: Path) -> None:
    """El canal de export de metadatos es `model_dump` puro: no debe requerir nada extra."""
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    assert snapshot is not None
    assert TwinSnapshot.model_validate(snapshot.model_dump(mode="json")) == snapshot


# --- métrica de latencia --------------------------------------------------- #
def test_dentro_del_presupuesto_de_latencia(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """Métrica acordada con el partner: ingesta de las tres modalidades < 60 s."""
    result = pipeline.run(CaseInput.from_case_dir(case_dir))
    assert result.latency_s < LATENCY_BUDGET_S
    assert result.within_budget


def test_secuencial_y_paralelo_dan_el_mismo_resultado(
    tmp_path: Path, case_dir: Path
) -> None:
    """La concurrencia es una optimización, no puede cambiar lo que se ingiere."""
    case = CaseInput.from_case_dir(case_dir)
    refs = []
    for parallel in (True, False):
        pipe = IngestionPipeline(ArtifactStore(tmp_path / f"art-{parallel}"), parallel=parallel)
        snapshot = pipe.run(case).snapshot
        assert snapshot is not None
        refs.append((snapshot.gaussian_field_ref, snapshot.surface_ref))
    assert refs[0] == refs[1]


# --- modalidades ausentes y fallidas --------------------------------------- #
def test_modalidad_no_aportada_es_missing(pipeline: IngestionPipeline, case_dir: Path) -> None:
    case = CaseInput.from_case_dir(case_dir)
    case.report = None
    result = pipeline.run(case)
    report = result.outcome(Modality.REPORT)
    assert report is not None and report.status is ModalityStatus.MISSING


def test_sin_cbct_no_hay_snapshot(pipeline: IngestionPipeline, case_dir: Path) -> None:
    """Un twin *es* el campo gaussiano más sus metadatos: sin campo no se emite contrato."""
    case = CaseInput.from_case_dir(case_dir)
    case.cbct = None
    result = pipeline.run(case)
    assert result.snapshot is None
    assert result.hitl_required
    assert any("Sin campo gaussiano" in r for r in result.hitl_reasons)


def test_una_modalidad_rota_no_tumba_las_demas(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """El motivo de que la ingesta sea fail-loud y no fail-fast."""
    roto = tmp_path / "roto.obj"
    roto.write_text("esto no es un OBJ\n")
    case = CaseInput.from_case_dir(case_dir)
    case.mesh = roto

    result = pipeline.run(case)

    mesh = result.outcome(Modality.MESH)
    assert mesh is not None and mesh.status is ModalityStatus.FAILED
    assert result.snapshot is not None          # el CBCT sí entró
    assert result.snapshot.surface_ref is None  # y la malla se declara ausente
    assert Modality.MESH not in result.snapshot.modalities


def test_el_fallo_queda_en_cuarentena(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    roto = tmp_path / "roto.obj"
    roto.write_text("esto no es un OBJ\n")
    case = CaseInput.from_case_dir(case_dir)
    case.mesh = roto
    pipeline.run(case)
    assert list((tmp_path / "quarantine").glob("*.json"))


# --- human-in-the-loop ----------------------------------------------------- #
def test_caso_limpio_no_requiere_revision(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    assert not pipeline.run(CaseInput.from_case_dir(case_dir)).hitl_required


def test_confianza_baja_dispara_el_gate(tmp_path: Path, case_dir: Path) -> None:
    """El agente reporta confianza; la decisión de parar vive en el orquestador."""
    pipe = IngestionPipeline(ArtifactStore(tmp_path / "art"), hitl_threshold=0.95)
    result = pipe.run(CaseInput.from_case_dir(case_dir))
    assert result.hitl_required
    assert any("confianza" in r for r in result.hitl_reasons)


def test_un_hallazgo_descartado_del_informe_dispara_el_gate(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """Cierre del diseño: el `report-agent` no decide nada, solo declara que
    descartó algo bajando la confianza; es aquí donde eso para el flujo.

    Sin esto, un pH mal tecleado por el clínico (`74` por `7.4`) desaparecería
    del twin sin que nadie se enterara."""
    informe = tmp_path / "informe.txt"
    informe.write_text("Diente 16: pH 5.1\nDiente 47: pH 74\n", encoding="utf-8")
    case = CaseInput.from_case_dir(case_dir)
    case.report = informe

    result = pipeline.run(case)

    assert result.hitl_required
    assert any("confianza" in r for r in result.hitl_reasons)


def test_una_modalidad_fallida_dispara_el_gate(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    roto = tmp_path / "roto.obj"
    roto.write_text("no es un OBJ\n")
    case = CaseInput.from_case_dir(case_dir)
    case.mesh = roto
    assert pipeline.run(case).hitl_required


# --- anonimización y serie temporal ---------------------------------------- #
def test_el_paciente_se_seudonimiza(pipeline: IngestionPipeline, case_dir: Path) -> None:
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-REAL-001"))
    assert result.patient_pseudonym
    assert "PAC-REAL-001" not in result.patient_pseudonym


def test_el_twin_acumula_snapshots(pipeline: IngestionPipeline, case_dir: Path) -> None:
    """La serie temporal del paciente se construye añadiendo adquisiciones."""
    case = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    case.timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    twin, _ = pipeline.run_into_twin(case)

    case2 = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    case2.acquisition_id = "acq-002"
    case2.timestamp = datetime(2026, 6, 1, tzinfo=UTC)
    twin, _ = pipeline.run_into_twin(case2, twin)

    assert isinstance(twin, PatientDigitalTwin)
    assert len(twin.snapshots) == 2
    latest = twin.latest()
    assert latest is not None and latest.acquisition_id == "acq-002"


def test_la_serie_de_ph_se_reconstruye(pipeline: IngestionPipeline, case_dir: Path) -> None:
    """La consulta que sostiene la evaluación de la evolución clínica del paciente."""
    case = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    twin, _ = pipeline.run_into_twin(case)

    serie = twin.series("16", "ph")
    assert len(serie) == 1
    instante, valor = serie[0]
    # El pH lo fija el informe sintético; el instante, la fecha que declara.
    assert valor == pytest.approx(5.2)
    assert isinstance(instante, datetime)


def test_no_se_mezclan_pacientes(pipeline: IngestionPipeline, case_dir: Path) -> None:
    twin, _ = pipeline.run_into_twin(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    with pytest.raises(ValueError, match="otro paciente"):
        pipeline.run_into_twin(
            CaseInput.from_case_dir(case_dir, patient_id="PAC-002"), twin
        )


# --- descubrimiento del caso ----------------------------------------------- #
def test_from_case_dir_descubre_las_modalidades(case_dir: Path) -> None:
    case = CaseInput.from_case_dir(case_dir)
    assert case.acquisition_id == "acq-001"
    assert case.mesh is not None and case.mesh.suffix == ".obj"
    assert case.cbct is not None and case.cbct.is_dir()
    assert case.report is not None


def test_from_case_dir_en_directorio_vacio(tmp_path: Path) -> None:
    case = CaseInput.from_case_dir(tmp_path)
    assert (case.mesh, case.cbct, case.report) == (None, None, None)
