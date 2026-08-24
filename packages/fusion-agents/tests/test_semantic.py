"""`semantic-fusion-agent`: anclaje de observaciones al FDI y gate HITL (ADR 004)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from core_schemas import (
    ClinicalAttributes,
    Hallazgo,
    Modality,
    ModalityStatus,
    PatientDigitalTwin,
    Provenance,
    RegionalObservation,
    TwinSnapshot,
)
from fusion_agents import SemanticFusionAgent, insert_snapshot


def _obs(
    fdi: str,
    *,
    confianza: float = 1.0,
    ph: float = 6.5,
    hallazgos: list | None = None,
) -> RegionalObservation:
    return RegionalObservation(
        region_id=fdi,
        attributes=ClinicalAttributes(ph=ph, hallazgos=hallazgos or []),
        timestamp=datetime.now(UTC),
        provenance=Provenance(
            source_file="informe.pdf",
            modality=Modality.REPORT,
            agent="report-agent@1.0.0",
            confidence=confianza,
        ),
    )


def _snap(observaciones: list[RegionalObservation], *, aid: str = "A1", **kw) -> TwinSnapshot:
    base = dict(
        acquisition_id=aid,
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:abc",
        regional=observaciones,
        provenance=Provenance(
            source_file="caso/", modality=Modality.MESH, agent="mesh-agent@1.0.0"
        ),
    )
    base.update(kw)
    return TwinSnapshot(**base)


@pytest.fixture
def agente() -> SemanticFusionAgent:
    return SemanticFusionAgent()


# --- anclaje normal --------------------------------------------------------- #
def test_ancla_la_observacion_cuando_el_diente_existe(agente):
    out = agente.fuse(_snap([_obs("46")]), detected={"46": 0.95})
    assert out.ok
    assert not out.hitl_required
    assert out.snapshot.regional[0].region_id == "46"


def test_la_confianza_es_el_eslabon_mas_debil(agente):
    """min(), no producto: encadenar productos hundiría todo bajo el umbral."""
    out = agente.fuse(_snap([_obs("46", confianza=0.9)]), detected={"46": 0.8})
    assert out.snapshot.regional[0].provenance.confidence == pytest.approx(0.8)


def test_la_confianza_baja_del_informe_tambien_manda(agente):
    out = agente.fuse(_snap([_obs("46", confianza=0.75)]), detected={"46": 0.99})
    assert out.snapshot.regional[0].provenance.confidence == pytest.approx(0.75)


# --- conflicto: la decisión que el ADR 004 §2.4 NO toma --------------------- #
def test_conflicto_conserva_el_fdi_del_informe(agente):
    """No se elige ganador: la fuente clínica manda, y se marca."""
    out = agente.fuse(_snap([_obs("46")]), detected={"45": 0.9})
    assert out.ok  # el conflicto no es un fallo del agente
    assert out.snapshot.regional[0].region_id == "46"


def test_conflicto_pone_la_confianza_a_cero_y_pide_revision(agente):
    out = agente.fuse(_snap([_obs("46")]), detected={"45": 0.9})
    assert out.snapshot.regional[0].provenance.confidence == 0.0
    assert out.hitl_required
    assert "46" in out.hitl_reasons[0]


def test_confianza_bajo_el_umbral_pide_revision_sin_conflicto(agente):
    """El gate no es solo para conflictos: también para anclajes poco fiables."""
    out = agente.fuse(_snap([_obs("46", confianza=0.5)]), detected={"46": 0.9})
    assert out.hitl_required
    assert out.snapshot.regional[0].provenance.confidence == pytest.approx(0.5)


def test_umbral_configurable():
    permisivo = SemanticFusionAgent(hitl_threshold=0.4)
    out = permisivo.fuse(_snap([_obs("46", confianza=0.5)]), detected={"46": 0.9})
    assert not out.hitl_required


# --- trazabilidad ----------------------------------------------------------- #
def test_conserva_la_procedencia_del_informe(agente):
    """El valor sigue viniendo del PDF: perder source_file rompería la trazabilidad."""
    out = agente.fuse(_snap([_obs("46")]), detected={"46": 0.9})
    prov = out.snapshot.regional[0].provenance
    assert prov.source_file == "informe.pdf"
    assert prov.modality is Modality.REPORT
    assert prov.agent == "report-agent@1.0.0"


def test_el_snapshot_registra_quien_fusiono(agente):
    """Quién derivó el snapshot sí es la fusión, y ahí es donde consta."""
    out = agente.fuse(_snap([_obs("46")]), detected={"46": 0.9})
    assert out.snapshot.provenance.agent == "semantic-fusion-agent@0.1.0"


# --- no mutar (ADR 004 §2.5) ------------------------------------------------ #
def test_no_muta_el_snapshot_de_entrada(agente):
    entrada = _snap([_obs("46")])
    antes = entrada.model_dump_json()
    agente.fuse(entrada, detected={"45": 0.9})
    assert entrada.model_dump_json() == antes


def test_conserva_el_acquisition_id(agente):
    out = agente.fuse(_snap([_obs("46")], aid="VISITA-7"), detected={"46": 0.9})
    assert out.snapshot.acquisition_id == "VISITA-7"


# --- fail-loud -------------------------------------------------------------- #
def test_sin_segmentacion_se_declara_missing(agente):
    """Sin dientes detectados no hay contra qué validar: se declara, no se inventa."""
    out = agente.fuse(_snap([_obs("46")]), detected={})
    assert out.status is ModalityStatus.MISSING
    assert out.snapshot is None
    assert "segmentation-agent" in out.detail


def test_sin_observaciones_es_ok_y_no_pide_revision(agente):
    out = agente.fuse(_snap([]), detected={"46": 0.9})
    assert out.ok
    assert out.snapshot.regional == []
    assert not out.hitl_required


def test_el_fallo_no_se_propaga_como_excepcion(agente, tmp_path):
    """Contrato fail-loud: el error es un dato, no una excepción hacia el orquestador."""
    con_cuarentena = SemanticFusionAgent(quarantine_dir=tmp_path)
    out = con_cuarentena.fuse(_snap([_obs("46")]), detected={"46": "no-es-un-numero"})
    assert out.status is ModalityStatus.FAILED
    assert out.snapshot is None
    assert out.quarantine_ref is not None
    assert list(tmp_path.glob("*.json"))


def test_la_cuarentena_no_guarda_dato_clinico(agente, tmp_path):
    con_cuarentena = SemanticFusionAgent(quarantine_dir=tmp_path)
    out = con_cuarentena.fuse(_snap([_obs("46", ph=5.1)]), detected={"46": object()})
    registro = json.loads(Path(out.quarantine_ref).read_text())
    assert registro["acquisition_id"] == "A1"
    assert "5.1" not in json.dumps(registro)  # el pH no se filtra a la cuarentena


def test_mide_latencia(agente):
    out = agente.fuse(_snap([_obs("46")]), detected={"46": 0.9})
    assert out.latency_s > 0


# --- serie temporal (ADR 004 §2.5 · issue #33) ------------------------------ #
def _twin(snaps: list[TwinSnapshot]) -> PatientDigitalTwin:
    return PatientDigitalTwin(patient_id="P1", snapshots=snaps)


def test_insertar_visita_nueva_crece_la_serie():
    twin = _twin([_snap([], aid="A1")])
    nuevo = insert_snapshot(twin, _snap([], aid="A2"))
    assert [s.acquisition_id for s in nuevo.snapshots] == ["A1", "A2"]


def test_reejecutar_la_fusion_no_duplica_la_visita():
    """La garantía de idempotencia: mismo acquisition_id → reemplaza, no añade."""
    original = _snap([_obs("46")], aid="A1")
    twin = _twin([original])
    refusionado = _snap([_obs("46", confianza=0.3)], aid="A1")
    nuevo = insert_snapshot(twin, refusionado)
    assert len(nuevo.snapshots) == 1
    assert nuevo.snapshots[0].regional[0].provenance.confidence == pytest.approx(0.3)


def test_la_serie_queda_ordenada_por_tiempo():
    ahora = datetime.now(UTC)
    twin = _twin([_snap([], aid="A2", timestamp=ahora)])
    antigua = _snap([], aid="A1", timestamp=ahora - timedelta(days=30))
    assert [s.acquisition_id for s in insert_snapshot(twin, antigua).snapshots] == ["A1", "A2"]


def test_insertar_no_muta_el_twin_de_entrada():
    twin = _twin([_snap([], aid="A1")])
    insert_snapshot(twin, _snap([], aid="A2"))
    assert len(twin.snapshots) == 1


# --- dientes que el informe da por AUSENTES --------------------------------- #
def test_un_diente_declarado_ausente_que_no_se_encuentra_NO_es_desacuerdo():
    """El falso positivo que esto arregla, y que avisaba justo cuando el pipeline
    acertaba.

    «El informe lo referencia» no es «el informe dice que existe»: una ficha dental
    habla de las 32 posiciones, incluidas las que declara ausentes. Medido sobre un caso
    real: el informe daba la 28 por ausente con confianza 0,877, el escáner no la traía
    —el paciente no tiene cordales— y el gate lo denunciaba como conflicto.
    """
    obs = _obs("28", hallazgos=[Hallazgo.AUSENTE])
    salida = SemanticFusionAgent().fuse(_snap([obs]), detected={"27": 0.9})

    assert not any("28" in m for m in salida.hitl_reasons), salida.hitl_reasons


def test_un_diente_declarado_ausente_que_SI_se_encuentra_es_desacuerdo():
    """El otro lado, que antes pasaba callado: uno de los dos se equivoca sobre si al
    paciente le falta una pieza, y eso no es un matiz."""
    obs = _obs("28", hallazgos=[Hallazgo.AUSENTE])
    salida = SemanticFusionAgent().fuse(_snap([obs]), detected={"28": 0.9})

    assert any("AUSENTE" in m and "28" in m for m in salida.hitl_reasons), salida.hitl_reasons


def test_un_diente_normal_que_no_se_encuentra_sigue_siendo_desacuerdo():
    """El arreglo no puede desactivar el aviso de verdad: si el informe habla de un
    diente con hallazgos y la segmentación no lo encuentra, eso hay que mirarlo."""
    obs = _obs("26", hallazgos=[Hallazgo.CARIES])
    salida = SemanticFusionAgent().fuse(_snap([obs]), detected={"27": 0.9})

    assert any("26" in m and "no lo encontró" in m for m in salida.hitl_reasons)
