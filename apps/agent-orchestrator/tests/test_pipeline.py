"""El pipeline de ingesta: ensamblado del contrato, gate humano y fail-loud."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from agent_orchestrator import LATENCY_BUDGET_S, CaseInput, IngestionPipeline
from analysis_agents import DEFAULT_CODES
from core_schemas import Modality, ModalityStatus, PatientDigitalTwin, TwinSnapshot
from ingestion_agents import ArtifactStore, synthetic
from ingestion_agents.ontology import all_fdi_codes


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
    case.reports = []
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
    case.reports = [informe]

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
    assert case.reports


def test_from_case_dir_en_directorio_vacio(tmp_path: Path) -> None:
    case = CaseInput.from_case_dir(tmp_path)
    assert (case.mesh, case.cbct, case.reports) == (None, None, [])


# --- imagen: 0..N fotos → image_refs (lista) ------------------------------- #
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _fotos(dir_: Path, n: int) -> list[Path]:
    dir_.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = dir_ / f"intraoral_{i}.jpg"
        Image.new("RGB", (48, 32), (10 * i, 100, 150)).save(p)
        out.append(p)
    return out


def test_las_fotos_van_a_image_refs(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """Con CBCT presente y N fotos, el snapshot lleva N image_refs."""
    fotos = _fotos(tmp_path / "fotos", 3)
    case = CaseInput.from_case_dir(case_dir)
    case.images = fotos

    result = pipeline.run(case)

    assert result.snapshot is not None
    assert len(result.snapshot.image_refs) == 3
    assert all(ref.startswith("sha256:") for ref in result.snapshot.image_refs)
    assert Modality.IMAGE in result.snapshot.modalities  # sin duplicar
    assert len(result.image_outcomes) == 3 and all(o.ok for o in result.image_outcomes)


def test_las_referencias_de_foto_resuelven(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    case = CaseInput.from_case_dir(case_dir)
    case.images = _fotos(tmp_path / "fotos", 2)
    snap = pipeline.run(case).snapshot
    assert snap is not None
    for ref in snap.image_refs:
        assert pipeline.store.exists(ref)


def test_sin_fotos_image_refs_vacio(pipeline: IngestionPipeline, case_dir: Path) -> None:
    snap = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    assert snap is not None and snap.image_refs == []


def test_from_case_dir_descubre_fotos(tmp_path: Path) -> None:
    """Bite2Text guarda las fotos en `intraoral-photo/`."""
    root = tmp_path / "F9999"
    _fotos(root / "intraoral-photo", 4)
    case = CaseInput.from_case_dir(root)
    assert len(case.images) == 4


def test_una_foto_rota_no_tumba_las_demas(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """Fail-loud también por foto: una corrupta se declara, las buenas entran."""
    buenas = _fotos(tmp_path / "fotos", 2)
    rota = tmp_path / "fotos" / "rota.jpg"
    rota.write_bytes(b"no soy un jpeg")
    case = CaseInput.from_case_dir(case_dir)
    case.images = [*buenas, rota]

    result = pipeline.run(case)

    assert result.snapshot is not None
    assert len(result.snapshot.image_refs) == 2  # solo las buenas
    assert any(o.status is ModalityStatus.FAILED for o in result.image_outcomes)
    assert result.hitl_required  # la foto fallida dispara el gate


# --- fusión enganchada al orquestador (ADR 004) ----------------------------- #
def _nube(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 10.0, (n, 3)) * np.array([1.0, 0.6, 0.3])


def test_sin_entradas_de_fusion_el_resultado_no_cambia(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """`run()` sigue siendo solo ingesta: la fusión no se cuela sin pedirla."""
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    assert result.fusion == []
    assert result.snapshot.provenance.transform is None


def test_la_fusion_geometrica_deja_la_transformacion(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    nube = _nube()
    fusionado = pipeline.fuse(result, registration=(nube, nube))

    assert len(fusionado.fusion) == 1
    assert fusionado.snapshot.provenance.transform is not None
    assert fusionado.snapshot.provenance.confidence == pytest.approx(1.0)


def test_la_fusion_semantica_ancla_las_observaciones(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    fdi = {o.region_id: 0.95 for o in result.snapshot.regional}
    fusionado = pipeline.fuse(result, detected=fdi)

    assert fusionado.snapshot.provenance.agent.startswith("semantic-fusion-agent")
    assert not fusionado.hitl_required


def test_el_conflicto_de_fdi_llega_al_gate_del_orquestador(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """La cadena completa: el agente marca con confianza 0 y el orquestador lo eleva.

    El informe sintético es todo maxilar (11, 16, 21, 26) y aquí no se detecta nada de esa
    arcada, así que el motivo es **de arcada** y no diente a diente — ver
    `test_una_arcada_sin_cubrir_es_un_motivo_y_no_dieciseis`.
    """
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    fusionado = pipeline.fuse(result, detected={"99": 0.9})  # ningún FDI del informe

    assert fusionado.hitl_required
    assert any("no cubrió NINGÚN diente" in m for m in fusionado.hitl_reasons)


def test_las_dos_etapas_corren_en_el_orden_del_pipeline(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    nube = _nube()
    fdi = {o.region_id: 0.95 for o in result.snapshot.regional}
    fusionado = pipeline.fuse(result, registration=(nube, nube), detected=fdi)

    agentes = [o.agent for o in fusionado.fusion]
    assert agentes[0].startswith("geometric-fusion-agent")
    assert agentes[1].startswith("semantic-fusion-agent")
    # La transformación de la etapa 1 sobrevive a la etapa 2.
    assert fusionado.snapshot.provenance.transform is not None


# --- preparación del registro dentro del orquestador ------------------------ #
def test_con_la_malla_el_orquestador_elige_las_nubes_el_solo(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """La prueba de que la #39 está cerrada: `fuse` sin `registration`, solo la ruta.

    Antes, para registrar un caso había que traer las dos nubes ya elegidas —aislar la
    arcada, quedarse con la corona, submuestrear— y eso vivía en un script. Si esto pasa,
    el pegamento ya no hace falta.
    """
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    fusionado = pipeline.fuse(result, malla=case_dir / "arcada.obj")

    assert [o.agent.split("@")[0] for o in fusionado.fusion] == ["geometric-fusion-agent"]
    assert fusionado.snapshot.provenance.transform is not None


def test_un_escaneo_que_no_dice_su_arcada_llega_al_gate(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """No basta con registrar igualmente: hay que decir contra qué se registró.

    Sin arcada el ICP encaja contra las dos y puede meter la mandíbula en el maxilar
    puntuando bien —está medido: 0,490 vs 0,509 mm, un 3,8 %—. El residuo no lo va a
    decir, así que lo dice el gate.
    """
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    fusionado = pipeline.fuse(result, malla=case_dir / "arcada.obj")

    assert any("no declara arcada" in m for m in fusionado.hitl_reasons)


def test_las_nubes_dadas_a_mano_mandan_sobre_la_preparacion(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """`registration=` sigue siendo la puerta de un registro preparado fuera."""
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    nube = _nube()
    fusionado = pipeline.fuse(result, registration=(nube, nube), malla=case_dir / "x.obj")

    assert fusionado.snapshot.provenance.confidence == pytest.approx(1.0)
    # No se preparó nada, así que no hay nada que declarar sobre la preparación.
    assert not any("no declara arcada" in m for m in fusionado.hitl_reasons)


def test_sin_malla_no_hay_nada_que_preparar_y_no_es_un_fallo(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """Una adquisición solo-CBCT es normal: `None`, no una excepción ni un motivo."""
    result = pipeline.run(
        CaseInput(acquisition_id="solo-cbct", cbct=case_dir / "cbct")
    )
    assert result.snapshot is not None and result.snapshot.surface_ref is None
    assert pipeline.prepara_registro(result, malla=case_dir / "arcada.obj") is None


def test_la_arcada_se_puede_declarar_a_mano_cuando_el_nombre_no_la_trae(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """Porque la etiqueta es del operador, y a veces no está o está mal."""
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    preparado = pipeline.prepara_registro(result, malla=case_dir / "arcada.obj")
    assert preparado is not None and preparado[2]["arcada"] is None

    declarado = pipeline.prepara_registro(
        result, malla=case_dir / "arcada.obj", arcada="maxilar"
    )
    assert declarado is not None and declarado[2]["arcada"] == "maxilar"


def test_fuse_no_muta_el_resultado_de_entrada(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    antes = result.snapshot.model_dump_json()
    pipeline.fuse(result, registration=(_nube(), _nube()))
    assert result.fusion == []
    assert result.snapshot.model_dump_json() == antes


def test_un_fallo_de_fusion_conserva_el_snapshot_de_ingesta(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """La fusión enriquece: que falle no destruye lo que la ingesta sí consiguió."""
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))

    def registrador_roto(source, target):
        raise RuntimeError("boom")

    pipeline.geometric_fusion.registrar = registrador_roto
    fusionado = pipeline.fuse(result, registration=(_nube(), _nube()))

    assert fusionado.snapshot is not None
    assert fusionado.hitl_required
    assert any("falló" in m for m in fusionado.hitl_reasons)


def test_reprocesar_la_misma_adquisicion_no_duplica_la_visita(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """ADR 004 §2.5 · issue #33: la identidad de visita es el `acquisition_id`."""
    caso = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    twin, _ = pipeline.run_into_twin(caso)
    twin, _ = pipeline.run_into_twin(
        CaseInput.from_case_dir(case_dir, patient_id="PAC-001"), twin
    )
    assert len(twin.snapshots) == 1


def test_run_into_twin_acepta_las_entradas_de_fusion(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    caso = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    nube = _nube()
    twin, result = pipeline.run_into_twin(caso, registration=(nube, nube))
    assert twin.snapshots[0].provenance.transform is not None
    assert len(result.fusion) == 1


# --- segmentación enganchada entre las dos fusiones ------------------------- #
def _segmentador(fdis: list[str], *, prob: float = 0.99):
    """Modelo de juguete: parte el campo en tantas zonas como códigos FDI se pidan.

    Se corta por `x` ordenado para que cada zona salga **conexa** —si no, la
    agregación la partiría en trozos y el agente tendría razón al quejarse—. El
    índice de clase sale del convenio de `DEFAULT_CODES` (0 = encía, 1..32 =
    dentición permanente), que es el que el agente asume si no le pasan otro.
    """
    codigos = [all_fdi_codes().index(f) + 1 for f in fdis]
    n_clases = len(DEFAULT_CODES) + 1

    def segmentar(points: np.ndarray) -> np.ndarray:
        clases = np.zeros(len(points), dtype=int)
        for codigo, zona in zip(
            codigos, np.array_split(np.argsort(points[:, 0]), len(codigos)), strict=True
        ):
            clases[zona] = codigo
        p = np.full((len(points), n_clases), (1.0 - prob) / (n_clases - 1))
        p[np.arange(len(points)), clases] = prob
        return np.log(p)

    return segmentar


@pytest.fixture
def con_segmentacion(tmp_path: Path, case_dir: Path):
    """Pipeline con modelo de segmentación, ya ingerido.

    El modelo de juguete encuentra **exactamente** los dientes que cita el informe:
    así el camino limpio se distingue del conflicto, que tiene su propio test.
    """
    base = IngestionPipeline(
        ArtifactStore(tmp_path / "artifacts"), quarantine_dir=tmp_path / "quarantine"
    )
    result = base.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    fdis = sorted({o.region_id for o in result.snapshot.regional})
    pipeline = IngestionPipeline(
        base.store, quarantine_dir=tmp_path / "quarantine", segmenter=_segmentador(fdis)
    )
    return pipeline, result, fdis


def test_la_segmentacion_produce_el_ancla_sin_pasarla_a_mano(con_segmentacion) -> None:
    """Lo que antes era un `detected` inventado desde fuera ahora lo calcula el agente."""
    pipeline, result, fdis = con_segmentacion
    salida = pipeline.fuse(result)

    assert len(salida.analysis) == 1
    assert sorted(salida.analysis[0].detected) == fdis
    assert salida.snapshot.provenance.agent.startswith("semantic-fusion-agent")
    assert not salida.hitl_required


def test_el_diente_que_el_modelo_no_encuentra_llega_al_gate(con_segmentacion) -> None:
    """El informe cita cuatro dientes y el modelo encuentra otros: nadie decide solo.

    El modelo encuentra el 48 —mandibular— y el informe es todo maxilar, así que ninguna
    observación se ancla y el gate lo dice. Los cuatro códigos siguen nombrados, que es lo
    que hace el motivo accionable.
    """
    pipeline, result, fdis = con_segmentacion
    pipeline.segmentation.segmenter = _segmentador(["48"])  # ninguno de los del informe
    salida = pipeline.fuse(result)

    assert salida.hitl_required
    assert all(
        any(f"FDI {fdi}" in m or fdi in m for m in salida.hitl_reasons) for fdi in fdis
    )


def test_una_arcada_sin_cubrir_es_un_motivo_y_no_dieciseis(con_segmentacion) -> None:
    """La limpieza del gate, medida sobre lo que la motivó.

    Un caso clínico real con solo escaneo maxilar producía **22 motivos, 16 de ellos la
    misma frase** para dientes mandibulares que nadie había mirado. Entre ellos quedaban
    enterrados los que sí importaban —el registro sin converger, la confianza bajo umbral—.
    Un aviso que salta en bloque es como se desactiva un gate.
    """
    pipeline, result, fdis = con_segmentacion
    pipeline.segmentation.segmenter = _segmentador(["48"])
    salida = pipeline.fuse(result)

    de_arcada = [m for m in salida.hitl_reasons if "no cubrió NINGÚN diente" in m]
    por_diente = [m for m in salida.hitl_reasons if "no lo encontró" in m]
    assert len(de_arcada) == 1, "una línea por arcada, no una por diente"
    assert por_diente == [], "no se miró esa arcada: no hay desacuerdo que declarar"
    # Y no se pierde nada: los cuatro códigos siguen en el motivo.
    assert all(fdi in de_arcada[0] for fdi in fdis)


def test_en_la_arcada_que_si_se_miro_el_desacuerdo_va_diente_a_diente(
    con_segmentacion,
) -> None:
    """La otra mitad, y la que hace que la agrupación no sea una excusa para callar.

    Si la segmentación **sí** cubrió la arcada y aun así falta un diente que el informe
    cita, eso es un desacuerdo real entre dos fuentes clínicas y cada uno es una decisión
    distinta. Ahí sigue habiendo una línea por diente.
    """
    pipeline, result, fdis = con_segmentacion
    pipeline.segmentation.segmenter = _segmentador(["17"])  # maxilar, pero no del informe
    salida = pipeline.fuse(result)

    por_diente = [m for m in salida.hitl_reasons if "no lo encontró" in m]
    assert len(por_diente) == len(fdis)
    assert not any("no cubrió NINGÚN diente" in m for m in salida.hitl_reasons)


def test_las_etiquetas_revisadas_mandan_sobre_el_modelo(con_segmentacion) -> None:
    """El gate humano solo sirve para algo si la corrección sustituye al modelo."""
    pipeline, result, fdis = con_segmentacion
    salida = pipeline.fuse(result, detected={fdi: 0.95 for fdi in fdis})

    assert salida.analysis == []  # la segmentación ni se ejecuta
    assert salida.snapshot.gaussian_field_ref == result.snapshot.gaussian_field_ref


def test_las_tres_etapas_corren_en_el_orden_del_pipeline(con_segmentacion) -> None:
    pipeline, result, _fdis = con_segmentacion
    nube = _nube()
    salida = pipeline.fuse(result, registration=(nube, nube))

    assert [o.agent.split("@")[0] for o in salida.fusion] == [
        "geometric-fusion-agent",
        "semantic-fusion-agent",
    ]
    assert salida.analysis[0].agent.startswith("segmentation-agent")
    # La segmentación etiquetó el campo y la etapa siguiente conserva esa referencia.
    assert salida.snapshot.gaussian_field_ref != result.snapshot.gaussian_field_ref
    assert "region_id" in pipeline.store.load(salida.snapshot.gaussian_field_ref)


def test_sin_segmenter_la_etapa_no_existe(pipeline: IngestionPipeline, case_dir: Path) -> None:
    """Retrocompatible: sin modelo, el ancla sigue entrando a mano."""
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    salida = pipeline.fuse(result)

    assert pipeline.segmentation is None
    assert salida.analysis == []
    assert salida.fusion == []


def test_si_la_segmentacion_falla_no_se_ancla_contra_un_ancla_inexistente(
    tmp_path: Path, case_dir: Path
) -> None:
    """El peor final sería anclar el informe a dientes que nadie ha confirmado."""

    def revienta(points: np.ndarray) -> np.ndarray:
        raise RuntimeError("CUDA out of memory")

    pipeline = IngestionPipeline(
        ArtifactStore(tmp_path / "artifacts"),
        quarantine_dir=tmp_path / "quarantine",
        segmenter=revienta,
    )
    result = pipeline.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    salida = pipeline.fuse(result)

    assert salida.analysis[0].status is ModalityStatus.FAILED
    assert salida.fusion == []  # la fusión semántica no llega a correr
    assert salida.snapshot is not None  # y la ingesta sobrevive
    assert any("falló" in m for m in salida.hitl_reasons)


# --- la fábrica de segmentador (la costura registro → segmentación) --------- #
def test_la_fabrica_recibe_el_snapshot_YA_REGISTRADO(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """La costura que esto existe para coser.

    Un `Segmenter` del campo necesita las coronas del escáner movidas al marco del CBCT, y
    esa pose la calcula la fusión geométrica **dentro de `fuse()`**. Con el parámetro
    `segmenter` a secas, quien lo construía tenía que registrar por su cuenta: trabajo
    duplicado y, peor, una pose distinta de la que el snapshot declara y de la que se
    exporta en el STL — dos verdades sobre el mismo paciente sin nada que las compare.
    """
    vistos = []

    def fabrica(snapshot):
        vistos.append(snapshot.provenance.transform)
        return lambda puntos: np.log(
            np.full((len(puntos), max(DEFAULT_CODES) + 1), 1.0 / (max(DEFAULT_CODES) + 1))
        )

    pipe = IngestionPipeline(
        ArtifactStore(tmp_path / "art"), segmenter_factory=fabrica
    )
    resultado = pipe.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    pipe.fuse(resultado, registration=(_nube(), _nube()))

    assert len(vistos) == 1, "la fábrica se llama una vez, tras la fusión geométrica"
    assert vistos[0] is not None, "y con la transformación ya puesta en la procedencia"


def test_una_fabrica_que_devuelve_none_no_rompe_el_recorrido(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """Sin registro no se puede saber DÓNDE está cada corona, y no segmentar es la
    respuesta correcta — no segmentar mal."""
    pipe = IngestionPipeline(
        ArtifactStore(tmp_path / "art"), segmenter_factory=lambda s: None
    )
    resultado = pipe.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    salida = pipe.fuse(resultado)

    assert salida.analysis == []
    assert salida.snapshot is not None


# --- varios informes por adquisición ---------------------------------------- #
def test_dos_informes_aportan_los_dos_sus_observaciones(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """El hueco que esto cierra, y costó dato clínico.

    Con `report` como campo único, el orquestador se quedaba con uno y el resto
    desaparecía **sin que nada lo dijera**. Medido sobre un caso real: tres PDF, y el que
    se perdía era un informe de oclusión/ATM con análisis de contacto dental — otra
    modalidad clínica, no un duplicado.
    """
    uno = tmp_path / "cbct.txt"
    uno.write_text("Diente 16: pH 5.1\n", encoding="utf-8")
    dos = tmp_path / "oclusion.txt"
    dos.write_text("Diente 47: pH 6.2\n", encoding="utf-8")

    case = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    case.reports = [uno, dos]
    result = pipeline.run(case)

    assert len(result.report_outcomes) == 2
    fdis = {o.region_id for o in result.snapshot.regional}
    assert {"16", "47"} <= fdis, "ninguno de los dos manda sobre el otro"


def test_un_informe_ilegible_se_declara_en_vez_de_desaparecer(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """Y el resto sigue entrando: fail-loud, no fail-fast.

    Un PDF escaneado sin capa de texto es normal en una carpeta de clínica. Lo que no es
    normal es que se filtre antes de que ningún agente lo vea — el `report-agent` ya sabe
    decir «no contiene texto extraíble», y que lo haya es justo lo que un clínico
    necesita saber.
    """
    bueno = tmp_path / "bueno.txt"
    bueno.write_text("Diente 16: pH 5.1\n", encoding="utf-8")
    # Un PDF de verdad pero sin capa de texto, que es lo que sale de un escáner de
    # sobremesa. Un `.txt` no sirve para este test: por esa rama el agente devuelve OK con
    # cero hallazgos, y hace bien — un informe sin pH es legítimo.
    ilegible = tmp_path / "escaneado.pdf"
    ilegible.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntrailer<<>>\n%%EOF\n")

    case = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    case.reports = [bueno, ilegible]
    result = pipeline.run(case)

    estados = {o.status for o in result.report_outcomes}
    assert ModalityStatus.FAILED in estados, "el ilegible se declara"
    assert result.snapshot is not None, "y no tumba al que sí se pudo leer"
    assert any(o.region_id == "16" for o in result.snapshot.regional)
    assert result.hitl_required


def test_sin_ningun_informe_la_modalidad_sigue_siendo_missing(
    pipeline: IngestionPipeline, case_dir: Path
) -> None:
    """`missing` = no había fichero, `failed` = lo había y no se pudo leer. La
    distinción no se pierde al pasar a lista."""
    case = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    case.reports = []
    result = pipeline.run(case)

    assert len(result.report_outcomes) == 1
    assert result.report_outcomes[0].status is ModalityStatus.MISSING


# --- el twin describe su propio campo ---------------------------------------- #
def test_el_snapshot_dice_QUE_es_cada_columna(pipeline: IngestionPipeline, case_dir: Path):
    """Lo que convierte «un fichero que sabemos leer» en un formato.

    Todo esto vivía en líneas `comment` del PLY: las personas las leen y los programas no.
    """
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    por_nombre = {c.nombre: c for c in snapshot.esquema_campo}

    assert {"x", "scale_0", "rot_0", "density"} <= set(por_nombre)
    assert por_nombre["density"].unidad == "normalised_sigma"
    assert "NO es opacidad" in por_nombre["density"].significado


def test_la_escala_declara_que_son_MILIMETROS_y_no_su_logaritmo(
    pipeline: IngestionPipeline, case_dir: Path
):
    """El detalle que evita una mala interpretación silenciosa.

    El PLY de facto de 3DGS usa `scale_0..2` con el MISMO nombre y guarda el logaritmo. Un
    visor estándar abriendo esto no fallaría: exponenciaría nuestros milímetros y
    renderizaría basura con buen aspecto. Por eso `escala` es una columna del esquema y no
    un comentario.
    """
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    escala = next(c for c in snapshot.esquema_campo if c.nombre == "scale_0")

    assert escala.unidad == "mm"
    assert escala.escala == "linear"
    assert snapshot.perfil_campo == "histora-twin/1.0", "y el perfil se declara para rechazarlo"


def test_un_campo_sin_segmentar_no_declara_region_id(
    pipeline: IngestionPipeline, case_dir: Path
):
    """Declarar de más sería tan mentira como declarar de menos."""
    snapshot = pipeline.run(CaseInput.from_case_dir(case_dir)).snapshot
    assert not any(c.nombre == "region_id" for c in snapshot.esquema_campo)


def test_el_esquema_SIGUE_al_campo_cuando_una_etapa_lo_cambia(
    case_dir: Path, tmp_path: Path
):
    """El que de verdad importa.

    Las etapas cambian las columnas: la segmentación añade `region_id`. Un esquema
    calculado solo en la ingesta describiría un artefacto que ya no es el referenciado, y
    eso es peor que no tener esquema — es un contrato que miente con precisión.
    """
    pipe = IngestionPipeline(
        ArtifactStore(tmp_path / "art"),
        segmenter=_segmentador(sorted({"11", "16"})),
    )
    resultado = pipe.run(CaseInput.from_case_dir(case_dir, patient_id="PAC-001"))
    assert not any(c.nombre == "region_id" for c in resultado.snapshot.esquema_campo)

    fusionado = pipe.fuse(resultado)
    region = next(
        (c for c in fusionado.snapshot.esquema_campo if c.nombre == "region_id"), None
    )
    assert region is not None, "la segmentación la añadió y el esquema tiene que decirlo"
    assert region.vocabulario == "ISO-3950"
    assert region.medido is False, "no es una medida: la derivó un agente"
    assert region.derivado_de == "segmentation-agent"



def test_el_orquestador_reenvia_TODO_lo_que_el_agente_de_UOS_acepta():
    """La firma de `exportar` enumera argumentos a mano, y una lista a mano se queda vieja.

    ⚠️ Costó una ejecución entera —451 s de entrenamiento incluidos— que murió en la última
    línea con `unexpected keyword argument 'sin_malla'`: el agente había ganado el
    parámetro y el orquestador no. Nada lo detectaba hasta que se ejecutaba el caso real
    con esa bandera puesta.

    Los que NO se reenvían están aquí uno a uno **y con su razón**. Añadir un parámetro al
    agente sin reenviarlo, o sin justificarlo aquí, rompe este test — que es exactamente
    cuando hay que enterarse.
    """
    import inspect

    from agent_orchestrator.pipeline import IngestionPipeline
    from uos.agente import UOSExportAgent

    # Los calcula el propio orquestador o no tienen sentido desde fuera.
    DELIBERADOS = {
        "campo",         # ruta del PLY del campo: la produce el canal de campo
        "compuesto",     # idem, el canal del compuesto
        "motivos",       # los acumula el pipeline de la fusión
        "previo",        # por defecto es el propio destino
        "pseudonimo",    # sale del snapshot
        "registrador",   # lo sabe la fusión geométrica
        # ⚠️ **Éste no es «lo calcula el orquestador»: es una decisión del FORMATO.**
        # Se reenviaba, con un `sin_originales: bool = False` propio en la firma de
        # `exportar`, y ese defecto duplicado le ganó al `= True` del agente: el
        # contenedor del caso clínico salió con el STL del escáner, nueve fotografías
        # del paciente y tres informes dentro —216 MB— mientras todos los tests del
        # agente seguían en verde. Que el `.uos` no lleve originales lo decide
        # `uos.agente` y sólo él; el orquestador no tiene voz aquí, y por eso no
        # aparece en su firma. Lo que sí hay es un test que mira los bytes del ZIP a
        # la salida del orquestador: `test_e2e.py::test_el_uos_del_recorrido_no_lleva_
        # NINGUN_original`, porque probar la pieza no prueba el montaje.
        "sin_originales",
    }
    del_agente = set(inspect.signature(UOSExportAgent._export).parameters) - {
        "self", "snapshot", "destination",
    }
    del_orquestador = set(inspect.signature(IngestionPipeline.exportar).parameters) - {
        "self", "result", "destino",
    }
    sin_reenviar = del_agente - del_orquestador - DELIBERADOS
    assert not sin_reenviar, (
        f"el agente de UOS acepta {sorted(sin_reenviar)} y `exportar` no lo reenvía: "
        "añádelo a la firma o justifícalo en DELIBERADOS"
    )


def test_lo_que_el_orquestador_reenvia_no_CONTRADICE_al_agente():
    """Un parámetro reenviado con otro valor por defecto es peor que uno olvidado.

    ⚠️ **Esto es lo que falló, y el test de reenvío no podía verlo.** `sin_originales`
    estaba en las dos firmas y se reenviaba —aquel test en verde— pero el agente lo
    declaraba `True` (el formato no lleva originales) y el orquestador `False`. Mientras
    la CLI pasaba la bandera a mano no se notaba; en cuanto se quitó, el valor cayó al
    defecto del orquestador y el contenedor del caso clínico salió con el STL del
    escáner, nueve fotografías del paciente y tres informes dentro, 216 MB.

    Un parámetro olvidado revienta con `TypeError` y se ve. Un defecto duplicado que
    discrepa NO revienta: emite el fichero equivocado en silencio, y cada test de la
    pieza sigue pasando. Por eso lo que se compara aquí son los VALORES, no los nombres.
    """
    import inspect

    from agent_orchestrator.pipeline import IngestionPipeline
    from uos.agente import UOSExportAgent

    agente = inspect.signature(UOSExportAgent._export).parameters
    orq = inspect.signature(IngestionPipeline.exportar).parameters

    discrepan = {
        n: (agente[n].default, orq[n].default)
        for n in set(agente) & set(orq)
        if agente[n].default is not inspect.Parameter.empty
        and orq[n].default is not inspect.Parameter.empty
        and agente[n].default != orq[n].default
    }
    assert not discrepan, (
        "el orquestador reenvía un valor por defecto distinto al del agente, así que "
        "el suyo gana y la decisión del agente no se aplica nunca: "
        + ", ".join(
            f"{n} (agente {a!r} / orquestador {o!r})"
            for n, (a, o) in sorted(discrepan.items())
        )
    )


def test_el_motivo_de_un_informe_ilegible_no_lleva_el_NOMBRE_del_fichero(
    pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path
) -> None:
    """⚠️ **Un nombre de fichero clínico es dato del paciente tan a menudo como no lo es.**

    Este motivo acaba en `review.reasons` y de ahí dentro del `.uos`, que se declara
    `pseudonymized`. Llevó primero la ruta entera —con el directorio del paciente— y luego
    el nombre del fichero, que parecía suficiente hasta que en un caso real uno de los
    informes se llamaba `APELLIDOS_NOMBRE_Informe_de_CBCT_...pdf`. No hay forma de saber
    cuál de los dos casos toca, así que va el hash: identifica el fichero sin poder llevar
    nada dentro. Es el mismo razonamiento por el que un asset externo se nombra
    `sha256:<hex>` en lugar de con su ruta.
    """
    bueno = tmp_path / "bueno.txt"
    bueno.write_text("Diente 16: pH 5.1\n", encoding="utf-8")
    apellidos = "PEREZ GOMEZ"
    ilegible = tmp_path / f"{apellidos}_MARIA_Informe_de_CBCT.pdf"
    # Un PDF VALIDO y sin capa de texto, que es lo que sale de un escaner de sobremesa.
    # Uno roto a mano no vale: falla antes, en el parser, y no llega a la rama que esto
    # comprueba.
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    with ilegible.open("wb") as fh:
        w.write(fh)

    case = CaseInput.from_case_dir(case_dir, patient_id="PAC-001")
    case.reports = [bueno, ilegible]
    result = pipeline.run(case)

    # El motivo viaja por dos sitios: el  del agente y las razones del gate de
    # revision humana, que es lo que acaba dentro del . Se miran los dos.
    razones = getattr(getattr(result.export, "review", None), "reasons", []) or []
    motivos = " ".join(
        [str(o.detail or "") for o in result.report_outcomes] + [str(r) for r in razones]
    )
    assert "no contiene texto extraíble" in motivos, "el fallo se sigue declarando"
    for trozo in (apellidos, "PEREZ", "GOMEZ", "MARIA"):
        assert trozo not in motivos, f"«{trozo}» sale del contenedor dentro del motivo"
    assert "sha256:" in motivos, "y se identifica por su hash, no por su nombre"
