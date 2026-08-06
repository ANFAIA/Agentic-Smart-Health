"""`segmentation-agent`: etiquetas FDI por gaussiana, confianza y gate HITL.

El almacén de estos tests es el `ArtifactStore` **de verdad**, no un doble: la
propiedad que más importa aquí —volver a segmentar el mismo campo devuelve la misma
referencia— es del direccionamiento por contenido, y un doble la daría por buena
sin demostrarla.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from analysis_agents import DEFAULT_CODES, SegmentationAgent
from core_schemas import (
    ClinicalAttributes,
    Modality,
    ModalityStatus,
    Provenance,
    RegionalObservation,
    TwinSnapshot,
)
from fusion_agents import SemanticFusionAgent
from ingestion_agents import ArtifactStore

# Clases del modelo de juguete: 0 encía, 1 → FDI 11, 2 → FDI 12 (DEFAULT_CODES).
GUM, A, B = 0, 1, 2
N_CLASES = 3


# --------------------------------------------------------------------------- #
# Material sintético
# --------------------------------------------------------------------------- #
def _nube(centro: tuple[float, float, float], n: int, *, semilla: int, radio: float = 1.0):
    rng = np.random.default_rng(semilla)
    return np.asarray(centro, dtype=np.float64) + rng.normal(scale=radio, size=(n, 3))


def _arcada() -> tuple[np.ndarray, np.ndarray]:
    """Dos dientes bien separados y un trozo de encía. Devuelve `(puntos, clases)`."""
    encia = _nube((0.0, -40.0, 0.0), 60, semilla=1, radio=3.0)
    diente_a = _nube((0.0, 0.0, 0.0), 50, semilla=2)
    diente_b = _nube((40.0, 0.0, 0.0), 50, semilla=3)
    puntos = np.vstack([encia, diente_a, diente_b])
    clases = np.array([GUM] * 60 + [A] * 50 + [B] * 50)
    return puntos, clases


def _segmentador(clases: np.ndarray, *, prob: float = 0.99, n_clases: int = N_CLASES):
    """Modelo de juguete: pone `prob` en la clase indicada y reparte el resto.

    Devuelve log-probabilidades normalizadas de verdad, así que la confianza de cada
    diente sale exactamente `prob` (media geométrica de una constante).
    """

    def segmentar(points: np.ndarray) -> np.ndarray:
        p = np.full((len(points), n_clases), (1.0 - prob) / (n_clases - 1))
        p[np.arange(len(points)), clases] = prob
        return np.log(p)

    return segmentar


def _campo(store: ArtifactStore, puntos: np.ndarray) -> str:
    n = len(puntos)
    return store.put(
        centers=puntos.astype(np.float32),
        scales=np.full((n, 3), 0.15, dtype=np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        density=np.full(n, 0.5, dtype=np.float32),
    )


def _snap(ref: str, *, aid: str = "A1", regional=()) -> TwinSnapshot:
    return TwinSnapshot(
        acquisition_id=aid,
        timestamp=datetime.now(UTC),
        gaussian_field_ref=ref,
        regional=list(regional),
        provenance=Provenance(
            source_file="caso/cbct", modality=Modality.CBCT, agent="cbct-agent@0.1.0"
        ),
    )


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "store")


@pytest.fixture
def caso(store):
    """El caso normal: dos dientes limpios. Devuelve `(agente, snapshot, puntos)`."""
    puntos, clases = _arcada()
    snapshot = _snap(_campo(store, puntos))
    agente = SegmentationAgent(store, segmenter=_segmentador(clases))
    return agente, snapshot, puntos


# --------------------------------------------------------------------------- #
# Camino feliz
# --------------------------------------------------------------------------- #
def test_encuentra_los_dientes_y_los_nombra(caso):
    agente, snapshot, _ = caso
    out = agente.analyze(snapshot)
    assert out.ok
    assert out.n_teeth == 2
    assert set(out.detected) == {"11", "12"}


def test_el_mapa_detected_es_lo_que_pide_la_fusion_semantica(caso, store):
    """La costura real: la salida de este agente entra tal cual en el siguiente."""
    agente, _, puntos = caso
    observacion = RegionalObservation(
        region_id="11",
        attributes=ClinicalAttributes(ph=6.2),
        timestamp=datetime.now(UTC),
        provenance=Provenance(
            source_file="informe.pdf", modality=Modality.REPORT, agent="report-agent@1.0.0"
        ),
    )
    segmentado = agente.analyze(_snap(_campo(store, puntos), regional=[observacion]))
    anclado = SemanticFusionAgent().fuse(segmentado.snapshot, detected=segmentado.detected)

    assert anclado.ok
    assert not anclado.hitl_required
    assert anclado.snapshot.regional[0].provenance.confidence == pytest.approx(0.99)


def test_la_confianza_es_la_media_geometrica_de_la_probabilidad(store):
    puntos, clases = _arcada()
    agente = SegmentationAgent(store, segmenter=_segmentador(clases, prob=0.80))
    out = agente.analyze(_snap(_campo(store, puntos)))
    assert out.detected["11"] == pytest.approx(0.80, abs=1e-6)


def test_registra_quien_segmento(caso):
    agente, snapshot, _ = caso
    out = agente.analyze(snapshot)
    assert out.snapshot.provenance.agent == "segmentation-agent@0.1.0"
    assert out.snapshot.acquisition_id == "A1"


def test_mide_latencia(caso):
    agente, snapshot, _ = caso
    assert agente.analyze(snapshot).latency_s > 0


# --------------------------------------------------------------------------- #
# El artefacto etiquetado
# --------------------------------------------------------------------------- #
def test_persiste_region_id_en_un_campo_nuevo(caso, store):
    agente, snapshot, _ = caso
    out = agente.analyze(snapshot)

    assert out.snapshot.gaussian_field_ref != snapshot.gaussian_field_ref
    region = store.load(out.snapshot.gaussian_field_ref)["region_id"]
    assert set(np.unique(region).tolist()) == {0, 11, 12}
    assert int(np.count_nonzero(region == 11)) == 50


def test_la_etiqueta_es_aditiva_y_no_toca_la_geometria(caso, store):
    """Reversibilidad: segmentar no puede alterar ni un byte del campo original."""
    agente, snapshot, _ = caso
    out = agente.analyze(snapshot)

    antes = store.load(snapshot.gaussian_field_ref)
    despues = store.load(out.snapshot.gaussian_field_ref)
    for clave in ("centers", "scales", "rotations", "density"):
        assert np.array_equal(antes[clave], despues[clave])
    assert store.exists(snapshot.gaussian_field_ref)  # el blob previo sigue ahí


def test_segmentar_dos_veces_da_la_misma_referencia(caso):
    agente, snapshot, _ = caso
    primera = agente.analyze(snapshot)
    segunda = agente.analyze(snapshot)
    assert primera.snapshot.gaussian_field_ref == segunda.snapshot.gaussian_field_ref


def test_no_muta_el_snapshot_de_entrada(caso):
    agente, snapshot, _ = caso
    antes = snapshot.model_dump_json()
    agente.analyze(snapshot)
    assert snapshot.model_dump_json() == antes


# --------------------------------------------------------------------------- #
# Gate de human-in-the-loop
# --------------------------------------------------------------------------- #
def test_confianza_baja_pide_revision(store):
    puntos, clases = _arcada()
    agente = SegmentationAgent(store, segmenter=_segmentador(clases, prob=0.5))
    out = agente.analyze(_snap(_campo(store, puntos)))
    assert out.ok  # baja confianza no es un fallo: es una entrega que hay que mirar
    assert out.hitl_required
    assert all("bajo el umbral" in m for m in out.hitl_reasons)


def test_el_mismo_fdi_en_dos_instancias_pide_revision(store):
    """Dos trozos separados con el mismo código: uno de los dos no es ese diente."""
    puntos, _ = _arcada()
    clases = np.array([GUM] * 60 + [A] * 100)  # los dos blobs se llaman igual
    agente = SegmentationAgent(store, segmenter=_segmentador(clases))
    out = agente.analyze(_snap(_campo(store, puntos)))

    assert out.n_teeth == 2
    assert set(out.detected) == {"11"}
    assert any("más de una instancia" in m for m in out.hitl_reasons)


def test_la_fragmentacion_descartada_pide_revision(store):
    """La agregación tira los trozos pequeños; que lo haga en silencio es el riesgo."""
    encia = _nube((0.0, -40.0, 0.0), 60, semilla=1, radio=3.0)
    diente = _nube((0.0, 0.0, 0.0), 50, semilla=2)
    astillas = _nube((40.0, 0.0, 0.0), 20, semilla=3, radio=25.0)  # dispersas: nada conecta
    puntos = np.vstack([encia, diente, astillas])
    clases = np.array([GUM] * 60 + [A] * 50 + [B] * 20)

    agente = SegmentationAgent(store, segmenter=_segmentador(clases))
    out = agente.analyze(_snap(_campo(store, puntos)))

    assert out.unassigned_fraction == pytest.approx(20 / 70)
    assert any("fragmentada" in m for m in out.hitl_reasons)


def test_una_clase_sin_codigo_fdi_se_declara(store):
    """Un modelo con otro convenio de clases no puede etiquetar en silencio."""
    puntos, clases = _arcada()
    agente = SegmentationAgent(store, segmenter=_segmentador(clases), codes={A: 11})
    out = agente.analyze(_snap(_campo(store, puntos)))

    assert set(out.detected) == {"11"}
    assert any("no tiene código FDI" in m for m in out.hitl_reasons)
    region = store.load(out.snapshot.gaussian_field_ref)["region_id"]
    assert set(np.unique(region).tolist()) == {0, 11}


def test_umbral_configurable(store):
    puntos, clases = _arcada()
    permisivo = SegmentationAgent(
        store, segmenter=_segmentador(clases, prob=0.5), hitl_threshold=0.4
    )
    assert not permisivo.analyze(_snap(_campo(store, puntos))).hitl_required


# --------------------------------------------------------------------------- #
# Fail-loud
# --------------------------------------------------------------------------- #
def test_sin_dientes_se_declara_missing(store):
    puntos, _ = _arcada()
    todo_encia = np.zeros(len(puntos), dtype=int)
    agente = SegmentationAgent(store, segmenter=_segmentador(todo_encia))
    out = agente.analyze(_snap(_campo(store, puntos)))

    assert out.status is ModalityStatus.MISSING
    assert out.snapshot is None
    assert out.detected == {}
    assert out.hitl_required


def test_los_logits_sin_normalizar_se_rechazan(store):
    """El fallo caro: las etiquetas saldrían bien y las confianzas serían falsas."""
    puntos, clases = _arcada()

    def logits(points: np.ndarray) -> np.ndarray:
        p = np.full((len(points), N_CLASES), 0.005)
        p[np.arange(len(points)), clases] = 0.99
        return np.log(p) * 4.0  # escalados: mismo argmax, otra escala

    out = SegmentationAgent(store, segmenter=logits).analyze(_snap(_campo(store, puntos)))
    assert out.status is ModalityStatus.FAILED
    assert "log_softmax" in out.detail


def test_una_referencia_colgante_es_un_fallo_declarado(store, tmp_path):
    agente = SegmentationAgent(
        store, segmenter=_segmentador(np.zeros(1, dtype=int)), quarantine_dir=tmp_path / "q"
    )
    out = agente.analyze(_snap("sha256:" + "de" * 32))

    assert out.status is ModalityStatus.FAILED
    assert out.snapshot is None
    assert out.quarantine_ref is not None


def test_el_fallo_del_modelo_no_se_propaga_como_excepcion(store, tmp_path):
    puntos, _ = _arcada()

    def revienta(points: np.ndarray) -> np.ndarray:
        raise RuntimeError("CUDA out of memory")

    agente = SegmentationAgent(store, segmenter=revienta, quarantine_dir=tmp_path / "q")
    out = agente.analyze(_snap(_campo(store, puntos)))

    assert out.status is ModalityStatus.FAILED
    assert "CUDA out of memory" in out.detail
    assert out.detected == {}  # el fallo no cambia la FORMA de la respuesta


def test_la_cuarentena_no_guarda_las_coordenadas(store, tmp_path):
    """Soberanía del dato: el registro de fallo es diagnóstico, no una copia del twin."""
    puntos, _ = _arcada()

    def revienta(points: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")

    agente = SegmentationAgent(store, segmenter=revienta, quarantine_dir=tmp_path / "q")
    out = agente.analyze(_snap(_campo(store, puntos), aid="VISITA-7"))

    registro = json.loads(Path(out.quarantine_ref).read_text(encoding="utf-8"))
    assert registro["acquisition_id"] == "VISITA-7"
    assert f"{puntos[0][0]:.4f}" not in json.dumps(registro)


def test_sin_cuarentena_configurada_sigue_devolviendo(store):
    def revienta(points: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")

    puntos, _ = _arcada()
    out = SegmentationAgent(store, segmenter=revienta).analyze(_snap(_campo(store, puntos)))
    assert out.status is ModalityStatus.FAILED
    assert out.quarantine_ref is None


@pytest.mark.parametrize(
    ("salida", "esperado"),
    [
        (lambda n: np.zeros((n - 1, N_CLASES)), "debe devolver (N, C)"),
        (lambda n: np.zeros((n, 1)), "hacen falta al"),
        (lambda n: np.full((n, N_CLASES), np.nan), "no finitos"),
    ],
    ids=["forma-incorrecta", "una-sola-clase", "no-finito"],
)
def test_una_salida_de_modelo_invalida_se_declara(store, salida, esperado):
    puntos, _ = _arcada()
    agente = SegmentationAgent(store, segmenter=lambda p: salida(len(p)))
    out = agente.analyze(_snap(_campo(store, puntos)))
    assert out.status is ModalityStatus.FAILED
    assert esperado in out.detail


def test_un_artefacto_sin_centers_no_es_un_campo_gaussiano(store):
    ref = store.put(densidad=np.ones(3, dtype=np.float32))
    agente = SegmentationAgent(store, segmenter=_segmentador(np.zeros(1, dtype=int)))
    out = agente.analyze(_snap(ref))
    assert out.status is ModalityStatus.FAILED
    assert "no es un campo gaussiano" in out.detail


# --------------------------------------------------------------------------- #
# Convenio de clases
# --------------------------------------------------------------------------- #
def test_el_mapeo_por_defecto_es_la_denticion_permanente():
    assert DEFAULT_CODES[1] == 11
    assert DEFAULT_CODES[32] == 48
    assert len(DEFAULT_CODES) == 32
    assert 0 not in DEFAULT_CODES  # el 0 es la encía, no un diente
