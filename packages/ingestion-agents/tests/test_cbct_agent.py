"""`cbct-agent`: serie DICOM → campo gaussiano semilla, con seudonimización."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from core_schemas import Modality, ModalityStatus, Support
from ingestion_agents import ArtifactStore, CBCTAgent, synthetic
from ingestion_agents.cbct_agent import HU_SATURATION, pseudonymize


# --- seudonimización ------------------------------------------------------- #
def test_el_seudonimo_es_estable() -> None:
    """El mismo paciente debe dar el mismo seudónimo entre adquisiciones:
    es lo que permite montar su serie temporal sin conocer su identidad."""
    assert pseudonymize("PAC-001", salt="s") == pseudonymize("PAC-001", salt="s")


def test_pacientes_distintos_seudonimos_distintos() -> None:
    assert pseudonymize("PAC-001", salt="s") != pseudonymize("PAC-002", salt="s")


def test_el_seudonimo_depende_de_la_sal() -> None:
    """Sin la sal no se puede reidentificar: la sal es el dato a proteger (RGPD)."""
    assert pseudonymize("PAC-001", salt="a") != pseudonymize("PAC-001", salt="b")


def test_el_seudonimo_no_contiene_el_identificador() -> None:
    assert "PAC-001" not in pseudonymize("PAC-001", salt="s")


def test_sal_desde_el_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASH_PSEUDONYM_SALT", "sal-de-test")
    assert pseudonymize("PAC-001") == pseudonymize("PAC-001", salt="sal-de-test")


# --- ingesta --------------------------------------------------------------- #
def test_ingesta_de_la_serie_sintetica(cbct_dir: Path, store: ArtifactStore) -> None:
    outcome = CBCTAgent(store).ingest(cbct_dir)

    assert outcome.status is ModalityStatus.OK
    assert outcome.modality is Modality.CBCT
    assert outcome.support is Support.VOLUMETRIC

    arrays = store.load(outcome.artifact_ref or "")
    assert set(arrays) == {"centers", "scales", "rotations", "density"}
    n = arrays["centers"].shape[0]
    assert (arrays["scales"].shape, arrays["rotations"].shape) == ((n, 3), (n, 4))
    assert outcome.n_primitives == n


def test_la_densidad_esta_normalizada(cbct_dir: Path, store: ArtifactStore) -> None:
    """σ ≥ 0 es un invariante del contrato (`GaussianPrimitive.density`)."""
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    density = arrays["density"]
    assert density.min() >= 0.0
    assert density.max() <= 1.0


def test_el_esmalte_satura_la_densidad(cbct_dir: Path, store: ArtifactStore) -> None:
    """El esmalte sintético está a 2200 HU, por encima de la saturación: debe dar σ=1."""
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    assert synthetic._HU_ENAMEL > HU_SATURATION
    assert float(arrays["density"].max()) == pytest.approx(1.0)


def test_la_semilla_no_inventa_rotacion(cbct_dir: Path, store: ArtifactStore) -> None:
    """Cuaternión identidad (w,x,y,z): la anisotropía la aprende el optimizador
    RGS, no la ingesta."""
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    rotations = arrays["rotations"]
    np.testing.assert_allclose(rotations, np.tile([1.0, 0, 0, 0], (len(rotations), 1)))


def test_el_campo_queda_centrado(cbct_dir: Path, store: ArtifactStore) -> None:
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    np.testing.assert_allclose(arrays["centers"].mean(axis=0), 0.0, atol=1e-3)


def test_seudonimiza_el_patient_id_del_dicom(cbct_dir: Path, store: ArtifactStore) -> None:
    """El identificador directo del DICOM nunca debe salir del agente."""
    agent = CBCTAgent(store)
    agent.ingest(cbct_dir)
    assert agent.patient_pseudonym is not None
    assert "SYNTH-0001" not in agent.patient_pseudonym


def test_el_submuestreo_respeta_el_tope(cbct_dir: Path, store: ArtifactStore) -> None:
    outcome = CBCTAgent(store, max_primitives=500).ingest(cbct_dir)
    assert outcome.n_primitives is not None and outcome.n_primitives <= 500
    # Se declara que el campo es una submuestra bajando la confianza.
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.9


def test_umbral_inalcanzable_falla_en_vez_de_devolver_vacio(
    cbct_dir: Path, store: ArtifactStore
) -> None:
    """Un campo vacío exportado en silencio sería peor que un fallo declarado."""
    outcome = CBCTAgent(store, hu_threshold=1e6).ingest(cbct_dir)
    assert outcome.status is ModalityStatus.FAILED
    assert "umbral" in (outcome.detail or "")


def test_un_corte_suelto_no_es_una_serie(cbct_dir: Path, store: ArtifactStore) -> None:
    outcome = CBCTAgent(store).ingest(next(cbct_dir.glob("*.dcm")))
    assert outcome.status is ModalityStatus.FAILED
    assert "directorio" in (outcome.detail or "")


def test_directorio_sin_dicom(tmp_path: Path, store: ArtifactStore) -> None:
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    outcome = CBCTAgent(store).ingest(vacio)
    assert outcome.status is ModalityStatus.FAILED
    assert ".dcm" in (outcome.detail or "")


def test_ingesta_reproducible(cbct_dir: Path, store: ArtifactStore) -> None:
    """Submuestreo por paso uniforme, no aleatorio: la misma serie da la misma referencia."""
    a = CBCTAgent(store, max_primitives=1000).ingest(cbct_dir)
    b = CBCTAgent(store, max_primitives=1000).ingest(cbct_dir)
    assert a.artifact_ref == b.artifact_ref


def test_el_orden_de_los_cortes_no_depende_del_nombre(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """Se ordena por posición física: un nombre de fichero engañoso no debe deformar el volumen."""
    volume, spacing = synthetic.build_volume(spacing=2.5)
    serie = tmp_path / "serie"
    synthetic.write_dicom_series(serie, volume, spacing)

    referencia = CBCTAgent(store).ingest(serie).artifact_ref

    # Se renombran los cortes en orden inverso; el contenido (y su ImagePositionPatient)
    # no cambia, así que el volumen reconstruido debe ser idéntico.
    revuelta = tmp_path / "revuelta"
    revuelta.mkdir()
    slices = sorted(serie.glob("*.dcm"))
    for i, src in enumerate(reversed(slices)):
        (revuelta / f"z{i:04d}.dcm").write_bytes(src.read_bytes())

    assert CBCTAgent(store).ingest(revuelta).artifact_ref == referencia


@pytest.mark.skipif(
    os.environ.get("ASH_PSEUDONYM_SALT") is not None, reason="sal ya definida en el entorno"
)
def test_sal_por_defecto_es_de_desarrollo() -> None:
    """Debe ser evidente que no vale para datos de paciente."""
    assert pseudonymize("PAC-001") == pseudonymize(
        "PAC-001", salt="dev-salt-no-usar-en-produccion"
    )
