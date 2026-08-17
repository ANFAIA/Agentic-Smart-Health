"""`field-export-agent` — regeneración del campo gaussiano desde el twin (fase 6).

El test que manda es el **round-trip real**: serie DICOM sintética → `cbct-agent` →
artefacto → `field-export-agent` → PLY → relectura. Medir la reversibilidad contra un
almacén de mentira no mediría nada, igual que en `test_stl_export.py`.

Lo demás son los fallos que un exportador tiene que **declarar en vez de escribir**:
referencia colgante, artefacto sin las claves del campo, un artefacto viejo al que se le
pide el marco del CBCT, y un PLY truncado.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from core_schemas import (
    Modality,
    ModalityIngestion,
    ModalityStatus,
    Provenance,
    TwinSnapshot,
)
from export_agents import (
    REVERSIBILITY_BUDGET_MM,
    FieldExportAgent,
    densidad_a_hu,
    lee_ply,
)
from ingestion_agents import ArtifactStore, CBCTAgent, synthetic
from ingestion_agents.cbct_agent import HU_SATURATION


@pytest.fixture(scope="session")
def cbct_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Serie DICOM sintética propia, generada una vez por sesión.

    No se reutiliza la del conftest de `ingestion-agents`: ese paquete es una dependencia
    de test, no un proveedor de fixtures, y `test_stl_export.py` ya escribe su propio OBJ
    por lo mismo. Espaciado grueso para que la suite siga siendo rápida.
    """
    raiz = tmp_path_factory.mktemp("caso-campo")
    codigos = synthetic.upper_arch_codes()
    volumen, sp = synthetic.build_volume(codigos, spacing=1.2)
    return synthetic.write_dicom_series(raiz / "cbct", volumen, sp, patient_id="SYNTH-0001")


def _snapshot(field_ref: str, **kw) -> TwinSnapshot:
    base: dict = dict(
        acquisition_id="ACQ-001",
        timestamp=datetime.now(UTC),
        gaussian_field_ref=field_ref,
        provenance=Provenance(
            source_file="caso/", modality=Modality.CBCT, agent="agent-orchestrator@0.1.0"
        ),
    )
    base.update(kw)
    return TwinSnapshot(**base)


@pytest.fixture
def ingerido(cbct_dir: Path, tmp_path: Path) -> tuple[ArtifactStore, TwinSnapshot, Path]:
    """El camino real: un CBCT ingerido de verdad y su snapshot apuntando al blob."""
    store = ArtifactStore(tmp_path / "artifacts")
    salida = CBCTAgent(store).ingest(cbct_dir)
    assert salida.ok and salida.artifact_ref
    return store, _snapshot(salida.artifact_ref), tmp_path


# --- el round-trip ---------------------------------------------------------- #
def test_round_trip_cbct_twin_ply(ingerido) -> None:
    """CBCT → twin → PLY: las posiciones vuelven exactas y la medida lo confirma."""
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "salida" / "campo.ply"

    salida = FieldExportAgent(store).export(snapshot, destino)

    assert salida.ok, salida.detail
    assert salida.path == destino and destino.exists()
    assert salida.format == "ply" and salida.frame == "twin"
    assert salida.n_vertices == len(store.load(snapshot.gaussian_field_ref)["centers"])
    # Se escribe `double` justamente para que esto sea 0: si no lo es, hay un bug de
    # formato que buscar, no un redondeo que tolerar.
    assert salida.max_deviation_mm == 0.0
    assert salida.within_budget and salida.max_deviation_mm < REVERSIBILITY_BUDGET_MM
    assert not salida.hitl_required


def test_el_marco_cbct_devuelve_las_coordenadas_reales(ingerido) -> None:
    """`frame="cbct"` suma `origin`: es la tarea de «volver a mm reales» de la issue."""
    store, snapshot, tmp_path = ingerido
    arrays = store.load(snapshot.gaussian_field_ref)

    centrado = FieldExportAgent(store, frame="twin").export(snapshot, tmp_path / "c.ply")
    real = FieldExportAgent(store, frame="cbct").export(snapshot, tmp_path / "r.ply")
    assert centrado.ok and real.ok

    a = lee_ply(tmp_path / "c.ply")
    b = lee_ply(tmp_path / "r.ply")
    desplazamiento = np.column_stack([b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"]])
    # El desplazamiento es constante y vale exactamente `origin`.
    assert np.abs(desplazamiento - arrays["origin"]).max() < 1e-9
    # Y el campo centrado sí está centrado, mientras el real no.
    assert np.abs(np.mean([a["x"], a["y"], a["z"]], axis=1)).max() < 1e-6
    assert np.abs(arrays["origin"]).max() > 1.0


def test_pedir_el_marco_cbct_sin_origin_falla_en_alto(ingerido) -> None:
    """Un artefacto viejo no se exporta callado en el marco equivocado.

    El desplazamiento depende del dato: ninguna versión del agente lo recomputa. Entregar
    el campo centrado diciendo que va en coordenadas del CBCT desplazaría todo lo que se
    midiese encima, y con muy buen aspecto.
    """
    store, snapshot, tmp_path = ingerido
    arrays = store.load(snapshot.gaussian_field_ref)
    viejo = store.put(**{k: v for k, v in arrays.items() if k not in ("origin", "hu_range")})

    salida = FieldExportAgent(store, frame="cbct").export(_snapshot(viejo), tmp_path / "v.ply")
    assert salida.status is ModalityStatus.FAILED
    assert salida.detail is not None and "origin" in salida.detail
    assert "reingerir" in salida.detail
    assert not (tmp_path / "v.ply").exists() or salida.path is None

    # El mismo artefacto viejo SÍ se exporta en su propio marco: la limitación es del
    # cambio de coordenadas, no del fichero.
    assert FieldExportAgent(store, frame="twin").export(_snapshot(viejo), tmp_path / "w.ply").ok


# --- lo que el PLY declara, y lo que a propósito no ------------------------- #
def test_el_ply_no_se_disfraza_de_splat_de_inria(ingerido) -> None:
    """Sin `opacity` ni armónicos: un visor de splats no debe poder inventarle color."""
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "campo.ply"
    assert FieldExportAgent(store).export(snapshot, destino).ok

    cabecera = destino.read_bytes().split(b"end_header")[0].decode("ascii")
    assert "property float density" in cabecera
    for inventado in ("opacity", "f_dc_0", "f_rest_0", "red", "green", "blue"):
        assert inventado not in cabecera, f"el PLY declara `{inventado}`, que el CBCT no mide"
    # Y dice qué es `density`, para que nadie la lea como transparencia.
    assert "Beer-Lambert" in cabecera and "NO opacidad" in cabecera


def test_la_cabecera_lleva_lo_que_hace_falta_para_invertir(ingerido) -> None:
    """Un fichero que no dice sus unidades no es reversible aunque los bytes estén bien."""
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "campo.ply"
    assert FieldExportAgent(store, frame="cbct").export(snapshot, destino).ok

    cabecera = destino.read_bytes().split(b"end_header")[0].decode("ascii")
    assert "field-export-agent@" in cabecera
    assert "ACQ-001" in cabecera
    assert "frame cbct" in cabecera
    assert "origin_mm" in cabecera and "hu_range" in cabecera


def test_la_densidad_se_puede_devolver_a_hounsfield(ingerido) -> None:
    store, snapshot, _ = ingerido
    arrays = store.load(snapshot.gaussian_field_ref)
    hu = densidad_a_hu(arrays["density"], arrays["hu_range"])
    assert hu.min() >= float(arrays["hu_range"][0])
    assert float(hu.max()) == pytest.approx(HU_SATURATION)


# --- los fallos que se declaran en vez de escribirse ----------------------- #
def test_referencia_colgante_es_un_fallo(ingerido) -> None:
    store, _, tmp_path = ingerido
    salida = FieldExportAgent(store).export(
        _snapshot("sha256:" + "0" * 64), tmp_path / "no.ply"
    )
    assert salida.status is ModalityStatus.FAILED
    assert salida.detail is not None and "colgante" in salida.detail.lower()


def test_un_artefacto_que_no_es_campo_es_un_fallo(ingerido) -> None:
    """Un blob de malla tiene `positions`, no `centers`: no se exporta como campo."""
    store, _, tmp_path = ingerido
    ref = store.put(positions=np.zeros((3, 3)), faces=np.zeros((1, 3), dtype=np.int32))
    salida = FieldExportAgent(store).export(_snapshot(ref), tmp_path / "no.ply")
    assert salida.status is ModalityStatus.FAILED
    assert salida.detail is not None and "centers" in salida.detail


def test_un_ply_truncado_no_se_lee_como_si_estuviera_completo(ingerido) -> None:
    """La relectura tiene que cazar el truncamiento, no devolver menos puntos callada."""
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "campo.ply"
    assert FieldExportAgent(store).export(snapshot, destino).ok

    crudo = destino.read_bytes()
    destino.write_bytes(crudo[: len(crudo) - 40])
    with pytest.raises(ValueError, match="truncado|miente"):
        lee_ply(destino)


def test_un_marco_inventado_no_se_acepta(ingerido) -> None:
    store, snapshot, tmp_path = ingerido
    with pytest.raises(ValueError, match="frame debe ser"):
        FieldExportAgent(store, frame="mundial")  # type: ignore[arg-type]
    salida = FieldExportAgent(store).export(snapshot, tmp_path / "x.ply", frame="mundial")
    assert salida.status is ModalityStatus.FAILED


# --- el gate de revisión humana -------------------------------------------- #
def test_un_twin_parcial_llega_marcado_al_fichero(ingerido) -> None:
    """El invariante del ADR 001: un snapshot parcial no llega callado a exportación."""
    store, snapshot, tmp_path = ingerido
    roto = snapshot.model_copy(
        update={
            "ingestion": [
                ModalityIngestion(
                    modality=Modality.IMAGE,
                    status=ModalityStatus.FAILED,
                    detail="la foto no se pudo decodificar",
                )
            ]
        }
    )
    destino = tmp_path / "parcial.ply"
    salida = FieldExportAgent(store).export(roto, destino)

    assert salida.ok, "un twin parcial se exporta; lo que no se hace es callarlo"
    assert salida.hitl_required
    assert any("image" in m for m in salida.hitl_reasons)
    assert b"PARCIAL" in destino.read_bytes().split(b"end_header")[0]
