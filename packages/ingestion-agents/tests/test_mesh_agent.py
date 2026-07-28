"""`mesh-agent`: parseo del OBJ intraoral, normales y color."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from core_schemas import Modality, ModalityStatus, Support
from ingestion_agents import ArtifactStore, MeshAgent
from ingestion_agents.mesh_agent import parse_obj, vertex_normals

# Un tetraedro con color por vértice: mínimo cerrado con normales bien definidas.
_TETRAEDRO = """\
# test
v 0 0 0 1.0 0.0 0.0
v 1 0 0 0.0 1.0 0.0
v 0 1 0 0.0 0.0 1.0
v 0 0 1 1.0 1.0 1.0
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


@pytest.fixture
def tetraedro(tmp_path: Path) -> Path:
    path = tmp_path / "tetra.obj"
    path.write_text(_TETRAEDRO)
    return path


# --- parser ---------------------------------------------------------------- #
def test_parse_obj_lee_posiciones_color_y_caras(tetraedro: Path) -> None:
    mesh = parse_obj(tetraedro)
    assert mesh["positions"].shape == (4, 3)
    assert mesh["colors"].shape == (4, 3)
    assert mesh["faces"].shape == (4, 3)


def test_las_caras_quedan_0_indexadas(tetraedro: Path) -> None:
    """El OBJ indexa desde 1; numpy desde 0. Confundirlos desplaza toda la malla."""
    faces = parse_obj(tetraedro)["faces"]
    assert faces.min() == 0
    assert faces.max() == 3


def test_obj_sin_color_es_valido(tmp_path: Path) -> None:
    path = tmp_path / "sin-color.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert "colors" not in parse_obj(path)


def test_indices_con_barras(tmp_path: Path) -> None:
    """Formato `f v/vt/vn`, habitual cuando el escáner exporta coordenadas de textura."""
    path = tmp_path / "barras.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n")
    np.testing.assert_array_equal(parse_obj(path)["faces"], [[0, 1, 2]])


def test_los_quads_se_triangulan(tmp_path: Path) -> None:
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    assert parse_obj(path)["faces"].shape == (2, 3)


def test_obj_sin_vertices_es_error(tmp_path: Path) -> None:
    path = tmp_path / "vacio.obj"
    path.write_text("# solo un comentario\n")
    with pytest.raises(ValueError, match="no contiene vértices"):
        parse_obj(path)


def test_color_parcial_es_error(tmp_path: Path) -> None:
    """Media malla con color sería un artefacto silenciosamente corrupto."""
    path = tmp_path / "parcial.obj"
    path.write_text("v 0 0 0 1 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match="Color parcial"):
        parse_obj(path)


# --- normales -------------------------------------------------------------- #
def test_normales_unitarias_y_alineadas(tetraedro: Path) -> None:
    mesh = parse_obj(tetraedro)
    normals = vertex_normals(mesh["positions"], mesh["faces"])
    assert normals.shape == mesh["positions"].shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


def test_vertice_suelto_queda_con_normal_nula(tmp_path: Path) -> None:
    """Sin caras no hay orientación: se deja en cero en vez de inventar una dirección."""
    path = tmp_path / "suelto.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nf 1 2 3\n")
    mesh = parse_obj(path)
    normals = vertex_normals(mesh["positions"], mesh["faces"])
    np.testing.assert_array_equal(normals[3], [0.0, 0.0, 0.0])


# --- agente ---------------------------------------------------------------- #
def test_ingesta_del_caso_sintetico(mesh_path: Path, store: ArtifactStore) -> None:
    outcome = MeshAgent(store).ingest(mesh_path)

    assert outcome.status is ModalityStatus.OK
    assert outcome.modality is Modality.MESH
    assert outcome.support is Support.SURFACE
    assert outcome.n_primitives is not None and outcome.n_primitives > 0

    arrays = store.load(outcome.artifact_ref or "")
    assert set(arrays) == {"positions", "faces", "normals", "colors_rgb8"}
    assert arrays["colors_rgb8"].dtype == np.uint8


def test_el_conteo_de_vertices_se_preserva(mesh_path: Path, store: ArtifactStore) -> None:
    """Color, normales y futuras etiquetas FDI deben seguir alineados índice a índice."""
    outcome = MeshAgent(store).ingest(mesh_path)
    arrays = store.load(outcome.artifact_ref or "")
    n = arrays["positions"].shape[0]
    assert arrays["normals"].shape[0] == n
    assert arrays["colors_rgb8"].shape[0] == n
    assert outcome.n_primitives == n


def test_round_trip_de_superficie_sin_perdida(mesh_path: Path, store: ArtifactStore) -> None:
    """🔒 Guardarraíl de reversibilidad: la superficie se conserva con error CERO.

    Es el requisito que una nube splatteada no cumple (~1 mm) y que la métrica del
    brief exige (<0,1 mm). Se compara el artefacto contra el fichero de origen
    reparseado: si el agente remuestreara o redondeara, este test lo caza.
    """
    outcome = MeshAgent(store).ingest(mesh_path)
    arrays = store.load(outcome.artifact_ref or "")
    origen = parse_obj(mesh_path)

    assert arrays["positions"].dtype == np.float64
    error_max = float(np.abs(arrays["positions"] - origen["positions"]).max())
    assert error_max == 0.0
    np.testing.assert_array_equal(arrays["faces"], origen["faces"])


def test_la_topologia_completa_se_persiste(mesh_path: Path, store: ArtifactStore) -> None:
    """Sin las caras no hay STL que regenerar: la superficie no es solo una nube."""
    arrays = store.load(MeshAgent(store).ingest(mesh_path).artifact_ref or "")
    assert arrays["faces"].size > 0
    assert arrays["faces"].max() < arrays["positions"].shape[0]


def test_sin_color_baja_la_confianza(tmp_path: Path, store: ArtifactStore) -> None:
    """Falta el aporte propio de la modalidad: se declara, no se inventa un color."""
    path = tmp_path / "sin-color.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    outcome = MeshAgent(store).ingest(path)
    assert outcome.ok
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.5


def test_el_gris_placeholder_no_se_persiste_como_color(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """Teeth3DS+ escribe un gris constante: es placeholder del exportador, no apariencia
    clínica. Persistirlo haría que la fusión pintara las gaussianas con un color
    que nadie midió, así que se trata como ausencia (`color_superficie = None`)."""
    path = tmp_path / "gris.obj"
    gris = " 0.501961 0.501961 0.501961"
    path.write_text(f"v 0 0 0{gris}\nv 1 0 0{gris}\nv 0 1 0{gris}\nf 1 2 3\n")

    outcome = MeshAgent(store).ingest(path)

    assert "colors_rgb8" not in store.load(outcome.artifact_ref or "")
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.6


def test_teeth3ds_real_no_aporta_color(store: ArtifactStore) -> None:
    """Comprobación sobre el dataset real, no sobre un fixture fabricado."""
    real = Path("data/raw/teeth3ds/3D_scans_per_patient_obj_files")
    scans = sorted(real.glob("*/*.obj")) if real.is_dir() else []
    if not scans:
        pytest.skip("Teeth3DS+ no está en data/raw (gitignored)")

    outcome = MeshAgent(store).ingest(scans[0])

    assert outcome.ok
    assert "colors_rgb8" not in store.load(outcome.artifact_ref or "")
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.6


def test_color_real_mantiene_confianza_plena(tetraedro: Path, store: ArtifactStore) -> None:
    outcome = MeshAgent(store).ingest(tetraedro)
    assert outcome.provenance is not None and outcome.provenance.confidence == 1.0


def test_rechaza_otras_extensiones(tmp_path: Path, store: ArtifactStore) -> None:
    path = tmp_path / "escaneo.stl"
    path.write_bytes(b"solid\n")
    outcome = MeshAgent(store).ingest(path)
    assert outcome.status is ModalityStatus.FAILED
    assert "solo ingiere OBJ" in (outcome.detail or "")


def test_ingesta_reproducible(mesh_path: Path, store: ArtifactStore) -> None:
    """Determinismo = el mismo escaneo produce la misma referencia. Requisito de trazabilidad."""
    a = MeshAgent(store).ingest(mesh_path)
    b = MeshAgent(store).ingest(mesh_path)
    assert a.artifact_ref == b.artifact_ref
