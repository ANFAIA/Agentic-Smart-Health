"""El compuesto: dientes del CBCT + encía del escáner, y que se note cuál es cuál.

Estos tests existen porque el compuesto anterior **no lo era**: el fichero llevaba el
comentario `compuesto dientes-CBCT + encia-IOS` y luego `element vertex 498407`,
exactamente el tamaño del campo del CBCT. La encía se contaba, se anunciaba y no se
escribía. Aquí se ata que eso no pueda volver a pasar sin que un test caiga.
"""

from __future__ import annotations

import numpy as np
import pytest
from core_schemas import (
    Modality,
    ModalityStatus,
    Provenance,
    RigidTransform,
    TwinSnapshot,
)
from export_agents import ORIGEN_CBCT, ORIGEN_IOS, CompositeExportAgent
from export_agents.field import lee_ply


class _Almacen:
    def __init__(self, **refs: dict) -> None:
        self._refs = refs

    def load(self, ref: str) -> dict:
        return self._refs[ref]

    def put(self, **arrays):  # pragma: no cover - no lo usa este agente
        return "sha256:" + "0" * 64


def _campo(n: int = 40, con_region: bool = True) -> dict:
    rng = np.random.default_rng(0)
    campo = {
        "centers": rng.normal(0, 5, (n, 3)),
        "scales": np.full((n, 3), 0.15),
        "rotations": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "density": rng.uniform(0, 1, n),
        "origin": np.array([10.0, 20.0, 30.0]),
    }
    if con_region:
        campo["region_id"] = np.where(np.arange(n) < 12, 36, 0).astype(np.int16)
    return campo


def _snapshot(con_superficie: bool = True, con_transform: bool = True) -> TwinSnapshot:
    prov = Provenance(
        source_file="x", modality=Modality.CBCT, agent="t", confidence=1.0,
        transform=RigidTransform(rotation=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0))
        if con_transform else None,
    )
    return TwinSnapshot(
        acquisition_id="acq-1",
        timestamp="2026-08-20T00:00:00Z",
        gaussian_field_ref="sha256:" + "a" * 64,
        surface_ref=("sha256:" + "b" * 64) if con_superficie else None,
        provenance=prov,
    )


@pytest.fixture
def almacen():
    rng = np.random.default_rng(1)
    return _Almacen(**{
        "sha256:" + "a" * 64: _campo(),
        "sha256:" + "b" * 64: {"positions": rng.normal(0, 5, (25, 3))},
    })


def test_la_encia_ESTA_en_el_fichero(almacen, tmp_path):
    """El fallo exacto que esto arregla: el compuesto anterior no llevaba encía."""
    salida = CompositeExportAgent(almacen).export(_snapshot(), tmp_path / "c.ply")
    assert salida.ok, salida.detail

    ply = lee_ply(salida.path)
    assert len(ply["x"]) == 40 + 25, "las dos mitades, no solo el campo"
    assert salida.n_vertices == 65


def test_cada_gaussiana_dice_de_donde_viene(almacen, tmp_path):
    """Un compuesto que no distingue sus dos mitades miente por omisión: una densidad
    medida en el CBCT y una forma medida por el escáner no son la misma clase de dato."""
    salida = CompositeExportAgent(almacen).export(_snapshot(), tmp_path / "c.ply")
    origen = lee_ply(salida.path)["origen"]

    assert int((origen == ORIGEN_CBCT).sum()) == 40
    assert int((origen == ORIGEN_IOS).sum()) == 25


def test_la_encia_no_trae_densidad_inventada(almacen, tmp_path):
    """El escáner intraoral NO mide atenuación. Ponerle una σ plausible la haría
    indistinguible de una gaussiana medida, y un DRR del compuesto integraría un número
    que nadie midió."""
    salida = CompositeExportAgent(almacen).export(_snapshot(), tmp_path / "c.ply")
    ply = lee_ply(salida.path)
    assert np.all(ply["density"][ply["origen"] == ORIGEN_IOS] == 0.0)
    assert np.any(ply["density"][ply["origen"] == ORIGEN_CBCT] > 0.0)


def test_el_fdi_sobrevive_al_compuesto(almacen, tmp_path):
    """`region_id` es lo único que el pipeline sabe de anatomía, y ya se perdió una vez
    entre dos agentes que pasaban sus propios tests."""
    salida = CompositeExportAgent(almacen).export(_snapshot(), tmp_path / "c.ply")
    ply = lee_ply(salida.path)
    assert int((ply["region_id"] == 36).sum()) == 12
    assert np.all(ply["region_id"][ply["origen"] == ORIGEN_IOS] == 0)


def test_mide_lo_que_escribe(almacen, tmp_path):
    """La medida es producto tanto como el fichero. Y aquí más: el compuesto mezcla dos
    fuentes con marcos distintos, y un error de encaje entre ellas es justo lo que un
    número de round-trip caza y una inspección visual no."""
    salida = CompositeExportAgent(almacen).export(_snapshot(), tmp_path / "c.ply")
    assert salida.max_deviation_mm is not None
    assert salida.within_budget


def test_sin_escaner_es_MISSING_y_no_un_fallo(almacen, tmp_path):
    """Una adquisición solo-CBCT es normal. El campo solo ya lo exporta otro canal."""
    salida = CompositeExportAgent(almacen).export(
        _snapshot(con_superficie=False), tmp_path / "c.ply"
    )
    assert salida.status is ModalityStatus.MISSING
    assert not salida.path


def test_sin_registro_tambien_es_MISSING(almacen, tmp_path):
    """Componer sin registrar dejaría la encía en el sistema del escáner y los dientes en
    el del CBCT: dos objetos sueltos en el mismo fichero. Pero es entrada que falta, no un
    fallo — la fusión geométrica simplemente no corrió."""
    salida = CompositeExportAgent(almacen).export(
        _snapshot(con_transform=False), tmp_path / "c.ply"
    )
    assert salida.status is ModalityStatus.MISSING
    assert "fusión geométrica" in (salida.detail or "")


def test_un_campo_sin_segmentar_lo_declara(tmp_path):
    """Sin `region_id` el fichero sale, pero es campo y encía SIN anatomía, y eso va al
    gate en vez de pasar por un compuesto completo."""
    rng = np.random.default_rng(2)
    almacen = _Almacen(**{
        "sha256:" + "a" * 64: _campo(con_region=False),
        "sha256:" + "b" * 64: {"positions": rng.normal(0, 5, (25, 3))},
    })
    salida = CompositeExportAgent(almacen).export(_snapshot(), tmp_path / "c.ply")

    assert salida.ok
    assert any("código FDI" in m for m in salida.hitl_reasons)
