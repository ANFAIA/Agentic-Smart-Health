"""El campo ajustado tiene que entrar en el twin sin mentir sobre lo que es."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from core_schemas import Modality, Provenance, TwinSnapshot
from gaussian_engine import PERFIL, ajusta_campo, esquema
from ingestion_agents.store import ArtifactStore

pytest.importorskip("torch", reason="el ajuste necesita torch (extra `gpu`)")


@pytest.fixture
def almacen(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "store")


def _campo(store: ArtifactStore, n: int = 800) -> str:
    rng = np.random.default_rng(0)
    pos = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
    return store.put(
        centers=pos,
        scales=np.full((n, 3), 0.15, dtype=np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        density=np.exp(-0.5 * (np.linalg.norm(pos, axis=1) / 2.0) ** 2).astype(np.float32),
        origin=np.array([10.0, 20.0, 30.0]),
        hu_range=np.array([300.0, 2000.0]),
    )


def _snap(ref: str) -> TwinSnapshot:
    return TwinSnapshot(
        acquisition_id="A1",
        timestamp=datetime.now(UTC),
        gaussian_field_ref=ref,
        provenance=Provenance(
            source_file="caso/cbct", modality=Modality.CBCT, agent="cbct-agent@0.1.0"
        ),
    )


def test_el_campo_ajustado_se_declara_derivado_y_no_pisa_al_semilla(almacen):
    """Los dos ficheros usan los mismos nombres de columna con distinto significado. Si el
    ajustado no llevara perfil propio, un lector no tendría forma de saber si `scale_0` es
    el vóxel que produjo la gaussiana o una forma que alguien optimizó."""
    ref = _campo(almacen)
    original = _snap(ref)

    nuevo, r = ajusta_campo(original, almacen, n_objetivo=40, iteraciones=50,
                            dispositivo="cpu")

    assert nuevo.perfil_campo == PERFIL != original.perfil_campo
    assert nuevo.gaussian_field_ref != original.gaussian_field_ref
    # El semilla sigue donde estaba y con lo que tenía.
    assert original.gaussian_field_ref == ref
    assert len(almacen.load(ref)["centers"]) == 800
    assert 0.0 <= r.rmse < 1.0


def test_el_esquema_avisa_de_que_la_escala_ya_no_es_una_medida(almacen):
    """Es la trampa del campo derivado: quien mida sobre estas escalas mide un ajuste."""
    columnas = {c.nombre: c.significado for c in esquema(47.0)}

    assert "AJUSTADO" in columnas["scale_0"]
    assert "47" in columnas["scale_0"]
    assert "NO medido" in columnas["scale_0"]


def test_lo_que_hace_reversible_el_campo_viaja_al_derivado(almacen):
    """`origin` deshace el centrado y `hu_range` la normalización. Perderlos al derivar
    dejaría el fichero sin forma de volver a las coordenadas y las unidades del CBCT."""
    nuevo, _ = ajusta_campo(_snap(_campo(almacen)), almacen, n_objetivo=40,
                            iteraciones=50, dispositivo="cpu")

    derivado = almacen.load(nuevo.gaussian_field_ref)
    assert np.array_equal(derivado["origin"], [10.0, 20.0, 30.0])
    assert np.array_equal(derivado["hu_range"], [300.0, 2000.0])


def test_el_error_declarado_en_el_esquema_es_el_que_se_midio(almacen):
    """El número del aviso no puede ser decorativo: sale del `Ajuste` que se devuelve."""
    nuevo, r = ajusta_campo(_snap(_campo(almacen)), almacen, n_objetivo=40,
                            iteraciones=50, dispositivo="cpu")

    aviso = next(c.significado for c in nuevo.esquema_campo if c.nombre == "scale_0")
    assert f"{r.rmse_hu:.0f}" in aviso


def test_un_snapshot_sin_campo_se_declara_en_vez_de_inventarlo(almacen):
    """El ajuste refina un campo existente; no lo crea desde el volumen."""
    snap = _snap(_campo(almacen)).model_copy(update={"gaussian_field_ref": None})

    with pytest.raises(ValueError, match="no tiene campo"):
        ajusta_campo(snap, almacen, n_objetivo=10, dispositivo="cpu")
