"""Cada agente contra el catálogo de casos límite de `edge_cases.py`.

La suite de conformidad comprueba que el contrato se cumple con entradas buenas y
con un par de entradas rotas improvisadas. Esto lo sistematiza: **cada agente se
enfrenta a todos los casos de su modalidad y a todos los genéricos**, y el catálogo
declara qué debe pasar con cada uno y por qué.

Lo que se vigila no es solo que no reviente. Es que **no devuelva `OK` con un dato
inventado**, que es el fallo caro: un `FAILED` se ve en el `TwinSnapshot` y detiene
la revisión; una resonancia ingerida como si fuera un CBCT produce densidades
plausibles y falsas que nadie mira dos veces.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from core_schemas import ModalityStatus
from ingestion_agents import ArtifactStore, edge_cases
from ingestion_agents.base import BaseIngestionAgent
from ingestion_agents.cbct_agent import CBCTAgent
from ingestion_agents.edge_cases import EdgeCase
from ingestion_agents.image_agent import ImageAgent
from ingestion_agents.mesh_agent import MeshAgent
from ingestion_agents.report_agent import ReportAgent

CONSTRUCTORES: dict[str, Callable[[ArtifactStore, Path], BaseIngestionAgent]] = {
    "mesh": lambda store, q: MeshAgent(store, quarantine_dir=q),
    "cbct": lambda store, q: CBCTAgent(store, quarantine_dir=q),
    "image": lambda store, q: ImageAgent(store, quarantine_dir=q),
    "report": lambda store, q: ReportAgent(quarantine_dir=q),
}

# (modalidad, caso) para cada agente y cada caso que le corresponde.
PAREJAS = [
    pytest.param(modalidad, caso, id=f"{modalidad}-{caso.name}")
    for modalidad in CONSTRUCTORES
    for caso in edge_cases.by_modality(modalidad)
]


@pytest.mark.parametrize(("modalidad", "caso"), PAREJAS)
def test_caso_limite(modalidad: str, caso: EdgeCase, tmp_path: Path) -> None:
    agente = CONSTRUCTORES[modalidad](
        ArtifactStore(tmp_path / "artifacts"), tmp_path / "cuarentena"
    )
    trabajo = tmp_path / "entrada"
    trabajo.mkdir()

    fuente = caso.build(trabajo)
    salida = agente.ingest(fuente)  # si esto lanza, el test falla solo

    assert salida.status in tuple(ModalityStatus)
    if caso.expected is not None and caso.modality != "any":
        assert salida.status is caso.expected, f"{caso.name}: {caso.why} (detalle: {salida.detail})"
    if salida.status is not ModalityStatus.OK:
        assert salida.detail, f"{caso.name}: fallo mudo, no se puede diagnosticar"
    if salida.status is ModalityStatus.FAILED:
        assert salida.quarantine_ref, f"{caso.name}: fallo sin registro de cuarentena"
    if salida.status is ModalityStatus.OK:
        assert salida.provenance is not None


@pytest.mark.parametrize("caso", edge_cases.CASES, ids=[c.name for c in edge_cases.CASES])
def test_el_catalogo_se_materializa(caso: EdgeCase, tmp_path: Path) -> None:
    """Un caso que no se puede escribir no prueba nada: se vigila el propio catálogo."""
    ruta = caso.build(tmp_path)
    assert ruta.exists() or ruta.is_symlink(), f"{caso.name} no produjo nada en disco"
    assert caso.why.strip(), "un caso límite sin justificación acaba borrado por ruido"
