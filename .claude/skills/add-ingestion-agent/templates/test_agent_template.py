"""Plantilla de tests de un agente de ingesta — cópiala junto al agente.

Cubre el mínimo del Definition of Done: caso OK + caso corrupto (fail-loud) +
Provenance. Añade los asserts específicos de tu modalidad. Objetivo: cobertura >80%.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core_schemas import ModalityStatus

# from <modalidad>_agent import AGENT_NAME, MODALITY, ingest  # TODO: import real


@pytest.fixture
def caso_ok(tmp_path: Path) -> Path:
    """Un fichero válido de la modalidad. TODO: genera/copia una muestra real."""
    f = tmp_path / "caso.<ext>"
    # TODO: escribe un fichero mínimo válido (o copia una muestra de data/).
    raise NotImplementedError


def test_ingesta_ok_produce_contrato(caso_ok: Path) -> None:
    out = ingest(caso_ok)
    assert out.ingestion.status is ModalityStatus.OK
    assert out.provenance is not None
    assert out.provenance.agent == AGENT_NAME
    assert out.provenance.modality is MODALITY
    # TODO: asserts específicos (gaussian_field_ref resoluble / regional no vacío…)


def test_fichero_ausente_es_missing(tmp_path: Path) -> None:
    out = ingest(tmp_path / "no_existe.<ext>")
    assert out.ingestion.status is ModalityStatus.MISSING
    assert out.provenance is None


def test_fichero_corrupto_es_failed_sin_excepcion(tmp_path: Path) -> None:
    """Fail-loud: un fichero ilegible NO lanza; devuelve FAILED con detalle."""
    malo = tmp_path / "corrupto.<ext>"
    malo.write_bytes(b"\x00\x01 no es un fichero valido de la modalidad \xff")
    out = ingest(malo)  # no debe lanzar
    assert out.ingestion.status is ModalityStatus.FAILED
    assert out.ingestion.detail  # explica por qué falló
