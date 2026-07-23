"""Ontología clínica mínima: el ancla FDI y los rangos plausibles."""

from __future__ import annotations

import pytest
from ingestion_agents import ontology


@pytest.mark.parametrize("code", ["11", "18", "48", "31", "55", "85"])
def test_codigos_validos(code: str) -> None:
    assert ontology.is_valid_fdi(code)


@pytest.mark.parametrize(
    "code",
    [
        "19",  # posición 9 no existe
        "10",  # posición 0 no existe
        "56",  # los cuadrantes temporales solo llegan a la posición 5
        "90",  # cuadrante 9 no existe
        "1",   # un solo dígito
        "116",  # tres dígitos
        "1a",
        "",
    ],
)
def test_codigos_invalidos(code: str) -> None:
    assert not ontology.is_valid_fdi(code)


def test_descomposicion_de_un_molar_superior_derecho() -> None:
    info = ontology.describe("16")
    assert info.quadrant == 1
    assert info.tooth_type is ontology.ToothType.MOLAR
    assert info.dentition is ontology.Dentition.PERMANENT
    assert (info.arch, info.side) == ("superior", "derecho")


def test_descomposicion_de_un_incisivo_inferior_izquierdo() -> None:
    info = ontology.describe("31")
    assert info.tooth_type is ontology.ToothType.INCISOR
    assert (info.arch, info.side) == ("inferior", "izquierdo")


def test_los_temporales_no_tienen_premolares() -> None:
    """En dentición temporal la posición 4 es molar, no premolar (ISO 3950)."""
    assert ontology.describe("54").tooth_type is ontology.ToothType.MOLAR
    assert ontology.describe("14").tooth_type is ontology.ToothType.PREMOLAR


def test_describe_rechaza_lo_inexistente() -> None:
    with pytest.raises(ValueError, match="inexistente"):
        ontology.describe("19")


def test_conteo_de_codigos() -> None:
    assert len(ontology.all_fdi_codes()) == 32
    assert len(ontology.all_fdi_codes(primary=True)) == 52


def test_rango_plausible_de_ph() -> None:
    """El contrato acota lo que es un pH; la ontología, lo que es un pH creíble."""
    assert ontology.PH.accepts(5.5)
    assert not ontology.PH.accepts(1.2)
    assert not ontology.PH.accepts(74.0)
