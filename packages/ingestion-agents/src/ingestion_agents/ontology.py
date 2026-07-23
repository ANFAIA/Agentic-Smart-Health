"""Ontología clínica mínima — vocabulario controlado de la ingesta.

Entregable de la Semana 3-4. Es deliberadamente **mínima**: solo el vocabulario
que la ingesta necesita para no inventarse nada, no una ontología dental
completa. Dos partes:

1. **Numeración ISO-FDI** (ISO 3950): dos dígitos, `cuadrante` + `posición`. Es
   el **ancla semántica** de todo el pipeline — el `region_id` que une densidad
   (CBCT), color (malla) y pH (informe) del mismo diente (ADR 001 §4.6). Aquí se
   define qué códigos existen, en qué cuadrante caen y qué tipo de diente son.

2. **Atributos regionales** y su rango plausible. Sirve para que el
   `report-agent` **rechace** una extracción absurda (un pH de 74 por un `7.4`
   mal leído) en vez de escribirla en el Digital Twin.

> **Frontera.** Esto es vocabulario, no conocimiento clínico. No modela
> patologías ni relaciones; cuando haga falta razonar sobre ellas, el sitio es un
> grafo/RAG aparte, no este módulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ToothType(StrEnum):
    """Tipo morfológico del diente (derivado de la posición FDI)."""

    INCISOR = "incisivo"
    CANINE = "canino"
    PREMOLAR = "premolar"
    MOLAR = "molar"


class Dentition(StrEnum):
    PERMANENT = "permanente"  # cuadrantes 1-4
    PRIMARY = "temporal"      # cuadrantes 5-8


@dataclass(frozen=True)
class ToothInfo:
    """Descomposición semántica de un código FDI."""

    fdi: str
    quadrant: int
    position: int
    dentition: Dentition
    tooth_type: ToothType
    upper: bool
    right: bool

    @property
    def arch(self) -> str:
        return "superior" if self.upper else "inferior"

    @property
    def side(self) -> str:
        return "derecho" if self.right else "izquierdo"


# Posición dentro del cuadrante → tipo. Permanente: 8 posiciones; temporal: 5.
_PERMANENT_TYPES = {
    1: ToothType.INCISOR, 2: ToothType.INCISOR,
    3: ToothType.CANINE,
    4: ToothType.PREMOLAR, 5: ToothType.PREMOLAR,
    6: ToothType.MOLAR, 7: ToothType.MOLAR, 8: ToothType.MOLAR,
}
_PRIMARY_TYPES = {
    1: ToothType.INCISOR, 2: ToothType.INCISOR,
    3: ToothType.CANINE,
    4: ToothType.MOLAR, 5: ToothType.MOLAR,
}
# Cuadrantes superiores (1,2 permanentes; 5,6 temporales) y derechos (1,4,5,8).
_UPPER_QUADRANTS = {1, 2, 5, 6}
_RIGHT_QUADRANTS = {1, 4, 5, 8}


def is_valid_fdi(code: str) -> bool:
    """¿Es `code` un código FDI existente? (mismo criterio que `FDICode` del contrato)."""
    if len(code) != 2 or not code.isdigit():
        return False
    quadrant, position = int(code[0]), int(code[1])
    if quadrant in (1, 2, 3, 4):
        return 1 <= position <= 8
    if quadrant in (5, 6, 7, 8):
        return 1 <= position <= 5
    return False


def describe(code: str) -> ToothInfo:
    """Descompone un código FDI. Lanza `ValueError` si no existe."""
    if not is_valid_fdi(code):
        raise ValueError(f"Código FDI inexistente: {code!r} (ISO 3950).")
    quadrant, position = int(code[0]), int(code[1])
    dentition = Dentition.PERMANENT if quadrant <= 4 else Dentition.PRIMARY
    types = _PERMANENT_TYPES if dentition is Dentition.PERMANENT else _PRIMARY_TYPES
    return ToothInfo(
        fdi=code,
        quadrant=quadrant,
        position=position,
        dentition=dentition,
        tooth_type=types[position],
        upper=quadrant in _UPPER_QUADRANTS,
        right=quadrant in _RIGHT_QUADRANTS,
    )


def all_fdi_codes(*, primary: bool = False) -> list[str]:
    """Todos los códigos FDI válidos (permanentes; con `primary=True`, también temporales)."""
    codes = [f"{q}{p}" for q in (1, 2, 3, 4) for p in range(1, 9)]
    if primary:
        codes += [f"{q}{p}" for q in (5, 6, 7, 8) for p in range(1, 6)]
    return codes


# --------------------------------------------------------------------------- #
# Atributos regionales admitidos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttributeSpec:
    """Rango plausible de un atributo regional, para validar la extracción."""

    name: str
    unit: str
    minimum: float
    maximum: float
    description: str

    def accepts(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


# Rango clínico plausible del pH en boca. Es más estrecho que el 0-14 del
# contrato **a propósito**: el contrato acota lo que es un pH; la ontología acota
# lo que es un pH *creíble en un informe dental*. Un 7.4 leído como 74 lo caza
# el contrato; un 1.2 lo caza esto.
PH = AttributeSpec(
    name="ph",
    unit="pH",
    minimum=3.0,
    maximum=9.0,
    description="pH de la superficie/placa del diente. <5.5 favorece desmineralización.",
)

ATTRIBUTES: dict[str, AttributeSpec] = {PH.name: PH}
