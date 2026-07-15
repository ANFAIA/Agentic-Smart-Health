"""
Modelos base del Digital Twin dental — Agentic Smart Health.

Traducción a Pydantic v2 del diseño conceptual de la Semana 1:

  1. Cómo RGS (CBCT + 3DGS) guarda la densidad radiológica en la primitiva
     gaussiana  →  `GaussianPrimitive` (θₙ = {c, Σ, σ}).
  2. Extensión con metadatos clínicos de distinto soporte geométrico:
       · densidad  σ            → volumétrico  (por gaussiana)
       · color_superficie       → superficial  (STL, solo la cáscara)
       · pH                     → regional     (informe, capa dispersa por FDI)
  3. Soporte de series temporales para evaluar la evolución clínica:
       modelo híbrido = snapshots por adquisición (geometría/densidad,
       reversibilidad) + observaciones regionales timestamped (evolución de
       atributos como el pH a lo largo del tiempo).

Nota de arquitectura: los arrays masivos de gaussianas (potencialmente
millones) viven como tensores/nubes de puntos en `3dgs-engine`. Aquí se
define el *contrato* de datos y los metadatos clínicos; `GaussianPrimitive`
documenta la unidad canónica y sirve para (de)serialización de conjuntos
pequeños, no para almacenar el campo completo en memoria como objetos Pydantic.

Las decisiones de diseño que justifican esta estructura están registradas en
`docs/architecture/001-digital-twin-core-schemas.md` (ADR 001).

Ref.: Lin et al., "Residual Gaussian Splatting for Ultra Sparse-View CBCT
Reconstruction", arXiv:2604.27552v1 (2026).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Vocabulario controlado
# --------------------------------------------------------------------------- #
class Modality(str, Enum):
    """Fuente de la que procede un dato ingerido."""

    CBCT = "cbct"      # DICOM  → densidad σ (volumétrico)
    STL = "stl"        # malla  → color_superficie (superficial)
    REPORT = "report"  # PDF    → pH y otros atributos regionales
    IMAGE = "image"    # foto 2D


class Support(str, Enum):
    """Soporte geométrico sobre el que está definido un atributo clínico.

    Es la distinción clave del diseño: los tres atributos NO comparten soporte.
    Vocabulario controlado; el soporte se codifica *estructuralmente* según en
    qué modelo vive cada atributo (ver ADR 001, §4.1).
    """

    VOLUMETRIC = "volumetric"  # todo el volumen, por gaussiana (σ)
    SURFACE = "surface"        # solo la cáscara 2-manifold (color_superficie)
    REGIONAL = "regional"      # un valor por zona/diente (pH)


# Código ISO-FDI de dos dígitos. Permanente: [1-4][1-8]; temporal: [5-8][1-5].
FDICode = Annotated[
    str,
    Field(pattern=r"^([1-4][1-8]|[5-8][1-5])$", description="Diente en numeración ISO-FDI, p. ej. '16'."),
]


# --------------------------------------------------------------------------- #
# Trazabilidad (requisito de transparencia del proyecto: RGPD/HIPAA)
# --------------------------------------------------------------------------- #
class Provenance(BaseModel):
    """Procedencia de un valor: qué fichero, qué agente y con qué confianza.

    Se adjunta a cada observación para garantizar la explicabilidad exigida:
    "qué dato se ingirió, qué transformación se aplicó y por qué".
    """

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(description="Ruta o URI del fichero de origen.")
    modality: Modality
    agent: str = Field(description="Agente de ingesta que produjo el valor.")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# Atributos por-punto: la primitiva gaussiana extendida  (θₙ⁺)
# --------------------------------------------------------------------------- #
class Color(BaseModel):
    """Color RGB de superficie (canal de apariencia reintroducido desde el STL)."""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)


class GaussianPrimitive(BaseModel):
    """Primitiva gaussiana extendida  θₙ⁺  del Digital Twin.

    Núcleo heredado de RGS (Eq. 2/3):
        center     cₙ ∈ ℝ³
        scale/rot  → covarianza Σₙ
        density    σₙ ≥ 0   ← la "densidad" radiológica (reemplaza la opacidad α;
                              los armónicos esféricos se descartan por isotropía).

    Extensión clínica:
        color_superficie  → RGB de la malla STL (soporte SUPERFICIAL; None si la
                            gaussiana no cae en la banda ε de la superficie).
        region_id         → etiqueta FDI del diente; ancla semántica que une esta
                            primitiva con la capa regional (pH) y con las demás
                            modalidades.
    """

    model_config = ConfigDict(extra="forbid")

    # --- geometría (heredada del 3DGS estándar) ---
    center: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotation: tuple[float, float, float, float] = Field(description="Cuaternión (w, x, y, z).")

    # --- densidad radiológica: soporte VOLUMÉTRICO ---
    density: float = Field(ge=0.0, description="σₙ ≥ 0, contribución de atenuación (Beer-Lambert).")

    # --- color de superficie: soporte SUPERFICIAL ---
    color_superficie: Color | None = None

    # --- ancla semántica hacia la capa regional ---
    region_id: FDICode | None = None


# --------------------------------------------------------------------------- #
# Atributos regionales: la capa dispersa  (pH y demás)  — soporte REGIONAL
# --------------------------------------------------------------------------- #
class ClinicalAttributes(BaseModel):
    """Metadatos clínicos definidos por zona/diente (no por punto).

    Un valor por región FDI. Extensible: hoy el pH; mañana movilidad, sangrado,
    profundidad de sondaje, etc. Mantener aquí solo lo que sea genuinamente
    regional (el color y la densidad viven en `GaussianPrimitive`).
    """

    model_config = ConfigDict(extra="forbid")

    ph: float | None = Field(default=None, ge=0.0, le=14.0)


class RegionalObservation(BaseModel):
    """Una medición regional en un instante concreto (unidad de la serie temporal).

    La evolución clínica de un atributo (p. ej. el pH del diente 16) se reconstruye
    reuniendo las observaciones de esa `region_id` a través de los snapshots.
    """

    model_config = ConfigDict(extra="forbid")

    region_id: FDICode
    attributes: ClinicalAttributes
    timestamp: datetime
    provenance: Provenance


# --------------------------------------------------------------------------- #
# Series temporales: snapshot por adquisición + envoltorio del paciente
# --------------------------------------------------------------------------- #
class TwinSnapshot(BaseModel):
    """Estado completo del Digital Twin en una adquisición (visita/escaneo).

    Snapshot-céntrico por reversibilidad: cada snapshot es autocontenido y basta
    para regenerar STL/imágenes de esa fecha. El campo gaussiano masivo no se
    embebe: se referencia por hash/URI al almacén de `3dgs-engine`.
    """

    model_config = ConfigDict(extra="forbid")

    acquisition_id: str
    timestamp: datetime
    modalities: list[Modality] = Field(default_factory=list)
    gaussian_field_ref: str = Field(description="Hash/URI del campo gaussiano en 3dgs-engine.")
    n_primitives: int | None = Field(default=None, ge=0)
    regional: list[RegionalObservation] = Field(default_factory=list)
    provenance: Provenance


class PatientDigitalTwin(BaseModel):
    """Gemelo digital del paciente: secuencia temporal de snapshots.

    `patient_id` es un seudónimo (nunca un identificador directo), acorde con la
    soberanía del dato y RGPD/HIPAA.
    """

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(description="Identificador seudonimizado del paciente.")
    snapshots: list[TwinSnapshot] = Field(default_factory=list)

    def latest(self) -> TwinSnapshot | None:
        """Snapshot más reciente por timestamp."""
        return max(self.snapshots, key=lambda s: s.timestamp, default=None)

    def series(self, region_id: str, attribute: str = "ph") -> list[tuple[datetime, float]]:
        """Serie temporal ``(instante, valor)`` de un atributo regional.

        Recorre todos los snapshots y extrae, para la región FDI pedida, el valor
        del atributo indicado. Es la consulta que sostiene la Tarea 3: evaluar la
        evolución clínica del paciente a lo largo del tiempo.
        """
        out: list[tuple[datetime, float]] = []
        for snap in self.snapshots:
            for obs in snap.regional:
                if obs.region_id == region_id:
                    value = getattr(obs.attributes, attribute, None)
                    if value is not None:
                        out.append((obs.timestamp, value))
        return sorted(out, key=lambda tv: tv[0])
