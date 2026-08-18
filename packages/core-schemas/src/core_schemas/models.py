"""
Modelos base del Digital Twin dental — Agentic Smart Health.

Traducción a Pydantic v2 del diseño conceptual de la Semana 1:

  1. Cómo RGS (CBCT + 3DGS) guarda la densidad radiológica en la primitiva
     gaussiana  →  `GaussianPrimitive` (θₙ = {c, Σ, σ}).
  2. Extensión con metadatos clínicos de distinto soporte geométrico:
       · densidad  σ            → volumétrico  (por gaussiana)
       · color_superficie       → superficial  (malla intraoral, solo la cáscara)
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

import math
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Versión del contrato de datos (SemVer). Se serializa en cada `TwinSnapshot`
# para que un JSON persistido declare bajo qué esquema se escribió y no quede
# "huérfano" si el contrato (o el formato del campo gaussiano) evoluciona.
#
# 1.3.0 — `Provenance.transform` (ADR 004): los agentes de fusión registran la
#         transformación rígida que aplicaron, para que sea invertible. Aditivo y
#         opcional: un JSON 1.2.0 sigue validando.
# 1.4.0 — `ClinicalAttributes` gana `n_raices`, `n_conductos` y `hallazgos`. Lo pidió
#         un informe real: los de CBCT con IA no traen pH —que era lo único que la
#         capa regional sabía recoger— sino anatomía radicular y hallazgos por pieza.
#         Todo opcional y con defecto, así que un JSON 1.3.0 sigue validando.
SCHEMA_VERSION = "1.4.0"


# --------------------------------------------------------------------------- #
# Vocabulario controlado
# --------------------------------------------------------------------------- #
class Modality(str, Enum):
    """Fuente de la que procede un dato ingerido."""

    CBCT = "cbct"      # DICOM        → densidad σ (volumétrico)
    MESH = "mesh"      # malla intraoral (OBJ/PLY, color por vértice) → color_superficie
    REPORT = "report"  # PDF          → pH y otros atributos regionales
    IMAGE = "image"    # foto 2D


class ModalityStatus(str, Enum):
    """Resultado de la ingesta de una modalidad en un snapshot.

    Hace explícito el fallo/ausencia: un snapshot parcial deja de ser
    indistinguible de uno completo. Sin esto, «falta la malla» y «el agente de
    malla falló» serían el mismo silencio (ver ADR 001, manejo de fallos de ingesta).
    """

    OK = "ok"            # ingerida y traducida al contrato
    MISSING = "missing"  # no se aportó el fichero de esta modalidad
    FAILED = "failed"    # se intentó pero falló (corrupto, no parseable…)


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
    Field(
        pattern=r"^([1-4][1-8]|[5-8][1-5])$",
        description="Diente en numeración ISO-FDI, p. ej. '16'.",
    ),
]


# --------------------------------------------------------------------------- #
# Resultado de ingesta por modalidad (manejo explícito de fallos/ausencias)
# --------------------------------------------------------------------------- #
class ModalityIngestion(BaseModel):
    """Estado de la ingesta de una modalidad concreta en un snapshot.

    Es el registro *fail-loud* del borde de ingesta: el orquestador anota aquí
    el resultado de cada modalidad que intentó (o esperaba) ingerir, de modo
    que un snapshot parcial lo declare en vez de llegar callado a exportación.
    """

    model_config = ConfigDict(extra="forbid")

    modality: Modality
    status: ModalityStatus
    detail: str | None = Field(
        default=None, description="Motivo si status != ok (p. ej. 'DICOM corrupto')."
    )


# --------------------------------------------------------------------------- #
# Trazabilidad (requisito de transparencia del proyecto: RGPD/HIPAA)
# --------------------------------------------------------------------------- #
class RigidTransform(BaseModel):
    """Transformación **rígida** aplicada a un valor derivado (ADR 004 §2.2).

    Rígida a propósito. El registro malla↔CBCT alinea dos medidas del **mismo
    objeto físico**, ambas en milímetros reales, así que no hay escala ni cizalla
    que representar. Una matriz 4×4 general sí podría codificarlas, y una escala
    espuria metida por un ICP con un mal día rompería en silencio la garantía de
    reversibilidad. Esta forma la hace **imposible de expresar**.

    Además invertirla es exacto y barato —el conjugado del cuaternión más una
    rotación de la traslación— sin el mal condicionamiento de invertir una 4×4.
    """

    model_config = ConfigDict(extra="forbid")

    rotation: tuple[float, float, float, float] = Field(
        description="Cuaternión (w, x, y, z) normalizado; misma convención que GaussianPrimitive."
    )
    translation: tuple[float, float, float] = Field(description="Traslación en mm.")
    rms_mm: float | None = Field(
        None, ge=0.0, description="Residuo RMS del registro, en mm. Alimenta la confianza."
    )

    @field_validator("rotation")
    @classmethod
    def _cuaternion_normalizado(cls, v: tuple[float, float, float, float]):
        """Un cuaternión sin normalizar codifica una escala encubierta."""
        norma = math.sqrt(sum(c * c for c in v))
        if not math.isclose(norma, 1.0, abs_tol=1e-6):
            raise ValueError(f"el cuaternión debe estar normalizado (norma = {norma:.6f})")
        return v

    def inverse(self) -> RigidTransform:
        """La transformación que deshace esta. Hace la reversibilidad auditable.

        Para un cuaternión unitario la inversa es su conjugado, y la traslación
        inversa es `-R⁻¹·t`.
        """
        w, x, y, z = self.rotation
        conj = (w, -x, -y, -z)
        rx, ry, rz = _rotar(conj, self.translation)
        return RigidTransform(rotation=conj, translation=(-rx, -ry, -rz), rms_mm=self.rms_mm)


def _rotar(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rota el vector `v` por el cuaternión unitario `q` = (w, x, y, z).

    Fórmula estándar `v + 2w·(u×v) + 2·u×(u×v)` con `u` la parte vectorial: evita
    construir la matriz de rotación y no necesita numpy (este paquete solo depende
    de pydantic).
    """
    w, ux, uy, uz = q
    vx, vy, vz = v
    tx = 2 * (uy * vz - uz * vy)
    ty = 2 * (uz * vx - ux * vz)
    tz = 2 * (ux * vy - uy * vx)
    return (
        vx + w * tx + (uy * tz - uz * ty),
        vy + w * ty + (uz * tx - ux * tz),
        vz + w * tz + (ux * ty - uy * tx),
    )


class Provenance(BaseModel):
    """Procedencia de un valor: qué fichero, qué agente y con qué confianza.

    Se adjunta a cada observación para garantizar la explicabilidad exigida:
    "qué dato se ingirió, qué transformación se aplicó y por qué".
    """

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(description="Ruta o URI del fichero de origen.")
    modality: Modality
    agent: str = Field(description="Agente de ingesta que produjo el valor.")
    # `default=` y no el primer posicional, en los dos: MyPy reconoce el valor por
    # defecto de un campo solo cuando viene por palabra clave (así está definido
    # `dataclass_transform`). Con el posicional da los campos por **obligatorios** y
    # marca error en cada `Provenance(...)` que no los pase — que son casi todos.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    transform: RigidTransform | None = Field(
        default=None,
        description=(
            "Transformación rígida que el agente aplicó a este valor. `None` en ingesta "
            "(no transforma nada); la pueblan los agentes de fusión para que el cambio "
            "sea reversible. ADR 004 §2.2."
        ),
    )


# --------------------------------------------------------------------------- #
# Atributos por-punto: la primitiva gaussiana extendida  (θₙ⁺)
# --------------------------------------------------------------------------- #
class Color(BaseModel):
    """Color RGB de superficie (canal de apariencia reintroducido desde la malla intraoral).

    El color lo aporta el escáner intraoral como color por vértice en la malla
    (OBJ/PLY en el dataset Teeth3DS+). Un STL «pelado» no lleva color: por eso la
    fuente es la malla, no el formato STL.
    """

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
        color_superficie  → RGB de la malla intraoral (soporte SUPERFICIAL; None si
                            la gaussiana no cae en la banda ε de la superficie).
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
class Hallazgo(str, Enum):
    """Vocabulario controlado de hallazgos clínicos por diente.

    **Controlado y no texto libre**, por la misma razón que `Modality` o `Support`: un
    hallazgo tiene que poder compararse entre visitas y entre proveedores. «Signos de
    caries», «caries» y «lesión cariosa» son la misma cosa para un clínico y tres cosas
    distintas para un `==`, y una serie temporal construida sobre cadenas libres no se
    puede consultar.

    Los seis de aquí son los que **están medidos** sobre informes reales de CBCT con IA.
    Añadir uno es aditivo y no rompe nada; inventarlos por adelantado, en cambio, llena
    el vocabulario de términos que ningún extractor produce.
    """

    AUSENTE = "ausente"
    CARIES = "caries"
    RESTAURACION = "restauracion"
    CALCULO_PULPAR = "calculo_pulpar"
    PERDIDA_OSEA_PERIODONTAL = "perdida_osea_periodontal"
    APARATO_ORTODONCICO = "aparato_ortodoncico"


class ClinicalAttributes(BaseModel):
    """Metadatos clínicos definidos por zona/diente (no por punto).

    Un valor por región FDI. Extensible: hoy el pH, la anatomía radicular y los
    hallazgos; mañana movilidad, sangrado, profundidad de sondaje. Mantener aquí solo lo
    que sea genuinamente regional (el color y la densidad viven en `GaussianPrimitive`).

    **Por qué la anatomía radicular vive aquí y no en el campo gaussiano.** El número de
    raíces y de conductos es una propiedad *del diente*, no de un punto: no hay forma de
    repartir «3 conductos» entre las gaussianas de la pieza. Y tiene un uso que ningún
    otro atributo da — es **verificable contra la geometría**: un segmentador de CBCT que
    encuentre dos raíces donde el informe dice tres está fallando de una forma concreta y
    medible, sin necesidad de anotar el volumen.
    """

    model_config = ConfigDict(extra="forbid")

    ph: float | None = Field(default=None, ge=0.0, le=14.0)
    n_raices: int | None = Field(
        default=None, ge=1, le=5,
        description="Raíces de la pieza. El máximo anatómico es 4-5 en molares superiores.",
    )
    n_conductos: int | None = Field(
        default=None, ge=1, le=8,
        description="Conductos radiculares. Puede superar al número de raíces.",
    )
    hallazgos: list[Hallazgo] = Field(
        default_factory=list,
        description="Hallazgos del informe para esta pieza, en vocabulario controlado.",
    )


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
    para regenerar la malla/imágenes de esa fecha. El campo gaussiano masivo no se
    embebe: se referencia por hash/URI al almacén de `3dgs-engine`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Versión del contrato bajo el que se escribió este snapshot (SemVer).",
    )
    acquisition_id: str
    timestamp: datetime
    modalities: list[Modality] = Field(
        default_factory=list,
        description="Modalidades presentes (status OK). El resultado completo por "
        "modalidad —incluidas las que faltan o fallaron— vive en `ingestion`.",
    )
    ingestion: list[ModalityIngestion] = Field(
        default_factory=list,
        description="Log autoritativo del resultado de ingesta por modalidad (fail-loud).",
    )
    gaussian_field_ref: str = Field(
        description="Hash/URI del campo gaussiano en 3dgs-engine. Invariante fail-loud: "
        "al cargar/exportar hay que validar que el blob referenciado existe; una "
        "referencia colgante es un error, no un modelo vacío silencioso.",
    )
    surface_ref: str | None = Field(
        default=None,
        description="Hash/URI de la malla intraoral ingerida (posiciones, normales, "
        "color por vértice). Existe porque el `mesh-agent` produce un artefacto "
        "propio ANTES de la fusión geométrica: hasta que el color se transfiera a "
        "las gaussianas de la banda ε, la superficie no tiene otro sitio donde "
        "vivir dentro del contrato. Mismo invariante fail-loud que "
        "`gaussian_field_ref`: una referencia colgante es un error.",
    )
    image_refs: list[str] = Field(
        default_factory=list,
        description="Hash/URI de las fotos intraorales ingeridas (píxeles RGB, sin EXIF). "
        "Es una **lista** porque una adquisición trae varias fotos (p. ej. 5 en "
        "Bite2Text). Como el `surface_ref`, es apariencia **pre-fusión**: la foto "
        "existe en el contrato antes de que la fusión geométrica proyecte su color "
        "sobre las gaussianas. Mismo invariante fail-loud: una referencia colgante "
        "es un error.",
    )
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
