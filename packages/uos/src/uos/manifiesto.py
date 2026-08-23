"""El manifiesto: `manifest.json`. Es el contrato del contenedor, y se valida como tal.

Cada campo del spec v0.2 §4 con su tipo. Lo que NO se declara aqui no puede entrar en un
`.uos`, que es justo el punto: un lector tiene que poder negarse en vez de adivinar.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

UOS_VERSION = "0.2"

# Media type propuesto (§10). Draft: el arbol `vnd.` se registra en IANA cuando el spec
# se publique, y hasta entonces la identificacion positiva es que la PRIMERA entrada del
# ZIP sea `manifest.json` con un `uos_version`.
MEDIA_TYPE = "application/vnd.histora.uos"


class EstadoPHI(StrEnum):
    """Anonimizacion como ESTADO EXPLICITO (§2.6), no como suposicion.

    Un `.uos` dice en que estado esta y los visores lo respetan —banner, politicas de
    export—. Sin este campo, «no se si lleva datos identificables» y «no los lleva» serian
    indistinguibles, que es la peor forma de manejar PHI.
    """

    IDENTIFIED = "identified"
    PSEUDONYMIZED = "pseudonymized"
    ANONYMIZED = "anonymized"
    QUARANTINED = "quarantined"


class Clase(StrEnum):
    """`kind` de un asset (§4.1). El nivel UOS-Core solo exige los tres primeros."""

    VOLUME = "volume"
    MESH_GS_SCENE = "mesh_gs_scene"
    IMAGE2D = "image2d"
    SIGNAL = "signal"
    DERIVED_SEG = "derived_seg"
    DOCUMENT = "document"


# Orden de carga progresiva (§4.1): menor = antes. La escena de malla primero porque es
# lo que permite ensenar algo util habiendo leido solo el manifiesto y el asset mas ligero.
PRIORIDAD = {
    Clase.MESH_GS_SCENE: 10,
    Clase.IMAGE2D: 20,
    Clase.VOLUME: 30,
    Clase.SIGNAL: 30,
    Clase.DERIVED_SEG: 40,
    Clase.DOCUMENT: 50,
}


class Regulatorio(BaseModel):
    """La capa regulatoria del asset (§1.1). Layer 3 es SaMD y se puede desmontar."""

    model_config = ConfigDict(extra="forbid")
    layer: int = Field(default=1, ge=1, le=3)
    status: str | None = None
    jurisdictions: list[str] = Field(default_factory=list)


class Adquisicion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time: datetime | None = None
    device: dict[str, str] = Field(default_factory=dict)


class Asset(BaseModel):
    """El sobre comun a todo asset (§4.1).

    `sha256` no es decorativo: es lo que hace la procedencia verificable (§2.5) y lo que
    permite decir que el DICOM que sale es byte-identico al que entro.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Clase
    visit: str
    uri: str
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    frame: str
    acquisition: Adquisicion = Field(default_factory=lambda: Adquisicion())
    load_priority: int = 30
    regulatory: Regulatorio = Field(default_factory=lambda: Regulatorio())

    @field_validator("uri")
    @classmethod
    def _relativa_y_ascii(cls, v: str) -> str:
        """Rutas internas relativas, ASCII y sin `..` (§3).

        No es purismo: un `..` en una ruta de ZIP es la travesia de directorios clasica, y
        un lector que la resuelva ingenuamente escribe fuera del destino.
        """
        if not v.isascii() or v.startswith("/") or ".." in v.split("/"):
            raise ValueError(
                f"uri {v!r}: las rutas internas son relativas, ASCII y sin '..'"
            )
        return v


class Frame(BaseModel):
    """Un sistema de coordenadas con nombre. El canonico es el hub geometrico (§2.2)."""

    model_config = ConfigDict(extra="forbid")
    id: str
    description: str = ""
    units: str = "mm"
    handedness: str = "right"


class Registro(BaseModel):
    """Relacion espacial entre dos frames. **Objeto de primera clase** (§6).

    Toda relacion es explicita, auditable y firmable. La transformada lleva puntos de
    `source_frame` a `target_frame`, en milimetros y mano derecha.

    ⚠️ `verified_by` vacio en un registro `auto_dl` significa PROVISIONAL, y el visor tiene
    que indicarlo. Un alineamiento automatico que nadie ha mirado no es lo mismo que uno
    firmado, y presentarlos igual seria exactamente el fallo callado que el spec evita.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    source_frame: str
    target_frame: str
    transform_4x4_row_major: list[float] = Field(min_length=16, max_length=16)
    method: str
    rms_error_mm: float | None = None
    computed: datetime | None = None
    operator: str | None = None
    verified_by: str | None = None
    regulatory: Regulatorio = Field(default_factory=lambda: Regulatorio())

    @property
    def provisional(self) -> bool:
        """Automatico y sin verificar por una persona."""
        return self.method == "auto_dl" and not self.verified_by


class Visita(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    date: str
    label: str = ""


class Sujeto(BaseModel):
    """El paciente, SIEMPRE por seudonimo. El nombre no entra en un `.uos`."""

    model_config = ConfigDict(extra="forbid")
    pseudonym: str
    fhir_patient: str | None = None


class Procedencia(BaseModel):
    """Cadena de hashes entre versiones del caso (§8). `.uos` es append-only logico."""

    model_config = ConfigDict(extra="forbid")
    prev_manifest_sha256: str | None = None
    chain: str | None = None


class Manifiesto(BaseModel):
    """`manifest.json`. DEBE ser la primera entrada fisica del ZIP (§3)."""

    model_config = ConfigDict(extra="forbid")
    uos_version: str = UOS_VERSION
    case_id: str
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    generator: dict[str, str]
    phi_state: EstadoPHI
    subject: Sujeto
    canonical_frame: Frame
    frames: list[Frame] = Field(default_factory=list)
    visits: list[Visita] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    registrations: list[Registro] = Field(default_factory=list)
    fhir_map: dict[str, dict[str, str]] = Field(default_factory=dict)
    provenance: Procedencia = Field(default_factory=lambda: Procedencia())

    def json_canonico(self) -> str:
        """JSON estable: mismas claves, mismo orden, misma salida.

        Hace falta para la cadena de hashes: si el mismo manifiesto se serializara distinto
        entre ejecuciones, `prev_manifest_sha256` cambiaria sin que cambiara nada.
        """
        return self.model_dump_json(indent=1, exclude_none=False)
