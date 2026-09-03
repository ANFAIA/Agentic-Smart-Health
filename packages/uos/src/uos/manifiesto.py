"""El manifiesto: `manifest.json`. Es el contrato del contenedor, y se valida como tal.

Cada campo del spec v0.2 §4 con su tipo. Lo que NO se declara aqui no puede entrar en un
`.uos`, que es justo el punto: un lector tiene que poder negarse en vez de adivinar.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UOS_VERSION = "0.2"

# Media type propuesto (§10). Draft: el arbol `vnd.` se registra en IANA cuando el spec
# se publique, y hasta entonces la identificacion positiva es que la PRIMERA entrada del
# ZIP sea `manifest.json` con un `uos_version`.
MEDIA_TYPE = "application/vnd.histora.uos"


class PHIState(StrEnum):
    """Anonimizacion como ESTADO EXPLICITO (§2.6), no como suposicion.

    Un `.uos` dice en que estado esta y los visores lo respetan —banner, politicas de
    export—. Sin este campo, «no se si lleva datos identificables» y «no los lleva» serian
    indistinguibles, que es la peor forma de manejar PHI.
    """

    IDENTIFIED = "identified"
    PSEUDONYMIZED = "pseudonymized"
    ANONYMIZED = "anonymized"
    QUARANTINED = "quarantined"


class AssetKind(StrEnum):
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
    AssetKind.MESH_GS_SCENE: 10,
    AssetKind.IMAGE2D: 20,
    AssetKind.VOLUME: 30,
    AssetKind.SIGNAL: 30,
    AssetKind.DERIVED_SEG: 40,
    AssetKind.DOCUMENT: 50,
}


class Clearance(BaseModel):
    """Que dice UNA jurisdiccion sobre este asset (B-5).

    Sustituye al par `status` + `jurisdictions`, que no se podia leer: `status` era texto
    libre y `jurisdictions: []` era ambiguo —¿ninguna, o no declarado?—, que es justo la
    ambiguedad que el formato prohibe en `fdi_targets` y en `derivation`. Un estado sin
    jurisdiccion no significa nada: «investigational» es una afirmacion frente a UN
    regulador, no una propiedad del fichero.
    """

    model_config = ConfigDict(extra="forbid")
    jurisdiction: str
    regime: str
    status: ClearanceStatus
    #: El numero de expediente o de autorizacion, cuando existe. `None` NO es "no hay":
    #: es "no consta aqui", igual que `weights_sha256` en el sidecar de `derived/`.
    reference: str | None = None


class ClearanceStatus(StrEnum):
    """Vocabulario CERRADO. `status` era `str | None` y por tanto texto libre: dos
    emisores escribian «investigational» y «Investigational (EU MDR)» y ningun lector
    podia compararlos."""

    NO_ES_PRODUCTO = "not_a_device"
    INVESTIGACION = "investigational"
    PRESENTADO = "submitted"
    AUTORIZADO = "cleared"
    RETIRADO = "withdrawn"


class Regulatory(BaseModel):
    """La capa regulatoria del asset (§1.1) y lo que un regulador dice de el.

    **Las tres capas.** 1 es lo adquirido y su transcripcion; 2 es lo COMPUTADO por un
    procedimiento determinista y reproducible a partir de capa 1, sin modelo entrenado —
    registraciones automaticas, conversiones de formato, submuestreos, color medido por
    pieza—; 3 es salida de modelo. El 2 no existia: el documento admitia `1..3` y solo
    definia el 1 y el 3, asi que todo el computo determinista viajaba como capa 1 sin que
    nadie lo dijera (B-5).

    **La 2 no se desmonta y la 3 si.** La 3 vive solo bajo `derived/` porque borrar ese
    directorio tiene que quitar toda la inferencia. La 2 puede vivir fuera, pero **tiene
    que declarar `derived_from`**: si es reproducible, se tiene que poder decir a partir
    de que, o la afirmacion no se puede comprobar.

    ⚠️ **`clearances: []` significa NO DECLARADO**, por definicion escrita y no por
    convencion. El validador avisa por cada asset de capa 3 que llegue vacio.
    """

    model_config = ConfigDict(extra="forbid")
    layer: int = Field(default=1, ge=1, le=3)
    clearances: list[Clearance] = Field(default_factory=list)


class PurposeOfUse(StrEnum):
    """Para que se emitio ESTE contenedor (B-4). Vocabulario cerrado.

    Un `.uos` que sale hacia un laboratorio protesico, hacia un colega para una segunda
    opinion y hacia un pipeline de entrenamiento son **tres actos juridicos distintos**, y
    el contenedor no distinguia ninguno. El proposito es lo primero que pregunta cualquier
    revision de proteccion de datos y es lo que decide que capas pueden viajar; sin el,
    cada receptor tiene que suponerlo.
    """

    TRATAMIENTO = "treatment"
    FABRICACION = "lab_manufacturing"
    SEGUNDA_OPINION = "second_opinion"
    INVESTIGACION = "research"
    ENTRENAMIENTO = "model_training"


class Consent(BaseModel):
    """Para que consintio el paciente. `scope` es lo que limita `purpose_of_use`."""

    model_config = ConfigDict(extra="forbid")
    #: Referencia al recurso `Consent` de FHIR R4 cuando el caso vive en un servidor.
    #: `None` mientras no exista, misma regla que `FHIRResource.resource`.
    fhir_consent: str | None = None
    scope: list[PurposeOfUse] = Field(default_factory=list)
    obtained: datetime | None = None


class Tool(BaseModel):
    """Que programa aplico la de-identificacion, para poder repetirla o auditarla."""

    model_config = ConfigDict(extra="forbid")
    name: str
    version: str | None = None
    sha256: str | None = None


class Deidentification(BaseModel):
    """QUE se hizo para de-identificar, en el vocabulario de DICOM PS3.15 Anexo E (B-3).

    ⚠️ **`phi_state` solo no puede sostener lo que afirma.** Trata la de-identificacion
    como una propiedad de las etiquetas DICOM, y el contenedor lleva cosas que identifican
    a una persona sin ninguna etiqueta: `scene/field.ply` es la densidad del CBCT con
    tejido blando incluido, y de ahi se reconstruye una **superficie facial** —«imagen
    comparable» a una fotografia de cara completa bajo HIPAA Safe Harbor, y dato biometrico
    bajo el RGPD—. La denticion identifica por si sola: de eso vive la odontologia forense.

    Por eso `pseudonymized` y `anonymized` exigen este bloque. No cambia lo que el
    contenedor lleva; cambia que diga **que se hizo** en lugar de **como quedo**, que es la
    unica de las dos afirmaciones que alguien puede comprobar.
    """

    model_config = ConfigDict(extra="forbid")
    profile: str
    #: Las opciones con nombre del Anexo E: `CleanDescriptors`,
    #: `CleanRecognizableVisualFeatures`, `RetainLongitudinalTemporalInformationModifiedDates`...
    options: list[str] = Field(default_factory=list)
    #: Sobre que assets se ejecuto. El validador avisa por cada `volume` o `image2d` que
    #: no aparezca aqui: no estar en la lista significa que no se le aplico nada.
    applied_to: list[str] = Field(default_factory=list)
    tool: Tool | None = None
    #: ⚠️ **El desplazamiento de fechas es la CLAVE de re-identificacion.** Solo viaja si
    #: el contenedor ya se declara `identified`; en cualquier otro estado va a `null`,
    #: porque publicarlo deshace la medida que dice haber aplicado.
    date_shift_days: int | None = None
    note: str | None = None

    #: La opcion de PS3.15 que quita la superficie facial reconstruible (el «defacing»).
    LIMPIA_RASGOS: ClassVar[str] = "CleanRecognizableVisualFeatures"


class Device(BaseModel):
    """El equipo, con claves FIJAS y **sin numero de serie** (B-3).

    Era `map str -> str` sin restriccion, y un mapa libre en un contenedor clinico acaba
    conteniendo el serial del equipo — que bajo HIPAA Safe Harbor es identificador
    directo, en la misma lista que el nombre. Si un flujo lo necesita, va en
    `deidentification.note` y el contenedor se declara `identified`.
    """

    model_config = ConfigDict(extra="forbid")
    manufacturer: str | None = None
    model: str | None = None
    software_version: str | None = None


class Acquisition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time: datetime | None = None
    device: Device = Field(default_factory=lambda: Device())


class Projection(BaseModel):
    """Que clase de imagen 2D es y a que piezas apunta (§5.3).

    `fdi_targets` va VACIO cuando no se sabe, que con una foto suelta de una carpeta de
    clinica es lo normal: nadie anoto a que diente apunta. Vacio significa «no consta»,
    no «ninguno» — y adivinarlo desde los pixeles es justo el trabajo que el proyecto no
    da por hecho.
    """

    model_config = ConfigDict(extra="forbid")
    type: str
    fdi_targets: list[str] = Field(default_factory=list)


class Part(BaseModel):
    """Un fichero dentro de un asset que es un DIRECTORIO (una serie DICOM).

    Existe para que la verificacion sea POR FICHERO. Un solo hash del conjunto dice que
    algo cambio; estos dicen cual de los 397 cortes, que es la diferencia entre «esta serie
    no cuadra» y «el corte 214 esta corrupto».

    ⚠️ **Y el hash del FICHERO no es la identidad del corte (D-3).** `sha256` cubre la
    cabecera entera, asi que cualquier de-identificacion —el paso que todo flujo clinico da,
    y que B-3 ademas exige— reescribe etiquetas y cambia el hash. La trazabilidad que el
    §3.4.2 promete («quien tenga la serie puede probar que es la de este caso, corte a
    corte») se rompia exactamente en el paso mas comun.

    DICOM ya resolvio que identifica una instancia: el **SOP Instance UID** `(0008,0018)`,
    que sobrevive a la de-identificacion cuando se elige retener UIDs, y el contenido de
    pixeles `(7FE0,0010)`, que la de-identificacion no toca salvo que se limpie a proposito.
    Los dos viajan al lado del hash del fichero, y la verificacion pasa a tener dos niveles
    que se reportan por separado: **identidad** (UID + pixeles) y **bytes exactos** (hash
    del fichero). Un corte de-identificado que conserva identidad pasa el primero y falla el
    segundo, y eso es informacion, no un error.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    #: `(0008,0018)`. La identidad clinica del corte, estable a traves de la de-identificacion.
    sop_instance_uid: str | None = None
    #: SHA-256 sobre el VALOR del elemento PixelData, sin cabecera.
    pixel_data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


#: Una uri que no es una ruta sino la identidad del fichero. Ver `Asset._direccion_y_custodia`.
DIRECCION = re.compile(r"sha256:[0-9a-f]{64}")


def direccion_de_contenido(sha256: str) -> str:
    """`sha256:<hex>` — como se nombra un asset que no viaja dentro del contenedor."""
    return f"sha256:{sha256}"


class Asset(BaseModel):
    """El sobre comun a todo asset (§4.1).

    `sha256` no es decorativo: es lo que hace la procedencia verificable (§2.5) y lo que
    permite decir que el DICOM que sale es byte-identico al que entro.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: AssetKind
    visit: str
    uri: str
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    frame: str
    acquisition: Acquisition = Field(default_factory=lambda: Acquisition())
    load_priority: int = 30
    regulatory: Regulatory = Field(default_factory=lambda: Regulatory())
    # Solo en assets que son un directorio (`uri` acabada en `/`). Ver `Part` y
    # `digesto_de_partes`.
    # ⚠️ §5.3. `projection` describe QUE tipo de imagen es y a que dientes apunta; `pose`
    # —donde estaba la camara— es explicitamente opcional en el spec y aqui no se emite:
    # una foto intraoral de una carpeta de clinica no trae pose, y calcularla exige la
    # fusion foto↔malla, que esta medida y no converge barata sin calibracion.
    projection: Projection | None = None

    parts: list[Part] = Field(default_factory=list)
    # Sidecar que describe el asset sin obligar a parsearlo. Lo pide §5.2 para el volumen:
    # un visor web no deberia necesitar un parser DICOM completo para saber que le llega.
    sidecar_uri: str | None = None
    # ⚠️ **De que otros assets sale este. No hay § que lo defina, y hace falta.**
    #
    # `scene/scene.glb` es la malla del escaner reindexada a glTF: MISMA geometria,
    # 220.085 triangulos, los mismos que el STL. Pero el manifiesto lo declaraba como un
    # asset suelto de tipo `mesh_gs_scene` —sin sidecar y sin ningun enlace al `asset.ios`
    # que se declara FUERA por su hash—, asi que quien recibiera el contenedor no tenia
    # como saber que esa malla es una conversion del escaner y no una fuente independiente.
    # El dato estaba, pero dentro del propio GLB (`extras.uos_source_asset`): habia que
    # parsear dieciocho megas de glTF para enterarse de algo que el manifiesto puede decir
    # en una linea.
    #
    # Es informacion que SUMA: un lector que no conozca el campo lo ignora y abre el caso
    # igual. Por eso va en `extensions_used` y nunca en `extensions_required`.
    derived_from: list[str] = Field(default_factory=list)
    #: D-3 · `(0020,000E)` y `(0020,000D)`. Lo que DICOM define como identidad de la serie
    #: y del estudio, para que un PACS pueda casar este asset con lo que ya tiene.
    series_instance_uid: str | None = None
    study_instance_uid: str | None = None
    # ⚠️ **El asset NO viaja dentro del contenedor: solo su identidad.** Es lo que hace
    # TODO original adquirido —DICOM, STL, fotos, informes—: se referencia por su direccion
    # de contenido y se acredita por `sha256`. No es un perfil ni una variante; es el
    # formato, y por eso el defecto de este campo es lo unico que sorprende: `False` porque
    # describe el sobre, no una recomendacion.
    #
    # Lo que el contenedor afirma de un asset asi es «se que fichero es», no «lo tengo». La
    # verificacion byte a byte que el §1.1 del borrador pide del DICOM adquirido no se puede
    # dar por construccion, y no se finge: `parts` sigue viajando para que quien SI lo
    # custodie pueda demostrar que no le falta un corte.
    #
    # El validador lo dice UNA vez por contenedor y no una por asset: si todos los
    # originales son externos siempre, un aviso por cada uno no distingue nada.
    external: bool = False

    @field_validator("uri")
    @classmethod
    def _relativa_y_ascii(cls, v: str) -> str:
        """Rutas internas relativas, ASCII y sin `..` (§3).

        No es purismo: un `..` en una ruta de ZIP es la travesia de directorios clasica, y
        un lector que la resuelva ingenuamente escribe fuera del destino.

        La direccion de contenido (`sha256:<hex>`) se acepta aparte: no es una ruta, asi
        que las reglas de ruta no se le aplican. Que solo pueda usarla un asset externo lo
        comprueba `_direccion_y_custodia`.
        """
        if DIRECCION.fullmatch(v):
            return v
        if not v.isascii() or v.startswith("/") or ".." in v.split("/"):
            raise ValueError(
                f"uri {v!r}: las rutas internas son relativas, ASCII y sin '..'"
            )
        return v

    @model_validator(mode="after")
    def _direccion_y_custodia(self) -> Asset:
        """Un asset externo se nombra por su CONTENIDO; uno interno, por su sitio.

        **Por que la uri de un externo es el hash.** Un asset que no viaja no tiene «sitio
        dentro del contenedor», asi que una ruta seria una promesa sobre un ZIP en el que
        no esta. Lo unico que sigue siendo cierto de el es **que fichero es**, y eso es
        exactamente lo que dice una direccion de contenido. Es la misma convencion que el
        `ArtifactStore` del proyecto usa desde el principio (`sha256:<hex>`), asi que un
        resolvedor que ya tenga un almacen direccionado por contenido no necesita saber
        nada de UOS para servirlo.

        ⚠️ Y tiene una propiedad que una ruta no tiene: **no puede llevar dato de
        paciente**. La ruta local de un caso clinico lleva el directorio del paciente; un
        hash no lleva nada. Referenciar saca ficheros del contenedor, no identidades.

        Se comprueba en los dos sentidos, y el segundo importa igual: una direccion de
        contenido en un asset que SI viaja seria un asset imposible de localizar dentro del
        ZIP.
        """
        es_direccion = bool(DIRECCION.fullmatch(self.uri))
        if self.external and not es_direccion:
            raise ValueError(
                f"asset {self.id}: es externo y su uri {self.uri!r} es una ruta. Un asset "
                "que no viaja se nombra por su contenido: `sha256:<hex>`."
            )
        if es_direccion:
            if not self.external:
                raise ValueError(
                    f"asset {self.id}: su uri es una direccion de contenido pero el asset "
                    "viaja dentro; entonces no habria forma de encontrarlo en el ZIP."
                )
            if self.uri.split(":", 1)[1] != self.sha256:
                raise ValueError(
                    f"asset {self.id}: la direccion de contenido y el campo `sha256` no "
                    "son el mismo hash."
                )
        return self


def digesto_de_partes(partes: list[Part]) -> str:
    """El `sha256` de un asset-directorio: hash sobre `nombre\0hash\n` ordenado.

    El spec da un `sha256` por asset y no dice como se calcula cuando el asset es una serie
    entera. Se define aqui, y **se define en vez de elegirse en el escritor**: si el
    validador lo calculara de otra forma, un contenedor valido daria invalido y nadie
    sabria cual de los dos tiene razon.

    ⚠️ **Sobre la IDENTIDAD de la serie, no sobre una serializacion suya (D-3).** Iba sobre
    el nombre y el hash del fichero, y los dos cambian al de-identificar: reescribir
    cabeceras cambia el hash, y el paso de de-identificacion es el mas comun de un flujo
    clinico. Un digesto asi identificaba «esta copia concreta de esta serie», no la serie.

    Cuando los cortes declaran su `sop_instance_uid` y su `pixel_data_sha256` se usan esos:
    el UID es la identidad que DICOM define y el hash de pixeles es lo que la
    de-identificacion no toca. Cuando no los declaran se cae al nombre y al hash del
    fichero, que es lo que habia — un contenedor antiguo sigue verificando, y el que trae
    identidad la usa. Los nombres siguen en `parts[]` como dato de ordenacion.
    """
    import hashlib

    h = hashlib.sha256()
    identidad = all(p.sop_instance_uid and p.pixel_data_sha256 for p in partes)
    if identidad:
        for p in sorted(partes, key=lambda x: x.sop_instance_uid or ""):
            h.update(f"{p.sop_instance_uid}\0{p.pixel_data_sha256}\n".encode())
        return h.hexdigest()
    for p in sorted(partes, key=lambda x: x.name):
        h.update(f"{p.name}\0{p.sha256}\n".encode())
    return h.hexdigest()


class AnatomicalConvention(StrEnum):
    """La convencion de ejes del frame (D-2).

    ⚠️ **«Diestro» fija la quiralidad, no la orientacion.** DICOM es LPS; un escaner
    intraoral usa un sistema arbitrario del aparato; glTF es Y-arriba sin significado
    anatomico. Los tres pueden ser diestros y no coincidir en nada util: un lector que
    reciba el frame del escaner no sabe cual de sus direcciones es anterior, superior o
    derecha del paciente. Cualquier medida clinica sobre el modelo —un angulo, una
    distancia a una estructura— necesita saberlo, y hasta ahora habia que mirar la imagen.
    """

    #: +X izquierda del paciente, +Y posterior, +Z superior. Es lo que DICOM impone.
    LPS = "LPS"
    #: +X derecha, +Y anterior, +Z superior. Frecuente en neuroimagen.
    RAS = "RAS"
    #: Sistema propio del aparato, sin significado anatomico declarado.
    DISPOSITIVO = "device"


class OcclusionRecord(StrEnum):
    """Como se registro la relacion entre arcadas (D-9).

    ⚠️ **Es la registracion clinicamente mas importante de un caso dental y el formato no
    la nombraba.** Mandibula<->maxila —el registro de mordida— es lo que decide si dos
    arcadas se pueden mirar juntas, y un formato dental que no le da nombre invita a que
    cada escritor la llame distinto. El caso de referencia es solo maxilar y por eso no
    aparecia; eso es una razon para reservarla, no para omitirla.

    Y **el silencio no es «no hay»**: un caso con dos arcadas declara como la registro o
    declara `not_recorded`, que es una afirmacion distinta de no decir nada.
    """

    #: Escaneo lateral adicional con los dientes en contacto.
    ESCANEO_MORDIDA = "bite_scan"
    #: Articulador o registro fisico llevado a digital.
    ARTICULADOR = "articulator"
    #: No se registro. Dicho, no callado.
    NO_REGISTRADA = "not_recorded"
    #: El caso trae una sola arcada, asi que no hay relacion que registrar.
    NO_APLICA = "single_arch"


class RegistrationFitness(StrEnum):
    """Para que sirve una registracion, medido y no supuesto (D-9).

    ⚠️ **`rms_error_mm` es un promedio global y no decide un uso clinico.** Para cirugia
    guiada de implantes lo que importa es el error maximo local en la zona de interes:
    0,666 mm de RMS es aceptable para visualizar y no para planificar. Un lector
    **MUST NOT** suponer aptitud para un uso que no este en la lista, y la lista vacia
    significa NO DECLARADO.
    """

    VISUALIZACION = "visualization"
    MEDICION = "measurement"
    CIRUGIA_GUIADA = "guided_surgery"


class SiteKind(StrEnum):
    """Sitios que NO son un diente (D-9).

    `uos_fdi` solo etiqueta dientes, asi que no habia forma de senalar un lecho de
    implante, una zona edentula ni un pilar protesico — que es de lo que trata media
    rehabilitacion. Se declara el vocabulario aunque este emisor todavia no los produzca:
    reservarlo es lo que evita que cada escritor invente el suyo.
    """

    LECHO_IMPLANTE = "implant_site"
    EDENTULO = "edentulous"
    PONTICO = "pontic"
    PILAR = "abutment"


class Frame(BaseModel):
    """Un sistema de coordenadas con nombre. El canonico es el hub geometrico (§2.2).

    ⚠️ **Un frame es POR ADQUISICION, no por aparato (D-2).** Dos escaneos del mismo
    paciente en visitas distintas son frames distintos aunque salgan del mismo escaner.
    Compartir frame equivale a afirmar que las dos nubes ya estan en el mismo espacio, y
    entre dos visitas eso es falso: el paciente se movio, la mordida cambio, o las dos.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    description: str = ""
    units: str = "mm"
    handedness: str = "right"
    #: D-2. `None` es NO DECLARADO y no equivale a `device`.
    anatomical: AnatomicalConvention | None = None
    #: D-1 · el identificador que DICOM ya define para un sistema de coordenadas, etiqueta
    #: `(0020,0052)`. `frame.ct_001` es una cadena que se invento el escritor: un lector
    #: que reciba la serie por otro canal no tiene forma de saber que es ESA serie salvo
    #: por confianza. El UID es global y unico, y **se LEE de la serie, nunca se inventa**
    #: (misma regla que ya rige para `orientation` en el sidecar del volumen).
    dicom_frame_of_reference_uid: str | None = None


class Registration(BaseModel):
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
    #: D-9 · el error MAXIMO local, que es el que decide un uso clinico. El RMS es un
    #: promedio y se reparte: puede ser bueno y esconder una zona mala.
    max_error_mm: float | None = None
    #: Error medido en puntos que NO se usaron para calcular la registracion (TRE). El RMS
    #: se mide sobre los puntos que si se usaron y por tanto es sistematicamente optimista.
    target_registration_error_mm: float | None = None
    #: La region donde se midio el TRE, porque un TRE sin region no dice nada.
    tre_region: str | None = None
    #: ⚠️ Vacio significa NO DECLARADO, y un lector **MUST NOT** suponer aptitud para un
    #: uso que no este aqui. Es la misma regla que `fdi_targets` y `clearances`.
    fit_for: list[RegistrationFitness] = Field(default_factory=list)
    computed: datetime | None = None
    operator: str | None = None
    verified_by: str | None = None
    #: ⚠️ **Sin defecto a proposito (B-5).** Con `default_factory` toda registracion
    #: llegaba con `layer: 1` puesto y no habia forma de distinguir «se declaro capa 1» de
    #: «nadie lo declaro». Una registracion calculada por una maquina es computo, no
    #: adquisicion, y el validador exige que lo diga cuando `operator` empieza por `auto:`.
    regulatory: Regulatory | None = None

    #: Prefijo con el que una maquina firma `operator`. Ver `provisional`.
    #: `ClassVar` para que pydantic no lo tome por un campo del manifiesto.
    AUTO: ClassVar[str] = "auto:"

    #: D-9 · el id RESERVADO del registro de mordida. Reservarlo es lo que evita que cada
    #: escritor lo llame distinto y que dos contenedores no se puedan comparar.
    OCLUSION: ClassVar[str] = "reg.mandible_to_maxilla"

    @property
    def provisional(self) -> bool:
        """Automatico y sin verificar por una persona.

        ⚠️ **Mira QUIEN lo calculo, no COMO.** Antes exigia `method == "auto_dl"`, y la
        consecuencia es que no cubria ni nuestro propio caso: la unica registracion que
        emitimos declara `method: "icp_surface"` con `operator: "auto:geometric-fusion-
        agent@0.2.0"` y `verified_by` vacio. Automatica, sin revisar, 0,666 mm de residuo
        — y no disparaba la regla que existe para ella.

        Una salvaguarda escrita contra el nombre de UN algoritmo deja de funcionar en
        cuanto alguien usa otro, que es siempre. Lo que decide si una alineacion es
        provisional es si la miro una persona; `method` describe la tecnica y es otro dato.
        """
        return bool(self.operator and self.operator.startswith(self.AUTO)) and not (
            self.verified_by
        )


class Visit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    date: str
    label: str = ""


class Subject(BaseModel):
    """El paciente, SIEMPRE por seudonimo. El nombre no entra en un `.uos`."""

    model_config = ConfigDict(extra="forbid")
    #: ⚠️ **HMAC con clave, NUNCA un hash simple del identificador clinico.** El espacio de
    #: identificadores de una clinica es pequeno, asi que un hash sin clave se invierte por
    #: diccionario y el seudonimo no seudonimiza nada. Ver `cbct_agent.pseudonymize`.
    pseudonym: str
    fhir_patient: str | None = None
    consent: Consent | None = None


class FHIRResource(BaseModel):
    """A que recurso FHIR R4 corresponde un asset (§9), para el conector con el PMS.

    ⚠️ **`resource` es una referencia y `resource_type` un TIPO, y no son lo mismo.** El
    ejemplo del spec escribe `"resource": "ImagingStudy/is-9911"`, o sea una referencia a un
    recurso que existe en un servidor concreto. Nosotros no tenemos ese servidor: el caso no
    ha pasado por ningun PMS y no hay identificador que citar. Declarar uno inventado seria
    exactamente el fallo del ADR 003 —plausible, silencioso y ya dentro del contrato—,
    porque un conector que lo leyera intentaria resolverlo.

    Asi que se declara lo que SI se sabe: el tipo de recurso al que corresponde cada asset.
    Es una afirmacion de tipo, verdadera hoy y sin servidor, y es lo que un conector
    necesita para saber que crear. `resource` queda para cuando el caso viva en un PMS de
    verdad y alguien pueda rellenarlo.
    """

    model_config = ConfigDict(extra="forbid")
    resource_type: str
    resource: str | None = None
    note: str = ""


class Extension(BaseModel):
    """Una extension del formato, DECLARADA (no hay §; es propuesta nuestra).

    **Por que hace falta.** UOS se apoya en glTF, que trae `extensionsUsed` /
    `extensionsRequired` desde la 1.0 y los mantiene sin cambios en la 2.0: un lector abre
    el fichero, ve que extensiones trae, y sabe si puede leerlo entero, en parte o nada.
    UOS v0.2 **no hereda ese mecanismo a nivel de contenedor**: ni el manifiesto ni el
    sobre de asset tienen donde
    decir «esto es una extension, se llama asi, y si no la entiendes ignorala».

    La consecuencia practica la vimos implementando: nuestras extensiones —la capa clinica,
    los descriptores de gaussianas medidas— viven en directorios propios y **un lector
    ajeno las ignora sin enterarse de que las ignora**. Eso convierte un formato abierto en
    uno que solo el emisor lee entero, que es justo lo que UOS existe para evitar.

    ⚠️ La distincion entre `used` y `required` es el corazon del mecanismo, no burocracia:
    `required` dice «sin entender esto NO abras el fichero». Una extension que solo anade
    informacion va en `used` y nunca en `required` — si estuviera, un visor conforme se
    negaria a abrir un caso que podria enseniar perfectamente.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    uri: str | None = None
    schema_id: str | None = None
    description: str = ""


class Provenance(BaseModel):
    """Chain de hashes entre versiones del caso (§8). `.uos` es append-only logico."""

    model_config = ConfigDict(extra="forbid")
    prev_manifest_sha256: str | None = None
    chain: str | None = None


class Manifest(BaseModel):
    """`manifest.json`. DEBE ser la primera entrada fisica del ZIP (§3)."""

    model_config = ConfigDict(extra="forbid")
    uos_version: str = UOS_VERSION
    case_id: str
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    generator: dict[str, str]
    phi_state: PHIState
    #: ⚠️ Obligatorio cuando `phi_state` no es `identified` (B-3): declarar un estado sin
    #: decir que medidas lo produjeron es una afirmacion que nadie puede comprobar.
    deidentification: Deidentification | None = None
    #: Para que se emitio ESTE contenedor (B-4). Tiene que estar dentro de
    #: `subject.consent.scope`: no se puede emitir para algo que el paciente no consintio.
    purpose_of_use: PurposeOfUse | None = None
    subject: Subject
    canonical_frame: Frame
    frames: list[Frame] = Field(default_factory=list)
    visits: list[Visit] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    registrations: list[Registration] = Field(default_factory=list)
    #: D-9 · como se registro la relacion entre arcadas. `None` es no declarado y el
    #: validador avisa: en un caso de dos arcadas es la registracion que mas importa.
    occlusion: OcclusionRecord | None = None
    fhir_map: dict[str, FHIRResource] = Field(default_factory=dict)
    # Extensiones del formato. Ver `Extension` — es propuesta nuestra, no v0.2.
    extensions: dict[str, Extension] = Field(default_factory=dict)
    extensions_used: list[str] = Field(default_factory=list)
    # Vacio a proposito en todo lo que emitimos: nada de lo nuestro impide abrir el caso.
    extensions_required: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=lambda: Provenance())

    def json_canonico(self) -> str:
        """JSON estable: mismas claves, mismo orden, misma salida.

        Hace falta para la cadena de hashes: si el mismo manifiesto se serializara distinto
        entre ejecuciones, `prev_manifest_sha256` cambiaria sin que cambiara nada.
        """
        return self.model_dump_json(indent=1, exclude_none=False)
