"""`cbct-agent` — serie DICOM (CBCT) → soporte **volumétrico** (densidad σ).

Modalidad `cbct`, soporte `VOLUMETRIC`. Worker **determinista**: DICOM es un
estándar con esquema: parsearlo es código tipado, no razonamiento.

**Alcance honesto.** Este agente es *ingesta*: envuelve la reconstrucción,
**no** reimplementa el algoritmo residual de RGS (Lin et al., arXiv:2604.27552).
Lo que produce es un **campo gaussiano semilla**: cada vóxel con atenuación
relevante se convierte en una gaussiana isótropa cuyo σ sale de la atenuación
normalizada. Es exactamente la inicialización que un optimizador RGS refinaría
después — y ya es un `gaussian_field_ref` válido del contrato, así que el twin
sintético de la Semana 4 se puede montar sin esperar al motor.

**Anonimización.** El DICOM viene cargado de identificadores directos (nombre,
fecha de nacimiento, ID de paciente). El agente **nunca** los propaga al
contrato: extrae solo geometría e intensidades, y del identificador de paciente
deriva un **seudónimo** (HMAC truncado). Ver `pseudonymize`.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core_schemas import Modality, Support

from ingestion_agents.base import BaseIngestionAgent, IngestionOutput
from ingestion_agents.store import ArtifactStore

# Aire ≈ -1000 HU, esmalte ≳ 2000 HU. Por debajo del umbral no hay tejido que
# modelar: sembrar gaussianas ahí solo añade millones de primitivas invisibles.
DEFAULT_HU_THRESHOLD = 300.0

# --- recorte a la region dental --------------------------------------------- #
#
# **Por que existe.** El campo se siembra muestreando UNIFORMEMENTE todo lo que pasa de
# 300 HU, y un CBCT dental de FOV completo es una CABEZA: craneo, mandibula y cervicales.
# Los 500.000 puntos se reparten por todo ese hueso y a los dientes les tocan las migajas.
# Medido sobre un caso real: de 493.932 gaussianas solo 28.652 caian en zona dental, y el
# compuesto acababa con **el 7 % del volumen de cada diente** — filamentos, no dientes.
#
# El recorte no adivina donde esta la dentadura: la LOCALIZA por el esmalte, que es lo mas
# denso de la cabeza, y se queda con la masa contigua mayor.
#
# ⚠️ El umbral se eligio midiendo, y los dos extremos fallan de formas distintas:
#
#     HU >= 1500 -> la masa mayor mide 112 x 97 x 86 mm: es el craneo, no la dentadura
#     HU >= 1800 -> 61 x 60 x 42 mm  ← las dos arcadas en oclusion
#     HU >= 2600 -> 10 x 20 x 13 mm: un EMPASTE. El volumen llega a 13.626 HU y por
#                   encima de 2600 el metal domina al esmalte
HU_ESMALTE = 1800.0

# Rejilla con la que se agrupa. A 5 mm la dentadura es una sola masa contigua y el ruido
# disperso no llega a formar grupo.
PASO_AGRUPACION_MM = 5.0

# Margen alrededor del esmalte. La raiz NO es esmalte y queda fuera de la masa localizada:
# un diente entero mide 20-25 mm y la corona ocupa los 8-9 superiores, asi que hacen falta
# ~12 mm para que el apice entre en la caja.
MARGEN_RAIZ_MM = 12.0
# HU a partir del cual σ satura a 1.0 (esmalte/metal). Normaliza el rango útil.
HU_SATURATION = 2000.0

# Modalidades DICOM cuyos valores SÍ son unidades Hounsfield. El estándar no tiene
# un código propio para CBCT —los equipos dentales lo emiten como `CT`—, pero algún
# fabricante escribe `CBCT`, así que se aceptan las dos. Todo lo demás (MR, PT, US,
# DX) usa otra escala y no se puede leer como densidad.
ACCEPTED_MODALITIES = frozenset({"CT", "CBCT"})

_PSEUDONYM_SALT_ENV = "ASH_PSEUDONYM_SALT"

# Lo que una exportación clínica deja junto a los cortes y no pretende ser uno:
# informes, miniaturas, visores. Todo lo demás se trata como candidato a corte.
_EXTENSIONES_ACCESORIAS = frozenset(
    {".txt", ".xml", ".json", ".html", ".htm", ".csv", ".log", ".ini", ".md",
     ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".pdf", ".exe", ".bat", ".dll"}
)


def pseudonymize(patient_id: str, *, salt: str | None = None) -> str:
    """Seudónimo estable y no reversible de un identificador de paciente.

    HMAC-SHA256 con sal secreta, truncado a 16 hex. Estable (el mismo paciente da
    el mismo seudónimo entre adquisiciones, que es lo que permite la serie
    temporal del `PatientDigitalTwin`) y no reversible sin la sal (RGPD: es
    seudonimización, no anonimización — la sal es el dato a proteger).

    La sal se lee de ``ASH_PSEUDONYM_SALT``. Si no está definida se usa una de
    desarrollo: sirve para datos sintéticos, **nunca** para datos de paciente.
    """
    key = (salt or os.environ.get(_PSEUDONYM_SALT_ENV, "dev-salt-no-usar-en-produccion")).encode()
    return hmac.new(key, patient_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _es_dicom(path: Path) -> bool:
    """¿Es este fichero un DICOM? Se mira la **firma**, no el nombre.

    El estándar reserva un preámbulo de 128 bytes seguido de la marca `DICM`, y esa
    es la única forma fiable de reconocerlo. Filtrar por la extensión `.dcm` deja
    fuera series enteras: los exportadores clínicos escriben a menudo los cortes sin
    extensión (`i0000567`, `IM_0001`), y con 578 ficheros así el agente veía un
    directorio vacío y declaraba un fallo que no era del dato sino nuestro.

    Encontrado sobre CBCT real (Carestream CS 9600) — ver `edge_cases.py`.
    """
    try:
        with path.open("rb") as fh:
            return fh.read(132)[128:] == b"DICM"
    except OSError:
        return False


@dataclass(frozen=True)
class Serie:
    """Una serie DICOM ya leída y ordenada."""

    volume: np.ndarray
    """`(n_cortes, filas, columnas)` en unidades Hounsfield."""
    spacing: tuple[float, float, float]
    """Tamaño de vóxel `(x, y, z)` en mm. El de z es el **nominal**."""
    patient_id: str
    z: np.ndarray
    """Posición real de cada corte en mm. **No** se asume `índice × espaciado`: un
    corte ausente desplazaría todo lo que hay por encima."""
    huecos: int
    """Cortes que faltan en la serie, deducidos del espaciado."""
    z_es_superior: bool
    """Si `z` sale de `ImagePositionPatient[2]`, y por tanto **crece hacia craneal**.

    `IPP` viene ya en el sistema del paciente (LPS: +x izquierda, +y posterior, +z
    craneal), así que el equipo resuelve la orientación al escribirlo y `PatientPosition`
    no la cambia. Con esto `z` alta es **maxilar** y `z` baja **mandíbula**, sin adivinar.

    Es `False` cuando la serie no trae `IPP` y hubo que ordenar por `InstanceNumber`: ahí
    el sentido del eje es **desconocido** y quien separe arcadas por altura no puede
    nombrarlas. Se expone precisamente para que no se dé por supuesto lo que no consta.
    """


def _caja_dental(
    volumen: np.ndarray, spacing, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """`(min, max)` en mm de la caja que contiene la dentadura, o `None` si no la localiza.

    Se localiza por el **esmalte**, que es lo mas denso de una cabeza, y quedandose con la
    masa contigua mayor. Es lo que distingue la dentadura de un empaste suelto: los dos
    superan el umbral, pero solo una forma un bloque de 60 mm.

    Devuelve `None` en vez de una caja mala si no encuentra nada agrupable — un CBCT sin
    dientes, o con un umbral que no le vale. Quien llama sigue con el volumen entero, que
    es el comportamiento de siempre.
    """
    from scipy import ndimage

    o = np.argwhere(volumen >= HU_ESMALTE)
    if len(o) < 1000:
        return None
    p = np.column_stack([o[:, 2] * spacing[0], o[:, 1] * spacing[1], z[o[:, 0]]])

    g = np.floor((p - p.min(axis=0)) / PASO_AGRUPACION_MM).astype(int)
    masa = np.zeros(g.max(axis=0) + 1, dtype=np.int32)
    np.add.at(masa, (g[:, 0], g[:, 1], g[:, 2]), 1)
    # `>= 3` para que una celda con dos voxeles sueltos no encadene dos masas distintas.
    etiquetas, n = ndimage.label(masa >= 3)
    if n == 0:
        return None

    tam = ndimage.sum(masa, etiquetas, range(1, n + 1))
    mayor = int(np.argmax(tam)) + 1
    dentro = etiquetas[g[:, 0], g[:, 1], g[:, 2]] == mayor
    q = p[dentro]
    return q.min(axis=0) - MARGEN_RAIZ_MM, q.max(axis=0) + MARGEN_RAIZ_MM


def _read_series(directory: Path) -> Serie:
    """Lee una serie DICOM, la ordena y comprueba que no tenga agujeros."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - entorno sin la dependencia
        raise RuntimeError(
            "El `cbct-agent` necesita `pydicom` (dependencia de `ingestion-agents`)."
        ) from exc

    # Todo lo que no sea claramente accesorio se considera candidato a corte, y un
    # candidato que no supere la firma es un fallo, no algo que saltarse. El defecto
    # por omisión es **sospechar**: si se ignorase en silencio, un corte truncado
    # desaparecería de la serie y el volumen saldría con medio maxilar de menos.
    candidatos = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() not in _EXTENSIONES_ACCESORIAS
        and p.name.upper() != "DICOMDIR"
    ]
    files = [p for p in candidatos if _es_dicom(p)]
    ilegibles = [p.name for p in candidatos if p not in set(files)]
    if ilegibles:
        raise ValueError(
            f"{len(ilegibles)} fichero(s) del directorio no son DICOM legible: "
            f"{', '.join(ilegibles[:5])}{'…' if len(ilegibles) > 5 else ''}. "
            "Un corte que no se puede leer no se ignora: dejaría un agujero en el volumen."
        )
    if not files:
        raise ValueError(
            f"No hay ficheros DICOM en {directory}: ninguno lleva la firma `DICM` "
            "en el byte 128 (se comprueba el contenido, no la extensión)."
        )

    slices = [pydicom.dcmread(str(p)) for p in files]
    # Orden por posición física si está; si no, por número de instancia. Un orden
    # equivocado deforma el volumen en silencio, así que no se deja al azar.
    def _z(ds: object) -> float:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0))

    # Que la altura salga de `IPP` o de `InstanceNumber` cambia lo que se puede afirmar:
    # solo en el primer caso el eje tiene sentido anatómico. Se registra, no se supone.
    z_es_superior = all(
        getattr(ds, "ImagePositionPatient", None) is not None for ds in slices
    )

    slices.sort(key=_z)

    # Cuántos cortes faltan, deducido del espaciado. No es fatal —una exportación
    # clínica real puede perder alguno— pero sí tiene que constar: baja la confianza
    # y el orquestador decide. Lo que no se hace es fingir que la serie está
    # completa, que es lo que pasaría si nadie mirase.
    posiciones = np.asarray([_z(ds) for ds in slices], dtype=np.float64)
    huecos = 0
    if len(posiciones) > 2:
        pasos = np.diff(posiciones)
        paso = float(np.median(pasos))
        if paso > 0:
            anchos = pasos[pasos > paso * 1.5]
            huecos = int(round(float((anchos / paso).sum()) - len(anchos)))

    first = slices[0]
    rows, cols = int(first.Rows), int(first.Columns)
    volume = np.empty((len(slices), rows, cols), dtype=np.float32)
    for i, ds in enumerate(slices):
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise ValueError("Cortes de tamaño heterogéneo: la serie no es un volumen único.")
        try:
            arr = ds.pixel_array.astype(np.float32)
        except RuntimeError as exc:
            # `pydicom` no descomprime solo: necesita un plugin por sintaxis de
            # transferencia. El mensaje suyo habla de plugins genéricos; aquí se
            # nombra la sintaxis concreta, que es lo que dice qué instalar.
            sintaxis = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
            raise ValueError(
                f"Píxeles comprimidos que no se pueden descomprimir "
                f"({getattr(sintaxis, 'name', sintaxis)}): falta el decodificador. "
                f"JPEG → `pylibjpeg-libjpeg`; JPEG 2000 → `pylibjpeg-openjpeg`."
            ) from exc
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume[i] = arr * slope + intercept  # → unidades Hounsfield

    # La modalidad no es un adorno: este agente interpreta los valores como
    # unidades Hounsfield. Una resonancia con extensión `.dcm` se leería igual de
    # bien y produciría densidades plausibles y falsas — el fallo caro, el que
    # nadie mira dos veces porque el resultado *parece* un CBCT.
    modality = str(getattr(first, "Modality", "") or "").upper()
    if modality and modality not in ACCEPTED_MODALITIES:
        raise ValueError(
            f"Modalidad DICOM '{modality}': este agente solo ingiere "
            f"{'/'.join(sorted(ACCEPTED_MODALITIES))}. Sus valores no son unidades Hounsfield."
        )

    px = getattr(first, "PixelSpacing", [1.0, 1.0])
    dz = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    if len(slices) > 1:
        measured = abs(_z(slices[1]) - _z(slices[0]))
        if measured > 0:
            dz = measured  # el espaciado real manda sobre el declarado
    spacing = (float(px[1]), float(px[0]), dz)  # (x, y, z) en mm

    # Un vóxel de tamaño cero no tiene coordenadas en mm (el campo saldría
    # colapsado en un punto) y uno negativo **espeja el volumen**: intercambia
    # izquierda y derecha del paciente sin avisar. Ninguno de los dos puede pasar
    # por bueno en algo que va a acabar guiando una decisión clínica.
    if not all(np.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(
            f"Espaciado inválido {spacing} mm: cada eje tiene que ser finito y positivo."
        )

    return Serie(
        volume=volume,
        spacing=spacing,
        patient_id=str(getattr(first, "PatientID", "") or ""),
        z=posiciones,
        huecos=huecos,
        z_es_superior=z_es_superior,
    )


class CBCTAgent(BaseIngestionAgent):
    """Ingiere una serie DICOM y siembra el campo gaussiano de densidad."""

    name = "cbct-agent"
    version = "0.1.0"
    modality = Modality.CBCT
    support = Support.VOLUMETRIC

    def __init__(
        self,
        store: ArtifactStore,
        *,
        hu_threshold: float = DEFAULT_HU_THRESHOLD,
        max_primitives: int = 500_000,
        recorte_dental: bool = False,
        quarantine_dir: str | Path | None = None,
    ) -> None:
        super().__init__(quarantine_dir=quarantine_dir)
        self.store = store
        self.hu_threshold = hu_threshold
        self.max_primitives = max_primitives
        # Opt-in, y a proposito: no todo CBCT que entre aqui es de FOV completo, y recortar
        # uno que ya venga acotado a la dentadura solo quitaria margen sin ganar nada.
        # Quien sabe que su serie es de cabeza entera lo pide.
        self.recorte_dental = recorte_dental
        self.patient_pseudonym: str | None = None

    def _ingest(self, source: Path) -> IngestionOutput:
        if not source.is_dir():
            raise ValueError(
                "`cbct-agent` ingiere el **directorio** de una serie DICOM, no un corte suelto."
            )

        serie = _read_series(source)
        volume, spacing = serie.volume, serie.spacing
        self.patient_pseudonym = pseudonymize(serie.patient_id) if serie.patient_id else None

        occupied = np.argwhere(volume >= self.hu_threshold)  # (M, 3) índices (z, y, x)
        if occupied.size == 0:
            raise ValueError(
                f"Ningún vóxel supera el umbral de {self.hu_threshold} HU: "
                "serie vacía, mal reescalada o umbral inadecuado."
            )

        recorte = None
        if self.recorte_dental:
            caja = _caja_dental(volume, spacing, serie.z)
            if caja is not None:
                lo, hi = caja
                mundo_todo = np.column_stack([
                    occupied[:, 2] * spacing[0],
                    occupied[:, 1] * spacing[1],
                    serie.z[occupied[:, 0]],
                ])
                dentro = np.all((mundo_todo >= lo) & (mundo_todo <= hi), axis=1)
                if dentro.sum() >= 1000:
                    recorte = (int(occupied.shape[0]), int(dentro.sum()), hi - lo)
                    occupied = occupied[dentro]

        # Submuestreo determinista si el volumen da más primitivas de las pedidas:
        # paso uniforme (no aleatorio) para que la ingesta sea reproducible.
        confidence = 1.0
        if occupied.shape[0] > self.max_primitives:
            step = int(np.ceil(occupied.shape[0] / self.max_primitives))
            occupied = occupied[::step]
            confidence = 0.9  # el campo es una submuestra, no el volumen completo

        # Una serie con cortes ausentes se ingiere, pero no se da por completa: la
        # confianza cae y el gate de human-in-the-loop del orquestador la ve.
        if serie.huecos:
            confidence = min(confidence, 0.8)

        sx, sy, _ = spacing
        # La z sale de la **posición real** de cada corte, no de `índice × espaciado`:
        # con un corte ausente, multiplicar por el índice desplazaría todo lo que hay
        # por encima del hueco. Es un error silencioso de 0,3 mm por corte perdido.
        mundo = np.column_stack(
            [
                occupied[:, 2] * sx,
                occupied[:, 1] * sy,
                serie.z[occupied[:, 0]],
            ]
        ).astype(np.float64)
        # Centrado en el origen, como el 3DGS estándar. El desplazamiento se GUARDA:
        # restarlo y olvidarlo hace el campo irreversible, porque depende del dato y no
        # se puede recomputar desde el fichero exportado. Con `origin` a mano,
        # `mundo = centers + origin` devuelve las coordenadas reales del CBCT.
        origin = mundo.mean(axis=0)
        centers = (mundo - origin).astype(np.float32)

        hu = volume[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        density = np.clip(
            (hu - self.hu_threshold) / (HU_SATURATION - self.hu_threshold), 0.0, 1.0
        ).astype(np.float32)

        # Gaussianas isótropas del tamaño del vóxel (½ arista): la semilla no
        # inventa anisotropía que el CBCT no midió; eso lo aprende el optimizador.
        scales = np.tile(
            np.asarray(spacing, dtype=np.float32) * 0.5, (centers.shape[0], 1)
        )
        # Cuaternión identidad (w, x, y, z) — sin rotación en la semilla.
        rotations = np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (centers.shape[0], 1)
        )

        return self._success(
            source,
            confidence=confidence,
            artifact_ref=self.store.put(
                centers=centers,
                scales=scales,
                rotations=rotations,
                density=density,
                # Lo que hace falta para deshacer las dos normalizaciones. Va en el
                # artefacto y no en `Provenance` por dos razones: `Provenance.transform`
                # ya significa otra cosa (el alineamiento de la fusión geométrica, ADR
                # 004) y reutilizarlo colisionaría; y el esquema declara el snapshot
                # **autocontenido**, así que lo que hace reversible un blob viaja con él.
                origin=origin,
                hu_range=np.asarray([self.hu_threshold, HU_SATURATION], dtype=np.float64),
            ),
            n_primitives=int(centers.shape[0]),
            detail=(
                None if recorte is None else
                f"recorte dental: {recorte[1]:,} de {recorte[0]:,} vóxeles de tejido duro "
                f"({recorte[1] / recorte[0]:.1%}) en una caja de "
                f"{recorte[2][0]:.0f}×{recorte[2][1]:.0f}×{recorte[2][2]:.0f} mm. El resto "
                "es cráneo y cervicales: sin recortar, el muestreo uniforme deja los "
                "dientes con el 7% de su volumen."
            ),
        )
