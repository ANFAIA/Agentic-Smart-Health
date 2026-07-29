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
from pathlib import Path

import numpy as np
from core_schemas import Modality, Support

from ingestion_agents.base import BaseIngestionAgent, IngestionOutput
from ingestion_agents.store import ArtifactStore

# Aire ≈ -1000 HU, esmalte ≳ 2000 HU. Por debajo del umbral no hay tejido que
# modelar: sembrar gaussianas ahí solo añade millones de primitivas invisibles.
DEFAULT_HU_THRESHOLD = 300.0
# HU a partir del cual σ satura a 1.0 (esmalte/metal). Normaliza el rango útil.
HU_SATURATION = 2000.0

_PSEUDONYM_SALT_ENV = "ASH_PSEUDONYM_SALT"


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


def _read_series(directory: Path) -> tuple[np.ndarray, tuple[float, float, float], str]:
    """Lee una serie DICOM y devuelve (volumen HU, espaciado mm (x,y,z), patient_id crudo)."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - entorno sin la dependencia
        raise RuntimeError(
            "El `cbct-agent` necesita `pydicom` (dependencia de `ingestion-agents`)."
        ) from exc

    files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".dcm")
    if not files:
        raise ValueError(f"No hay ficheros .dcm en {directory}")

    slices = [pydicom.dcmread(str(p)) for p in files]
    # Orden por posición física si está; si no, por número de instancia. Un orden
    # equivocado deforma el volumen en silencio, así que no se deja al azar.
    def _z(ds: object) -> float:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=_z)

    first = slices[0]
    rows, cols = int(first.Rows), int(first.Columns)
    volume = np.empty((len(slices), rows, cols), dtype=np.float32)
    for i, ds in enumerate(slices):
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise ValueError("Cortes de tamaño heterogéneo: la serie no es un volumen único.")
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume[i] = arr * slope + intercept  # → unidades Hounsfield

    px = getattr(first, "PixelSpacing", [1.0, 1.0])
    dz = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    if len(slices) > 1:
        measured = abs(_z(slices[1]) - _z(slices[0]))
        if measured > 0:
            dz = measured  # el espaciado real manda sobre el declarado
    spacing = (float(px[1]), float(px[0]), dz)  # (x, y, z) en mm

    return volume, spacing, str(getattr(first, "PatientID", "") or "")


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
        quarantine_dir: str | Path | None = None,
    ) -> None:
        super().__init__(quarantine_dir=quarantine_dir)
        self.store = store
        self.hu_threshold = hu_threshold
        self.max_primitives = max_primitives
        self.patient_pseudonym: str | None = None

    def _ingest(self, source: Path) -> IngestionOutput:
        if not source.is_dir():
            raise ValueError(
                "`cbct-agent` ingiere el **directorio** de una serie DICOM, no un corte suelto."
            )

        volume, spacing, raw_patient_id = _read_series(source)
        self.patient_pseudonym = pseudonymize(raw_patient_id) if raw_patient_id else None

        occupied = np.argwhere(volume >= self.hu_threshold)  # (M, 3) índices (z, y, x)
        if occupied.size == 0:
            raise ValueError(
                f"Ningún vóxel supera el umbral de {self.hu_threshold} HU: "
                "serie vacía, mal reescalada o umbral inadecuado."
            )

        # Submuestreo determinista si el volumen da más primitivas de las pedidas:
        # paso uniforme (no aleatorio) para que la ingesta sea reproducible.
        confidence = 1.0
        if occupied.shape[0] > self.max_primitives:
            step = int(np.ceil(occupied.shape[0] / self.max_primitives))
            occupied = occupied[::step]
            confidence = 0.9  # el campo es una submuestra, no el volumen completo

        sx, sy, sz = spacing
        centers = np.column_stack(
            [
                occupied[:, 2] * sx,
                occupied[:, 1] * sy,
                occupied[:, 0] * sz,
            ]
        ).astype(np.float32)
        centers -= centers.mean(axis=0)  # centrado en el origen, como el 3DGS estándar

        hu = volume[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        density = np.clip(
            (hu - self.hu_threshold) / (HU_SATURATION - self.hu_threshold), 0.0, 1.0
        ).astype(np.float32)

        # Gaussianas isótropas del tamaño del vóxel (½ arista): la semilla no
        # inventa anisotropía que el CBCT no midió; eso lo aprende el optimizador.
        scales = np.tile(
            np.asarray([sx, sy, sz], dtype=np.float32) * 0.5, (centers.shape[0], 1)
        )
        # Cuaternión identidad (w, x, y, z) — sin rotación en la semilla.
        rotations = np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (centers.shape[0], 1)
        )

        return self._success(
            source,
            confidence=confidence,
            artifact_ref=self.store.put(
                centers=centers, scales=scales, rotations=rotations, density=density
            ),
            n_primitives=int(centers.shape[0]),
        )
