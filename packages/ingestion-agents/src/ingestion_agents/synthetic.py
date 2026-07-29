"""Generador de casos **sintéticos** coherentes (malla + CBCT + informe).

El brief fija «datos sintéticos primero»: el Digital Twin de la Semana 4 se monta
sin datos de paciente. Este módulo genera un caso completo donde las **tres
modalidades describen la misma boca**, que es lo que hace falta para probar la
ingesta de punta a punta (y lo que un dataset público suelto no da: nadie publica
CBCT + malla + informe del mismo paciente).

Modelo: una arcada parabólica de 16 dientes (cuadrantes 1 y 2), cada diente un
elipsoide cuyo tamaño depende de su tipo FDI. La **misma** geometría se
materializa de dos formas — malla de superficie (OBJ) y volumen de atenuación
(serie DICOM) — y el informe cuelga un pH de algunos de esos dientes.

No pretende ser anatómicamente realista: pretende ser **coherente entre
modalidades y reproducible** (semilla fija), que es lo que la ingesta necesita
validar.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ingestion_agents import ontology

# Semieje (mm) del elipsoide según tipo de diente. Un molar es más ancho y más
# corto que un incisivo; basta con eso para que las modalidades se distingan.
_RADII_BY_TYPE = {
    ontology.ToothType.INCISOR: (2.6, 1.8, 5.0),
    ontology.ToothType.CANINE: (2.8, 2.6, 6.0),
    ontology.ToothType.PREMOLAR: (3.4, 3.2, 4.6),
    ontology.ToothType.MOLAR: (4.6, 4.4, 4.2),
}
# Atenuación aproximada (HU): esmalte muy denso, encía casi tejido blando.
_HU_ENAMEL = 2200.0
_HU_DENTIN = 1400.0
_HU_GUM = 120.0
_HU_AIR = -1000.0

_TOOTH_COLOR = (0.94, 0.92, 0.86)  # marfil
_GUM_COLOR = (0.78, 0.45, 0.45)    # encía


def upper_arch_codes() -> list[str]:
    """Los 16 dientes permanentes de la arcada superior, de derecha a izquierda.

    Cuadrante 1 (derecha del paciente) va del 18 al 11; el 2 (izquierda) del 21
    al 28. Recorrer la arcada en orden anatómico y no numérico es lo que hace que
    el diente 11 y el 21 queden juntos en el centro, como en una boca real.
    """
    return [f"1{p}" for p in range(8, 0, -1)] + [f"2{p}" for p in range(1, 9)]


def _tooth_centers(codes: list[str], *, width: float = 30.0, depth: float = 22.0) -> np.ndarray:
    """Centros de los dientes sobre una parábola (el arco dental)."""
    t = np.linspace(-1.0, 1.0, len(codes))
    x = t * width
    y = depth * (1.0 - t**2)
    z = np.zeros_like(t)
    return np.column_stack([x, y, z]).astype(np.float64)


def _ellipsoid(
    center: np.ndarray, radii: tuple[float, float, float], n_theta: int = 16, n_phi: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Malla UV de un elipsoide: (posiciones, caras triangulares 0-indexadas)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, np.pi, n_phi)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    pos = np.column_stack(
        [
            center[0] + radii[0] * np.sin(pp).ravel() * np.cos(tt).ravel(),
            center[1] + radii[1] * np.sin(pp).ravel() * np.sin(tt).ravel(),
            center[2] + radii[2] * np.cos(pp).ravel(),
        ]
    )
    faces: list[tuple[int, int, int]] = []
    for i in range(n_theta):
        i_next = (i + 1) % n_theta  # cierra el anillo en θ
        for j in range(n_phi - 1):
            a = i * n_phi + j
            b = i_next * n_phi + j
            faces.append((a, b, a + 1))
            faces.append((b, b + 1, a + 1))
    return pos, np.asarray(faces, dtype=np.int32)


def _arch_parts(codes: list[str]) -> list[tuple[np.ndarray, tuple[float, float, float], bool]]:
    """(centro, semiejes, es_diente) de cada primitiva de la arcada, dientes + encía."""
    centers = _tooth_centers(codes)
    parts: list[tuple[np.ndarray, tuple[float, float, float], bool]] = []
    for code, center in zip(codes, centers, strict=True):
        parts.append((center, _RADII_BY_TYPE[ontology.describe(code).tooth_type], True))
        # Reborde de encía: mismo sitio, más ancho y desplazado hacia apical.
        gum_center = center + np.asarray([0.0, 0.0, -4.5])
        parts.append((gum_center, (4.2, 4.0, 3.0), False))
    return parts


# --------------------------------------------------------------------------- #
# Malla intraoral (OBJ)
# --------------------------------------------------------------------------- #
def write_mesh_obj(path: Path, codes: list[str] | None = None) -> Path:
    """Escribe la arcada como OBJ con color por vértice (formato del escáner)."""
    codes = codes or upper_arch_codes()
    all_pos: list[np.ndarray] = []
    all_col: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0
    for center, radii, is_tooth in _arch_parts(codes):
        pos, faces = _ellipsoid(center, radii)
        all_pos.append(pos)
        colour = _TOOTH_COLOR if is_tooth else _GUM_COLOR
        all_col.append(np.tile(np.asarray(colour), (pos.shape[0], 1)))
        all_faces.append(faces + offset)
        offset += pos.shape[0]

    positions = np.vstack(all_pos)
    colors = np.vstack(all_col)
    faces = np.vstack(all_faces)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Caso sintético — Agentic Smart Health (ingestion_agents.synthetic)\n")
        for (x, y, z), (r, g, b) in zip(positions, colors, strict=True):
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for a, b, c in faces + 1:  # el OBJ indexa desde 1
            fh.write(f"f {a} {b} {c}\n")
    return path


# --------------------------------------------------------------------------- #
# CBCT (serie DICOM)
# --------------------------------------------------------------------------- #
def build_volume(
    codes: list[str] | None = None, *, spacing: float = 0.6, margin: float = 6.0
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Voxeliza la **misma** arcada en un volumen de HU. Devuelve (volumen[z,y,x], espaciado)."""
    codes = codes or upper_arch_codes()
    parts = _arch_parts(codes)

    lo = np.min([c - np.asarray(r) for c, r, _ in parts], axis=0) - margin
    hi = np.max([c + np.asarray(r) for c, r, _ in parts], axis=0) + margin
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1

    grid = [lo[i] + np.arange(dims[i]) * spacing for i in range(3)]
    gx, gy, gz = np.meshgrid(*grid, indexing="ij")
    volume = np.full(gx.shape, _HU_AIR, dtype=np.float32)

    for center, radii, is_tooth in parts:
        # Distancia elipsoidal normalizada: <=1 dentro de la primitiva.
        d2 = (
            ((gx - center[0]) / radii[0]) ** 2
            + ((gy - center[1]) / radii[1]) ** 2
            + ((gz - center[2]) / radii[2]) ** 2
        )
        inside = d2 <= 1.0
        if is_tooth:
            # Esmalte en la cáscara, dentina en el núcleo: el gradiente radial es
            # lo que hace que el volumen tenga estructura interna que la malla no
            # puede tener (justo el aporte propio del CBCT).
            volume[inside] = np.where(
                d2[inside] > 0.55, _HU_ENAMEL, _HU_DENTIN
            ).astype(np.float32)
        else:
            volume[inside] = np.maximum(volume[inside], _HU_GUM)

    # (x, y, z) → (z, y, x), que es el orden corte-fila-columna de una serie DICOM.
    return np.transpose(volume, (2, 1, 0)), (spacing, spacing, spacing)


def write_dicom_series(
    out_dir: Path,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    patient_id: str = "SYNTH-0001",
    study_date: datetime | None = None,
) -> Path:
    """Escribe el volumen como serie DICOM CT (un fichero por corte)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    out_dir.mkdir(parents=True, exist_ok=True)
    study_date = study_date or datetime.now(UTC)
    study_uid, series_uid = generate_uid(), generate_uid()
    intercept = -1024.0  # desplaza los HU al rango sin signo habitual del CT

    for k in range(volume.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.preamble = b"\0" * 128
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"

        # Identificadores: sintéticos por construcción. Aun así el `cbct-agent`
        # los seudonimiza al ingerir, para que el camino sea el mismo el día que
        # entre un DICOM real.
        ds.PatientID = patient_id
        ds.PatientName = "SINTETICO^CASO"
        ds.StudyDate = study_date.strftime("%Y%m%d")

        ds.Rows, ds.Columns = int(volume.shape[1]), int(volume.shape[2])
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # [fila, columna]
        ds.SliceThickness = float(spacing[2])
        ds.ImagePositionPatient = [0.0, 0.0, float(k * spacing[2])]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.InstanceNumber = k + 1

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # sin signo
        ds.RescaleIntercept = intercept
        ds.RescaleSlope = 1.0
        stored = np.clip(volume[k] - intercept, 0, 65535).astype(np.uint16)
        ds.PixelData = stored.tobytes()

        pydicom.dcmwrite(out_dir / f"slice_{k:04d}.dcm", ds, enforce_file_format=True)

    return out_dir


# --------------------------------------------------------------------------- #
# Informe clínico
# --------------------------------------------------------------------------- #
def write_report(
    path: Path, ph_by_fdi: dict[str, float] | None = None, *, date: datetime | None = None
) -> Path:
    """Escribe un informe clínico sintético en texto (mismo formato que un PDF extraído)."""
    date = date or datetime.now(UTC)
    ph_by_fdi = ph_by_fdi or {"16": 5.2, "21": 6.8, "26": 5.9, "11": 7.1}

    lines = [
        "INFORME ODONTOLÓGICO (CASO SINTÉTICO)",
        "=====================================",
        f"Fecha: {date.strftime('%Y-%m-%d')}",
        "Paciente: caso sintético generado para validación de la ingesta.",
        "",
        "Hallazgos por diente (numeración ISO-FDI):",
    ]
    for code, value in sorted(ph_by_fdi.items()):
        info = ontology.describe(code)
        riesgo = "riesgo de desmineralización" if value < 5.5 else "dentro de rango"
        lines.append(
            f"  - Diente {code} ({info.tooth_type.value} {info.arch} {info.side}): "
            f"pH {value:.1f} — {riesgo}."
        )
    lines += ["", "Sin hallazgos radiológicos adicionales.", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Caso completo
# --------------------------------------------------------------------------- #
def write_case(
    root: Path,
    *,
    patient_id: str = "SYNTH-0001",
    with_dicom: bool = True,
    spacing: float = 0.6,
) -> dict[str, Path]:
    """Genera el caso completo y devuelve `{modalidad: ruta}` listo para la ingesta."""
    root = Path(root)
    codes = upper_arch_codes()
    paths = {
        "mesh": write_mesh_obj(root / "scan_upper.obj", codes),
        "report": write_report(root / "informe.txt"),
    }
    if with_dicom:
        volume, sp = build_volume(codes, spacing=spacing)
        paths["cbct"] = write_dicom_series(root / "cbct", volume, sp, patient_id=patient_id)
    return paths
