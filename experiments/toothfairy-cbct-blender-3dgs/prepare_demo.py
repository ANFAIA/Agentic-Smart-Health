"""DICOM → volumen numpy para el experimento de capas de la demo.

    uv run python prepare_demo.py

Lee la serie DICOM de la demo, la ordena por posición física, apila en HU y la
baja a 0,30 mm (paso 2×) para seguir el protocolo del doc (vóxel 0,30 mm) y no
agotar la GPU. Escribe `volume.npy` (float32, orden z,y,x) y `meta.json` con el
espaciado efectivo. Sin etiquetas: la máscara de "dónde acaba el paciente" se
sintetiza en el orquestador como `HU > 300`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SERIE = Path("/home/lgarbayo/anfaia/histora/another_patient/CASE-6997CD_files")
PAQUETE = Path("/home/lgarbayo/agentic-smart-health/data/processed/demo-capas")
FACTOR = 2  # 0,15 → 0,30 mm


def main() -> int:
    import pydicom

    slices = sorted(
        (pydicom.dcmread(str(p)) for p in SERIE.glob("*.dcm")),
        key=lambda ds: float(getattr(ds, "ImagePositionPatient", [0, 0, 0])[2]),
    )
    first = slices[0]
    rows, cols = int(first.Rows), int(first.Columns)
    vol = np.empty((len(slices), rows, cols), dtype=np.float32)
    for i, ds in enumerate(slices):
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        vol[i] = ds.pixel_array.astype(np.float32) * slope + intercept

    sx = float(first.PixelSpacing[0]) * FACTOR
    sy = float(first.PixelSpacing[1]) * FACTOR
    sz = float(first.SliceThickness) * FACTOR

    vol = vol[::FACTOR, ::FACTOR, ::FACTOR]
    print(f"volumen {vol.shape} · spacing {sx:.3f}/{sy:.3f}/{sz:.3f} mm · "
          f"HU [{vol.min():.0f}, {vol.max():.0f}]")

    PAQUETE.mkdir(parents=True, exist_ok=True)
    np.save(PAQUETE / "volume.npy", vol)
    (PAQUETE / "meta.json").write_text(json.dumps({
        "shape": list(vol.shape), "spacing": [sx, sy, sz], "world_offset_mm": [0.0, 0.0, 0.0],
    }), encoding="utf-8")
    print(f"escrito {PAQUETE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
