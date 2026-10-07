#!/usr/bin/env python
"""Separa las series de un ZIP DICOM en copias de trabajo y audita su extracción.

No combina marcos ni modifica cabeceras. Guardar la salida en data/processed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pydicom

RESUMEN_EN = "Separates DICOM ZIP series into hashed working copies without rewriting data."


def prepara(archivo: Path, extraido: Path, destino: Path) -> dict:
    groups: dict[str, dict] = {}
    missing, changed, duplicates = [], [], []
    with ZipFile(archivo) as archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            name = info.filename
            if not name.lower().endswith(".dcm"):
                continue
            raw = archive.read(info)
            ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
            uid = str(ds.SeriesInstanceUID)
            id_ = hashlib.sha256(uid.encode()).hexdigest()[:12]
            group = groups.setdefault(id_, {"instances": [], "frames": set(),
                                           "study": set(), "sop": {}})
            base = Path(name).name
            target = destino / f"series-{id_}" / base
            if target.exists() and target.read_bytes() != raw:
                raise ValueError("Colisión de nombres con contenidos diferentes")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            local = extraido / base
            if not local.exists():
                missing.append(base)
            elif hashlib.sha256(local.read_bytes()).hexdigest() != digest:
                changed.append(base)
            sop = str(ds.SOPInstanceUID)
            if sop in group["sop"]:
                if group["sop"][sop] != digest:
                    raise ValueError("SOPInstanceUID repetido con contenidos diferentes")
                duplicates.append({"file": base, "sha256": digest})
                continue
            group["sop"][sop] = digest
            group["study"].add(str(ds.StudyInstanceUID))
            group["frames"].add(str(ds.FrameOfReferenceUID))
            group["instances"].append({"file": base, "sha256": digest,
                                       "instance": int(ds.InstanceNumber),
                                       "position_mm": [float(v) for v in ds.ImagePositionPatient]})
    if not groups:
        raise ValueError("ZIP sin instancias DICOM")
    result = {
        "archive_sha256": hashlib.sha256(archivo.read_bytes()).hexdigest(),
        "missing_from_previous_extraction": missing, "changed_in_previous_extraction": changed,
        "identical_duplicate_entries": duplicates,
        "series": {k: {"directory": str(destino / f"series-{k}"),
                       "count": len(g["instances"]), "frames": len(g["frames"]),
                       "studies": len(g["study"]), "instances": g["instances"]}
                   for k, g in groups.items()},
        "policy": "Separate original series; no UID rewriting, interpolation or frame fusion",
    }
    (destino / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", type=Path, required=True)
    parser.add_argument("--extraido", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    result = prepara(args.archivo, args.extraido, args.salida)
    print(json.dumps({"series": {k: g["count"] for k, g in result["series"].items()},
                      "missing_from_previous_extraction":
                          result["missing_from_previous_extraction"],
                      "changed_count": len(result["changed_in_previous_extraction"])}, indent=2))


if __name__ == "__main__":
    main()
