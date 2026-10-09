#!/usr/bin/env python
"""Audita fuentes y compara dos ejecuciones; no acredita calibración clínica.

Los informes locales pueden contener metadatos del caso: guardar bajo data/processed.
La repetibilidad comprobada se limita a los payloads binarios y al registro geométrico
de dos ejecuciones con las mismas fuentes. No compara identidades de contenedor,
timestamps ni entrenamiento nuevo de apariencia cuando se reutiliza un archivo.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

RESUMEN_EN = "Audits CBCT source hashes and compares independent complete-case executions."
RAIZ = Path(__file__).resolve().parent.parent
for src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(src))

from uos import read_fidelity, read_manifest, validate  # noqa: E402
from uos.contenedor import digesto_de_partes, partes_de, sha256  # noqa: E402


def audita_cabeceras(cbct: Path) -> dict:
    """Cuenta identidades y examina muestreo sin publicar identificadores del paciente."""
    import pydicom

    headers = [pydicom.dcmread(p, stop_before_pixels=True)
               for p in sorted(cbct.glob("*.dcm"))]
    if not headers:
        raise ValueError("No hay cortes .dcm que auditar")
    counts = {k: len({str(getattr(ds, k, "") or "") for ds in headers} - {""})
              for k in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID")}
    groups: dict[str, list] = {}
    for ds in headers:
        groups.setdefault(str(getattr(ds, "SeriesInstanceUID", "") or ""), []).append(ds)
    series = []
    for datasets in groups.values():
        positions = [getattr(ds, "ImagePositionPatient", None) for ds in datasets]
        spacing = None
        gaps = None
        if all(p is not None for p in positions) and len(positions) > 1:
            z = np.sort([float(p[2]) for p in positions])
            delta = np.diff(z)
            spacing = [float(delta.min()), float(delta.max())]
            step = float(np.median(delta))
            if step > 0:
                gaps = int(np.maximum(np.rint(delta / step).astype(int) - 1, 0).sum())
        series.append({"instances": len(datasets), "slice_spacing_min_max_mm": spacing,
                       "estimated_missing_slices": gaps})
    tags = ("PixelSpacing", "SliceThickness", "ImageOrientationPatient", "RescaleSlope",
            "RescaleIntercept", "RescaleType", "PixelSpacingCalibrationType",
            "PixelSpacingCalibrationDescription", "DateOfLastCalibration")
    return {
        "instances": len(headers), "identity_counts": counts,
        "single_identity": all(n == 1 for n in counts.values()), "series": series,
        "declarations": {k: sorted({str(getattr(ds, k, "") or "") for ds in headers})
                         for k in tags},
        "scope": "Header declarations and estimated missing slices; no independent calibration",
    }


def audita_fuente(uos: Path, cbct: Path) -> dict:
    """Comprueba todos los ficheros de la fuente contra la referencia exportada."""
    manifest = read_manifest(uos)
    report = validate(uos)
    ct = next((a for a in manifest.assets if a.id == "asset.ct_001"), None)
    if ct is None:
        raise ValueError("No se ha exportado la referencia de procedencia CBCT")
    parts = partes_de(cbct)
    headers = audita_cabeceras(cbct)
    expected = {(p.name, p.sha256, p.bytes, p.pixel_data_sha256) for p in parts}
    actual = {(p.name, p.sha256, p.bytes, p.pixel_data_sha256) for p in ct.parts}
    records = read_fidelity(uos).assets
    with zipfile.ZipFile(uos) as archive:
        sidecar = json.loads(archive.read(ct.sidecar_uri))
        originals_absent = all(a.uri not in archive.namelist()
                               for a in manifest.assets if a.external)
        dicom_absent = not any(n.lower().endswith(".dcm") for n in archive.namelist())
    verified = (report.valid and headers["single_identity"]
                and ct.external and originals_absent and dicom_absent
                and expected == actual and ct.sha256 == digesto_de_partes(parts)
                and records["asset.field"].sources == [ct.id]
                and records["asset.field"].source_status == "resolved")
    return {
        "verified": verified, "valid_uos": report.valid,
        "uos_sha256": sha256(uos), "series_sha256": ct.sha256,
        "parts": len(parts), "all_source_hashes_match": expected == actual,
        "source_headers": headers,
        "originals_absent": originals_absent and dicom_absent,
        "field_sources": records["asset.field"].sources,
        "dimensions": sidecar["dimensions"], "spacing_mm": sidecar["spacing_mm"],
        "calibration": {
            "intensity": sidecar["calibration_status"],
            "calibrated_hu": sidecar["calibrated_hu"],
            "geometry": "unknown", "colour": "unknown",
            "scope": "Header sampling and rescale are not independent calibration evidence",
        },
        "warnings": len(report.warnings),
    }


def registro_de(uos: Path) -> dict | None:
    """El `run-record.json` que `caso_completo.py` deja junto a `export/`, si existe."""
    ruta = uos.parent.parent / "run-record.json"
    return json.loads(ruta.read_text()) if ruta.is_file() else None


def revision_comparada(a: Path, b: Path) -> dict:
    """Mismo código en las dos ejecuciones: HEAD y hash del diff, ambos registrados.

    Las ejecuciones anteriores al registro no lo tienen: se declaran `unrecorded` y no
    hacen fallar la comparación, porque ese dato no existía, no porque no coincida.
    """
    ra, rb = registro_de(a), registro_de(b)
    if ra is None or rb is None:
        return {"code_revision": "unrecorded", "same_code_revision": None}
    ca, cb = ra.get("code", {}), rb.get("code", {})
    claves = ("git_head", "git_diff_sha256", "untracked_code")
    same = (ca.get("git_head") is not None
            and all(ca.get(k) == cb.get(k) for k in claves))
    return {"code_revision": "recorded", "same_code_revision": same,
            "git_head": [ca.get("git_head"), cb.get("git_head")],
            "git_dirty": [ca.get("git_dirty"), cb.get("git_dirty")],
            "seed": [ra.get("seed"), rb.get("seed")]}


def compara(a: Path, b: Path) -> dict:
    """Criterio fijado: identidad exacta de binarios y transformadas, mismas fuentes.

    Los JSON se inventarían como comparación de bytes separada, sin atribuirles
    repetibilidad semántica al excluir fechas de manera genérica.
    """
    ma, mb = read_manifest(a), read_manifest(b)
    fa, fb = read_fidelity(a).assets, read_fidelity(b).assets
    aa, ab = {x.id: x for x in ma.assets}, {x.id: x for x in mb.assets}
    same_inventory = aa.keys() == ab.keys()
    same_sources = {k: x.sha256 for k, x in aa.items() if x.external} == {
        k: x.sha256 for k, x in ab.items() if x.external}
    processes_a = {k: r.process.model_dump(mode="json") for k, r in fa.items()
                   if r.process is not None}
    processes_b = {k: r.process.model_dump(mode="json") for k, r in fb.items()
                   if r.process is not None}
    same_process_records = processes_a == processes_b
    binary, metadata = {}, {}
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        for id_ in sorted(aa.keys() & ab.keys()):
            x, y = aa[id_], ab[id_]
            if x.external or y.external:
                continue
            identical = za.read(x.uri) == zb.read(y.uri)
            target = metadata if x.media_type == "application/json" else binary
            target[id_] = {"identical": identical, "bytes_a": x.bytes, "bytes_b": y.bytes}
    ra = {r.id: r.transform_4x4_row_major for r in ma.registrations}
    rb = {r.id: r.transform_4x4_row_major for r in mb.registrations}
    same_registrations = ra == rb
    displacement = None
    if ra.keys() == rb.keys() and ra:
        displacement = max(float(np.max(np.abs(np.asarray(ra[k]) - np.asarray(rb[k]))))
                           for k in ra)
    revision = revision_comparada(a, b)
    verified = (validate(a).valid and validate(b).valid and same_inventory and same_sources
                and bool(binary) and all(x["identical"] for x in binary.values())
                and same_registrations and same_process_records
                and revision["same_code_revision"] is not False)
    return {
        "verified": verified, "runs": 2,
        "criterion": "Exact binary payloads and registration matrices; identical source hashes",
        "scope": "These two runs on this environment only; excludes JSON semantic equivalence "
                 "and new appearance training",
        "same_inventory": same_inventory, "same_sources": same_sources,
        "same_process_records": same_process_records,
        "same_registrations": same_registrations,
        **revision,
        "max_registration_coefficient_difference": displacement,
        "binary_assets": binary, "metadata_bytes": metadata,
        "container_bytes_identical": sha256(a) == sha256(b),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uos", type=Path, required=True)
    parser.add_argument("--cbct", type=Path, required=True)
    parser.add_argument("--compara", type=Path)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    output = {"source_audit": audita_fuente(args.uos, args.cbct)}
    if args.compara:
        output["source_audit_second"] = audita_fuente(args.compara, args.cbct)
        output["repeatability"] = compara(args.uos, args.compara)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Informe: {args.salida}")
    return 0 if all(v["verified"] for v in output.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
