"""Write and validate fidelity, loss inheritance and processing provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from uos.contenedor import asset_de_bytes, read_manifest_from
from uos.fidelidad import (
    ASSET_ID,
    EXTENSION,
    SCHEMA,
    URI,
    AssetRecord,
    Audit,
    Claim,
    ClinicalAssessment,
    Loss,
    Metric,
    Process,
    State,
    inherit,
)
from uos.manifiesto import Asset, AssetKind, Extension, Manifest, Regulatory

if TYPE_CHECKING:
    from uos.validador import Report


def unknown_records(manifest: Manifest) -> dict[str, AssetRecord]:
    """Legacy containers carry no implicit fidelity or reproducibility guarantees."""
    return {a.id: AssetRecord(origin="unknown") for a in manifest.assets if a.id != ASSET_ID}


def read_fidelity(path: Path) -> Audit:
    with zipfile.ZipFile(path) as z:
        manifest, _ = read_manifest_from(z.read("manifest.json"))
        extension = manifest.extensions.get(EXTENSION)
        if extension is None:
            return Audit(assets=unknown_records(manifest))
        if extension.version != "1.0" or extension.schema_id != SCHEMA or extension.uri is None:
            raise ValueError("unsupported fidelity/provenance extension")
        return Audit.model_validate_json(z.read(extension.uri))


def remove_inference(source: Path, destination: Path) -> Path:
    """Create a successor without inferred payloads, preserving lineage and integrity.

    Saved views are cleared because their framing may have used inferred tooth labels.
    This operates on one container; it makes no claim about deleting older copies.
    """
    from uos.procedencia import CHAIN, encadena, lee_version_previa
    from uos.validador import validate
    from uos.version import puede_reemitir
    from uos.vistas import VIEWS

    if source.resolve() == destination.resolve():
        raise ValueError("inference removal must create a separate successor")
    report = validate(source)
    if not report.valid:
        raise ValueError(f"cannot transform an invalid container: {report.errors}")
    with zipfile.ZipFile(source) as z:
        manifest, ignored = read_manifest_from(z.read("manifest.json"))
        if ignored or not puede_reemitir(manifest.uos_version):
            raise ValueError("cannot rewrite a newer container while ignoring unknown fields")
        old_hash, chain, warning = lee_version_previa(source)
        if warning:
            raise ValueError(warning)
        retained = [a for a in manifest.assets if a.regulatory.layer != 3]
        ids = {a.id for a in retained}
        removed = [a for a in manifest.assets if a.id not in ids]
        excluded = {a.uri for a in removed} | {a.sidecar_uri for a in removed if a.sidecar_uri}
        content = {name: z.read(name) for name in z.namelist()
                   if name != "manifest.json" and name not in excluded
                   and not name.startswith("derived/")}
    manifest.assets = retained
    manifest.fhir_map = {key: item for key, item in manifest.fhir_map.items()
                         if key == "case" or key in ids}
    manifest.extensions = {key: item for key, item in manifest.extensions.items()
                           if item.uri not in excluded and key != "histora_clinical_inference"}
    manifest.extensions_used = [key for key in manifest.extensions_used
                                if key in manifest.extensions]
    manifest.extensions_required = [key for key in manifest.extensions_required
                                    if key in manifest.extensions]
    if EXTENSION in manifest.extensions:
        audit = read_fidelity(source)
        audit.assets = {key: record for key, record in audit.assets.items() if key in ids}
        inherit(audit.assets)
        raw_audit = audit.model_dump_json(by_alias=True, indent=2).encode()
        content[URI] = raw_audit
        for asset in retained:
            if asset.id == ASSET_ID:
                asset.sha256 = hashlib.sha256(raw_audit).hexdigest()
                asset.bytes = len(raw_audit)
    if VIEWS in content:
        content[VIEWS] = b'{"views": []}'
    manifest.provenance.prev_manifest_sha256 = old_hash
    manifest.provenance.chain = CHAIN
    raw_manifest = manifest.json_canonico()
    content[CHAIN] = encadena(
        case_id=manifest.case_id, manifiesto_json=raw_manifest,
        previo_sha256=old_hash, cadena_previa=chain, generator=manifest.generator,
        assets=len(retained), note="inference removed; label-dependent saved views cleared",
    ).model_dump_json().encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("manifest.json", raw_manifest)
            for name, payload in sorted(content.items()):
                z.writestr(name, payload)
        successor_report = validate(temporary)
        if not successor_report.valid:
            raise ValueError(f"invalid inference-free successor: {successor_report.errors}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def append_audit(
    assets: list[Asset], records: dict[str, AssetRecord],
    extras: dict[str, str | bytes], extensions: dict[str, Extension],
) -> None:
    inherit(records)
    audit = Audit(assets=records)
    crudo = audit.model_dump_json(by_alias=True, indent=2)
    extras[URI] = crudo
    reference = assets[0]
    assets.append(asset_de_bytes(
        crudo.encode(), URI, id_=ASSET_ID, kind=AssetKind.DOCUMENT,
        visit=reference.visit, frame=reference.frame, media_type="application/json",
    ))
    extensions[EXTENSION] = Extension(
        name=EXTENSION, version="1.0", uri=URI, schema_id=SCHEMA,
        description="Per-asset losses, evidence and processing provenance; no clinical guarantee",
    )


def record_for(asset: Asset, *, agent: str) -> AssetRecord:
    """Base record, supplemented with process information known to the writer."""
    acquired = asset.external and asset.regulatory.layer == 1
    return AssetRecord(
        origin="acquired" if acquired else (
            "inferred" if asset.regulatory.layer == 3 else "computed"
        ),
        sources=list(asset.derived_from),
        source_status="not_applicable" if acquired else (
            "resolved" if asset.derived_from else "unresolved"
        ),
        process=None if acquired else Process(operation="conversion", agent=agent),
    )


def build_records(
    assets: list[Asset], extras: dict[str, str | bytes], *, agent: str,
    process_records: dict[str, Process] | None = None,
) -> dict[str, AssetRecord]:
    """Describe emitted representations, keeping unavailable process details unknown."""
    ids = {a.id for a in assets}
    records: dict[str, AssetRecord] = {}
    sources_by_role = {
        "asset.scene": ["asset.ios"],
        "asset.field": ["asset.ct_001"],
        "asset.field_fit": ["asset.field"],
        "asset.composite": ["asset.ct_001", "asset.ios"],
        "asset.gs": ["asset.ios"],
    }
    for asset in assets:
        if not asset.external and asset.regulatory.layer != 3:
            asset.regulatory = Regulatory(**{**asset.regulatory.model_dump(), "layer": 2})
        if asset.id in sources_by_role and not asset.derived_from:
            asset.derived_from = [s for s in sources_by_role[asset.id] if s in ids]
        clinical_unresolved = False
        if asset.id == "asset.clinical":
            references: set[str] = set()

            def collect(node, references=references) -> None:
                nonlocal clinical_unresolved
                if isinstance(node, dict):
                    if "source_ref" in node:
                        if node["source_ref"]:
                            references.add(node["source_ref"])
                        else:
                            clinical_unresolved = True
                    references.update(node.get("derived_from") or [])
                    for child in node.values():
                        collect(child)
                elif isinstance(node, list):
                    for child in node:
                        collect(child)

            collect(json.loads(extras[asset.uri]))
            resolved = {a.uri: a.id for a in assets}
            asset.derived_from = sorted({resolved[ref] for ref in references if ref in resolved})
            clinical_unresolved |= bool(references - resolved.keys())
        record = record_for(asset, agent=agent)
        records[asset.id] = record
        if clinical_unresolved:
            record.source_status = "unresolved"
        record.prior_history_unknown = record.origin == "acquired" or not record.sources
        record.prior_history_unknown |= clinical_unresolved
        meta = {}
        if asset.sidecar_uri and asset.sidecar_uri in extras:
            meta = json.loads(extras[asset.sidecar_uri])
        if meta.get("source_status") == "unresolved":
            record.source_status = "unresolved"
            record.prior_history_unknown = True
        compression = meta.get("compression_history")
        if compression:
            record.current_encoding = Claim.model_validate(compression["current_encoding"])
            record.prior_history_unknown = compression["prior_history_unknown"]
            record.introduced_losses = [Loss.model_validate(x) for x in compression["losses"]]
            record.history = Claim(state=State.DECLARED, value=compression["scope"])
        if meta.get("spacing_mm"):
            record.sampling = Claim(
                state=State.DECLARED,
                value=f"sample spacing in mm: {meta['spacing_mm']}; not effective resolution",
            )
        if record.origin == "acquired":
            continue
        process = record.process
        assert process is not None
        process.software = {"python": platform.python_version(), "writer": agent}
        process.operation = {
            "asset.scene": "mesh-to-glTF",
            "asset.field": "DICOM-to-Gaussian-seed",
            "asset.field_fit": "Gaussian-density-optimization",
            "asset.composite": "CBCT-IOS-composition",
            "asset.gs": "Gaussian-appearance-optimization",
            "asset.clinical": "report-transcription-and-regional-colour",
        }.get(asset.id, "model-inference" if record.origin == "inferred" else "conversion")
        if asset.id in ("asset.scene", "asset.field", "asset.composite", "asset.clinical"):
            process.reproducibility = "deterministic"
        if asset.id in ("asset.gs", "asset.field_fit", "asset.appearance"):
            process.reproducibility = "unknown" if asset.id == "asset.field_fit" else "stochastic"
            record.introduced_losses.append(Loss(
                id=f"{asset.id}:reconstruction", origin_asset=asset.id,
                kind="reconstruction", method=process.operation,
            ))
        if (meta.get("role") == "real appearance trained with gsplat"
                and asset.regulatory.layer != 3):
            record.origin = "mixed"
            process.operation = "mesh-to-glTF-and-Gaussian-appearance-optimization"
            process.reproducibility = "stochastic"
            record.introduced_losses.append(Loss(
                id=f"{asset.id}:appearance-reconstruction", origin_asset=asset.id,
                kind="reconstruction", method="per-case-Gaussian-appearance-optimization",
            ))
        model = meta.get("model") or {}
        if isinstance(model, dict):
            process.model = model.get("name")
            process.weights_sha256 = model.get("weights_sha256")
        subsampling = meta.get("subsampling")
        subsampling = subsampling or meta.get("submuestreo")
        if subsampling:
            process.parameters = subsampling
            record.introduced_losses.append(Loss(
                id=f"{asset.id}:subsampling", origin_asset=asset.id,
                kind="subsampling", method="Gaussian-seed-subsampling",
                input_shape=([subsampling["from"]] if subsampling.get("from", 0) > 0 else None),
                output_shape=([subsampling["to"]] if subsampling.get("to", 0) > 0 else None),
            ))
        if process_records and asset.id in process_records:
            override = process_records[asset.id]
            if process.model and override.model != process.model:
                raise ValueError(f"processing record contradicts model for {asset.id}")
            record.process = override.model_copy(deep=True)
        if "processing" in meta:
            record.process = Process.model_validate(meta["processing"])
            if record.origin == "inferred":
                record.process.model = process.model
        if asset.id == "asset.field_fit" and "reconstruction_error_hu" in meta:
            record.metrics.append(Metric(
                name="density_fit_rmse", value=meta["reconstruction_error_hu"],
                unit="source_grey_value", method="RMSE against input Gaussian density samples",
                reference_asset="asset.field", scope="fitted input samples, not independent TRE "
                "or calibrated HU; no diagnostic eligibility inferred",
            ))
    if process_records and set(process_records) - ids:
        raise ValueError("processing records reference assets not emitted")
    return records


def record_image_encoding(record: AssetRecord, path: Path, asset_id: str) -> None:
    """Identify supported JPEG DCT markers, rather than infer a codec from a suffix."""
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            return
        # Only header segments before SOS are needed. Never read image data or EXIF values.
        while marker := stream.read(2):
            if len(marker) != 2 or marker[0] != 0xff or marker[1] in (0xda, 0xd9):
                return
            size = stream.read(2)
            if len(size) != 2:
                return
            length = int.from_bytes(size, "big")
            if length < 2:
                return
            if marker[1] in (0xc0, 0xc1, 0xc2):
                record.current_encoding = Claim(state=State.DECLARED, value="JPEG-DCT-lossy")
                record.introduced_losses.append(Loss(
                    id=f"{asset_id}:jpeg-dct", origin_asset=asset_id,
                    kind="compression", method=f"JPEG SOF 0x{marker[1]:02x}",
                ))
                return
            stream.seek(length - 2, 1)


def record_mesh_precision(record: AssetRecord, positions, source: str) -> None:
    """Measure the local float64→float32 conversion, without claiming scanner accuracy."""
    import numpy as np

    original = np.asarray(positions, dtype=np.float64)
    delta = original.astype(np.float32).astype(np.float64) - original
    maximum = float(np.max(np.linalg.norm(delta, axis=1))) if len(original) else 0.0
    record.metrics.append(Metric(
        name="max_vertex_rounding_error", value=maximum, unit="mm",
        method="max Euclidean norm of float64 minus float32 positions",
        reference_asset=source, scope="conversion input coordinates only; not scanner accuracy",
    ))
    if maximum:
        record.introduced_losses.append(Loss(
            id="asset.scene:float32", origin_asset="asset.scene", kind="quantization",
            method="float64-to-float32-coordinate-conversion",
        ))


def dicom_losses(headers: list, asset_id: str) -> tuple[list[Loss], bool, Claim]:
    """Read declarations across ALL instances; 00/absence never erases another 01.

    Lossy Image Compression is lifetime history, not the current Transfer Syntax.
    Metadata are declarations rather than independent clinical evidence (PS3.3 C.7.6.1).
    """
    unknown = not headers
    losses: list[Loss] = []
    syntaxes: set[str] = set()
    for i, ds in enumerate(headers):
        syntax = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
        if syntax:
            syntaxes.add(syntax)
        flag = str(getattr(ds, "LossyImageCompression", ""))
        unknown |= flag not in ("00", "01")
        if flag != "01":
            continue

        def values(name: str, ds=ds) -> list:
            value = getattr(ds, name, None)
            if value is None:
                return []
            return list(value) if not isinstance(value, str | float | int) else [value]

        methods = values("LossyImageCompressionMethod")
        ratios = values("LossyImageCompressionRatio")
        unknown |= not methods or len(methods) != len(ratios)
        for j in range(max(1, len(methods), len(ratios))):
            ratio = None
            if j < len(ratios):
                try:
                    candidate = float(ratios[j])
                    if candidate >= 1 and candidate < float("inf"):
                        ratio = candidate
                    else:
                        unknown = True
                except (ValueError, TypeError):
                    unknown = True
            losses.append(Loss(
                id=f"{asset_id}:dicom:{i}:{j}", origin_asset=asset_id, kind="compression",
                method=str(methods[j]) if j < len(methods) else "DICOM:unspecified-lossy",
                ratio=ratio,
            ))
    encoding = Claim(
        state=State.DECLARED, value="DICOM Transfer Syntax: " + ",".join(sorted(syntaxes)),
    ) if syntaxes else Claim()
    return losses, unknown, encoding


def validate_audit(z: zipfile.ZipFile, manifest: Manifest, report: Report) -> None:
    extension = manifest.extensions.get(EXTENSION)
    if extension is None:
        return
    if extension.version != "1.0" or extension.schema_id != SCHEMA:
        report.error("20", "unsupported fidelity/provenance extension version")
        return
    target = next((a for a in manifest.assets if a.uri == extension.uri), None)
    if target is None or target.id != ASSET_ID or target.external:
        report.error("20", "fidelity/provenance must reference its embedded metadata asset")
        return
    try:
        audit = Audit.model_validate_json(z.read(target.uri))
    except (ValueError, KeyError) as exc:
        report.error("20", f"invalid fidelity/provenance contract: {exc}")
        return
    assets = {a.id: a for a in manifest.assets if a.id != ASSET_ID}
    records = audit.assets
    if set(assets) != set(records):
        report.error("20", "fidelity/provenance inventory differs from payload assets")
        return
    canonical = {key: record.model_copy(deep=True) for key, record in records.items()}
    try:
        inherit(canonical)
    except ValueError as exc:
        report.error("21", str(exc))
        return
    events: dict[str, Loss] = {}
    for id_, record in records.items():
        asset = assets[id_]
        path = f"fidelity.assets[{id_}]"
        if record.sources != asset.derived_from:
            report.error("21", "process sources differ from manifest derived_from", path)
        dangling = [s for s in record.sources if s not in assets]
        if dangling and record.source_status != "unresolved":
            report.error("21", f"unresolved sources reported as resolved: {dangling}", path)
        if record.source_status == "not_applicable" and (
            record.sources or record.origin != "acquired"
        ):
            report.error("21", "not_applicable sources only apply to acquired originals", path)
        if record.source_status == "resolved" and not record.sources:
            report.error("21", "resolved provenance requires sources", path)
        if record.inherited_losses != canonical[id_].inherited_losses:
            report.error("20", "inherited losses were removed, added or altered", path)
        if record.prior_history_unknown != canonical[id_].prior_history_unknown:
            report.error("20", "unknown upstream history was cleared", path)
        if not record.prior_history_unknown:
            if record.origin == "acquired" and record.history.state is State.UNKNOWN:
                report.error("20", "known original history requires declaration or evidence", path)
            if record.source_status == "unresolved":
                report.error("20", "unresolved provenance cannot establish known history", path)
        for loss in record.introduced_losses:
            if loss.origin_asset != id_:
                report.error("20", "introduced loss names a different origin asset", path)
            if loss.id in events:
                report.error("20", f"duplicate loss event id {loss.id}", path)
            events[loss.id] = loss
        if record.origin == "inferred":
            if asset.regulatory.layer != 3:
                report.error("21", "inferred output must remain in layer 3", path)
            if record.process is None or not record.process.model:
                report.error("21", "inferred output must identify its model", path)
            if asset.sidecar_uri is not None:
                try:
                    meta = json.loads(z.read(asset.sidecar_uri))
                    if meta.get("source_assets") != record.sources:
                        report.error("21", "sidecar sources contradict processing provenance", path)
                    if (record.process
                            and (meta.get("model") or {}).get("name") != record.process.model):
                        report.error("21", "sidecar model contradicts processing provenance", path)
                except (ValueError, KeyError, TypeError, AttributeError):
                    report.error("21", "inference sidecar cannot be inspected", path)
        if asset.regulatory.layer == 3 and record.origin != "inferred":
            report.error("21", "layer 3 cannot be relabelled as acquired or computed", path)
        if record.origin in ("computed", "mixed") and asset.regulatory.layer != 2:
            report.error("21", "computed/mixed representations must declare layer 2", path)
        if record.origin in ("computed", "mixed", "transcribed", "inferred"):
            if record.process is None:
                report.error("21", "processed output has no process record", path)
        if record.origin == "acquired" and (asset.regulatory.layer != 1 or record.process):
            report.error("21", "acquired origin contradicts its processing/layer", path)
        if record.source_status == "unresolved":
            report.warn("21", "processing provenance has unresolved sources", path)
        process = record.process
        if process and process.reproducibility == "unknown":
            report.warn("21", "reproducibility has not been established", path)
        registration_ids = {r.id for r in manifest.registrations}
        if set(record.registrations_used) - registration_ids:
            report.error("21", "geometric provenance references missing registrations", path)
        claims: list[Claim | Metric | Loss | ClinicalAssessment] = [
            record.current_encoding, record.history, record.calibration, record.sampling,
            record.geometric_uncertainty, *record.metrics, *record.introduced_losses,
            *record.inherited_losses, *record.clinical_assessments,
        ]
        evidence = [e for claim in claims for e in claim.evidence]
        if process:
            evidence += (process.reviews + process.reproducibility_evidence
                         + process.repeatability.evidence)
            if process.reproducibility == "stochastic" and not process.seeds:
                report.warn("21", "stochastic process has no recorded seeds", path)
            if process.model and not process.weights_sha256:
                report.warn("21", "model weights are not identified", path)
        if record.prior_history_unknown:
            report.warn("20", "prior loss history is unknown", path)
        for e in evidence:
            if e.reference_asset not in assets:
                report.error("20", f"evidence references missing asset {e.reference_asset}", path)
        for metric in record.metrics:
            if metric.reference_asset not in assets:
                report.error("20", "metric comparison references a missing asset", path)
        # A v0.3 hash only establishes bytes, not external evidence or clinical truth.
        if evidence:
            report.warn("20", "evidence/reviewer declarations are not authenticated; "
                        "technical validation does not certify clinical eligibility", path)
        for source in record.sources:
            if source in assets and assets[source].regulatory.layer == 3:
                if asset.regulatory.layer != 3:
                    report.error("21", "non-removable output depends on layer-3 inference", path)


def write_schema(root: Path) -> Path:
    schema = Audit.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = SCHEMA
    schema["title"] = "UOS fidelity and provenance extension v1.0"
    target = root / "schemas/uos-fidelity-provenance-1.0.schema.json"
    target.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    return target
