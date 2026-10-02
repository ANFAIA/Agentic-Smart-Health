"""Loss inheritance, evidence boundaries and provenance checked against emitted bytes."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from uos import Asset, Frame, Manifest, Subject, Visit, validate, write_uos
from uos.auditoria import append_audit, dicom_losses, read_fidelity, record_image_encoding
from uos.contenedor import asset_de_bytes
from uos.fidelidad import (
    ASSET_ID,
    URI,
    AssetRecord,
    Audit,
    Claim,
    ClinicalAssessment,
    Evidence,
    Loss,
    Metric,
    Process,
    inherit,
)
from uos.manifiesto import AssetKind, PHIState, Regulatory


def asset(id_, sources=(), layer=1):
    return asset_de_bytes(
        b"synthetic", f"data/{id_}.txt", id_=id_, kind=AssetKind.DOCUMENT,
        frame="master", visit="v1", media_type="text/plain",
        derived_from=list(sources), regulatory=Regulatory(layer=layer),
    )


def container(tmp_path, assets: list[Asset], records=None, mutate=None):
    extras = {a.uri: b"synthetic" for a in assets}
    extensions = {}
    if records is not None:
        append_audit(assets, records, extras, extensions)
        if mutate:
            document = json.loads(extras[URI])
            mutate(document)
            raw = json.dumps(document).encode()
            extras[URI] = raw
            assets[-1] = assets[-1].model_copy(update={
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            })
    manifest = Manifest(
        case_id="urn:uuid:synthetic", generator={"name": "test", "version": "1"},
        phi_state=PHIState.IDENTIFIED, subject=Subject(pseudonym="synthetic"),
        canonical_frame=Frame(id="master"), visits=[Visit(id="v1", date="2026-10-02")],
        assets=assets, extensions=extensions, extensions_used=list(extensions),
    )
    return write_uos(tmp_path / "case.uos", manifest, [], extras=extras)


def lineage():
    loss = Loss(id="input:jpeg", origin_asset="input", kind="compression", method="JPEG")
    records = {
        "input": AssetRecord(origin="acquired", source_status="not_applicable",
                             introduced_losses=[loss]),
        "converted": AssetRecord(origin="computed", sources=["input"], source_status="resolved",
                                 prior_history_unknown=False,
                                 process=Process(operation="conversion", agent="test@1")),
        "reconstructed": AssetRecord(
            origin="computed", sources=["converted"], source_status="resolved",
            prior_history_unknown=False,
            process=Process(operation="reconstruction", agent="test@1"),
        ),
    }
    assets = [asset("input"), asset("converted", ["input"], 2),
              asset("reconstructed", ["converted"], 2)]
    return assets, records


def test_legacy_has_unknown_fidelity_without_rewriting(tmp_path):
    path = container(tmp_path, [asset("input")])
    raw = path.read_bytes()
    record = read_fidelity(path).assets["input"]
    assert record.origin == "unknown" and record.prior_history_unknown
    assert record.current_encoding.state == "unknown"
    assert not record.clinical_assessments
    assert path.read_bytes() == raw


def test_losses_and_unknown_history_survive_multiple_conversions(tmp_path):
    assets, records = lineage()
    path = container(tmp_path, assets, records)
    assert validate(path).valid
    record = read_fidelity(path).assets["reconstructed"]
    assert record.prior_history_unknown
    assert [loss.id for loss in record.inherited_losses] == ["input:jpeg"]
    assert not record.clinical_assessments


@pytest.mark.parametrize("tamper", ["remove", "alter", "unknown"])
def test_validator_rejects_laundering_of_losses(tmp_path, tamper):
    assets, records = lineage()

    def mutate(document):
        entry = document["assets"]["reconstructed"]
        if tamper == "remove":
            entry["inherited_losses"] = []
        elif tamper == "alter":
            entry["inherited_losses"][0]["method"] = "lossless"
        else:
            entry["prior_history_unknown"] = False

    report = validate(container(tmp_path, assets, records, mutate))
    assert not report.valid
    assert any(e.code == "UOS-E-020" for e in report.errors)


def test_cycles_are_rejected_instead_of_hanging():
    _, records = lineage()
    records["input"].sources = ["reconstructed"]
    with pytest.raises(ValueError, match="cyclic"):
        inherit(records)


def test_evidence_is_required_and_clinical_scope_cannot_be_inferred():
    with pytest.raises(ValidationError):
        Claim(state="verified", value="lossless")
    with pytest.raises(ValidationError):
        ClinicalAssessment(task="root fracture detection", conclusion="eligible", state="verified")
    with pytest.raises(ValidationError):
        Metric(name="error", value=float("nan"), unit="mm", method="comparison",
               reference_asset="input", scope="vertices")


def test_evidence_reference_is_validated_but_not_authenticated(tmp_path):
    assets, records = lineage()
    evidence = Evidence(reference_asset="input", method="comparison", scope="pixels",
                        result="equal", verifier="declared-reader")
    records["converted"].current_encoding = Claim(
        state="verified", value="PNG encoding", evidence=[evidence],
    )
    path = container(tmp_path, assets, records)
    assert validate(path).valid
    assert any("not authenticated" in w for w in validate(path).warnings)
    assets, records = lineage()
    evidence.reference_asset = "missing-evidence"
    records["converted"].current_encoding = Claim(
        state="verified", value="PNG encoding", evidence=[evidence],
    )
    assert not validate(container(tmp_path, assets, records)).valid


def test_mismatched_inventory_and_sources_are_rejected(tmp_path):
    assets, records = lineage()
    path = container(tmp_path, assets, records,
                     lambda doc: doc["assets"]["converted"].update(sources=["input", "ghost"]))
    assert any(e.code == "UOS-E-021" for e in validate(path).errors)


def test_inference_cannot_be_reclassified_by_human_review(tmp_path):
    assets, records = lineage()
    records["converted"].origin = "inferred"
    records["converted"].process.model = "test:model"
    records["converted"].process.reviews = [Evidence(
        reference_asset="input", method="human review", scope="transcription",
        result="approved", verifier="reviewer",
    )]
    report = validate(container(tmp_path, assets, records))
    assert any("layer 3" in e for e in report.errors)


def test_dicom_reads_all_instances_and_preserves_lifetime_history():
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    headers = []
    for flag in ("00", "01", None):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        if flag:
            ds.LossyImageCompression = flag
        if flag == "01":
            ds.LossyImageCompressionMethod = ["ISO_10918_1", "ISO_15444_1"]
            ds.LossyImageCompressionRatio = [8, 4]
        headers.append(ds)
    losses, unknown, encoding = dicom_losses(headers, "ct")
    assert unknown and encoding.state == "declared"
    assert [loss.ratio for loss in losses] == [8, 4]
    assert [loss.method for loss in losses] == ["ISO_10918_1", "ISO_15444_1"]
    assert all(loss.state == "declared" for loss in losses)


def test_jpeg_compression_is_detected_from_bytes_and_history_remains_unknown(tmp_path):
    from PIL import Image

    path = tmp_path / "renamed.bin"
    Image.new("RGB", (8, 8)).save(path, format="JPEG")
    record = AssetRecord(origin="acquired", source_status="not_applicable")
    record_image_encoding(record, path, "photo")
    assert record.current_encoding.value == "JPEG-DCT-lossy"
    assert record.prior_history_unknown
    assert record.introduced_losses[0].ratio is None


def test_published_extension_schema_matches_contract():
    import jsonschema

    root = Path(__file__).resolve().parents[3]
    schema = json.loads((root / "schemas/uos-fidelity-provenance-1.0.schema.json").read_text())
    generated = Audit.model_json_schema(by_alias=True)
    generated.update({"$schema": schema["$schema"], "$id": schema["$id"],
                      "title": schema["title"]})
    assert schema == generated
    _, records = lineage()
    inherit(records)
    jsonschema.validate(Audit(assets=records).model_dump(mode="json", by_alias=True), schema)
    assert ASSET_ID not in records
