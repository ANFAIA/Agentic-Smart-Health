"""Versioned evidence contract; container integrity never certifies clinical eligibility.

Stored in an optional extension asset, leaving the v0.3 manifest schema unchanged.
Absence of this extension means unknown fidelity and unknown reproducibility.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

EXTENSION = "uos_fidelity_provenance"
SCHEMA: Literal["uos-fidelity-provenance/1.0"] = "uos-fidelity-provenance/1.0"
URI = "metadata/fidelity-provenance.json"
ASSET_ID = "asset.fidelity_provenance"


class State(StrEnum):
    UNKNOWN = "unknown"
    DECLARED = "declared"
    VERIFIED = "verified"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Evidence(Contract):
    reference_asset: str = Field(min_length=1)
    method: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    result: str = Field(min_length=1)
    verifier: str = Field(min_length=1)


class Claim(Contract):
    state: State = State.UNKNOWN
    value: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _supported(self) -> Claim:
        if self.state is State.UNKNOWN and (self.value is not None or self.evidence):
            raise ValueError("unknown claims cannot assert a value or evidence")
        if self.state is not State.UNKNOWN and not self.value:
            raise ValueError("a declared or verified claim needs a value")
        if self.state is State.VERIFIED and not self.evidence:
            raise ValueError("verified claims require scoped evidence")
        return self


class Loss(Contract):
    id: str = Field(min_length=1)
    origin_asset: str = Field(min_length=1)
    kind: Literal["compression", "quantization", "subsampling", "reconstruction", "selection"]
    method: str = Field(min_length=1)
    state: State = State.DECLARED
    ratio: float | None = Field(default=None, ge=1)
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _supported(self) -> Loss:
        if self.state is State.UNKNOWN:
            raise ValueError("an unknown history is recorded separately, not as a known loss")
        if self.state is State.VERIFIED and not self.evidence:
            raise ValueError("verified losses require scoped evidence")
        for shape in (self.input_shape, self.output_shape):
            if shape is not None and (not shape or any(n <= 0 for n in shape)):
                raise ValueError("dimensions must be positive")
        return self


class Metric(Contract):
    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    method: str = Field(min_length=1)
    reference_asset: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    state: State = State.DECLARED
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _supported(self) -> Metric:
        if self.state is State.UNKNOWN:
            raise ValueError("an unmeasured metric must be absent")
        if self.state is State.VERIFIED and not self.evidence:
            raise ValueError("verified metrics require scoped evidence")
        return self


class ClinicalAssessment(Contract):
    task: str = Field(min_length=1)
    conclusion: Literal["unknown", "eligible", "ineligible"] = "unknown"
    state: State = State.UNKNOWN
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _supported(self) -> ClinicalAssessment:
        if (self.conclusion == "unknown") != (self.state is State.UNKNOWN):
            raise ValueError("unknown task eligibility must remain unknown")
        if self.conclusion != "unknown" and not self.evidence:
            raise ValueError("clinical eligibility requires task-specific evidence")
        return self


class Process(Contract):
    operation: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    reproducibility: Literal["unknown", "deterministic", "stochastic"] = "unknown"
    reproducibility_evidence: list[Evidence] = Field(default_factory=list)
    software: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    seeds: dict[str, int] = Field(default_factory=dict)
    model: str | None = None
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # A review records an action; it never overwrites origin or model.
    reviews: list[Evidence] = Field(default_factory=list)
    repeatability: Claim = Field(default_factory=Claim)


class AssetRecord(Contract):
    origin: Literal["unknown", "acquired", "transcribed", "computed", "inferred", "mixed"]
    sources: list[str] = Field(default_factory=list)
    source_status: Literal["resolved", "unresolved", "not_applicable"] = "unresolved"
    process: Process | None = None
    current_encoding: Claim = Field(default_factory=Claim)
    # True unless a history has explicitly been documented. Never reset by conversion.
    prior_history_unknown: bool = True
    history: Claim = Field(default_factory=Claim)
    calibration: Claim = Field(default_factory=Claim)
    sampling: Claim = Field(default_factory=Claim)
    geometric_uncertainty: Claim = Field(default_factory=Claim)
    registrations_used: list[str] = Field(default_factory=list)
    introduced_losses: list[Loss] = Field(default_factory=list)
    inherited_losses: list[Loss] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    clinical_assessments: list[ClinicalAssessment] = Field(default_factory=list)


class Audit(Contract):
    schema_id: Literal["uos-fidelity-provenance/1.0"] = Field(default=SCHEMA, alias="schema")
    # The extension asset itself is excluded to avoid a self-referential lineage/hash.
    assets: dict[str, AssetRecord]


def inherit(records: dict[str, AssetRecord]) -> None:
    """Propagate known losses and uncertainty; reject cyclic derivation graphs."""
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(id_: str) -> None:
        if id_ in visiting:
            raise ValueError(f"cyclic provenance at {id_}")
        if id_ in done:
            return
        visiting.add(id_)
        record = records[id_]
        losses: dict[str, Loss] = {}
        for source in record.sources:
            if source not in records:
                record.prior_history_unknown = True
                record.source_status = "unresolved"
                continue
            visit(source)
            parent = records[source]
            record.prior_history_unknown |= parent.prior_history_unknown
            for loss in (*parent.inherited_losses, *parent.introduced_losses):
                if loss.id in losses and losses[loss.id] != loss:
                    raise ValueError(f"conflicting loss event {loss.id}")
                losses[loss.id] = loss
        record.inherited_losses = [losses[key] for key in sorted(losses)]
        visiting.remove(id_)
        done.add(id_)

    for id_ in records:
        visit(id_)
