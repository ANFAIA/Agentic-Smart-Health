"""CLI del orquestador: ingiere un caso y escribe el `TwinSnapshot`.

Uso típico (Demo Semana 4 — primer twin sintético, sin datos de paciente):

    uv run python apps/agent-orchestrator/main.py --demo

O sobre un caso ya en disco:

    uv run python apps/agent-orchestrator/main.py --case data/interim/mi-caso \\
        --out data/processed/twin.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from agent_orchestrator import CaseInput, IngestionPipeline
from ingestion_agents import ArtifactStore, synthetic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingesta multimodal → TwinSnapshot")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", type=Path, help="Directorio con las modalidades del caso.")
    source.add_argument(
        "--demo", action="store_true", help="Genera un caso sintético y lo ingiere."
    )
    parser.add_argument("--store", type=Path, default=Path("data/interim/artifacts"))
    parser.add_argument("--quarantine", type=Path, default=Path("data/interim/quarantine"))
    parser.add_argument("--out", type=Path, help="Ruta del JSON del snapshot.")
    parser.add_argument(
        "--report-backend", choices=("rules", "llm"), default="rules",
        help="`rules` es determinista y no necesita red ni clave de API.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    case_dir = args.case
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.demo:
        tmp = tempfile.TemporaryDirectory(prefix="ash-demo-")
        case_dir = Path(tmp.name) / "caso-sintetico"
        synthetic.write_case(case_dir)
        print(f"· Caso sintético generado en {case_dir}")

    pipeline = IngestionPipeline(
        ArtifactStore(args.store),
        quarantine_dir=args.quarantine,
        report_backend=args.report_backend,
    )
    result = pipeline.run(CaseInput.from_case_dir(case_dir))

    print(f"\n{'modalidad':<10} {'estado':<8} {'confianza':>10} {'latencia':>10}  detalle")
    for outcome in result.outcomes:
        confidence = f"{outcome.provenance.confidence:.2f}" if outcome.provenance else "—"
        print(
            f"{outcome.modality.value:<10} {outcome.status.value:<8} {confidence:>10} "
            f"{outcome.latency_s:>9.2f}s  {outcome.detail or ''}"
        )

    budget = "dentro" if result.within_budget else "FUERA"
    print(f"\nLatencia total: {result.latency_s:.2f} s ({budget} del presupuesto de 60 s)")
    print(f"Paciente (seudónimo): {result.patient_pseudonym}")

    if result.hitl_required:
        print("\n⚠ Requiere revisión humana antes de persistir:")
        for reason in result.hitl_reasons:
            print(f"  · {reason}")

    if result.snapshot is None:
        print("\n✗ No se formó ningún TwinSnapshot (ver motivos arriba).")
        return 1

    snapshot = result.snapshot
    print(
        f"\n✓ TwinSnapshot {snapshot.acquisition_id} · schema {snapshot.schema_version}\n"
        f"  gaussian_field_ref: {snapshot.gaussian_field_ref}\n"
        f"  surface_ref:        {snapshot.surface_ref}\n"
        f"  primitivas: {snapshot.n_primitives} · "
        f"observaciones regionales: {len(snapshot.regional)}"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  escrito en {args.out}")

    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
