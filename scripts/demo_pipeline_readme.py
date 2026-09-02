#!/usr/bin/env python
"""demo_pipeline_readme.py — Genera la animacion del README desde el pipeline sintetico.

    uv run python scripts/demo_pipeline_readme.py

El README necesita ensenar el recorrido sin publicar dato clinico. Este script ejecuta el
pipeline sobre `ingestion_agents.synthetic`, escribe un directorio de trabajo inspeccionable
si se pide `--keep-workdir`, y actualiza dos artefactos versionables:

- `docs/assets/pipeline-demo-events.json`: resumen de la ejecucion.
- `docs/assets/pipeline-demo.svg`: explorador de directorios animado para GitHub.

No versiona DICOM, PLY, STL ni `.uos`: esos ficheros se generan en `/tmp` y se resumen.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
for src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(src))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))

import numpy as np  # noqa: E402
from agent_orchestrator import CaseInput, IngestionPipeline, PipelineResult  # noqa: E402
from ingestion_agents import ArtifactStore, synthetic  # noqa: E402
from ingestion_agents.mesh_agent import parse_obj  # noqa: E402
from ingestion_agents.ontology import all_fdi_codes  # noqa: E402
from uos import valida  # noqa: E402
from uos.contenedor import MANIFIESTO  # noqa: E402

ASSETS = RAIZ / "docs" / "assets"
SVG = ASSETS / "pipeline-demo.svg"
EVENTS = ASSETS / "pipeline-demo-events.json"


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    files: list[str]
    metrics: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic ASH pipeline and regenerate the README animation."
    )
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    parser.add_argument("--workdir", type=Path, help="Directory for generated demo files.")
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the generated DICOM/STL/PLY/UOS demo directory for inspection.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.8,
        help="Synthetic CBCT spacing in mm. Larger values make the public demo faster.",
    )
    return parser


def _write_photo(path: Path) -> Path:
    """Foto sintetica minima para que el `image-agent` tambien participe."""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 360), (238, 234, 226))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((70, 105, 570, 260), radius=70, fill=(184, 105, 112))
    for i, x in enumerate(range(140, 515, 50)):
        tone = 232 - (i % 3) * 7
        draw.rounded_rectangle((x, 95, x + 38, 210), radius=18, fill=(tone, tone - 6, 205))
        draw.line((x + 7, 112, x + 30, 112), fill=(255, 255, 245), width=2)
    draw.arc((105, 80, 535, 310), start=198, end=342, fill=(120, 77, 82), width=6)
    img.save(path)
    return path


def _segmentador(fdis: list[str], *, prob: float = 0.99):
    """Doble explicito del modelo: segmenta el campo sintetico en zonas por eje x."""
    codigos = [all_fdi_codes().index(fdi) + 1 for fdi in fdis]
    n_clases = len(all_fdi_codes()) + 1

    def segmentar(points: np.ndarray) -> np.ndarray:
        clases = np.zeros(len(points), dtype=int)
        for codigo, zona in zip(
            codigos, np.array_split(np.argsort(points[:, 0]), len(codigos)), strict=True
        ):
            clases[zona] = codigo
        p = np.full((len(points), n_clases), (1.0 - prob) / (n_clases - 1))
        p[np.arange(len(points)), clases] = prob
        return np.log(p)

    return segmentar


def _nubes_del_twin(
    pipeline: IngestionPipeline, result: PipelineResult
) -> tuple[np.ndarray, np.ndarray]:
    snapshot = result.snapshot
    if snapshot is None or snapshot.surface_ref is None or snapshot.gaussian_field_ref is None:
        raise RuntimeError("La ingesta no produjo las dos nubes necesarias para fusion.")
    malla = pipeline.store.load(snapshot.surface_ref)["positions"].astype(np.float64)
    campo = pipeline.store.load(snapshot.gaussian_field_ref)
    centros = campo["centers"].astype(np.float64) + campo["origin"]
    return (
        malla[:: max(1, len(malla) // 4000)],
        centros[:: max(1, len(centros) // 4000)],
    )


def _etiquetas_ios_sinteticas(mesh: Path) -> np.ndarray | None:
    """Etiquetas FDI por vertice, coherentes con `synthetic.write_mesh_obj`.

    El generador escribe, para cada diente, primero la corona y luego su encia. La etiqueta
    de la corona es el FDI; la de la encia es 0. Si el layout cambia, se devuelve `None` y
    la demo sigue siendo valida, solo con menos vistas en el `.uos`.
    """
    n_vertices = len(parse_obj(mesh)["positions"])
    codigos = synthetic.upper_arch_codes()
    partes = len(codigos) * 2
    if n_vertices % partes:
        return None
    por_parte = n_vertices // partes
    etiquetas: list[int] = []
    for codigo in codigos:
        etiquetas.extend([int(codigo)] * por_parte)
        etiquetas.extend([0] * por_parte)
    return np.asarray(etiquetas, dtype=np.int32)


def _json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _short_ref(ref: str | None) -> str:
    if not ref:
        return "-"
    return ref[:18] + "..."


def _outputs(result: PipelineResult) -> list[dict[str, Any]]:
    return [
        {
            "agent": out.agent,
            "status": out.status.value,
            "detail": out.detail,
            "path": str(out.path) if out.path else None,
            "paths": [str(p) for p in out.paths],
            "max_deviation_mm": out.max_deviation_mm,
            "psnr_db": out.psnr_db,
            "ssim": out.ssim,
        }
        for out in result.exports
    ]


def run_demo(workdir: Path, spacing: float) -> tuple[list[Stage], dict[str, Any]]:
    if workdir.exists():
        shutil.rmtree(workdir)
    case_dir = workdir / "input"
    synthetic.write_case(case_dir, spacing=spacing)
    photo = _write_photo(case_dir / "intraoral-photo" / "synthetic-smile.png")

    cbct_slices = len(list((case_dir / "cbct").glob("*.dcm")))
    stages = [
        Stage(
            "input",
            "0. Synthetic raw case",
            [
                "demo-case/input/scan_upper.obj",
                f"demo-case/input/cbct/{cbct_slices} DICOM slices",
                "demo-case/input/informe.txt",
                "demo-case/input/intraoral-photo/synthetic-smile.png",
            ],
            [
                "public synthetic data",
                f"CBCT spacing {spacing:.1f} mm",
                "no patient file enters git",
            ],
        )
    ]

    base = IngestionPipeline(
        ArtifactStore(workdir / "01_ingestion" / "artifacts"),
        quarantine_dir=workdir / "quarantine",
    )
    ingested = base.run(
        CaseInput.from_case_dir(
            case_dir, acquisition_id="acq-README", patient_id="SYNTH-README"
        )
    )
    if ingested.snapshot is None:
        raise RuntimeError("La ingesta sintetica no produjo TwinSnapshot.")
    _json(
        workdir / "01_ingestion" / "twin_snapshot.json",
        ingested.snapshot.model_dump(mode="json"),
    )
    _json(
        workdir / "01_ingestion" / "outcomes.json",
        [o.model_dump(mode="json") for o in ingested.outcomes],
    )
    artifact_count = len(list((workdir / "01_ingestion" / "artifacts").glob("*.npz")))
    stages.append(
        Stage(
            "ingestion",
            "1. Ingestion agents",
            [
                "demo-case/01_ingestion/twin_snapshot.json",
                "demo-case/01_ingestion/outcomes.json",
                f"demo-case/01_ingestion/artifacts/{artifact_count} content-addressed blobs",
            ],
            [
                "mesh-agent / cbct-agent / report-agent / image-agent",
                f"surface_ref {_short_ref(ingested.snapshot.surface_ref)}",
                f"gaussian_field_ref {_short_ref(ingested.snapshot.gaussian_field_ref)}",
            ],
        )
    )

    fdis = sorted({obs.region_id for obs in ingested.snapshot.regional})
    complete = IngestionPipeline(
        base.store,
        quarantine_dir=workdir / "quarantine",
        segmenter=_segmentador(fdis),
    )
    fused = complete.fuse(ingested, registration=_nubes_del_twin(base, ingested))
    _json(
        workdir / "02_fusion" / "fusion.json",
        [o.model_dump(mode="json") for o in fused.fusion],
    )
    _json(
        workdir / "02_fusion" / "segmentation.json",
        [o.model_dump(mode="json") for o in fused.analysis],
    )
    geo = next((o for o in fused.fusion if o.agent.startswith("geometric-fusion-agent")), None)
    seg = next((o for o in fused.analysis if o.agent.startswith("segmentation-agent")), None)
    rms = (
        fused.snapshot.provenance.transform.rms_mm
        if fused.snapshot is not None and fused.snapshot.provenance.transform is not None
        else None
    )
    stages.append(
        Stage(
            "fusion",
            "2. Fusion and segmentation",
            [
                "demo-case/02_fusion/fusion.json",
                "demo-case/02_fusion/segmentation.json",
            ],
            [
                f"geometric-fusion-agent {geo.status.value if geo else 'missing'}",
                f"rms_error_mm {rms:.3f}" if rms is not None else "rms_error_mm -",
                f"segmentation-agent {seg.status.value if seg else 'missing'}",
                "semantic-fusion-agent attaches report findings to FDI regions",
            ],
        )
    )

    export_dir = workdir / "03_exports"
    exported = complete.exportar(
        fused,
        export_dir,
        malla=case_dir / "scan_upper.obj",
        etiquetas_ios=_etiquetas_ios_sinteticas(case_dir / "scan_upper.obj"),
        imagenes=[photo],
        informes=[case_dir / "informe.txt"],
        cbct=case_dir / "cbct",
    )
    _json(workdir / "03_exports" / "exports.json", _outputs(exported))
    ok_exports = [e.agent.split("@")[0] for e in exported.exports if e.ok]
    stages.append(
        Stage(
            "exports",
            "3. Export agents",
            [
                f"demo-case/03_exports/{exported.snapshot.acquisition_id}.stl",
                f"demo-case/03_exports/{exported.snapshot.acquisition_id}.ply",
                f"demo-case/03_exports/{exported.snapshot.acquisition_id}-visor/",
                "demo-case/03_exports/render/",
                "demo-case/03_exports/exports.json",
            ],
            [
                f"{len(ok_exports)} export channels wrote files",
                f"reversible {str(exported.reversible).lower()}",
                "STL / PLY / PNG / viewer package / UOS",
            ],
        )
    )

    uos_out = exported.export("uos-export-agent")
    if uos_out is None or uos_out.path is None or not uos_out.path.exists():
        raise RuntimeError("El exportador UOS no produjo contenedor.")
    uos_dir = workdir / "04_uos"
    uos_dir.mkdir(parents=True, exist_ok=True)
    public_uos = uos_dir / "acq-README.uos"
    shutil.copy2(uos_out.path, public_uos)
    with zipfile.ZipFile(public_uos) as zf:
        manifest = json.loads(zf.read(MANIFIESTO))
    _json(uos_dir / "manifest.json", manifest)
    validation = valida(public_uos)
    (uos_dir / "validation.txt").write_text(
        "\n".join(
            [
                f"valid: {validation.valido}",
                "levels: " + ", ".join(validation.niveles),
                f"errors: {len(validation.errores)}",
                f"warnings: {len(validation.avisos)}",
                f"views: {validation.vistas}",
                f"external acquired assets: {validation.externos}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stages.append(
        Stage(
            "uos",
            "4. UOS container",
            [
                "demo-case/04_uos/acq-README.uos",
                "demo-case/04_uos/manifest.json",
                "demo-case/04_uos/validation.txt",
            ],
            [
                "levels " + ", ".join(validation.niveles),
                f"errors {len(validation.errores)}",
                f"views {validation.vistas}",
                f"external acquired assets {validation.externos}",
            ],
        )
    )

    summary = {
        "workdir": "demo-case (generated in /tmp by default; use --workdir to choose it)",
        "synthetic": True,
        "cbct_slices": cbct_slices,
        "stages": [
            {"key": s.key, "title": s.title, "files": s.files, "metrics": s.metrics}
            for s in stages
        ],
        "agents": {
            "ingestion": [o.agent for o in ingested.outcomes],
            "fusion": [o.agent for o in fused.fusion],
            "analysis": [o.agent for o in fused.analysis],
            "exports": [o.agent for o in exported.exports],
        },
        "validation": {
            "valid": validation.valido,
            "levels": list(validation.niveles),
            "errors": validation.errores,
            "warnings": validation.avisos,
            "views": validation.vistas,
            "external_acquired_assets": validation.externos,
        },
    }
    return stages, summary


def _svg_text(x: int, y: int, text: str, *, size: int = 15, fill: str = "#1f2933") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="ui-monospace, SFMono-Regular, Menlo, '
        f"Consolas, monospace\" font-size=\"{size}\" fill=\"{fill}\">"
        f"{html.escape(text)}</text>"
    )


def _reveal(begin: float, total: int, *, fade: float = 0.35) -> str:
    """Animacion acumulativa: oculto hasta `begin`, visible hasta cerrar el ciclo."""
    start = begin / total
    visible = min((begin + fade) / total, 0.92)
    return (
        '<animate attributeName="opacity" values="0;0;1;1;0" '
        f'keyTimes="0;{start:.4f};{visible:.4f};0.9400;1" '
        f'dur="{total}s" repeatCount="indefinite"/>'
    )


def render_svg(stages: list[Stage]) -> str:
    total = 18
    row_bits = []
    y = 190
    for i, stage in enumerate(stages):
        begin = 0.1 + i * 3.2
        rows = [("dir", stage.title)] + [
            ("file", "  " + file.replace("demo-case/", "")) for file in stage.files
        ]
        for j, (kind, label) in enumerate(rows):
            color = "#2563eb" if kind == "dir" else "#334155"
            weight = "700" if kind == "dir" else "500"
            row_begin = begin + j * 0.12
            row_bits.append(
                f'<text x="82" y="{y}" opacity="0" '
                f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
                f'font-size="15" font-weight="{weight}" fill="{color}">'
                f"{html.escape(label)}"
                f"{_reveal(row_begin, total)}"
                f'<animate attributeName="fill" values="{color};#0f766e;{color}" '
                f'begin="{row_begin:.1f}s" dur="1.2s" repeatCount="indefinite"/>'
                "</text>"
            )
            y += 22

    cards = []
    for i, stage in enumerate(stages):
        x = 54 + i * 205
        begin = i * 3.2
        cards.append(
            f'<g><rect x="{x}" y="82" width="178" height="54" rx="8" fill="#ffffff" '
            f'stroke="#cbd5e1" stroke-width="1.2"/>'
            f'<rect x="{x}" y="82" width="178" height="54" rx="8" fill="#dbeafe" opacity="0">'
            f'<animate attributeName="opacity" values="0;0.95;0" begin="{begin:.1f}s" '
            f'dur="3.2s" repeatCount="indefinite"/></rect>'
            f'{_svg_text(x + 16, 106, stage.title.split(". ", 1)[-1], size=13, fill="#0f172a")}'
            f'{_svg_text(x + 16, 126, stage.key, size=12, fill="#64748b")}</g>'
        )

    metric_lines = []
    y = 198
    for i, stage in enumerate(stages):
        begin = 0.1 + i * 3.2
        group = [_svg_text(760, y, stage.title, size=14, fill="#0f172a")]
        y += 22
        for metric in stage.metrics[:3]:
            wrapped = textwrap.wrap(metric, width=34)[:2]
            for j, line in enumerate(wrapped):
                prefix = "- " if j == 0 else "  "
                group.append(_svg_text(778, y, prefix + line, size=13, fill="#475569"))
                y += 18
        y += 12
        metric_lines.append(
            f'<g opacity="0">{_reveal(begin, total)}{"".join(group)}</g>'
        )

    header = _svg_text(
        126,
        58,
        "Agentic Smart Health: synthetic case -> TwinSnapshot -> UOS",
        size=15,
        fill="#e2e8f0",
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 760"
  role="img" aria-labelledby="title desc">
  <title id="title">Agentic Smart Health synthetic pipeline demo</title>
  <desc id="desc">Animated directory view showing synthetic raw dental files becoming
  a TwinSnapshot, fusion outputs, exports and a UOS container.</desc>
  <rect width="1120" height="760" fill="#f8fafc"/>
  <rect x="36" y="34" width="1048" height="692" rx="10" fill="#ffffff" stroke="#cbd5e1"/>
  <rect x="36" y="34" width="1048" height="38" rx="10" fill="#0f172a"/>
  <circle cx="62" cy="53" r="5" fill="#ef4444"/>
  <circle cx="80" cy="53" r="5" fill="#f59e0b"/>
  <circle cx="98" cy="53" r="5" fill="#22c55e"/>
  {header}
  <rect x="54" y="146" width="650" height="540" rx="8" fill="#f8fafc" stroke="#dbe3ea"/>
  <rect x="738" y="146" width="306" height="540" rx="8" fill="#f8fafc" stroke="#dbe3ea"/>
  {_svg_text(78, 174, "demo-case/", size=17, fill="#0f172a")}
  {_svg_text(760, 174, "Measured run summary", size=17, fill="#0f172a")}
  <rect x="54" y="710" width="1010" height="5" rx="2.5" fill="#e2e8f0"/>
  <rect x="54" y="710" width="0" height="5" rx="2.5" fill="#2563eb">
    <animate attributeName="width" from="0" to="1010" dur="{total}s" repeatCount="indefinite"/>
  </rect>
  {''.join(cards)}
  {''.join(row_bits)}
  {''.join(metric_lines)}
</svg>
"""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assets = args.assets_dir
    svg = assets / SVG.name
    events = assets / EVENTS.name

    temp: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir is not None:
        workdir = args.workdir
    elif args.keep_workdir:
        workdir = Path(tempfile.mkdtemp(prefix="ash-readme-pipeline-")) / "demo-case"
    else:
        temp = tempfile.TemporaryDirectory(prefix="ash-readme-pipeline-")
        workdir = Path(temp.name) / "demo-case"

    try:
        stages, summary = run_demo(workdir, args.spacing)
        assets.mkdir(parents=True, exist_ok=True)
        _json(events, summary)
        svg.write_text(render_svg(stages), encoding="utf-8")
        print(f"events: {events}")
        print(f"svg:    {svg}")
        print(f"workdir: {workdir}")
    finally:
        if temp is not None and not args.keep_workdir:
            temp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
