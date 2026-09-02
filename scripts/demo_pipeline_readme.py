#!/usr/bin/env python
"""demo_pipeline_readme.py — Genera la animacion del README desde el pipeline sintetico.

    uv run python scripts/demo_pipeline_readme.py

El README necesita ensenar el recorrido sin publicar dato clinico. Este script ejecuta el
pipeline sobre `ingestion_agents.synthetic`, escribe un directorio de trabajo inspeccionable
si se pide `--keep-workdir`, y actualiza dos artefactos versionables:

- `docs/assets/pipeline-demo-events.json`: resumen de la ejecucion.
- `docs/assets/pipeline-demo.svg`: diagrama de flujo animado para GitHub.

No versiona DICOM, PLY, STL ni `.uos`: esos ficheros se generan en `/tmp` y se resumen.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import tempfile
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


def _format_kb(size: int) -> str:
    return f"{size / 1024:.1f} KB"


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
        zip_infos = zf.infolist()
    zip_names = [info.filename for info in zip_infos]
    scene_entries = sum(name.startswith("scene/") for name in zip_names)
    derived_entries = sum(name.startswith("derived/") for name in zip_names)
    clinical_entries = sum(name.startswith("clinical/") for name in zip_names)
    provenance_entries = sum(name.startswith("provenance/") for name in zip_names)
    manifest_size = next(info.file_size for info in zip_infos if info.filename == MANIFIESTO)
    views_size = next(info.file_size for info in zip_infos if info.filename == "views.json")
    manifest_path = uos_dir / "manifest.json"
    _json(manifest_path, manifest)
    validation = valida(public_uos)
    validation_path = uos_dir / "validation.txt"
    validation_path.write_text(
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
                "demo-case/04_uos/acq-README.uos/",
                "acq-README.uos/clinical/",
                "acq-README.uos/derived/",
                "acq-README.uos/manifest.json",
                "acq-README.uos/provenance/",
                "acq-README.uos/scene/",
                "acq-README.uos/views.json",
            ],
            [
                "levels " + ", ".join(validation.niveles),
                f"errors {len(validation.errores)}",
                f"views {validation.vistas}",
                f"external acquired assets {validation.externos}",
                f"assets {len(manifest['assets'])}",
                f"zip_entries {len(zip_infos)}",
                f"uos_size_kb {public_uos.stat().st_size // 1024}",
                f"manifest_size {_format_kb(manifest_size)}",
                f"views_size {_format_kb(views_size)}",
                f"scene_entries {scene_entries}",
                f"derived_entries {derived_entries}",
                f"clinical_entries {clinical_entries}",
                f"provenance_entries {provenance_entries}",
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


def _svg_text(
    x: int,
    y: int,
    text: str,
    *,
    size: int = 15,
    fill: str = "#dbeafe",
    weight: int = 500,
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="ui-monospace, SFMono-Regular, Menlo, '
        f'Consolas, monospace" font-size="{size}" font-weight="{weight}" fill="{fill}">'
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


def _stage(stages: list[Stage], key: str) -> Stage:
    try:
        return next(stage for stage in stages if stage.key == key)
    except StopIteration as exc:
        raise ValueError(f"Stage missing from pipeline summary: {key}") from exc


def _metric(stage: Stage, prefix: str, fallback: str) -> str:
    return next((metric for metric in stage.metrics if metric.startswith(prefix)), fallback)


def _metric_tail(stage: Stage, prefix: str, fallback: str) -> str:
    metric = _metric(stage, prefix, "")
    if not metric:
        return fallback
    return metric.removeprefix(prefix).strip()


def _flow_card(
    x: int,
    y: int,
    *,
    index: int,
    title: str,
    subtitle: str,
    lines: tuple[str, str],
    begin: float,
    total: int,
) -> str:
    dot = 17 + index
    return f"""
  <g opacity="0">
    {_reveal(begin, total, fade=0.45)}
    <rect x="{x}" y="{y}" width="184" height="126" rx="10" fill="#0f172a"
      stroke="#334155" stroke-width="1.2"/>
    <circle cx="{x + 22}" cy="{y + 27}" r="12" fill="#020617" stroke="#64748b"/>
    {_svg_text(x + 17, y + 32, str(index), size=13, fill="#f8fafc", weight=700)}
    {_svg_text(x + 44, y + 29, title, size=14, fill="#f8fafc", weight=700)}
    {_svg_text(x + 44, y + 50, subtitle, size=11, fill="#93c5fd")}
    {_svg_text(x + 20, y + 82, lines[0], size=12, fill="#cbd5e1")}
    {_svg_text(x + 20, y + 104, lines[1], size=12, fill="#cbd5e1")}
    <circle cx="{x + 160}" cy="{y + 27}" r="4" fill="#22c55e"/>
    <text x="{x + 150}" y="{y + 110}" font-family="ui-monospace, SFMono-Regular, Menlo,
      Consolas, monospace" font-size="{dot}" fill="#1e293b" opacity="0.42">.</text>
  </g>"""


def _arrow(x1: int, x2: int, y: int, *, begin: float, total: int) -> str:
    return f"""
  <g opacity="0">
    {_reveal(begin, total, fade=0.25)}
    <line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#38bdf8" stroke-width="3"
      stroke-linecap="round"/>
    <path d="M {x2} {y} l -9 -6 v 12 z" fill="#38bdf8"/>
  </g>"""


def _turn_arrow(*, begin: float, total: int) -> str:
    return f"""
  <g opacity="0">
    {_reveal(begin, total, fade=0.25)}
    <path d="M 594 304 C 594 352 258 338 258 386" fill="none"
      stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>
    <path d="M 258 386 l -7 -10 h 14 z" fill="#38bdf8"/>
  </g>"""


def _dir_row(y: int, name: str, meta: str, *, begin: float, total: int) -> str:
    return f"""
    <g opacity="0">
      {_reveal(begin, total)}
      <rect x="730" y="{y - 18}" width="326" height="34" rx="6" fill="#111827"
        stroke="#1f2937"/>
      {_svg_text(750, y + 4, name, size=14, fill="#e5e7eb", weight=700)}
      {_svg_text(925, y + 4, meta, size=12, fill="#7dd3fc")}
    </g>"""


def render_svg(stages: list[Stage]) -> str:
    total = 20
    starts = [0.1, 3.6, 7.1, 10.6, 14.1]
    ingestion = _stage(stages, "ingestion")
    fusion = _stage(stages, "fusion")
    uos = _stage(stages, "uos")
    artifacts = next(
        (
            path.rsplit("/", 1)[-1].replace(" content-addressed blobs", " blobs")
            for path in ingestion.files
            if "content-addressed blobs" in path
        ),
        "content blobs",
    )
    segmentation = _metric_tail(fusion, "segmentation-agent", "-")
    levels = _metric_tail(uos, "levels", "UOS-Core")
    errors = _metric_tail(uos, "errors", "0")
    views = _metric_tail(uos, "views", "8")
    external_assets = _metric_tail(uos, "external acquired assets", "3")
    assets = _metric_tail(uos, "assets", "8")
    zip_entries = _metric_tail(uos, "zip_entries", "11")
    uos_size = _metric_tail(uos, "uos_size_kb", "-")
    manifest_size = _metric_tail(uos, "manifest_size", "-")
    views_size = _metric_tail(uos, "views_size", "-")
    scene_entries = _metric_tail(uos, "scene_entries", "-")
    derived_entries = _metric_tail(uos, "derived_entries", "-")
    clinical_entries = _metric_tail(uos, "clinical_entries", "-")
    provenance_entries = _metric_tail(uos, "provenance_entries", "-")
    cards = [
        _flow_card(
            54,
            178,
            index=1,
            title="Raw case",
            subtitle="synthetic input",
            lines=("OBJ / DICOM / report", "photo · synthetic only"),
            begin=starts[0],
            total=total,
        ),
        _flow_card(
            278,
            178,
            index=2,
            title="Ingestion",
            subtitle="four agents",
            lines=("TwinSnapshot", artifacts),
            begin=starts[1],
            total=total,
        ),
        _flow_card(
            502,
            178,
            index=3,
            title="Fusion",
            subtitle="geometry + FDI",
            lines=("rigid registration ok", f"segmentation {segmentation}"),
            begin=starts[2],
            total=total,
        ),
        _flow_card(
            166,
            390,
            index=4,
            title="Exports",
            subtitle="materialised twin",
            lines=("STL / PLY / render", "viewer + UOS"),
            begin=starts[3],
            total=total,
        ),
        _flow_card(
            390,
            390,
            index=5,
            title="UOS",
            subtitle="one container",
            lines=(f"{assets} assets · {zip_entries} entries", f"{views} views · {errors} errors"),
            begin=starts[4],
            total=total,
        ),
    ]
    arrows = [
        _arrow(238, 274, 241, begin=starts[1], total=total),
        _arrow(462, 498, 241, begin=starts[2], total=total),
        _turn_arrow(begin=starts[3], total=total),
        _arrow(350, 386, 453, begin=starts[4], total=total),
    ]
    directory = f"""
  <g opacity="0">
    {_reveal(starts[4] + 0.2, total)}
    <rect x="700" y="162" width="386" height="500" rx="14" fill="#020617"
      stroke="#38bdf8" stroke-width="1.4"/>
    <rect x="700" y="162" width="386" height="42" rx="14" fill="#0f172a"/>
    <circle cx="724" cy="183" r="5" fill="#ef4444"/>
    <circle cx="744" cy="183" r="5" fill="#f59e0b"/>
    <circle cx="764" cy="183" r="5" fill="#22c55e"/>
    {_svg_text(788, 188, "demo-case/04_uos/acq-README.uos/", size=14, fill="#e5e7eb", weight=700)}
    <g opacity="0">
      {_reveal(starts[4] + 0.5, total)}
      {_svg_text(750, 238, f"opened ZIP · {uos_size} KB", size=12, fill="#93c5fd",
                 weight=700)}
    </g>
    {_dir_row(276, "clinical/", f"{clinical_entries} entry", begin=starts[4] + 0.8,
              total=total)}
    {_dir_row(328, "derived/", f"{derived_entries} entries", begin=starts[4] + 1.0,
              total=total)}
    {_dir_row(380, "manifest.json", manifest_size, begin=starts[4] + 1.2, total=total)}
    {_dir_row(432, "provenance/", f"{provenance_entries} entry", begin=starts[4] + 1.4,
              total=total)}
    {_dir_row(484, "scene/", f"{scene_entries} entries", begin=starts[4] + 1.6,
              total=total)}
    {_dir_row(536, "views.json", views_size, begin=starts[4] + 1.8, total=total)}
    <g opacity="0">
      {_reveal(starts[4] + 2.2, total)}
      <rect x="730" y="588" width="326" height="48" rx="8" fill="#052e2b"
        stroke="#0f766e"/>
      {_svg_text(754, 617, f"{levels} · v0.2 · {external_assets} external refs",
                 size=13, fill="#99f6e4")}
    </g>
  </g>"""
    header = _svg_text(
        62,
        74,
        "Agentic Smart Health pipeline",
        size=24,
        fill="#f8fafc",
        weight=800,
    )
    subhead = _svg_text(
        64,
        104,
        "A synthetic dental case becomes a traceable .uos container",
        size=14,
        fill="#93c5fd",
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 760"
  role="img" aria-labelledby="title desc">
  <title id="title">Agentic Smart Health synthetic pipeline demo</title>
  <desc id="desc">Dark animated flow diagram showing synthetic raw dental files becoming
  a TwinSnapshot, fusion outputs, exports and a UOS directory.</desc>
  <defs>
    <radialGradient id="bg" cx="50%" cy="30%" r="80%">
      <stop offset="0%" stop-color="#1e3a8a"/>
      <stop offset="45%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#020617"/>
    </radialGradient>
  </defs>
  <rect width="1120" height="760" fill="url(#bg)"/>
  <rect x="34" y="34" width="1052" height="692" rx="20" fill="#020617"
    stroke="#1e293b" stroke-width="1.4"/>
  {header}
  {subhead}
  <rect x="56" y="130" width="560" height="2" rx="1" fill="#1e293b"/>
  <rect x="56" y="130" width="0" height="2" rx="1" fill="#38bdf8">
    <animate attributeName="width" from="0" to="980" dur="{total}s"
      repeatCount="indefinite"/>
  </rect>
  {''.join(cards)}
  {''.join(arrows)}
  {directory}
  <g opacity="0">
    {_reveal(starts[4] + 2.1, total)}
    <rect x="54" y="682" width="562" height="36" rx="10" fill="#0f172a"
      stroke="#334155"/>
    {_svg_text(76, 705, "Final artifact: one .uos file with scene, provenance and views",
              size=14, fill="#e0f2fe", weight=700)}
  </g>
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
