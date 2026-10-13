#!/usr/bin/env python
"""Puntúa el margen corona-encía de un caso contra líneas marcadas por una persona.

    uv run python scripts/puntua_margen.py \\
        --anotacion anotador-1/ --stl-cargado-como lps \\
        --candidato actual=caso.uos --candidato pliegue=etiquetas.npz:despues \\
        --salida data/processed/<caso>/margen-anotado

Cada `--anotacion` es una carpeta con un `.mrk.json` de 3D Slicer por pieza: una curva
cerrada sobre el margen gingival, en un fichero cuyo nombre lleva el código FDI. Con dos
carpetas —dos personas, o la misma en dos días— se mide además cuánto distan entre sí.

Cada `--candidato` es un etiquetado que se quiere puntuar: un `.uos`, del que se leen la
escena y `derived/seg_teeth.bin`, o un `.npz` con `V`, `F` y la clave indicada tras `:`.
Todos deben indexar la misma malla, que es la del STL que se anotó.

El protocolo de marcado está en `docs/spec/uos-protocolo-margen-gingival.md`. El resultado
se guarda aparte: no se escribe en ningún `.uos`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

RESUMEN_EN = "Scores a case's crown-gum margin against closed curves a person traced."
RAIZ = Path(__file__).resolve().parent.parent
for src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(src))

from analysis_agents.margen_anotado import (  # noqa: E402
    CONTACTO_MM,
    acuerdo_entre_anotaciones,
    comprueba_sobre_malla,
    distancia_a_las_curvas,
    etiquetas_de_curvas,
    lee_anotacion,
)
from analysis_agents.margen_metricas import metricas_de_arcada  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "mide_segmentacion", Path(__file__).with_name("mide_segmentacion.py"))
assert _spec and _spec.loader
_ms = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, _ms)
_spec.loader.exec_module(_ms)


def lee_candidato(texto: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """`NOMBRE=ruta.uos` o `NOMBRE=ruta.npz:clave` → nombre, posiciones, caras, etiquetas."""
    nombre, _, resto = texto.partition("=")
    if not nombre or not resto:
        raise SystemExit(f"--candidato {texto!r}: se espera NOMBRE=ruta")
    if resto.endswith(".uos"):
        with zipfile.ZipFile(resto) as z:
            if _ms.SEGMENTACION not in z.namelist():
                raise SystemExit(f"{resto} no trae `{_ms.SEGMENTACION}`")
            pos, tri = _ms.lee_escena(z.read(_ms.ESCENA))
            etq = np.frombuffer(z.read(_ms.SEGMENTACION), dtype="<i2").astype(np.int64)
    else:
        ruta, _, clave = resto.rpartition(":")
        if not ruta.endswith(".npz") or not clave:
            raise SystemExit(f"--candidato {texto!r}: se espera un `.uos` o `ruta.npz:clave`")
        with np.load(ruta) as datos:
            pos, tri = datos["V"].astype(np.float64), datos["F"].astype(np.int64)
            etq = datos[clave].astype(np.int64)
    if len(etq) != len(pos):
        raise SystemExit(f"{nombre}: {len(etq):,} etiquetas para {len(pos):,} vértices")
    return nombre, pos, tri, etq


def puntua(pos: np.ndarray, tri: np.ndarray, candidatos: dict[str, np.ndarray],
           anotaciones: dict[str, dict[int, np.ndarray]]) -> dict[str, Any]:
    """Todas las medidas: cada candidato contra cada anotación, y las anotaciones entre sí."""
    resultado: dict[str, Any] = {
        "schema": "histora-margin-annotation/1.0",
        "mesh": {"vertices": int(len(pos)), "faces": int(len(tri))},
        "contact_threshold_mm": CONTACTO_MM,
        "annotations": {}, "scores": {}, "between_annotations": {},
        "scope": ("Agreement with lines one or more people traced on this case only. Not "
                  "clinical validation and not general accuracy of the segmentation. Lines "
                  "used to tune a method can no longer score it."),
    }
    for nombre, curvas in anotaciones.items():
        sobre = comprueba_sobre_malla(pos, curvas)
        anotado, registros = etiquetas_de_curvas(pos, tri, curvas)
        resultado["annotations"][nombre] = {**sobre, **registros,
                                            "teeth": sorted(int(f) for f in curvas)}
        utiles = {f for f in curvas if registros["per_tooth"][str(f)]["usable"]}
        resultado["scores"][nombre] = {
            cand: {"curves": distancia_a_las_curvas(pos, tri, etq, curvas, anotado, utiles),
                   "labels": metricas_de_arcada(pos, tri, etq, anotado)}
            for cand, etq in candidatos.items()}
    for a, b in combinations(anotaciones, 2):
        resultado["between_annotations"][f"{a} / {b}"] = acuerdo_entre_anotaciones(
            anotaciones[a], anotaciones[b])
    return resultado


def _mm(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.2f}"


def _pct(valor: float | None) -> str:
    return "—" if valor is None else f"{100 * valor:.1f} %"


def informe_md(resultado: dict[str, Any]) -> str:
    n = len(resultado["annotations"])
    lineas = [
        "# Margen corona-encía frente a líneas anotadas", "",
        f"Acuerdo con las líneas de {n} anotación(es) sobre este caso. No es validación "
        "clínica ni exactitud general de la segmentación. Unas líneas que se usen para "
        "ajustar un método dejan de servir para puntuarlo.", "",
    ]
    for anot, datos in resultado["annotations"].items():
        malas = {f: r["reason"] for f, r in datos["per_tooth"].items() if not r["usable"]}
        hueco = max(r["max_control_gap_mm"] for r in datos["per_tooth"].values())
        lineas += [f"## Anotación `{anot}`", "",
                   f"{len(datos['teeth'])} piezas: {datos['teeth']}. Puntos de control a "
                   f"{_mm(datos['control_to_mesh_p50_mm'])} mm de la malla (mediana); mayor "
                   f"separación entre dos puntos consecutivos, {_mm(hueco)} mm.", ""]
        if malas:
            lineas += [f"Líneas que no encierran una región: {malas}. Su distancia al margen "
                       "se mide, pero sin signo, y las tasas por etiqueta cuentan esas "
                       "piezas como encía.", ""]
        if datos["disputed_vertices"]:
            lineas += [f"{datos['disputed_vertices']} vértices quedaban dentro de dos "
                       "líneas y se asignaron a la pieza de centro más cercano.", ""]
        notas = resultado["scores"][anot]
        lineas += ["| Candidato | Margen → línea p50 (mm) | p95 (mm) | Línea → margen p50 "
                   "(mm) | Encía marcada diente | Corona marcada encía | Piezas con < 85 % "
                   "de corona |", "|---|---:|---:|---:|---:|---:|---:|"]
        for cand, s in notas.items():
            c, e = s["curves"], s["labels"]
            lineas.append(
                f"| {cand} | {_mm(c['pred_to_curve']['p50_mm'])} "
                f"| {_mm(c['pred_to_curve']['p95_mm'])} "
                f"| {_mm(c['curve_to_pred']['p50_mm'])} | {_pct(e['false_tooth_rate'])} "
                f"| {_pct(e['false_gum_rate'])} "
                f"| {e['teeth']['under_retained']} de {e['teeth']['expert']} |")
        lineas += ["", "Por pieza: distancia mediana del margen a la línea (mm, con el signo "
                   "medio entre paréntesis: + desborde, − sobre-recorte) y corona conservada.",
                   "", "| FDI | " + " | ".join(f"{c}: margen | {c}: corona" for c in notas)
                   + " |", "|---|" + "---:|---:|" * len(notas)]
        for fdi in datos["teeth"]:
            celdas = []
            for s in notas.values():
                c = s["curves"]["per_tooth"].get(str(fdi))
                e = s["labels"]["per_tooth"].get(str(fdi))
                signo = None if c is None else c["signed_mean_mm"]
                celdas.append("—" if c is None else _mm(c["pred_to_curve"]["p50_mm"])
                              + ("" if signo is None else f" ({signo:+.2f})"))
                celdas.append("—" if e is None else _pct(e["retained"]))
            lineas.append(f"| {fdi} | " + " | ".join(celdas) + " |")
        lineas.append("")
        for cand, s in notas.items():
            c = s["curves"]
            if c["only_annotated"] or c["only_predicted"]:
                lineas += [f"`{cand}`: piezas solo anotadas {c['only_annotated']}; solo en "
                           f"el etiquetado {c['only_predicted']}.", ""]
    lineas += ["## Acuerdo entre anotaciones", ""]
    if not resultado["between_annotations"]:
        lineas += ["Una sola anotación: no se puede medir cuánto varía la línea, así que no "
                   "hay suelo con el que comparar las distancias de arriba.", ""]
    for par, d in resultado["between_annotations"].items():
        lineas += [f"- `{par}`: distancia simétrica entre líneas, mediana "
                   f"{_mm(d['overall']['p50_mm'])} mm, p95 {_mm(d['overall']['p95_mm'])} mm. "
                   "Un margen que diste de la línea menos que esto no se distingue de "
                   "correcto.", ""]
    return "\n".join(lineas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anotacion", type=Path, action="append", required=True,
                    help="Carpeta con un `.mrk.json` por pieza. Repetible.")
    ap.add_argument("--stl-cargado-como", choices=("lps", "ras"), required=True,
                    help="Cómo se cargó el STL en el visor de marcado. No se adivina.")
    ap.add_argument("--candidato", action="append", required=True,
                    help="NOMBRE=caso.uos o NOMBRE=fichero.npz:clave. Repetible.")
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    candidatos: dict[str, np.ndarray] = {}
    pos = tri = None
    for texto in args.candidato:
        nombre, p, t, etq = lee_candidato(texto)
        if pos is None:
            pos, tri = p, t
        elif p.shape != pos.shape or not np.array_equal(t, tri) or not np.allclose(p, pos):
            raise SystemExit(f"{nombre}: no indexa la misma malla que el primer candidato")
        candidatos[nombre] = etq
    assert pos is not None and tri is not None
    try:
        anotaciones = {c.name or str(c): lee_anotacion(c, args.stl_cargado_como)
                       for c in args.anotacion}
        if len(anotaciones) != len(args.anotacion):
            raise ValueError("dos carpetas de anotación con el mismo nombre")
        resultado = puntua(pos, tri, candidatos, anotaciones)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    resultado["stl_loaded_as"] = args.stl_cargado_como.upper()

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "margen-anotado.json").write_text(json.dumps(resultado, indent=2) + "\n")
    (args.salida / "margen-anotado.md").write_text(informe_md(resultado))
    for anot, notas in resultado["scores"].items():
        for cand, s in notas.items():
            print(f"{anot} · {cand}: margen a {_mm(s['curves']['pred_to_curve']['p50_mm'])} "
                  f"mm de la línea (mediana), "
                  f"{s['labels']['teeth']['under_retained']} piezas recortadas de más")
    print(f"Informe: {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
