#!/usr/bin/env python
"""Banco de la separación entre dientes: un checkpoint nuevo contra el de hoy, en la reserva.

    ~/.venvs/dental-gpu/bin/python scripts/benchmark_separacion.py predice \\
        --split dev --checkpoint data/processed/a3-checkpoints/normales-ce-baseline.pt
    uv run python scripts/benchmark_separacion.py evalua --split dev --base B.pt --candidato C.pt

**Qué responde.** Si un checkpoint candidato separa mejor cada diente de sus vecinos que el
de hoy, sin numerar peor, sin poner más fantasmas y sin mover el margen gingival. Las dos
predicciones se puntúan en el pipeline REAL: cada arcada se segmenta canonizada contra las
18 referencias de entrenamiento, se vota entre las 3 más confiadas
(`analysis_agents.pose_votada`) y se aplica la limpieza de `scripts/caso_completo.py`. La
pose original de Teeth3DS+ se guarda también, como techo, para medir cuánto depende el
resultado de la pose.

**Splits.** `dev` son los 20 pacientes de siempre (`benchmark_margen.particion`): sirve para
elegir. `reserva` son 60 pacientes del `private-testing-set` de 3DTeethSeg'22 que nadie ha
visto (`scripts/fetch_teeth3ds_osf.py`), y se evalúa **una sola vez**: `evalua --split
reserva` se niega a sobrescribir su propio resultado.

Las cifras son de Teeth3DS+, no de los casos clínicos propios.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RESUMEN_EN = "Scores tooth-to-tooth separation of a candidate FDI checkpoint against today's."
RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))

from analysis_agents.margen_metricas import (  # noqa: E402
    CORONA_MINIMA_CONSERVADA,
    agrega,
    identidad_de_piezas,
    metricas_de_arcada,
)
from analysis_agents.pose_votada import vota  # noqa: E402


def _carga(nombre: str, ruta: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(nombre, modulo)
    spec.loader.exec_module(modulo)
    return sys.modules[nombre]


bp = _carga("benchmark_pose", RAIZ / "scripts/benchmark_pose.py")
bm = bp.bm
osf = _carga("fetch_teeth3ds_osf", RAIZ / "scripts/fetch_teeth3ds_osf.py")

VERSION_PROTOCOLO = "1"
SALIDA = RAIZ / "data/processed/benchmark-separacion"
NATIVA = "nativa"

# ── Criterios de aceptación, fijados el 2026-11-20 ANTES de entrenar el candidato ──────
#
# C = checkpoint candidato, B = el de hoy (`normales-ce-baseline.pt`). Los dos en el
# pipeline real (voto entre 3 de 18 referencias + limpieza actual), arcada a arcada.
#
# Puerta: C tiene que cumplirlos TODOS en `dev`; si no, la reserva no se toca y el
# resultado se escribe como negativo. La reserva se evalúa UNA vez, con estos umbrales.
MEJORA_CONTACTO = 0.75       # K1: mediana de contactos de C ≤ 0,75 × la de B …
ARCADAS_QUE_MEJORAN = 0.70   # K1: … y la mediana por arcada mejora en ≥ 70 % de ellas
MEJORA_SANGRADO = 0.75       # K2: piezas que sangran > 10 % con el vecino: C ≤ 0,75 × B
#                              K3: piezas bien numeradas C ≥ B, y fantasmas C ≤ B
#                              K4: (techo nativo − voto) de C ≤ el de B
SUBIDA_FALSO_DIENTE = 0.01   # K5: falso diente medio ≤ B + 0,01 …
SUBIDA_FALSA_ENCIA = 0.005   # K5: … falsa encía ≤ B + 0,5 puntos …
PIEZAS_DANADAS = 0.005       # K5: … y piezas que pasan a < 85 % de corona ≤ 0,5 %
RETROCESO_FDI = 0.005        # K6: acierto FDI por vértice de diente ≥ B − 0,005


# ── Datos ───────────────────────────────────────────────────────────────────────

def arcadas(split: str, limite: int | None = None) -> list[tuple[str, str]]:
    if split == "reserva":
        ids = osf.ids_reserva()
        pares = [(p, a) for p in ids for a in bm.ARCADAS]
        return pares[:limite] if limite else pares
    return bm.arcadas_de(split, limite)


def carga(split: str, paciente: str, arcada: str):  # type: ignore[no-untyped-def]
    """`(posiciones, caras, etiquetas de experto)` como `benchmark_margen.carga_arcada`."""
    if split != "reserva":
        return bm.carga_arcada(paciente, arcada)
    malla = bm.parse_obj(osf.MALLAS / paciente / f"{paciente}_{arcada}.obj")
    pos = np.asarray(malla["positions"], dtype=np.float64)
    tri = np.asarray(malla["faces"], dtype=np.int64)
    datos = json.loads((osf.ETIQUETAS / paciente / f"{paciente}_{arcada}.json").read_text())
    experto = np.asarray(datos["labels"], dtype=np.int64)
    if len(experto) != len(pos):
        raise ValueError(f"{paciente}_{arcada}: {len(experto)} etiquetas, {len(pos)} vértices")
    return pos, tri, experto


def sha_de(checkpoint: Path) -> str:
    return hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()


def ruta(sha: str, split: str, paciente: str, arcada: str) -> Path:
    return SALIDA / sha[:12] / split / f"{paciente}_{arcada}.npz"


def registro(argv: list[str]) -> dict:
    cc = _carga("caso_completo", RAIZ / "scripts/caso_completo.py")
    from datetime import UTC, datetime

    return {"schema": "histora-run-record/1.0", "argv": list(argv),
            "code": cc.revision_de_codigo(), "environment": cc.entorno_de_software(),
            "started_at": datetime.now(UTC).isoformat(),
            "scope": "Identifies code and environment of one benchmark run"}


# ── Predicción (GPU) ────────────────────────────────────────────────────────────

def predice(args: argparse.Namespace) -> int:
    import torch
    from analysis_agents.pose_votada import confianza
    from fusion_agents.marco import a_marco_de, marco_canonico

    seg = _carga("segmentar_fdi", RAIZ / "scripts/segmentar_fdi.py")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, codigos, n_clases, meta = seg.carga_modelo(args.checkpoint, dev)
    sha = sha_de(args.checkpoint)
    print(f"modelo {meta['name']} · trozo {meta['num_points']} · sha256 {sha[:12]} · {dev}")
    refs = bp.referencias()
    marcos = {(r, a): marco_canonico(np.asarray(
        bm.parse_obj(bm.MALLAS / r / f"{r}_{a}.obj")["positions"], dtype=np.float64))[:2]
        for r in refs for a in bm.ARCADAS}
    pares = arcadas(args.split, args.limite)
    destino_registro = SALIDA / sha[:12] / args.split / "run-record.json"
    destino_registro.parent.mkdir(parents=True, exist_ok=True)
    destino_registro.write_text(json.dumps(registro(sys.argv), indent=2) + "\n")
    for k, (paciente, arcada) in enumerate(pares, 1):
        destino = ruta(sha, args.split, paciente, arcada)
        if destino.is_file():
            continue
        pos, tri, _ = carga(args.split, paciente, arcada)
        normales = seg.normales_vtk(pos.astype(np.float32), tri).astype(np.float64)
        origen = marco_canonico(pos)
        salida: dict[str, np.ndarray] = {"razon": np.float32(origen[2])}
        for r in [NATIVA, *refs]:
            if r == NATIVA:
                v_ent, n_ent = pos, normales
            else:
                v_ent, n_ent = a_marco_de(pos, origen[:2], marcos[(r, arcada)],
                                          normales=normales)
            torch.manual_seed(0)
            logp, instancias = seg.infiere(modelo, codigos, n_clases, pos, v_ent, n_ent,
                                           arcada, dev, trozo=meta["num_points"])
            etiquetas = seg.etiquetas_de(instancias, len(pos))
            salida[f"et_{r}"] = etiquetas
            salida[f"conf_{r}"] = np.float32(confianza(logp, etiquetas))
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destino, **salida)
        print(f"  [{k}/{len(pares)}] {paciente}_{arcada}: razón {float(origen[2]):.2f}",
              flush=True)
    return 0


# ── Evaluación (CPU) ────────────────────────────────────────────────────────────

def _etiquetas(z, refs: list[str]) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    """`(voto, nativa)`, las dos sin limpiar."""
    seg = {r: z[f"et_{r}"].astype(np.int64) for r in refs}
    conf = {r: float(z[f"conf_{r}"]) for r in refs}
    votada, _ = vota(seg, conf)
    return votada, z[f"et_{NATIVA}"].astype(np.int64)


def evalua_arcada(trabajo: tuple[str, str, str, dict[str, str]]) -> dict:
    split, paciente, arcada, shas = trabajo
    pos, tri, experto = carga(split, paciente, arcada)
    refs = bp.referencias()
    fila: dict = {"patient": paciente, "jaw": arcada, "models": {}}
    for papel, sha in shas.items():
        z = np.load(ruta(sha, split, paciente, arcada))
        votada, nativa = _etiquetas(z, refs)
        votada = bm.limpieza_actual(pos, votada)
        nativa = bm.limpieza_actual(pos, nativa)
        m = metricas_de_arcada(pos, tri, votada, experto)
        fila["models"][papel] = {"voted": m,
                                 "native_identity": identidad_de_piezas(nativa, experto)}
        fila["orientation_ratio"] = float(z["razon"])
    return fila


def _medias(filas: list[dict], papel: str, *claves: str) -> float:
    v = []
    for f in filas:
        x = f["models"][papel]["voted"]
        for c in claves:
            x = x[c]
        if x is not None:
            v.append(float(x))
    return float(np.mean(v)) if v else float("nan")


def criterios(filas: list[dict]) -> dict:
    """K1-K6 de C frente a B. Ver las constantes de arriba."""
    resumen = {p: agrega([f["models"][p]["voted"] for f in filas]) for p in ("B", "C")}
    sep = {p: resumen[p]["separation"] for p in ("B", "C")}
    ident = {p: resumen[p]["identity"] for p in ("B", "C")}
    nativas = {p: sum(f["models"][p]["native_identity"]["correct"] for f in filas)
               for p in ("B", "C")}
    pares = [(f["models"]["C"]["voted"]["separation"]["contact_p50_mm"],
              f["models"]["B"]["voted"]["separation"]["contact_p50_mm"]) for f in filas]
    pares = [(c, b) for c, b in pares if c is not None and b is not None]
    mejoran = float(np.mean([c < b for c, b in pares])) if pares else 0.0
    piezas = nuevas = 0
    for f in filas:
        antes = f["models"]["B"]["voted"]["per_tooth"]
        despues = f["models"]["C"]["voted"]["per_tooth"]
        piezas += len(antes)
        nuevas += sum(1 for t, v in antes.items() if v["retained"] >= CORONA_MINIMA_CONSERVADA
                      and despues[t]["retained"] < CORONA_MINIMA_CONSERVADA)
    ft = {p: _medias(filas, p, "false_tooth_rate") for p in ("B", "C")}
    fg = {p: _medias(filas, p, "false_gum_rate") for p in ("B", "C")}
    fdi = {p: _medias(filas, p, "tooth_fdi_agreement") for p in ("B", "C")}
    cb, cc = sep["B"]["contact_p50_mm_pooled"], sep["C"]["contact_p50_mm_pooled"]
    hueco = {p: nativas[p] - ident[p]["correct"] for p in ("B", "C")}
    k = {
        "K1_contact": {"B": cb, "C": cc, "arches_improved": mejoran, "arches": len(pares),
                       "pass": bool(cc <= MEJORA_CONTACTO * cb
                                    and mejoran >= ARCADAS_QUE_MEJORAN)},
        "K2_bleeding": {"B": sep["B"]["teeth_bleeding"], "C": sep["C"]["teeth_bleeding"],
                        "teeth": sep["B"]["teeth"],
                        "pass": bool(sep["C"]["teeth_bleeding"]
                                     <= MEJORA_SANGRADO * sep["B"]["teeth_bleeding"])},
        "K3_numbering": {"correct_B": ident["B"]["correct"], "correct_C": ident["C"]["correct"],
                         "ghost_B": ident["B"]["ghost"], "ghost_C": ident["C"]["ghost"],
                         "expert": ident["B"]["expert"],
                         "pass": bool(ident["C"]["correct"] >= ident["B"]["correct"]
                                      and ident["C"]["ghost"] <= ident["B"]["ghost"])},
        "K4_pose": {"native_B": nativas["B"], "native_C": nativas["C"],
                    "gap_B": hueco["B"], "gap_C": hueco["C"],
                    "pass": bool(hueco["C"] <= hueco["B"])},
        "K5_margin": {"false_tooth_B": ft["B"], "false_tooth_C": ft["C"],
                      "false_gum_B": fg["B"], "false_gum_C": fg["C"],
                      "newly_under_retained": nuevas, "teeth": piezas,
                      "pass": bool(ft["C"] <= ft["B"] + SUBIDA_FALSO_DIENTE
                                   and fg["C"] <= fg["B"] + SUBIDA_FALSA_ENCIA
                                   and nuevas <= PIEZAS_DANADAS * piezas)},
        "K6_tooth_fdi": {"B": fdi["B"], "C": fdi["C"],
                         "pass": bool(fdi["C"] >= fdi["B"] - RETROCESO_FDI)},
    }
    k["all_pass"] = all(v["pass"] for v in k.values() if isinstance(v, dict))
    return {"criteria": k, "summary": resumen}


def informe_md(salida: dict) -> str:
    k = salida["acceptance"]["criteria"]
    s = salida["acceptance"]["summary"]
    si = lambda v: "cumple" if v else "**NO cumple**"  # noqa: E731

    def fila(p: str) -> str:
        i, sp = s[p]["identity"], s[p]["separation"]
        return (f"| {p} | {sp['contact_p50_mm_pooled']:.2f} | {sp['contacts_over_1mm']} de "
                f"{sp['resolved']} | {sp['teeth_bleeding']} de {sp['teeth']} | {i['correct']} "
                f"de {i['expert']} | {i['shifted']} | {i['ghost']} | {i['split']} | "
                f"{i['missing']} | {s[p]['false_tooth_rate']['mean']:.3f} |")

    k1, k2, k3, k4, k5, k6 = (k[x] for x in ("K1_contact", "K2_bleeding", "K3_numbering",
                                              "K4_pose", "K5_margin", "K6_tooth_fdi"))
    return "\n".join([
        f"# Separación entre dientes · split `{salida['split']}`", "",
        f"{salida['arches']} arcadas de {salida['patients']} pacientes. B = "
        f"`{salida['checkpoints']['B'][:12]}`, C = `{salida['checkpoints']['C'][:12]}`. "
        f"Pipeline real: voto entre 3 de 18 referencias + limpieza. Protocolo "
        f"{salida['protocol']}.", "",
        "| | Contacto p50 (mm) | Contactos > 1 mm | Piezas que sangran > 10 % | Correctas "
        "| Desplazadas | Fantasmas | Fragmentos | Ausentes | Falso diente |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", fila("B"), fila("C"), "",
        "| Criterio | Medido | |", "|---|---|---|",
        f"| K1 contacto | {k1['B']:.3f} → {k1['C']:.3f} mm; mejora en el "
        f"{k1['arches_improved']:.0%} de {k1['arches']} arcadas | {si(k1['pass'])} |",
        f"| K2 sangrado | {k2['B']} → {k2['C']} de {k2['teeth']} | {si(k2['pass'])} |",
        f"| K3 numeración | correctas {k3['correct_B']} → {k3['correct_C']}; fantasmas "
        f"{k3['ghost_B']} → {k3['ghost_C']} | {si(k3['pass'])} |",
        f"| K4 pose | techo − voto {k4['gap_B']} → {k4['gap_C']} (techo {k4['native_B']} → "
        f"{k4['native_C']}) | {si(k4['pass'])} |",
        f"| K5 margen | falso diente {k5['false_tooth_B']:.3f} → {k5['false_tooth_C']:.3f}; "
        f"falsa encía {k5['false_gum_B']:.4f} → {k5['false_gum_C']:.4f}; dañadas "
        f"{k5['newly_under_retained']} de {k5['teeth']} | {si(k5['pass'])} |",
        f"| K6 FDI por vértice | {k6['B']:.4f} → {k6['C']:.4f} | {si(k6['pass'])} |", "",
        f"**Veredicto: {'cumple todos' if k['all_pass'] else 'NO cumple todos'}.**", "",
        "Cifras sobre Teeth3DS+; no miden los casos clínicos propios.", ""])


def evalua(args: argparse.Namespace) -> int:
    shas = {"B": sha_de(args.base), "C": sha_de(args.candidato)}
    destino = SALIDA / f"resumen-{args.split}-{shas['C'][:12]}.json"
    if args.split == "reserva" and destino.is_file():
        print(f"⚠ La reserva ya se evaluó con este candidato ({destino}). Se evalúa UNA vez.",
              file=sys.stderr)
        return 3
    pares = arcadas(args.split, args.limite)
    faltan = [(p, a, x) for p, a in pares for x in shas.values()
              if not ruta(x, args.split, p, a).is_file()]
    if faltan:
        print(f"Faltan {len(faltan)} predicciones; ejecuta `predice`.", file=sys.stderr)
        return 2
    rec = registro(sys.argv)
    trabajos = [(args.split, p, a, shas) for p, a in pares]
    with ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        filas = list(grupo.map(evalua_arcada, trabajos))
    salida = {
        "protocol": VERSION_PROTOCOLO, "split": args.split, "arches": len(filas),
        "patients": len({p for p, _ in pares}), "checkpoints": shas,
        "acceptance": criterios(filas), "run_record": rec,
        "per_arch": [{"patient": f["patient"], "jaw": f["jaw"],
                      "orientation_ratio": f["orientation_ratio"],
                      **{p: {"separation": {k: v for k, v in m["voted"]["separation"].items()
                                            if k != "per_tooth"},
                             "identity": {k: v for k, v in m["voted"]["identity"].items()
                                          if k != "per_label"},
                             "native_correct": m["native_identity"]["correct"]}
                         for p, m in f["models"].items()}} for f in filas],
        "scope": "Teeth3DS+ only; not the project's clinical cases",
    }
    if args.limite:
        destino = destino.with_name(destino.stem + f"-limite{args.limite}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    destino.with_suffix(".md").write_text(informe_md(salida))
    print(informe_md(salida))
    print(f"Informe: {destino}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    p = ordenes.add_parser("predice")
    p.add_argument("--checkpoint", type=Path, required=True)
    e = ordenes.add_parser("evalua")
    e.add_argument("--base", type=Path, default=bm.CHECKPOINT)
    e.add_argument("--candidato", type=Path, required=True)
    e.add_argument("--trabajos", type=int, default=8)
    for sub, funcion in ((p, predice), (e, evalua)):
        sub.add_argument("--split", choices=("dev", "reserva"), required=True)
        sub.add_argument("--limite", type=int, help="Solo las N primeras arcadas.")
        sub.set_defaults(funcion=funcion)
    args = ap.parse_args()
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
