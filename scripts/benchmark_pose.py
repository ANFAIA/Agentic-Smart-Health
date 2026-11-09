#!/usr/bin/env python
"""Banco de pruebas de la pose: numerar los dientes con una referencia o votando entre varias.

    ~/.venvs/dental-gpu/bin/python scripts/benchmark_pose.py predice --split dev
    uv run python scripts/benchmark_pose.py evalua --split dev

**Qué responde.** Si segmentar el mismo escaneo con varias poses —cada una canonizada
contra una arcada de referencia distinta— y votar entre las más confiadas
(`analysis_agents.pose_votada`) numera los dientes tan bien como la pose original de
Teeth3DS+, y mejor que una sola referencia, que es lo que hace hoy `segmentar_fdi.py`.

**Por qué Teeth3DS+ sirve aunque ya esté en la pose del modelo.** La canonización calcula
los ejes de la forma de la propia malla, así que girar un escaneo al azar apenas cambia el
resultado. Lo que cambia es la referencia: cada arcada tiene una forma algo distinta. Así
que aquí cada arcada se segmenta en su pose original —el techo, que un escáner real no
ofrece— y canonizada contra cada referencia.

Las referencias son los 18 primeros pacientes de ENTRENAMIENTO de la partición del modelo:
solo prestan su pose. La medida se hace sobre `dev` y `eval`, que el modelo no vio.

⚠️ La configuración —18 referencias, voto entre 3, sin la pose original— se eligió en
`dev`. `eval` se ejecuta una sola vez, con los criterios de abajo fijados antes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RESUMEN_EN = "Scores tooth numbering with one canonicalisation reference against voting."
RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))

from analysis_agents.margen_metricas import identidad_de_piezas  # noqa: E402
from analysis_agents.pose_votada import MEJORES, vota  # noqa: E402

_spec = importlib.util.spec_from_file_location("benchmark_margen",
                                               RAIZ / "scripts/benchmark_margen.py")
assert _spec and _spec.loader
bm = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, bm)
_spec.loader.exec_module(bm)

VERSION_PROTOCOLO = "1"
N_REFERENCIAS = 18
NATIVA = "nativa"
SALIDA = RAIZ / "data/processed/benchmark-pose"

# ── Criterios de aceptación, fijados el 2026-11-09 antes de ejecutar `eval` ─────────────
#
# V = votación entre las 3 más confiadas de 18 referencias, sin la pose original.
# Techo = pose original de Teeth3DS+. Práctica actual = una sola referencia; como no hay
# una referencia «de hoy», se compara con la mediana de las 18 en cada arcada.
DISTANCIA_AL_TECHO = 0.01        # P1: piezas correctas ≥ techo − 1 % de las del experto
ARCADAS_PEOR_QUE_MEDIANA = 0.10  # P2: peor que la mediana de una referencia en ≤ 10 %
ARCADAS_CON_ORDEN = 0.80         # P4: confianza ordena el acierto en ≥ 80 % de arcadas
RETROCESO_FDI = 0.01             # P5: acierto FDI por vértice ≥ techo − 0,01
RETROCESO_BINARIO = 0.005        # P6: acuerdo diente/encía ≥ techo − 0,005


def referencias() -> list[str]:
    return bm.particion(bm.pacientes())["train"][:N_REFERENCIAS]


def ruta(paciente: str, arcada: str) -> Path:
    return SALIDA / "pred" / f"{paciente}_{arcada}.npz"


def predice(args: argparse.Namespace) -> int:
    """Segmenta cada arcada en su pose y contra cada referencia. Necesita GPU."""
    import torch

    spec = importlib.util.spec_from_file_location("segmentar_fdi",
                                                  RAIZ / "scripts/segmentar_fdi.py")
    assert spec and spec.loader
    seg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = seg
    spec.loader.exec_module(seg)
    from analysis_agents.pose_votada import confianza
    from fusion_agents.marco import a_marco_de, marco_canonico

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, codigos, n_clases, _ = seg.carga_modelo(bm.CHECKPOINT, dev)
    refs = referencias()
    marcos = {(r, a): marco_canonico(np.asarray(
        bm.parse_obj(bm.MALLAS / r / f"{r}_{a}.obj")["positions"], dtype=np.float64))[:2]
        for r in refs for a in bm.ARCADAS}
    pares = bm.arcadas_de(args.split, args.limite)
    for k, (paciente, arcada) in enumerate(pares, 1):
        destino = ruta(paciente, arcada)
        if destino.is_file() and not args.rehace:
            continue
        pos, tri, _ = bm.carga_arcada(paciente, arcada)
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
                                           arcada, dev)
            etiquetas = seg.etiquetas_de(instancias, len(pos))
            salida[f"et_{r}"] = etiquetas
            salida[f"conf_{r}"] = np.float32(confianza(logp, etiquetas))
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destino, **salida)
        print(f"  [{k}/{len(pares)}] {paciente}_{arcada}: razón {float(origen[2]):.2f}",
              flush=True)
    return 0


def _resumen(etiquetas: np.ndarray, experto: np.ndarray) -> dict:
    i = identidad_de_piezas(etiquetas, experto)
    e = experto > 0
    return {"correct": i["correct"], "shifted": i["shifted"], "ghost": i["ghost"],
            "perfect": bool(i["perfect"]), "expert": i["expert"],
            "tooth_fdi": float((etiquetas[e] == experto[e]).mean()),
            "binary": float(((etiquetas > 0) == e).mean())}


def evalua_arcada(par: tuple[str, str]) -> dict:
    from scipy.stats import spearmanr

    paciente, arcada = par
    _, _, experto = bm.carga_arcada(paciente, arcada)
    z = np.load(ruta(paciente, arcada))
    refs = referencias()
    seg = {r: z[f"et_{r}"].astype(np.int64) for r in refs}
    conf = {r: float(z[f"conf_{r}"]) for r in refs}
    votada, registro = vota(seg, conf)
    por_ref = {r: _resumen(seg[r], experto) for r in refs}
    correctas = sorted(v["correct"] for v in por_ref.values())
    rho = spearmanr([conf[r] for r in refs], [por_ref[r]["tooth_fdi"] for r in refs])[0]
    return {
        "patient": paciente, "jaw": arcada, "orientation_ratio": float(z["razon"]),
        "native": _resumen(z[f"et_{NATIVA}"].astype(np.int64), experto),
        "voted": _resumen(votada, experto), "chosen": registro["chosen"],
        "single_reference_correct": {"min": correctas[0], "max": correctas[-1],
                                     "median": float(np.median(correctas))},
        "confidence_accuracy_spearman": None if rho != rho else float(rho),
    }


def criterios(filas: list[dict]) -> dict:
    expertas = sum(f["native"]["expert"] for f in filas)
    v = {k: sum(f["voted"][k] for f in filas) for k in ("correct", "ghost")}
    t = {k: sum(f["native"][k] for f in filas) for k in ("correct", "ghost")}
    mediana = sum(f["single_reference_correct"]["median"] for f in filas)
    peor = sum(f["voted"]["correct"] < f["single_reference_correct"]["median"] for f in filas)
    rhos = [f["confidence_accuracy_spearman"] for f in filas
            if f["confidence_accuracy_spearman"] is not None]
    fdi_v = float(np.mean([f["voted"]["tooth_fdi"] for f in filas]))
    fdi_t = float(np.mean([f["native"]["tooth_fdi"] for f in filas]))
    bin_v = float(np.mean([f["voted"]["binary"] for f in filas]))
    bin_t = float(np.mean([f["native"]["binary"] for f in filas]))
    c = {
        "P1_ceiling": {"voted": v["correct"], "native": t["correct"],
                       "pass": v["correct"] >= t["correct"] - DISTANCIA_AL_TECHO * expertas},
        "P2_vs_single": {"voted": v["correct"], "median_single": mediana,
                         "arches_worse": peor,
                         "pass": v["correct"] > mediana
                         and peor <= ARCADAS_PEOR_QUE_MEDIANA * len(filas)},
        "P3_ghosts": {"voted": v["ghost"], "native": t["ghost"],
                      "pass": v["ghost"] <= t["ghost"]},
        "P4_confidence_order": {"positive": sum(r > 0 for r in rhos), "arches": len(rhos),
                                "pass": sum(r > 0 for r in rhos) >= ARCADAS_CON_ORDEN * len(rhos)},
        "P5_tooth_fdi": {"voted": fdi_v, "native": fdi_t,
                         "pass": fdi_v >= fdi_t - RETROCESO_FDI},
        "P6_binary": {"voted": bin_v, "native": bin_t,
                      "pass": bin_v >= bin_t - RETROCESO_BINARIO},
    }
    c["all_pass"] = all(x["pass"] for x in c.values() if isinstance(x, dict))
    return c


def informe_md(salida: dict) -> str:
    c = salida["acceptance"]
    s = salida["totals"]

    def fila(nombre: str, d: dict) -> str:
        return (f"| {nombre} | {d['correct']} de {d['expert']} | {d['shifted']} | "
                f"{d['ghost']} | {d['perfect']} de {salida['arches']} |")

    lineas = [
        f"# Banco de pruebas de la pose · split `{salida['split']}`", "",
        f"{salida['arches']} arcadas de {salida['patients']} pacientes de Teeth3DS+ que el "
        f"modelo no vio. {N_REFERENCIAS} referencias de entrenamiento; votan las "
        f"{MEJORES} más confiadas. Protocolo {salida['protocol']}.", "",
        "| Método | Correctas | Desplazadas | Fantasmas | Arcadas perfectas |",
        "|---|---:|---:|---:|---:|",
        fila("Pose original (techo)", s["native"]),
        f"| Una sola referencia | {s['single_min']}–{s['single_max']} (mediana por arcada "
        f"sumada {s['single_median']:.0f}) | | | |",
        fila(f"Votación entre {MEJORES} de {N_REFERENCIAS}", s["voted"]), "",
        "## Criterios", "",
        *[f"- **{k}**: {'cumple' if v['pass'] else 'NO cumple'} · "
          + ", ".join(f"{a} {b:.4f}" if isinstance(b, float) else f"{a} {b}"
                      for a, b in v.items() if a != "pass")
          for k, v in c.items() if isinstance(v, dict)],
        "", f"**Veredicto:** {'cumple todos' if c['all_pass'] else 'NO cumple todos'}.", "",
        "Estas cifras son sobre Teeth3DS+. No miden los casos clínicos propios.", ""]
    return "\n".join(lineas)


def evalua(args: argparse.Namespace) -> int:
    pares = bm.arcadas_de(args.split, args.limite)
    faltan = [p for p in pares if not ruta(*p).is_file()]
    if faltan:
        print(f"Faltan {len(faltan)} predicciones; ejecuta `predice --split {args.split}`.",
              file=sys.stderr)
        return 2
    with ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        filas = list(grupo.map(evalua_arcada, pares))

    def suma(clave: str) -> dict:
        d = {k: int(sum(f[clave][k] for f in filas))
             for k in ("correct", "shifted", "ghost", "perfect", "expert")}
        return d

    salida = {
        "protocol": VERSION_PROTOCOLO, "split": args.split, "arches": len(filas),
        "patients": len({p for p, _ in pares}), "references": referencias(),
        "voters": MEJORES,
        "totals": {"native": suma("native"), "voted": suma("voted"),
                   "single_min": int(sum(f["single_reference_correct"]["min"] for f in filas)),
                   "single_max": int(sum(f["single_reference_correct"]["max"] for f in filas)),
                   "single_median": float(sum(f["single_reference_correct"]["median"]
                                              for f in filas))},
        "acceptance": criterios(filas),
        "per_arch": filas,
        "scope": "Teeth3DS+ held-out patients only; not the project's clinical cases",
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / f"resumen-{args.split}.json").write_text(json.dumps(salida, indent=2) + "\n")
    (SALIDA / f"resumen-{args.split}.md").write_text(informe_md(salida))
    print(informe_md(salida))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    for nombre, funcion in (("predice", predice), ("evalua", evalua)):
        sub = ordenes.add_parser(nombre)
        sub.add_argument("--split", choices=("dev", "eval"), required=True)
        sub.add_argument("--limite", type=int, help="Solo las N primeras arcadas.")
        sub.set_defaults(funcion=funcion)
        if nombre == "predice":
            sub.add_argument("--rehace", action="store_true")
        else:
            sub.add_argument("--trabajos", type=int, default=6)
    args = ap.parse_args()
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
