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
from typing import Any

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
from analysis_agents.testigo import rescata_coronas  # noqa: E402


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

# Las dos reservas viven en el árbol de OSF. La segunda (§6.11) se predice con este mismo
# script para que C2 corra exactamente igual que en la primera.
RESERVAS = {"reserva": osf.ids_reserva, "reserva-2": osf.ids_reserva_2,
            "reserva-3": osf.ids_reserva_3}


def arcadas(split: str, limite: int | None = None) -> list[tuple[str, str]]:
    if split in RESERVAS:
        ids = RESERVAS[split]()
        pares = [(p, a) for p in ids for a in bm.ARCADAS]
        return pares[:limite] if limite else pares
    return bm.arcadas_de(split, limite)


def carga(split: str, paciente: str, arcada: str):  # type: ignore[no-untyped-def]
    """`(posiciones, caras, etiquetas de experto)` como `benchmark_margen.carga_arcada`."""
    if split not in RESERVAS:
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


def ruta_borde(sha: str, split: str, paciente: str, arcada: str) -> Path:
    """§6.19: la predicción con el marco orientado por el borde. Solo existe donde la regla
    da la vuelta a la arcada; en las demás el marco es el mismo y se usa `ruta`."""
    return SALIDA / f"{sha[:12]}-borde" / split / f"{paciente}_{arcada}.npz"


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
    from analysis_agents.orientacion import marco_orientado
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
    i, n = (int(x) for x in args.parte.split("/"))
    pares = pares[i::n]  # la agregación es de un núcleo: se reparte en procesos
    # ⚠️ Con --orienta-borde el registro va al directorio del borde: escribirlo en el de
    # siempre sobrescribió el de la predicción normal de `dev` el 2026-11-24.
    carpeta = f"{sha[:12]}-borde" if args.orienta_borde else sha[:12]
    destino_registro = SALIDA / carpeta / args.split / "run-record.json"
    destino_registro.parent.mkdir(parents=True, exist_ok=True)
    destino_registro.write_text(json.dumps(registro(sys.argv), indent=2) + "\n")
    for k, (paciente, arcada) in enumerate(pares, 1):
        destino = (ruta_borde if args.orienta_borde else ruta)(sha, args.split, paciente, arcada)
        if destino.is_file():
            continue
        pos, tri, _ = carga(args.split, paciente, arcada)
        if args.orienta_borde:
            # §6.19: solo se predice donde el borde da la vuelta al marco; sin la pose nativa,
            # que no depende del marco y ya está en `ruta`.
            c, R, razon, volteado = marco_orientado(pos, tri)
            if not volteado:
                continue
            origen, poses = (c, R, razon), refs
        else:
            origen, poses = marco_canonico(pos), [NATIVA, *refs]
        normales = seg.normales_vtk(pos.astype(np.float32), tri).astype(np.float64)
        salida: dict[str, Any] = {"razon": np.float32(origen[2])}
        for r in poses:
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
        temporal = destino.with_suffix(".tmp.npz")
        np.savez_compressed(temporal, **salida)
        temporal.replace(destino)
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
    k: dict[str, Any] = {
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


# ── Elección en `dev`, fijada el 2026-11-20 antes de ver ningún candidato ────────────────
#
# Tres configuraciones como mucho, en este orden:
#   1. 16.384 puntos, ±15°;   2. 8.192 puntos, ±15°;
#   3. la resolución que gane entre 1 y 2, con ±30°.
# Gana la resolución con menor contacto p50 en `dev` entre las que cumplen K3 (no numerar
# peor); a igualdad, la de menos piezas que sangran. El candidato final es, de los tres, el
# de menor contacto p50 entre los que cumplen K1-K6 en `dev`. Si ninguno los cumple todos,
# la reserva no se toca. La semilla extra que preveía el plan no se entrena, por tiempo.

def _resumen_dev(checkpoint: Path) -> dict:
    ruta_resumen = SALIDA / f"resumen-dev-{sha_de(checkpoint)[:12]}.json"
    if not ruta_resumen.is_file():
        raise SystemExit(f"Falta {ruta_resumen}: ejecuta `evalua --split dev` con {checkpoint}.")
    return json.loads(ruta_resumen.read_text())["acceptance"]


def elige_candidato(resumenes: dict[str, dict], modo: str) -> dict:
    """`resolucion`: el mejor que cumple K3. `final`: el mejor que cumple K1-K6."""
    validos = {k: v for k, v in resumenes.items()
               if (v["criteria"]["K3_numbering"]["pass"] if modo == "resolucion"
                   else v["criteria"]["all_pass"])}
    if not validos:
        return {"mode": modo, "chosen": None,
                "reason": "ninguno cumple " + ("K3" if modo == "resolucion" else "K1-K6")}
    orden = sorted(validos, key=lambda k: (validos[k]["criteria"]["K1_contact"]["C"],
                                           validos[k]["criteria"]["K2_bleeding"]["C"]))
    return {"mode": modo, "chosen": orden[0], "ranking": orden,
            "contact_p50_mm": {k: validos[k]["criteria"]["K1_contact"]["C"] for k in orden}}


def elige(args: argparse.Namespace) -> int:
    resumenes = {str(c): _resumen_dev(c) for c in args.candidatos}
    eleccion = elige_candidato(resumenes, args.modo)
    eleccion["candidates"] = {k: {"all_pass": v["criteria"]["all_pass"],
                                  **{c: v["criteria"][c]["pass"] for c in v["criteria"]
                                     if c.startswith("K")}}
                              for k, v in resumenes.items()}
    destino = SALIDA / f"eleccion-{args.modo}.json"
    destino.write_text(json.dumps(eleccion, indent=2) + "\n")
    print(json.dumps(eleccion, indent=2))
    return 0 if eleccion["chosen"] else 4


# ── compara: §6.15, la lectura fijada antes de entrenar C2-s1 y D ─────────────────────────
#
# C2-s1 es C2 con otra semilla y D es C2 con trozos de 2.048 puntos. |C2-s1 − C2| es el ruido
# de semilla de cada cifra; B → D se atribuye a los datos y D → C2 a la resolución solo si la
# diferencia es MAYOR que ese ruido. Nada de esto elige ni cambia el defecto.

CIFRAS_6_15 = {  # cifra: (criterio, clave de B, clave del candidato)
    "contact_p50_mm": ("K1_contact", "B", "C"),
    "bleeding_teeth": ("K2_bleeding", "B", "C"),
    "correct": ("K3_numbering", "correct_B", "correct_C"),
    "ghosts": ("K3_numbering", "ghost_B", "ghost_C"),
    "pose_gap": ("K4_pose", "gap_B", "gap_C"),
    "false_tooth": ("K5_margin", "false_tooth_B", "false_tooth_C"),
    "false_gum": ("K5_margin", "false_gum_B", "false_gum_C"),
    # Respecto a B, así que la de B es 0 por definición.
    "newly_under_retained": ("K5_margin", None, "newly_under_retained"),
    "tooth_fdi": ("K6_tooth_fdi", "B", "C"),
}


def lee_6_15(aceptacion: dict[str, dict]) -> dict[str, Any]:
    """`aceptacion`: `{"C2": …, "C2-s1": …, "D": …}`, el bloque `acceptance` de cada resumen
    contra B. Devuelve, por cifra, los cuatro valores, el ruido y a qué se atribuye cada salto."""
    tabla: dict[str, Any] = {}
    for cifra, (k, clave_b, clave_c) in CIFRAS_6_15.items():
        v = {m: float(a["criteria"][k][clave_c]) for m, a in aceptacion.items()}
        b = float(aceptacion["C2"]["criteria"][k][clave_b]) if clave_b else 0.0
        ruido = abs(v["C2-s1"] - v["C2"])
        datos, resolucion = v["D"] - b, v["C2"] - v["D"]
        tabla[cifra] = {"B": b, **v, "seed_noise": ruido,
                        "data_effect": datos, "data_exceeds_noise": abs(datos) > ruido,
                        "resolution_effect": resolucion,
                        "resolution_exceeds_noise": abs(resolucion) > ruido}
    return {"figures": tabla,
            "all_pass_vs_B": {m: bool(a["criteria"]["all_pass"]) for m, a in aceptacion.items()}}


def informe_6_15(salida: dict) -> str:
    def f(x: float) -> str:
        return f"{x:.3f}" if abs(x) < 10 and x != int(x) else f"{x:.0f}"
    lineas = [f"# §6.15 · semilla y resolución frente a datos · `{salida['split']}`", "",
              "| Cifra | B | D (2.048) | C2 | C2-s1 | Ruido de semilla | Datos (B → D) "
              "| Resolución (D → C2) |", "|---|---:|---:|---:|---:|---:|---|---|"]
    def marca(efecto: float, supera: bool) -> str:
        return f"{f(efecto)} " + ("(**mayor que el ruido**)" if supera
                                  else "(no se distingue del ruido)")

    for cifra, t in salida["figures"].items():
        datos = marca(t["data_effect"], t["data_exceeds_noise"])
        res = marca(t["resolution_effect"], t["resolution_exceeds_noise"])
        lineas.append(f"| {cifra} | {f(t['B'])} | {f(t['D'])} | {f(t['C2'])} | {f(t['C2-s1'])} "
                      f"| {f(t['seed_noise'])} | {datos} | {res} |")
    lineas += ["", "K1-K6 contra B (solo información): " + ", ".join(
        f"{m} {'cumple' if ok else 'no cumple'}" for m, ok in salida["all_pass_vs_B"].items()),
        "", "Con dos semillas el ruido está mal estimado. Nada de esto cambia el defecto.", ""]
    return "\n".join(lineas)


def compara(args: argparse.Namespace) -> int:
    modelos = {"C2": args.c2, "C2-s1": args.c2_s1, "D": args.d}
    aceptacion = {}
    for nombre, ck in modelos.items():
        r = SALIDA / f"resumen-{args.split}-{sha_de(ck)[:12]}.json"
        if not r.is_file():
            raise SystemExit(f"Falta {r}: ejecuta `evalua --split {args.split}` con {ck}.")
        datos = json.loads(r.read_text())
        if datos["checkpoints"]["B"] != sha_de(args.base):
            raise SystemExit(f"⚠ {r} no se midió contra {args.base}.")
        aceptacion[nombre] = datos["acceptance"]
    salida: dict[str, Any] = {"section": "6.15", "split": args.split,
                              "checkpoints": {k: sha_de(v) for k, v in modelos.items()},
                              **lee_6_15(aceptacion), "run_record": registro(sys.argv)}
    destino = SALIDA / f"comparacion-6-15-{args.split}.json"
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    destino.with_suffix(".md").write_text(informe_6_15(salida))
    print(informe_6_15(salida))
    return 0


# ── §6.16: C2 con D de testigo, fijado antes de escribir este código ─────────────────────
#
# CT = C2 con `rescata_coronas` usando D de testigo, sobre las etiquetas votadas sin limpiar
# (y lo mismo en la pose nativa, para K4); luego la limpieza de siempre. R1: CT cumple K1-K6
# contra B. R2: CT no empeora a C2 más que el ruido de semilla de §6.15 en ese conjunto.
# Si CT cumple R1 y R2 en `dev` y en la reserva gastada, `reserva-2` se evalúa UNA vez con
# B, C2, D y CT juntos, y se lee con el ruido de la reserva gastada (también 120 arcadas).

# cifra de §6.15 → +1 si más es peor, −1 si más es mejor
SENTIDO_R2 = {"contact_p50_mm": 1, "bleeding_teeth": 1, "ghosts": 1, "false_tooth": 1,
              "false_gum": 1, "correct": -1, "tooth_fdi": -1}
RESULTADO_RESERVA_2_IDENTIDAD = RAIZ / "data/processed/identidad"


def reserva_2_gastada() -> Path | None:
    """El cerrojo común de `reserva-2`: el primer resultado escrito por cualquiera de los
    dos benchmarks que la evalúan. Mientras exista, ninguno vuelve a evaluarla."""
    hechos = sorted([*SALIDA.glob("testigo-reserva-2*.json"),
                     *SALIDA.glob("margen-reserva-2*.json"),
                     *RESULTADO_RESERVA_2_IDENTIDAD.glob("resumen-reserva-2*.json")])
    return hechos[0] if hechos else None


def evalua_arcada_testigo(trabajo: tuple[str, str, str, dict[str, str]]) -> dict:
    split, paciente, arcada, shas = trabajo
    pos, tri, experto = carga(split, paciente, arcada)
    refs = bp.referencias()
    crudas = {}
    for papel, sha in shas.items():
        z = np.load(ruta(sha, split, paciente, arcada))
        crudas[papel] = _etiquetas(z, refs)
        razon = float(z["razon"])
    votada, rescates = rescata_coronas(pos, tri, crudas["C2"][0], crudas["D"][0])
    nativa, _ = rescata_coronas(pos, tri, crudas["C2"][1], crudas["D"][1])
    crudas["CT"] = (votada, nativa)
    fila: dict = {"patient": paciente, "jaw": arcada, "orientation_ratio": razon,
                  "rescued": rescates, "models": {}}
    for papel, (v, n) in crudas.items():
        v, n = bm.limpieza_actual(pos, v), bm.limpieza_actual(pos, n)
        fila["models"][papel] = {"voted": metricas_de_arcada(pos, tri, v, experto),
                                 "native_identity": identidad_de_piezas(n, experto)}
    return fila


def _valores(aceptacion: dict) -> dict[str, float]:
    return {c: float(aceptacion["criteria"][k][clave_c])
            for c, (k, _, clave_c) in CIFRAS_6_15.items()}


def lee_6_16(aceptacion: dict[str, dict], ruido: dict[str, float]) -> dict[str, Any]:
    """`aceptacion`: `{"C2": …, "D": …, "CT": …}` contra B. R1 y R2 tal como se fijaron."""
    v = {m: _valores(a) for m, a in aceptacion.items()}
    r2 = {}
    for cifra, sentido in SENTIDO_R2.items():
        tope = v["C2"][cifra] + sentido * ruido[cifra]
        r2[cifra] = {"C2": v["C2"][cifra], "CT": v["CT"][cifra], "noise": ruido[cifra],
                     "pass": bool(sentido * (v["CT"][cifra] - tope) <= 0)}
    return {"values": v, "R1": bool(aceptacion["CT"]["criteria"]["all_pass"]),
            "R2": all(x["pass"] for x in r2.values()), "R2_figures": r2,
            "all_pass_vs_B": {m: bool(a["criteria"]["all_pass"]) for m, a in aceptacion.items()}}


def informe_6_16(salida: dict) -> str:
    def f(x: float) -> str:
        return f"{x:.3f}" if abs(x) < 10 and x != int(x) else f"{x:.0f}"
    v, r2 = salida["values"], salida["R2_figures"]
    lineas = [f"# §6.16 · C2 con D de testigo · `{salida['split']}`", "",
              f"Piezas rescatadas: {salida['rescued_teeth']} en {salida['rescued_arches']} "
              f"de {salida['arches']} arcadas.", "",
              "| Cifra | B | D | C2 | CT | Ruido | R2 |", "|---|---:|---:|---:|---:|---:|---|"]
    for cifra in CIFRAS_6_15:
        b = salida["B"][cifra]
        r = r2.get(cifra)
        marca = "—" if r is None else ("cumple" if r["pass"] else "**NO cumple**")
        ruido = "" if r is None else f(r["noise"])
        lineas.append(f"| {cifra} | {f(b)} | {f(v['D'][cifra])} | {f(v['C2'][cifra])} "
                      f"| {f(v['CT'][cifra])} | {ruido} | {marca} |")
    lineas += ["", "K1-K6 contra B: " + ", ".join(
        f"{m} {'cumple' if ok else 'no cumple'}" for m, ok in salida["all_pass_vs_B"].items()),
        "", f"**R1 {'cumple' if salida['R1'] else 'NO cumple'} · "
        f"R2 {'cumple' if salida['R2'] else 'NO cumple'}.** Nada de esto cambia el defecto.", ""]
    return "\n".join(lineas)


def testigo(args: argparse.Namespace) -> int:
    shas = {"B": sha_de(args.base), "C2": sha_de(args.c2), "D": sha_de(args.d)}
    destino = SALIDA / f"testigo-{args.split}.json"
    if args.split == "reserva-2":
        gastada = reserva_2_gastada()
        if gastada:
            print(f"⚠ `reserva-2` ya se evaluó ({gastada}). Se evalúa UNA vez.", file=sys.stderr)
            return 3
        for previo in ("dev", "reserva"):
            r = SALIDA / f"testigo-{previo}.json"
            if not r.is_file():
                raise SystemExit(f"⚠ falta {r}: CT se mide antes en `dev` y en la reserva gastada.")
            datos = json.loads(r.read_text())
            if datos["checkpoints"] != shas:
                raise SystemExit(f"⚠ {r} se midió con otros checkpoints; no sigo.")
            if not (datos["R1"] and datos["R2"]):
                raise SystemExit(f"⚠ CT no cumple R1 y R2 en {previo}: `reserva-2` no se toca.")
    pares = arcadas(args.split, args.limite)
    faltan = [(p, a, x) for p, a in pares for x in shas.values()
              if not ruta(x, args.split, p, a).is_file()]
    if faltan:
        print(f"Faltan {len(faltan)} predicciones; ejecuta `predice`.", file=sys.stderr)
        return 2
    rec = registro(sys.argv)
    trabajos = [(args.split, p, a, shas) for p, a in pares]
    with ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        filas = list(grupo.map(evalua_arcada_testigo, trabajos))
    aceptacion = {m: criterios([{**f, "models": {"B": f["models"]["B"], "C": f["models"][m]}}
                                for f in filas]) for m in ("C2", "D", "CT")}
    fuente = "reserva" if args.split == "reserva-2" else args.split
    comparacion = json.loads((SALIDA / f"comparacion-6-15-{fuente}.json").read_text())
    ruido = {c: float(t["seed_noise"]) for c, t in comparacion["figures"].items()}
    b = {c: (float(aceptacion["C2"]["criteria"][k][clave_b]) if clave_b else 0.0)
         for c, (k, clave_b, _) in CIFRAS_6_15.items()}
    salida: dict[str, Any] = {
        "section": "6.16", "split": args.split, "arches": len(filas), "checkpoints": shas,
        "noise_from": f"comparacion-6-15-{fuente}.json", "B": b,
        **lee_6_16(aceptacion, ruido),
        "rescued_teeth": sum(len(f["rescued"]) for f in filas),
        "rescued_arches": sum(1 for f in filas if f["rescued"]),
        "acceptance": aceptacion, "run_record": rec,
        "per_arch": [{"patient": f["patient"], "jaw": f["jaw"], "rescued": f["rescued"]}
                     for f in filas],
        "scope": "Teeth3DS+ only; not the project's clinical cases",
    }
    if args.limite:
        destino = destino.with_name(destino.stem + f"-limite{args.limite}.json")
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    destino.with_suffix(".md").write_text(informe_6_16(salida))
    print(informe_6_16(salida))
    return 0


# ── §6.17: `ancla_al_pliegue` sobre C2, fijado antes de escribir este código ─────────────
#
# Las variantes y los criterios A1-A6 son los de `benchmark_margen.py`, sin tocar. La base
# `M1` es C2 votado + limpieza actual. Se elige en `dev` y en la reserva gastada con A3 a la
# mitad de su límite; `reserva-2` se evalúa UNA vez con la elegida y los límites completos,
# y en esa misma pasada se miden C2 y D contra B con K1-K6.

VARIANTES_MARGEN = ("M5", "M7", "M8", "M9")
HOLGURA_MARGEN = bm.HOLGURA_DEV


def evalua_arcada_margen(trabajo: tuple[str, str, str, dict[str, str]]) -> dict:
    split, paciente, arcada, shas = trabajo
    pos, tri, experto = carga(split, paciente, arcada)
    z = np.load(ruta(shas["C2"], split, paciente, arcada))
    votada, _ = _etiquetas(z, bp.referencias())
    m1 = bm.limpieza_actual(pos, votada)
    etiquetas = {"M1": m1}
    for v in VARIANTES_MARGEN:
        etiquetas[v] = np.asarray(bm.REFINAMIENTOS[v][1](pos, tri, m1), dtype=np.int64)
    for e, (_, origen) in bm.SOBRE_EXPERTO.items():
        if origen in VARIANTES_MARGEN:
            metodo = bm.REFINAMIENTOS[origen][1]
            etiquetas[e] = np.asarray(metodo(pos, tri, experto), dtype=np.int64)
    fila: dict = {"patient": paciente, "jaw": arcada,
                  "methods": {k: metricas_de_arcada(pos, tri, v, experto)
                              for k, v in etiquetas.items()}}
    if "B" in shas:  # solo en `reserva-2`: C2 y D contra B, K1-K6
        fila["vs_B"] = {m: evalua_arcada((split, paciente, arcada,
                                          {"B": shas["B"], "C": shas[m]}))["models"]
                        for m in ("C2", "D")}
    return fila


def con_holgura(c: dict, holgura: float = HOLGURA_MARGEN) -> bool:
    """A1-A6 con A3 a `holgura` de su límite, como en §6.7."""
    if not c:
        return False
    a3 = c["A3_overtrim"]
    resto = all(v["pass"] for k, v in c.items() if k.startswith("A") and k != "A3_overtrim")
    return bool(resto and a3["false_gum_rise"] <= holgura * bm.SUBIDA_FALSA_ENCIA
                and a3["newly_under_retained_fraction"] <= holgura * bm.PIEZAS_DANADAS)


def elige_margen(dev: dict, reserva: dict) -> dict:
    """`dev`, `reserva`: la salida de `margen` en cada conjunto. Regla de §6.17."""
    pasan = [v for v in VARIANTES_MARGEN
             if dev["variants"][v]["slack_pass"] and reserva["variants"][v]["slack_pass"]]
    orden = sorted(pasan, key=lambda v: (reserva["variants"][v]["false_tooth"],
                                         reserva["variants"][v]["newly_under_retained"]))
    return {"passing": pasan, "chosen": orden[0] if orden else None}


# §6.18: desviación declarada. La variante la nombra quien decide, y solo se acepta si
# cumple A1-A6 con los límites completos en `dev` y en la reserva gastada.
DESVIACION_6_18 = "M8"


def elige_por_desviacion(dev: dict, reserva: dict, variante: str) -> dict:
    if variante != DESVIACION_6_18:
        raise SystemExit(f"⚠ §6.18 declara {DESVIACION_6_18}, no {variante}.")
    for nombre, s in (("dev", dev), ("reserva", reserva)):
        if not s["variants"][variante]["criteria"].get("all_pass"):
            raise SystemExit(f"⚠ {variante} no cumple A1-A6 completos en {nombre}; no sigo.")
    return {"passing": [variante], "chosen": variante, "deviation": "6.18"}


def informe_6_17(salida: dict) -> str:
    si = lambda ok: "cumple" if ok else "**no**"  # noqa: E731
    lineas = [f"# §6.17 · `ancla_al_pliegue` sobre C2 · `{salida['split']}`", "",
              f"Base `M1` (C2 + limpieza): encía marcada diente {salida['base_false_tooth']:.3f}.",
              "", "| Variante | Encía como diente | Subida falsa encía | Dañadas | A1 | A2 | A3 "
              "| A4 | A5 | A6 | Con holgura |", "|---|---:|---:|---:|---|---|---|---|---|---|---|"]
    for v, d in salida["variants"].items():
        c = d["criteria"]
        a = [si(c[k]["pass"]) for k in ("A1_margin", "A2_agreement", "A3_overtrim",
                                        "A4_labelled_fraction", "A5_width",
                                        "A6_truth_untouched")]
        lineas.append(f"| {v} | {d['false_tooth']:.3f} | "
                      f"{c['A3_overtrim']['false_gum_rise'] * 100:+.2f} pt | "
                      f"{d['newly_under_retained']} de {c['A3_overtrim']['teeth']} | "
                      + " | ".join(a) + f" | {si(d['slack_pass'])} |")
    if "vs_B" in salida:
        lineas += ["", "K1-K6 contra B: " + ", ".join(
            f"{m} {'cumple' if k['criteria']['all_pass'] else 'no cumple'}"
            for m, k in salida["vs_B"].items())]
    if "chosen" in salida:
        lineas += ["", f"**Elegida para `reserva-2`:** {salida['chosen'] or 'ninguna'}."]
    if "confirmed" in salida:
        lineas += ["", f"**{salida['candidate']} en `reserva-2`, A1-A6 con límites completos: "
                   f"{'cumple' if salida['confirmed'] else 'NO cumple'}.**"]
    return "\n".join(lineas + [""])


def margen(args: argparse.Namespace) -> int:
    shas = {"C2": sha_de(args.c2)}
    destino = SALIDA / f"margen-{args.split}.json"
    eleccion = None
    if args.split == "reserva-2":
        gastada = reserva_2_gastada()
        if gastada:
            print(f"⚠ `reserva-2` ya se evaluó ({gastada}). Se evalúa UNA vez.", file=sys.stderr)
            return 3
        previos = {}
        for previo in ("dev", "reserva"):
            r = SALIDA / f"margen-{previo}.json"
            if not r.is_file():
                raise SystemExit(f"⚠ falta {r}: la variante se elige antes en `dev` y en la "
                                 "reserva gastada.")
            previos[previo] = json.loads(r.read_text())
            if previos[previo]["checkpoints"]["C2"] != shas["C2"]:
                raise SystemExit(f"⚠ {r} se midió con otro C2; no sigo.")
        eleccion = (elige_por_desviacion(previos["dev"], previos["reserva"], args.desviacion)
                    if args.desviacion else elige_margen(previos["dev"], previos["reserva"]))
        if eleccion["chosen"] is None:
            raise SystemExit("⚠ ninguna variante pasó en `dev` y la reserva: no se toca.")
        if args.base is None or args.d is None:
            raise SystemExit("⚠ en `reserva-2` se miden también B y D: pasa --base y --d.")
        shas.update({"B": sha_de(args.base), "D": sha_de(args.d)})
    pares = arcadas(args.split, args.limite)
    faltan = [(p, a, x) for p, a in pares for x in shas.values()
              if not ruta(x, args.split, p, a).is_file()]
    if faltan:
        print(f"Faltan {len(faltan)} predicciones; ejecuta `predice`.", file=sys.stderr)
        return 2
    rec = registro(sys.argv)
    with ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        filas = list(grupo.map(evalua_arcada_margen,
                               [(args.split, p, a, shas) for p, a in pares]))
    variantes = {}
    for v in VARIANTES_MARGEN:
        c = bm.criterios(filas, v)
        variantes[v] = {
            "criteria": c, "slack_pass": con_holgura(c),
            "false_tooth": float(np.mean([f["methods"][v]["false_tooth_rate"] for f in filas])),
            "newly_under_retained": c["A3_overtrim"]["newly_under_retained"]}
    salida: dict[str, Any] = {
        "section": "6.17", "split": args.split, "arches": len(filas), "checkpoints": shas,
        "base_false_tooth": float(np.mean([f["methods"]["M1"]["false_tooth_rate"]
                                           for f in filas])),
        "variants": variantes, "run_record": rec,
        "scope": "Teeth3DS+ only; not the project's clinical cases",
    }
    if eleccion is not None:
        salida.update(eleccion)
        salida["candidate"] = eleccion["chosen"]
        salida["confirmed"] = bool(variantes[eleccion["chosen"]]["criteria"]["all_pass"])
        salida["vs_B"] = {m: criterios([{**f, "models": f["vs_B"][m]} for f in filas])
                          for m in ("C2", "D")}
    if args.limite:
        destino = destino.with_name(destino.stem + f"-limite{args.limite}.json")
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    destino.with_suffix(".md").write_text(informe_6_17(salida))
    print(informe_6_17(salida))
    return 0


# ── §6.19: la arcada al revés (`O`) y la fila del pliegue (`M10`), fijado en 4bfd94a ─────
#
# `O` = `analysis_agents.orientacion.marco_orientado` en la arcada que se segmenta; `M10` =
# `M8` + `devuelve_anillos`. La puerta de `M10` son A1-A6 con holgura en los tres conjuntos
# gastados; en `reserva-3`, A1-A6 con los límites completos. `O` se juzga con O1-O4.
SPLITS_6_19 = ("dev", "reserva", "reserva-2", "reserva-3")
GASTADOS_6_19 = ("dev", "reserva", "reserva-2")
CANDIDATO_6_19, BASE_6_19 = "M10", "M1"
EJE_NATIVO = np.array([0.0, 0.0, 1.0])  # en Teeth3DS+ el oclusal nativo es +z


def reserva_3_gastada() -> Path | None:
    """El cerrojo de `reserva-3`: el primer resultado de §6.19 escrito sobre ella."""
    hechos = sorted(SALIDA.glob("seccion-6-19-reserva-3*.json"))
    return hechos[0] if hechos else None


def _votada_borde(z, refs: list[str]) -> np.ndarray:  # type: ignore[no-untyped-def]
    seg = {r: z[f"et_{r}"].astype(np.int64) for r in refs}
    votada, _ = vota(seg, {r: float(z[f"conf_{r}"]) for r in refs})
    return votada


def evalua_arcada_6_19(trabajo: tuple[str, str, str, dict[str, str]]) -> dict:
    from analysis_agents.orientacion import marco_orientado
    from fusion_agents.marco import marco_canonico

    split, paciente, arcada, shas = trabajo
    pos, tri, experto = carga(split, paciente, arcada)
    refs = bp.referencias()
    z = np.load(ruta(shas["C2"], split, paciente, arcada))
    votada, nativa = _etiquetas(z, refs)
    m1 = bm.limpieza_actual(pos, votada)
    nativa = bm.limpieza_actual(pos, nativa)
    _, R0, _ = marco_canonico(pos)
    _, R, razon, volteado = marco_orientado(pos, tri)
    o = m1
    if volteado:
        o = bm.limpieza_actual(pos, _votada_borde(
            np.load(ruta_borde(shas["C2"], split, paciente, arcada)), refs))
    etiquetas = {"M1": m1}
    for v in ("M8", CANDIDATO_6_19):
        etiquetas[v] = np.asarray(bm.REFINAMIENTOS[v][1](pos, tri, m1), dtype=np.int64)
        e = bm.VERDAD_DE[v]
        etiquetas[e] = np.asarray(bm.REFINAMIENTOS[v][1](pos, tri, experto), dtype=np.int64)
    metodos = {k: metricas_de_arcada(pos, tri, v, experto) for k, v in etiquetas.items()}
    id_nativa = identidad_de_piezas(nativa, experto)
    modelos = {"C2": {"voted": metodos["M1"], "native_identity": id_nativa},
               "C2O": {"voted": metodos["M1"] if not volteado
                       else metricas_de_arcada(pos, tri, o, experto),
                       "native_identity": id_nativa}}
    if "B" in shas:
        zb = np.load(ruta(shas["B"], split, paciente, arcada))
        vb, nb = _etiquetas(zb, refs)
        modelos["B"] = {"voted": metricas_de_arcada(pos, tri, bm.limpieza_actual(pos, vb),
                                                    experto),
                        "native_identity": identidad_de_piezas(bm.limpieza_actual(pos, nb),
                                                               experto)}
    return {"patient": paciente, "jaw": arcada, "orientation_ratio": float(razon),
            "flipped": bool(volteado), "cos_canonical": float(R0[2] @ EJE_NATIVO),
            "cos_oriented": float(R[2] @ EJE_NATIVO), "methods": metodos, "models": modelos}


def criterios_6_19(filas: list[dict]) -> dict:
    """O1-O4 de `O` frente a C2 (O4 solo con B), fijados en §6.19."""
    piezas = perdidas = 0
    for f in filas:
        antes = f["models"]["C2"]["voted"]["per_tooth"]
        despues = f["models"]["C2O"]["voted"]["per_tooth"]
        piezas += len(antes)
        perdidas += sum(1 for t, v in antes.items() if v["retained"] >= CORONA_MINIMA_CONSERVADA
                        and despues[t]["retained"] < CORONA_MINIMA_CONSERVADA)
    correctas = {m: sum(f["models"][m]["voted"]["identity"]["correct"] for f in filas)
                 for m in ("C2", "C2O")}
    contrarias = sum(1 for f in filas if f["cos_oriented"] <= 0)
    o = {
        "flipped": sum(f["flipped"] for f in filas),
        "inverted_before": sum(1 for f in filas if f["cos_canonical"] <= 0),
        "O1_orientation": {"against_native": contrarias, "pass": contrarias == 0},
        "O2_no_crown_lost": {"teeth": piezas, "lost": perdidas, "pass": perdidas == 0},
        "O3_numbering": {**correctas, "pass": correctas["C2O"] >= correctas["C2"]},
    }
    o["confirmed"] = all(o[k]["pass"] for k in ("O1_orientation", "O2_no_crown_lost",
                                               "O3_numbering"))
    o["benefit_measured"] = o["flipped"] > 0
    if all("B" in f["models"] for f in filas):
        o["vs_B"] = {m: criterios([{**f, "models": {"B": f["models"]["B"],
                                                     "C": f["models"][m]}} for f in filas])
                     for m in ("C2", "C2O")}
        k5 = o["vs_B"]["C2O"]["criteria"]["K5_margin"]
        o["O4_crowns_vs_B"] = {"newly_under_retained": k5["newly_under_retained"],
                               "teeth": k5["teeth"],
                               "pass": k5["newly_under_retained"] <= PIEZAS_DANADAS * k5["teeth"]}
    return o


def puerta_6_19(sha_c2: str) -> dict:
    """La puerta de `M10`: A1-A6 con holgura en los tres conjuntos gastados, mismo C2."""
    puerta: dict[str, Any] = {}
    for s in GASTADOS_6_19:
        r = SALIDA / f"seccion-6-19-{s}.json"
        if not r.is_file():
            raise SystemExit(f"⚠ falta {r}: la puerta de §6.19 se mide antes en {GASTADOS_6_19}.")
        previo = json.loads(r.read_text())
        if previo["checkpoints"]["C2"] != sha_c2:
            raise SystemExit(f"⚠ {r} se midió con otro C2; no sigo.")
        puerta[s] = previo["M10"]["slack_pass"]
    puerta["pass"] = all(puerta[s] for s in GASTADOS_6_19)
    return puerta


def informe_6_19(r: dict) -> str:
    si = lambda b: "cumple" if b else "**no**"  # noqa: E731
    o = r["O"]
    lineas = [f"# §6.19 · La arcada al revés y la fila del pliegue · `{r['split']}`", "",
              f"{r['arches']} arcadas. El borde da la vuelta a {o['flipped']}; con "
              f"`marco_canonico` estaban al revés {o['inverted_before']}.", "",
              "| `O` frente a C2 | Valor | Veredicto |", "|---|---|---|",
              f"| O1 · eje contra el +z nativo | {o['O1_orientation']['against_native']} | "
              f"{si(o['O1_orientation']['pass'])} |",
              f"| O2 · coronas que C2 conserva y `O` pierde | {o['O2_no_crown_lost']['lost']} de "
              f"{o['O2_no_crown_lost']['teeth']} | {si(o['O2_no_crown_lost']['pass'])} |",
              f"| O3 · bien numeradas | {o['O3_numbering']['C2']} → {o['O3_numbering']['C2O']} | "
              f"{si(o['O3_numbering']['pass'])} |"]
    if "O4_crowns_vs_B" in o:
        o4 = o["O4_crowns_vs_B"]
        lineas.append(f"| O4 · coronas que pasan a < 85 % frente a B | "
                      f"{o4['newly_under_retained']} de {o4['teeth']} | {si(o4['pass'])} |")
    lineas += ["", f"**`O`: {'confirmada' if o['confirmed'] else 'no confirmada'}"
               + ("" if o["benefit_measured"] else "; la regla no dio la vuelta a ninguna "
                  "arcada, así que el beneficio no se ha medido") + ".**", ""]
    if "vs_B" in o:
        lineas.append("K1-K6 contra B: " + ", ".join(
            f"{'C2' if m == 'C2' else 'C2 con `O`'} {si(c['criteria']['all_pass'])}"
            for m, c in o["vs_B"].items()))
        lineas.append("")
    m10 = r.get("M10")
    if m10 is None:
        lineas.append("**`M10` no se lleva a esta reserva: no pasó la puerta.**")
    else:
        a3 = m10["criteria"]["A3_overtrim"]
        lineas += ["| `M10` frente a `M1` | A1 | A2 | A3 | A4 | A5 | A6 | Con holgura |",
                   "|---|---|---|---|---|---|---|---|",
                   "| " + f"subida falsa encía {100 * a3['false_gum_rise']:+.2f} pt, dañadas "
                   f"{a3['newly_under_retained']} de {a3['teeth']}, encía como diente "
                   f"{m10['false_tooth']:.3f} (M1 {r['base_false_tooth']:.3f}) | "
                   + " | ".join(si(m10["criteria"][k]["pass"]) for k in
                                ("A1_margin", "A2_agreement", "A3_overtrim",
                                 "A4_labelled_fraction", "A5_width", "A6_truth_untouched"))
                   + f" | {si(m10['slack_pass'])} |", "",
                   f"**`M10`, A1-A6 con límites completos: {si(m10['criteria']['all_pass'])}.**"]
    return "\n".join(lineas) + "\n"


def seccion_6_19(args: argparse.Namespace) -> int:
    shas = {"C2": sha_de(args.c2)}
    puerta = None
    if args.split == "reserva-3":
        gastada = reserva_3_gastada()
        if gastada:
            print(f"⚠ `reserva-3` ya se evaluó ({gastada}). Se evalúa UNA vez.", file=sys.stderr)
            return 3
        if args.base is None:
            raise SystemExit("⚠ en `reserva-3` se mide también B (O4): pasa --base.")
        puerta = puerta_6_19(shas["C2"])
    if args.base is not None:  # en un gastado, solo para ejercitar O4 antes de reserva-3
        shas["B"] = sha_de(args.base)
    pares = arcadas(args.split, args.limite)
    faltan = [(p, a, x) for p, a in pares for x in shas.values()
              if not ruta(x, args.split, p, a).is_file()]
    if faltan:
        print(f"Faltan {len(faltan)} predicciones; ejecuta `predice`.", file=sys.stderr)
        return 2
    rec = registro(sys.argv)
    with ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        filas = list(grupo.map(evalua_arcada_6_19, [(args.split, p, a, shas) for p, a in pares]))
    salida: dict[str, Any] = {
        "section": "6.19", "split": args.split, "arches": len(filas), "checkpoints": shas,
        "base_false_tooth": float(np.mean([f["methods"]["M1"]["false_tooth_rate"]
                                           for f in filas])),
        "O": criterios_6_19(filas), "gate": puerta, "run_record": rec,
        "per_arch": [{k: f[k] for k in ("patient", "jaw", "orientation_ratio", "flipped",
                                        "cos_canonical", "cos_oriented")} for f in filas],
        "scope": "Teeth3DS+ only; not the project's clinical cases",
    }
    if puerta is None or puerta["pass"]:
        c = bm.criterios(filas, CANDIDATO_6_19)
        salida["M10"] = {"criteria": c, "slack_pass": con_holgura(c),
                         "false_tooth": float(np.mean([f["methods"][CANDIDATO_6_19]
                                                       ["false_tooth_rate"] for f in filas]))}
        c8 = bm.criterios(filas, "M8")
        salida["M8_reference"] = {"criteria": c8, "slack_pass": con_holgura(c8)}
    destino = SALIDA / f"seccion-6-19-{args.split}.json"
    if args.limite:
        destino = destino.with_name(destino.stem + f"-limite{args.limite}.json")
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    destino.with_suffix(".md").write_text(informe_6_19(salida))
    print(informe_6_19(salida))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    p = ordenes.add_parser("predice")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parte", default="0/1", help="i/n: solo las arcadas i, i+n, …")
    e = ordenes.add_parser("evalua")
    e.add_argument("--base", type=Path, default=bm.CHECKPOINT)
    e.add_argument("--candidato", type=Path, required=True)
    e.add_argument("--trabajos", type=int, default=8)
    el = ordenes.add_parser("elige")
    el.add_argument("--modo", choices=("resolucion", "final"), required=True)
    el.add_argument("--candidatos", type=Path, nargs="+", required=True)
    el.set_defaults(funcion=elige)
    co = ordenes.add_parser("compara", help="§6.15: semilla y resolución frente a datos")
    co.add_argument("--split", choices=("dev", "reserva"), required=True)
    co.add_argument("--base", type=Path, default=bm.CHECKPOINT)
    co.add_argument("--c2", type=Path, required=True)
    co.add_argument("--c2-s1", type=Path, required=True)
    co.add_argument("--d", type=Path, required=True)
    co.set_defaults(funcion=compara)
    te = ordenes.add_parser("testigo", help="§6.16: C2 con D de testigo para las coronas")
    te.add_argument("--split", choices=("dev", "reserva", "reserva-2"), required=True)
    te.add_argument("--base", type=Path, default=bm.CHECKPOINT)
    te.add_argument("--c2", type=Path, required=True)
    te.add_argument("--d", type=Path, required=True)
    te.add_argument("--trabajos", type=int, default=6)
    te.add_argument("--limite", type=int, help="Solo las N primeras arcadas.")
    te.set_defaults(funcion=testigo)
    ma = ordenes.add_parser("margen", help="§6.17: ancla_al_pliegue sobre C2")
    ma.add_argument("--split", choices=("dev", "reserva", "reserva-2"), required=True)
    ma.add_argument("--c2", type=Path, required=True)
    ma.add_argument("--base", type=Path, help="B; solo en reserva-2")
    ma.add_argument("--d", type=Path, help="D; solo en reserva-2")
    ma.add_argument("--desviacion", help="§6.18: variante declarada para reserva-2 (M8)")
    ma.add_argument("--trabajos", type=int, default=6)
    ma.add_argument("--limite", type=int, help="Solo las N primeras arcadas.")
    ma.set_defaults(funcion=margen)
    s19 = ordenes.add_parser("seccion-6-19", help="§6.19: arcada al revés y fila del pliegue")
    s19.add_argument("--split", choices=SPLITS_6_19, required=True)
    s19.add_argument("--c2", type=Path, required=True)
    s19.add_argument("--base", type=Path, help="B, para O4; obligatorio en reserva-3")
    s19.add_argument("--trabajos", type=int, default=6)
    s19.add_argument("--limite", type=int, help="Solo las N primeras arcadas.")
    s19.set_defaults(funcion=seccion_6_19)
    # `reserva-2` se predice aquí y se evalúa una vez en total: con `testigo` (§6.16) o con
    # `benchmark_identidad.py evalua` (§6.11), lo que llegue antes. Ver `reserva_2_gastada`.
    p.add_argument("--orienta-borde", action="store_true",
                   help="§6.19: predice con el marco orientado por el borde, solo donde cambia")
    for sub, funcion, splits in ((p, predice, ("dev", "reserva", "reserva-2", "reserva-3")),
                                 (e, evalua, ("dev", "reserva"))):
        sub.add_argument("--split", choices=splits, required=True)
        sub.add_argument("--limite", type=int, help="Solo las N primeras arcadas.")
        sub.set_defaults(funcion=funcion)
    args = ap.parse_args()
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
