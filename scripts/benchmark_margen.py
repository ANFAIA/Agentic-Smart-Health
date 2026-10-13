#!/usr/bin/env python
"""Banco de pruebas de la frontera corona-encía contra las etiquetas de experto de Teeth3DS+.

    ~/.venvs/dental-gpu/bin/python scripts/benchmark_margen.py predice --split dev
    uv run python scripts/benchmark_margen.py evalua --split dev --trabajos 6

**Qué responde.** Si un método de frontera acerca las etiquetas del modelo a las del
experto, y a qué precio. Compara, sobre las mismas arcadas, las etiquetas tal cual salen
del modelo, la limpieza que hoy aplica el pipeline y cada refinamiento disponible.

**Dos órdenes, porque solo una necesita GPU.** `predice` corre el modelo y guarda sus
etiquetas; `evalua` las lee y mide, sin torch. Así un método nuevo se evalúa en segundos y
en cualquier máquina, y todos se miden sobre EXACTAMENTE las mismas predicciones.

**La partición es la del entrenamiento, y por eso no hay fuga.** El checkpoint se entrenó
con el 80 % de los pacientes (semilla 42); aquí solo se usa el 20 % restante, reproducido
con el mismo código. Ese 20 % se parte otra vez: `dev` para ajustar constantes y `eval`
para UNA ejecución final. Ajustar sobre `eval` lo gasta, y entonces hay que decirlo.

⚠️ **Lo que NO mide.** Teeth3DS+ es otro escáner, otra pose y otra población que los casos
clínicos propios, de los que no hay anotación. Un método que pase aquí está autorizado a
probarse sobre un caso real, no validado sobre él.

Los datos de Teeth3DS+ no se versionan (licencia de terceros) y la salida va a
`data/processed/benchmark-margen/`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RESUMEN_EN = "Scores crown-gum boundary methods against Teeth3DS+ expert labels."

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))

from analysis_agents import (  # noqa: E402
    absorbe_islas,
    afina_fronteras,
    quita_motas,
    rellena_etiquetas,
    rellena_huecos_interiores,
)
from analysis_agents.frontera import (  # noqa: E402
    ancla_al_pliegue,
    recorta_al_cuello,
    refina_fronteras,
    reparte_por_geodesica,
)
from analysis_agents.margen_metricas import (  # noqa: E402
    RESUMIBLES,
    agrega,
    compara,
    metricas_de_arcada,
)
from ingestion_agents.mesh_agent import parse_obj  # noqa: E402

# Cambia cuando cambie CUALQUIER cosa que haga incomparables dos ejecuciones: la partición,
# la inferencia, una métrica. Viaja en la salida.
VERSION_PROTOCOLO = "1"

DATOS = RAIZ / "data/raw/teeth3ds"
MALLAS = DATOS / "3D_scans_per_patient_obj_files"
ETIQUETAS = DATOS / "ground-truth_labels_instances"
SALIDA = RAIZ / "data/processed/benchmark-margen"
CHECKPOINT = RAIZ / "data/processed/a3-checkpoints/normales-ce-baseline.pt"

SEMILLA_PARTICION = 42
FRACCION_TEST = 0.20
PACIENTES_DEV = 20
# Arcadas de ENTRENAMIENTO para la comprobación de fuga: el modelo debe acertar claramente
# más en ellas que en `eval`. Si acierta igual, la partición no es la del entrenamiento.
PACIENTES_FUGA = 5
ARCADAS = ("upper", "lower")


# ── Partición ───────────────────────────────────────────────────────────────────

def pacientes() -> list[str]:
    """Pacientes con las dos arcadas y sus etiquetas, ordenados."""
    return sorted(
        d.name for d in MALLAS.iterdir() if d.is_dir() and all(
            (d / f"{d.name}_{a}.obj").is_file()
            and (ETIQUETAS / d.name / f"{d.name}_{a}.json").is_file() for a in ARCADAS))


def particion(ids: list[str]) -> dict[str, list[str]]:
    """`train`, `dev` y `eval`, con el mismo barajado que la celda 5 del notebook.

    ⚠️ Es el MISMO código que entrenó el checkpoint, no uno parecido: ids ordenados,
    `default_rng(42).shuffle`, y el primer 20 % es test. Cambiar cualquier detalle mete
    pacientes de entrenamiento en la evaluación sin que nada lo avise.
    """
    barajados = list(ids)
    np.random.default_rng(SEMILLA_PARTICION).shuffle(barajados)
    n_test = max(1, round(FRACCION_TEST * len(barajados)))
    test, train = barajados[:n_test], barajados[n_test:]
    return {"train": train, "dev": test[:PACIENTES_DEV], "eval": test[PACIENTES_DEV:],
            "fuga": train[:PACIENTES_FUGA]}


def huella(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def arcadas_de(split: str, limite: int | None) -> list[tuple[str, str]]:
    ids = particion(pacientes())[split]
    pares = [(p, a) for p in ids for a in ARCADAS]
    return pares[:limite] if limite else pares


# ── Datos ───────────────────────────────────────────────────────────────────────

def carga_arcada(paciente: str, arcada: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(posiciones, caras, etiquetas de experto)`. El OBJ se lee sin deduplicar."""
    malla = parse_obj(MALLAS / paciente / f"{paciente}_{arcada}.obj")
    pos = np.asarray(malla["positions"], dtype=np.float64)
    tri = np.asarray(malla["faces"], dtype=np.int64)
    datos = json.loads((ETIQUETAS / paciente / f"{paciente}_{arcada}.json").read_text())
    experto = np.asarray(datos["labels"], dtype=np.int64)
    if len(experto) != len(pos):
        raise ValueError(f"{paciente}_{arcada}: {len(experto)} etiquetas para {len(pos)} vértices")
    return pos, tri, experto


def ruta_prediccion(paciente: str, arcada: str) -> Path:
    return SALIDA / "pred" / f"{paciente}_{arcada}.npz"


# ── Métodos ─────────────────────────────────────────────────────────────────────

def limpieza_actual(pos: np.ndarray, etq: np.ndarray) -> np.ndarray:
    """Las cinco limpiezas que aplica `scripts/caso_completo.py`, en su mismo orden."""
    e = rellena_etiquetas(pos, etq)
    e, _ = absorbe_islas(pos, e)
    e, _ = afina_fronteras(pos, e)
    e, _ = quita_motas(pos, e)
    e, _ = rellena_huecos_interiores(pos, e)
    return np.asarray(e, dtype=np.int64)


# Cada método parte de las etiquetas YA limpias (`M1`), que es donde entraría en el
# pipeline. Un método nuevo se añade aquí y aparece en la tabla sin tocar nada más.
REFINAMIENTOS: dict[str, tuple[str, Callable[..., np.ndarray]]] = {
    "M2": ("limpieza + recorta_al_cuello", recorta_al_cuello),
    "M3": ("limpieza + refina_fronteras", refina_fronteras),
    "M4": ("limpieza + reparte_por_geodesica", reparte_por_geodesica),
    "M5": ("limpieza + ancla_al_pliegue", lambda p, f, e: ancla_al_pliegue(p, f, e)[0]),
    "M6": ("limpieza + reparte + ancla_al_pliegue",
           lambda p, f, e: ancla_al_pliegue(p, f, reparte_por_geodesica(p, f, e))[0]),
}
# Los que además se prueban sobre las etiquetas del EXPERTO: un método que estropea la
# verdad de campo no puede estar arreglando nada.
SOBRE_EXPERTO: dict[str, tuple[str, str]] = {"E2": ("experto + ancla_al_pliegue", "M5")}

MAYOR_ES_MEJOR = ("binary_agreement", "binary_iou", "fdi_agreement", "tooth_fdi_agreement")

DESCRIPCIONES = {
    "E0": "experto (referencia)", "E1": "experto + limpieza actual",
    "M0": "modelo, sin tocar", "M1": "modelo + limpieza actual (pipeline de hoy)",
    **{k: v[0] for k, v in REFINAMIENTOS.items()},
    **{k: v[0] for k, v in SOBRE_EXPERTO.items()},
}


def evalua_arcada(paciente: str, arcada: str) -> dict:
    """Todas las filas de la tabla para una arcada. Se llama en un proceso aparte."""
    pos, tri, experto = carga_arcada(paciente, arcada)
    guardado = np.load(ruta_prediccion(paciente, arcada))
    crudo = guardado["etiquetas"].astype(np.int64)
    if len(crudo) != len(pos):
        raise ValueError(f"{paciente}_{arcada}: la predicción no es de esta malla")
    tiempos: dict[str, float] = {}

    def cronometra(nombre: str, calcula: Callable[[], np.ndarray]) -> np.ndarray:
        t0 = time.perf_counter()
        resultado = calcula()
        tiempos[nombre] = round(time.perf_counter() - t0, 3)
        return resultado

    etiquetas = {"E0": experto, "M0": crudo}
    etiquetas["E1"] = cronometra("E1", lambda: limpieza_actual(pos, experto))
    etiquetas["M1"] = cronometra("M1", lambda: limpieza_actual(pos, crudo))
    for clave, (_, metodo) in REFINAMIENTOS.items():
        etiquetas[clave] = cronometra(
            clave, lambda m=metodo: np.asarray(m(pos, tri, etiquetas["M1"]), dtype=np.int64))
    for clave, (_, origen) in SOBRE_EXPERTO.items():
        metodo = REFINAMIENTOS[origen][1]
        etiquetas[clave] = cronometra(
            clave, lambda m=metodo: np.asarray(m(pos, tri, experto), dtype=np.int64))
    return {
        "patient": paciente, "jaw": arcada, "seconds": tiempos,
        "checkpoint_sha256": str(guardado["checkpoint_sha256"]),
        "methods": {k: metricas_de_arcada(pos, tri, v, experto)
                    for k, v in etiquetas.items()},
    }


def _evalua(par: tuple[str, str]) -> dict:
    return evalua_arcada(*par)


# ── Criterios de aceptación ─────────────────────────────────────────────────────
#
# Fijados ANTES de ejecutar `eval` y escritos aquí para que el veredicto lo dé el script.
# Comparan un candidato con el pipeline de hoy (`M1`), arcada a arcada.
#
# ⚠️ **A3 se precisó el 2026-10-13, antes de tocar `eval`, y se declara.** Decía «≤ 1 % de
# dientes conservan < 85 % de su corona» sin distinguir las piezas que el modelo ya traía
# rotas: en `dev` son 5 de 520 antes de aplicar ningún método. Lo que el criterio quiere
# impedir es que el método rompa piezas, así que cuenta las que pasan de ≥ 85 % a < 85 %.
CANDIDATO, BASE, SOBRE_VERDAD = "M5", "M1", "E2"
MEJORA_MINIMA_ARCADAS = 0.75
RETROCESO_TOLERADO = 0.005
ARCADAS_CON_RETROCESO = 0.10
SUBIDA_FALSA_ENCIA = 0.010
PIEZAS_DANADAS = 0.01
CORONA_MINIMA = 0.85
ETIQUETADO_DE_MENOS = -0.02
ACUERDO_SOBRE_VERDAD = 0.99


def _medida(fila: dict, metodo: str, *ruta: str) -> float | None:
    valor = fila["methods"][metodo]
    for clave in ruta:
        valor = valor.get(clave) if isinstance(valor, dict) else None
        if valor is None:
            return None
    return float(valor)


def _pares(filas: list[dict], *ruta: str) -> np.ndarray:
    """`(candidato, base)` por arcada, solo donde las dos medidas existen."""
    pares = [(_medida(f, CANDIDATO, *ruta), _medida(f, BASE, *ruta)) for f in filas]
    return np.array([(c, b) for c, b in pares if c is not None and b is not None])


def criterios(filas: list[dict]) -> dict:
    """El veredicto A1-A6 del candidato frente al pipeline de hoy."""
    if not all(CANDIDATO in f["methods"] and SOBRE_VERDAD in f["methods"] for f in filas):
        return {}
    margen = _pares(filas, "margin", "symmetric_p50_mm")
    d_margen = margen[:, 0] - margen[:, 1]
    acuerdos = {}
    for nombre in ("binary_agreement", "fdi_agreement"):
        par = _pares(filas, nombre)
        d = par[:, 0] - par[:, 1]
        acuerdos[nombre] = {"mean_diff": float(d.mean()),
                            "arches_below": float((d < -RETROCESO_TOLERADO).mean())}
    encia = _pares(filas, "false_gum_rate")
    piezas = nuevas = 0
    for f in filas:
        antes, despues = f["methods"][BASE]["per_tooth"], f["methods"][CANDIDATO]["per_tooth"]
        piezas += len(antes)
        nuevas += sum(1 for t, v in antes.items() if v["retained"] >= CORONA_MINIMA
                      and despues[t]["retained"] < CORONA_MINIMA)
    etiquetado = _pares(filas, "labelled_fraction", "paired_diff_area")
    anchos = {q: _pares(filas, "width_error_mm", q) for q in ("p50", "p90")}
    verdad = np.array([_medida(f, SOBRE_VERDAD, "binary_agreement") for f in filas])
    a = {
        "A1_margin": {
            "improved_fraction": float((d_margen < 0).mean()),
            "median_diff_mm": float(np.median(d_margen)),
            "pass": bool((d_margen < 0).mean() >= MEJORA_MINIMA_ARCADAS
                         and np.median(d_margen) < 0)},
        "A2_agreement": {
            **acuerdos,
            "pass": bool(all(v["mean_diff"] >= 0 and v["arches_below"] <= ARCADAS_CON_RETROCESO
                             for v in acuerdos.values()))},
        "A3_overtrim": {
            "false_gum_rise": float((encia[:, 0] - encia[:, 1]).mean()),
            "teeth": int(piezas), "newly_under_retained": int(nuevas),
            "newly_under_retained_fraction": float(nuevas / piezas) if piezas else None,
            "pass": bool((encia[:, 0] - encia[:, 1]).mean() <= SUBIDA_FALSA_ENCIA
                         and piezas and nuevas / piezas <= PIEZAS_DANADAS)},
        "A4_labelled_fraction": {
            "mean_abs_candidate": float(np.abs(etiquetado[:, 0]).mean()),
            "mean_abs_base": float(np.abs(etiquetado[:, 1]).mean()),
            "mean_candidate": float(etiquetado[:, 0].mean()),
            "pass": bool(np.abs(etiquetado[:, 0]).mean() < np.abs(etiquetado[:, 1]).mean()
                         and etiquetado[:, 0].mean() >= ETIQUETADO_DE_MENOS)},
        "A5_width": {
            **{q: {"candidate": float(v[:, 0].mean()), "base": float(v[:, 1].mean())}
               for q, v in anchos.items()},
            "pass": bool(all(v[:, 0].mean() <= v[:, 1].mean() for v in anchos.values()))},
        "A6_truth_untouched": {
            "agreement": float(verdad.mean()),
            "pass": bool(verdad.mean() >= ACUERDO_SOBRE_VERDAD)},
    }
    a["candidate"], a["baseline"] = CANDIDATO, BASE
    a["all_pass"] = all(v["pass"] for k, v in a.items() if k.startswith("A"))
    return a


def veredicto_md(c: dict) -> list[str]:
    if not c:
        return []
    si = lambda v: "cumple" if v else "**NO cumple**"  # noqa: E731
    a1, a2, a3 = c["A1_margin"], c["A2_agreement"], c["A3_overtrim"]
    a4, a5, a6 = c["A4_labelled_fraction"], c["A5_width"], c["A6_truth_untouched"]
    return [
        f"## Criterios de aceptación: `{c['candidate']}` frente a `{c['baseline']}`", "",
        "| Criterio | Medido | Resultado |", "|---|---|---|",
        f"| A1 margen | mejora en el {a1['improved_fraction']:.0%} de las arcadas; mediana "
        f"{a1['median_diff_mm']:+.2f} mm | {si(a1['pass'])} |",
        f"| A2 acuerdo | diente/encía {a2['binary_agreement']['mean_diff']:+.3f}, FDI "
        f"{a2['fdi_agreement']['mean_diff']:+.3f}; arcadas que retroceden "
        f"{a2['binary_agreement']['arches_below']:.0%} y "
        f"{a2['fdi_agreement']['arches_below']:.0%} | {si(a2['pass'])} |",
        f"| A3 sobre-recorte | falsa encía {a3['false_gum_rise'] * 100:+.2f} puntos; piezas "
        f"dañadas de nuevo {a3['newly_under_retained']} de {a3['teeth']} "
        f"({a3['newly_under_retained_fraction']:.1%}) | {si(a3['pass'])} |",
        f"| A4 fracción etiquetada | distancia al experto {a4['mean_abs_base']:.3f} → "
        f"{a4['mean_abs_candidate']:.3f}; sesgo {a4['mean_candidate']:+.3f} | {si(a4['pass'])} |",
        f"| A5 ancho | p50 {a5['p50']['base']:.2f} → {a5['p50']['candidate']:.2f} mm; p90 "
        f"{a5['p90']['base']:.2f} → {a5['p90']['candidate']:.2f} mm | {si(a5['pass'])} |",
        f"| A6 no daña la verdad | acuerdo {a6['agreement']:.4f} sobre etiquetas de experto "
        f"| {si(a6['pass'])} |",
        "", f"**Veredicto: {'se cumplen todos' if c['all_pass'] else 'NO se cumplen todos'}.**", ""]


# ── Órdenes ─────────────────────────────────────────────────────────────────────

def predice(args: argparse.Namespace) -> int:
    """Corre el modelo sobre las arcadas del split y guarda sus etiquetas. Necesita GPU."""
    import torch

    spec = importlib.util.spec_from_file_location("segmentar_fdi",
                                                  RAIZ / "scripts/segmentar_fdi.py")
    assert spec and spec.loader
    seg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = seg
    spec.loader.exec_module(seg)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, codigos, n_clases, meta = seg.carga_modelo(args.checkpoint, dev)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(f"modelo {meta['name']} · semilla {meta['seed']} · {dev} · sha256 {sha[:12]}")
    pares = arcadas_de(args.split, args.limite)
    for k, (paciente, arcada) in enumerate(pares, 1):
        destino = ruta_prediccion(paciente, arcada)
        if destino.is_file() and not args.rehace:
            continue
        pos, tri, _ = carga_arcada(paciente, arcada)
        normales = seg.normales_vtk(pos.astype(np.float32), tri).astype(np.float64)
        # ⚠️ La inferencia baraja los vértices en trozos, y sin semilla dos ejecuciones
        # dan etiquetas distintas: el banco compararía métodos sobre entradas diferentes.
        torch.manual_seed(0)
        # Sin canonizar: Teeth3DS+ YA está en la pose de entrenamiento.
        _, instancias = seg.infiere(modelo, codigos, n_clases, pos, pos, normales,
                                    arcada, dev)
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destino, etiquetas=seg.etiquetas_de(instancias, len(pos)),
                            checkpoint_sha256=sha)
        print(f"  [{k}/{len(pares)}] {paciente}_{arcada}: {len(instancias)} piezas")
    return 0


def _columna(resumen: dict, nombre: str, formato: str = "{:.3f}") -> str:
    valor = resumen.get(nombre, {}).get("mean")
    return "—" if valor is None else formato.format(valor)


def informe_md(salida: dict) -> str:
    lineas = [
        f"# Banco de pruebas del margen corona-encía · split `{salida['split']}`", "",
        f"{salida['arches']} arcadas de {salida['patients']} pacientes de Teeth3DS+ que el "
        f"modelo no vio al entrenar. Protocolo {salida['protocol']}. Medias por arcada.", "",
        "| Fila | Método | Acuerdo | Falso diente | Falsa encía | Margen p50 (mm) "
        "| Signo (mm) | Etiquetado − experto | Ancho p90 (mm) | Piezas < 85 % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for clave, resumen in salida["summary"].items():
        lineas.append(
            f"| {clave} | {DESCRIPCIONES.get(clave, clave)} "
            f"| {_columna(resumen, 'binary_agreement')} "
            f"| {_columna(resumen, 'false_tooth_rate')} "
            f"| {_columna(resumen, 'false_gum_rate')} "
            f"| {_columna(resumen, 'margin_p50_mm', '{:.2f}')} "
            f"| {_columna(resumen, 'margin_signed_mm', '{:+.2f}')} "
            f"| {_columna(resumen, 'labelled_area_paired_diff', '{:+.3f}')} "
            f"| {_columna(resumen, 'width_error_p90_mm', '{:.2f}')} "
            f"| {resumen['teeth_under_retained']} de {resumen['teeth_expert']} |")
    lineas += [
        "", "- **Falso diente:** de lo que el experto llama encía, cuánto se etiqueta diente.",
        "- **Falsa encía:** de lo que el experto llama diente, cuánto se etiqueta encía.",
        "- **Signo:** positivo si el margen predicho cae en encía del experto.",
        "- **Piezas < 85 %:** piezas que conservan menos del 85 % de su corona de experto.",
        "", *veredicto_md(salida.get("acceptance", {})),
        "Comprobación de la partición (modelo sin tocar, acierto FDI en vértices de "
        "diente y piezas con el código correcto): " + "; ".join(
            f"{jaw} {v['tooth_fdi_agreement']:.3f} y {v['teeth_majority_correct']:.3f}"
            for jaw, v in salida.get("leak_check", {}).items()
            if v["tooth_fdi_agreement"] is not None) + ".",
        "", "Estas cifras son sobre Teeth3DS+. No miden los casos clínicos propios, de los "
        "que no hay anotación.", ""]
    return "\n".join(lineas)


def evalua(args: argparse.Namespace) -> int:
    pares = arcadas_de(args.split, args.limite)
    faltan = [p for p in pares if not ruta_prediccion(*p).is_file()]
    if faltan:
        print(f"Faltan {len(faltan)} predicciones; ejecuta `predice --split {args.split}`.",
              file=sys.stderr)
        return 2
    cache = SALIDA / f"arcadas-{args.split}.jsonl"
    hechas: dict[tuple[str, str], dict] = {}
    if args.reanuda and cache.is_file():
        for linea in cache.read_text().splitlines():
            fila = json.loads(linea)
            hechas[(fila["patient"], fila["jaw"])] = fila
    pendientes = [p for p in pares if p not in hechas]
    SALIDA.mkdir(parents=True, exist_ok=True)
    modo = "a" if hechas else "w"
    with cache.open(modo) as f, ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        for k, fila in enumerate(grupo.map(_evalua, pendientes), 1):
            hechas[(fila["patient"], fila["jaw"])] = fila
            f.write(json.dumps(fila) + "\n")
            f.flush()
            print(f"  [{k}/{len(pendientes)}] {fila['patient']}_{fila['jaw']} "
                  f"({sum(fila['seconds'].values()):.0f} s)")
    filas = [hechas[p] for p in pares]
    claves = list(filas[0]["methods"])
    por_metodo = {k: [f["methods"][k] for f in filas] for k in claves}
    ids = sorted({p for p, _ in pares})
    salida = {
        "protocol": VERSION_PROTOCOLO, "split": args.split,
        "patients": len(ids), "arches": len(filas), "split_sha256": huella(ids),
        "checkpoint_sha256": sorted({f["checkpoint_sha256"] for f in filas}),
        "descriptions": DESCRIPCIONES,
        "summary": {k: agrega(v) for k, v in por_metodo.items()},
        # Cada método contra el pipeline de hoy, arcada a arcada.
        "paired_vs_M1": {
            k: [compara(v, por_metodo["M1"], nombre, ruta,
                        menor_es_mejor=nombre not in MAYOR_ES_MEJOR)
                for nombre, ruta in RESUMIBLES]
            for k, v in por_metodo.items() if k not in ("M1", "E0", "E1")},
        # La cifra del notebook de entrenamiento, para comprobar la partición: sobre sus
        # 20 maxilares de test dio 0,898 por punto de diente y 0,958 por pieza. En `fuga`
        # —pacientes de entrenamiento— tiene que salir claramente más alta.
        "leak_check": {
            jaw: {"tooth_fdi_agreement": agrega(
                      [f["methods"]["M0"] for f in filas if f["jaw"] == jaw]
                  )["tooth_fdi_agreement"].get("mean"),
                  "teeth_majority_correct": agrega(
                      [f["methods"]["M0"] for f in filas if f["jaw"] == jaw]
                  )["teeth_majority_correct_fraction"]}
            for jaw in ARCADAS},
        "acceptance": criterios(filas),
        "scope": "Teeth3DS+ held-out patients only; not the project's clinical cases",
    }
    destino_json = args.json or SALIDA / f"resumen-{args.split}.json"
    destino_md = args.md or SALIDA / f"resumen-{args.split}.md"
    destino_json.write_text(json.dumps(salida, indent=2) + "\n")
    destino_md.write_text(informe_md(salida))
    print("\n" + informe_md(salida))
    print(f"Informe: {destino_json} · {destino_md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    for nombre, funcion in (("predice", predice), ("evalua", evalua)):
        sub = ordenes.add_parser(nombre)
        sub.add_argument("--split", choices=("dev", "eval", "fuga"), required=True)
        sub.add_argument("--limite", type=int, help="Solo las N primeras arcadas.")
        sub.set_defaults(funcion=funcion)
        if nombre == "predice":
            sub.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
            sub.add_argument("--rehace", action="store_true",
                             help="Vuelve a predecir aunque exista la caché.")
        else:
            sub.add_argument("--trabajos", type=int, default=4)
            sub.add_argument("--reanuda", action="store_true",
                             help="Reutiliza las arcadas ya evaluadas de este split.")
            sub.add_argument("--json", type=Path)
            sub.add_argument("--md", type=Path)
    args = ap.parse_args()
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
