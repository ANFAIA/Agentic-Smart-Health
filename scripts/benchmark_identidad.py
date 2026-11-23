#!/usr/bin/env python
"""El verificador (E1) y la renumeración (E2) de §6.11, medidos contra C2, cada uno aparte.

    G=~/.venvs/dental-gpu/bin/python
    $G scripts/benchmark_identidad.py c2 --split dev-privado        # R y p₁ (GPU)
    $G scripts/benchmark_identidad.py puntua --split dev-privado --modelo …/pieza.pt
    uv run python scripts/benchmark_identidad.py elige --modelo …/pieza.pt
    uv run python scripts/benchmark_identidad.py evalua --modelo …/pieza.pt   # UNA vez

    # verificador v2 (§6.12): solo E1, con `--version 2` en puntua, elige, evalua y clinico
    # área de cada corona en las arcadas de entrenamiento de Teeth3DS+ (§6.13), sin GPU
    uv run python scripts/benchmark_identidad.py areas
    # y el umbral premolar/primer molar sobre las instancias de C2 en esas arcadas (§6.14)
    uv run python scripts/benchmark_identidad.py areas --c2
    # verificador v3 (§6.13): el mismo modelo y las puntuaciones de la v2, otra regla
    uv run python scripts/benchmark_identidad.py elige --version 3 --modelo …/pieza-v2.pt
    $G scripts/benchmark_identidad.py puntua --version 2 --split dev-privado --modelo …/pieza-v2.pt

**Los conjuntos** (§6.11 de `docs/research/segmentacion-fdi-escaner.md`):

- `dev-privado`: la primera reserva, ya gastada (120 arcadas del test privado). Aquí se
  elige τ y w y se mira todo lo que haga falta.
- `dev`: el de siempre; solo para V5 y N5, que piden no empeorar ahí.
- `reserva-2`: 60 pacientes del test privado que nadie ha mirado
  (`teeth3ds-reserva-2.txt`). Se evalúa UNA vez y el script se niega a repetirlo.

**R, la referencia,** es C2 en el pipeline real: las mismas predicciones cacheadas de
`benchmark_separacion.py`, el voto entre las 3 referencias más confiadas de 18 y la limpieza
de `caso_completo.py`. `c2` recalcula, solo para esas 3, la log-probabilidad de C2 que E2
necesita (p₁), y comprueba que las etiquetas salen idénticas a las cacheadas.

**La versión 2** (§6.12) recorta cada instancia desde su componente principal, guarda en
`pieza-v2/` y solo elige y evalúa E1. **La versión 3** (§6.13) usa ese mismo modelo y esas
mismas puntuaciones, sin `puntua` propio, y cambia solo la regla: `verifica_v3`. La reserva
nueva se evalúa una vez en total, sea con la versión que sea: `evalua` se niega si ya existe
cualquier resumen de ella.

Los recortes de cada pieza se ponen en la pose de la referencia más confiada, que es la pose
de Teeth3DS+ con la que se entrenó el modelo de pieza. El orden desde la línea media se mide
en el marco de la propia arcada (`fusion_agents.marco.marco_canonico`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

RESUMEN_EN = "Scores the per-tooth verifier and renumbering against C2, each on its own."
RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))

from analysis_agents.identidad import (  # noqa: E402
    MARGEN_RECORTE_MM,
    PESOS_C2,
    TAUS,
    areas_de_instancia,
    componente_principal,
    distancia_con_signo,
    instancias,
    p1_por_instancia,
    recorte,
    renumera,
    verifica,
    verifica_v3,
)
from analysis_agents.margen_metricas import (  # noqa: E402
    CORONA_MINIMA_CONSERVADA,
    identidad_de_piezas,
    separacion_interproximal,
)
from analysis_agents.pose_votada import MEJORES, vota  # noqa: E402


def _carga(nombre: str, ruta: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(nombre, modulo)
    spec.loader.exec_module(modulo)
    return sys.modules[nombre]


bs = _carga("benchmark_separacion", RAIZ / "scripts/benchmark_separacion.py")
ep = _carga("entrena_pieza", RAIZ / "scripts/entrena_pieza.py")
bm, bp = bs.bm, bs.bp
SALIDA = RAIZ / "data/processed/identidad"
# El nombre de cada conjunto aquí → el que usa la caché de `benchmark_separacion.py`.
SPLITS = {"dev": "dev", "dev-privado": "reserva", "reserva-2": "reserva-2"}
SEMILLA_RECORTE = 0

# ── Criterios, fijados en §6.11 (commit 4038be6) antes de escribir este script ──────
FANTASMAS_E1 = 0.5        # V1: fantasmas ≤ 0,5 × R
TOLERANCIA_PIEZAS = 0.005  # V2, V3, N3 y V5/N5: 0,5 % de las piezas del experto
DESPLAZADAS_E2 = 0.75     # N1: desplazadas ≤ 0,75 × R
RETROCESO_FDI = 0.005     # N4: acierto FDI por vértice de diente ≥ R − 0,005
SUBIDA_CONTACTO_MM = 0.02  # los dos juntos: contacto p50 ≤ R + 0,02 mm


def ruta_c2(split: str, paciente: str, arcada: str) -> Path:
    return SALIDA / "c2" / split / f"{paciente}_{arcada}.npz"


def ruta_pieza(split: str, paciente: str, arcada: str, version: int = 1) -> Path:
    # La versión 3 reutiliza las puntuaciones de la 2: solo cambia la regla.
    return SALIDA / ("pieza-v2" if version >= 2 else "pieza") / split / f"{paciente}_{arcada}.npz"


def ruta_eleccion(version: int) -> Path:
    return SALIDA / (f"eleccion-v{version}.json" if version >= 2 else "eleccion.json")


def recorte_de_instancia(pos, nrm, vs, arbol, rng, version: int):  # type: ignore[no-untyped-def]
    """El recorte que ve el modelo de pieza; desde la versión 2, desde la componente principal."""
    if version >= 2:
        vs = componente_principal(pos, vs)
    idx, d = distancia_con_signo(pos, vs, MARGEN_RECORTE_MM, arbol)
    return recorte(pos, nrm, idx, d, rng)


def arcadas(split: str, limite: int | None = None) -> list[tuple[str, str]]:
    return bs.arcadas(SPLITS[split], limite)


def parte(pares: list, cual: str) -> list:  # type: ignore[type-arg]
    """`i/n`: las arcadas i, i+n, … La agregación y la limpieza son de un núcleo."""
    i, n = (int(x) for x in cual.split("/"))
    return pares[i::n]


# ── c2: R, p₁ y el marco de cada arcada (GPU) ───────────────────────────────────

def c2(args: argparse.Namespace) -> int:
    import torch
    from fusion_agents.marco import a_marco_de, marco_canonico

    ep.firma_c2()
    seg = _carga("segmentar_fdi", RAIZ / "scripts/segmentar_fdi.py")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, codigos, n_clases, meta = seg.carga_modelo(ep.C2, dev)
    refs = bp.referencias()
    marcos = {(r, a): marco_canonico(np.asarray(
        bm.parse_obj(bm.MALLAS / r / f"{r}_{a}.obj")["positions"], dtype=np.float64))[:2]
        for r in refs for a in bm.ARCADAS}
    pares = parte(arcadas(args.split, args.limite), args.parte)
    (SALIDA / "c2" / args.split).mkdir(parents=True, exist_ok=True)
    (SALIDA / "c2" / args.split / "run-record.json").write_text(
        json.dumps(bs.registro(sys.argv), indent=2) + "\n")
    distintas = 0
    for k, (paciente, arcada) in enumerate(pares, 1):
        destino = ruta_c2(args.split, paciente, arcada)
        if destino.is_file():
            continue
        z = np.load(bs.ruta(ep.C2_SHA, SPLITS[args.split], paciente, arcada))
        pos, tri, _ = bs.carga(SPLITS[args.split], paciente, arcada)
        normales = seg.normales_vtk(pos.astype(np.float32), tri).astype(np.float64)
        origen = marco_canonico(pos)
        conf = {r: float(z[f"conf_{r}"]) for r in refs}
        votada, registro_voto = vota({r: z[f"et_{r}"].astype(np.int64) for r in refs}, conf)
        elegidas = registro_voto["chosen"]
        assert len(elegidas) == MEJORES
        logps = []
        for r in elegidas:
            v_ent, n_ent = a_marco_de(pos, origen[:2], marcos[(r, arcada)], normales=normales)
            torch.manual_seed(0)
            logp, inst = seg.infiere(modelo, codigos, n_clases, pos, v_ent, n_ent, arcada,
                                     dev, trozo=meta["num_points"])
            distintas += int((seg.etiquetas_de(inst, len(pos)) != z[f"et_{r}"]).sum())
            logps.append(logp)
        R = bm.limpieza_actual(pos, votada)
        p1 = p1_por_instancia(logps, codigos, R)
        cods = sorted(p1)
        temporal = destino.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporal, R=R.astype(np.int16), p1_codigos=np.array(cods, dtype=np.int16),
            p1=np.stack([p1[c] for c in cods]) if cods else np.zeros((0, 8)),
            elegidas=np.array(elegidas), centro=origen[0], marco=origen[1])
        temporal.replace(destino)
        print(f"  [{k}/{len(pares)}] {paciente}_{arcada}", flush=True)
    print(f"vértices con etiqueta distinta de la cacheada al recalcular: {distintas}")
    (SALIDA / "c2" / args.split / f"reproduccion-{args.parte.replace('/', 'de')}.json"
     ).write_text(json.dumps({"vertices_distintos": distintas}) + "\n")
    return 0


# ── puntua: el modelo de pieza sobre cada instancia de R (GPU) ───────────────────

def carga_pieza(ruta: Path, dev: str):  # type: ignore[no-untyped-def]
    import torch

    ck = torch.load(ruta, map_location=dev, weights_only=False)
    modelo = ep.construye_modelo().to(dev)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()
    return modelo, ck


def puntua(args: argparse.Namespace) -> int:
    import torch
    from fusion_agents.marco import a_marco_de, marco_canonico
    from scipy.spatial import cKDTree

    seg = _carga("segmentar_fdi", RAIZ / "scripts/segmentar_fdi.py")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, _ = carga_pieza(args.modelo, dev)
    refs = bp.referencias()
    marcos = {(r, a): marco_canonico(np.asarray(
        bm.parse_obj(bm.MALLAS / r / f"{r}_{a}.obj")["positions"], dtype=np.float64))[:2]
        for r in refs for a in bm.ARCADAS}
    pares = parte(arcadas(args.split, args.limite), args.parte)
    for k, (paciente, arcada) in enumerate(pares, 1):
        destino = ruta_pieza(args.split, paciente, arcada, args.version)
        if destino.is_file():
            continue
        z = np.load(ruta_c2(args.split, paciente, arcada))
        pos, tri, _ = bs.carga(SPLITS[args.split], paciente, arcada)
        normales = seg.normales_vtk(pos.astype(np.float32), tri).astype(np.float64)
        origen = (z["centro"], z["marco"])
        v1, n1 = a_marco_de(pos, origen, marcos[(str(z["elegidas"][0]), arcada)],
                            normales=normales)
        arbol = cKDTree(v1)
        rng = np.random.default_rng(SEMILLA_RECORTE)
        cods, recortes = [], []
        for c, vs in instancias(z["R"]).items():
            r = recorte_de_instancia(v1, n1, vs, arbol, rng, args.version)
            assert r is not None
            cods.append(c)
            recortes.append(r)
        p_diente, p2 = puntua_recortes(modelo, recortes, dev)
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_suffix(".tmp.npz")
        np.savez_compressed(temporal, codigos=np.array(cods, dtype=np.int16),
                            p_diente=p_diente, p2=p2)
        temporal.replace(destino)
        print(f"  [{k}/{len(pares)}] {paciente}_{arcada}: {len(cods)} piezas", flush=True)
    return 0


def puntua_recortes(modelo, recortes, dev: str, lote: int = 16):  # type: ignore[no-untyped-def]
    import torch

    pd, pp = [], []
    with torch.no_grad():
        for s in range(0, len(recortes), lote):
            X, P, b = ep.a_tensores(recortes[s: s + lote], dev)
            sal_d, sal_p = modelo(X, P, b)
            pd.append(torch.sigmoid(sal_d).cpu().numpy())
            pp.append(sal_p.cpu().numpy())
    if not pd:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 8), dtype=np.float32)
    return np.concatenate(pd).astype(np.float32), np.concatenate(pp).astype(np.float32)


# ── Variantes y métricas (CPU) ──────────────────────────────────────────────────

def variantes(R, pos, centro, marco, p_diente, p2, p1, tau=None, w=None,  # type: ignore[no-untyped-def]
              version=1, areas=None, registro=None):
    """`{nombre: etiquetas}`: R, E1 con cada τ, E2 con cada w y, si se dan, E1+E2.

    En la versión 3, E1 es `verifica_v3` con las `areas` de cada instancia, y lo que decide
    en cada τ se deja en `registro`.
    """
    salida = {"R": R}
    for t in TAUS:
        if version == 3:
            salida[f"E1_{t}"], reg = verifica_v3(R, p_diente, t, areas)
            if registro is not None:
                registro[str(t)] = {"abstains": reg["abstains"],
                                    "too_small": len(reg["too_small"])}
        else:
            salida[f"E1_{t}"] = verifica(R, p_diente, t)[0]
    for peso in PESOS_C2:
        salida[f"E2_{peso}"] = renumera(R, pos, centro, marco, p2, p1, peso)[0]
    if tau is not None and w is not None:
        salida["E1+E2"] = renumera(verifica(R, p_diente, tau)[0], pos, centro, marco,
                                   p2, p1, w)[0]
    return salida


def metricas(pred: np.ndarray, experto: np.ndarray) -> dict[str, Any]:
    ide = identidad_de_piezas(pred, experto)
    bien = sorted({e["expert"] for e in ide["per_label"].values() if e["status"] == "correct"})
    dientes = experto > 0
    return {
        **{k: ide[k] for k in ("correct", "shifted", "neighbour_shift", "ghost", "split",
                               "missing", "expert")},
        "correct_teeth": bien,
        "retained": {str(t): float((pred[experto == t] > 0).mean())
                     for t in (int(x) for x in np.unique(experto) if x > 0)},
        "tooth_fdi_agreement": float((pred[dientes] == experto[dientes]).mean())
        if dientes.any() else None,
    }


def evalua_arcada(trabajo: tuple[str, str, str, float | None, float | None, bool, int]
                  ) -> dict:
    split, paciente, arcada, tau, w, con_contacto, version = trabajo
    pos, tri, experto = bs.carga(SPLITS[split], paciente, arcada)
    z = np.load(ruta_c2(split, paciente, arcada))
    s = np.load(ruta_pieza(split, paciente, arcada, version))
    p1 = {int(c): v for c, v in zip(z["p1_codigos"], z["p1"], strict=True)}
    p_diente = {int(c): float(v) for c, v in zip(s["codigos"], s["p_diente"], strict=True)}
    p2 = {int(c): v for c, v in zip(s["codigos"], s["p2"], strict=True)}
    R = z["R"].astype(np.int64)
    v3: dict[str, Any] = {}
    vs = variantes(R, pos, z["centro"], z["marco"], p_diente, p2, p1, tau, w, version,
                   areas_de_instancia(pos, tri, R) if version == 3 else None, v3)
    fila: dict[str, Any] = {"patient": paciente, "jaw": arcada,
                            "variants": {k: metricas(v, experto) for k, v in vs.items()}}
    if version == 3:
        fila["v3"] = v3
    if con_contacto:
        fila["contact_p50_mm"] = {
            k: [c["p50_mm"] for c in separacion_interproximal(pos, tri, vs[k], experto)
                ["contacts"] if c.get("p50_mm") is not None]
            for k in ("R", "E1+E2") if k in vs}
    return fila


def suma(filas: list[dict], var: str, clave: str) -> int:
    return int(sum(f["variants"][var][clave] for f in filas))


def nuevas_bajo_corona(filas: list[dict], var: str) -> int:
    """Piezas que con R conservaban ≥ 85 % de su corona y con `var` ya no."""
    return sum(1 for f in filas for t, r in f["variants"]["R"]["retained"].items()
               if r >= CORONA_MINIMA_CONSERVADA
               and f["variants"][var]["retained"][t] < CORONA_MINIMA_CONSERVADA)


def rotas(filas: list[dict], var: str) -> int:
    """Piezas bien numeradas con R que dejan de estarlo con `var`: el fallo de §6.5."""
    return sum(len(set(f["variants"]["R"]["correct_teeth"])
                   - set(f["variants"][var]["correct_teeth"])) for f in filas)


def fdi_medio(filas: list[dict], var: str) -> float:
    v = [f["variants"][var]["tooth_fdi_agreement"] for f in filas]
    return float(np.mean([x for x in v if x is not None]))


def criterios_e1(filas: list[dict], var: str) -> dict[str, Any]:
    N = suma(filas, "R", "expert")
    g = {p: suma(filas, p, "ghost") for p in ("R", var)}
    m = {p: suma(filas, p, "missing") for p in ("R", var)}
    c = {p: suma(filas, p, "correct") for p in ("R", var)}
    nuevas = nuevas_bajo_corona(filas, var)
    return {
        "teeth": N,
        "V1_ghosts": {"R": g["R"], "E1": g[var], "pass": g[var] <= FANTASMAS_E1 * g["R"]},
        "V2_missing": {"R": m["R"], "E1": m[var],
                       "pass": m[var] <= m["R"] + TOLERANCIA_PIEZAS * N},
        "V3_crowns": {"newly_under_retained": nuevas, "pass": nuevas <= TOLERANCIA_PIEZAS * N},
        "V4_correct": {"R": c["R"], "E1": c[var], "pass": c[var] >= c["R"]},
    }


def criterios_e2(filas: list[dict], var: str) -> dict[str, Any]:
    N = suma(filas, "R", "expert")
    d = {p: suma(filas, p, "shifted") for p in ("R", var)}
    c = {p: suma(filas, p, "correct") for p in ("R", var)}
    rot = rotas(filas, var)
    f = {p: fdi_medio(filas, p) for p in ("R", var)}
    return {
        "teeth": N,
        "N1_shifted": {"R": d["R"], "E2": d[var], "pass": d[var] <= DESPLAZADAS_E2 * d["R"]},
        "N2_correct": {"R": c["R"], "E2": c[var], "pass": c[var] >= c["R"]},
        "N3_broken": {"broken": rot, "pass": rot <= TOLERANCIA_PIEZAS * N},
        "N4_tooth_fdi": {"R": f["R"], "E2": f[var], "pass": f[var] >= f["R"] - RETROCESO_FDI},
    }


def criterio_dev(filas: list[dict], var: str, fantasmas: bool) -> dict[str, Any]:
    """V5 / N5: en `dev`, no empeorar la numeración (y, para E1, tampoco los fantasmas)."""
    N = suma(filas, "R", "expert")
    c = {p: suma(filas, p, "correct") for p in ("R", var)}
    ok = c[var] >= c["R"] - TOLERANCIA_PIEZAS * N
    salida: dict[str, Any] = {"teeth": N, "correct_R": c["R"], "correct": c[var]}
    if fantasmas:
        g = {p: suma(filas, p, "ghost") for p in ("R", var)}
        salida.update(ghost_R=g["R"], ghost=g[var])
        ok = ok and g[var] <= g["R"]
    salida["pass"] = bool(ok)
    return salida


def _todas(c: dict[str, Any]) -> bool:
    return all(v["pass"] for k, v in c.items() if isinstance(v, dict) and "pass" in v)


def _filas(split: str, tau=None, w=None, contacto=False, trabajos=8, limite=None,  # type: ignore[no-untyped-def]
           version=1):
    pares = arcadas(split, limite)
    faltan = [p for p in pares if not ruta_pieza(split, *p, version).is_file()
              or not ruta_c2(split, *p).is_file()]
    if faltan:
        raise SystemExit(f"⚠ {split}: faltan {len(faltan)} arcadas por `c2` o `puntua`.")
    with ProcessPoolExecutor(max_workers=trabajos) as grupo:
        return list(grupo.map(evalua_arcada,
                              [(split, p, a, tau, w, contacto, version)
                               for p, a in pares]))


# ── elige: τ y w, solo en `dev-privado` y `dev` ─────────────────────────────────

def elige(args: argparse.Namespace) -> int:
    filas = _filas("dev-privado", trabajos=args.trabajos, version=args.version)
    filas_dev = _filas("dev", trabajos=args.trabajos, version=args.version)
    e1 = {}
    for t in TAUS:
        c = criterios_e1(filas, f"E1_{t}")
        c["V5_dev"] = criterio_dev(filas_dev, f"E1_{t}", fantasmas=True)
        c["all_pass"] = _todas(c)
        if args.version == 3:
            c["v3"] = {"dev-privado": resumen_v3(filas, t), "dev": resumen_v3(filas_dev, t)}
        e1[str(t)] = c
    e2 = {}
    # §6.12 y §6.13: desde la versión 2 solo es el verificador; E2 no se vuelve a evaluar.
    for w in (PESOS_C2 if args.version == 1 else ()):
        c = criterios_e2(filas, f"E2_{w}")
        c["N5_dev"] = criterio_dev(filas_dev, f"E2_{w}", fantasmas=False)
        c["all_pass"] = _todas(c)
        e2[str(w)] = c
    # Regla de §6.11: τ con menos fantasmas entre los que cumplen; si empatan, el menor.
    # w con menos desplazadas entre los que cumplen; si empatan, el mayor.
    validos_t = [t for t in TAUS if e1[str(t)]["all_pass"]]
    validos_w = [w for w in PESOS_C2 if str(w) in e2 and e2[str(w)]["all_pass"]]
    tau = min(validos_t, key=lambda t: (e1[str(t)]["V1_ghosts"]["E1"], t)) if validos_t \
        else None
    w = min(validos_w, key=lambda x: (e2[str(x)]["N1_shifted"]["E2"], -x)) if validos_w \
        else None
    salida = {"set": "dev-privado + dev", "version": args.version, "model": str(args.modelo),
              "model_sha256": bs.sha_de(args.modelo), "E1": e1, "E2": e2,
              "chosen": {"tau": tau, "w": w}, "run_record": bs.registro(sys.argv)}
    destino = ruta_eleccion(args.version)
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    destino.with_suffix(".md").write_text(informe_eleccion(salida))
    print(informe_eleccion(salida))
    return 0


def resumen_v3(filas: list[dict], tau: float) -> dict[str, int]:
    """Lo que decidió `verifica_v3` en un conjunto: informativo, no es criterio (§6.13)."""
    return {"arches_abstained": sum(f["v3"][str(tau)]["abstains"] for f in filas),
            "instances_too_small": sum(f["v3"][str(tau)]["too_small"] for f in filas)}


def si(x: bool) -> str:
    return "cumple" if x else "**NO cumple**"


def informe_eleccion(s: dict) -> str:
    v2 = {2: " · verificador v2 (§6.12)", 3: " · verificador v3 (§6.13)"}.get(
        s.get("version", 1), "")
    lineas = ["# Identidad por pieza" + v2
              + " · elección en `dev-privado` (y `dev` para V5/N5)", "",
              "| E1 τ | Fantasmas R → E1 | Ausentes | Coronas dañadas | Correctas | V5 dev "
              "| Todo |", "|---|---|---|---|---|---|---|"]
    for t, c in s["E1"].items():
        lineas.append(
            f"| {t} | {c['V1_ghosts']['R']} → {c['V1_ghosts']['E1']} "
            f"({si(c['V1_ghosts']['pass'])}) "
            f"| {c['V2_missing']['R']} → {c['V2_missing']['E1']} ({si(c['V2_missing']['pass'])}) "
            f"| {c['V3_crowns']['newly_under_retained']} ({si(c['V3_crowns']['pass'])}) "
            f"| {c['V4_correct']['R']} → {c['V4_correct']['E1']} ({si(c['V4_correct']['pass'])}) "
            f"| {c['V5_dev']['correct_R']} → {c['V5_dev']['correct']}, fantasmas "
            f"{c['V5_dev']['ghost_R']} → {c['V5_dev']['ghost']} ({si(c['V5_dev']['pass'])}) "
            f"| {si(c['all_pass'])} |")
    if s["E2"]:
        lineas += ["", "| E2 w | Desplazadas R → E2 | Correctas | Rotas | FDI por vértice "
                   "| N5 dev | Todo |", "|---|---|---|---|---|---|---|"]
    for w, c in s["E2"].items():
        lineas.append(
            f"| {w} | {c['N1_shifted']['R']} → {c['N1_shifted']['E2']} "
            f"({si(c['N1_shifted']['pass'])}) "
            f"| {c['N2_correct']['R']} → {c['N2_correct']['E2']} ({si(c['N2_correct']['pass'])}) "
            f"| {c['N3_broken']['broken']} ({si(c['N3_broken']['pass'])}) "
            f"| {c['N4_tooth_fdi']['R']:.4f} → {c['N4_tooth_fdi']['E2']:.4f} "
            f"({si(c['N4_tooth_fdi']['pass'])}) "
            f"| {c['N5_dev']['correct_R']} → {c['N5_dev']['correct']} ({si(c['N5_dev']['pass'])}) "
            f"| {si(c['all_pass'])} |")
    if s.get("version", 1) == 3:
        lineas += ["", "Informativo, no es criterio: arcadas en las que se abstiene e instancias "
                   "que no juzga por tamaño.", "", "| τ | `dev-privado` | `dev` |", "|---|---|---|"]
        for t, c in s["E1"].items():
            a, b = c["v3"]["dev-privado"], c["v3"]["dev"]
            lineas.append(f"| {t} | {a['arches_abstained']} · {a['instances_too_small']} "
                          f"| {b['arches_abstained']} · {b['instances_too_small']} |")
    ch = s["chosen"]
    lineas += ["", f"**Elegidos:** τ = {ch['tau']}, w = {ch['w']}. `None` = ninguno cumple; "
               "ese componente se cierra como negativo sin tocar la reserva.", ""]
    return "\n".join(lineas)


# ── evalua: la reserva nueva, UNA vez ───────────────────────────────────────────

def evalua(args: argparse.Namespace) -> int:
    destino = SALIDA / (f"resumen-reserva-2-v{args.version}.json" if args.version >= 2
                        else "resumen-reserva-2.json")
    gastada = bs.reserva_2_gastada()  # el cerrojo es común con `benchmark_separacion testigo`
    hechos = [gastada] if gastada else []
    if hechos:
        print(f"⚠ La reserva nueva ya se evaluó ({hechos[0]}). Se evalúa UNA vez, con la "
              "versión que sea.", file=sys.stderr)
        return 3
    eleccion = json.loads(ruta_eleccion(args.version).read_text())
    if eleccion["model_sha256"] != bs.sha_de(args.modelo):
        raise SystemExit("⚠ el modelo no es el que se eligió en dev-privado; no sigo.")
    tau, w = eleccion["chosen"]["tau"], eleccion["chosen"]["w"]
    if tau is None and w is None:
        raise SystemExit("⚠ ni E1 ni E2 pasaron en dev-privado: la reserva no se toca.")
    rec = bs.registro(sys.argv)
    filas = _filas("reserva-2", tau, w, contacto=tau is not None and w is not None,
                   trabajos=args.trabajos, version=args.version)
    salida: dict[str, Any] = {"set": "reserva-2", "version": args.version, "arches": len(filas),
                              "patients": len({f["patient"] for f in filas}),
                              "chosen": {"tau": tau, "w": w},
                              "model_sha256": eleccion["model_sha256"], "run_record": rec}
    if tau is not None:
        c = criterios_e1(filas, f"E1_{tau}")
        c["V5_dev"] = eleccion["E1"][str(tau)]["V5_dev"]
        c["all_pass"] = _todas(c)
        if args.version == 3:
            c["v3"] = resumen_v3(filas, tau)
        salida["E1"] = c
    if w is not None:
        c = criterios_e2(filas, f"E2_{w}")
        c["N5_dev"] = eleccion["E2"][str(w)]["N5_dev"]
        c["all_pass"] = _todas(c)
        salida["E2"] = c
    if tau is not None and w is not None:
        todos = {k: np.array([x for f in filas for x in f["contact_p50_mm"][k]])
                 for k in ("R", "E1+E2")}
        med = {k: float(np.median(v)) for k, v in todos.items()}
        salida["E1+E2"] = {
            "contact_p50_mm": med,
            "pass": med["E1+E2"] <= med["R"] + SUBIDA_CONTACTO_MM,
            "correct": {k: suma(filas, k, "correct") for k in ("R", "E1+E2")},
            "ghost": {k: suma(filas, k, "ghost") for k in ("R", "E1+E2")},
            "shifted": {k: suma(filas, k, "shifted") for k in ("R", "E1+E2")}}
    salida["per_arch"] = [{"patient": f["patient"], "jaw": f["jaw"],
                           **{k: {m: v[m] for m in ("correct", "shifted", "ghost", "missing")}
                              for k, v in f["variants"].items()}} for f in filas]
    salida["scope"] = "Teeth3DS+ only; not the project's clinical cases"
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    destino.with_suffix(".md").write_text(informe_reserva(salida))
    print(informe_reserva(salida))
    return 0


def informe_reserva(s: dict) -> str:
    lineas = ["# Identidad por pieza"
              + {2: " · verificador v2 (§6.12)", 3: " · verificador v3 (§6.13)"}.get(
                  s.get("version", 1), "")
              + " · `reserva-2`, evaluada una vez", "",
              f"{s['arches']} arcadas de {s['patients']} pacientes. τ = {s['chosen']['tau']}, "
              f"w = {s['chosen']['w']}, elegidos en `dev-privado`.", ""]
    if "E1" in s:
        c = s["E1"]
        lineas += ["| E1 | R | E1 | |", "|---|---:|---:|---|",
                   f"| V1 fantasmas | {c['V1_ghosts']['R']} | {c['V1_ghosts']['E1']} | "
                   f"{si(c['V1_ghosts']['pass'])} |",
                   f"| V2 ausentes | {c['V2_missing']['R']} | {c['V2_missing']['E1']} | "
                   f"{si(c['V2_missing']['pass'])} |",
                   f"| V3 coronas dañadas | | {c['V3_crowns']['newly_under_retained']} de "
                   f"{c['teeth']} | {si(c['V3_crowns']['pass'])} |",
                   f"| V4 correctas | {c['V4_correct']['R']} | {c['V4_correct']['E1']} | "
                   f"{si(c['V4_correct']['pass'])} |",
                   f"| V5 en `dev` | | | {si(c['V5_dev']['pass'])} |", "",
                   f"**E1: {'cumple todos' if c['all_pass'] else 'NO cumple todos'}.**", ""]
        if "v3" in c:
            lineas += [f"Se abstiene en {c['v3']['arches_abstained']} arcadas y deja sin juzgar "
                       f"{c['v3']['instances_too_small']} instancias por tamaño.", ""]
    if "E2" in s:
        c = s["E2"]
        lineas += ["| E2 | R | E2 | |", "|---|---:|---:|---|",
                   f"| N1 desplazadas | {c['N1_shifted']['R']} | {c['N1_shifted']['E2']} | "
                   f"{si(c['N1_shifted']['pass'])} |",
                   f"| N2 correctas | {c['N2_correct']['R']} | {c['N2_correct']['E2']} | "
                   f"{si(c['N2_correct']['pass'])} |",
                   f"| N3 rotas | | {c['N3_broken']['broken']} de {c['teeth']} | "
                   f"{si(c['N3_broken']['pass'])} |",
                   f"| N4 FDI por vértice | {c['N4_tooth_fdi']['R']:.4f} | "
                   f"{c['N4_tooth_fdi']['E2']:.4f} | {si(c['N4_tooth_fdi']['pass'])} |",
                   f"| N5 en `dev` | | | {si(c['N5_dev']['pass'])} |", "",
                   f"**E2: {'cumple todos' if c['all_pass'] else 'NO cumple todos'}.**", ""]
    if "E1+E2" in s:
        c = s["E1+E2"]
        lineas += [f"**Juntos:** contacto p50 {c['contact_p50_mm']['R']:.3f} → "
                   f"{c['contact_p50_mm']['E1+E2']:.3f} mm ({si(c['pass'])}); correctas "
                   f"{c['correct']['R']} → {c['correct']['E1+E2']}, fantasmas "
                   f"{c['ghost']['R']} → {c['ghost']['E1+E2']}, desplazadas "
                   f"{c['shifted']['R']} → {c['shifted']['E1+E2']}.", ""]
    lineas += ["Cifras sobre Teeth3DS+; no miden los casos clínicos propios.", ""]
    return "\n".join(lineas)


# ── clinico: E1 y E2 sobre una segmentación ya hecha, sin ajustar nada ──────────

def clinico(args: argparse.Namespace) -> int:
    """Aplica E1 (τ) y E2 (w) elegidos en `dev-privado` a la salida de `segmentar_fdi.py`.

    Parte de lo que ya se segmentó con C2 en §6.10 (`segmentacion_<arcada>.npz`, la columna
    `limpias`, y el voto de `fdi_<arcada>.json`), recalcula la log-probabilidad de las tres
    referencias que votaron y escribe un npz con `V`, `F` y una columna por variante, que es
    lo que lee `puntua_margen.py`. No escribe nombres de fichero del escaneo.
    """
    import torch
    from fusion_agents.marco import a_marco_de, marco_canonico
    from scipy.spatial import cKDTree

    seg = _carga("segmentar_fdi", RAIZ / "scripts/segmentar_fdi.py")
    eleccion = json.loads(ruta_eleccion(args.version).read_text())
    if eleccion["model_sha256"] != bs.sha_de(args.modelo):
        raise SystemExit("⚠ el modelo no es el que se eligió en dev-privado; no sigo.")
    tau, w = eleccion["chosen"]["tau"], eleccion["chosen"]["w"]
    ep.firma_c2()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c2_modelo, codigos, n_clases, meta = seg.carga_modelo(ep.C2, dev)
    pieza, _ = carga_pieza(args.modelo, dev)
    z = np.load(args.segmentacion)
    V, F = z["V"].astype(np.float64), z["F"].astype(np.int64)
    R = z["limpias"].astype(np.int64)
    voto = json.loads(args.fdi_json.read_text())["pose"]["vote"]["chosen"]
    rutas = {r.stem: r for r in seg.referencias_de_entrenamiento(18, args.arcada)}
    N = seg.normales_vtk(V.astype(np.float32), F).astype(np.float64)
    origen = marco_canonico(V)
    marcos = {r: marco_canonico(np.asarray(bm.parse_obj(rutas[r])["positions"],
                                           dtype=np.float64))[:2] for r in voto}
    logps, etqs, confs = [], {}, {}
    from analysis_agents.pose_votada import confianza

    for r in voto:
        v_ent, n_ent = a_marco_de(V, origen[:2], marcos[r], normales=N)
        torch.manual_seed(0)
        logp, inst = seg.infiere(c2_modelo, codigos, n_clases, V, v_ent, n_ent, args.arcada,
                                 dev, trozo=meta["num_points"])
        logps.append(logp)
        etqs[r] = seg.etiquetas_de(inst, len(V)).astype(np.int64)
        confs[r] = confianza(logp, etqs[r])
    # Solo votan esas tres, así que su voto tiene que reproducir el de §6.10.
    distintas = int((vota(etqs, confs)[0] != z["etiquetas"]).sum())
    p1 = p1_por_instancia(logps, codigos, R)
    v1, n1 = a_marco_de(V, origen[:2], marcos[voto[0]], normales=N)
    arbol, rng = cKDTree(v1), np.random.default_rng(SEMILLA_RECORTE)
    cods, recortes = [], []
    for c, vs in instancias(R).items():
        r_ = recorte_de_instancia(v1, n1, vs, arbol, rng, args.version)
        assert r_ is not None
        cods.append(c)
        recortes.append(r_)
    pd, pp = puntua_recortes(pieza, recortes, dev)
    p_diente = {c: float(v) for c, v in zip(cods, pd, strict=True)}
    p2 = {c: v for c, v in zip(cods, pp, strict=True)}
    columnas: dict[str, np.ndarray] = {"R": R}
    registro: dict[str, Any] = {"tau": tau, "w": w, "vote": voto,
                                "vertices_differing_from_cached_vote": distintas,
                                "per_instance": {str(c): {
                                    "p_tooth": round(p_diente[c], 4),
                                    "position_model": int(np.argmax(p2[c])) + 1,
                                    "position_c2": int(np.argmax(p1[c])) + 1}
                                    for c in cods}}
    if tau is not None and args.version == 3:
        columnas["E1"], registro["E1_v3"] = verifica_v3(
            R, p_diente, tau, areas_de_instancia(V, F, R))
    elif tau is not None:
        columnas["E1"], registro["E1_removed"] = verifica(R, p_diente, tau)
    if w is not None:
        columnas["E2"], registro["E2"] = renumera(R, V, origen[0], origen[1], p2, p1, w)
    if tau is not None and w is not None:
        columnas["E1+E2"] = renumera(columnas["E1"], V, origen[0], origen[1], p2, p1, w)[0]
    args.salida.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.salida / f"identidad_{args.arcada}.npz", V=V, F=F,
                        **{k.replace("+", "_"): v.astype(np.int16)
                           for k, v in columnas.items()})
    registro["codes"] = {k: sorted(instancias(v)) for k, v in columnas.items()}
    registro["run_record"] = bs.registro(sys.argv)
    (args.salida / f"identidad_{args.arcada}.json").write_text(
        json.dumps(registro, indent=2) + "\n")
    print(json.dumps({k: v for k, v in registro.items() if k != "run_record"}, indent=1))
    return 0


# ── areas: el tamaño de las coronas en entrenamiento (§6.13) ────────────────────

def _areas_de(par: tuple[str, str]) -> dict[str, float]:
    pos, tri, experto = bm.carga_arcada(*par)
    return {str(c): a for c, a in areas_de_instancia(pos, tri, experto).items()}


def _areas_c2_de(par: tuple[str, str]) -> dict[str, list[float]]:
    """Áreas de las instancias de C2 bien numeradas en una arcada de entrenamiento,
    con la limpieza actual, por posición."""
    minado = ep.MINADO / f"{par[0]}_{par[1]}.npz"
    if not minado.is_file():
        return {}
    pos, tri, experto = bm.carga_arcada(*par)
    c2 = bm.limpieza_actual(pos, np.load(minado)["et"].astype(np.int64))
    ide = identidad_de_piezas(c2, experto)
    a = areas_de_instancia(pos, tri, c2)
    salida: dict[str, list[float]] = {}
    for cod, e in ide["per_label"].items():
        if e["status"] == "correct":
            salida.setdefault(str(int(cod) % 10), []).append(a[int(cod)])
    return salida


def umbral_de_separacion(bajos: list[float], altos: list[float]) -> tuple[float, int]:
    """El área `t` que menos instancias clasifica mal, y cuántas: es molar lo que mide `≥ t`,
    así que un `bajo` falla si llega a `t` y un `alto` si no. Si empatan, el primer óptimo,
    a medio camino entre ese valor observado y el anterior."""
    b, a = np.sort(bajos), np.sort(altos)
    candidatos = np.unique(np.concatenate([b, a]))
    errores = (len(b) - np.searchsorted(b, candidatos, side="left")
               + np.searchsorted(a, candidatos, side="left"))
    k = int(np.argmin(errores))
    t = (candidatos[k - 1] + candidatos[k]) / 2 if k > 0 else candidatos[k]
    return float(t), int(errores[k])


def areas(args: argparse.Namespace) -> int:
    """La mediana del área de cada corona del experto en las arcadas de entrenamiento.

    Solo los pacientes de Teeth3DS+ de la partición `train`: los de OSF que también entran en
    el entrenamiento están en npz sin caras, y el área necesita las caras. No mira `dev`,
    `eval` ni ninguna reserva.
    """
    pares = [(p, a) for p in bm.particion(bm.pacientes())["train"] for a in bm.ARCADAS]
    if args.limite:
        pares = pares[: args.limite]
    if args.c2:
        return areas_c2(args, pares)
    por_codigo: dict[str, list[float]] = {}
    with ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        for k, fila in enumerate(grupo.map(_areas_de, pares, chunksize=4), 1):
            for c, a in fila.items():
                por_codigo.setdefault(c, []).append(a)
            if k % 100 == 0:
                print(f"  [{k}/{len(pares)}]", flush=True)
    resumen = {c: {"n": len(v), "p50_mm2": float(np.median(v)),
                   "p05_mm2": float(np.percentile(v, 5))}
               for c, v in sorted(por_codigo.items(), key=lambda x: int(x[0]))}
    incisivos = por_codigo.get("31", []) + por_codigo.get("41", [])
    salida = {"set": "Teeth3DS+ train (sin OSF)", "arches": len(pares), "by_code": resumen,
              "lower_central_incisor_p50_mm2": float(np.median(incisivos)),
              "run_record": bs.registro(sys.argv)}
    (SALIDA / "areas-entrenamiento.json").write_text(json.dumps(salida, indent=2) + "\n")
    print(json.dumps({k: v for k, v in salida.items() if k not in ("run_record",)}, indent=1))
    return 0


def areas_c2(args: argparse.Namespace, pares: list[tuple[str, str]]) -> int:
    """§6.14: el área que separa un premolar (4, 5) de un primer molar (6), por arcada,
    sobre las instancias de C2 bien numeradas en las arcadas de entrenamiento minadas.

    Son las instancias de C2 y no las coronas del experto porque C2 desborda sobre la encía y
    eso infla el área: el umbral tiene que estar en la escala que se va a medir.
    """
    por: dict[str, dict[str, list[float]]] = {a: {} for a in bm.ARCADAS}
    with ProcessPoolExecutor(max_workers=args.trabajos) as grupo:
        for k, ((_, arcada), fila) in enumerate(
                zip(pares, grupo.map(_areas_c2_de, pares, chunksize=4), strict=True), 1):
            for posicion, v in fila.items():
                por[arcada].setdefault(posicion, []).extend(v)
            if k % 100 == 0:
                print(f"  [{k}/{len(pares)}]", flush=True)
    salida: dict[str, Any] = {"set": "Teeth3DS+ train (sin OSF), instancias de C2 bien "
                              "numeradas con la limpieza actual", "arches": len(pares)}
    for arcada, d in por.items():
        premolares = d.get("4", []) + d.get("5", [])
        molares = d.get("6", [])
        t, errores = umbral_de_separacion(premolares, molares)
        salida[arcada] = {
            "by_position": {q: {"n": len(v), "p05_mm2": float(np.percentile(v, 5)),
                                "p50_mm2": float(np.median(v)),
                                "p95_mm2": float(np.percentile(v, 95))}
                            for q, v in sorted(d.items())},
            "premolar_vs_first_molar_mm2": t, "misclassified": errores,
            "of": len(premolares) + len(molares)}
    salida["run_record"] = bs.registro(sys.argv)
    (SALIDA / "areas-c2-entrenamiento.json").write_text(json.dumps(salida, indent=2) + "\n")
    print(json.dumps({k: v for k, v in salida.items() if k != "run_record"}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    c = ordenes.add_parser("c2")
    c.add_argument("--split", choices=tuple(SPLITS), required=True)
    c.add_argument("--limite", type=int)
    c.add_argument("--parte", default="0/1")
    p = ordenes.add_parser("puntua")
    p.add_argument("--split", choices=tuple(SPLITS), required=True)
    p.add_argument("--modelo", type=Path, required=True)
    p.add_argument("--limite", type=int)
    p.add_argument("--parte", default="0/1")
    p.add_argument("--version", type=int, choices=(1, 2), default=1)
    cl = ordenes.add_parser("clinico")
    cl.add_argument("--segmentacion", type=Path, required=True)
    cl.add_argument("--fdi-json", type=Path, required=True)
    cl.add_argument("--arcada", choices=("upper", "lower"), required=True)
    cl.add_argument("--modelo", type=Path, required=True)
    cl.add_argument("--salida", type=Path, required=True)
    cl.add_argument("--version", type=int, choices=(1, 2, 3), default=1)
    for nombre in ("elige", "evalua"):
        s = ordenes.add_parser(nombre)
        s.add_argument("--modelo", type=Path, required=True)
        s.add_argument("--trabajos", type=int, default=6)
        s.add_argument("--version", type=int, choices=(1, 2, 3), default=1)
    ar = ordenes.add_parser("areas")
    ar.add_argument("--trabajos", type=int, default=6)
    ar.add_argument("--limite", type=int)
    ar.add_argument("--c2", action="store_true",
                    help="instancias de C2 y umbral premolar/primer molar (§6.14)")
    args = ap.parse_args()
    return {"c2": c2, "puntua": puntua, "elige": elige, "evalua": evalua,
            "clinico": clinico, "areas": areas}[args.orden](args)


if __name__ == "__main__":
    raise SystemExit(main())
