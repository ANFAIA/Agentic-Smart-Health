#!/usr/bin/env python
"""El modelo de pieza de §6.11: mira cada pieza recortada y dice si es diente y cuál es.

    G=~/.venvs/dental-gpu/bin/python
    $G scripts/entrena_pieza.py mina                       # C2 sobre entrenamiento (GPU)
    uv run python scripts/entrena_pieza.py prepara         # regiones y negativos (CPU)
    $G scripts/entrena_pieza.py entrena --salida data/processed/identidad/pieza.pt

**Por qué existe.** C2 decide el número de cada diente sobre todo por su posición en la
arcada: cuando falta un premolar corre la hemiarcada, y detrás del 7 pone un «8» sobre la
encía. Este modelo ve una sola pieza a resolución de malla, sin saber dónde está en la
arcada, así que su evidencia es independiente de la de C2: si hay corona o solo encía, y si
es un incisivo, un premolar o un molar. Ver §6.11 de `docs/research/segmentacion-fdi-escaner.md`.

**Los datos.** Los mismos 540 pacientes de entrenamiento que C2
(`entrena_fdi_escaner.arcadas_de_entrenamiento`, que veta `dev`, `eval` y las dos reservas):

- positivos: cada pieza del experto, con la frontera movida hasta 1 mm hacia dentro o hacia
  fuera y, la mitad de las veces, hasta un 30 % de ella quitado por un lado, para que se
  parezca a lo que entrega C2;
- negativos: parches de encía del tamaño de un molar, detrás de la última pieza de cada
  cuadrante y en otras zonas cerca de los dientes, más los fantasmas que da C2 sobre estas
  mismas arcadas (`mina`). C2 se entrenó con ellas, así que da pocos.

El 10 % de los pacientes, con semilla fija, valida y elige la época; nunca `dev-privado`.

**La entrada** es el recorte de `analysis_agents.identidad.recorte`: la pieza y lo que hay a
menos de 2 mm, hasta 4.096 puntos, centrado en la pieza, en milímetros y en la pose de
Teeth3DS+. Por punto: la normal y si pertenece a la pieza. El tamaño no se normaliza: es
parte de lo que separa un premolar de un molar.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

RESUMEN_EN = "Trains the per-tooth crop model: is it a tooth, and which position in its quadrant."
RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))

from analysis_agents.identidad import (  # noqa: E402
    MARGEN_RECORTE_MM,
    POSICIONES,
    distancia_con_signo,
    instancias,
    recorte,
)


def _carga(nombre: str, ruta: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(nombre, modulo)
    spec.loader.exec_module(modulo)
    return sys.modules[nombre]


ef = _carga("entrena_fdi_escaner", RAIZ / "scripts/entrena_fdi_escaner.py")
bm = ef.bm
SALIDA = RAIZ / "data/processed/identidad"
MINADO = SALIDA / "mina"
REGIONES = SALIDA / "regiones"
C2 = RAIZ / "data/processed/a3-checkpoints/fdi-8k-rot15-s0.pt"
C2_SHA = "24a4cf569aa902138bdc765a9836197b771c8fca297854cd546e48e92f95946a"

DESPLAZA_MM = 1.0          # la frontera de una pieza se mueve hasta ±1 mm
QUITA_MAXIMO = 0.30        # y hasta un 30 % se quita por un lado, la mitad de las veces
RADIO_REGION = DESPLAZA_MM + MARGEN_RECORTE_MM
PARCHE_MM = (3.5, 5.5)     # radio de un parche de encía: el área de un molar, 40-95 mm²
DISTAL_MM = 8.0            # a qué distancia detrás de la última pieza se busca el parche
PARCHES_CERCA = 2          # parches de encía por arcada cerca de los dientes
VALIDACION = 0.10
SEMILLA = 0


def firma_c2() -> None:
    if hashlib.sha256(C2.read_bytes()).hexdigest() != C2_SHA:
        raise SystemExit(f"⚠ {C2} no es el C2 de §6.10; no sigo.")


# ── mina: C2 sobre las arcadas de entrenamiento (GPU) ──────────────────────────

def mina(args: argparse.Namespace) -> int:
    """Etiquetas de C2 en la pose nativa de cada arcada de entrenamiento, sin limpiar."""
    import torch

    firma_c2()
    seg = _carga("segmentar_fdi", RAIZ / "scripts/segmentar_fdi.py")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, codigos, n_clases, meta = seg.carga_modelo(C2, dev)
    arcadas = ef.arcadas_de_entrenamiento(True)
    if args.limite:
        arcadas = arcadas[: args.limite]
    MINADO.mkdir(parents=True, exist_ok=True)
    for k, (nombre, ruta) in enumerate(arcadas, 1):
        destino = MINADO / f"{nombre}.npz"
        if destino.is_file():
            continue
        z = np.load(ruta)
        pos = z["pos"].astype(np.float64)
        arcada = nombre.rsplit("_", 1)[1]
        torch.manual_seed(0)
        _, inst = seg.infiere(modelo, codigos, n_clases, pos, pos, z["nrm"], arcada, dev,
                              trozo=meta["num_points"])
        np.savez_compressed(destino, et=seg.etiquetas_de(inst, len(pos)))
        if k % 50 == 0:
            print(f"  [{k}/{len(arcadas)}] {nombre}", flush=True)
    return 0


# ── prepara: regiones con distancia con signo (CPU) ─────────────────────────────

def _region(pos, arbol, dentro, clase: int, posicion: int, tipo: int):  # type: ignore[no-untyped-def]
    idx, d = distancia_con_signo(pos, dentro, RADIO_REGION, arbol)
    return idx.astype(np.int32), d.astype(np.float16), (clase, posicion, tipo)


def parches_de_encia(pos: np.ndarray, experto: np.ndarray,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """Parches de encía del tamaño de un molar: uno detrás de la última pieza de cada
    cuadrante y `PARCHES_CERCA` cerca de los dientes. Solo vértices de encía del experto."""
    from scipy.spatial import cKDTree

    encia = np.flatnonzero(experto == 0)
    if len(encia) < 100:
        return []
    arbol_encia = cKDTree(pos[encia])
    inst = instancias(experto)
    cent = {c: pos[v].mean(0) for c, v in inst.items()}
    semillas = []
    for q in sorted({c // 10 for c in inst}):
        cs = sorted(c for c in inst if c // 10 == q)
        if len(cs) >= 2:
            ultimo, previo = cent[cs[-1]], cent[cs[-2]]
            sentido = (ultimo - previo) / max(np.linalg.norm(ultimo - previo), 1e-6)
            semillas.append(ultimo + DISTAL_MM * sentido)
    dientes = np.flatnonzero(experto > 0)
    if len(dientes):
        d, _ = cKDTree(pos[dientes]).query(pos[encia], distance_upper_bound=8.0)
        cerca = encia[(d > 3.0) & np.isfinite(d)]
        if len(cerca):
            semillas += list(pos[rng.choice(cerca, size=min(PARCHES_CERCA, len(cerca)),
                                            replace=False)])
    parches = []
    for s in semillas:
        _, j = arbol_encia.query(s)
        radio = rng.uniform(*PARCHE_MM)
        vecinos = arbol_encia.query_ball_point(pos[encia[j]], radio)
        if len(vecinos) >= 50:
            parches.append(encia[np.asarray(vecinos)])
    return parches


def prepara_arcada(trabajo: tuple[str, str]) -> str:
    from analysis_agents.margen_metricas import identidad_de_piezas
    from scipy.spatial import cKDTree

    nombre, ruta = trabajo
    destino = REGIONES / f"{nombre}.npz"
    if destino.is_file():
        return nombre
    z = np.load(ruta)
    pos = z["pos"].astype(np.float64)
    experto = z["labels"].astype(np.int64)
    arbol = cKDTree(pos)
    rng = np.random.default_rng(int(hashlib.sha256(nombre.encode()).hexdigest()[:8], 16))
    regiones = [_region(pos, arbol, v, 1, c % 10, 0) for c, v in instancias(experto).items()
                if 1 <= c % 10 <= POSICIONES]
    regiones += [_region(pos, arbol, v, 0, 0, 1) for v in parches_de_encia(pos, experto, rng)]
    fantasmas = 0
    minado = MINADO / f"{nombre}.npz"
    if minado.is_file():
        c2 = bm.limpieza_actual(pos, np.load(minado)["et"].astype(np.int64))
        ide = identidad_de_piezas(c2, experto)
        for cod, e in ide["per_label"].items():
            if e["status"] == "ghost":
                regiones.append(_region(pos, arbol, np.flatnonzero(c2 == int(cod)), 0, 0, 2))
                fantasmas += 1
    np.savez_compressed(
        destino, idx=np.concatenate([r[0] for r in regiones]),
        d=np.concatenate([r[1] for r in regiones]),
        cortes=np.cumsum([0] + [len(r[0]) for r in regiones]),
        etiqueta=np.array([r[2] for r in regiones], dtype=np.int16),
        minado=np.bool_(minado.is_file()))
    return f"{nombre}: {len(regiones)} regiones, {fantasmas} fantasmas de C2"


def prepara(args: argparse.Namespace) -> int:
    from multiprocessing import Pool

    arcadas = ef.arcadas_de_entrenamiento(True)
    if args.limite:
        arcadas = arcadas[: args.limite]
    faltan = [a for a in arcadas if not (MINADO / f"{a[0]}.npz").is_file()]
    if faltan and not args.sin_minar:
        raise SystemExit(f"⚠ faltan {len(faltan)} arcadas por minar; corre `mina` antes.")
    REGIONES.mkdir(parents=True, exist_ok=True)
    with Pool(args.procesos) as pool:
        for k, msg in enumerate(pool.imap_unordered(prepara_arcada, arcadas), 1):
            if k % 50 == 0 or k == len(arcadas):
                print(f"  [{k}/{len(arcadas)}] {msg}", flush=True)
    return 0


# ── entrena (GPU) ───────────────────────────────────────────────────────────────

def particion_validacion(nombres: list[str]) -> set[str]:
    """Pacientes de validación: el 10 % con semilla fija, por paciente y no por arcada."""
    pacientes = sorted({n.rsplit("_", 1)[0] for n in nombres})
    k = round(VALIDACION * len(pacientes))
    return set(np.random.default_rng(SEMILLA).choice(pacientes, size=k, replace=False))


def construye_modelo():  # type: ignore[no-untyped-def]
    """El clasificador: los bloques del Point Transformer de C2, bajando, y dos cabezas."""
    import torch
    from torch_geometric.nn import MLP, global_max_pool, global_mean_pool, knn_graph

    seg = _carga("segmentar_fdi", RAIZ / "scripts/segmentar_fdi.py")
    dims = (32, 64, 128, 256)

    class ModeloPieza(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entrada = MLP([4, dims[0]], plain_last=False)
            self.bloque0 = seg.TransformerBlock(dims[0], dims[0])
            self.bajadas = torch.nn.ModuleList(
                [seg.TransitionDown(dims[i], dims[i + 1], k=16) for i in range(len(dims) - 1)])
            self.bloques = torch.nn.ModuleList(
                [seg.TransformerBlock(dims[i + 1], dims[i + 1]) for i in range(len(dims) - 1)])
            self.cuerpo = MLP([2 * dims[-1], 256, 128], plain_last=False)
            self.diente = torch.nn.Linear(128, 1)
            self.posicion = torch.nn.Linear(128, POSICIONES)

        def forward(self, x, pos, batch):  # type: ignore[no-untyped-def]
            x = self.entrada(x)
            x = self.bloque0(x, pos, knn_graph(pos, k=16, batch=batch))
            for bajada, bloque in zip(self.bajadas, self.bloques, strict=True):
                x, pos, batch = bajada(x, pos, batch)
                x = bloque(x, pos, knn_graph(pos, k=16, batch=batch))
            h = self.cuerpo(torch.cat([global_max_pool(x, batch),
                                       global_mean_pool(x, batch)], dim=1))
            return self.diente(h).squeeze(1), torch.log_softmax(self.posicion(h), dim=1)

    return ModeloPieza()


ESCALA_POS = 10.0  # mm → unidades de la red; no normaliza el tamaño, solo lo escala


def a_tensores(recortes: list[tuple[np.ndarray, np.ndarray]], dev: str):  # type: ignore[no-untyped-def]
    import torch

    P = torch.from_numpy(np.concatenate([p for p, _ in recortes]) / ESCALA_POS).to(dev)
    X = torch.from_numpy(np.concatenate([x for _, x in recortes])).to(dev)
    b = torch.cat([torch.full((len(p),), i, dtype=torch.long)
                   for i, (p, _) in enumerate(recortes)]).to(dev)
    return X, P, b


def aumenta(pos, nrm, idx, d, rng, rotacion: float):  # type: ignore[no-untyped-def]
    """Un recorte de entrenamiento: frontera movida, a veces un lado quitado, girado."""
    r = recorte(pos, nrm, idx, d.astype(np.float64), rng,
                desplaza=rng.uniform(-DESPLAZA_MM, DESPLAZA_MM))
    if r is None:
        r = recorte(pos, nrm, idx, d.astype(np.float64), rng)
    assert r is not None
    P, X = r
    if rng.random() < 0.5:
        dentro = np.flatnonzero(X[:, 3] > 0)
        eje = rng.normal(size=3)
        proy = P[dentro] @ (eje / np.linalg.norm(eje))
        n = int(rng.uniform(0, QUITA_MAXIMO) * len(dentro))
        if 0 < n < len(dentro):
            X[dentro[np.argsort(proy)[-n:]], 3] = 0.0
    R = ef.rotacion_aleatoria(rng, rotacion)
    escala = rng.uniform(0.95, 1.05)
    X[:, :3] = X[:, :3] @ R.T
    return (P @ R.T * escala).astype(np.float32), X


def entrena(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    cc = _carga("caso_completo", RAIZ / "scripts/caso_completo.py")
    registro = {"schema": "histora-run-record/1.0", "argv": sys.argv,
                "code": cc.revision_de_codigo(), "environment": cc.entorno_de_software(),
                "seed": args.semilla}
    torch.manual_seed(args.semilla)
    rng = np.random.default_rng(args.semilla)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    arcadas = ef.arcadas_de_entrenamiento(True)
    if args.limite:
        arcadas = arcadas[: args.limite]
    datos, muestras = [], []
    for nombre, ruta in arcadas:
        z, r = np.load(ruta), np.load(REGIONES / f"{nombre}.npz")
        if not bool(r["minado"]) and not args.limite:
            raise SystemExit(f"⚠ {nombre} se preparó sin minar; rehaz `prepara`.")
        datos.append((nombre, z["pos"].astype(np.float32), z["nrm"].astype(np.float32),
                      r["idx"], r["d"], r["cortes"]))
        for j, e in enumerate(r["etiqueta"]):
            muestras.append((len(datos) - 1, j, int(e[0]), int(e[1]), int(e[2])))
    val_pac = particion_validacion([n for n, *_ in datos])
    es_val = np.array([datos[a][0].rsplit("_", 1)[0] in val_pac for a, *_ in muestras])
    entreno = [m for m, v in zip(muestras, es_val, strict=True) if not v]
    valida = [m for m, v in zip(muestras, es_val, strict=True) if v]
    tipos = np.bincount([m[4] for m in muestras], minlength=3)
    print(f"{len(datos)} arcadas · {len(entreno)} recortes de entreno, {len(valida)} de "
          f"validación · positivos {tipos[0]}, parches {tipos[1]}, fantasmas de C2 "
          f"{tipos[2]} · {time.perf_counter() - t0:.0f} s", flush=True)

    def recorte_de(m, rng_):  # type: ignore[no-untyped-def]
        a, j, *_ = m
        _, pos, nrm, idx, d, cortes = datos[a]
        s = slice(cortes[j], cortes[j + 1])
        return aumenta(pos, nrm, idx[s], d[s], rng_, args.rotacion)

    rng_val = np.random.default_rng(12345)
    lotes_val = [[recorte_de(m, rng_val) for m in valida[s: s + args.lote]]
                 for s in range(0, len(valida), args.lote)]
    y_val = [(torch.tensor([m[2] for m in valida[s: s + args.lote]], dtype=torch.float32),
              torch.tensor([max(m[3] - 1, 0) for m in valida[s: s + args.lote]]))
             for s in range(0, len(valida), args.lote)]

    modelo = construye_modelo().to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    pasos = args.epocas * math.ceil(args.por_epoca / args.lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=pasos,
                                                pct_start=0.05)

    def perdida(sal_d, sal_p, y_d, y_p):  # type: ignore[no-untyped-def]
        l_d = F.binary_cross_entropy_with_logits(sal_d, y_d)
        pos_ = y_d > 0.5
        l_p = F.nll_loss(sal_p[pos_], y_p[pos_]) if pos_.any() else sal_p.sum() * 0
        return l_d + l_p, l_d, l_p

    historia, mejor, mejor_estado = [], math.inf, None
    for epoca in range(args.epocas):
        modelo.train()
        te = time.perf_counter()
        elegidas = rng.choice(len(entreno), size=args.por_epoca, replace=False)
        suma = 0.0
        for s in range(0, len(elegidas), args.lote):
            ms = [entreno[i] for i in elegidas[s: s + args.lote]]
            X, P, b = a_tensores([recorte_de(m, rng) for m in ms], dev)
            y_d = torch.tensor([m[2] for m in ms], dtype=torch.float32, device=dev)
            y_p = torch.tensor([max(m[3] - 1, 0) for m in ms], device=dev)
            opt.zero_grad()
            loss, *_ = perdida(*modelo(X, P, b), y_d, y_p)
            loss.backward()
            opt.step()
            sched.step()
            suma += loss.item() * len(ms)
        modelo.eval()
        vd = vp = acc_d = acc_p = n_p = 0.0
        with torch.no_grad():
            for lote, (y_d, y_p) in zip(lotes_val, y_val, strict=True):
                X, P, b = a_tensores(lote, dev)
                sal_d, sal_p = modelo(X, P, b)
                y_d, y_p = y_d.to(dev), y_p.to(dev)
                _, l_d, l_p = perdida(sal_d, sal_p, y_d, y_p)
                pos_ = y_d > 0.5
                vd += l_d.item() * len(lote)
                vp += l_p.item() * int(pos_.sum())
                acc_d += float(((sal_d > 0) == pos_).sum())
                acc_p += float((sal_p.argmax(1)[pos_] == y_p[pos_]).sum())
                n_p += float(pos_.sum())
        val = vd / len(valida) + vp / max(n_p, 1)
        fila = {"epoch": epoca + 1, "train_loss": suma / len(elegidas), "val_loss": val,
                "val_tooth_acc": acc_d / len(valida), "val_position_acc": acc_p / max(n_p, 1),
                "seconds": round(time.perf_counter() - te, 1)}
        historia.append(fila)
        print(json.dumps(fila), flush=True)
        if val < mejor:
            mejor = val
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    nombres = sorted(n for n, *_ in datos)
    torch.save({
        "state_dict": mejor_estado, "name": "pieza · verificador + posición",
        "epoch": min(historia, key=lambda f: f["val_loss"])["epoch"],
        "seed": args.semilla, "history": historia, "run_record": registro,
        "rotation_deg": args.rotacion, "points": 4096, "margin_mm": MARGEN_RECORTE_MM,
        "position_scale": ESCALA_POS, "train_arches": len(nombres),
        "train_ids_sha256": hashlib.sha256("\n".join(nombres).encode()).hexdigest(),
        "validation_patients": sorted(val_pac),
    }, args.salida)
    print(f"→ {args.salida} ({(time.perf_counter() - t0) / 60:.0f} min)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    m = ordenes.add_parser("mina")
    m.add_argument("--limite", type=int)
    p = ordenes.add_parser("prepara")
    p.add_argument("--limite", type=int)
    p.add_argument("--procesos", type=int, default=min(6, os.cpu_count() or 1))
    p.add_argument("--sin-minar", action="store_true", help="Solo para pruebas de humo.")
    e = ordenes.add_parser("entrena")
    e.add_argument("--salida", type=Path, required=True)
    e.add_argument("--epocas", type=int, default=20)
    e.add_argument("--por-epoca", type=int, default=8000)
    e.add_argument("--lote", type=int, default=16)
    e.add_argument("--lr", type=float, default=1e-3)
    e.add_argument("--rotacion", type=float, default=15.0)
    e.add_argument("--semilla", type=int, default=SEMILLA)
    e.add_argument("--limite", type=int)
    args = ap.parse_args()
    return {"mina": mina, "prepara": prepara, "entrena": entrena}[args.orden](args)


if __name__ == "__main__":
    raise SystemExit(main())
