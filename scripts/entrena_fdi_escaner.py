#!/usr/bin/env python
"""Entrenar el modelo FDI del escáner con más resolución, rotaciones y más pacientes.

    ~/.venvs/dental-gpu/bin/python scripts/entrena_fdi_escaner.py \\
        --puntos 8192 --epocas 60 --rotacion 15 --salida data/processed/a3-checkpoints/x.pt

**Por qué existe.** El checkpoint de agosto (`normales-ce-baseline.pt`) salió del notebook
`exercise-point-transformer-teeth3ds.ipynb`: trozos de 2.048 puntos, 20 épocas, 240
pacientes y ninguna aumentación. Sobre `dev`, el 32 % de las piezas cede o toma más del
10 % de su área al vecino y la frontera entre dientes cae a 0,72 mm de la del experto (§6.10
de `docs/research/segmentacion-fdi-escaner.md`). Con 2.048 puntos sobre una arcada de
~9.500 mm² hay unos 2 mm entre puntos y la malla tiene 0,17 mm entre vértices: el modelo no
puede poner la frontera más fina que lo que ve.

**Qué cambia y qué no.** La arquitectura, la entrada (normales), la pérdida (NLL con pesos
por frecuencia mediana) y la normalización son las del notebook, y `Net` se importa de
`segmentar_fdi.py` para que haya una sola definición. Cambian:

- `--puntos`: el tamaño de trozo. La inferencia lo lee del checkpoint (`num_points`).
- `--rotacion`: un giro aleatorio de hasta ± N grados sobre un eje aleatorio, a posición y
  normales a la vez, porque la pose que da la canonización nunca es exactamente la de
  Teeth3DS+. **Nunca un espejo**: cambiaría el lado y con él el número FDI.
- Más pacientes: los 240 de entrenamiento de siempre más los 300 del `public-training-set-2`
  (`scripts/fetch_teeth3ds_osf.py entrenamiento`). `dev`, `eval` y la reserva quedan fuera,
  y se comprueba.

La partición es la de `benchmark_margen.particion`, sin tocar.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RESUMEN_EN = "Trains the scanner FDI point model with larger chunks, rotations and more data."
RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))


def _carga(nombre: str, ruta: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    assert spec and spec.loader
    modulo = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(nombre, modulo)
    spec.loader.exec_module(modulo)
    return sys.modules[nombre]


bm = _carga("benchmark_margen", RAIZ / "scripts/benchmark_margen.py")
osf = _carga("fetch_teeth3ds_osf", RAIZ / "scripts/fetch_teeth3ds_osf.py")
CACHE_LOCAL = RAIZ / "data/processed/teeth3ds-local-npz"
PESO_MAXIMO = 50.0  # el tope del notebook para el peso de una clase rara


# ── Datos ───────────────────────────────────────────────────────────────────────

def arcadas_de_entrenamiento(con_osf: bool) -> list[tuple[str, Path]]:
    """`(nombre, npz)` de cada arcada de entrenamiento. Convierte a npz lo que falte."""
    particion = bm.particion(bm.pacientes())
    vetados = set(particion["dev"]) | set(particion["eval"])
    vetados |= set(osf.ids_reserva())
    salida = []
    for p in particion["train"]:
        for a in bm.ARCADAS:
            destino = CACHE_LOCAL / f"{p}_{a}.npz"
            if not destino.is_file():
                osf.a_npz(bm.MALLAS / p / f"{p}_{a}.obj",
                          bm.ETIQUETAS / p / f"{p}_{a}.json", destino)
            salida.append((f"{p}_{a}", destino))
    if con_osf:
        extra = sorted(osf.NPZ.glob("*.npz"))
        if not extra:
            raise SystemExit("No hay pacientes de OSF; ejecuta "
                             "`fetch_teeth3ds_osf.py entrenamiento` o usa --sin-osf.")
        salida += [(f.stem, f) for f in extra]
    fuga = {n.rsplit("_", 1)[0] for n, _ in salida} & vetados
    if fuga:
        raise SystemExit(f"⚠ {len(fuga)} pacientes de evaluación en el entrenamiento: {fuga}")
    return salida


def normaliza(pos: np.ndarray) -> np.ndarray:
    """`NormalizeScale` de PyG: centra en la media y escala a la bola unidad (max |x|)."""
    c = pos - pos.mean(0)
    return (c / np.abs(c).max()).astype(np.float32)


def rotacion_aleatoria(rng: np.random.Generator, grados: float) -> np.ndarray:
    """Giro de ángulo uniforme en ±`grados` sobre un eje uniforme en la esfera. det = +1."""
    if grados <= 0:
        return np.eye(3, dtype=np.float32)
    eje = rng.normal(size=3)
    eje /= np.linalg.norm(eje)
    ang = math.radians(rng.uniform(-grados, grados))
    k = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]])
    return (np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k).astype(np.float32)


# ── Entrenamiento ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puntos", type=int, default=8192)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--rotacion", type=float, default=0.0, help="Grados, ± sobre eje aleatorio.")
    ap.add_argument("--escala", type=float, default=0.05, help="Jitter de escala, ± fracción.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--sin-osf", action="store_true", help="Solo los 240 pacientes de siempre.")
    ap.add_argument("--limite", type=int, help="Solo N arcadas (prueba de humo).")
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    seg = _carga("segmentar_fdi", RAIZ / "scripts/segmentar_fdi.py")
    cc = _carga("caso_completo", RAIZ / "scripts/caso_completo.py")
    registro = {"schema": "histora-run-record/1.0", "argv": sys.argv,
                "code": cc.revision_de_codigo(), "environment": cc.entorno_de_software(),
                "seed": args.semilla}
    torch.manual_seed(args.semilla)
    rng = np.random.default_rng(args.semilla)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    arcadas = arcadas_de_entrenamiento(not args.sin_osf)
    if args.limite:
        arcadas = arcadas[: args.limite]
    datos = []
    for nombre, ruta in arcadas:
        z = np.load(ruta)
        datos.append((nombre, normaliza(z["pos"]), z["nrm"].astype(np.float32),
                      z["labels"].astype(np.int64)))
    base = torch.load(bm.CHECKPOINT, map_location="cpu", weights_only=False)
    codigos = sorted(set(base["codes"]) | {int(c) for *_, y in datos for c in np.unique(y)})
    a_indice = np.full(max(codigos) + 1, -1, dtype=np.int64)
    a_indice[codigos] = np.arange(len(codigos))
    datos = [(n, p, x, a_indice[y]) for n, p, x, y in datos]
    K = len(codigos)
    print(f"{len(datos)} arcadas en {time.perf_counter() - t0:.0f} s · {K} clases", flush=True)

    cuentas = np.zeros(K)
    for *_, y in datos:
        cuentas += np.bincount(y, minlength=K)
    frec = cuentas / cuentas.sum()
    presentes = cuentas > 0
    pesos = np.zeros(K)
    pesos[presentes] = np.minimum(np.median(frec[presentes]) / frec[presentes], PESO_MAXIMO)
    W = torch.tensor(pesos, dtype=torch.float32, device=dev)

    modelo = seg.Net(3, K, dim_model=list(seg.DIM_MODELO), k=16).to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    pasos = args.epocas * math.ceil(len(datos) / args.lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=pasos,
                                                pct_start=0.05)
    historia = []
    for epoca in range(args.epocas):
        modelo.train()
        orden = rng.permutation(len(datos))
        perdida = aciertos = total = 0.0
        te = time.perf_counter()
        for s in range(0, len(orden), args.lote):
            pos_l, x_l, y_l, b_l = [], [], [], []
            for j, i in enumerate(orden[s: s + args.lote]):
                _, pos, nrm, y = datos[i]
                sel = rng.choice(len(pos), size=args.puntos, replace=len(pos) < args.puntos)
                R = rotacion_aleatoria(rng, args.rotacion)
                escala = rng.uniform(1 - args.escala, 1 + args.escala)
                pos_l.append(torch.from_numpy(pos[sel] @ R.T * escala))
                x_l.append(torch.from_numpy(nrm[sel] @ R.T))
                y_l.append(torch.from_numpy(y[sel]))
                b_l.append(torch.full((args.puntos,), j, dtype=torch.long))
            pos_t, x_t = torch.cat(pos_l).to(dev), torch.cat(x_l).to(dev)
            y_t, b_t = torch.cat(y_l).to(dev), torch.cat(b_l).to(dev)
            opt.zero_grad()
            salida = modelo(x_t, pos_t, b_t)
            loss = F.nll_loss(salida, y_t, weight=W)
            loss.backward()
            opt.step()
            sched.step()
            perdida += float(loss) * len(b_l)
            diente = y_t != 0
            aciertos += float((salida.argmax(1)[diente] == y_t[diente]).sum())
            total += float(diente.sum())
        fila = {"epoch": epoca + 1, "loss": perdida / len(datos),
                "train_tooth_acc": aciertos / max(total, 1),
                "seconds": round(time.perf_counter() - te, 1)}
        historia.append(fila)
        print(json.dumps(fila), flush=True)

    nombres = sorted(n for n, *_ in datos)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": modelo.state_dict(), "cols": [0, 1, 2], "in_ch": 3, "K": K,
        "codes": codigos, "dim_model": list(seg.DIM_MODELO), "num_points": args.puntos,
        "name": f"normales · CE · {args.puntos} pts · rot ±{args.rotacion:g}° · "
                f"{len(datos)} arcadas",
        "seed": args.semilla,
        "augmentation": {"rotation_deg": args.rotacion, "scale": args.escala, "mirror": False},
        "epochs": args.epocas, "lr": args.lr, "batch": args.lote,
        "train_arches": len(nombres),
        "train_ids_sha256": hashlib.sha256("\n".join(nombres).encode()).hexdigest(),
        "history": historia, "run_record": registro,
    }, args.salida)
    print(f"→ {args.salida} ({(time.perf_counter() - t0) / 60:.0f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
