#!/usr/bin/env python
"""entrena_gs_escaner.py — 3DGS de verdad sobre la superficie del escaner.

    ~/.venvs/dental-gpu/bin/python scripts/entrena_gs_escaner.py --stl <x.stl> --fdi <x.npy>

**Por que existe.** El canal del visor escribe los vertices del escaner como splats con
forma puesta a mano —esferas primero, discos despues— y no se ve como una superficie.
Comparado en el mismo visor contra un 3DGS entrenado de verdad, la diferencia no esta en
el tamano ni en la opacidad ni en la proporcion del splat: esta en que aquel optimizo
POSICION, FORMA, COLOR y OPACIDAD contra 1.600 renders durante 6.000 pasos hasta 31,5 dB.
Ninguna constante cierra esa brecha desde una nube de vertices fija.

**Que hace.** La receta del notebook 07, como script: renders EEVEE del STL con pose
conocida → gsplat con control adaptativo de densidad → PLY. El color de partida es el
FALSO COLOR por pieza, no el del paciente, asi que cada diente sale de su tono y el
resultado sigue diciendo que el color no es una medida.

⚠️ **Lo que sale es APARIENCIA, no medida.** Las gaussianas finales no son los vertices
del escaner: el optimizador las mueve, las divide y las poda. Es un artefacto de
presentacion —`views` en el vocabulario de UOS, no `assets`— y no se mide encima.

**Pero la etiqueta FDI si es exacta**, y no por transferirla del vertice mas cercano. El
codigo viaja como un PARAMETRO MAS con tasa de aprendizaje CERO: `DefaultStrategy` lo
arrastra por la densificacion igual que a las escalas, asi que al dividir una gaussiana los
hijos heredan el codigo del padre y al podar desaparece con ella. Comprobado: tras dividir
y podar, todas las etiquetas siguen siendo enteros originales. Cada gaussiana desciende de
UN vertice de UNA pieza, sin votar ni inferir nada — que es lo que hace fiable pinchar un
diente en el visor.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RESUMEN_EN = "Trains real 3DGS over the scanner surface."

RAIZ = Path(__file__).resolve().parent.parent
for src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(src))


def color_por_pieza(etiquetas: np.ndarray) -> np.ndarray:
    """RGB en [0,1] por vertice: el mismo falso color por FDI que usa el visor."""
    import colorsys

    from export_agents.visor import _COLOR_ENCIA, _LUZ_PIEZA, _SATURACION_PIEZA, tono_de

    rgb = np.tile(_COLOR_ENCIA, (len(etiquetas), 1)).astype(np.float64)
    for codigo in sorted(set(int(f) for f in np.unique(etiquetas)) - {0}):
        rgb[etiquetas == codigo] = colorsys.hls_to_rgb(
            tono_de(codigo), _LUZ_PIEZA, _SATURACION_PIEZA
        )
    return rgb


def escribe_ply_color(destino: Path, pos: np.ndarray, caras: np.ndarray, rgb: np.ndarray) -> None:
    """PLY ASCII con color por vertice, que es lo que el renderizador de Blender espera."""
    c = (rgb * 255).clip(0, 255).astype(np.uint8)
    with destino.open("w", encoding="ascii") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pos)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"element face {len(caras)}\nproperty list uchar int vertex_indices\n"
                 "end_header\n")
        for p, q in zip(pos, c, strict=True):
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]} {q[1]} {q[2]}\n")
        for f in caras:
            fh.write(f"3 {f[0]} {f[1]} {f[2]}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stl", type=Path, required=True)
    ap.add_argument("--fdi", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--vistas", type=int, default=1600)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--iteraciones", type=int, default=6000)
    ap.add_argument("--semillas", type=int, default=150_000)
    ap.add_argument("--salta-render", action="store_true",
                    help="Reutiliza los renders que ya haya en la salida.")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    from analysis_agents import rellena_etiquetas
    from ingestion_agents.mesh_agent import parse_stl

    malla = parse_stl(args.stl)
    pos, caras = malla["positions"].astype(np.float64), malla["faces"]
    etq = rellena_etiquetas(pos, np.load(args.fdi).astype(np.int64))
    print(f"malla {len(pos):,} vertices · {len(caras):,} caras · "
          f"{len(set(etq[etq > 0].tolist()))} piezas")

    coloreado = args.salida / "escaner_color.ply"
    if not args.salta_render:
        escribe_ply_color(coloreado, pos, caras, color_por_pieza(etq))
        t0 = time.perf_counter()
        r = subprocess.run(
            ["blender", "--background", "--python",
             str(RAIZ / "scripts" / "blender_render_views.py"), "--",
             "--scan", str(coloreado), "--out", str(args.salida),
             "--views", str(args.vistas), "--res", str(args.res),
             "--samples", "16", "--elevations", "8"],
            capture_output=True, text=True,
        )
        if not (args.salida / "transforms.json").exists():
            print(r.stdout[-2000:] or r.stderr[-2000:])
            raise SystemExit("Blender no produjo transforms.json")
        print(f"Blender: {time.perf_counter() - t0:.0f}s")

    T = json.loads((args.salida / "transforms.json").read_text())
    print(f"vistas {len(T['frames'])} · fov_x {T['camera_angle_x']:.3f} · "
          f"escala {T['scan_scale']:.4f}")

    entrena(args, T, pos, etq)
    return 0


def entrena(args, T: dict, pos: np.ndarray, etq: np.ndarray) -> None:
    """El bucle de gsplat. Import aqui dentro: torch y gsplat solo hacen falta a partir de aqui."""
    import torch
    from gsplat import DefaultStrategy, rasterization
    from PIL import Image
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    dev = "cuda"
    # Normalizacion IDENTICA a la de Blender: si difiere, las camaras no miran al modelo.
    P = pos.astype(np.float32)
    centro = (P.min(0) + P.max(0)) / 2
    P = (P - centro) * (2.0 / (P.max(0) - P.min(0)).max())
    rng = np.random.default_rng(0)
    sel = rng.choice(len(P), min(args.semillas, len(P)), replace=False)
    means0 = torch.tensor(P[sel], device=dev)
    n0 = len(sel)
    d = torch.cdist(means0[:2000], means0[:2000]).topk(4, largest=False).values[:, 1:].mean()
    col0 = torch.tensor(color_por_pieza(etq)[sel], dtype=torch.float32, device=dev).clamp(.01, .99)
    fdi0 = torch.tensor(etq[sel], dtype=torch.float32, device=dev).unsqueeze(1)

    # ParameterDict con un Adam POR clave: es lo que `DefaultStrategy` necesita para anadir
    # y quitar filas —densificar y podar— sin descuadrar el estado del optimizador.
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means0),
        "scales": torch.nn.Parameter(torch.log(torch.full((n0, 3), float(d), device=dev))),
        "quats": torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=dev).repeat(n0, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n0,), 0.1, device=dev))),
        "colors": torch.nn.Parameter(torch.logit(col0)),
        # El codigo FDI, como parametro con tasa CERO. Ver el docstring del modulo: es lo
        # que hace que la etiqueta sea exacta por construccion en vez de heredada.
        "fdi": torch.nn.Parameter(fdi0),
    }).to(dev)

    W = H = T["w"]
    fx = 0.5 * W / math.tan(0.5 * T["camera_angle_x"])
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)
    flip = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                        dtype=torch.float32)  # GL -> CV
    vm = torch.stack([
        torch.linalg.inv(torch.tensor(f["transform_matrix"], dtype=torch.float32) @ flip)
        for f in T["frames"]]).to(dev)

    def imagen(k: int) -> torch.Tensor:
        # Bajo demanda: a 1024 px las 1600 imagenes son ~20 GB y no caben en RAM.
        rgba = np.asarray(Image.open(args.salida / T["frames"][k]["file_path"]).convert("RGBA"),
                          np.float32) / 255.0
        return torch.from_numpy(rgba[..., :3] * rgba[..., 3:4]).to(dev)   # sobre negro

    idx = np.arange(len(T["frames"]))
    test = idx[::4]
    train = np.setdiff1d(idx, test)
    rng2 = np.random.default_rng(1)
    ev_test = rng2.choice(test, min(24, len(test)), replace=False)

    def render(v):
        # `fdi` no entra en el rasterizador: no es apariencia, solo viaja.
        out, _, info = rasterization(
            params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.sigmoid(params["colors"]), v[None], K[None], W, H, packed=False)
        return out[0], info

    def psnr(js) -> float:
        with torch.no_grad():
            return float(np.mean([
                -10 * math.log10(((render(vm[j])[0] - imagen(j)) ** 2).mean().item() + 1e-10)
                for j in js]))

    lr = {"means": 1e-3, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 1e-2,
          "fdi": 0.0}
    opt = {k: torch.optim.Adam([v], lr=lr[k]) for k, v in params.items()}
    # Densificacion suave —la siembra desde la malla ya es densa—, poda de las casi
    # transparentes y reset periodico de opacidad. El reset+poda es lo que evita la neblina
    # y deja un campo que se ve en cualquier visor al umbral estandar.
    strat = DefaultStrategy(refine_start_iter=500, refine_stop_iter=args.iteraciones // 2,
                            reset_every=1500, refine_every=100, verbose=False)
    strat.check_sanity(params, opt)
    estado = strat.initialize_state(scene_scale=1.0)

    torch.manual_seed(0)
    t0 = time.perf_counter()
    for it in range(args.iteraciones):
        i = int(np.random.choice(train))
        out, info = render(vm[i])
        strat.step_pre_backward(params, opt, estado, it, info)
        tgt = imagen(i)
        a, b = out.permute(2, 0, 1).unsqueeze(0), tgt.permute(2, 0, 1).unsqueeze(0)
        perdida = 0.8 * (out - tgt).abs().mean() + 0.2 * (1.0 - ssim(a, b, data_range=1.0))
        for o in opt.values():
            o.zero_grad()
        perdida.backward()
        for o in opt.values():
            o.step()
        strat.step_post_backward(params, opt, estado, it, info, packed=False)
        if it % 500 == 0 or it == args.iteraciones - 1:
            print(f"  it {it:5d}  N={params['means'].shape[0]:>7,}  "
                  f"perdida {perdida.item():.4f}  retenidas {psnr(ev_test):.2f} dB", flush=True)
    print(f"entrenamiento {time.perf_counter() - t0:.0f}s · "
          f"{params['means'].shape[0]:,} gaussianas finales")

    guarda(args, params, T, centro, torch)


def guarda(args, params, T: dict, centro: np.ndarray, torch) -> None:
    """Un PLY por capa, en coordenadas del ESCANER, mas los centroides por pieza.

    Se escribe aqui en vez de reutilizar `_escribe_inria` del exportador porque aquel esta
    atado al `TwinSnapshot` y a la transferencia densidad→alfa, y este campo no tiene
    ninguna de las dos: su opacidad la aprendio el optimizador.

    Se parte en capas —coronas y encia— porque es como el visor las conmuta, y se emite el
    centroide de cada pieza porque es lo que hace posible pinchar un diente. Ese centroide
    sale de gaussianas con etiqueta EXACTA, no de un reparto por cercania.
    """
    from export_agents.visor import _PROPIEDADES_INRIA, C0

    with torch.no_grad():
        escala = 1.0 / float(T["scan_scale"])
        pos = params["means"].cpu().numpy().astype(np.float64) * escala + centro
        col = torch.sigmoid(params["colors"]).cpu().numpy().astype(np.float64)
        esc = params["scales"].cpu().numpy().astype(np.float64) + np.log(escala)
        rot = torch.nn.functional.normalize(params["quats"], dim=-1).cpu().numpy()
        opa = params["opacities"].cpu().numpy().astype(np.float64)
        fdi = params["fdi"].cpu().numpy().ravel().round().astype(np.int64)

    columnas = {
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "nx": np.zeros(len(pos)), "ny": np.zeros(len(pos)), "nz": np.zeros(len(pos)),
        "opacity": opa,
        **{f"f_dc_{i}": (col[:, i] - 0.5) / C0 for i in range(3)},
        **{f"scale_{i}": esc[:, i] for i in range(3)},
        **{f"rot_{i}": rot[:, i].astype(np.float64) for i in range(4)},
    }

    for capa, m in (("coronas", fdi > 0), ("encia", fdi == 0)):
        destino = args.salida / f"gs_escaner-{capa}.ply"
        cabecera = [
            "ply", "format binary_little_endian 1.0",
            "comment perfil INRIA 3DGS grado 0 - APARIENCIA, no medida",
            "comment las gaussianas NO son los vertices del escaner: el optimizador las",
            "comment movio, dividio y podo. No hay correspondencia 1:1 con lo medido.",
            "comment la ETIQUETA FDI si es exacta: viajo como parametro con tasa cero, asi",
            "comment que cada gaussiana desciende de un vertice de una sola pieza.",
            f"comment entrenado contra {len(T['frames'])} renders EEVEE del STL, "
            f"{args.iteraciones} iteraciones",
            "comment f_dc_* = falso color por FDI - un escaner intraoral no da este color",
            "comment coordenadas en mm del escaner; scan_scale/offset en transforms.json",
            f"element vertex {int(m.sum())}",
            *(f"property float {p}" for p in _PROPIEDADES_INRIA),
            "end_header",
        ]
        texto = "\n".join(cabecera) + "\n"
        if not texto.isascii():
            raise ValueError("la cabecera del PLY tiene caracteres no ASCII")
        datos = np.column_stack([columnas[p][m] for p in _PROPIEDADES_INRIA]).astype(np.float32)
        with destino.open("wb") as f:
            f.write(texto.encode("ascii"))
            f.write(datos.tobytes())
        print(f"escrito {destino.name} ({int(m.sum()):,} gaussianas)")

    piezas = sorted(set(int(c) for c in np.unique(fdi)) - {0})
    lado = {
        "perfil_ply": "inria-3dgs/grado-0 (apariencia, entrenado)",
        "vistas": len(T["frames"]), "iteraciones": args.iteraciones,
        "scan_scale": T["scan_scale"], "scan_offset": T.get("scan_offset"),
        "gaussianas": {c: int((fdi == c).sum()) for c in piezas},
        "centroides": {str(c): pos[fdi == c].mean(axis=0).round(4).tolist() for c in piezas},
    }
    (args.salida / "escaner_3dgs.json").write_text(json.dumps(lado, indent=2))
    print(f"{len(piezas)} piezas con etiqueta exacta · "
          f"{int((fdi > 0).sum()):,} gaussianas de corona, {int((fdi == 0).sum()):,} de encia")


if __name__ == "__main__":
    sys.exit(main())
