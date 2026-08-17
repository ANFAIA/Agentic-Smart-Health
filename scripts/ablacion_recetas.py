#!/usr/bin/env python
"""Ablación de la receta de entrenamiento: qué aporta cada pieza.

Entrena el MISMO caso cuatro veces, cambiando una sola cosa cada vez, y mide el
PSNR sobre las MISMAS vistas retenidas. Es el experimento que decidió la receta
del notebook 06, y el que produce la tabla del README.

    A  L1 sola, sin densificar, sin armónicos   (la receta de partida)
    B  + densificación y poda (DefaultStrategy)
    C  + SSIM en la pérdida
    D  + armónicos esféricos de grado 2         (la que usa el 06)

Uso:
    ~/.venvs/dental-gpu/bin/python scripts/ablacion_recetas.py
    ~/.venvs/dental-gpu/bin/python scripts/ablacion_recetas.py --iters 3000 --figura

Requiere GPU y el entorno del notebook 06 (torch + gsplat), no el `.venv` del
proyecto. Necesita que el notebook 05 haya generado el paquete del caso (corrida
densa, en `data/processed/teeth3ds-dense/`).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vtk
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from vtk.util.numpy_support import vtk_to_numpy

vtk.vtkObject.GlobalWarningDisplayOff()

# Raiz del repo: se ancla en .git, igual que los notebooks. data/ esta gitignored,
# asi que no sirve de ancla en un clon limpio.
ROOT = Path(__file__).resolve().parent
while ROOT != ROOT.parent and not (ROOT / ".git").exists():
    ROOT = ROOT.parent

CADA_N = 8          # 1 de cada 8 vistas se aparta y no se entrena con ella
UMBRAL_OPACIDAD = 0.3   # para contar gaussianas "utiles"
UMBRAL_ESCALA = 5.0     # mm

dev = torch.device("cuda")


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def leer_png(path: Path) -> np.ndarray:
    r = vtk.vtkPNGReader()
    r.SetFileName(str(path))
    r.Update()
    im = r.GetOutput()
    d = im.GetDimensions()
    a = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(d[1], d[0], -1)
    return a[::-1, :, :3].astype(np.float32) / 255.0


def cargar(case_dir: Path):
    meta = json.loads((case_dir / "transforms.json").read_text())
    W, H = meta["w"], meta["h"]
    gt = torch.from_numpy(
        np.stack([leer_png(case_dir / f["file_path"]) for f in meta["frames"]])).to(dev)
    c2w = torch.from_numpy(np.stack(
        [np.array(f["transform_matrix"], np.float32) for f in meta["frames"]])).to(dev)
    # OpenGL -> OpenCV, que es lo que espera gsplat
    flip = torch.diag(torch.tensor([1., -1., -1., 1.], device=dev))
    viewmats = torch.linalg.inv(c2w @ flip)
    K = torch.tensor([[meta["fl_x"], 0, meta["cx"]],
                      [0, meta["fl_y"], meta["cy"]], [0, 0, 1]],
                     dtype=torch.float32, device=dev)
    Ks = K.unsqueeze(0).repeat(len(meta["frames"]), 1, 1)

    pr = vtk.vtkPLYReader()
    pr.SetFileName(str(case_dir / "init.ply"))
    pr.Update()
    init = vtk_to_numpy(pr.GetOutput().GetPoints().GetData()).astype(np.float32)
    pos = c2w[:, :3, 3].cpu().numpy()
    return gt, viewmats, Ks, W, H, init, pos


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def vista_frontal(pos: np.ndarray, idx_test: np.ndarray) -> int:
    """Vista retenida más próxima a (azimut 0°, elevación −10°): la frontal.

    Anclada por ÁNGULO y no por posición en la lista de retenidas: el índice
    depende de cuántas vistas tenga la rejilla, el ángulo no. Es la misma que
    enseña el panel «frontal» del notebook 06, así que las cifras son comparables.
    """
    d = pos - pos.mean(0)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    elev = np.degrees(np.arcsin(d[:, 1]))
    azim = np.degrees(np.arctan2(d[:, 0], d[:, 2])) % 360
    d_az = np.abs((azim[idx_test] - 0 + 180) % 360 - 180)
    return int(idx_test[np.argmin(d_az + np.abs(elev[idx_test] + 10))])


def construir(init: np.ndarray, extent: float, sh_grado: int, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(init)
    params = {
        "means": torch.nn.Parameter(torch.from_numpy(init).to(dev)),
        "scales": torch.nn.Parameter(torch.full((n, 3), np.log(extent * 0.005), device=dev)),
        "quats": torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=dev).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.full((n,), float(np.log(0.1 / 0.9)), device=dev)),
    }
    # Con grado 0 el color es un RGB plano; con grado > 0 son coeficientes SH.
    forma = (n, 3) if sh_grado == 0 else (n, (sh_grado + 1) ** 2, 3)
    params["colors"] = torch.nn.Parameter(torch.zeros(forma, device=dev))

    lr = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3,
          "opacities": 5e-2, "colors": 2.5e-3}
    # DefaultStrategy exige UN optimizador por parametro, no un Adam con grupos.
    opts = {k: torch.optim.Adam([{"params": [v], "lr": lr[k]}]) for k, v in params.items()}
    return params, opts


def render(params, viewmats, Ks, idx, W, H, sh_grado):
    extra = {"sh_degree": sh_grado} if sh_grado > 0 else {}
    colors = params["colors"] if sh_grado > 0 else torch.sigmoid(params["colors"])
    return rasterization(
        means=params["means"],
        quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors, viewmats=viewmats[idx], Ks=Ks[idx],
        width=W, height=H, **extra)


@torch.no_grad()
def psnr_medio(params, viewmats, Ks, gt, idx, W, H, sh_grado) -> float:
    vals = []
    for i in range(0, len(idx), 8):
        trozo = [int(j) for j in idx[i:i + 8]]
        pred, _, _ = render(params, viewmats, Ks, trozo, W, H, sh_grado)
        vals.append(float(-10 * torch.log10(F.mse_loss(pred.clamp(0, 1), gt[trozo]))))
    return float(np.mean(vals))


def utiles(params) -> int:
    """Gaussianas que aportan algo: opacas y de tamaño razonable."""
    with torch.no_grad():
        op = torch.sigmoid(params["opacities"]).cpu().numpy()
        sc = torch.exp(params["scales"]).max(1).values.cpu().numpy()
    return int(((op > UMBRAL_OPACIDAD) & (sc < UMBRAL_ESCALA)).sum())


def entrenar(datos, densificar: bool, usar_ssim: bool, sh_grado: int, iters: int):
    gt, viewmats, Ks, W, H, init, pos = datos
    n_vistas = gt.shape[0]
    idx_test = np.arange(0, n_vistas, CADA_N)
    idx_train = np.setdiff1d(np.arange(n_vistas), idx_test)
    extent = float(np.linalg.norm(init.max(0) - init.min(0)))

    params, opts = construir(init, extent, sh_grado)
    estrategia = estado = None
    if densificar:
        estrategia = DefaultStrategy(refine_stop_iter=int(iters * 0.75))
        estrategia.check_sanity(params, opts)
        estado = estrategia.initialize_state(scene_scale=extent)

    t0 = time.time()
    for paso in range(iters):
        cam = int(np.random.choice(idx_train))
        img, _, info = render(params, viewmats, Ks, [cam], W, H, sh_grado)
        if estrategia:
            estrategia.step_pre_backward(params, opts, estado, paso, info)

        pred = img[0]
        loss = F.l1_loss(pred, gt[cam])
        if usar_ssim:
            a = pred.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            b = gt[cam].permute(2, 0, 1).unsqueeze(0)
            loss = 0.8 * loss + 0.2 * (1.0 - ssim_fn(a, b, data_range=1.0))

        for o in opts.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts.values():
            o.step()
        if estrategia:
            # rasterization usa packed=True por defecto: el info viene empaquetado.
            estrategia.step_post_backward(params, opts, estado, paso, info, packed=True)

    vista = vista_frontal(pos, idx_test)
    return {
        "gauss": params["means"].shape[0],
        "utiles": utiles(params),
        # Media sobre TODAS las retenidas: la metrica defendible.
        "psnr": psnr_medio(params, viewmats, Ks, gt, idx_test, W, H, sh_grado),
        # Y la de la vista concreta que sale en la figura, para que la tabla y
        # la imagen hablen del mismo numero.
        "psnr_vista": psnr_medio(params, viewmats, Ks, gt, np.array([vista]), W, H, sh_grado),
        "vista": vista,
        "s": time.time() - t0,
        "params": params,
        "sh": sh_grado,
    }


RECETAS = [
    ("A · actual",            False, False, 0),
    ("B · + densificación",   True,  False, 0),
    ("C · + SSIM",            True,  True,  0),
    ("D · + armónicos gr. 2", True,  True,  2),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caso", default="01A6GW4A_lower_3dgs")
    ap.add_argument("--iters", type=int, default=9000)
    ap.add_argument("--figura", action="store_true",
                    help="escribe la comparación visual en data/processed/_ablacion/")
    args = ap.parse_args()

    case_dir = ROOT / "data/processed/teeth3ds-dense" / args.caso
    if not case_dir.exists():
        raise SystemExit(f"Falta {case_dir} — ejecuta antes el notebook 05.")

    datos = cargar(case_dir)
    n_vistas = datos[0].shape[0]
    print(f"caso {args.caso} · {n_vistas} vistas "
          f"({n_vistas - len(range(0, n_vistas, CADA_N))} entren. / "
          f"{len(range(0, n_vistas, CADA_N))} retenidas) · {args.iters} iteraciones\n")
    print(f"{'receta':<24}{'gauss.':>10}{'útiles':>9}{'%':>6}"
          f"{'PSNR ret':>10}{'PSNR vista':>12}{'s':>7}")
    print("-" * 78)

    filas = []
    for nombre, densificar, ssim, sh in RECETAS:
        r = entrenar(datos, densificar, ssim, sh, args.iters)
        r["receta"] = nombre
        filas.append(r)
        print(f"{nombre:<24}{r['gauss']:>10,}{r['utiles']:>9,}"
              f"{100 * r['utiles'] / r['gauss']:>5.0f}%{r['psnr']:>10.2f}"
              f"{r['psnr_vista']:>12.2f}{r['s']:>7.0f}")

    base = filas[0]["psnr"]
    print("-" * 78)
    print(f"  vista de la figura: r_{filas[0]['vista']:04d} (frontal, retenida)")
    for r in filas:
        print(f"  {r['receta']:<24} media {r['psnr']:>6.2f} dB ({r['psnr'] - base:+.2f} vs A)"
              f"  ·  esa vista {r['psnr_vista']:>6.2f} dB")

    if args.figura:
        escribir_figura(filas, datos, case_dir, args)


def escribir_figura(filas, datos, case_dir, args) -> None:
    """Fila superior: la vista completa. Inferior: el mismo detalle ampliado.

    El recorte es donde se ve la diferencia — en la vista completa las cuatro
    recetas parecen parecidas; en los molares no.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt, viewmats, Ks, W, H, _, _ = datos
    vista = filas[0]["vista"]   # la misma que mide la tabla
    ref = gt[vista].cpu().numpy()
    # Cuadrante de molares, en fraccion de la imagen para que aguante otro tamano.
    zy = slice(int(0.375 * H), int(0.825 * H))
    zx = slice(int(0.150 * W), int(0.600 * W))

    n = len(filas) + 1
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.6))
    axes[0, 0].imshow(ref)
    axes[0, 0].set_title("malla (referencia)", fontsize=12, pad=8)
    axes[1, 0].imshow(ref[zy, zx])

    for k, r in enumerate(filas, start=1):
        with torch.no_grad():
            img, _, _ = render(r["params"], viewmats, Ks, [vista], W, H, r["sh"])
        im = img[0].clamp(0, 1).cpu().numpy()
        axes[0, k].imshow(im)
        axes[0, k].set_title(f"{r['receta']}\n{r['psnr_vista']:.2f} dB · "
                             f"{r['gauss']:,} gauss.", fontsize=11, pad=8)
        axes[1, k].imshow(im[zy, zx])

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("vista completa", fontsize=10)
    axes[1, 0].set_ylabel("detalle · molares", fontsize=10)
    fig.suptitle(f"Ablación de la receta · vista retenida r_{vista:04d} "
                 f"({W}x{H}, {args.iters} iteraciones)", fontsize=14)
    fig.tight_layout()

    salida = ROOT / "data/processed/_ablacion"
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / "ablacion.png"
    fig.savefig(destino, dpi=110)
    print(f"\nfigura: {destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
