#!/usr/bin/env python
"""Capas CRUZADAS (clase ∩ HU) para la demo: el encendido clínico diente/hueso.

    ~/.venvs/dental-gpu/bin/python run_layers_cruzadas.py [--smoke]

Igual que `run_layers.py` pero con el mapa de clases anatómicas que faltaba: el
U-Net de diente ya entrenado (`data/processed/cbct-diente/modelo.pt`) segmenta el
CBCT de la demo, y las capas son `diente-esmalte`, `diente-dentina`,
`hueso-cortical`, `hueso-trabecular` (se deja fuera `hueso-medular`, que sin una
segmentación de médula ósea no se puede aislar). `hueso` = tejido duro (HU ≥ 300)
que NO es diente; `diente` = la máscara del U-Net (umbral 0,530).

CRITERIO fijado ANTES: PSNR(suma de capas) − PSNR(campo único) > 0,5 dB sobre el
mismo holdout. Entregable clínico: encender diente y hueso por separado.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT = Path("/home/lgarbayo/agentic-smart-health/experiments/toothfairy-cbct-blender-3dgs")
RAIZ = Path("/home/lgarbayo/agentic-smart-health")
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(RAIZ / "scripts"))

from run_layers import PAQUETE, gpu, escribe_transforms  # noqa: E402
from tf_pipeline.volume_io import Volume  # noqa: E402
from tf_pipeline import bands  # noqa: E402
from gs.render_drr import _mu_volume, drr  # noqa: E402

CHECKPOINT = RAIZ / "data/processed/cbct-diente/modelo.pt"
F1_UMBRAL = 0.530
HU_PACIENTE = 300.0
CAPAS = ["diente-esmalte", "diente-dentina", "hueso-cortical", "hueso-trabecular"]


def mascara_diente(vol: np.ndarray, checkpoint: Path) -> np.ndarray:
    """Máscara de diente del U-Net entrenado, sobre el volumen ya a 0,30 mm."""
    from entrena_diente_cbct import HU_MAX, HU_MIN, PARCHE, UNet3D, normaliza

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=dev)
    modelo = UNet3D().to(dev)
    modelo.load_state_dict(ckpt["modelo"])
    modelo.eval()
    print(f"U-Net: {checkpoint.name} · F1 val {ckpt.get('f1_parche', float('nan')):.3f} "
          f"· HU {HU_MIN:g}..{HU_MAX:g}")

    pad = [(0, (-s) % PARCHE) for s in vol.shape]
    hu = normaliza(np.pad(vol, pad))
    prob = np.zeros(hu.shape, dtype=np.float32)
    with torch.no_grad():
        for z in range(0, hu.shape[0], PARCHE):
            for y in range(0, hu.shape[1], PARCHE):
                for x in range(0, hu.shape[2], PARCHE):
                    s = (slice(z, z + PARCHE), slice(y, y + PARCHE), slice(x, x + PARCHE))
                    xt = torch.from_numpy(hu[s]).unsqueeze(0).unsqueeze(0).to(dev)
                    prob[s] = torch.sigmoid(modelo(xt))[0, 0].cpu().numpy()
    recorte = tuple(slice(0, s) for s in vol.shape)
    mascara = prob[recorte] > F1_UMBRAL
    print(f"máscara de diente: {int(mascara.sum()):,} vóxeles "
          f"({100 * mascara.mean():.1f}% del volumen)")
    return mascara


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    vol = np.load(PAQUETE / "volume.npy")
    meta_json = json.loads((PAQUETE / "meta.json").read_text(encoding="utf-8"))
    spacing = tuple(meta_json["spacing"])
    world_offset = np.zeros(3, np.float32)
    volume = Volume(array=vol, spacing=spacing, fmt="metaimage")

    diente = mascara_diente(vol, CHECKPOINT)
    # clases: 0 fondo/aire, 1 hueso, 2 diente (las que `CAPAS_CRUZADAS` espera).
    clases = np.where(diente, 2, np.where(vol >= HU_PACIENTE, 1, 0)).astype(np.int16)
    labels = Volume(array=clases, spacing=spacing, fmt="metaimage")
    print(f"labels: diente {int((clases == 2).sum()):,} · hueso {int((clases == 1).sum()):,}")

    sx, sy, sz = spacing
    nz, ny, nx = vol.shape
    extent = np.array([nx * sx, ny * sy, nz * sz])
    n_vistas = 64 if args.smoke else 252
    meta = escribe_transforms(n_vistas, extent / 2, float(np.linalg.norm(extent) / 2 * 1.6))

    bandas = ["todo"] + CAPAS
    for band in bandas:
        arrays, summary = bands.seed_field(volume, labels, band, world_offset=world_offset,
                                           max_primitives=500_000)
        bands.save_band(PAQUETE, arrays, band)
        print(f"  siembra {band}: {summary.n_primitives:,} pts · {summary.volume_cm3:.1f} cm3")

    device = torch.device("cuda")
    frames = meta["frames"]
    for band in bandas:
        mascara = bands.band_mask(volume, labels, band)
        arr = np.where(mascara, vol, -1000.0).astype(np.float32)
        mu = torch.from_numpy(_mu_volume(arr)).to(device)
        pila = [drr(mu, spacing, world_offset, meta,
                    np.asarray(f["transform_matrix"], np.float32), n_steps=640,
                    device=device).cpu() for f in frames]
        tau = torch.stack(pila).numpy().astype(np.float32)
        tau_ref = float(np.percentile(tau, 99.9))
        nombre = "drr.npz" if band == "todo" else f"drr_{band}.npz"
        np.savez_compressed(PAQUETE / nombre, tau=tau, tau_ref=np.float32(tau_ref))
        print(f"  drr {band}: tau_ref {tau_ref:.2f}")
    import shutil

    shutil.copy(PAQUETE / "drr.npz", PAQUETE / "drr_todo.npz")

    iters = 50 if args.smoke else 7000
    for band in bandas:
        gpu("train_volumetric.py", "--package", str(PAQUETE), "--band", band,
            "--iters", str(iters), "--n-init", "500000", "--holdout-every", "8")

    if args.smoke:
        print("smoke: se salta la medición.")
        return 0

    # --- suma de las 4 capas cruzadas sobre el MISMO holdout ------------------ #
    from gs.compose_layers import cargar_campo, render_tau
    from gsplat import rasterization  # noqa: F401  (render_tau lo usa)

    holdout = list(range(0, n_vistas, 8))
    c2w = torch.from_numpy(
        np.stack([np.asarray(meta["frames"][i]["transform_matrix"], np.float32)
                  for i in holdout])).to(device)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    viewmats = torch.linalg.inv(c2w @ flip)
    K = torch.tensor([[meta["fl_x"], 0.0, meta["cx"]], [0.0, meta["fl_y"], meta["cy"]],
                      [0.0, 0.0, 1.0]], device=device)
    Ks = K.unsqueeze(0).repeat(len(holdout), 1, 1)

    with np.load(PAQUETE / "drr.npz") as d:
        gt = d["tau"][holdout]
        ref_total = float(d["tau_ref"])

    suma = np.zeros_like(gt)
    for capa in CAPAS:
        campo = cargar_campo(PAQUETE, capa, device)
        tau = render_tau(campo, viewmats, Ks, meta["w"], meta["h"]).cpu().numpy()
        suma += tau * campo["tau_ref"]
    mse = float(np.mean((np.clip(suma / ref_total, 0, 1)
                         - np.clip(gt / ref_total, 0, 1)) ** 2))
    suma_db = -10 * np.log10(max(mse, 1e-20))

    unico = json.loads((PAQUETE / "metrics_volumetric_todo.json").read_text(encoding="utf-8"))
    unico_db = unico["psnr_holdout"]
    delta = suma_db - unico_db
    print(f"\nPSNR suma 4 capas cruzadas {suma_db:.2f} dB · campo único {unico_db:.2f} dB · "
          f"Δ {delta:+.2f} dB")
    print("CRITERIO (>0,5 dB):", "CUMPLIDO" if delta > 0.5 else "FALLIDO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
