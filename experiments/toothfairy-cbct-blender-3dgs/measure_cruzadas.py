#!/usr/bin/env python
"""Medir las 4 capas cruzadas ya entrenadas contra la DRR completa, mismo holdout.

    ~/.venvs/dental-gpu/bin/python measure_cruzadas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

EXPERIMENT = Path("/home/lgarbayo/agentic-smart-health/experiments/toothfairy-cbct-blender-3dgs")
PAQUETE = Path("/home/lgarbayo/agentic-smart-health/data/processed/demo-capas")
sys.path.insert(0, str(EXPERIMENT))

CAPAS = ["diente-esmalte", "diente-dentina", "hueso-cortical", "hueso-trabecular"]
_ALPHA_MIN = 6e-3
_EPS = 1e-6
_TAU_FLOOR = 1e-4


def cargar(capa: str, device: torch.device) -> dict:
    with np.load(PAQUETE / f"splats_volumetric_{capa}.npz") as d:
        return {
            "means": torch.from_numpy(d["means"]).float().to(device),
            "scales": torch.from_numpy(d["scales"]).float().to(device),
            "quats": torch.from_numpy(d["quats"]).float().to(device),
            "sigma": torch.from_numpy(d["density"]).float().to(device),
            "tau_ref": float(d["tau_ref"]),
        }


@torch.no_grad()
def render_tau(campo: dict, viewmats: torch.Tensor, Ks: torch.Tensor,
               width: int, height: int) -> torch.Tensor:
    alpha = (1.0 - torch.exp(-campo["sigma"])).clamp(_ALPHA_MIN, 1.0 - _EPS)
    _, alphas, _ = rasterization(
        means=campo["means"], quats=campo["quats"] / campo["quats"].norm(dim=-1, keepdim=True),
        scales=campo["scales"], opacities=alpha,
        colors=torch.ones(campo["means"].shape[0], 1, device=campo["means"].device),
        sh_degree=None, viewmats=viewmats, Ks=Ks, width=width, height=height,
    )
    return -torch.log((1.0 - alphas[..., 0]).clamp_min(_TAU_FLOOR))


def main() -> int:
    device = torch.device("cuda")
    meta = json.loads((PAQUETE / "transforms.json").read_text(encoding="utf-8"))
    w, h = meta["w"], meta["h"]
    n_vistas = len(meta["frames"])
    holdout = list(range(0, n_vistas, 8))

    c2w = torch.from_numpy(
        np.stack([np.asarray(meta["frames"][i]["transform_matrix"], np.float32)
                  for i in holdout])).float().to(device)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=device))
    viewmats = torch.linalg.inv(c2w @ flip).float()
    K = torch.tensor([[meta["fl_x"], 0.0, meta["cx"]], [0.0, meta["fl_y"], meta["cy"]],
                      [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    Ks = K.unsqueeze(0).repeat(len(holdout), 1, 1).float()

    with np.load(PAQUETE / "drr.npz") as d:
        gt = d["tau"][holdout]
        ref_total = float(d["tau_ref"])

    suma = np.zeros_like(gt)
    for capa in CAPAS:
        campo = cargar(capa, device)
        tau = render_tau(campo, viewmats, Ks, w, h).cpu().numpy()
        suma += tau * campo["tau_ref"]
    mse = float(np.mean((np.clip(suma / ref_total, 0, 1)
                         - np.clip(gt / ref_total, 0, 1)) ** 2))
    suma_db = -10 * np.log10(max(mse, 1e-20))

    unico = json.loads((PAQUETE / "metrics_volumetric_todo.json").read_text(encoding="utf-8"))
    unico_db = unico["psnr_holdout"]
    delta = suma_db - unico_db
    print(f"PSNR suma 4 capas cruzadas {suma_db:.2f} dB · campo único {unico_db:.2f} dB · "
          f"Δ {delta:+.2f} dB")
    print("CRITERIO (>0,5 dB):", "CUMPLIDO" if delta > 0.5 else "FALLIDO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
