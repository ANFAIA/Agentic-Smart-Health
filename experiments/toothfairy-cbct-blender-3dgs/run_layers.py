#!/usr/bin/env python
"""Descomponer el campo de densidad de la demo en capas por HU y medir.

    ~/.venvs/dental-gpu/bin/python run_layers.py [--smoke]

Replica el hallazgo 4 de `docs/research/3dgs-volumetrico-cbct.md` sobre el CBCT de
la demo, sin mapa de etiquetas: partición `PARTICION_HU` (4 tramos de densidad) con
máscara de paciente sintetizada como `HU > 300`. Se entrena el campo único de control
(`todo`) y las 4 capas con la MISMA rama volumétrica (DRR, Beer-Lambert) y se compone
la suma contra la MISMA DRR completa.

CRITERIO fijado ANTES de correr: PSNR(suma de 4 capas) − PSNR(campo único) > 0,5 dB
sobre el mismo holdout. Si no, FALLIDO con esas palabras.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT = Path("/home/lgarbayo/agentic-smart-health/experiments/toothfairy-cbct-blender-3dgs")
PAQUETE = Path("/home/lgarbayo/agentic-smart-health/data/processed/demo-capas")
PY_GPU = "/home/lgarbayo/.venvs/dental-gpu/bin/python"
sys.path.insert(0, str(EXPERIMENT))

from tf_pipeline.volume_io import Volume  # noqa: E402
from tf_pipeline import bands  # noqa: E402
from gs.render_drr import _mu_volume, drr  # noqa: E402

HU_PACIENTE = 300.0  # máscara de "dónde acaba el paciente" (tejido ≥ blando)
BANDAS = ["todo", "densidad-baja", "densidad-media", "densidad-alta", "densidad-muy-alta"]
ANCHO = 800
FOV_RAD = np.deg2rad(30.0)


def escribe_transforms(n_vistas: int, centro: np.ndarray, radio: float) -> None:
    """Órbita alrededor del eje Y (misma familia de matrices que el ToothFairy)."""
    fl = (ANCHO / 2) / np.tan(FOV_RAD / 2)
    frames = []
    for i in range(n_vistas):
        th = 2 * np.pi * i / n_vistas
        c, s = np.cos(th), np.sin(th)
        pos = centro + radio * np.array([c, 0.0, s])
        rot = [[0.0, -s, c], [1.0, 0.0, 0.0], [0.0, c, s]]
        m = [[rot[0][0], rot[0][1], rot[0][2], float(pos[0])],
             [rot[1][0], rot[1][1], rot[1][2], float(pos[1])],
             [rot[2][0], rot[2][1], rot[2][2], float(pos[2])],
             [0.0, 0.0, 0.0, 1.0]]
        frames.append({"file_path": f"drr_{i:04d}.png", "azimuth_deg": np.rad2deg(th),
                       "elevation_deg": 0.0, "transform_matrix": m})
    meta = {"camera_model": "OPENCV", "convention": "opengl (c2w: -Z adelante, +Y arriba)",
            "w": ANCHO, "h": ANCHO, "fl_x": fl, "fl_y": fl, "cx": ANCHO / 2, "cy": ANCHO / 2,
            "camera_angle_x": FOV_RAD, "camera_angle_y": FOV_RAD,
            "orbit_radius": float(radio), "frames": frames}
    (PAQUETE / "transforms.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def gpu(script: str, *extra: str) -> None:
    cmd = [PY_GPU, str(EXPERIMENT / "gs" / script), *extra]
    env = {"PYTHONPATH": str(EXPERIMENT), "PATH": "/usr/bin:/bin"}
    env.update({k: v for k, v in __import__("os").environ.items()})
    print(f"· {Path(script).name} {' '.join(extra)}")
    if subprocess.run(cmd, text=True, env=env).returncode != 0:
        raise RuntimeError(f"{script} falló")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    vol = np.load(PAQUETE / "volume.npy")
    meta_json = json.loads((PAQUETE / "meta.json").read_text(encoding="utf-8"))
    spacing = tuple(meta_json["spacing"])
    world_offset = np.zeros(3, np.float32)
    volume = Volume(array=vol, spacing=spacing, fmt="metaimage")
    labels = Volume(array=(vol > HU_PACIENTE).astype(np.int16), spacing=spacing, fmt="metaimage")

    sx, sy, sz = spacing
    nz, ny, nx = vol.shape
    extent = np.array([nx * sx, ny * sy, nz * sz])
    centro = extent / 2
    radio = float(np.linalg.norm(extent) / 2 * 1.6)

    n_vistas = 64 if args.smoke else 252
    meta = escribe_transforms(n_vistas, centro, radio)
    print(f"transforms: {n_vistas} vistas · extent {np.round(extent, 1)} mm · radio {radio:.0f}")

    # --- siembra ----------------------------------------------------------- #
    for band in BANDAS:
        arrays, summary = bands.seed_field(volume, labels, band, world_offset=world_offset,
                                           max_primitives=500_000)
        bands.save_band(PAQUETE, arrays, band)
        print(f"  siembra {band}: {summary.n_primitives:,} pts · "
              f"{summary.volume_cm3:.1f} cm3 · HU [{summary.hu_min:.0f}, {summary.hu_max:.0f}]")

    # --- DRR por banda + completa ----------------------------------------- #
    device = torch.device("cuda")
    frames = meta["frames"]
    for band in BANDAS:
        mascara = bands.band_mask(volume, labels, band)
        arr = np.where(mascara, vol, -1000.0).astype(np.float32)
        mu = torch.from_numpy(_mu_volume(arr)).to(device)
        pila = []
        for f in frames:
            pila.append(drr(mu, spacing, world_offset, meta,
                            np.asarray(f["transform_matrix"], np.float32),
                            n_steps=640, device=device).cpu())
        tau = torch.stack(pila).numpy().astype(np.float32)
        tau_ref = float(np.percentile(tau, 99.9))
        nombre = "drr.npz" if band == "todo" else f"drr_{band}.npz"
        np.savez_compressed(PAQUETE / nombre, tau=tau, tau_ref=np.float32(tau_ref))
        print(f"  drr {band}: tau_ref {tau_ref:.2f} · tau max {tau.max():.2f}")
    # train_volumetric --band todo lee drr_todo.npz (la completa, igual que drr.npz).
    shutil.copy(PAQUETE / "drr.npz", PAQUETE / "drr_todo.npz")

    # --- entrenamiento ----------------------------------------------------- #
    iters = 50 if args.smoke else 7000
    for band in BANDAS:
        gpu("train_volumetric.py", "--package", str(PAQUETE), "--band", band,
            "--iters", str(iters), "--n-init", "500000", "--holdout-every", "8")

    # --- composición y veredicto ------------------------------------------ #
    if args.smoke:
        print("smoke: se salta la composición (iters de prueba).")
        return 0
    holdout = [str(i) for i in range(0, n_vistas, 8)]
    gpu("compose_layers.py", "--package", str(PAQUETE), "--conjunto", "densidad",
        "--views", *holdout)
    capas_json = json.loads((PAQUETE / "capas" / "capas.json").read_text(encoding="utf-8"))
    suma_db = capas_json["suma_vs_drr_completa"]["psnr_db"]
    unico = json.loads((PAQUETE / "metrics_volumetric_todo.json").read_text(encoding="utf-8"))
    unico_db = unico["psnr_holdout"]
    delta = suma_db - unico_db
    print(f"\nPSNR suma 4 capas {suma_db:.2f} dB · campo único {unico_db:.2f} dB · "
          f"Δ {delta:+.2f} dB")
    print("CRITERIO (>0,5 dB):", "CUMPLIDO" if delta > 0.5 else "FALLIDO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
