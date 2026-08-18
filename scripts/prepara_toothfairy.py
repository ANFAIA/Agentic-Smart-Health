#!/usr/bin/env python
"""prepara_toothfairy.py — Descarga ToothFairy2 caso a caso y lo deja entrenable.

    uv run python scripts/prepara_toothfairy.py --n 120 --destino ~/anfaia/toothfairy2

**Por qué no un `git clone` del dataset.** El espejo publica los volúmenes en MetaImage
**sin comprimir y en coma flotante**: 686 MB por caso, 315 GB los 480, y el disco tiene
218 GB libres. Descargarlo entero no cabe.

Y no hace falta: los HU son enteros, así que `float → int16` es **sin pérdida real**
(medido: 0,5 HU de redondeo máximo, frente a los ±300 de ruido dentro de una meseta que
mide la propia ficha del CBCT). Recortando además a la caja de la anatomía etiquetada
—que es el 37 % del volumen; el resto es aire— cada caso baja a **32 MB**. Los 480 caben
en 15 GB.

**Reanudable a propósito.** Descargar 480 casos son horas: el script salta lo que ya
está convertido, así que se puede parar y seguir. Y borra el `.mha` en cuanto lo
convierte, para que el pico de disco sea un caso, no la colección.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "experiments/toothfairy-cbct-blender-3dgs"))

from tf_pipeline.volume_io import read_volume  # noqa: E402

REPO = "yeldafrt72/ToothFairy2_Complete_6Class"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def casos_disponibles() -> list[str]:
    """Los identificadores que el dataset publica, preguntándoselo a la API."""
    import json
    import urllib.request

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}") as r:
        info = json.load(r)
    imgs = {
        f["rfilename"].split("/")[-1].replace("_0000.mha", "")
        for f in info["siblings"] if f["rfilename"].startswith("imagesTr/")
    }
    labs = {
        f["rfilename"].split("/")[-1].replace(".mha", "")
        for f in info["siblings"] if f["rfilename"].startswith("labelsTr")
    }
    # Solo los que tienen las DOS mitades: un volumen sin etiqueta no entrena nada.
    return sorted(imgs & labs)


def descarga(url: str, destino: Path) -> bool:
    r = subprocess.run(
        ["curl", "-sfL", "--retry", "3", "-o", str(destino), url],
        capture_output=True,
    )
    return r.returncode == 0 and destino.exists() and destino.stat().st_size > 0


def convierte(img: Path, lab: Path, salida: Path) -> dict:
    """`.mha` crudos → un `.npz` con el volumen en int16 y las etiquetas en uint8."""
    vol, etq = read_volume(img), read_volume(lab)
    hu = np.rint(vol.array).astype(np.int16)
    etiq = etq.array.astype(np.uint8)
    if hu.shape != etiq.shape:
        raise ValueError(f"volumen {hu.shape} y etiquetas {etiq.shape} no cuadran")

    # Recorte a la caja de lo etiquetado: fuera solo hay aire, y ocupa el 63 %.
    nz = np.argwhere(etiq > 0)
    if len(nz) == 0:
        raise ValueError("el mapa de etiquetas está vacío")
    lo, hi = nz.min(0), nz.max(0) + 1
    rec = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))

    np.savez_compressed(
        salida, hu=hu[rec], lab=etiq[rec],
        spacing=np.asarray(vol.spacing, dtype=np.float64), origen=lo,
    )
    return {"forma": tuple(hi - lo), "mb": salida.stat().st_size / 1e6}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--destino", type=Path, default=Path.home() / "anfaia/toothfairy2")
    ap.add_argument("--n", type=int, default=0, help="Cuántos casos (0 = todos).")
    args = ap.parse_args()

    args.destino.mkdir(parents=True, exist_ok=True)
    tmp = args.destino / "_tmp"
    tmp.mkdir(exist_ok=True)

    casos = casos_disponibles()
    if args.n:
        casos = casos[: args.n]
    print(f"{len(casos)} caso(s) a preparar → {args.destino}")

    hechos = fallos = 0
    t0 = time.time()
    for i, caso in enumerate(casos, 1):
        salida = args.destino / f"{caso}.npz"
        if salida.exists():
            hechos += 1
            continue
        img, lab = tmp / f"{caso}_0000.mha", tmp / f"{caso}.mha"
        try:
            if not descarga(f"{BASE}/imagesTr/{caso}_0000.mha", img):
                raise RuntimeError("no se pudo descargar el volumen")
            if not descarga(f"{BASE}/labelsTr_6cls/{caso}.mha", lab):
                raise RuntimeError("no se pudo descargar las etiquetas")
            info = convierte(img, lab, salida)
            hechos += 1
            print(f"[{i}/{len(casos)}] {caso} · {info['forma']} · {info['mb']:.0f} MB "
                  f"· {(time.time() - t0) / 60:.0f} min")
        except Exception as e:  # noqa: BLE001
            fallos += 1
            print(f"[{i}/{len(casos)}] {caso} ✗ {type(e).__name__}: {e}")
            salida.unlink(missing_ok=True)
        finally:
            # El `.mha` se borra SIEMPRE: el pico de disco es un caso, no la coleccion.
            img.unlink(missing_ok=True)
            lab.unlink(missing_ok=True)

    total = sum(f.stat().st_size for f in args.destino.glob("*.npz")) / 1e9
    print(f"\n{hechos} preparado(s) · {fallos} fallo(s) · {total:.1f} GB en disco")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
