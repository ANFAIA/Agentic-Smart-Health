#!/usr/bin/env python
"""umbral_vs_verdad.py — ¿Cuánto diente recupera un umbral, contra una verdad conocida?

    uv run python scripts/umbral_vs_verdad.py --volumen F_001_0000.mha --etiquetas F_001.mha

**Por qué existe.** Sobre `histora` está medido que umbralizar el CBCT **no separa** la
raíz del hueso alveolar: a cualquier umbral que incluya dentina, diente y hueso salen
como un solo bloque de 80-110 mm. Pero ese experimento no tiene verdad de referencia, así
que solo puede decir «no se separa», no *cuánto* se pierde ni **por qué**.

Aquí sí la hay: ToothFairy2 trae el mapa de 6 clases, donde el diente (esmalte + dentina)
y el hueso (cortical, trabecular, medular) están anotados por separado. Con eso las
preguntas se vuelven medibles:

1. **¿Cuánto del diente anotado captura cada umbral?** (recall)
2. **¿Cuánto de lo que captura es hueso?** (precisión) — un umbral que se lleva la mitad
   de la mandíbula tiene recall perfecto y no sirve de nada.
3. **¿Existe ALGÚN umbral que separe?** Es decir, con recall alto y precisión alta a la
   vez. Si no existe, queda demostrado que el problema no es elegir mejor el número: es
   que la información no está en la intensidad, y hace falta un modelo con prior de forma.

La respuesta a la 3 es la que decide si merece la pena entrenar un segmentador.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "experiments/toothfairy-cbct-blender-3dgs"))

from tf_pipeline.volume_io import read_volume  # noqa: E402

# Mapa de 6 clases del espejo de ToothFairy2, tal como lo documenta `tf_pipeline/bands.py`.
CLASE_FONDO = 0
CLASES_HUESO = (1,)
CLASES_DIENTE = (2, 3, 4, 5)

BARRIDO_HU = (300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2500)


def f1(recall: float, precision: float) -> float:
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--volumen", type=Path, required=True)
    ap.add_argument("--etiquetas", type=Path, required=True)
    args = ap.parse_args()

    vol = read_volume(args.volumen)
    etq = read_volume(args.etiquetas)
    hu, lab = vol.array, etq.array
    if hu.shape != lab.shape:
        print(f"✗ volumen {hu.shape} y etiquetas {lab.shape} no cuadran.")
        return 1

    sx, sy, sz = vol.spacing
    print(f"volumen {hu.shape} · vóxel {sx:.3f} x {sy:.3f} x {sz:.3f} mm "
          f"· HU {hu.min():.0f}–{hu.max():.0f}")

    diente = np.isin(lab, CLASES_DIENTE)
    hueso = np.isin(lab, CLASES_HUESO)
    mm3 = sx * sy * sz
    print(f"anotación: diente {diente.sum():,} vóxeles ({diente.sum() * mm3 / 1000:.1f} cm³) "
          f"· hueso {hueso.sum():,} ({hueso.sum() * mm3 / 1000:.1f} cm³)")

    # El solapamiento de HU entre las dos clases es la respuesta directa: si las
    # distribuciones se pisan, ningún umbral las separa por definición.
    hu_d, hu_h = hu[diente], hu[hueso]
    print(f"\nHU del diente  p5 {np.percentile(hu_d, 5):.0f} · mediana "
          f"{np.median(hu_d):.0f} · p95 {np.percentile(hu_d, 95):.0f}")
    print(f"HU del hueso   p5 {np.percentile(hu_h, 5):.0f} · mediana "
          f"{np.median(hu_h):.0f} · p95 {np.percentile(hu_h, 95):.0f}")

    print("\n--- qué recupera cada umbral del diente anotado ---")
    print(f"{'HU':>6} {'recall':>8} {'precision':>10} {'F1':>7} {'del hueso':>10}")
    mejor = None
    for u in BARRIDO_HU:
        sel = hu >= u
        n = int(sel.sum())
        if n == 0:
            continue
        tp = int((sel & diente).sum())
        rec = tp / int(diente.sum())
        pre = tp / n
        # Cuánto del hueso anotado se cuela: es lo que hace inservible al umbral.
        col = int((sel & hueso).sum()) / int(hueso.sum())
        f = f1(rec, pre)
        if mejor is None or f > mejor[1]:
            mejor = (u, f, rec, pre)
        print(f"{u:>6} {rec:>7.1%} {pre:>9.1%} {f:>6.3f} {col:>9.1%}")

    u, f, rec, pre = mejor
    print(f"\nmejor umbral por F1: {u} HU → recall {rec:.1%} · precisión {pre:.1%} "
          f"(F1 {f:.3f})")

    # El veredicto. Un umbral solo sirve si captura casi todo el diente Y casi nada más:
    # con recall 0,95 y precisión 0,50, la mitad de lo que se recorta es hueso.
    if rec >= 0.90 and pre >= 0.90:
        print("→ El umbral SEPARA. No hace falta modelo: basta elegir bien el número.")
        return 0
    print(
        "→ NINGÚN umbral separa diente de hueso.\n"
        f"  El mejor compromiso deja {1 - rec:.0%} del diente fuera y mete {1 - pre:.0%}\n"
        "  de material que no es diente. No es un problema de elegir mejor el número:\n"
        "  las dos clases COMPARTEN rango de intensidad, así que la información que las\n"
        "  distingue no está en el HU. Un segmentador con prior de forma sí puede — es\n"
        "  exactamente lo que aporta, y esto lo cuantifica."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
