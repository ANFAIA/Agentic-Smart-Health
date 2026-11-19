#!/usr/bin/env python
"""mide_toothfairy2_entrenamiento.py — Qué vio el modelo de ToothFairy2 al entrenar.

    ~/.venvs/nnunet-tf2/bin/python scripts/mide_toothfairy2_entrenamiento.py \\
        --zip ~/anfaia/toothfairy2-oficial/ToothFairy2_Dataset.zip \\
        --salida data/processed/toothfairy2-entrenamiento/medidas.json

La primera ejecución sobre el caso B salió de distribución por la altura de la caja: llegaba
a +40 mm sobre el plano oclusal y una muestra de 16 volúmenes de entrenamiento no pasaba de
+31. Este script mide eso sobre **todas** las etiquetas del zip oficial, sin extraerlo, para
que la caja del experimento siguiente y sus umbrales salgan de una regla escrita antes
(`docs/research/toothfairy2-caso-b.md`) y no de mirar el resultado.

Por volumen, en el array `(z, y, x)` de ToothFairy2 (z crece hacia caudal, medido en
`segmenta_cbct_toothfairy2.py`):

- **plano oclusal superior**: el vóxel más caudal de los dientes 11–28, la misma definición
  que en el caso clínico (el vértice más caudal de las coronas superiores del escáner);
- **altura de la caja** por encima y por debajo de ese plano, en mm;
- **mandíbula por encima del plano oclusal**, en mm³: en oclusión la rama y la apófisis
  coronoides suben por encima, así que no es cero, y su rango es la referencia;
- **contactos entre arcadas**: caras de vóxel compartidas entre el grupo superior (maxilar y
  11–28) y el inferior (mandíbula y 31–48).

Los volúmenes sin dientes superiores no tienen plano oclusal y se cuentan aparte. Solo se
leen etiquetas: ninguna imagen sale del zip.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))

from segmenta_cbct_toothfairy2 import (  # noqa: E402
    INFERIOR,
    SUPERIOR,
    contactos,
    lee_mha_etiquetas,
    revision_de_codigo,
    sha256_de,
)

RESUMEN_EN = "Measures box height and arch separation over ToothFairy2 training labels."

DIENTES_SUPERIORES = [*range(11, 19), *range(21, 29)]


def espaciado(z: zipfile.ZipFile, nombre: str) -> tuple[float, float, float]:
    with z.open(nombre) as f:
        cab = f.read(2048).split(b"ElementDataFile")[0].decode(errors="replace")
    linea = next(lin for lin in cab.splitlines() if lin.startswith("ElementSpacing"))
    sx, sy, sz = (float(v) for v in linea.split("=")[1].split())
    return sx, sy, sz


def mide(a: np.ndarray, sp: tuple[float, float, float]) -> dict:
    sx, sy, sz = sp
    sup = np.isin(a, DIENTES_SUPERIORES)
    fila: dict = {"shape_zyx": list(a.shape), "spacing_xyz": [sx, sy, sz],
                  "upper_teeth": int(len(set(np.unique(a[sup]).tolist())))}
    if not sup.any():
        return fila
    k_oclusal = int(np.nonzero(sup.any(axis=(1, 2)))[0].max())     # lo más caudal
    fila["top_above_occlusal_mm"] = round(k_oclusal * sz, 2)
    fila["bottom_below_occlusal_mm"] = round((a.shape[0] - 1 - k_oclusal) * sz, 2)
    mand = a[:k_oclusal] == 1
    fila["lower_jawbone_above_occlusal_mm3"] = round(float(mand.sum()) * sx * sy * sz, 1)
    fila["upper_lower_contact_faces"] = contactos(a, SUPERIOR, INFERIOR)
    return fila


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()
    if args.salida.exists():
        raise SystemExit(f"{args.salida} ya existe: una medida no se pisa.")

    casos = {}
    with zipfile.ZipFile(args.zip) as z:
        nombres = sorted(n for n in z.namelist()
                         if n.startswith("Dataset112_ToothFairy2/labelsTr/")
                         and n.endswith(".mha"))
        for i, nombre in enumerate(nombres):
            caso = Path(nombre).stem
            casos[caso] = mide(lee_mha_etiquetas(z, nombre), espaciado(z, nombre))
            if i % 40 == 0:
                print(f"  {i + 1}/{len(nombres)} {caso} {casos[caso]}", flush=True)

    con_plano = {k: v for k, v in casos.items() if "top_above_occlusal_mm" in v}

    def pct(clave: str) -> dict:
        x = np.array([v[clave] for v in con_plano.values()], dtype=np.float64)
        return {p: round(float(np.percentile(x, q)), 2)
                for p, q in (("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75),
                             ("p95", 95), ("max", 100))}

    resumen = {
        "cases": len(casos), "cases_with_upper_teeth": len(con_plano),
        "spacings_xyz": sorted({tuple(v["spacing_xyz"]) for v in casos.values()}),
        **{clave: pct(clave) for clave in ("top_above_occlusal_mm",
                                           "bottom_below_occlusal_mm",
                                           "lower_jawbone_above_occlusal_mm3",
                                           "upper_lower_contact_faces")},
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps({
        "scope": "ToothFairy2 training labels (Dataset112 zip), labels only.",
        "code": revision_de_codigo(),
        "zip_sha256": sha256_de(args.zip),
        "summary": resumen, "cases": casos}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(resumen, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
