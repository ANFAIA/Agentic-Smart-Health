#!/usr/bin/env python
"""¿Se puede decir «esta pieza se desplazó X mm»? Referencia leave-one-out y umbral.

Las matrices por diente de `fusion_agents.por_diente` daban un residuo robusto pero un
desplazamiento inservible: se medía contra el registro global del arco, y ese registro se
mueve al tocar el `trim` del ICP, así que **todas** las traslaciones se movían con él.
Medido aquí: la mediana pasa de 0,171 a 0,738 mm solo con cambiar ese hiperparámetro.

Este script valida el arreglo y, sobre todo, **fija el umbral por debajo del cual no se
puede afirmar nada**. Son dos preguntas distintas y las dos hacen falta.

## Decisión 1 · la referencia se ajusta con los DEMÁS dientes

Para medir el diente *X* se reajusta el marco con todos los dientes menos X
(`desplazamientos_relativos`). Quita dos contaminaciones del registro global:

- **X entraba en su propia referencia.** Si X se mueve, arrastra el marco contra el que se
  le mide y su desplazamiento sale infravalorado y repartido entre los demás.
- **La encía entraba en la referencia.** Y la encía cambia de forma entre dos momentos
  —inflamación, retracción, una higiene— sin que ningún diente se haya movido.

Lo que se mide pasa a ser desplazamiento **relativo dentro del arco**. No es una
limitación que se acepta a regañadientes: en un escaneo intraoral **no existe** marco
absoluto, porque el escáner no ve ninguna estructura fija.

## Decisión 2 · el control nulo NO es el par pre/post

Comparar `PREVIO` con `POST HIGIENE` parece el control obvio —una higiene no mueve
dientes— pero **sí quita cálculo**, así que la superficie cambia de verdad y el ICP lo lee
como desplazamiento. El control limpio son los **dos escaneos independientes de la misma
visita**, donde no cambió ni la biología ni los depósitos.

Sin ese control cualquier cifra de desplazamiento es indistinguible de la
reproducibilidad del escáner, y no habría umbral que citar.

## Lo que sale

    referencia    global : mediana 0,171 → 0,738 mm al cambiar el trim (4,3×)
    referencia  relativo : mediana 0,158 → 0,182 mm                    (1,2×)
    peor diente          : Δ 0,696 mm  →  Δ 0,107 mm

Y el control nulo da la MISMA distribución que el par real (p90 0,349 frente a 0,342 mm),
así que en este paciente **ningún diente se movió de forma detectable** — lo esperable en
un par pre/post higiene, o sea el test nulo pasa. Umbral de detección: **~0,4 mm**.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
for _paquete in ("ingestion-agents", "fusion-agents", "core-schemas", "tooth-aggregation"):
    sys.path.insert(0, str(RAIZ / f"packages/{_paquete}/src"))

from fusion_agents.por_diente import (  # noqa: E402
    desplazamientos_relativos,
    registra_dientes,
    transfiere_etiquetas,
)
from fusion_agents.registration import apply, icp, quaternion_to_matrix  # noqa: E402
from ingestion_agents.mesh_agent import parse_stl, vertex_normals  # noqa: E402
from tooth_aggregation import aggregate_teeth  # noqa: E402

TRIMS = (0.7, 1.0)
"""Los dos valores entre los que se compara. Cualquier patrón que no sobreviva a los dos
es del ajuste, no del paciente — está medido que el signo de una correlación se invierte.
"""

CLASES_MANDIBULARES = slice(16, 32)
"""Índices de clase de 31-38 y 41-48 en la tabla `codigos.npy`.

Enmascarar el maxilar es **decodificación restringida**: la arcada no es una hipótesis a
resolver, viene en el fichero. Sin esto un molar inferior sale etiquetado como 25.
"""

PERCENTIL_RUIDO = 90


def segmenta(malla: dict, logprob: np.ndarray, codigos: np.ndarray) -> np.ndarray:
    """Códigos FDI por vértice, restringidos a la arcada mandibular."""
    V = np.asarray(malla["positions"], dtype=np.float64)
    logp = logprob.astype(np.float64).copy()
    permitidas = np.zeros(logp.shape[1], dtype=bool)
    permitidas[0] = True                      # encía
    permitidas[CLASES_MANDIBULARES] = True
    logp[:, ~permitidas] = -np.inf

    fdi = np.zeros(len(V), dtype=np.int64)
    tabla = {i: int(c) for i, c in enumerate(codigos)}
    for inst in aggregate_teeth(V, logp, codes=tabla):
        if inst.fdi:
            fdi[inst.vertices] = inst.fdi
    return fdi


def _por_patron(directorio: Path, patron: str) -> Path:
    """Localiza un STL por patrón en vez de escribir su nombre.

    🔒 Los nombres que exporta el escáner llevan las **iniciales del paciente** («I.F.S.
    POST HIGIENE LowerJawScan.stl»), así que no pueden aparecer en un repositorio público:
    son dato identificativo, igual que el EXIF que el `image-agent` descarta. Se resuelven
    en tiempo de ejecución sobre el directorio local, que no está versionado.

    Si no hay exactamente una coincidencia devuelve la ruta literal, para que el chequeo de
    existencia de `main` dé un mensaje claro en vez de fallar aquí con un `IndexError`.
    """
    encontrados = sorted(directorio.glob(patron)) if directorio.is_dir() else []
    return encontrados[0] if len(encontrados) == 1 else directorio / patron


def _rigida(fuente: np.ndarray, objetivo: np.ndarray, trim: float):
    r = icp(fuente, objetivo, trim=trim)
    return quaternion_to_matrix(r.rotation), np.asarray(r.translation), r


def compara_referencias(Vo, Vd, fdi_d, Nd) -> dict[str, dict[float, dict[int, float]]]:
    """Traslación por diente con las dos referencias, a los dos `trim`."""
    fuera: dict[str, dict[float, dict[int, float]]] = {"global": {}, "relativo": {}}
    for trim in TRIMS:
        rot, trans, r = _rigida(Vo, Vd, trim)
        print(f"\n─── trim {trim} · registro global {r.rms_efectivo_mm:.3f} mm")
        viejo = registra_dientes(Vo, Vd, fdi_d, Nd, rot_global=rot, trans_global=trans)
        nuevo = desplazamientos_relativos(
            Vo, Vd, fdi_d, Nd, rot_global=rot, trans_global=trans, trim_referencia=trim
        )
        fuera["global"][trim] = {d.fdi: d.traslacion_mm for d in viejo}
        fuera["relativo"][trim] = {d.fdi: d.traslacion_mm for d in nuevo}

        por_fdi = {d.fdi: d for d in viejo}
        print(f"{'FDI':>4} {'n':>7} {'GLOBAL':>8} {'RELATIVA':>9} {'rms ref':>8} "
              f"{'rms dte':>8} {'rot°':>6} {'cond':>7}")
        for d in nuevo:
            v = por_fdi.get(d.fdi)
            print(f"{d.fdi:>4} {d.n_diente:>7,} "
                  f"{(v.traslacion_mm if v else float('nan')):>8.3f} "
                  f"{d.traslacion_mm:>9.3f} {d.rms_referencia_mm:>8.3f} "
                  f"{d.rms_diente_mm:>8.3f} {d.rotacion_deg:>6.2f} {d.condicion:>7.0f}")
    return fuera


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    raiz = Path.home() / "anfaia" / "histora"
    visita = raiz / "CASE-EB5070_files"
    etiquetado = _por_patron(raiz, "*POST HIGIENE LowerJawScan.stl")
    previo = visita / "PREVIO LowerJawScan.stl"
    repetido = _por_patron(visita, "* ESCANEADO PREVIO LowerJawScan.stl")

    ap.add_argument("--etiquetado", type=Path, default=etiquetado,
                    help="STL sobre el que se segmentó (el de mejor calidad).")
    ap.add_argument("--logprob", type=Path,
                    default=Path.home() / "anfaia/fdi/logp_lower_despues.npy",
                    help="Log-probabilidades densas del segmentador, (N, C).")
    ap.add_argument("--codigos", type=Path,
                    default=Path.home() / "anfaia/fdi/codigos.npy",
                    help="Tabla índice de clase → código FDI.")
    ap.add_argument("--par", type=Path, nargs=2, default=[previo, etiquetado],
                    help="Par origen/destino sobre el que medir el desplazamiento.")
    ap.add_argument("--control", type=Path, nargs=2, default=[repetido, previo],
                    help="Par de la MISMA visita: el control nulo que fija el suelo.")
    ap.add_argument("--salida", type=Path, default=Path.home() / "anfaia/relativo",
                    help="Dónde escribir el JSON.")
    args = ap.parse_args()

    faltan = [
        p for p in [args.etiquetado, args.logprob, args.codigos, *args.par, *args.control]
        if not p.exists()
    ]
    if faltan:
        print("✗ No encuentro el dato:\n    " + "\n    ".join(str(p) for p in faltan)
              + "\n  Es dato clínico y NO está versionado (ver .gitignore).", file=sys.stderr)
        return 2

    malla_etq = parse_stl(args.etiquetado)
    V_etq = np.asarray(malla_etq["positions"], dtype=np.float64)
    fdi_etq = segmenta(malla_etq, np.load(args.logprob), np.load(args.codigos))
    print(f"{args.etiquetado.name}: {(fdi_etq > 0).sum():,} vertices con FDI en "
          f"{len(set(fdi_etq[fdi_etq > 0].tolist()))} dientes")

    resultados: dict[str, object] = {}
    for nombre, (p_o, p_d) in (("par", args.par), ("control", args.control)):
        Vo = np.asarray(parse_stl(p_o)["positions"], dtype=np.float64)
        malla_d = parse_stl(p_d)
        Vd = np.asarray(malla_d["positions"], dtype=np.float64)
        Nd = vertex_normals(Vd, malla_d["faces"])
        if p_d.resolve() == args.etiquetado.resolve():
            fdi_d = fdi_etq
        else:
            # Se etiqueta UNA vez y se transfiere. Segmentar cada escaneo por su cuenta da
            # a un mismo diente extensiones distintas y el ICP alinea superficies que no
            # se corresponden: medido, rotaciones de hasta 39°.
            rot, trans, _ = _rigida(V_etq, Vd, 0.9)
            fdi_d = transfiere_etiquetas(Vd, apply(rot, trans, V_etq), fdi_etq)

        print(f"\n{'═' * 74}\n{nombre.upper()}: {p_o.name}  →  {p_d.name}")
        print(f"  {len(Vo):,} → {len(Vd):,} vertices · destino con "
              f"{(fdi_d > 0).sum():,} etiquetados")
        resultados[nombre] = compara_referencias(Vo, Vd, fdi_d, Nd)

    print(f"\n{'═' * 74}\nSENSIBILIDAD AL TRIM — decide si la referencia relativa sirve")
    a, b = TRIMS
    for cual in ("global", "relativo"):
        x, y = resultados["par"][cual][a], resultados["par"][cual][b]
        comunes = sorted(set(x) & set(y))
        mx, my = np.median([x[f] for f in comunes]), np.median([y[f] for f in comunes])
        peor = max(abs(x[f] - y[f]) for f in comunes)
        print(f"  referencia {cual:>9}: mediana {mx:.3f} → {my:.3f} mm "
              f"({max(mx, my) / max(min(mx, my), 1e-9):.1f}×) · peor diente Δ {peor:.3f} mm")

    print(f"\n{'═' * 74}\nFALSOS POSITIVOS SOBRE EL CONTROL NULO — la prueba más dura")
    print("  Dos escaneos de la misma visita. Todo desplazamiento que se informe aquí es\n"
          "  falso por construcción, así que el máximo es el error del método.")
    for cual in ("global", "relativo"):
        for trim in TRIMS:
            v = np.array(list(resultados["control"][cual][trim].values()))
            print(f"    referencia {cual:>9} · trim {trim}: mediana {np.median(v):.3f} mm · "
                  f"MAXIMO {v.max():.3f} mm")

    print(f"\n{'═' * 74}\nUMBRAL DE DETECCION — el control nulo es el que manda")
    umbrales = {}
    for trim in TRIMS:
        ruido = np.array(list(resultados["control"]["relativo"][trim].values()))
        real = np.array(list(resultados["par"]["relativo"][trim].values()))
        umbral = float(np.percentile(ruido, PERCENTIL_RUIDO))
        umbrales[str(trim)] = umbral
        print(f"  trim {trim}: ruido p{PERCENTIL_RUIDO} {umbral:.3f} mm · par real p"
              f"{PERCENTIL_RUIDO} {np.percentile(real, PERCENTIL_RUIDO):.3f} mm · "
              f"dientes por encima del ruido: {(real > umbral).sum()}/{len(real)}")
    print("\n  El control nulo son DOS escaneos de la misma visita: no cambió nada, así\n"
          "  que todo lo que informa es reproducibilidad. Por debajo de ese umbral no se\n"
          "  puede escribir «esta pieza se desplazó», por mucho que la cifra sea estable.")

    args.salida.mkdir(parents=True, exist_ok=True)
    destino = args.salida / "desplazamiento_relativo.json"
    destino.write_text(json.dumps({
        "trims": list(TRIMS),
        "umbral_deteccion_mm": umbrales,
        "percentil_ruido": PERCENTIL_RUIDO,
        "resultados": {
            k: {cual: {str(t): {str(f): v for f, v in d.items()} for t, d in por.items()}
                for cual, por in r.items()}
            for k, r in resultados.items()
        },
    }, indent=2), encoding="utf-8")
    print(f"\n→ {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
