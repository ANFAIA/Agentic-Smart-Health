#!/usr/bin/env python
"""Dos preguntas de diseño sobre el registro por diente, medidas en vez de argumentadas.

Las dos salen de la propuesta de diseño del registro por diente, y las dos cambian el
producto según la respuesta.

## Pregunta 1 · «encuentro la matriz para 3 dientes, la promedio y la uso para el resto»

Si funcionase, el artefacto de agente sería **una** matriz en vez de trece, y el trabajo de
segmentar catorce piezas dejaría de hacer falta. Así que merece una medida, no un juicio.

Se prueba de las **dos** formas en que se puede leer la frase, porque una es un
espantapájaros y la otra es la buena:

- **cruda** — promediar las 4×4 tal cual. Es lo que dice la frase, y tiene un problema que
  no es del método sino de las coordenadas: la traslación de una rígida depende del origen,
  así que una rotación pequeña alrededor de un origen lejano produce un desplazamiento
  grande al aplicarla a otra pieza.
- **local** — descomponer cada movimiento en «rotación alrededor del centroide del propio
  diente» + «desplazamiento del centroide», promediar eso y aplicarlo al diente *j*
  alrededor de **su** centroide. Es la lectura física de la misma idea.

Y no se eligen tres dientes a dedo: se prueban **todas** las combinaciones de 3 sobre los
13 y se informa la distribución. Elegir un trío concreto permitiría contar el resultado que
más convenga.

Las rotaciones se promedian proyectando la media a SO(3) por SVD, que es la rotación más
cercana en norma de Frobenius. Promediar matrices sin proyectar no devuelve una rotación.

## Pregunta 2 · la escala, el 7º grado de libertad

Un diente no cambia de tamaño entre dos escaneos, así que una escala distinta de 1 solo
puede ser del escáner. Si es consistente, el producto debería vigilarla.

⚠️ **Tres estimadores se descartaron por medirlos, y conviene que quede escrito** — el
orden en que fallan es informativo:

1. **ICP con escala, sobre el arco completo.** Colapsa: encoger acerca todos los puntos al
   centroide del objetivo y baja la distancia al vecino más próximo *pase lo que pase con la
   escala real*. Medido, A→B da 0,904 y B→A da 1,016; el producto sale **0,913** en vez de
   1. Una medida asimétrica no es una medida. Y las diagonales de las dos mallas son 88,25 y
   88,50 mm, así que un −9,6 % es imposible por construcción.
2. **ICP con escala, por diente.** Menos malo porque arranca ya alineado, pero la desviación
   entre piezas es de ~8.000 ppm y sube a **70.000** al quitar el recorte. Y el sesgo va
   siempre al mismo lado, que es la firma del mismo colapso.
3. **Razón entre distancias de centroides de diente.** No usa proximidad sino etiquetas, así
   que parecía inmune. Pero la razón individual dispersa **±6-8 %**: el centroide depende de
   qué vértices quedaron etiquetados, y las etiquetas son transferidas.

Lo que sí funciona no **optimiza** la escala, la **lee**: con la rígida fijada, un error de
escala `s` desplaza cada punto radialmente en `(s−1)·r`. Eso es una recta de pendiente
`s−1` sobre decenas de miles de puntos emparejados, y no puede colapsar porque no se le
permite mover la rígida.

El **par nulo** —dos escaneos de la misma visita— da el suelo: allí la escala es 1 por
construcción. Y se mide en los dos sentidos: si no cambia de signo, no es una escala.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RESUMEN_EN = "Measures design questions around per-tooth registration and scaling."

RAIZ = Path(__file__).resolve().parent.parent
for _paquete in ("ingestion-agents", "fusion-agents", "core-schemas", "tooth-aggregation"):
    sys.path.insert(0, str(RAIZ / f"packages/{_paquete}/src"))

from fusion_agents.por_diente import (  # noqa: E402
    MIN_PUNTOS,
    TRIM_LOCAL,
    transfiere_etiquetas,
)
from fusion_agents.registration import apply, icp, quaternion_to_matrix  # noqa: E402
from ingestion_agents.mesh_agent import parse_stl  # noqa: E402
from tooth_aggregation import aggregate_teeth  # noqa: E402

CLASES_MANDIBULARES = slice(16, 32)
EMPAREJADO_MM = 0.5
RADIO_MINIMO_MM = 1.0
"""Cerca del centroide la dirección radial es ruido: dividir por un radio ~0 la amplifica."""


def segmenta(malla: dict, logprob: np.ndarray, codigos: np.ndarray) -> np.ndarray:
    """Códigos FDI por vértice, con la arcada enmascarada (decodificación restringida)."""
    V = np.asarray(malla["positions"], dtype=np.float64)
    logp = logprob.astype(np.float64).copy()
    permitidas = np.zeros(logp.shape[1], dtype=bool)
    permitidas[0] = True
    permitidas[CLASES_MANDIBULARES] = True
    logp[:, ~permitidas] = -np.inf
    fdi = np.zeros(len(V), dtype=np.int64)
    for inst in aggregate_teeth(V, logp, codes={i: int(c) for i, c in enumerate(codigos)}):
        if inst.fdi:
            fdi[inst.vertices] = inst.fdi
    return fdi


def _por_patron(directorio: Path, patron: str) -> Path:
    """Localiza un STL por patrón en vez de escribir su nombre.

    🔒 Los nombres que exporta el escáner llevan las **iniciales del paciente**, así que no
    pueden aparecer en un repositorio público: son dato identificativo, igual que el EXIF
    que el `image-agent` descarta. Se resuelven sobre el directorio local, no versionado.
    """
    encontrados = sorted(directorio.glob(patron)) if directorio.is_dir() else []
    return encontrados[0] if len(encontrados) == 1 else directorio / patron


def _rigida(fuente: np.ndarray, objetivo: np.ndarray, trim: float):
    r = icp(fuente, objetivo, trim=trim)
    return quaternion_to_matrix(r.rotation), np.asarray(r.translation), r


def _rms(fuente: np.ndarray, arbol: cKDTree, trim: float = TRIM_LOCAL) -> float:
    d, _ = arbol.query(fuente)
    d = np.sort(d)[: max(3, int(len(d) * trim))]
    return float(np.sqrt((d**2).mean()))


def media_so3(rotaciones: list[np.ndarray]) -> np.ndarray:
    """Media de rotaciones proyectada a SO(3). Sin proyectar no sería una rotación."""
    U, _, Vt = np.linalg.svd(np.mean(rotaciones, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    return R


def firma_radial(X: np.ndarray, Y: np.ndarray, *, trim: float) -> dict[str, float]:
    """Escala leída de su firma radial, con la rígida fijada. `{ppm, error, r2, n}`."""
    rot, trans, r = _rigida(X, Y, trim)
    Xa = apply(rot, trans, X)
    d, idx = cKDTree(Y).query(Xa)
    emp = d < EMPAREJADO_MM
    P, Q = Xa[emp], Y[idx[emp]]
    v = P - P.mean(0)
    radio = np.linalg.norm(v, axis=1)
    ok = radio > RADIO_MINIMO_MM
    v, radio, residuo = v[ok], radio[ok], (Q - P)[ok]
    radial = np.einsum("ij,ij->i", residuo, v / radio[:, None])

    # Regresión SIN término constante: una escala pura no tiene desplazamiento radial fijo.
    pendiente = float((radio @ radial) / (radio @ radio))
    resto = radial - pendiente * radio
    n = len(radio)
    error = float(np.sqrt((resto**2).sum() / (n - 1) / (radio**2).sum()))
    ss = float(((radial - radial.mean()) ** 2).sum())
    return {
        "ppm": pendiente * 1e6,
        "error_ppm": error * 1e6,
        # ⚠️ El error estándar supone residuos independientes y NO lo son: los vértices
        # vecinos están correlacionados. Con n ~ 75.000 sale una significancia enorme que
        # no significa nada. El suelo de verdad es el del par nulo.
        "r2": 1.0 - float((resto**2).sum()) / ss if ss > 0 else float("nan"),
        "n": n,
        "rms_registro_mm": r.rms_efectivo_mm,
        "mm_en_60": pendiente * 60,
    }


def piezas_registradas(Vo, Vd, fdi_d, *, trim_global: float, semilla: int = 0) -> dict:
    """Una rígida por diente con validación cruzada, y lo necesario para promediarlas."""
    rot_g, trans_g, _ = _rigida(Vo, Vd, trim_global)
    Vo_al = apply(rot_g, trans_g, Vo)
    fdi_o = transfiere_etiquetas(Vo_al, Vd, fdi_d)

    rng = np.random.default_rng(semilla)
    fuera: dict[int, dict] = {}
    for c in sorted({int(x) for x in np.unique(fdi_d) if x}):
        src, tgt = Vo_al[fdi_o == c], Vd[fdi_d == c]
        if len(src) < 2 * MIN_PUNTOS or len(tgt) < MIN_PUNTOS:
            continue
        orden = rng.permutation(len(src))
        ajuste, retenida = src[orden[::2]], src[orden[1::2]]
        arbol = cKDTree(tgt)
        R, t, _ = _rigida(ajuste, tgt, TRIM_LOCAL)
        centro = src.mean(0)
        fuera[c] = {
            "R": R, "t": t, "centro": centro, "arbol": arbol, "retenida": retenida,
            # Desplazamiento del centroide: el movimiento local, sin la dependencia
            # del origen que arruina el promedio de las 4×4 crudas.
            "d": R @ centro + t - centro,
            "rms_global": _rms(retenida, arbol),
            "rms_propio": _rms(apply(R, t, retenida), arbol),
        }
    return fuera


def promedio_de_tres(piezas: dict) -> dict[str, dict]:
    """Todas las combinaciones de 3: promediar y medir en los dientes que quedan fuera."""
    codigos = sorted(piezas)
    acumulado: dict[str, list[tuple[float, float, float]]] = {"cruda": [], "local": []}
    for trio in itertools.combinations(codigos, 3):
        R_avg = media_so3([piezas[c]["R"] for c in trio])
        t_avg = np.mean([piezas[c]["t"] for c in trio], axis=0)
        d_avg = np.mean([piezas[c]["d"] for c in trio], axis=0)
        for c in codigos:
            if c in trio:
                continue
            p = piezas[c]
            ret, arbol = p["retenida"], p["arbol"]
            acumulado["cruda"].append(
                (_rms(ret @ R_avg.T + t_avg, arbol), p["rms_propio"], p["rms_global"])
            )
            local = (ret - p["centro"]) @ R_avg.T + p["centro"] + d_avg
            acumulado["local"].append((_rms(local, arbol), p["rms_propio"], p["rms_global"]))

    fuera: dict[str, dict] = {}
    for lectura, datos in acumulado.items():
        prom = np.array([x[0] for x in datos])
        propio = np.array([x[1] for x in datos])
        glob = np.array([x[2] for x in datos])
        fuera[lectura] = {
            "n_evaluaciones": len(datos),
            "promediada_mm": {
                "p10": float(np.percentile(prom, 10)),
                "mediana": float(np.median(prom)),
                "p90": float(np.percentile(prom, 90)),
                "max": float(prom.max()),
            },
            "propia_mediana_mm": float(np.median(propio)),
            "global_mediana_mm": float(np.median(glob)),
            "veces_gana_la_propia": int((propio < prom).sum()),
            "veces_peor_que_no_tocar_nada": int((prom > glob).sum()),
        }
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

    ap.add_argument("--etiquetado", type=Path, default=etiquetado)
    ap.add_argument("--logprob", type=Path,
                    default=Path.home() / "anfaia/fdi/logp_lower_despues.npy")
    ap.add_argument("--codigos", type=Path, default=Path.home() / "anfaia/fdi/codigos.npy")
    ap.add_argument("--par", type=Path, nargs=2, default=[previo, etiquetado])
    ap.add_argument("--control", type=Path, nargs=2, default=[repetido, previo],
                    help="Par de la MISMA visita: el suelo de ruido.")
    ap.add_argument("--trim", type=float, default=0.7, help="Recorte del ICP global.")
    ap.add_argument("--salida", type=Path, default=Path.home() / "anfaia/hipotesis")
    args = ap.parse_args()

    faltan = [p for p in [args.etiquetado, args.logprob, args.codigos, *args.par, *args.control]
              if not p.exists()]
    if faltan:
        print("✗ No encuentro el dato:\n    " + "\n    ".join(str(p) for p in faltan)
              + "\n  Es dato clínico y NO está versionado (ver .gitignore).", file=sys.stderr)
        return 2

    malla_etq = parse_stl(args.etiquetado)
    V_etq = np.asarray(malla_etq["positions"], dtype=np.float64)
    fdi_etq = segmenta(malla_etq, np.load(args.logprob), np.load(args.codigos))
    print(f"{args.etiquetado.name}: {(fdi_etq > 0).sum():,} vértices con FDI")

    informe: dict[str, object] = {"trim_global": args.trim}
    for nombre, (p_o, p_d) in (("par", args.par), ("control", args.control)):
        Vo = np.asarray(parse_stl(p_o)["positions"], dtype=np.float64)
        Vd = np.asarray(parse_stl(p_d)["positions"], dtype=np.float64)
        if p_d.resolve() == args.etiquetado.resolve():
            fdi_d = fdi_etq
        else:
            rot, trans, _ = _rigida(V_etq, Vd, 0.9)
            fdi_d = transfiere_etiquetas(Vd, apply(rot, trans, V_etq), fdi_etq)

        piezas = piezas_registradas(Vo, Vd, fdi_d, trim_global=args.trim)
        print(f"\n{'═' * 78}\n{nombre.upper()}: {p_o.name} → {p_d.name}")
        print(f"  {len(piezas)} dientes utilizables")

        p1 = promedio_de_tres(piezas)
        print(f"\n  PREGUNTA 1 · promediar 3 matrices "
              f"({len(list(itertools.combinations(sorted(piezas), 3)))} tríos)")
        for lectura, r in p1.items():
            m = r["promediada_mm"]
            print(f"    lectura {lectura:>6}: promediada {m['mediana']:.3f} mm "
                  f"(p10 {m['p10']:.3f} · p90 {m['p90']:.3f} · max {m['max']:.3f})")
            print(f"                    propia {r['propia_mediana_mm']:.3f} · "
                  f"global {r['global_mediana_mm']:.3f} · gana la propia "
                  f"{r['veces_gana_la_propia']}/{r['n_evaluaciones']} · PEOR que no tocar "
                  f"nada {r['veces_peor_que_no_tocar_nada']}/{r['n_evaluaciones']}")

        escalas = {
            "ida": firma_radial(Vo, Vd, trim=args.trim),
            "vuelta": firma_radial(Vd, Vo, trim=args.trim),
        }
        print("\n  PREGUNTA 2 · escala por firma radial (rígida fijada)")
        for sentido, e in escalas.items():
            print(f"    {sentido:>7}: {e['ppm']:+8.0f} ppm ±{e['error_ppm']:.0f} · "
                  f"R² {e['r2']:.4f} · {e['mm_en_60']:+.4f} mm en 60 · "
                  f"{e['n']:,} puntos, registro {e['rms_registro_mm']:.3f} mm")
        suma = escalas["ida"]["ppm"] + escalas["vuelta"]["ppm"]
        veredicto = (
            "coherente: cambia de signo"
            if abs(suma) < abs(escalas["ida"]["ppm"])
            else "⚠ NO cambia de signo: entonces no es una escala"
        )
        print(f"    ida + vuelta = {suma:+.0f} ppm  ({veredicto})")
        informe[nombre] = {"promedio_de_tres": p1, "escala": escalas}

    args.salida.mkdir(parents=True, exist_ok=True)
    destino = args.salida / "promedio_y_escala.json"
    destino.write_text(json.dumps(informe, indent=2), encoding="utf-8")
    print(f"\n→ {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
