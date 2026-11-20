#!/usr/bin/env python
"""registro_entre_cbct.py — registra un CBCT sobre otro del mismo paciente (E3).

    uv run python scripts/registro_entre_cbct.py \\
        --cbct1 DIR_SERIE_1 --cbct2 DIR_SERIE_2 --arcada upper --salida DIR

Es la pieza secundaria de la evaluación de `scripts/precision_registro.py` (E3, plan §5):
allí P4 varía el ESCANEO contra un mismo CBCT; E3 varía el CBCT contra un mismo escaneo, y
para eso hace falta la transformación CBCT₂→CBCT₁, que no sale de ningún escáner.

**Qué hace.** La misma maquinaria punto a plano del registro anatómico
(`fusion_agents.anatomico`), con el escáner sustituido por el otro CBCT:

1. Cada CBCT da su marco (`marco_cbct`: ejes LPS con el superior corregido por el plano
   del esmalte) y su contorno de diente a `UMBRAL_CONTORNO` en la losa inclinada de la
   arcada (`contornos_losa`). Nada más se lee del volumen.
2. **Arranque**: la rotación que lleva el marco de CBCT₂ sobre el de CBCT₁
   (`por_cortes.pose_inicial`), con la traslación que hace coincidir el centroide de los
   contornos en planta y los planos oclusales en altura. Un solo arranque, sin azar.
3. **Ajuste**: ICP punto a plano recortado (`TRIM`) del contorno de CBCT₂ al de CBCT₁, con
   las normales del DESTINO, hasta que el paso mueve menos de `TOL_MM`.
4. **Dejar fuera cada diente** y reajustar desde la solución: la navaja de esos reajustes
   es la condición de validez de E3 (dientes que se han movido entre las dos adquisiciones
   la disparan). Si quien llama da la asignación de diente de cada punto de CBCT₁
   (`precision_registro.py` la transfiere desde el escáner registrado), se deja fuera un
   diente; si no, un **sextante** por posición angular alrededor del centroide del
   contorno en planta, que es una aproximación gruesa y así se declara.

⚠️ **Las normales, y por qué no son las del plan tal cual.** `contornos_con_normales` da la
normal 2-D en el plano del corte. El plan proponía elevarla a 3-D con z = 0. Con TODAS las
normales del destino horizontales, la columna de la traslación vertical de JᵀWJ es cero:
`paso_punto_plano` (mínimos cuadrados de norma mínima) no mueve nunca t_z, la altura queda
fijada por el arranque —es decir, por los dos planos del esmalte— y la navaja en altura
sale cero. Eso no es precisión: es una dirección que el ajuste no ve presentada como si
estuviera medida. Por defecto (`NORMALES = "pca"`) la normal se estima en 3-D por PCA de
los `K_NORMAL` vecinos del contorno, que abarcan varios cortes y ven la pendiente de la
pared; el signo se toma de la normal 2-D. Donde los vecinos caen en un solo corte, se usa
la 2-D elevada con z = 0 y se cuenta. `--normales plano` reproduce la aproximación del plan
para documentar la diferencia; el condicionamiento de la traslación va siempre al informe.

**Lo que no hace.**

- No comprueba que los dos CBCT sean del mismo paciente ni de la misma visita: eso lo
  dice `scripts/inventario_series_cbct.py` y lo juzga una persona.
- No es un TRE: un sesgo común a los dos contornos (el isonivel, el medio paso del borde)
  se cancela.
- Es rígido: si los dientes se han movido entre las adquisiciones, ninguna pose encaja
  todas las piezas, y eso es lo que mide la navaja, no lo que corrige.
- El contorno de CBCT₂ incluye lo que su losa ve (raíz incluida); el de CBCT₁ igual. Si
  los campos de visión difieren mucho, la cobertura (`coverage`) lo dice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from fusion_agents.anatomico import (  # noqa: E402
    ANGULO_ANATOMICO_MAX,
    CORRESPONDENCIAS_MINIMAS,
    ITER_FINO,
    TOL_MM,
    TOPE_PASO,
    Plano,
    angulo_grados,
    contornos_losa,
    huella_constantes,
    marco_cbct,
)
from fusion_agents.por_cortes import (  # noqa: E402
    MINIMO_CONTORNO,
    TRIM,
    UMBRAL_CONTORNO,
    pose_inicial,
)
from fusion_agents.registration import (  # noqa: E402
    DEFAULT_OVERLAP_MM,
    informacion_6dof,
    paso_punto_plano,
)

RESUMEN_EN = "Registers one CBCT onto another of the same patient with point-to-plane ICP."

MAX_PRIMITIVAS = 1_500_000
"""El tope de `caso_completo.py` y de `precision_registro.py`."""

NORMALES = "pca"
"""Cómo se lleva a 3-D la normal del contorno destino: `pca` (por defecto) o `plano` (la
2-D con z = 0, la del plan, que deja la altura sin medir; ver el módulo)."""

K_NORMAL = 16
"""Vecinos 3-D para la PCA de la normal: con cortes de 0,3–0,6 mm y puntos de borde cada
~0,3 mm, 16 vecinos abarcan dos o tres cortes."""

SEPARACION_Z_MIN_MM = 0.05
"""Si los vecinos no se separan al menos esto en z, están en un corte y la PCA no ve la
pendiente: se usa la normal 2-D."""

ANGULO_SEXTANTE_GRADOS = 35.0
"""Sin asignación de diente: un punto es del sextante anterior si su dirección en planta
desde el centroide del contorno está a menos de esto del eje anterior (≈ los caninos)."""

MINIMO_GRUPO = 200
"""Un diente o sextante con menos puntos de destino que esto no se deja fuera: su
reajuste no se distinguiría del de no quitar nada."""

TOL_COBERTURA_MM = DEFAULT_OVERLAP_MM


# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Variante:
    """El contorno de un CBCT con la losa referida a UNA de las anclas de su plano."""

    ancla: float            # altura sobre el plano del CBCT, mm
    plano: Plano            # el plano llevado a esa altura, con una sola ancla
    P: np.ndarray           # contorno a UMBRAL_CONTORNO en la losa de la arcada
    N2: np.ndarray          # su normal 2-D hacia fuera


@dataclass(frozen=True, eq=False)
class CbctPreparado:
    """Lo que el registro entre CBCT lee de un volumen: marco, plano y contornos."""

    base: np.ndarray        # filas: derecha, anterior, superior, en mm del DICOM
    plano: Plano
    informe_marco: dict
    variantes: tuple[Variante, ...]


def prepara(centros: np.ndarray, hu: np.ndarray, arcada: str) -> CbctPreparado:
    """Marco y contornos de un CBCT: uno por ancla del plano (`Plano.anclas`).

    `marco_cbct` deja la altura del contacto entre arcadas sin decidir cuando el esmalte
    tiene dos picos (moda y valle), y `prepara_anatomico` lo resuelve arrancando desde las
    dos y puntuando contra el escáner. Aquí no hay escáner que desempate, así que cada
    ancla da su propia losa —referida a ESA altura, sin el ensanche que `contornos_losa`
    aplica para cubrir las dos— y `registra_entre_cbct` prueba las combinaciones.
    """
    centros = np.asarray(centros, dtype=np.float64)
    base, plano, informe = marco_cbct(centros, hu)
    variantes = []
    for ancla in plano.anclas:
        p = Plano(plano.punto + ancla * plano.normal, plano.normal, (0.0,))
        P, N2 = contornos_losa(centros, hu, UMBRAL_CONTORNO, p, arcada)
        if len(P) >= MINIMO_CONTORNO:
            variantes.append(Variante(ancla=float(ancla), plano=p, P=P, N2=N2))
    if not variantes:
        raise ValueError("ninguna losa del CBCT tiene contorno suficiente")
    return CbctPreparado(base=base, plano=plano, informe_marco=informe,
                         variantes=tuple(variantes))


def normales_3d(P: np.ndarray, N2: np.ndarray, modo: str = NORMALES) -> tuple[np.ndarray, dict]:
    """`(N3, informe)`: la normal 3-D de cada punto de contorno. Ver el módulo."""
    P = np.asarray(P, dtype=np.float64)
    elevada = np.c_[N2, np.zeros(len(N2))]
    elevada /= np.maximum(np.linalg.norm(elevada, axis=1, keepdims=True), 1e-300)
    if modo == "plano":
        return elevada, {"mode": "plano", "fraction_2d": 1.0}
    if modo != "pca":
        raise ValueError(f"modo de normales desconocido: {modo!r} (pca o plano)")
    _, j = cKDTree(P).query(P, k=min(K_NORMAL, len(P)))
    vecinos = P[j]
    centrado = vecinos - vecinos.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centrado, centrado)
    _, vec = np.linalg.eigh(cov)
    n = vec[:, :, 0]
    n *= np.where(np.einsum("ij,ij->i", n[:, :2], N2) < 0, -1.0, 1.0)[:, None]
    plano = np.ptp(vecinos[:, :, 2], axis=1) < SEPARACION_Z_MIN_MM
    n[plano] = elevada[plano]
    return n, {"mode": "pca", "k": K_NORMAL, "fraction_2d": round(float(plano.mean()), 4)}


def grupos_por_sextante(P: np.ndarray, base: np.ndarray) -> np.ndarray:
    """1 = lado −derecha (izquierda del paciente), 2 = anterior, 3 = lado +derecha."""
    centro = P.mean(axis=0)
    x = (P - centro) @ base[0]
    y = (P - centro) @ base[1]
    angulo = np.degrees(np.arctan2(x, y))
    g = np.full(len(P), 2, dtype=np.int64)
    g[angulo > ANGULO_SEXTANTE_GRADOS] = 3
    g[angulo < -ANGULO_SEXTANTE_GRADOS] = 1
    return g


def pose_de_partida(
    c1: CbctPreparado, v1: Variante, c2: CbctPreparado, v2: Variante
) -> tuple[np.ndarray, np.ndarray]:
    """Marco con marco; centroides del contorno en planta y las dos anclas en altura."""
    R0 = pose_inicial(c2.base, c1.base)
    sup = c1.base[2]
    planta = v1.P.mean(axis=0) - R0 @ v2.P.mean(axis=0)
    planta = planta - (planta @ sup) * sup
    vertical = float((v1.plano.punto - R0 @ v2.plano.punto) @ sup)
    return R0, planta + vertical * sup


def cobertura(q: np.ndarray, arbol_dst: cKDTree, dst: np.ndarray) -> tuple[float, float, float]:
    """`(C, C_origen, C_destino)`: fracción a ≤ `TOL_COBERTURA_MM` en cada sentido; C, el
    mínimo. Es lo que elige la combinación de anclas, no el residuo que se ajusta."""
    a = float(np.mean(arbol_dst.query(q, workers=-1)[0] <= TOL_COBERTURA_MM))
    b = float(np.mean(cKDTree(q).query(dst, workers=-1)[0] <= TOL_COBERTURA_MM))
    return min(a, b), a, b


def ajusta(
    R: np.ndarray, t: np.ndarray, src: np.ndarray, arbol: cKDTree, dst: np.ndarray,
    normales: np.ndarray, *, permitido: np.ndarray | None = None, iters: int = ITER_FINO,
    tol: float = TOL_MM,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """ICP punto a plano recortado de `src` (CBCT₂) a `dst` (CBCT₁), desde `(R, t)`.

    `permitido` marca los puntos de destino que pueden emparejarse (dejar un diente fuera).
    """
    convergido, hechas = False, 0
    for k in range(1, iters + 1):
        hechas = k
        q = src @ R.T + t
        d, j = arbol.query(q, workers=-1)
        ok = np.ones(len(q), dtype=bool) if permitido is None else permitido[j]
        idx = np.flatnonzero(ok)
        if len(idx) < CORRESPONDENCIAS_MINIMAS:
            break
        usa = idx[d[idx] <= np.quantile(d[idx], TRIM)]
        dr, dt = paso_punto_plano(q[usa], dst[j[usa]], normales[j[usa]], tope=TOPE_PASO)
        R, t = dr @ R, dr @ t + dt
        movido = float(np.max(np.linalg.norm(q[usa] @ dr.T + dt - q[usa], axis=1)))
        if movido < tol:
            convergido = True
            break
    return R, t, hechas, convergido


def _condicionamiento(q: np.ndarray, dst: np.ndarray, normales: np.ndarray,
                      base: np.ndarray) -> dict:
    valores, vectores, sigma = informacion_6dof(q, dst, normales)
    m = (vectores * valores) @ vectores.T
    tras = m[3:, 3:] - m[3:, :3] @ np.linalg.pinv(m[:3, :3]) @ m[:3, 3:]
    lt, vt = np.linalg.eigh(tras)
    peor = base @ vt[:, 0]
    return {
        "translation_eigenvalues": [float(x) for x in lt],
        "worst_direction_anatomical": dict(zip(("right", "anterior", "superior"),
                                               (round(float(c), 3) for c in peor),
                                               strict=True)),
        "worst_axis": ["right", "anterior", "superior"][int(np.argmax(np.abs(peor)))],
        "residual_sigma_mm": round(float(sigma), 4),
        "formal_sigma_worst_mm": (round(float(sigma / math.sqrt(lt[0])), 5)
                                  if lt[0] > 1e-12 else None),
        "formal_is_optimistic": True,
    }


def registra_entre_cbct(
    c1: CbctPreparado, c2: CbctPreparado, *,
    asigna_dientes: Callable[[np.ndarray], np.ndarray] | None = None,
    puntos_eval: np.ndarray | None = None, normales: str = NORMALES,
) -> dict:
    """La transformación CBCT₂→CBCT₁ (mm DICOM a mm DICOM) y su informe.

    Se ajusta desde cada combinación de anclas (una losa de cada CBCT, ver `prepara`) y se
    queda la de mayor cobertura simétrica C; empates de menos de 1e-3, la primera en orden.
    `asigna_dientes(P1)` da el código de diente de cada punto del contorno de CBCT₁
    (0 = sin asignar); sin él se deja fuera un sextante cada vez. `puntos_eval`: dónde se
    mide la navaja, en mm de CBCT₁ (por defecto, el contorno elegido de CBCT₁).
    """
    candidatas = []
    for v1 in c1.variantes:
        N1, info_normales = normales_3d(v1.P, v1.N2, normales)
        arbol = cKDTree(v1.P)
        for v2 in c2.variantes:
            R0, t0 = pose_de_partida(c1, v1, c2, v2)
            R, t, iters, conv = ajusta(R0, t0, v2.P, arbol, v1.P, N1)
            C, Ca, Cb = cobertura(v2.P @ R.T + t, arbol, v1.P)
            candidatas.append({"v1": v1, "v2": v2, "N1": N1, "arbol": arbol,
                               "normales": info_normales, "R0": R0, "t0": t0, "R": R,
                               "t": t, "iterations": iters, "converged": conv, "C": C,
                               "C_source": Ca, "C_target": Cb})
    mejor_c = max(c["C"] for c in candidatas)
    elegida = next(c for c in candidatas if mejor_c - c["C"] < 1e-3)
    v1, v2, N1, arbol = elegida["v1"], elegida["v2"], elegida["N1"], elegida["arbol"]
    R, t = elegida["R"], elegida["t"]

    q = v2.P @ R.T + t
    d, j = arbol.query(q, workers=-1)
    r = np.einsum("ij,ij->i", q - v1.P[j], N1[j])
    usa = d <= np.quantile(d, TRIM)
    angulos = [angulo_grados(R @ c2.base[i], c1.base[i]) for i in range(3)]
    veto = None
    if max(angulos) > ANGULO_ANATOMICO_MAX:
        veto = ("los ejes de CBCT₂ llevados a CBCT₁ quedan a "
                + "/".join(f"{a:.0f}" for a in angulos)
                + f"° de los de CBCT₁ (máximo {ANGULO_ANATOMICO_MAX:.0f}°)")

    # Dejar fuera un diente (o un sextante) cada vez, desde la solución.
    if asigna_dientes is not None:
        grupos = np.asarray(asigna_dientes(v1.P), dtype=np.int64)
        modo = "teeth"
    else:
        grupos = grupos_por_sextante(v1.P, c1.base)
        modo = "sextants_by_angle"
    X = v1.P if puntos_eval is None else np.asarray(puntos_eval, dtype=np.float64)
    # Δ_k es lo que cada reajuste mueve un punto de CBCT₁: X se lleva a CBCT₂ con la
    # inversa de la solución y vuelve con cada reajuste.
    Xen2 = (X - t) @ R
    base_x = Xen2 @ R.T + t
    deltas, por_grupo = [], {}
    for g in sorted(int(x) for x in np.unique(grupos) if x > 0):
        fuera = grupos == g
        if int(fuera.sum()) < MINIMO_GRUPO:
            por_grupo[str(g)] = {"n_points": int(fuera.sum()), "skipped": "pocos puntos"}
            continue
        Rk, tk, _, ck = ajusta(R, t, v2.P, arbol, v1.P, N1, permitido=~fuera)
        delta = (Xen2 @ Rk.T + tk) - base_x
        deltas.append(delta)
        norma = np.linalg.norm(delta, axis=1)
        por_grupo[str(g)] = {"n_points": int(fuera.sum()), "converged": ck,
                             "median_mm": round(float(np.median(norma)), 4),
                             "max_mm": round(float(norma.max()), 4)}
    if len(deltas) >= 2:
        D = np.stack(deltas)
        dm = D - D.mean(axis=0)
        n = len(deltas)
        sigma = np.sqrt((n - 1) / n * np.sum(dm**2, axis=(0, 2)))
        sigma_v = np.sqrt((n - 1) / n * np.sum((dm @ c1.base[2]) ** 2, axis=0))
        navaja = {"n_groups": n, "sigma_median_mm": round(float(np.median(sigma)), 4),
                  "sigma_p95_mm": round(float(np.percentile(sigma, 95)), 4),
                  "sigma_vertical_p95_mm": round(float(np.percentile(sigma_v, 95)), 4)}
    else:
        navaja = {"n_groups": len(deltas), "sigma_median_mm": None, "sigma_p95_mm": None,
                  "sigma_vertical_p95_mm": None}

    return {
        "R": R, "t": t,
        "informe": {
            "method": "scripts/registro_entre_cbct.py",
            "constants_sha256": huella_constantes(),
            "normals": elegida["normales"],
            "anchor_combinations": [
                {"anchor_cbct1_mm": c["v1"].ancla, "anchor_cbct2_mm": c["v2"].ancla,
                 "C": round(c["C"], 4), "C_source": round(c["C_source"], 4),
                 "C_target": round(c["C_target"], 4), "iterations": c["iterations"],
                 "converged": c["converged"], "chosen": c is elegida}
                for c in candidatas
            ],
            "start": {"rotation": elegida["R0"].tolist(),
                      "translation_mm": elegida["t0"].tolist()},
            "iterations": elegida["iterations"],
            "converged": elegida["converged"],
            "n_points": {"cbct1": int(len(v1.P)), "cbct2": int(len(v2.P))},
            "rms_point_plane_mm": round(float(np.sqrt(np.mean(r**2))), 4),
            "rms_point_plane_trimmed_mm": round(float(np.sqrt(np.mean(r[usa] ** 2))), 4),
            "coverage": round(elegida["C"], 4),
            "axis_angles_deg": [round(a, 3) for a in angulos],
            "veto": veto,
            "conditioning": _condicionamiento(q[usa], v1.P[j[usa]], N1[j[usa]], c1.base),
            "leave_one_out": {"mode": modo, "groups": por_grupo, "jackknife": navaja},
            "cbct1_frame": c1.informe_marco,
            "cbct2_frame": c2.informe_marco,
            "limitations": [
                "rígido: dientes movidos entre adquisiciones no se corrigen, se miden en la "
                "navaja",
                "un arranque por combinación de anclas, marco con marco; la combinación se "
                "elige por cobertura, sin búsqueda de cuencas",
                "precisión entre dos contornos del mismo isonivel, no un TRE",
                *([] if modo == "teeth" else [
                    "sin asignación de diente: se deja fuera un sextante por ángulo en "
                    f"planta (±{ANGULO_SEXTANTE_GRADOS:.0f}° del anterior), no por pieza"]),
                *([] if normales == "pca" else [
                    "normales 2-D con z = 0: la traslación vertical no está restringida "
                    "por el ajuste y queda la del arranque"]),
            ],
        },
    }


# --------------------------------------------------------------------------- #
def ingiere_cbct(pipe: Any, ruta: Path, nombre: str) -> tuple[dict, np.ndarray]:
    """`(campo, hu)` de un CBCT, cargados como `IngestionPipeline.prepara_registro`."""
    from agent_orchestrator import CaseInput
    from export_agents import densidad_a_hu

    r = pipe.run(CaseInput(acquisition_id=nombre, cbct=ruta, mesh=None, reports=[],
                           images=[]))
    if r.snapshot is None or not r.snapshot.gaussian_field_ref:
        raise SystemExit(f"{nombre}: la ingesta del CBCT no produjo campo.")
    campo = pipe.store.load(r.snapshot.gaussian_field_ref)
    if "hu_range" not in campo or "density" not in campo:
        raise SystemExit(f"{nombre}: el campo no trae HU.")
    return campo, densidad_a_hu(campo["density"], campo["hu_range"])


def centros_de(campo: dict) -> np.ndarray:
    c = np.asarray(campo["centers"], dtype=np.float64)
    return c + np.asarray(campo["origin"], dtype=np.float64) if "origin" in campo else c


def _a_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _a_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_a_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _a_json(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cbct1", type=Path, required=True, help="Directorio de la serie destino.")
    ap.add_argument("--cbct2", type=Path, required=True, help="Directorio de la serie origen.")
    ap.add_argument("--arcada", choices=("upper", "lower"), required=True)
    ap.add_argument("--normales", choices=("pca", "plano"), default=NORMALES)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    from agent_orchestrator import IngestionPipeline
    from caso_completo import entorno_de_software, revision_de_codigo
    from ingestion_agents import ArtifactStore

    if args.salida.exists():
        raise SystemExit(f"{args.salida} ya existe: una ejecución no se pisa.")
    args.salida.mkdir(parents=True)
    registro = {"schema": "histora-run-record/1.0",
                "argv": [a if "/" not in a else "<ruta local>" for a in sys.argv],
                "code": revision_de_codigo(), "environment": entorno_de_software(),
                "constants_sha256": huella_constantes(),
                "started_at": datetime.now(UTC).isoformat()}
    (args.salida / "run-record.json").write_text(json.dumps(registro, indent=2) + "\n")

    pipe = IngestionPipeline(ArtifactStore(args.salida / "artifacts"),
                             quarantine_dir=args.salida / "quarantine",
                             cbct_recorte_dental=True, cbct_max_primitivas=MAX_PRIMITIVAS,
                             segmenter_factory=None)
    campo1, hu1 = ingiere_cbct(pipe, args.cbct1, "cbct1")
    campo2, hu2 = ingiere_cbct(pipe, args.cbct2, "cbct2")
    c1 = prepara(centros_de(campo1), hu1, args.arcada)
    c2 = prepara(centros_de(campo2), hu2, args.arcada)
    salida = registra_entre_cbct(c1, c2, normales=args.normales)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = salida["R"], salida["t"]
    resultado = {"transform_cbct2_to_cbct1_4x4": m.tolist(), **salida["informe"]}
    texto = json.dumps(_a_json(resultado), indent=2, ensure_ascii=False) + "\n"
    (args.salida / "resultado.json").write_text(texto)
    registro["finished_at"] = datetime.now(UTC).isoformat()
    registro["outputs"] = {"resultado.json": hashlib.sha256(texto.encode()).hexdigest()}
    (args.salida / "run-record.json").write_text(json.dumps(registro, indent=2) + "\n")

    inf = salida["informe"]
    print(f"· CBCT₂→CBCT₁: {inf['iterations']} iteraciones, convergido {inf['converged']}, "
          f"rms {inf['rms_point_plane_mm']} mm, cobertura {inf['coverage']}, "
          f"ejes {inf['axis_angles_deg']}°, veto {inf['veto']}")
    print(f"· navaja ({inf['leave_one_out']['mode']}): {inf['leave_one_out']['jackknife']}")
    print(f"· peor dirección: {inf['conditioning']['worst_axis']} "
          f"(λ {inf['conditioning']['translation_eigenvalues']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
