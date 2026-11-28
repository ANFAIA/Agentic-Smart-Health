#!/usr/bin/env python
"""registro_navaja_arranques.py — Navaja por diente y arranques desplazados del registro
escáner↔CBCT, sin anotador (medidas 2 y 3 de los casos A y B).

    uv run python scripts/registro_navaja_arranques.py --caso A --referencia <ejecución R1>
    uv run python scripts/registro_navaja_arranques.py --caso B \\
        --referencia <ejecución de referencia del caso B>

Protocolo, criterios y lectura: `docs/research/registro-navaja-arranques.md`. Este script solo
ejecuta lo que allí está fijado, **una vez por caso**, desde un commit con el árbol limpio. No
tiene etapas sueltas ni modo de desarrollo: se niega a pisar una salida que ya existe.

**Mide precisión, no exactitud.** Lo que compartan todas las ejecuciones —la isolínea de 1400,
el borde a medio vóxel, la altura como dirección peor medida— se cancela y no aparece.

**Los datos no se reingieren.** Se cargan del almacén de artefactos de la ejecución de
referencia (el campo semilla del CBCT con `origin` y `hu_range`, y la superficie del escáner en
el orden de ingesta) y de su contenedor `.uos` (las etiquetas de `derived/seg_teeth` y
`reg.ct_to_ios`). Todo movimiento se hace en memoria y en float64.

**Marcos.** Toda pose es una 4×4 que lleva coordenadas del STL del escáner, en el orden de
ingesta, a milímetros absolutos del DICOM (`centers + origin`). `a @ b` aplica primero `b`.
`reg.ct_to_ios` es la inversa de la pose del registro.

**Nada de la medida se imprime antes de quedar escrito.** Cada etapa escribe su JSON y solo
después resume en pantalla. Ningún nombre de fichero del proveedor ni ruta de paciente llega a
las salidas: el `argv` se reconstruye de los argumentos YA interpretados, con cada ruta
sustituida por su papel, y el analizador no acepta abreviaturas.

**Orden (§4).** V0 → E1 → lo demás. Si V0 o E1 fallan, se escribe su JSON y el
`run-record.json` con `aborted` y no se calcula nada más.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import precision_registro  # noqa: E402
from consistencia_registro import _de_icp, _sha, aplica  # noqa: E402
from export_agents import densidad_a_hu  # noqa: E402
from export_agents.anatomia import anatomical_frame  # noqa: E402
from fusion_agents.geometric import GeometricFusionAgent  # noqa: E402
from fusion_agents.por_cortes import EJES_LPS, SerieDensa, prepara_por_cortes  # noqa: E402
from fusion_agents.preparacion import (  # noqa: E402
    HU_ARBITRO,
    HU_ESMALTE,
    nubes_para_registro,
    plano_oclusal_del_esmalte,
)
from fusion_agents.registration import EPSILON_IOS_CBCT_MM, icp  # noqa: E402
from precision_registro import (  # noqa: E402
    P1_DIRECCIONES,
    VUELTA_MM,
    _4x4,
    _a_json,
    _angulo_4x4,
    _gira,
    _git_rastrea,
    movimientos,
)

RESUMEN_EN = ("Runs the pre-registered leave-one-tooth-out and displaced-start precision "
              "measures of the scan-to-CBCT registration on cases A and B.")

PROTOCOLO = "docs/research/registro-navaja-arranques.md"

# =========================================================================== #
# Constantes del protocolo (§2–§4). El documento las fija; aquí solo se copian.
# =========================================================================== #
DIENTES_ESPERADOS = {
    "A": (11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27),
    "B": (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27),
}
"""§2: los dientes que se esperan. Si las etiquetas de referencia traen otros, se miden los
que haya y se declara: no se elige ninguno."""
CONTENEDOR_SHA256 = {
    "A": "239e258e077d7b53a561740d67d8188705992dd00a3e992f38b5d05819f8034a",
    "B": "c1baaf680771b675fe83e74be473fb0c377920a5dd0b333197ad0cabd06d82f7",
}
"""§2: la sha256 del contenedor `.uos` de la ejecución de referencia de cada caso."""
COMMIT_REFERENCIA = {"A": "57eb021", "B": "eef9d27"}
"""§2: el commit (prefijo) que tiene que dar el `run-record.json` de la referencia, con
`git_dirty: false`."""
ANOTACIONES = "data/processed/puntos-anatomicos/anotaciones"
"""§6: la carpeta de entregas, relativa a la copia principal. No es un argumento."""
SALIDAS = "data/processed/registro-navaja-arranques"
"""§8: la carpeta de salidas, relativa a la copia principal; cada una lleva el commit."""
ARCADA_B = "maxilar"
"""La que `caso_completo.py` lee del nombre `escaneo-superior.stl` del caso B."""

V0_MAX_MM = 0.01
"""§3.1: desplazamiento máximo de las coronas entre la pose reconstruida y la de referencia."""
SEMILLA_PIPELINE = 0
"""La semilla que usa el pipeline en el camino clásico (`nubes_para_registro` e `icp`)."""
SEMILLAS_B = tuple(range(20))
"""§3.2–§3.3: la misma semilla va a `nubes_para_registro` y a `icp`."""
HU_RESIDUO_B = HU_ARBITRO
"""§3.2: el residuo de X en el caso B se mide contra las gaussianas con HU ≥ 1800."""

ANGULOS_GRADOS = (2.0, 5.0, 10.0)
ANGULOS_K1_GRADOS = (2.0, 5.0)
DIRECCIONES = P1_DIRECCIONES
"""Las 8 direcciones de los vértices de un cubo, en (derecha, anterior, superior) del CBCT."""
ALTURAS_MM = (-5.0, -3.0, -1.0, 1.0, 3.0, 5.0)
ALTURAS_K1_MM = (-3.0, -1.0, 1.0, 3.0)
TRASLACIONES_MM = (1.0, 3.0, 5.0)
TRASLACIONES_K1_MM = (1.0, 3.0)
EJES_TRASLACION = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
"""Los ejes del DICOM: ±x es ∓derecha, ±y es ∓anterior, ±z es ±superior (LPS)."""

N_MOVIMIENTOS_B = 10
GIRO_B_PASO_GRADOS = 36.0
GIRO_B_INICIO_GRADOS = 10.0
SEMILLA_MOVIMIENTO_B = 1001
TRASLACION_B_MAX_MM = 100.0
"""§3.4: el k-ésimo gira 36°·k + 10° alrededor de z y traslada en x/y lo que dé
`default_rng(1001 + k).uniform(-100, 100, 2)`, sin traslación en z."""

N1_FACTOR = 2.0
N2_MAX_MM = EPSILON_IOS_CBCT_MM
K1_MAX_MM = 0.1
E1_MAX_MM = 0.01
UMBRAL_VUELTA_MM = VUELTA_MM
"""§3.5: una ejecución es vuelta si la MEDIANA del desplazamiento de TODAS las coronas del
escaneo (X incluido en las ejecuciones sin X) respecto a T₀ es estrictamente mayor que esto."""
MIN_PARES_VALIDOS_B = 10
"""§3.5: un diente del caso B con menos pares válidos queda indeterminado."""
SUELO_PERCENTIL = 95.0
"""§3.6: X supera el suelo si su movimiento pasa del p95 de su propio suelo de semilla."""
FRACCION_VUELTAS_ILEGIBLE = 0.5
"""§3.5: si MÁS de esta fracción de las ejecuciones completas desde cero son vuelta, nada
del caso B se lee, K1 incluido."""

MODULOS_HUELLA = ("fusion_agents.por_cortes", "fusion_agents.preparacion",
                  "fusion_agents.registration", "fusion_agents.geometric",
                  "export_agents.anatomia")
"""§8: se importan y se toman los VALORES resueltos de sus nombres en mayúsculas."""
CONSTANTES_PRECISION = ("SEMILLAS_SO3", "TRASLACION_MAX_MM", "GIROS_FIJOS",
                        "TRASLACIONES_FIJAS_MM", "VUELTA_MM", "P1_DIRECCIONES")
"""§8: las de `precision_registro` que usan `movimientos()`, `VUELTA_MM` y las direcciones."""
FUERA_DE_HUELLA = ("RAIZ",)
"""Constantes propias que dependen de dónde está la copia, no de la medida."""


# =========================================================================== #
# Transformaciones
# =========================================================================== #
def inversa(m: np.ndarray) -> np.ndarray:
    """Inversa de una 4×4 rígida, sin invertir la matriz entera."""
    R = np.asarray(m, dtype=np.float64)[:3, :3]
    return _4x4(R.T, -R.T @ np.asarray(m, dtype=np.float64)[:3, 3])


def pose_de(resultado: Any) -> np.ndarray:
    """`RegistrationResult` → 4×4, por el mismo cuaternión que guarda el pipeline."""
    return _de_icp(resultado)


def giro_en_torno(R: np.ndarray, centro: np.ndarray) -> np.ndarray:
    """La 4×4 que gira `R` alrededor de `centro` (deja `centro` quieto)."""
    c = np.asarray(centro, dtype=np.float64)
    return _4x4(R, c - R @ c)


def eje_anatomico(direccion: Sequence[float]) -> np.ndarray:
    """(derecha, anterior, superior) del CBCT → vector en el DICOM (LPS), como en P1."""
    return EJES_LPS.T @ np.asarray(direccion, dtype=np.float64)


def giro_z(grados: float) -> np.ndarray:
    """Giro alrededor de z con ceros EXACTOS en la fila y la columna de z: así la z de cada
    punto no cambia ni en el último bit, y el histograma del plano oclusal es el mismo."""
    a = np.radians(grados)
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def desplazamiento(m1: np.ndarray, m2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Norma, vértice a vértice, entre `X` llevado con `m1` y con `m2`."""
    return np.linalg.norm(aplica(m1, X) - aplica(m2, X), axis=1)


def mueve_campo(campo: dict[str, np.ndarray], G: np.ndarray) -> dict[str, np.ndarray]:
    """El campo con sus coordenadas ABSOLUTAS movidas por `G`, en float64 y con origen 0."""
    absolutas = (np.asarray(campo["centers"], dtype=np.float64)
                 + np.asarray(campo["origin"], dtype=np.float64))
    return {"centers": aplica(G, absolutas), "origin": np.zeros(3)}


def movimientos_b() -> list[tuple[str, np.ndarray]]:
    """Los 10 movimientos de la equivarianza del caso B (§3.4), en orden fijo."""
    salida = []
    for k in range(N_MOVIMIENTOS_B):
        grados = GIRO_B_PASO_GRADOS * k + GIRO_B_INICIO_GRADOS
        txy = np.random.default_rng(SEMILLA_MOVIMIENTO_B + k).uniform(
            -TRASLACION_B_MAX_MM, TRASLACION_B_MAX_MM, 2)
        salida.append((f"z{grados:.0f}-{k}", _4x4(giro_z(grados), np.r_[txy, 0.0])))
    return salida


# =========================================================================== #
# Estadísticos
# =========================================================================== #
def estadisticos(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=np.float64)
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "median_mm": float(np.median(d)),
            "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max()),
            "mean_mm": float(d.mean())}


def movimiento(m_con: np.ndarray, m_sin: np.ndarray, X: np.ndarray) -> dict:
    """§3.2: cuánto se mueve cada vértice de `X` entre la pose completa y la pose sin X.

    La componente vertical es la media de la z (superior +, LPS) de `sin − con`."""
    if len(X) == 0:
        return {"n": 0, "median_mm": None, "max_mm": None, "mean_vertical_mm": None}
    dif = aplica(m_sin, X) - aplica(m_con, X)
    n = np.linalg.norm(dif, axis=1)
    return {"n": int(len(X)), "median_mm": float(np.median(n)), "max_mm": float(n.max()),
            "mean_vertical_mm": float(dif[:, 2].mean())}


def etiqueta_del_contorno(pose: np.ndarray, corona: np.ndarray, etq_corona: np.ndarray,
                          destino: np.ndarray) -> np.ndarray:
    """La etiqueta del vértice de corona más cercano a cada punto de contorno, con `pose`.

    Con la pose completa, es la máscara de la variante (b): circular, y se declara."""
    _, j = cKDTree(aplica(pose, corona)).query(destino)
    return np.asarray(etq_corona)[j]


def residuo_contorno(pose: np.ndarray, contorno: np.ndarray,
                     corona: np.ndarray) -> float | None:
    """§3.2, caso A: mediana sin truncar de contorno de X → corona de X movida con `pose`."""
    if len(contorno) == 0 or len(corona) == 0:
        return None
    return float(np.median(cKDTree(aplica(pose, corona)).query(contorno)[0]))


def residuo_esmalte(pose: np.ndarray, corona: np.ndarray, arbol: cKDTree) -> float | None:
    """§3.2, caso B: mediana sin truncar de la corona de X movida → gaussiana HU ≥ 1800."""
    if len(corona) == 0:
        return None
    return float(np.median(arbol.query(aplica(pose, corona))[0]))


def es_vuelta(pose: np.ndarray, T0: np.ndarray, coronas: np.ndarray) -> tuple[bool, float]:
    """§3.5: `(vuelta, mediana del desplazamiento de las coronas respecto a T₀)`."""
    m = float(np.median(desplazamiento(pose, T0, coronas)))
    return m > UMBRAL_VUELTA_MM, m


def _mediana(xs: Iterable[float | None]) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return float(np.median(v)) if v else None


# =========================================================================== #
# Criterios (§4)
# =========================================================================== #
def agrega_pares(pares: Sequence[dict], dientes: Sequence[int]) -> dict[int, dict]:
    """Por diente, las medianas entre semillas de los pares SIN vuelta (§3.5).

    Cada par es `{"tooth", "valid", "residual_full_mm", "residual_without_mm", "movement"}`;
    `valid` es falso si cualquiera de los dos registros es una vuelta. En el caso A hay un
    par por diente y siempre es válido.
    """
    salida: dict[int, dict] = {}
    for x in dientes:
        suyos = [p for p in pares if int(p["tooth"]) == int(x)]
        buenos = [p for p in suyos if p["valid"]]
        salida[int(x)] = {
            "n_pairs": len(suyos), "n_valid": len(buenos),
            "residual_full_mm": _mediana(p["residual_full_mm"] for p in buenos),
            "residual_without_mm": _mediana(p["residual_without_mm"] for p in buenos),
            "movement_median_mm": _mediana(p["movement"]["median_mm"] for p in buenos),
            "movement_max_mm": (max(p["movement"]["max_mm"] for p in buenos)
                                if buenos else None),
            "movement_mean_vertical_mm": _mediana(p["movement"]["mean_vertical_mm"]
                                                  for p in buenos),
        }
    return salida


def evalua_n1(por_diente: dict[int, dict]) -> dict:
    """N1: mediana entre dientes del residuo sin X ≤ 2 × la del residuo con la pose completa."""
    validos = {x: v for x, v in por_diente.items()
               if v["residual_full_mm"] is not None and v["residual_without_mm"] is not None}
    fuera = sorted(int(x) for x in por_diente if x not in validos)
    if not validos:
        return {"pass": None, "teeth_used": [], "teeth_without_data": fuera}
    sin = float(np.median([v["residual_without_mm"] for v in validos.values()]))
    con = float(np.median([v["residual_full_mm"] for v in validos.values()]))
    return {"pass": bool(sin <= N1_FACTOR * con), "median_without_mm": sin,
            "median_full_mm": con, "limit_mm": N1_FACTOR * con,
            "ratio": (sin / con) if con > 0 else None,
            "teeth_used": sorted(int(x) for x in validos), "teeth_without_data": fuera}


def evalua_n2(por_diente: dict[int, dict]) -> dict:
    """N2: ningún diente se mueve más de ε (la mediana de sus vértices de corona)."""
    validos = {x: v for x, v in por_diente.items() if v["movement_median_mm"] is not None}
    fuera = sorted(int(x) for x in por_diente if x not in validos)
    if not validos:
        return {"pass": None, "teeth_over": [], "teeth_without_data": fuera}
    encima = sorted(int(x) for x, v in validos.items() if v["movement_median_mm"] > N2_MAX_MM)
    return {"pass": not encima, "limit_mm": N2_MAX_MM, "teeth_over": encima,
            "max_movement_median_mm": max(v["movement_median_mm"] for v in validos.values()),
            "teeth_without_data": fuera}


def _falla_k1(f: dict) -> bool:
    return bool(f["flip"] or not f["converged"] or f["displacement_median_mm"] > K1_MAX_MM)


def evalua_k1(filas: Sequence[dict]) -> dict:
    """K1: toda ejecución con criterio vuelve a ≤ 0,1 mm, converge y no es vuelta (§3.5, §4).

    Cada fila es `{"with_criterion", "displacement_median_mm", "flip", "converged"}`; `flip`
    es verdadero si la ejecución perturbada o su referencia sin perturbar es una vuelta, así
    que una referencia vuelta hace fallar todos sus arranques. `converged = False` es fallo y
    se cuenta aparte."""
    con = [f for f in filas if f["with_criterion"]]
    fallos = [f for f in con if _falla_k1(f)]
    return {"pass": bool(con) and not fallos, "limit_mm": K1_MAX_MM, "n": len(con),
            "failures": len(fallos),
            "failures_by_flip": sum(bool(f["flip"]) for f in fallos),
            "failures_not_converged": sum(not f["converged"] for f in fallos),
            "failures_by_distance": sum(f["displacement_median_mm"] > K1_MAX_MM
                                        for f in fallos),
            "max_displacement_median_mm": (max(f["displacement_median_mm"] for f in con)
                                           if con else None)}


def resume_sin_criterio(filas: Sequence[dict]) -> dict:
    """Los arranques sin criterio (10°, 5 mm): los mismos recuentos que K1, sin veredicto."""
    salida = evalua_k1([{**f, "with_criterion": True} for f in filas])
    salida.pop("pass")
    salida.pop("limit_mm")
    return {**salida, "criterion": "sin criterio"}


def lee_suelo(por_diente: dict[int, dict], suelo: dict[int, Sequence[float]]) -> dict[int, dict]:
    """§3.5–§3.6, caso B: por diente, si su movimiento sin X supera su suelo de semilla.

    `por_diente[x]["movement_median_mm"]` es la mediana entre semillas (pares válidos) del
    movimiento de X sin X; `suelo[x]`, la mediana del desplazamiento de las coronas de X entre
    la pose completa con s y T₀, para cada s = 1…19 que no sea vuelta. X supera el suelo si lo
    primero es mayor que el p95 de lo segundo. Con menos de 10 pares válidos, o sin suelo, X
    queda indeterminado."""
    salida: dict[int, dict] = {}
    for x, v in por_diente.items():
        valores = [float(s) for s in suelo.get(int(x), [])]
        p95 = float(np.percentile(valores, SUELO_PERCENTIL)) if valores else None
        mov = v["movement_median_mm"]
        if v["n_valid"] < MIN_PARES_VALIDOS_B or mov is None or p95 is None:
            estado = "indeterminado"
        else:
            estado = "supera" if mov > p95 else "no supera"
        salida[int(x)] = {"state": estado, "n_valid": int(v["n_valid"]),
                          "movement_median_mm": mov, "floor_p95_mm": p95,
                          "floor_n_seeds": len(valores)}
    return salida


def legibilidad_b(completos_desde_cero: Sequence[dict]) -> dict:
    """§3.5: si MÁS de la mitad de las ejecuciones completas desde cero son vuelta respecto a
    T₀, T₀ no representa al camino clásico y nada del caso B se lee, K1 incluido."""
    n = len(completos_desde_cero)
    vueltas = sum(bool(r["flip"]) for r in completos_desde_cero)
    legible = n > 0 and vueltas <= FRACCION_VUELTAS_ILEGIBLE * n
    return {"readable": bool(legible), "n_full_scratch": n, "flips_full_scratch": vueltas}


def tasa(filas: Sequence[dict]) -> dict:
    n = len(filas)
    k = sum(bool(r["flip"]) for r in filas)
    return {"n": n, "flips": k, "rate": (k / n) if n else None}


def evalua_e1(filas: Sequence[dict]) -> dict:
    """E1: el desplazamiento máximo de las coronas respecto a la pose esperada ≤ 0,01 mm."""
    peor = max((f["max_mm"] for f in filas), default=None)
    return {"pass": bool(filas) and peor is not None and peor <= E1_MAX_MM,
            "limit_mm": E1_MAX_MM, "n": len(filas), "max_mm": peor}


# =========================================================================== #
# Caso A: `fusion_agents.por_cortes`
# =========================================================================== #
def prepara_a(campo: dict, hu: np.ndarray, V: np.ndarray, etq: np.ndarray,
              plano: float | None = None) -> tuple[np.ndarray, np.ndarray, Any, dict]:
    """Lo que hace `IngestionPipeline.prepara_registro` con `registro="auto"` y etiquetas.

    `plano` solo existe para los tests sintéticos; en la medida es `None`, el valor por
    defecto de `prepara_por_cortes`, y no se pasa."""
    base, motivo = anatomical_frame(np.asarray(V), np.asarray(etq, dtype=np.int64))
    if base is None:
        raise ValueError(f"sin base anatómica del escáner: {motivo}")
    extra = {} if plano is None else {"plano": plano}
    return prepara_por_cortes(
        campo, hu, np.asarray(V)[np.asarray(etq) > 0],
        np.stack([base.derecha, base.anterior, base.superior]), base.arcada, **extra)


def _fila_registro(r: Any) -> dict:
    return {"iterations": int(r.iterations), "converged": bool(r.converged),
            "rms_mm": float(r.rms_mm), "rms_overlap_mm": r.rms_overlap_mm,
            "overlap_fraction": r.overlap_fraction}


def navaja_a(reg: Any, corona: np.ndarray, etq_corona: np.ndarray, destino: np.ndarray,
             T0: np.ndarray, dientes: Sequence[int]) -> dict:
    """§3.2, caso A: `rot0` y `plano` fijos; para cada X, el registrador sobre la corona sin X.

    (b) principal: sale también el contorno de X (el que, con la pose completa, cae más
    cerca de la corona de X). (a) descriptiva: el contorno de X se queda."""
    etq_corona = np.asarray(etq_corona)
    arranque = replace(reg, max_iter=0)
    S0 = pose_de(arranque(corona, destino))
    del_contorno = etiqueta_del_contorno(T0, corona, etq_corona, destino)
    filas, pares = [], {"b": [], "a": []}
    for x in dientes:
        es_x = etq_corona == x
        Xv, sin = corona[es_x], corona[~es_x]
        mascara = del_contorno == x
        contorno_x = destino[mascara]
        r_con = residuo_contorno(T0, contorno_x, Xv)
        fila: dict[str, Any] = {"tooth": int(x), "n_crown_vertices": int(es_x.sum()),
                                "n_contour_points": int(mascara.sum()),
                                "residual_full_mm": r_con}
        for variante, objetivo in (("b", destino[~mascara]), ("a", destino)):
            r = reg(sin, objetivo)
            P = pose_de(r)
            S = pose_de(arranque(sin, objetivo))
            fila[variante] = {
                "pose": P, "start_pose": S, **_fila_registro(r),
                "residual_without_mm": residuo_contorno(P, contorno_x, Xv),
                "movement": movimiento(T0, P, Xv),
                "start_movement": movimiento(S0, S, Xv),
                "crown_displacement_vs_full": estadisticos(desplazamiento(P, T0, corona)),
            }
            pares[variante].append({"tooth": int(x), "valid": True, "residual_full_mm": r_con,
                                    "residual_without_mm": fila[variante]["residual_without_mm"],
                                    "movement": fila[variante]["movement"]})
        filas.append(fila)
    sin_contorno = [f["tooth"] for f in filas if f["n_contour_points"] == 0]
    return {"full_start_pose": S0, "teeth": filas,
            "teeth_without_contour": sin_contorno,
            "teeth_without_contour_note": "fuera de N1 (no hay residuo), dentro de N2 (§3.2)",
            "per_tooth": {v: agrega_pares(pares[v], dientes) for v in pares}}


def arranques_a(reg: Any, corona: np.ndarray, destino: np.ndarray, T0: np.ndarray, *,
                angulos: Sequence[float] = ANGULOS_GRADOS,
                direcciones: Sequence[Sequence[int]] = DIRECCIONES,
                alturas: Sequence[float] = ALTURAS_MM) -> list[dict]:
    """§3.3, caso A: `rot0 ← Rδ·rot0` (24) y `plano ← plano + dz` (6)."""
    filas = []
    casos: list[tuple[dict, Any]] = []
    for grados in angulos:
        for d in direcciones:
            R = _gira(eje_anatomico(d), grados)
            casos.append(({"kind": "rotation", "angle_deg": float(grados),
                           "direction_anatomical": list(d),
                           "with_criterion": float(grados) in ANGULOS_K1_GRADOS},
                          replace(reg, rot0=R @ np.asarray(reg.rot0, dtype=np.float64))))
    for dz in alturas:
        casos.append(({"kind": "height", "dz_mm": float(dz),
                       "with_criterion": float(dz) in ALTURAS_K1_MM},
                      replace(reg, plano=float(reg.plano) + float(dz))))
    for meta, r_p in casos:
        r = r_p(corona, destino)
        P = pose_de(r)
        d = desplazamiento(P, T0, corona)
        filas.append({**meta, "pose": P,
                      "start_pose": pose_de(replace(r_p, max_iter=0)(corona, destino)),
                      **_fila_registro(r), "crown_displacement": estadisticos(d),
                      "displacement_median_mm": float(np.median(d)), "flip": False,
                      "angle_to_reference_deg": _angulo_4x4(P @ inversa(T0))})
    return filas


def equivariancia_a(campo: dict, hu: np.ndarray, V: np.ndarray, etq: np.ndarray,
                    T: np.ndarray, movs: Sequence[tuple[str, np.ndarray, np.ndarray]],
                    plano: float | None = None) -> list[dict]:
    """§3.4, caso A: el escaneo movido por G; se espera `T' = T∘G⁻¹`."""
    filas = []
    for nombre, R, t in movs:
        G = _4x4(R, t)
        V2 = aplica(G, np.asarray(V, dtype=np.float64))
        corona2, destino2, reg2, _ = prepara_a(campo, hu, V2, etq, plano)
        r = reg2(corona2, destino2)
        P = pose_de(r)
        d = desplazamiento(P, T @ inversa(G), corona2)
        filas.append({"motion": nombre, "G": G, "pose": P, **_fila_registro(r),
                      "max_mm": float(d.max()), "median_mm": float(np.median(d))})
    return filas


# =========================================================================== #
# Caso B: el ICP clásico con árbitro
# =========================================================================== #
def registra_b(campo: dict, hu: np.ndarray, V: np.ndarray, semilla: int,
               previa: np.ndarray | None = None) -> dict:
    """El camino clásico del pipeline con la semilla `semilla` en sus dos puntos de azar.

    `previa` mueve el escaneo en memoria antes de registrar (arranque en caliente o
    perturbado). La pose devuelta es la EFECTIVA: del escaneo sin mover al DICOM."""
    Vm = np.asarray(V) if previa is None else aplica(previa, np.asarray(V, dtype=np.float64))
    origen, destino, informe = nubes_para_registro(
        campo, Vm, arcada=ARCADA_B, hu=hu, dos_arcadas=True,
        registrar=partial(icp, seed=semilla), semilla=semilla)
    r = icp(origen, destino, seed=semilla)
    W = pose_de(r)
    return {"seed": int(semilla), "pose": W if previa is None else W @ previa,
            "hu_target": informe.get("hu_objetivo"),
            "arbiter_score": informe.get("puntuacion_arbitro"),
            "arbiter_scores": informe.get("puntuaciones"),
            "n_target": informe.get("n_objetivo"),
            "occlusal_plane_mm": informe.get("plano_oclusal_mm"), **_fila_registro(r)}


def _marca(run: dict, T0: np.ndarray, coronas: np.ndarray) -> dict:
    vuelta, m = es_vuelta(run["pose"], T0, coronas)
    return {**run, "flip": vuelta, "crown_displacement_vs_T0_median_mm": m}


def navaja_b(campo: dict, hu: np.ndarray, V: np.ndarray, etq: np.ndarray, T0: np.ndarray,
             arbol_esmalte: cKDTree, *, semillas: Sequence[int] = SEMILLAS_B,
             dientes: Sequence[int], base0: dict | None = None,
             progreso: Any = None) -> dict:
    """§3.2, caso B: desde cero (principal) y en caliente desde T₀ (descriptiva).

    Para cada semilla s, la pose sin X se compara con la pose completa con la misma s.
    `base0` es la ejecución completa desde cero con la semilla del pipeline, ya hecha en V0.
    """
    etq = np.asarray(etq)
    coronas = np.asarray(V)[etq > 0]
    corona_de = {int(x): np.asarray(V)[etq == x] for x in dientes}
    ejecuciones: dict[str, list[dict]] = {"scratch": [], "warm": []}
    pares: dict[str, list[dict]] = {"scratch": [], "warm": []}
    for s in semillas:
        if progreso:
            progreso(f"navaja: semilla {s}")
        for modo, previa in (("scratch", None), ("warm", T0)):
            if modo == "scratch" and base0 is not None and int(s) == base0["seed"]:
                completo = _marca(base0, T0, coronas)
            else:
                completo = _marca(registra_b(campo, hu, V, s, previa), T0, coronas)
            ejecuciones[modo].append({**completo, "without": None})
            for x in dientes:
                sin = _marca(registra_b(campo, hu, V[etq != x], s, previa), T0, coronas)
                ejecuciones[modo].append({**sin, "without": int(x)})
                Xv = corona_de[int(x)]
                pares[modo].append({
                    "seed": int(s), "tooth": int(x),
                    "flip_full": completo["flip"], "flip_without": sin["flip"],
                    "valid": not (completo["flip"] or sin["flip"]),
                    "residual_full_mm": residuo_esmalte(completo["pose"], Xv, arbol_esmalte),
                    "residual_without_mm": residuo_esmalte(sin["pose"], Xv, arbol_esmalte),
                    "movement": movimiento(completo["pose"], sin["pose"], Xv),
                })
    return {"runs": ejecuciones, "pairs": pares,
            "per_tooth": {m: agrega_pares(pares[m], dientes) for m in pares}}


def suelo_b(completos: Sequence[dict], T0: np.ndarray, V: np.ndarray, etq: np.ndarray,
            dientes: Sequence[int]) -> dict:
    """§3.6: por diente, la mediana del desplazamiento de sus vértices de corona entre la pose
    completa con la semilla s y T₀, para cada s = 1…19 que no sea vuelta."""
    etq = np.asarray(etq)
    coronas = np.asarray(V)[etq > 0]
    validos = [c for c in completos if c["seed"] != SEMILLA_PIPELINE and not c["flip"]]
    por_semilla = []
    for c in validos:
        por_semilla.append({"seed": c["seed"], "crowns": estadisticos(
            desplazamiento(c["pose"], T0, coronas)), "per_tooth_median_mm": {
            int(x): float(np.median(desplazamiento(c["pose"], T0, np.asarray(V)[etq == x])))
            for x in dientes}})
    valores = {int(x): [p["per_tooth_median_mm"][int(x)] for p in por_semilla]
               for x in dientes}
    return {
        "seeds": [c["seed"] for c in completos if c["seed"] != SEMILLA_PIPELINE],
        "flips": sorted(c["seed"] for c in completos
                        if c["seed"] != SEMILLA_PIPELINE and c["flip"]),
        "per_seed": por_semilla,
        "per_tooth_values_mm": valores,
        "per_tooth_p95_mm": {x: (float(np.percentile(v, SUELO_PERCENTIL)) if v else None)
                             for x, v in valores.items()},
        "crowns_median_of_medians_mm": _mediana(p["crowns"]["median_mm"] for p in por_semilla),
    }


def arbol_esmalte_maxilar(campo: dict, hu: np.ndarray) -> tuple[cKDTree, dict]:
    """§3.2, caso B: las gaussianas con HU ≥ 1800 del lado maxilar del plano de separación.

    El plano es el de `nubes_para_registro` con `dos_arcadas=True`: la misma función
    (`plano_oclusal_del_esmalte`) sobre la misma entrada (la z de HU ≥ `HU_ESMALTE`), y el
    mismo lado (`z ≥ plano` para el maxilar)."""
    centros = (np.asarray(campo["centers"], dtype=np.float64)
               + np.asarray(campo["origin"], dtype=np.float64))
    hu = np.asarray(hu)
    plano = plano_oclusal_del_esmalte(centros[hu >= HU_ESMALTE][:, 2])
    elegidas = (hu >= HU_RESIDUO_B) & (centros[:, 2] >= plano)
    if not elegidas.any():
        raise SystemExit("No hay esmalte del lado maxilar del plano: no hay residuo de B.")
    return cKDTree(centros[elegidas]), {"split_plane_mm": float(plano),
                                        "n_enamel_maxillary": int(elegidas.sum()),
                                        "hu_min": HU_RESIDUO_B, "side": "z >= plane"}


def perturbaciones_b(centro: np.ndarray, *, angulos: Sequence[float] = ANGULOS_GRADOS,
                     direcciones: Sequence[Sequence[int]] = DIRECCIONES,
                     traslaciones: Sequence[float] = TRASLACIONES_MM,
                     ejes: Sequence[Sequence[int]] = EJES_TRASLACION
                     ) -> list[tuple[dict, np.ndarray]]:
    """§3.3, caso B: P en el DICOM. Los giros, alrededor del centroide de las coronas en T₀."""
    salida = []
    for grados in angulos:
        for d in direcciones:
            salida.append(({"kind": "rotation", "angle_deg": float(grados),
                            "direction_anatomical": list(d),
                            "with_criterion": float(grados) in ANGULOS_K1_GRADOS},
                           giro_en_torno(_gira(eje_anatomico(d), grados), centro)))
    for mm in traslaciones:
        for e in ejes:
            salida.append(({"kind": "translation", "mm": float(mm), "axis_dicom": list(e),
                            "with_criterion": float(mm) in TRASLACIONES_K1_MM},
                           _4x4(np.eye(3), float(mm) * np.asarray(e, dtype=np.float64))))
    return salida


def arranques_b(campo: dict, hu: np.ndarray, V: np.ndarray, etq: np.ndarray,
                T0: np.ndarray, referencias: dict[int, dict],
                perturbaciones: Sequence[tuple[dict, np.ndarray]], *,
                semillas: Sequence[int] = SEMILLAS_B, progreso: Any = None) -> list[dict]:
    """§3.3, caso B: el escaneo movido con P·T₀; cada resultado frente al arranque en
    caliente sin perturbar con la misma semilla (`referencias[s]`)."""
    coronas = np.asarray(V)[np.asarray(etq) > 0]
    filas = []
    for i, (meta, P) in enumerate(perturbaciones):
        if progreso:
            progreso(f"arranques: perturbación {i + 1}/{len(perturbaciones)}")
        for s in semillas:
            run = _marca(registra_b(campo, hu, V, s, P @ T0), T0, coronas)
            ref = referencias[int(s)]
            d = desplazamiento(run["pose"], ref["pose"], coronas)
            filas.append({**meta, "P": P, **run, "reference_flip": ref["flip"],
                          "reference_converged": ref["converged"],
                          "displacement_median_mm": float(np.median(d)),
                          "displacement_max_mm": float(d.max()),
                          "flip": bool(run["flip"] or ref["flip"]),
                          "run_flip": run["flip"]})
    return filas


def equivariancia_b(campo: dict, hu: np.ndarray, V: np.ndarray, etq: np.ndarray,
                    T: np.ndarray, movs: Sequence[tuple[str, np.ndarray]]) -> list[dict]:
    """§3.4, caso B: escaneo y campo movidos a la vez por G; se espera `G·T·G⁻¹`."""
    coronas = np.asarray(V, dtype=np.float64)[np.asarray(etq) > 0]
    filas = []
    for nombre, G in movs:
        run = registra_b(mueve_campo(campo, G), hu,
                         aplica(G, np.asarray(V, dtype=np.float64)), SEMILLA_PIPELINE)
        d = desplazamiento(run["pose"], G @ T @ inversa(G), aplica(G, coronas))
        filas.append({"motion": nombre, "G": G, **run, "max_mm": float(d.max()),
                      "median_mm": float(np.median(d))})
    return filas


# =========================================================================== #
# Entradas
# =========================================================================== #
@dataclass
class Datos:
    """Lo que se carga de la ejecución de referencia. Nada se reingiere."""

    campo: dict[str, np.ndarray]
    hu: np.ndarray
    V: np.ndarray
    etq: np.ndarray
    T_ref: np.ndarray
    entradas: dict


def lee_contenedor(ruta: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """`(T_ref, etiquetas, meta de seg_teeth, hashes)` del `.uos` de referencia."""
    with zipfile.ZipFile(ruta) as z:
        manifiesto = json.loads(z.read("manifest.json"))
        crudo = z.read("derived/seg_teeth.bin")
        meta_crudo = z.read("derived/seg_teeth.meta.json")
    reg = next((r for r in manifiesto.get("registrations", [])
                if r.get("id") == "reg.ct_to_ios"), None)
    if reg is None:
        raise SystemExit("El contenedor de referencia no trae `reg.ct_to_ios`.")
    ct_a_ios = np.asarray(reg["transform_4x4_row_major"], dtype=np.float64).reshape(4, 4)
    etq = np.frombuffer(crudo, dtype="<i2").astype(np.int64)
    hashes = {"seg_teeth_sha256": hashlib.sha256(crudo).hexdigest(),
              "seg_teeth_meta_sha256": hashlib.sha256(meta_crudo).hexdigest(),
              "reference_rms_mm": reg.get("rms_error_mm"),
              "reference_transform_4x4_ct_to_ios": ct_a_ios.tolist()}
    return np.linalg.inv(ct_a_ios), etq, json.loads(meta_crudo), hashes


def _sha_posiciones(posiciones: np.ndarray) -> str:
    """El T-4 del `.uos`: sha256 de las posiciones en float32, en el orden de la escena."""
    return hashlib.sha256(np.ascontiguousarray(posiciones, dtype=np.float32).tobytes()
                          ).hexdigest()


def elige_campo(npzs: Sequence[Path]) -> tuple[Path, str]:
    """El campo SEMILLA del `cbct-agent`: sin `region_id` y con todas las rotaciones en la
    identidad. El segmentado lleva `region_id` y el ajustado rota sus elipsoides.

    Si hubiera más de uno, vale el único cuyos `centers` son los de un campo segmentado: la
    segmentación corre sobre el campo con el que se registró, dentro del mismo `fuse`."""
    candidatos, segmentados = [], []
    for p in npzs:
        with np.load(p, allow_pickle=False) as z:
            claves = set(z.files)
            if not {"centers", "density", "origin", "hu_range", "rotations"} <= claves:
                continue
            if "region_id" in claves:
                segmentados.append(hashlib.sha256(z["centers"].tobytes()).hexdigest())
                continue
            rot = z["rotations"]
            if rot.ndim == 2 and rot.shape[1] == 4 and np.all(rot == np.array([1, 0, 0, 0])):
                candidatos.append(p)
    if len(candidatos) == 1:
        return candidatos[0], "only_seed_field"
    if len(candidatos) > 1:
        iguales = []
        for p in candidatos:
            with np.load(p, allow_pickle=False) as z:
                if hashlib.sha256(z["centers"].tobytes()).hexdigest() in segmentados:
                    iguales.append(p)
        if len(iguales) == 1:
            return iguales[0], "centers_of_segmented_field"
    raise SystemExit(f"El almacén de referencia tiene {len(candidatos)} campos semilla y no "
                     "se puede decir cuál se registró: no se elige a ciegas.")


def elige_superficie(npzs: Sequence[Path], n_etiquetas: int,
                     sha_posiciones: str | None) -> tuple[Path, str]:
    """La superficie del escáner a la que indexan las etiquetas de `derived/seg_teeth`."""
    candidatos = []
    for p in npzs:
        with np.load(p, allow_pickle=False) as z:
            if {"positions", "faces"} <= set(z.files):
                candidatos.append(p)
    if sha_posiciones is not None:
        iguales = []
        for p in candidatos:
            with np.load(p, allow_pickle=False) as z:
                if _sha_posiciones(z["positions"]) == sha_posiciones:
                    iguales.append(p)
        if len(iguales) != 1:
            raise SystemExit(f"{len(iguales)} superficies del almacén tienen las posiciones "
                             "que declara `derived/seg_teeth` (se espera una).")
        return iguales[0], "source_positions_sha256"
    iguales = []
    for p in candidatos:
        with np.load(p, allow_pickle=False) as z:
            if len(z["positions"]) == n_etiquetas:
                iguales.append(p)
    if len(iguales) != 1:
        raise SystemExit(f"{len(iguales)} superficies del almacén tienen tantos vértices como "
                         "etiquetas (se espera una) y el contenedor no declara su hash.")
    return iguales[0], "vertex_count"


def comprueba_referencia(directorio: Path, caso: str) -> tuple[Path, dict]:
    """§2: la ejecución de referencia es la fijada, o no se ejecuta.

    Su `run-record.json` tiene que dar el commit de §2 con `git_dirty: false`, y su contenedor
    la sha256 de §2. Devuelve el contenedor y lo que se guarda de ella en las entradas. Los
    mensajes no llevan rutas."""
    rr = directorio / "run-record.json"
    if not rr.is_file():
        raise SystemExit("--referencia: no hay run-record.json; no se puede atar la "
                         "ejecución de referencia a su commit (§2).")
    try:
        codigo = json.loads(rr.read_text(encoding="utf-8")).get("code") or {}
    except (json.JSONDecodeError, AttributeError) as exc:
        raise SystemExit("--referencia: run-record.json ilegible "
                         f"({type(exc).__name__}).") from None
    head = codigo.get("git_head") or ""
    if not (isinstance(head, str) and head.startswith(COMMIT_REFERENCIA[caso])):
        raise SystemExit(f"--referencia: su run-record da el commit {head[:12] or 'ninguno'}, "
                         f"no {COMMIT_REFERENCIA[caso]} (§2).")
    if codigo.get("git_dirty") is not False:
        raise SystemExit("--referencia: su run-record no declara `git_dirty: false` (§2).")
    contenedores = sorted((directorio / "export").glob("*.uos"))
    if len(contenedores) != 1:
        raise SystemExit(f"--referencia: export/ tiene {len(contenedores)} contenedores .uos "
                         "(se espera uno).")
    sha = _sha(contenedores[0])
    if sha != CONTENEDOR_SHA256[caso]:
        raise SystemExit(f"--referencia: el contenedor tiene SHA-256 {sha[:12]}…, no la de "
                         f"§2 ({CONTENEDOR_SHA256[caso][:12]}…).")
    return contenedores[0], {"reference_uos_sha256": sha,
                             "reference_run_record_sha256": _sha(rr),
                             "reference_git_head": head, "reference_git_dirty": False}


def carga_referencia(directorio: Path, contenedor: Path, de_referencia: dict) -> Datos:
    """Campo semilla y superficie del almacén; etiquetas y `reg.ct_to_ios` del `.uos`."""
    npzs = sorted((directorio / "artifacts").glob("*.npz"))
    if not npzs:
        raise SystemExit("--referencia: no hay almacén de artefactos en artifacts/.")
    T_ref, etq, meta, hashes = lee_contenedor(contenedor)
    sha_pos = (meta.get("encoding") or {}).get("source_positions_sha256")
    p_campo, campo_por = elige_campo(npzs)
    p_sup, como = elige_superficie(npzs, len(etq), sha_pos)
    with np.load(p_campo, allow_pickle=False) as z:
        campo = {"centers": z["centers"], "origin": z["origin"]}
        hu = densidad_a_hu(z["density"], z["hu_range"])
    with np.load(p_sup, allow_pickle=False) as z:
        V = z["positions"]
    if len(V) != len(etq):
        raise SystemExit(f"Las etiquetas ({len(etq)}) no corresponden a los {len(V)} vértices.")
    entradas = {
        "field_artifact": p_campo.name, "field_npz_sha256": _sha(p_campo),
        "field_matched_by": campo_por,
        "surface_artifact": p_sup.name, "surface_npz_sha256": _sha(p_sup),
        "surface_matched_by": como, **de_referencia, **hashes,
        "n_field": int(len(campo["centers"])), "n_vertices": int(len(V)),
        "n_crown_vertices": int((etq > 0).sum()),
    }
    return Datos(campo=campo, hu=hu, V=V, etq=etq, T_ref=T_ref, entradas=entradas)


def dientes_de(etq: np.ndarray) -> tuple[int, ...]:
    """§2: los códigos que traen las etiquetas. Se miden estos, sean o no los esperados."""
    return tuple(sorted(int(c) for c in np.unique(np.asarray(etq)) if c > 0))


def declara_dientes(caso: str, etq: np.ndarray) -> dict:
    encontrados = dientes_de(etq)
    return {"expected": list(DIENTES_ESPERADOS[caso]), "found": list(encontrados),
            "match_expected": encontrados == DIENTES_ESPERADOS[caso]}


# =========================================================================== #
# Disciplina: huellas, guardas, salidas
# =========================================================================== #
_tipos_de_dato = (bool, int, float, str, tuple, list, dict, set, frozenset, type(None),
                  np.ndarray, np.generic)


def _valor_json(x: Any) -> Any:
    """El valor resuelto de una constante, en JSON estable. Lo que no sea dato, error."""
    if isinstance(x, dict):
        return {str(k): _valor_json(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (set, frozenset)):
        return sorted((_valor_json(v) for v in x), key=repr)
    if isinstance(x, (list, tuple)):
        return [_valor_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _valor_json(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)                # `json` escribe el repr: sin pérdida
    if isinstance(x, (int, str)) or x is None:
        return x
    raise TypeError(f"constante no serializable: {type(x).__name__}")


def valores_resueltos(espacio: dict[str, Any], nombres: Iterable[str] | None = None,
                      excluye: Iterable[str] = ()) -> dict[str, Any]:
    """Los nombres en mayúsculas de un espacio de nombres que son datos, con su valor."""
    quiero = None if nombres is None else set(nombres)
    fuera = set(excluye)
    return {k: _valor_json(v) for k, v in sorted(espacio.items())
            if k.isupper() and k not in fuera and (quiero is None or k in quiero)
            and isinstance(v, _tipos_de_dato)}


def _huella(valores: dict) -> str:
    return hashlib.sha256(json.dumps(valores, sort_keys=True).encode()).hexdigest()


def huella_constantes() -> dict:
    """§8: sha256 de los VALORES resueltos de las constantes en mayúsculas de los módulos de
    `MODULOS_HUELLA`, de las de `precision_registro` que usa esta medida y de las propias."""
    por_modulo = {m: valores_resueltos(vars(importlib.import_module(m)))
                  for m in MODULOS_HUELLA}
    por_modulo["precision_registro"] = valores_resueltos(vars(precision_registro),
                                                         CONSTANTES_PRECISION)
    faltan = set(CONSTANTES_PRECISION) - set(por_modulo["precision_registro"])
    if faltan:
        raise SystemExit(f"precision_registro no define {sorted(faltan)}.")
    propias = valores_resueltos(globals(), excluye=FUERA_DE_HUELLA)
    return {"sha256": _huella(por_modulo),
            "per_module_sha256": {m: _huella(v) for m, v in por_modulo.items()},
            "values": por_modulo, "script_sha256": _huella(propias),
            "script_values": propias}


def checkout_principal() -> Path:
    """La copia principal del repositorio (la del `--git-common-dir`), también desde un
    worktree: allí están los datos."""
    try:
        comun = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=RAIZ,
            check=True, capture_output=True, text=True).stdout.strip()
        return Path(comun).parent
    except (subprocess.CalledProcessError, OSError):
        return RAIZ


def comprueba_anotaciones(carpeta: Path) -> None:
    """§6: no se ejecuta si hay cualquier cosa que no sea una carpeta bajo las entregas."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise SystemExit(f"{ANOTACIONES} no es una carpeta en la copia principal: no se puede "
                         "comprobar que las entregas del caso A siguen sin abrir.")
    for p in carpeta.rglob("*"):
        if not p.is_dir() or p.is_symlink():
            raise SystemExit("La carpeta de entregas del caso A ya tiene algún fichero: estas "
                             "medidas se ejecutan antes de abrir ninguna entrega (§6).")


def guarda_anotaciones(caso: str) -> dict | None:
    """§6: la carpeta fija de entregas, comprobada por el script. Solo el caso A. Se guarda la
    ruta relativa y que estaba vacía, nunca los nombres de sus subcarpetas."""
    if caso != "A":
        return None
    comprueba_anotaciones(checkout_principal() / ANOTACIONES)
    return {"path": ANOTACIONES, "empty": True}


def _revision() -> dict:
    from caso_completo import revision_de_codigo

    return revision_de_codigo()


def _entorno() -> dict:
    from caso_completo import entorno_de_software

    return entorno_de_software()


def _protocolo_sin_cambios() -> bool:
    """`git diff --quiet HEAD -- <protocolo>`: ni cambios en el árbol ni en el índice."""
    try:
        return subprocess.run(["git", "diff", "--quiet", "HEAD", "--", PROTOCOLO], cwd=RAIZ,
                              capture_output=True).returncode == 0
    except OSError:
        return False


def _blob_del_protocolo() -> str:
    return subprocess.run(["git", "rev-parse", f"HEAD:{PROTOCOLO}"], cwd=RAIZ, check=True,
                          capture_output=True, text=True).stdout.strip()


def comprueba_protocolo() -> dict:
    """§6: el documento está versionado y sin cambios; se guarda su blob de git."""
    if not _git_rastrea(PROTOCOLO):
        raise SystemExit(f"{PROTOCOLO} no está en git: se fija antes de medir (§6).")
    if not _protocolo_sin_cambios():
        raise SystemExit(f"{PROTOCOLO} tiene cambios sin confirmar: se mide desde el commit "
                         "que lo fija (§6).")
    return {"path": PROTOCOLO, "git_blob": _blob_del_protocolo()}


def salida_por_defecto(caso: str, git_head: str) -> Path:
    """`<copia principal>/data/processed/registro-navaja-arranques/<caso>-<commit[:12]>`."""
    return checkout_principal() / SALIDAS / f"{caso}-{git_head[:12]}"


def prepara_salida(ruta: Path) -> Path:
    """Crea la carpeta de salida; se niega si ya existe (una ejecución no se pisa)."""
    ruta = Path(ruta).expanduser()
    if ruta.exists():
        raise SystemExit(f"{ruta.name} ya existe: una ejecución no se pisa (§6).")
    ruta.mkdir(parents=True)
    return ruta


def escribe_etapa(salida: Path, nombre: str, datos: dict, meta: dict) -> Path:
    """Un JSON por etapa, nunca sobrescrito."""
    ruta = Path(salida) / f"{nombre}.json"
    if ruta.exists():
        raise SystemExit(f"{ruta.name} ya existe: cada etapa se escribe una vez.")
    cuerpo = {"stage": nombre, **meta, "written_at": datetime.now(UTC).isoformat(), **datos}
    ruta.write_text(json.dumps(_a_json(cuerpo), indent=1, ensure_ascii=False) + "\n")
    return ruta


def escribe_registro(salida: Path, registro: dict) -> None:
    """`run-record.json`, el único fichero que se reescribe: al empezar y al terminar."""
    ruta = Path(salida) / "run-record.json"
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_a_json(registro), indent=2, ensure_ascii=False) + "\n")
    tmp.replace(ruta)


def argv_registrado(args: argparse.Namespace) -> list[str]:
    """§8: los argumentos YA interpretados, con cada ruta sustituida por su papel. No se
    copia nada del `argv` crudo, así que ninguna forma de escribir una ruta se escapa."""
    salida = ["scripts/registro_navaja_arranques.py", "--caso", str(args.caso),
              "--referencia", "<referencia>"]
    if args.salida is not None:
        salida += ["--salida", "<salida>"]
    return salida


def analizador() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--caso", choices=("A", "B"), required=True)
    ap.add_argument("--referencia", type=Path, required=True,
                    help="Ejecución de referencia de §2: con run-record.json, artifacts/ y "
                         "export/*.uos.")
    ap.add_argument("--salida", type=Path,
                    help=f"Por defecto <copia principal>/{SALIDAS}/<caso>-<commit[:12]>.")
    return ap


def _si(valor: bool | None) -> str:
    return "indeterminado" if valor is None else ("sí" if valor else "NO")


def _progreso(texto: str) -> None:
    print(f"  · {texto}", file=sys.stderr, flush=True)


# =========================================================================== #
# Ejecución por caso
# =========================================================================== #
def ejecuta_a(datos: Datos, salida: Path, meta: dict) -> dict:
    dientes = dientes_de(datos.etq)
    corona, destino, reg, informe = prepara_a(datos.campo, datos.hu, datos.V, datos.etq)
    # Comprobaciones que pueden detener la ejecución: antes de escribir V0 (§3.1).
    if reg.arcada != "upper":
        raise SystemExit("El caso A es un maxilar y la base anatómica dice otra cosa.")
    r = reg(corona, destino)
    T0 = pose_de(r)
    v0 = float(desplazamiento(T0, datos.T_ref, corona).max())
    v0_datos = {"pass": v0 <= V0_MAX_MM, "max_mm": v0, "limit_mm": V0_MAX_MM,
                "pose": T0, "reference_pose": datos.T_ref, **_fila_registro(r),
                "preparation_report": informe, "arch": reg.arcada,
                "rot0": reg.rot0, "occlusal_plane_mm": reg.plano}
    escribe_etapa(salida, "V0", v0_datos, meta)
    print(f"· V0: máx {v0:.2e} mm (límite {V0_MAX_MM})")
    if v0 > V0_MAX_MM:
        return {"aborted": "V0", "stages": ["V0"]}

    _progreso("equivarianza")
    equiv = equivariancia_a(datos.campo, datos.hu, datos.V, datos.etq, T0, movimientos())
    e1 = evalua_e1(equiv)
    escribe_etapa(salida, "equivarianza", {"E1": e1, "expected": "T' = T∘G⁻¹",
                                           "motions": equiv}, meta)
    print(f"· equivarianza: E1 {_si(e1['pass'])} (máx {e1['max_mm']:.2e} mm)")
    if not e1["pass"]:
        return {"aborted": "E1", "stages": ["V0", "equivarianza"]}

    _progreso("navaja")
    etq_c = datos.etq[datos.etq > 0]
    nav = navaja_a(reg, corona, etq_c, destino, T0, dientes)
    n1, n2 = evalua_n1(nav["per_tooth"]["b"]), evalua_n2(nav["per_tooth"]["b"])
    escribe_etapa(salida, "navaja", {
        "full_pose": T0, "mask_note": "la máscara de (b) sale de la pose completa: circular",
        "N1": n1, "N2": n2, **nav}, meta)
    print(f"· navaja (b): N1 {_si(n1['pass'])} · N2 {_si(n2['pass'])}"
          f" · dientes que superan {N2_MAX_MM} mm: {n2['teeth_over']}"
          f" · sin contorno: {nav['teeth_without_contour']}")

    _progreso("arranques")
    arr = arranques_a(reg, corona, destino, T0)
    k1 = evalua_k1(arr)
    escribe_etapa(salida, "arranques", {"unperturbed_pose": T0, "K1": k1, "runs": arr}, meta)
    print(f"· arranques: K1 {_si(k1['pass'])} ({k1['failures']} fallos de {k1['n']}, "
          f"{k1['failures_not_converged']} sin converger)")

    criterios = {
        "N1": n1, "N2": n2, "K1": k1, "E1": e1,
        "teeth_without_contour": nav["teeth_without_contour"],
        "descriptive": {
            "variant_a_N1": evalua_n1(nav["per_tooth"]["a"]),
            "variant_a_N2": evalua_n2(nav["per_tooth"]["a"]),
            "rotations_10deg": resume_sin_criterio([f for f in arr
                                                    if f.get("angle_deg") == 10.0]),
            "heights_5mm": resume_sin_criterio([f for f in arr
                                                if abs(f.get("dz_mm", 0.0)) == 5.0]),
            "start_movement_b": {t["tooth"]: t["b"]["start_movement"] for t in nav["teeth"]},
        },
    }
    escribe_etapa(salida, "criterios", criterios, meta)
    print("· criterios: " + " · ".join(f"{k} {_si(criterios[k]['pass'])}"
                                       for k in ("N1", "N2", "K1", "E1")))
    return {"stages": ["V0", "equivarianza", "navaja", "arranques", "criterios"]}


SIN_CRITERIO_B = {"pass": None, "verdict": "sin criterio",
                  "reason": "§3.2: en el caso B la navaja es descriptiva; se lee solo contra el "
                            "suelo de semilla (§3.6)."}


def ejecuta_b(datos: Datos, salida: Path, meta: dict) -> dict:
    dientes = dientes_de(datos.etq)
    coronas = datos.V[datos.etq > 0]
    # Lo que hizo `--registro auto`: por cortes se niega con `SerieDensa` y cae al clásico.
    try:
        prepara_a(datos.campo, datos.hu, datos.V, datos.etq)
        por_cortes = "no se negó: el pipeline habría registrado por cortes"
    except SerieDensa as exc:
        por_cortes = f"SerieDensa: {exc}"
    except ValueError as exc:
        por_cortes = f"ValueError: {exc}"
    base0 = registra_b(datos.campo, datos.hu, datos.V, SEMILLA_PIPELINE)
    # Comprobaciones que pueden detener la ejecución: antes de escribir V0 (§3.1). El residuo
    # usa el plano de separación del pipeline; tiene que ser el que dio la reconstrucción.
    arbol, esmalte = arbol_esmalte_maxilar(datos.campo, datos.hu)
    if base0["occlusal_plane_mm"] is None or (float(base0["occlusal_plane_mm"])
                                              != esmalte["split_plane_mm"]):
        raise SystemExit("El plano de separación del residuo no es el que usó la "
                         "reconstrucción del registro (§3.2).")
    T = base0["pose"]
    v0 = float(desplazamiento(T, datos.T_ref, coronas).max())
    v0_datos = {"pass": v0 <= V0_MAX_MM, "max_mm": v0, "limit_mm": V0_MAX_MM,
                "reference_pose": datos.T_ref, "run": base0, "por_cortes": por_cortes}
    escribe_etapa(salida, "V0", v0_datos, meta)
    print(f"· V0: máx {v0:.2e} mm (límite {V0_MAX_MM}) · {por_cortes.split(':')[0]}")
    if v0 > V0_MAX_MM:
        return {"aborted": "V0", "stages": ["V0"]}
    # T₀ es la pose de referencia, la del contenedor; V0 garantiza que la reconstruida (T)
    # está a ≤ 0,01 mm de ella.
    T0 = datos.T_ref

    _progreso("equivarianza")
    equiv = equivariancia_b(datos.campo, datos.hu, datos.V, datos.etq, T, movimientos_b())
    e1 = evalua_e1(equiv)
    escribe_etapa(salida, "equivarianza", {"E1": e1, "expected": "G·T·G⁻¹",
                                           "motions": equiv}, meta)
    print(f"· equivarianza: E1 {_si(e1['pass'])} (máx {e1['max_mm']:.2e} mm)")
    if not e1["pass"]:
        return {"aborted": "E1", "stages": ["V0", "equivarianza"]}

    nav = navaja_b(datos.campo, datos.hu, datos.V, datos.etq, T0, arbol, dientes=dientes,
                   base0=base0, progreso=_progreso)
    completos = {m: [r for r in nav["runs"][m] if r["without"] is None] for m in nav["runs"]}
    sin_x = {m: [r for r in nav["runs"][m] if r["without"] is not None] for m in nav["runs"]}
    legible = legibilidad_b(completos["scratch"])
    suelo = suelo_b(completos["scratch"], T0, datos.V, datos.etq, dientes)
    escribe_etapa(salida, "suelo", suelo, meta)
    lectura = {m: lee_suelo(nav["per_tooth"][m], suelo["per_tooth_values_mm"])
               for m in ("scratch", "warm")}
    vueltas = {m: {"full": tasa(completos[m]), "without": tasa(sin_x[m])} for m in completos}
    escribe_etapa(salida, "navaja", {"reference_pose_T0": T0, "residual_enamel": esmalte,
                                     "readable": legible, "flip_rates": vueltas,
                                     "seed_floor_reading": lectura, **nav}, meta)
    if legible["readable"]:
        estados = [v["state"] for v in lectura["scratch"].values()]
        print("· navaja (desde cero, descriptiva): "
              + " · ".join(f"{e} {estados.count(e)}"
                           for e in ("supera", "no supera", "indeterminado")))
    else:
        print(f"· caso B no legible: {legible['flips_full_scratch']} de "
              f"{legible['n_full_scratch']} ejecuciones completas desde cero son vuelta (§3.5)")

    centro = aplica(T0, coronas).mean(axis=0)
    referencias = {r["seed"]: r for r in completos["warm"]}
    arr = arranques_b(datos.campo, datos.hu, datos.V, datos.etq, T0, referencias,
                      perturbaciones_b(centro), progreso=_progreso)
    k1 = evalua_k1(arr)
    if not legible["readable"]:
        k1 = {**k1, "pass": None, "not_readable": "§3.5: T₀ no representa al camino clásico"}
    escribe_etapa(salida, "arranques", {"rotation_centre_dicom_mm": centro,
                                        "reference": "arranque en caliente sin perturbar, "
                                                     "misma semilla",
                                        "K1": k1, "runs": arr}, meta)
    if legible["readable"]:
        print(f"· arranques: K1 {_si(k1['pass'])} ({k1['failures']} fallos de {k1['n']}, "
              f"{k1['failures_by_flip']} por vuelta, "
              f"{k1['failures_not_converged']} sin converger)")

    todas = [r for m in nav["runs"] for r in nav["runs"][m]] + list(arr)
    umbrales: dict[str, int] = {}
    for r in todas:
        clave = str(r["hu_target"])
        umbrales[clave] = umbrales.get(clave, 0) + 1
    criterios = {
        "readable": legible,
        "N1": SIN_CRITERIO_B, "N2": SIN_CRITERIO_B, "K1": k1, "E1": e1,
        "descriptive": {
            "seed_floor_reading": lectura,
            "flip_rates": vueltas,
            "flips_perturbed": sum(bool(r["run_flip"]) for r in arr),
            "rotations_10deg": resume_sin_criterio([f for f in arr
                                                    if f.get("angle_deg") == 10.0]),
            "translations_5mm": resume_sin_criterio([f for f in arr if f.get("mm") == 5.0]),
            "arbiter_thresholds": umbrales,
        },
        "reading": "Si el movimiento de X no supera el suelo de semilla, no se puede atribuir "
                   "al diente; si lo supera, es movimiento por encima del ruido de semilla, "
                   "sin causa atribuida (§5).",
    }
    escribe_etapa(salida, "criterios", criterios, meta)
    if legible["readable"]:
        print(f"· criterios: K1 {_si(k1['pass'])} · E1 {_si(e1['pass'])} · "
              "N1 y N2 sin criterio")
    return {"stages": ["V0", "equivarianza", "suelo", "navaja", "arranques", "criterios"]}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    args = analizador().parse_args(argv[1:])

    # --- Disciplina, antes de tocar ningún dato (§2, §6).
    codigo = _revision()
    if codigo["git_head"] is None or codigo["git_dirty"]:
        raise SystemExit("El árbol de código está sucio (o no hay git): la medida se ejecuta "
                         "desde un commit limpio (§6).")
    protocolo = comprueba_protocolo()
    anotaciones = guarda_anotaciones(args.caso)
    por_defecto = args.salida is None
    salida = (salida_por_defecto(args.caso, codigo["git_head"]) if por_defecto
              else args.salida.expanduser())
    esperado = f"{args.caso}-{codigo['git_head'][:12]}"
    if salida.name != esperado:
        raise SystemExit(f"La salida tiene que llamarse {esperado}: el caso y el commit van "
                         "en el nombre, para que nada se repita en silencio (§6).")
    if salida.exists():
        raise SystemExit(f"{salida.name} ya existe: una ejecución no se pisa (§6).")
    referencia = args.referencia.expanduser()
    contenedor, de_referencia = comprueba_referencia(referencia, args.caso)

    registro: dict[str, Any] = {
        "schema": "histora-run-record/1.0", "script": "scripts/registro_navaja_arranques.py",
        "case": args.caso, "status": "running", "argv": argv_registrado(args),
        "output": f"{SALIDAS}/{salida.name}" if por_defecto else "<salida>",
        "protocol": protocolo, "code": codigo, "environment": _entorno(),
        "agents": {GeometricFusionAgent.name: GeometricFusionAgent.version},
        "seeds": list(SEMILLAS_B) if args.caso == "B" else [],
        "pipeline_seed": SEMILLA_PIPELINE,
        "constants": None, "inputs": None, "teeth": None,
        "annotations_guard": anotaciones,
        "started_at": datetime.now(UTC).isoformat(), "finished_at": None,
        "scope": "Precision of the scan-to-CBCT registration (leave-one-tooth-out and "
                 "displaced starts). Not accuracy, not a TRE.",
    }
    salida = prepara_salida(salida)
    escribe_registro(salida, registro)

    # A partir de aquí todo queda en disco: un intento que falla al cargar los datos también.
    estado, tipo_error = "error", "desconocido"
    resultado: dict = {}
    try:
        datos = carga_referencia(referencia, contenedor, de_referencia)
        dientes = declara_dientes(args.caso, datos.etq)
        huellas = huella_constantes()
        registro.update({"constants": huellas, "inputs": datos.entradas, "teeth": dientes})
        escribe_registro(salida, registro)
        if not dientes["match_expected"]:
            print(f"· dientes: se esperaban {dientes['expected']} y hay {dientes['found']}; se "
                  "miden los que hay (§2).")
        meta = {"case": args.caso, "protocol": protocolo, "code": codigo, "teeth": dientes,
                "constants_sha256": huellas["sha256"],
                "script_constants_sha256": huellas["script_sha256"],
                "inputs": {k: v for k, v in datos.entradas.items() if k.endswith("sha256")}}
        resultado = (ejecuta_a if args.caso == "A" else ejecuta_b)(datos, salida, meta)
        estado, tipo_error = ("aborted" if "aborted" in resultado else "ok"), None
    except BaseException as exc:
        tipo_error = type(exc).__name__
        raise
    finally:
        registro.update(resultado)
        registro["status"] = estado
        registro["error_type"] = tipo_error
        registro["finished_at"] = datetime.now(UTC).isoformat()
        registro["outputs"] = {p.name: _sha(p) for p in sorted(salida.glob("*.json"))
                               if p.name != "run-record.json"}
        escribe_registro(salida, registro)
    if "aborted" in resultado:
        if resultado["aborted"] == "V0":
            print("V0 no se cumple: el procedimiento no reproduce el registro de referencia. "
                  "Se detiene sin medir; la causa va en una enmienda.")
        else:
            print("E1 no se cumple: es una prueba del programa. Se detiene sin calcular N1, "
                  "N2 ni K1; la causa va en una enmienda.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
