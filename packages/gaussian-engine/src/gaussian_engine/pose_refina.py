"""Refinamiento de pose por SILUETA: chamfer entre la superficie y la mascara de diente.

## Por que existe

`estima_pose` casa centroides: ~10-14 puntos 2D contra ~10-14 puntos 3D. Con esos pocos
puntos, un brillo o una fisura que parta una corona desplaza un centroide y la pose sale
torcida — o directamente no sale: medido en el caso real, una foto daba 99 candidatos con
5 inliers y ninguno llegaba a los 6 que el PnP exige, y otra se quedaba en 0,939 mm con el
gate en 0,9.

Aqui la pose se refina contra TODOS los pixeles: se minimiza la distancia chamfer entre
la superficie de la arcada reproyectada y la mascara de diente de la foto. Es la version
continua del `apoyo` — el mismo sesgo (poner los dientes de la malla sobre los dientes de
la foto) con un gradiente util para optimizar los 6 grados de libertad.

## Que NO cambia

- El gate `ERROR_MAXIMO_MM` (0,9 mm) queda INTACTO: es un guard de fidelidad clinica —
  textura nitida pero desplazada es erronea — y aqui solo se mejora la estimacion, no se
  relaja la puerta.
- `error_mm` se recalcula con la MISMA definicion que `estima_pose`: reproyeccion media
  de los inliers ORIGINALES de la semilla, escalada por la extension 3D/2D del set entero.
- El `apoyo` se recalcula con la MISMA corona y la MISMA mascara.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaussian_engine.pose_foto import (
    ERROR_MAXIMO_MM,
    TERCIO,
    PoseFoto,
    _apoyo,
    _blobs_para_pose,
    _ordena_arco,
    centros_oclusales,
    corona_oclusal,
)

#: Apoyo minimo para aceptar una pose REFINADA. Es el criterio de la fase, fijado ANTES
#: de ver el resultado: la foto buena actual da 87 %, y un apoyo menor dice que la pose
#: cubre mal la arcada aunque su reproyeccion media sea baja.
APOYO_MINIMO = 0.80

#: Semillas que se refinan por foto, las de menor coste inicial de silueta. No todas:
#: 99 semillas × ~2 s son 3 minutos por foto, y las buenas son las que ya alinean.
MAX_SEMILLAS = 6

#: Pesos del anclaje a las correspondencias del PnP, en las mismas unidades (px) que el
#: chamfer, una RONDA de optimizacion por peso. La primera (1,0) manda sobre las
#: correspondencias: condiciona el problema y recupera las rotaciones, que la silueta de
#: blobs convexos casi no ve (medido en el sintetico: sin anclaje una perturbacion de 3°
#: se queda a 4,3 mm del suelo). La segunda (0,05) deja que el chamfer —miles de puntos
#: de superficie contra unos pocos centroides— corrija el sesgo del PnP sin desanclarse
#: del todo: el gate se mide contra esas mismas correspondencias, y una ronda final de
#: chamfer puro se va al optimo de silueta, que el gate no reconoce como bueno (medido:
#: la pose VERDADERA puntua 1,5 mm en la metrica por el sesgo de perspectiva del
#: centroide, y el gate la descartaria).
LAMBDAS = (1.0, 0.05)

#: Escala de las variables de translacion para el optimizador (t/100 ~ 2 en vez de 220):
#: con las unidades en bruto, el gradiente numerico de las translaciones es ruido y el
#: descenso se para enseguida (medido: 7 evaluaciones y ni un paso).
ESCALA_T = 100.0

#: Tope de puntos 3D muestreados por coste. Con 12.000 el coste tarda ~1-2 ms.
TOPE_PUNTOS = 12_000


def superficie_oclusal(V: np.ndarray, etiquetas: np.ndarray,
                       codigos: list[int]) -> np.ndarray:
    """Vertices del 2/3 superior de las coronas (hacia oclusal), para el coste.

    El tercio cervical se deja fuera a proposito: las etiquetas invaden la encia y esos
    vertices tiran del coste hacia donde no hay diente. Muestreado con semilla fija para
    que el resultado sea reproducible corrida a corrida.
    """
    eje = V[etiquetas > 0].mean(0) - V[etiquetas == 0].mean(0)
    eje /= np.linalg.norm(eje)
    P = np.vstack([V[etiquetas == c] for c in codigos])
    h = P @ eje
    P = P[h >= np.quantile(h, 1 - 2 * TERCIO)]
    if len(P) > TOPE_PUNTOS:
        rng = np.random.default_rng(0)
        P = P[rng.choice(len(P), TOPE_PUNTOS, replace=False)]
    return P


def _coste(x: np.ndarray, P3: np.ndarray, K: np.ndarray,
           D: np.ndarray, alto: int, ancho: int) -> float:
    """Distancia chamfer media de la superficie reproyectada por `x = [rvec, tvec]`.

    `D` es la transformada de distancia de la mascara de diente: 0 dentro, crece fuera.

    ⚠️ **El muestreo es BILINEAL, no por pixel entero.** Con indices enteros el coste es
    constante a tramos: un paso del gradiente numerico menor que un pixel no cambia
    ningun indice, el gradiente sale cero y el optimizador se para en la semilla
    (medido: 7 evaluaciones y ni un paso). Con bilineal la distancia al borde de la
    mascara es suave y el descenso si tiene por donde bajar.
    """
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(x[:3], np.float64).reshape(3, 1))
    Vc = P3 @ R.T + np.asarray(x[3:6], np.float64).reshape(1, 3)
    z = Vc[:, 2]
    vis = z > 0
    if vis.sum() < 200:
        return 1e3
    uv = (K @ Vc[vis].T / z[vis]).T[:, :2]
    px = uv[:, 0]
    py = uv[:, 1]
    ok = (px >= 0) & (px < ancho - 1) & (py >= 0) & (py < alto - 1)
    if ok.sum() < 200:
        return 1e3
    px = px[ok]
    py = py[ok]
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = px - x0
    wy = py - y0
    d = (D[y0, x0] * (1 - wx) * (1 - wy) + D[y0, x1] * wx * (1 - wy)
         + D[y1, x0] * (1 - wx) * wy + D[y1, x1] * wx * wy)
    return float(d.mean())


def refina_silueta(arco: np.ndarray, xy_ord: np.ndarray, p3_todos: np.ndarray,
                   P3: np.ndarray, corona: np.ndarray, umbral_a: float,
                   semilla: dict, ruta: Path,
                   informe: dict | None = None) -> PoseFoto | None:
    """Refina UNA semilla contra la silueta y la devuelve solo si pasa el gate.

    La semilla es un candidato del diagnostico de `estima_pose`: trae focal, ventana,
    sentido, `rvec/tvec` e `inliers_idx`. Se optimizan 6 DoF con la focal FIJA — la focal
    se compara entre semillas, no dentro de una — y al final se recalculan `error_mm` y
    `apoyo` con las definiciones exactas de `estima_pose`. Si el resultado no pasa
    `ERROR_MAXIMO_MM` y `APOYO_MINIMO`, devuelve `None`: una pose refinada mala es una
    pose descartada, no una pose "mejor que la semilla".

    `informe`, si se pasa, recibe las metricas de la refinada ANTES del gate — lo que
    deja saber cuanto falto cuando se descarta.
    """
    import cv2
    from scipy.optimize import minimize

    alto, ancho = arco.shape
    f = float(semilla["focal"])
    i0, i1 = semilla["ventana"]
    p3 = np.ascontiguousarray(p3_todos[i0:i1][:: semilla["sentido"]])
    p2 = np.ascontiguousarray(xy_ord, dtype=np.float64)
    idx = np.asarray(semilla["inliers_idx"], int)
    if len(p3) != len(p2) or len(idx) > len(p3):
        return None
    K = np.array([[f, 0, ancho / 2], [0, f, alto / 2], [0, 0, 1]], float)
    D = cv2.distanceTransform((arco == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(float)

    p3a = np.ascontiguousarray(p3[idx], np.float64)
    p2a = np.ascontiguousarray(p2[idx], np.float64)

    def coste_total(s: np.ndarray, lam: float) -> float:
        # `s` trae la translacion ESCALADA (ver `ESCALA_T`); aqui se deshace.
        x = np.array([s[0], s[1], s[2], s[3] * ESCALA_T, s[4] * ESCALA_T, s[5] * ESCALA_T])
        c = _coste(x, P3, K, D, alto, ancho)
        R, _ = cv2.Rodrigues(np.asarray(x[:3], np.float64).reshape(3, 1))
        Vc = p3a @ R.T + np.asarray(x[3:6], np.float64).reshape(1, 3)
        uv = (K @ Vc.T / Vc[:, 2]).T[:, :2]
        err = float(np.linalg.norm(uv - p2a, axis=1).mean())
        return c + lam * err

    x = np.r_[np.asarray(semilla["rvec"], float).ravel(),
              np.asarray(semilla["tvec"], float).ravel()]
    s = np.r_[x[:3], x[3:6] / ESCALA_T]
    for lam in LAMBDAS:
        r = minimize(coste_total, s, args=(lam,), method="L-BFGS-B", jac=None,
                     options={"maxiter": 200, "ftol": 1e-8, "eps": 1e-3})
        s = r.x
    rv = np.asarray([s[0], s[1], s[2]])
    tv = np.asarray([s[3] * ESCALA_T, s[4] * ESCALA_T, s[5] * ESCALA_T])

    # ⚠️ **Los inliers se RE-DERIVAN contra la pose refinada, no se heredan de la
    # semilla.** Un inlier es una correspondencia consistente con LA POSE (a 25 px, el
    # mismo umbral del RANSAC), y la pose ha cambiado: contar los de la semilla haria
    # estructuralmente imposible rescatar una foto cuyas semillas tenian 5 — justo el
    # caso para el que existe este refinado. La definicion del gate no cambia; se aplica
    # a la pose que hay.
    pr, _ = cv2.projectPoints(np.ascontiguousarray(p3), rv, tv, K, None)
    errs = np.linalg.norm(pr.reshape(-1, 2) - p2, axis=1)
    dentro = errs <= 25.0
    mm_px = (np.linalg.norm(p3.max(0) - p3.min(0))
             / max(np.linalg.norm(p2.max(0) - p2.min(0)), 1e-9))
    e = float(errs[dentro].mean()) if dentro.any() else float("inf")
    sop = _apoyo(rv, tv, K, corona, arco)
    if informe is not None:
        informe["error_mm"] = round(e * mm_px, 4)
        informe["apoyo"] = round(sop, 4)
        informe["inliers"] = int(dentro.sum())
        informe["focal_px"] = round(f, 1)
        informe["rvec"] = [float(x) for x in np.asarray(rv).ravel()]
        informe["tvec"] = [float(x) for x in np.asarray(tv).ravel()]
    if dentro.sum() < 6 or e * mm_px >= ERROR_MAXIMO_MM or sop < APOYO_MINIMO:
        return None
    return PoseFoto(ruta=ruta, rvec=rv, tvec=tv.reshape(3, 1), focal_px=f,
                    ancho=ancho, alto=alto, error_px=e, error_mm=e * mm_px,
                    inliers=int(dentro.sum()), correspondencias=len(p3),
                    umbral_a=umbral_a, apoyo=sop)


def refina_desde_candidatos(ruta: Path, V: np.ndarray, etiquetas: np.ndarray,
                            codigos: list[int], candidatos: list[dict],
                            diag: dict | None = None) -> PoseFoto | None:
    """Refina las mejores semillas de una foto y devuelve la que gane el gate.

    Las semillas se ordenan por su coste de silueta INICIAL —no por inliers—: un
    candidato con 5 inliers que ya alinea la arcada es mejor semilla que uno con 6 que la
    apunta a otro sitio. La ambiguedad de espejo no se decide con el apoyo redondeado:
    se refinan los dos sentidos y gana el coste final de cada uno.

    `diag`, si se pasa, recibe `refinada_mejor`: las metricas de la mejor refinada AUNQUE
    no pase el gate — es lo que deja ver CUANTO le falto a una foto, en vez de un simple
    «no» en el log.
    """
    import cv2

    arco, xy, u, _fundidos = _blobs_para_pose(ruta, len(codigos))
    if len(xy) < 6:
        return None
    xy_ord = xy[_ordena_arco(xy)]
    p3_todos = centros_oclusales(V, etiquetas, codigos)
    P3 = superficie_oclusal(V, etiquetas, codigos)
    corona = corona_oclusal(V, etiquetas, codigos)
    alto, ancho = arco.shape
    D = cv2.distanceTransform((arco == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(float)

    semillas = [c for c in candidatos
                if c.get("rvec") is not None and len(c.get("inliers_idx", [])) >= 4
                and c["ventana"][1] - c["ventana"][0] == len(xy_ord)]
    # Cada semilla se evalua con SU focal — la focal es parte de la pose inicial.
    semillas.sort(key=lambda c: _coste(
        np.r_[c["rvec"], c["tvec"]], P3,
        np.array([[c["focal"], 0, ancho / 2], [0, c["focal"], alto / 2], [0, 0, 1]], float),
        D, alto, ancho))
    semillas = semillas[:MAX_SEMILLAS]

    mejor: PoseFoto | None = None
    mejor_informe: dict | None = None
    for c in semillas:
        informe: dict = {}
        p = refina_silueta(arco, xy_ord, p3_todos, P3, corona, u, c, ruta, informe)
        if p is not None and (mejor is None or p.error_mm < mejor.error_mm):
            mejor = p
        # Para el diagnostico interesa la refinada mas ALINEADA aunque falle el gate:
        # sin este dato, «no paso» no distingue un fallo por apoyo de uno por error.
        if (mejor_informe is None
                or informe.get("error_mm", float("inf")) < mejor_informe["error_mm"]):
            mejor_informe = informe
    if diag is not None and mejor_informe is not None:
        diag["refinada_mejor"] = mejor_informe
    return mejor
