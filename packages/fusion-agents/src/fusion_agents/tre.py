"""Error en puntos anatómicos (TRE) de un registro CBCT↔escáner, con puntos marcados fuera.

**Por qué existe.** El residuo de ICP mide lo bien que encajan las superficies que el propio
ICP emparejó, no si una cúspide del CBCT cae sobre la misma cúspide del escáner. En el caso
real dio 0,680 mm sobre un 3,6 % de solapamiento: una cifra de aspecto excelente que no dice
nada de la arcada. El TRE se mide con puntos que **el registro no usó**, marcados por una
persona en cada modalidad.

**No hay interfaz de marcado propia.** Los puntos se marcan en 3D Slicer o en el visor que
use el especialista, y se exportan como *markups* (`.mrk.json`) o CSV. Este módulo los lee,
los empareja por etiqueta y mide.

⚠️ **La trampa de los sistemas de coordenadas.** DICOM y el `.mrk.json` de Slicer usan LPS;
el mundo de Slicer es RAS. Un STL no declara ninguno, y Slicer pregunta al cargarlo si sus
coordenadas son LPS o RAS. Equivocarse invierte los ejes x e y: el TRE sale de decenas de
milímetros y parece un mal registro cuando es un mal convenio. Por eso cada fichero declara
su sistema y quien llama declara cómo se cargó el STL; no se adivina ninguno de los dos.

Etiquetas: `<FDI>-<punto>`, por ejemplo `16-mv` o `11-i`. El FDI da la región; una etiqueta
sin FDI se mide igual y cae en la región `sin_region`.

El resultado no se escribe en ningún `.uos`: rellenar `target_registration_error_mm` exige
una versión sucesora y la identidad de quien verificó.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from fusion_agents.registration import apply, kabsch

SISTEMAS = ("LPS", "RAS")
_INVIERTE_XY = np.array([-1.0, -1.0, 1.0])


def _normaliza_sistema(sistema: str | None) -> str | None:
    if sistema is None:
        return None
    s = sistema.strip().upper()
    if s not in SISTEMAS:
        raise ValueError(f"Sistema de coordenadas desconocido: {sistema!r}")
    return s


def lee_puntos(ruta: Path, sistema: str | None = None) -> dict[str, np.ndarray]:
    """Puntos de un fichero, **en LPS**, por etiqueta.

    `.mrk.json` declara su sistema en `coordinateSystem`; si además se pasa `sistema` y no
    coincide, es un error. Un CSV `etiqueta,x,y,z` no declara nada: `sistema` es obligatorio.
    """
    pedido = _normaliza_sistema(sistema)
    ruta = Path(ruta)
    crudos: list[tuple[str, list[float]]] = []
    if ruta.name.endswith(".json"):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        marcas = datos.get("markups") or []
        if len(marcas) != 1:
            raise ValueError(f"{ruta.name}: se esperaba una lista de puntos, hay {len(marcas)}")
        declarado = _normaliza_sistema(marcas[0].get("coordinateSystem"))
        if declarado is None and pedido is None:
            raise ValueError(f"{ruta.name} no declara su sistema de coordenadas")
        if declarado is not None and pedido is not None and declarado != pedido:
            raise ValueError(f"{ruta.name} declara {declarado} y se pidió {pedido}")
        origen = declarado or pedido
        for p in marcas[0].get("controlPoints", []):
            crudos.append((str(p["label"]).strip(), [float(v) for v in p["position"]]))
    else:
        if pedido is None:
            raise ValueError(f"{ruta.name}: un CSV no declara su sistema; indícalo")
        origen = pedido
        with ruta.open(newline="", encoding="utf-8") as f:
            for fila in csv.reader(f):
                if not fila or fila[0].strip().lower() in ("", "label", "etiqueta"):
                    continue
                crudos.append((fila[0].strip(), [float(v) for v in fila[1:4]]))
    puntos: dict[str, np.ndarray] = {}
    for etiqueta, xyz in crudos:
        if etiqueta in puntos:
            raise ValueError(f"{ruta.name}: etiqueta repetida {etiqueta!r}")
        v = np.asarray(xyz, dtype=np.float64)
        puntos[etiqueta] = v * _INVIERTE_XY if origen == "RAS" else v
    return puntos


def a_coordenadas_stl(puntos_lps: dict[str, np.ndarray],
                      stl_cargado_como: str) -> dict[str, np.ndarray]:
    """Lleva puntos LPS marcados sobre un STL a las coordenadas crudas del fichero.

    Si Slicer cargó el STL como LPS, las coordenadas LPS de los puntos son las del fichero.
    Si lo cargó como RAS, el fichero está en RAS y hay que invertir x e y.
    """
    modo = _normaliza_sistema(stl_cargado_como)
    if modo == "LPS":
        return dict(puntos_lps)
    return {k: v * _INVIERTE_XY for k, v in puntos_lps.items()}


def region(etiqueta: str) -> str:
    """Cuadrante FDI y sector a partir de `<FDI>-<punto>`; `sin_region` si no hay FDI."""
    fdi = etiqueta.split("-", 1)[0].strip()
    if len(fdi) != 2 or not fdi.isdigit() or fdi[0] not in "1234" or fdi[1] not in "12345678":
        return "sin_region"
    sector = "anterior" if fdi[1] in "123" else "posterior"
    return f"Q{fdi[0]}-{sector}"


def consenso(observadores: list[dict[str, np.ndarray]]) -> tuple[dict[str, np.ndarray], dict]:
    """Media de los observadores por etiqueta y su dispersión.

    Solo entran las etiquetas que marcaron **todos**: promediar un punto que un observador
    no encontró con otro que sí, mezclaría dos acuerdos distintos. Las demás se declaran.
    """
    if not observadores:
        raise ValueError("Hacen falta puntos de al menos un observador")
    comunes = set.intersection(*(set(o) for o in observadores))
    todas = set.union(*(set(o) for o in observadores))
    media = {k: np.mean([o[k] for o in observadores], axis=0) for k in sorted(comunes)}
    dispersion: dict = {"observers": len(observadores),
                        "not_marked_by_all": sorted(todas - comunes)}
    if len(observadores) > 1:
        por_punto = {k: max(float(np.linalg.norm(a[k] - b[k]))
                            for a, b in combinations(observadores, 2))
                     for k in sorted(comunes)}
        valores = list(por_punto.values())
        dispersion.update(
            max_pairwise_mm_per_point=por_punto,
            mean_max_pairwise_mm=float(np.mean(valores)) if valores else None,
            max_pairwise_mm=float(np.max(valores)) if valores else None)
    return media, dispersion


def _resumen(errores: dict[str, float]) -> dict:
    if not errores:
        return {"n": 0}
    v = np.array(list(errores.values()))
    regiones: dict[str, list[float]] = {}
    for k, e in errores.items():
        regiones.setdefault(region(k), []).append(e)
    return {
        "n": len(v), "rms_mm": float(np.sqrt(np.mean(v ** 2))),
        "mean_mm": float(v.mean()), "max_mm": float(v.max()),
        "per_point_mm": dict(sorted(errores.items())),
        "per_region": {r: {"n": len(x), "rms_mm": float(np.sqrt(np.mean(np.square(x)))),
                           "max_mm": float(np.max(x))} for r, x in sorted(regiones.items())},
    }


def _empareja(cbct: dict[str, np.ndarray], ios: dict[str, np.ndarray]) -> tuple[list[str], dict]:
    comunes = sorted(set(cbct) & set(ios))
    return comunes, {"only_cbct": sorted(set(cbct) - set(ios)),
                     "only_ios": sorted(set(ios) - set(cbct))}


def tre_de_registro(cbct: dict[str, np.ndarray], ios: dict[str, np.ndarray],
                    transform_4x4: np.ndarray) -> dict:
    """TRE de un registro existente: TODOS los puntos son de evaluación, porque no lo ajustaron.

    `cbct` en el marco de origen del registro (coordenadas DICOM, LPS) e `ios` en el de
    destino (coordenadas del STL). La matriz lleva de uno al otro.
    """
    t = np.asarray(transform_4x4, dtype=np.float64).reshape(4, 4)
    if not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("La transformada no es afín homogénea")
    comunes, sin_pareja = _empareja(cbct, ios)
    errores = {k: float(np.linalg.norm(apply(t[:3, :3], t[:3, 3], cbct[k][None])[0] - ios[k]))
               for k in comunes}
    return {"mode": "existing_registration", "unpaired": sin_pareja, **_resumen(errores)}


def tre_por_puntos(cbct: dict[str, np.ndarray], ios: dict[str, np.ndarray],
                   ajuste: list[str]) -> dict:
    """Registro rígido por puntos de `ajuste`; TRE en los demás y *leave-one-out* en todos.

    Sirve para saber si un registro por puntos alcanzaría el criterio, no para sustituir al
    registro actual. El error en los puntos de ajuste se informa aparte como FLE: es un
    residuo, no un error de registro.
    """
    comunes, sin_pareja = _empareja(cbct, ios)
    fit = [k for k in comunes if k in set(ajuste)]
    faltan = sorted(set(ajuste) - set(comunes))
    if len(fit) < 3:
        raise ValueError(f"Hacen falta al menos 3 puntos de ajuste emparejados, hay {len(fit)}")
    rot, tr = kabsch(np.array([cbct[k] for k in fit]), np.array([ios[k] for k in fit]))

    def err(k: str, r: np.ndarray = rot, t: np.ndarray = tr) -> float:
        return float(np.linalg.norm(apply(r, t, cbct[k][None])[0] - ios[k]))

    evaluacion = {k: err(k) for k in comunes if k not in set(fit)}
    loo: dict[str, float] = {}
    if len(comunes) >= 4:
        for k in comunes:
            resto = [x for x in comunes if x != k]
            r, t = kabsch(np.array([cbct[x] for x in resto]), np.array([ios[x] for x in resto]))
            loo[k] = err(k, r, t)
    matriz = np.eye(4)
    matriz[:3, :3], matriz[:3, 3] = rot, tr
    return {
        "mode": "landmark_fit", "fit_labels": fit, "fit_labels_unpaired": faltan,
        "unpaired": sin_pareja, "transform_4x4_row_major": matriz.ravel().tolist(),
        "fiducial_residual": _resumen({k: err(k) for k in fit}),
        "evaluation": _resumen(evaluacion),
        "leave_one_out": _resumen(loo),
    }
