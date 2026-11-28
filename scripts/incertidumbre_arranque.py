#!/usr/bin/env python
"""incertidumbre_arranque.py — ¿Cubre la incertidumbre que declara el registro por cortes el
error de corona verdadero del fantasma? Y la que declara para R1, el registro del caso A.

    uv run python scripts/incertidumbre_arranque.py fantasma
    uv run python scripts/incertidumbre_arranque.py r1 --referencia <ejecución R1>

Protocolo, batería, criterios y lectura: `docs/research/registro-incertidumbre-arranque.md`.
Este script solo ejecuta lo que allí está fijado, **una vez por subcomando**, desde un commit
con el árbol limpio y el documento sin cambios. Se niega a pisar una salida que ya existe y a
repetir un subcomando que ya tiene salida desde cualquier commit.

**La incertidumbre declarada** es la que `fusion_agents.por_cortes.prepara_por_cortes` deja en
su informe, redondeada como la ve el gate: la dispersión entre umbrales de contorno
(`threshold_spread_mm`) y la dispersión entre arranques girados (`start_spread_mm`). La
primaria es U = √(U_umbral² + U_arranque²) con las medianas. **Es precisión**: ninguna de las
dos ve un sesgo que compartan todas las poses.

- `fantasma`: la batería fija del protocolo sobre `fantasma_arcada`, que tiene verdad. Por
  configuración, el error verdadero E de las coronas frente a U, y si la pose devuelta es la
  misma con la medida de arranque encendida y apagada (P1). P1, P2 y P4 en `criterios.json`,
  con la razón de tiempos del fantasma solo descriptiva.
- `r1`: carga R1 como lo hace `scripts/registro_navaja_arranques.py` (sin reingerir), comprueba
  que la pose lo reproduce (V0), que la medida de arranque no la mueve (P1) y cuánto tiempo le
  añade (P3), y escribe la U declarada para R1. Se niega si hay cualquier fichero en la carpeta
  de entregas de las anotadoras: esa U se fija antes de ver ningún punto.

Si la declaración se queda encendida lo decide `decision_de_envio`, que junta las dos partes:
P1 y P4 del fantasma (`criterios.json`) y P1 y P3 de R1 (`declaracion.json`).

**Nada de la medida se imprime antes de quedar escrito**, y ninguna ruta de paciente llega a
las salidas: el `argv` se reconstruye de los argumentos ya interpretados, con cada ruta
sustituida por su papel, y el analizador no acepta abreviaturas.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import registro_navaja_arranques as rn  # noqa: E402
from consistencia_registro import _sha  # noqa: E402
from export_agents.anatomia import anatomical_frame  # noqa: E402
from fusion_agents.geometric import GeometricFusionAgent  # noqa: E402
from fusion_agents.por_cortes import (  # noqa: E402
    UMBRAL_CONTORNO,
    UMBRALES_INCERTIDUMBRE,
    banda,
    motivos,
    prepara_por_cortes,
)
from precision_registro import _git_rastrea  # noqa: E402

RESUMEN_EN = ("Runs the pre-registered check of whether the uncertainty the slice "
              "registration declares (contour threshold and start orientation) covers the "
              "true crown error on the synthetic phantom, and declares it for case A's R1.")

PROTOCOLO = "docs/research/registro-incertidumbre-arranque.md"

# =========================================================================== #
# Constantes del protocolo. El documento las fija; aquí solo se copian.
# =========================================================================== #
SALIDAS = "data/processed/incertidumbre-arranque"
"""§8: la carpeta de salidas, relativa a la copia principal; cada una lleva el commit."""
FANTASMA = "packages/fusion-agents/tests/fantasma_arcada.py"
"""El fantasma con verdad. Vive con los tests de `fusion-agents` y se carga por su ruta."""

CONTENEDOR_R1_SHA256 = "239e258e077d7b53a561740d67d8188705992dd00a3e992f38b5d05819f8034a"
COMMIT_R1 = "57eb021"
"""§2.2: R1 es `run-6` del caso A. Los mismos valores que fija la medida 3; si allí cambiaran,
este script se niega en vez de seguir al otro."""

V0_MAX_MM = 0.01
"""§3.3: desplazamiento máximo de las coronas entre la pose reconstruida y la de R1."""
P1_MAX_MM = 0.0
"""§4: la pose con la medida de arranque es la misma que sin ella, sin tolerancia."""
P2_FRACCION = 0.8
"""§4: E ≤ U en al menos esta fracción, redondeada hacia arriba, de las configuraciones de la
batería **con gradiente** en la banda (`"gradiente": True` en `BATERIA`)."""
P3_LIMITE_S = 120.0
"""§4, P3 sobre R1: segundos que la medida de arranque puede añadir a `prepara_por_cortes`
(mínimo de dos con ella menos mínimo de dos sin ella). Un presupuesto de ingeniería frente a
una ejecución del caso completo que ya tarda muchos minutos, no una prueba científica."""
ORDEN_P3 = (False, True, False, True)
"""§3.3: el orden fijo de las cuatro llamadas cronometradas en R1 (`mide_arranque`), después
de una llamada sin la medida que calienta y no se cronometra."""

MOVIMIENTO_ESCANER = {"eje": (0.3, 1.0, 0.2), "grados": 97.0,
                      "traslacion_mm": (40.0, -12.0, 80.0)}
"""§2.1: el escáner está en su propio marco, `fichero = R·mundo + t`; el de `test_anatomico.py`."""

LOSA = {"paso": 0.4, "z_desde": -3.0, "z_hasta": 9.0, "paso_z": 0.45, "quita": (2.2, 3.9),
        "losa_x": (-30.0, 30.0), "losa_y": (30.0, 46.0), "losa_z": 0.15, "hu": 2200.0,
        "origen": (3.0, -2.0, 1.5)}
"""§2.1, F1: el campo del caso A sintético de `tests/test_registro_navaja_arranques.py`, con
una losa de esmalte lejos de la arcada para que el plano salga del modo, como en el pipeline."""

_DISPERSA_08 = {"paso": 0.4, "paso_z": 0.8, "quita": (2.0, 4.0)}
_DISPERSA_045 = {"paso": 0.4, "paso_z": 0.45, "quita": (2.2, 3.9), "raices": "realistas"}
_CAPA = {**_DISPERSA_045, "esmalte": "capa", "desenfoque_mm": 0.18}

BATERIA: tuple[dict[str, Any], ...] = (
    {"id": "F1", "campo": "losa", "escaneo": {}, "plano": None, "gradiente": False},
    {"id": "F2", "campo": _DISPERSA_08, "escaneo": {}, "plano": 0.0, "gradiente": False},
    {"id": "F3", "campo": _DISPERSA_045, "escaneo": {}, "plano": 0.0, "gradiente": False},
    {"id": "F4", "campo": _CAPA, "escaneo": {}, "plano": 0.0, "gradiente": True},
    {"id": "F5", "campo": {**_CAPA, "oclusion": "sobremordida"}, "escaneo": {}, "plano": 0.0,
     "gradiente": True},
    {"id": "F6", "campo": _CAPA,
     "escaneo": {"encia": "cuello", "corona": "corta", "desborde_mm": 0.0}, "plano": 0.0,
     "gradiente": True},
    {"id": "F7", "campo": _CAPA,
     "escaneo": {"encia": "cuello", "corona": "corta", "desborde_mm": 1.0}, "plano": 0.0,
     "gradiente": True},
    {"id": "F8", "campo": _CAPA,
     "escaneo": {"encia": "cuello", "corona": "corta", "desborde_mm": 3.0}, "plano": 0.0,
     "gradiente": True},
    {"id": "F9", "campo": _DISPERSA_045,
     "escaneo": {"encia": "cuello", "corona": "corta", "desborde_mm": 3.0}, "plano": 0.0,
     "gradiente": False},
    {"id": "F10", "campo": _CAPA,
     "escaneo": {"encia": "cuello", "corona": "normal", "desborde_mm": 3.0}, "plano": 0.0,
     "gradiente": True},
    {"id": "F11", "campo": {**_DISPERSA_08, "inclinacion": {"eje": (1.0, 0.0, 0.0),
                                                          "grados": 10.0}},
     "escaneo": {}, "plano": 0.0, "gradiente": False},
    {"id": "F12", "campo": {**_DISPERSA_045, "esmalte": "cascara", "dentina": 1200.0},
     "escaneo": {}, "plano": 0.0, "gradiente": True},
)
"""§2.1: la batería fija. `campo` son los argumentos de `fantasma_arcada.campo_cbct` (o `"losa"`,
ver `LOSA`), `escaneo` los de `escaneo_superior`, y `plano` el que se pasa a
`prepara_por_cortes` (`None`: el modo del esmalte, como en el pipeline).

`gradiente` (§2.1, §4): si los cortes de la banda tienen algún vóxel entre el umbral de
contorno más bajo y el más alto (1100 ≤ HU < 1900). Sin ninguno, los tres umbrales dibujan el
mismo contorno, el registro es determinista y `threshold_spread_mm` sale 0 por construcción:
esas configuraciones se informan aparte y **no entran en P2**. Se deduce del código de
`campo_cbct`; un test lo comprueba construyendo solo los campos, sin registrar, y
`evalua_config` lo vuelve a contar sobre la banda real de cada configuración y lo informa."""


def minimo_cubiertas(n: int) -> int:
    """§4: el 80 % de `n`, redondeado hacia arriba, como entero (7 → 6)."""
    return math.ceil(round(P2_FRACCION * n, 9))


GRUPO_P2 = tuple(c["id"] for c in BATERIA if c["gradiente"])
"""§4: las configuraciones en las que se lee P2. Sale de `BATERIA`, no de una lista a mano."""
P2_MINIMO = minimo_cubiertas(len(GRUPO_P2))
"""§4: cuántas de `GRUPO_P2` tienen que quedar cubiertas: 6 de las 7."""

FUERA_DE_HUELLA = ("RAIZ",)


# =========================================================================== #
# El fantasma
# =========================================================================== #
def fantasma_arcada() -> ModuleType:
    """`fantasma_arcada`, cargado por su ruta (no es un paquete) y una sola vez."""
    if "fantasma_arcada" in sys.modules:
        return sys.modules["fantasma_arcada"]
    spec = importlib.util.spec_from_file_location("fantasma_arcada", RAIZ / FANTASMA)
    if spec is None or spec.loader is None:
        raise SystemExit(f"No se puede cargar {FANTASMA}.")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["fantasma_arcada"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


@dataclass
class Fantasma:
    """Una configuración construida. `verdad` es dónde está cada vértice de `V` en el CBCT."""

    campo: dict[str, np.ndarray]
    hu: np.ndarray
    V: np.ndarray
    etq: np.ndarray
    corona: np.ndarray
    verdad: np.ndarray
    plano: float | None


def _campo_losa(fa: ModuleType) -> tuple[dict[str, np.ndarray], np.ndarray]:
    p = LOSA
    gx, gy = np.meshgrid(np.arange(-32, 32, p["paso"]), np.arange(-16, 20, p["paso"]),
                         indexing="ij")
    xy = np.c_[gx.ravel(), gy.ravel()]
    sup, fs = fa.centros_coronas("upper")
    inf, fi = fa.centros_coronas("lower")
    capas = []
    for z in np.round(np.arange(p["z_desde"], p["z_hasta"], p["paso_z"]), 3):
        if p["quita"][0] < z < p["quita"][1]:
            continue
        q = np.c_[xy, np.full(len(xy), z)]
        capas.append(q[fa._dentro(q, sup, fs, True) | fa._dentro(q, inf, fi, False)])
    lx, ly = np.meshgrid(np.arange(*p["losa_x"], p["paso"]), np.arange(*p["losa_y"], p["paso"]),
                         indexing="ij")
    capas.append(np.c_[lx.ravel(), ly.ravel(), np.full(lx.size, p["losa_z"])])
    c = np.vstack(capas)
    origen = np.asarray(p["origen"], dtype=np.float64)
    return ({"centers": (c - origen).astype(np.float32), "origin": origen},
            np.full(len(c), p["hu"]))


def campo_de(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Solo el campo de la configuración `config`: `(campo, hu, inclinacion)`, sin escanear.

    `inclinacion` es la matriz con la que se giró la boca (la identidad si no se giró); la
    batería la guarda como eje y grados para que entre en la huella."""
    fa = fantasma_arcada()
    if config["campo"] == "losa":
        campo, hu = _campo_losa(fa)
        return campo, hu, np.eye(3)
    argumentos = dict(config["campo"])
    inclinacion = np.eye(3)
    if "inclinacion" in argumentos:
        giro = argumentos.pop("inclinacion")
        inclinacion = fa.rotacion(np.asarray(giro["eje"]), giro["grados"])
        argumentos["inclinacion"] = inclinacion
    campo, hu = fa.campo_cbct(**argumentos)
    return campo, hu, inclinacion


def construye(config: dict[str, Any]) -> Fantasma:
    """La configuración `config` de `BATERIA`, con el escáner en su propio marco."""
    fa = fantasma_arcada()
    campo, hu, inclinacion = campo_de(config)
    Vw, _, etq = fa.escaneo_superior(**config["escaneo"])
    # La corona verdadera: el mismo escaneo sin desborde (el desborde solo cambia etiquetas).
    sin_desborde = {**config["escaneo"], **({"desborde_mm": 0.0}
                                             if "desborde_mm" in config["escaneo"] else {})}
    Vw0, _, etq0 = fa.escaneo_superior(**sin_desborde)
    if not np.array_equal(Vw, Vw0):
        raise ValueError("el desborde cambió los vértices del escaneo, no solo las etiquetas")
    m = MOVIMIENTO_ESCANER
    rot = fa.rotacion(np.asarray(m["eje"]), m["grados"])
    return Fantasma(campo=campo, hu=hu,
                    V=fa.mueve(Vw, rot, np.asarray(m["traslacion_mm"])),
                    etq=np.asarray(etq, dtype=np.int64), corona=np.asarray(etq0) > 0,
                    verdad=fa.mueve(Vw, inclinacion, np.zeros(3)), plano=config["plano"])


# =========================================================================== #
# La medida
# =========================================================================== #
def entrada(V: np.ndarray, etq: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """`(corona, ejes, arcada)` que el pipeline pasa a `prepara_por_cortes`: los vértices del
    escáner con FDI > 0 (el desborde incluido) y la base anatómica de sus etiquetas."""
    base, motivo = anatomical_frame(np.asarray(V), np.asarray(etq, dtype=np.int64))
    if base is None:
        raise ValueError(f"sin base anatómica del escáner: {motivo}")
    return (np.asarray(V)[np.asarray(etq) > 0],
            np.stack([base.derecha, base.anterior, base.superior]), base.arcada)


def prepara_desde(campo: dict, hu: np.ndarray, ent: tuple[np.ndarray, np.ndarray, str], *,
                  mide_arranque: bool, plano: float | None = None) -> tuple:
    """`prepara_por_cortes` sobre una `entrada` ya calculada: lo único que cronometra P3."""
    corona, ejes, arcada = ent
    extra = {} if plano is None else {"plano": plano}
    return prepara_por_cortes(campo, hu, corona, ejes, arcada, mide_arranque=mide_arranque,
                              **extra)


def prepara(campo: dict, hu: np.ndarray, V: np.ndarray, etq: np.ndarray, *,
            mide_arranque: bool, plano: float | None = None) -> tuple:
    """Lo que hace `IngestionPipeline.prepara_registro` (`rn.prepara_a`), con el interruptor
    de la medida de arranque.

    **Repite a propósito el cuerpo de `rn.prepara_a`**, partido en `entrada` y
    `prepara_desde`: P3 necesita la base anatómica fuera del cronómetro, y P1 la llamada con
    la medida apagada, y `rn.prepara_a` no admite ninguna de las dos cosas sin tocar el script
    de una medida gastada. Donde basta la llamada del pipeline tal cual (V0 de R1) se usa
    `rn.prepara_a`. Un test comprueba, con la medida encendida y apagada, que esto da lo mismo
    que `rn.prepara_a` bit a bit."""
    return prepara_desde(campo, hu, entrada(V, etq), mide_arranque=mide_arranque, plano=plano)


def voxeles_entre_umbrales(campo: dict, hu: np.ndarray, banda_mm: Sequence[float]) -> int:
    """§2.1: vóxeles de los cortes de la banda con HU entre el umbral más bajo (incluido) y el
    más alto (excluido). Con 0, los tres umbrales dibujan el mismo contorno."""
    umbrales = (*UMBRALES_INCERTIDUMBRE, UMBRAL_CONTORNO)
    origen = np.asarray(campo.get("origin", np.zeros(3)), dtype=np.float64)
    z = np.round(np.asarray(campo["centers"], dtype=np.float64)[:, 2] + origen[2], 3)
    hu = np.asarray(hu)
    en = (z >= banda_mm[0]) & (z <= banda_mm[1])
    return int((en & (hu >= min(umbrales)) & (hu < max(umbrales))).sum())


def _componente(informe: dict, clave: str, estadistico: str) -> float | None:
    bloque = informe.get(clave)
    return None if not bloque else float(bloque[estadistico])


def incertidumbre(informe: dict) -> dict:
    """§3.2: U con las medianas (primaria), con los máximos y en altura, de lo declarado.

    Una componente que el informe no trae cuenta como 0 y se dice en `missing`."""
    salida: dict[str, Any] = {"missing": [c for c in ("threshold_spread_mm", "start_spread_mm")
                                          if not informe.get(c)]}
    for nombre, estadistico in (("U_mm", "median"), ("U_max_mm", "max"),
                                ("U_vertical_mm", "vertical")):
        th = _componente(informe, "threshold_spread_mm", estadistico) or 0.0
        st = _componente(informe, "start_spread_mm", estadistico) or 0.0
        salida[nombre] = math.hypot(th, st)
    salida["threshold_spread_mm"] = informe.get("threshold_spread_mm")
    salida["start_spread_mm"] = informe.get("start_spread_mm")
    return salida


def _identico(a: Any, b: Any) -> bool:
    """Igualdad byte a byte: arrays por tipo, forma y bytes; dicts y secuencias por elemento;
    lo demás por tipo y valor."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_identico(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_identico(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def misma_pose(con: tuple, sin: tuple) -> dict:
    """§3.2, P1: lo devuelto con la medida de arranque y sin ella, y la pose de cada uno.

    Del registrador se comparan **todos** sus campos (`dataclasses.fields`, `extra`
    incluido), no una lista escrita a mano."""
    corona, destino, reg, _ = con
    corona2, destino2, reg2, _ = sin
    identicos = {
        "origin": _identico(np.asarray(corona), np.asarray(corona2)),
        "target": _identico(np.asarray(destino), np.asarray(destino2)),
        "registrar_type": type(reg) is type(reg2),
        **{f"registrar.{c.name}": _identico(getattr(reg, c.name), getattr(reg2, c.name))
           for c in fields(reg)},
    }
    P, P2 = rn.pose_de(reg(corona, destino)), rn.pose_de(reg2(corona2, destino2))
    maximo = float(rn.desplazamiento(P, P2, corona).max())
    return {"pass": bool(all(identicos.values()) and maximo <= P1_MAX_MM),
            "max_mm": maximo, "limit_mm": P1_MAX_MM, "identical": identicos, "_pose": P}


def evalua_config(config: dict[str, Any]) -> dict:
    """§3.2: una configuración de la batería. Sin y con la medida, cronometradas.

    La base anatómica se calcula una vez, fuera del cronómetro; una llamada sin la medida,
    sin cronometrar, calienta; después se cronometra solo `prepara_por_cortes`, sin la medida
    y luego con ella, una vez cada una."""
    f = construye(config)
    ent = entrada(f.V, f.etq)
    prepara_desde(f.campo, f.hu, ent, mide_arranque=False, plano=f.plano)  # calentamiento
    t = time.perf_counter()
    sin = prepara_desde(f.campo, f.hu, ent, mide_arranque=False, plano=f.plano)
    t_sin = time.perf_counter() - t
    t = time.perf_counter()
    con = prepara_desde(f.campo, f.hu, ent, mide_arranque=True, plano=f.plano)
    t_con = time.perf_counter() - t
    corona, destino, reg, informe = con
    p1 = misma_pose(con, sin)
    P = p1.pop("_pose")
    r = reg(corona, destino)
    # La banda exacta del registro (la del informe va redondeada a 0,01 mm).
    entre = voxeles_entre_umbrales(f.campo, f.hu, banda(reg.arcada, reg.plano))
    dif = rn.aplica(P, f.V[f.corona]) - f.verdad[f.corona]
    norma = np.linalg.norm(dif, axis=1)
    error = {"n_crown_vertices": int(f.corona.sum()),
             "median_mm": float(np.median(norma)), "p95_mm": float(np.percentile(norma, 95)),
             "max_mm": float(norma.max()),
             "vertical_median_mm": float(np.median(dif[:, 2])),
             "vertical_mean_mm": float(dif[:, 2].mean())}
    u = incertidumbre(informe)
    arranque = _componente(informe, "start_spread_mm", "median") or 0.0
    return {"config": config, "registered": True, "pose": P, **rn._fila_registro(r),
            "n_labelled_crown_vertices": int(len(corona)),
            "band_voxels_between_thresholds": entre,
            "gradient_declaration_holds": (entre > 0) == bool(config["gradiente"]),
            "error": error, "uncertainty": u,
            "covered": bool(error["median_mm"] <= u["U_mm"]),
            "covered_by_start_alone": bool(error["median_mm"] <= arranque),
            "covered_max": bool(error["median_mm"] <= u["U_max_mm"]),
            "covered_vertical": bool(abs(error["vertical_median_mm"]) <= u["U_vertical_mm"]),
            "P1": p1, "seconds_without_starts": t_sin, "seconds_with_starts": t_con,
            "preparation_report": informe, "gate_reasons": motivos(informe)}


def evalua_bateria(bateria: Sequence[dict[str, Any]], salida: Path, meta: dict) -> list[dict]:
    """Cada configuración, en orden, con su JSON escrito antes de pasar a la siguiente. Una
    que lanza **cualquier** `Exception` (`SerieDensa`, otro `ValueError`, o un fallo al medir
    después de registrar) se escribe con su tipo, su mensaje y su traza, y se sigue: cuenta
    como no cubierta (§2.1). Una interrupción (`KeyboardInterrupt`, `SystemExit`) no se
    captura y termina la ejecución como error."""
    filas = []
    for config in bateria:
        try:
            fila = evalua_config(config)
        except Exception as exc:  # noqa: BLE001 — §2.1: se escribe y se sigue
            fila = {"config": config, "registered": False, "error_type": type(exc).__name__,
                    "message": str(exc), "traceback": traceback.format_exc()}
        rn.escribe_etapa(salida, config["id"], fila, meta)
        filas.append(fila)
        if fila["registered"]:
            print(f"· {config['id']}: E {fila['error']['median_mm']:.3f} mm · U "
                  f"{fila['uncertainty']['U_mm']:.3f} mm · P1 {rn._si(fila['P1']['pass'])}")
        else:
            print(f"· {config['id']}: no se registra ({fila['error_type']})")
    return filas


def criterios(filas: Sequence[dict]) -> dict:
    """§4: la parte del fantasma. P1 y P4 deciden si la declaración se queda; P2 solo la
    redacción de la línea del umbral; la razón de tiempos del fantasma es descriptiva.

    P2 se lee en las configuraciones con `gradiente` (§2.1); una que lanza cuenta como no
    cubierta dentro de su grupo. Las demás se informan aparte, sin criterio."""
    hechas = [f for f in filas if f["registered"]]
    lanzadas = [{"id": f["config"]["id"], "error_type": f["error_type"],
                 "message": f["message"]} for f in filas if not f["registered"]]
    p1 = {"pass": bool(hechas) and all(f["P1"]["pass"] for f in hechas),
          "limit_mm": P1_MAX_MM, "n": len(hechas),
          "max_mm": max((f["P1"]["max_mm"] for f in hechas), default=None),
          "failing": [f["config"]["id"] for f in hechas if not f["P1"]["pass"]]}

    grupo = [f for f in filas if f["config"]["gradiente"]]
    cubiertas = [f["config"]["id"] for f in grupo if f["registered"] and f["covered"]]
    minimo = minimo_cubiertas(len(grupo))
    p2 = {"pass": bool(grupo) and len(cubiertas) >= minimo,
          "group": "gradient: band voxels between the lowest and highest contour threshold",
          "ids": [f["config"]["id"] for f in grupo], "n": len(grupo),
          "fraction_required": P2_FRACCION, "minimum_covered": minimo,
          "covered": len(cubiertas), "covered_ids": cubiertas,
          "not_registered_count_as_uncovered": [f["config"]["id"] for f in grupo
                                                if not f["registered"]],
          "descriptive": {
              "covered_max": sum(f["covered_max"] for f in grupo if f["registered"]),
              "covered_vertical": sum(f["covered_vertical"] for f in grupo
                                      if f["registered"])}}

    def fila_sin_gradiente(f: dict) -> dict:
        if not f["registered"]:
            return {"id": f["config"]["id"], "registered": False, "covered": False}
        u = f["uncertainty"]
        return {"id": f["config"]["id"], "registered": True,
                "E_mm": f["error"]["median_mm"],
                "U_threshold_mm": (u["threshold_spread_mm"] or {}).get("median"),
                "U_start_mm": (u["start_spread_mm"] or {}).get("median"),
                "covered_by_start_alone": f["covered_by_start_alone"],
                "covered": f["covered"]}

    resto = [f for f in filas if not f["config"]["gradiente"]]
    sin_gradiente = {"criterion": None, "ids": [f["config"]["id"] for f in resto],
                     "n": len(resto),
                     "covered": sum(bool(f["registered"] and f["covered"]) for f in resto),
                     "rows": [fila_sin_gradiente(f) for f in resto]}
    grupos_declarados = {"holds": all(f["gradient_declaration_holds"] for f in hechas),
                         "mismatched": [f["config"]["id"] for f in hechas
                                        if not f["gradient_declaration_holds"]]}

    t_sin = sum(f["seconds_without_starts"] for f in hechas)
    t_con = sum(f["seconds_with_starts"] for f in hechas)
    razon = {"criterion": None,
             "note": "descriptive only: the cost criterion is P3 on R1 (declaracion.json)",
             "seconds_without": t_sin, "seconds_with": t_con,
             "factor": (t_con / t_sin) if t_sin > 0 else None,
             "per_config_factor": {f["config"]["id"]: f["seconds_with_starts"]
                                   / f["seconds_without_starts"] for f in hechas}}

    def sana(f: dict) -> bool:
        a = (f.get("preparation_report") or {}).get("start_spread_mm")
        return bool(a) and all(math.isfinite(float(a[k])) for k in ("median", "max",
                                                                     "vertical")) \
            and isinstance(a.get("not_converged"), int)

    p4 = {"pass": bool(hechas) and all(sana(f) for f in hechas), "n": len(hechas),
          "failing": [f["config"]["id"] for f in hechas if not sana(f)],
          "not_converged_starts": sum(int(f["preparation_report"]["start_spread_mm"]
                                          ["not_converged"]) for f in hechas if sana(f)),
          "base_not_converged": [f["config"]["id"] for f in hechas if not f["converged"]]}
    return {"P1": p1, "P2": p2, "P4": p4, "no_gradient_descriptive": sin_gradiente,
            "gradient_declaration": grupos_declarados,
            "phantom_time_ratio_descriptive": razon, "not_registered": lanzadas,
            "phantom_part": {
                "decides_shipping": {"P1_phantom": p1["pass"], "P4_phantom": p4["pass"]},
                "decides_wording_only": {"P2": p2["pass"],
                                         "threshold_line_must_say_precision": not p2["pass"]},
                "other_part": "declaracion.json of the r1 subcommand (P1_r1, P3_r1)",
                "rule": REGLA_DE_ENVIO}}


REGLA_DE_ENVIO = (
    "§5: la declaración se queda encendida por defecto si y solo si P1 pasa en el fantasma, "
    "P1 pasa en R1 o no se evalúa allí (y se dice), P3 pasa en R1 y P4 pasa en el fantasma. "
    "P3 sin evaluar (r1 no se ejecuta o se detiene en V0) es que no se queda. P2 solo decide "
    "la redacción de la línea del umbral.")


def decision_de_envio(fantasma: dict, r1: dict | None) -> dict:
    """§5: la decisión del commit de resultados, juntando `criterios.json` (`fantasma`) y
    `declaracion.json` (`r1`; `None` si no existe porque `r1` no se ejecutó o se detuvo en
    V0). Cada parte nombra lo que decide: `phantom_part` y `r1_part`."""
    parte = fantasma["phantom_part"]["decides_shipping"]
    de_r1 = (r1 or {}).get("r1_part", {}).get("decides_shipping", {})
    p1_r1, p3_r1 = de_r1.get("P1_r1"), de_r1.get("P3_r1")
    evaluados = (("P1_phantom", parte["P1_phantom"]), ("P4_phantom", parte["P4_phantom"]),
                 ("P1_r1", p1_r1), ("P3_r1", p3_r1))
    se_queda = (parte["P1_phantom"] is True and parte["P4_phantom"] is True
                and p1_r1 is not False and p3_r1 is True)
    declara = []
    if p1_r1 is None:
        declara.append("P1 no se evaluó en R1: esa parte se decide con el P1 del fantasma")
    if p3_r1 is None:
        declara.append("P3 no se evaluó: el coste en un caso real es desconocido y la "
                       "declaración no se queda")
    return {"ships": se_queda, "mide_arranque_default": se_queda,
            "failed": [k for k, v in evaluados if v is False],
            "not_evaluated": [k for k, v in evaluados if v is None], "declared": declara,
            "threshold_line_must_say_precision":
                fantasma["phantom_part"]["decides_wording_only"][
                    "threshold_line_must_say_precision"],
            "rule": REGLA_DE_ENVIO}


# =========================================================================== #
# R1
# =========================================================================== #
def ejecuta_r1(datos: rn.Datos, salida: Path, meta: dict) -> dict:
    # V0 con la llamada del pipeline tal cual (la medida de arranque encendida por defecto).
    con = rn.prepara_a(datos.campo, datos.hu, datos.V, datos.etq)
    corona, destino, reg, informe = con
    if reg.arcada != "upper":
        raise SystemExit("El caso A es un maxilar y la base anatómica dice otra cosa.")
    r = reg(corona, destino)
    T0 = rn.pose_de(r)
    v0 = float(rn.desplazamiento(T0, datos.T_ref, corona).max())
    rn.escribe_etapa(salida, "V0", {"pass": v0 <= V0_MAX_MM, "max_mm": v0,
                                    "limit_mm": V0_MAX_MM, "pose": T0,
                                    "reference_pose": datos.T_ref, **rn._fila_registro(r)},
                     meta)
    print(f"· V0: máx {v0:.2e} mm (límite {V0_MAX_MM})")
    if v0 > V0_MAX_MM:
        return {"aborted": "V0", "stages": ["V0"]}
    p3, sin = cronometra(datos)
    p1 = misma_pose(con, sin)
    p1.pop("_pose")
    u = incertidumbre(informe)
    rn.escribe_etapa(salida, "declaracion", {
        "P1": p1, "P3": p3, **u, "preparation_report": informe,
        "gate_reasons": motivos(informe),
        "r1_part": {"decides_shipping": {"P1_r1": p1["pass"], "P3_r1": p3["pass"]},
                    "other_part": "criterios.json of the fantasma subcommand (P1_phantom, "
                                  "P4_phantom; P2 decides the wording only)",
                    "rule": REGLA_DE_ENVIO},
        "population": "U: medianas sobre los vértices del escáner con FDI > 0 que recibe el "
                      "registro, juntando los 2 umbrales o los 8 arranques.",
        "reading": "§5: U es precisión declarada; se compara con la mediana del TRE de R1 "
                   "en la enmienda del 2026-11-28 del pre-registro del TRE."}, meta)
    print(f"· R1: U {u['U_mm']:.2f} mm (umbral {u['threshold_spread_mm']}, arranque "
          f"{u['start_spread_mm']}) · P1 {rn._si(p1['pass'])} · P3 "
          f"+{p3['extra_seconds']:.1f} s ({rn._si(p3['pass'])})")
    return {"stages": ["V0", "declaracion"]}


def p3_de(sin: Sequence[float], con: Sequence[float]) -> dict:
    """§4, P3: el mínimo con la medida menos el mínimo sin ella, frente a `P3_LIMITE_S`."""
    extra = min(con) - min(sin)
    return {"pass": bool(extra <= P3_LIMITE_S), "limit_s": P3_LIMITE_S,
            "extra_seconds": extra, "min_seconds_without": min(sin),
            "min_seconds_with": min(con), "seconds_without": list(sin),
            "seconds_with": list(con)}


def cronometra(datos: rn.Datos) -> tuple[dict, tuple]:
    """§3.3, P3 sobre R1: la base anatómica una vez, fuera del cronómetro; una llamada sin la
    medida que calienta y no se cronometra; después solo `prepara_por_cortes`, en el orden
    fijo `ORDEN_P3` (sin, con, sin, con), y de cada variante el mínimo de sus dos tiempos.

    Devuelve P3 y el último resultado sin la medida, que es el que compara P1."""
    ent = entrada(datos.V, datos.etq)
    prepara_desde(datos.campo, datos.hu, ent, mide_arranque=False)  # calentamiento
    tiempos: dict[bool, list[float]] = {False: [], True: []}
    sin: tuple = ()
    for mide in ORDEN_P3:
        t = time.perf_counter()
        resultado = prepara_desde(datos.campo, datos.hu, ent, mide_arranque=mide)
        tiempos[mide].append(time.perf_counter() - t)
        if not mide:
            sin = resultado
    p3 = p3_de(tiempos[False], tiempos[True])
    p3["order"] = ["with" if m else "without" for m in ORDEN_P3]
    p3["warm_up"] = "one untimed call without the start measurement"
    return p3, sin


# =========================================================================== #
# Disciplina: huellas, guardas, salidas
# =========================================================================== #
def huella_constantes() -> dict:
    """§8: sha256 de los valores resueltos de las constantes de los módulos que registran,
    de las del fantasma y de las de este script."""
    por_modulo = {m: rn.valores_resueltos(vars(importlib.import_module(m)))
                  for m in rn.MODULOS_HUELLA}
    por_modulo["fantasma_arcada"] = rn.valores_resueltos(vars(fantasma_arcada()))
    propias = rn.valores_resueltos(globals(), excluye=FUERA_DE_HUELLA)
    return {"sha256": rn._huella(por_modulo),
            "per_module_sha256": {m: rn._huella(v) for m, v in por_modulo.items()},
            "values": por_modulo, "script_sha256": rn._huella(propias),
            "script_values": propias}


def checkout_principal() -> Path:
    return rn.checkout_principal()


def _revision() -> dict:
    return rn._revision()


def _entorno() -> dict:
    return rn._entorno()


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


def comprueba_r1(referencia: Path) -> tuple[Path, dict]:
    """§2.2: R1 es la fijada, o no se ejecuta (commit, árbol limpio y sha256 del contenedor)."""
    if rn.CONTENEDOR_SHA256["A"] != CONTENEDOR_R1_SHA256 or rn.COMMIT_REFERENCIA["A"] != COMMIT_R1:
        raise SystemExit("La referencia del caso A de la medida 3 ya no es R1: no se sigue.")
    return rn.comprueba_referencia(referencia, "A")


def salida_de(sub: str, git_head: str) -> Path:
    """`<copia principal>/data/processed/incertidumbre-arranque/<sub>-<commit[:12]>`."""
    return checkout_principal() / SALIDAS / f"{sub}-{git_head[:12]}"


def argv_registrado(args: argparse.Namespace) -> list[str]:
    """§8: los argumentos ya interpretados, con cada ruta sustituida por su papel."""
    salida = ["scripts/incertidumbre_arranque.py", str(args.sub)]
    if args.sub == "r1":
        salida += ["--referencia", "<referencia>"]
    return salida


def analizador() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="sub", required=True)
    sub.add_parser("fantasma", allow_abbrev=False,
                   help="La batería fija sobre el fantasma: P1, P2 y P4.")
    r1 = sub.add_parser("r1", allow_abbrev=False,
                        help="La incertidumbre declarada para R1 (caso A) y P3.")
    r1.add_argument("--referencia", type=Path, required=True,
                    help="La ejecución R1: con run-record.json, artifacts/ y export/*.uos.")
    return ap


def ejecuta_fantasma(salida: Path, meta: dict) -> dict:
    filas = evalua_bateria(BATERIA, salida, meta)
    c = criterios(filas)
    rn.escribe_etapa(salida, "criterios", c, meta)
    print("· criterios: " + " · ".join(f"{k} {rn._si(c[k]['pass'])}"
                                       for k in ("P1", "P2", "P4"))
          + f" · cubiertas {c['P2']['covered']} de {c['P2']['n']} con gradiente"
          f" (mínimo {c['P2']['minimum_covered']})")
    return {"stages": [f["config"]["id"] for f in filas] + ["criterios"]}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    args = analizador().parse_args(argv[1:])

    # --- Disciplina, antes de tocar ningún dato (§6, §8).
    codigo = _revision()
    if codigo["git_head"] is None or codigo["git_dirty"]:
        raise SystemExit("El árbol de código está sucio (o no hay git): la medida se ejecuta "
                         "desde un commit limpio (§6).")
    protocolo = comprueba_protocolo()
    anotaciones = None
    if args.sub == "r1":
        rn.comprueba_anotaciones(checkout_principal() / rn.ANOTACIONES)
        anotaciones = {"path": rn.ANOTACIONES, "empty": True}
    salida = salida_de(args.sub, codigo["git_head"])
    previas = sorted(p.name for p in salida.parent.glob(f"{args.sub}-*")) \
        if salida.parent.is_dir() else []
    if salida.exists() or previas:
        raise SystemExit(f"ya existe una ejecución de `{args.sub}` ({', '.join(previas)}): "
                         "una ejecución no se pisa ni se repite desde otro commit (§6).")
    referencia = contenedor = de_referencia = None
    if args.sub == "r1":
        referencia = args.referencia.expanduser()
        contenedor, de_referencia = comprueba_r1(referencia)

    registro: dict[str, Any] = {
        "schema": "histora-run-record/1.0", "script": "scripts/incertidumbre_arranque.py",
        "subcommand": args.sub, "status": "running", "argv": argv_registrado(args),
        "output": f"{SALIDAS}/{salida.name}", "protocol": protocolo, "code": codigo,
        "environment": _entorno(),
        "agents": {GeometricFusionAgent.name: GeometricFusionAgent.version},
        "constants": None, "inputs": None, "annotations_guard": anotaciones,
        "started_at": datetime.now(UTC).isoformat(), "finished_at": None,
        "scope": "Whether the declared precision of the slice registration covers the true "
                 "crown error on a synthetic phantom; the declared value for case A's R1. "
                 "Not accuracy, not a TRE.",
    }
    salida.mkdir(parents=True)
    rn.escribe_registro(salida, registro)

    estado, tipo_error = "error", "desconocido"
    resultado: dict = {}
    try:
        huellas = huella_constantes()
        meta: dict[str, Any] = {"subcommand": args.sub, "protocol": protocolo, "code": codigo,
                                "constants_sha256": huellas["sha256"],
                                "script_constants_sha256": huellas["script_sha256"]}
        registro["constants"] = huellas
        if args.sub == "r1":
            assert referencia is not None and contenedor is not None
            datos = rn.carga_referencia(referencia, contenedor, de_referencia or {})
            registro.update({"inputs": datos.entradas,
                             "teeth": rn.declara_dientes("A", datos.etq)})
            meta["inputs"] = {k: v for k, v in datos.entradas.items() if k.endswith("sha256")}
            rn.escribe_registro(salida, registro)
            resultado = ejecuta_r1(datos, salida, meta)
        else:
            registro["inputs"] = {"phantom_sha256": _sha(RAIZ / FANTASMA)}
            meta["inputs"] = registro["inputs"]
            rn.escribe_registro(salida, registro)
            resultado = ejecuta_fantasma(salida, meta)
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
        rn.escribe_registro(salida, registro)
    if "aborted" in resultado:
        print("V0 no se cumple: el procedimiento no reproduce R1. Se detiene sin declarar; la "
              "causa va en una enmienda.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
