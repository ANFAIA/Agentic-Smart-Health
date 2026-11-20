#!/usr/bin/env python
"""precision_registro.py — evaluación preregistrada del registro anatómico escáner↔CBCT.

    uv run python scripts/precision_registro.py \\
        --cbct DIR_SERIE --principal A.stl --repetido B.stl \\
        [--post C.stl --visualizacion D.stl] [--cbct2 DIR_SERIE_2 --e3-misma-denticion si] \\
        --etiquetas DIR_ETIQUETAS \\
        --huella <huella_constantes()> --huella-protocolo <huella_protocolo()> \\
        --salida data/processed/<caso>/precision-registro-1 [--solo V0,P0-semilla,...]

    uv run python scripts/precision_registro.py --huellas     # imprime las dos huellas

Protocolo: `docs/research/registro-anatomico-protocolo.md`. Congela las constantes de este
módulo (`huella_protocolo()`) y las de `fusion_agents.anatomico` (`huella_constantes()`);
este script se niega a correr si alguna no coincide con la que se le pasa, si el árbol de
código está sucio o si el protocolo no está en git. Solo ejecuta lo que allí está fijado.

**Qué mide.** Precisión del registro anatómico (`fusion_agents.anatomico`), no exactitud.
Lo que compartan los dos caminos que se comparan —el mismo CBCT, el mismo isonivel, un
sesgo de pose común— se cancela y no aparece.

**Etapas.** Cada una escribe `<salida>/<etapa>.json` y se niega a correr si ya existe: cada
prueba se ejecuta una vez. Se pueden lanzar por separado (`--solo`) para reanudar, pero
todas las invocaciones de una salida deben llevar el mismo HEAD, las mismas huellas y las
mismas entradas, y eso se comprueba contra los `run-record*.json` anteriores.

- `V0`: el camino del pipeline (`IngestionPipeline.prepara_registro(registro="anatomico")`),
  `prepara_anatomico` y la construcción directa de este script dan la MISMA transformación
  byte a byte para el escaneo principal. Si no, se aborta: las demás etapas no medirían
  el registro que corre en el pipeline.
- `P0-semilla`, `P0-orden`, `P0-movimiento`, `P0-teselado`, `P0-etiquetas`: invariancia,
  sobre todos los escaneos dados. Si una de las cuatro primeras falla, se aborta y ninguna
  etapa de precisión corre (`P0-etiquetas` se informa aparte: atribuye el fallo al
  segmentador y no bloquea la afirmación del registro).
- `P1` robustez al arranque, `P2` sensibilidad a las etiquetas, `P3` suelo de muestreo
  (`P3a` cortes pares/impares, `P3b` tablero de la banda), sobre principal y repetido.
- `P4` (y `P4b`, en el mismo fichero aparte) la medida primaria entre PREVIO y ESCANEADO
  PREVIO («repetido»); necesita P0–P3 escritos.
- `E3` el segundo CBCT; `P5` descriptivas; `resumen` agrega los veredictos en
  `resultado.json`.

Nada de una etapa se imprime antes de que su fichero esté escrito.

**Decisiones que este script toma y el protocolo congela.**

1 · **Etiquetas por escaneo**, nunca transferidas: `--etiquetas DIR` con `DIR/<rol>/` tal
como lo deja `scripts/segmentar_fdi.py --salida DIR/<rol> --referencias-entrenamiento 18`
(un `etiquetas_<arcada>.npy` y su `fdi_<arcada>.json`, cuyo sha256 va al registro). Se
limpian UNA vez, sobre el escaneo original, con la cadena de `caso_completo.py`
(`LIMPIEZA_ETIQUETAS`); las variantes de P0 llevan esas etiquetas limpias transportadas
por coordenadas, de modo que P0 mide el registro y no la limpieza. `P0-etiquetas` es la
excepción: recalcula y limpia, de extremo a extremo.

2 · **Registros que solo necesitan la pose no calculan la incertidumbre.**
`prepara_anatomico` la calcula siempre (unos 16 reajustes). Aquí el registrador se
construye como lo construye `prepara_anatomico` —`Escaner.construye`, `marco_cbct`,
`contornos_losa`, `RegistroAnatomico`— pero solo con el contorno a `UMBRAL_CONTORNO`, y se
llama sin `incertidumbre()`. Los contornos a 1100 y 1900 solo los lee `incertidumbre`, y la
normal 2-D del de 1400 solo el residuo con signo, que no entra en S. V0 comprueba que el
atajo da los mismos bytes que el pipeline, así que no es una suposición.

3 · **P0-orden tiene dos variantes y el criterio se aplica a las dos.** `parse_stl`
deduplica con `np.unique`, que ORDENA los vértices: reescribir el STL con los vértices
permutados y reingerirlo devuelve exactamente el mismo array y no prueba nada sobre el
orden de vértices; solo sobrevive el orden de las caras (y su rotación cíclica). Por eso
cada semilla se prueba (a) reescrita en STL binario y reingerida con `MeshAgent`, como pide
el plan, y (b) permutada en memoria —vértices, normales, etiquetas y caras— y registrada
sin pasar por la ingesta, que es donde el orden de vértices llega de verdad al registro.

4 · **P1 perturba la rejilla de arranques, no la referencia anatómica.** `RegistroPerturbado`
gira TODOS los arranques (los 81 de rejilla y los de vuelta) por la misma rotación; `rot0`
—la referencia de `coherencia` para el ángulo al arranque y la detección de vueltas— no
se toca, porque perturbarla cambiaría qué se considera vuelta y no dónde empieza la
búsqueda. No se edita `anatomico.py`: es una subclase.

5 · **Legibilidad en P4** (de `consistencia_registro.py`, adaptada a etiquetas propias): una
pieza c de A se lee si `n_A(c) / (n_B(c)·|V_A|/|V_B|) ≥ LEGIBLE_COBERTURA_MIN` y la mediana
sobre sus vértices del suelo f(x) es ≤ `LEGIBLE_SUELO_MAX_MM`, con
f(x) = √(P3a(x)²/2 + P3b(x)²/2 + suelo_directo(x)²) (plan §5, P3).

6 · **Calibración en P4** sobre los vértices de BANDA de A en piezas legibles (U solo está
definida en la banda): U_A(x) de `incertidumbre()`; U_B en el vértice de banda de B más
cercano a A→B(x). «Mediana de U» es la mayor de las dos medianas.

7 · **Una vuelta** es un veto o un desplazamiento máximo de las coronas de más de
`VUELTA_MM` respecto al registro de base del mismo escaneo (corregido el movimiento).

Ningún nombre de fichero del proveedor queda escrito en las salidas: cada escaneo entra
por un enlace duro con el nombre de su rol (`<salida>/trabajo/escaneos/<rol>/`), y se
identifica por ese rol y su sha256. Las rutas no se escriben en `argv`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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

from consistencia_registro import (  # noqa: E402
    Superficie,
    _digesto_serie,
    _resumen,
    _sha,
    directo,
)
from fusion_agents.anatomico import (  # noqa: E402
    Escaner,
    RegistroAnatomico,
    contornos_losa,
    huella_constantes,
    marco_cbct,
)
from fusion_agents.por_cortes import (  # noqa: E402
    MINIMO_CONTORNO,
    UMBRAL_CONTORNO,
    pose_inicial,
)
from fusion_agents.registration import quaternion_to_matrix, rotacion_de_vector  # noqa: E402

RESUMEN_EN = ("Runs the pre-registered precision evaluation of the anatomical "
              "scan-to-CBCT registration.")

PROTOCOLO = "docs/research/registro-anatomico-protocolo.md"

# =========================================================================== #
# Constantes del protocolo. Todas entran en `huella_protocolo()`; el documento del
# protocolo las cita por su nombre.
# =========================================================================== #

# --- Ingesta: la de `caso_completo.py` y `consistencia_registro.py`.
MAX_PRIMITIVAS = 1_500_000

# --- Etiquetas.
REFERENCIAS_ENTRENAMIENTO = 18
"""`segmentar_fdi.py --referencias-entrenamiento`: el `fdi_<arcada>.json` de cada escaneo
debe listar exactamente estas referencias de pose (`pose.references`)."""
LIMPIEZA_ETIQUETAS = ("rellena_etiquetas", "absorbe_islas", "afina_fronteras",
                      "quita_motas", "rellena_huecos_interiores")
"""La cadena de `caso_completo.py`, en su orden, antes de que nadie use las etiquetas."""

# --- Vuelta (P0–P4).
VUELTA_MM = 5.0
"""Desplazamiento máximo de las coronas, respecto al registro de base del mismo escaneo, a
partir del cual un registro cuenta como vuelta (además de cualquier veto)."""

# --- Semilla de los instrumentos que sí tienen azar: el registro directo A→B
# (`consistencia_registro.directo`, ICP y barrido global) y el camino clásico de P5.
SEMILLA_INSTRUMENTOS = 0

# --- P0-semilla.
SEMILLAS_P0 = tuple(range(20))
"""`np.random.seed` y `random.seed` antes de cada registro, en el mismo proceso."""
HIJOS_P0 = ((11, 101), (12, 102))
"""Dos procesos hijos: (PYTHONHASHSEED, np.random.seed). Criterio: los 20 + 2 + el de base
dan transformaciones idénticas byte a byte (cuaternión y traslación en float64)."""

# --- P0-orden.
SEMILLAS_ORDEN = tuple(range(2001, 2011))
"""`default_rng(s)`: permutación de vértices, permutación de caras y rotación cíclica de
cada cara (conserva el sentido), en ese orden de extracción."""
P0_ORDEN_MAX_MM = 0.01

# --- P0-movimiento: 10 movimientos rígidos del fichero, las etiquetas viajan con los
# vértices (por coordenadas, tras reingerir).
SEMILLAS_SO3 = tuple(range(1001, 1007))
"""`default_rng(s)`: cuaternión uniforme (4 normales normalizadas) y traslación uniforme en
±`TRASLACION_MAX_MM` por eje, en ese orden."""
TRASLACION_MAX_MM = 100.0
GIROS_FIJOS = (("x", 90.0), ("y", 90.0), ("z", 90.0), ("z", 180.0))
"""Giros sobre los ejes DEL FICHERO, con la traslación de igual índice."""
TRASLACIONES_FIJAS_MM = ((100.0, -60.0, 30.0), (-80.0, 100.0, -100.0),
                         (40.0, 90.0, -70.0), (-100.0, -100.0, 100.0))
P0_MOVIMIENTO_MAX_MM = 0.01

# --- P0-teselado: subdivisión 1→4 por punto medio.
P0_TESELADO_MEDIANA_MM = 0.05
P0_TESELADO_P95_MM = 0.10

# --- P0-etiquetas: P0-movimiento con las etiquetas recalculadas (GPU, opcional).
P0_ETIQUETAS_CONCORDANCIA_MIN = 0.99
P0_ETIQUETAS_MEDIANA_MM = 0.1

# --- P1: la rejilla de arranques girada alrededor de las 8 direcciones de los vértices de
# un cubo, en los ejes anatómicos (derecha, anterior, superior) del CBCT.
P1_ANGULOS_GRADOS = (10.0, 20.0, 30.0)
P1_ANGULOS_CON_CRITERIO = (10.0, 20.0)
P1_DIRECCIONES = tuple((a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))
P1_MISMA_CUENCA_MM = 0.05
"""Desplazamiento MÁXIMO de las coronas respecto al registro de base."""

# --- P2: etiquetas modificadas.
P2_MEDIANA_MM = 0.2
SEXTANTES = {
    "upper": {"derecho_posterior": (14, 15, 16, 17, 18),
              "anterior": (11, 12, 13, 21, 22, 23),
              "izquierdo_posterior": (24, 25, 26, 27, 28)},
    "lower": {"derecho_posterior": (44, 45, 46, 47, 48),
              "anterior": (31, 32, 33, 41, 42, 43),
              "izquierdo_posterior": (34, 35, 36, 37, 38)},
}
ORDEN_ARCADA = {
    "upper": (18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28),
    "lower": (48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38),
}
"""Desplazamiento de códigos: cada código pasa al siguiente de esta lista (hacia la
izquierda del paciente); el último, que no tiene siguiente, pasa a 0 (encía)."""

# --- P3.
TABLERO_MM = 1.0
"""P3b: casilla del tablero 3-D en (derecha, anterior, superior) del escáner, desde su
centro; la paridad de ⌊x/1⌋+⌊y/1⌋+⌊z/1⌋ separa las dos mitades."""

# --- P4.
LEGIBLE_COBERTURA_MIN = 0.5
LEGIBLE_SUELO_MAX_MM = 0.2
OBJETIVO_MEDIANA_MM = 0.6
"""(ii) nivel «objetivo»: dos vóxeles en planta."""
OBJETIVO_P95_MM = 1.2
"""(ii): dos vóxeles más la PSF de 425 µm, redondeado."""
REFERENCIA_ANTIGUA_D50_MM = 2.14
MEJORA_D50_MM = 1.07
"""Nivel «mejora», informado aparte: la mitad de la referencia del camino antiguo."""
PIEZA_MEDIANA_MAX_MM = 1.0
"""(iii) = `DEFAULT_OVERLAP_MM`."""
DIFERENCIA_ANGULO_MAX_GRADOS = 1.0
DIFERENCIA_CENTROIDE_MAX_MM = 0.5
CALIBRACION_K = 2.0
CALIBRACION_FRACCION_MIN = 0.80
CALIBRACION_FRACCION_VACIA = 0.99
CALIBRACION_U_MEDIANA_MAX_MM = 1.0
VERTICAL_MEDIANA_MAX_MM = 0.6
DIENTES_SESGO_P4B = (11, 21, 22, 26)

# --- E3.
E3_NAVAJA_P95_MAX_MM = 0.3
E3_MEDIANA_MAX_MM = 0.8
E3_P95_MAX_MM = 1.5

# --- Etapas.
ETAPAS = ("V0", "P0-semilla", "P0-orden", "P0-movimiento", "P0-teselado", "P0-etiquetas",
          "P1", "P2", "P3", "P4", "E3", "P5", "resumen")
P0_OBLIGATORIAS = ("P0-semilla", "P0-orden", "P0-movimiento", "P0-teselado")
PRIMARIAS = ("V0", *P0_OBLIGATORIAS, "P1", "P2", "P3", "P4")

# Identificadores del caso B que ya están en el repositorio. Con `--desarrollo` (árbol
# sucio, sin huella) el script se niega a leer cualquier ruta que los contenga: hasta
# congelar el protocolo, el desarrollo solo usa el fantasma y el caso A.
_CASO_B = ("CASE-EB5070", "CASE-64A77C", "CASE-C872C6", "caso-agosto", "agosto_upper")
_NO_PROTOCOLO = ("RAIZ", "RESUMEN_EN", "PROTOCOLO", "ETAPAS", "P0_OBLIGATORIAS",
                 "PRIMARIAS")


def huella_protocolo() -> str:
    """sha256 de los valores de las constantes de este módulo que deciden la evaluación."""
    valores = {n: v for n, v in globals().items()
               if n.isupper() and not n.startswith("_") and n not in _NO_PROTOCOLO}
    texto = json.dumps(valores, sort_keys=True, default=str)
    return hashlib.sha256(texto.encode()).hexdigest()


# =========================================================================== #
# Utilidades
# =========================================================================== #
def _a_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _a_json(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_a_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _a_json(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, Path):
        return x.name
    return x


def _r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def resumen(d: np.ndarray) -> dict:
    """`_resumen` de `consistencia_registro` más el máximo."""
    salida = _resumen(np.asarray(d))
    if len(d):
        salida["max_mm"] = _r(np.max(d), 6)
    return salida


def _gira(eje: np.ndarray, grados: float) -> np.ndarray:
    eje = np.asarray(eje, dtype=np.float64)
    return rotacion_de_vector(eje / np.linalg.norm(eje) * math.radians(grados))


def _cuaternion_a_matriz(q: np.ndarray) -> np.ndarray:
    return quaternion_to_matrix(tuple(float(c) for c in q / np.linalg.norm(q)))


def _4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, t
    return m


def _angulo_4x4(m: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(m[:3, :3]) - 1) / 2, -1, 1))))


# =========================================================================== #
# CBCT, escaneos y registros
# =========================================================================== #
@dataclass
class Cbct:
    """Un campo CBCT y lo que el registro anatómico lee de él, calculado una vez."""

    centros: np.ndarray
    hu: np.ndarray
    _marcos: dict = field(default_factory=dict, repr=False)
    _contornos: dict = field(default_factory=dict, repr=False)

    @classmethod
    def de_campo(cls, campo: dict, hu: np.ndarray) -> Cbct:
        c = np.asarray(campo["centers"], dtype=np.float64)
        if "origin" in campo:
            c = c + np.asarray(campo["origin"], dtype=np.float64)
        return cls(c, np.asarray(hu))

    def prepara(self, arcada: str, umbrales: tuple[float, ...] = (UMBRAL_CONTORNO,), *,
                fresco: bool = False) -> tuple[np.ndarray, Any, dict, dict]:
        """`(base, plano, informe_marco, contornos)` como en `prepara_anatomico`.

        `fresco` lo recalcula todo, sin memoria: P0-semilla quiere ver cualquier azar que
        hubiera en el marco o en los contornos, no solo en la búsqueda.
        """
        if fresco or "marco" not in self._marcos:
            marco = marco_cbct(self.centros, self.hu)
            if fresco:
                contornos = {float(u): contornos_losa(self.centros, self.hu, u, marco[1],
                                                      arcada) for u in umbrales}
                return (*marco, contornos)
            self._marcos["marco"] = marco
        base, plano, informe = self._marcos["marco"]
        contornos = {}
        for u in umbrales:
            clave = (arcada, float(u))
            if clave not in self._contornos:
                self._contornos[clave] = contornos_losa(self.centros, self.hu, u, plano,
                                                        arcada)
            contornos[float(u)] = self._contornos[clave]
        return base, plano, informe, contornos


@dataclass
class Escaneo:
    rol: str
    V: np.ndarray
    F: np.ndarray
    N: np.ndarray
    cruda: np.ndarray
    etq: np.ndarray
    arcada: str
    sha256: str
    enlace: Path | None = None
    ref: str | None = None
    limpieza: dict = field(default_factory=dict)

    @property
    def corona(self) -> np.ndarray:
        return self.etq > 0


def ejes_de(V: np.ndarray, etq: np.ndarray) -> Any:
    from export_agents.anatomia import anatomical_frame

    base, motivo = anatomical_frame(np.asarray(V), np.asarray(etq, dtype=np.int64))
    if base is None:
        raise ValueError(motivo)
    return base


def limpia(V: np.ndarray, cruda: np.ndarray) -> tuple[np.ndarray, dict]:
    """La cadena `LIMPIEZA_ETIQUETAS` de `caso_completo.py`, en su orden."""
    from analysis_agents import (
        absorbe_islas,
        afina_fronteras,
        quita_motas,
        rellena_etiquetas,
        rellena_huecos_interiores,
    )

    V = np.asarray(V, dtype=np.float64)
    e = rellena_etiquetas(V, np.asarray(cruda, dtype=np.int64))
    e, islas = absorbe_islas(V, e)
    e, fronteras = afina_fronteras(V, e)
    e, motas = quita_motas(V, e)
    e, huecos = rellena_huecos_interiores(V, e)
    return np.asarray(e, dtype=np.int64), {
        "chain": list(LIMPIEZA_ETIQUETAS),
        "islands_absorbed": [list(map(int, x)) for x in islas],
        "boundary_vertices_moved": int(fronteras), "specks_removed": int(motas),
        "holes_filled": int(huecos)}


@dataclass(frozen=True, eq=False)
class RegistroPerturbado(RegistroAnatomico):
    """`RegistroAnatomico` con TODA la rejilla de arranques girada por `giro` (P1).

    La rotación se compone por la izquierda (en el marco del CBCT) con cada arranque, de
    rejilla y de vuelta; las traslaciones las recalcula `traslacion_inicial` como siempre.
    `rot0` no cambia: es la referencia anatómica, no el arranque.
    """

    giro: np.ndarray = field(default_factory=lambda: np.eye(3))

    def arranques(self) -> list[tuple[str, np.ndarray, float]]:
        return [(n, self.giro @ r, dz) for n, r, dz in super().arranques()]


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray
    resultado: Any
    registrador: RegistroAnatomico
    segundos: float

    def aplica(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.R.T + self.t

    def bytes_hex(self) -> str:
        r = self.resultado
        return np.asarray([*r.rotation, *r.translation], dtype=np.float64).tobytes().hex()

    def m4(self) -> np.ndarray:
        return _4x4(self.R, self.t)

    def informe(self) -> dict:
        """La búsqueda completa: arranques, cuencas, puntuaciones, márgenes y veto."""
        b = self.registrador.cache["ultima"]
        r = self.resultado

        def cuenca(c: dict) -> dict:
            salida = {k: v for k, v in c.items() if k != "pose"}
            salida["pose"] = {"R": c["pose"][0], "t": c["pose"][1]}
            return salida

        busqueda = {k: v for k, v in b.items()
                    if k not in ("result", "P", "por_corte", "basins", "chosen")}
        if isinstance(busqueda.get("ambiguous"), dict):
            busqueda["ambiguous"] = {k: v for k, v in busqueda["ambiguous"].items()
                                     if k != "pose"}
        return _a_json({
            "seconds": _r(self.segundos, 2),
            "transform_bytes_sha256": hashlib.sha256(bytes.fromhex(self.bytes_hex()))
            .hexdigest(),
            "rotation_wxyz": list(r.rotation), "translation_mm": list(r.translation),
            "veto": r.veto, "rms_mm": r.rms_mm, "rms_overlap_mm": r.rms_overlap_mm,
            "overlap_fraction": r.overlap_fraction, "iterations": r.iterations,
            "converged": r.converged,
            "chosen": cuenca(b["chosen"]),
            "basins": [cuenca(c) for c in b["basins"]],
            **busqueda,
        })


def construye(cbct: Cbct, V: np.ndarray, F: np.ndarray, N: np.ndarray | None,
              etq: np.ndarray, *, umbrales: tuple[float, ...] = (UMBRAL_CONTORNO,),
              fresco: bool = False, escaner: Escaner | None = None,
              clase: type = RegistroAnatomico, **extra: Any,
              ) -> tuple[RegistroAnatomico, np.ndarray, np.ndarray]:
    """`(registrador, origen, destino)` construidos como en `prepara_anatomico`."""
    if escaner is None:
        escaner = Escaner.construye(V, F, etq, ejes_de(V, etq), N)
    if escaner.banda.sum() < MINIMO_CONTORNO:
        raise ValueError(
            f"la banda de corona del escáner tiene solo {int(escaner.banda.sum())} vértices")
    base_cbct, plano, _, contornos = cbct.prepara(escaner.arcada, umbrales, fresco=fresco)
    pose_inicial(escaner.base, base_cbct)
    destino = contornos[float(UMBRAL_CONTORNO)][0]
    if len(destino) < MINIMO_CONTORNO:
        raise ValueError(f"solo {len(destino)} puntos de contorno en la losa del CBCT")
    reg = clase(escaner=escaner, base_cbct=base_cbct, plano=plano, contornos=contornos,
                **extra)
    return reg, escaner.V[escaner.banda], destino


def ejecuta(reg: RegistroAnatomico, origen: np.ndarray, destino: np.ndarray,
            t0: float | None = None) -> Pose:
    inicio = time.perf_counter() if t0 is None else t0
    res = reg(origen, destino)
    return Pose(quaternion_to_matrix(res.rotation), np.asarray(res.translation, float), res,
                reg, time.perf_counter() - inicio)


def registra(cbct: Cbct, V: np.ndarray, F: np.ndarray, N: np.ndarray | None,
             etq: np.ndarray, **kw: Any) -> Pose:
    """Solo la pose: sin contornos de incertidumbre ni `incertidumbre()`."""
    t0 = time.perf_counter()
    reg, origen, destino = construye(cbct, V, F, N, etq, **kw)
    return ejecuta(reg, origen, destino, t0)


def desplazamiento(a: Pose, Xa: np.ndarray, b: Pose, Xb: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a.aplica(Xa) - b.aplica(Xb), axis=1)


def es_vuelta(p: Pose, d: np.ndarray) -> bool:
    return bool(p.resultado.veto) or (len(d) > 0 and float(np.max(d)) > VUELTA_MM)


# =========================================================================== #
# Variantes del fichero (P0)
# =========================================================================== #
def rueda_caras(F: np.ndarray, giro: np.ndarray) -> np.ndarray:
    """Rotación cíclica de cada cara (a,b,c)→(b,c,a)…: conserva el sentido."""
    F = F.copy()
    for k in (1, 2):
        m = giro == k
        F[m] = np.roll(F[m], -k, axis=1)
    return F


def permutacion(semilla: int, nv: int, nf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)
    return rng.permutation(nv), rng.permutation(nf), rng.integers(0, 3, nf)


def movimientos() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Los 10 movimientos de P0-movimiento, en orden fijo."""
    salida = []
    for s in SEMILLAS_SO3:
        rng = np.random.default_rng(s)
        q = rng.normal(size=4)
        t = rng.uniform(-TRASLACION_MAX_MM, TRASLACION_MAX_MM, 3)
        salida.append((f"so3-{s}", _cuaternion_a_matriz(q), t))
    ejes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
    for (eje, grados), t in zip(GIROS_FIJOS, TRASLACIONES_FIJAS_MM, strict=True):
        salida.append((f"{eje}{grados:.0f}", _gira(ejes[eje], grados), np.asarray(t)))
    return salida


def subdivide(V: np.ndarray, F: np.ndarray, etq: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1→4 por punto medio. El punto medio lleva la etiqueta si sus extremos coinciden."""
    F = np.asarray(F, dtype=np.int64)
    aristas = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    unicas, inversa = np.unique(aristas, axis=0, return_inverse=True)
    inversa = np.asarray(inversa).ravel()
    medio = len(V) + inversa.reshape(3, -1).T          # (ab, bc, ca) de cada cara
    Vn = np.vstack([V, 0.5 * (V[unicas[:, 0]] + V[unicas[:, 1]])])
    en = np.concatenate([etq, np.where(etq[unicas[:, 0]] == etq[unicas[:, 1]],
                                       etq[unicas[:, 0]], 0)])
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    ab, bc, ca = medio[:, 0], medio[:, 1], medio[:, 2]
    Fn = np.concatenate([np.c_[a, ab, ca], np.c_[ab, b, bc], np.c_[ca, bc, c],
                         np.c_[ab, bc, ca]])
    return Vn, Fn, en


def mapea_etiquetas(V_nuevo: np.ndarray, X_escrito: np.ndarray, etiquetas: np.ndarray
                    ) -> tuple[np.ndarray, dict]:
    """Las etiquetas de `X_escrito` (vértices tal cual se escribieron, antes del float32)
    sobre los vértices reingeridos, por coincidencia EXACTA de coordenadas en float32."""
    x32 = np.asarray(X_escrito, dtype=np.float64).astype(np.float32).astype(np.float64)
    d, j = cKDTree(V_nuevo).query(x32)
    lab = np.zeros(len(V_nuevo), dtype=np.int64)
    lab[j] = etiquetas
    cubiertos = np.zeros(len(V_nuevo), dtype=bool)
    cubiertos[j] = True
    return lab, {"max_match_distance_mm": float(d.max()),
                 "merged_vertices": int(len(j) - len(np.unique(j))),
                 "unmatched_new_vertices": int((~cubiertos).sum())}


def escribe_stl(ruta: Path, V: np.ndarray, F: np.ndarray) -> str:
    from export_agents.stl import write_binary_stl

    ruta.parent.mkdir(parents=True, exist_ok=True)
    write_binary_stl(ruta, np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64),
                     header="precision_registro variante")
    return _sha(ruta)


# =========================================================================== #
# Contexto de una invocación
# =========================================================================== #
@dataclass
class Contexto:
    args: argparse.Namespace
    salida: Path
    trabajo: Path
    cbct: Cbct
    campo: dict
    hu: np.ndarray
    escaneos: dict[str, Escaneo]
    ingesta: Callable[[Path], tuple[np.ndarray, np.ndarray, np.ndarray, str | None]]
    pipe: Any = None
    resultado: Any = None          # PipelineResult de la ingesta del CBCT + principal
    registro_ejecucion: str = ""
    _base: dict = field(default_factory=dict)

    def base(self, rol: str) -> Pose:
        """El registro de base del escaneo (solo pose), una vez por invocación."""
        if rol not in self._base:
            e = self.escaneos[rol]
            self._base[rol] = registra(self.cbct, e.V, e.F, e.N, e.etq)
        return self._base[rol]

    def con_repetido(self) -> list[str]:
        return [r for r in ("principal", "repetido") if r in self.escaneos]


def _meta(ctx: Contexto, etapa: str) -> dict:
    return {"stage": etapa, "protocol": PROTOCOLO, "run_record": ctx.registro_ejecucion,
            "constants_sha256": huella_constantes(), "protocol_sha256": huella_protocolo(),
            "development_run": bool(ctx.args.desarrollo),
            "finished_at": datetime.now(UTC).isoformat()}


def escribe_etapa(ctx: Contexto, etapa: str, datos: dict) -> Path:
    ruta = ctx.salida / f"{etapa}.json"
    if ruta.exists():
        raise SystemExit(f"{ruta.name} ya existe: cada prueba se ejecuta una vez.")
    texto = json.dumps(_a_json({**_meta(ctx, etapa), **datos}), indent=1,
                       ensure_ascii=False) + "\n"
    ruta.write_text(texto)
    return ruta


def lee_etapa(salida: Path, etapa: str) -> dict | None:
    ruta = salida / f"{etapa}.json"
    return json.loads(ruta.read_text()) if ruta.exists() else None


def _base_comun(ctx: Contexto, rol: str) -> dict:
    """La huella de la pose de base: debe ser la misma en todas las etapas."""
    return {"role": rol, "base_transform_hex": ctx.base(rol).bytes_hex()}


# =========================================================================== #
# V0
# =========================================================================== #
def etapa_v0(ctx: Contexto) -> dict:
    """Pipeline, `prepara_anatomico` y el atajo de este script: los mismos bytes."""
    from fusion_agents.anatomico import prepara_anatomico

    e = ctx.escaneos["principal"]
    t0 = time.perf_counter()
    preparado = ctx.pipe.prepara_registro(ctx.resultado, malla=e.enlace, dos_arcadas=True,
                                          etiquetas_origen=e.etq, registro="anatomico")
    if preparado is None:
        raise SystemExit("V0: el pipeline no preparó el registro (¿sin malla?).")
    o_p, d_p, inf_p = preparado
    if inf_p.get("method") != "fusion_agents.anatomico":
        raise SystemExit("V0: el pipeline no usó el registro anatómico: "
                         + str(inf_p.get("sin_anatomico")))
    res_p = inf_p["registrador"](o_p, d_p)
    seg_p = time.perf_counter() - t0

    t0 = time.perf_counter()
    o_d, d_d, reg_d, inf_d = prepara_anatomico(
        ctx.campo, ctx.hu, e.V, e.F, e.etq, ejes_de(e.V, e.etq), normales=e.N,
        arcada_nombre="maxilar" if e.arcada == "upper" else "mandibular")
    res_d = reg_d(o_d, d_d)
    seg_d = time.perf_counter() - t0

    base = ctx.base("principal")

    def hexa(r: Any) -> str:
        return np.asarray([*r.rotation, *r.translation], dtype=np.float64).tobytes().hex()

    iguales = {
        "pipeline_vs_prepara_anatomico": hexa(res_p) == hexa(res_d),
        "prepara_anatomico_vs_script": hexa(res_d) == base.bytes_hex(),
        "source_identical": bool(np.array_equal(o_p, o_d)
                                 and np.array_equal(o_d, base.registrador.escaner.V[
                                     base.registrador.escaner.banda])),
        "target_identical": bool(np.array_equal(d_p, d_d)),
    }
    return {
        "pass": all(iguales.values()),
        "checks": iguales,
        "transform_hex": {"pipeline": hexa(res_p), "prepara_anatomico": hexa(res_d),
                          "script": base.bytes_hex()},
        "seconds": {"pipeline_with_uncertainty": _r(seg_p, 1),
                    "prepara_anatomico_with_uncertainty": _r(seg_d, 1),
                    "script_pose_only": _r(base.segundos, 1)},
        "report_prepara_anatomico": inf_d,
        "search_script": base.informe(),
    }


# =========================================================================== #
# P0
# =========================================================================== #
def _hijo(npz: Path, destino: Path, semilla: int) -> None:
    """Proceso hijo de P0-semilla: reconstruye todo desde los arrays y registra."""
    np.random.seed(semilla)
    random.seed(semilla)
    a = np.load(npz)
    cbct = Cbct(a["centros"], a["hu"])
    p = registra(cbct, a["V"], a["F"], a["N"], a["etq"], fresco=True)
    destino.write_text(json.dumps({"transform_hex": p.bytes_hex(),
                                   "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                                   "numpy_seed": semilla, "seconds": p.segundos}))


def etapa_p0_semilla(ctx: Contexto) -> dict:
    por_escaneo = {}
    for rol, e in ctx.escaneos.items():
        base = ctx.base(rol)
        filas = []
        for s in SEMILLAS_P0:
            np.random.seed(s)
            random.seed(s)
            p = registra(ctx.cbct, e.V, e.F, e.N, e.etq, fresco=True)
            filas.append({"seed": s, "transform_hex": p.bytes_hex(),
                          "seconds": _r(p.segundos, 1), "veto": p.resultado.veto,
                          "search": p.informe() if s == SEMILLAS_P0[0] else None})
        npz = ctx.trabajo / f"p0-semilla-{rol}.npz"
        np.savez(npz, centros=ctx.cbct.centros, hu=ctx.cbct.hu, V=e.V, F=e.F, N=e.N,
                 etq=e.etq)
        hijos = []
        for hashseed, semilla in HIJOS_P0:
            out = ctx.trabajo / f"p0-semilla-{rol}-hijo-{hashseed}.json"
            entorno = dict(os.environ, PYTHONHASHSEED=str(hashseed))
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_hijo",
                            str(npz), str(out), str(semilla)], env=entorno, check=True,
                           cwd=RAIZ)
            hijos.append(json.loads(out.read_text()))
        npz.unlink()
        todas = [base.bytes_hex()] + [f["transform_hex"] for f in filas] + \
                [h["transform_hex"] for h in hijos]
        por_escaneo[rol] = {
            "base_transform_hex": base.bytes_hex(),
            "identical": len(set(todas)) == 1,
            "n_distinct": len(set(todas)),
            "flips": sum(1 for f in filas if f["veto"]) + (1 if base.resultado.veto else 0),
            "runs": filas, "children": hijos,
        }
    return {"pass": all(v["identical"] for v in por_escaneo.values()),
            "criterion": "las 20 semillas, los 2 procesos hijos y el registro de base dan "
                         "bytes idénticos",
            "flips": sum(v["flips"] for v in por_escaneo.values()),
            "scans": por_escaneo}


def etapa_p0_orden(ctx: Contexto) -> dict:
    por_escaneo = {}
    for rol, e in ctx.escaneos.items():
        base = ctx.base(rol)
        X = e.V[e.corona]
        filas = []
        for s in SEMILLAS_ORDEN:
            p, fp, giro = permutacion(s, len(e.V), len(e.F))
            inv = np.argsort(p)
            Vp, Np, Ep = e.V[p], e.N[p], e.etq[p]
            Fp = rueda_caras(inv[e.F][fp], giro)
            # (b) en memoria: el orden de vértices llega al registro.
            pm = registra(ctx.cbct, Vp, Fp, Np, Ep)
            dm = desplazamiento(pm, X, base, X)
            # (a) STL binario reescrito y reingerido.
            ruta = ctx.trabajo / "p0-orden" / rol / f"{s}" / f"escaneo-{_nombre_arcada(e)}.stl"
            sha = escribe_stl(ruta, Vp, Fp)
            V2, F2, N2, _ = ctx.ingesta(ruta)
            lab, mapeo = mapea_etiquetas(V2, Vp, Ep)
            ps = registra(ctx.cbct, V2, F2, N2, lab)
            ds = desplazamiento(ps, X, base, X)
            ruta.unlink()
            filas.append({
                "seed": s, "stl_sha256": sha, "label_mapping": mapeo,
                "in_memory": {"crown_displacement": resumen(dm), "veto": pm.resultado.veto,
                              "flip": es_vuelta(pm, dm), "seconds": _r(pm.segundos, 1),
                              "search": pm.informe()},
                "stl_reingested": {"crown_displacement": resumen(ds),
                                   "veto": ps.resultado.veto, "flip": es_vuelta(ps, ds),
                                   "seconds": _r(ps.segundos, 1), "search": ps.informe()},
                "pass": bool(dm.max() <= P0_ORDEN_MAX_MM and ds.max() <= P0_ORDEN_MAX_MM),
            })
        por_escaneo[rol] = {
            **_base_comun(ctx, rol),
            "pass": all(f["pass"] for f in filas),
            "flips": sum(f["in_memory"]["flip"] + f["stl_reingested"]["flip"] for f in filas),
            "max_crown_displacement_mm": max(
                max(f["in_memory"]["crown_displacement"]["max_mm"],
                    f["stl_reingested"]["crown_displacement"]["max_mm"]) for f in filas),
            "runs": filas,
        }
    return {"pass": all(v["pass"] for v in por_escaneo.values()),
            "criterion": f"máx. desplazamiento de las coronas ≤ {P0_ORDEN_MAX_MM} mm en las "
                         "dos variantes de cada semilla",
            "flips": sum(v["flips"] for v in por_escaneo.values()),
            "scans": por_escaneo}


def _nombre_arcada(e: Escaneo) -> str:
    return "superior" if e.arcada == "upper" else "inferior"


def _escaneo_movido(ctx: Contexto, e: Escaneo, nombre: str, R: np.ndarray, t: np.ndarray,
                    carpeta: str) -> tuple[Path, np.ndarray, tuple, dict, str]:
    X = e.V @ R.T + t
    ruta = ctx.trabajo / carpeta / e.rol / nombre / f"escaneo-{_nombre_arcada(e)}.stl"
    sha = escribe_stl(ruta, X, e.F)
    ingerido = ctx.ingesta(ruta)
    lab, mapeo = mapea_etiquetas(ingerido[0], X, e.etq)
    return ruta, X, ingerido, {"labels": lab, "mapping": mapeo}, sha


def etapa_p0_movimiento(ctx: Contexto) -> dict:
    por_escaneo = {}
    for rol, e in ctx.escaneos.items():
        base = ctx.base(rol)
        X0 = e.V[e.corona]
        filas = []
        for nombre, R, t in movimientos():
            ruta, X, (V2, F2, N2, _), m, sha = _escaneo_movido(ctx, e, nombre, R, t,
                                                               "p0-movimiento")
            p = registra(ctx.cbct, V2, F2, N2, m["labels"])
            d = desplazamiento(p, X[e.corona], base, X0)
            ruta.unlink()
            filas.append({"motion": nombre, "rotation": R, "translation_mm": t,
                          "stl_sha256": sha, "label_mapping": m["mapping"],
                          "crown_displacement": resumen(d), "veto": p.resultado.veto,
                          "flip": es_vuelta(p, d), "seconds": _r(p.segundos, 1),
                          "search": p.informe(),
                          "pass": bool(d.max() <= P0_MOVIMIENTO_MAX_MM
                                       and not p.resultado.veto)})
        por_escaneo[rol] = {**_base_comun(ctx, rol),
                            "pass": all(f["pass"] for f in filas),
                            "flips": sum(f["flip"] for f in filas), "runs": filas}
    return {"pass": all(v["pass"] for v in por_escaneo.values()),
            "criterion": f"deshecho el movimiento, máx. desplazamiento de las coronas ≤ "
                         f"{P0_MOVIMIENTO_MAX_MM} mm y ningún veto",
            "flips": sum(v["flips"] for v in por_escaneo.values()),
            "scans": por_escaneo}


def etapa_p0_teselado(ctx: Contexto) -> dict:
    por_escaneo = {}
    for rol, e in ctx.escaneos.items():
        base = ctx.base(rol)
        X = e.V[e.corona]
        Vn, Fn, En = subdivide(e.V, e.F, e.etq)
        ruta = ctx.trabajo / "p0-teselado" / rol / f"escaneo-{_nombre_arcada(e)}.stl"
        sha = escribe_stl(ruta, Vn, Fn)
        V2, F2, N2, _ = ctx.ingesta(ruta)
        lab, mapeo = mapea_etiquetas(V2, Vn, En)
        p = registra(ctx.cbct, V2, F2, N2, lab)
        d = desplazamiento(p, X, base, X)
        ruta.unlink()
        por_escaneo[rol] = {
            **_base_comun(ctx, rol), "stl_sha256": sha, "label_mapping": mapeo,
            "n_vertices": [int(len(e.V)), int(len(V2))],
            "crown_displacement": resumen(d), "veto": p.resultado.veto,
            "flip": es_vuelta(p, d), "seconds": _r(p.segundos, 1), "search": p.informe(),
            "pass": bool(np.median(d) <= P0_TESELADO_MEDIANA_MM
                         and np.percentile(d, 95) <= P0_TESELADO_P95_MM),
        }
    return {"pass": all(v["pass"] for v in por_escaneo.values()),
            "criterion": f"mediana ≤ {P0_TESELADO_MEDIANA_MM} mm y p95 ≤ "
                         f"{P0_TESELADO_P95_MM} mm",
            "flips": sum(v["flip"] for v in por_escaneo.values()),
            "scans": por_escaneo}


def etapa_p0_etiquetas(ctx: Contexto) -> dict:
    """P0-movimiento con las etiquetas recalculadas por `segmentar_fdi.py` (GPU)."""
    python_gpu = Path(ctx.args.python_gpu).expanduser()
    if not python_gpu.exists():
        raise SystemExit(f"P0-etiquetas necesita el entorno de GPU ({python_gpu}).")
    por_escaneo = {}
    for rol, e in ctx.escaneos.items():
        base = ctx.base(rol)
        X0 = e.V[e.corona]
        filas = []
        for nombre, R, t in movimientos():
            ruta, X, (V2, F2, N2, _), m, sha = _escaneo_movido(ctx, e, nombre, R, t,
                                                               "p0-etiquetas")
            cruda_mapeada, _ = mapea_etiquetas(V2, X, e.cruda)
            subprocess.run([str(python_gpu), str(RAIZ / "scripts/segmentar_fdi.py"),
                            "--escaneo", str(ruta), "--arcada", e.arcada,
                            "--salida", str(ruta.parent),
                            "--referencias-entrenamiento", str(REFERENCIAS_ENTRENAMIENTO)],
                           check=True, cwd=RAIZ, stdout=subprocess.DEVNULL)
            nueva = np.load(ruta.parent / f"etiquetas_{e.arcada}.npy").astype(np.int64)
            if len(nueva) != len(V2):
                raise SystemExit(f"P0-etiquetas: {rol}/{nombre} etiquetas de otro tamaño.")
            concordancia = float(np.mean(nueva == cruda_mapeada))
            limpias, _ = limpia(V2, nueva)
            try:
                p = registra(ctx.cbct, V2, F2, N2, limpias)
            except ValueError as exc:
                filas.append({"motion": nombre, "label_agreement": concordancia,
                              "error": str(exc), "pass": False, "flip": False})
                continue
            d = desplazamiento(p, X[e.corona], base, X0)
            ruta.unlink()
            filas.append({"motion": nombre, "stl_sha256": sha,
                          "label_agreement": _r(concordancia, 5),
                          "crown_displacement": resumen(d), "veto": p.resultado.veto,
                          "flip": es_vuelta(p, d), "search": p.informe(),
                          "pass": bool(concordancia >= P0_ETIQUETAS_CONCORDANCIA_MIN
                                       and np.median(d) <= P0_ETIQUETAS_MEDIANA_MM)})
        por_escaneo[rol] = {**_base_comun(ctx, rol),
                            "pass": all(f["pass"] for f in filas),
                            "flips": sum(f["flip"] for f in filas), "runs": filas}
    return {"pass": all(v["pass"] for v in por_escaneo.values()),
            "criterion": f"concordancia por vértice ≥ {P0_ETIQUETAS_CONCORDANCIA_MIN} y "
                         f"mediana del desplazamiento ≤ {P0_ETIQUETAS_MEDIANA_MM} mm",
            "scope": "atribuye el fallo al segmentador: no bloquea la afirmación del "
                     "registro, sí la de extremo a extremo",
            "flips": sum(v["flips"] for v in por_escaneo.values()),
            "scans": por_escaneo}


# =========================================================================== #
# P1, P2, P3
# =========================================================================== #
def etapa_p1(ctx: Contexto) -> dict:
    por_escaneo = {}
    for rol in ctx.con_repetido():
        e = ctx.escaneos[rol]
        base = ctx.base(rol)
        reg0 = base.registrador
        X = e.V[e.corona]
        origen = reg0.escaner.V[reg0.escaner.banda]
        destino = reg0.contornos[float(UMBRAL_CONTORNO)][0]
        filas = []
        for grados in P1_ANGULOS_GRADOS:
            for direccion in P1_DIRECCIONES:
                eje = reg0.base_cbct.T @ np.asarray(direccion, dtype=np.float64)
                giro = _gira(eje, grados)
                reg = RegistroPerturbado(escaner=reg0.escaner, base_cbct=reg0.base_cbct,
                                         plano=reg0.plano, contornos=reg0.contornos,
                                         giro=giro)
                p = ejecuta(reg, origen, destino)
                d = desplazamiento(p, X, base, X)
                filas.append({"angle_deg": grados, "direction_anatomical": direccion,
                              "crown_displacement": resumen(d), "veto": p.resultado.veto,
                              "flip": es_vuelta(p, d),
                              "same_basin": bool(d.max() <= P1_MISMA_CUENCA_MM),
                              "seconds": _r(p.segundos, 1), "search": p.informe()})
        con_criterio = [f for f in filas if f["angle_deg"] in P1_ANGULOS_CON_CRITERIO]
        por_escaneo[rol] = {
            **_base_comun(ctx, rol),
            "pass": all(f["same_basin"] for f in con_criterio),
            "same_basin_by_angle": {
                str(g): sum(f["same_basin"] for f in filas if f["angle_deg"] == g)
                for g in P1_ANGULOS_GRADOS},
            "flips": sum(f["flip"] for f in filas), "runs": filas}
    return {"pass": all(v["pass"] for v in por_escaneo.values()),
            "criterion": f"el 100 % de las perturbaciones de {P1_ANGULOS_CON_CRITERIO}° "
                         f"acaban a ≤ {P1_MISMA_CUENCA_MM} mm (máx. de coronas) de la base; "
                         "30° se informa",
            "flips": sum(v["flips"] for v in por_escaneo.values()),
            "scans": por_escaneo}


def etiquetas_p2(etq: np.ndarray, arcada: str) -> list[tuple[str, np.ndarray]]:
    salida = []
    for nombre, codigos in SEXTANTES[arcada].items():
        salida.append((f"sin_{nombre}", np.where(np.isin(etq, codigos), 0, etq)))
    orden = ORDEN_ARCADA[arcada]
    siguiente = {c: (orden[i + 1] if i + 1 < len(orden) else 0) for i, c in enumerate(orden)}
    desplazada = np.array([siguiente.get(int(c), 0) if c > 0 else 0 for c in etq],
                          dtype=np.int64)
    salida.append(("codigos_desplazados_un_diente", desplazada))
    return salida


def etapa_p2(ctx: Contexto) -> dict:
    por_escaneo = {}
    for rol in ctx.con_repetido():
        e = ctx.escaneos[rol]
        base = ctx.base(rol)
        X = e.V[e.corona]
        cod = e.etq[e.corona]
        filas = []
        for nombre, mod in etiquetas_p2(e.etq, e.arcada):
            try:
                p = registra(ctx.cbct, e.V, e.F, e.N, mod)
            except ValueError as exc:
                filas.append({"modification": nombre, "error": str(exc), "pass": False,
                              "flip": False})
                continue
            d = desplazamiento(p, X, base, X)
            filas.append({
                "modification": nombre, "crown_displacement": resumen(d),
                "per_tooth_median_mm": {int(c): _r(np.median(d[cod == c]))
                                        for c in np.unique(cod)},
                "veto": p.resultado.veto, "flip": es_vuelta(p, d),
                "seconds": _r(p.segundos, 1), "search": p.informe(),
                "pass": bool(np.median(d) <= P2_MEDIANA_MM)})
        por_escaneo[rol] = {**_base_comun(ctx, rol),
                            "pass": all(f["pass"] for f in filas),
                            "flips": sum(f["flip"] for f in filas), "runs": filas}
    return {"pass": all(v["pass"] for v in por_escaneo.values()),
            "criterion": f"mediana del desplazamiento de las coronas ≤ {P2_MEDIANA_MM} mm "
                         "en cada modificación",
            "flips": sum(v["flips"] for v in por_escaneo.values()),
            "scans": por_escaneo}


def mitades_por_corte(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras de los cortes pares e impares del contorno (por altura z, ordenada)."""
    z = np.round(P[:, 2], 3)
    rango = np.searchsorted(np.unique(z), z)
    return rango % 2 == 0, rango % 2 == 1


def mitades_tablero(esc: Escaner) -> tuple[np.ndarray, np.ndarray]:
    c = (esc.V - esc.centro) @ esc.base.T
    paridad = np.floor(c / TABLERO_MM).astype(np.int64).sum(axis=1) % 2
    return esc.banda & (paridad == 0), esc.banda & (paridad == 1)


def etapa_p3(ctx: Contexto) -> dict:
    por_escaneo = {}
    for rol in ctx.con_repetido():
        e = ctx.escaneos[rol]
        base = ctx.base(rol)
        reg0 = base.registrador
        esc = reg0.escaner
        X = e.V[e.corona]
        cod = e.etq[e.corona]
        destino = reg0.contornos[float(UMBRAL_CONTORNO)][0]
        origen = esc.V[esc.banda]
        poses, vueltas, informes = {}, 0, {}
        for nombre, m in zip(("cortes_pares", "cortes_impares"), mitades_por_corte(destino),
                             strict=True):
            reg = RegistroAnatomico(escaner=esc, base_cbct=reg0.base_cbct, plano=reg0.plano,
                                    contornos=reg0.contornos)
            poses[nombre] = ejecuta(reg, origen, destino[m])
            informes[nombre] = {"n_contour_points": int(m.sum()),
                                "n_slices": int(len(np.unique(np.round(destino[m, 2], 3))))}
        for nombre, banda in zip(("tablero_0", "tablero_1"), mitades_tablero(esc),
                                 strict=True):
            mitad = dataclasses.replace(esc, banda=banda, banda_cara=banda[esc.F].all(axis=1))
            reg = RegistroAnatomico(escaner=mitad, base_cbct=reg0.base_cbct,
                                    plano=reg0.plano, contornos=reg0.contornos)
            poses[nombre] = ejecuta(reg, mitad.V[mitad.banda], destino)
            informes[nombre] = {"n_band_vertices": int(banda.sum())}
        for nombre, p in poses.items():
            d = desplazamiento(p, X, base, X)
            vueltas += es_vuelta(p, d)
            informes[nombre].update(crown_displacement_vs_base=resumen(d),
                                    veto=p.resultado.veto, flip=es_vuelta(p, d),
                                    seconds=_r(p.segundos, 1), search=p.informe())
        d3a = desplazamiento(poses["cortes_pares"], X, poses["cortes_impares"], X)
        d3b = desplazamiento(poses["tablero_0"], X, poses["tablero_1"], X)
        npz = ctx.trabajo / f"p3-{rol}.npz"
        np.savez(npz, d3a=d3a, d3b=d3b, corona=np.flatnonzero(e.corona))
        por_escaneo[rol] = {
            **_base_comun(ctx, rol),
            "per_vertex_file": npz.name, "per_vertex_sha256": _sha(npz),
            "P3a_even_odd_slices": {
                "discrepancy": resumen(d3a),
                "per_tooth": {int(c): resumen(d3a[cod == c]) for c in np.unique(cod)}},
            "P3b_band_checkerboard": {
                "discrepancy": resumen(d3b),
                "per_tooth": {int(c): resumen(d3b[cod == c]) for c in np.unique(cod)}},
            "halves": informes, "flips": int(vueltas)}
    return {"descriptive": True,
            "note": "P3 no tiene criterio de paso: es el suelo f(x) de P4",
            "flips": sum(v["flips"] for v in por_escaneo.values()),
            "scans": por_escaneo}


# =========================================================================== #
# P4 y P4b
# =========================================================================== #
def _requiere(ctx: Contexto, etapas: tuple[str, ...], quien: str) -> dict[str, dict]:
    salida = {}
    for et in etapas:
        datos = lee_etapa(ctx.salida, et)
        if datos is None:
            raise SystemExit(f"{quien} necesita {et}.json escrito antes.")
        salida[et] = datos
    for et in (*P0_OBLIGATORIAS, "V0"):
        datos = lee_etapa(ctx.salida, et)
        if datos is None or not datos.get("pass"):
            raise SystemExit(f"{quien}: {et} no está escrito o no pasó; no se mide precisión.")
    return salida


def mide_par(ctx: Contexto, a: str, b: str, *, U: dict | None = None,
             suelo_p3: Path | None = None) -> dict:
    """d(x) = ‖A→CBCT(x) − B→CBCT(A→B(x))‖ sobre las coronas de A, y lo que P4 juzga."""
    A, B = ctx.escaneos[a], ctx.escaneos[b]
    pa, pb = ctx.base(a), ctx.base(b)
    corona = A.V[A.corona]
    cod = A.etq[A.corona]
    np.random.seed(SEMILLA_INSTRUMENTOS)
    m_ab, suelo = directo(corona, Superficie(B.V, B.F))
    en_b = corona @ m_ab[:3, :3].T + m_ab[:3, 3]
    llega_a = pa.aplica(corona)
    llega_b = pb.aplica(en_b)
    dv = llega_a - llega_b
    d = np.linalg.norm(dv, axis=1)
    sup = pa.registrador.base_cbct[2]
    delta = pb.m4() @ m_ab @ np.linalg.inv(pa.m4())
    centro = llega_a.mean(axis=0)

    if suelo_p3 is not None:
        f3 = np.load(suelo_p3)
        if not np.array_equal(f3["corona"], np.flatnonzero(A.corona)):
            raise SystemExit("P4: el suelo de P3 no corresponde a las coronas de A.")
        f = np.sqrt(f3["d3a"] ** 2 / 2 + f3["d3b"] ** 2 / 2 + suelo**2)
    else:
        f = suelo
    rho = len(A.V) / len(B.V)
    n_b = {int(c): int((B.etq == c).sum()) for c in np.unique(B.etq[B.etq > 0])}
    por_pieza, legibles = {}, []
    for c in np.unique(cod):
        sel = cod == c
        cobertura = float(sel.sum() / (n_b[int(c)] * rho)) if n_b.get(int(c)) else 0.0
        legible = bool(cobertura >= LEGIBLE_COBERTURA_MIN
                       and np.median(f[sel]) <= LEGIBLE_SUELO_MAX_MM)
        if legible:
            legibles.append(int(c))
        por_pieza[int(c)] = {"discrepancy": resumen(d[sel]),
                             "vertical_abs": resumen(np.abs(dv[sel] @ sup)),
                             "floor_direct": resumen(suelo[sel]), "floor_f": resumen(f[sel]),
                             "coverage": _r(cobertura, 3), "readable": legible}
    en = np.isin(cod, legibles)
    salida = {
        "pair": [a, b],
        "direct_floor": resumen(suelo),
        "readable_teeth": legibles,
        "discrepancy_all_teeth": resumen(d),
        "discrepancy_readable": resumen(d[en]),
        "vertical_abs_readable": resumen(np.abs(dv[en] @ sup)),
        "rigid_difference": {"angle_deg": _r(_angulo_4x4(delta), 4),
                             "translation_at_crown_centroid_mm": _r(np.linalg.norm(
                                 delta[:3, :3] @ centro + delta[:3, 3] - centro), 4)},
        "direct_transform_4x4": m_ab,
        "per_tooth": por_pieza,
    }
    if U is not None:
        esc_a = pa.registrador.escaner
        esc_b = pb.registrador.escaner
        pos = -np.ones(len(A.V), dtype=np.int64)
        pos[np.flatnonzero(esc_a.banda)] = np.arange(int(esc_a.banda.sum()))
        en_banda = (pos[np.flatnonzero(A.corona)] >= 0) & en
        ua = U[a][pos[np.flatnonzero(A.corona)][en_banda]]
        dist_b, jb = cKDTree(B.V[esc_b.banda]).query(en_b[en_banda])
        ub = U[b][jb]
        limite = CALIBRACION_K * np.sqrt(ua**2 + ub**2 + f[en_banda] ** 2)
        fraccion = float(np.mean(d[en_banda] <= limite)) if en_banda.any() else float("nan")
        mediana_u = max(float(np.median(ua)), float(np.median(ub))) if en_banda.any() \
            else float("nan")
        salida["calibration"] = {
            "n_vertices": int(en_banda.sum()), "fraction_within": _r(fraccion, 4),
            "U_A_median_mm": _r(np.median(ua), 4) if en_banda.any() else None,
            "U_B_median_mm": _r(np.median(ub), 4) if en_banda.any() else None,
            "U_median_mm": _r(mediana_u, 4),
            "B_band_match_distance": resumen(dist_b),
        }
    return salida


def etapa_p4(ctx: Contexto) -> tuple[dict, dict]:
    from fusion_agents.anatomico import prepara_anatomico

    previas = _requiere(ctx, ("P1", "P2", "P3"), "P4")
    if "repetido" not in ctx.escaneos:
        raise SystemExit("P4 necesita --repetido.")
    U, informes, incert = {}, {}, {}
    for rol in ("principal", "repetido"):
        e = ctx.escaneos[rol]
        t0 = time.perf_counter()
        o, dd, reg, inf = prepara_anatomico(
            ctx.campo, ctx.hu, e.V, e.F, e.etq, ejes_de(e.V, e.etq), normales=e.N,
            arcada_nombre="maxilar" if e.arcada == "upper" else "mandibular")
        res = reg(o, dd)
        hexa = np.asarray([*res.rotation, *res.translation], np.float64).tobytes().hex()
        if hexa != ctx.base(rol).bytes_hex():
            raise SystemExit(f"P4: el registro de {rol} con incertidumbre no es el de base.")
        inc = reg.incertidumbre()
        if round(float(np.median(inc["_U"])), 3) != inf["uncertainty"]["U_median_mm"]:
            raise SystemExit(f"P4: la incertidumbre de {rol} no se repite.")
        U[rol] = np.asarray(inc["_U"])
        incert[rol] = inc
        informes[rol] = {"seconds": _r(time.perf_counter() - t0, 1), "report": inf,
                         "veto": res.veto}
    suelo_p3 = ctx.trabajo / "p3-principal.npz"
    if _sha(suelo_p3) != previas["P3"]["scans"]["principal"]["per_vertex_sha256"]:
        raise SystemExit("P4: el fichero por vértice de P3 no es el que P3 escribió.")
    par = mide_par(ctx, "principal", "repetido", U=U, suelo_p3=suelo_p3)

    vueltas = {et: int(previas[et].get("flips", 0)) for et in ("P1", "P2", "P3")}
    for et in P0_OBLIGATORIAS:
        vueltas[et] = int(lee_etapa(ctx.salida, et).get("flips", 0))
    vueltas["P4"] = sum(1 for r in informes.values() if r["veto"])
    lect = par["discrepancy_readable"]
    cal = par["calibration"]
    por_pieza_ok = all(par["per_tooth"][c]["discrepancy"]["median_mm"] <= PIEZA_MEDIANA_MAX_MM
                       for c in par["readable_teeth"])
    hay = lect.get("n", 0) > 0
    if not hay or cal["n_vertices"] == 0:
        veredicto_cal = "sin vértices legibles"
    elif cal["fraction_within"] > CALIBRACION_FRACCION_VACIA and \
            cal["U_median_mm"] > CALIBRACION_U_MEDIANA_MAX_MM:
        veredicto_cal = "vacía"
    elif cal["fraction_within"] < CALIBRACION_FRACCION_MIN:
        veredicto_cal = "exceso de confianza"
    elif cal["U_median_mm"] > CALIBRACION_U_MEDIANA_MAX_MM:
        veredicto_cal = "demasiado ancha"
    else:
        veredicto_cal = "calibrada"
    criterios = {
        "i_zero_flips_P0_P4": sum(vueltas.values()) == 0,
        "ii_target": bool(hay and lect["median_mm"] <= OBJETIVO_MEDIANA_MM
                          and lect["p95_mm"] <= OBJETIVO_P95_MM),
        "iii_per_tooth": bool(hay and por_pieza_ok),
        "iv_rigid_difference": bool(
            par["rigid_difference"]["angle_deg"] <= DIFERENCIA_ANGULO_MAX_GRADOS
            and par["rigid_difference"]["translation_at_crown_centroid_mm"]
            <= DIFERENCIA_CENTROIDE_MAX_MM),
        "v_calibration": veredicto_cal == "calibrada",
    }
    p4 = {
        "pass": all(criterios.values()),
        "criteria": criterios,
        "improvement_level": {"D50_mm": lect.get("median_mm"),
                              "threshold_mm": MEJORA_D50_MM,
                              "reference_old_path_mm": REFERENCIA_ANTIGUA_D50_MM,
                              "pass": bool(hay and lect["median_mm"] < MEJORA_D50_MM)},
        "vertical_verdict": {
            "median_abs_mm": par["vertical_abs_readable"].get("median_mm"),
            "threshold_mm": VERTICAL_MEDIANA_MAX_MM,
            "pass": bool(hay and par["vertical_abs_readable"]["median_mm"]
                         <= VERTICAL_MEDIANA_MAX_MM),
            "if_fails": "la precisión vale en planta pero no en altura"},
        "calibration_verdict": veredicto_cal,
        "flips_by_stage": vueltas,
        "flips": vueltas["P4"],
        "pair": par,
        "registrations": informes,
        "note": "si (ii)–(v) fallan se informa como fallo y NO se repite",
    }
    residuos = {r: incert[r].get("signed_residual") or {} for r in incert}
    comunes = sorted(set(residuos["principal"]) & set(residuos["repetido"]))
    dif = {int(k): abs(residuos["principal"][k]["median"] - residuos["repetido"][k]["median"])
           for k in comunes}
    p4b = {
        "descriptive": True,
        "signed_residual_median_mm": {
            r: {int(k): v["median"] for k, v in res.items()} for r, res in residuos.items()},
        "abs_difference_per_tooth_mm": dif,
        "median_abs_difference_mm": float(np.median(list(dif.values()))) if dif else None,
        "teeth_of_interest": {
            int(k): {r: residuos[r][k]["median"] if k in residuos[r] else None
                     for r in residuos} for k in DIENTES_SESGO_P4B},
        "scale_scanner_to_cbct": {r: incert[r].get("scale_scanner_to_cbct") for r in incert},
        "reading": "si los sesgos de las piezas de interés coinciden entre los dos escaneos, "
                   "son de la boca o del CBCT, no del escaneo",
    }
    return p4, p4b


# =========================================================================== #
# E3
# =========================================================================== #
def etapa_e3(ctx: Contexto) -> dict:
    import registro_entre_cbct as e3

    _requiere(ctx, (), "E3")
    if ctx.args.cbct2 is None:
        raise SystemExit("E3 necesita --cbct2.")
    if ctx.args.e3_misma_denticion is None:
        raise SystemExit("E3 necesita --e3-misma-denticion (si/no/desconocido), leído del "
                         "inventario de `scripts/inventario_series_cbct.py`.")
    campo2, hu2 = e3.ingiere_cbct(ctx.pipe, ctx.args.cbct2, "cbct2")
    cbct2 = Cbct.de_campo(campo2, hu2)
    e = ctx.escaneos["principal"]
    p1 = ctx.base("principal")
    p2 = registra(cbct2, e.V, e.F, e.N, e.etq)
    X = e.V[e.corona]
    esc = p1.registrador.escaner

    def asigna(P: np.ndarray) -> np.ndarray:
        q = (P - p1.t) @ p1.R            # contorno de CBCT₁ al marco del escáner
        dist, j = esc.arbol.query(q)
        return np.where(esc.banda[j] & (dist <= 3.0), esc.etiquetas[j], 0)

    c1 = e3.prepara(ctx.cbct.centros, ctx.cbct.hu, e.arcada)
    c2 = e3.prepara(cbct2.centros, cbct2.hu, e.arcada)
    t0 = time.perf_counter()
    s = e3.registra_entre_cbct(c1, c2, asigna_dientes=asigna, puntos_eval=p1.aplica(X))
    segundos = time.perf_counter() - t0
    navaja = s["informe"]["leave_one_out"]["jackknife"]
    validez = {
        "same_dentition_declared": ctx.args.e3_misma_denticion,
        "jackknife_p95_mm": navaja["sigma_p95_mm"],
        "jackknife_threshold_mm": E3_NAVAJA_P95_MAX_MM,
        "valid": bool(ctx.args.e3_misma_denticion == "si"
                      and navaja["sigma_p95_mm"] is not None
                      and navaja["sigma_p95_mm"] <= E3_NAVAJA_P95_MAX_MM
                      and not s["informe"]["veto"] and not p2.resultado.veto),
    }
    salida = {"validity": validez, "cbct2_to_cbct1": s["informe"],
              "cbct2_to_cbct1_seconds": _r(segundos, 1),
              "transform_cbct2_to_cbct1_4x4": _4x4(s["R"], s["t"]),
              "scan_to_cbct2": {"search": p2.informe(), "veto": p2.resultado.veto},
              "cbct2_files_sha256": _digesto_serie(ctx.args.cbct2)}
    if not validez["valid"]:
        salida["verdict"] = "no interpretable"
        return salida
    llega_1 = p1.aplica(X)
    llega_21 = p2.aplica(X) @ s["R"].T + s["t"]
    d = np.linalg.norm(llega_1 - llega_21, axis=1)
    cod = e.etq[e.corona]
    salida.update(
        discrepancy=resumen(d),
        per_tooth={int(c): resumen(d[cod == c]) for c in np.unique(cod)},
        **{"pass": bool(np.median(d) <= E3_MEDIANA_MAX_MM
                        and np.percentile(d, 95) <= E3_P95_MAX_MM)},
        verdict="interpretable")
    return salida


# =========================================================================== #
# P5
# =========================================================================== #
def _clasico(ctx: Contexto, ref: str, enlace: Path) -> np.ndarray:
    """El camino clásico del pipeline sobre la superficie `ref`, como `caso_completo`."""
    snap = ctx.resultado.snapshot.model_copy(update={"surface_ref": ref})
    res = dataclasses.replace(ctx.resultado, snapshot=snap, registration_report=None)
    np.random.seed(SEMILLA_INSTRUMENTOS)
    fus = ctx.pipe.fuse(res, malla=enlace, dos_arcadas=True, registro="clasico")
    t = fus.snapshot.provenance.transform if fus.snapshot else None
    if t is None:
        raise SystemExit("P5: el camino clásico no dejó transformación.")
    return _4x4(quaternion_to_matrix(t.rotation), np.asarray(t.translation, float))


def etapa_p5(ctx: Contexto) -> dict:
    _requiere(ctx, (), "P5")
    salida: dict[str, Any] = {"descriptive": True, "pairs": []}
    for a, b in (("post", "visualizacion"), ("post", "principal")):
        if a in ctx.escaneos and b in ctx.escaneos:
            par = mide_par(ctx, a, b)
            par["note"] = "legibilidad con el suelo directo solo (no hay P3 de estos escaneos)"
            salida["pairs"].append(par)

    e = ctx.escaneos["principal"]
    if e.ref is None:
        raise SystemExit("P5 necesita el escaneo principal en el almacén del pipeline.")
    X = e.V[e.corona]
    m0 = _clasico(ctx, e.ref, e.enlace)

    def disp(m: np.ndarray, Xm: np.ndarray) -> np.ndarray:
        return np.linalg.norm(Xm @ m[:3, :3].T + m[:3, 3] - (X @ m0[:3, :3].T + m0[:3, 3]),
                              axis=1)

    orden, movs = [], []
    arrays = ctx.pipe.store.load(e.ref)
    for s in SEMILLAS_ORDEN:
        p, fp, giro = permutacion(s, len(e.V), len(e.F))
        inv = np.argsort(p)
        nuevos = {"positions": arrays["positions"][p], "normals": arrays["normals"][p],
                  "faces": rueda_caras(inv[np.asarray(arrays["faces"], np.int64)][fp],
                                       giro).astype(arrays["faces"].dtype)}
        for k, v in arrays.items():
            if k not in nuevos and len(v) == len(e.V):
                nuevos[k] = v[p]
        ref = ctx.pipe.store.put(**nuevos)
        d = disp(_clasico(ctx, ref, e.enlace), X)
        orden.append({"seed": s, "crown_displacement": resumen(d),
                      "flip": bool(d.max() > VUELTA_MM)})
    for nombre, R, t in movimientos():
        ruta, Xm, (_, _, _, ref), _, _ = _escaneo_movido(ctx, e, nombre, R, t, "p5-clasico")
        m = _clasico(ctx, ref, ruta)
        movs.append({"motion": nombre,
                     "crown_displacement": resumen(disp(m, Xm[e.corona])),
                     "flip": bool(disp(m, Xm[e.corona]).max() > VUELTA_MM)})
        ruta.unlink()
    salida["classic_path"] = {
        "base_transform_4x4": m0, "order_in_memory": orden, "motion_stl": movs,
        "flips": sum(f["flip"] for f in orden + movs),
        "note": "el orden se prueba en memoria: reingerido desde STL el orden de vértices "
                "desaparece (np.unique) y no habría nada que medir"}
    salida["not_run"] = ["el barrido vertical por diente de registro-por-cortes.md §3: "
                         "no forma parte de este script"]
    return salida


# =========================================================================== #
# Resumen
# =========================================================================== #
def etapa_resumen(ctx: Contexto) -> dict:
    if (ctx.salida / "resultado.json").exists():
        raise SystemExit("resultado.json ya existe.")
    faltan = [et for et in PRIMARIAS if lee_etapa(ctx.salida, et) is None]
    if faltan:
        raise SystemExit("resumen: faltan " + ", ".join(faltan))
    etapas = {}
    for et in ETAPAS[:-1]:
        datos = lee_etapa(ctx.salida, et)
        if datos is None:
            continue
        etapas[et] = {"file_sha256": _sha(ctx.salida / f"{et}.json"),
                      "pass": datos.get("pass"), "descriptive": datos.get("descriptive",
                                                                          False),
                      "flips": datos.get("flips")}
    p4 = lee_etapa(ctx.salida, "P4") or {}
    return {
        "scope": "Precisión del registro anatómico escáner↔CBCT frente a elecciones "
                 "declaradas y entre escaneos de la misma dentición. No es exactitud ni TRE.",
        "stages": etapas,
        "P0_pass": all(etapas[et]["pass"] for et in P0_OBLIGATORIAS),
        "P4_criteria": p4.get("criteria"),
        "P4_improvement_level": p4.get("improvement_level"),
        "P4_vertical_verdict": p4.get("vertical_verdict"),
        "P4_calibration_verdict": p4.get("calibration_verdict"),
    }


# =========================================================================== #
# Entrada
# =========================================================================== #
def _redacta(argv: list[str]) -> list[str]:
    return ["<ruta local>" if ("/" in a or Path(a).exists()) and not a.startswith("--")
            else a for a in argv]


def _git_rastrea(ruta: str) -> bool:
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", ruta], cwd=RAIZ, check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    return True


def _entrada_escaneo(ruta: Path, salida: Path) -> Path:
    """Enlace duro con el nombre del rol: el nombre del proveedor no llega a ningún sitio."""
    salida.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(ruta.resolve(), salida)
    except OSError:
        shutil.copyfile(ruta.resolve(), salida)
    return salida


def _etiquetas_de(directorio: Path, rol: str, desarrollo: bool) -> tuple[np.ndarray, dict]:
    carpeta = directorio / rol
    npys = sorted(carpeta.glob("etiquetas_*.npy"))
    if len(npys) != 1:
        raise SystemExit(f"--etiquetas: {rol}/ debe tener exactamente un etiquetas_*.npy.")
    info: dict[str, Any] = {"labels_sha256": _sha(npys[0])}
    jsons = sorted(carpeta.glob("fdi_*.json"))
    if jsons:
        cfg = json.loads(jsons[0].read_text())
        refs = (cfg.get("pose") or {}).get("references") or []
        info.update(config_sha256=_sha(jsons[0]), n_pose_references=len(refs),
                    model=cfg.get("modelo"))
        if len(refs) != REFERENCIAS_ENTRENAMIENTO and not desarrollo:
            raise SystemExit(f"--etiquetas: {rol} se segmentó con {len(refs)} referencias de "
                             f"pose, no con {REFERENCIAS_ENTRENAMIENTO}.")
    elif not desarrollo:
        raise SystemExit(f"--etiquetas: {rol}/ no trae su fdi_*.json de configuración.")
    return np.load(npys[0]).astype(np.int64), info


def _registros_previos(salida: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(salida.glob("run-record*.json"))]


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--_hijo":
        _hijo(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--huellas", action="store_true",
                    help="Imprime huella_constantes() y huella_protocolo() y sale.")
    ap.add_argument("--cbct", type=Path)
    ap.add_argument("--cbct2", type=Path)
    ap.add_argument("--principal", type=Path, help="PREVIO (escaneo A).")
    ap.add_argument("--repetido", type=Path, help="ESCANEADO PREVIO (escaneo B).")
    ap.add_argument("--post", type=Path)
    ap.add_argument("--visualizacion", type=Path)
    ap.add_argument("--etiquetas", type=Path,
                    help="DIR con DIR/<rol>/etiquetas_<arcada>.npy de segmentar_fdi.py.")
    ap.add_argument("--e3-misma-denticion", choices=("si", "no", "desconocido"))
    ap.add_argument("--huella", help="huella_constantes() congelada en el protocolo.")
    ap.add_argument("--huella-protocolo", help="huella_protocolo() congelada en el protocolo.")
    ap.add_argument("--salida", type=Path)
    ap.add_argument("--solo", help="Etapas separadas por comas: " + ",".join(ETAPAS))
    ap.add_argument("--con-etiquetas-gpu", action="store_true",
                    help="Incluye P0-etiquetas (necesita GPU y ~/.venvs/dental-gpu).")
    ap.add_argument("--python-gpu", default="~/.venvs/dental-gpu/bin/python")
    ap.add_argument("--desarrollo", action="store_true",
                    help="Permite árbol sucio y sin huellas. Los resultados NO valen como "
                         "protocolo y se marcan así. Rechaza rutas del caso B.")
    ap.add_argument("--recorta", metavar="S,O,M",
                    help="Solo con --desarrollo: usa las S primeras semillas de P0-semilla, "
                         "las O primeras de P0-orden y los M primeros movimientos.")
    args = ap.parse_args()

    if args.huellas:
        print(f"huella_constantes  {huella_constantes()}")
        print(f"huella_protocolo   {huella_protocolo()}")
        return 0
    for nombre in ("cbct", "principal", "etiquetas", "salida"):
        if getattr(args, nombre) is None:
            ap.error(f"--{nombre} es obligatorio")

    from caso_completo import entorno_de_software, revision_de_codigo

    # --- Disciplina, antes de tocar ningún dato.
    codigo = revision_de_codigo()
    if args.desarrollo:
        rutas = [str(Path(p).expanduser().resolve()) for p in
                 (args.cbct, args.cbct2, args.principal, args.repetido, args.post,
                  args.visualizacion, args.etiquetas) if p is not None]
        if any(ident in r for r in rutas for ident in _CASO_B):
            raise SystemExit("--desarrollo no lee datos del caso B: el protocolo no se ha "
                             "ejecutado.")
    else:
        if codigo["git_head"] is None or codigo["git_dirty"]:
            raise SystemExit("El árbol de código está sucio (o no hay git): el protocolo "
                             "se ejecuta desde un commit limpio.")
        if not _git_rastrea(PROTOCOLO):
            raise SystemExit(f"{PROTOCOLO} no está en git: se congela antes de medir.")
        if args.huella != huella_constantes():
            raise SystemExit("huella_constantes() no coincide con --huella: las constantes "
                             "de fusion_agents.anatomico no son las del protocolo.")
        if args.huella_protocolo != huella_protocolo():
            raise SystemExit("huella_protocolo() no coincide con --huella-protocolo.")

    if args.recorta:
        if not args.desarrollo:
            raise SystemExit("--recorta solo existe con --desarrollo.")
        global SEMILLAS_P0, SEMILLAS_ORDEN, SEMILLAS_SO3, GIROS_FIJOS, TRASLACIONES_FIJAS_MM
        n_s, n_o, n_m = (int(x) for x in args.recorta.split(","))
        SEMILLAS_P0, SEMILLAS_ORDEN = SEMILLAS_P0[:n_s], SEMILLAS_ORDEN[:n_o]
        SEMILLAS_SO3 = SEMILLAS_SO3[:n_m]
        GIROS_FIJOS = GIROS_FIJOS[:max(0, n_m - 6)]
        TRASLACIONES_FIJAS_MM = TRASLACIONES_FIJAS_MM[:max(0, n_m - 6)]

    if args.solo:
        etapas = [s.strip() for s in args.solo.split(",") if s.strip()]
        desconocidas = [s for s in etapas if s not in ETAPAS]
        if desconocidas:
            raise SystemExit("etapas desconocidas: " + ", ".join(desconocidas))
    else:
        etapas = [et for et in ETAPAS
                  if (et != "P0-etiquetas" or args.con_etiquetas_gpu)
                  and (et != "E3" or args.cbct2 is not None)
                  and (et != "P5" or args.post is not None)
                  and (et not in ("P4",) or args.repetido is not None)]
    etapas = [et for et in ETAPAS if et in etapas]       # siempre en orden de protocolo
    salida = args.salida
    for et in etapas:
        if (salida / f"{et}.json").exists() or (et == "P4" and (salida / "P4b.json").exists()) \
                or (et == "resumen"
                                                 and (salida / "resultado.json").exists()):
            raise SystemExit(f"{et} ya está escrito en {salida}: cada prueba se ejecuta una "
                             "vez.")

    # --- Entradas: hashes primero, y comparados con las invocaciones anteriores.
    rutas_escaneo = {r: getattr(args, r) for r in
                     ("principal", "repetido", "post", "visualizacion")
                     if getattr(args, r) is not None}
    entradas = {
        "scans_sha256": {r: _sha(p.expanduser()) for r, p in rutas_escaneo.items()},
        "cbct_files_sha256": _digesto_serie(args.cbct.expanduser()),
    }
    if args.cbct2 is not None:
        entradas["cbct2_files_sha256"] = _digesto_serie(args.cbct2.expanduser())
    etiquetas_crudas, entradas["labels"] = {}, {}
    for r in rutas_escaneo:
        etiquetas_crudas[r], entradas["labels"][r] = _etiquetas_de(
            args.etiquetas.expanduser(), r, args.desarrollo)
    salida.mkdir(parents=True, exist_ok=True)
    previos = _registros_previos(salida)
    for p in previos:
        for clave, valor in (("code_head", codigo["git_head"]),
                             ("constants_sha256", huella_constantes()),
                             ("protocol_sha256", huella_protocolo()),
                             ("development_run", bool(args.desarrollo))):
            if p.get(clave) != valor:
                raise SystemExit(f"Una invocación anterior en {salida} tiene otro {clave}: "
                                 "no se mezclan ejecuciones.")
        for clave, valor in entradas.items():
            if clave in p.get("inputs", {}) and p["inputs"][clave] != valor \
                    and clave != "labels":
                raise SystemExit(f"Una invocación anterior usó otras entradas ({clave}).")
        for r, v in entradas["labels"].items():
            if r in p.get("inputs", {}).get("labels", {}) and p["inputs"]["labels"][r] != v:
                raise SystemExit(f"Una invocación anterior usó otras etiquetas para {r}.")

    from agent_orchestrator import IngestionPipeline
    from fusion_agents.geometric import GeometricFusionAgent
    from ingestion_agents import ArtifactStore
    from ingestion_agents.cbct_agent import CBCTAgent
    from ingestion_agents.mesh_agent import MeshAgent

    nombre_registro = "run-record.json" if not previos else \
        f"run-record.{len(previos) + 1}.json"
    registro = {
        "schema": "histora-run-record/1.0",
        "script": "scripts/precision_registro.py",
        "argv": _redacta(sys.argv), "stages": etapas,
        "code": codigo, "code_head": codigo["git_head"],
        "environment": entorno_de_software(),
        "agents": {CBCTAgent.name: CBCTAgent.version, MeshAgent.name: MeshAgent.version,
                   GeometricFusionAgent.name: GeometricFusionAgent.version},
        "constants_sha256": huella_constantes(), "protocol_sha256": huella_protocolo(),
        "protocol": PROTOCOLO, "development_run": bool(args.desarrollo),
        "inputs": entradas, "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None, "outputs": {},
        "scope": "Identifica código, constantes y entradas de una invocación; no es "
                 "calibración física",
    }
    (salida / nombre_registro).write_text(json.dumps(registro, indent=2,
                                                     ensure_ascii=False) + "\n")
    trabajo = salida / "trabajo"
    trabajo.mkdir(exist_ok=True)

    # --- Ingesta: el CBCT una vez, con el principal; los demás escaneos al mismo almacén.
    from agent_orchestrator import CaseInput
    from core_schemas import Modality
    from export_agents import densidad_a_hu
    from ingestion_agents.mesh_agent import parse_stl

    pipe = IngestionPipeline(ArtifactStore(salida / "artifacts"),
                             quarantine_dir=salida / "quarantine",
                             cbct_recorte_dental=True, cbct_max_primitivas=MAX_PRIMITIVAS,
                             segmenter_factory=None)
    enlaces = {}
    for r, p in rutas_escaneo.items():
        V0 = parse_stl(p.expanduser())["positions"]
        if len(etiquetas_crudas[r]) != len(V0):
            raise SystemExit(f"Las etiquetas de {r} no corresponden a sus vértices.")
        arcada = ejes_de(V0, etiquetas_crudas[r]).arcada
        nombre = "superior" if arcada == "upper" else "inferior"
        enlaces[r] = _entrada_escaneo(p.expanduser(),
                                      trabajo / "escaneos" / r / f"escaneo-{nombre}.stl")
    print("· ingesta del CBCT y del escaneo principal")
    caso_dir = trabajo / "casos" / "entrada"
    caso_dir.mkdir(parents=True, exist_ok=True)
    resultado = pipe.run(CaseInput(acquisition_id=caso_dir.name, cbct=args.cbct.expanduser(),
                                   mesh=enlaces["principal"], reports=[], images=[]))
    if resultado.snapshot is None or resultado.snapshot.surface_ref is None:
        raise SystemExit("La ingesta no produjo snapshot con superficie.")
    campo = pipe.store.load(resultado.snapshot.gaussian_field_ref)
    if "hu_range" not in campo or "density" not in campo:
        raise SystemExit("El campo no trae HU.")
    hu = densidad_a_hu(campo["density"], campo["hu_range"])
    agente_malla = pipe.agents[Modality.MESH]

    def ingesta(ruta: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
        out = agente_malla.ingest(ruta)
        if out.artifact_ref is None:
            raise SystemExit(f"La ingesta de {ruta.name} falló: {out.ingestion.detail}")
        a = pipe.store.load(out.artifact_ref)
        return (np.asarray(a["positions"], np.float64), np.asarray(a["faces"], np.int64),
                np.asarray(a["normals"], np.float64), out.artifact_ref)

    escaneos = {}
    for r in rutas_escaneo:
        if r == "principal":
            ref = resultado.snapshot.surface_ref
            a = pipe.store.load(ref)
            V, F, N = (np.asarray(a["positions"], np.float64),
                       np.asarray(a["faces"], np.int64), np.asarray(a["normals"], np.float64))
        else:
            V, F, N, ref = ingesta(enlaces[r])
        cruda = etiquetas_crudas[r]
        if len(cruda) != len(V):
            raise SystemExit(f"Las etiquetas de {r} no corresponden a sus vértices.")
        etq, actas = limpia(V, cruda)
        escaneos[r] = Escaneo(rol=r, V=V, F=F, N=N, cruda=cruda, etq=etq,
                              arcada=ejes_de(V, etq).arcada,
                              sha256=entradas["scans_sha256"][r], enlace=enlaces[r], ref=ref,
                              limpieza=actas)
    (trabajo / "limpieza-etiquetas.json").write_text(json.dumps(
        {r: e.limpieza for r, e in escaneos.items()}, indent=1) + "\n")

    ctx = Contexto(args=args, salida=salida, trabajo=trabajo, cbct=Cbct.de_campo(campo, hu),
                   campo=campo, hu=hu, escaneos=escaneos, ingesta=ingesta, pipe=pipe,
                   resultado=resultado, registro_ejecucion=nombre_registro)

    funciones: dict[str, Callable[[Contexto], Any]] = {
        "V0": etapa_v0, "P0-semilla": etapa_p0_semilla, "P0-orden": etapa_p0_orden,
        "P0-movimiento": etapa_p0_movimiento, "P0-teselado": etapa_p0_teselado,
        "P0-etiquetas": etapa_p0_etiquetas, "P1": etapa_p1, "P2": etapa_p2, "P3": etapa_p3,
        "E3": etapa_e3, "P5": etapa_p5, "resumen": etapa_resumen,
    }
    codigo_salida = 0
    for et in etapas:
        if et not in ("V0", *P0_OBLIGATORIAS, "P0-etiquetas", "resumen"):
            for previa in ("V0", *P0_OBLIGATORIAS):
                datos = lee_etapa(salida, previa)
                if datos is None or not datos.get("pass"):
                    raise SystemExit(f"{et}: {previa} no está escrito o no pasó; no se "
                                     "escribe ninguna cifra de precisión.")
        print(f"· {et}")
        if et == "P4":
            p4, p4b = etapa_p4(ctx)
            escritos = [escribe_etapa(ctx, "P4", p4), escribe_etapa(ctx, "P4b", p4b)]
        elif et == "resumen":
            datos = etapa_resumen(ctx)
            ruta = salida / "resultado.json"
            ruta.write_text(json.dumps(_a_json({**_meta(ctx, "resumen"), **datos}),
                                       indent=1, ensure_ascii=False) + "\n")
            escritos = [ruta]
        else:
            escritos = [escribe_etapa(ctx, et, funciones[et](ctx))]
        registro["outputs"].update({p.name: _sha(p) for p in escritos})
        (salida / nombre_registro).write_text(json.dumps(registro, indent=2,
                                                         ensure_ascii=False) + "\n")
        for p in escritos:
            imprime(p)
        datos = json.loads(escritos[0].read_text())
        if et in ("V0", *P0_OBLIGATORIAS) and not datos.get("pass"):
            print(f"  ✗ {et} no pasa: se aborta sin medir precisión.")
            codigo_salida = 3
            break
    registro["finished_at"] = datetime.now(UTC).isoformat()
    (salida / nombre_registro).write_text(json.dumps(registro, indent=2,
                                                     ensure_ascii=False) + "\n")
    return codigo_salida


def imprime(ruta: Path) -> None:
    """Resumen legible de un fichero YA escrito."""
    d = json.loads(ruta.read_text())
    linea = f"  {ruta.name}: "
    if "pass" in d:
        linea += "pasa" if d["pass"] else "NO pasa"
    elif d.get("descriptive"):
        linea += "descriptiva"
    if "flips" in d and d["flips"] is not None:
        linea += f" · vueltas {d['flips']}"
    print(linea)
    for rol, s in (d.get("scans") or {}).items():
        extra = {k: s[k] for k in ("pass", "n_distinct", "max_crown_displacement_mm",
                                   "same_basin_by_angle", "flips") if k in s}
        if "crown_displacement" in s:
            extra["crown_displacement"] = s["crown_displacement"]
        for k in ("P3a_even_odd_slices", "P3b_band_checkerboard"):
            if k in s:
                extra[k] = s[k]["discrepancy"]
        print(f"    {rol}: {extra}")
    if "criteria" in d:
        print(f"    criterios {d['criteria']} · mejora {d['improvement_level']['pass']} · "
              f"vertical {d['vertical_verdict']['pass']} · calibración "
              f"{d['calibration_verdict']}")
        par = d["pair"]
        print(f"    legibles {par['readable_teeth']} · d {par['discrepancy_readable']} · "
              f"{par['rigid_difference']}")
    if "validity" in d:
        print(f"    validez {d['validity']} · {d.get('verdict')} · {d.get('discrepancy')}")


if __name__ == "__main__":
    raise SystemExit(main())
