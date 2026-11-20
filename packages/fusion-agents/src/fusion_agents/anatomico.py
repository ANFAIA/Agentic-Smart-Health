"""Registrar el escáner contra un CBCT, denso o disperso, sin azar y sin vueltas posibles.

**Por qué existe.** La medida 1 (`docs/research/consistencia-registro-caso-b.md` §5) dejó
dos cosas medidas sobre el camino de siempre para series densas (`preparacion` + ICP
desde la identidad sobre una submuestra aleatoria de 4.000 vértices + árbitro de umbral):

- la pose cambia 1–3 mm con la **semilla** de la submuestra y con el **orden** de los
  vértices del fichero, sin que cambie ningún dato;
- en ~30 % de semillas da la **vuelta** (≈180°), y el rms del ICP no lo nota (0,679 frente
  a 0,664–0,680 mm). Lo nota el solapamiento, pero nadie lo usaba para elegir.

Este módulo hace el registro de forma que esas dos cosas no puedan pasar, y dice cuánto se
fía de sí mismo diente a diente. Diseño y justificación: `docs/research/registro-anatomico.md`.

**Las cuatro decisiones.**

1 · **Primero se orienta y después se ajusta, sin azar.** El escáner trae sus ejes de las
etiquetas FDI (`export_agents.anatomia.anatomical_frame`, que pasa quien llama); el CBCT,
los de la cabecera DICOM (LPS) tal cual, sin estimar la inclinación del plano oclusal
(`marco_cbct` explica por qué). La pose inicial es la rotación de unos ejes a otros
(`por_cortes.pose_inicial`). No hay submuestra aleatoria en ningún paso: mover el STL,
reordenar sus vértices o cambiar una semilla no puede cambiar el resultado.

2 · **Solo se compara lo que las dos modalidades ven.** En el escáner, una banda **por
diente**: los vértices de corona entre 1,5 y 6 mm por debajo de la cúspide de *ese* diente,
medidos a lo largo del eje oclusal (`banda_escaner`). Sigue la curva de Spee y cualquier
inclinación de la cabeza, que una losa axial sobre el CBCT no sigue. En el CBCT, los
**bordes** del diente en cada corte (`por_cortes.contornos_con_normales`, umbral fijo 1400),
no el volumen: una corona hundida 1 mm deja el borde a 1 mm de la superficie del escáner,
y la comprobación corona→esmalte de una cara no lo veía. Y solo los de **corona**: los que
tienen esmalte cerca en su corte (`junto_al_esmalte`). La raíz es dentina, tan ancha como
el cuello, y en el caso A atraía la banda (`docs/research/registro-anatomico.md` §7–§8).

3 · **El ajuste va del contorno al escáner, punto a plano.** Cada punto de contorno busca
el vértice más próximo entre TODOS los del escáner; solo cuenta si ese vértice es de la
banda. Así las cúspides de la arcada contraria y el hueso cervical se caen solos —su
vecino es oclusal o encía— sin un umbral de altura que decidirlo.

4 · **Varias salidas fijas y una vara distinta para elegir.** 81 arranques deterministas
alrededor de la pose anatómica (±20° por eje × ±2 mm en altura), más 9 hipótesis de vuelta
de 180° que se ajustan siempre y **nunca pueden ganar**: están para medir por cuánto
pierden. Las poses se agrupan en cuencas y se elige por `puntua_secciones`, que no es la
función que se ajusta: la distancia simétrica, dentro de cada corte real del CBCT, entre
la sección exacta del escáner y el contorno, con todos los dientes pesando igual.

⚠️ **Precisión, no exactitud.** Todo lo que este módulo declara —sensibilidad al umbral,
navaja dejando fuera cada diente, ambigüedad de cuenca, condicionamiento, escala— mide
cuánto se mueve la pose ante elecciones declaradas. Un sesgo común a todas ellas (el
isonivel 1400, el medio paso del borde, que los dientes se hayan movido entre el CBCT y el
escaneo) no lo ve. Eso solo lo acotan los puntos anatómicos (`fusion_agents.tre`).

⚠️ **La altura es la dirección peor medida.** Con cortes horizontales y la banda en las
paredes de la corona, la altura solo se ve en cuánto se estrecha la sección hacia la
cúspide. Se declara con el bloque de traslación de JᵀWJ, no se corrige.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from fusion_agents.por_cortes import (
    ALTO_BANDA_MM,
    EJES_LPS,
    MARGEN_MORDIDA_MM,
    MINIMO_CONTORNO,
    RADIO_ARCADA_MM,
    TRIM,
    UMBRAL_CONTORNO,
    UMBRALES_INCERTIDUMBRE,
    contornos_con_normales,
    cortes,
    describe_cortes,
    pose_inicial,
)
from fusion_agents.preparacion import HU_ARBITRO, plano_oclusal_del_esmalte
from fusion_agents.registration import (
    DEFAULT_OVERLAP_MM,
    RegistrationResult,
    informacion_6dof,
    matrix_to_quaternion,
    paso_punto_plano,
    paso_similitud_punto_plano,
    rotacion_de_vector,
)

# --------------------------------------------------------------------------- #
# Constantes. Ninguna sale del caso B: o se heredan de `por_cortes` (fijadas en el
# caso A) o son a priori, y están congeladas en el protocolo
# (`docs/research/registro-anatomico-protocolo.md`) con su huella (`huella_constantes`).
# --------------------------------------------------------------------------- #

# Esmalte con el que se sitúa la altura del plano oclusal, y la ventana alrededor de la moda
# con la que se toma su centro en planta.
HU_PLANO = HU_ARBITRO
BANDA_PLANO_MM = 6.0
# Perfil de altura del esmalte: suavizado, separación mínima entre dos picos para hablar
# de dos arcadas, y cuánto más bajo que el menor de los dos tiene que ser el valle.
SUAVIZADO_PERFIL_MM = 1.0
SEPARACION_PICOS_MM = 3.0
VALLE_RELATIVO = 0.75
# Un segundo pico solo es la otra arcada si tiene al menos esta fracción de la altura del
# primero. A priori: dos arcadas en oclusión tienen cantidades de esmalte por corte del
# mismo orden —una a la que le faltaran tres de cada cuatro piezas seguiría pasando—, y un
# pico de unas pocas gaussianas sueltas no es una arcada sino ruido sobre un perfil casi
# vacío, que el suavizado normalizado por corte convierte en máximo local. Sin este mínimo,
# el valle entre el plano y ese ruido entraba como ancla y ensanchaba la losa hasta la raíz
# (`docs/research/registro-anatomico.md` §7–§8). En el fantasma de dos arcadas el segundo
# pico es el 0,98 del primero.
PICO_SECUNDARIO_RELATIVO = 0.25

# Solo se ajusta contra contorno de CORONA: un punto de contorno cuenta si en su corte hay
# esmalte (HU ≥ `HU_ESMALTE`) a menos de `RADIO_ESMALTE_MM`. La corona es esmalte y la raíz
# dentina: el contorno a 1400 rodea a las dos, y la raíz, con la anchura del cuello y bajo
# una encía que las etiquetas del escáner a veces cuentan como diente, es justo lo que el
# ajuste confundía con la banda de corona (§7). El umbral es el mismo del árbitro y del
# plano (`preparacion.HU_ARBITRO`). El radio es a priori: el grosor del esmalte en las
# paredes axiales de una corona (1–2,5 mm, menos hacia el cuello) más un vóxel; una pared a
# más de eso del esmalte más próximo de su corte no es pared de corona. No mueve ningún
# punto: el contorno sigue a 1400, con su medio paso de 8 vecinos; solo elige cuáles cuentan.
HU_ESMALTE = HU_ARBITRO
RADIO_ESMALTE_MM = 1.5
# Qué cuenta como corona es una elección, y U dice cuánto mueve la pose: se reajusta con la
# máscara a medio milímetro menos y más (un vóxel de 0,3–0,5 mm, a priori).
RADIOS_ESMALTE_INCERTIDUMBRE = (1.0, 2.0)

# Caras de banda demasiado inclinadas para ser pared de corona (§11 de la nota). Cuando las
# etiquetas FDI del escáner desbordan sobre la encía, esa encía entra en la banda con caras
# inclinadas (|n·oclusal| ≈ 0,60, frente a 0,21–0,24 en las paredes de la corona), y en el
# ajuste punto a plano todas tiran de la arcada hacia la raíz. El umbral sale de cada diente
# del propio escaneo: la distribución de |n·oclusal| de sus caras en la parte ALTA de la
# banda (`MARGEN_MORDIDA_MM`–`ALTO_BANDA_SUPERIOR_MM` bajo su cúspide), en su percentil
# `PERCENTIL_INCLINACION` más `MARGEN_INCLINACION`. Las caras de banda por encima no se
# emparejan ni asignan contorno a S; C sigue contándolas, para no cambiar lo que veta.
# ⚠️ La primera versión, fijada antes de medir, tomaba el percentil 95 más 0,15: con coronas
# clínicas de 5 mm la papila desbordada ya ocupa el 21 % de esa franja alta, el p95 cae en
# encía y no se quita nada (§11.1). La mediana aguanta hasta la mitad de contaminación; el
# margen de 0,20 se puso DESPUÉS de ver el fantasma (corona: mediana ≈ 0,3, p99 ≈ 0,45 junto
# a las cúspides; encía ≥ 0,56). Ninguno mira C ni el caso A.
FILTRA_INCLINACION = True
ALTO_BANDA_SUPERIOR_MM = 3.5
PERCENTIL_INCLINACION = 50.0
MARGEN_INCLINACION = 0.20

# Losa de contornos del CBCT, en altura sobre el plano oclusal (horizontal) hacia la raíz de
# la arcada escaneada. Ancha a propósito: la banda del escáner (1,5–6 mm sobre cada
# cúspide) tiene que caber aunque la boca esté inclinada ~10° —±4,4 mm en los extremos de
# una arcada de 50 mm— y el plano desplazado 2 mm. Lo que entra de más (cúspides de la otra
# arcada, raíz, hueso) se cae solo en el ajuste: su punto más próximo no es de la banda.
LOSA_CBCT_MM = (-4.0, 14.0)

# Arranques: combinaciones de estos ángulos alrededor de derecha, anterior y superior del
# CBCT, por cada uno de estos desplazamientos en altura.
REJILLA_GRADOS = (-20.0, 0.0, 20.0)
DESPLAZAMIENTOS_VERTICALES_MM = (-2.0, 0.0, 2.0)

# Etapa gruesa sobre un representante del contorno por vóxel de este lado.
REJILLA_GRUESA_MM = 1.0
# Dos poses son la misma cuenca si el rms del desplazamiento de la banda es menor que esto.
RADIO_CUENCA_GRUESA_MM = 0.5
RADIO_CUENCA_MM = 0.1

# Sección exacta del escáner muestreada cada tanto; un punto de contorno a más de
# `TOPE_SECCION_MM` de la sección no es corona de esta arcada.
PASO_SECCION_MM = 0.1
TOPE_SECCION_MM = 3.0

# Elegibilidad de una cuenca.
COBERTURA_MIN = 0.5
TOL_COBERTURA_MM = DEFAULT_OVERLAP_MM
ANGULO_ANATOMICO_MAX = 30.0
ALTURA_CUSPIDES_MM = 5.0

# Señales que van al gate.
ANGULO_VUELTA = 150.0
# Márgenes de S en mm, no en proporción: S tiene un suelo que pone el tamaño de vóxel
# (0,15 mm a 0,4 mm de vóxel, 0,11 a 0,3 en el fantasma), así que una proporción fija
# saltaría con un vóxel y no con otro para el mismo desajuste. Medido en el fantasma, un
# desajuste lateral uniforme de 0,2 mm sube S 0,024–0,030 mm y uno de 0,5 mm, 0,15.
MARGEN_VUELTA_S_MM = 0.05
MARGEN_VUELTA_C = 0.05
MARGEN_AMBIGUEDAD_S_MM = 0.03
DESPLAZAMIENTO_AMBIGUO_MM = 1.0
ESCALA_MAX = 0.005
DISCREPANCIA_MARCOS_GRADOS = 30.0
INCERTIDUMBRE_MAX_MM = DEFAULT_OVERLAP_MM
RESIDUO_FIRMADO_MAX_MM = 0.5

# Iteraciones y parada.
ITER_GRUESO = 60
ITER_MEDIO = 40
ITER_FINO = 200
# Cuánto por encima del margen de ambigüedad puede quedar una cuenca tras el tramo corto
# y aun así refinarse hasta converger: con menos iteraciones su S todavía no es la final.
HOLGURA_FINALISTA_MM = 0.05
TOL_MM = 1e-4
# La etapa gruesa solo tiene que dejar cada arranque en su cuenca.
TOL_GRUESO_MM = 1e-3
ITER_ESCALA = 30
TOPE_PASO = (math.radians(5.0), 2.0)
CORRESPONDENCIAS_MINIMAS = 100
# Vecinos cuyo abanico de caras se examina para el punto más próximo sobre la malla, y
# cuántas poses atrás se mira para reconocer un ciclo del ICP.
K_VECINOS = 6
MEMORIA_CICLO = 8


def huella_constantes() -> str:
    """sha256 de los valores de todas las constantes que deciden el registro.

    Va al informe y a `registro.json`: un resultado que no lleva la huella del protocolo
    no se puede atribuir a él, y un script de evaluación se niega a correr si no coincide.
    """
    valores = {
        nombre: globals()[nombre]
        for nombre in sorted(globals())
        if nombre.isupper() and not nombre.startswith("_") and nombre != "EJES_LPS"
    }
    valores["EJES_LPS"] = EJES_LPS.tolist()
    texto = json.dumps(valores, sort_keys=True, default=float)
    return hashlib.sha256(texto.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Marcos y bandas
# --------------------------------------------------------------------------- #
def _unitario(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _ortogonaliza(v: np.ndarray, *previas: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).copy()
    for p in previas:
        v = v - (v @ p) * p
    return _unitario(v)


def angulo_grados(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.clip(_unitario(a) @ _unitario(b), -1.0, 1.0))
    return math.degrees(math.acos(c))


def angulo_rotacion_grados(r: np.ndarray) -> float:
    """Ángulo de una rotación 3×3, en grados."""
    c = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(c))


@dataclass(frozen=True)
class Plano:
    """Plano oclusal del CBCT: un punto y su normal hacia superior, en mm del DICOM.

    `anclas` son las alturas sobre este plano donde puede estar el contacto entre arcadas:
    `(0,)` si la altura del esmalte tiene un pico, y además la del valle si tiene dos. La
    búsqueda arranca desde todas y la elección la decide la puntuación, no el perfil.
    """

    punto: np.ndarray
    normal: np.ndarray
    anclas: tuple[float, ...] = (0.0,)

    def altura(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.punto) @ self.normal


def _perfil_z(z: np.ndarray) -> dict:
    """Perfil de la altura del esmalte: uno o dos picos, y dónde.

    Se cuenta el esmalte **por corte** y se suaviza con un núcleo gaussiano de
    `SUAVIZADO_PERFIL_MM` normalizado por los cortes que hay cerca (Nadaraya–Watson). Un
    histograma de anchura fija no sirve: con cortes cada 0,3–0,4 mm unas casillas reciben
    dos cortes y otras ninguno, y salen picos de muestreo; y en una serie dispersa los
    huecos parecerían valles.
    """
    zs, cuenta = np.unique(np.round(np.asarray(z, dtype=np.float64), 3), return_counts=True)
    rejilla = np.arange(zs.min(), zs.max() + 1e-9, 0.1)
    k = np.exp(-0.5 * ((rejilla[:, None] - zs[None]) / SUAVIZADO_PERFIL_MM) ** 2)
    perfil = (k @ cuenta) / np.maximum(k.sum(axis=1), 1e-300)
    picos = [
        i for i in range(1, len(perfil) - 1)
        if perfil[i] >= perfil[i - 1] and perfil[i] > perfil[i + 1]
    ]
    picos = sorted(picos, key=lambda i: -perfil[i])
    salida: dict[str, Any] = {"mode_mm": round(float(rejilla[int(np.argmax(perfil))]), 2)}
    segundo = next(
        (j for j in picos[1:] if abs(rejilla[j] - rejilla[picos[0]]) >= SEPARACION_PICOS_MM),
        None,
    )
    if segundo is not None:
        salida["second_peak_ratio"] = round(float(perfil[segundo] / perfil[picos[0]]), 4)
        if perfil[segundo] < PICO_SECUNDARIO_RELATIVO * perfil[picos[0]]:
            salida["second_peak_ignored_mm"] = round(float(rejilla[segundo]), 2)
            segundo = None
    if picos and segundo is not None:
        a, b = sorted((picos[0], segundo))
        valle = a + int(np.argmin(perfil[a : b + 1]))
        salida.update(
            bimodal=bool(perfil[valle] < VALLE_RELATIVO * min(perfil[a], perfil[b])),
            peaks_mm=[round(float(rejilla[a]), 2), round(float(rejilla[b]), 2)],
            valley_mm=round(float(rejilla[valle]), 2),
            valley_ratio=round(float(perfil[valle] / min(perfil[a], perfil[b])), 3),
        )
    else:
        salida["bimodal"] = False
    return salida


def marco_cbct(
    centros: np.ndarray, hu: np.ndarray, arcada: str = "upper"
) -> tuple[np.ndarray, Plano, dict]:
    """`(base, plano, informe)`: ejes del paciente en el CBCT y la altura del plano oclusal.

    `base` son los ejes LPS del DICOM (derecha −x, anterior −y, superior +z). **No se
    estima la inclinación del plano oclusal en el CBCT**, y se probó: un plano por PCA al
    esmalte depende tanto de la ventana que en el fantasma leyó entre 0,2° y 40° para una
    inclinación real de 10° —con las dos arcadas en la ventana toma por inclinación el
    resalte entre ellas; con una ventana definida por el propio plano se refuerza a sí
    misma—. La inclinación la absorben la rejilla de arranques (±20°), la banda POR DIENTE
    del escáner y una losa de contornos ancha (`LOSA_CBCT_MM`); la vertical con que se
    declara la incertidumbre es el eje superior del escáner ya registrado, que sí está
    medido (etiquetas FDI).

    **La altura del plano es la moda del esmalte, y si el perfil tiene dos picos el valle
    entra como segunda ancla, no como sustituta.** En oclusión las caras oclusales de las
    dos arcadas se tocan y, si el esmalte se concentra en ellas, sale un pico en el plano
    oclusal: la moda (`preparacion.plano_oclusal_del_esmalte`, estable entre umbrales en el
    caso B). Pero si cada arcada forma su propio pico, la moda cae en el ecuador de UNA de
    las dos coronas y el contacto es el valle: con la moda sola, el fantasma de dos arcadas
    dejó el plano 6 mm dentro de la arcada contraria. Y con el valle solo, el caso A (serie
    dispersa) tomó un valle entre el plano oclusal y un pico menor 11 mm por encima, y la
    losa se quedó sin contorno. Ninguna de las dos reglas sirve sola, así que no se elige:
    se arranca desde las dos y decide la puntuación.

    `arcada` no cambia nada aquí; se acepta para que la firma no dependa de la versión.
    """
    del arcada
    centros = np.asarray(centros, dtype=np.float64)
    hu = np.asarray(hu)
    esmalte = centros[hu >= HU_PLANO]
    if len(esmalte) < MINIMO_CONTORNO:
        raise ValueError(
            f"solo {len(esmalte)} gaussianas de esmalte (HU ≥ {HU_PLANO:.0f}): no hay con "
            "qué situar el plano oclusal"
        )
    perfil = _perfil_z(esmalte[:, 2])
    referencia = perfil["mode_mm"]
    cerca = esmalte[np.abs(esmalte[:, 2] - referencia) <= BANDA_PLANO_MM]
    centro_xy = (cerca if len(cerca) else esmalte)[:, :2].mean(axis=0)
    punto = np.array([centro_xy[0], centro_xy[1], referencia])
    anclas = [0.0]
    if perfil["bimodal"]:
        anclas.append(round(float(perfil["valley_mm"] - referencia), 3))
    informe: dict[str, Any] = {
        "enamel_mode_mm_histogram": round(float(plano_oclusal_del_esmalte(esmalte[:, 2])), 2),
        "enamel_profile": perfil,
        "reference_height_mm": round(float(referencia), 2),
        "plane_point_mm": [round(float(c), 3) for c in punto],
        "plane_normal": EJES_LPS[2].tolist(),
        "anchors_mm": anclas,
    }
    return EJES_LPS.copy(), Plano(punto, EJES_LPS[2].copy(), tuple(anclas)), informe


def normales_vertice(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Normales por vértice ponderadas por área, en float64. Su signo no se usa."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = np.zeros_like(V)
    cara = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for i in range(3):
        np.add.at(n, F[:, i], cara)
    norma = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, norma, out=np.zeros_like(n), where=norma > 0)


def banda_escaner(
    V: np.ndarray, etiquetas: np.ndarray, oclusal: np.ndarray, centro: np.ndarray
) -> tuple[np.ndarray, dict[int, float]]:
    """`(mascara, puntas)`: la banda de corona de CADA diente y la altura de su cúspide.

    La altura se mide a lo largo del eje oclusal de la arcada (de la encía a las coronas),
    no de un eje por diente: el eje de un diente sacado de su propia nube se midió con 60°
    de error (ficha del `segmentation-agent`). La cúspide es el percentil 99 de la altura
    de sus vértices, para que un vértice suelto no mande. Un vértice está en la banda si
    queda entre `MARGEN_MORDIDA_MM` y `ALTO_BANDA_MM` por debajo de la cúspide de su diente.
    """
    V = np.asarray(V, dtype=np.float64)
    etq = np.asarray(etiquetas, dtype=np.int64)
    h = (V - np.asarray(centro, dtype=np.float64)) @ _unitario(oclusal)
    banda = np.zeros(len(V), dtype=bool)
    puntas: dict[int, float] = {}
    for c in np.unique(etq[etq > 0]):
        m = etq == c
        punta = float(np.percentile(h[m], 99))
        puntas[int(c)] = punta
        banda |= m & (h <= punta - MARGEN_MORDIDA_MM) & (h >= punta - ALTO_BANDA_MM)
    return banda, puntas


def caras_de_pared(
    altura: np.ndarray, inclinacion: np.ndarray, etiqueta: np.ndarray, en_banda: np.ndarray,
    puntas: Mapping[int, float],
) -> np.ndarray:
    """Máscara de caras de banda cuya |n·oclusal| cabe en la pared de corona de su diente.

    `altura` es la del centroide de cada cara sobre el eje oclusal; `inclinacion`,
    |n·oclusal|. Por diente, el umbral es el percentil `PERCENTIL_INCLINACION` de la
    inclinación de sus caras de banda entre `MARGEN_MORDIDA_MM` y `ALTO_BANDA_SUPERIOR_MM`
    bajo la cúspide, más `MARGEN_INCLINACION`. Un diente sin caras en esa franja se deja
    entero: sin referencia propia no se quita nada. Fuera de la banda devuelve `False`.
    """
    apta = en_banda.copy()
    for c, punta in puntas.items():
        m = en_banda & (etiqueta == c)
        prof = punta - altura
        ref = m & (prof >= MARGEN_MORDIDA_MM) & (prof <= ALTO_BANDA_SUPERIOR_MM)
        if not ref.any():
            continue
        umbral = float(np.percentile(inclinacion[ref], PERCENTIL_INCLINACION)) + MARGEN_INCLINACION
        apta[m & (inclinacion > umbral)] = False
    return apta


def limites_losa(arcada: str) -> tuple[float, float]:
    """Alturas sobre el plano entre las que se toman contornos, hacia la raíz de la arcada."""
    if arcada == "upper":
        return LOSA_CBCT_MM
    if arcada == "lower":
        return -LOSA_CBCT_MM[1], -LOSA_CBCT_MM[0]
    raise ValueError(f"arcada desconocida: {arcada!r} (se espera 'upper' o 'lower')")


def junto_al_esmalte(
    P: np.ndarray, centros: np.ndarray, hu: np.ndarray, radio: float | None = None
) -> np.ndarray:
    """Máscara de los puntos de contorno `P` con esmalte a menos de `radio` EN SU CORTE.

    En su corte y no en 3D: con una serie dispersa el corte vecino puede estar a 2 mm, y en
    3D una pared de raíz justo por encima del cuello vería el esmalte del corte de abajo.
    Los cortes se identifican por su z redondeada a la micra, como en `por_cortes`.
    `radio=None` lee `RADIO_ESMALTE_MM` al llamar, no al importar: así la huella de las
    constantes y lo que se ejecuta no pueden separarse.
    """
    radio = RADIO_ESMALTE_MM if radio is None else float(radio)
    P = np.asarray(P, dtype=np.float64)
    salida = np.zeros(len(P), dtype=bool)
    if len(P) == 0:
        return salida
    centros = np.asarray(centros, dtype=np.float64)
    esmalte = centros[np.asarray(hu) >= HU_ESMALTE]
    if len(esmalte) == 0:
        return salida
    z_e = np.round(esmalte[:, 2], 3)
    z_p = np.round(P[:, 2], 3)
    for z in np.unique(z_p):
        e = esmalte[z_e == z]
        if len(e) == 0:
            continue
        m = np.flatnonzero(z_p == z)
        d = cKDTree(e[:, :2]).query(P[m, :2], distance_upper_bound=radio)[0]
        salida[m] = d <= radio
    return salida


def contornos_losa(
    centros: np.ndarray, hu: np.ndarray, umbral: float, plano: Plano, arcada: str,
    *, solo_esmalte: bool = True, radio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`(P, N2)`: contornos de corona en la losa inclinada y dentro de la huella de la arcada.

    ⚠️ El contorno se detecta sobre la envolvente AXIAL de la losa y solo después se filtra
    por la altura inclinada. Detectarlo dentro de la losa ya recortada crearía bordes
    falsos en sus caras: un punto de esmalte cuyo vecino quedó fuera de la losa parecería
    borde sin serlo.

    Con `solo_esmalte` (por defecto) se quedan solo los puntos con esmalte cerca en su corte
    (`junto_al_esmalte`): las paredes de raíz y de hueso no entran ni en el ajuste ni en la
    puntuación. La misma regla se aplica a los contornos a 1100 y 1900 de la incertidumbre,
    así que lo que mide la sensibilidad al umbral sigue siendo solo el umbral.
    """
    centros = np.asarray(centros, dtype=np.float64)
    hu = np.asarray(hu)
    lo, hi = limites_losa(arcada)
    lo, hi = lo + min(plano.anclas), hi + max(plano.anclas)
    alturas = plano.altura(centros)
    esmalte = centros[(hu >= HU_ARBITRO) & (alturas >= lo) & (alturas <= hi)]
    if len(esmalte) < 10:
        return np.empty((0, 3)), np.empty((0, 2))
    n = plano.normal
    if n[2] <= 1e-6:
        raise ValueError("el plano oclusal es vertical: no hay losa axial que lo envuelva")
    # Para cada (x, y) del esmalte, la z del plano a altura h es z0 + (h − a)/n_z.
    a = (esmalte[:, :2] - plano.punto[:2]) @ n[:2]
    z_min = plano.punto[2] + (lo - float(a.max())) / n[2]
    z_max = plano.punto[2] + (hi - float(a.min())) / n[2]
    puntos, normales = contornos_con_normales(centros, hu, umbral, z_min, z_max,
                                              desplazamiento="8-vecinos")
    if len(puntos) == 0:
        return puntos, normales
    h = plano.altura(puntos)
    dentro = (h >= lo) & (h <= hi)
    dentro &= cKDTree(esmalte[:, :2]).query(puntos[:, :2])[0] < RADIO_ARCADA_MM
    if solo_esmalte:
        dentro &= junto_al_esmalte(puntos, centros, hu, radio)
    return puntos[dentro], normales[dentro]


def rejilla_gruesa(P: np.ndarray, paso: float = REJILLA_GRUESA_MM) -> np.ndarray:
    """Índices de un representante por vóxel: el punto más cercano al centroide del vóxel.

    Los empates se rompen por coordenadas, no por índice, y la rejilla está en mm del
    DICOM: solo depende del CBCT, nunca del escáner ni de su orden.
    """
    P = np.asarray(P, dtype=np.float64)
    if len(P) == 0:
        return np.empty(0, dtype=np.int64)
    clave = np.floor(P / paso).astype(np.int64)
    _, inversa = np.unique(clave, axis=0, return_inverse=True)
    inversa = np.asarray(inversa).ravel()
    centroide = np.zeros((int(inversa.max()) + 1, 3))
    np.add.at(centroide, inversa, P)
    centroide /= np.bincount(inversa)[:, None]
    d = np.linalg.norm(P - centroide[inversa], axis=1)
    orden = np.lexsort((P[:, 2], P[:, 1], P[:, 0], d, inversa))
    primero = np.r_[True, inversa[orden][1:] != inversa[orden][:-1]]
    return np.sort(orden[primero])


def seccion_malla(
    V: np.ndarray, F: np.ndarray, z: float, *, caras: np.ndarray | None = None,
    paso: float = PASO_SECCION_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """`(puntos_xy, cara)`: la intersección exacta de la malla con el plano horizontal `z`.

    Cada cara que cruza el plano deja un segmento entre sus dos aristas cortadas
    (interpolación lineal). El segmento se muestrea cada `paso` mm por su longitud, en los
    puntos medios de cada tramo, de modo que la densidad no depende del teselado y dos
    caras vecinas no repiten su punto común. `cara` dice de qué cara sale cada punto.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if caras is None:
        caras = np.arange(len(F))
    tri = F[caras]
    d = V[:, 2] - z
    dt = d[tri]
    arriba = dt >= 0
    cruza = arriba.any(axis=1) & ~arriba.all(axis=1)
    tri, dt, arriba, caras = tri[cruza], dt[cruza], arriba[cruza], np.asarray(caras)[cruza]
    if len(tri) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=np.int64)
    puntos = np.empty((len(tri), 3, 2))
    corta = np.zeros((len(tri), 3), dtype=bool)
    for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        corta[:, k] = arriba[:, a] != arriba[:, b]
        da, db = dt[:, a], dt[:, b]
        den = np.where(corta[:, k], da - db, 1.0)
        f = np.where(corta[:, k], da / den, 0.0)
        va, vb = V[tri[:, a], :2], V[tri[:, b], :2]
        puntos[:, k] = va + f[:, None] * (vb - va)
    dos = np.argsort(~corta, axis=1, kind="stable")[:, :2]
    fila = np.arange(len(tri))
    p, q = puntos[fila, dos[:, 0]], puntos[fila, dos[:, 1]]
    largo = np.linalg.norm(q - p, axis=1)
    n = np.maximum(1, np.ceil(largo / paso).astype(np.int64))
    cual = np.repeat(fila, n)
    inicio = np.cumsum(n) - n
    k = np.arange(int(n.sum())) - np.repeat(inicio, n)
    frac = (k + 0.5) / n[cual]
    return p[cual] + frac[:, None] * (q[cual] - p[cual]), caras[cual]


def _mas_cercano_en_triangulos(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Punto más próximo a cada `p` sobre su triángulo `(a, b, c)`. Ericson, *Real-Time
    Collision Detection* §5.1.5, vectorizado por regiones de Voronoi del triángulo."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    salida = np.empty_like(p)
    hecho = np.zeros(len(p), dtype=bool)

    def pon(mascara: np.ndarray, valor: np.ndarray) -> None:
        m = mascara & ~hecho
        salida[m] = valor[m]
        hecho[m] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        pon((d1 <= 0) & (d2 <= 0), a)
        pon((d3 >= 0) & (d4 <= d3), b)
        pon((d6 >= 0) & (d5 <= d6), c)
        v = d1 / (d1 - d3)
        pon((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        w = d2 / (d2 - d6)
        pon((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        pon((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w2[:, None] * (c - b))
        den = 1.0 / (va + vb + vc)
        pon(np.ones(len(p), dtype=bool), a + (vb * den)[:, None] * ab + (vc * den)[:, None] * ac)
    return salida


# --------------------------------------------------------------------------- #
# El escáner, preparado una vez
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Escaner:
    """Todo lo del escáner que el registro consulta, en el marco del propio fichero."""

    V: np.ndarray
    F: np.ndarray
    N: np.ndarray
    etiquetas: np.ndarray
    banda: np.ndarray
    base: np.ndarray  # filas: derecha, anterior, superior
    oclusal: np.ndarray
    centro: np.ndarray
    arcada: str
    puntas: dict[int, np.ndarray]  # punto de cúspide por diente, en el marco del fichero
    arbol: cKDTree = field(repr=False, compare=False)
    etiqueta_cara: np.ndarray = field(repr=False, compare=False)
    banda_cara: np.ndarray = field(repr=False, compare=False)
    ajuste_cara: np.ndarray = field(repr=False, compare=False)
    normal_cara: np.ndarray = field(repr=False, compare=False)
    caras_de: np.ndarray = field(repr=False, compare=False)
    inicio: np.ndarray = field(repr=False, compare=False)

    def mas_cercano(
        self, q: np.ndarray, *, solo_banda: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(punto, cara, distancia)`: el punto más próximo sobre la malla a cada `q`.

        Se examinan las caras de los `K_VECINOS` vértices más próximos. Con `solo_banda`,
        los puntos sin ningún vértice de la banda entre esos vecinos no se proyectan
        (`cara = −1`, distancia infinita): para que su cara más próxima fuera de la banda,
        sus tres vértices —todos de la banda— tendrían que quedar fuera de los `K_VECINOS`
        más próximos. Son la mayoría del contorno (raíz, hueso, la otra arcada), y es solo
        para no gastar tiempo en ellos.

        Un empate exacto —un punto que proyecta sobre una arista compartida— se resuelve por
        geometría (la
        cara cuya normal está más alineada con el vector), nunca por el índice de la cara,
        para que reordenar el fichero no cambie el resultado.
        """
        q = np.asarray(q, dtype=np.float64)
        _, vecinos = self.arbol.query(q, k=K_VECINOS, workers=_hilos(q))
        mejor_d = np.full(len(q), np.inf)
        mejor_p = np.zeros_like(q)
        mejor_c = np.full(len(q), -1, dtype=np.int64)
        mejor_a = np.full(len(q), -1.0)
        filas = np.arange(len(q))
        if solo_banda:
            filas = np.flatnonzero(self.banda[vecinos].any(axis=1))
        for col in range(vecinos.shape[1]):
            v = vecinos[filas, col]
            n = self.inicio[v + 1] - self.inicio[v]
            for t in range(int(n.max()) if len(n) else 0):
                local = np.flatnonzero(n > t)
                if len(local) == 0:
                    break
                tiene = filas[local]
                c = self.caras_de[self.inicio[v[local]] + t]
                tri = self.V[self.F[c]]
                p = _mas_cercano_en_triangulos(q[tiene], tri[:, 0], tri[:, 1], tri[:, 2])
                dif = q[tiene] - p
                d = np.linalg.norm(dif, axis=1)
                alin = np.abs(np.einsum("ij,ij->i", dif, self.normal_cara[c]))
                alin = alin / np.maximum(d, 1e-300)
                mejora = (d < mejor_d[tiene] - 1e-12) | (
                    (np.abs(d - mejor_d[tiene]) <= 1e-12) & (alin > mejor_a[tiene])
                )
                i = tiene[mejora]
                mejor_d[i], mejor_p[i], mejor_c[i], mejor_a[i] = (
                    d[mejora], p[mejora], c[mejora], alin[mejora]
                )
        return mejor_p, mejor_c, mejor_d

    @classmethod
    def construye(
        cls, V: np.ndarray, F: np.ndarray, etiquetas: np.ndarray, ejes: Any,
        normales: np.ndarray | None = None,
    ) -> Escaner:
        """`ejes` es la base de `anatomical_frame`: `centro`, `oclusal`, `derecha`,
        `anterior`, `superior` y `arcada`."""
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)
        etq = np.asarray(etiquetas, dtype=np.int64)
        if len(etq) != len(V):
            raise ValueError(
                f"las etiquetas del escáner ({len(etq)}) no corresponden a sus {len(V)} vértices"
            )
        N = normales_vertice(V, F) if normales is None else np.asarray(normales, np.float64)
        oclusal = _unitario(ejes.oclusal)
        centro = np.asarray(ejes.centro, dtype=np.float64)
        banda, alturas_punta = banda_escaner(V, etq, oclusal, centro)
        h = (V - centro) @ oclusal
        puntas = {}
        for c, punta in alturas_punta.items():
            m = (etq == c) & (h >= punta)
            puntas[c] = V[m].mean(axis=0)
        e = etq[F]
        etiqueta_cara = np.where((e[:, 0] == e[:, 1]) & (e[:, 1] == e[:, 2]), e[:, 0], 0)
        normal_cara = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        norma = np.linalg.norm(normal_cara, axis=1, keepdims=True)
        normal_cara = np.divide(normal_cara, norma, out=np.zeros_like(normal_cara),
                                where=norma > 0)
        banda_cara = banda[F].all(axis=1)
        ajuste_cara = banda_cara.copy()
        if FILTRA_INCLINACION:
            ajuste_cara &= caras_de_pared(
                h[F].mean(axis=1), np.abs(normal_cara @ oclusal), etiqueta_cara, banda_cara,
                alturas_punta,
            )
        orden = np.argsort(F.ravel(), kind="stable")
        inicio = np.searchsorted(F.ravel()[orden], np.arange(len(V) + 1))
        base = np.stack([_unitario(ejes.derecha), _unitario(ejes.anterior),
                         _unitario(ejes.superior)])
        return cls(
            V=V, F=F, N=N, etiquetas=etq, banda=banda, base=base, oclusal=oclusal,
            centro=centro, arcada=str(ejes.arcada), puntas=puntas, arbol=cKDTree(V),
            etiqueta_cara=etiqueta_cara, banda_cara=banda_cara, ajuste_cara=ajuste_cara,
            normal_cara=normal_cara, caras_de=orden // 3, inicio=inicio,
        )

    def dientes_en_banda(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.etiquetas[self.banda]) if c > 0)


def marco_geometrico(V: np.ndarray, etiquetas: np.ndarray) -> np.ndarray:
    """El anterior del escáner SIN usar las etiquetas para orientarlo.

    El eje oclusal es el menor de la nube con el signo de las etiquetas, como en
    `anatomical_frame`; el anterior, el del vértice de la parábola de **grado 2** ajustada
    a la cresta, cuyos
    brazos van hacia posterior. No es `marco.marco_canonico`, que lee `coef[0]` de un
    ajuste de grado 4 —el término cuártico, no la apertura— y en los escaneos superiores
    del caso B da el anterior al revés (ver el *issue* abierto).
    """
    V = np.asarray(V, dtype=np.float64)
    etq = np.asarray(etiquetas)
    centro = V.mean(axis=0)
    ejes = np.linalg.svd(V - centro, full_matrices=False)[2]
    oclusal = ejes[2]
    if (V[etq > 0].mean(axis=0) - V[etq == 0].mean(axis=0)) @ oclusal < 0:
        oclusal = -oclusal
    P = (V - centro) @ np.stack([ejes[0], ejes[1], oclusal]).T
    cresta = P[:, 2] > np.percentile(P[:, 2], 80)
    a = np.polyfit(P[cresta, 0], P[cresta, 1], 2)[0]
    return -ejes[1] if a > 0 else ejes[1]


# --------------------------------------------------------------------------- #
# El registrador
# --------------------------------------------------------------------------- #
def _por_corte(P: np.ndarray, N2: np.ndarray | None) -> list[tuple[float, np.ndarray, Any]]:
    z = np.round(np.asarray(P)[:, 2], 3)
    salida = []
    for valor in np.unique(z):
        m = z == valor
        salida.append((float(valor), P[m, :2], None if N2 is None else N2[m]))
    return salida


def _hilos(q: np.ndarray) -> int:
    """Hilos para una consulta al árbol: en paralelo solo compensa con muchos puntos. El
    resultado de la consulta no depende de esto, solo el tiempo."""
    return -1 if len(q) > 8000 else 1


def _inversa(rot: np.ndarray, tras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rot.T, -rot.T @ tras


@dataclass(frozen=True, eq=False)
class RegistroAnatomico:
    """Registrador (`fusion_agents.registration.Registrar`) anatómico por banda de escáner.

    `__call__(source, target)`: `target` es el contorno del CBCT (mm del DICOM) contra el
    que se ajusta y se puntúa; `source` debe ser la banda del escáner (`V[banda]`) y solo
    se usa como clave de memoria —la geometría completa del escáner la lleva `escaner`,
    porque el ajuste consulta TODOS sus vértices y la puntuación sus caras—.

    La búsqueda se memoriza por el sha256 de las dos entradas: `prepara_anatomico` la
    ejecuta para poder declarar su incertidumbre antes de la fusión, y la llamada de
    `GeometricFusionAgent` la recupera en vez de repetirla.
    """

    escaner: Escaner
    base_cbct: np.ndarray
    plano: Plano
    contornos: dict[float, tuple[np.ndarray, np.ndarray]]
    cache: dict = field(default_factory=dict, compare=False, repr=False)
    # Contorno a `UMBRAL_CONTORNO` con la máscara de corona a otros radios, por radio.
    contornos_corona: dict[float, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, compare=False, repr=False
    )

    # --- piezas -------------------------------------------------------- #
    @property
    def rot0(self) -> np.ndarray:
        return pose_inicial(self.escaner.base, self.base_cbct)

    def arranques(self) -> list[tuple[str, np.ndarray, float]]:
        """81 arranques de rejilla y 9 hipótesis de vuelta por ancla del plano, en orden fijo."""
        r0 = self.rot0
        der, ant, sup = self.base_cbct
        salida = []
        for a in REJILLA_GRADOS:
            for b in REJILLA_GRADOS:
                for c in REJILLA_GRADOS:
                    giro = (rotacion_de_vector(der * math.radians(a))
                            @ rotacion_de_vector(ant * math.radians(b))
                            @ rotacion_de_vector(sup * math.radians(c)))
                    for ancla in self.plano.anclas:
                        for dz in DESPLAZAMIENTOS_VERTICALES_MM:
                            salida.append((f"rejilla({a:+.0f},{b:+.0f},{c:+.0f})",
                                           giro @ r0, ancla + dz))
        for nombre, eje in zip(("derecha", "anterior", "superior"), self.base_cbct, strict=True):
            vuelta = rotacion_de_vector(eje * math.pi) @ r0
            for ancla in self.plano.anclas:
                for dz in DESPLAZAMIENTOS_VERTICALES_MM:
                    salida.append((f"vuelta-{nombre}", vuelta, ancla + dz))
        return salida

    def traslacion_inicial(self, rot: np.ndarray, dz: float, P: np.ndarray) -> np.ndarray:
        """Centra la banda sobre el contorno en planta y pone las cúspides en el plano + dz."""
        esc = self.escaner
        der, ant, sup = self.base_cbct
        planta = np.stack([der, ant]).T
        banda = esc.V[esc.banda] @ rot.T
        corona = esc.V[esc.etiquetas > 0] @ rot.T
        dp = (P @ planta).mean(axis=0) - (banda @ planta).mean(axis=0)
        cuspide = float(np.percentile(self.plano.altura(corona),
                                      1 if esc.arcada == "upper" else 99))
        return der * dp[0] + ant * dp[1] + sup * (dz - cuspide)

    def empareja(
        self, q: np.ndarray, *, superficie: bool = True, excluye: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`(validos, punto, normal, distancia)` de cada punto `q` (marco del escáner).

        Con `superficie`, el punto más próximo SOBRE LA MALLA (Ericson, entre las caras de
        los `K_VECINOS` vértices más próximos) y la normal de esa cara; la pareja vale si la
        cara es de la banda. Sin `superficie`, el vértice más próximo y su normal; vale si
        el vértice es de la banda. La etapa gruesa usa vértices (rápido, solo tiene que
        dejar cada arranque en su cuenca); el ajuste fino y todo lo que se declara, la
        superficie: emparejar con el vértice sesga la altura según el espaciado de la malla
        —0,05 a 0,17 mm en el fantasma con distintos teselados de la misma superficie—, y
        la altura es justo la dirección peor restringida.
        """
        esc = self.escaner
        if not superficie:
            d, j = esc.arbol.query(q, workers=_hilos(q))
            ok = esc.banda[j]
            if excluye is not None:
                ok &= esc.etiquetas[j] != excluye
            return ok, esc.V[j], esc.N[j], d
        punto, cara, d = esc.mas_cercano(q, solo_banda=True)
        ok = (cara >= 0) & esc.ajuste_cara[np.maximum(cara, 0)]
        if excluye is not None:
            ok &= esc.etiqueta_cara[cara] != excluye
        return ok, punto, esc.normal_cara[cara], d

    def afina(
        self, rot: np.ndarray, tras: np.ndarray, P: np.ndarray, *, iters: int = ITER_FINO,
        tol: float = TOL_MM, excluye: int | None = None, superficie: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """ICP punto a plano recortado del contorno `P` al escáner, desde `(rot, tras)`.

        Trabaja con la pose inversa: el contorno se lleva al marco del escáner y se empareja
        con TODO el escáner (`empareja`); la pareja cuenta solo si cae en la banda (y no en
        el diente `excluye`). Devuelve la pose escáner→CBCT, las iteraciones y si convergió.

        ⚠️ **Ciclos.** Con recorte, el conjunto de parejas puede alternar entre dos estados y
        el ICP oscilar con periodo 2 o más sin llegar nunca a la tolerancia. Si la pose
        vuelve a una de las `MEMORIA_CICLO` anteriores (a menos de `tol`), se da por
        convergido y se devuelve el elemento del ciclo de menor coste recortado: así el
        resultado no depende de en qué paridad se agotan las iteraciones.
        """
        ri, ti = _inversa(rot, tras)
        lo, hi = P.min(axis=0), P.max(axis=0)
        sonda = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                          for z in (lo[2], hi[2])])
        historia: list[tuple[np.ndarray, np.ndarray, float]] = []
        convergido, k = False, 0
        while k < iters:
            k += 1
            q = P @ ri.T + ti
            ok, punto, normal, d = self.empareja(q, superficie=superficie, excluye=excluye)
            idx = np.flatnonzero(ok)
            if len(idx) < CORRESPONDENCIAS_MINIMAS:
                break
            usa = idx[d[idx] <= np.quantile(d[idx], TRIM)]
            coste = float(np.mean(d[usa]))
            dr, dt = paso_punto_plano(q[usa], punto[usa], normal[usa], tope=TOPE_PASO)
            historia.append((ri, ti, coste))
            ri, ti = dr @ ri, dr @ ti + dt
            movido = float(np.max(np.linalg.norm(q[usa] @ dr.T + dt - q[usa], axis=1)))
            if movido < tol:
                convergido = True
                break
            aqui = sonda @ ri.T + ti
            for atras in range(2, min(MEMORIA_CICLO, len(historia)) + 1):
                r0, t0, _ = historia[-atras]
                if float(np.max(np.linalg.norm(sonda @ r0.T + t0 - aqui, axis=1))) < tol:
                    ciclo = historia[-atras:]
                    ri, ti, _ = min(ciclo, key=lambda e: e[2])
                    convergido = True
                    break
            if convergido:
                break
        rot, tras = _inversa(ri, ti)
        return rot, tras, k, convergido

    def correspondencias(
        self, rot: np.ndarray, tras: np.ndarray, P: np.ndarray, excluye: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`(q, punto, normal, residuo)` de las parejas en banda: contorno en el marco del
        escáner, su punto sobre la malla, la normal de esa cara y la distancia punto a plano
        con signo según esa normal (cuyo sentido es el del fichero, no anatómico)."""
        ri, ti = _inversa(rot, tras)
        q = P @ ri.T + ti
        ok, punto, normal, _ = self.empareja(q, excluye=excluye)
        q, punto, normal = q[ok], punto[ok], normal[ok]
        return q, punto, normal, np.einsum("ij,ij->i", q - punto, normal)

    def desplazamiento(self, a: tuple, b: tuple) -> np.ndarray:
        """Desplazamiento por vértice de la banda entre dos poses, en mm del DICOM."""
        x = self.escaner.V[self.escaner.banda]
        return (x @ a[0].T + a[1]) - (x @ b[0].T + b[1])

    def rms_desplazamiento(self, a: tuple, b: tuple) -> float:
        return float(np.sqrt(np.mean(np.sum(self.desplazamiento(a, b) ** 2, axis=1))))

    def puntua_secciones(
        self, rot: np.ndarray, tras: np.ndarray, por_corte: list, *, solo: int | None = None
    ) -> dict:
        """S, C y el detalle por diente de una pose. Ver `docs/research/registro-anatomico.md`.

        Para cada corte del CBCT: se corta el escáner movido en esa z exacta; por diente,
        la media de dos medianas —sección→contorno y contorno asignado→sección—; un punto de
        contorno se asigna al diente de su punto de sección más próximo si ese punto es de
        banda y está a ≤ `TOPE_SECCION_MM`. S_t es la mediana sobre cortes; S, la mediana
        sobre dientes, todos pesando igual. C es el mínimo de (a) la fracción de la sección
        de banda con contorno a ≤ 1 mm y (b) entre los puntos de contorno cuyo punto de
        sección más próximo es de banda, la fracción a ≤ 1 mm.

        Con normales en `por_corte` se mide además el residuo con signo por diente:
        + la sección del escáner queda fuera del diente del CBCT (flota), − dentro (hundida).
        """
        esc = self.escaner
        vt = esc.V @ rot.T + tras
        zf = vt[esc.F][:, :, 2]
        zmin, zmax = zf.min(axis=1), zf.max(axis=1)
        por_diente: dict[int, list[float]] = {}
        firmados: dict[int, list[np.ndarray]] = {}
        a_num = a_den = b_num = b_den = 0
        for z, c2, n2 in por_corte:
            caras = np.flatnonzero((zmin <= z) & (zmax >= z))
            if len(caras) == 0 or len(c2) == 0:
                continue
            s2, cara = seccion_malla(vt, esc.F, z, caras=caras)
            if len(s2) == 0:
                continue
            lab = esc.etiqueta_cara[cara]
            en_banda = esc.banda_cara[cara] & (lab > 0)
            if solo is not None:
                en_banda &= lab == solo
            # S solo con caras de pared: las demasiado inclinadas ni puntúan ni reciben
            # contorno (`FILTRA_INCLINACION`). C, con la banda entera.
            de_pared = en_banda & esc.ajuste_cara[cara]
            d_s, j_s = cKDTree(c2).query(s2)
            d_c, j_c = cKDTree(s2).query(c2)
            a_den += int(en_banda.sum())
            a_num += int((d_s[en_banda] <= TOL_COBERTURA_MM).sum())
            cerca_banda = en_banda[j_c]
            # El lado del contorno se cuenta sobre TODOS los puntos cuyo punto de sección
            # más próximo es de banda, como antes de la máscara de esmalte. Restringirlo a
            # los asignados (≤ `TOPE_SECCION_MM`) se propuso para que la pose del registro
            # por cortes del caso A fuera elegible, y en el fantasma solo subía C en poses
            # equivocadas: elegir así es elegir con el método antiguo (§8 de la nota).
            b_den += int(cerca_banda.sum())
            b_num += int((d_c[cerca_banda] <= TOL_COBERTURA_MM).sum())
            asignado = de_pared[j_c] & (d_c <= TOPE_SECCION_MM)
            diente_c = lab[j_c]
            for f in np.unique(lab[de_pared]):
                s_f = de_pared & (lab == f)
                c_f = asignado & (diente_c == f)
                if c_f.any():
                    por_diente.setdefault(int(f), []).append(
                        0.5 * (float(np.median(d_s[s_f])) + float(np.median(d_c[c_f])))
                    )
                if n2 is not None:
                    util = s_f & (d_s <= TOPE_SECCION_MM)
                    if util.any():
                        firmados.setdefault(int(f), []).append(np.einsum(
                            "ij,ij->i", s2[util] - c2[j_s[util]], n2[j_s[util]]
                        ))
        s_t = {f: float(np.median(v)) for f, v in sorted(por_diente.items())}
        cobertura_a = a_num / a_den if a_den else 0.0
        cobertura_b = b_num / b_den if b_den else 0.0
        salida = {
            "S": float(np.median(list(s_t.values()))) if s_t else float("inf"),
            "C": float(min(cobertura_a, cobertura_b)),
            "C_scanner_side": round(float(cobertura_a), 4),
            "C_contour_side": round(float(cobertura_b), 4),
            "per_tooth": s_t,
        }
        if firmados:
            residuo = {}
            for f, partes in sorted(firmados.items()):
                v = np.concatenate(partes)
                med = float(np.median(v))
                residuo[f] = {
                    "median": med,
                    "mad": float(np.median(np.abs(v - med))),
                    "p95_abs": float(np.percentile(np.abs(v), 95)),
                    "n": int(len(v)),
                }
            salida["signed_residual"] = residuo
        return salida

    def coherencia(self, rot: np.ndarray, tras: np.ndarray, puntuacion: Mapping) -> dict:
        """Elegibilidad de una pose: anatomía, cobertura y altura de las cúspides."""
        angulos = [
            angulo_grados(rot @ self.escaner.base[i], self.base_cbct[i]) for i in range(3)
        ]
        puntas = np.stack(list(self.escaner.puntas.values())) @ rot.T + tras
        h = float(np.median(self.plano.altura(puntas)))
        altura = min((h - a for a in self.plano.anclas), key=abs)
        motivos = []
        if max(angulos) > ANGULO_ANATOMICO_MAX:
            motivos.append(
                "ejes anatómicos a " + "/".join(f"{a:.0f}" for a in angulos)
                + f"° de los del CBCT (máximo {ANGULO_ANATOMICO_MAX:.0f}°)"
            )
        if puntuacion["C"] < COBERTURA_MIN:
            motivos.append(f"cobertura {puntuacion['C']:.2f} (mínimo {COBERTURA_MIN})")
        if abs(altura) > ALTURA_CUSPIDES_MM:
            motivos.append(
                f"cúspides a {altura:+.1f} mm del plano oclusal del CBCT más cercano (máximo "
                f"±{ALTURA_CUSPIDES_MM:.0f}): encajaría contra la otra arcada"
            )
        return {
            "eligible": not motivos,
            "why_not": motivos,
            "axis_angles_deg": [round(a, 2) for a in angulos],
            "cusp_height_mm": round(altura, 3),
            "angle_to_start_deg": round(angulo_rotacion_grados(rot @ self.rot0.T), 2),
        }

    # --- búsqueda ------------------------------------------------------- #
    def __call__(self, source: np.ndarray, target: np.ndarray) -> RegistrationResult:
        source = np.ascontiguousarray(source, dtype=np.float64)
        target = np.ascontiguousarray(target, dtype=np.float64)
        clave = hashlib.sha256(source.tobytes() + target.tobytes()).hexdigest()
        if clave not in self.cache:
            self.cache[clave] = self._busca(target)
        self.cache["ultima"] = self.cache[clave]
        return self.cache[clave]["result"]

    def _normales_de(self, P: np.ndarray) -> np.ndarray | None:
        for p, n2 in self.contornos.values():
            if p.shape == P.shape and np.array_equal(p, P):
                return n2
        return None

    def _busca(self, P: np.ndarray) -> dict:
        grueso = P[rejilla_gruesa(P)]
        por_corte = _por_corte(P, self._normales_de(P))

        # 1 · Etapa gruesa desde cada arranque.
        poses = []
        for nombre, rot, dz in self.arranques():
            tras = self.traslacion_inicial(rot, dz, P)
            r, t, _, _ = self.afina(rot, tras, grueso, iters=ITER_GRUESO, tol=TOL_GRUESO_MM,
                                   superficie=False)
            poses.append((nombre, r, t))

        # 2 · Cuencas gruesas, en el orden de los arranques; el primero de cada una la
        # representa.
        gruesas: list[dict] = []
        for nombre, r, t in poses:
            for c in gruesas:
                if self.rms_desplazamiento((r, t), c["pose"]) < RADIO_CUENCA_GRUESA_MM:
                    c["starts"].append(nombre)
                    break
            else:
                gruesas.append({"pose": (r, t), "starts": [nombre]})

        # 3 · Tramo corto con todo el contorno para todas, y reagrupación. Una cuenca que
        # en ese tramo no ha convergido suele estar deslizándose por el arco, y no es un
        # mínimo: no merece 200 iteraciones si no puede ganar.
        cuencas: list[dict] = []
        for c in gruesas:
            r0, t0 = c["pose"]
            r, t, iteraciones, convergido = self.afina(r0, t0, P, iters=ITER_MEDIO,
                                                       superficie=False)
            for d in cuencas:
                if self.rms_desplazamiento((r, t), d["pose"]) < RADIO_CUENCA_GRUESA_MM:
                    d["starts"] += c["starts"]
                    break
            else:
                cuencas.append({"pose": (r, t), "starts": list(c["starts"]),
                                "iterations": iteraciones, "converged": convergido,
                                "refined": False})

        def puntua(c: dict) -> None:
            r0, t0 = c["pose"]
            c["score"] = self.puntua_secciones(r0, t0, por_corte)
            c["check"] = self.coherencia(r0, t0, c["score"])
            c["flip_start"] = all(s.startswith("vuelta") for s in c["starts"])

        def refina(c: dict) -> None:
            if c["refined"]:
                return
            # El tramo corto emparejó con vértices; el fino, con la superficie, siempre.
            r0, t0 = c["pose"]
            r, t, iteraciones, convergido = self.afina(r0, t0, P, iters=ITER_FINO)
            c.update(pose=(r, t), iterations=c["iterations"] + iteraciones,
                     converged=convergido)
            c["refined"] = True
            puntua(c)

        for c in cuencas:
            puntua(c)

        # 4 · Hasta converger solo las que pueden ganar: mientras haya una elegible sin
        # refinar cuya S del tramo corto quede a menos de `MARGEN_AMBIGUEDAD_S_MM` +
        # `HOLGURA_FINALISTA_MM` de la mejor ya refinada (o ninguna refinada todavía), se
        # refina la de menor S. Y siempre la mejor vuelta, para medir su margen.
        while True:
            hechas = [c for c in cuencas if c["refined"] and c["check"]["eligible"]]
            mejor = min((c["score"]["S"] for c in hechas), default=math.inf)
            limite = mejor + MARGEN_AMBIGUEDAD_S_MM + HOLGURA_FINALISTA_MM
            pendientes = sorted(
                (c for c in cuencas if not c["refined"] and c["check"]["eligible"]
                 and c["score"]["S"] <= limite),
                key=lambda c: c["score"]["S"],
            )
            if not pendientes:
                break
            refina(pendientes[0])
        vueltas_cortas = [
            c for c in cuencas
            if not c["check"]["eligible"] and c["check"]["angle_to_start_deg"] > ANGULO_VUELTA
        ]
        if vueltas_cortas:
            refina(min(vueltas_cortas, key=lambda c: c["score"]["S"]))

        # Las refinadas que han caído en la misma cuenca fina se funden.
        finas: list[dict] = []
        for c in cuencas:
            if not c["refined"]:
                finas.append(c)
                continue
            for d in finas:
                if d["refined"] and self.rms_desplazamiento(c["pose"], d["pose"]) < RADIO_CUENCA_MM:
                    d["starts"] += c["starts"]
                    break
            else:
                finas.append(c)
        cuencas = finas

        elegibles = [c for c in cuencas if c["refined"] and c["check"]["eligible"]]
        if elegibles:
            mejor_s = min(c["score"]["S"] for c in elegibles)
            empatadas = [c for c in elegibles if c["score"]["S"] - mejor_s < 1e-3]
            elegida = min(empatadas, key=lambda c: c["check"]["angle_to_start_deg"])
            veto = None
        else:
            elegida = min(cuencas, key=lambda c: max(c["check"]["axis_angles_deg"]))
            refina(elegida)
            veto = (
                "ninguna pose candidata es anatómicamente coherente con el CBCT ("
                + "; ".join(elegida["check"]["why_not"] or ["no queda ninguna"])
                + "): se devuelve la más cercana a la orientación anatómica, sin confianza"
            )

        # 5 · Señales: vuelta no excluida por los datos y ambigüedad de cuenca.
        vueltas = [
            c for c in cuencas
            if c["refined"] and not c["check"]["eligible"]
            and c["check"]["angle_to_start_deg"] > ANGULO_VUELTA
        ]
        margen_vuelta = None
        if vueltas and math.isfinite(elegida["score"]["S"]):
            v = min(vueltas, key=lambda c: c["score"]["S"])
            margen_vuelta = {
                "S_difference_mm": round(v["score"]["S"] - elegida["score"]["S"], 4),
                "C_difference": round(elegida["score"]["C"] - v["score"]["C"], 3),
                "angle_deg": v["check"]["angle_to_start_deg"],
            }
        ambigua = None
        for c in elegibles:
            if c is elegida:
                continue
            if c["score"]["S"] <= elegida["score"]["S"] + MARGEN_AMBIGUEDAD_S_MM:
                lejos = self.rms_desplazamiento(c["pose"], elegida["pose"])
                if lejos > DESPLAZAMIENTO_AMBIGUO_MM and (
                    ambigua is None or lejos > ambigua["rms_mm"]
                ):
                    ambigua = {"rms_mm": round(lejos, 3), "S": c["score"]["S"],
                               "pose": c["pose"]}

        rot, tras = elegida["pose"]
        _, _, _, r = self.correspondencias(rot, tras, P)
        cerca = np.abs(r) < DEFAULT_OVERLAP_MM
        resultado = RegistrationResult(
            rotation=matrix_to_quaternion(rot),
            translation=(float(tras[0]), float(tras[1]), float(tras[2])),
            rms_mm=float(np.sqrt(np.mean(r**2))) if len(r) else float("inf"),
            iterations=int(elegida["iterations"]),
            converged=bool(elegida["converged"]),
            rms_overlap_mm=float(np.sqrt(np.mean(r[cerca] ** 2))) if cerca.any() else None,
            overlap_fraction=float(elegida["score"]["C"]),
            veto=veto,
        )
        return {
            "result": resultado,
            "chosen": elegida,
            "basins": cuencas,
            "flip_margin": margen_vuelta,
            "ambiguous": ambigua,
            "n_starts": len(poses),
            "por_corte": por_corte,
            "P": P,
        }

    # --- incertidumbre -------------------------------------------------- #
    def incertidumbre(self) -> dict:
        """Lo que la pose elegida admite moverse ante elecciones declaradas. Ver el módulo.

        Todas las reestimaciones parten de la pose elegida (solo ajuste fino, sin búsqueda).
        Por vértice de la banda: sensibilidad al umbral h, navaja dejando fuera cada diente
        σ_J, ambigüedad de cuenca a y radio de la máscara de corona c (reajuste con
        `RADIOS_ESMALTE_INCERTIDUMBRE`); U = √(h² + σ_J² + a² + c²). Además, el residuo con signo
        por diente, el condicionamiento de la traslación y la escala que pediría un modelo
        de semejanza.
        """
        busqueda = self.cache["ultima"]
        rot, tras = busqueda["chosen"]["pose"]
        P = busqueda["P"]
        # La vertical es la del escáner registrado: el CBCT no la mide (ver `marco_cbct`).
        sup = _unitario(rot @ self.escaner.base[2])
        banda_fdi = self.escaner.etiquetas[self.escaner.banda]

        # Umbral.
        h_thr = np.zeros(int(self.escaner.banda.sum()))
        h_thr_v = np.zeros_like(h_thr)
        umbral: dict[str, Any] = {}
        for u, (pu, _) in sorted(self.contornos.items()):
            if pu.shape == P.shape and np.array_equal(pu, P):
                continue
            if len(pu) < MINIMO_CONTORNO:
                umbral[str(int(u))] = "sin contorno suficiente"
                continue
            r_u, t_u, _, conv = self.afina(rot, tras, pu)
            d = self.desplazamiento((r_u, t_u), (rot, tras))
            norma = np.linalg.norm(d, axis=1)
            h_thr = np.maximum(h_thr, norma)
            h_thr_v = np.maximum(h_thr_v, np.abs(d @ sup))
            umbral[str(int(u))] = {"median_mm": round(float(np.median(norma)), 3),
                                   "max_mm": round(float(norma.max()), 3), "converged": conv}

        # Navaja dejando fuera cada diente.
        dientes = self.escaner.dientes_en_banda()
        deltas, prediccion = [], {}
        for k in dientes:
            r_k, t_k, _, _ = self.afina(rot, tras, P, excluye=k)
            deltas.append(self.desplazamiento((r_k, t_k), (rot, tras)))
            sola = self.puntua_secciones(r_k, t_k, busqueda["por_corte"], solo=k)
            pred = {"S_t": sola["per_tooth"].get(k)}
            if "signed_residual" in sola and k in sola["signed_residual"]:
                pred["signed_median"] = sola["signed_residual"][k]["median"]
            prediccion[k] = pred
        if len(deltas) >= 2:
            D = np.stack(deltas)
            dm = D - D.mean(axis=0)
            n = len(deltas)
            sigma_j = np.sqrt((n - 1) / n * np.sum(dm**2, axis=(0, 2)))
            sigma_j_v = np.sqrt((n - 1) / n * np.sum((dm @ sup) ** 2, axis=0))
        else:
            sigma_j = sigma_j_v = np.zeros_like(h_thr)

        # Ambigüedad de cuenca.
        amb = busqueda["ambiguous"]
        if amb is not None:
            d = self.desplazamiento(amb["pose"], (rot, tras))
            a, a_v = np.linalg.norm(d, axis=1), np.abs(d @ sup)
        else:
            a = a_v = np.zeros_like(h_thr)

        # Qué cuenta como corona: el radio de la máscara de esmalte.
        c_cor = np.zeros_like(h_thr)
        c_cor_v = np.zeros_like(h_thr)
        corona: dict[str, Any] = {}
        for radio, (pr, _) in sorted(self.contornos_corona.items()):
            if len(pr) < MINIMO_CONTORNO:
                corona[str(radio)] = "sin contorno suficiente"
                continue
            r_c, t_c, _, conv = self.afina(rot, tras, pr)
            d = self.desplazamiento((r_c, t_c), (rot, tras))
            norma = np.linalg.norm(d, axis=1)
            c_cor = np.maximum(c_cor, norma)
            c_cor_v = np.maximum(c_cor_v, np.abs(d @ sup))
            corona[str(radio)] = {"median_mm": round(float(np.median(norma)), 3),
                                  "max_mm": round(float(norma.max()), 3), "converged": conv}

        U = np.sqrt(h_thr**2 + sigma_j**2 + a**2 + c_cor**2)
        U_v = np.sqrt(h_thr_v**2 + sigma_j_v**2 + a_v**2 + c_cor_v**2)
        por_diente = {}
        for k in dientes:
            m = banda_fdi == k
            por_diente[k] = {
                "U_median_mm": round(float(np.median(U[m])), 3),
                "U_p95_mm": round(float(np.percentile(U[m], 95)), 3),
                "U_vertical_median_mm": round(float(np.median(U_v[m])), 3),
                "threshold_median_mm": round(float(np.median(h_thr[m])), 3),
                "jackknife_median_mm": round(float(np.median(sigma_j[m])), 3),
                "basin_median_mm": round(float(np.median(a[m])), 3),
                "crown_mask_median_mm": round(float(np.median(c_cor[m])), 3),
                "left_out_prediction": prediccion.get(k),
            }

        # Condicionamiento de la traslación (complemento de Schur sobre la rotación).
        q, punto, normal, _ = self.correspondencias(rot, tras, P)
        valores, vectores, sigma = informacion_6dof(q, punto, normal)
        matriz = (vectores * valores) @ vectores.T
        trans = matriz[3:, 3:] - matriz[3:, :3] @ np.linalg.pinv(matriz[:3, :3]) @ matriz[:3, 3:]
        lt, vt = np.linalg.eigh(trans)
        peor = vt[:, 0]
        # La pose se ajusta en el marco del escáner: su traslación se lleva al del CBCT.
        peor_cbct = self.base_cbct @ (rot @ peor)
        condicionamiento = {
            "translation_eigenvalues": [float(x) for x in lt],
            "worst_direction_anatomical": {
                "right": round(float(peor_cbct[0]), 3),
                "anterior": round(float(peor_cbct[1]), 3),
                "superior": round(float(peor_cbct[2]), 3),
            },
            "worst_axis": ["right", "anterior", "superior"][int(np.argmax(np.abs(peor_cbct)))],
            "residual_sigma_mm": round(float(sigma), 4),
            "formal_sigma_worst_mm": round(float(sigma / math.sqrt(max(lt[0], 1e-300))), 5),
            "formal_is_optimistic": True,
        }

        # Escala: semejanza de 7 GdL desde la pose elegida, solo para declarar.
        ri, ti = _inversa(rot, tras)
        s_total = 1.0
        for _ in range(ITER_ESCALA):
            qq = P @ ri.T + ti
            valido, punto, normal, d = self.empareja(qq)
            ok = np.flatnonzero(valido)
            if len(ok) < CORRESPONDENCIAS_MINIMAS:
                break
            usa = ok[d[ok] <= np.quantile(d[ok], TRIM)]
            r7, t7, s7 = paso_similitud_punto_plano(qq[usa], punto[usa], normal[usa])
            ri, ti = s7 * (r7 @ ri), s7 * (r7 @ ti) + t7
            s_total *= s7
        escala = 1.0 / s_total

        return {
            "per_tooth": por_diente,
            "U_median_mm": round(float(np.median(U)), 3),
            "U_p95_mm": round(float(np.percentile(U, 95)), 3),
            "U_vertical_median_mm": round(float(np.median(U_v)), 3),
            "threshold": umbral,
            "crown_mask_radius": corona,
            "conditioning": condicionamiento,
            "scale_scanner_to_cbct": round(escala, 5),
            "signed_residual": busqueda["chosen"]["score"].get("signed_residual"),
            "_U": U,
        }


# --------------------------------------------------------------------------- #
# Preparación para el orquestador
# --------------------------------------------------------------------------- #
def _json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def prepara_anatomico(
    campo: dict[str, np.ndarray],
    hu: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    etiquetas: np.ndarray,
    ejes: Any,
    *,
    normales: np.ndarray | None = None,
    arcada_nombre: str | None = None,
) -> tuple[np.ndarray, np.ndarray, RegistroAnatomico, dict]:
    """`(origen, destino, registrador, informe)` para `GeometricFusionAgent`.

    `ejes` es la base de `export_agents.anatomia.anatomical_frame(V, etiquetas)`; la pasa
    quien llama para no hacer depender este paquete del de exportación. `origen` es la
    banda del escáner y `destino` el contorno del CBCT a `UMBRAL_CONTORNO`. La búsqueda y
    la incertidumbre ya se han calculado al volver, y van en `informe`.

    Lanza `ValueError` si no se puede (sin esmalte, sin contorno, ejes en espejo, banda
    vacía). Quien llama decide si cae a otro camino, y lo declara.
    """
    centros = np.asarray(campo["centers"], dtype=np.float64)
    if "origin" in campo:
        centros = centros + np.asarray(campo["origin"], dtype=np.float64)
    hu = np.asarray(hu)
    escaner = Escaner.construye(V, F, etiquetas, ejes, normales)
    if escaner.banda.sum() < MINIMO_CONTORNO:
        raise ValueError(
            f"la banda de corona del escáner tiene solo {int(escaner.banda.sum())} vértices"
        )
    base_cbct, plano, informe_marco = marco_cbct(centros, hu, escaner.arcada)
    pose_inicial(escaner.base, base_cbct)  # lanza si las quiralidades no coinciden

    contornos = {}
    for u in (UMBRAL_CONTORNO, *UMBRALES_INCERTIDUMBRE):
        contornos[float(u)] = contornos_losa(centros, hu, u, plano, escaner.arcada)
    destino = contornos[float(UMBRAL_CONTORNO)][0]
    if len(destino) < MINIMO_CONTORNO:
        raise ValueError(f"solo {len(destino)} puntos de contorno en la losa del CBCT")

    corona = {
        float(r): contornos_losa(centros, hu, UMBRAL_CONTORNO, plano, escaner.arcada, radio=r)
        for r in RADIOS_ESMALTE_INCERTIDUMBRE
    }
    registrador = RegistroAnatomico(
        escaner=escaner, base_cbct=base_cbct, plano=plano, contornos=contornos,
        contornos_corona=corona,
    )
    origen = escaner.V[escaner.banda]
    resultado = registrador(origen, destino)
    busqueda = registrador.cache["ultima"]
    incertidumbre = registrador.incertidumbre()

    anterior_g = marco_geometrico(escaner.V, escaner.etiquetas)
    marcos = {
        # Solo el anterior: el eje oclusal sin etiquetas sería el mismo cálculo que hace
        # `anatomical_frame`, y una comprobación que no puede fallar no comprueba nada.
        "anterior_deg": round(angulo_grados(anterior_g, escaner.base[1]), 2),
    }
    rot_elegida = busqueda["chosen"]["pose"][0]
    inclinacion = angulo_grados(rot_elegida @ escaner.base[2], EJES_LPS[2])
    lo, hi = limites_losa(escaner.arcada)
    lo, hi = lo + min(plano.anclas), hi + max(plano.anclas)
    z_losa = destino[:, 2]
    informe: dict[str, Any] = {
        "method": "fusion_agents.anatomico",
        "constants_sha256": huella_constantes(),
        "arch": escaner.arcada,
        "cbct_frame": informe_marco,
        "scanner_band": {
            "n_vertices": int(escaner.banda.sum()),
            "teeth": escaner.dientes_en_banda(),
            "mm_below_cusp": [MARGEN_MORDIDA_MM, ALTO_BANDA_MM],
        },
        "contour": {
            "threshold": UMBRAL_CONTORNO,
            "slab_mm": [lo, hi],
            "crown_only": {"enamel_hu": HU_ESMALTE, "radius_mm": RADIO_ESMALTE_MM},
            "n_points": {str(int(u)): int(len(p)) for u, (p, _) in contornos.items()},
            "n_coarse": int(len(rejilla_gruesa(destino))),
            **describe_cortes(cortes(centros), float(z_losa.min()), float(z_losa.max())),
        },
        "search": {
            "n_starts": busqueda["n_starts"],
            "basins": [
                {
                    "starts": len(c["starts"]),
                    "flip_only": c["flip_start"],
                    "refined": c["refined"],
                    "iterations": c["iterations"],
                    "converged": c["converged"],
                    "S": c["score"]["S"],
                    "C": c["score"]["C"],
                    **c["check"],
                }
                for c in busqueda["basins"]
            ],
            "chosen": {
                "starts": len(busqueda["chosen"]["starts"]),
                "S": busqueda["chosen"]["score"]["S"],
                "C": busqueda["chosen"]["score"]["C"],
                "per_tooth_S": busqueda["chosen"]["score"]["per_tooth"],
                **busqueda["chosen"]["check"],
            },
            "flip_margin": busqueda["flip_margin"],
            "ambiguous": None if busqueda["ambiguous"] is None else {
                k: v for k, v in busqueda["ambiguous"].items() if k != "pose"
            },
            "veto": resultado.veto,
        },
        "frame_check": marcos,
        "tilt_from_registration_deg": round(inclinacion, 2),
        "uncertainty": incertidumbre,
    }
    if arcada_nombre is not None:
        esperada = {"maxilar": "upper", "mandibular": "lower"}.get(arcada_nombre)
        if esperada is not None and esperada != escaner.arcada:
            informe["arch_mismatch"] = (
                f"el nombre del fichero dice {arcada_nombre} y las etiquetas FDI dicen "
                f"{escaner.arcada}: se registra según las etiquetas"
            )
    return origen, destino, registrador, _json(informe)


def motivos(informe: Mapping[str, Any]) -> list[str]:
    """Lo que el gate tiene que saber de un registro anatómico. Siempre algo."""
    inc = informe.get("uncertainty", {})
    cond = inc.get("conditioning", {})
    salida = [
        "registro escáner↔CBCT anatómico por banda de escáner: esto es precisión del "
        "registro frente a elecciones declaradas, no un TRE; la dirección peor condicionada "
        f"es {cond.get('worst_axis', 'la altura')}"
    ]
    busqueda = informe.get("search", {})
    if busqueda.get("veto"):
        salida.append(str(busqueda["veto"]))
    margen = busqueda.get("flip_margin")
    if margen and margen.get("S_difference_mm") is not None and (
        margen["S_difference_mm"] <= MARGEN_VUELTA_S_MM
        or margen["C_difference"] < MARGEN_VUELTA_C
    ):
        salida.append(
            f"una vuelta de {margen['angle_deg']:.0f}° puntúa casi igual (S "
            f"{margen['S_difference_mm']:+.3f} mm, cobertura "
            f"{-margen['C_difference']:+.2f}): los datos no la excluyen, solo la anatomía"
        )
    amb = busqueda.get("ambiguous")
    if amb:
        salida.append(
            f"pose ambigua: otra cuenca puntúa a menos de {MARGEN_AMBIGUEDAD_S_MM} mm de S y mueve "
            f"las coronas {amb['rms_mm']:.2f} mm"
        )
    marcos = informe.get("frame_check", {})
    for eje, clave in (("anterior", "anterior_deg"),):
        if marcos.get(clave) is not None and marcos[clave] > DISCREPANCIA_MARCOS_GRADOS:
            salida.append(
                f"el eje {eje} del escáner medido con etiquetas y sin ellas difiere "
                f"{marcos[clave]:.0f}°: etiquetas espejadas o mal numeradas, o escaneo "
                "degenerado"
            )
    altos = [
        f"{k} ({v['U_median_mm']:.2f} mm)"
        for k, v in inc.get("per_tooth", {}).items()
        if v["U_median_mm"] > INCERTIDUMBRE_MAX_MM
    ]
    if altos:
        salida.append(
            f"incertidumbre del registro por encima de {INCERTIDUMBRE_MAX_MM} mm en "
            + ", ".join(altos)
        )
    sesgados = [
        f"{k} ({v['median']:+.2f} mm)"
        for k, v in (inc.get("signed_residual") or {}).items()
        if abs(v["median"]) > RESIDUO_FIRMADO_MAX_MM
    ]
    if sesgados:
        salida.append(
            "residuo con signo por encima de "
            f"{RESIDUO_FIRMADO_MAX_MM} mm (+ flota, − hundido) en " + ", ".join(sesgados)
        )
    escala = inc.get("scale_scanner_to_cbct")
    if escala is not None and abs(escala - 1) > ESCALA_MAX:
        salida.append(
            f"una semejanza pediría escalar el escáner ×{escala:.4f}: ningún modelo rígido "
            "encaja todas las piezas a la vez"
        )
    if informe.get("arch_mismatch"):
        salida.append(str(informe["arch_mismatch"]))
    return salida
