"""Registration rígido de dos nubes de puntos (ADR 004 §2.6).

El algoritmo vive **detrás de un `Protocol`** y no dentro del agente, por tres
motivos: se puede sustituir sin tocar el contrato, se puede testear el agente con
un registrador trivial, y permite enchufar una etapa gruesa distinta cuando el dato
real diga que hace falta.

**Qué hay implementado y qué no.** Aquí está el **ICP multiescala** (la etapa fina),
en numpy + scipy. La etapa **gruesa** del ADR —RANSAC sobre descriptores FPFH— *no*
está: es la que da una pose inicial cuando las dos nubes están muy desalineadas, y
depende de una librería pesada (Open3D). El `Protocol` deja el hueco preparado.

Consecuencia práctica, dicha sin rodeos: **el ICP de aquí converge solo si la pose
inicial ya está razonablemente cerca**. Con mallas derivadas del propio volumen eso
se cumple por construcción; con un intraoral y un CBCT capturados por separado,
habrá que medirlo antes de dar el registro por bueno.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from core_schemas import RigidTransform
from scipy.spatial import cKDTree

# Banda de tolerancia del ADR 004 §2.6. No es la métrica de 0.1 mm del brief: esa
# mide reversibilidad de UNA malla; esta, alineamiento entre DOS modalidades.
#
# ⚠ **ε es por par de modalidades, no una constante del paquete.** Lo dice la
# aritmética del gate: `clamp(1 − rms/ε) ≥ 0.7` equivale a `rms ≤ 0.3·ε`, o sea
# 0,15 mm con este valor. Un CBCT de vóxel 0,30 mm y PSF 425 µm no puede alcanzar
# eso ni en teoría, así que con ε = 0,5 la fusión intraoral↔CBCT **no podría pasar
# el gate nunca**. Este 0,5 vale para el caso fácil —malla derivada del propio
# volumen, alineada por construcción—; para modalidades capturadas por separado hay
# que usar el ε de su par.
DEFAULT_EPSILON_MM = 0.5

# Medido sobre el paciente de `histora` (CS 9600 + escáner intraoral) con
# `scripts/registro_ios_cbct.py`: el mejor registro alcanzable es 0,452 mm sobre la
# población solapada. ε se lee como «el error a partir del cual el resultado deja de
# servir», no como «la precisión que quiero»: para medir una recesión de 1-3 mm, 1,5
# mm de error ya no sirve. Con este valor, ese mejor registro da confianza 0,70.
EPSILON_IOS_CBCT_MM = 1.5

# Distancia por debajo de la cual se considera que un punto tiene contrapartida en la
# otra nube. Es lo que separa «el registro está mal» de «el registro está bien y las
# dos superficies no cubren lo mismo» — el escáner trae paladar y encía, y el esmalte
# del CBCT no.
DEFAULT_OVERLAP_MM = 1.0


@dataclass(frozen=True)
class RegistrationResult:
    """Lo que devuelve un registrador: la transformación y cómo de bien encajó."""

    rotation: tuple[float, float, float, float]  # cuaternión (w, x, y, z)
    translation: tuple[float, float, float]  # mm
    rms_mm: float
    """Residuo sobre la nube COMPLETA. Con solapamiento parcial **no mide el
    registro**: mide cuánto de `source` no tiene contrapartida posible."""
    iterations: int
    converged: bool

    rms_overlap_mm: float | None = None
    """Residuo sobre los puntos que sí tienen contrapartida (< `overlap_mm`). Es el
    número del que debe salir la confianza. `None` si el registrador no lo mide."""
    overlap_fraction: float | None = None
    """Fracción de `source` con contrapartida. Un solapamiento ridículo invalida el
    registro **por bajo que salga el rms**: con pocos puntos siempre se encuentra
    alguna pose que los acerca."""
    veto: str | None = None
    """Por qué esta pose **no** se puede dar por buena aunque el ajuste haya convergido.
    Lo pone un registrador que sabe algo que el residuo no ve —p. ej. que ninguna pose
    candidata es anatómicamente coherente—. Con veto, el agente da confianza 0 y lo manda
    al gate. `None` en los registradores que no vetan, que son todos los anteriores."""

    @property
    def rms_efectivo_mm(self) -> float:
        """El residuo que significa algo. Cae al de la nube completa si no hay otro.

        Medido en `histora`: 4,98 mm sobre la nube completa frente a **0,452 mm**
        sobre la población solapada, para el mismo registro.
        """
        return self.rms_mm if self.rms_overlap_mm is None else self.rms_overlap_mm

    def to_rigid_transform(self) -> RigidTransform:
        """Puente al contrato: lo que se guarda en `Provenance.transform`.

        Viaja el residuo **efectivo**: es el que describe la calidad del registro, y
        es el que alguien leerá en la `Provenance` dentro de un año.
        """
        return RigidTransform(
            rotation=self.rotation, translation=self.translation, rms_mm=self.rms_efectivo_mm
        )


class Registrar(Protocol):
    """Superficie mínima de un algoritmo de registro.

    `source` y `target` son `(N, 3)` en milímetros. Devuelve la transformación que
    lleva `source` sobre `target`.
    """

    def __call__(self, source: np.ndarray, target: np.ndarray) -> RegistrationResult: ...


# --------------------------------------------------------------------------- #
# Piezas
# --------------------------------------------------------------------------- #
def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transformación rígida óptima entre dos nubes **ya emparejadas** punto a punto.

    Solución cerrada por SVD (Kabsch/Umeyama). La corrección con el determinante
    evita que la SVD devuelva una **reflexión** en vez de una rotación: sería una
    solución de menor error que no corresponde a ningún movimiento físico.
    """
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    h = (source - cs).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, ct - rot @ cs


def matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Matriz de rotación 3×3 → cuaternión (w, x, y, z).

    Método de Shepperd: elige la rama según la traza para no dividir por algo
    cercano a cero, que es donde la conversión ingenua pierde precisión.
    """
    tr = float(np.trace(rot))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (
            0.25 * s,
            (rot[2, 1] - rot[1, 2]) / s,
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[1, 0] - rot[0, 1]) / s,
        )
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        q = (
            (rot[2, 1] - rot[1, 2]) / s,
            0.25 * s,
            (rot[0, 1] + rot[1, 0]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
        )
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        q = (
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[0, 1] + rot[1, 0]) / s,
            0.25 * s,
            (rot[1, 2] + rot[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        q = (
            (rot[1, 0] - rot[0, 1]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
            (rot[1, 2] + rot[2, 1]) / s,
            0.25 * s,
        )
    norma = math.sqrt(sum(c * c for c in q))
    return (q[0] / norma, q[1] / norma, q[2] / norma, q[3] / norma)


def quaternion_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Cuaternión (w, x, y, z) → matriz de rotación 3×3. Inversa de `matrix_to_quaternion`."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def apply(rot: np.ndarray, trans: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ rot.T + trans


# --------------------------------------------------------------------------- #
# Punto a plano
# --------------------------------------------------------------------------- #
def rotacion_de_vector(w: np.ndarray) -> np.ndarray:
    """Rotación exacta (Rodrigues) de un vector de rotación `w` en radianes."""
    w = np.asarray(w, dtype=np.float64)
    angulo = float(np.linalg.norm(w))
    if angulo < 1e-15:
        return np.eye(3)
    k = w / angulo
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angulo) * kx + (1.0 - math.cos(angulo)) * (kx @ kx)


def _sistema_punto_plano(
    src: np.ndarray, dst: np.ndarray, normales: np.ndarray, pesos: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(J, b, w, centro)` del problema linealizado, en coordenadas centradas en `src`.

    Centrar en el centroide del origen hace que el condicionamiento no dependa de dónde
    está el marco: una arcada a 100 mm del origen del fichero tendría si no un brazo de
    palanca enorme que mezcla rotación y traslación.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = np.asarray(normales, dtype=np.float64)
    w = np.ones(len(src)) if pesos is None else np.asarray(pesos, dtype=np.float64)
    centro = (w[:, None] * src).sum(axis=0) / max(float(w.sum()), 1e-300)
    p = src - centro
    jac = np.c_[np.cross(p, n), n]
    b = -np.einsum("ij,ij->i", src - dst, n)
    return jac, b, w, centro


def paso_punto_plano(
    src: np.ndarray,
    dst: np.ndarray,
    normales_dst: np.ndarray,
    pesos: np.ndarray | None = None,
    *,
    tope: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Un paso de mínimos cuadrados punto a plano: `(R, t)` que acerca `src` a los planos.

    Minimiza Σ wᵢ ((R·srcᵢ + t − dstᵢ)·nᵢ)² linealizando la rotación. Es el paso de
    Chen–Medioni: con el plano tangente del destino, un punto que desliza sobre la
    superficie no cuenta como error, y eso es lo que deja converger dos muestreos
    distintos de la misma superficie — el contorno de un corte y los vértices de una malla.
    La rotación se reconstruye exacta con Rodrigues, no se deja linealizada.

    `tope = (radianes, mm)` limita el tamaño de UN paso, escalando el incremento entero
    si se pasa. Con una dirección mal condicionada —la altura, cuando casi todo el contorno
    es pared vertical— un paso sin tope puede saltar lejos; el tope no cambia el punto fijo
    al que se converge, solo cómo se llega.
    """
    jac, b, w, centro = _sistema_punto_plano(src, dst, normales_dst, pesos)
    jtw = jac.T * w
    x = np.linalg.lstsq(jtw @ jac, jtw @ b, rcond=None)[0]
    if tope is not None:
        exceso = max(
            float(np.linalg.norm(x[:3])) / tope[0], float(np.linalg.norm(x[3:])) / tope[1]
        )
        if exceso > 1.0:
            x = x / exceso
    rot = rotacion_de_vector(x[:3])
    return rot, centro - rot @ centro + x[3:]


def informacion_6dof(
    src: np.ndarray, dst: np.ndarray, normales_dst: np.ndarray, pesos: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """`(autovalores, autovectores, sigma)` de JᵀWJ en la solución.

    Dice qué dirección de la pose está peor restringida por los datos. Los tres
    primeros componentes son rotación (rad), los tres últimos traslación (mm); un
    autovector se lee por sus pesos en cada bloque. `sigma` es la desviación robusta del
    residuo (1,4826·MAD). σ/√λ es un error formal **optimista**: supone residuos
    independientes y no lo son —puntos vecinos de un mismo contorno comparten error—.
    """
    jac, b, w, _ = _sistema_punto_plano(src, dst, normales_dst, pesos)
    valores, vectores = np.linalg.eigh((jac.T * w) @ jac)
    sigma = 1.4826 * float(np.median(np.abs(b - np.median(b)))) if len(b) else float("nan")
    return valores, vectores, sigma


def paso_similitud_punto_plano(
    src: np.ndarray, dst: np.ndarray, normales_dst: np.ndarray, pesos: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Como `paso_punto_plano` con un 7.º grado de libertad: escala alrededor del centroide.

    Solo para diagnosticar. Devuelve `(R, t, s)` con `s·R·src + t` acercándose al
    destino. Si la escala que pide el ajuste se aleja de 1, ningún modelo rígido puede
    encajar todas las piezas a la vez, y eso se declara: nunca se guarda en la
    transformación.
    """
    jac, b, w, centro = _sistema_punto_plano(src, dst, normales_dst, pesos)
    p = np.asarray(src, dtype=np.float64) - centro
    n = np.asarray(normales_dst, dtype=np.float64)
    jac7 = np.c_[jac, np.einsum("ij,ij->i", p, n)]
    jtw = jac7.T * w
    x = np.linalg.lstsq(jtw @ jac7, jtw @ b, rcond=None)[0]
    rot = rotacion_de_vector(x[:3])
    s = 1.0 + float(x[6])
    return rot, centro - s * (rot @ centro) + x[3:6], s


# --------------------------------------------------------------------------- #
# ICP multiescala
# --------------------------------------------------------------------------- #
def icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    scales: tuple[float, ...] = (0.1, 0.35, 1.0),
    max_iter: int = 60,
    tol_mm: float = 1e-6,
    seed: int = 0,
    trim: float = 1.0,
    overlap_mm: float = DEFAULT_OVERLAP_MM,
) -> RegistrationResult:
    """Iterative Closest Point multiescala, con rechazo opcional de atípicos.

    En cada escala se usa una fracción creciente de `source`: las primeras pasadas
    son baratas y sacan al registro del grueso del desalineamiento, las últimas
    afinan con todos los puntos. Sale más rápido y cae en menos mínimos locales que
    iterar siempre con la nube completa.

    El submuestreo va con semilla fija: dos ejecuciones sobre el mismo dato tienen
    que dar la misma transformación, o la confianza que se deriva del residuo no
    significaría nada.

    **`trim` es la fracción de correspondencias que se usa para ajustar** en cada
    iteración, quedándose con las más cercanas. Con `1.0` (por defecto) el
    comportamiento es el clásico punto-a-punto. Bajarlo es imprescindible cuando las
    dos superficies **no cubren lo mismo**: registrando un escaneo intraoral contra
    el esmalte de un CBCT, la mayoría del escaneo es paladar y encía, y sin recorte
    esos puntos —que no tienen contrapartida— arrastran el ajuste.

    ⚠ **Recortar tiene un coste que no es obvio: destruye la capacidad de
    discriminar entre poses.** A una pose equivocada le basta con que la fracción
    conservada caiga cerca. Medido: con `trim=0.35` la búsqueda global no recupera ni
    una transformación conocida sobre la MISMA nube; con `0.70` y `1.0` sí. Por eso
    `icp_global` criba flojo y refina fuerte, y por eso un único valor de recorte
    para todo el registro es un error.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"`source` debe ser (N, 3), recibido {source.shape}")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"`target` debe ser (N, 3), recibido {target.shape}")
    if len(source) < 3 or len(target) < 3:
        raise ValueError("hacen falta al menos 3 puntos en cada nube para fijar una pose")

    rng = np.random.default_rng(seed)
    arbol = cKDTree(target)
    rot, trans = np.eye(3), np.zeros(3)
    iteraciones, convergio = 0, False

    for escala in scales:
        n = max(3, int(len(source) * escala))
        sub = source if n >= len(source) else source[rng.choice(len(source), n, replace=False)]
        previo = math.inf

        for _ in range(max_iter):
            iteraciones += 1
            movido = apply(rot, trans, sub)
            dist, idx = arbol.query(movido)
            usados = _mas_cercanos(dist, trim)
            d_rot, d_trans = kabsch(movido[usados], target[idx[usados]])
            rot, trans = d_rot @ rot, d_rot @ trans + d_trans

            rms = float(np.sqrt(np.mean(dist[usados] ** 2)))
            if abs(previo - rms) < tol_mm:
                convergio = True
                break
            previo = rms

    # El residuo se reporta sobre la nube COMPLETA, no sobre el submuestreo: es el
    # número del que sale la confianza, y tiene que ser honesto.
    dist_final, _ = arbol.query(apply(rot, trans, source))
    solapados = dist_final < overlap_mm
    return RegistrationResult(
        rotation=matrix_to_quaternion(rot),
        translation=(float(trans[0]), float(trans[1]), float(trans[2])),
        rms_mm=float(np.sqrt(np.mean(dist_final**2))),
        iterations=iteraciones,
        converged=convergio,
        rms_overlap_mm=(
            float(np.sqrt(np.mean(dist_final[solapados] ** 2))) if solapados.any() else None
        ),
        overlap_fraction=float(solapados.mean()),
    )


def _mas_cercanos(dist: np.ndarray, trim: float) -> np.ndarray:
    """Índices de la fracción `trim` de correspondencias más cercanas."""
    if trim >= 1.0:
        return np.arange(len(dist))
    k = max(3, int(len(dist) * trim))
    return np.argpartition(dist, k - 1)[:k]


def icp_global(
    source: np.ndarray,
    target: np.ndarray,
    *,
    poses: int = 500,
    criba: int = 2_000,
    fino: int = 20_000,
    mejores: int = 10,
    trim: float = 0.35,
    trim_criba: float = 0.70,
    overlap_mm: float = DEFAULT_OVERLAP_MM,
    seed: int = 0,
) -> RegistrationResult:
    """Registration **con etapa gruesa**: barre SO(3) al azar y refina las mejores poses.

    Es el hueco que el ADR 004 dejaba para RANSAC-FPFH, resuelto por fuerza bruta.
    Basta, y no arrastra Open3D: la traslación la resuelve el propio ICP alineando
    centroides, así que lo único que hay que acertar es la rotación, y unos cientos
    de muestras de SO(3) la encuentran. Medido: segundos.

    **Criba y refinado usan recortes distintos a propósito** (ver el aviso de `icp`).
    Se criba con `trim_criba` —flojo, para que las poses malas se noten— y se refina
    con `trim` —fuerte, para sobrevivir a la parte de `source` que no tiene
    contrapartida—.

    Cumple el `Protocol` `Registrar`, así que se enchufa en el
    `GeometricFusionAgent` sin tocar el agente.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) < 3 or len(target) < 3:
        raise ValueError("hacen falta al menos 3 puntos en cada nube para fijar una pose")

    rng = np.random.default_rng(seed)
    c_f, c_o = source.mean(axis=0), target.mean(axis=0)
    rotaciones = _rotaciones_aleatorias(poses, rng)

    def muestra(p: np.ndarray, n: int) -> np.ndarray:
        return p if len(p) <= n else p[rng.choice(len(p), n, replace=False)]

    cri_f, cri_o = muestra(source, criba), muestra(target, criba)
    puntuadas = []
    for i, r in enumerate(rotaciones):
        inicial = (cri_f - c_f) @ r.T + c_o
        res = icp(inicial, cri_o, max_iter=25, trim=trim_criba, overlap_mm=overlap_mm)
        puntuadas.append((res.rms_efectivo_mm, i))
    puntuadas.sort()

    fin_f = muestra(source, fino)
    finalistas = []
    for _, i in puntuadas[:mejores]:
        inicial = (fin_f - c_f) @ rotaciones[i].T + c_o
        res = icp(inicial, target, trim=trim, overlap_mm=overlap_mm)
        finalistas.append((res.rms_efectivo_mm, i, res))
    finalistas.sort(key=lambda t: t[0])
    _, mejor, _ = finalistas[0]

    # La pose ganadora se reejecuta sobre `source` COMPLETO para que la
    # transformación devuelta lleve dentro la alineación inicial y el resultado sea
    # aplicable tal cual, no relativo a un submuestreo.
    inicial = (source - c_f) @ rotaciones[mejor].T + c_o
    afinado = icp(inicial, target, trim=trim, overlap_mm=overlap_mm)
    rot = quaternion_to_matrix(afinado.rotation) @ rotaciones[mejor]
    trans = quaternion_to_matrix(afinado.rotation) @ (c_o - rotaciones[mejor] @ c_f) + np.asarray(
        afinado.translation
    )
    return RegistrationResult(
        rotation=matrix_to_quaternion(rot),
        translation=(float(trans[0]), float(trans[1]), float(trans[2])),
        rms_mm=afinado.rms_mm,
        iterations=afinado.iterations,
        converged=afinado.converged,
        rms_overlap_mm=afinado.rms_overlap_mm,
        overlap_fraction=afinado.overlap_fraction,
    )


def _rotaciones_aleatorias(n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` rotaciones uniformes sobre SO(3), vía cuaterniones normalizados.

    Se genera aquí y no con `scipy.spatial.transform` para no añadir superficie de
    dependencia a un paquete que hoy solo usa `cKDTree` de scipy.
    """
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.stack([quaternion_to_matrix(tuple(fila)) for fila in q])
