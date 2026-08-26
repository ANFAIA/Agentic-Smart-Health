"""Pipeline de agregación punto → diente.

Todo opera sobre `ndarray`; no hay estado ni GPU. Convenio: las etiquetas son
**índices de clase**, con `gum_class` (0 por defecto) reservado para la encía.
El mapeo índice → código FDI es del llamante (parámetro `codes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ToothInstance:
    """Un diente detectado: qué vértices lo forman y con qué código."""

    label: int
    """Índice de clase asignado (nunca `gum_class`)."""

    vertices: np.ndarray = field(repr=False)
    """Índices de los vértices de la malla que forman la instancia."""

    confidence: float
    """Log-probabilidad media de `label` sobre los vértices de la instancia."""

    fdi: int | None = None
    """Código FDI, si se pasó `codes` al agregar. `None` si se trabaja en índices."""

    @property
    def size(self) -> int:
        return int(self.vertices.size)


# --------------------------------------------------------------------------- #
# Grafo kNN
# --------------------------------------------------------------------------- #
def _knn_edges(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Aristas del grafo kNN y sus longitudes.

    `cKDTree` devuelve el propio punto como primer vecino (distancia 0), así que
    se pide `k+1` y se descarta esa columna.
    """
    n = len(points)
    if n < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty(0)
    kk = min(k + 1, n)
    dist, idx = cKDTree(points).query(points, k=kk)
    dist, idx = np.atleast_2d(dist)[:, 1:], np.atleast_2d(idx)[:, 1:]
    src = np.repeat(np.arange(n), idx.shape[1])
    return np.vstack([src, idx.ravel()]), dist.ravel()


def typical_spacing(points: np.ndarray, k: int = 8) -> float:
    """Espaciado típico de la nube (mediana de la longitud de arista kNN).

    Da la **escala** de la malla, para que los umbrales de distancia puedan
    expresarse en múltiplos de ella en vez de en unidades absolutas: así no
    dependen del tamaño de la arcada ni de cómo esté escalado el escaneo.
    """
    _, dist = _knn_edges(points, k)
    return float(np.median(dist)) if dist.size else 0.0


# --------------------------------------------------------------------------- #
# 0 · decodificación: de log-probabilidades a etiquetas
# --------------------------------------------------------------------------- #
def suaviza_contiguidad(
    points: np.ndarray,
    logprob: np.ndarray,
    *,
    beta: float = 2.0,
    k: int = 8,
    pasadas: int = 12,
) -> np.ndarray:
    """Etiquetas del `logprob` decodificadas **con la restricción de que un diente es
    contiguo**, en vez de un `argmax` por punto.

    Un `argmax` decide cada vértice por su cuenta: nada en él dice que las etiquetas
    de una malla tengan que formar parches. Medido sobre un maxilar real de 112.067
    vértices, esa decisión suelta deja **119 componentes conexas por diente** — cientos
    de motas del código de una pieza salpicadas sobre las demás.

    Esto es ICM (*iterated conditional modes*): en cada pasada, cada vértice se queda
    con la etiqueta que maximiza `logprob + beta · (fracción de vecinos que ya la
    llevan)`. El primer término es lo que el modelo cree; el segundo, lo único que el
    modelo no puede saber mirando un punto aislado. Converge en pocas pasadas y **no
    reentrena nada**: es el posterior del propio modelo, decodificado mejor.

    ⚠️ **Quita las motas y NO mueve la frontera, y esa distinción es el resultado.**
    Sobre ese mismo maxilar: componentes por pieza 119 → 30 e islas 6,8 % → 1,5 %, pero
    las 9 coronas de 15 que se pasan de su ancho anatómico siguen siendo 9, y el exceso
    medio se mueve **una centésima de milímetro** (+2,81 → +2,80 mm). Una frontera mal
    puesta no es ruido de decodificación: es el modelo equivocándose con confianza justo
    en el punto de contacto, y ahí el suavizado no tiene nada que corregir. Sirve para
    separar las dos causas, no para arreglar las dos.

    Args:
        points: `(N, 3)` posiciones de los vértices.
        logprob: `(N, C)` log-probabilidad por punto y clase.
        beta: peso del término de contigüidad. `0` reproduce el `argmax` exacto.
        k: vecinos del grafo kNN, el mismo que usa el resto del paquete.
        pasadas: tope de iteraciones; para antes si nada cambia.

    Returns:
        `(N,)` índices de clase.
    """
    labels = np.asarray(logprob).argmax(axis=1)
    if beta <= 0 or len(points) < 2:
        return labels
    edges, _ = _knn_edges(np.asarray(points, dtype=np.float64), k)
    if edges.shape[1] == 0:
        return labels
    n, clases = logprob.shape
    # Grafo simétrico y normalizado por grado, para que `beta` signifique lo mismo en un
    # vértice del borde —con menos vecinos— que en uno del interior.
    src = np.concatenate([edges[0], edges[1]])
    dst = np.concatenate([edges[1], edges[0]])
    grado = np.maximum(np.bincount(src, minlength=n), 1)
    for _ in range(pasadas):
        apoyo = np.zeros((n, clases))
        np.add.at(apoyo, (src, labels[dst]), 1.0)
        nuevas = (logprob + beta * (apoyo / grado[:, None])).argmax(axis=1)
        if np.array_equal(nuevas, labels):
            break
        labels = nuevas
    return labels


# --------------------------------------------------------------------------- #
# 1 · semántica -> instancias
# --------------------------------------------------------------------------- #
def connected_labels(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 8,
    min_size: int = 30,
    gum_class: int = 0,
) -> list[np.ndarray]:
    """Componentes conexas de puntos no-encía que comparten etiqueta predicha.

    Se construye el grafo kNN sobre los puntos de diente y se **cortan** las
    aristas cuyos extremos tienen etiquetas distintas; cada componente resultante
    con al menos `min_size` puntos es un diente candidato. Las componentes
    menores se descartan por ruido.

    Devuelve índices **sobre el array original**, no sobre el subconjunto.
    """
    fg = np.flatnonzero(labels != gum_class)
    if fg.size < min_size:
        return []
    edges, _ = _knn_edges(points[fg], k)
    if edges.shape[1] == 0:
        return []
    sub = labels[fg]
    keep = sub[edges[0]] == sub[edges[1]]
    edges = edges[:, keep]
    adj = coo_matrix(
        (np.ones(edges.shape[1], dtype=np.int8), (edges[0], edges[1])),
        shape=(fg.size, fg.size),
    )
    ncomp, comp = connected_components(adj, directed=False)
    out = [fg[np.flatnonzero(comp == c)] for c in range(ncomp)]
    return [c for c in out if c.size >= min_size]


# --------------------------------------------------------------------------- #
# 2 · fusión de fragmentos
# --------------------------------------------------------------------------- #
def merge_fragments(
    points: np.ndarray,
    components: list[np.ndarray],
    labels: np.ndarray,
    *,
    spacing: float,
    mult: float = 12.0,
) -> list[np.ndarray]:
    """Une componentes de la **misma clase** cuyos puntos más cercanos estén a
    menos de `mult` × `spacing`.

    Por qué existe: un diente real sale a menudo partido en varias componentes
    (una franja de puntos mal etiquetados lo corta). Mientras eso pase, «una
    instancia = un diente» es falso, y cualquier razonamiento que dependa de esa
    premisa —señaladamente `assign_unique`— produce errores inventados.

    El umbral es **relativo** al espaciado de la malla, y la unión es transitiva
    (union-find): A-B y B-C pegados implican A-B-C en la misma instancia.

    Con `mult <= 0` no fusiona nada, que es el control para medir su efecto.
    """
    if len(components) < 2 or mult <= 0 or spacing <= 0:
        return components

    eps = mult * spacing
    lab = [majority_label(labels[c]) for c in components]
    parent = list(range(len(components)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    trees = [cKDTree(points[c]) for c in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if lab[i] != lab[j] or find(i) == find(j):
                continue
            # Distancia mínima exacta entre los dos conjuntos de puntos.
            if trees[i].query(points[components[j]], k=1)[0].min() < eps:
                parent[find(i)] = find(j)

    groups: dict[int, list[np.ndarray]] = {}
    for i, c in enumerate(components):
        groups.setdefault(find(i), []).append(c)
    return [np.concatenate(v) for v in groups.values()]


def majority_label(labels: np.ndarray) -> int:
    """Etiqueta más frecuente. Empate → la de índice menor (determinista)."""
    return int(np.bincount(labels).argmax())


# --------------------------------------------------------------------------- #
# 3 · unicidad FDI (opcional)
# --------------------------------------------------------------------------- #
def assign_unique(
    instances: list[ToothInstance],
    logprob: np.ndarray,
    *,
    gum_class: int = 0,
) -> list[ToothInstance]:
    """Reasigna etiquetas para que **ninguna se repita**, maximizando la
    confianza total (asignación húngara).

    Qué optimiza exactamente: la **suma** sobre instancias de la log-probabilidad
    media de la etiqueta que se le asigna. Es la etiquetación conjunta más
    probable de la arcada bajo la restricción de unicidad — no la mejor etiqueta
    de cada instancia por separado, que es lo que ya da el voto mayoritario.

    La matriz es **rectangular** (hay más clases que instancias): solo exige que
    no se repitan, no que se usen todas, así que sigue valiendo con dentición
    incompleta.

    ⚠️ Medido: aplicarlo **sin fusionar fragmentos antes** empeora el resultado.
    Ver el docstring del paquete.
    """
    if not instances:
        return []

    cost = np.stack([logprob[i.vertices].mean(axis=0) for i in instances])
    keep = [c for c in range(cost.shape[1]) if c != gum_class]
    rows, cols = linear_sum_assignment(-cost[:, keep])

    out = list(instances)
    for r, c in zip(rows, cols, strict=True):
        label = keep[c]
        out[r] = ToothInstance(
            label=label,
            vertices=instances[r].vertices,
            confidence=float(cost[r, label]),
            fdi=instances[r].fdi,
        )
    return out


# --------------------------------------------------------------------------- #
# Pipeline completo
# --------------------------------------------------------------------------- #
def aggregate_teeth(
    points: np.ndarray,
    logprob: np.ndarray,
    *,
    gum_class: int = 0,
    k: int = 8,
    min_size: int = 30,
    merge_mult: float = 12.0,
    beta: float = 2.0,
    enforce_unique: bool = False,
    codes: dict[int, int] | None = None,
) -> list[ToothInstance]:
    """De log-probabilidades por punto a **lista de dientes**.

    Args:
        points: `(N, 3)` posiciones de los vértices.
        logprob: `(N, C)` log-probabilidad por punto y clase (salida del modelo).
        gum_class: índice de la clase encía, que no genera instancias.
        k: vecinos del grafo kNN.
        min_size: puntos mínimos para que una componente cuente como diente.
        merge_mult: umbral de fusión en múltiplos del espaciado. `0` la desactiva.
        beta: contigüidad al decodificar (ver `suaviza_contiguidad`). `0` vuelve al
            `argmax` por punto, que es lo que hacía esta función antes.
        enforce_unique: impone «un código por arcada» (ver `assign_unique`).
            **Desactivado por defecto**: medido, empeora salvo que la fusión
            deje de verdad una instancia por diente.
        codes: mapeo índice de clase → código FDI, para rellenar `ToothInstance.fdi`.

    Returns:
        Instancias ordenadas de mayor a menor tamaño.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"`points` debe ser (N, 3), recibido {points.shape}")
    if logprob.ndim != 2 or len(logprob) != len(points):
        raise ValueError(f"`logprob` debe ser (N, C) con N={len(points)}, recibido {logprob.shape}")

    labels = suaviza_contiguidad(points, logprob, beta=beta, k=k)
    comps = connected_labels(points, labels, k=k, min_size=min_size, gum_class=gum_class)
    if not comps:
        return []

    comps = merge_fragments(
        points,
        comps,
        labels,
        spacing=typical_spacing(points[labels != gum_class], k),
        mult=merge_mult,
    )

    instances = []
    for c in comps:
        label = majority_label(labels[c])
        instances.append(
            ToothInstance(
                label=label,
                vertices=c,
                confidence=float(logprob[c, label].mean()),
                fdi=None if codes is None else codes.get(label),
            )
        )

    if enforce_unique:
        instances = assign_unique(instances, logprob, gum_class=gum_class)
        if codes is not None:
            instances = [
                ToothInstance(i.label, i.vertices, i.confidence, codes.get(i.label))
                for i in instances
            ]

    return sorted(instances, key=lambda i: i.size, reverse=True)
