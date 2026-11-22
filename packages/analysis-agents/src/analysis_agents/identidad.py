"""La identidad de cada pieza, decidida mirando la pieza: verificador y renumeración.

**Por qué existe.** El modelo de puntos (C2, §6.10 de
`docs/research/segmentacion-fdi-escaner.md`) separa bien los dientes, pero decide el número
de cada uno sobre todo por su posición en la arcada. En la reserva gastada eso deja 75
terceros molares sobre la encía y 13 hemiarcadas corridas un puesto. Y con las instancias
perfectas del experto numeraría igual (1.528 frente a 1.531): lo que falla no es la
frontera, es qué número cree que lleva cada pieza (§6.11).

Aquí está la parte que no necesita GPU, pre-registrada en §6.11 antes de escribirla:

- `recorte`: la pieza y lo que la rodea a menos de `MARGEN_RECORTE_MM`, a resolución de
  malla y en milímetros, que es lo que ve el modelo de pieza.
- `componente_principal`: la mayor componente conexa de una instancia, de la que sale el
  recorte del verificador v2 (§6.12). Una etiqueta de C2 puede aparecer también a 40 mm,
  y un recorte centrado en la media de las dos mira entre ambas.
- `verifica` (**E1**): una instancia con `p(diente) < τ` pasa a encía.
- `verifica_v3` (§6.13): E1, salvo en una arcada donde rechazaría más de la mitad de las
  instancias, que es una arcada en la que C2 falló entera, y salvo en instancias de menos
  de `AREA_MINIMA_MM2`.
- `renumera` (**E2**): por cuadrante, las instancias en orden desde la línea media reciben la
  secuencia de posiciones estrictamente creciente que maximiza
  `Σ [log p₂(posición) + w · log p₁(posición)]`.

El cuadrante lo sigue poniendo C2: de las 103 piezas desplazadas de la reserva gastada, 76
caen en la vecina del mismo cuadrante. Las fronteras de C2 no se tocan.

Todo es numpy; la inferencia la hace quien llama, con torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Lo que se recorta alrededor de cada pieza: el borde de la vecina y de la encía es parte de
# lo que distingue un diente de un parche de encía.
MARGEN_RECORTE_MM = 2.0
# Tope de puntos por recorte. Una corona de molar tiene ~5.000 vértices a 0,17 mm.
PUNTOS_RECORTE = 4096
POSICIONES = 8
# Los valores que se prueban en `dev-privado`, fijados en §6.11 antes de construir nada.
TAUS = (0.3, 0.5, 0.7)
PESOS_C2 = (0.0, 0.5, 1.0)
# Dos vértices de una instancia están unidos si distan menos de esto (§6.12). La malla va a
# ~0,2 mm, así que una corona entera es una componente y un trozo a 40 mm es otra.
RADIO_COMPONENTE_MM = 1.0
# Por debajo de esto una instancia no se juzga (§6.13): la mitad de la mediana del área de la
# corona más pequeña, el incisivo central inferior, 78,0 mm² en 480 arcadas de entrenamiento
# de Teeth3DS+ (`benchmark_identidad.py areas`, desde 5b1f39b). Fijado antes de evaluar.
AREA_MINIMA_MM2 = 39.0


def instancias(etiquetas: np.ndarray) -> dict[int, np.ndarray]:
    """`código → índices de vértice`. Una instancia es un código: así cuenta
    `margen_metricas.identidad_de_piezas`, y así se compara."""
    e = np.asarray(etiquetas, dtype=np.int64)
    return {int(c): np.flatnonzero(e == c) for c in np.unique(e) if c > 0}


def componente_principal(pos: np.ndarray, vertices: np.ndarray,
                         radio: float = RADIO_COMPONENTE_MM) -> np.ndarray:
    """Los vértices de la mayor componente conexa de `vertices`, unidos a menos de `radio`.

    No mira las caras: el caché de entrenamiento no las tiene, y así entrenar e inferir
    usan exactamente lo mismo. Si empatan, la de menor índice.
    """
    v = np.asarray(vertices, dtype=np.int64)
    if len(v) <= 1:
        return v
    pares = cKDTree(np.asarray(pos, dtype=np.float64)[v]).query_pairs(
        radio, output_type="ndarray")
    grafo = coo_matrix((np.ones(len(pares), dtype=np.int8), (pares[:, 0], pares[:, 1])),
                       shape=(len(v), len(v)))
    _, comp = connected_components(grafo, directed=False)
    return v[comp == int(np.argmax(np.bincount(comp)))]


def areas_de_instancia(pos: np.ndarray, caras: np.ndarray,
                       etiquetas: np.ndarray) -> dict[int, float]:
    """`código → área en mm²` de cada instancia: un tercio de cada cara a cada vértice.

    Es el tamaño con el que el verificador v3 (§6.13) decide qué no juzga. Por área y no
    por número de vértices, porque la resolución de la malla va de 0,15 a 0,2 mm entre
    escáneres y el número de vértices de una misma corona varía casi al doble.
    """
    pos = np.asarray(pos, dtype=np.float64)
    tri = np.asarray(caras, dtype=np.int64)
    a = 0.5 * np.linalg.norm(np.cross(pos[tri[:, 1]] - pos[tri[:, 0]],
                                      pos[tri[:, 2]] - pos[tri[:, 0]]), axis=1)
    por_vertice = np.zeros(len(pos))
    np.add.at(por_vertice, tri.ravel(), np.repeat(a / 3.0, 3))
    return {c: float(por_vertice[v].sum()) for c, v in instancias(etiquetas).items()}


def distancia_con_signo(pos: np.ndarray, dentro: np.ndarray, radio: float,
                        arbol: cKDTree | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`(índices, d)` de los vértices a menos de `radio` de la región `dentro`.

    `d < 0` dentro de la región (menos la distancia al vértice de fuera más cercano) y
    `d > 0` fuera (la distancia a la región). Sirve para mover la frontera de una pieza
    hacia dentro o hacia fuera sin recalcular nada: la pieza con la frontera desplazada δ
    es `d ≤ δ`.
    """
    pos = np.asarray(pos, dtype=np.float64)
    dentro = np.asarray(dentro, dtype=np.int64)
    if len(dentro) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    arbol = arbol if arbol is not None else cKDTree(pos)
    lo, hi = pos[dentro].min(0) - radio, pos[dentro].max(0) + radio
    caja = np.flatnonzero(np.all((pos >= lo) & (pos <= hi), axis=1))
    marca = np.zeros(len(pos), dtype=bool)
    marca[dentro] = True
    fuera_caja = caja[~marca[caja]]
    d_fuera, _ = cKDTree(pos[dentro]).query(pos[fuera_caja], distance_upper_bound=radio)
    cerca = np.isfinite(d_fuera)
    if len(fuera_caja) and cerca.any():
        d_dentro, _ = cKDTree(pos[fuera_caja]).query(pos[dentro])
    else:
        d_dentro = np.full(len(dentro), radio)
    indices = np.concatenate([dentro, fuera_caja[cerca]])
    d = np.concatenate([-np.minimum(d_dentro, radio), d_fuera[cerca]])
    return indices, d


def recorte(pos: np.ndarray, nrm: np.ndarray, indices: np.ndarray, d: np.ndarray,
            rng: np.random.Generator, *, desplaza: float = 0.0,
            margen: float = MARGEN_RECORTE_MM,
            max_puntos: int = PUNTOS_RECORTE) -> tuple[np.ndarray, np.ndarray] | None:
    """`(P, X)` del recorte de una pieza: posiciones centradas en mm y `[normal, ∈ pieza]`.

    `indices` y `d` son los de `distancia_con_signo`. `desplaza` mueve la frontera de la
    pieza (positivo hacia fuera), que es como se imita en entrenamiento una frontera de C2.
    Devuelve `None` si la pieza desplazada se queda vacía.
    """
    en_pieza = d <= desplaza
    en_recorte = d <= desplaza + margen
    if not en_pieza.any():
        return None
    sel = indices[en_recorte]
    flag = en_pieza[en_recorte].astype(np.float32)
    if len(sel) > max_puntos:
        k = rng.choice(len(sel), size=max_puntos, replace=False)
        sel, flag = sel[k], flag[k]
    centro = pos[indices[en_pieza]].mean(0)
    P = (np.asarray(pos)[sel] - centro).astype(np.float32)
    X = np.concatenate([np.asarray(nrm)[sel].astype(np.float32), flag[:, None]], axis=1)
    return P, X


def verifica(etiquetas: np.ndarray, p_diente: dict[int, float],
             tau: float) -> tuple[np.ndarray, list[int]]:
    """**E1.** Las instancias con `p(diente) < τ` pasan a encía. `(etiquetas, quitadas)`."""
    e = np.asarray(etiquetas, dtype=np.int64).copy()
    quitadas = sorted(c for c, p in p_diente.items() if p < tau and (e == c).any())
    for c in quitadas:
        e[e == c] = 0
    return e, quitadas


def verifica_v3(etiquetas: np.ndarray, p_diente: dict[int, float], tau: float,
                areas: dict[int, float], area_minima: float = AREA_MINIMA_MM2,
                ) -> tuple[np.ndarray, dict[str, Any]]:
    """**E1 v3** (§6.13). `(etiquetas, registro)`.

    Como `verifica`, con dos excepciones fijadas antes de evaluar:

    - Si más de la mitad de las instancias puntuadas tienen `p(diente) < τ`, no se quita
      ninguna y `registro["abstains"]` lo dice. Un escaneo de arcada es dientes casi entero;
      cuando el modelo dice lo contrario, lo que ha fallado es la segmentación, y borrar
      deja sin corona los dientes reales sobre los que montaba.
    - Una instancia de menos de `area_minima` mm² no se juzga: un fragmento así no le da
      instancia a nadie si se quita.
    """
    e = np.asarray(etiquetas, dtype=np.int64).copy()
    presentes = sorted(c for c in instancias(e) if c in p_diente)
    rechazadas = [c for c in presentes if p_diente[c] < tau]
    pequenas = [c for c in rechazadas if areas[c] < area_minima]
    registro: dict[str, Any] = {"instances": len(presentes), "rejected": len(rechazadas),
                                "abstains": 2 * len(rechazadas) > len(presentes),
                                "too_small": pequenas, "removed": []}
    if registro["abstains"]:
        return e, registro
    for c in rechazadas:
        if c not in pequenas:
            e[e == c] = 0
            registro["removed"].append(c)
    return e, registro


def asigna_posiciones(puntos: np.ndarray) -> list[int] | None:
    """Posiciones 1…8 estrictamente crecientes que maximizan la suma de `puntos[i, p-1]`.

    `puntos` es `(m, 8)`, con las instancias ya en orden desde la línea media. Huecos
    permitidos: una pieza que falta deja su posición libre. Con más de 8 instancias no hay
    asignación posible y se devuelve `None`, para que quien llama no toque ese cuadrante.
    """
    S = np.asarray(puntos, dtype=np.float64)
    m = len(S)
    if m == 0:
        return []
    if m > POSICIONES or S.shape[1] != POSICIONES:
        return None
    # mejor[i, p]: la mejor suma de las i+1 primeras con la i-ésima en la posición p.
    mejor = np.full((m, POSICIONES), -np.inf)
    previa = np.zeros((m, POSICIONES), dtype=np.int64)
    mejor[0] = S[0]
    for i in range(1, m):
        acumulado = np.maximum.accumulate(mejor[i - 1])
        donde = np.zeros(POSICIONES, dtype=np.int64)
        for p in range(1, POSICIONES):
            donde[p] = donde[p - 1] if mejor[i - 1, p] <= acumulado[p - 1] else p
        mejor[i, 1:] = acumulado[:-1] + S[i, 1:]
        previa[i, 1:] = donde[:-1]
    if not np.isfinite(mejor[-1]).any():
        return None
    p = int(np.argmax(mejor[-1]))
    salida = [p]
    for i in range(m - 1, 0, -1):
        p = int(previa[i, p])
        salida.append(p)
    return [q + 1 for q in reversed(salida)]


def orden_desde_linea_media(centroides: np.ndarray, centro: np.ndarray, R: np.ndarray,
                            contrarios: np.ndarray | None = None) -> np.ndarray:
    """Índices de `centroides` de la pieza más cercana a la línea media a la más lejana.

    `centro` y `R` son el marco de la arcada (`fusion_agents.marco.marco_canonico`, filas
    `[lateral, anterior, oclusal]`), medido sobre la propia malla. El ángulo respecto al eje
    anterior en el plano oclusal, en valor absoluto, es monótono a lo largo de cada
    hemiarcada en U; eso da el ORDEN. El SENTIDO no se fía del signo de ese eje.

    ⚠️ **Por qué.** `marco_canonico` fija los signos para que dos arcadas se puedan poner una
    sobre otra, no para que «anterior» sea anatómico. En `dev-privado` el orden salía
    invertido en el maxilar —39 cuadrantes al revés y 18 bien en 30 arcadas— y la
    renumeración, obligada a numerar en orden creciente, desnumeraba la hemiarcada entera.
    Así que el sentido se mide: el extremo de la línea media es el más cercano a las piezas
    del cuadrante contrario (`contrarios`). Solo sin ellas se usa el eje anterior.
    """
    v = np.asarray(centroides, dtype=np.float64) - np.asarray(centro, dtype=np.float64)
    lateral, anterior = v @ np.asarray(R)[0], v @ np.asarray(R)[1]
    orden = np.argsort(np.abs(np.arctan2(lateral, anterior)), kind="stable")
    if contrarios is not None and len(contrarios) and len(orden) > 1:
        otros = np.asarray(contrarios, dtype=np.float64)
        cerca = np.linalg.norm(np.asarray(centroides)[:, None, :] - otros[None, :, :],
                               axis=2).min(axis=1)
        if cerca[orden[-1]] < cerca[orden[0]]:
            orden = orden[::-1]
    return orden


def p1_por_instancia(logps: list[np.ndarray], codigos_modelo: list[int],
                     etiquetas: np.ndarray) -> dict[int, np.ndarray]:
    """`código → (8,)` log-probabilidad de C2 para cada posición del cuadrante de la pieza.

    Por referencia: la log-probabilidad media de cada código del cuadrante sobre los
    vértices de la instancia, normalizada entre las 8 posiciones. Después, la media entre
    las referencias que votaron. Un código que el modelo no tiene recibe −∞.
    """
    indice = {int(c): k for k, c in enumerate(codigos_modelo)}
    salida: dict[int, np.ndarray] = {}
    for c, vs in instancias(etiquetas).items():
        q = c // 10
        cols = [indice.get(q * 10 + p) for p in range(1, POSICIONES + 1)]
        por_ref = []
        for logp in logps:
            m = np.array([np.asarray(logp)[vs, k].astype(np.float64).mean()
                          if k is not None else -np.inf for k in cols])
            finitos = np.isfinite(m)
            if finitos.any():
                m = m - np.logaddexp.reduce(m[finitos])
            por_ref.append(m)
        salida[c] = np.mean(np.stack(por_ref), axis=0) if por_ref else np.zeros(POSICIONES)
    return salida


def renumera(etiquetas: np.ndarray, pos: np.ndarray, centro: np.ndarray, R: np.ndarray,
             p2: dict[int, np.ndarray], p1: dict[int, np.ndarray],
             w: float) -> tuple[np.ndarray, dict[str, Any]]:
    """**E2.** Renumera cada cuadrante con `asigna_posiciones`. `(etiquetas, registro)`.

    `p2` es la log-probabilidad de posición del modelo de pieza y `p1` la de C2
    (`p1_por_instancia`). Con `w = 0` solo cuenta el modelo de pieza. Un cuadrante con más
    de 8 instancias se deja como estaba y se declara.
    """
    e = np.asarray(etiquetas, dtype=np.int64)
    salida = e.copy()
    inst = instancias(e)
    cambios: dict[str, int] = {}
    sin_tocar: list[int] = []
    for q in sorted({c // 10 for c in inst}):
        codigos = sorted(c for c in inst if c // 10 == q)
        cent = np.stack([np.asarray(pos)[inst[c]].mean(0) for c in codigos])
        # El cuadrante contrario de la misma arcada: 1↔2, 3↔4.
        contrario = {1: 2, 2: 1, 3: 4, 4: 3}.get(q)
        otros = [np.asarray(pos)[inst[c]].mean(0) for c in inst if c // 10 == contrario]
        orden = [codigos[i] for i in orden_desde_linea_media(
            cent, centro, R, np.stack(otros) if otros else None)]
        S = np.stack([np.asarray(p2[c], dtype=np.float64)
                      + (w * np.asarray(p1[c], dtype=np.float64) if w else 0.0)
                      for c in orden])
        posiciones = asigna_posiciones(S)
        if posiciones is None:
            sin_tocar.append(q)
            continue
        for c, p in zip(orden, posiciones, strict=True):
            nuevo = q * 10 + p
            salida[inst[c]] = nuevo
            if nuevo != c:
                cambios[str(c)] = nuevo
    return salida, {"renumbered": cambios, "quadrants_untouched": sin_tocar}
