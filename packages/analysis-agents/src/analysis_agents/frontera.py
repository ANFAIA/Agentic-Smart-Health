"""Afinar la frontera de cada pieza sobre el escáner, y decir de cuáles te puedes fiar.

**Para qué.** El visor deja seleccionar una pieza y enseñar lo que el contenedor sabe de
ella. Eso sólo vale si al seleccionarla sale **esa** pieza: si arrastra medio diente vecino,
lo que se enseña al lado es correcto y lo que se ve no, que es la peor combinación.

⚠️ **Son dos errores distintos y hace falta atacarlos por separado.** Medido sobre un caso
real, la etiqueta de una corona se desborda por el cuello hacia la encía (error *apical*) y
además cruza el punto de contacto hacia la vecina (error *mesiodistal*). Un corte en altura
arregla el primero y no toca el segundo; una competencia entre piezas arregla el segundo y
no toca el primero.

⚠️ **Y la concavidad NO puede construir la frontera, sólo juzgarla.** Se probó: quitar todo
vértice que llegue a la encía sin cruzar un pliegue da una razón de concavidad de **1,88**
—exactamente la del experto— con el **87 %** de las etiquetas mal. La malla está llena de
pliegues (fisuras oclusales, troneras) y el criterio se satisface por cualquiera de ellos.
Cualquier método que optimice esa razón va a mentir; aquí se usa sólo para elegir la altura
dentro de una ventana que fija la anatomía.
"""

from __future__ import annotations

import numpy as np

from analysis_agents.dental import altura_admitida

# Cuánto puede apartarse el ancho mesiodistal medido del de tabla antes de que la pieza deje
# de considerarse bien recortada, en mm.
#
# ⚠️ **Calibrado sobre etiquetas de experto, y eso cambió el diagnóstico.** El gate contaba
# «coronas más anchas de lo admitido» y daba 11 de 14 — pero las etiquetas de experto de
# Teeth3DS+ dan **86 de 111 (77 %)** con ese mismo recuento. Contar no mide un defecto: lo
# que separa es la MAGNITUD. Sobre 188 coronas de experto, `|medido - tabla|` da p50 0,48,
# p90 1,46 y **p95 1,92 mm**; nuestro caso daba p50 2,38 y p90 7,35.
EXCESO_ADMITIDO_MM = 1.9

# Anillos que se retiran del borde de cada etiqueta para quedarse con su núcleo.
ANILLOS_NUCLEO = 4

# Media ventana, en mm, dentro de la que se busca el cuello alrededor de la altura de tabla.
VENTANA_CUELLO_MM = 2.5


def _aristas(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Las aristas en los DOS sentidos: cada vertice tiene que ver a todos sus vecinos."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    return e[:, 0], e[:, 1]


def concavidad(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Indicador adimensional por vertice: positivo = concavo.

    Para cada vecino `u` de `v` se mide cuanto se sale `u` del plano tangente en `v`, en
    unidades de la propia arista. Con normales hacia fuera, un pliegue —que es lo que hay
    donde el diente sale de la encia— da positivo y una cuspide da negativo. Es una media
    de cosenos: no hay que ajustar una cuadrica ni elegir un radio.
    """
    a, b, c = (pos[tri[:, i]] for i in range(3))
    n_cara = np.cross(b - a, c - a)          # su modulo es 2x el area: pondera solo
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, tri[:, k], n_cara)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    i, j = _aristas(tri)
    d = pos[j] - pos[i]
    largo = np.linalg.norm(d, axis=1)
    coseno = np.einsum("ij,ij->i", d, n[i]) / np.maximum(largo, 1e-12)
    suma, cuenta = np.zeros(len(pos)), np.zeros(len(pos))
    np.add.at(suma, i, coseno)
    np.add.at(cuenta, i, 1.0)
    return suma / np.maximum(cuenta, 1.0)


# Por debajo de esto la razon es ruido: no hay bastante frontera diente/encia que medir.
MINIMO_VERTICES_MARGEN = 50


def margen_gingival(tri: np.ndarray, etq: np.ndarray) -> np.ndarray:
    """Máscara de los vértices de DIENTE que tocan encía: el margen, visto desde la corona.

    ⚠️ El contacto interproximal —diente contra diente— queda FUERA a propósito: ahí no hay
    margen gingival, y meterlo mezclaría dos fronteras distintas en una cifra.
    """
    i, j = _aristas(tri)
    marca = np.zeros(len(etq), dtype=bool)
    marca[i[(etq[i] != 0) & (etq[j] == 0)]] = True
    return marca


def razon_de_concavidad(pos: np.ndarray, tri: np.ndarray, etq: np.ndarray) -> float | None:
    """Cuánto más cóncavo es el margen gingival que la malla entera. `None` si no lo hay.

    ⚠️ **Sirve para JUZGAR una frontera, no para construirla** (ver la cabecera): un método
    que la optimice llega a la cifra del experto con las etiquetas mal. Vivía en
    `scripts/mide_segmentacion.py`; está aquí para que el banco de pruebas y el falsador
    midan con la misma función.
    """
    marca = margen_gingival(tri, etq)
    if int(marca.sum()) < MINIMO_VERTICES_MARGEN:
        return None
    c = concavidad(pos, tri)
    return float(c[marca].mean() / max(float(np.abs(c).mean()), 1e-12))


def nucleos(tri: np.ndarray, etq: np.ndarray, anillos: int = ANILLOS_NUCLEO) -> np.ndarray:
    """Lo que queda de cada etiqueta tras retirarle `anillos` de su borde. `-1` = retirado.

    Sirve para dos cosas a la vez: quitar el borde dudoso y **borrar las islas sueltas**,
    que en un maxilar real son decenas por código y bastan para inflar un ancho medido.
    """
    i, j = _aristas(tri)
    e = etq.copy()
    for _ in range(anillos):
        borde = np.zeros(len(etq), bool)
        borde[i[e[i] != e[j]]] = True
        e[borde & (e > 0)] = -1
    return e


def reparte_por_geodesica(pos: np.ndarray, tri: np.ndarray,
                          etq: np.ndarray) -> np.ndarray:
    """Cada vértice de diente se queda con la pieza cuyo NÚCLEO tiene más cerca por la malla.

    ⚠️ **Distancia geodésica pura, sin penalizar la tronera.** Se probó a encarecer el paso
    por los sitios cóncavos —la tronera interproximal es un valle— y sale PEOR: reconstruir
    la frontera diente-diente de Teeth3DS+ desde sólo los núcleos da **0,995** de acuerdo
    con el experto sin penalización y 0,986 con ella. La superficie ya separa las piezas; el
    término de curvatura sólo añade ruido.

    ⚠️ **Y tiene un límite que hay que tener escrito:** la frontera queda a medio camino
    entre los núcleos, así que un desborde UNIFORME no se corrige — el núcleo se desplaza
    con la etiqueta. Medido sobre el caso real, esta función sola no cambia el veredicto de
    ninguna pieza (4/14) y sólo baja el `|exceso|` mediano de 2,60 a 2,38 mm. Lo que sirve
    es combinarla con `recorta_al_cuello`: 4/14 -> **7/14**. Lo que sí arregla ella es el
    desborde irregular y las islas, que es lo que impide que el recorte elija bien.

    La encía no entra: aquí se decide de qué diente es cada vértice de diente, no dónde
    acaba el diente.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    piezas = [int(t) for t in np.unique(etq) if t > 0]
    if len(piezas) < 2:
        return etq.copy()
    i, j = _aristas(tri)
    peso = np.linalg.norm(pos[j] - pos[i], axis=1)
    nuc = nucleos(tri, etq)
    n = len(pos)
    fi, fj, fw = [i], [j], [peso]
    for k, t in enumerate(piezas):
        s = np.nonzero(nuc == t)[0]
        if not len(s):
            s = np.nonzero(etq == t)[0]
        fi += [np.full(len(s), n + k), s]
        fj += [s, np.full(len(s), n + k)]
        fw += [np.zeros(len(s)), np.zeros(len(s))]
    g = coo_matrix((np.concatenate(fw), (np.concatenate(fi), np.concatenate(fj))),
                   shape=(n + len(piezas),) * 2).tocsr()
    d = dijkstra(g, indices=list(range(n, n + len(piezas))), directed=False)[:, :n]
    ganador = np.asarray(piezas)[np.argmin(d, axis=0)]
    e = etq.copy()
    e[etq > 0] = ganador[etq > 0]
    return e


def eje_oclusal(pos: np.ndarray, etq: np.ndarray) -> np.ndarray:
    """Del centroide de la encía al de los dientes. Se MIDE sobre la arcada entera.

    ⚠️ **No vale el eje propio de cada pieza.** El PCA de una etiqueta desbordada lo domina
    el faldón que baja por la encía, así que sale inclinado y el corte se va con él: medido,
    el mismo recorte con eje por pieza da 0,832 de acuerdo donde con este da 0,908.
    """
    d = pos[etq > 0].mean(0) - pos[etq == 0].mean(0)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def recorta_al_cuello(pos: np.ndarray, tri: np.ndarray, etq: np.ndarray, *,
                      ventana: float = VENTANA_CUELLO_MM, paso: float = 0.1) -> np.ndarray:
    """Cede a la encía lo que cada pieza tiene por debajo de su cuello.

    La altura de corona de tabla fija **dónde buscar** —±`ventana` mm— y la concavidad
    decide **dónde exactamente** dentro de esa franja. Cada criterio sujeta al otro: la
    anatomía sola ya se midió que se Goodhartea, y la concavidad sola no sabe parar.
    """
    return _recorta(pos, tri, etq, concavidad(pos, tri), ventana, paso)


def _recorta(pos, tri, etq, c, ventana, paso):
    eje = eje_oclusal(pos, etq)
    h = pos @ eje
    e = etq.copy()
    for t in [int(x) for x in np.unique(etq) if x > 0]:
        m = etq == t
        cota = altura_admitida(t)
        if cota is None or int(m.sum()) < 200:
            continue
        centro = float(np.percentile(h[m], 98)) - cota
        mejor, mejor_c = centro, -np.inf
        for z in np.arange(centro - ventana, centro + ventana + 1e-9, paso):
            anillo = m & (np.abs(h - z) < paso)
            if anillo.sum() < 15:
                continue
            v = float(c[anillo].mean())
            if v > mejor_c:
                mejor, mejor_c = float(z), v
        e[m & (h < mejor)] = 0
    return e


# ── Anclar el margen al pliegue cervical ─────────────────────────────────────────
#
# Las constantes se ajustaron sobre `dev` del banco de pruebas (20 pacientes de Teeth3DS+
# que el modelo no vio) y NO sobre `eval`. Ver `scripts/benchmark_margen.py`.
#
# Cuánto abarata el corte pasar por un pliegue. La capacidad de una arista es su longitud
# dual por `exp(-PESO_PLIEGUE · concavidad normalizada)`: con 2, un pliegue como el del
# margen de un experto (≈1,8 veces la concavidad media) cuesta unas 35 veces menos. Entre
# 1 y 3 el resultado apenas cambia.
PESO_PLIEGUE = 2.0
# Concavidad normalizada a partir de la que una arista del corte cuenta como «en pliegue».
UMBRAL_PLIEGUE = 1.0
# A partir de esta concavidad media, el margen ACTUAL ya se considera en el surco y no se
# toca. El margen de un experto da 1,34 en su percentil 5.
MARGEN_YA_EN_PLIEGUE = 1.0
# Evidencia mínima para mover el margen de una pieza: concavidad media del corte y fracción
# de su longitud dentro de pliegue.
#
# ⚠️ **Más exigentes que el propio experto, a propósito.** El margen de un experto da 1,34
# y 0,65 en su percentil 5. Con esos valores el método dejaba 14 piezas de 520 con menos
# del 85 % de su corona; con estos, 9, de las que 5 ya lo estaban antes de tocarlas. El
# precio es declinar más (193 de 544) y corregir menos: un surco dudoso se deja como está.
FUERZA_MINIMA_PLIEGUE = 1.5
COBERTURA_MINIMA_PLIEGUE = 0.75
# La parte de la corona que NUNCA se corta: los vértices cuya distancia a la encía, medida
# sobre la malla, es al menos esta fracción de la máxima de su pieza.
#
# ⚠️ **No sale de la altura de tabla, y se probó.** La primera versión protegía todo lo que
# quedaba a menos de «altura de tabla − 2,5 mm» de la cúspide. Sobre Teeth3DS+ las coronas
# reales miden 2 mm MENOS que la tabla (mediana), así que esa zona contenía encía del
# experto en el 95 % de las piezas y el corte no podía seguir el surco: declinaba 117 de
# 165. Con la distancia a la encía, la contaminación mediana es del 0,1 % y no hace falta
# ni tabla ni eje oclusal, que sobre etiquetas desbordadas tampoco sale bien.
FRACCION_NUCLEO = 0.6
# Una pieza por debajo de esta fracción del tamaño mediano no se toca: es una esquirla.
FRACCION_PIEZA_MINIMA = 0.15

_CAPACIDAD_INFINITA = 10 ** 8
_ESCALA_CAPACIDAD = 1000.0


def _aristas_con_dual(pos: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aristas únicas `(m, 2)` y su longitud DUAL: lo que mide un corte al cruzarlas.

    Un corte en el grafo de vértices secciona aristas, y la curva que traza pasa por los
    centros de las caras de cada lado. El centroide de un triángulo dista de su arista un
    tercio de la altura, así que la longitud es `2·(A1 + A2) / (3·|arista|)`. Con el número
    de aristas a secas, el corte más barato sería el de menos aristas, no el más corto.
    """
    a, b, c = (pos[tri[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pares = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    unicas, inverso = np.unique(pares, axis=0, return_inverse=True)
    suma = np.zeros(len(unicas))
    np.add.at(suma, inverso.ravel(), np.tile(area, 3))
    largo = np.linalg.norm(pos[unicas[:, 1]] - pos[unicas[:, 0]], axis=1)
    return unicas, 2.0 * suma / (3.0 * np.maximum(largo, 1e-12))


def _corte_minimo(n: int, i: np.ndarray, j: np.ndarray, capacidad: np.ndarray,
                  fuentes: np.ndarray, a_sumidero: np.ndarray) -> np.ndarray:
    """Lado de la fuente (`bool`, `n`) del corte mínimo entre `fuentes` y el sumidero.

    `i, j, capacidad` son aristas no dirigidas entre nodos `0..n-1`; `a_sumidero[k]` es la
    capacidad del nodo `k` hacia el sumidero (0 si no lo toca).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    fuente, sumidero = n, n + 1
    toca = np.flatnonzero(a_sumidero > 0)
    filas = np.concatenate([i, j, np.full(len(fuentes), fuente), toca])
    cols = np.concatenate([j, i, fuentes, np.full(len(toca), sumidero)])
    caps = np.concatenate([capacidad, capacidad,
                           np.full(len(fuentes), _CAPACIDAD_INFINITA), a_sumidero[toca]])
    grafo = coo_matrix((caps.astype(np.int32), (filas, cols)),
                       shape=(n + 2, n + 2)).tocsr()
    flujo = maximum_flow(grafo, fuente, sumidero).flow
    residual = (grafo - flujo).tocsr()
    residual.data = np.where(residual.data > 0, 1, 0).astype(np.int32)
    residual.eliminate_zeros()
    # ⚠️ **De los cortes mínimos, el más cercano al sumidero.** Suele haber varios del mismo
    # coste —uno a cada lado del pliegue— y recorrer el residual desde la fuente da el más
    # alejado de la encía: una etiqueta que ya está en el surco perdería un anillo en cada
    # pasada. Recorrerlo hacia atrás desde el sumidero da el que menos mueve.
    llegan = breadth_first_order(residual.T.tocsr(), sumidero, directed=True,
                                 return_predecessors=False)
    lado = np.ones(n + 2, dtype=bool)
    lado[llegan] = False
    return lado[:n]


def _entero(capacidad: np.ndarray, referencia: float) -> np.ndarray:
    """Capacidades a entero, que es lo que pide `maximum_flow`, sin que ninguna quede en 0."""
    return np.clip(np.rint(capacidad / referencia * _ESCALA_CAPACIDAD), 1, 10 ** 7)


def ancla_al_pliegue(
    pos: np.ndarray, tri: np.ndarray, etq: np.ndarray, *,
    nucleo: float = FRACCION_NUCLEO, peso: float = PESO_PLIEGUE,
    fuerza_minima: float = FUERZA_MINIMA_PLIEGUE,
    cobertura_minima: float = COBERTURA_MINIMA_PLIEGUE,
    en_pliegue: float = MARGEN_YA_EN_PLIEGUE,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Lleva el margen de cada pieza al surco cervical, **o la deja como está**.

    Se busca el corte más barato que separa el núcleo de las coronas de la encía que hoy
    las rodea, con las aristas baratas dentro de un pliegue. Es un lazo sobre la
    superficie, no un plano: las caras vestibular y lingual pueden cortarse a alturas
    distintas, que es como es un margen gingival.

    ⚠️ **Un solo corte para toda la arcada, no uno por pieza, y se midió por qué.** Cuando
    la etiqueta de una pieza invade a la vecina —el «16» del modelo cubriendo un tercio del
    17—, el corte por pieza encuentra el surco ENTRE los dos dientes, que es un pliegue
    real y nítido, y manda a la encía media corona ajena. Con todos los dientes en el mismo
    grafo ese surco no separa nada: a los dos lados hay diente.

    Lo que lo distingue de `recorta_al_cuello`, que son los motivos por los que aquel no se
    aplica:

    - **Sabe declinar, pieza a pieza.** Si el tramo del corte que toca a una pieza no cae
      en un pliegue, esa pieza recupera su frontera y se dice. Aquel cortaba a la altura de
      tabla.
    - **No usa la altura de tabla ni el eje oclusal.** Lo que nunca se corta es lo más
      alejado de la encía sobre la propia malla (`FRACCION_NUCLEO`). Nada fija dónde cae el
      margen: una corona corta de adolescente y una larga por recesión se tratan igual.
    - **Solo cede.** Un vértice pasa de diente a encía, nunca al revés: el error medido es
      de un solo signo (el 45 % de la encía sale diente y menos del 1 % al contrario).
    - **No toca lo que ya está bien.** Una pieza cuyo margen ya cae en un pliegue se deja.

    Devuelve las etiquetas y, por pieza, un registro serializable con lo que hizo y la
    evidencia que tenía, para que la puerta de revisión humana lo enseñe.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    etq = np.asarray(etq, dtype=np.int64)
    salida = etq.copy()
    piezas = [int(t) for t in np.unique(etq) if t > 0]
    if not piezas or not (etq == 0).any():
        return salida, {}

    def registro(**campos: object) -> dict:
        base = {"moved": 0, "moved_fraction": 0.0, "evidence": None, "coverage": None,
                "concavity_before": None, "declined": False, "reason": None}
        return {**base, **campos}

    c = concavidad(pos, tri)
    s = c / max(float(np.abs(c).mean()), 1e-12)
    aristas, dual = _aristas_con_dual(pos, tri)
    s_arista = 0.5 * (s[aristas[:, 0]] + s[aristas[:, 1]])
    coste = _entero(dual * np.exp(-peso * np.clip(s_arista, 0.0, 4.0)),
                    float(np.median(dual)) or 1.0)
    largo = np.linalg.norm(pos[aristas[:, 1]] - pos[aristas[:, 0]], axis=1)
    ea, eb = etq[aristas[:, 0]], etq[aristas[:, 1]]

    dientes = np.flatnonzero(etq > 0)
    n = len(dientes)
    nodo = np.full(len(etq), -1, dtype=np.int64)
    nodo[dientes] = np.arange(n)
    dentro = np.flatnonzero((ea > 0) & (eb > 0))            # diente-diente, de la pieza que sea
    borde = np.flatnonzero((ea > 0) != (eb > 0))             # diente-encía: el margen de hoy
    gi, gj = nodo[aristas[dentro, 0]], nodo[aristas[dentro, 1]]
    del_diente = np.where(ea[borde] > 0, aristas[borde, 0], aristas[borde, 1])
    nb = nodo[del_diente]
    pieza_borde = etq[del_diente]

    # Distancia a la encía sobre la malla, dentro de la masa dental. La parte de cada pieza
    # que supera `nucleo` de su propio máximo es la que nunca se corta.
    masa = coo_matrix((largo[dentro], (gi, gj)), shape=(n, n)).tocsr()
    a_encia = dijkstra(masa, directed=False, indices=np.unique(nb), min_only=True)
    a_encia[~np.isfinite(a_encia)] = 0.0                     # islas que no tocan encía
    codigo = etq[dientes]
    suelo = FRACCION_PIEZA_MINIMA * float(np.median([(codigo == t).sum() for t in piezas]))
    es_fuente = np.zeros(n, dtype=bool)
    registros: dict[int, dict] = {}
    for t in piezas:
        suyos = codigo == t
        tope = float(a_encia[suyos].max())
        if suyos.sum() < suelo:
            registros[t] = registro(declined=True, reason="fragment")
            es_fuente |= suyos                              # una esquirla no se toca
        elif tope <= 0.0:
            registros[t] = registro(reason="no_gum_contact")
            es_fuente |= suyos
        else:
            es_fuente |= suyos & (a_encia >= nucleo * tope)

    a_sumidero = np.zeros(n)
    np.add.at(a_sumidero, nb, coste[borde])
    lado = _corte_minimo(n, gi, gj, coste[dentro], np.flatnonzero(es_fuente), a_sumidero)

    cruzan = lado[gi] != lado[gj]
    corte_dentro = dentro[cruzan]
    pieza_corte = np.where(lado[gi[cruzan]], codigo[gi[cruzan]], codigo[gj[cruzan]])
    sigue = lado[nb]                                         # tramo que no se ha movido
    for t in piezas:
        if t in registros:
            continue
        suyos = codigo == t
        mio = pieza_borde == t
        if not mio.any():
            registros[t] = registro(reason="no_gum_contact")
            salida[dientes[suyos]] = t
            continue
        antes = float(np.average(s_arista[borde[mio]], weights=dual[borde[mio]]))
        cedidos = dientes[suyos & ~lado]
        corte = np.concatenate([corte_dentro[pieza_corte == t], borde[mio & sigue]])
        datos: dict[str, object] = {"concavity_before": round(antes, 3)}
        if len(corte):
            datos["evidence"] = round(
                float(np.average(s_arista[corte], weights=dual[corte])), 3)
            datos["coverage"] = round(float(
                dual[corte][s_arista[corte] >= UMBRAL_PLIEGUE].sum() / dual[corte].sum()), 3)
        if not len(cedidos):
            registros[t] = registro(**datos)
        elif antes >= en_pliegue:
            # ⚠️ Un margen que ya está en un pliegue no se mueve. Sin esto el corte busca
            # igualmente el lazo más barato y a veces lo encuentra más arriba, en un surco
            # de la propia corona: sobre etiquetas de experto las dejaba en 0,981.
            registros[t] = registro(reason="already_in_crease", **datos)
        elif (not len(corte) or float(datos["evidence"]) < fuerza_minima  # type: ignore[arg-type]
              or float(datos["coverage"]) < cobertura_minima):  # type: ignore[arg-type]
            registros[t] = registro(declined=True, reason="weak_crease", **datos)
        else:
            salida[cedidos] = 0
            registros[t] = registro(moved=int(len(cedidos)),
                                    moved_fraction=round(len(cedidos) / int(suyos.sum()), 4),
                                    **datos)
    return salida, registros


def refina_fronteras(pos: np.ndarray, tri: np.ndarray, etq: np.ndarray) -> np.ndarray:
    """Los dos refinamientos, en el orden que importa.

    Primero se reparte entre piezas y luego se recorta el cuello: al revés, el recorte se
    haría sobre etiquetas que todavía invaden a la vecina y elegiría la altura mirando la
    concavidad de un diente que no es.
    """
    return recorta_al_cuello(pos, tri, reparte_por_geodesica(pos, tri, etq))


def calidad_por_pieza(pos: np.ndarray, tri: np.ndarray,
                      etq: np.ndarray) -> dict[int, dict]:
    """Por pieza, si su recorte está dentro de lo que hacen las etiquetas de experto.

    ⚠️ **Esto es lo que hace honesta la selección en el visor.** Sin este dato, seleccionar
    una pieza mal recortada devuelve medio diente vecino y nada lo dice; con él, el visor
    puede enseñar la pieza Y que su frontera no es de fiar, que es información clínica.

    El umbral no es una opinión: sale del `p95` de `|ancho medido - tabla|` sobre 188
    coronas etiquetadas por experto. Ver `EXCESO_ADMITIDO_MM`.
    """
    from export_agents.anatomia import anchos_de_corona

    salida: dict[int, dict] = {}
    for fdi, (medido, tabla) in anchos_de_corona(pos, tri, etq).items():
        exceso = float(medido - tabla)
        salida[int(fdi)] = {
            "mesiodistal_mm": round(float(medido), 2),
            "table_mm": round(float(tabla), 2),
            "excess_mm": round(exceso, 2),
            "within_expert_range": bool(abs(exceso) <= EXCESO_ADMITIDO_MM),
        }
    return salida
