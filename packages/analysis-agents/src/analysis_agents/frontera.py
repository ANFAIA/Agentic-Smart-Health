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
