"""Un `Segmenter` para el campo del CBCT, hecho con las dos medidas que tenemos.

**Por qué hace falta.** `SegmentationAgent` está implementado desde hace tiempo, pero
`IngestionPipeline` lo construye **solo si se le pasa un `segmenter`**, y no había
ninguno. Así que la etapa no corría: `region_id` no se poblaba, la fusión semántica se
declaraba `MISSING` y los hallazgos del informe nunca llegaban a colgarse de un diente.
Mientras tanto, el compuesto —el entregable del proyecto— se montaba en un script que usa
**uno de los diez agentes**.

**La idea, y por qué son dos medidas y no una.** Ninguna de las dos fuentes basta sola:

- El **modelo del CBCT** sabe *qué* es diente y ve por debajo de la encía, que es lo único
  que ve la raíz. Pero es **binario**: no distingue el 36 del 37. Y no puede: los dientes
  se tocan en el punto de contacto interproximal, así que ni la conectividad ni el umbral
  de decisión los separan — medido, la componente de 40 mm sobrevive a los seis umbrales
  y a dos modelos con 24 puntos de precisión de diferencia
  (`docs/research/segmentacion-diente-cbct.md` §4).
- El **escáner intraoral** trae los dientes **ya separados y con nombre**, porque la
  frontera diente-encía sí está resuelta en una superficie de decenas de µm. Pero solo ve
  corona: por debajo del margen gingival no hay dato.

Así que uno dice **qué** y el otro dice **cuál**. Esta clase los junta: la probabilidad
del modelo decide diente contra encía, y el FDI sale de la corona etiquetada más cercana.
La separación entre dientes la pone el escáner, no la conectividad del volumen.

**Lo que NO puede hacer, y cómo se rodea.** Sin ayuda las piezas salen largas —27-32 mm
contra 20-25 anatómicos— porque el ligamento periodontal mide 0,15-0,38 mm frente a un
vóxel de 0,30 y por debajo de la cresta ósea **no hay frontera que resolver**. Eso no lo
arregla ningún clasificador, y está medido que no lo arregla; ver la ficha citada arriba.

Con `recorta_por_longitud` el ápice se corta a la longitud que la anatomía admite para ese
tipo de diente, medida sobre el eje propio de la pieza. ⚠️ Es un **prior, no una medida**:
a partir de ahí dónde acaba la raíz es algo que se supone, y medir longitud radicular sobre
el resultado sería medir lo que se ha supuesto. Por eso va apagado por defecto y lo pide
quien monta el caso, no el agente.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from analysis_agents.segmentation import DEFAULT_CODES, GUM_CLASS

# A más de esto de cualquier corona etiquetada, un punto se queda SIN nombre y cuenta como
# encía. Un diente entero mide ~22 mm y la corona ocupa los 8 superiores, así que el ápice
# de una raíz queda a ~15 mm de su propia corona. Con menos, las raíces se quedarían mudas;
# con mucho más, el hueso de alrededor heredaría el FDI del diente vecino.
RADIO_NOMBRE_MM = 16.0

# Cuanto puede quedar una gaussiana **al otro lado** de su corona, hacia la oclusion.
#
# ⚠️ Sin esta cota el nombrado funde cada diente con el que lo ocluye, y se ve: piezas de
# 44 y 47 mm cuando un diente mide 20-25, cruzando la encia por arriba y por abajo. La
# causa es que el radio de nombre es ISOTROPO y en oclusion la corona superior esta a 0-2
# mm de la inferior, asi que el diente de abajo entero —corona mas 15 mm de raiz— cae
# dentro de los 16 mm y hereda el FDI de arriba.
#
# La cota no es simetrica porque la anatomia no lo es: **la raiz se aleja de la oclusion**.
# Un diente se extiende desde su corona HACIA el hueso, y su punto mas oclusal es la propia
# corona que el escaner midio. Lo que queda mas alla de esa corona, hacia la boca, no es
# ese diente: es el que lo muerde. Los 3 mm son para el grosor del esmalte que el escaner
# no ve y el error de registro (medido: p50 0,8-4 mm segun el caso).
TOLERANCIA_OCLUSAL_MM = 3.0

# Coronas candidatas que se miran por punto. Con una sola no bastaria: la mas cercana puede
# ser la del diente que ocluye, y entonces el punto se quedaria mudo en vez de tomar la
# suya, que esta un poco mas lejos pero del lado correcto.
K_CORONAS = 16

# A esta distancia de una corona etiquetada, **manda el escaner y no el modelo**.
#
# ⚠️ Es donde estaba el hueco que se veia: la corona del escaner y la raiz del CBCT
# aparecian como dos objetos separados. Medido, no era desajuste de registro —el 94 % de
# las coronas tiene material del CBCT a menos de 2 mm, y la mediana es 0,68— sino que
# **el modelo no llamaba diente a ese material**: a 1 mm de una corona hay 5.215
# gaussianas nombradas y 2.927 sin nombrar, un 56 % mas, y eso con la capa decimada al 25 %.
#
# La justificacion no es de conveniencia: ahi la superficie la **midio un escaner intraoral
# con exactitud de decenas de micras**, y la opinion de un clasificador sobre un voxel de
# 0,30 mm es evidencia mas debil que esa medida. El origen del dato no cambia —esas
# gaussianas siguen siendo del CBCT y su `origen` lo dice— lo que cambia es quien decide
# que son diente.
RADIO_CORONA_MM = 1.0

# Longitud total (corona + raiz) por tipo de diente, en mm. De tablas anatomicas.
#
# ⚠️ **Esto es un PRIOR, no una medida, y cambia el estatuto del apice.** Con el recorte
# puesto, donde acaba la raiz deja de ser algo que el CBCT midio y pasa a ser algo que se
# supone. Sirve para que el compuesto no arrastre hueso alveolar; **no sirve para medir
# longitud radicular**, que seria circular — se mediria lo que se ha supuesto.
#
# Hace falta porque el modelo no puede parar donde acaba el diente: por debajo de la
# cresta osea no hay frontera que segmentar (ligamento periodontal 0,15-0,38 mm frente a
# un voxel de 0,30). Medido en este caso, el 16 salia con 34,3 mm cuando un molar superior
# entero mide ~20. Los 14 de mas son seno maxilar y hueso.
#
# El canino es el mas largo de la boca, y por eso tiene su propia entrada: recortarlo con
# la cota del incisivo le quitaria apice de verdad.
#
# ⚠️ La cota se mide a lo largo del eje de CADA PIEZA, no de uno global. Con un eje unico
# para toda la arcada el recorte apenas mordia: medido, el 16 se quedaba en 36 mm y el 26
# en 41 tras recortar a 22. La razon es geometrica — recortar a 22 mm *a lo largo del eje
# global* no deja la pieza de 22 mm de largo si la pieza esta inclinada respecto a ese eje,
# y los molares superiores lo estan. Ver `_ejes_por_pieza`.
LONGITUD_MM: dict[str, float] = {
    "incisivo": 24.0,
    "canino": 28.0,
    "premolar": 23.0,
    "molar": 22.0,
}

# ⚠️ **La cota se mide sobre el eje GLOBAL de la arcada, y no por falta de intentarlo.**
# Un eje por pieza deberia cortar mejor —los molares superiores estan inclinados— pero el
# escaner no contiene esa direccion, y esta medido. Tres estimadores sacados de la corona,
# los tres fallan:
#
#   * PCA de la corona, componente mas alineada con el eje global: angulos de 17 a 48
#     grados contra el global y extension axial de la corona entre 5,0 y 19,6 mm.
#   * normal del anillo cervical (menor varianza de la banda que toca encia): 33 a 90.
#   * centroide de la corona -> centroide del anillo cervical: p50 51 grados, max 87.
#
# El criterio de rechazo es el mismo para los tres: la corona clinica mide 7-9 mm, asi que
# su extension a lo largo del eje verdadero tiene que caer ahi. Ninguno lo consigue. La
# razon es que el escaner ve un CASQUETE: por debajo del margen gingival no hay dato, asi
# que apenas hay extension axial que detectar y las direcciones principales las manda el
# contorno de la corona, que es una propiedad del diente pero no SU EJE.
#
# Puesto en produccion el eje por pieza empeoro el resultado, que es lo que zanja la
# discusion: mediana 26,7 mm contra 25,8 con el global, y ocho piezas desbordadas contra
# seis. Se deja escrito para que nadie lo reintente sin datos nuevos — el eje tendria que
# venir del CBCT (donde la raiz si se ve) o de una anotacion, no de la corona.

# Percentil con el que se fija el extremo oclusal de cada pieza, para medir la longitud
# desde ahi. NO el minimo: las etiquetas del escaner traen vertices sueltos y uno solo mal
# colocado corre la cota entera.
PERCENTIL_OCLUSAL = 2.0


# Suelo y techo de probabilidad. **No es cosmética**: `SegmentationAgent` rechaza valores
# no finitos, y `log(0)` es `-inf`. Pasa en cuanto el modelo devuelve exactamente 1,0 para
# un punto —cosa que hace— porque entonces la columna de encía vale `1 - p = 0`. Sin el
# recorte la etapa entera sale `FAILED`, que es lo que pasó al conectarla.
#
# El error que mete en la suma es `~1e-12 · C`, muy por debajo de la tolerancia de `1e-3`
# con la que el agente comprueba que esto son log-probabilidades de verdad.
_PISO = 1e-12


# Vecinos con los que se decide si un vertice sin etiquetar es un HUECO o un BORDE.
K_VECINDARIO = 16
# Fraccion del vecindario que tiene que estar etiquetada, y fraccion que tiene que
# coincidir en la misma pieza. Ver `rellena_etiquetas`.
FRACCION_RODEADO = 0.85
FRACCION_ACUERDO = 0.85


def rellena_etiquetas(
    vertices: np.ndarray,
    etiquetas: np.ndarray,
    *,
    k: int = K_VECINDARIO,
    rodeado: float = FRACCION_RODEADO,
    acuerdo: float = FRACCION_ACUERDO,
) -> np.ndarray:
    """Cierra los agujeros del etiquetado del escaner. No extiende las piezas.

    El segmentador del escaner —otro modelo, `segmentar_fdi.py`— deja vertices sueltos sin
    etiquetar dentro de una corona, y esos salen pintados como encia. Un vertice **rodeado**
    de una misma pieza es esa pieza.

    ⚠️ **La distancia sola no vale, y por eso el criterio es el vecindario.** En el margen
    gingival la encia toca la corona de verdad: rellenar por radio se come el margen, que
    es justo la frontera clinica que interesa conservar. Medido sobre un caso real: por
    radio de 0,5 mm entrarian 10.128 vertices; exigiendo ademas que el vecindario este
    rodeado, quedan **1.851**. La diferencia son 8.000 vertices de margen que habrian
    pasado a ser diente sin serlo.

    Que el numero salga pequeno es el resultado, no un fallo del metodo: los huecos de
    verdad son pocos, y lo que parecia hueco era la banda del margen.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=np.float64)
    e = np.asarray(etiquetas).astype(np.int64).copy()
    sin = np.flatnonzero(e == 0)
    if len(sin) == 0 or (e > 0).sum() < k:
        return e

    _, idx = cKDTree(v).query(v[sin], k=min(k, len(v) - 1) + 1)
    vecinos = e[idx[:, 1:]]
    frac_etiquetado = (vecinos > 0).mean(axis=1)
    moda = np.array([
        np.bincount(x[x > 0]).argmax() if (x > 0).any() else 0 for x in vecinos
    ])
    frac_moda = np.array([(x == m).mean() for x, m in zip(vecinos, moda, strict=True)])

    rellena = (frac_etiquetado >= rodeado) & (frac_moda >= acuerdo) & (moda > 0)
    e[sin[rellena]] = moda[rellena]
    return e


class SegmentadorDental:
    """`Segmenter`: `(N, 3)` mm → `(N, C)` log-probabilidades, columna 0 = encía.

    `probabilidad_en` es la única dependencia de torch, y va **por fuera** a propósito:
    quien tenga GPU calcula el volumen de probabilidad y pasa aquí un callable. Así este
    módulo entra en el paquete sin arrastrar torch a todo el que importe `analysis_agents`,
    y se puede probar con una función de dos líneas.

    `coronas` y `etiquetas` son los vértices de corona del escáner **ya registrados en el
    marco del CBCT** y su código FDI. Registrarlos es de la fusión geométrica; aquí solo se
    consultan.
    """

    def __init__(
        self,
        probabilidad_en: Callable[[np.ndarray], np.ndarray],
        coronas: np.ndarray,
        etiquetas: np.ndarray,
        *,
        codes: dict[int, int] | None = None,
        radio_nombre_mm: float = RADIO_NOMBRE_MM,
        direccion_raiz: np.ndarray | None = None,
        tolerancia_oclusal_mm: float = TOLERANCIA_OCLUSAL_MM,
        radio_corona_mm: float = RADIO_CORONA_MM,
        recorta_por_longitud: bool = False,
    ) -> None:
        coronas = np.asarray(coronas, dtype=np.float64)
        etiquetas = np.asarray(etiquetas)
        if len(coronas) != len(etiquetas):
            raise ValueError(
                f"{len(coronas)} coronas y {len(etiquetas)} etiquetas: tiene que haber "
                "una etiqueta por vértice."
            )
        con_nombre = etiquetas > 0
        if not con_nombre.any():
            raise ValueError(
                "ninguna corona trae código FDI: sin nombres esto no puede decir CUÁL es "
                "cada diente, que es la mitad del trabajo que hace."
            )
        self.probabilidad_en = probabilidad_en
        self.coronas = coronas[con_nombre]
        self.etiquetas = etiquetas[con_nombre].astype(int)
        self.radio_nombre_mm = radio_nombre_mm
        self.codes = dict(DEFAULT_CODES if codes is None else codes)
        # FDI → índice de columna. Se invierte el mapa del agente para no depender del
        # orden de `all_fdi_codes()`.
        self._columna = {fdi: col for col, fdi in self.codes.items()}
        self._arbol = cKDTree(self.coronas)
        self.tolerancia_oclusal_mm = tolerancia_oclusal_mm
        self.radio_corona_mm = radio_corona_mm
        self.direccion_raiz: np.ndarray | None = (
            None if direccion_raiz is None
            else np.asarray(direccion_raiz, dtype=np.float64)
            / max(float(np.linalg.norm(direccion_raiz)), 1e-9)
        )
        # Cota apical por pieza: (eje propio, origen, avance maximo admitido).
        # Vacio si no se pidio el recorte o si no hay direccion con la que orientarlo.
        self._cota_apical: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        if recorta_por_longitud and self.direccion_raiz is not None:
            self._cota_apical = self._construye_cotas(self.direccion_raiz)
    def _construye_cotas(
        self, global_: np.ndarray
    ) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
        """Por pieza: su eje, su extremo oclusal y hasta donde puede llegar su apice."""
        from ingestion_agents.ontology import describe, is_valid_fdi

        cotas: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for codigo in np.unique(self.etiquetas):
            fdi_txt = str(int(codigo))
            if not is_valid_fdi(fdi_txt):
                continue
            largo = LONGITUD_MM.get(describe(fdi_txt).tooth_type.value)
            if largo is None:
                continue
            corona = self.coronas[self.etiquetas == codigo]
            eje = global_
            origen = corona.mean(axis=0)
            # El extremo oclusal por percentil, no por minimo: las etiquetas del escaner
            # traen vertices sueltos y uno mal colocado corre la cota entera.
            oclusal = float(
                np.percentile((corona - origen) @ eje, PERCENTIL_OCLUSAL)
            )
            cotas[int(codigo)] = (eje, origen, oclusal + largo)
        return cotas

    def _nombra(self, puntos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FDI por punto, o 0.

        Con `direccion_raiz`, un punto solo puede pertenecer a un diente si NO esta mas
        alla del extremo oclusal **de ese diente**. Un diente se extiende desde su corona
        hacia el hueso; lo que queda por delante de la corona, hacia la boca, es el diente
        que lo muerde.

        ⚠️ La referencia es el extremo de la PIEZA, no el vertice mas cercano, y la
        diferencia importa: comparando contra el vertice mas cercano, un punto profundo al
        que le falla su propio diente prueba con el siguiente candidato y acaba
        engancharse a un incisivo, que cuelga mas bajo que los molares. Por pieza no hay
        escapatoria — cada diente lleva su propia cota.
        """
        if self.direccion_raiz is None:
            d, vecino = self._arbol.query(puntos)
            return np.where(d <= self.radio_nombre_mm, self.etiquetas[vecino], 0), d

        # Se miran varias coronas candidatas, de la mas cercana en adelante, y gana la
        # primera que ademas este del lado correcto. Con una sola no bastaria: la mas
        # cercana suele ser la del diente que ocluye, y el punto se quedaria mudo en vez
        # de tomar la suya, que esta algo mas lejos pero es la buena.
        k = min(K_CORONAS, len(self.coronas))
        d, vecino = self._arbol.query(puntos, k=k)
        if k == 1:
            d, vecino = d[:, None], vecino[:, None]
        # La comparacion es LOCAL: contra el vertice candidato, no contra una cota de la
        # pieza entera. Medido, y la diferencia no es pequena — 24,2 mm de altura mediana
        # frente a 26,8 con la cota por pieza, y 6 desbordadas frente a 7. Un molar
        # inclinado tiene su corona repartida a lo largo del eje de la raiz, asi que una
        # cota unica para toda la pieza queda demasiado holgada justo donde mas aprieta.
        fuera = np.zeros(len(puntos), dtype=np.int64)
        cerca = np.full(len(puntos), np.inf)
        pendiente = np.ones(len(puntos), dtype=bool)
        for j in range(k):
            idx = np.flatnonzero(pendiente & (d[:, j] <= self.radio_nombre_mm))
            if not len(idx):
                continue
            corona = self.coronas[vecino[idx, j]]
            avance = (puntos[idx] - corona) @ self.direccion_raiz
            ok = avance >= -self.tolerancia_oclusal_mm
            fuera[idx[ok]] = self.etiquetas[vecino[idx[ok], j]]
            cerca[idx[ok]] = d[idx[ok], j]
            pendiente[idx[ok]] = False
        return fuera, cerca

    def __call__(self, points: np.ndarray) -> np.ndarray:
        puntos = np.asarray(points, dtype=np.float64)
        n, c = len(puntos), max(self.codes) + 1
        p = np.clip(
            np.asarray(self.probabilidad_en(puntos), dtype=np.float64),
            _PISO, 1.0 - _PISO,
        )
        if p.shape != (n,):
            raise ValueError(
                f"`probabilidad_en` devolvió {p.shape} para {n} puntos: se espera una "
                "probabilidad de diente por punto."
            )

        fdi, dist_corona = self._nombra(puntos)

        # Mas alla de la longitud que el tipo de diente admite, no es ese diente: es el
        # hueso que lo rodea. Ver `LONGITUD_MM` — y el apice pasa a ser SUPUESTO.
        for codigo, (eje, origen, cota) in self._cota_apical.items():
            idx = np.flatnonzero(fdi == codigo)
            if not len(idx):
                continue
            fdi[idx[(puntos[idx] - origen) @ eje > cota]] = 0

        # Junto a una corona medida, el escaner manda. Ver `RADIO_CORONA_MM`.
        junto_a_corona = (fdi > 0) & (dist_corona <= self.radio_corona_mm)
        p = np.where(junto_a_corona, np.maximum(p, 0.5 + _PISO), p)
        columna = np.array([self._columna.get(int(f), GUM_CLASS) for f in fdi])

        # Un punto sin nombre es encía **aunque el modelo diga diente**: si nada lo
        # reclama, no se le puede colgar un hallazgo clínico. Declarar «diente sin saber
        # cuál» sería inventar la mitad que falta.
        sin_nombre = columna == GUM_CLASS
        p = np.where(sin_nombre, _PISO, p)

        prob = np.full((n, c), _PISO)
        prob[:, GUM_CLASS] = 1.0 - p
        filas = np.flatnonzero(~sin_nombre)
        prob[filas, columna[filas]] = p[filas]
        return np.log(prob)


# Tamaño por debajo del cual una pieza del escaner se sospecha ISLA, en fraccion del
# tamano mediano de las piezas del caso. Medido sobre un caso clinico: el parche fantasma
# estaba en 0,06 y el diente real mas pequeno —un premolar mal cubierto— en 0,25, o sea
# cuatro veces por encima. El umbral va en fraccion y no en vertices porque el numero de
# vertices depende de la resolucion del escaner, y la proporcion entre piezas no.
FRACCION_ISLA = 0.15

# Y cuanto tienen que coincidir sus vecinos ajenos para que la absorcion sea de UNA pieza
# concreta. El tamano solo no basta: el 17 del mismo caso tambien tiene el 100% de sus
# vecinos en el 16 —esta al final de la arcada y solo tiene un vecino— y es un diente de
# verdad. Es el tamano el que separa; esto solo decide EN QUE pieza se absorbe.
ACUERDO_ISLA = 0.90


def absorbe_islas(
    vertices: np.ndarray,
    etiquetas: np.ndarray,
    *,
    k: int = K_VECINDARIO,
    fraccion: float = FRACCION_ISLA,
    acuerdo: float = ACUERDO_ISLA,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Absorbe las piezas diminutas metidas dentro de otra. Devuelve `(etiquetas, actas)`.

    El fallo que arregla, y que sólo se vio porque alguien contó los dientes del escaneo:
    el segmentador del escáner puso el código de un tercer molar a **275 vértices pegados
    a la cara distal del vecino** —mediana 0,85 mm de distancia—. No era un diente: era un
    parche. Y el daño no se quedaba ahí, porque esas etiquetas son las SEMILLAS con las
    que se nombra el CBCT: el código creció hacia la raíz y aterrizó sobre la tuberosidad
    del maxilar, que tiene tamaño y forma de molar. A partir de ahí el fantasma viajaba al
    visor, a las vistas y a la capa clínica como una pieza más.

    ⚠️ **Cada absorción se devuelve como acta, no se hace en silencio.** Un tercer molar
    parcialmente erupcionado se parece mucho a esto: pequeño y con un solo vecino. El
    tamaño los separa con holgura en el caso medido, pero la decisión de que un diente
    deja de existir es de las que un clínico tiene que poder mirar, así que sale en el
    gate y no en un log.

    `absorbe_islas` es lo contrario de `rellena_etiquetas` y por eso van juntas: aquella
    da etiqueta a lo que no la tiene, ésta se la quita a lo que la tiene mal.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=np.float64)
    e = np.asarray(etiquetas).astype(np.int64).copy()
    piezas = sorted({int(x) for x in np.unique(e) if x > 0})
    if len(piezas) < 2 or len(v) != len(e):
        return e, []

    mediana = float(np.median([int((e == f).sum()) for f in piezas]))
    if mediana <= 0:
        return e, []

    _, vecinos = cKDTree(v).query(v, k=min(k + 1, len(v)))
    vecinos = np.atleast_2d(vecinos)[:, 1:]

    actas: list[tuple[int, int, int]] = []
    for fdi in piezas:
        suyos = np.flatnonzero(e == fdi)
        if len(suyos) >= fraccion * mediana:
            continue
        # Sólo los vecinos que son de OTRA pieza: la encía no absorbe, porque una isla
        # rodeada de encía es un diente pequeño y suelto, no un trozo de su vecino.
        ajenos = e[vecinos[suyos]].ravel()
        ajenos = ajenos[(ajenos > 0) & (ajenos != fdi)]
        if len(ajenos) == 0:
            continue
        codigos, cuenta = np.unique(ajenos, return_counts=True)
        if cuenta.max() / len(ajenos) < acuerdo:
            continue
        destino = int(codigos[cuenta.argmax()])
        e[suyos] = destino
        actas.append((fdi, destino, len(suyos)))
    return e, actas
