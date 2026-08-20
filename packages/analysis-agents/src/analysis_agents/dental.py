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

**Lo que NO hace.** No recorta la raíz. Las piezas siguen saliendo largas —27-32 mm contra
20-25 anatómicos— porque el ligamento periodontal mide 0,15-0,38 mm frente a un vóxel de
0,30 y por debajo de la cresta ósea **no hay frontera que resolver**. Eso no lo arregla
ningún clasificador, y está medido que no lo arregla; ver la ficha citada arriba.
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

# Suelo y techo de probabilidad. **No es cosmética**: `SegmentationAgent` rechaza valores
# no finitos, y `log(0)` es `-inf`. Pasa en cuanto el modelo devuelve exactamente 1,0 para
# un punto —cosa que hace— porque entonces la columna de encía vale `1 - p = 0`. Sin el
# recorte la etapa entera sale `FAILED`, que es lo que pasó al conectarla.
#
# El error que mete en la suma es `~1e-12 · C`, muy por debajo de la tolerancia de `1e-3`
# con la que el agente comprueba que esto son log-probabilidades de verdad.
_PISO = 1e-12


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

        d, vecino = self._arbol.query(puntos)
        fdi = np.where(d <= self.radio_nombre_mm, self.etiquetas[vecino], 0)
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
