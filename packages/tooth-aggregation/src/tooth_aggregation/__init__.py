"""Agregación **punto → diente** para el `segmentation-agent`.

Un modelo de segmentación de nubes de puntos devuelve una etiqueta **por punto**.
Un agente tiene que entregar **dientes**: instancias, cada una con su código FDI.
Este paquete es ese salto, y **solo** ese salto.

Deliberadamente **no depende de `torch`**: el *forward* del modelo (GPU) es del
llamante, que pasa aquí las log-probabilidades ya calculadas como `ndarray`. Así
esto se instala y se testea en el workspace normal, sin el venv de GPU.

El pipeline es:

1. `connected_labels` — de semántica a **instancias**: componentes conexas del
   grafo kNN restringido a aristas cuyos dos extremos comparten etiqueta predicha.
2. `merge_fragments` — une componentes de la **misma clase** que están **pegadas**.
   No es cosmético: sin esto la detección parte dientes en trozos.
3. `assign_unique` *(opcional)* — impone «un FDI por arcada» con asignación
   húngara: el reparto que maximiza la confianza total sin repetir código.

**Dos lecciones medidas** sobre Teeth3DS+, porque condicionan cómo se usa esto:

- **La métrica honesta es el acierto por DIENTE, no por punto.** El acierto por
  punto *subestima* la identificación por diente; la ficha del agente debe
  declarar la segunda.
- **`assign_unique` sin fusionar antes EMPEORA.** La restricción de unicidad
  presupone «una instancia = un diente»; con fragmentos esa premisa es falsa y
  forzar un FDI distinto por trozo *inventa* errores. Fusionando se recupera casi
  todo, pero seguía por debajo del voto mayoritario simple, así que
  `enforce_unique` viene **desactivado por defecto** a propósito.

Las cifras concretas viven **solo** en la ficha del experimento
(`notebooks/exercise-point-transformer-teeth3ds.md`), que es la fuente única: se
re-miden en cada corrida, y duplicarlas aquí las desincronizaría.
"""

from tooth_aggregation.aggregate import (
    ToothInstance,
    aggregate_teeth,
    assign_unique,
    connected_labels,
    majority_label,
    merge_fragments,
    suaviza_contiguidad,
    typical_spacing,
)

__all__ = [
    "ToothInstance",
    "aggregate_teeth",
    "assign_unique",
    "connected_labels",
    "majority_label",
    "merge_fragments",
    "suaviza_contiguidad",
    "typical_spacing",
]
