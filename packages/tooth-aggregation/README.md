# `tooth-aggregation` — agregación punto → diente

Un modelo de segmentación de nubes de puntos devuelve una etiqueta **por punto**.
El `segmentation-agent` tiene que entregar **dientes**: instancias, cada una con su
código FDI. Este paquete es ese salto, y solo ese salto.

Extraído de [`notebooks/exercise-point-transformer-teeth3ds.ipynb`](../../notebooks/exercise-point-transformer-teeth3ds.md)
(sección A4), donde se midió sobre Teeth3DS+ completo.

## Sin `torch`, a propósito

El *forward* del modelo (GPU, `torch` + `pyg`) es del **llamante**, que pasa aquí las
log-probabilidades ya calculadas como `ndarray`. La agregación es geometría y
combinatoria pura: `numpy` + `scipy`.

Esto no es un detalle de estilo. El stack de GPU vive en un venv aparte
(`~/.venvs/dental-gpu`, ver [`notebooks/README.md` §04](../../notebooks/README.md)) que
está **fuera** del workspace `uv`. Un paquete que importara `torch` no se instalaría
aquí ni podría testearse en CI.

## Uso

```python
from tooth_aggregation import aggregate_teeth

# logprob (N, C) sale del modelo; points (N, 3) es la malla.
dientes = aggregate_teeth(points, logprob, codes={1: 11, 2: 12, ...})

for d in dientes:
    print(d.fdi, d.size, d.confidence)
```

Piezas sueltas, si hace falta control fino: `connected_labels` (instancias),
`merge_fragments` (fusión), `assign_unique` (unicidad FDI), `typical_spacing`.

## Dos cosas medidas que condicionan cómo se usa

**La métrica honesta es el acierto por DIENTE, no por punto.** En Teeth3DS+ el acierto
por punto *subestima* la identificación por diente. La ficha del agente debe declarar la
segunda.

**`enforce_unique` viene desactivado por defecto**, y no por prudencia genérica. La
restricción «un FDI por arcada» presupone *una instancia = un diente*; como la detección
parte dientes en fragmentos, esa premisa es falsa y forzar un código distinto por trozo
**inventa** errores. Medido sobre Teeth3DS+: activarlo **sin fusionar antes empeora** el
acierto por diente; con fusión se recupera casi todo, pero seguía **sin superar al voto
mayoritario simple**. Actívalo solo si mides que en tu caso la fusión deja de verdad una
instancia por diente.

> Las cifras concretas están **solo** en la
> [ficha del experimento](../../notebooks/exercise-point-transformer-teeth3ds.md), que es la
> fuente única: se re-miden en cada corrida y duplicarlas aquí las desincronizaría.

## Tests

```bash
uv run pytest packages/tooth-aggregation
```

Datos sintéticos (rejillas regulares, para poder razonar los umbrales a mano): que las
manchas lejanas no se fusionen, que los vecinos de distinta clase tampoco, que la fusión
sea transitiva, y que la húngara encuentre el **óptimo global** y no la solución voraz.
