# Protocolo de líneas de margen gingival para puntuar la segmentación

**Fecha:** 13 de octubre de 2026. **Estado:** borrador; se ratifica con quien vaya a marcar.

## Para qué

El banco de [`scripts/benchmark_margen.py`](../../scripts/benchmark_margen.py) puntúa la
frontera corona-encía contra Teeth3DS+, que trae la etiqueta de un experto por vértice. Un
caso propio no la trae: hasta ahora solo se podía decir que algo parecía mal, nunca que
estaba bien. Este protocolo pide lo más barato que una persona puede aportar —**una línea
cerrada por pieza**— y [`scripts/puntua_margen.py`](../../scripts/puntua_margen.py) la
convierte en una medida.

## Qué se envía y qué no

- Solo el STL del escaneo, **con un nombre neutro**. Los nombres de fábrica suelen llevar
  el número de caso. Antes de enviarlo se comprueba que la cabecera de 80 bytes no trae
  texto identificativo.
- **No se envía la salida del programa**, ni imágenes de ella, ni la lista de piezas que
  detecta. Quien anota a ciegas da una referencia; quien ha visto la propuesta tiende a
  reproducirla, y la medida sale mejor de lo que es.
- Un escaneo sin nombre sigue siendo un dato de un paciente. Enseñarlo a un tercero tiene
  que estar cubierto por el permiso con el que se obtuvo.

## Qué se marca

Una curva cerrada por cada diente presente, sobre el margen gingival **tal como se ve en el
escaneo**: la corona clínica visible, no el límite amelocementario. Es el mismo criterio
con el que están hechas las etiquetas de Teeth3DS+, y por eso las medidas son comparables.

- Entre dientes se sigue el borde de la papila; donde dos piezas se tocan sin encía visible,
  la línea pasa por el punto de contacto.
- Con encía retraída se marca donde termina el tejido blando, y se anota.
- Donde el margen no se ve con claridad se marca igualmente y se anota la pieza y la cara.
- Cada línea se nombra con el código FDI que le dé quien anota.

## Marcado en 3D Slicer

1. Cargar el STL. Slicer pregunta el sistema de coordenadas del fichero: elegir **LPS** y
   anotarlo.
2. En el módulo *Markups*, crear una *Closed Curve* y **renombrarla con el código FDI antes
   de guardar**: el fichero toma el nombre de la curva, y es lo único que dice de qué pieza
   es.
3. Hacer clic sobre el modelo dando la vuelta completa a la corona. De 15 a 25 puntos por
   pieza, más juntos donde la línea gira.
4. Guardar. Sale un `.mrk.json` por curva, por ejemplo `16.mrk.json`.

## Anotadores

- Una anotación ya permite medir. **Dos** —dos personas, o la misma en dos días sin mirar
  la primera— dan además cuánto varía la línea, que es el suelo de la medida: un margen
  que diste de la línea menos que eso no se distingue de correcto.
- Junto a los ficheros se piden unas notas: especialidad, programa y versión, tiempo
  empleado, y en qué piezas o caras hubo duda.

## Medida

```
uv run python scripts/puntua_margen.py \
    --anotacion anotador-1/ --anotacion anotador-2/ --stl-cargado-como lps \
    --candidato actual=<caso>.uos --candidato pliegue=<etiquetas>.npz:despues \
    --salida data/processed/<caso>/margen-anotado
```

Cada candidato es un etiquetado sobre la misma malla que se anotó. El informe da, por
candidato y por pieza:

- la distancia del margen a la línea, con signo: positivo si nuestro margen cae sobre la
  encía (desborde), negativo si cae sobre la corona (sobre-recorte);
- las mismas tasas que el banco —encía marcada como diente, corona marcada como encía,
  piezas que conservan menos de **0,85** de su corona— tomando como etiqueta de referencia <!--const:CORONA_MINIMA_CONSERVADA-->
  lo que cada línea encierra.

Límites que el script declara y no esconde:

- La línea se sigue como polilínea entre puntos de control; el informe da la mayor
  separación entre dos puntos consecutivos.
- La región de cada pieza se obtiene con un lazo de vértices, así que su resolución es una
  arista de la malla.
- Un tramo de línea a menos de **0,5 mm** de la línea de otra pieza se trata como contacto <!--const:CONTACTO_MM-->
  interproximal y no cuenta como margen gingival. Es un convenio, no una medida.
- Si las líneas quedan a más de **1,0 mm** de la malla, se rechazan y se dice si el <!--const:DISTANCIA_MAXIMA_A_MALLA_MM-->
  convenio contrario (LPS/RAS) las pondría sobre ella. No se corrige solo.
- Una línea que no encierra ninguna región se declara y su pieza queda sin etiqueta.

## Qué no dice

Es el acuerdo con las líneas de una o dos personas sobre **un** caso. No es validación
clínica ni exactitud general de la segmentación.

## Si se modifica el método

Si un método se ajusta a la vista de estas líneas, dejan de servir para puntuarlo: se usaron
para decidir. Hay que pedir una anotación nueva, o reservar piezas, antes de medir la
versión corregida.
