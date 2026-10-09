# Protocolo de puntos anatómicos para medir el registro CBCT↔escáner

**Fecha:** 10 de octubre de 2026. **Estado:** borrador; se ratifica con los especialistas
que vayan a marcar. Mide el criterio A1 de la [ficha de uso previsto](uos-uso-previsto-v0.1.md).

## Para qué

El residuo de ICP mide cuánto encajan las superficies que el propio ICP emparejó. No dice si
una cúspide del CBCT cae sobre la misma cúspide del escáner. Para eso se marcan puntos
anatómicos en las dos modalidades, **sin usarlos para registrar**, y se mide la distancia
entre cada pareja después de aplicar el registro. Esa distancia es el TRE.

## Requisito previo

Las dos modalidades deben cubrir las mismas piezas. En el caso actual hay que aclarar antes
con la clínica qué representa cada serie: con huecos de varios milímetros entre cortes,
una cúspide puede caer donde no hay imagen. Un punto que no se ve con claridad no se marca.

## Puntos

Solo coronas: son lo único visible en las dos modalidades. Raíces y hueso no aparecen en
el escaneo intraoral.

| Etiqueta | Punto | Piezas |
|---|---|---|
| `<FDI>-mv` | Vértice de la cúspide mesiovestibular | Primeros molares: 16, 26 (36, 46 en mandíbula) |
| `<FDI>-c` | Vértice de la cúspide | Caninos: 13, 23 (33, 43) |
| `<FDI>-v` | Vértice de la cúspide vestibular | Primeros premolares: 14, 24 (34, 44) |
| `<FDI>-i` | Punto medio del borde incisal | Incisivos centrales: 11, 21 (31, 41) |

Son ocho puntos por arcada, repartidos entre cuadrantes y entre sector anterior y posterior.
La herramienta informa el error por región a partir del código FDI de la etiqueta.

Las etiquetas deben escribirse exactamente igual en las dos modalidades. Un punto con
etiqueta distinta queda sin pareja y se declara, no se descarta en silencio.

## Observadores

- Al menos **dos observadores**, que marcan por separado y sin ver los puntos del otro.
- Cada observador marca CBCT y escáner en sesiones distintas.
- Solo cuentan los puntos que marcaron todos los observadores. El informe lista los
  excluidos.
- El acuerdo entre observadores se informa junto al TRE: un TRE menor que el desacuerdo
  entre observadores no se puede distinguir de cero.

## Marcado en 3D Slicer

1. Cargar la serie DICOM desde el módulo DICOM.
2. Cargar el STL del escáner. Slicer pregunta el sistema de coordenadas del fichero: elegir
   **LPS** y anotarlo. Si se elige RAS, también vale, pero hay que anotarlo.
3. Crear una lista de puntos por modalidad con el módulo *Markups* y nombrar cada punto con
   su etiqueta.
4. Guardar cada lista como `.mrk.json`. Slicer escribe el sistema en el propio fichero.

⚠️ **Anotar cómo se cargó el STL no es opcional.** El STL no declara sistema de
coordenadas. Si se mide con el convenio equivocado, el TRE sale de decenas de milímetros y
parece un registro muy malo. En una prueba con el caso real, el mismo conjunto de puntos dio
0,00 mm con el convenio correcto y 98 mm con el contrario.

## Medida

```
uv run python scripts/tre_puntos.py --uos <caso>.uos \
    --cbct-puntos obs1-cbct.mrk.json --cbct-puntos obs2-cbct.mrk.json \
    --ios-puntos obs1-ios.mrk.json --ios-puntos obs2-ios.mrk.json \
    --stl-cargado-como lps --salida data/processed/<caso>/tre
```

El script evalúa el registro que el `.uos` declara desde el volumen DICOM hacia el escáner.
Con `--ajuste 16-mv,26-mv,13-c,23-c` calcula además un registro rígido con esos puntos y
mide el error en los demás y por *leave-one-out*. Esa segunda medida dice si un registro por
puntos alcanzaría el criterio. No sustituye al registro del contenedor.

Los resultados quedan en `data/processed/`. El script no escribe en el `.uos`: declarar un
TRE en el manifiesto exige una versión sucesora y la identidad de quien verificó.

## Si se modifica el algoritmo

Si el registro se cambia a la vista de estos resultados, los mismos puntos ya no sirven para
evaluarlo: se usaron para decidir. Hay que reservar puntos nuevos, o marcar de nuevo, antes
de medir la versión corregida.
