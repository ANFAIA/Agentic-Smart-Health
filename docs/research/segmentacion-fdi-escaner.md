# Segmentar FDI sobre el escáner intraoral — por qué no está resuelto

> **Estado (2026-08-25):** resultado **negativo y medido**. La etapa corre, produce 14
> códigos FDI en un maxilar de 14 piezas y los produce **en el orden anatómico correcto**,
> pero la frontera de cada pieza está mal: 11 de las 14 se pueden descartar por anatomía
> sin necesidad de verdad de campo. Se cierra como está, declarado, porque lo que falta no
> es afinar un parámetro.

Complementa `segmentacion-diente-cbct.md`, que mide el otro lado del problema: allí el
clasificador dice **qué** es diente y no puede decir **cuál**; aquí el escáner dice cuál y
no acierta dónde acaba.

## 1 · De dónde salen las etiquetas

`scripts/segmentar_fdi.py`, con el checkpoint `normales-ce-baseline.pt` entrenado sobre
**Teeth3DS+**. Sobre el test de Teeth3DS+ ese modelo da **0,932 de FDI por diente**.

Ese 0,932 **no se transfiere** a estos casos, y no es una precaución retórica:

- otro escáner intraoral, con otra resolución y otra convención de ejes — medido, el eje
  oclusal de Teeth3DS+ va en `+z` y el de estos escaneos en `+y`, **coseno 0,004**;
- otras bocas, con patología y con piezas ausentes, que es justo lo que el dataset no
  tiene en la misma proporción;
- y sobre todo: **no hay etiquetas de estos pacientes**. Nadie ha marcado diente a diente
  sus escaneos. Así que no existe la cifra de acierto. No es que sea mala: no existe.

## 2 · Lo único que se puede medir sin verdad de campo

Plausibilidad anatómica, con `scripts/mide_segmentacion.py`:

```
uv run python scripts/mide_segmentacion.py CASO.uos
```

⚠️ **Es una prueba de falsación, no una nota.** Puede demostrar que una pieza está mal;
no puede demostrar que esté bien. Una corona del tamaño correcto con el nombre del vecino
la pasa entera — hay un test que fija exactamente eso por escrito
(`tests/test_mide_segmentacion.py::test_la_cota_es_SUPERIOR_no_una_nota`).

Dos criterios:

1. **Tamaño contra el tipo.** La diagonal de la caja propia de la pieza —en sus ejes
   principales, recortando el 1 % de cada extremo para que una mota no mida por todos—
   contra la diagonal de la caja anatómica de esa corona (Wheeler: mesiodistal ×
   vestibulolingual × altura clínica). Salta a partir de 1,30.
2. **Simetría contralateral.** El 16 y el 26 son la misma pieza de la misma boca. Esta
   prueba no usa ninguna constante de población: la referencia es el propio paciente.

## 3 · El resultado, sobre un maxilar de 14 piezas

| FDI | área mm² | diagonal | anatómica | razón | |
|---|---|---|---|---|---|
| 11 | 255,6 | 21,3 | 15,2 | **1,40** | se pasa |
| 12 | 117,1 | 16,7 | 12,6 | **1,32** | se pasa |
| 13 | 149,3 | 15,1 | 14,8 | 1,02 | |
| 14 | 132,1 | 15,2 | 14,2 | 1,07 | |
| 15 | 294,9 | 22,1 | 14,2 | **1,55** | se pasa |
| 16 | 269,2 | 19,4 | 16,7 | 1,17 | |
| 17 | 119,1 | 20,8 | 15,8 | **1,31** | se pasa |
| 21 | 265,5 | 21,3 | 15,2 | **1,40** | se pasa |
| 22 | 169,3 | 19,8 | 12,6 | **1,57** | se pasa |
| 23 | 158,0 | 17,2 | 14,8 | 1,16 | |
| 24 | 34,8 | 11,2 | 14,2 | 0,78 | |
| 25 | 164,9 | 17,7 | 14,2 | 1,25 | |
| 26 | 352,1 | 28,6 | 16,7 | **1,72** | se pasa |
| 27 | 358,6 | 27,2 | 15,8 | **1,72** | se pasa |

| par | áreas mm² | razón | |
|---|---|---|---|
| 11/21 | 255,6 · 265,5 | 1,04 | |
| 12/22 | 117,1 · 169,3 | 1,45 | |
| 13/23 | 149,3 · 158,0 | 1,06 | |
| 14/24 | 132,1 · **34,8** | **3,80** | asimétrico |
| 15/25 | 294,9 · 164,9 | **1,79** | asimétrico |
| 16/26 | 269,2 · 352,1 | 1,31 | |
| 17/27 | 119,1 · **358,6** | **3,01** | asimétrico |

**11 de 14 descartadas. Cota SUPERIOR de piezas correctas: 21 %.**

## 4 · Qué dicen estos números que no dice «está mal»

**4.1 · Lo que sobra es cuerpo, no motas.** Sin recortar extremos las diagonales salen
1,5–3 mm mayores y **saltan exactamente las mismas piezas**. No es ruido salpicado que un
filtro quite: es superficie contigua mal asignada.

**4.2 · El error va hacia la encía, no hacia el vecino.** Era la hipótesis natural —el
contacto interproximal no tiene borde geométrico— y la medida no la sostiene como causa
principal. Un incisivo central mide 10,5 mm de corona; el 11 y el 21 miden **21,3 mm de
diagonal**, el doble. Eso no cabe en un punto de contacto: es margen gingival y encía
adherida heredando el código de la corona. Se ve directamente en el visor, y es la misma
observación que «no toda la encía se colorea»: **no está sin colorear, está coloreada de
diente**. Medido, el 67,1 % del área de la malla lleva código FDI, cuando 14 coronas
suman ~2.900 mm² de los 4.366 mm² del escaneo.

**4.3 · Y el error simétrico existe.** El 24 tiene **el 26 % del área de su contralateral**
y una diagonal de 11,2 mm: no se ha comido a nadie, ha perdido casi toda su corona.
Los dos modos de fallo conviven en la misma arcada.

**4.4 · El orden anatómico sí sale bien.** Los 14 códigos aparecen ordenados a lo largo
del arco, sin cuadrantes imposibles. Es lo que `segmentar_fdi.py` comprueba y lo que la
canonización de pose arregló. **Es necesario y no es suficiente**, y confundir las dos
cosas es lo que haría pasar esta etapa por resuelta.

## 5 · Por qué no se arregla con este planteamiento

El modelo decide **por vértice**, y un vértice del margen gingival es —geométricamente—
indistinguible de un vértice de cuello: misma curvatura, mismas normales, misma vecindad.
La frontera diente-encía no está en la geometría de la superficie, está en el **color** y
en la **anatomía del festoneado cervical**, y ninguna de las dos entra hoy en la decisión:

- el escaneo se ingiere como STL, **sin color por vértice**, así que la señal que un
  clínico usa para ver dónde acaba el esmalte no llega al modelo;
- la limpieza posterior (`analysis_agents.dental`) tiene **prohibido mover el margen
  gingival** por diseño, y esa prohibición es correcta: un suavizado que empuja el margen
  unas décimas cambia una medida clínica. Puede cerrar huecos, absorber islas y afilar la
  frontera **entre piezas contiguas**, y eso hizo (área con dos dientes 308 → 92 mm²);
  no puede inventar el borde que el modelo no puso.

## 6 · Lo que habría que hacer, y en qué orden

1. **Meter el color en la ingesta.** Es la señal que falta, no un adorno. La vía medida
   son las **fotos clínicas que el contenedor ya lleva**: un umbral sobre `a*` separa
   diente de encía con 3,4–4,3 σ y traza el festoneado cervical
   (`frontera-encia-desde-foto.md`). Lo que falta ahí es la pose de cámara, no el color.
   El color por vértice del escáner nativo es la alternativa, y es peor: depende del
   modelo de escáner y tiene ~112.000 muestras contra los 15 Mpx de una foto.
2. **Anclar el margen a la cresta de curvatura cervical** en vez de a la salida del
   modelo: es una curva detectable sobre la propia malla y es la definición clínica.
3. **Anotar unas pocas arcadas propias.** Sin verdad de campo de estos escáneres no habrá
   nunca una cifra de acierto, solo cotas superiores como la de §3. Diez arcadas
   etiquetadas cambian más que otro entrenamiento sobre Teeth3DS+.
4. Y solo después, reentrenar. Está medido en `segmentacion-diente-cbct.md` que subir la
   precisión del clasificador **no** arregló el defecto que motivó entrenarlo; no hay
   razón para esperar que aquí sí.

## 7 · Consecuencia sobre el producto

`derived/seg_teeth.bin` sale con `regulatory.layer: 3` y `status: investigational`, que
es exactamente lo que es. El contenedor **sigue siendo válido sin él**: borrar `derived/`
y sus entradas del manifiesto es una operación soportada, y lo que queda —escaneo, CBCT,
campo medido, informe, registro, procedencia— no depende de esta etapa.

Es decir: la segmentación es la pieza no resuelta, y el formato está construido para que
no resuelta signifique *separable*, no *contaminante*.
