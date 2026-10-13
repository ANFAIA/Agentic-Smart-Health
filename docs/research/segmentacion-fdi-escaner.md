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

Tres criterios:

1. **Tamaño contra el tipo.** La diagonal de la caja propia de la pieza —en sus ejes
   principales, recortando el 1 % de cada extremo para que una mota no mida por todos—
   contra la diagonal de la caja anatómica de esa corona (Wheeler: mesiodistal ×
   vestibulolingual × altura clínica). Salta a partir de 1,30.
2. **Simetría contralateral.** El 16 y el 26 son la misma pieza de la misma boca. Esta
   prueba no usa ninguna constante de población: la referencia es el propio paciente.
3. **La frontera, ¿cae en el pliegue?** Las dos anteriores miran el **tamaño** de cada
   pieza y las dos son ciegas a dónde está puesto el borde: una corona puede tener la
   diagonal correcta y la simetría correcta con la frontera corrida dos milímetros sobre la
   encía. Ésta mide la concavidad media de los vértices que tocan encía dividida por la
   concavidad típica de cualquier punto de la malla — el denominador la hace inmune a la
   escala del escáner y a la densidad del mallado.

   ⚠️ **Y su umbral sí está calibrado**, a diferencia del 1,30 de arriba. Sobre 12 maxilares
   de Teeth3DS+ con anotación de experto la razón sale **+1,82** (rango 1,50–2,25): el
   margen gingival es un pliegue y se mide. Nuestras etiquetas dan **−0,06**, y eso no es
   «la frontera está algo mal puesta» sino **está puesta donde la superficie no hace nada**.
   Es un diagnóstico distinto y más útil, porque explica por qué afinar el decodificador no
   la mueve (ver A9 en `notebooks/exercise-point-transformer-teeth3ds.md`). Devuelve `None`
   —no cero— cuando la escena no trae frontera diente/encía que medir.

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

### 6.1 · El banco de pruebas, y la línea base que da (2026-10-13)

Las cifras de frontera de este documento salieron de notebooks y ejecuciones sueltas, cada
una con su regla. Desde el 13 de octubre hay una sola, versionada:
[`scripts/benchmark_margen.py`](../../scripts/benchmark_margen.py), con las medidas en
`analysis_agents.margen_metricas`. Puntúa cualquier método de frontera contra las etiquetas
de experto de Teeth3DS+, sobre pacientes que el modelo no vio al entrenar.

**La partición es la del entrenamiento**, reproducida con su mismo código. El 20 % de test
se parte en `dev` (20 pacientes, para ajustar) y `eval` (40, para una sola ejecución
final). Comprobado con la cifra del notebook: el acierto FDI sobre vértices de diente da
**0,899** en los 20 maxilares de `dev`, contra el 0,898 que el notebook registró, y sube a
0,950 en cinco pacientes de entrenamiento. Sin esa diferencia la partición estaría mal.

Línea base sobre `dev`, 40 arcadas, medias por arcada:

| Método | Acuerdo diente/encía | Falso diente | Falsa encía | Margen p50 | Piezas con < 85 % de corona |
|---|---:|---:|---:|---:|---:|
| Modelo, sin tocar | 0,809 | 0,448 | 0,009 | 2,08 mm | 5 de 520 |
| + limpieza actual (pipeline de hoy) | 0,808 | 0,451 | 0,008 | 2,38 mm | 4 de 520 |
| + `recorta_al_cuello` | 0,831 | 0,363 | 0,029 | 1,61 mm | 21 de 520 |
| + `refina_fronteras` | 0,831 | 0,361 | 0,032 | 1,61 mm | 23 de 520 |
| + `reparte_por_geodesica` | 0,808 | 0,451 | 0,008 | 2,38 mm | 4 de 520 |

*Falso diente*: de lo que el experto llama encía, cuánto se etiqueta diente. *Falsa encía*:
de lo que llama diente, cuánto se etiqueta encía. *Margen p50*: distancia simétrica entre
el margen predicho y el del experto.

Tres cosas que esto mide por primera vez:

- **El error es casi todo de un signo.** El 45 % de la encía del experto sale etiquetada
  como diente y menos del 1 % de la corona sale como encía. Por eso el acierto sobre
  vértices de diente es alto y no dice nada del margen.
- **La limpieza que aplica el pipeline empeora el margen**: de 2,08 a 2,38 mm. Rellenar
  huecos avanza hacia la encía. Hace lo que promete entre piezas contiguas, y este coste
  no estaba medido.
- **`recorta_al_cuello` mejora la media y rompe piezas.** Acerca el margen en las 40
  arcadas, y a cambio deja 21 piezas con menos del 85 % de su corona, cinco veces más que
  antes. Es la misma bimodalidad que se vio sobre el caso clínico, ahora con número: una
  media que mejora mientras un puñado de piezas pierde media corona.

⚠️ Estas cifras son de Teeth3DS+: otro escáner, otra pose y otra población. Un método que
pase aquí está autorizado a probarse sobre un caso propio, no validado sobre él. Las
exploraciones anteriores ya habían mirado algunas arcadas de este test.

### 6.2 · Anclar el margen al surco cervical: resultado (2026-10-13)

Es el punto 2 de la lista de arriba, hecho: `analysis_agents.frontera.ancla_al_pliegue`.
Busca el corte más barato que separa el núcleo de las coronas de la encía que hoy las
rodea, con las aristas baratas dentro de un pliegue, y **deja como está la pieza donde no
encuentra surco**. No reentrena nada ni usa color.

Tres decisiones de diseño salieron de medir sobre `dev`, no de pensarlo:

- **Qué parte de la corona nunca se corta.** La primera versión la fijaba con la altura de
  corona de tabla. Las coronas de Teeth3DS+ miden 2 mm menos que la tabla, así que esa
  zona contenía encía del experto en el 95 % de las piezas y el método declinaba 117 de
  165. Ahora es lo más alejado de la encía sobre la propia malla: contaminación mediana
  del 0,1 %, sin tabla y sin eje oclusal.
- **Un corte para toda la arcada, no uno por pieza.** Cuando una etiqueta invade a la
  vecina, el corte por pieza encuentra el surco *entre* los dos dientes, que es un pliegue
  real, y manda a la encía media corona ajena.
- **Un margen que ya está en un pliegue no se toca.** Sin esa regla, aplicado a etiquetas
  de experto las dejaba en 0,981 de acuerdo.

Resultado, medias por arcada. `dev` sirvió para ajustar; `eval` se ejecutó una sola vez,
con las constantes ya fijadas:

| | Acuerdo diente/encía | Falso diente | Falsa encía | Margen p50 | Piezas con < 85 % de corona |
|---|---:|---:|---:|---:|---:|
| `dev` · pipeline de hoy | 0,808 | 0,451 | 0,008 | 2,38 mm | 4 de 520 |
| `dev` · + `ancla_al_pliegue` | 0,906 | 0,202 | 0,018 | 0,59 mm | 9 de 520 |
| `eval` · pipeline de hoy | 0,808 | 0,450 | 0,009 | 2,42 mm | 13 de 1065 |
| `eval` · + `recorta_al_cuello` | 0,826 | 0,354 | 0,044 | 1,63 mm | 70 de 1065 |
| `eval` · + `ancla_al_pliegue` | **0,939** | **0,114** | 0,024 | **0,17 mm** | 26 de 1065 |

Aplicado a las etiquetas del experto en `eval` las deja en 0,9988 de acuerdo.

⚠️ **El veredicto es NEGATIVO, por un criterio de seis.** Los criterios se fijaron antes
de ejecutar y los calcula el propio script:

| Criterio | `eval` | |
|---|---|---|
| A1 · el margen mejora en ≥ 75 % de las arcadas | 100 %; mediana −2,36 mm | cumple |
| A2 · el acuerdo no retrocede | +0,131; ninguna arcada retrocede | cumple |
| A3 · falsa encía ≤ +1,0 punto y ≤ 1 % de piezas dañadas | **+1,51 puntos; 13 de 1065 (1,2 %)** | **no cumple** |
| A4 · fracción etiquetada más cerca del experto | 0,119 → 0,033 | cumple |
| A5 · el ancho no empeora | p90 2,88 → 2,41 mm | cumple |
| A6 · no daña la verdad de campo (≥ 0,99) | 0,9988 | cumple |

Es decir: corrige el defecto dominante —la encía etiquetada como diente pasa del 45 % al
11 %— y a cambio recorta de más un poco por encima de lo que se había fijado como
tolerable. En `dev` cumplía los seis; en `eval`, no. Por eso **queda disponible y sin
aplicar**: `scripts/caso_completo.py --margen-gingival` vale `no` por defecto, `propone`
calcula y declara sin cambiar nada, y `aplica` lo cambia. `eval` queda **gastado**: un
intento posterior tiene que decirlo o reservar pacientes nuevos.

Dos precisiones que se hicieron antes de tocar `eval` y que hay que tener escritas:

- A3 decía «≤ 1 % de dientes conservan < 85 % de su corona» sin distinguir las piezas que
  el modelo ya traía rotas (5 de 520 en `dev`). Cuenta las que el método rompe: las que
  pasan de ≥ 85 % a < 85 %.
- Los umbrales de evidencia son más exigentes que el propio experto (1,5 y 0,75 frente a su
  percentil 5 de 1,34 y 0,65), para declinar los surcos dudosos.

**Sobre el caso clínico, que solo se puede falsar.** Aplicado a las etiquetas del escáner
de `another_patient` tras la limpieza: mueve 6 piezas y declina 8; la malla etiquetada baja
del 69,6 % al 61,1 % (el experto anda entre 44 y 60 %); la razón de concavidad del margen
sube de −0,06 a +0,24; siguen siendo 14 piezas. El número de piezas dentro del rango de
ancho del experto **no cambia: 4 de 14**, y una pieza, el 12, pierde el 53 % de su
etiqueta. No hay anotación con la que decir si eso es acierto o daño; la comparación
visual del antes y el después queda para revisión humana y no se versiona.

⚠️ El orden importa: va **después** de `rellena_huecos_interiores`. Un hueco de encía
dentro de una corona es encía para el corte, y puesto antes declinaba las 14 piezas.

### 6.3 · Puntuar un caso propio con líneas marcadas a mano (2026-10-13)

Un caso propio no trae la etiqueta de un experto, y por eso el párrafo de arriba termina
en «no hay anotación con la que decirlo». Lo más barato que una persona puede aportar es
una curva cerrada por pieza sobre el margen gingival, trazada en 3D Slicer.
[`scripts/puntua_margen.py`](../../scripts/puntua_margen.py) las lee y da, por pieza, la
distancia de nuestro margen a la línea con su signo —desborde o sobre-recorte— y las mismas
tasas que el banco, tomando como referencia lo que cada línea encierra. Con dos anotaciones
da además cuánto distan entre sí, que es el suelo de la medida. El marcado está en
[el protocolo](../spec/uos-protocolo-margen-gingival.md).

**No hay ninguna anotación todavía**: la herramienta está probada con mallas sintéticas y,
sobre la malla del caso, solo con líneas fabricadas a partir de nuestras propias etiquetas,
que comprueban que funciona y no dicen nada del margen. Cuando la haya, medirá un caso y
uno o dos anotadores; no es validación clínica.

## 7 · Consecuencia sobre el producto

`derived/seg_teeth.bin` sale con `regulatory.layer: 3` y `status: investigational`, que
es exactamente lo que es. El contenedor **sigue siendo válido sin él**: borrar `derived/`
y sus entradas del manifiesto es una operación soportada, y lo que queda —escaneo, CBCT,
campo medido, informe, registro, procedencia— no depende de esta etapa.

Es decir: la segmentación es la pieza no resuelta, y el formato está construido para que
no resuelta signifique *separable*, no *contaminante*.
