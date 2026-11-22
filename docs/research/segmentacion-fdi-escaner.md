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

### 6.4 · La numeración, medida aparte del margen (2026-11-09)

Cruzar las curvas revisadas del caso clínico con nuestras etiquetas sacó un error más grave
que el margen: un «17» casi entero sobre la encía, un «24» de 41 mm² y, entre medias, los
números corridos un puesto. El acuerdo FDI por vértice apenas lo ve, así que el banco
cuenta ahora la identidad de cada etiqueta (`margen_metricas.identidad_de_piezas`): si cae
sobre la pieza del experto de su número, sobre otra (y si es la vecina), sobre la encía
(fantasma), si es un fragmento de más, y qué piezas no tiene nadie.

Línea base sobre `dev`, 40 arcadas y 520 piezas del experto:

| Fila | Correctas | Desplazadas (a la vecina) | Fantasmas | Fragmentos | Ausentes | Arcadas perfectas |
|---|---:|---:|---:|---:|---:|---:|
| Modelo, sin tocar | 498 | 22 (21) | 42 | 11 | 11 | 12 de 40 |
| + limpieza actual | 498 | 22 (21) | 24 | 11 | 11 | 21 de 40 |

La limpieza quita la mitad de los fantasmas y no toca los desplazamientos, que casi siempre
son al vecino. La predicción se regeneró guardando la log-probabilidad por vértice: las
etiquetas salieron idénticas a las de octubre en 119 de 120 arcadas, y en la otra difieren
en el 0,01 % de los vértices.

**En el caso clínico la pose pesa.** Su razón de orientación es 0,62, dudosa, y la
numeración depende de la arcada de Teeth3DS+ que se usa como referencia para canonizarla:
con la de agosto salen corridas 4 piezas; con otra, 13 de 14, todas un puesto. Un
desplazamiento uniforme está ordenado, así que imponer el orden de la arcada no basta para
corregirlo: hace falta algo que ancle cada número a su tipo de diente.

### 6.5 · Renumerar por orden y ancho: resultado negativo (2026-11-09)

Se probó, sin reentrenar, a elegir la secuencia de códigos de máxima verosimilitud que
respeta el orden de la arcada: programación dinámica sobre las instancias ordenadas, con
la log-probabilidad media del modelo más un término por lo verosímil que es el ancho de
cada pieza para cada código, huecos permitidos, fantasmas a encía y fragmentos. Los anchos
de referencia se midieron con el mismo estimador sobre 80 pacientes de **entrenamiento**.

**No mejora nada en `dev`, y no se evaluó en `eval`.** Con peso de ancho 0,25 deja la
numeración igual (498 de 520 correctas); con 0,5 baja a 496 y con 1,0 a 486, y empieza a
renumerar dientes bien numerados (sobre etiquetas de experto, 517 de 520 conservan su
número con 0,5). La causa está medida: los errores del modelo en Teeth3DS+ son de alta
confianza —la pieza mal numerada da −0,4 de log-probabilidad media a su código y −3 al
correcto— y el ancho de una corona varía demasiado entre pacientes para vencer eso. En el
caso clínico es peor: el método prefiere mandar piezas a la encía. Queda descartado.

**Lo que sí explica el caso clínico es la pose.** Se repitió la segmentación con seis
arcadas de entrenamiento distintas como referencia para canonizar, y se comparó con las
curvas revisadas:

| Referencia | Correctas | Desplazadas | Fantasmas | Confianza media en dientes |
|---|---:|---:|---:|---:|
| `WKDWFB4W` | 12 | 0 | 2 | −0,298 |
| `01A6HE9H` | 12 | 0 | 2 | −0,312 |
| `01M24NSZ` | 12 | 0 | 2 | −0,316 |
| `22AEG48N` | 11 | 0 | 3 | −0,318 |
| `EJWZZZRF` | 10 | 1 | 4 | −0,327 |
| `01FC2D4A` | 9 | 3 | 3 | −0,353 |
| `013FHA7K` | 0 | 11 | 4 | −0,389 |

La ejecución de agosto, con otra referencia, daba 7 correctas y 4 desplazadas. En las siete,
la monotonía del orden sale entre 0,94 y 1,00, así que no distingue las buenas de las malas.
La **confianza media del modelo sobre los dientes** sí las ordena igual que el acierto, sin
mirar ninguna anotación. Es un solo caso y unas curvas revisadas, no anotadas a ciegas: es
una hipótesis para medir, no un resultado.

### 6.6 · Varias referencias y voto: resultado en `eval` (2026-11-09)

Aumentación en tiempo de inferencia: el escaneo se canoniza contra 18 arcadas de
entrenamiento, se segmenta con cada una y cada vértice toma el código que le dan las 3
segmentaciones más confiadas (`analysis_agents.pose_votada`). El banco es
[`scripts/benchmark_pose.py`](../../scripts/benchmark_pose.py). La configuración se eligió
en `dev` y los criterios se fijaron en un commit antes de ejecutar `eval`, una sola vez.

| `eval`, 80 arcadas, 1065 piezas | Correctas | Desplazadas | Fantasmas | Arcadas perfectas |
|---|---:|---:|---:|---:|
| Pose original de Teeth3DS+ (techo) | 1017 | 66 | 52 | 30 |
| Una sola referencia (lo que se hace hoy) | 363–1009; mediana por arcada, 896 | | | |
| Voto entre 3 de 18 referencias | 979 | 93 | 57 | 31 |

**No cumple tres de seis criterios**: queda a 38 piezas del techo (se admitían 11), con 5
fantasmas más y 0,025 menos de acierto por vértice. **Cumple** los otros tres: gana a la
mediana de una sola referencia por 83 piezas y solo pierde contra ella en 2 arcadas; la
confianza ordena las referencias como su acierto en 78 de 80 arcadas; y no mueve el margen.

La pérdida frente al techo se concentra: en 64 de 80 arcadas el voto numera igual que la
pose original, en 4 mejor y en 12 peor, y dos arcadas con razón de orientación alta (0,85 y
0,76) se llevan 24 de las 38 piezas perdidas. Con razón ≤ 0,6 queda a 7 piezas del techo
(599 frente a 606).

Lectura: el techo es la pose en que se entrenó el modelo, y un escáner real no la da. Frente
a lo que el pipeline puede hacer con un escaneo real —una referencia—, el voto es mejor y
mucho más estable; no alcanza el techo cuando la orientación de la propia malla es ambigua.
En el caso clínico, con razón 0,62, numera 12 de 14 piezas como las curvas revisadas y
ninguna desplazada, frente a 7 y 4 con la referencia de agosto. `eval` queda gastado también
para esta hipótesis.

### 6.7 · Recortar menos corona: segundo intento, parado en `dev` (2026-11-09)

`ancla_al_pliegue` falló A3 en `eval` por recortar corona de más (§6.2). En `dev` el daño se
concentra: de 346 piezas movidas, el 95 % de lo que cede era encía del experto y solo 3
pierden más del 15 % de su corona, dos últimos molares y un canino, donde el corte sigue un
pliegue fuerte que no es el cuello. Se probaron dos formas de recortar menos: proteger más
corona (`M7`, núcleo 0,4) y exigir un pliegue más marcado (`M8`, fuerza ≥ 2,0).

Antes de ejecutarlas se fijó una regla en `scripts/benchmark_margen.py`
(`VARIANTES_SEGUNDO_INTENTO`): como `eval` había salido unas 1,5 veces peor que `dev`, una
variante solo pasa a `eval` si en `dev` cumple A1-A6 con A3 a la mitad de su límite.

| `dev`, 40 arcadas | Falso diente | Subida de falsa encía | Piezas dañadas | ¿Holgura? |
|---|---:|---:|---:|---|
| `M5`, el método | 0,202 | +1,00 puntos | 5 de 520 | no |
| `M7`, núcleo 0,4 | 0,295 | +0,60 | 2 (0,38 %) | no |
| `M8`, pliegue ≥ 2,0 | 0,306 | +0,58 | 4 (0,77 %) | no |

**Ninguna pasa, así que `eval` no se ha tocado.** Las tres cumplen A1-A6 en `dev`, y las dos
variantes pagan el recorte de menos dejando un 50 % más de encía etiquetada como diente.

**Sobre el caso clínico, con una línea trazada a ciegas.** Una periodoncista marcó las 14
piezas en 3D Slicer sin ver la salida del pipeline. Su línea y la de la segunda anotación,
propuesta por una IA y revisada por otra dentista, distan 0,15 mm de mediana: es el suelo de
la medida. Contra la línea a ciegas, con la numeración votada:

| | Margen → línea, p50 | Encía marcada diente | Corona marcada encía | Piezas < 85 % |
|---|---:|---:|---:|---:|
| Sin método | 3,53 mm | 51,9 % | 0,0 % | 0 de 14 |
| `M5` | **0,19 mm** | 21,1 % | 1,6 % | 0 de 14 |
| `M7` | 3,03 mm | 47,3 % | 0,3 % | 0 de 14 |
| `M8` | 1,80 mm | 36,8 % | 0,9 % | 0 de 14 |

`M5` mueve 11 de 14 piezas y las deja todas a menos de 0,3 mm de la línea; declina el 11, el 12 y
el 22, que siguen desbordados entre 1,8 y 4 mm. La pieza que más corona pierde conserva el
94 %. La numeración votada coincide pieza a pieza con la línea a ciegas.

Lectura: en este caso el método hace lo que promete y su defecto en Teeth3DS+, recortar de
más, no aparece. Es **un caso y una anotadora**: no deshace el fallo de A3 sobre 80 arcadas
y no es validación clínica. Las variantes prudentes no se adoptan: no pasan la regla en
`dev` y en el caso devuelven casi todo el desborde. El método sigue apagado por defecto.

### 6.8 · Las piezas que el método declina: tercer intento, parado en `dev` (2026-11-10)

En el caso clínico, `M5` deja el 11, el 12 y el 22 desbordados entre 3,5 y 4 mm porque su
surco no llega a los umbrales (`weak_crease`). En `dev` pasa lo mismo a escala: declina 193
de 544 piezas, y esas concentran casi todo el error que queda.

| `dev`, `M5` | Piezas | Margen → experto, mediana | Encía marcada diente, mediana |
|---|---:|---:|---:|
| Declinadas | 193 | 2,16 mm | 54 mm² |
| Movidas | 351 | 0,08 mm | 1,2 mm² |

Los umbrales del método (fuerza 1,5, cobertura 0,75) están por encima, a propósito, del
percentil 5 del margen del experto (1,34 y 0,65). `M9` acepta las piezas con un surco tan
claro como ese. Los criterios se fijaron antes de ejecutar, comparando con `M5` porque
ninguno de los dos cumple A3 frente a `M1`. Al elegir la variante ya se conocían los valores
del caso: con estos umbrales pasarían el 11 y el 12, y no el 22.

| Criterio, `M9` frente a `M5` | Límite | Resultado | |
|---|---|---|---|
| R1 · margen de las piezas que ahora acepta | baja a la mitad y mejora el 75 % | 41 piezas: 1,57 → 0,12 mm, mejoran todas | pasa |
| R2 · piezas que pasan a < 85 % de corona | ≤ 0,5 % | 5 de 520, 0,96 % | **falla** |
| R3 · subida de falsa encía | ≤ 0,5 puntos | +0,36 | pasa |
| R4 · acuerdo sobre etiquetas del experto | ≥ 0,99 | 0,997 | pasa |

**Falla R2, así que no se aplica al caso clínico y las líneas de las anotadoras no se han
usado.** Las cinco piezas dañadas son de la arcada inferior: cuatro molares (36 y 37) y un
canino (43), con un surco aceptado de cobertura 0,65-0,74 que no es el cuello. En la arcada
superior no se daña ninguna. Esa observación es posterior al resultado: una regla que la use
estaría ajustada sobre `dev`, y su única prueba independiente que queda es el caso clínico.

### 6.9 · `M9` en el maxilar, probado sobre el caso clínico (2026-11-10)

Por decisión del responsable del proyecto, `M9` se probó una vez sobre el caso clínico, que
es maxilar, aun habiendo fallado R2 en `dev`. Se declara como lo que es: **la prueba que
decide**, con las dos anotaciones que hasta ahora solo habían medido.

| Contra la línea a ciegas | `M5` | `M9` |
|---|---:|---:|
| 11, margen → línea | 3,54 mm | 0,22 mm |
| 12 | 4,14 mm | 0,15 mm |
| 22, sigue declinado | 1,75 mm | 1,75 mm |
| Caso, mediana | 0,19 mm | 0,15 mm |
| Encía marcada diente | 21,1 % | 15,9 % |
| Corona marcada encía | 1,6 % | 1,7 % |
| Piezas < 85 % | 0 de 14 | 0 de 14 |

Contra la segunda anotación sale lo mismo (11: 3,62 → 0,34 mm; 12: 4,23 → 0,14 mm). La pieza
que más corona pierde conserva el 98 %.

**Qué se adopta.** `--margen-gingival aplica-experto` en `scripts/caso_completo.py`: el listón
del surco baja al del experto (`analysis_agents.frontera.umbrales_de_pliegue`) solo si todas
las piezas etiquetadas son del maxilar; si no, se usan los umbrales de siempre y se dice.
`aplica` sigue igual y el valor por defecto sigue siendo `no`.

**Lo que no se sabe.** La restricción al maxilar sale de mirar el daño en `dev` después del
resultado, y la elección se hizo sobre las dos anotaciones de este caso, que dejan de servir
para medirla. Confirmarla exige una línea nueva trazada a ciegas, sobre este caso u otro
maxilar. Es un caso.

### 6.10 · La separación entre dientes, y un modelo reentrenado para ella (2026-11-20)

**Diagnóstico, sobre `dev` y antes de proponer nada.** Con el pipeline real —voto entre 3
de 18 referencias y la limpieza de `caso_completo.py`— y las predicciones ya cacheadas, se
contó cada fallo por pieza del experto (40 arcadas, 520 piezas):

| Fallo | Medida | Piezas afectadas |
|---|---|---:|
| Numeración | etiqueta emparejada con otra pieza, o pieza sin etiqueta | 22 (4 %) |
| **Separación** | la pieza cede o toma más del 10 % de su área a una vecina | **165 (32 %)** |
| | … más del 30 % | 36 |
| Frontera interproximal | distancia a la del experto, mediana por contacto | 0,72 mm; 146 de 439 contactos > 1 mm |

La pose no lo explica: con la pose original de Teeth3DS+ salen 147 piezas y 0,72 mm. El
sangrado es una franja pegada al vecino (mediana 0,4-0,5 mm de la pieza dueña), no
fragmentos sueltos. Y es lo que cabe esperar de cómo ve el modelo la arcada: trozos de
2.048 puntos, que sobre unos 9.500 mm² dejan ~2,15 mm entre puntos, cuando la malla tiene
0,17 mm entre vértices. El error de frontera crece con ese espaciado (Spearman 0,36 sobre
40 arcadas): es una correlación modesta y se toma como hipótesis, no como causa probada.

Así que el fallo que se ataca es la **separación**, y por la vía de la causa: reentrenar el
mismo Point Transformer con más puntos por trozo. La numeración y el margen no son el
objetivo, pero no pueden empeorar.

**El conjunto reservado.** Los 300 pacientes descargados son todos el
`public-training-set-1` de 3DTeethSeg'22 y están gastados: 240 entrenaron el modelo y 60 son
`dev` y `eval`. El proyecto OSF `xctdy` tiene el dataset entero en siete zips (ver
`dataset-teeth3ds.md` §3); de los 600 pacientes que faltan, 300 son el `private-testing-set`
del reto, que no ha visto **nadie** con datos públicos. La reserva son 60 de ellos, elegidos
con semilla fija por [`scripts/fetch_teeth3ds_osf.py`](../../scripts/fetch_teeth3ds_osf.py)
y versionados en [`teeth3ds-reserva.txt`](teeth3ds-reserva.txt) antes de bajar ninguno. Los
otros 300, el `public-training-set-2`, se usan para entrenar.

**El candidato.** [`scripts/entrena_fdi_escaner.py`](../../scripts/entrena_fdi_escaner.py):
la arquitectura, la entrada y la pérdida del notebook; trozos de 8.192 puntos, giros
aleatorios de hasta ±N° (nunca espejo, que cambiaría el lado FDI) y 1.080 arcadas en vez de
480. Se prueban como mucho tres valores de giro y se elige en `dev`.

**La métrica nueva.** `margen_metricas.separacion_interproximal`: por cada par de piezas del
experto que se tocan, la distancia simétrica entre su frontera y la predicha; por pieza, el
área cedida y tomada. Empareja las etiquetas con las piezas por solape, como
`identidad_de_piezas`, así que un número cambiado no cuenta como frontera mal puesta.

**Criterios, fijados en este commit antes de entrenar** (`scripts/benchmark_separacion.py`).
C = candidato, B = `normales-ce-baseline.pt`, los dos en el pipeline real:

| | Criterio |
|---|---|
| K1 | mediana de todos los contactos de C ≤ 0,75 × la de B, y mejora en ≥ 70 % de las arcadas |
| K2 | piezas que sangran > 10 %: C ≤ 0,75 × B |
| K3 | piezas bien numeradas C ≥ B, y fantasmas C ≤ B |
| K4 | la distancia del voto al techo de la pose nativa no crece |
| K5 | falso diente ≤ B + 0,01, falsa encía ≤ B + 0,5 puntos, piezas que pasan a < 85 % de corona ≤ 0,5 % |
| K6 | acierto FDI por vértice de diente ≥ B − 0,005 |

**Precisión del protocolo, fijada antes de ver ningún candidato** (2026-11-20, mientras
entrenaba el primero). Las configuraciones son tres y en este orden: 16.384 puntos con
±15°; 8.192 con ±15°; y la resolución ganadora con ±30°. Gana la resolución con menor
contacto p50 en `dev` entre las que cumplen K3; el candidato final es, de las tres, la de
menor contacto p50 entre las que cumplen K1-K6 en `dev` (`benchmark_separacion.py elige`).
La semilla extra que preveía el plan no se entrena, por tiempo.

Si C no los cumple todos en `dev`, la reserva no se toca y esto se cierra como negativo. La
reserva se evalúa **una vez** (el script se niega a repetirlo). Si cumple todo, C pasa a ser
el checkpoint por defecto de `segmentar_fdi.py`; si no, queda disponible y declarado.

Las anotaciones de las doctoras (14 curvas del maxilar del caso A) solo entran al final,
como falsación: dan la identidad de cada pieza pero no la frontera entre dientes que se
tocan, y ya se usaron para numerar.

**Resultado en `dev` (2026-11-20).** Los tres candidatos entrenaron 60 épocas con semilla 0
sobre las 1.080 arcadas y se midieron con `benchmark_separacion.py evalua --split dev` contra
la base, desde `6602771` con el árbol limpio:

| | Contacto p50 | Contactos > 1 mm | Sangran > 10 % | Correctas | Fantasmas | Falso diente | Falsa encía | FDI por vértice | K1-K6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B, base | 0,75 mm | 152 de 428 | 165 de 520 | 498 | 19 | 0,454 | 0,0070 | 0,884 | — |
| C1, 16.384 pts, ±15° | 0,22 mm | 6 de 437 | 22 | 506 | 15 | 0,257 | 0,0021 | 0,936 | falla K4 |
| **C2, 8.192 pts, ±15°** | **0,16 mm** | 11 de 438 | 19 | 508 | 13 | 0,283 | 0,0017 | 0,946 | **cumple** |
| C3, 8.192 pts, ±30° | 0,23 mm | 13 de 439 | 18 | 509 | 14 | 0,305 | 0,0015 | 0,945 | falla K4 |

Los tres mejoran el contacto en las 40 arcadas. C1 y C3 fallan solo K4, y por una pieza
cada uno: el voto se queda una pieza por debajo de lo que daría la pose nativa (506 frente a
507, 509 frente a 510), cuando con la base coincidían. El criterio pedía que esa distancia no
creciera y creció, así que quedan fuera, aunque sea por poco. Por la regla fijada, la
resolución ganadora fue 8.192 (0,16 frente a 0,22 mm) y el candidato final es **C2**. Que
8.192 puntos separe mejor que 16.384 contradice la hipótesis de partida, según la cual más
resolución sería mejor. Con una sola semilla no se puede saber si es una diferencia real o
ruido de entrenamiento, y no se ha investigado.

C2 pasó la puerta, así que la reserva se predijo con B y con C2 y se evaluó una vez.

**Resultado en la reserva (2026-11-20): negativo, por un criterio.** 120 arcadas de 60
pacientes del `private-testing-set`, evaluadas con `benchmark_separacion.py evalua --split
reserva` desde `6602771` con el árbol limpio (`run-record.json` junto a cada predicción):

| Criterio | B | C2 | |
|---|---:|---:|---|
| K1 contacto p50 | 0,86 mm | 0,17 mm; mejora en las 118 arcadas con contactos | cumple |
| K2 piezas que sangran > 10 % | 789 de 1.657 | 103 | cumple |
| K3 bien numeradas · fantasmas | 1.365 · 106 | 1.531 · 95 | cumple |
| K4 techo − voto | 42 | 14 | cumple |
| K5 falso diente · falsa encía | 0,358 · 0,065 | 0,255 · 0,019 | cumple |
| K5 piezas que **pasan** a < 85 % de corona | — | **10 de 1.657 (0,60 %)** | **no cumple** (límite 0,5 %) |
| K6 FDI por vértice de diente | 0,738 | 0,888 | cumple |

Cinco criterios y medio se cumplen con holgura. Lo que falla es la cláusula de daño de K5:
10 piezas que con la base conservaban al menos el 85 % de su corona bajan de ese umbral con
C2, cuando el límite era 8 (0,5 % de 1.657). El total de piezas por debajo del 85 % baja de
152 a 33, pero el criterio contaba solo las que empeoran, a propósito: un método que arregla
muchas piezas y estropea unas pocas no debe compensar unas con otras. Así que, por la regla
fijada, **C2 no pasa a ser el checkpoint por defecto**. Queda disponible y declarado
(`segmentar_fdi.py --checkpoint`), igual que `M9` y `ancla_al_pliegue`.

Dos cosas que se ven en la reserva y no estaban previstas. La base rinde mucho peor aquí que
en `dev`: el 82 % de las piezas bien numeradas frente al 96 %, y 74 de 120 arcadas con algún
fantasma. Son pacientes del test privado del reto, que nadie ha visto con datos públicos, y es
la primera medida de la base fuera de la distribución con la que se entrenó. Lo que C2 gana
también es mayor aquí: 166 piezas bien numeradas más, frente a 10 en `dev`. Y el margen
mejora sin tocarlo: la mediana pasa de 2,23 a 0,96 mm del experto, efecto secundario de las
1.080 arcadas y de los trozos más finos, no de ninguna regla de margen. No se ha separado qué
parte de la mejora viene de los trozos y qué parte del doble de datos. Para eso haría falta
entrenar 2.048 puntos con las 1.080 arcadas, y ese experimento no estaba en el protocolo.

**Falsación en los casos clínicos, sin ajustar nada.** Se segmentaron el maxilar del caso A
y las dos arcadas del caso B con B y con C2. Las 14 curvas de margen de las dos anotadoras del
caso A ya estaban gastadas, así que no se leen como validación:

- **Identidad, caso A.** Con C2 las 14 piezas anotadas se emparejan con su propio número y
  ninguna queda sin etiqueta. Con B se emparejaban por solape igual, pero a la 21 (y a la 17,
  según una de las anotaciones) no le quedaba etiqueta mayoritaria. C2 añade además una
  **18** de 109 mm² que ninguna anotadora marcó. No se sabe si existe esa pieza; si no existe,
  es un fantasma del tipo que K3 cuenta.
- **Margen, caso A.** Hay 3,53 y 3,64 mm de la línea con B, y 0,88 y 0,98 mm con C2, sin
  ninguna pieza por debajo del 85 % de corona. Entre las dos anotadoras hay unos 0,15 mm.
  Coincide en dirección con la reserva; no es una medida independiente.
- **Caso B**, sin anotación. Las piezas que superan el ancho anatómico pasan de 8 a 5 en el
  maxilar y de 2 a 1 en la mandíbula. C2 añade una **38** de 35 mm² en la mandíbula, de nuevo
  sin forma de saber si es real.

Nada de lo clínico contradice la reserva. La pega coincide con la que la reserva ya cuenta:
los fantasmas bajan poco (106 → 95) y en los dos casos aparece un tercer molar que nadie ha
confirmado.

**Lo que queda abierto.** (1) El daño de corona de K5: mirar las 10 piezas de la reserva no
vale para ajustar, porque la reserva ya está gastada; un intento nuevo necesita su propio
conjunto (quedan 240 pacientes del test privado sin bajar). (2) Distinguir resolución de
datos, como se dice arriba. (3) Una segunda semilla de C2, que el plan preveía y no se
entrenó. (4) Los fantasmas, que el reentrenamiento apenas toca.

**Decisión (2026-11-22): C2 pasa a ser el checkpoint por defecto de `segmentar_fdi.py`.** La
toma el responsable del proyecto a la vista de esta tabla. **No la dicta el protocolo**, que
decía lo contrario, y no se reescribe el resultado: C2 sigue sin cumplir K5. Lo que se acepta
es un intercambio declarado. Se ganan 166 piezas bien numeradas de 1.657, el contacto pasa de
0,86 a 0,17 mm y la mediana del margen de 2,23 a 0,96 mm. A cambio, 10 piezas que con la base
conservaban al menos el 85 % de su corona dejan de conservarlo. Las cifras son las del
pipeline medido, con `--referencias-entrenamiento 18`. El checkpoint de agosto sigue
disponible con `--checkpoint data/processed/a3-checkpoints/normales-ce-baseline.pt`, y los
bancos de pruebas (`benchmark_margen.py`, `benchmark_separacion.py`) siguen nombrando su base
explícitamente, así que ninguna cifra anterior cambia.

### 6.11 · La identidad de cada pieza: plan, criterios y resultado negativo (2026-11-20 · 2026-11-22)

**De dónde sale.** C2 (§6.10) separa bien los dientes, pero en la reserva deja 95
fantasmas y 103 piezas con otro número. Este apartado ataca eso, y solo eso. Se escribe y
se versiona antes de escribir el código y de entrenar nada.

**La reserva de §6.10 pasa a ser desarrollo.** Ya se evaluó una vez, así que no puede
volver a medir nada. Sí puede servir para elegir: son pacientes del test privado, donde el
modelo falla cuatro veces más que en `dev`. Aquí se la llama `dev-privado`: 120 arcadas,
1.657 piezas. `dev` está cerca del techo para la numeración (508 de 520 con C2) y no
distinguiría una mejora de ruido. Lo que se elija se mide en una **reserva nueva**: 60
pacientes del `private-testing-set` que no están en la primera, elegidos con semilla
20261121 por `fetch_teeth3ds_osf.py reserva-2` y versionados en
[`teeth3ds-reserva-2.txt`](teeth3ds-reserva-2.txt) en este commit, antes de bajar ninguno.

**Diagnóstico en `dev-privado`**, con C2 en el pipeline real (voto entre 3 de 18
referencias y la limpieza actual), mirado después de gastar esa reserva:

| Fallo | Cuántos | Cómo se reparten |
|---|---:|---|
| Fantasmas | 95 | **75 son un «8»**: un tercer molar puesto sobre la encía detrás del último diente. Mediana de 64 mm², tamaño de diente; solo 17 bajan de 30 mm² |
| Desplazadas | 103 | **84 están en 13 arcadas** con tres o más: hemiarcadas enteras corridas un puesto. 76 caen en la pieza vecina |
| Ausentes | 48 | piezas reales sin etiqueta propia (mediana de 120 mm²) |
| Coronas dañadas respecto a la base | 10 | **6 en una sola arcada**, que pierde el frente entero |

En `dev`, más pequeño, el patrón es el mismo: 13 fantasmas, y 10 de las 11 desplazadas son
de un solo paciente.

**Lo que este diagnóstico descarta.** Se pensó en un método de dos etapas que primero
detecta cada diente y después lo segmenta, porque daría mejores instancias. Se midió el
techo de esa idea: con las instancias **del experto**, perfectas, y el número por mayoría
de C2 dentro de cada una, salen 1.528 de 1.657 piezas bien numeradas, contra las 1.531 de
hoy. En `dev`, 510 frente a 508. **Las fronteras ya no son lo que numera mal**; el error
está en qué número cree el modelo que lleva cada pieza. Mejorar las instancias no lo arregla.

**La hipótesis.** El modelo de puntos ve cada pieza a 2 mm de resolución de trozo y decide su
número sobre todo por la posición en la arcada. Por eso, cuando falta un premolar, corre la
hemiarcada, y por eso pone un «8» en la encía detrás del 7. Un segundo modelo que mire **cada
pieza recortada, a resolución de malla**, puede ver lo que el primero no ve: si hay corona
(cúspides, esmalte convexo) o solo encía, y si es un premolar o un molar. Esta evidencia es
independiente de la posición. Lo que tumbó §6.5 fue que el ancho de la corona era demasiado
débil para vencer errores de alta confianza. Aquí la prueba es la misma: si la forma vista
de cerca no basta, sale negativo.

**Lo que se construye**, sobre las instancias de C2, sin tocar sus fronteras:

- **Recorte por pieza.** Los vértices de la instancia y los que quedan a menos de 2 mm de
  ella, hasta 4.096 puntos, con normales. Va centrado en la pieza y en el sistema de la pose
  votada, porque la orientación distingue un incisivo de un molar.
- **El modelo de pieza.** Un clasificador de nube de puntos con dos cabezas: `p(diente)` y
  `p(posición 1…8 dentro del cuadrante)`. El cuadrante lo sigue poniendo C2: 76 de las 103
  desplazadas caen en la vecina del mismo cuadrante. Se entrena con los 540 pacientes de
  entrenamiento de §6.10 y nunca con `dev`, `eval` ni ninguna reserva (lo comprueba
  `entrena_fdi_escaner.arcadas_de_entrenamiento`):
  - positivos: las piezas del experto, con la frontera dilatada o erosionada hasta 1 mm y
    hasta un 30 % de la pieza quitado, para parecerse a lo que entrega C2;
  - negativos: parches de encía del tamaño de un molar, por detrás de la última pieza y en
    otras zonas, más los fantasmas que dé C2 en entrenamiento.

  Para parar a tiempo se valida con el 10 % de los pacientes de entrenamiento. Una semilla.
- **E1, el verificador.** Antes de numerar, una instancia con `p(diente) < τ` pasa a encía.
- **E2, la renumeración.** Por cuadrante, las instancias se ordenan a lo largo de la arcada
  desde la línea media. Se elige la secuencia de posiciones estrictamente creciente, con
  huecos permitidos, que maximiza Σ [log p₂(posición) + w · log p₁(posición)]: p₂ es el
  modelo de pieza y p₁ la log-probabilidad media de C2 en la instancia.

**Elección, solo en `dev-privado`.** τ ∈ {0,3; 0,5; 0,7} y w ∈ {0; 0,5; 1}, como mucho
nueve combinaciones, sin más valores después de verlas:

- τ: el que deja menos fantasmas entre los que cumplen V1-V5; si empatan, el menor.
- w: el que deja menos desplazadas entre los que cumplen N1-N5; si empatan, el mayor.

E1 y E2 se juzgan **por separado**, cada uno con sus criterios. Si solo pasa uno, se adopta
ese. Si ninguno pasa en `dev-privado`, la reserva nueva no se toca y esto se cierra como
negativo.

**Criterios.** R es la referencia: C2 en el pipeline real, la misma ejecución que en §6.10.
N es el número de piezas del experto en el conjunto medido.

| | E1 · verificador |
|---|---|
| V1 | fantasmas ≤ 0,5 × R |
| V2 | ausentes ≤ R + 0,5 % de N: el verificador no borra dientes reales |
| V3 | piezas que pasan a < 85 % de corona respecto a R ≤ 0,5 % de N |
| V4 | bien numeradas ≥ R |
| V5 | en `dev`: bien numeradas ≥ R − 0,5 % de N, y fantasmas ≤ R |

| | E2 · renumeración |
|---|---|
| N1 | desplazadas ≤ 0,75 × R |
| N2 | bien numeradas ≥ R |
| N3 | piezas bien numeradas con R que pasan a mal numeradas ≤ 0,5 % de N: es el fallo de §6.5 |
| N4 | acierto FDI por vértice de diente ≥ R − 0,005 |
| N5 | en `dev`: bien numeradas ≥ R − 0,5 % de N |

Los dos juntos, si los dos pasan: la frontera entre dientes (contacto p50) ≤ R + 0,02 mm,
porque ninguno debería moverla.

**Una sola evaluación en la reserva nueva**, desde un commit con el árbol limpio y con su
`run-record.json`. El script se negará a repetirla, como en §6.10. Después, la falsación en los casos
clínicos sin ajustar nada: el caso A tiene que seguir con sus 14 piezas anotadas bien
numeradas, y se dirá qué pasa con la 18 que nadie marcó; el caso B se describe.

**Lo que este plan no ataca, dicho para que nadie lo suponga.**

- Las 48 ausentes: si una pieza real no tiene instancia propia, renumerar no se la da.
- La arcada que pierde el frente: es un fallo de C2 en una arcada entera, no de identidad.
- Que C2 pase a ser el checkpoint por defecto: falló K5 en §6.10 y eso no cambia aquí. Si E1
  o E2 pasan, se añaden al camino opt-in de C2; decidir qué es el defecto es del responsable
  del proyecto.

**Coste previsto.**

- Bajar la reserva nueva: ~1,5 GB y unas dos horas, por el límite de OSF.
- Entrenar el modelo de pieza: menos de una hora de GPU.
- Predecir C2 en la reserva nueva: ~1,2 h.
- Elegir y evaluar sobre las predicciones cacheadas: minutos.

**Resultado (2026-11-22): negativo en `dev-privado`, así que la reserva nueva no se ha tocado.**

Se construyó lo pre-registrado, sin cambiar ningún criterio:
[`analysis_agents.identidad`](../../packages/analysis-agents/src/analysis_agents/identidad.py),
[`scripts/entrena_pieza.py`](../../scripts/entrena_pieza.py) y
[`scripts/benchmark_identidad.py`](../../scripts/benchmark_identidad.py).

- **El modelo de pieza.** Se entrenó desde `b820698` con el árbol limpio, sobre las 1.080
  arcadas de entrenamiento:
  - 16.845 recortes para entrenar y 1.863 para validar;
  - 14.141 piezas del experto, 4.133 parches de encía y 434 fantasmas de C2 minados sobre
    esas mismas arcadas;
  - 20 épocas de 8.000 recortes, semilla 0, unos 20 minutos de GPU.

  Se quedó la época 19, la de menor pérdida de validación. En validación acierta si es
  diente en el 99,5 % de los recortes y la posición en el 91,7 %.
- **R, la referencia.** Sale de las predicciones cacheadas de §6.10, así que es idéntica a la
  de entonces. Al recalcular las tres referencias que votan para obtener p₁, 7 vértices de
  las 160 arcadas de `dev` y `dev-privado` salen con otra etiqueta. Es no-determinismo de la
  GPU y no afecta a R.

**Un error de implementación, encontrado en `dev-privado` y corregido** (`6afd214`). En el
maxilar, el orden desde la línea media salía invertido: el signo del eje anterior de
`marco_canonico` sirve para superponer arcadas, no es anatómico. En 30 arcadas había 39
cuadrantes al revés y 18 bien. La primera elección dio, por eso, una renumeración que
bajaba las piezas bien numeradas de 1.531 a 416. Ahora el extremo de la línea media se mide:
es el más cercano al cuadrante contrario. Las rejillas de τ y w no cambiaron. La elección con
el error se guarda junto a la buena (`eleccion-con-orden-invertido.json`). La tabla es la de
después de corregirlo, desde `6afd214` con el árbol limpio.

| E1 τ | Fantasmas | Ausentes | Coronas dañadas | Bien numeradas | V5 en `dev` | |
|---|---|---|---|---|---|---|
| R | 95 | 48 | — | 1.531 | 508 · 13 fantasmas | |
| 0,3 | **8** | 84 ✗ | 45 ✗ | 1.500 ✗ | 503 · 3 ✗ | no |
| 0,5 | 2 | 96 ✗ | 61 ✗ | 1.492 ✗ | 499 · 1 ✗ | no |
| 0,7 | 0 | 117 ✗ | 83 ✗ | 1.472 ✗ | 495 · 1 ✗ | no |

| E2 w | Desplazadas | Bien numeradas | Rotas | FDI por vértice | N5 en `dev` | |
|---|---|---|---|---|---|---|
| R | 103 | 1.531 | — | 0,888 | 508 | |
| 0 | 121 ✗ | 1.513 ✗ | 28 ✗ | 0,879 ✗ | 507 | no |
| 0,5 | 116 ✗ | 1.518 ✗ | 19 ✗ | 0,881 ✗ | 507 | no |
| 1 | 113 ✗ | 1.521 ✗ | 15 ✗ | 0,883 | 508 | no |

`dev-privado` tiene 120 arcadas y 1.657 piezas, así que el 0,5 % permite 8.

**Por qué falla el verificador.** Separa bien los fantasmas: la mediana de su `p(diente)` es
0,006. Pero con τ = 0,3 descarta 49 instancias que no son fantasmas, y 31 de ellas son
dientes bien numerados y de tamaño completo (mediana de 5.036 vértices). Un 36 de 6.646
vértices sale con 0,001. No es la pose ni el signo de las normales: en cuatro de esas
piezas, la pose nativa de Teeth3DS+ y las normales invertidas dan lo mismo, y el modelo
acierta su posición. Una explicación posible, no probada, es que los positivos se
entrenaron con la frontera del experto movida ±1 mm, y el borde de C2 desborda varios
milímetros sobre la encía, igual que los fantasmas.

**Por qué falla la renumeración.** La evidencia de la forma no es independiente de la de C2
donde haría falta. Frente a la posición del experto:
- en las 1.531 piezas que C2 numera bien, el modelo de pieza acierta 1.366 (89 %);
- en las 85 que C2 desplaza dentro del mismo cuadrante, solo 27 (32 %).

Ve lo mismo que C2 justo donde C2 se equivoca, y de las 13 hemiarcadas corridas E2 arregla 2.
Es el riesgo que este apartado declaró al empezar: si la forma vista de cerca no basta, sale
negativo. Sale negativo, igual que el ancho en §6.5.

**Lo que esto deja.**

- **La reserva nueva sigue sin gastar.** Se lanzó su predicción con C2 (solo etiquetas,
  reanudable, en la caché de `benchmark_separacion.py`) para que un intento posterior no
  tenga que repetirla. Nadie ha comparado esas etiquetas con las del experto, ni se ha
  ejecutado `c2`, `puntua` ni `evalua` sobre ella.
- **Los casos clínicos no se ejecutaron.** No hay ningún componente que aplicar.
- **C2 y el modelo por defecto no cambian.**
- **Abierto:**
  - un verificador entrenado con instancias de C2 emparejadas con el experto, en vez de la
    pieza del experto movida ±1 mm. Es una hipótesis nueva que necesita su propio
    pre-registro, y `dev-privado` ya se ha mirado para esta. Es §6.12;
  - la numeración corrida no se arregla con información local de la pieza. Haría falta algo
    que cuente piezas a lo largo de la arcada y vea el hueco de la que falta, no qué forma
    tiene cada una.

### 6.12 · El verificador, entrenado con los recortes de C2: plan, criterios y resultado negativo (2026-11-22)

**De dónde sale.** El verificador de §6.11 (E1) quitaba 87 de los 95 fantasmas con τ = 0,3,
pero también 31 dientes reales bien numerados, y las ausentes subían de 48 a 84. Este
apartado prueba la explicación que §6.11 dejó sin probar. Se escribe y se versiona antes de
tocar el código y de entrenar nada. Desde §6.10 C2 es además el modelo por defecto, así que
lo que se mide aquí mejora o no el camino por defecto.

**Diagnóstico, en `dev-privado`.** Es el conjunto de elección, y mirarlo es lo que le
corresponde. La pregunta: en las 1.531 piezas que C2 numera bien, ¿cuánto se sale su
instancia del diente del experto? Se mide con el percentil 95 de la distancia, al diente del
experto, de los vértices de C2 que caen fuera de él (medianas por grupo):

| Piezas bien numeradas | N | Desborde | Desborde > 1 mm | Parte sobre encía | Parte sobre el vecino |
|---|---:|---:|---:|---:|---:|
| Que E1 quitó (`p(diente)` < 0,3) | 31 | 6,3 mm | 87 % | 0,29 | 0,01 |
| Que E1 dejó (≥ 0,7) | 1.472 | 1,3 mm | 72 % | 0,15 | 0,01 |

- **El desborde es sobre la encía, no sobre el vecino**, y en las piezas quitadas es cinco
  veces mayor. El modelo de §6.11 solo vio fronteras movidas ±1 mm, así que una pieza real
  con 6 mm de encía pegada le pareció un fantasma.
- **5 de las 31 llevan un trozo suelto a más de 30 mm** del diente: la misma etiqueta
  aparece en otra parte de la arcada, y el recorte, centrado en la media, mira entre los dos.
- **6 de las 31 son fragmentos** que cubren menos del 20 % de su diente. Esto no los ataca.
- **En entrenamiento el desborde existe, pero su cola es corta.** En 216 de las 1.080
  arcadas de entrenamiento (una de cada cinco), con C2 en su pose nativa y la limpieza
  actual, la mediana es 1,15 mm y el percentil 90 es 2,5 mm. En `dev-privado` son 1,36 y
  5,8 mm. Entrenar con las instancias de C2 acerca la distribución, pero no llega a la cola.

**La hipótesis.** El verificador falló por lo que vio al entrenar y no por lo que puede ver.
Con recortes que desbordan como los de C2 debería distinguir una corona con encía pegada de
un parche de encía sin corona, que es lo que es un fantasma.

**Lo que se construye.** Solo cambia lo que se lista. La arquitectura, las 20 épocas de
8.000 recortes, el lote de 16, `lr` 1e-3, la rotación de ±15°, la semilla 0, el 10 % de
pacientes de validación y los 540 pacientes de entrenamiento son los de §6.11.

- **Positivos nuevos:** las instancias de C2 sobre las arcadas de entrenamiento, con la
  limpieza actual y con su propia frontera. Entran todas las que `identidad_de_piezas` no
  clasifica como fantasma, con la posición de la pieza del experto con la que se emparejan.
  Se mantienen los positivos del experto (±1 mm, hasta un 30 % quitado) y los negativos de
  §6.11 (parches de encía y fantasmas de C2).
- **Desborde sobre encía como aumento.** La mitad de las veces, la pieza de un positivo crece
  sobre los vértices de encía del experto que quedan a menos de *s* mm de ella, con *s*
  uniforme entre 0 y 13 mm. El 13 sale del diagnóstico: cubre el desborde de 26 de las 31
  piezas quitadas. De las otras 5, cuatro son de las del trozo suelto y una desborda 14,8 mm.
- **El recorte sale de la componente principal de la instancia**: la mayor componente
  conexa de sus vértices, unidos cuando están a menos de 1 mm. Es igual al entrenar y al
  inferir. La etiqueta no cambia: E1 sigue quitando o dejando la instancia entera.

**Elección, solo en `dev-privado`.** τ ∈ {0,3; 0,5; 0,7}, sin más valores después de verlos,
con la regla de §6.11: el que deja menos fantasmas entre los que cumplen V1-V5; si empatan,
el menor. E2 no se vuelve a evaluar: el modelo sigue aprendiendo la posición, pero no se usa.
**Si ningún τ pasa, la reserva nueva no se toca** y esto se cierra como negativo.

**Criterios.** Son los V1-V5 de §6.11, sin cambiar ninguno. R es C2 en el pipeline real, que
ahora es también el defecto. N es el número de piezas del experto en el conjunto medido.

| | E1 v2 · verificador |
|---|---|
| V1 | fantasmas ≤ 0,5 × R |
| V2 | ausentes ≤ R + 0,5 % de N |
| V3 | piezas que pasan a < 85 % de corona respecto a R ≤ 0,5 % de N |
| V4 | bien numeradas ≥ R |
| V5 | en `dev`: bien numeradas ≥ R − 0,5 % de N, y fantasmas ≤ R |

**Una sola evaluación en la reserva nueva**, de V1-V4, desde un commit con el árbol limpio y
con su `run-record.json`. El script se negará a repetirla. Con eso la reserva queda gastada
también para cualquier intento posterior. Quedan unos 180 pacientes del test privado sin
bajar. Después, la falsación en los casos clínicos, sin ajustar nada: el caso A tiene que
seguir con sus 14 piezas anotadas bien numeradas, y se dirá qué pasa con la 18 que nadie
marcó; el caso B se describe.

**Lo que este plan no ataca.**

- Las 103 desplazadas: un verificador decide si hay diente, no cuál es.
- Las 48 ausentes y los 6 fragmentos pequeños: quitar un fragmento no le da instancia a su
  diente.
- Las 10 coronas dañadas de K5.
- Qué es el defecto: si E1 v2 pasa, se añade como opción sobre C2, y adoptarlo por defecto
  lo decide el responsable del proyecto.

**Coste previsto.**

- Regiones de entrenamiento con radio de 15 mm: unos 4 GB, de los 20 GB libres.
- Entrenar: menos de una hora de GPU.
- Puntuar `dev`, `dev-privado` y la reserva nueva: minutos por conjunto. Antes hay que terminar
  la predicción de C2 en la reserva nueva, que ya está en marcha, y su paso `c2`.

**Resultado (2026-11-22): negativo en `dev-privado`, más cerca que §6.11. La reserva nueva no
se ha tocado.**

Se construyó lo pre-registrado, sin cambiar ningún criterio ni añadir valores de τ:
`entrena_pieza.py --version 2` y `benchmark_identidad.py --version 2`, con
`analysis_agents.identidad.componente_principal`. El modelo se entrenó desde `768a630` con el
árbol limpio, en 100 minutos de GPU; la mejor época fue la 18 (99,6 % de acierto en diente y
92,5 % en posición sobre validación). La elección se ejecutó desde `7279aed`, también limpio.

Dos cosas se apartan de lo escrito, ninguna en el método:

- **El desborde se calcula al vuelo** en cada recorte en vez de guardarse en regiones de
  15 mm: esas regiones no cabían ni en disco ni en RAM. El aumento es el mismo; lo que cambia
  es el coste, unos 80 ms por recorte desbordado.
- **`elige` falló la primera vez** antes de escribir nada, porque la regla de w recorría E2
  aunque la versión 2 no lo evalúa. Se corrigió en `7279aed` y se repitió; no había ningún
  resultado que ver.

| E1 v2 τ | Fantasmas (V1) | Ausentes (V2) | Coronas dañadas (V3) | Bien numeradas (V4) | V5 en `dev` | |
|---|---|---|---|---|---|---|
| R | 95 | 48 | — | 1.531 | 508 · 13 | |
| 0,3 | 18 | 59 ✗ | 13 ✗ | 1.525 ✗ | 508 · 6 | no |
| 0,5 | 14 | 62 ✗ | 14 ✗ | 1.523 ✗ | 508 · 1 | no |
| 0,7 | 11 | 62 ✗ | 15 ✗ | 1.523 ✗ | 507 · 1 | no |
| §6.11, τ = 0,3 | 8 | 84 | 45 | 1.500 | 503 · 3 | no |

Los límites, con N = 1.657: V2 permite 56 ausentes, V3 8 coronas y V4 exige 1.531. Ningún τ
pasa, así que por la regla fijada **la reserva nueva no se evalúa** y los casos clínicos no se
ejecutan.

**Lo que la hipótesis acertó.** Con τ = 0,3 el v2 ya no quita ningún diente entero bien
numerado. De las 31 de §6.11, 25 eran dientes enteros y no queda ninguna; las 6 bien
numeradas que quita ahora son fragmentos que cubren a lo sumo el 19 % de su diente, como los
otros 6 de entonces. Las ausentes pasan de 84 a 59 y las coronas
dañadas de 45 a 13. La explicación de §6.11 era correcta: el modelo fallaba por lo que vio al
entrenar.

**Lo que queda, mirado en `dev-privado` después de la elección.** Con τ = 0,3 quita 100
instancias: 77 fantasmas, 6 fragmentos bien numerados y 17 desplazadas. De esas 17:

- 11 son fragmentos que cubren menos del 25 % del diente al que caen. Al quitarlos, ese
  diente se queda sin etiqueta que lo empareje cuando su cuerpo lleva otro número, y cuenta
  como ausente.
- 6 son **dientes enteros** (cubren del 55 % al 95 %) que C2 numeró con el código del otro
  lado o del cuadrante opuesto: un 46 llamado 36, un 41 llamado 31, un 37 llamado 48. El
  modelo les da `p(diente)` entre 0,01 y 0,14. No se ha mirado por qué; la sospecha, sin
  probar, es que esas instancias montan sobre dos piezas a ambos lados de la línea media o
  del cuadrante.

Las 11 ausentes nuevas salen de esos dos grupos.

**Lo que esto deja.**

- **La reserva nueva sigue sin gastar.** La predicción de C2 sobre ella está terminada y en
  caché (solo etiquetas). No se ha ejecutado `c2`, `puntua` ni `evalua` sobre ella.
- **El defecto no cambia**: C2 sin verificador.
- **Abierto:** que el verificador no quite una instancia que cubre la mayor parte de un diente
  real, y que los fragmentos se fundan con su pieza en vez de borrarse. Las dos cosas
  necesitan su propio pre-registro, y `dev-privado` ya se ha mirado para esta.

## 7 · Consecuencia sobre el producto

`derived/seg_teeth.bin` sale con `regulatory.layer: 3` y `status: investigational`, que
es exactamente lo que es. El contenedor **sigue siendo válido sin él**: borrar `derived/`
y sus entradas del manifiesto es una operación soportada, y lo que queda —escaneo, CBCT,
campo medido, informe, registro, procedencia— no depende de esta etapa.

Es decir: la segmentación es la pieza no resuelta, y el formato está construido para que
no resuelta signifique *separable*, no *contaminante*.

Desde el 2026-11-22 las etiquetas por defecto salen de C2 (§6.10), por decisión del responsable
y no por haber pasado los criterios. Eso no cambia la capa ni el estado: siguen siendo
inferencia de un modelo, en `derived/`, `investigational`.
