# `exercise-point-transformer-teeth3ds.ipynb` — Segmentación de dientes (FDI) por punto

Prototipo del **`segmentation-agent`**: dado el escaneo intraoral como nube de puntos,
etiquetar **cada punto con su diente (FDI)** o encía. Reproduce el modelo y el bucle del
[ejemplo oficial de Point Transformer (PyG)](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_segmentation.py);
lo propio es el *loader* (mallas `.obj` + labels `.json` → PyG), la **loss ponderada por
clase**, el diagnóstico **`tooth_acc`**, y la agregación **punto → diente** de A4.

> Es un **spike de validación técnica**, no el sistema final ni un resultado clínico: sirve para
> de-arriesgar una decisión de arquitectura (**¿cómo entra el CBCT en el sistema?**).

---

## El experimento

| | |
|---|---|
| **Tarea** | segmentación semántica por punto — 32 clases (encía + 31 FDI) |
| **Datos** | **Teeth3DS+ completo**: 600 mallas / 300 pacientes (`data/raw/teeth3ds/`, gitignored) |
| **Muestra** | ~117k vértices/malla → submuestreo a **2048 puntos** en entrenamiento |
| **Split** | **por paciente** 240 train / 60 test (80/20, `seed 42`, sin fuga) |
| **Modelo** | Point Transformer de segmentación, 4,59M parámetros |
| **Semillas** | **3 por configuración** — se reporta **media ± desviación** |
| **Loss** | `nll_loss` **ponderada por clase** (*median-frequency balancing*, peso topado a 50) |
| **Épocas / HW** | 20 · RTX 5070 (sm_120), torch cu128 · ~45 min el notebook completo |

### Por qué multi-semilla

Es la decisión metodológica que sostiene todo lo demás. La varianza entre corridas
(no-determinismo GPU + submuestreo aleatorio) **es del orden de los efectos que queremos medir**,
así que con una sola semilla el **signo** de las diferencias pequeñas no es reproducible. Ocurrió:
corridas sucesivas del mismo código dieron a A3 deltas de signo opuesto (ver §A3). No eran
resultados en conflicto, eran muestras de una distribución centrada cerca de cero.

Los veredictos del notebook se **derivan** comparando cada diferencia con la dispersión observada,
en vez de afirmarse por escrito.

### Por qué la loss ponderada

Sin pesos, la **encía** (clase 0, **44%** de los puntos) domina el gradiente y el modelo colapsa
prediciéndola en todo. Con frecuencia inversa un diente pesa **~147×** más que la encía.

### Por qué `tooth_acc` (y no `mIoU`)

- **`tooth_acc`** = acierto **solo** en los puntos de diente. Es el número honesto.
- **`mIoU`** queda **inflado** por la convención «parte ausente en la muestra → IoU 1.0».

---

## Resultados — A2 · ablación de features

**Pregunta:** ¿qué feature por punto arregla la generalización, y aporta algo el gris del CBCT?

| config | in_ch | train | **test** (media ± desv, 3 semillas) |
|---|---|---|---|
| pos-only (geometría) | 1 | 0.773 | **0.133 ± 0.094** |
| **normales** | 3 | 0.815 | **0.748 ± 0.059** |
| gris CBCT sintético | 1 | 0.703 | **0.649 ± 0.044** |
| normal + gris | 4 | 0.820 | **0.769 ± 0.049** |

- **`pos-only` no transfiere.** Con input por punto constante el modelo se apoya en la **posición
  absoluta**, que cambia de una arcada a otra. Memoriza el train (0.773) y no generaliza. Es lo
  único de A2 que aguanta en todas las corridas sin matices.
- **Un descriptor LOCAL por punto rompe la degeneración** → **normales, 0.748**.
- **El gris no aporta señal medible**: +0.021 sobre las normales, por debajo de la dispersión
  (±0.059).

> **El gris NO resta — corrección.** Una versión anterior de esta ficha afirmaba que añadir el
> gris **restaba**, apoyándose en dos corridas (−0.066 y −0.053). La tercera dio **+0.021**: el
> signo **cambia entre ejecuciones**, así que lo único defendible es que **no aporta señal
> independiente**. Que un gris derivado de la propia malla no añada nada es lo esperable —
> afirmar que perjudica activamente era ir más allá del dato.

### Por qué el eje real es «descriptor local», no «qué feature»

Un **descriptor local** es un valor asociado a un punto que resume **cómo es la superficie a su
alrededor**, calculado solo desde su vecindario. Su propiedad definitoria es que **no depende de la
posición**: dos puntos con forma parecida alrededor reciben valores parecidos, estén donde estén.
`(x, y, z)` no es un descriptor — no dice nada de la forma, solo de la ubicación.

Con esa lente, las tres configuraciones dejan de ser tres features sueltas y forman una escala:

| config | ¿descriptor local? | resultado |
|---|---|---|
| `pos-only` | **no**, solo posición | colapsa |
| `normales` | sí — orientación de la superficie | funciona |
| `gris sintético` | **sí, sin pretenderlo** | funciona casi igual |

El gris se fabrica **voxelizando la propia malla y desenfocando** (`_synth_volume`), así que lo que
acaba midiendo es **grosor y densidad local de la superficie**: es un descriptor local con otro
envoltorio. Eso explica de golpe los dos resultados que si no parecen raros:

- **por qué el gris SOLO ya funciona** (0.649, cerca de las normales) — es un descriptor local válido;
- **por qué sumado a las normales no aporta** — ambos describen la **misma geometría local**, así que
  el segundo canal es información que el modelo ya tenía.

> **La conclusión de A2, en una frase:** *cualquier descriptor local decente sirve; lo que no sirve es
> no tener ninguno.* Las normales se adoptan por ser **las más simples**, no por ser mejores.
>
> Esto también fija el listón para el CBCT real, y es más alto de lo que suele asumirse: no basta con
> que «aporte información». Tiene que aportar algo que **no sea geometría local de la superficie** —
> es decir, la **densidad interna** (esmalte/dentina/hueso), que un muestreo en la superficie
> descarta por construcción. El mismo concepto reaparece en el registro de DDMF, que empareja
> modalidades con **FPFH**, otro descriptor local.

---

## Resultados — A3 · boundary + centroid loss (estilo IOSNet/DDMF)

Dos pérdidas extra que atacan el **borde diente-encía**, sobre la mejor config (normales):

| | `tooth_acc` | **frontera** |
|---|---|---|
| normales · CE | 0.708 ± 0.131 | 0.603 ± 0.104 |
| + boundary + centroid | 0.593 ± 0.162 | 0.519 ± 0.143 |
| **Δ** | **−0.116** (disp. ±0.162) | **−0.084** (disp. ±0.143) |

**Ambas diferencias caen dentro del ruido entre semillas.** No hay efecto medible.

> Esto **corrige una conclusión anterior** de este documento, que reportaba `+0.034` global y
> `+0.051` en frontera a partir de una única corrida.
>
> El histórico deja poco margen de duda: en cinco ejecuciones del mismo código el Δ global ha
> salido `+0.034`, `−0.048`, `−0.035`, `+0.017` y `−0.116` — **cambia de signo** entre corridas.
> No son resultados en conflicto: son muestras de una distribución centrada cerca de cero.
>
> **Atención a las dispersiones de esta corrida** (±0.131 y ±0.162, frente a ±0.03–0.06 en las
> anteriores): no es solo que el efecto sea pequeño, es que **el entrenamiento en sí es
> inestable** a 20 épocas. Ver §A4.
>
> Matiz importante: **no es que las pérdidas no funcionen** — es que a esta escala (20 épocas,
> 2048 puntos, 3 semillas) su efecto es **indistinguible del ruido**. Es un resultado sobre nuestra
> capacidad de medir, no sobre la técnica.

---

## Resultados — A4 · de puntos a DIENTES

A2/A3 miden acierto **por punto** sobre 2048 puntos. A4 cierra los cinco huecos que quedan entre
eso y un agente, **reutilizando el modelo de A3 sin reentrenar**:

1. **Inferencia densa** sobre los ~117k vértices (la malla se parte en trozos de 2048 — mismo
   tamaño y distribución que en entrenamiento — y se acumulan log-probs en 3 pasadas).
2. **Semántica → instancia**: componentes conexas del grafo kNN restringido a aristas con la misma
   etiqueta predicha.
3. **Métrica por diente**: voto mayoritario sobre los vértices del diente real.
4. **Unicidad FDI**: asignación **húngara** instancia→FDI (maximiza la confianza total con la
   condición de que ningún FDI se repita), precedida de **fusión de fragmentos**.
5. **Desglose** por arcada y por dentición completa/incompleta + taxonomía del error.

Todo el análisis se corre para **los dos modelos de A3**: como A3 no consigue distinguirlos,
reportar solo uno confundiría el método con la lotería de la semilla.

### Reparto de papeles — qué hace la red y qué hace la agregación

Conviene tenerlo claro porque marca la frontera entre el modelo y el agente: **el Point Transformer
solo hace la mitad**.

| | Point Transformer | agregación (`packages/tooth-aggregation`) |
|---|---|---|
| qué asigna a cada punto | una **categoría** (una de las 32 clases) | una **pertenencia a objeto** |
| «este punto es un 36» | ✅ | |
| «estos puntos son **el mismo** 36» | | ✅ |
| aprendizaje | red de 4,59M parámetros | ninguno: geometría y combinatoria |

La red hace segmentación **semántica**, no **de instancia**. Si pinta dos manchas separadas como
`36`, no tiene forma de expresar si son un diente partido en dos o dos dientes distintos: esa
pregunta no existe en su salida. El dataset trae `instances` en el nombre de la carpeta de etiquetas,
pero aquí solo se lee `labels`, la parte semántica.

**Pero la agregación no descubre los dientes por su cuenta.** Usa las etiquetas de la red como
criterio de corte: construye el grafo kNN y **elimina las aristas cuyos extremos tienen etiqueta
distinta**; lo que queda conectado es un objeto. Es decir, **la red dibuja las fronteras y la
agregación las lee como objetos**. Un error de la red en el borde entre el `36` y el `37` se hereda
tal cual — la agregación no mira la geometría por su cuenta, solo la conectividad.

Y el **nombre** de cada instancia sale del voto mayoritario de las etiquetas de la red dentro del
grupo. Así que la agregación decide **la agrupación**, no el código. Lo único que puede reescribir
nombres es la **húngara**, y precisamente por eso viene desactivada por defecto: al tocar los
códigos con instancias fragmentadas hacía más daño que bien.

> En una frase: **la red dice qué es cada punto; la agregación dice quién va con quién.**

### El número que importa

| modelo de A3 | por **punto** | **FDI por DIENTE** | mejor con fusión |
|---|---|---|---|
| normales · CE | 0.849 | **0.932** (1478/1585) | 0.933 |
| normales · CE+boundary+centroid | 0.473 | **0.490** (776/1585) | 0.486 |
| **spread** | 0.376 | **0.443** | 0.447 |

- **El acierto por punto subestima la identificación por diente** (0.85 vs 0.93): son métricas
  distintas y la ficha del agente debe declarar la segunda, no la primera. Esto sí se repite en
  todas las corridas.
- **Qué valida y qué no ese 0.932.** Se calcula sobre los vértices del **diente real**
  (`pred[y == c]`), así que responde a «dada la extensión verdadera del diente, ¿lo nombra bien?».
  Mide **la mitad de nombrar**, no la de **delimitar**: no evalúa si la detección de instancias
  encontró los dientes correctos. El otro número lo recuerda — quedan **15,8 instancias por arcada
  frente a 13,2 dientes reales**. Es la pregunta correcta para de-arriesgar la decisión de
  arquitectura, pero en producción, sin etiqueta real, la delimitación también contará.
- **El modelo `CE+boundary+centroid` de esta corrida está DEGENERADO**, no es una variante peor:
  falla en distinguir maxilar de mandíbula. Acierta 0.789 en `upper` pero **0.193 en `lower`**, y
  sus confusiones dominantes son `42→22`, `44→24`, `41→21`, `43→23` — mapea sistemáticamente la
  arcada inferior a códigos de la superior. Su 0.490 no mide la agregación, mide un modelo roto.

> **Corrección: el *spread* NO acota la lotería de semilla.** Una versión anterior de esta ficha
> concluyó, con un *spread* de 0.001, que la cifra por diente era «atribuible al método, no a qué
> variante o semilla tocó». Esta corrida da **0.443**. Lo que la comparación mide en realidad es
> **la estabilidad del entrenamiento**, y el veredicto es el contrario: con 20 épocas una semilla
> puede producir un modelo que nunca aprende a separar arcadas.
>
> Lo que sí sobrevive: **dado un modelo sano**, la agregación punto→diente rinde de forma
> consistente — 0.914, 0.956 y 0.932 en tres corridas. La inestabilidad está en el entrenamiento,
> no en el pipeline de A4.

### Fusión de fragmentos + unicidad FDI

La restricción «un FDI por arcada» presupone **una instancia = un diente**, y la detección parte
dientes en fragmentos. Se fusionan antes de repartir (componentes de la misma clase predicha cuyos
puntos más cercanos distan menos de `mult` × el espaciado típico de la malla):

Barrido del umbral (modelo `normales · CE`; el otro da la misma forma). **13,2 dientes reales
por arcada** es la referencia contra la que hay que leer la columna de instancias:

| umbral | inst/arcada | FDI duplicados | arcadas afectadas | **FDI/diente** |
|---|---|---|---|---|
| 0 (sin fusión) | 18.3 | 313 | 80/120 | 0.873 |
| 3× | 16.9 | 152 | 63/120 | 0.902 |
| 6× | 16.6 | 110 | 58/120 | 0.913 |
| 12× | 16.3 | 75 | 46/120 | 0.922 |
| 24× | 16.0 | 37 | 29/120 | 0.930 |
| **48×** | **15.8** | **12** | **11/120** | **0.933** |

- **La fusión funciona, y el diagnóstico era correcto**: el problema no era la húngara, eran los
  fragmentos. De 0× a 48× el acierto sube +0.061 y los FDI duplicados se desploman de **313 a 12**
  (de 80 arcadas afectadas a 11). Este `+0.06` se repite en ambos modelos, también en el degenerado.
- **La brecha se cierra**: 0.933 frente al voto mayoritario 0.932 → **+0.001**. Empatan. En la
  corrida anterior salió −0.001; el signo es ruido, la conclusión es que **igualan**.
- **Sigue en el borde del rango.** 48× es el mejor valor probado y la curva aún no se ha aplanado;
  no sabemos si el óptimo está más allá o si a partir de ahí empieza a fusionar dientes distintos.
  Quedan 14,5 instancias por arcada frente a 13,2 reales, así que **algo de fragmentación
  persiste**. El código avisa explícitamente cuando el mejor umbral cae en el extremo.

### En qué se equivoca

Solo del modelo sano (`CE`); el degenerado tiene otro perfil de error, ver arriba.

| tipo | n | % |
|---|---|---|
| **vecino** (16→15/17) | 66 | 62% |
| **otra arcada** (maxilar↔mandíbula) | 40 | 37% |
| espejo | 1 | 1% |

Top confusiones: `36→37` (5), `46→47` (5), `17→16` (5), `46→45` (4), `31→11` (4), `44→24` (4).

| desglose | | |
|---|---|---|
| arcada | upper **0.958** | lower **0.907** |
| dentición | completa ≥14 **0.975** | incompleta <14 **0.856** |

- El error dominante es el **desplazamiento del conteo posicional** (vecino, 62%), y la
  **dentición incompleta penaliza ~12 puntos** (0.856 vs 0.975) — consistente con que la
  numeración FDI se apoya en contar posiciones y se descoloca cuando faltan piezas. **Es el
  hallazgo más accionable de A4**, y el que mejor se repite entre corridas: dice dónde vigilar.
- La proporción de confusiones **entre arcadas oscila mucho** entre ejecuciones (14% y 37% en dos
  corridas del modelo sano): es un fallo poco frecuente en absoluto, así que su porcentaje sobre
  el total de errores es inestable.

#### Por qué el fallo es el vecino: FDI es ordinal, no morfológico

**Dónde ocurre.** No es un error en el *borde* entre dos dientes: es que **un diente entero recibe
el número de su vecino**. El modelo lo detecta y lo delimita bien, y lo llama `45` cuando era `46`.
La figura de A5 lo enseña en crudo — toda la arcada correcta y **un solo diente** marcado
`45 (era 46)`.

**Por qué.** El segundo dígito FDI es *cuántas posiciones llevas contando desde la línea media*
(1 incisivo central … 8 cordal). Para decir «esto es un 46» hacen falta dos cosas: el **cuadrante**,
y que sea **el sexto contando desde el centro**. Lo segundo es **contar**, y contar exige ver la
secuencia entera.

Pero el modelo mira **vecindarios locales** — es lo que le dan las normales, y es justo lo que le
permite generalizar entre pacientes (ver §A2). El precio es que **un primer molar y un segundo molar
son casi idénticos de forma**: localmente no hay nada que distinga un `46` de un `47`. El ordinal
tiene que deducirlo de algo parecido a «cuán avanzado voy por el arco», y si esa estimación se
desplaza **una casilla**, el número se desplaza una casilla. De ahí que el error sea casi siempre el
**vecino inmediato** y casi nunca un diente lejano.

Es el mismo mecanismo que la simetría entre arcadas (abajo), visto en el otro eje: **el modelo
acierta *qué* es cada diente y falla en *dónde* está en la secuencia.**

**Por qué empeora con piezas ausentes.** Si falta un diente, la referencia del conteo se rompe: lo
que aprendió como «la sexta mancha del arco» pasa a ser la séptima pieza real. Eso es lo que mide el
desglose (0.856 vs 0.975), y encaja con que en una corrida la confusión estrella fuera `37→38` —
segundo molar contra cordal, la posición donde la presencia del cordal es más variable.

> **Consecuencia práctica.** Un escaneo intraoral **no puede ver** un diente ausente ni uno incluido
> sin erupcionar: solo captura superficie visible en boca, así que la referencia para contar **no
> está en el dato**. Cualquier mejora sustancial de la numeración pasa por darle al modelo contexto
> global de la arcada, no por afinar la geometría local. Ahí es donde una fuente que vea la dentición
> completa —panorámica, CBCT, o el propio informe clínico (`fdi-consistency-agent`, ADR 003)— aporta
> algo que la malla no tiene.

#### El error entre arcadas es una simetría, no ruido

La **matriz de confusión por diente** del notebook (un panel por modelo, escala `log1p`) enseña algo
que las listas de confusiones no dejan ver como patrón: además de la diagonal principal y de las
celdas contiguas —los errores de vecino— aparece **una segunda diagonal paralela y desplazada**.

Corresponde a confusiones del tipo `31→11`, `42→22`, `44→24`, en las que **la posición se conserva y
solo cambia el cuadrante** (3→1, 4→2). Es decir: el modelo identifica correctamente **qué** diente es
y se equivoca en **dónde** está. Eso apunta a que lo que le falta es la **orientación global de la
arcada**, no la forma local del diente — coherente con que las features sean descriptores locales,
que por construcción son invariantes a la posición.

**Y es el diagnóstico visual del modelo degenerado.** En el panel del modelo sano esa segunda
diagonal es tenue; en el del `CE+boundary+centroid` de esta corrida sale **casi tan brillante como la
principal**. Ahí se ve de un vistazo lo que las cifras dicen por separado (0.193 en `lower` frente a
0.789 en `upper`): el modelo aprendió a numerar dientes pero nunca aprendió en qué mandíbula estaba.

---

## Resultados — A5 · vista cualitativa

Dos arcadas de test con el código FDI escrito sobre cada diente (real vs predicho, vista oclusal
por PCA). Se eligen **una arcada perfecta y la primera con algún fallo**, no solo el caso bonito.
La figura queda embebida en el `.ipynb` y se escribe también a `data/processed/fdi-preview/`.

Lo que enseña y las tablas no cuentan:

- El fallo típico es **un diente corrido una posición** (`45` donde iba `46`), con el resto de la
  arcada intacto. No es un modelo confundido: es un desplazamiento local del conteo.
- La predicción **«engorda» los dientes hacia la encía**: los límites diente-encía predichos son
  más generosos que los reales. Eso no penaliza en `tooth_acc` (que solo mira puntos de diente)
  pero sí importaría para cualquier medición de volumen o de margen gingival.

---

## Conclusiones

1. **La palanca de generalización es un descriptor geométrico LOCAL por punto** (normales). No es
   «más datos» ni «el CBCT». `pos-only` se apoya en posición absoluta y no transfiere.

2. **El gris por vértice no es la vía del CBCT — y aquí incluso restó.** Con un gris *sintético*
   (proxy geométrico derivado de la propia malla) el canal extra no aportó señal medible frente a normales
   solas. Dos matices que hay que decir siempre: es un gris sintético, así que **no valida ni
   refuta el CBCT real**; y muestrear un volumen en las posiciones de los vértices **exige registro
   CBCT↔malla igual que la fusión**, así que no ahorra el bloqueo de no tener pares CBCT+IOS.
   El listón para el CBCT real es **superar a normales-solo** aportando la **densidad interna**
   (esmalte/dentina/hueso) que el muestreo en superficie descarta por construcción.

3. **El valor del CBCT se realiza por FUSIÓN.** El paper líder —**DDMF**
   ([arXiv:2203.05784](https://arxiv.org/abs/2203.05784), *Patterns* 2023, **503 pacientes CBCT+IOS
   emparejados**)— segmenta cada modalidad por separado y las **fusiona por registro**
   (RANSAC-FPFH + ICP multiescala → **0.17 mm** ASSD); el gris nunca es feature de vértice.
   Además la fusión **se descompone**: su rama CBCT puede desarrollarse con datos no emparejados,
   cosa que el gris por vértice no permite. → el `cbct-agent` va por segmentación propia + registro.

4. **El `segmentation-agent` es construible YA, solo con geometría** — 0.91-0.96 por diente, sin
   dependencia del CBCT ni de datos emparejados. Es el desbloqueo práctico del experimento.

5. **La receta de IOSNet (boundary+centroid) no da efecto medible a esta escala.**

### Caveats metodológicos

- **3 semillas** es poco para estimar dispersión: los ± son orientativos.
- **`pos-only`** oscila `[0.000, 0.310, 0.038]` → de esa fila solo vale lo cualitativo.
- **`mIoU` no informativo** aquí (convención parte-ausente → 1.0).
- **Un solo dataset y un solo tipo de escáner**: nada de esto dice cómo generaliza a otro escáner
  ni contra etiquetas clínicas reales.
- **El entrenamiento es inestable a 20 épocas**, y no es un caveat teórico: una semilla produjo
  un modelo que nunca aprendió a separar maxilar de mandíbula (0.193 de acierto en `lower`).
  Cualquier uso serio necesita más épocas, o descartar modelos degenerados con un control
  automático — p.ej. exigir un mínimo de acierto por arcada antes de dar el modelo por bueno.
- **No hay comparación contra DDMF/IOSNet en el mismo split**, así que el experimento dice si la
  geometría basta, no si el modelo es bueno en absoluto.

---

## Cómo reproducirlo

Requiere el **kernel GPU dedicado** `Dental GPU (3DGS)` (torch cu128 + `pyg-lib`, más `scipy`) —
ver [`README.md` §04](README.md). Necesita el dataset en `data/raw/teeth3ds/` (gitignored).

```bash
~/.venvs/dental-gpu/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/exercise-point-transformer-teeth3ds.ipynb
# seguir el progreso de un run largo (nbconvert no vuelca nada hasta el final):
tail -f notebooks/.run-progress.log
```

---

## Conexión con el proyecto

- Es el **esqueleto del `segmentation-agent`**. Contrato de entrada: nube de puntos + **descriptor
  local por punto** (normales), con guardarraíl explícito de **nunca posición absoluta sola**.
  Contrato de salida: **instancias con FDI**, y métrica declarada **FDI por diente**.
- Da la **evidencia medida** de que el CBCT entra por **fusión**; el canal de gris queda
  **swap-ready** en el código para probarlo con volumen real en cuanto haya pares.
- **Dónde poner al humano (HIL):** el error se concentra en el **desplazamiento posicional** y en
  la **dentición incompleta** (−0.10). Cruzar el FDI predicho contra el informe clínico es el
  `fdi-consistency-agent` (ADR 003).
- **Aviso de implementación:** imponer unicidad FDI con asignación húngara **sin fusionar
  fragmentos antes** empeora el resultado (0.873 vs 0.932); con fusión agresiva (48×) empatan.

### Siguiente paso — conversión a `segmentation-agent` (valor de futuro)

El desbloqueo práctico de este spike es que **el agente es construible ya, solo con geometría**:
sin dependencia del `cbct-agent` ni de datos emparejados. Lo que este notebook deja listo para
esa conversión, y que **no** hay que volver a decidir:

- **Entrada**: nube de puntos + normales por vértice (`vtkPolyDataNormals`, sin splitting, ya en
  el *loader*). Guardarraíl: **nunca posición absoluta sola**.
- **Salida**: **instancias con FDI**, vía componentes conexas + voto mayoritario. No etiquetas
  por punto.
- **Métrica a declarar en la ficha del agente**: **FDI por diente** (0.91-0.96 según corrida), no `tooth_acc`.
- **Loss**: ponderada por clase, no opcional (encía al 44%).
- **Enrutado a HIL**: zona molar y arcadas con dentición incompleta, que concentran el error.
- **Aviso**: no imponer unicidad FDI sin fusionar fragmentos antes.

Lo que **queda fuera** del alcance medido aquí y habrá que resolver en el agente: `Provenance`,
fail-loud, reversibilidad, tests, y la ficha en `AGENTS.md`.

**Estado.** A1 (split real) · A2 (ablación, multi-semilla) · A3 (boundary+centroid: sin efecto
medible) · A4 (punto→diente, fusión + unicidad, barrido hasta 48× sobre los dos modelos) ·
A5 (vista cualitativa) ✅. **Experimento cerrado.** Los pesos de A3 quedan en
`data/processed/a3-checkpoints/` (gitignored), así que iterar sobre A4/A5 ya no exige reentrenar.

Sin acotar, y ya sin impacto en las conclusiones: (a) el umbral de fusión óptimo sigue cayendo en
el extremo del rango probado (48×); (b) **la estabilidad del entrenamiento**, que es el hallazgo
que este último run destapó y el primer problema a resolver si se retoma el modelo.

---

## A6 · El modelo depende de la POSE del fichero — hallazgo al aplicarlo fuera de Teeth3DS+

Al llevar el checkpoint de A3 a un escáner propio (`histora`), el modelo etiquetó una **arcada
mandibular con códigos de maxilar**: `17`, `27`, `28`, `11`, `21`. El mismo síntoma que A4
describe como «modelo degenerado», pero **con el modelo sano**.

**No era el modelo, era la pose.** Dos controles lo separan:

| control | resultado |
|---|---|
| el checkpoint sobre 4 inferiores de Teeth3DS+ **con etiquetas** | `tooth_acc` 0.80 / 0.84 / 0.93 / 0.94 — sano |
| eje oclusal de Teeth3DS+ frente al de `histora` | **coseno 0.004** — perpendiculares |

Teeth3DS+ escribe el eje oclusal en **+z** y este escáner en **+y**. El *loader* aplica
`NormalizeScale`, que centra y escala pero **no rota**, así que la orientación absoluta del
fichero llega intacta al modelo. Y A2 ya lo había medido sin sacar esta consecuencia: `pos-only`
colapsa porque el modelo se apoya en la posición absoluta. Las normales arreglan el *descriptor*,
pero **la geometría de la atención sigue viviendo en las coordenadas del fichero**.

### El arreglo

Canonizar la pose antes de inferir, con `fusion_agents.marco`: eje oclusal de `marco_arcada`,
anterior por el signo de la parábola del arco, y lateral **derivado** (`x = y × z`) — elegirlo
por separado espejaría la arcada y cambiaría los `3x` por los `4x`, que es un error que no avisa.

Sobre `histora` inferior, con el mismo checkpoint y sin reentrenar:

| | FDI a lo largo del arco | monotonía |
|---|---|---|
| sin canonizar | `17, 17, 36, 28, 27, 11, 21` | — (10 de 16 de la arcada equivocada) |
| **canonizado** | `47, 48, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38` | **ρ +1.00** |

Se añade además **decodificación restringida**: enmascarar los códigos de la arcada contraria
antes de agregar. La arcada no es una hipótesis que el modelo deba resolver —viene en el fichero—,
y dejarle emitirlos es darle libertad para fallar en algo que no estaba en duda. Bajó las
instancias de 23 a 16.

> ⚠️ **La canonización tiene su propio suelo.** `marco_arcada` devuelve una razón de orientación
> y hay que mirarla: en la autocomprobación sobre Teeth3DS+, una malla con razón **0.61** se
> hundió de 0.808 a **0.197**, mientras que las de 0.40-0.43 quedaron intactas. Canonizar una
> arcada cuyo marco es dudoso es peor que no canonizarla.

**Consecuencia para el agente:** la ficha del `segmentation-agent` tiene que declarar que su
entrada es una arcada **en pose canónica**, no una malla cualquiera. Sin eso, el 0.93 de FDI por
diente no significa nada fuera del dataset con el que se midió. Implementado en
[`scripts/segmentar_fdi.py`](../scripts/segmentar_fdi.py); el uso posterior de las etiquetas está
en [`registro-por-diente-histora.md`](registro-por-diente-histora.md).

## A7 · La FRONTERA — lo que ninguna métrica de aquí estaba midiendo

Todo lo anterior mide **identificación**: ¿el código mayoritario de cada diente es el
correcto? Nada mide **dónde acaba un diente y empieza el de al lado**. Y son cosas
distintas: una etiqueta que se pasa dos milímetros al vecino no cambia una mayoría, así
que puede convivir con 0,96 de FDI por diente sin que ningún número de esta ficha se
mueva. Es exactamente lo que un clínico ve al encender una pieza en el visor y arrastra
un trozo de la contigua.

La medida que sí lo ve es el **ancho mesiodistal** de cada corona etiquetada contra la
tabla anatómica (`export_agents.anatomia.anchos_de_corona`). Dos detalles del cálculo no
son opcionales: se toma solo la **componente conexa mayor** de cada etiqueta —con todos
sus vértices el 27 de un caso real «mide» 41 mm, porque se está midiendo hasta la mota
más lejana— y el **rango del 1 al 99 %**, no el máximo.

### Calibración: ¿cuánto marca sobre etiquetas de experto?

Sobre 330 coronas anotadas de 25 maxilares de Teeth3DS+:

| | exceso mediano | coronas > +1,5 mm |
|---|---|---|
| **verdad anotada** | **+0,28 mm** | **6 %** |

O sea que el umbral de +1,5 mm deja fuera la variación anatómica normal. La medida no
está sesgada.

### El modelo, sobre el split de TEST (20 maxilares, 80/20 semilla 42)

| decodificación | acierto/punto | FDI/diente | exceso mediano | coronas anchas |
|---|---|---|---|---|
| `argmax` por punto | 0.898 | 0.958 | +0,96 mm | **33 %** |
| contigüidad `beta=2` | **0.908** | **0.962** | +0,92 mm | **33 %** |

**Un tercio de las coronas tiene la frontera fuera de tolerancia con 0,958 de FDI por
diente.** Cinco veces la tasa natural. Las dos métricas que esta ficha publica no lo ven.

### La decodificación con contigüidad y qué separa

`tooth_aggregation.suaviza_contiguidad` decodifica el mismo `logprob` con ICM sobre el
grafo kNN, añadiendo el único hecho que el `argmax` por punto ignora: **un diente es un
parche contiguo**. No reentrena nada.

- **Quita las motas**: sobre un maxilar real de 112.067 vértices, componentes conexas por
  pieza **119 → 30** e islas **6,8 % → 1,5 %**. En test, +1,0 punto de acierto por punto
  y +0,4 de FDI por diente — no degrada nada, por eso es el defecto de `aggregate_teeth`.
- **No mueve la frontera**: coronas anchas 33 % → 33 %, exceso mediano −0,04 mm. En el
  caso propio, 9 de 15 antes y después con el exceso medio moviéndose una centésima.

Y esa separación **es el resultado**. El término de contigüidad pesa como mucho `beta`,
así que no puede con una diferencia de log-probabilidad mayor: donde el modelo se
equivoca **con confianza** —y en el punto de contacto interproximal se equivoca con
confianza— no hay nada que decodificar mejor. La frontera pide otro modelo, no otro
decodificador.

### Fuera de Teeth3DS+

Sobre el maxilar de `histora` (otro escáner, otra pose, dentición con implantes) las
coronas anchas suben de **33 % a 60 %** (9 de 15), con excesos de hasta +9,1 mm en el 27.
Es la misma caída que A6 documenta para la pose: el número de test no se transfiere por
decreto.
