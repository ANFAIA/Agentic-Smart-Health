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

### El número que importa

| modelo de A3 | por **punto** | **FDI por DIENTE** | mejor con fusión |
|---|---|---|---|
| normales · CE | 0.849 | **0.932** (1478/1585) | 0.933 |
| normales · CE+boundary+centroid | 0.473 | **0.490** (776/1585) | 0.486 |
| **spread** | 0.376 | **0.443** | 0.447 |

- **El acierto por punto subestima la identificación por diente** (0.85 vs 0.93): son métricas
  distintas y la ficha del agente debe declarar la segunda, no la primera. Esto sí se repite en
  todas las corridas.
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
