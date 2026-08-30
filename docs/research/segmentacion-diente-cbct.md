# Segmentar el diente en el CBCT — hasta dónde llega un clasificador, y dónde deja de llegar

> **Estado (2026-08-20):** resultado **negativo**, medido y reproducible. Un segmentador
> mejor en el banco de pruebas **no** arregla el defecto que motivó entrenarlo, y el que
> gana el banco **pierde en la tarea real**. Se documenta porque la conclusión cambia el
> plan: el recorte de la raíz hay que atacarlo por anatomía, no por clasificación.

## 1 · La pregunta

El compuesto CBCT+IOS entrega cada diente con su raíz, pero **la mitad de las piezas salen
demasiado largas**: 27-37 mm cuando un diente entero mide 20-25. Los milímetros de más son
hueso alveolar pegado por debajo del ápice.

La hipótesis era de precisión. El primer segmentador tenía **recall 0,948 y precisión
0,588**: marcaba de diente casi todo lo que tocaba hueso. Parecía evidente que subir la
precisión partiría las piezas.

Se entrenó un segundo modelo bajando la fracción de parches centrados en diente
(`--frac-diente` 0,50 → 0,25), que es la palanca directa sobre ese desequilibrio.

## 2 · Los dos modelos, en el banco de pruebas

Evaluación sobre **volúmenes enteros** de ToothFairy2 con ventana deslizante — el único
número comparable con el listón del umbral de HU.

| | `modelo.pt` | `modelo_p025.pt` | listón (umbral HU) |
|---|---|---|---|
| `--frac-diente` | 0,50 | 0,25 | — |
| **F1 (volumen)** | 0,726 | **0,822** | 0,530 |
| recall | 0,948 | 0,949 | — |
| **precisión** | **0,588** | **0,724** | — |
| F1 (parche) | 0,956 | 0,955 | — |

El segundo modelo hace exactamente lo que se le pidió: **+13,6 puntos de precisión sin
tocar el recall**, y +0,096 de F1 sobre volumen entero.

⚠️ El F1 por parches (0,955) es el mismo en los dos y **no distingue nada**: con parches
centrados en diente al 50 % los positivos son ~30 % frente al ~1 % de un volumen real.
Es la métrica que hay que ignorar, y está aquí solo para que se vea por qué.

## 3 · El experimento controlado

Mismo comando, mismo CBCT (`histora`, 0,300 mm isótropo, 498.407 primitivas), mismos dos
escaneos intraorales, mismas etiquetas FDI. **La única variable es el checkpoint.**

```
scripts/composicion_cbct_ios.py --cbct <serie> --ios <upper> <lower>
    --fdi <etiquetas_upper> <etiquetas_lower> --esperados 28 --modelo <checkpoint>
```

| | `modelo.pt` (pre 0,588) | `modelo_p025.pt` (pre 0,724) |
|---|---|---|
| gaussianas marcadas diente | 58.533 | 48.848 |
| **pieza mayor** | **40,5 mm** | **40,2 mm** |
| dientes nombrados | **27** de 28 | 20 de 28 |
| con raíz / desbordadas | 13 / 13 | 11 / 9 |
| registro maxilar | **p50 0,81 mm · 78 % bajo 2** | p50 4,02 mm · 30 % |
| registro mandibular | **p50 0,77 mm · 80 % bajo 2** | p50 3,52 mm · 29 % |

## 4 · Tres resultados negativos

**4.1 · La pieza de 40 mm no se mueve.** 40,5 → 40,2 mm. Veinticuatro puntos de precisión
no la tocan.

**4.2 · Ni el umbral de decisión la toca.** Barrido completo en los dos modelos:

| umbral | 0,50 | 0,60 | 0,70 | 0,80 | 0,90 | 0,95 |
|---|---|---|---|---|---|---|
| mayor pieza, `modelo.pt` | 40,5 | 40,2 | 40,2 | 40,2 | 40,2 | 40,2 |
| mayor pieza, `modelo_p025.pt` | 40,2 | 40,2 | 40,2 | 40,2 | 39,9 | 37,5 |

Doce configuraciones, dos clasificadores separados por 24 puntos de precisión, **la misma
componente**. El modelo no duda de esos vóxeles: ahí hay diente de verdad, y hay hueso de
verdad, y no hay frontera entre ellos que ninguna probabilidad pueda cortar.

**4.3 · Las desbordadas son demasiado uniformes para ser ruido.** Con `modelo_p025.pt`
van de 27,6 a 31,8 mm — cada una arrastra 5-8 mm de hueso, y **casi lo mismo en todas**.
Si fuera imprecisión del clasificador habría dispersión. Y son casi todas molares y
premolares (15, 16, 27, 35, 36, 37, 46, 47), que es donde la raíz es más larga y más
superficie toca hueso.

Encaja con el muro físico que el pipeline documenta desde el principio: **el ligamento
periodontal mide 0,15-0,38 mm y el vóxel 0,30**. Por debajo de la cresta ósea no hay borde
que resolver. Ningún clasificador inventa un borde que el CBCT no midió.

## 5 · El daño colateral: la máscara alimenta el registro

El modelo «mejor» **degrada el registro cinco veces** (p50 0,81 → 4,02 mm), con el mismo
CBCT y los mismos escaneos. La causa está en el propio script: el objetivo del ICP es

```
máscara_de_diente ∩ HU ≥ 1200
```

es decir, **la salida del segmentador es la entrada del registro**. Máscara distinta →
objetivo distinto → pose distinta → el nombrado por vecindad falla → 7 dientes se quedan
mudos (27 → 20).

Y lo contraintuitivo: `modelo_p025.pt` aporta **más** gaussianas de corona en el maxilar
(11.154 frente a 9.340) y aun así registra peor. No es cantidad — es que mete cosas
equivocadas cerca de las coronas.

⚠️ **Y el rms del ICP no se enteró de nada.** Los cuatro residuos caben en 0,06 mm:

| | maxilar | mandibular |
|---|---|---|
| ejecución **buena** (p50 0,81 / 0,77) | rms 0,642 | rms 0,593 |
| ejecución **mala** (p50 4,02 / 3,52) | rms **0,591** | rms 0,655 |

En el maxilar el residuo **premia a la ejecución mala**; en la mandíbula lo ordena bien
pero por 0,06 mm. Mientras tanto la calidad real difiere **cinco veces**. Quien eligiera
el modelo por el rms del registro elegiría el equivocado. Es la cuarta vez que este modo
de fallo aparece medido en el proyecto (ver
[`fusion_agents.preparacion`](../../packages/fusion-agents/src/fusion_agents/preparacion.py)).

### 5.1 · Resuelto, y lo que costó

El objetivo del ICP ya no toca la máscara del modelo. El plano oclusal sale del **modo de
la z del esmalte** (`plano_oclusal_del_esmalte`) y el objetivo se elige **midiendo**, no
fijando una constante.

| | acoplado | desacoplado ingenuo | **desacoplado + árbitro** |
|---|---|---|---|
| FDI con `modelo.pt` | 27 | 25 | **27** |
| FDI con `modelo_p025.pt` | **20** | 24 | **27** |
| maxilar corona→diente | 0,81 mm · 78 % | 4,49 · 27 % | 1,06 · 71 % |
| mandibular | 0,77 mm · 80 % | 1,48 · 60 % | **0,56 · 90 %** |

**El hueco de 7 dientes desaparece**, y el registro sale bit a bit idéntico con los dos
checkpoints (mismo objetivo, mismo rms): la etapa ya no depende del segmentador.

Tres cosas que hubo que aprender por el camino:

**a · El acoplamiento hacía trabajo real.** La máscara excluía el hueso del objetivo del
ICP. Quitarla sin más (`HU ≥ 1200` sobre el campo entero) mete cortical del paladar y
hunde el maxilar a 4,49 mm. La salida libre de modelo es el **esmalte**: ningún hueso
llega a esa densidad.

**b · Pero ningún umbral vale para las dos arcadas, y el barrido no es monótono.**

```
maxilar     1200 → 6,84 mm    1400 → 1,73    1600 → 1,53    1800 → 8,13 ⚠    1900 → 1,64
mandibular  1200 → 7,35 mm    1400 → 7,41    1600 → 7,45    1800 → 0,73      1900 → 0,73
```

El maxilar **falla en 1800 entre dos vecinos buenos** — firma de mínimo local del ICP, no
de mala elección de objetivo. Fijar una constante ahí sería ajustar a un paciente. Por eso
se prueban los cinco y gana el que puntúe mejor contra el árbitro.

**c · Y el árbitro no puede ser el rms.** Quinta aparición del mismo modo de fallo: el rms
de esas diez poses cabe en **0,614-0,668 mm** mientras su calidad real va de **0,73 a
8,13**. La distancia corona→esmalte sí discrimina, y no conoce al modelo.

**Lo que esto NO arregló:** las desbordadas vuelven a 12 (eran 13 acopladas). La bajada a 7
del desacoplamiento ingenuo era un artefacto de que, con el registro malo, se nombraban
menos gaussianas y las piezas salían más cortas. El §4 se mantiene entero.

## 6 · Lo que NO se puede concluir

Se corrió también sobre el segundo paciente, y **esa ejecución no aporta nada** por dos
motivos, los dos de invocación y no del método:

- Se lanzó **sin `--fdi`**. Sin etiquetas, `f = ones(len(V))` y por tanto `f > 0` selecciona
  **todos los vértices, encía incluida**. Los `p50 6,00 mm` que imprimió no miden el
  registro: miden encía contra dientes, y la encía no tiene contrapartida por construcción.
  El `0 vértices de encía` del compuesto es el mismo artefacto (`encia_total` cuenta
  `f == 0`).
- Su CBCT es **0,15 × 0,15 × 0,45 mm**, anisótropo, así que se activa el remuestreo al
  espaciado de entrenamiento y entra una variable nueva. (El `SliceThickness` del DICOM
  dice 0,150 y miente; el espaciado bueno sale de las posiciones, `IPP`.)

Queda anotado para no reutilizar esos números.

## 7 · Qué se saca de aquí

**El hueco banco↔tarea es el hallazgo.** `modelo_p025.pt` gana en ToothFairy2 (F1 0,822
frente a 0,726) y **pierde en el caso clínico** (20 dientes frente a 27). Un banco de
pruebas que puntúa vóxeles no puntúa lo que el pipeline necesita, que es *piezas
separables con la identidad correcta*.

Decisiones que salen de esto:

1. **`modelo.pt` sigue siendo el de la composición.** `modelo_p025.pt` se conserva como el
   que gana el banco, con esta ficha al lado explicando por qué no se usa.
2. **El recorte de la raíz no es un problema de clasificación.** Hay que atacarlo por
   anatomía: cortar a una longitud esperada desde la corona registrada, que es medida, en
   vez de esperar que el clasificador encuentre un borde que no existe. Eso convierte el
   ápice en **supuesto**, no medido, y como tal tendría que declararse en la procedencia —
   sirve para que el compuesto deje de arrastrar mandíbula, y **no** sirve para medir
   longitud radicular, que sería circular.
3. **Desacoplar registro y segmentación.** Que el objetivo del ICP dependa de la máscara
   del modelo acopla dos etapas que deberían ser independientes, y es cómo un cambio de
   checkpoint estropeó el registro sin que nadie lo pidiera.
4. **El F1 sobre volumen no se guarda en el checkpoint**, solo el de parches — que es
   justo el que no sirve. Vive únicamente en el log del entrenamiento.

## 8 · Reproducir

```bash
# entrenamiento (GPU): 16.000 pasos, ~33 min en una RTX 5070
~/.venvs/dental-gpu/bin/python scripts/entrena_diente_cbct.py \
    --datos ~/anfaia/toothfairy2 --frac-diente 0.25 --pasos 16000

# el listón que hay que batir —el umbral de HU contra verdad anotada, F1 0,530—
# se midió con un script que ya no vive en el repositorio: dependía del banco
# de ToothFairy2, cuya licencia no permite que nada suyo viaje aquí.

# la comparación de §3: mismo comando, cambiando solo --modelo
~/.venvs/dental-gpu/bin/python scripts/composicion_cbct_ios.py \
    --cbct <serie> --ios <upper.stl> <lower.stl> --fdi <upper.npy> <lower.npy> \
    --esperados 28 --modelo <checkpoint>
```

Los ficheros crudos y los checkpoints viven fuera del repositorio (`~/anfaia/`,
`data/processed/`), como el resto del dato clínico — ver
[`dataset-histora.md`](dataset-histora.md).
