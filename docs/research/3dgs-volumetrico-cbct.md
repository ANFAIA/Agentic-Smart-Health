# 3DGS volumétrico sobre CBCT: qué se midió y qué se aprendió

**Estado**: exploración cerrada. Los resultados de abajo son medidas, no
estimaciones, y cada una dice sobre qué se tomó.

**Los datos y el código no se publican.** El trabajo corrió sobre una cohorte CBCT
cuya licencia es CC BY-NC-SA y cuya descarga exige registro y aceptar un acuerdo de
uso, así que ni los volúmenes ni los campos derivados de ellos pueden vivir en este
repositorio. Lo que sí es nuestro y sí se publica es esto: **lo aprendido**. Este
documento está escrito para ser autosuficiente — no depende de ningún notebook ni
de ningún fichero que no esté aquí.

---

## Por qué se hizo

El [ADR 001 §7.1](../architecture/001-digital-twin-core-schemas.md) define la
primitiva del gemelo digital para rayos X en términos muy concretos: la opacidad
`α ∈ [0,1]` pasa a **densidad `σ ≥ 0` sin cota**, los armónicos esféricos de color
se **descartan** (la atenuación es isótropa), y el campo modela
`μ(x) = Σᵢ σᵢ·G(x | cᵢ, Σᵢ)`. El docstring del `cbct-agent` afirma además que su
salida es «exactamente la inicialización que un optimizador RGS refinaría después».

Ninguna de las dos afirmaciones estaba comprobada. Esta exploración las somete a
prueba con un CBCT real, un rasterizador real (`gsplat`) y una verdad de campo que
respeta la física: **radiografías sintéticas** (DRR), es decir integrales de línea
`τ = ∫μ ds`, en vez de renders de superficie.

## Las dos ramas, y por qué la distinción no es un matiz

|  | fotométrica | volumétrica |
|---|---|---|
| Atenuación | opacidad `α ∈ [0,1]` | **densidad `σ ≥ 0`** |
| Apariencia | 9 coeficientes SH por canal | **ninguna** |
| Verdad de campo | renders de superficie | **DRR** del volumen medido |
| Color | el del material del render — **inventado** | **`None`**, por construcción |
| Composición | α-blending, depende del orden | **Beer-Lambert**, orden-independiente |

La primera versión de este trabajo entrenó contra renders de una superficie opaca
derivada del propio CBCT. Producía una **cáscara con color inventado**: se veía
bien y era exactamente lo contrario de lo que el ADR pide. Corregirlo —pasar a DRR,
tirar los armónicos y declarar `color_superficie = None` como manda el
[ADR 004 §2.8](../architecture/004-fusion.md)— fue el punto de inflexión de todo lo
demás.

---

## Hallazgos

### 1. `gsplat` impone una ventana estrecha a σ, y hay que calibrarla midiendo

El rasterizador **descarta toda gaussiana cuya opacidad efectiva caiga por debajo
de ~1/255** (medido: con opacidad `1e-3` la imagen sale exactamente cero, así que
la primitiva no contribuye *ni recibe gradiente*: está muerta sin avisar). Por
arriba, el alfa acumulado satura en 0,9999 en `float32`, lo que topa `τ` en ≈ 9,2.

Con la σ cruda derivada de los HU, cientos de gaussianas por rayo saturan ese techo
y el modelo **se clava en 19 dB** con la predicción topada en 0,087 frente a 1,54
de la verdad de campo. La corrección es **un escalar global** sobre σ —una elección
de unidades, no de anatomía: la *forma* que midió el CBCT se respeta— calibrado
contra la media de la DRR. Con él, 19 → 43 dB.

> **Consecuencia de arquitectura.** El ADR 001 asume que la primitiva volumétrica
> puede alojarse en un rasterizador de splatting. Puede, pero **con una ventana de
> σ prestada de un modelo fotométrico**, y esa ventana es la que limita todo lo
> demás (ver hallazgo 3). Un rasterizador que integre σ sin suelo ni saturación es
> una decisión de producto pendiente, no un detalle de implementación.

### 2. Más presupuesto de primitivas no compra calidad; la empeora

Barrido sobre un caso, mismo entrenamiento, solo cambia cuántas gaussianas se
siembran:

| sembradas | % en el suelo al acabar | **vivas** | PSNR |
|---|---|---|---|
| 20 000 | 61,6 % | 7 685 | 43,07 |
| 40 000 | 76,6 % | 9 373 | 43,16 |
| 80 000 | 86,2 % | 11 078 | 43,13 |
| 160 000 | 93,3 % | 10 721 | 40,55 |
| 320 000 | 97,4 % | 8 287 | 40,18 |
| 418 727 | 98,0 % | 8 298 | 42,44 |

El campo vivo se estanca en ~8–11 k pase lo que pase: la calibración global reparte
la misma atenuación total entre más primitivas, así que cuantas más se siembran, más
caen al suelo. **Ese número no es una constante del rasterizador** — depende del
rango dinámico de la imagen objetivo (ver hallazgo 4).

Una **densificación adaptada a σ** —podar lo muerto, dividir por gradiente
posicional y reajustar la escala global multiplicando— da el mismo resultado con
**2,4× menos primitivas** (33 891 frente a 80 000, 43,60 frente a 43,12 dB). Es
eficiencia, no techo. La heurística estándar de 3DGS no sirve aquí: hace *opacity
reset*, que sobre una densidad física no significa nada.

### 3. Más resolución tampoco, y se puede decir por qué

Con la cámara a 291 mm y el encuadre usado, 400 px son 0,390 mm por píxel y 800 px
son 0,195, **frente a un vóxel de 0,300 mm**. El salto 400 → 800 recupera señal que
el CBCT sí tiene; a partir de ahí es aumento vacío, porque la DRR es interpolación
trilineal de esa misma rejilla. Medido: la rama volumétrica **no mejora** al doblar
la resolución (43,1 dB en ambas) y la fotométrica **empeora** 1,3 dB — los mismos
splats tienen que explicar el doble de detalle.

Y trocear las vistas en parches (*crop* como aumento de datos) no aporta nada por
construcción: son los mismos píxeles reorganizados; no cambia ni la información ni
el reparto del campo.

### 4. Lo que sí funciona: **un campo por tejido**, cada uno con su escala de σ

La propuesta era dividir el volumen por rangos de densidad y darle a cada rango su
propia dimensión. Se comprobó así: **particiones disjuntas que cubren toda la
anatomía**, de modo que la integral de línea sea aditiva y la suma de las capas
tenga que reproducir **exactamente** la radiografía completa — lo que permite medir
todos los modelos **sobre la misma imagen**, que es la única comparación honesta.

La aditividad se verificó antes de comparar nada: error máximo `3,6·10⁻⁷` sobre τ
normalizado.

Resultado, cinco capas (clase anatómica ∩ rango de HU) frente al campo único, cada
caso contra su propio control y a su propia resolución:

| caso | 1 campo | 5 capas | Δ | vivas 1 campo | vivas 5 capas |
|---|---|---|---|---|---|
| A | 43,23 | 47,62 | **+4,39 dB** | 11 013 | 176 865 |
| B | 42,02 | 44,10 | **+2,08 dB** | 22 887 | 185 787 |
| C | 37,44 | 44,85 | **+7,41 dB** | 13 672 | 133 628 |

Con un ruido entre corridas de ~0,5 dB, la ganancia es señal, y **replica en los
tres casos**.

**El mecanismo está medido, no supuesto.** No es que cada campo tenga un cupo fijo:
un campo entrenado contra la radiografía de *un solo tejido* atenúa mucho menos, así
que tras calibrar σ sobrevive al suelo mucha más primitiva. **16 veces más gaussianas
vivas** repartidas en cinco capas que en el campo único. Dar ese mismo presupuesto a
un solo campo no funciona (320 k sembradas → 40,18 dB): hay que **repartirlo en
campos con su propia escala de σ**.

De regalo, una capacidad que el campo único no puede tener de ninguna manera:
**encender y apagar tejidos**. Como la atenuación se suma, ver un subconjunto es
sumar los campos que toque, sin reentrenar nada.

### 5. Concentrar el presupuesto en el espacio **no** funciona (hipótesis refutada)

Si el estancamiento fuera «por campo», restringir un campo a una región le daría el
mismo cupo para una fracción del espacio. Se probó con una caja de 25 mm sobre un
molar y cámaras acercadas hasta llenar el cuadro (31 µm/píxel):

| modelo | sembradas en la región | vivas | PSNR |
|---|---|---|---|
| campo global recortado a la región | 5 803 | 1 404 | 25,56 |
| campo dedicado, siembra completa | 29 114 | 1 533 | 44,50 |
| campo dedicado, **misma densidad de siembra** | 5 803 | 1 461 | 45,02 |

**La concentración no aporta nada**: sembrar 5× más primitivas en la región da el
mismo resultado (45,02 con la siembra escasa frente a 44,50 con la densa), y las
vivas no se multiplican. El estancamiento es **por región proyectada**.

Los 19 dB de diferencia vienen entonces de otra cosa: la **especialización** —
entrenar contra la radiografía de esa región con su propia calibración de σ—, el
mismo mecanismo del hallazgo 4 aplicado al espacio. Y sale barato: 5 803 gaussianas
y 22 s para el «zoom» de una pieza, lo que lo convierte en una función viable
(«enséñame este diente») y no en un lujo computacional.

*(Cautela: el control hace una tarea para la que no se entrenó, así que esos 19 dB
miden «lo mal que se extrae una región de un modelo global», no calidad absoluta.)*

### 6. El ingestor registrado bastaba: no hacía falta un agente nuevo

Materializando el volumen como serie DICOM, el **`cbct-agent` del repositorio** da
exactamente la misma nube que un agente escrito a medida (Δ centros = 0,0 mm) y la
misma superficie derivada, byte a byte. Lo que sí falta en producto es el **lector**
de formatos volumétricos (`.mha`, `.nii.gz`), no un agente.

---

## Límites, y son del dato

- **La segmentación de entrada pone el techo.** Las capas por tejido heredan sus
  errores: en un caso hay **0,78 cm³ con densidad de esmalte etiquetados como
  hueso**, y el 99,3 % en **una sola región** de 11 mm — tamaño de pieza dental, no
  ruido. El objeto denso completo (2,32 cm³) lo reparte el mapa entre diente (46 %),
  hueso (33 %) y fondo (17 %), y está saturado al 67 % frente al 22–33 % de las
  piezas bien etiquetadas: huele a metal (restauración) que satura el detector y
  funde piezas vecinas. **No rompe la partición** —sigue siendo disjunta y sumando
  exacto—, lo que falla es la etiqueta. La partición por densidad pura no sufre esto
  porque no afirma nada anatómico que pueda ser falso.
- **La encía no está, y no es un problema de método.** Por densidad, el tejido blando
  del campo de visión forma **una sola población** (mediana −23 HU, desviación 92 HU,
  histograma unimodal): ningún umbral separa encía de mejilla, labio o lengua, y
  además se tocan. Un CBCT está optimizado para tejido duro. La encía sale de fundir
  con un **escaneo intraoral**, que sí mide su superficie y su color — registrando por
  el esmalte, queda acotada entre esa superficie y el hueso alveolar. Eso está
  bloqueado por datos: hace falta **CBCT e intraoral del mismo paciente**, y las
  cohortes disponibles traen una cosa o la otra, nunca las dos.
- **La rama fotométrica no es fotogrametría clínica.** Sus «fotos» son renders de una
  superficie derivada del propio volumen: comparten geometría con la verdad de campo.
  Falta la pose real, la iluminación real y el ruido del sensor.
- **Este montaje no valida la fusión geométrica**, y el ADR 004 ya lo anticipa en su
  tabla de alternativas descartadas: una malla derivada del propio volumen queda
  alineada por construcción, así que no hay nada que registrar.
- **Los umbrales son decisiones clínicas disfrazadas de parámetros.** Dónde acaba el
  hueso y cuánto interior entra en el campo los fija un número, y ninguno está
  calibrado contra una referencia clínica.

---

## Qué debería recoger el producto

1. **Lector de volúmenes dentro del `cbct-agent`**, como despacho por formato
   (directorio ⇒ DICOM, fichero ⇒ MetaImage/NIfTI), igual que el `mesh-agent`
   despacha entre OBJ y STL. Son pocas líneas y elimina todo el andamiaje externo.
2. **Decidir si el centrado entra en el contrato.** El agente aplica un `world_offset`
   y lo descarta, así que cualquier consumidor que necesite alinear su salida con
   otra cosa tiene que reconstruirlo fuera. (No es que los artefactos no sean
   alineables: para eso está el registro de la capa de fusión. Es que el marco no
   viaja con el dato.)
3. **Un ADR sobre el rasterizador.** La ventana de σ del splatting fotométrico
   (suelo en ~1/255, techo en τ≈9,2) es lo que topa la capacidad efectiva del campo,
   y ni más presupuesto ni más resolución ni concentración espacial lo mueven. Lo
   único que lo mueve es **varios campos especializados**. Eso es una restricción de
   arquitectura y merece quedar escrita antes de construir encima.
4. **La descomposición por tejido como característica, no como truco.** Reconstruye
   mejor de forma reproducible y habilita el visionado por capas, que es como un
   clínico querría mirar el twin. Depende del `segmentation-agent` para que los
   nombres signifiquen algo; sin él, la partición por densidad es la versión honesta.

## Replicación sobre el caso clínico real (demo, 2026-08-28)

Los hallazgos 4 y 5 se replicaron sobre el **caso clínico real de la demo** (serie
DICOM de la clínica, no la cohorte CC BY-NC-SA), para comprobar que el margen no era
propio de la cohorte. Protocolo: DICOM → vóxel 0,30 mm → 252 vistas en órbita, 800×800,
7000 iteraciones, holdout 1/8, y el control entrenado con la MISMA rama volumétrica
(DRR) para que la comparación sea sobre la misma imagen.

**Partición por densidad sola** (`PARTICION_HU`, sin etiquetas — máscara de paciente
sintetizada como HU > 300):

| | PSNR holdout |
|---|---|
| campo único (DRR) | 37,35 dB |
| suma de 4 capas por HU | **39,83 dB** |
| **Δ** | **+2,48 dB** |

Replica el hallazgo 4 dentro de su horquilla (+2,08 a +7,41). La descomposición por
densidad no necesita segmentación y recupera el margen en el caso real.

**Capas cruzadas** (diente/hueso, con el U-Net de diente entrenado del proyecto,
F1 de validación 0,956):

| | PSNR holdout |
|---|---|
| campo único (DRR) | 37,32 dB |
| suma de 4 capas cruzadas | 36,07 dB |
| **Δ** | **−1,25 dB** |

La partición por CLASE **resta** en el caso real, a diferencia de la cohorte (donde el
mapa de 6 clases venía del dataset). Confirma el techo declarado en el hallazgo 4: la
calidad de las capas cruzadas está topada por la **segmentación de entrada** — el esmalte
que el U-Net se pierde cae en `hueso-cortical` con la normalización de σ equivocada y la
suma reproduce peor que un campo único. El encendido clínico por tejido queda condicionado
a una segmentación de diente mejor.

⚠️ Alcance: el campo de densidad que lleva hoy el `.uos` de la demo es la **semilla cruda**
del `cbct-agent` (perfil `ash-twin/1.0`, submuestreo 3×3×1), sin entrenar — ni DRR ni
`ajusta_campo`. Estos números son de la rama volumétrica y no alteran el contenedor actual
— dicen cuánto ganaría la demo si adoptara el campo en capas.

## Nota de reproducibilidad

El código de la exploración y los artefactos derivados están **fuera del control de
versiones a propósito** (`.gitignore`), por la licencia de la cohorte. Las cifras de
este documento provienen de tres volúmenes CBCT reales, con 252 vistas por caso y
una de cada ocho retenida para evaluación; los PSNR son siempre sobre vistas
retenidas, y las comparaciones entre modelos, siempre sobre la misma imagen.
