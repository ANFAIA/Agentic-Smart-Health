# Registro escáner↔CBCT sin anotador: navaja por diente y arranques desplazados (casos A y B)

> **Estado (2026-11-28): pre-registro.** Fijado en este commit **antes de ejecutar nada** sobre
> los casos A y B. Son las medidas 2 y 3 de las que el proyecto se propuso hacer sin anotador; la
> 1 es [`consistencia-registro-caso-b.md`](consistencia-registro-caso-b.md). Las dos miden la
> **precisión** del registro: cuánto depende la pose de cada diente que entra en el ajuste y del
> punto de partida. **No miden su exactitud.** Un sesgo que compartan todas las ejecuciones (la
> isolínea de 1400 HU <!--const:UMBRAL_CONTORNO-->, el borde a medio vóxel, la altura como dirección peor medida) se cancela y
> no aparece. No sustituyen al TRE con puntos anatómicos marcados por una persona
> ([`registro-caso-a-puntos-preregistro.md`](registro-caso-a-puntos-preregistro.md)).

## 1 · Las preguntas

El registro del caso A (R1) da 0,555 mm de rms sobre los puntos de contorno que tienen pareja,
con una confianza de 0,63. El solapamiento de R1 no consta; en la reejecución con C3 es del
55,8 %, y V0 lo volverá a medir. El rms es un residuo **dentro de la muestra**: mide lo bien que
la pose se ajusta a los dientes con los que se calculó. No dice nada de un diente que no hubiera
entrado en el ajuste. Las dos medidas preguntan:

- **Medida 2 · navaja por diente.** Si se repite el registro dejando fuera la corona de un diente
  X, ¿cuánto se mueve X entre las dos poses? ¿Y cuánto peor ajusta la pose sin X al propio X que
  la pose con todos?
- **Medida 3 · arranques desplazados.** Si el registro empieza desde una pose desplazada una
  cantidad conocida, ¿vuelve a la misma pose?
- **Comprobación de cordura · equivarianza.** Si se mueve rígidamente lo que el método trata como
  datos, ¿el resultado sigue exactamente ese movimiento? Es una prueba del programa, no de la
  precisión.

Se llama «navaja» y no «leave-one-out» porque el artículo ya usa *leave-one-out* para otra cosa
(un marco para medir el movimiento de un diente entre dos escaneos). Es el término de
`fusion_agents.anatomico`, cuya navaja es otra: parte de la pose elegida y no repite la búsqueda,
así que no sirve para el camino por cortes ni para el clásico.

## 2 · Los datos, y qué registro se mide en cada caso

| | Caso A | Caso B |
|---|---|---|
| Serie CBCT | `c1bd42bdd9d7`, 301 cortes, dispersa | `CASE-64A77C`, 578 cortes, densa |
| Escaneo | maxilar | maxilar, POST HIGIENE (el de `run-5-modelo`) |
| Camino de registro (`--registro auto`) | `fusion_agents.por_cortes`, determinista | ICP clásico desde la identidad, con árbitro de umbral HU y submuestra aleatoria de 4000 <!--const:MUESTRA--> vértices |
| Ejecución de referencia | **R1** = `run-6` (`57eb021`, árbol limpio), la que puntuarán los puntos anatómicos | `demo-c3/caso-b` (`eef9d27`, árbol limpio); su matriz es idéntica byte a byte a la de `run-5-modelo` |
| SHA-256 del contenedor de referencia | `239e258e077d7b53a561740d67d8188705992dd00a3e992f38b5d05819f8034a` | `c1baaf680771b675fe83e74be473fb0c377920a5dd0b333197ad0cabd06d82f7` |
| Dientes | los códigos de `seg_teeth` del contenedor; se esperan 14 (17…27) | los códigos de `seg_teeth` del contenedor; se esperan 13 (11–16, 21–27) |

El script se niega a ejecutarse si el contenedor no tiene esa SHA-256, o si el `run-record.json`
de la ejecución de referencia no da ese commit con `git_dirty: false`. Si los dientes no son los
esperados, se miden los que haya y se declara: no se elige ninguno.

**Etiquetas.** Las de `derived/seg_teeth` del contenedor de referencia. En el caso A son las que
usó el registro tras la limpieza y el margen. En el caso B el registro no usa etiquetas: aquí
solo definen las coronas y el diente X. No se recalculan: volver a ejecutar `caso_completo.py`
hoy daría otras (desde el 2026-11-24 `--margen-gingival aplica` calcula `M10`, y la segmentación
por defecto es C3).

**Los datos del caso no se reingieren.** Se cargan del almacén de artefactos de la ejecución de
referencia: el campo semilla del CBCT (con `origin` y `hu_range`) y la superficie del escáner en
el orden de ingesta. Leer un STL movido reordena los vértices (`parse_stl` deduplica con
`np.unique`), y en el camino clásico eso cambia la submuestra. Todos los movimientos se hacen
en memoria.

## 3 · Procedimiento, fijado antes de ejecutar

[`scripts/registro_navaja_arranques.py`](../../scripts/registro_navaja_arranques.py), una vez por
caso. Por diseño **no modifica `packages/fusion-agents`**: llama a sus funciones desde fuera.

### 3.1 · V0, la puerta de reproducción

Antes de medir nada, el script reconstruye el registro de referencia con el código de este
commit y lo compara con el `reg.ct_to_ios` del contenedor de referencia:

- **Caso A:** `anatomical_frame` y `prepara_por_cortes` con las etiquetas de R1, y el registrador
  aplicado a la corona y al contorno. Así lo hace `IngestionPipeline.prepara_registro`.
- **Caso B:** `nubes_para_registro` y `icp`, los dos con semilla 0, que es lo que usa el
  pipeline: `caso_completo.py --semilla` no llega al registro.

**V0 se cumple** si, sobre los vértices de corona, el desplazamiento máximo entre las dos poses
es de 0,01 mm o menos. Si no se cumple, el script escribe `V0.json` (con `pass: false`) y
`run-record.json` (con `aborted: "V0"`) y se detiene sin medir nada más. La causa se corrige en
una enmienda con su commit, y V0 se repite **una sola vez**, en una salida nueva. Si vuelve a
fallar, la medida de ese caso no se hace y se escribe así.

### 3.2 · Medida 2: navaja por diente

**Caso A (por cortes).** El destino (contorno de esmalte en la franja, plano oclusal) depende solo
del CBCT. La pose de partida depende de las etiquetas. Para que la navaja mida el ajuste y no el
cambio de ejes, se calcula **una vez** con todas las etiquetas: `rot0` y `plano` quedan fijos y,
para cada diente X, el mismo registrador se aplica a la corona sin X. Hay dos variantes:

- **(b) principal: el contorno de X también sale.** Son los puntos de contorno cuyo vértice de
  corona más cercano, con la pose completa, lleva la etiqueta X. Usar la pose completa para elegir
  la máscara es circular, y se declara. Sin esto, el contorno de X se empareja con las coronas
  vecinas y el recorte del 20 % solo lo quita en parte.
- **(a) descriptiva: el contorno de X se queda.** Mide el arrastre que deja un diente que el
  escáner no ve pero el CBCT sí.

El arranque en x/y y en altura se recalcula desde la corona sin X: es lo que hace el método. Por
cada X se guarda también la pose de arranque sin iterar (`max_iter=0`), para separar lo que se
mueve el arranque de lo que se mueve el ajuste.

**Caso B (clásico), solo descriptiva.** La única forma de dejar fuera X es quitar sus vértices de
los que entran en `nubes_para_registro`. No se puede quitar el esmalte de X del lado del CBCT sin
segmentarlo o sin usar la pose completa. Quitar vértices cambia la submuestra de todos los
demás, así que **no hay pares exactos**: el ruido de semilla ya medido en este camino (1,00 y
1,85 mm de mediana) es del orden del límite de N2, y lo dominaría por construcción. Por eso en el
caso B la navaja **no tiene criterio** y se lee solo contra el suelo de semilla (§3.6). Cada
condición se ejecuta con 20 semillas (0–19), y la misma semilla se usa en `nubes_para_registro`
y en `icp`:

- **Desde cero (principal).** Arranca en la identidad, como el pipeline. Para cada semilla s, la
  pose sin X se compara con la pose completa con esa misma semilla.
- **En caliente (descriptiva).** El escaneo se mueve antes con la pose de referencia T₀, así que
  el ICP afina desde ella. Es lo análogo a la navaja de `anatomico`.

**Residuo de X**, el mismo estadístico con la pose completa y con la pose sin X:
- **Caso A:** la mediana de la distancia de los puntos de contorno de X (la máscara de (b)) a los
  vértices de la corona de X movidos con esa pose. Va de contorno a escáner, en el sentido en que
  ajusta el método.
- **Caso B (descriptivo):** la mediana de la distancia de los vértices de corona de X, movidos
  con esa pose, a la gaussiana del CBCT más cercana con HU ≥ 1800 <!--const:HU_ARBITRO-->, solo
  del lado maxilar del plano de separación. Es de un solo lado: un vértice hundido en el esmalte
  da ≈0, y puntúa casi igual dos poses separadas 2 mm (`consistencia-registro-caso-b.md` §5).
  Además es el mismo conjunto que maximiza el árbitro. Por eso no lleva criterio.

Se usa la mediana sin truncar y no el rms sobre lo solapado. Ese rms, por construcción, solo
cuenta distancias de menos de 1 mm y no puede crecer más allá. Comparar el diente fuera con él
daría por cumplido cualquier resultado.

**Movimiento de X:** la norma del desplazamiento de cada vértice de corona de X entre la pose
completa y la pose sin X. Se informa la mediana, el máximo y la componente vertical media.

**Dientes sin contorno (caso A).** Si la máscara de (b) no da ningún punto de contorno a X, X no
entra en N1 (no hay residuo) pero sí en N2 (su movimiento se mide igual), y se lista.

### 3.3 · Medida 3: arranques desplazados

**Caso A.** La pose de partida del método son `rot0` (sale de los ejes FDI del escaneo) y la
altura del plano oclusal (sale del histograma del esmalte). Las dos dependen de los datos, y eso
es lo que se perturba:

- **Giros:** `rot0 ← Rδ·rot0`, con ángulos de **2°, 5° y 10°** alrededor de las 8 direcciones de
  los vértices de un cubo en los ejes anatómicos del CBCT. Son las direcciones de P1 del protocolo
  de `anatomico`. En total, 24 arranques.
- **Altura:** `plano ← plano + dz`, con dz = **±1, ±3 y ±5 mm**. En total, 6 arranques.

La traslación en x/y no se perturba. El método la recalcula siempre desde los datos (centra la
corona sobre el contorno), así que un arranque desplazado en x/y no es un estado al que el
pipeline pueda llegar. Hacerlo obligaría a copiar el bucle del ICP en el script, y eso mediría la
copia y no el código.

**Caso B.** El ICP clásico arranca en la identidad, a 110° y 61,6 mm de la solución. Perturbar
esa identidad unos grados no cambia la pregunta: eso ya lo dice la variabilidad entre semillas.
La pregunta útil es la **cuenca**: si el ICP empieza cerca de la solución, desplazado una
cantidad conocida, ¿vuelve a ella? El escaneo se mueve en memoria con P·T₀:

- **Giros:** P con los mismos 24 giros que en el caso A, alrededor del centroide de las coronas
  movidas con T₀. Alrededor del origen DICOM el giro añadiría decenas de milímetros de traslación.
- **Traslaciones:** P con **1, 3 y 5 mm** en las 6 direcciones de los ejes. En total, 18.

Se usan las 20 semillas. Cada resultado se compara con el del arranque en caliente sin
perturbar con la misma semilla, así que la submuestra es la misma y el par es exacto.

### 3.4 · Equivarianza

- **Caso A:** 10 movimientos rígidos propios (los de `movimientos()` de `precision_registro.py`)
  aplicados al escaneo en memoria, recalculando los ejes y `prepara_por_cortes`. Se espera
  `T' = T∘G⁻¹`.
- **Caso B:** mover solo el escaneo **es** perturbar el arranque, así que no puede ser una prueba
  de equivarianza. Se mueven a la vez el escaneo y el campo del CBCT con G y se espera `G·T·G⁻¹`.
  Solo valen giros alrededor de z y traslaciones en x/y, porque la separación de arcadas y el
  histograma del plano dependen de z. Son 10 movimientos: el k-ésimo (k = 0…9) gira 36°·k + 10°
  alrededor de z y traslada en x/y lo que dé `default_rng(1001 + k)` uniforme en ±100 mm, sin
  traslación en z. Se aplican en float64 a las coordenadas absolutas del campo
  (`centers + origin`) y a los vértices del escaneo, con la semilla 0.

### 3.5 · Vueltas (caso B)

Una ejecución cuenta como **vuelta** si la mediana del desplazamiento de todos los vértices de
corona del escaneo (X incluido en las ejecuciones sin X) respecto a T₀ es mayor de 5 mm. Es el
valor de `VUELTA_MM` de `precision_registro.py`, pero allí se aplica al máximo y aquí a la
mediana, así que los recuentos no son comparables con los de ese protocolo ni con los de la
medida 1. Las vueltas se cuentan aparte en cada condición y no entran en las medianas:

- **Navaja (descriptiva):** un par en el que cualquiera de los dos registros es una vuelta no
  entra en las medianas. Se informa la tasa de vueltas sin X frente a la de las ejecuciones
  completas. Un diente con menos de 10 pares válidos queda **indeterminado**.
- **K1:** una vuelta cuenta como fallo. Si la referencia en caliente de una semilla es una
  vuelta, sus 28 arranques del criterio cuentan como fallo.
- **Si más de la mitad de las 20 ejecuciones completas desde cero son vuelta respecto a T₀**, T₀
  no representa al camino clásico y nada del caso B se lee, K1 incluido.

Se guarda el umbral HU que elige el árbitro en cada ejecución: los de 1200 y 1900 empatan casi
(0,368 frente a 0,358).

### 3.6 · Suelo de semilla (caso B)

Es el desplazamiento de las coronas de cada diente entre la pose completa con la semilla s y T₀,
para s = 1…19 que no sean vuelta. Es la referencia contra la que se lee el movimiento de X en el
caso B: **X supera el suelo** si la mediana entre semillas de su movimiento sin X es mayor que el
percentil 95 de su propio desplazamiento en el suelo. La medida 1 dio 1,00 y 1,85 mm para otros
escaneos, pero con otro estadístico (todos los pares de semillas, variando solo la submuestra de
`nubes_para_registro`), así que no es la misma cifra.

## 4 · Criterios, fijados en este commit

| | Criterio | Caso A | Caso B |
|---|---|---|---|
| **N1** | Mediana entre dientes del residuo de X con la pose sin X ≤ 2 × la mediana entre dientes del residuo de X con la pose completa | variante (b) | sin criterio (§3.2) |
| **N2** | Ningún diente se mueve más de 1,5 mm <!--const:EPSILON_IOS_CBCT_MM--> (la mediana del desplazamiento de sus vértices de corona, el ε escáner↔CBCT) | variante (b) | sin criterio (§3.2) |
| **K1** | Con giros de hasta 5° y desplazamientos de hasta 3 mm, el registro vuelve a 0,1 mm o menos (mediana del desplazamiento de las coronas) de la pose sin perturbar | 16 giros y 4 alturas | 16 giros y 12 traslaciones × 20 semillas, pares exactos (la submuestra va por índice); una vuelta es fallo |
| **E1** | Equivarianza: el desplazamiento máximo de las coronas respecto a la pose esperada es de 0,01 mm o menos | 10 movimientos | 10 movimientos |

Los 10° y los 5 mm, la variante (a) del caso A, la navaja del caso B (desde cero y en caliente),
las poses de arranque, los recuentos de vueltas y los umbrales del árbitro se informan sin
criterio.

**Cómo se lee cada criterio.** Son de paso o no paso por caso, sin compensación entre ellos.
Una ejecución de K1 que termina sin converger (`converged = false`) cuenta como fallo y se
informa aparte: con un límite de 0,1 mm, el criterio de parada también puede hacerlo fallar, y la
lectura «depende del arranque» vale para las dos causas.

**E1 va antes que todo lo demás.** Es una prueba del programa. Si falla, el script escribe
`equivarianza.json` y `run-record.json` (con `aborted: "E1"`) y se detiene **sin calcular N1, N2
ni K1**. La enmienda se hace sin haberlos visto, y E1 se repite una sola vez.

## 5 · Qué se dirá en cada caso

- **Si el caso A cumple N1, N2 y K1:** el registro por cortes del caso A es estable frente a
  quitar un diente y frente al arranque, dentro de esos márgenes. Se dirá con estas palabras y
  sin decir que es exacto. La dispersión entre umbrales de contorno (1,61 mm de mediana en R1)
  sigue siendo la otra mitad de la incertidumbre y no la mide esto.
- **Si falla N2 (caso A):** se dirá qué dientes se mueven más de 1,5 mm <!--const:EPSILON_IOS_CBCT_MM-->
  sin ellos. Esos dientes no quedan fijados por su vecindad: su posición en el gemelo la decide
  su propio contorno.
- **Si falla N1 (caso A):** el residuo de un diente que no entra en el ajuste es más del doble
  que cuando entra, así que el residuo dentro de la muestra no representa a un diente fuera de
  ella. No es una medida de exactitud.
- **Si falla K1:** el resultado depende del arranque y se dirá desde qué tamaño de desplazamiento.
- **Caso B, navaja:** si el movimiento de X no supera el suelo de semilla (§3.6), se dirá que
  **no se puede atribuir al diente**: lo que se mueve el registro clásico con la submuestra tapa
  lo que mueve un diente. Si lo supera, se informará como movimiento por encima del ruido de
  semilla, sin atribuirle causa.
- **Caso B, K1:** igual que en el caso A, sobre los pares exactos.

**Lo que se espera, dicho antes.** En el caso A, con solo 4 cortes en la franja, se espera que
se muevan más los dientes de los extremos (17 y 27, con más brazo de palanca) y los que dejan
pocos puntos de contorno. En el caso B se espera que el ruido de semilla domine la navaja y que
K1 falle en los giros de 5°, porque el ICP clásico no tiene recorte (`trim=1.0`) y lo arrastra la
encía sin pareja. Escribirlo ahora es para que no parezca una lectura hecha a la vista del
resultado.

## 6 · Lo que no se hará

- **No se ajusta nada con estos resultados.** Ni `por_cortes`, ni el camino clásico, ni una
  constante, un umbral o una guarda, ni el camino por defecto. Tampoco se elige entre R1, R2 y
  R2b. Si algo se cambiara después de verlos, los puntos anatómicos del caso A dejarían de medir
  el método cambiado ([`registro-caso-a-puntos-preregistro.md`](registro-caso-a-puntos-preregistro.md) §8).
- **No se usan para excluir puntos anatómicos** ni entran en la lectura fijada de §7 de ese
  pre-registro.
- **Se ejecutan antes de abrir ninguna entrega de las anotadoras.** La carpeta de entregas está
  fijada aquí, `data/processed/puntos-anatomicos/anotaciones/` del checkout principal. El script
  la comprueba él mismo, sin recibirla como argumento, y se niega a ejecutar el caso A si hay
  cualquier fichero debajo. Hoy está vacía. El run-record guarda la ruta relativa y que estaba
  vacía, no los nombres de sus subcarpetas.
- **En el caso B no se ejecuta `fusion_agents.anatomico`.** Su protocolo exige congelarse antes
  ([`registro-anatomico-protocolo.md`](registro-anatomico-protocolo.md)). Estas ejecuciones del
  camino clásico sobre B se añaden a la exposición previa de su §3.
- **Una sola ejecución** por caso, desde este commit (o desde una enmienda) con el árbol limpio.
  El script se niega si este documento tiene cambios sin confirmar, y guarda su blob de git. La
  salida va a una carpeta con el commit en el nombre, el script se niega a sobrescribirla, y nada
  se repite en silencio: un intento fallido queda en disco junto al siguiente.

## 7 · Lo que no se podrá afirmar

- **El TRE o la exactitud.** Esto es precisión.
- **Independencia en el caso A.** `por_cortes` se ajustó sobre esta misma serie
  ([`registro-por-cortes.md`](registro-por-cortes.md) §5).
- **Que se generalice.** Son dos casos con dos métodos distintos.
- **Que cierre el `\todo` del artículo** sobre la exactitud del registro. Se puede citar como
  precisión con su nombre.

## 8 · Huellas y salidas

- El `run-record.json` de cada caso se escribe al empezar (`status: "running"`) y se reescribe
  al terminar, también si algo falla (`ok`, `aborted` o `error`, con el tipo en
  `error_type`). Un fallo al cargar los datos de referencia también queda en disco. Guarda:
  - `git_head`, `git_dirty` y el blob de este documento;
  - la sha256 de los valores resueltos de las constantes en mayúsculas de `por_cortes`,
    `preparacion`, `registration`, `geometric`, `export_agents.anatomia` y de las que usa
    `precision_registro.movimientos()`, leídas sin tocar los módulos;
  - las sha256 de las entradas: artefactos, contenedor de referencia, etiquetas y el
    `run-record.json` de la ejecución de referencia con su `git_head`;
  - las semillas y la versión de `geometric-fusion-agent`;
  - los argumentos ya interpretados, con las rutas sustituidas por su papel (el analizador no
    acepta abreviaturas, para que ninguna ruta se escape).
- Salidas en `data/processed/registro-navaja-arranques/<caso>-<commit>/`. Hay un JSON por etapa,
  que se escribe antes de imprimir nada.

## 9 · Resultados

_Se escriben aquí después de la ejecución, con el commit y el árbol limpio._
