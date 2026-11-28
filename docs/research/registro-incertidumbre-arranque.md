# Registro por cortes: declarar cuánto depende la pose del arranque, y si lo declarado cubre el error del fantasma

> **Estado (2026-11-28): ejecutado; resultados en §9.** P1, P3 y P4 pasan: la declaración del
> arranque **se queda**, encendida por defecto. **P2 falla (0 de 7)**: lo declarado no cubre el
> error del fantasma, así que la línea del umbral dice ya que es precisión y no acota el error.
> R1 declara U = 1,62 mm. El pre-registro se fijó en `2fe061d` **antes de ejecutar** la batería
> del fantasma (§2.1) y la declaración de R1 (§3.3). Las dos se ejecutan una sola vez, desde este
> commit y con el árbol limpio. **Ya se ha visto** el resultado del caso A en la medida 3
> ([`registro-navaja-arranques.md`](registro-navaja-arranques.md) §9.1), y se ha visto el tiempo
> de una configuración de la batería, F1 (§5, «Lo que ya se ha visto»). **Ejecutar F1 antes de
> este commit fue contra la regla** de que la batería solo se ejecuta desde él: se ejecutó una
> vez, al implementar, y se vio su P1 (0,0 mm) y sus dos tiempos (×9,3); **E y U de F1 no se
> imprimieron ni se miraron**, y su P1 entra en P1 con ese dato conocido. **Enmienda del
> 2026-11-28, antes de este commit, por decisión que el responsable del proyecto delegó:** el
> P3 del borrador (en el fantasma, tiempo con arranques ≤ 5× sin ellos) **se retira** y se
> sustituye por un límite absoluto sobre R1 (§4). Se dice cómo fue: aquel P3 estaba mal
> elegido, porque la razón depende de la estructura (3 registros pasan a 11) y de cuántas
> iteraciones necesitan los arranques girados, no del coste para el usuario, y 0,17 s, la
> llamada entera a `prepara_por_cortes` sin la medida en F1, no dice nada del coste en el
> producto; y el cambio se hizo **después** de ver el ×9,3 de F1. **El cambio de criterio
> cambia la decisión esperada:** con el P3 y la regla del borrador, la declaración muy
> probablemente no se habría quedado (`mide_arranque` a `False`); con el P3 nuevo se predice
> que se queda (§5). En la misma enmienda P2 se acota a las
> configuraciones en las que el umbral puede mover el contorno (§2.1, §4). El tamaño del giro,
> 5° <!--const:GIRO_ARRANQUE_GRADOS-->, y las 8 direcciones vienen del pre-registro de la
> medida 3, fijado antes de ejecutar nada sobre el caso A; aquí se eligieron **después** de ver
> su resultado, pero no se ajustaron a él. Todo lo que se declara es **precisión**, no exactitud.
> No sustituye al TRE con puntos anatómicos
> ([`registro-caso-a-puntos-preregistro.md`](registro-caso-a-puntos-preregistro.md)).

## 1 · Las preguntas

La medida 3 mostró que el registro por cortes (`fusion_agents.por_cortes`) del caso A no vuelve a
la misma pose cuando su rotación de partida `rot0` se gira 2–5°: se queda a 0,11–0,41 mm, con el
mismo rms. El residuo no distingue esas poses. La corrección honesta en el producto **no es
cambiar la pose** sino **declararlo**, junto a lo que ya se declara: cuánto se mueve la pose con
el umbral del contorno (1100 y 1900 frente a 1400 HU <!--const:UMBRAL_CONTORNO-->, en
`threshold_spread_mm`, que va al gate).

Con eso el registro declara dos dispersiones. Las dos son precisión: ninguna ve un sesgo que
compartan todas las poses. Las preguntas:

- **Principal.** En el fantasma `fantasma_arcada`, que **sí tiene verdad**, ¿cubre la
  incertidumbre declarada el error verdadero de las coronas?
- **Que la pose no cambia.** Con la medida de arranque encendida, ¿es la pose devuelta la misma,
  bit a bit, que con ella apagada? En el fantasma y en R1.
- **Coste.** ¿Cuánto tiempo añaden a `prepara_por_cortes` los ocho registros de más en un
  caso real, R1? En el fantasma solo se informa la razón, sin criterio.
- **R1.** ¿Qué incertidumbre declara el código para R1, el registro del caso A que puntuarán los
  puntos anatómicos? Se fija aquí, antes de que llegue ningún punto, para poder compararla con
  su TRE (enmienda del 2026-11-28 del pre-registro del TRE).

## 2 · Los datos

### 2.1 · La batería del fantasma, fija

Doce configuraciones de `packages/fusion-agents/tests/fantasma_arcada.py`, tal como lo deja este
commit. Todas son series con huecos en la banda, que es donde `por_cortes` registra: en una serie
densa lanza `SerieDensa` y el pipeline sigue por el camino clásico. Las nueve primeras tras F1
repiten configuraciones que ya usan los tests o las notas; F11 y F12 son combinaciones nuevas de
parámetros existentes, porque la cabeza inclinada y la cáscara de esmalte solo se habían usado en
series densas. Ninguna se ha ajustado.

`campo` son los argumentos de `campo_cbct` (siempre `paso=0.4`); `escaneo`, los de
`escaneo_superior`. «0,45/2,2–3,9» es `paso_z=0.45, quita=(2.2, 3.9)`; «0,8/2–4» es `paso_z=0.8,
quita=(2.0, 4.0)`. «Capa» es `esmalte="capa", desenfoque_mm=0.18`.

| | Campo | Escaneo | Plano | De dónde sale |
|---|---|---|---|---|
| F1 | losa: la del caso A sintético de la medida 3 (abajo) | `escaneo_superior()` | modo del esmalte | `tests/test_registro_navaja_arranques.py`, `caso_a_sintetico` |
| F2 | 0,8/2–4, raíces cónicas, esmalte macizo | `escaneo_superior()` | 0,0 | `test_anatomico.py`, `test_recupera_la_pose_en_una_serie_dispersa` |
| F3 | 0,45/2,2–3,9, `raices="realistas"` | `escaneo_superior()` | 0,0 | [`registro-anatomico.md`](registro-anatomico.md) §8.1 |
| F4 | F3 + capa | `escaneo_superior()` | 0,0 | §8.1 |
| F5 | F4 + `oclusion="sobremordida"` (valores de libro) | `escaneo_superior()` | 0,0 | §10 |
| F6 | F4 | `encia="cuello", corona="corta", desborde_mm=0.0` | 0,0 | §11 |
| F7 | F4 | `encia="cuello", corona="corta", desborde_mm=1.0` | 0,0 | §11 |
| F8 | F4 | `encia="cuello", corona="corta", desborde_mm=3.0` | 0,0 | §11 |
| F9 | F3 (esmalte macizo, sin desenfoque) | `encia="cuello", corona="corta", desborde_mm=3.0` | 0,0 | §11 |
| F10 | F4 | `encia="cuello", corona="normal", desborde_mm=3.0` | 0,0 | §11 |
| F11 | F2 + `inclinacion` = 10° alrededor de x | `escaneo_superior()` | 0,0 | nueva: la inclinación de `test_anatomico.py` §7 sobre la serie de F2 |
| F12 | 0,45/2,2–3,9, `raices="realistas", esmalte="cascara", dentina=1200.0` | `escaneo_superior()` | 0,0 | nueva: la cáscara de [`registro-anatomico.md`](registro-anatomico.md) §12 sobre la serie de F3 |

**F1, la losa.** Rejilla de 0,4 mm en x ∈ [−32, 32) e y ∈ [−16, 20), cortes cada 0,45 mm de −3 a
9 mm sin los que caen entre 2,2 y 3,9, las coronas de las dos arcadas (`_dentro`, sin raíces) a
2200 HU, y una losa de esmalte a 2200 HU en x ∈ [−30, 30), y ∈ [30, 46), z = 0,15 mm, lejos de
la arcada. Origen (3, −2, 1,5) y `centers` en float32. Es la única configuración en la que el
plano oclusal sale del modo del esmalte, como en el pipeline: la losa pone el modo en el plano del
fantasma.

**El plano en las demás.** Con `campo_cbct`, el modo de la altura del esmalte cae dentro de la
arcada inferior (−5,3 a −8,0 mm frente al plano verdadero en 0, comprobado al preparar la
batería sin registrar nada; es el fallo que [`registro-anatomico.md`](registro-anatomico.md)
describe en su §4), la banda queda en la
arcada equivocada y la serie deja de ser dispersa en ella. Por eso F2–F12 reciben `plano=0.0`, el
plano del fantasma, como hace `test_por_cortes.py`. **El error del plano estimado no entra** ni en
E ni en U de esas configuraciones, **salvo en F11**: allí `plano=0.0` es el plano de la boca
*antes* de inclinarla. `campo_cbct` gira la boca 10° alrededor de x por el origen, y las coronas
van de y ≈ −10 mm (incisivos) a y ≈ +5 mm (molares), así que el plano oclusal verdadero se separa
de z = 0 hasta unos 1,7 mm en los incisivos y 0,9 mm en los molares. Ese error de plano es parte
de F11, y la banda la recibe.

**El escáner y la verdad.** En todas, el escaneo está en su propio marco: `fichero = R·mundo + t`
con R = 97° alrededor de (0,3, 1,0, 0,2) y t = (40, −12, 80) mm, el movimiento de
`test_anatomico.py`. El registro es equivariante a ese movimiento (E1 de la medida 3, 6·10⁻¹³ mm),
así que un solo movimiento basta. La verdad de cada vértice es su posición en el mundo del CBCT;
con `inclinacion` (F11), la boca girada con la misma matriz que el campo. Las **coronas
verdaderas** son los vértices con FDI > 0 del mismo escaneo sin desborde: con `desborde_mm > 0`
el registro usa también la encía etiquetada como diente, que es lo que hace el segmentador real,
pero el error se mide solo sobre corona. El desborde solo cambia etiquetas; el script comprueba
que los vértices son los mismos.

Si una configuración lanza **cualquier excepción** (`SerieDensa`, otro `ValueError`, o un fallo
al medir después de registrar), se escribe su JSON con el tipo, el mensaje y la traza, se sigue
con las demás y **cuenta como no cubierta** dentro de su grupo (abajo). Solo una interrupción
(`KeyboardInterrupt`, `SystemExit`) termina la ejecución, que queda como `error` en su
`run-record.json`.

**Dos grupos: con y sin gradiente en la banda.** El destino de cada umbral son los puntos de
borde de `hu ≥ umbral` en los cortes de la banda (`contornos`), filtrados por la cercanía al
esmalte de ≥ 1800 HU, que no depende del umbral. Si **ningún vóxel de la banda tiene
1100 ≤ HU < 1900**, los tres umbrales eligen los mismos puntos, el destino es el mismo y, como el
registro es determinista, `threshold_spread_mm` sale 0,00: **U_umbral = 0 por construcción** y U
se reduce a U_arranque. Eso se sigue del código, no de ejecutar nada:

- en `campo_cbct`, sin desenfoque y con esmalte macizo, los únicos valores son 2200 (corona) y
  la dentina de la raíz (1600). La corona superior llega hasta 3 mm por encima de su centro
  (z < 9 mm) y ahí empieza la raíz, cónica o realista; con plano 0 la banda es
  z ∈ [1,5; 6] mm, y la corona inferior acaba en z = 0,5 mm. En la banda solo hay 2200;
- en F11 la boca gira 10° alrededor de x: un punto de la rejilla en la banda (z ≤ 6 mm,
  y ≥ −16 mm) cae, en la boca sin girar, a z = 6·cos 10° + 16·sen 10° ≈ 8,7 mm como mucho, por
  debajo de los 9 mm de la raíz. Tampoco entra raíz;
- F1 es todo 2200, esté donde esté el plano.

Son **F1, F2, F3, F9 y F11**: cinco. El borrador de este documento decía cuatro (F1, F2, F3 y
F9) y daba F11 como posible gradiente sin hacer la cuenta de arriba. En las otras siete hay
valores entre los umbrales: con capa y desenfoque (F4–F8, F10), la dentina de la corona a 1600 y
los bordes desenfocados; en F12, el núcleo de la cáscara a 1200. Se comprobó, antes de este
commit, **construyendo solo los campos**, sin escanear ni registrar nada: 0 vóxeles entre 1100 y
1900 HU en la banda de F2, F3, F9 y F11, un solo valor (2200) en todo el campo de F1, y entre
1956 y 2693 vóxeles en la banda de cada una de las otras siete. El grupo va en la batería
(`"gradiente"` en `BATERIA`); un test lo repite, y el script lo vuelve a contar sobre la banda
real de cada configuración y lo informa (`gradient_declaration` en `criterios.json`), sin que
eso cambie los grupos fijados aquí.

### 2.2 · R1

R1 es `run-6` del caso A, la ejecución que puntuarán los puntos anatómicos: commit `57eb021`,
árbol limpio, contenedor con SHA-256
`239e258e077d7b53a561740d67d8188705992dd00a3e992f38b5d05819f8034a`. Se carga **sin reingerir**,
con los cargadores de `scripts/registro_navaja_arranques.py` (importados, no copiados): el campo
semilla y la superficie del escáner del almacén de artefactos, y las etiquetas de
`derived/seg_teeth` y `reg.ct_to_ios` del contenedor. Es exactamente la entrada de la medida 3,
cuyo V0 reprodujo R1 a 2·10⁻¹⁴ mm.

## 3 · Procedimiento, fijado antes de ejecutar

### 3.1 · Lo que cambia en `por_cortes`

`prepara_por_cortes`, después de los umbrales, registra otra vez desde 8 arranques girados:
`rot0 ← Rδ·rot0`, con Rδ un giro de 5° <!--const:GIRO_ARRANQUE_GRADOS--> alrededor de cada
diagonal del cubo (±1, ±1, ±1)/√3 en los ejes anatómicos del CBCT (derecha, anterior, superior)
llevados al DICOM con `EJES_LPS`. Es la construcción de la medida 3, en el mismo orden; un test
comprueba que las matrices son las mismas bit a bit. Cada arranque es un registrador **nuevo**
(`dataclasses.replace`) sobre el **mismo destino** (el contorno de 1400 HU). El arranque en x/y y
en altura se recalcula desde los datos, como en la medida 3.

Por cada arranque, el desplazamiento de cada vértice de corona respecto a la pose de base, igual
que se hace con los umbrales. «Vértice de corona» es aquí lo que el registro recibe como origen:
**los vértices del escáner con FDI > 0**, con el desborde sobre la encía incluido cuando lo hay
(F7–F10), porque es lo que hace el segmentador real. En `informe["start_spread_mm"]`:
`angle_deg`, `n_starts`, la mediana y el máximo de todos esos desplazamientos juntos (los 8
arranques sobre los mismos vértices), `vertical` (el mayor, entre arranques,
del valor absoluto de la media de Δz) redondeados a 0,01 mm como los del umbral, y
`not_converged`, cuántos arranques terminan sin converger. Los que no convergen **entran** en la
dispersión y se cuentan.

En el gate, una línea más: cuánto se mueven las coronas con la orientación de partida (8
arranques girados 5° <!--const:GIRO_ARRANQUE_GRADOS-->), mediana, máximo y altura, y que **es
precisión y no acota el error**. La línea del umbral no cambia en esta versión (§5).

Lo que **no** cambia: el registrador, el origen y el destino que se devuelven son los de antes, y
la pose sale de ellos; la confianza; el contrato UOS (el número viaja en los motivos del gate y en
el informe del registro). `prepara_por_cortes(..., mide_arranque=False)` lo apaga para quien no
quiera pagar los ocho registros; por defecto está encendido.

### 3.2 · Por configuración del fantasma

1. **Tiempos (descriptivos).** La base anatómica del escáner (`anatomical_frame`) se calcula una
   vez, fuera del cronómetro. Una llamada a `prepara_por_cortes` sin la medida, **sin
   cronometrar**, calienta. Después se cronometra **solo** `prepara_por_cortes`
   (`time.perf_counter`, una vez cada una): primero **sin** la medida de arranque, luego **con**
   ella. Las llamadas son las del pipeline (las etiquetas del escáner y su base), más el `plano`
   de §2.1. La razón se informa sin criterio: el coste que decide es P3, en R1 (§3.3).
2. **P1.** Los dos resultados devuelven el mismo origen y destino byte a byte, un registrador del
   mismo tipo con **todos** sus campos idénticos (`dataclasses.fields`: `rot0`, `plano`, `arcada`,
   `trim`, `max_iter`, `tol_mm`, `overlap_mm` y `extra`), y la pose de cada uno (el registrador
   aplicado al origen y al destino) mueve las coronas 0,0 mm de una a otra.
3. **Error verdadero E.** La pose con la medida aplicada a los vértices de corona verdaderos;
   E = mediana, sobre esos vértices, de |corona registrada − corona verdadera|. También el p95, el
   máximo, y la componente vertical con signo (mediana y media de Δz en el DICOM; positiva hacia
   +S, que en la arcada superior es hacia la raíz).
4. **Incertidumbre declarada U**, de los números del informe, redondeados como los ve el gate:
   - U_umbral = mediana de `threshold_spread_mm`; U_arranque = mediana de `start_spread_mm`.
     Cada una es la mediana sobre **los vértices con FDI > 0 que recibe el registro** (§3.1),
     juntando los 2 umbrales o los 8 arranques. **E se mide sobre otra población**: solo las
     coronas verdaderas (§2.1). En F1–F6, F11 y F12 coinciden; en F7–F10 la de U lleva además la
     encía desbordada;
   - **primaria: U = √(U_umbral² + U_arranque²)**;
   - con los máximos, U_máx = √(máx_umbral² + máx_arranque²), y en altura con `vertical`: se
     informan, sin criterio.
   - Si el informe no trae una de las dos (los dos umbrales alternativos con menos de
     500 puntos de contorno <!--const:MINIMO_CONTORNO-->), esa componente cuenta 0 y se dice.
5. **Cubierta** si E ≤ U. En las configuraciones sin gradiente (§2.1) se informa además si
   E ≤ U_arranque, que allí es lo mismo.

### 3.3 · R1

1. Antes de cargar nada, el script se niega si hay **cualquier fichero** bajo
   `data/processed/puntos-anatomicos/anotaciones/` de la copia principal (hoy solo tiene dos
   subcarpetas vacías). La U de R1 se fija antes de ver ningún punto.
2. Comprueba R1 como la medida 3 (commit, `git_dirty: false`, SHA-256 del contenedor) y se niega
   si esa medida ya no apunta a R1.
3. **V0.** Prepara con la llamada del pipeline tal cual (`rn.prepara_a`, con la medida de
   arranque encendida por defecto) y registra. Si las
   coronas quedan a más de 0,01 mm de `reg.ct_to_ios` de R1, escribe `V0.json` y se detiene.
4. **Tiempo (P3).** La base anatómica del escáner se calcula una vez, fuera del cronómetro. Una
   llamada a `prepara_por_cortes` sin la medida, **sin cronometrar**, calienta. Después se
   cronometra **solo** `prepara_por_cortes` cuatro veces, en este orden fijo: **sin, con, sin,
   con** (`ORDEN_P3`). De cada variante se queda el **mínimo** de sus dos tiempos, y P3 mide la
   diferencia: mínimo con la medida menos mínimo sin ella.
5. **P1 sobre R1**, como en §3.2: lo que devuelve la llamada del pipeline de V0 frente a la
   última llamada sin la medida del paso 4.
6. Escribe U, U_máx, U en altura, `threshold_spread_mm`, `start_spread_mm`, P1, P3 con los dos
   mínimos y su diferencia, el informe y los motivos del gate en `declaracion.json`.

## 4 · Criterios, fijados en este commit

| | Criterio | Fantasma | R1 |
|---|---|---|---|
| **P1** | La pose con la medida de arranque es la misma que sin ella: desplazamiento máximo de las coronas 0,0 mm, y origen, destino y registrador idénticos byte a byte | cada configuración que registra | sí |
| **P2** | E ≤ U (primaria) en al menos el 80 %, redondeado hacia arriba, de las configuraciones **con gradiente** (§2.1: F4, F5, F6, F7, F8, F10 y F12): **6 de las 7**. Una que lanza cuenta como no cubierta. Las cinco sin gradiente (F1, F2, F3, F9, F11) se informan aparte (E, U_arranque y si E ≤ U_arranque), sin criterio | sí | — |
| **P3** | Coste: el tiempo que la medida de arranque añade a `prepara_por_cortes` (§3.3: mínimo de dos con ella menos mínimo de dos sin ella, la base anatómica fuera del cronómetro, tras una llamada de calentamiento) es como mucho **120 s** | — (la razón se informa sin criterio) | sí |
| **P4** | Cordura: toda configuración que registra trae `start_spread_mm` con mediana, máximo y altura finitos y `not_converged` entero. Un arranque sin converger **no** es fallo: se cuenta | sí | — |
| **V0** | La pose reconstruida está a 0,01 mm o menos de `reg.ct_to_ios` | — | puerta |

P1 sobre R1 es parte de P1: si falla en el fantasma o en R1, P1 falla. Cada fichero nombra la
parte que decide: `criterios.json` lleva `phantom_part` (`P1_phantom` y `P4_phantom` deciden;
P2 solo la redacción) y `declaracion.json` lleva `r1_part` (`P1_r1` y `P3_r1`). La decisión de
§5 los junta con `decision_de_envio` del script. **Si P1 no se evalúa en R1** (`r1` no se ejecuta
antes de la primera entrega de puntos, o se detiene en V0), esa parte de P1 se decide con el
fantasma y se dice; pero entonces **P3 tampoco se evalúa**, y sin P3 la declaración no se queda
(§5). U_máx, la altura y el tiempo del fantasma se informan sin criterio.

**Por qué 120 s, dicho antes.** Lo que se espera sale de la medida 3 sobre el caso A, con las
horas a las que escribió cada etapa (`written_at`). Su etapa de arranques hizo 30 registros
completos (24 giros y 6 alturas, 2275 iteraciones en total) en 61,2 s, unos 2 s por registro
y 0,027 s por iteración; la de la navaja, 28 registros en 46,6 s, lo mismo por iteración. Los
ocho arranques de 5° de esa etapa son los mismos ocho de aquí y sumaron 574 iteraciones: unos
**16 s**. Es una cuenta, no una medida de `prepara_por_cortes`, pero sale de la misma
ejecución y del mismo caso. **El límite de 120 s, unas 7 veces eso, se mantiene sabiéndolo**,
como presupuesto de ingeniería frente a una ejecución del caso completo que ya tarda muchos
minutos, **no como una prueba científica**. Se dice cómo se llegó: el límite se fijó al
redactar la enmienda con una cuenta equivocada, unos 30 registros en los 3 minutos de toda la
ejecución y unos 50 s para los 8 arranques, cuando la ejecución hizo unos 102 registros (V0 4,
equivarianza 40, navaja 28, arranques 30). La cuenta se corrigió en una revisión, antes de
este commit, y el límite no se cambió.

## 5 · Qué se dirá en cada caso

**La regla de decisión.** La declaración **se queda, encendida por defecto**, si y solo si P1
pasa en el fantasma, P1 pasa en R1 (o no se evalúa allí, y se dice), **P3 pasa en R1** y P4 pasa
en el fantasma. **P3 sin evaluar** (`r1` no se ejecuta o se detiene en V0) **es no se queda**,
porque su coste en un caso real sería desconocido. Si no se queda, en el commit de resultados el
valor por defecto de `mide_arranque` pasa a `False`, el pipeline deja de declarar el arranque, y
`AGENTS.md` lo dice con el criterio que falló o que no se evaluó. **P2 no decide si se queda;
decide la redacción:**

- **Si P2 pasa:** no se dice nada más allá de «en el fantasma cubrió el error en X de las 7
  configuraciones con gradiente». No es una cota del error en un caso real.
- **Si P2 falla:** en el commit de resultados, la línea del umbral de `motivos()` que hoy termina
  «Esa es su incertidumbre, no el residuo» se cambia para decir que es precisión y que **no acota
  el error**, y `AGENTS.md` lo dice. La línea del arranque ya lo dice.

**R1.** Se informa su U con estas palabras: «la incertidumbre que el registro declara para R1»,
sin llamarla error. Su lectura frente al TRE es la de la enmienda del 2026-11-28 del pre-registro
del TRE, y nada más.

**Lo que se espera, dicho antes.** Solo lo que se sigue del código:
- **En F1, F2, F3, F9 y F11, U_umbral = 0,00 exactamente** (§2.1), así que allí U = U_arranque.
  Si E queda por debajo de U_arranque no se predice: el registro por cortes tiene un sesgo vertical conocido
  hacia la raíz (0,37 mm con rejilla de 0,3 mm, docstring de `por_cortes`), que comparten todas
  las poses y que ninguna dispersión ve, pero su tamaño en estas configuraciones no se sigue del
  código.
- **P2 no se predice.** En las siete con gradiente el umbral cambia el contorno y U_umbral
  no es 0 por construcción, pero si U llega a E depende de cuánto lo mueva frente al sesgo, y eso
  no se puede leer del código.
- **P3:** unos 16 s en R1 (§4), muy por debajo del límite. Sale de las iteraciones de la
  medida 3, no de cronometrar `prepara_por_cortes`.
- **P1 y P4 pasan.** Los registros de más no tocan lo que se devuelve, y el registrador no tiene
  azar.
- **R1:** U_umbral ≈ 1,61 mm (la mediana de `threshold_spread_mm` que ya dio el V0 de la medida 3)
  y U_arranque del orden de 0,2–0,4 mm (los mismos ocho arranques de 5° de la medida 3: 0,18–0,41 mm
  de mediana por arranque), así que U ≈ 1,6–1,7 mm, dominada por el umbral.
  Queda por encima de los 1,5 mm del criterio A1 del TRE. <!--const:EPSILON_IOS_CBCT_MM-->

**Lo que ya se ha visto.**
- La medida 3 completa sobre el caso A (§9.1 de su nota), y con ella el `threshold_spread_mm` de
  R1 (1,61 mm de mediana) y los ocho arranques de 5°. La U de R1 es casi predecible, y se dice.
- Al implementar, F1 se ejecutó **una vez** para medir el tiempo: P1 0,0 mm, 0,17 s sin la medida
  y 1,54 s con ella (×9,3). Ese tiempo se tomó con el procedimiento anterior a este commit (la
  base anatómica dentro del cronómetro y sin llamada de calentamiento); §3.2 lo cambió después,
  a raíz de una revisión, sin volver a ejecutar nada. **E y U de F1 no se imprimieron ni se
  miraron.** Con ese ×9,3 a la vista era previsible que el P3 del borrador (5× en el fantasma)
  fallara, y el propio borrador lo predecía; con su regla, eso apagaba la declaración. **Se
  cambió después de verlo**, por la enmienda de la cabecera, y con el P3 nuevo se predice lo
  contrario: que se queda. El P3 nuevo se mide en R1, sobre el que ese tiempo no informa.
- Al preparar la enmienda se construyeron los **campos** de las doce configuraciones, sin
  escaneo ni registro, para contar los vóxeles de la banda entre 1100 y 1900 HU (§2.1). Eso es
  la entrada, no el resultado: no dice nada de E ni de U_arranque.
- Los tests unitarios ejecutan la medida sobre fantasmas que **no** están en la batería (el de
  `test_por_cortes.py` y una losa más gruesa que la de F1); comprueban estructura y P1, no E ni U.

## 6 · Lo que no se hará

- **No se cambia la pose.** Ni `por_cortes`, ni su arranque, ni una constante, ni la confianza. Los
  registros de más solo se miden. Si la pose de R1 cambiara, sus puntos anatómicos dejarían de
  medirla ([`registro-caso-a-puntos-preregistro.md`](registro-caso-a-puntos-preregistro.md) §8).
- **No se ajusta U con el fantasma.** Ni el giro, ni las direcciones, ni la combinación de las dos
  dispersiones, ni un factor que las haga cubrir. Si P2 falla, cambia la redacción, no el número.
- **No se corrige el sesgo vertical restando**, como ya dice `por_cortes`: sería ajustar a un
  sintético.
- **No se toca la medida 3** ni su script: está gastada. Su lógica no cambia, pero **su
  comportamiento sí**: `rn.prepara_a` llama a `prepara_por_cortes` sin el interruptor, así que
  ahora paga los ocho registros de más y deja `start_spread_mm` en el informe de su V0, y la
  huella de constantes que guardaría una nueva ejecución cambia, porque incluye las de
  `por_cortes`. Sus tests tardan unos 35 s más; se dejan así a propósito, para que sigan
  ejecutando lo que el script hace de verdad.
- **Una sola ejecución** de cada subcomando, desde este commit con el árbol limpio. El script se
  niega si este documento no está en git o tiene cambios sin confirmar, y guarda su blob. Se niega
  si ya existe **cualquier** salida de ese subcomando (`<subcomando>-*`), venga del commit que
  venga: después del commit de resultados no se puede repetir. Un intento fallido queda en disco
  y también bloquea; repetir exigiría una enmienda que lo diga.
- **`r1` no se ejecuta después de que llegue ningún punto anatómico.** El script lo comprueba.

## 7 · Lo que no se podrá afirmar

- **Que U acota el error en un caso real.** Aunque P2 pase, el fantasma no tiene el ruido, los
  artefactos ni la anatomía de un CBCT real, y el plano se le da en F2–F12.
- **El TRE o la exactitud de R1.** La U de R1 es precisión declarada.
- **Que el coste medido en R1 valga para otro caso**: depende de cuántas iteraciones necesiten
  los arranques girados, y R1 es un solo caso.
- **Validación clínica.** Nada de esto lo es.

## 8 · Huellas y salidas

- [`scripts/incertidumbre_arranque.py`](../../scripts/incertidumbre_arranque.py), subcomandos
  `fantasma` y `r1`. Se niega con el árbol de código sucio, con este documento sin versionar o
  con cambios, con cualquier salida ya existente de ese subcomando y, `r1`, con cualquier
  fichero en la carpeta de
  entregas o con otra ejecución de referencia. El analizador no acepta abreviaturas.
- El `run-record.json` se escribe al empezar (`status: "running"`) y se reescribe al terminar,
  también si algo falla (`ok`, `aborted` o `error`, con el tipo en `error_type`). Guarda:
  `git_head`, `git_dirty`, el blob de este documento; la sha256 de los valores resueltos de las
  constantes en mayúsculas de `por_cortes`, `preparacion`, `registration`, `geometric`,
  `export_agents.anatomia`, `fantasma_arcada` y del propio script (la batería incluida); la sha256
  del fichero del fantasma o, en `r1`, las de las entradas de R1; la versión de
  `geometric-fusion-agent`; la guarda de entregas (`r1`), y los argumentos ya interpretados con
  las rutas sustituidas por su papel.
- Salidas en `data/processed/incertidumbre-arranque/<subcomando>-<commit[:12]>/`: `fantasma` escribe
  un JSON por configuración (`F1.json` … `F12.json`) y `criterios.json` (P1, P2, P4, las
  configuraciones sin gradiente y la razón de tiempos, con `phantom_part`); `r1`, `V0.json` y
  `declaracion.json` (U, P1 y P3, con `r1_part`). Cada JSON se escribe antes de imprimir nada de
  él. La decisión del commit de resultados sale de `decision_de_envio` sobre los dos.

## 9 · Resultados

Ejecutado una sola vez cada subcomando el 2026-11-28 desde `2fe061d`, con el árbol limpio
(`git_dirty: false`, blob del protocolo `5112bd7`). Los dos terminan con `status: ok`. Salidas en
`data/processed/incertidumbre-arranque/fantasma-2fe061d6d7a4/` y `r1-2fe061d6d7a4/` (sin
versionar). Las dos ejecuciones se solaparon en el tiempo: el cronómetro de P3 en R1 compartió
CPU con la batería, lo que solo puede alargar los tiempos, es decir, juega en contra de P3.

### 9.1 · Criterios

| | Resultado | Límite | |
|---|---|---|---|
| V0 (R1) | máx 2·10⁻¹⁴ mm | ≤ 0,01 mm | sí |
| P1 fantasma | 0,0 mm en las 12; origen, destino y todos los campos del registrador idénticos | 0,0 mm | **sí** |
| P1 R1 | 0,0 mm; ídem | 0,0 mm | **sí** |
| P2 | E ≤ U en **0 de 7** configuraciones con gradiente | ≥ 6 de 7 | **no** |
| P3 (R1) | +14,8 s (mínimos 5,0 s sin y 19,8 s con) | ≤ 120 s | **sí** |
| P4 | las 12 finitas, 0 arranques sin converger | — | **sí** |

**Decisión (§5):** la declaración **se queda, encendida por defecto** (`mide_arranque=True`).
Como P2 falla, la línea del umbral de `motivos()` deja de decir «Esa es su incertidumbre, no el
residuo» y dice «Es precisión, no el residuo: no acota el error», y `AGENTS.md` lo dice.

### 9.2 · El fantasma

| | Grupo | E (mm) | E vertical (mm) | U_umbral | U_arranque | U (mm) | E ≤ U |
|---|---|---:|---:|---:|---:|---:|---|
| F1 | sin gradiente | 0,82 | 0,81 | 0,00 | 0,41 | 0,41 | no |
| F2 | sin gradiente | 0,45 | 0,45 | 0,00 | 0,61 | 0,61 | sí |
| F3 | sin gradiente | 0,61 | 0,61 | 0,00 | 0,01 | 0,01 | no |
| F4 | gradiente | 0,61 | 0,61 | 0,01 | 0,01 | 0,01 | no |
| F5 | gradiente | 0,61 | 0,61 | 0,02 | 0,05 | 0,05 | no |
| F6 | gradiente | 0,62 | 0,62 | 0,26 | 0,01 | 0,26 | no |
| F7 | gradiente | 0,61 | 0,61 | 0,25 | 0,12 | 0,28 | no |
| F8 | gradiente | 0,61 | 0,61 | 0,25 | 0,24 | 0,35 | no |
| F9 | sin gradiente | 0,61 | 0,61 | 0,00 | 0,24 | 0,24 | no |
| F10 | gradiente | 0,61 | 0,61 | 0,01 | 0,03 | 0,03 | no |
| F11 | sin gradiente | 1,55 | 1,37 | 0,00 | 0,38 | 0,38 | no |
| F12 | gradiente | 0,70 | 0,69 | 0,05 | 0,50 | 0,50 | no |

E vertical es la mediana de la componente vertical del error; U_umbral y U_arranque son medianas en mm.

- **El error del fantasma es casi todo un desplazamiento vertical sistemático.** En las
  configuraciones con capa de esmalte el error de las coronas es 0,61 mm de mediana con un p95 de
  0,61 y un máximo de 0,61 (F4), y su componente vertical es 0,61: es la misma traslación hacia la
  raíz para todo el arco. Es el sesgo de altura que el docstring de `por_cortes` ya describía
  (0,37 mm con rejilla de 0,3 mm; aquí la rejilla es de 0,4). Ninguna dispersión lo ve, porque lo
  comparten todas las poses. En las doce, la componente vertical es casi todo el error.
- **Las dispersiones son pequeñas donde el error es sesgo:** U entre 0,01 y 0,50 mm en el grupo
  con gradiente. Con el máximo en lugar de la mediana (descriptivo) se cubrirían 6 de 7; con la
  componente vertical, 1 de 7.
- **F11** (cabeza inclinada 10° con el plano dado antes de la inclinación) tiene el mayor error,
  1,55 mm (1,37 mm vertical, máximo 2,69 mm), que incluye el error del plano declarado en §2.1.
- **F2** es la única configuración cubierta (sin gradiente, por el arranque: 0,45 ≤ 0,61 mm).
- **Coste en el fantasma (descriptivo):** ×9,3 en total (2,3 s sin, 21,1 s con).

### 9.3 · R1

| | Mediana | Máximo | Altura |
|---|---:|---:|---:|
| Umbral (1100/1900 frente a 1400) | 1,61 mm | 3,37 mm | 1,58 mm |
| Arranque (8 × 5°) | 0,21 mm | 0,54 mm | 0,04 mm |
| **U** (√ de la suma de cuadrados de las medianas) | **1,62 mm** | | |

Es lo que se predijo (§5): U ≈ 1,6–1,7 mm, dominada por el umbral, y por encima de los 1,5 mm del
criterio A1 del TRE. El arranque en R1 mueve sobre todo en planta (0,04 mm en altura). Con la
regla de la enmienda del 2026-11-28 del pre-registro del TRE, esta U = 1,62 mm es la que se
comparará con la mediana del TRE de R1 cuando lleguen los puntos.

### 9.4 · Lo que se dice

- El registro por cortes declara ahora cuánto depende su pose del arranque, a un coste de unos
  15 s por caso disperso, sin cambiar la pose ni la confianza.
- **Lo declarado es precisión y no acota el error.** En el fantasma, que tiene verdad, el error
  real de las coronas quedó por encima de lo declarado en las siete configuraciones en que el
  umbral mueve el contorno, porque es un sesgo vertical que comparten todas las poses. El
  producto lo dice ya así en las dos líneas de la puerta.
- Nada de esto mide la exactitud del caso A. Eso sigue siendo el TRE con puntos anatómicos.
