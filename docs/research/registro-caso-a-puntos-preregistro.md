# Pre-registro: TRE del caso A con puntos anatómicos

**Fecha:** 2026-11-20. **Estado:** fijado **antes de recibir ningún punto**. A la fecha de
este commit no se ha enviado el encargo a nadie y la carpeta de anotaciones está vacía.
Enmendado el 2026-11-28 (§10), también antes de recibir ningún punto.

Este documento fija qué se va a medir con los puntos anatómicos del caso A, contra qué
registros, con qué comandos y qué se podrá concluir. Se escribe antes porque lo que se usa
para elegir deja de valer para medir: si esto se decidiera después de ver los puntos, el
TRE resultante estaría elegido con ellos.

Protocolo de marcado: [`uos-protocolo-puntos-anatomicos.md`](../spec/uos-protocolo-puntos-anatomicos.md).
Contexto: [`registro-anatomico.md`](registro-anatomico.md), §7 a §9 y §11.

## 1 · Qué pregunta contesta

El registro anatómico del caso A sale **vetado por cobertura** (C = 0,43 frente a un mínimo
de 0,5; [`registro-anatomico.md`](registro-anatomico.md) §8.2). Sin verdad no se puede
saber si es un registro malo o una guarda mal calibrada para series reales. Los puntos
marcados por personas son la única referencia independiente disponible para este caso.

Pregunta secundaria: el fantasma muestra un sesgo vertical hacia la raíz de 0,4–4,4 mm
cuando la encía etiquetada entra en la banda (§11), y el veto no lo ve. La componente
vertical del TRE es la única medida del caso real que puede mostrarlo.

## 2 · Datos que se marcan

- **Caso A, arcada superior.** Serie CBCT `c1bd42bdd9d7`: 301 cortes de una adquisición
  con huecos (paso mediano 0,15 mm, salto máximo 6,15 mm). Las cabeceras no tienen ninguna
  etiqueta identificable con valor; comprobado con `scripts/inventario_series_cbct.py`
  sobre el ZIP de envío, que escribió el inventario sin necesitar
  `--permitir-identificables`. Salida guardada en
  `data/processed/puntos-anatomicos/inventario-zip/` (`inventario.json`, SHA-256
  `a76c51556608a80d5a5e23c79cb2fda1b1dd352a61ce1014c3794c23fcb67e39`).
- **Cortes reales en la zona de las coronas** (altura en mm, la que muestra Slicer en la
  vista axial), los mismos que lista la guía y comprobados hoy contra las cabeceras:
  24,94 · 25,09 · 26,29 · 27,19 · 27,64 · 27,94 · 28,54 · 28,84 · 28,99 · 29,29 · 29,89 ·
  30,94 · 31,09 · 31,39 · 31,84 · 32,14 · 32,59.
- **Paquete preparado para el envío** (`data/processed/envio/`), el mismo para las dos
  observadoras:

  | Fichero | SHA-256 |
  |---|---|
  | `caso-A-puntos.zip` (304 ficheros: `cbct/`, `escaneo-superior.stl`, guía ES/EN en PDF) | `1f94159a0dcd59b524e9e1be7c550a8990e9bcde1d5e24695b896eb8c0555fe4` |
  | `escaneo-superior.stl` | `abca87e971dab2d6a49d60edfe7f73f95ce62410603df86e72373a8c874197cb` |
  | `instrucciones.pdf` | `f1c422dca14f31910e66f52c56568ede4cbf32c47d3610dd846089459ecbbfe7` |
  | `instructions-en.pdf` | `dad88c930c272012499b9c9829053d7818e03a49d1c252ddb7ff5abd1d74b449` |

  El contenido del ZIP coincide fichero a fichero con los ficheros del paquete que incluye
  (los `.md` de la guía no van). Los nombres internos van por papel; ninguno es el del
  fabricante.

## 3 · Observadoras

- **Dos observadoras independientes**, especialistas que ya conocen 3D Slicer. Sus
  nombres no constan en el repositorio: aquí son la observadora 1 y la 2. Si aceptan,
  aparecerán con su nombre en los agradecimientos del artículo, **nunca como autoras**.
- Cada una marca **sin ver los puntos de la otra ni hablarlo con ella** hasta haber
  entregado, y marca el CBCT y el escáner en sesiones distintas.
- Ninguna ve la superposición que calcula el programa.
- Cada una entrega en su propia carpeta (`anotaciones/<observadora>/cbct.mrk.json` y
  `escaner.mrk.json`). Al recibir cada entrega se anota su SHA-256 y la fecha, antes de
  ejecutar nada.
- **No se ejecuta la puntuación hasta tener las dos entregas**, o hasta que el usuario dé
  por cerrada la espera, anotándolo con fecha en este documento. Con una sola entrega se
  informa su TRE como **no conforme al protocolo**, que exige dos observadores: sin suelo
  entre observadoras, sin la lectura de §7 y sin veredicto sobre el veto.

## 4 · Registros que se puntúan

Los dos, con los mismos puntos y en la misma ejecución de la puntuación. Se informan los
dos sea cual sea el resultado.

**R1 · Registro por cortes, `run-6`** (`fusion_agents.por_cortes`, el camino `auto` en una
serie dispersa). Es un artefacto ya generado; se puntúa tal cual.
- Código: commit `57eb021f7b74f2b174449d4f9ca279507a691369`, árbol limpio (su
  `run-record.json`).
- Contenedor: `data/processed/verification-20261109/series-c1bd42bdd9d7/run-6/export/another_patient.uos`,
  SHA-256 `239e258e077d7b53a561740d67d8188705992dd00a3e992f38b5d05819f8034a`.
- Registro: `reg.ct_to_ios`.
- Comando con que se generó:

  ```
  uv run python scripts/caso_completo.py \
      --caso <caso A> \
      --cbct data/processed/verification-20261007/input/series-c1bd42bdd9d7 \
      --salida data/processed/verification-20261109/series-c1bd42bdd9d7/run-6 \
      --semilla 0 \
      --fdi data/processed/fdi-votada/another_upper/etiquetas_upper.npy \
      --ajusta-campo --gs-apariencia data/processed/gs-escaner-full --entrena-apariencia \
      --lado-foto <foto 1>=11 --lado-foto <foto 2>=16 \
      --margen-gingival aplica
  ```

  La ruta del directorio del caso y los nombres de las fotos están literales en el
  `run-record.json`; aquí no se copian.

**R2 · Registro anatómico** (`fusion_agents.anatomico`, rama `feat/registro-banda-escaner`).
- Código: HEAD de la rama, `543e28952ff6d24cfe993c6fc1a2910731a27f71`.
- Huellas (`uv run python scripts/precision_registro.py --huellas` en ese commit):
  - `huella_constantes` `6e0a24b30e59f5813b09fcd2b445a2a3fb2426230e9979026d06e9cd07ff2b68`
  - `huella_protocolo` `b4a345e95b24b8c20ee36aec167aed1510f37dffef0c7b144e65b581e4bacfef`
- Comando: el de R1 con `--registro anatomico` y otra salida, desde una copia limpia de ese
  commit:

  ```
  systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 \
  uv run python scripts/caso_completo.py \
      <los mismos argumentos que R1, salvo --salida> \
      --registro anatomico \
      --salida data/processed/verification-20261120/series-c1bd42bdd9d7/anatomico-543e289
  ```

- Su `run-record.json` tiene que decir `git_head` = `543e289…` y `git_dirty: false`;
  `--huellas` tiene que dar los dos valores de arriba; tiene que existir `registro.json`
  (solo lo escribe el camino anatómico) y el informe de registro no puede llevar la clave
  `sin_anatomico`, que es la que deja el orquestador cuando cae al ICP de siempre. El
  `.uos` dice `icp_surface` en los dos casos y no sirve para distinguirlos. Si algo de esto
  falla, el contenedor no es R2.
- **R2 se ejecuta y su `.uos` se anota aquí con su SHA-256 antes de abrir ninguna
  entrega.** Si la ejecución falla por memoria u otra causa, se repite una sola vez con el
  mismo límite y se informa.
- El método no tiene azar (§2 de `registro-anatomico.md`): la semilla solo afecta a la
  apariencia, que no interviene en el registro. Lo que se puntúa es `reg.ct_to_ios`.
- El registro vetado se escribe igual en el contenedor, con confianza 0 y el motivo en la
  puerta humana: se puntúa **esa** pose, la que el método devuelve.
- Si la ejecución no reproduce el veto de §8.2 (C = 0,43), se declara la diferencia y se
  puntúa lo que salga; no se repite hasta obtener el valor esperado.

**R2, ejecutado el 2026-11-20, 13:03–13:25.** `run-record.json`: `git_head` `543e2895…`,
`git_dirty: false`. `registro.json` existe, con `method` `fusion_agents.anatomico` y
`constants_sha256` igual a la `huella_constantes` de arriba; no hay `sin_anatomico`.
Contenedor `data/processed/verification-20261120/series-c1bd42bdd9d7/anatomico-543e289/export/another_patient.uos`,
SHA-256 `c84b978db2b83bd62c8268be05df44d0a05d5c9ed4d47af359833494fcc15b50`.
**Diferencia declarada:** la búsqueda queda vetada, como en §8.2, pero el motivo dice
cobertura 0,50 (mínimo 0,5) y no 0,43. No se ha investigado ni se repite: se puntúa esta
pose.

**R2b · Registro anatómico con el filtro de inclinación** (§11.1 de `registro-anatomico.md`).
Añadido el 2026-11-20 a las 13:05, **antes de enviar la petición y sin ninguna entrega**. El
filtro deja de emparejar las caras de banda más inclinadas que la pared de su corona, que en
el fantasma corrigen el sesgo hacia la raíz de §11. R2 no cambia y se sigue puntuando tal
cual: R2b no lo sustituye.
- Código: `00887406…` de la misma rama (`git rev-parse 0088740` da el hash completo).
- Huellas (`--huellas` en ese commit):
  - `huella_constantes` `c2c75645343a866384487a963a3fee8818e520f60756b6254b97e5448d0233f9`
  - `huella_protocolo` `b4a345e95b24b8c20ee36aec167aed1510f37dffef0c7b144e65b581e4bacfef`
- Comando: el de R2 desde una copia limpia de `0088740`, con salida
  `data/processed/verification-20261120/series-c1bd42bdd9d7/anatomico-0088740`.
- Las mismas comprobaciones que R2 (`git_head`, `git_dirty: false`, huellas, `registro.json`,
  sin `sin_anatomico`), y el mismo requisito: su `.uos` se anota aquí con su SHA-256 antes de
  abrir ninguna entrega.
- La misma regla de lectura que R1 y R2 (§7). **R2 frente a R2b no elige nada**: que uno
  cumpla y el otro no se informa como resultado de este caso, no como motivo para encender
  el filtro por defecto. El margen del filtro se eligió viendo el fantasma, y el filtro tiene
  un límite medido (coronas de 4 mm con la franja de referencia contaminada) en el que no
  quita nada.
- **R2b, ejecutado el 2026-11-22, 14:00–14:46** (el 20 se lanzó a las 13:25 y se paró en el
  corte antes de terminar; aquella salida parcial se borró). `run-record.json`: `git_head`
  `0088740f…`, `git_dirty: false`. `registro.json` existe, con `method`
  `fusion_agents.anatomico` y `constants_sha256` igual a la huella de arriba; no hay
  `sin_anatomico`. Contenedor
  `data/processed/verification-20261120/series-c1bd42bdd9d7/anatomico-0088740/export/another_patient.uos`,
  SHA-256 `3bbd881602bd2b46e232835adec98a8c4124da0f4d6f8b9e6f7abac051c8d736`.
  Queda vetado, como R2: cobertura 0,48 (mínimo 0,5), frente a 0,50 de R2. Se puntúa esta
  pose.

**Secundario, no es un tercer método.** `--ajuste 16-mv,26-mv,13-c,23-c`, en la ejecución
de R1 (no depende del registro del contenedor), calcula un registro rígido con esos cuatro
puntos, su error en los demás y un *leave-one-out* sobre todos los puntos comunes. Dice si
un registro por puntos alcanzaría el criterio A1 en este caso; no se compara con R1 ni R2
para elegir.

## 5 · Comando de puntuación

Se ejecuta una vez por registro con las dos observadoras, la observadora 1 **siempre
primero** (la herramienta identifica cada fichero por su nombre, que es el mismo para las
dos), y otra vez por observadora, con un solo `--cbct-puntos` y un solo `--ios-puntos`,
para la sensibilidad de §6.

**Herramienta fijada.** Se puntúa con `scripts/tre_puntos.py` y `fusion_agents.tre` tal como
están en el commit de este pre-registro. Lo que §6 pide y la herramienta no da (agregados
por tipo de diente, componente vertical, U por punto) se añade en un script commiteado
**antes de abrir ninguna entrega**. Cualquier cambio posterior se declara y se informan los
resultados con las dos versiones.

**Convenio del STL, fijado sin mirar el registro.** Cada observadora declara si cargó el STL
como LPS o RAS. Antes de puntuar se comprueba con una prueba que no depende de ningún
registro: la distancia de sus puntos del escáner a la superficie del STL, en coordenadas del
fichero, tiene que ser < 0,5 mm con el convenio declarado. Si falla con el declarado y se
cumple con el contrario, se usa el contrario y se dice. Si las dos usaron convenios
distintos, las listas se pasan al mismo convenio antes de promediar (cambio de signo en x e
y, commiteado antes de puntuar).

```
uv run python scripts/tre_puntos.py \
    --uos <contenedor de R1 o R2> \
    --cbct-puntos data/processed/puntos-anatomicos/anotaciones/<obs1>/cbct.mrk.json \
    --cbct-puntos data/processed/puntos-anatomicos/anotaciones/<obs2>/cbct.mrk.json \
    --ios-puntos data/processed/puntos-anatomicos/anotaciones/<obs1>/escaner.mrk.json \
    --ios-puntos data/processed/puntos-anatomicos/anotaciones/<obs2>/escaner.mrk.json \
    --stl-cargado-como <lps|ras, lo que declaren> \
    --cbct-serie data/processed/verification-20261007/input/series-c1bd42bdd9d7 \
    --salida data/processed/puntos-anatomicos/puntuacion-<fecha>/<R1|R2>
```

`--cbct-serie` excluye los puntos del CBCT a más de 0,05 mm de un corte real <!--const:TOLERANCIA_CORTE_MM-->
y los declara.

## 6 · Qué se informa

Por registro (R1, R2 y R2b), todo y siempre:
- **TRE por punto** sobre la media de las dos observadoras (`per_point_mm` de `tre.json`), y
  **por tipo de diente** (molar 16/26, premolar 14/24, canino 13/23, incisivo 11/21), con
  mediana y máximo agregados a partir de esos valores. La tabla por región que trae
  `tre_puntos.py` (cuadrante × sector, RMS y máximo) se adjunta tal cual.
- **Componente vertical del TRE**, con signo, sobre el eje z del DICOM (S en LPS, el mismo
  eje del suelo de altura), positiva hacia +S, que en la arcada superior es hacia la raíz.
  No depende de ningún eje estimado.
- **Suelo de altura por punto** en la serie dispersa: la mitad del mayor hueco con los
  cortes vecinos.
- **Concordancia entre observadoras**, por punto (`max_pairwise_mm_per_point` de
  `tre.json`) y por tipo de diente (agregada a partir de esos valores), en el CBCT y en el
  escáner por separado, y su componente vertical en el CBCT. Es el suelo de la medida: un TRE menor que el desacuerdo entre
  observadoras no se distingue de cero.
- **Puntos excluidos** y por qué: no marcados por las dos, fuera de un corte real, o
  señalados como dudosos por la propia observadora (en la descripción del punto o en sus
  notas). La lista de exclusiones por notas se fija y se anota antes de ejecutar la
  puntuación. Los ficheros recibidos no se editan.
- TRE de cada observadora por separado, como sensibilidad; no sustituye al de la media.
- **Criterio A1** de la [ficha de uso previsto](../spec/uos-uso-previsto-v0.1.md): TRE máximo
  ≤ 1,5 mm en cada punto. Se dice si cada registro lo cumple. <!--const:EPSILON_IOS_CBCT_MM-->

## 7 · Lectura fijada de antemano

**Incertidumbre por punto.** Para cada punto común *k*:

U_k = √(d_cbct,k² + d_ios,k² + s_k²)

con *d* el desacuerdo entre las dos observadoras en cada modalidad y *s_k* el mayor de sus
dos suelos de altura. El punto **cumple** A1 si TRE_k + U_k ≤ 1,5 mm, **falla** si <!--const:EPSILON_IOS_CBCT_MM-->
TRE_k − U_k > 1,5 mm, y si no es **indeterminado**. <!--const:EPSILON_IOS_CBCT_MM-->

**Por registro, la misma regla para R1, R2 y R2b.** Un registro **cumple** si cumplen todos los
puntos comunes, con n ≥ 6 y al menos uno por cuadrante; **falla** si falla al menos uno; en
cualquier otro caso, incluido n < 6, es **indeterminado**.

Lectura para la pregunta del veto:
- **R2 cumple**: el veto fue una falsa alarma en este caso. Es compatible con una guarda
  demasiado estricta **en esta serie**; no dice nada de otras series y **no fija un mínimo
  nuevo**: recalibrar la guarda exige otro conjunto que no sean estos puntos.
- **R2 falla**: el veto acertó; el registro es malo en este caso.
- **R2 indeterminado**: no se puede decir, y se dice así.

**Sesgo vertical.** Si la media de la componente vertical de R2 hacia la raíz supera la suma
del suelo medio de altura y del desacuerdo vertical medio entre observadoras en el CBCT, es
coherente con el sesgo de §11. No prueba su causa.

## 8 · Lo que los puntos no hacen

- **No ajustan nada.** Ningún registro, constante, umbral ni guarda se elige o se cambia con
  estos puntos.
- **Si un registro se cambia después de verlos**, estos puntos dejan de valer para medir el
  método cambiado. Su TRE exige puntos nuevos o un marcado nuevo, reservado antes de mirar.
- **No se usan para elegir el camino por defecto** entre R1, R2 y R2b. Una elección así exigiría
  puntos nuevos.
- No se escriben en el `.uos`: declarar un TRE en el manifiesto exige una versión sucesora y
  la identidad de quien verificó.

## 9 · Qué se podrá concluir, y qué no

- **Sí:** el TRE de R1, R2 y R2b en este caso, con su incertidumbre, y si el veto de R2 fue una
  falsa alarma o un acierto **aquí**.
- **No es validación clínica.** Es un solo caso, una sola arcada, como mucho ocho puntos,
  en una serie dispersa y adquirida en oclusión.
- **No mide el método en general**, ni su exactitud en otros equipos o series.
- Las observadoras no son una verdad perfecta: su desacuerdo es el límite de lo que se
  puede afirmar, no un error que se pueda restar.
- **No toca el caso B**, que tiene su propio protocolo de precisión, ciego y pendiente
  ([`registro-anatomico-protocolo.md`](registro-anatomico-protocolo.md)).

## 10 · Enmienda del 2026-11-28, antes de recibir ningún punto

Hecha **antes de ver ningún punto**: la petición ya se envió a las observadoras, pero a la fecha
de esta enmienda no ha llegado ninguna entrega y la carpeta de anotaciones solo tiene
subcarpetas vacías, sin ningún fichero. Lo que no se menciona aquí queda como estaba.

**Motivo.** El registro por cortes declara ahora, además de lo que se mueve la pose con el umbral
del contorno, lo que se mueve con la orientación de partida
([`registro-incertidumbre-arranque.md`](registro-incertidumbre-arranque.md)). La pose que
devuelve no cambia ni en un bit, así que R1 sigue siendo el mismo registro y se puntúa tal cual.
Lo que cambia es que el código declara para R1 una incertidumbre, U, y esa cifra se puede
contrastar con los puntos.

**Lo que se añade: una lectura descriptiva, solo para R1.** Se compara la **mediana del TRE de
R1** (la mediana, entre los puntos comunes, del TRE por punto sobre la media de las dos
observadoras de §6; con una sola entrega, la mediana del TRE por punto de esa observadora,
informada como no conforme al protocolo igual que el resto) con la **U declarada para R1** por
`scripts/incertidumbre_arranque.py r1`, ejecutado una sola vez desde el commit del pre-registro
de esa nota, antes de abrir ninguna entrega: el campo `U_mm` de
`data/processed/incertidumbre-arranque/r1-<commit[:12]>/declaracion.json`. La redacción queda
fijada ahora:

- TRE mediano ≤ U: «compatible con la incertidumbre declarada».
- TRE mediano > U: «la incertidumbre declarada no cubre el error en este caso».

**Lo que esta lectura vale, dicho antes.**

- **La U de R1 ya es casi conocida.** La nota de la incertidumbre la predice en ≈ 1,6–1,7 mm
  (su §5), con resultados de la medida 3 sobre el caso A que ya se han visto: la dispersión
  entre umbrales de R1, 1,61 mm de mediana, y los mismos ocho arranques de 5°, 0,18–0,41 mm. Es
  **más que el 1,5 mm de A1** (§6, §7). <!--const:EPSILON_IOS_CBCT_MM--> Así que el umbral de
  esta lectura estaba prácticamente fijado al escribirla, y «compatible» sale para cualquier TRE
  mediano hasta unos 1,6–1,7 mm, **también con un R1 que falle A1**. «Compatible» no dice nada
  sobre A1 y **no es evidencia a favor de R1**: solo dice que la precisión que el código declara
  no queda por debajo del error medido.
- **No se agregan las mismas cosas.** U es una mediana de desplazamientos **entre poses**
  calculados sobre todos los vértices de corona etiquetados del escáner; el TRE mediano es una
  mediana **entre puntos anatómicos**, como mucho ocho, de la distancia a lo que marcan las
  observadoras. Las dos poblaciones son distintas, y la comparación es descriptiva por eso.

**Lo que no cambia.** Ni A1, ni la regla por punto de §7 (la U_k de las observadoras es otra cosa:
el desacuerdo entre ellas y el suelo de altura, no la U del registro), ni las lecturas de R2 y
R2b, ni la del sesgo vertical, ni la herramienta de §5, ni nada más. La lectura nueva no decide
si R1 cumple A1, no se usa para excluir puntos y no ajusta nada (§8). Si `r1` no se ejecuta
antes de la primera entrega, o se detiene en su V0, esta lectura no se hace y se dice.
