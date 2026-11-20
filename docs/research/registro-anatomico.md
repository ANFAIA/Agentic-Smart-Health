# Registro escáner↔CBCT anatómico por banda de escáner

> **Estado (2026-11-20): implementado en `fusion_agents.anatomico` y opcional
> (`--registro anatomico`). Funciona en el fantasma y FALLABA en el caso A (§7). La
> corrección de §8 —solo contorno de corona, el valle solo con otra arcada de verdad, y
> la cobertura contada sobre lo que S asigna— quita el veto en el caso A sin empeorar el
> fantasma, pero el caso A no tiene verdad y su cobertura queda cerca del mínimo.** El
> protocolo del caso B sigue suspendido sin congelar hasta revisar §8.
> El caso B no se ha registrado con este método: su evaluación de precisión está
> pre-registrada en [`registro-anatomico-protocolo.md`](registro-anatomico-protocolo.md) y
> se ejecuta después de commitear ese protocolo. `registro="auto"` no cambia hasta que el
> protocolo lo justifique, y en commit aparte.
>
> Todo lo que este registro declara es **precisión**: cuánto se mueve la pose ante
> elecciones declaradas. No es exactitud ni TRE.

## 1 · Por qué

La medida 1 ([`consistencia-registro-caso-b.md`](consistencia-registro-caso-b.md) §5) dejó
dos hechos medidos sobre el camino que se usaba para series densas (`preparacion` + ICP
desde la identidad sobre una submuestra aleatoria de 4.000 vértices + árbitro de umbral):

- la pose del caso B cambia **1–3 mm** con la semilla de la submuestra y con el orden de los
  vértices del fichero, sin que cambie ningún dato;
- en ~30 % de semillas da la **vuelta** (≈180°), con el mismo rms que una pose buena
  (0,679 frente a 0,664–0,680 mm).

Con eso el registro no sirve para afirmar nada por debajo de unos 2 mm. Hacían falta tres
cosas: que no dependa de nada arbitrario, que no pueda elegir una vuelta, y que lo elija un
criterio que distinga poses.

## 2 · El método

El diseño salió de un panel de tres propuestas independientes puntuadas por dos jueces
(anatómica por banda, búsqueda global con consenso, objetivo bilateral contra superficie
fija). Se eligió la primera y se le injertaron piezas de las otras dos (ver §6).

1. **Primero se orienta y después se ajusta, sin azar.**
   - El escáner trae sus ejes de las etiquetas FDI (`anatomical_frame`).
   - El CBCT trae los de la cabecera DICOM (LPS), **sin estimar la inclinación del plano
     oclusal**. Se probó a estimarla y no es fiable (§4).
   - La altura del plano oclusal en el CBCT es la moda del perfil de altura del esmalte.
     Si el perfil tiene dos picos, el valle entra como **segunda ancla**, no como
     sustituta: se arranca desde las dos y decide la puntuación (§4). El segundo pico
     tiene que llegar a la cuarta parte del primero para contar como otra arcada (§8).
   - La pose inicial es la rotación de unos ejes a otros. No hay submuestra aleatoria en
     ningún paso.
2. **Solo se compara lo que las dos modalidades ven.**
   - En el escáner se toma una banda **por diente**: la corona entre 1,5 y 6 mm por debajo
     de la cúspide de *ese* diente, a lo largo del eje oclusal de la arcada. Sigue la curva
     de Spee y la inclinación de la cabeza, que una losa sobre el CBCT no sigue.
   - En el CBCT se toman los **bordes** del diente en cada corte, a un umbral fijo de 1400,
     dentro de una losa de −4 a +14 mm alrededor de las anclas. La losa es ancha a
     propósito: tiene que caber la banda aunque la boca esté inclinada unos 10°.
   - De esos bordes solo cuentan los de **corona**: los que tienen esmalte (≥ 1800) a
     menos de 1,5 mm en su corte. La raíz es dentina y no entra (§8).
   - El borde se sitúa a medio paso de vóxel hacia fuera también en diagonal (8 vecinos).
     Con solo 4 vecinos quedaba sesgado hacia dentro en las esquinas (§4).
3. **El ajuste va del contorno al escáner, punto a plano, con recorte.**
   - Cada punto de contorno busca su punto más próximo entre **todo** el escáner, y solo
     cuenta si ese punto es de la banda. Así las cúspides de la arcada contraria, la raíz y
     el hueso cervical se caen solos, sin un umbral de altura que decidirlo.
   - Hay tres tramos:
     - un tramo grueso, sobre un representante del contorno por milímetro, que solo tiene
       que dejar cada arranque en su cuenca;
     - un tramo medio emparejando con el vértice más próximo;
     - un tramo fino, solo para las cuencas finalistas, que empareja con el punto más
       próximo **sobre la superficie** de la malla y con la normal de su cara. Emparejar
       con vértices sesgaba la altura según el teselado (§4).
   - Si el ICP entra en un ciclo, se detecta y se devuelve el elemento del ciclo de menor
     coste.
4. **Varias salidas fijas y una vara distinta para elegir.**
   - Hay 81 arranques alrededor de la pose anatómica (±20° por eje × ±2 mm en altura) por
     cada ancla.
   - Hay además 9 hipótesis de vuelta de 180° por ancla, que se ajustan siempre y nunca
     pueden ganar: están para medir por cuánto pierden.
   - Las poses se agrupan en cuencas. Gana la de menor **S**: la distancia simétrica,
     dentro de cada corte real del CBCT, entre la sección exacta del escáner y el contorno,
     con todos los dientes pesando igual.
   - S no es lo que se ajusta (punto a plano con recorte) ni lo que se evalúa (§5 del
     protocolo).

**Elegibilidad de una cuenca.** Tiene que cumplir las tres condiciones:
- los tres ejes anatómicos a ≤ 30° de los del CBCT;
- cobertura C ≥ 0,5, con el lado del contorno contado sobre los puntos que S asigna a un
  diente (§8);
- las cúspides a ≤ 5 mm de la altura de alguna ancla.

Si ninguna cuenca las cumple, se devuelve la más cercana a la anatomía con un **veto**, que
deja la confianza en 0 y lo manda al gate (`RegistrationResult.veto`).

**Lo que declara además de la pose** (`informe["uncertainty"]`, por diente):

| Componente | Qué mide |
|---|---|
| Sensibilidad al umbral | Cuánto se mueve la pose al reajustar contra contornos a 1100 y a 1900 |
| Navaja | Cuánto se mueve al dejar fuera cada diente, y el error de predicción de ese diente |
| Ambigüedad de cuenca | Si hay una segunda cuenca casi igual de buena |
| Máscara de corona | Cuánto se mueve al reajustar con el radio de esmalte a 1,0 y a 2,0 mm en vez de 1,5 (§8) |
| Residuo con signo | + si el escáner flota, − si se hunde |
| Condicionamiento | La dirección de traslación peor restringida, en ejes anatómicos |
| Escala | La que pediría una semejanza; solo se informa, nunca se aplica |
| Inclinación | El ángulo entre el eje oclusal del escáner registrado y el superior del CBCT. Incluye el error del eje que sale de las etiquetas FDI (0,99° en el fantasma) |

U = √(umbral² + navaja² + ambigüedad² + máscara²) por vértice. La vertical con que se descompone es el
eje superior del escáner ya registrado, que sí está medido (etiquetas FDI).

Los márgenes de la vuelta y de la ambigüedad van en milímetros de S, no en proporción. S
tiene un suelo que pone el tamaño de vóxel, así que una proporción fija saltaría con un
vóxel y no con otro para el mismo desajuste.

## 3 · Por qué la tabla de `histora` no descartaba este enfoque en series densas

`registro-por-cortes.md` §4 justificaba no usar el registro por cortes en series densas con
1,41 frente a 2,24 mm de mediana corona→esmalte en el maxilar de `histora`. Esa comparación
no discrimina, por cuatro motivos:

1. **La métrica es de un solo lado.** Un vértice hundido en el diente da ≈0 igual que uno
   sobre la superficie: en 11–15 del caso B puntúa igual dos poses separadas 2 mm
   (consistencia §5).
2. **Es casi el criterio del propio camino antiguo.** Su árbitro
   (`preparacion.puntua_contra_esmalte`) elige el umbral por la fracción de vértices a menos
   de 2 mm del esmalte ≥ 1800.
3. **Era una sola semilla**, y la dispersión entre semillas del camino antiguo
   (RIC 0,62–3,05 mm) cubre la diferencia. El propio `por_cortes.py` cita 0,81 mm y
   1,41 mm para el mismo camino sobre el mismo caso.
4. **La banda era una losa axial** con el plano inclinado unos 4°.

Esto no demuestra que el método nuevo sea mejor en series densas. Dice que la tabla no lo
decidía, y que lo decide el protocolo.

## 4 · Desarrollo: qué se probó antes de congelar, y sobre qué

Solo sobre el fantasma sintético (`packages/fusion-agents/tests/fantasma_arcada.py`) y el
caso A. Ninguna constante sale del caso B: o se heredan de `por_cortes`, fijadas sobre el
caso A, o son a priori.

**El fantasma tuvo que hacerse más realista para que las pruebas significaran algo:**
- **Con dientes idénticos, el arco se desliza un diente entero.** Ahora cada pieza tiene su
  anchura y la arcada inferior va desplazada media pieza, como en una oclusión real.
- **Con coronas simétricas de arriba abajo, una vuelta encaja casi igual.** Ahora la corona
  se corta en el cuello. Una vuelta sigue puntuando cerca y la descarta la anatomía, que es
  para lo que está el veto.

**Lo que el fantasma y el caso A cambiaron del método:**
- **La altura del plano.** Con la moda sola, el fantasma de dos arcadas dejó el plano 6 mm
  dentro de la arcada contraria: cada arcada formaba su pico y la moda caía en el ecuador
  de una. Con el valle solo, el caso A (serie dispersa) tomó un valle entre el plano
  oclusal y un pico menor 11 mm por encima, y la losa se quedó sin contorno. Ninguna regla
  sirve sola, así que entran las dos como anclas. El perfil se cuenta por corte y se
  suaviza con un núcleo de 1 mm: un histograma de anchura fija daba picos falsos de
  muestreo.
- **La inclinación del CBCT no se estima.** Un plano por PCA al esmalte leyó entre 0,2° y
  40° para una inclinación real de 10°, según la ventana. Con las dos arcadas en la ventana
  tomaba por inclinación el resalte entre ellas; con una ventana definida por el propio
  plano se reforzaba a sí mismo. La inclinación la absorben la rejilla de arranques, la
  banda por diente y la losa ancha. Con la boca del fantasma inclinada 10° el registro
  cae en la pose correcta a menos de 1° y la inclinación se mide después, sobre el escáner
  registrado.
- **Tres sesgos encontrados por la revisión de geometría, corregidos antes de congelar:**
  - el borde de 4 vecinos quedaba hacia dentro en las esquinas: ahora 8 vecinos;
  - emparejar con vértices movía la altura 0,05–0,17 mm según el teselado de la misma
    superficie: ahora el tramo fino empareja con la superficie;
  - el ICP podía oscilar entre dos poses sin converger: ahora el ciclo se detecta.

**Resultado en el fantasma**, con rejilla de 0,4 mm y el escaneo movido 97° y 85 mm a su
propio marco:
- Antes de las tres correcciones, el error de la banda era de 0,172 mm de mediana, todo en
  altura y hacia la raíz.
- Con ellas, la revisión lo midió en ~0,06 mm.
- La U de mediana es de centésimas de milímetro, y la dirección peor condicionada es la
  superior.

Que U sea tan pequeña en el fantasma es lo esperado. Es un caso sin ruido, con la misma
forma en las dos modalidades. No dice nada sobre el caso B.

**Caso A:** ver §7.

## 5 · Lo que este método no resuelve

- **Exactitud.** Un sesgo común a todos los reajustes no aparece en U: el isonivel 1400, o
  que los dientes se hayan movido entre la sesión del CBCT y la del escaneo. Solo lo acotan
  los puntos anatómicos
  ([`uos-protocolo-puntos-anatomicos.md`](../spec/uos-protocolo-puntos-anatomicos.md)).
- **Ningún modelo rígido encaja todas las piezas** si el escaneo y el CBCT no tienen la
  misma forma, como PREVIO frente a POST (consistencia §2). La escala y la navaja lo
  cuantifican; no lo corrigen.
- **Depende de las etiquetas FDI.** Fijan el marco grueso y la banda. Una numeración
  corrida un diente desplaza la banda de cada pieza; P2 del protocolo lo acota.
- **Es más lento**: unos 80 s por registro en el fantasma, frente a segundos del camino
  antiguo. Por eso sigue siendo opcional.

## 6 · Qué se injertó de las otras propuestas

- De la búsqueda con consenso:
  - el segundo CBCT como eje de precisión independiente (E3). En el caso B no existe: la
    segunda serie es la misma adquisición, y E3 queda como control de reproducibilidad
    (protocolo §2);
  - los dos niveles de veredicto, «mejora» y «objetivo»;
  - la prueba de calibración;
  - el signo de la arcada.
- Del objetivo bilateral:
  - el residuo con signo por diente;
  - el diagnóstico de escala de 7 GdL;
  - la prueba de teselado;
  - la sospecha sobre la moda del plano, que el fantasma confirmó (§4).

**Dos hallazgos laterales, fuera de este cambio:**
- `marco.marco_canonico` lee el signo del término cuártico de un ajuste de grado 4, no la
  apertura del arco. En los escaneos superiores del caso B da el anterior al revés; lo usa
  `segmentar_fdi.py`. Aquí se usa un ajuste de grado 2 solo como comprobación cruzada
  (`marco_geometrico`).
- `preparacion.plano_oclusal_del_esmalte` usa la moda. Este módulo no la toca: añade el
  valle como segunda ancla cuando el perfil es bimodal y lo declara.

## 7 · Caso A: el método falla, y por qué

Ejecución de desarrollo del 2026-11-20 sobre el maxilar del caso A. La serie es dispersa
(paso mediano 0,45 mm, hueco máximo 1,95 mm) y las etiquetas FDI son las votadas. **No hay
verdad en el caso A**: la referencia es el registro por cortes (`por_cortes`, rms 1,65 mm,
solapamiento 0,56), que tampoco es exacto. Nada de esta sección se usa para elegir
constantes comparando con esa pose.

**El método tal como está, con 180 arranques (dos anclas):**
- **Veto.** Ninguna de las 35 cuencas es elegible. La devuelta tiene cobertura C = 0,21 y
  S = 1,47 mm, y queda a 14,0 mm de mediana del registro por cortes.
- **Una vuelta puntúa mejor.** Su S es 0,71 mm menor; solo la descarta la anatomía.
- **Incertidumbre declarada de 20 a 43 mm** por diente: el propio método dice que no sabe
  dónde está.
- **Tiempo: 22–24 min** por registro con incertidumbre, frente a unos 80 s en el fantasma.

**Por qué, medido desde la pose del registro por cortes:**

| Contorno | S en esa pose | C en esa pose | Afinando desde ahí, la pose se mueve |
|---|---|---|---|
| Dos anclas, losa −4 a +25,6 mm | 0,86 mm | 0,40 | 10,4 mm |
| Solo la moda, −4 a +14 mm | 0,86 mm | 0,40 | 7,6 mm; las cúspides acaban a +8,7 mm |
| Solo la moda, −3 a +9 mm | 0,86 mm | 0,40 | 3,3 mm |
| Solo la moda, −2 a +7 mm | 0,78 mm | 0,42 | 1,2 mm, sin converger en 200 iteraciones |

Esto deja tres fallos de diseño, que el fantasma no tenía cómo mostrar:
1. **La banda de corona del escáner encaja contra las raíces.** Una losa que llega 14 mm por
   encima del plano oclusal mete raíz y cuello, y en un diente real tienen una anchura
   parecida a la de la corona cerca de la encía. El ajuste sube la arcada hacia ellas, en la
   dirección peor condicionada (la vertical). Cuanto más alta la losa, más se desplaza. En
   el fantasma las raíces eran conos finos y no competían.
2. **La segunda ancla es espuria aquí.** El perfil del esmalte es bimodal (picos a 29,1 y
   42,7 mm), pero las cúspides están en el primero; el valle (+11,6 mm) solo ensancha la losa
   hasta casi 30 mm.
3. **La cobertura mínima de 0,5 es inalcanzable** en esta serie incluso en la pose de
   referencia: el lado del contorno da 0,40, porque la losa trae mucho borde que el escáner
   no ve. Con eso el veto salta siempre.

**Consecuencia:** el protocolo del caso B queda **suspendido sin congelar**. Ejecutarlo
sería medir la precisión de un método que ya se sabe que falla en el único caso real de
desarrollo, y a ~22 min por registro serían unas 50 h. Antes hace falta:
- reproducir estos fallos en el fantasma, con raíces tan anchas como el cuello, porque solo
  ahí hay verdad;
- corregirlos contra el fantasma;
- volver a pasar el caso A, solo como comprobación de que no hay veto y de lo que declara;
- reducir el tiempo por registro.

## 8 · Corrección «esmalte»: la corona es esmalte y la raíz dentina

Tres cambios, cada uno con una regla fijada a priori o sobre el fantasma. **Ninguna
constante sale del caso A**, y su distancia al registro por cortes no se ha usado para
elegir nada: se da abajo solo como información.

**1 · Solo se ajusta contra contorno de corona** (`junto_al_esmalte`). El contorno sigue a
1400 y en el mismo sitio (medio paso de 8 vecinos), pero un punto solo cuenta si en **su
corte** hay esmalte (HU ≥ 1800, el umbral del árbitro y del plano) a menos de 1,5 mm. El
radio es el grosor del esmalte en las paredes axiales (1–2,5 mm, menos hacia el cuello)
más un vóxel. En su corte y no en 3D: en una serie dispersa el corte vecino puede estar a
2 mm, y en 3D una pared de raíz justo encima del cuello vería el esmalte de abajo. La misma
regla se aplica a los contornos a 1100 y 1900 de la incertidumbre. Y el radio es una
elección, así que entra en U: se reajusta con 1,0 y 2,0 mm (`RADIOS_ESMALTE_INCERTIDUMBRE`).

Se eligió esto y no subir el umbral del ajuste, por lo que mide el fantasma con una capa
de esmalte que se adelgaza hasta el cuello y un desenfoque gaussiano (opciones
`esmalte="capa"` y `desenfoque_mm` de `fantasma_arcada`): desde la verdad, con σ = 0,3 mm,
ajustar contra el contorno a 1900 deja la corona **1,93 mm** hacia la raíz, y a 1400,
0,28 mm. La máscara no mueve ningún punto, así que no cambia ese sesgo: en todos los
umbrales y desenfoques probados (σ 0, 0,18 y 0,3 mm) la pose desde la verdad es la misma
con máscara y sin ella.

**2 · El valle solo es ancla si el segundo pico es otra arcada.** El segundo pico tiene que
llegar a la cuarta parte del primero (`PICO_SECUNDARIO_RELATIVO`). Dos arcadas en oclusión
tienen esmalte por corte del mismo orden; en el fantasma de dos arcadas la proporción es
0,98. En el caso A el «pico» de 42,7 mm tenía **3·10⁻⁴** de la altura del primero: unas
pocas gaussianas sobre un perfil casi vacío. Cualquier umbral entre 0,01 y 0,9 decide lo
mismo en los dos.

**3 · Propuesta y revertida: contar el lado del contorno de C solo sobre lo que S asigna a
un diente** (a ≤ 3 mm de la sección). La verificación independiente midió que en el
fantasma, la única verdad, eso solo sube C en poses equivocadas (deslizamientos de 3 a
4,5 mm en planta) y nunca en la verdadera. Además, la motivación escrita era que la pose del
registro por cortes del caso A no llegaba al mínimo, y elegir una guarda por la pose del
método antiguo es justo lo que no se permite. Se revirtió en un commit propio: C sigue
siendo el de §2.

### 8.1 · En el fantasma (la única verdad)

Error de corona frente a la verdad, todos los vértices de corona; la vertical positiva va
hacia la raíz. Rejilla de 0,4 mm.

| Fantasma | Veto | Elegida: S / C | Mediana | p95 | Vertical | Desde la verdad |
|---|---|---|---|---|---|---|
| Raíces realistas (antes de §8) | no | 0,150 / 0,973 | 0,082 | 0,105 | +0,081 | 0,084 mm |
| Raíces realistas | no | 0,148 / 0,974 | 0,081 | 0,083 | +0,076 | 0,084 mm |
| Realistas, serie dispersa (0,45 mm, hueco 2,2–3,9) | no | 0,150 / 0,977 | 0,080 | 0,092 | +0,077 | 0,078 mm |
| Realistas, capa de esmalte, σ = 0,3 mm | no | 0,150 / 0,970 | 0,436 | 0,503 | +0,434 | 0,285 mm |
| Ídem sin máscara (comparación) | no | 0,150 / 0,970 | 0,437 | 0,503 | +0,435 | 0,285 mm |
| Realistas, capa, σ = 0,18 mm, serie dispersa | no | 0,146 / 0,977 | 0,065 | 0,078 | +0,065 | 0,078 mm |
| Ídem sin máscara (comparación) | no | 0,150 / 0,977 | 0,080 | 0,092 | +0,077 | 0,078 mm |

- En la verdad: S 0,148 y C 0,975 (lado del contorno 1,0) con raíces realistas; S 0,158 y
  C 0,975 con desenfoque.
- **Desde la verdad no hay deslizamiento**: con la losa normal y con una que llega a toda
  la raíz (−4 a +25,6), y arrancando 2, 4 y 6 mm hacia la raíz, el ajuste vuelve a
  0,085 mm. Con la máscara el contorno es **idéntico** con las dos losas: la altura de la
  losa deja de importar (test).
- **Vuelta**: con raíces realistas y sin desenfoque una vuelta puntúa 0,003 mm de S mejor,
  como ya pasaba con los conos (−0,0023); la descarta la anatomía. Con desenfoque, la
  máscara le baja la cobertura 0,29 respecto a la elegida (0,005 sin máscara): ahí ya la
  excluyen los datos.
- **U**: en el fantasma nítido, ~0,03 mm (navaja; el umbral no mueve nada porque el campo
  es binario). Con σ = 0,3 mm, U ≈ 1,6 mm por diente, casi todo vertical y casi todo del
  reajuste a 1900. Cubre el error real (0,44 mm) con holgura. Con y sin máscara sale igual,
  y cambiar el radio de la máscara a 1,0 o 2,0 mm no mueve la pose (0,000 mm).
- **Tiempo**: 80–120 s por registro con incertidumbre, según la carga de la máquina; lo
  mismo que antes de §8.
- **El desenfoque deja un sesgo vertical** de 0,28–0,44 mm hacia la raíz: el isonivel 1400
  de un borde desenfocado cae dentro del diente, y con paredes casi verticales eso es
  altura. §8 no lo corrige ni lo pretende; es exactitud (§5) y solo lo acotan los puntos
  anatómicos.

### 8.2 · En el caso A (sin verdad: solo veto, U y tiempo)

Misma serie y etiquetas que §7, por el camino del orquestador (`prepara_registro`,
`registro="anatomico"`), desde el commit con las reglas 1 y 2 y la 3 revertida:
- **Vetado otra vez, ahora solo por cobertura:** C = 0,43 frente al mínimo de 0,5.
  - Ancla única en la moda: el valle de §7 queda descartado por la regla 2.
  - Se devuelve la cuenca más cercana a la orientación anatómica, con confianza 0.
- **La pose ya no se va a las raíces.** Queda a 3,1 mm de mediana del registro por cortes
  (p95 5,4), frente a 14 mm en §7. Es solo información, no un criterio.
- **U de 1,6 a 4,0 mm** por diente (en §7, de 20 a 43); los peores son los segundos
  molares, 17 y 27. Residuo con signo de −0,62 mm en el 21.
- **Una vuelta puntúa casi igual** (S −0,039 mm, C −0,02): la descarta la anatomía, no los
  datos.
- **Tiempo: 666 s** con incertidumbre, frente a 22–24 min en §7.

Con la regla 3 (la ejecución del agente que la propuso, antes de revertirla) el caso A no
salía vetado: C = 0,558, S = 0,617 mm, U mediana 2,14 mm, a 3,2 mm del registro por cortes,
en 882–908 s. La diferencia entre vetado y no vetado es solo esa regla, y la regla no se
sostiene en el fantasma.

**Lectura.** La máscara de esmalte quita el mecanismo que llevaba la banda a las raíces, y
el método pasa de no saber dónde está (U de 20–43 mm) a declarar 1,6–4 mm. Lo que queda es
una pregunta que el caso A no puede contestar sin verdad: si C < 0,5 en este caso es un
registro malo o una guarda mal calibrada para series reales. La única forma limpia de
saberlo es medir el TRE del caso A con puntos anatómicos marcados por una persona
([`uos-protocolo-puntos-anatomicos.md`](../spec/uos-protocolo-puntos-anatomicos.md)), o un
fantasma que reproduzca C ≈ 0,4 en la pose verdadera, que ninguno de los probados
(raíces anchas, serie dispersa, capa de esmalte, desenfoque) hace: todos dan C ≈ 0,97 en la
verdad.

### 8.3 · Lo que sigue débil

- **El caso A sigue vetado por cobertura (C = 0,43).** No se ha bajado el mínimo: no hay
  verdad con la que fijarlo, y cambiar la guarda hasta que el caso pase sería elegir con él.
- **El fantasma no reproduce el fallo de §7** ni con esto: con o sin máscara la pose desde
  la verdad es la misma. Que la máscara sea lo que quita el deslizamiento en el caso A es
  una lectura de su mecanismo (sin máscara, 7,6 mm y las cúspides sobre la raíz; con ella,
  3,1 mm), no una medida contra verdad.
- **Depende de que el esmalte llegue a 1800.** En un equipo con valores de gris por debajo
  (esmalte saturado bajo o mal calibrado) la máscara vaciaría el contorno y
  `prepara_anatomico` fallaría por falta de puntos, que es un fallo declarado; si solo
  quedara el esmalte de las cúspides, la banda perdería su parte cervical. En el sentido
  contrario, hueso cortical ≥ 1800 junto a una raíz volvería a meter raíz.
- **Con la máscara, S distingue peor la arcada contraria.** En el fantasma disperso con capa
  de esmalte y σ = 0,18 mm, dos cuencas con la banda sobre las coronas inferiores (a 14 y
  16 mm de la verdad) puntúan S 0,137 y 0,142, por debajo de la buena (0,146); sin máscara
  quedaban por encima (≥ 0,179), porque la raíz de la otra arcada contaba en su contra. Las
  excluyen la cobertura y la altura de las cúspides, no S. Con una sola ancla eso aguanta
  mientras las cúspides queden a más de 5 mm del plano, pero es un margen que antes daba S y
  ahora da solo la anatomía.
- **La sobremordida no explica el veto.** Se pensó que la otra arcada en las alturas de la
  banda, que es esmalte y la máscara no quita, bajaría C. Probado en §10: en la pose
  verdadera C pasa de 0,975 a 0,94–0,97, no a 0,4.
- **El desenfoque es un sesgo de exactitud de 0,3–0,4 mm** en el fantasma con σ = 0,3 mm,
  que U declara a través del umbral pero el método no corrige.

## 9 · Dónde se quedó (2026-11-20, corte de las 13:30) y cómo retomar

**Estado.**
- La rama `feat/registro-banda-escaner` tiene el método con la máscara de esmalte (§8),
  264 tests verdes en la última pasada completa más los 3 de §11 por separado, y ruff, mypy
  y los guardianes limpios.
- El protocolo del caso B sigue **sin congelar**. Sus huellas (§7 del protocolo) son
  anteriores a §8 y no valen: hay que recalcularlas con `--huellas` al congelar.

**Lo que bloquea el caso B**, en orden:
1. **El veto por cobertura en el caso A (C = 0,43).** Para decidir si es el registro o la
   guarda hace falta verdad. Hay dos caminos:
   - puntos anatómicos del caso A, marcados por una persona según el protocolo de puntos.
     Requiere pedirlo a las anotadoras, y eso solo con permiso explícito del usuario;
   - un fantasma que dé C ≈ 0,4 en la pose verdadera.
     - La sobremordida queda descartada (§10).
     - Las etiquetas FDI desbordadas sobre la encía (§11) son el único candidato abierto:
       bajan el lado del contorno hasta 0,63–0,84 cuando entra encía vestibular en la
       banda, sin llegar a 0,5.
     - **Primer diagnóstico, hecho el 2026-11-20 a las 12:30.** Es una medida, no una
       regla, y no elige nada. En la pose que devuelve el caso A, que está vetada:
       - lado del escáner 0,74, lado del **contorno 0,43**: el bajo es el del contorno, como
         en §7;
       - desglose aproximado (vértice más próximo en 3D, no la sección por corte de
         `puntua_secciones`): los contornos cuyo vértice de banda está a menos de 1,5 mm de
         encía sin etiqueta quedan a ≤ 1 mm el 59 % de las veces (581 puntos), y el resto
         el 49 % (11 851 puntos).
       - Así que la falta de cobertura **no se concentra junto a la encía**: la mitad del
         contorno de toda la banda queda a más de 1 mm del escáner.
       - Eso apunta a una pose equivocada, o a una forma que no casa entre las dos
         modalidades, más que al desborde de etiquetas. Al estar medido en una pose vetada,
         no distingue entre las dos cosas. Se distinguirían con puntos anatómicos.
2. **La encía etiquetada en la banda sesga la altura sin que salte el veto** (§11):
   - 0,4–1,1 mm en serie dispersa;
   - hasta 0,8–4,4 mm en las sondas con encía vestibular, incluso en serie densa.
   Es lo primero a corregir, con uno de los dos remedios geométricos de §11, verificado
   contra el fantasma con el criterio fijado antes. El modelo FDI reentrenado, que baja la
   encía etiquetada como diente del 45 % al 28 % en `dev`, ataca la misma causa por el otro
   lado. Si su evaluación en la reserva lo confirma, hay que reetiquetar A y B con él.
3. **El sesgo vertical por desenfoque** (0,3–0,7 mm hacia la raíz en el fantasma con
   σ = 0,3–0,4 mm). Es exactitud, no precisión, así que no bloquea el protocolo de
   precisión, pero tiene que estar declarado en el artículo.
4. **Tiempo:** en el caso A, 255 s por registro sin incertidumbre y 11 min con ella. El
   protocolo del caso B son ~125 registros, casi todos sin incertidumbre: unas 9–10 h, más
   si la serie densa de B trae más contorno.

**Ramas locales del workflow del 2026-11-20, no fusionadas y no subidas:**
- `fix/losa-sigue-pose` (5c24a77…67b29ba): un filtro de orientación de normales que en el
  fantasma con capa de esmalte de 1,2 mm baja el deslizamiento de 1,19 a 0,067 mm,
  confirmado por ablación. Con esmalte de 0,6–1,0 mm deja 0,22–0,32 mm. La «ventana que
  sigue a la pose» de esa misma rama no aporta nada y su guarda empeora un arranque; no
  debe entrar. Es el candidato siguiente para el sesgo de capa fina.
- `worktree-wf_3f408c80-68c-3` (b0f6720): S bilateral con desfase. La verificación lo dejó
  en cosmético y debilita las guardas; descartado.
- Los informes completos de los agentes y verificadores están en el diario del workflow
  `wf_3f408c80-68c` de la sesión.

## 10 · Sobremordida: probada en el fantasma, no baja C

**Pregunta.** En una mordida real los incisivos inferiores quedan 2–3 mm por lingual de los
superiores y los solapan 2–4 mm en altura, y las cúspides vestibulares inferiores entran
1–2 mm en las fosas superiores. Esa otra arcada es esmalte a las alturas de la banda: la
máscara de §8 no la quita y podía contar contra el lado del contorno de C. En el fantasma las
dos arcadas apenas se tocaban (puntas a ±0,5 mm de z = 0, incisivos borde a borde).

**La opción.** `fantasma_arcada.campo_cbct(oclusion="sobremordida")`, por defecto
`"plana"` (lo de siempre). Solo mueve la arcada inferior, con valores de libro fijados
antes de medir: resalte de 2,5 mm por lingual en los incisivos, sobremordida de 3 mm,
intercuspidación posterior de 1,5 mm, y el canino a medias. El desplazamiento lingual de los
posteriores es el que ya da la arcada inferior, más estrecha. Donde las coronas se tocan el
volumen las une, como en un CBCT con la boca cerrada. Test de la geometría y test del
resultado en `test_anatomico.py`.

**Medido con el código actual** (rejilla de 0,4 mm, raíces realistas). C es el mínimo de
los dos lados, y S va en mm. Cada fila es la pose verdadera, y la búsqueda completa como
el orquestador, sin incertidumbre:

| Fantasma | Verdad: S / C | Lado escáner / contorno | Búsqueda: veto, error mediano | Afinar desde la verdad |
|---|---|---|---|---|
| Sin sobremordida | 0,148 / 0,975 | 0,975 / 1,000 | — (§8.1: no, 0,081 mm) | 0,084 mm |
| Sobremordida 3 mm, resalte 2,5, posterior 1,5 | 0,149 / 0,972 | 0,975 / 0,972 | no, 0,061 mm | 0,044 mm |
| Peor plausible: 4 mm, resalte 2, posterior 2 | 0,150 / 0,954 | 0,970 / 0,954 | — | 0,043 mm |
| Ídem + capa de esmalte, σ = 0,3 mm, serie dispersa | 0,160 / 0,940 | 0,969 / 0,940 | no, 0,304 mm (vertical +0,30) | 0,300 mm |
| Capa, σ = 0,18 mm, serie dispersa, valores de libro | 0,153 / 0,950 | 0,976 / 0,950 | — | 0,037 mm |
| Cota: incisivos separados del todo (resalte 4,5 mm, fuera de lo plausible) | 0,150 / 0,892 | 0,975 / 0,892 | — | — |

La serie dispersa es paso 0,45 mm con un hueco de 2,2 a 3,9 mm, como en §8.1. El 0,30 mm de
la fila con desenfoque es el sesgo vertical de §8.1, no la sobremordida.

**Lectura.** La sobremordida baja el lado del contorno de 1,0 a 0,94–0,97, y ni separando
los incisivos del todo baja de 0,89. Afecta solo a 4–6 dientes de 14, y solo a la parte
baja de la banda (desde 1,5 mm sobre la punta). **No reproduce C ≈ 0,4 en la verdad.** Por
eso no se propone ninguna regla nueva de cobertura: no hay una pose verdadera con C baja
contra la que diseñarla, y cambiar C sin ella sería volver a la regla 3 de §8.

De paso, sobre las poses equivocadas, con y sin sobremordida:
- Un deslizamiento en planta de 2 mm en x da C ≈ 0,57–0,58 y es elegible.
- Uno de 4,5 mm en −x da C 0,49–0,52, y es elegible en tres de los cinco fantasmas.
- Los de 3 mm en y, 4,5 en −y y 3+3 en xy dan C 0,22–0,47 y no lo son.
- 4 mm hacia la raíz conserva C 0,64–0,95. Con una sola ancla lo veta la altura de las
  cúspides; con dos es elegible, y solo S lo deja atrás (0,23 frente a 0,15).

C no es lo que separa una pose a 2 mm de la buena; lo hace S. El mínimo de 0,5 está cerca de
lo que da un deslizamiento de 3–4,5 mm. Bajarlo dejaría pasar esos deslizamientos en el
fantasma, que es la única verdad.

**Qué queda.** La C = 0,43 del caso A sigue sin explicación con verdad. El camino limpio es
el TRE con puntos anatómicos marcados por una persona (§8.2). El fantasma tampoco tiene
fosas ni incisivos en pala: son elipsoides. Una oclusión más fiel cambiaría el detalle de la
tabla, pero difícilmente lo bastante para pasar de 0,95 a 0,4.


## 11 · Etiquetas FDI desbordadas sobre la encía: candidato abierto para C, y un sesgo de altura que el veto no ve

**Pregunta.** En Teeth3DS+ el segmentador del escáner etiqueta como diente el 45 % de la
encía del experto (`scripts/benchmark_margen.py`), típicamente 1–3 mm más allá del margen.
Si esa encía cae en la banda (1,5–6 mm bajo la cúspide) y no tiene contorno de diente a
menos de 1 mm, el lado del escáner de C baja. En el fantasma no podía pasar: las coronas
asomaban 9 mm y la encía era una cinta 1 mm por encima, etiquetada 0.

**La opción.** `fantasma_arcada.escaneo_superior(encia="cuello", corona=..., desborde_mm=...)`,
por defecto `encia="cinta"` (lo de siempre). Valores de libro fijados antes de medir:
- la encía nace en el margen de cada corona: sale a 45° hacia apical hasta 1 mm de grosor
  y sube paralela al eje del diente 5 mm más;
- el margen es festoneado: la papila sube 2,5 mm hacia oclusal en mesial y distal;
- corona clínica de 9 mm en vestibular (`corona="normal"`) o de 6,5 mm (`"corta"`);
- `desborde_mm`: cuánta encía, medida sobre la superficie desde el margen, lleva la etiqueta
  de su diente;
- el CBCT no cambia: la encía es tejido blando.

**Medido con el código actual** (rejilla 0,4 mm, raíces realistas). «Serie dispersa» es paso
0,45 mm con un hueco de 2,2 a 3,9 mm, como en §8.1. Error de corona frente a la verdad sobre
los vértices de corona de verdad (sin la encía desbordada); vertical positiva hacia la raíz.
La búsqueda es la del orquestador, sin incertidumbre.

| Escaneo | CBCT | Encía en banda | Verdad: S / C (escáner / contorno) | Afinar desde la verdad | Búsqueda: veto, error mediano (p95) |
|---|---|---|---|---|---|
| Cinta de siempre | denso | 0 | 0,148 / 0,975 (0,975 / 1,000) | 0,084 mm | — (§8.1: no, 0,081) |
| Corona 9 mm, desborde 0 | denso | 0 | 0,148 / 0,975 (0,975 / 1,000) | 0,084 mm | no, 0,081 (0,083) |
| Corona 9 mm, desborde 1, 2 o 3 | denso | 0 | idéntico al anterior | 0,084 mm | no, 0,081 (0,083) con 3 mm |
| Corona 6,5 mm, desborde 0 | denso | 0 | 0,150 / 0,996 (0,996 / 1,000) | 0,076 mm | no, 0,077 (0,085) |
| Corona 6,5 mm, desborde 1 | denso | 778 de 5744 | 0,157 / 0,976 (0,996 / 0,976) | 0,101 mm | no, 0,130 (0,166) |
| Corona 6,5 mm, desborde 2 | denso | 868 de 5834 | 0,158 / 0,976 (0,996 / 0,976) | 0,100 mm | no, 0,104 (0,127) |
| Corona 6,5 mm, desborde 3 | denso | 876 de 5842 | 0,158 / 0,976 (0,996 / 0,976) | 0,100 mm | no, 0,104 (0,128) |
| Corona 6,5 mm, desborde 0 | disperso, capa | 0 | 0,150 / 1,000 (1,000 / 1,000) | 0,041 mm | no, 0,074 (0,077) |
| Corona 6,5 mm, desborde 1 | disperso, capa | 778 | 0,163 / 0,959 (1,000 / 0,959) | 0,407 mm | no, **0,417** (0,533), vertical +0,41 |
| Corona 6,5 mm, desborde 2 | disperso, capa | 868 | 0,163 / 0,959 (1,000 / 0,959) | 0,412 mm | no, **0,548** (0,732), vertical +0,54 |
| Corona 6,5 mm, desborde 3 | disperso | 876 | 0,163 / 0,959 (1,000 / 0,959) | 0,412 mm | no, **0,552** (0,732), vertical +0,54 |
| Corona 6,5 mm, desborde 3 | disperso, capa | 876 | 0,163 / 0,959 (1,000 / 0,959) | 0,412 mm | no, **1,062** (1,532), vertical +1,03 |
| Corona 9 mm, desborde 3 | disperso, capa | 0 | 0,148 / 0,976 (0,976 / 1,000) | 0,078 mm | no, 0,066 (0,078) |

«Capa» es capa de esmalte con σ = 0,18 mm. En la verdad y al afinar da exactamente lo mismo
que el esmalte macizo (comprobado con desborde 0 y 3 en denso, y 3 en disperso): a ese
desenfoque el contorno a 1400 y la máscara no cambian en este fantasma. La búsqueda, en
cambio, puede acabar en otra cuenca: con desborde 3 y serie dispersa, 0,55 mm con esmalte
macizo y **1,06 mm** con capa, las dos sin veto y con C 0,96–0,97. En esta última U sale
1,34 mm de mediana, casi todo vertical y casi todo del reajuste a 1900: el gate lo recibe
por incertidumbre, no por veto.

**Sensibilidad al grosor de la encía, añadida después de ver la tabla**, no un valor
elegido: la encía palatina real tiene 2–3 mm.
- Corona 6,5 mm, desborde 3, denso: con 2 mm, C 0,973 en la verdad y desliza 0,092 mm; con
  3 mm, C 0,972 y 0,091 mm.
- Ídem en serie dispersa con 2 mm: C 0,949, desliza **1,12 mm**, y la búsqueda da 1,16 mm
  (vertical +1,12) sin veto, C 0,96.

**Lectura** (corregida tras la verificación independiente, que reprodujo la tabla exacta):
1. **Con esta geometría el desborde no lleva C a 0,4 en la verdad, pero no queda
   descartado.**
   - El lado del escáner no baja (0,99–1,00). El que baja es el del **contorno**, de 1,0 a
     0,96–0,98: la encía etiquetada queda encima del esmalte del CBCT y se lleva como punto
     de sección más próximo contornos que están a más de 1 mm.
   - Es el mismo lado que estaba bajo en el caso A en §7. Una primera redacción de esta
     sección decía lo contrario; era un error.
   - Si se descarta es solo por magnitud, y esa magnitud depende de la geometría probada.
     Con coronas de 6,5 mm y la banda acabando 6 mm bajo la cúspide, solo entra la
     **papila**.
   - El verificador probó coronas clínicas de 4–5 mm y encía de 3 mm, que meten encía
     vestibular y palatina en la banda. C en la verdad baja a **0,84–0,63**, siempre por el
     lado del contorno.
   - Esas sondas exageran: en este fantasma la unión amelocementaria está 9,5 mm bajo la
     cúspide, así que bajo esa encía queda esmalte que en una corona corta real sería en
     buena parte dentina.
   - La tendencia va hacia el veto sin llegar a 0,5. Queda como **candidato debilitado pero
     abierto** para el C = 0,43 del caso A.
   - Tampoco se sabe qué lado es el bajo en el caso A después de la máscara: §8.2 no lo
     desglosa. Es el primer diagnóstico que hay que sacar al retomar.
2. **Sí aparece otro fallo, que el veto no ve: la encía etiquetada tira de la altura.**
   - Sus caras están inclinadas (|n·oclusal| 0,60, frente a 0,21–0,24 en las paredes de la
     corona) y todas tiran en el mismo sentido.
   - En la serie densa, con solo papila en la banda, son el 9 % de las parejas y el efecto es
     de 0,02 mm.
   - En la serie dispersa el hueco quita cortes de la mitad de la banda y la encía pasa al
     17 % de las parejas. El ajuste baja entonces la arcada 0,41 mm hacia la raíz desde la
     verdad, y 1,1 mm con encía de 2 mm. Basta 1 mm de desborde, porque la papila ya está en
     la banda.
   - En las sondas del verificador con encía vestibular en la banda, el deslizamiento desde
     la verdad llega a **0,8–4,4 mm incluso en serie densa**. Ahí C queda en 0,75–0,93, la
     pose es elegible y **el veto no salta**.
   - Este sesgo es más grave que lo que la tabla sola sugiere, y es exactitud, no
     precisión: un protocolo de precisión entre dos escaneos con el mismo tipo de desborde
     no lo vería.

**Qué no se ha cambiado, y qué se podría decidir después.** No se toca ninguna regla: no
hay tiempo de diseñar y verificar una antes del corte, y la pregunta de C sigue abierta. Un
remedio con principio para el punto 2 no debería salir del caso A sino del escáner mismo:
la encía desbordada se distingue por geometría (normal inclinada respecto al eje oclusal,
fuera del contorno convexo de la corona en planta, o más allá del pliegue cervical que ya
busca `analysis_agents.frontera.ancla_al_pliegue`). Es lo primero a corregir al retomar. Candidatos a probar contra el fantasma,
con el criterio fijado antes: (a) recortar la banda de cada diente al pliegue cervical en
vez de a 6 mm fijos; (b) no emparejar contra caras de la banda con |n·oclusal| por encima de
lo que tienen las paredes de la corona del propio diente. Cualquiera de los dos tiene que
dejar la tabla de §8.1 igual y bajar el deslizamiento de las filas dispersas; no puede
elegirse mirando C ni la pose del caso A.

Tests: `test_la_encia_al_cuello_solo_entra_en_la_banda_con_coronas_cortas`,
`test_el_desborde_de_etiquetas_sobre_la_encia_no_baja_la_cobertura_en_la_verdad` y
`test_el_desborde_sobre_la_encia_lleva_el_ajuste_hacia_la_raiz_sin_bajar_c`.

## 12 · Remedio (a): la banda acaba en el pliegue cervical de cada diente

**Decisión: queda APAGADO por defecto** (`BANDA_AL_PLIEGUE = False`), porque uno de los
criterios fijados antes de medir falla (K3, abajo). Encenderlo es una decisión que tiene que
tomar alguien sabiendo qué se cambia.

**Qué cambia si se enciende.** `pliegue_cervical` (en `anatomico.py`, activado por `BANDA_AL_PLIEGUE`):
por diente y por sector en planta (`SECTORES_PLIEGUE` = 12, porque el margen es festoneado),
se recorren sus vértices de 1,5 a 6 mm bajo la cúspide, de arriba abajo, siguiendo la
elevación de la normal hacia fuera. La pared de una corona es convexa: bajando, su normal
solo gira hacia apical. El pliegue es el primer vértice cuya elevación supera en
`GIRO_PLIEGUE_GRADOS` = 20° la mínima vista hasta él, es decir, donde la superficie se abre
hacia la encía. La banda del sector acaba una arista mediana de la propia malla del diente
por encima (`ARISTAS_SOBRE_PLIEGUE`). Un sector sin pliegue conserva 6 mm; un diente con
pliegue en menos de la mitad de sus sectores lo conserva en todos y va al informe
(`scanner_band.teeth_without_clear_crease`). No se añade motivo al gate: con coronas largas
(o la cinta de siempre) el pliegue está por debajo de 6 mm y **todos** los dientes salen
«sin pliegue», que es lo correcto y no un aviso.

**Por qué con la normal y no con `analysis_agents.frontera.concavidad`.** Se probó primero
la concavidad discreta con el umbral heredado de `frontera` (1,0 normalizado): en el
fantasma no encontró ni un pliegue. Medido: normalizada por la media de la malla, la pared
de la corona da ≈ −1 (convexa, radio de pocos mm con aristas de 0,44 mm) y el pliegue no
pasa de +0,2–0,6, porque la media de cosenos sobre todos los vecinos diluye una sola arista
doblada. Además depende del sentido de giro de las caras, que en el fantasma es hacia
dentro. La normal orientada con el vector radial del diente sí lo ve. El umbral de 20° es a priori
(entre el ruido de normales y los ~45° con que sale la encía); no se tocó tras medir.

**Criterios fijados antes de medir** y resultado (rejilla 0,4 mm, raíces realistas; afinar
desde la verdad, `rms_desplazamiento` de la banda):

| | Escaneo | CBCT | Banda fija (encía en banda) | Al pliegue (encía en banda) | Criterio |
|---|---|---|---|---|---|
| K1 | corona 6,5, desborde 3, encía 1 mm | disperso, capa σ 0,18 | **0,412** (876) | **0,073** (0) | < 0,10 ✔ |
| K1 | corona 6,5, desborde 3, encía 2 mm | disperso, capa σ 0,18 | **1,119** (906) | **0,083** (0) | < 0,15 ✔ |
| K2 | corona 5, desborde 3, encía 3 mm | denso | **3,310** (3440) | **0,082** (0) | < 0,2 ✔ |
| sonda | corona 4, desborde 3, encía 3 mm | denso | 4,256 (5178) | 0,097 (306) | — |
| sonda | corona 5, desborde 3, encía 2 mm | denso | 3,118 (3446) | 0,082 (0) | — |

Con la corona de 4 mm quedan 306 vértices de encía en la banda: en interproximal la papila
sube 2,5 mm y el pliegue queda por encima de 1,5 mm bajo la cúspide, fuera de la ventana
donde se busca. Aun así el deslizamiento baja de 4,3 a 0,10 mm.

Búsqueda completa del orquestador con el pliegue (sin incertidumbre): corona 6,5,
desborde 3, encía 1 mm, disperso con capa → 0,075 mm de error mediano de corona (p95 0,081),
sin veto, frente a 1,06 con la banda fija (§11). Corona 5, encía 3 mm, denso → 0,084
(0,104), sin veto.

**K3, sin regresión.**
- La cinta de siempre y las coronas de 9 mm (desborde 0 y 3) dan **la misma banda, vértice
  a vértice**, con el pliegue activado y sin él: ningún sector encuentra pliegue por encima
  de 6 mm. Como el resto del registro es determinista y solo consume la banda, las filas de
  §8.1 (densa y dispersa realistas) y las de corona 9 mm de §11 no cambian: no hay nada que
  volver a correr.
- Corona 6,5 mm sin desborde sí cambia: la banda pierde 760 de 4966 vértices (el último
  anillo, una arista por encima del pliegue en el margen, sobre todo junto a la papila).
  Denso: afinar desde la verdad 0,082 mm frente a 0,076, búsqueda 0,082 (0,085) frente a
  0,077 (0,085), sin veto: dentro de 0,01.
  Disperso con capa: búsqueda 0,076 (0,081) frente a 0,074 (0,077), sin veto, dentro de
  0,01. Pero **afinar desde la verdad pasa de 0,041 a 0,074 mm**: esa casilla de K3 falla.
  No es el margen sobre el pliegue: con `ARISTAS_SOBRE_PLIEGUE` = 0 (sonda posterior, no
  usada para elegir) da 0,082. El 0,041 de la banda fija parece la excepción —denso da
  0,076 con ella, y la búsqueda dispersa ya acababa en 0,074—, pero eso es una lectura, y
  el criterio estaba fijado antes.
- Determinismo: la banda es idéntica tras permutar los vértices del fichero (orden por
  profundidad con desempate por posición, no por índice).

**Lo que sigue débil.**
- Solo se ha probado en el fantasma, cuya encía sale a 45° de un margen nítido. En un
  escaneo real el pliegue puede ser romo, el surco puede no estar escaneado y la encía
  desbordada puede no abrirse hacia fuera; ahí el diente se declara sin pliegue y vuelve el
  sesgo de §11 sin aviso en el gate.
- Una corona con una concavidad propia por encima del margen (un cuello en reloj de arena,
  una restauración, un incisivo con el cíngulo marcado en palatino) cortaría la banda antes
  de tiempo. Sin datos con verdad no se sabe cuánto.
- No se ha corrido el caso A ni se ha mirado C con el remedio: no puede elegirse por él.
- La huella de constantes (`huella_constantes`) cambia: el protocolo del caso B sigue
  suspendido y sin congelar, así que no rompe ninguna congelación.

Tests: `test_la_banda_al_pliegue_cervical_deja_fuera_la_encia_desbordada` (fija K1) y los
tres de §11, que ahora fijan la banda vieja con `BANDA_AL_PLIEGUE` apagado.
