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

**3 · El lado del contorno de C se cuenta sobre lo que S asigna a un diente**, a ≤ 3 mm
(`TOPE_SECCION_MM`) de la sección del escáner. Es la misma frontera que S ya usaba («a más
de eso no es corona de esta arcada»); C contaba también lo que S descartaba: la otra
arcada en sobremordida, un cordal que el escaneo no trae.

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

Misma serie y etiquetas que §7, con `prepara_anatomico` completo (búsqueda e
incertidumbre):
- **Sin veto.** Ancla única en la moda (el valle de §7 queda descartado por la regla 2).
  21 825 puntos de contorno a 1400 frente a 45 040 sin máscara.
- Elegida: S 0,617 mm, C 0,558 (lado del escáner 0,82, del contorno 0,56). Ejes a 4,0 /
  4,0 / 1,1° del CBCT. Tres cuencas elegibles de catorce; la siguiente, a 0,115 mm de S.
- La vuelta puntúa ahora **peor** (S +0,093 mm, C −0,07); en §7 puntuaba 0,71 mm mejor.
- **U mediana 2,14 mm** (p95 3,92; vertical 1,69). Por diente, de 1,26 mm (12) a 3,96 mm
  (17); la domina la navaja (1,1–3,1 mm), y los segundos molares son los peores. El
  reajuste a 1100 no converge en 200 iteraciones y mueve 0,98 mm de mediana; a 1900, 0,35.
  El radio de la máscara mueve 0,24 mm (1,0 mm) y 0,47 mm (2,0 mm) de mediana, hasta 1,1 mm
  en el 27: aquí sí es una elección que pesa, y por eso entra en U. En §7 la U declarada era
  de 20 a 43 mm.
- **Tiempo: 882–908 s** con la máquina compartida (registros del fantasma y la batería de
  tests en paralelo), frente a 22–24 min en §7. Sigue siendo lento.
- Solo como información, nunca como criterio: la pose queda a 3,2 mm rms de banda del
  registro por cortes. Desde esa pose, el ajuste se mueve 3,1 mm con la máscara y 7,6 mm
  sin ella (cúspides a +8,7 mm del plano: la banda sobre las raíces otra vez).

### 8.3 · Lo que sigue débil

- **C queda a 0,06 del mínimo en el caso A**, y varias cuencas no elegidas rondan 0,47–0,52.
  Otro caso con peor acuerdo entre modalidades puede volver a vetarse. No se ha bajado el
  mínimo: no hay verdad con la que fijarlo. Afinando desde la pose del registro por cortes
  (solo información), C sale 0,57, 0,56 y 0,52 con radio 1,0, 1,5 y 2,0 mm.
- **El fantasma no reproduce el fallo de §7** ni con esto: con o sin máscara la pose desde
  la verdad es la misma. Que la máscara sea lo que quita el deslizamiento en el caso A es
  una lectura de su mecanismo (sin máscara, 7,6 mm y las cúspides sobre la raíz; con ella,
  3,1 mm), no una medida contra verdad.
- **Depende de que el esmalte llegue a 1800.** En un equipo con valores de gris por debajo
  (esmalte saturado bajo o mal calibrado) la máscara vaciaría el contorno y
  `prepara_anatomico` fallaría por falta de puntos, que es un fallo declarado; si solo
  quedara el esmalte de las cúspides, la banda perdería su parte cervical. En el sentido
  contrario, hueso cortical ≥ 1800 junto a una raíz volvería a meter raíz.
- **La sobremordida no está en el fantasma.** La otra arcada en las alturas de la banda es
  esmalte y la máscara no la quita; la regla 3 solo deja de contarla en C si está a más de
  3 mm.
- **El desenfoque es un sesgo de exactitud de 0,3–0,4 mm** en el fantasma con σ = 0,3 mm,
  que U declara a través del umbral pero el método no corrige.
