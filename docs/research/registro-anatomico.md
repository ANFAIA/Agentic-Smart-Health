# Registro escáner↔CBCT anatómico por banda de escáner

> **Estado (2026-11-20): implementado en `fusion_agents.anatomico` y opcional
> (`--registro anatomico`). Funciona en el fantasma y FALLA en el caso A (§7).** El
> protocolo del caso B está suspendido sin congelar hasta corregirlo. La corrección de §8
> (ventana en el marco del escáner, filtro de sentido, valle y cobertura) corrige en el
> fantasma un deslizamiento de 1,2 mm que sí tiene verdad, pero el caso A **sigue vetado**.
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
     sustituta: se arranca desde las dos y decide la puntuación (§4).
   - La pose inicial es la rotación de unos ejes a otros. No hay submuestra aleatoria en
     ningún paso.
2. **Solo se compara lo que las dos modalidades ven.**
   - En el escáner se toma una banda **por diente**: la corona entre 1,5 y 6 mm por debajo
     de la cúspide de *ese* diente, a lo largo del eje oclusal de la arcada. Sigue la curva
     de Spee y la inclinación de la cabeza, que una losa sobre el CBCT no sigue.
   - En el CBCT se toman los **bordes** del diente en cada corte, a un umbral fijo de 1400,
     dentro de una losa de −4 a +14 mm alrededor de las anclas. La losa es ancha a
     propósito: tiene que caber la banda aunque la boca esté inclinada unos 10°.
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
- cobertura C ≥ 0,5;
- las cúspides a ≤ 5 mm de la altura de alguna ancla.

Si ninguna cuenca las cumple, se devuelve la más cercana a la anatomía con un **veto**, que
deja la confianza en 0 y lo manda al gate (`RegistrationResult.veto`).

**Lo que declara además de la pose** (`informe["uncertainty"]`, por diente):

| Componente | Qué mide |
|---|---|
| Sensibilidad al umbral | Cuánto se mueve la pose al reajustar contra contornos a 1100 y a 1900 |
| Navaja | Cuánto se mueve al dejar fuera cada diente, y el error de predicción de ese diente |
| Ambigüedad de cuenca | Si hay una segunda cuenca casi igual de buena |
| Residuo con signo | + si el escáner flota, − si se hunde |
| Condicionamiento | La dirección de traslación peor restringida, en ejes anatómicos |
| Escala | La que pediría una semejanza; solo se informa, nunca se aplica |
| Inclinación | El ángulo entre el eje oclusal del escáner registrado y el superior del CBCT. Incluye el error del eje que sale de las etiquetas FDI (0,99° en el fantasma) |

U = √(umbral² + navaja² + ambigüedad²) por vértice. La vertical con que se descompone es el
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

## 8 · Corrección: comparar en el marco del escáner, y solo lo que mira hacia fuera

Cambios del 2026-11-20, tras §7. **Todo lo que decide se midió contra la verdad del
fantasma**; el caso A solo se usa para comprobar si hay veto, qué incertidumbre declara y
cuánto tarda. Su distancia al registro por cortes se da como información, nunca como
criterio.

**Lo que cambia en el ajuste (`empareja`, `afina`):**
- **Ventana que sigue a la pose.** Un punto de contorno, llevado al marco del escáner por
  la pose de *esa* iteración, solo cuenta si su profundidad bajo la cúspide de su diente
  cae en la banda (1,5–6 mm), con 1 mm de rampa a cada lado (`MARGEN_VENTANA_MM`). La rampa
  es para que entrar y salir de la ventana no haga saltar el objetivo. La losa fija del
  CBCT ya no decide qué se compara: solo acota cuánto contorno se calcula.
- **Guarda contra una ventana que encoge.** Si la suma de pesos cae por debajo de la mitad
  de la inicial (`FRACCION_VENTANA_MIN`), se para, se devuelve la última pose con la
  ventana llena y no se declara convergida.
- **Filtro de sentido.** La pareja solo vale si la normal hacia fuera del contorno y la de
  la malla hacia fuera del diente forman menos de 90°. Las normales de la malla se orientan
  por diente y por mayoría contra la dirección radial de su banda, porque el sentido del
  fichero no es anatómico.

**Lo que cambia en la elección:**
- **S y C no cuentan bordes interiores** (mismo filtro de sentido) ni contorno a más de
  `TOPE_SECCION_MM` de la sección. Antes, el lado del contorno de C contaba cualquier punto
  cuyo punto de sección más próximo fuera de banda, a cualquier distancia.
- **El valle solo es ancla si los dos picos de esmalte caben a una altura de corona de él.**
  La altura de corona es la del escáner (`Escaner.alto_corona`): la mediana entre dientes de
  la cúspide menos el percentil 1 de su altura. Si el valle fuera el contacto entre arcadas,
  cada pico sería el esmalte de una de ellas, y ese esmalte no puede quedar más lejos del
  contacto de lo que mide la corona.

### 8.1 · Lo que encontró el fantasma

El fantasma de raíces anchas no reproducía el fallo (§7 y el test de raíces). **Lo
reproduce una cáscara de esmalte**: la corona con 1,2 mm de esmalte a 2200 y dentina a 1200,
por debajo del umbral de contorno de 1400 (`campo_cbct(..., dentina=1200, esmalte_mm=1.2)`).
Entonces el isonivel dibuja también la unión amelodentinaria, por dentro de cada corona. Esos
puntos tienen como más próximo una pared de banda a ~1,2 mm, tiran de ella hacia dentro, y la
única forma rígida de acercarlas es subir hasta donde la corona del escáner es más estrecha:

| Fantasma (verdad conocida) | Código | C en la verdad | Afinando desde la verdad: corona, mediana | vertical |
|---|---|---|---|---|
| Cáscara de esmalte, dentina 1200 | anterior | 0,73 | **1,19 mm** | +1,17 mm (hacia la raíz) |
| Cáscara de esmalte, dentina 1200 | solo la ventana | 0,98 | 1,20 mm | +1,18 mm |
| Cáscara de esmalte, dentina 1200 | este | 0,98 | 0,067 mm | +0,066 mm |
| Raíces anchas, todo ≥ 1600 | anterior | 0,975 | 0,085 mm | +0,080 mm |
| Raíces anchas, todo ≥ 1600 | este | 0,975 | 0,088 mm | +0,082 mm |

La dirección (hacia la raíz) es la del caso A. **Lo que corrige el deslizamiento es el filtro
de sentido, no la ventana**: con la ventana sola la cáscara se sigue deslizando lo mismo,
porque la unión amelodentinaria está a la altura de la banda y la ventana la deja pasar.

### 8.2 · Búsqueda completa en el fantasma, contra la verdad

Rejilla de 0,4 mm, escaneo movido 97° y 85 mm. Error en vértices de corona; vertical = +z,
hacia la raíz de la arcada superior.

| Fantasma | Veto | S / C elegida | S / C en la verdad | Corona: mediana · p95 | Vertical (mediana) | Afinando desde la verdad | Margen de la vuelta (S) |
|---|---|---|---|---|---|---|---|
| Raíces anchas, denso | no | 0,148 / 0,974 | 0,148 / 0,975 | 0,084 · 0,085 mm | +0,078 mm | 0,087 mm | +0,009 mm |
| Raíces anchas, disperso (0,45 mm, hueco 2,2–3,9) | no | 0,150 / 0,977 | 0,148 / 0,976 | 0,085 · 0,096 mm | +0,082 mm | 0,083 mm | +0,008 mm |
| Cáscara de esmalte, dentina 1200 | no | 0,150 / 0,983 | 0,151 / 0,983 | 0,065 · 0,073 mm | +0,064 mm | 0,066 mm | **−0,0015 mm** |

En la cáscara, la vuelta puntúa 0,0015 mm mejor que la buena y solo la descarta la anatomía;
`motivos` lo manda al gate. Ya pasaba en el fantasma cónico con el código anterior (§7).

### 8.3 · Caso A: sigue vetado

Misma ejecución de desarrollo que §7, con este código. **No hay verdad.**

- El valle (+11,6 mm) queda fuera como ancla: un pico está a 11,6 mm de él y la corona del
  escáner mide 11,2 mm. **El margen es de 0,4 mm.** La regla no se eligió mirando el caso A,
  pero con una corona medida medio milímetro más alta el valle habría entrado.
- **Veto otra vez.** Ninguna de las 44 cuencas es elegible. La devuelta tiene S 0,91, C 0,48
  y las cúspides a +8,8 mm de la moda. Queda a 7,8 mm (rms de la banda) del registro por
  cortes, frente a 14 mm antes; es solo información.
- **Una vuelta puntúa mejor** (S 0,036 mm menor), como antes.
- **El método no ve la altura en este caso.** Trasladando la pose devuelta en vertical de −12
  a +4 mm, S se queda entre 0,85 y 1,03 y C entre 0,43 y 0,48. En el fantasma S tiene un
  mínimo claro en 0,15. Ni el filtro de sentido ni la ventana dan al caso A una señal
  vertical que no tiene.
- Una comprobación ajena a la referencia: en la pose devuelta, la corona del escáner queda a
  5,7 mm de mediana del esmalte ≥ 1800 (0,5 % a menos de 1 mm); en la del registro por cortes,
  a 1,6 mm (33 %). Apunta a que la devuelta está mal, pero es el criterio del árbitro antiguo
  (§3) y no se usa para elegir.
- **Incertidumbre declarada: U de 2,2 mm de mediana** (p95 4,1 mm; vertical 1,05 mm), entre
  1,7 y 4,0 mm por diente, frente a 20–43 mm antes. La escala que pediría una semejanza es
  ×1,024: ningún modelo rígido encaja. **U está subestimada**: el reajuste a 1900 no tiene
  ninguna pareja en esa pose (la corona queda lejos del esmalte), así que no mide nada; antes
  eso se registraba como «0 mm», y ahora el informe dice «no medido». Y el ajuste fino desde la
  pose devuelta no converge en 200 iteraciones: las parejas alternan entre dos conjuntos casi
  iguales y la pose deriva despacio, sin volver a una anterior.
- **Tiempo:** 577 s de búsqueda y 737 s de incertidumbre, unos 22 min en total, igual que
  antes. La máquina estaba compartida con otros dos procesos de registro. En el fantasma, con
  carga ajena de 10–14, son 95 s sin incertidumbre y 128 s con ella.

Lo que no se ha resuelto: por qué S no distingue alturas en el caso A cuando en el fantasma
sí. La cáscara de esmalte reproduce el deslizamiento y su dirección, pero no ese S plano.
Candidatos sin probar: etiquetas FDI votadas que desbordan a la encía (la corona mide 11,2 mm
según ellas), el desenfoque del CBCT y su umbral sin calibrar, o una mordida abierta en el
CBCT.

### 8.4 · Constantes nuevas

| Constante | Valor | De dónde sale |
|---|---|---|
| `MARGEN_VENTANA_MM` | 1,0 mm | A priori: la tolerancia de cobertura (`DEFAULT_OVERLAP_MM`), dos o tres vóxeles. |
| `FRACCION_VENTANA_MIN` | 0,5 | A priori: la mitad. En un fantasma recortado a propósito, la masa baja legítimamente a 0,53: hay poco margen. |
| `Escaner.alto_corona` | medido | No es una constante: sale de las etiquetas FDI de cada escáner. |

No cambia ninguna constante existente. La losa sigue en −4 a +14 mm.
