# Registro escáner↔CBCT anatómico por banda de escáner

> **Estado (2026-11-20): implementado en `fusion_agents.anatomico` y opcional
> (`--registro anatomico`). Funciona en el fantasma y FALLA en el caso A (§7).** La
> corrección de §8 reproduce el deslizamiento en un fantasma desenfocado y lo reduce allí,
> pero el caso A sigue vetado. El protocolo del caso B está suspendido sin congelar.
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
     sustituta: se arranca desde las dos y decide la puntuación (§4). Desde §8, solo si
     cae en la mitad central entre los dos picos.
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
     con todos los dientes pesando igual. Desde §8, la sección es la de la corona entera
     (no solo la banda) y S descuenta un desfase común del isonivel.
   - S no es lo que se ajusta (punto a plano con recorte) ni lo que se evalúa (§5 del
     protocolo).

**Elegibilidad de una cuenca.** Tiene que cumplir las tres condiciones:
- los tres ejes anatómicos a ≤ 30° de los del CBCT;
- cobertura C ≥ 0,5 (desde §8, solo el lado del escáner: la fracción de la sección de
  corona con borde a ≤ 1 mm);
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

## 8 · Corrección: objetivo bilateral sobre la corona y desfase del isonivel

Desarrollo del 2026-11-20, posterior a §7. **La verdad solo está en el fantasma**; el caso A
se usa para comprobar veto, incertidumbre declarada y tiempo, y su distancia al registro
por cortes se da solo como información.

**Lo que no reprodujeron las raíces anchas, lo reproduce el desenfoque.** Con raíces del
ancho del cuello y una serie dispersa el fantasma no se deslizaba (protocolo de
reproducción, commit `907b26f`). Lo que faltaba es que el borde a 1400 no está sobre la
superficie que mide el escáner. Con una PSF gaussiana de σ = 0,4 mm sobre el fantasma de
raíces realistas (`campo_cbct(desenfoque_mm=0.4)`), el borde cae dentro del diente: poco en
las paredes de esmalte (2200), bastante en la dentina (1600) y en las cúspides, que son
estrechas. Con el código de §7:
- afinando desde la verdad, la arcada sube **0,70 mm hacia la raíz** (1,34° de giro);
- y S puntúa mejor la pose deslizada (0,156 mm) que la verdad (0,193).

Es la firma del caso A en pequeño: deslizamiento hacia la raíz y S de acuerdo con él. Con
grises más bajos (esmalte 1900, dentina 1550) el deslizamiento desde la verdad es de 1,66 mm,
y desde 8 mm hacia la raíz el ajuste no vuelve (15–19 mm).

**El mecanismo.** Por debajo del ecuador de la corona la sección se estrecha hacia la
cúspide. Si el borde del CBCT queda dentro de la superficie, el ajuste encuentra una sección
del escáner más estrecha subiendo la arcada: confunde encoger con subir. Cuanto más alta la
losa y más se parece la raíz a la banda, más lejos llega.

**Lo que cambia** (`fusion_agents.anatomico`):
1. **S sobre la corona entera** (de la cúspide a 6 mm por debajo), no solo la banda, y en
   los dos sentidos dentro del corte. Lo que una pose deslizada hacia la raíz deja sin
   pareja es la parte oclusal de la corona del escáner, metida dentro de una raíz. En el
   fantasma sin desenfoque, 8 mm hacia la raíz subían S con la banda de 0,148 a 0,295 mm;
   con la corona y el desfase, de 0,115 a 0,362 (2 mm: a 0,205).
2. **S descuenta un desfase común del isonivel** (`DESFASE_EN_S`): mide `|r − b|`, con `b`
   la mediana del residuo con signo de la sección de corona contra la normal del borde.
   Con desenfoque, sin él S prefería la pose deslizada (0,156 frente a 0,193 en la
   verdad); con él prefiere la verdad (0,116). El desfase va al informe
   (`isolevel_offset_mm`).
3. **C es solo el lado del escáner.** El lado del contorno cuenta bordes que el escáner no
   puede ver (arcada contraria, hueso por encima del umbral) y no mide si la pose cubre lo
   que sí ve. Se sigue informando.
4. **El valle solo es ancla si cae en la mitad central entre los dos picos**
   (`VALLE_CENTRAL = 0,25`), regla a priori: el contacto entre dos coronas de altura
   parecida queda cerca del punto medio. Un valle pegado a un pico es un hombro del perfil.
   En el fantasma de dos arcadas el valle está en el punto medio exacto y se conserva.
5. **El AJUSTE no cambia por defecto.** Se implementaron y se midieron dos cambios del
   objetivo, y los dos quedan apagados:
   - **Sentido escáner→contorno** (`BILATERAL`): corona del escáner contra el borde del
     corte, un vértice por vóxel de 0,5 mm en ejes anatómicos (por área: con todos los
     vértices el polo de la corona pesaba una quinta parte y torcía la pose 0,6–0,8°). Sin
     desenfoque apenas cambia nada (0,078 frente a 0,085 mm). Con desenfoque, junto al
     desfase, empeora: 0,49–1,16 mm frente a 0,20–0,33, porque la erosión de las cúspides
     no es un desfase común y este sentido la lee como altura.
   - **Desfase en el ajuste** (`DESFASE_EN_AJUSTE`, con las normales del escáner orientadas
     hacia fuera: en la cúspide, la exterior apunta a oclusal). Con desenfoque lleva la
     pose de 0,71 a 0,31 mm de la verdad. Pero en el cono de la corona subir se parece a
     encoger, el valle altura–desfase queda casi plano y **el mismo escaneo subdividido
     1→4 converge a 0,19 mm del original** (−0,09 frente a +0,09 mm en altura). Eso rompe
     P0-teselado del protocolo; una pose que depende del teselado no se puede declarar.

**Resultado en el fantasma, contra la verdad** (rejilla de 0,4 mm, búsqueda completa sin
incertidumbre, coronas; «§8 + desfase en el ajuste» no es el valor por defecto):

| Fantasma | Código | Veto | Mediana | p95 | Vertical | S elegida / verdad | C elegida / verdad | Vuelta, ΔS |
|---|---|---|---|---|---|---|---|---|
| Raíces realistas | §7 | no | 0,082 | 0,105 | +0,081 | 0,150 / 0,148 | 0,973 / 0,975 | +0,007 |
| Raíces realistas | §8 | no | 0,082 | 0,105 | +0,081 | 0,114 / 0,115 | 0,980 / 0,981 | +0,034 |
| Raíces realistas | §8 + desfase en el ajuste | no | 0,099 | 0,102 | +0,094 | 0,115 / 0,115 | 0,981 / 0,981 | +0,017 |
| Realistas + desenfoque 0,4 | §7 | no | 0,714 | 0,846 | +0,710 | 0,156 / 0,193 | 0,968 / 0,975 | −0,001 |
| Realistas + desenfoque 0,4 | §8 | no | 0,714 | 0,846 | +0,710 | 0,118 / 0,116 | 0,977 / 0,974 | +0,007 |
| Realistas + desenfoque 0,4 | §8 + desfase en el ajuste | no | 0,310 | 0,364 | +0,308 | 0,120 / 0,116 | 0,975 / 0,974 | +0,005 |

**Lectura honesta.** Por defecto la pose del fantasma no cambia: el ajuste es el mismo y
la cuenca elegida también. Lo que cambia es que S ya no premia el deslizamiento (con
desenfoque, la verdad puntúa mejor que la cuenca elegida, 0,116 frente a 0,118) y que la
vuelta pierde por más. Pero **con el borde desplazado ninguna salida llega a la verdad**:
las 43 cuencas del fantasma desenfocado se deslizan, y una puntuación mejor no puede elegir
una cuenca que no existe. El deslizamiento solo se corrige en el ajuste, y el único cambio
de ajuste que lo corrige aquí rompe la estabilidad ante el teselado.

**Caso A** (sin verdad; maxilar, votadas, serie dispersa):

Una búsqueda completa con incertidumbre, con los valores por defecto de §8 (ancla única: el
valle de +11,6 mm queda fuera por estar a 0,35 de la separación del punto medio entre
picos):
- **Sigue habiendo veto**, ahora solo por altura: las cúspides acaban a +8,7 mm del plano
  oclusal. La cobertura ya no es el motivo: C = 0,62 en la elegida y 0,78 en la pose del
  registro por cortes (antes 0,21 y 0,40).
- **La pose sigue deslizándose hacia la raíz**: afinando desde la pose del registro por
  cortes se mueve 7,6 mm, y la devuelta queda a 7,6 mm de mediana de esa pose (vertical
  +7,2; solo información). S sigue prefiriendo la deslizada (0,452 frente a 0,532 mm).
- **La incertidumbre declarada es enorme y lo dice**: U mediana 11,0 mm (vertical 10,2),
  p95 11,8 mm; el reajuste a 1900 se mueve 10,9 mm.
- La vuelta pierde ahora por S +0,056 mm (antes ganaba por 0,71).
- **Tiempo: 549 s** con incertidumbre, frente a 22–24 min: la mitad de arranques (una
  ancla) y una losa de 18 mm en vez de 29,6. Máquina compartida.

Con el desfase también en el ajuste (`DESFASE_EN_AJUSTE`, no por defecto) el resultado es
el mismo en lo esencial: veto por altura (+8,6 mm), 6,6 mm de deslizamiento desde la pose
del registro por cortes, U mediana 14,6 mm, 915 s.

**Diagnóstico, sin selección.** Medido en la pose del registro por cortes, el residuo con
signo de la corona del escáner contra el borde en el corte cambia con la altura bajo la
cúspide: −0,34 mm en el primer milímetro, −0,06 a 1–3 mm, +0,44 a 4–6 mm. El desfase común
estimado es +0,03 mm: no hay un desfase común que corregir, hay un gradiente de casi 0,8 mm
entre cúspide y cuello. Es compatible con el esmalte que adelgaza hacia el cuello y deja la
zona cervical por debajo de 1400 tras el desenfoque; también con que esa pose de referencia
esté baja. Sin verdad no se puede separar, y el fantasma, de esmalte uniforme, no lo
reproduce todavía.

**Lo que sigue sin resolver.**
- Un desfase común no modela una erosión que depende de la curvatura y del tejido. Con
  grises bajos (esmalte 1900, dentina 1550) y σ = 0,8 mm, el ajuste desde la verdad se va
  1,0 mm aun con el desfase en el ajuste (2,8 mm con el sentido inverso), y 6,6 mm sin él.
- El desfase y la altura están acoplados en el cono de la corona; un sesgo de la PSF
  deja un sesgo de altura que ni U ni el desfase declarado distinguen.
- El tiempo por registro no baja con el método: en el fantasma, 110 s con incertidumbre,
  igual que el código de §7 con la misma carga. En el caso A baja a 549 s porque el valle
  ya no es ancla, no porque el registro sea más rápido.
- El caso A sigue sin registrarse. El protocolo del caso B sigue suspendido.
