# Registro escáner↔CBCT anatómico por banda de escáner

> **Estado (2026-11-20): implementado en `fusion_agents.anatomico` y opcional
> (`--registro anatomico`).** Desarrollado sobre un fantasma sintético y sobre el caso A.
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
   - El CBCT trae los suyos de la cabecera DICOM (LPS). El eje superior se corrige con la
     normal de un plano ajustado al esmalte (`marco_cbct`).
   - La pose inicial es la rotación de unos ejes a otros. No hay submuestra aleatoria en
     ningún paso.
2. **Solo se compara lo que las dos modalidades ven.**
   - En el escáner se toma una banda **por diente**: la corona entre 1,5 y 6 mm por debajo
     de la cúspide de *ese* diente, a lo largo del eje oclusal de la arcada.
   - En el CBCT se toman los **bordes** del diente en cada corte, a un umbral fijo de 1400,
     dentro de una losa inclinada de −2 a +12 mm sobre el plano oclusal.
3. **El ajuste va del contorno al escáner, punto a plano.**
   - Cada punto de contorno busca su vértice más próximo entre **todos** los del escáner,
     y solo cuenta si ese vértice es de la banda.
   - Así las cúspides de la arcada contraria y el hueso cervical se caen solos, sin un
     umbral de altura que decidirlo.
4. **Varias salidas fijas y una vara distinta para elegir.**
   - Hay 81 arranques alrededor de la pose anatómica (±20° por eje × ±2 mm en altura).
   - Hay además 9 hipótesis de vuelta de 180° que se ajustan siempre y nunca pueden ganar.
   - Las poses se agrupan en cuencas. Gana la de menor **S**: la distancia simétrica,
     dentro de cada corte real del CBCT, entre la sección exacta del escáner y el contorno,
     con todos los dientes pesando igual.
   - S no es lo que se ajusta (punto a plano con recorte) ni lo que se evalúa (§5 del
     protocolo).

**Elegibilidad de una cuenca.** Tiene que cumplir las tres condiciones:
- los tres ejes anatómicos a ≤ 30° de los del CBCT;
- cobertura C ≥ 0,5;
- las cúspides a ≤ 5 mm del plano oclusal del CBCT.

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

U = √(umbral² + navaja² + ambigüedad²) por vértice.

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

**Sobre el fantasma salieron tres cosas:**
- **La moda de la altura del esmalte puede caer en la arcada contraria.**
  - Si cada arcada forma su propio pico de esmalte, la moda queda en el ecuador de una de
    las dos.
  - Con la moda, el plano quedó 6 mm dentro de la arcada de abajo, y el escáner superior
    encajó contra los dientes inferiores.
  - Eso sí lo rechazó el veto de altura de las cúspides.
  - Ahora la altura de referencia es el **valle** si el perfil tiene dos picos y la moda si
    tiene uno.
  - El perfil se cuenta por corte y se suaviza con un núcleo de 1 mm. Un histograma de
    anchura fija daba picos falsos de muestreo con cortes de 0,4 mm.
- **Con dientes idénticos el arco se desliza un diente entero.** El fantasma tiene ahora
  anchuras distintas por pieza y la arcada inferior desplazada media pieza, como en una
  oclusión real.
- **Con coronas simétricas de arriba abajo, una vuelta encaja casi igual.** El fantasma
  corta ahora la corona en el cuello. Una vuelta sigue puntuando cerca (S ×1,03) y la
  descarta la anatomía, que es para lo que está el veto.

**Resultado en el fantasma** (rejilla de 0,4 mm, escaneo movido 97° y 85 mm a su propio
marco):
- error de la banda: 0,172 mm de mediana, todo en altura y hacia la raíz. Es el sesgo de
  medio paso del borde ya documentado en `registro-por-cortes.md` §5;
- U de mediana: 0,014 mm;
- dirección peor condicionada: la superior.

**Caso A:** ver §7.

## 5 · Lo que este método no resuelve

- **Exactitud.** Un sesgo común a todos los reajustes no aparece en U: el isonivel 1400, el
  medio paso del borde, o que los dientes se hayan movido entre la sesión del CBCT y la del
  escaneo. Solo lo acotan los puntos anatómicos
  ([`uos-protocolo-puntos-anatomicos.md`](../spec/uos-protocolo-puntos-anatomicos.md)).
- **Ningún modelo rígido encaja todas las piezas** si el escaneo y el CBCT no tienen la
  misma forma, como PREVIO frente a POST (consistencia §2). ŝ y la navaja lo cuantifican; no
  lo corrigen.
- **Depende de las etiquetas FDI.** Fijan el marco grueso y la banda. Una numeración
  corrida un diente desplaza la banda de cada pieza; P2 del protocolo lo acota.

## 6 · Qué se injertó de las otras propuestas

- De la búsqueda con consenso:
  - el segundo CBCT como eje de precisión independiente (E3);
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
- `preparacion.plano_oclusal_del_esmalte` usa la moda. Este módulo no la toca: la sustituye
  por el valle cuando el perfil es bimodal y lo declara.

## 7 · Caso A

_Pendiente: se completa con la ejecución de desarrollo sobre el caso A antes de congelar el
protocolo._
