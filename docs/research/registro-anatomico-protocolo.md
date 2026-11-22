# Protocolo: precisión del registro anatómico escáner↔CBCT en el caso B

> **Estado (2026-11-20): BORRADOR, SUSPENDIDO SIN CONGELAR.** El método falla en el caso A
> ([`registro-anatomico.md`](registro-anatomico.md) §7), así que este protocolo no se ejecuta
> hasta corregirlo. Las huellas de §7 son las del código que falla y **no valen para
> ejecutar**: se recalculan y el protocolo se congela en un commit propio, antes de registrar
> ningún escaneo del caso B con `fusion_agents.anatomico`.
>
> Lo que sigue es el diseño de la evaluación. Cuando se congele, se congelará en un commit
> propio. Cada prueba se ejecuta una sola vez, desde ese commit y
> con el árbol limpio. Las enmiendas siguen el procedimiento de §8.
>
> Mide **precisión**: cuánto cambia la pose ante cambios que no deberían cambiarla. No mide
> exactitud ni TRE. El método está descrito en [`registro-anatomico.md`](registro-anatomico.md).

## 1 · La pregunta

La medida 1 ([`consistencia-registro-caso-b.md`](consistencia-registro-caso-b.md) §5) dejó
la precisión del registro antiguo del caso B en 1–3 mm. La pose variaba con la semilla y con
el orden de los vértices, y daba la vuelta en ~30 % de las semillas.

Sobre el registro nuevo se hacen tres preguntas:
1. ¿Es invariante a lo que no son datos (semilla, orden, marco del fichero, teselado)?
2. ¿Es robusto al arranque y a las etiquetas?
3. ¿Qué precisión tiene entre dos escaneos independientes de la misma boca contra el mismo
   CBCT, y cuánto de eso es suelo de medida?

## 2 · Datos

**CBCT.** `CASE-64A77C`, la serie de `run-5-modelo`: 578 cortes de 0,3 mm.

**Escaneos superiores.** Entran por enlace duro con el nombre de su papel; el nombre del
proveedor no llega a ninguna salida.

| Papel | Escaneo | Para qué |
|---|---|---|
| `principal` | PREVIO | escaneo A de la medida primaria |
| `repetido` | ESCANEADO PREVIO | escaneo B; misma sesión, adquisición independiente (encaja con A a 0,063 mm) |
| `post` | POST HIGIENE | P5, descriptivo |
| `visualizacion` | Visualization 18-28 | P5, descriptivo |

**Etiquetas FDI.** Cada escaneo lleva las suyas; ninguna se transfiere desde POST.
- Se generaron antes de este commit con
  `segmentar_fdi.py --arcada upper --referencias-entrenamiento 18` (en
  `data/processed/caso-agosto/etiquetas-precision/<papel>/`).
- Son información solo del escáner: no tocan el CBCT ni el registro.
- Se limpian una vez con la cadena de `caso_completo.py` (`LIMPIEZA_ETIQUETAS`).
- **Se declara antes de ejecutar lo que ya se vio en ellas:**
  - PREVIO sale *no consistente* (ρ = 0,89, razón de orientación 0,62, «dudosa»). Tiene una
    instancia de 1.698 vértices etiquetada 27 en la línea media, entre el 11 y el 21.
  - ESCANEADO PREVIO, Visualization y POST salen consistentes (ρ = 0,99).
  - No se corrige nada a mano: eso sería elegir. P2 mide cuánto mueven la pose las
    etiquetas. Si P4 falla y P2 apunta a las etiquetas, se dirá así.

**Segundo CBCT.** Se comprobó en las cabeceras antes de escribir esto:
- `CASE-C872C6` **no es un segundo CBCT**. Tiene el mismo estudio, la misma serie y el mismo
  marco de referencia que `CASE-64A77C`, y 577 de sus 578 cortes con PixelData idéntico byte
  a byte.
- E3 se ejecuta, pero se lee como **control de reproducibilidad** con la misma adquisición
  menos un corte, donde se espera una discrepancia ≈ 0. No es un eje de precisión del lado
  del CBCT.
- Ese eje no se puede medir con los datos que hay, y se declara así.

## 3 · Exposición previa al caso B

Esto es lo que se vio del caso B antes de congelar el protocolo:
- **La medida 1 completa**, con el registro antiguo; motivó el diseño.
- **Un agente del panel de diseño midió el perfil de altura del esmalte del CBCT**: moda
  41,1 mm y valle 38,7 mm, más una inclinación de 4,1° por PCA que luego se comprobó que no
  es fiable (ver `registro-anatomico.md` §4).
- **Las etiquetas FDI de §2** y la identidad de las series CBCT.

Ninguna constante sale del caso B. Las de `fusion_agents.anatomico` se heredan de
`por_cortes`, que se fijaron sobre el caso A, o son a priori. Se desarrollaron solo sobre el
fantasma sintético y el caso A.

## 4 · Procedimiento

`scripts/precision_registro.py`, en este orden. Cada etapa escribe `<salida>/<etapa>.json`
**antes** de imprimir nada y se niega a sobrescribir.

El script **se niega a correr** si:
- el árbol está sucio;
- `huella_constantes()` (las de `anatomico`) o `huella_protocolo()` (las del script) no
  coinciden con las de §7.

Los registros que solo necesitan la pose no calculan la incertidumbre. V0 comprueba que ese
atajo da los mismos bytes que el pipeline.

| Etapa | Qué hace | Criterio de paso |
|---|---|---|
| **V0** | Para el escaneo principal compara tres caminos: `IngestionPipeline.prepara_registro(registro="anatomico")`, `prepara_anatomico` y la construcción del script | transformaciones idénticas byte a byte; si no, se aborta |
| **P0-semilla** | 20 semillas (`SEMILLAS_P0`) y 2 procesos con otro `PYTHONHASHSEED` (`HIJOS_P0`) | 22 de 22 idénticas byte a byte |
| **P0-orden** | 10 permutaciones de vértices, caras y rotación cíclica de caras (`SEMILLAS_ORDEN`), reescritas y reingeridas y también en memoria | desplazamiento máximo de las coronas ≤ `P0_ORDEN_MAX_MM` = 0,01 mm |
| **P0-movimiento** | 6 movimientos uniformes de SO(3) (`SEMILLAS_SO3`) y 4 giros fijos (`GIROS_FIJOS`), con traslaciones de hasta ±100 mm | ≤ `P0_MOVIMIENTO_MAX_MM` = 0,01 mm y ningún veto |
| **P0-teselado** | subdivisión 1→4 por punto medio | mediana ≤ 0,05 mm y p95 ≤ 0,10 mm |
| **P0-etiquetas** | P0-movimiento con las etiquetas **recalculadas** por el segmentador (GPU) | concordancia por vértice ≥ 99 % y mediana ≤ 0,1 mm; si falla se atribuye al segmentador y no bloquea |
| **P1** | la rejilla de arranques girada 10°, 20° y 30° alrededor de las 8 direcciones de un cubo | las de 10° y 20° acaban en la misma cuenca (≤ `P1_MISMA_CUENCA_MM` = 0,05 mm); la de 30° se informa |
| **P2** | sin las etiquetas de cada sextante (3 ejecuciones) y con todos los códigos corridos una pieza (1) | mediana del desplazamiento de las coronas ≤ `P2_MEDIANA_MM` = 0,2 mm |
| **P3** | suelo de muestreo: cortes pares frente a impares (P3a) y la banda en tablero de 1 mm (P3b) | descriptivo; entra en el suelo f(x) de P4 |
| **P4** | la medida primaria (§5) | (i)–(v) y veredicto vertical |
| **P4b** | concordancia del residuo con signo por diente entre PREVIO y ESCANEADO PREVIO | descriptivo |
| **E3** | control con la misma adquisición menos un corte (§2), con `--e3-misma-denticion si`; el registro entre CBCT es `scripts/registro_entre_cbct.py` | se interpreta solo si la navaja entre CBCT tiene p95 ≤ `E3_NAVAJA_P95_MAX_MM` = 0,3 mm y nada sale vetado; entonces mediana ≤ 0,8 mm y p95 ≤ 1,5 mm. **No entra en el veredicto** (§2): un fallo aquí es un fallo de reproducibilidad del software, no de precisión |
| **P5** | POST ↔ Visualization y POST ↔ PREVIO; el camino clásico bajo P0-orden y P0-movimiento | descriptivo |

**Las 9 etapas con criterio son P0 a P4 y deben pasar todas.**
- Si falla P0-semilla, P0-orden, P0-movimiento o P0-teselado, es un fallo de software: se
  aborta sin escribir ninguna cifra de precisión.
- Una **vuelta** es un veto, o un desplazamiento de las coronas de más de `VUELTA_MM` = 5 mm
  respecto al registro de base del mismo escaneo.

## 5 · Medida primaria (P4)

Para cada vértice x de corona de A:

d(x) = ‖A→CBCT(x) − B→CBCT(A→B(x))‖

A→B es el registro directo entre los dos escaneos (`consistencia_registro.directo`, con
`SEMILLA_INSTRUMENTOS` = 0).

**Piezas legibles.** Una pieza se lee si cumple las dos condiciones:
- cobertura ≥ `LEGIBLE_COBERTURA_MIN` = 0,5;
- mediana del suelo f(x) ≤ `LEGIBLE_SUELO_MAX_MM` = 0,2 mm, con
  f(x) = √(P3a²/2 + P3b²/2 + suelo directo²).

**Criterios de paso, fijados ahora:**
- **(i)** Cero vueltas en P0–P4.
- **(ii) Nivel «objetivo»:** mediana de d en las piezas legibles ≤ `OBJETIVO_MEDIANA_MM` =
  0,6 mm y p95 ≤ `OBJETIVO_P95_MM` = 1,2 mm.
  - El nivel **«mejora»** se informa aparte: mediana < `MEJORA_D50_MM` = 1,07 mm, la mitad de
    los 2,14 mm del camino antiguo.
- **(iii)** La mediana de d de cada pieza legible ≤ `PIEZA_MEDIANA_MAX_MM` = 1,0 mm.
- **(iv)** Diferencia rígida entre los dos caminos ≤ 1,0° y ≤ 0,5 mm en el centroide de las
  coronas. El camino antiguo daba 3,8° y 1,59 mm.
- **(v) Calibración:** ≥ 80 % de los vértices legibles cumplen d(x) ≤ 2·√(U_A² + U_B² + f²),
  y la mediana de U ≤ 1,0 mm.
  - Con ≥ 99 % de vértices y la mediana de U por encima de 1,0 mm, la incertidumbre se
    declara «vacía».
  - Con menos del 80 %, se declara «exceso de confianza».
- **Vertical, veredicto aparte:** mediana de |d·superior| ≤ `VERTICAL_MEDIANA_MAX_MM` =
  0,6 mm. Si falla, la precisión vale en planta y no en altura.

Si (ii)–(v) fallan, se informa el fallo y **no se repite** con otras constantes.

## 6 · Lo que no se afirmará

- **Exactitud ni TRE.** Un sesgo común a los dos escaneos se cancela en d: el isonivel 1400,
  o que los dientes se hayan movido entre el CBCT y el escaneo. Solo lo acotan los puntos
  anatómicos marcados por una persona.
- Validez clínica, aptitud para cirugía guiada, o generalización más allá de este caso.
- Precisión del lado del CBCT: no hay un segundo CBCT (§2).
- Que el método nuevo sea «mejor» en exactitud. Solo se comparan precisiones medidas con el
  mismo protocolo.

## 7 · Huellas congeladas

`uv run python scripts/precision_registro.py --huellas` imprime estas dos. El script
compara sus valores con los que se le pasan.

- `huella_constantes()` (`fusion_agents.anatomico`):
  `b770021b4a5c54b8b507c65c34b10d1969794bafe0e82ea09166554002ce049c`
- `huella_protocolo()` (`scripts/precision_registro.py`):
  `b4a345e95b24b8c20ee36aec167aed1510f37dffef0c7b144e65b581e4bacfef`

Los valores que cubren están en el apéndice. Si una huella no coincide, el script no corre:
cambiar una constante después de este commit exige una enmienda (§8).

**Invocación**, una vez por etapa y en el orden de §4, siempre a la misma salida
`data/processed/caso-agosto/precision-registro-1`:

```
uv run python scripts/precision_registro.py --solo <ETAPAS> [--con-etiquetas-gpu] \
    --cbct <serie CASE-64A77C> --principal <PREVIO> --repetido <ESCANEADO PREVIO> \
    --post <POST HIGIENE> --visualizacion <Visualization 18-28> \
    --etiquetas data/processed/caso-agosto/etiquetas-precision \
    --cbct2 <serie CASE-C872C6> --e3-misma-denticion si \
    --huella <huella_constantes> --huella-protocolo <huella_protocolo> \
    --salida data/processed/caso-agosto/precision-registro-1
```

Los escaneos entran por enlace duro con el nombre de su papel. `--solo` acepta varias etapas
separadas por comas. P0-etiquetas necesita la GPU; si está ocupada se ejecuta al final, y
si no se puede ejecutar se declara.

## 8 · Enmiendas

Una enmienda se hace en un commit nuevo, fechada y con su motivo, antes de ejecutar la etapa
afectada. Si una etapa ya se ejecutó, su resultado se conserva y se informa junto al nuevo:
nada se repite en silencio.

## 9 · Resultados

_Se escriben aquí después de la ejecución, tal como salgan._

## Apéndice · Valores congelados

### `fusion_agents.anatomico` (incluidas las heredadas de `por_cortes`, `preparacion` y `registration`)

| Constante | Valor |
|---|---|
| `ALTO_BANDA_MM` | `6.0` |
| `ALTURA_CUSPIDES_MM` | `5.0` |
| `ANGULO_ANATOMICO_MAX` | `30.0` |
| `ANGULO_VUELTA` | `150.0` |
| `BANDA_PLANO_MM` | `6.0` |
| `COBERTURA_MIN` | `0.5` |
| `CORRESPONDENCIAS_MINIMAS` | `100` |
| `DEFAULT_OVERLAP_MM` | `1.0` |
| `DESPLAZAMIENTOS_VERTICALES_MM` | `[-2.0, 0.0, 2.0]` |
| `DESPLAZAMIENTO_AMBIGUO_MM` | `1.0` |
| `DISCREPANCIA_MARCOS_GRADOS` | `30.0` |
| `EJES_LPS` | `[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]` |
| `ESCALA_MAX` | `0.005` |
| `HOLGURA_FINALISTA_MM` | `0.05` |
| `HU_ARBITRO` | `1800.0` |
| `HU_PLANO` | `1800.0` |
| `INCERTIDUMBRE_MAX_MM` | `1.0` |
| `ITER_ESCALA` | `30` |
| `ITER_FINO` | `200` |
| `ITER_GRUESO` | `60` |
| `ITER_MEDIO` | `40` |
| `K_VECINOS` | `6` |
| `LOSA_CBCT_MM` | `[-4.0, 14.0]` |
| `MARGEN_AMBIGUEDAD_S_MM` | `0.03` |
| `MARGEN_MORDIDA_MM` | `1.5` |
| `MARGEN_VUELTA_C` | `0.05` |
| `MARGEN_VUELTA_S_MM` | `0.05` |
| `MEMORIA_CICLO` | `8` |
| `MINIMO_CONTORNO` | `500` |
| `PASO_SECCION_MM` | `0.1` |
| `RADIO_ARCADA_MM` | `4.0` |
| `RADIO_CUENCA_GRUESA_MM` | `0.5` |
| `RADIO_CUENCA_MM` | `0.1` |
| `REJILLA_GRADOS` | `[-20.0, 0.0, 20.0]` |
| `REJILLA_GRUESA_MM` | `1.0` |
| `RESIDUO_FIRMADO_MAX_MM` | `0.5` |
| `SEPARACION_PICOS_MM` | `3.0` |
| `SUAVIZADO_PERFIL_MM` | `1.0` |
| `TOL_COBERTURA_MM` | `1.0` |
| `TOL_GRUESO_MM` | `0.001` |
| `TOL_MM` | `0.0001` |
| `TOPE_PASO` | `[0.08726646259971647, 2.0]` |
| `TOPE_SECCION_MM` | `3.0` |
| `TRIM` | `0.8` |
| `UMBRALES_INCERTIDUMBRE` | `[1100.0, 1900.0]` |
| `UMBRAL_CONTORNO` | `1400.0` |
| `VALLE_RELATIVO` | `0.75` |

### `scripts/precision_registro.py`

| Constante | Valor |
|---|---|
| `UTC` | `"UTC"` |
| `MINIMO_CONTORNO` | `500` |
| `UMBRAL_CONTORNO` | `1400.0` |
| `MAX_PRIMITIVAS` | `1500000` |
| `REFERENCIAS_ENTRENAMIENTO` | `18` |
| `LIMPIEZA_ETIQUETAS` | `["rellena_etiquetas", "absorbe_islas", "afina_fronteras", "quita_motas", "rellena_huecos_interiores"]` |
| `VUELTA_MM` | `5.0` |
| `SEMILLA_INSTRUMENTOS` | `0` |
| `SEMILLAS_P0` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]` |
| `HIJOS_P0` | `[[11, 101], [12, 102]]` |
| `SEMILLAS_ORDEN` | `[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]` |
| `P0_ORDEN_MAX_MM` | `0.01` |
| `SEMILLAS_SO3` | `[1001, 1002, 1003, 1004, 1005, 1006]` |
| `TRASLACION_MAX_MM` | `100.0` |
| `GIROS_FIJOS` | `[["x", 90.0], ["y", 90.0], ["z", 90.0], ["z", 180.0]]` |
| `TRASLACIONES_FIJAS_MM` | `[[100.0, -60.0, 30.0], [-80.0, 100.0, -100.0], [40.0, 90.0, -70.0], [-100.0, -100.0, 100.0]]` |
| `P0_MOVIMIENTO_MAX_MM` | `0.01` |
| `P0_TESELADO_MEDIANA_MM` | `0.05` |
| `P0_TESELADO_P95_MM` | `0.1` |
| `P0_ETIQUETAS_CONCORDANCIA_MIN` | `0.99` |
| `P0_ETIQUETAS_MEDIANA_MM` | `0.1` |
| `P1_ANGULOS_GRADOS` | `[10.0, 20.0, 30.0]` |
| `P1_ANGULOS_CON_CRITERIO` | `[10.0, 20.0]` |
| `P1_DIRECCIONES` | `[[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]` |
| `P1_MISMA_CUENCA_MM` | `0.05` |
| `P2_MEDIANA_MM` | `0.2` |
| `SEXTANTES` | `{"upper": {"derecho_posterior": [14, 15, 16, 17, 18], "anterior": [11, 12, 13, 21, 22, 23], "izquierdo_posterior": [24, 25, 26, 27, 28]}, "lower": {"derecho_posterior": [44, 45, 46, 47, 48], "anterior": [31, 32, 33, 41, 42, 43], "izquierdo_posterior": [34, 35, 36, 37, 38]}}` |
| `ORDEN_ARCADA` | `{"upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28], "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]}` |
| `TABLERO_MM` | `1.0` |
| `LEGIBLE_COBERTURA_MIN` | `0.5` |
| `LEGIBLE_SUELO_MAX_MM` | `0.2` |
| `OBJETIVO_MEDIANA_MM` | `0.6` |
| `OBJETIVO_P95_MM` | `1.2` |
| `REFERENCIA_ANTIGUA_D50_MM` | `2.14` |
| `MEJORA_D50_MM` | `1.07` |
| `PIEZA_MEDIANA_MAX_MM` | `1.0` |
| `DIFERENCIA_ANGULO_MAX_GRADOS` | `1.0` |
| `DIFERENCIA_CENTROIDE_MAX_MM` | `0.5` |
| `CALIBRACION_K` | `2.0` |
| `CALIBRACION_FRACCION_MIN` | `0.8` |
| `CALIBRACION_FRACCION_VACIA` | `0.99` |
| `CALIBRACION_U_MEDIANA_MAX_MM` | `1.0` |
| `VERTICAL_MEDIANA_MAX_MM` | `0.6` |
| `DIENTES_SESGO_P4B` | `[11, 21, 22, 26]` |
| `E3_NAVAJA_P95_MAX_MM` | `0.3` |
| `E3_MEDIANA_MAX_MM` | `0.8` |
| `E3_P95_MAX_MM` | `1.5` |
