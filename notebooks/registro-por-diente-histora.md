# Registro por diente sobre `histora` — qué se sostiene y qué no

Ficha del experimento que responde a una pregunta de arquitectura: **¿merece la pena
registrar cada diente por separado, o el registro global del arco ya es lo mejor que se
puede sacar?**

Importa porque cambia el producto. Un registro global entrega una rígida para todo el
arco; uno por diente entrega **una 4×4 por pieza**, que es pequeña, invertible,
auditable y aprobable de una en una por un clínico. Es un artefacto de agente; una nube
de puntos no lo es.

> Es un **spike de validación** sobre un solo paciente, no un resultado clínico.

**Código:** [`packages/fusion-agents/src/fusion_agents/por_diente.py`](../packages/fusion-agents/src/fusion_agents/por_diente.py)
y [`scripts/segmentar_fdi.py`](../scripts/segmentar_fdi.py).

---

## El número

Residuo **en la mitad de vértices retenida** —la que la transformación no vio— sobre los
dos escáneres mandibulares de `histora`, 13 dientes con código FDI:

| `trim` del ICP global | global | **por diente** | mejora |
|---|---|---|---|
| 0,7 | 0,154 mm | **0,129 mm** | 13/13 |
| 1,0 | 0,273 mm | **0,148 mm** | 13/13 |

**Lo que sostiene la técnica no es que baje, es que sea estable.** El registro global se
mueve un 77 % al tocar un hiperparámetro de ajuste; el registro por diente, un 15 %. Eso
es lo que distingue capturar geometría de absorber la arbitrariedad del ajuste global.

---

## Tres correcciones, dos de ellas a conclusiones propias

### 1 · El `trim` fabrica correlaciones espaciales — corrige una conclusión anterior

Al mirar el patrón espacial de los desplazamientos apareció que la traslación por diente
crecía con la distancia al centro del arco (ρ +0,67, p 0,033), simétrico y con mínimo en
el medio. Se interpretó como **deriva de cosido del escáner**: el ICP fija la pose donde
hay más puntos y el error se acumula hacia los extremos.

**Es falso, y el propio método lo fabricaba.** `trim=0.7` descarta el 30 % de
correspondencias peores, que están precisamente en los extremos del arco, así que ajusta
la pose al centro *por construcción*. Repitiendo sin recorte:

| arcada | `trim` 0,7 | `trim` 1,0 |
|---|---|---|
| inferior | ρ **+0,67** (p 0,033) | ρ **−0,73** (p 0,016) |
| superior | ρ +0,71 (p 0,022) | ρ +0,41 (p 0,244) |

El signo **se invierte** al cambiar un solo hiperparámetro. Una correlación que hace eso
no describe el paciente, describe el ajuste.

> **Regla que queda:** ningún patrón espacial de los desplazamientos por diente se cuenta
> sin haberlo comprobado a los dos valores de `trim`. Por eso el parámetro se expone en
> `registra_dientes` en vez de fijarse dentro.

### 2 · Segmentar los dos momentos por separado alinea superficies distintas

Etiquetando cada escaneo por su cuenta, un mismo diente cubre extensiones distintas en
cada momento y el ICP empareja cosas que no se corresponden:

| | por separado | etiqueta **transferida** |
|---|---|---|
| residuo por diente | 0,374 mm | **0,129 mm** |
| rotaciones | 0,9 – **39°** | 0,4 – **4,2°** |
| el `46` | 3,26 mm / 39° | 0,33 mm / 2,5° |

Una rotación de 39° en un molar no es anatomía, es desajuste de recorte. Se etiqueta
**una vez**, sobre el escaneo de mejor calidad, y se transfiere por vecindad al otro ya
alineado (`transfiere_etiquetas`).

### 3 · La arcada superior no es un control limpio

Se corrió como control y hay que declarar sus tres avisos: `marco_arcada` devuelve razón
de orientación **0,75** (su propio umbral dice desconfiar), el registro global es
**0,517 mm con 0,83 de solape** frente a 0,275/0,95 del inferior, y las rotaciones por
diente salen de **8–17°**. El registro por diente mejora igualmente 11/11, pero las
cifras absolutas de esa arcada no se pueden usar.

---

## Cómo se separan los dientes

**Por código FDI**, con el Point Transformer de
[`exercise-point-transformer-teeth3ds.md`](exercise-point-transformer-teeth3ds.md) más la
agregación de `tooth-aggregation`. Sobre `histora` salen **14 dientes comunes a los dos
momentos** (31-37, 41-47) con el orden anatómico exacto a lo largo del arco (ρ ±1,00).

### El camino que se descartó, y funcionaba

Antes de tener FDI se separaron los dientes por **mínimos del perfil de altura a lo largo
del arco**. Funciona: 12 troneras, 13 segmentos, anchura mediana **8,0 mm**, que es
exactamente un diente inferior. Dos cosas que costaron y merecen quedar escritas:

- **Por islas conexas NO funciona.** El escáner rellena el punto de contacto
  interproximal y la malla queda unida: salían 6 bloques, no 14 dientes.
- **El perfil crudo tampoco.** Lo domina la curva de Spee —7,6 mm de amplitud frente a
  2,0 mm de la señal útil—, así que buscar mínimos ahí encuentra los valles de la forma
  del arco y no las troneras. Hay que quitarle la tendencia con una media móvil de 15 mm,
  más ancha que un diente y más estrecha que el arco.

Se abandonó porque un segmento de arco **no tiene identidad anatómica**: sirve para medir
un residuo, no para decir «el 36 se movió» ni para injertar la raíz correcta.

---

## Cómo leer la salida, y qué no se puede publicar todavía

**Lo robusto es el residuo, no el desplazamiento.** La traslación mediana por diente pasa
de **0,152 a 0,471 mm** solo con cambiar el `trim`, porque se mide *contra* el registro
global: si la referencia se mueve, todas las traslaciones se mueven con ella.

**Las rotaciones no se usan.** Un parche de esmalte liso desliza sobre sí mismo sin
penalizar el residuo. `DienteRegistrado.condicion` declara cuándo pasa; conviene mirarlo
antes que la cifra.

Así que las matrices **sirven ya** para bajar el suelo de ruido de una medida. Para
afirmar «esta pieza se desplazó X mm» hacía falta fijar la referencia — resuelto en la
sección siguiente.

---

## La referencia leave-one-out, y el umbral por debajo del cual no se afirma nada

**Código:** [`desplazamientos_relativos`](../packages/fusion-agents/src/fusion_agents/por_diente.py)
· **experimento:** [`scripts/desplazamiento_relativo.py`](../scripts/desplazamiento_relativo.py)

Para medir el diente *X*, el marco se reajusta con **todos los dientes menos X**. Quita
dos contaminaciones del registro global de golpe: X entraba en su propia referencia (si se
mueve, tira del marco contra el que se le mide y reparte su movimiento entre los demás), y
la **encía** entraba también (y cambia de forma entre dos momentos sin que ningún diente
se haya movido).

Lo que se mide pasa a ser desplazamiento **relativo dentro del arco**. No es una
concesión: en un escaneo intraoral **no existe** marco absoluto, porque el escáner no ve
ninguna estructura fija.

### El resultado

| referencia | mediana al cambiar el `trim` | factor | peor diente Δ |
|---|---|---|---|
| global | 0,171 → 0,738 mm | 4,3× | 0,696 mm |
| **leave-one-out** | 0,158 → 0,182 mm | **1,2×** | **0,107 mm** |

El peor diente pasa de moverse 0,696 mm al tocar un hiperparámetro a moverse 0,107 —
por debajo del residuo de la propia medida. El desplazamiento ya es tan estable como el
dato permite.

### El control nulo, que es lo que fija el umbral

`PREVIO → POST HIGIENE` **no** es un control nulo: una higiene no mueve dientes pero sí
quita cálculo, así que la superficie cambia de verdad y el ICP lo lee como desplazamiento.
El control limpio son los **dos escaneos independientes de la misma visita** (87.417 y
93.860 vértices), donde no cambió ni la biología ni los depósitos. Todo lo que se informe
ahí es falso por construcción:

| referencia | `trim` | mediana | **máximo** |
|---|---|---|---|
| global | 0,7 | 0,112 | 0,295 |
| global | 1,0 | **1,356** | **2,027** |
| leave-one-out | 0,7 | 0,106 | 0,389 |
| leave-one-out | 1,0 | 0,156 | 0,535 |

⚠️ **A `trim` 0,7 la referencia global sale mejor** (0,295 frente a 0,389). Lo que la hace
inservible no es ser peor: es que a 1,0 informa **2,027 mm de desplazamiento sobre dos
escaneos de la misma boca**, y no hay forma de saber de antemano en qué régimen estás. La
referencia relativa no gana en todos los ajustes — gana en **no depender del ajuste**, que
es lo que permite citar un umbral.

### Y el par real no lo supera

| | mediana | p90 | max |
|---|---|---|---|
| control nulo | 0,106 / 0,156 | 0,349 / 0,375 | 0,389 / 0,535 |
| `PREVIO → POST HIGIENE` | 0,158 / 0,182 | 0,342 / 0,308 | 0,398 / 0,398 |

Indistinguibles. **Ningún diente se movió de forma detectable**, que es lo esperable en un
par pre/post higiene: el test nulo **pasa**. Y deja el número que hacía falta —

> **Umbral de detección: ~0,4 mm** (p90 del control nulo 0,35–0,38; máximo 0,39–0,54). Por
> debajo de eso no se escribe «esta pieza se desplazó», por estable que sea la cifra.

Eso es lo que le da regla a la visualización por color que pide el producto: se pinta un
diente cuando su desplazamiento relativo supera el umbral **a los dos `trim`**, y no antes.

---

## El injerto de raíz del CBCT — resultado NEGATIVO

Se intentó lo obvio con estas matrices: conservar la corona del escáner (0,138 mm de
espaciado) e injertarle la raíz del CBCT, que es lo único que el escáner no puede ver.

**Casi toda la cadena funciona:**

| paso | resultado |
|---|---|
| aislar la arcada mandibular pese al metal | 49,5 × 36,9 × 19,5 mm ✓ |
| registrar IOS ↔ CBCT (rígida compuesta, verificada) | 0,488 mm ✓ |
| transferir los códigos FDI al esmalte del CBCT | 60 %, 11 dientes ✓ |
| **crecer la etiqueta hacia la raíz** | **✗** |

El último paso no sale. Se creció desde la corona por vóxeles de dentina (HU ≥ 1100) con
dos cotas anatómicas —25 mm de largo máximo y solo hacia apical—, y el resultado es:

```
 FDI      mm3   largo   ancho  veredicto
  31      515    18.5    13.5  ok
  33      258    11.7    11.8  ok
  34     1826    27.9    24.0  DESBORDADO (topó la cota)
  41     1877    27.9    24.7  DESBORDADO (topó la cota)
  42      632    17.8    23.0  ok
  47     1108    25.8    20.8  DESBORDADO (topó la cota)
```

**Los seis desbordados topan la cota, los seis.** Lo que los paró fue el recorte
geométrico, no un borde anatómico — ese era el criterio fijado de antemano para
distinguir señal de recorte, y sale recorte. Y los que pasan por volumen fallan por
**anchura**: el `42` es un incisivo, mide 5-6 mm de ancho, y sale con 23,0.

**Por qué, y no es del algoritmo.** El ligamento periodontal es la única frontera real
entre raíz y hueso y mide **0,15–0,38 mm**; el vóxel de este CBCT es **0,30 mm**. La
frontera no está muestreada. Ninguna cota geométrica crea un borde que el dato no
contiene: solo esconde la fuga.

---

## Margen gingival → cresta ósea — también NEGATIVO, y corrige lo anterior

> ⚠️ **Corrección.** Aquí decía: «esto NO bloquea la medida periodontal del proyecto;
> margen → cresta necesita el borde hueso ↔ tejido blando, que son cientos de HU». Se
> midió y **es falso**. El borde hueso ↔ blando sí es resoluble, pero la medida no lo
> necesita en el vacío: lo necesita **junto a la raíz**, y ahí vuelve a hacer falta
> distinguir raíz de hueso. Es el mismo muro, no una frontera distinta.

Se lanzó un rayo desde cada margen hacia apical, separado lateralmente del diente para no
medir el propio esmalte, barriendo la separación de 1,5 a 3,0 mm. **21/21 secciones** con
margen detectado, y el marco verificado (cresta a 2000 HU, base a 129: el eje oclusal
apunta donde debe).

| lado | sitios con cresta (HU≥700) | dispersión en el barrido |
|---|---|---|
| vestibular | **1/21** | — |
| lingual | 6/21 | **3,4 – 8,7 mm** |

**Los dos lados fallan, y por motivos distintos.** A vestibular no hay señal mineralizada:
el máximo de HU tiene mediana **137/126/105** a 1/2/3 mm de separación, durante 10 mm
apicales. A lingual sí la hay (11/21 sitios pasan de 700 HU), pero la cresta se desplaza
hasta **8,7 mm** al mover el rayo 1,5 mm, y varios sitios dan 0,00 mm porque el rayo
**arranca ya dentro** de tejido duro.

El criterio estaba fijado antes de correrlo —«una medida que cambia mucho con el barrido
no es una medida»— y no lo pasa. Así que **no se puede decir «el método funciona y falta el
dato»**: el control interno lingual, donde el hueso sí está, tampoco da un número.

**La causa es la del injerto.** Para saber si un vóxel duro es cresta ósea o es raíz hace
falta el ligamento periodontal, y no está muestreado. Medido: el tejido duro más cercano a
cada margen está a 1,4 mm (HU≥400) o 2,9 mm (HU≥1000), pero su componente **apical es
negativa** (−0,7 a −1,05 mm) — está *coronal* al margen. Es el diente, no el hueso, y a
ningún umbral se separan.

### Un hallazgo aparte, que es pregunta para los doctores

La asimetría vestibular/lingual es demasiado marcada para ser solo ruido, y el caso es de
**recesión**. Una tabla vestibular fina o dehiscente en el sector anteroinferior es lo
esperable clínicamente, y por debajo de 0,30 mm el promedio parcial la borra. **No se
afirma que el paciente no tenga hueso vestibular**: se afirma que en este CBCT no hay
señal ≥200 HU ahí, y que decidir entre «no existe» y «no se ve» no es cosa del script.

Verificado que no es un signo invertido, que es el error que ya apareció tres veces en
esta sesión: las normales salen hacia fuera por dos vías independientes (el volumen con
signo crece al inflar la malla, y el producto con el radial del arco da +0,95 en 21/21).

---

## Dos preguntas de diseño, respondidas con números

**Experimento:** [`scripts/promedio_y_escala.py`](../scripts/promedio_y_escala.py)

### 1 · Promediar la matriz de 3 dientes y usarla para el resto — **NO funciona**

Importaba porque si funcionase, el artefacto sería **una** matriz en vez de trece y
segmentar catorce piezas dejaría de hacer falta. Se probó de las dos formas en que se puede
leer la frase, y sobre **las 286 combinaciones de 3** entre los 13 dientes — no un trío
elegido a dedo, que permitiría contar el resultado que más convenga.

| lectura | promediada | propia | global | gana la propia | **peor que no tocar nada** |
|---|---|---|---|---|---|
| cruda (las 4×4 tal cual) | 0,288 mm | 0,127 | 0,155 | 2860/2860 | **2629/2860 (92 %)** |
| local (rotación sobre el centroide propio) | 0,180 mm | 0,127 | 0,155 | 2860/2860 | **2071/2860 (72 %)** |

**El ajuste propio de cada diente gana en el 100 % de los casos**, y ni el mejor decil de
tríos (p10 = 0,131 mm en la lectura buena) llega al 0,127 del ajuste propio. Peor: aplicar
la matriz promediada **empeora** respecto a dejar el diente donde lo puso el registro global
en el 72 % de los casos. Un movimiento promedio es el movimiento equivocado para todos.

Dos cosas que merece la pena dejar escritas, porque el «no» no es lo único útil aquí:

- **Si se va a promediar, hay que hacerlo en coordenadas locales.** La lectura cruda sale
  0,288 y la local 0,180 — un 60 % peor solo por el sistema de referencia. La traslación de
  una rígida depende del origen, así que una rotación pequeña alrededor de un origen lejano
  produce un desplazamiento grande al aplicarla a otra pieza. Aun así, ninguna funciona.
- **No es heterogeneidad biológica.** En el control nulo —dos escaneos de la misma visita,
  donde nada se movió— pasa lo mismo (0,128 frente a 0,104, peor que no tocar nada en el
  70 %). Lo que capturan las matrices por diente es geometría local y ruido de escaneo, y
  eso es específico de cada pieza por naturaleza. No hay nada que promediar.

### 2 · La escala (7º grado de libertad) — **no se puede medir, y no aporta**

Un diente no cambia de tamaño entre dos escaneos, así que una escala ≠ 1 solo puede ser del
escáner. Añadida como 7º grado de libertad por diente, la mejora en la mitad retenida es de
**0,0003 mm** (0,1274 → 0,1271). Eso ya la descarta, pero el proceso de medirla dio algo más
útil: **cuatro estimadores, tres roto**s.

| estimador | qué pasó |
|---|---|
| ICP con escala, arco completo | **colapsa**: A→B da 0,904 y B→A 1,016, producto **0,913** ≠ 1 |
| ICP con escala, por diente | desviación entre piezas ~8.000 ppm, sube a **70.000** sin recorte |
| razón de distancias entre centroides de diente | razón individual dispersa **±6-8 %** |
| **firma radial con la rígida fijada** | se comporta — es el que se usa |

El colapso del primero es instructivo: **encoger acerca todos los puntos al centroide del
objetivo y baja la distancia al vecino más próximo pase lo que pase con la escala real**. Y
las diagonales de las dos mallas son 88,25 y 88,50 mm, así que un −9,6 % es imposible por
construcción — el estimador estaba dando un número absurdo con toda tranquilidad.

El que funciona no **optimiza** la escala, la **lee**: con la rígida fijada, un error de
escala `s` desplaza cada punto radialmente en `(s−1)·r`, y eso es una recta sobre ~75.000
puntos emparejados que no puede colapsar porque no se le deja mover la rígida.

| par | ida | vuelta | suma | veredicto |
|---|---|---|---|---|
| control nulo (misma visita) | +432 ppm | −608 ppm | −177 | ✅ cambia de signo |
| `PREVIO → POST HIGIENE` | **−1550 ppm** | −162 ppm | −1712 | ❌ **no cambia de signo** |

**En el control nulo el estimador se comporta como una escala** (cambia de signo al invertir
el par) y da el suelo: **±500 ppm**, o 0,03 mm sobre 60 mm de arco. **En el par real no.**
Los −1550 ppm no son una escala: si lo fueran, el sentido inverso daría +1550.

> **Conclusión: no se añade el grado de libertad.** Aporta 0,0003 mm, su estimación directa
> es inestable hasta dar −9,6 %, y el único estimador fiable dice que lo que hay en el par
> real no es una escala.

**Hipótesis de qué es** —y es hipótesis, no medida—: la higiene **quita cálculo**, así que
la superficie se mueve hacia dentro donde había depósitos. Eso imita un encogimiento en un
sentido y no se invierte al cambiar de dirección. Encaja con que el control nulo, donde no
se quitó nada, sí sea simétrico. Comprobarlo pide localizar los depósitos, que no se ha
hecho.

Y el `R²` de la regresión radial es **0,006-0,012** en todos los casos: aunque hubiera
escala, explicaría el 1 % de la varianza del residuo. El otro 99 % es ruido de escaneo.

---

## Qué arcada del CBCT es cuál — y una conclusión propia retirada

> ⚠️ **Retirado.** Se afirmó aquí y en dos docstrings que «en esta serie la z crece hacia
> los pies, así que `arcada_superior` devuelve la mandíbula», y que el 0,452 mm commiteado
> se había medido contra la arcada equivocada. **Las dos afirmaciones son falsas.** Salían
> de un heurístico de anchura de arco que, reimplementado, daba la respuesta contraria.

`Serie.z` es `ImagePositionPatient[2]`, e **`IPP` viene ya en coordenadas del paciente**
(LPS: +z craneal). No es una convención del fichero ni depende de `PatientPosition`: el
equipo la resuelve al exportar. Así que z alta es maxilar, y `arcada_superior` era correcta
desde el principio. `Serie.z_es_superior` deja constancia de cuándo esto vale — es `False`
si la serie no trae `IPP` y hubo que ordenar por `InstanceNumber`.

Confirmado por **anatomía**, independiente de la metadata: alejándose del plano oclusal,

| desde el plano oclusal | hacia z baja | hacia z alta |
|---|---|---|
| aire (<−500 HU) a 15 mm | 0,0 % | 10,8 % |
| aire a 33 mm | **0,0 %** | **51,5 %** |
| hueso (≥400 HU) a 33 mm | 23,0 % | 2,5 % |

Hacia z alta aparece una cavidad de aire creciente **dentro** del hueso: seno maxilar y
fosa nasal. Hacia z baja, hueso continuo y cero aire en 33 mm: cuerpo mandibular.

**Lo que sí quedó medido es que el residuo del registro no discrimina la arcada.** Se
registró el escaneo mandibular contra los dos lóbulos: **0,490 mm** contra z_baja y
**0,509** contra z_alta, un 3,8 % de diferencia. Una arcada dental se parece bastante a
otra arcada dental, así que «registrar contra los dos y quedarse con el que ajusta» —que
es lo que yo había puesto en su lugar— **tampoco vale**. Ahora el lóbulo se elige por el
`IPP` cruzado con la etiqueta del nombre del STL, y el segundo ajuste se conserva solo
como aviso: si el otro lóbulo ajusta mejor que el que dice la anatomía, salta.

> **Regla que queda:** un eje anatómico no se supone nunca, se lee de la metadata o se
> mide. En esta sesión el mismo error apareció **tres veces** —eje oclusal en la
> segmentación FDI, eje apical en el crecimiento radicular, y este— y las tres veces el
> síntoma fue un número plausible, no una excepción.

---

## Lo que este experimento pide

Van **tres preguntas medidas** y bloqueadas por la misma resolución: la unión
amelocementaria, la delimitación raíz/hueso y la **cresta ósea junto a la raíz**. Un CBCT
de **FOV pequeño a 0,08 mm** —3,8× más fino, y por debajo del ligamento periodontal— las
desbloquearía.

> ⚠️ **Corrección.** Aquí decía «cuatro preguntas, todas medidas», incluyendo la frontera
> **esmalte/dentina**. Esa cuarta **no está medida**: en el repositorio solo aparecía como un
> umbral heredado (`HU ≥ 1100`) usado para crecer la etiqueta en el injerto, que es un
> parámetro, no una medida de si el borde se resuelve.

Se intentó medirla y **fallaron tres planteamientos**, los tres por el mismo motivo de
fondo — no aislar un diente:

1. **Perfiles desde la superficie del escáner hacia dentro.** El control positivo no medía
   nada: el rayo arrancaba *en* la malla, o sea ya dentro del esmalte (1388 HU a
   profundidad 0), y nunca cruzaba el borde exterior.
2. **Promediar esos perfiles por diente.** Difumina la unión **por construcción**: el
   esmalte mide ~2,5 mm en una cúspide y ~0,5 en el cuello, así que la media de muchos
   escalones a distinta profundidad es una rampa. La rampa que salió (1380 → 840 HU en
   1,5 mm) era eso, no anatomía.
3. **Profundidad con signo por transformada de distancia sobre el CBCT.** Quita el error
   del registro, pero la máscara de tejido duro (`HU ≥ 900`) **fusiona los dientes con el
   hueso alveolar**, así que la profundidad es «hacia dentro del blob» y no «hacia dentro
   de un diente». Se nota en que el HU **sube** con la profundidad (995 → 1669), al revés
   de la anatomía: el esmalte es la capa exterior y es la más densa.

De ese tercer intento sí queda un número utilizable, porque no depende de aislar el diente:
el **borde más abrupto que este CBCT resuelve** —esmalte contra tejido blando— tiene un
gradiente de **1050 HU/mm**, o sea que transiciona en ~1 vóxel, y el **ruido dentro de una
meseta es ±300 HU**. Con eso, un contraste esmalte/dentina de los 700-1000 HU que da la
literatura sería 2-3× el ruido: detectable en principio. Lo que falta no es contraste, es
**aislar la corona de un diente** del hueso que la rodea.

Medirla de verdad pide una máscara por pieza en el propio CBCT (no transferida desde el
escáner, que mete 0,49 mm) y, para poder llamar «esmalte» y «dentina» a los dos niveles,
una calibración — los grises de un CBCT no son HU.

Y una que no es de resolución sino clínica, para los doctores: **¿hay tabla ósea vestibular
en el sector anteroinferior de este paciente?** Si la respuesta es que no, la medida
periodontal que el proyecto persigue no es «margen → cresta» ahí, sino la propia
dehiscencia — y eso cambia el producto, no solo el método.
