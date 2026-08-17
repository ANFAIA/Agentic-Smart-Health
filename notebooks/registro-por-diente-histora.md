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

Así que las matrices **sirven ya** para bajar el suelo de ruido de una medida, y **no
sirven todavía** para afirmar «esta pieza se desplazó X mm». Para eso hace falta fijar la
referencia contra algo que no se mueva, y eso no está resuelto.

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

> ⚠️ **Esto NO bloquea la medida periodontal del proyecto.** Margen gingival → cresta
> ósea necesita el borde **hueso ↔ tejido blando**, que son cientos de HU de diferencia,
> no el borde raíz ↔ hueso. Son fronteras distintas y solo una es imposible aquí.

---

## Lo que este experimento pide

Van **tres preguntas distintas** bloqueadas por la misma resolución, todas medidas: la
unión amelocementaria, la frontera esmalte/dentina y la delimitación raíz/hueso. Un CBCT
de **FOV pequeño a 0,08 mm** —3,8× más fino, y por debajo del ligamento periodontal—
desbloquearía las tres.
