# Numeración FDI en el CBCT con el modelo oficial de ToothFairy2: caso B

> **Estado (2026-11-19):** la ejecución 1 es técnicamente correcta y **no sirve** para la
> identidad de los dientes superiores: el modelo mezcla las dos arcadas. Se atribuye a que
> el caso queda fuera de lo que vio al entrenar. El experimento 2 queda **escrito antes de
> ejecutarse** (§3). Nada de esto es un TRE ni una validación clínica: es la salida de un
> modelo externo (Capa 3) descrita con números.

## 1 · Qué se probó y por qué

El segmentador propio del CBCT es binario, y la separación entre piezas en el CBCT sale de
las semillas del escáner, que arrastran sus errores. ToothFairy2 (Bolelli et al., MedIA
2026; CVPR 2025) etiqueta cada diente por su código FDI en 480 CBCT. Su sitio publica
checkpoints de nnU-Net. Si uno de ellos separa bien las piezas de **nuestro** CBCT, la
numeración del CBCT deja de depender del escáner.

- **Checkpoint**: el entrenado sin espejo izquierda-derecha (`onlyMirror01`), elegido antes
  de ver ninguna salida porque el espejo lateral sin intercambio de etiquetas confunde 1x
  con 2x. Es una **PlainConvUNet de 30,8 M de parámetros** aunque la carpeta diga ResEncL,
  y una foto a **mitad de entrenamiento**: época 266 de 1000, un solo fold. Dice interno 0,89.
- **Licencia**: la web dice CC BY-NC-SA 4.0 y los `dataset.json` dicen CC-BY-SA 4.0. Se
  aplica la no comercial.
- **Código**: [`scripts/segmenta_cbct_toothfairy2.py`](../../scripts/segmenta_cbct_toothfairy2.py).

## 2 · Ejecución 1 (commit 439c0ad)

`data/processed/caso-agosto/toothfairy2-run-1/`. Caja de 283×400×400 vóxeles: de 45 mm por
debajo a 40 mm por encima del plano oclusal superior, centrada en las coronas registradas.

**Lo que se verificó** (cuatro revisiones independientes, reproducidas en CPU):

- La inferencia es la del nnU-Net 2.8.1 oficial: red idéntica a la que reconstruye nnU-Net,
  0 vóxeles distintos frente a `predict_single_npy_array` en un subvolumen, y una pasada en
  CPU coincide con la de GPU en el 99,995 % de los vóxeles.
- La orientación es correcta en los tres ejes. La de ToothFairy2 se midió sobre sus
  etiquetas, porque sus cabeceras declaran identidad. Las coronas 1x del escáner caen sobre
  etiquetas derechas y las 2x sobre izquierdas.

**Lo que salió mal:**

| Medida | Ejecución 1 | Etiquetas de ToothFairy2 (3 casos) |
|---|---:|---:|
| Caras de contacto maxilar–mandíbula | 10.034 | 0 |
| Caras de contacto entre grupo superior e inferior | 17.850 | 0 |
| Mandíbula por encima del plano oclusal | 24,1 cm³ | 6,8–9,5 cm³ |

Las coronas 22 y 23 del escáner caen sobre las etiquetas 32 y 33. La numeración superior
va corrida un puesto respecto a la del escáner; son dos modelos en desacuerdo y ninguno
es la referencia.

**Por qué, probablemente:** el caso queda fuera de distribución de dos formas.

1. **Altura.** La caja sube a +40 mm sobre el plano oclusal; una muestra de 16 volúmenes
   de entrenamiento llega como mucho a +31. Por encima de +30 mm el modelo llama
   «mandíbula» a 8,5 cm³.
2. **Grises.** La caja llega a 20.794; el máximo de primer plano visto en entrenamiento es
   5.264. El z-score no recorta, así que esos vóxeles inflan la desviación. Las etiquetas
   11 (94 %), 46 (95 %), «Crown» (81 %) e «Implant» (83 %) caen casi enteras sobre ellos.
   Qué objeto es lo tiene que decir una persona.

Además, la normalización se calcula sobre la caja: otra caja cambia lo que ve la red.

## 3 · Experimento 2: protocolo fijado antes de ejecutar

**Paso previo, medido y no elegido.** [`scripts/mide_toothfairy2_entrenamiento.py`](../../scripts/mide_toothfairy2_entrenamiento.py)
mide sobre las 480 etiquetas del zip, con la misma definición de plano oclusal que en el
caso clínico, la altura de cada volumen por encima y por debajo de ese plano, el volumen de
mandíbula por encima y los contactos entre arcadas.

**La caja** sale de una regla, no de mirar resultados: por encima, la **mediana** de la
altura de entrenamiento sobre el plano oclusal; por debajo, la **mediana** de la altura
por debajo; las dos redondeadas a 0,5 mm. En el plano no cambia (400×400).

**El recorte de grises** lleva la entrada al rango de primer plano de entrenamiento que
declaran los planes del checkpoint, [−1000, 5264], antes del preprocesado oficial. Es una
desviación de nnU-Net y se declara.

**Tres variantes, todas informadas; la principal es 2c:**

| Variante | Caja | Grises |
|---|---|---|
| 2a | la de la regla | sin recortar |
| 2b | la de la ejecución 1 | recortados |
| 2c (principal) | la de la regla | recortados |

2a y 2b existen para saber a qué se debe un cambio, no para elegir.

**Criterios, sin mirar el escáner.** 2c pasa si cumple los tres:

- **C1.** Caras de contacto entre grupo superior (maxilar y 11–28) e inferior (mandíbula y
  31–48): **≤ 1.785**, el 10 % de la ejecución 1.
- **C2.** Mandíbula por encima del plano oclusal: **≤ el p95** de las etiquetas de
  entrenamiento.
- **C3.** Las premisas de orientación se cumplen: izquierda menos derecha > 0 en x LPS
  (la única que prueba la inversión), incisivos menos molares < 0 en y, superiores menos
  inferiores > 0 en z.

**Lo que no se hará:** elegir variante por su parecido con el escáner, ni tocar la caja,
el recorte o los umbrales después de ver una salida. Si 2c falla, el resultado es negativo
y se escribe aquí: el checkpoint no sirve para este caso sin reentrenar.

**Si 2c pasa**, comparar su numeración con la del escáner es otro paso con sus propios
criterios. Las dos son salidas de modelos; la referencia para decidir cuál acierta son los
puntos anatómicos marcados por una persona, que todavía no existen.

### 3.1 · Enmienda del 2026-11-19, antes de ejecutar ninguna variante

Una revisión del código, hecha después de medir el entrenamiento y **antes de ninguna
inferencia del experimento 2**, encontró que el protocolo de arriba no medía lo que dice.
Se corrige así; lo que no se menciona queda como estaba.

**Lo que midió el entrenamiento** (`data/processed/toothfairy2-entrenamiento/medidas.json`,
commit 9bbd4c3): de 480 volúmenes, 356 tienen algún diente superior y con ellos se calcula
la regla. Por encima del plano oclusal llegan a una mediana de **10,2 mm** (p95 26,7; máx.
32,1); por debajo, 42,6 mm. **La caja de la regla es, por tanto, de −42,5 a +10,0 mm.**
Consecuencia que hay que decir: deja fuera casi todas las raíces superiores. El modelo
apenas las vio al entrenar, así que este experimento solo puede responder si separa las
arcadas y numera **coronas**, no raíces superiores.

La regla se mantiene como se escribió, aunque tiene defectos conocidos, porque cambiarla
ahora sería elegir a la vista de los datos:
- un diente mal etiquetado mueve el plano oclusal (P_061, P_067, P_101);
- 30 de los 356 casos tienen solo 1–3 dientes superiores;
- contando solo los casos con al menos 10 dientes superiores, la mediana sube a 13,2 mm.

**1. Ventana común.** Medir cada variante en su propia caja premiaba a la caja corta: la
ejecución 1, que es el fallo, recortada a la caja nueva ya cumplía C2 (5,6 cm³ frente a un
umbral de 9,5). Todos los criterios se calculan ahora **en la misma ventana de alturas,
[−42,5, +10,0] mm**, en las tres variantes (`--ventana-criterios -42.5 10.0`, bloque
`arch_separation.window` de `resumen.json`). En esa ventana, la ejecución 1 da:
- 5.932 caras de contacto entre grupos;
- 22 entre maxilar y mandíbula;
- 5.647 mm³ de mandíbula por encima del plano oclusal;
- 643 mm³ de puente, corona e implante.

**2. Criterios corregidos.** 2c pasa si cumple todos:

- **C1.** Contactos entre grupo superior (maxilar y 11–28) e inferior (mandíbula y 31–48)
  en la ventana: **≤ 593**, el 10 % de la ejecución 1 en la misma ventana. El 1.785 de
  arriba era el 30 % de lo comparable.
- **C1b.** Puente, corona e implante (8, 9, 10) en la ventana: **≤ 643 mm³**, lo de la
  ejecución 1. Sin esto, C1 podría cumplirse con solo reetiquetar la zona confusa como una
  clase que no pertenece a ningún grupo.
- **C2.** Mandíbula con altura en (0, +10] mm: **≤ el p95 de los volúmenes de
  entrenamiento que llegan a +10 mm**, medido en esa misma franja
  (`lower_jawbone_in_window_mm3` de `medidas-ventana.json`, commit d8f27df): **≤ 5.042
  mm³**, p95 de los 180 volúmenes que llegan a +10 mm (mediana 3.739). La ejecución 1 da
  5.647 en esa franja: el criterio la habría rechazado. Los volúmenes de entrenamiento no están recortados en el plano y la
  caja sí, así que el umbral es, si acaso, indulgente.
- **C3.** `orientation_premises.holds == true` en `resumen.json`:
  - izquierda menos derecha > 0 en x LPS, con los dientes 21–28 y 31–38 frente a 11–18 y
    41–48;
  - incisivos (11, 21, 31, 41) menos primeros molares (16, 26, 36, 46) < 0 en y;
  - superiores menos inferiores > 0 en z;
  - todo con centroides ponderados por vóxeles, sin las etiquetas de menos de 50 mm³, y
    falla si falta algún grupo.

**3. Comprobación del registro.** Exigir el 50 % de coronas junto a tejido denso abortaba
las tres variantes: el caso B da 0,470, medido antes de inferir. Pasa a exigir **≥ 0,30 y
al menos 10 veces más que si se hubiera leído el registro del marco centrado** (que da
0,005). Lo que distingue las dos hipótesis es la razón, no el umbral absoluto.

**4. Una ejecución por variante**, en GPU, en `toothfairy2-run-2a`, `-2b` y `-2c`. La
primera ejecución completa de cada una es su resultado.

## 4 · Resultados del experimento 2

Pendiente.
