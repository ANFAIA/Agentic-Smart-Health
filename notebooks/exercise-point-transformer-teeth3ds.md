# `exercise-point-transformer-teeth3ds.ipynb` — Segmentación de dientes (FDI) por punto

Prototipo del **`segmentation-agent`**: dado el escaneo intraoral como nube de puntos,
etiquetar **cada punto con su diente (FDI)** o encía. Reproduce el modelo y el bucle del
[ejemplo oficial de Point Transformer (PyG)](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_segmentation.py);
lo único propio es el *loader* (mallas `.obj` + labels `.json` → PyG), la **loss ponderada por
clase**, el diagnóstico **`tooth_acc`** y el IoU calculado a mano.

> Es un **spike de validación técnica**, no el sistema final ni un resultado clínico: sirve para
> de-arriesgar una decisión de arquitectura (¿basta la geometría para segmentar FDI?).

---

## El experimento

| | |
|---|---|
| **Tarea** | segmentación semántica por punto — 32 clases (encía + 31 FDI) |
| **Datos** | **Teeth3DS+ completo**: 600 mallas / 300 pacientes (`data/raw/teeth3ds/`, gitignored) |
| **Muestra** | ~117k vértices/malla → submuestreo a **2048 puntos** |
| **Split** | **por paciente** 240 train / 60 test (80/20, `seed 42`, sin fuga entre train y test) |
| **Modelo** | Point Transformer de segmentación, ~4.6M parámetros, **`pos-only`** (solo geometría XYZ) |
| **Loss** | `nll_loss` **ponderada por clase** (*median-frequency balancing*, peso topado a 50) |
| **Épocas / HW** | 20 · RTX 5070 (sm_120), torch cu128 · ~4.5 s/época (~108 s total) |

### Por qué la loss ponderada

Sin pesos, la **encía** (clase 0, **44%** de los puntos) domina el gradiente y el modelo colapsa
prediciéndola en todo. Con frecuencia inversa (peso = `mediana(freq)/freq`, topado a 50) un diente
pesa **~147×** más que la encía, de modo que el modelo no puede «ganar» ignorando los dientes.

### Por qué `tooth_acc` (y no `mIoU`)

- **`tooth_acc`** = acierto **solo** en los puntos de diente (no-encía). Es el número honesto: mide si
  el modelo segmenta dientes o solo pinta encía.
- **`mIoU`** queda **inflado** por la convención «parte ausente en la muestra → IoU 1.0», así que se
  mantiene en ~0.4–0.5 sin informar de nada. Por eso el diagnóstico principal es `tooth_acc`.

---

## Resultados

| Métrica | Época 1 | Época 20 |
|---|---|---|
| `loss` | 2.88 | **0.57** (satura) |
| `acc` global | 0.10 | 0.70 |
| **`tooth_acc` train** | 0.16 | **0.81** |
| **`tooth_acc` test** | 0.03 | **0.055** (plano ~0.05, rango 0.001–0.15, sin tendencia) |

- **El modelo aprende el train**: `tooth_acc(train)` sube de 0.16 a **0.81** y la loss satura → con
  480 mallas hay señal de sobra y la optimización funciona.
- **No generaliza**: `tooth_acc(test)` se queda **plano en ~0.05**, sin tendencia ascendente. El hueco
  train/test es enorme (0.81 vs 0.05).

---

## Conclusiones

1. **El cuello de botella NO es la cantidad de datos.** Con 480 mallas de entrenamiento el modelo
   aprende el train sin problema; lo que falla es la **generalización a pacientes nuevos**.

2. **La geometría sola tiene techo.** Un modelo `pos-only` **memoriza** el train pero **no identifica
   el FDI** en arcadas no vistas: la posición no distingue la simetría izquierda/derecha (11↔21) ni
   dientes vecinos casi idénticos. Ese es el límite estructural del experimento.

3. **La palanca son *features por punto*** (comentario #2 del jefe). Añadir el **gris del CBCT** —o al
   menos normales de superficie— como canal de entrada (`in_channels>1`) da al modelo la información
   que la geometría no tiene (la densidad discrimina esmalte/dentina/hueso/encía). Hoy bloqueado por
   falta de **CBCT emparejado** con la malla → vía **sintética** (voxelizar la malla a un campo de
   atenuación; issue G).

### Caveats metodológicos

- **`tooth_acc(test)` ruidoso** (bota 0.001 → 0.15 → 0.03): en parte es **artefacto de medición** —
  `FixedPoints` re-submuestrea 2048 puntos aleatorios *también en test*, así que cada época evalúa
  sobre un sorteo distinto. Un eval **determinista** (o promediar varios sorteos) lo limpiaría, pero
  **no** cerrará el hueco de generalización.
- **`mIoU` no informativo** aquí (convención parte-ausente → 1.0).

---

## Cómo reproducirlo

Requiere el **kernel GPU dedicado** `Dental GPU (3DGS)` (torch cu128 + `pyg-lib`) — ver
[`README.md` §04](README.md) para el montaje del venv. Necesita el dataset en
`data/raw/teeth3ds/` (gitignored).

```bash
~/.venvs/dental-gpu/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/exercise-point-transformer-teeth3ds.ipynb
# o interactivo (selecciona el kernel "Dental GPU (3DGS)"):
~/.venvs/dental-gpu/bin/jupyter notebook
```

---

## Conexión con el proyecto

- Es el **esqueleto del `segmentation-agent`** (entrada nube de puntos → salida FDI por punto).
- Da la **evidencia medida** de por qué ese agente necesita el **twin fusionado**: sin features del
  CBCT sobre la malla, la segmentación FDI no generaliza.
- Validar el FDI predicho contra el informe clínico sería el `fdi-consistency-agent` (ADR 003).

**Siguiente paso (A2):** cablear un canal de feature **CBCT-sintético** (gris por vértice desde un
campo de atenuación voxelizado de la malla) y re-medir `tooth_acc(test)`.
