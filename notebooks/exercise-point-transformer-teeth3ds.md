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
| **Modelo** | Point Transformer de segmentación, ~4.6M parámetros |
| **Features** | ablación: `pos-only` (geometría) · **normales** · **gris CBCT sintético** (swap-ready) · normal+gris |
| **Loss** | `nll_loss` **ponderada por clase** (*median-frequency balancing*, peso topado a 50) |
| **Épocas / HW** | 20 · RTX 5070 (sm_120), torch cu128 · ~4×100 s (4 configs) |

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

## Resultados — ablación de features (A2)

Primero se vio que **`pos-only` no generaliza** (memoriza el train, `tooth_acc` train ~0.76 pero test
~0.08). La pregunta de A2: **¿qué feature por punto arregla la generalización, y cuánto aporta cada
una?** Ablación con el **mismo split/semilla** (`tooth_acc` test = media de 5 sorteos):

| config | in_ch | train | **test** | Δ vs pos-only |
|---|---|---|---|---|
| pos-only (geometría) | 1 | 0.759 | **0.078** | — |
| **normales** | 3 | 0.831 | **0.842** | **+0.765** |
| gris CBCT sintético | 1 | 0.770 | **0.793** | +0.715 |
| normal + gris | 4 | 0.806 | 0.713 | +0.635 |

- **`pos-only` no transfiere** (test 0.08): con input por punto constante el modelo se apoya solo en la
  **posición absoluta**, que cambia de una arcada a otra.
- **Cualquier descriptor LOCAL por punto rompe la degeneración**: normales 0.84, gris 0.79.
- **Son redundantes**: `normal+gris` (0.71) **no supera** a ninguno solo — combinar incluso perjudica
  (−0.13 vs normales). Un buen descriptor local **basta**.

---

## Conclusiones

1. **La palanca no es «más datos» ni «el CBCT»**, sino dar al modelo **un descriptor geométrico LOCAL
   por punto**. Las **normales** son lo más simple y lo mejor (`tooth_acc` test **0.84**). `pos-only`
   se apoya en posición absoluta y no generaliza.

2. **El gris CBCT *sintético* no es inerte, pero no valida el CBCT real.** Solo, sube a 0.79 — pero es
   un **proxy geométrico** (grosor/densidad local de la malla), redundante con las normales. Que un
   gris *sintético* ya casi sature la mejora significa que **con pura geometría se llega a ~0.84**; el
   **listón para el CBCT real** es **superar normales-solo** aportando la **densidad interna**
   (esmalte/dentina/hueso) que la geometría no tiene.

3. **El valor del CBCT se realiza por FUSIÓN, no pintando gris por vértice.** El paper líder —**DDMF**
   ([arXiv:2203.05784](https://arxiv.org/abs/2203.05784), *Patterns* 2023, **503 pacientes CBCT+IOS
   emparejados**)— segmenta cada modalidad por separado y las **fusiona por registro** (RANSAC-FPFH +
   ICP multiescala → **0.17 mm** ASSD); el gris nunca es feature de vértice. Su rama de malla (IOSNet)
   separa encía+FDI con **CE + centroid loss + boundary loss**, sin intensidad. → el `cbct-agent` va
   por segmentación propia + registro, no por gris-por-vértice.

### Caveats metodológicos

- **`tooth_acc(test)`** final = **media de 5 sorteos** de `FixedPoints` (que re-submuestrea también en
  test) → quita la varianza de submuestreo.
- **`mIoU` no informativo** aquí (convención parte-ausente → 1.0); por eso el diagnóstico es `tooth_acc`.

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
- Da la **evidencia medida** de que el CBCT entra por **fusión** (segmentar CBCT aparte + registro
  CBCT↔malla, estilo DDMF), **no** pintando gris por vértice. Ese es el diseño del `cbct-agent`.
- El canal de gris queda **swap-ready** en el código (muestreo trilineal de un volumen); cuando haya
  CBCT real registrado, solo se sustituye el volumen sintético.
- Validar el FDI predicho contra el informe clínico sería el `fdi-consistency-agent` (ADR 003).

**Siguiente paso (A3):** **boundary loss** (frontera diente-encía, top-5% puntos) + **centroid loss**
al estilo IOSNet/DDMF — mejora el borde con la encía usando Teeth3DS+, **sin CBCT**.
