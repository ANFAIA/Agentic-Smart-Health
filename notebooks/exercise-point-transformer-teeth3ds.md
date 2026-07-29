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

## Resultados — A2 · ablación de features

Primero se vio que **`pos-only` no generaliza** (memoriza el train pero el `tooth_acc` test se queda
en ~0.01–0.08). La pregunta de A2: **¿qué feature por punto arregla la generalización, y cuánto aporta
cada una?** Ablación con el **mismo split/semilla** (`tooth_acc` test = media de 5 sorteos):

| config | in_ch | train | **test** |
|---|---|---|---|
| pos-only (geometría) | 1 | 0.78 | **0.01** |
| normales | 3 | 0.83 | **0.76** |
| gris CBCT sintético | 1 | 0.73 | **0.74** |
| normal + gris | 4 | 0.83 | **0.76** |

- **`pos-only` no transfiere** (test ~0.01–0.08): con input por punto constante el modelo se apoya solo
  en la **posición absoluta**, que cambia de una arcada a otra.
- **Cualquier descriptor LOCAL por punto rompe la degeneración** → salto a **~0.74–0.84**.
- Las tres configs con features son **estadísticamente indistinguibles** y el gris **no aporta** sobre
  las normales → **redundantes**. Un buen descriptor local **basta**.

> **Varianza.** Los valores oscilan **±~0.05–0.1 entre ejecuciones** (no-determinismo GPU + submuestreo
> aleatorio). Lo robusto es lo **cualitativo** (pos-only cae; una feature local lo arregla; el gris no
> suma), no el 2º decimal. Se adopta **normales** por ser la más simple.

## Resultados — A3 · boundary + centroid loss (estilo IOSNet/DDMF)

Sobre la mejor config (**normales, sin CBCT**), dos pérdidas extra que atacan el **borde diente-encía**:
**boundary loss** (CE sobre el 10% de puntos frontera) + **centroid loss** (centro predicho ≈ centro
real por diente). Se mide además el acierto **en la frontera**:

| | `tooth_acc` | **frontera** |
|---|---|---|
| normales · CE | 0.758 | 0.657 |
| + boundary + centroid | 0.792 | 0.707 |
| **Δ** | **+0.034** | **+0.051** |

Mejora **modesta pero consistente**, y **mayor en la frontera** (+0.05) que global (+0.03) — justo
donde se diseñó. Coste: ~172 s vs ~99 s (la centroid loss añade overhead).

---

## Conclusiones

1. **La palanca no es «más datos» ni «el CBCT»**, sino dar al modelo **un descriptor geométrico LOCAL
   por punto**. Las **normales** son la opción más simple y tan buena como cualquier otra feature
   (~0.76–0.84). `pos-only` se apoya en posición absoluta y no generaliza.

2. **El gris CBCT *sintético* no es inerte, pero no valida el CBCT real.** Solo, sube a ~0.74 — pero es
   un **proxy geométrico** (grosor/densidad local de la malla), redundante con las normales. Que un
   gris *sintético* ya sature la mejora significa que **con pura geometría se llega a ~0.8**; el
   **listón para el CBCT real** es **superar a normales-solo** aportando la **densidad interna**
   (esmalte/dentina/hueso) que la geometría no tiene.

3. **El valor del CBCT se realiza por FUSIÓN, no pintando gris por vértice.** El paper líder —**DDMF**
   ([arXiv:2203.05784](https://arxiv.org/abs/2203.05784), *Patterns* 2023, **503 pacientes CBCT+IOS
   emparejados**)— segmenta cada modalidad por separado y las **fusiona por registro** (RANSAC-FPFH +
   ICP multiescala → **0.17 mm** ASSD); el gris nunca es feature de vértice. Su rama de malla (IOSNet)
   separa encía+FDI con **CE + centroid loss + boundary loss**, sin intensidad. → el `cbct-agent` va
   por segmentación propia + registro, no por gris-por-vértice.

4. **Las pérdidas boundary/centroid (A3) mejoran el borde con la encía** usando solo Teeth3DS+ (+0.05
   en frontera) — la receta de IOSNet, sin CBCT.

### Caveats metodológicos

- **Varianza ±~0.05–0.1 entre ejecuciones** (no-determinismo GPU + submuestreo): fiarse de lo
  cualitativo, no del 2º decimal.
- **`tooth_acc(test)`** final = **media de 5 sorteos** de `FixedPoints` (que re-submuestrea también en
  test) → quita parte de la varianza de submuestreo.
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

**Estado.** A1 (split real) · A2 (ablación de features) · A3 (boundary + centroid loss) ✅. El canal de
gris queda **swap-ready** para el CBCT real. Siguientes palancas naturales cuando haya datos: **CBCT
real por fusión** (registro CBCT↔malla estilo DDMF) para el `cbct-agent`, y validar el FDI contra el
informe (`fdi-consistency-agent`).
