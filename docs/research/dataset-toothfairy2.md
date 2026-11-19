# Dataset — ToothFairy2 (CBCT con estructuras maxilofaciales etiquetadas)

> **Estado (2026-11-19):** descargado de la fuente oficial y usado en dos cosas:
> la orientación y las alturas de entrenamiento se miden sobre sus etiquetas, y su
> checkpoint oficial se probó sobre el caso B, con **resultado negativo**
> ([`toothfairy2-caso-b.md`](toothfairy2-caso-b.md)). Nada de él está versionado.

## 1. Identidad

- **Nombre:** ToothFairy2, de AImageLab, Universidad de Módena y Reggio Emilia.
- **Fuente:** <https://ditto.ing.unimore.it/toothfairy2/>. Pide cuenta y aceptar la
  licencia.
- **Contenido:** 480 CBCT en `.mha`, en HU y a 0,3 mm isótropos. El set A tiene 417
  volúmenes con prefijo P y el set B 63 con prefijo F. Formato nnU-Net (`Dataset112`).
- **Etiquetas:** 42 clases según la web. Los códigos de diente coinciden con FDI (11–48).
  Hay además mandíbula (1), maxilar (2), canales alveolares (3 y 4), senos (5 y 6),
  faringe (7), puente (8), corona (9) e implante (10). El `dataset.json` del checkpoint
  tiene 49 entradas contiguas porque incluye seis «NA».

## 2. Licencia

**No coinciden las fuentes, y se aplica la más restrictiva:**

| Fuente | Dice |
|---|---|
| Pie de la web oficial | CC BY-NC-SA 4.0 |
| `dataset.json` del zip | CC-BY-SA 4.0 |
| `dataset_json` dentro del checkpoint | CC-BY-SA 4.0 |

Se aplica **CC BY-NC-SA 4.0**: no comercial, compartir igual y cita obligatoria.
Cualquier peso entrenado con estos datos hereda esas condiciones. Afecta también a
`data/processed/cbct-diente/modelo.pt`, que se entrenó sobre un espejo público de
150 de estos volúmenes (ver [`segmentacion-diente-cbct.md`](segmentacion-diente-cbct.md)).

**Citas obligatorias** (pestaña BibTeX de la web):

- F. Bolelli et al., *Multi-structure segmentation in CBCT volumes: The ToothFairy2
  challenge*, Medical Image Analysis, 104095, 2026. doi:10.1016/j.media.2026.104095
- F. Bolelli et al., *Segmenting Maxillofacial Structures in CBCT Volumes*, CVPR 2025.
  doi:10.1109/CVPR52734.2025.00494
- L. Lumetti et al., *Enhancing Patch-Based Learning for the Segmentation of the
  Mandibular Canal*, IEEE Access, 2024. doi:10.1109/ACCESS.2024.3408629

## 3. Copia local (fuera del repositorio)

`~/anfaia/toothfairy2-oficial/`, con su `SHA256SUMS`:

| Fichero | Tamaño | SHA-256 |
|---|---:|---|
| `ToothFairy2_Dataset.zip` | 26,1 GB (109,5 GB descomprimido) | `a720914e715f9761a8063d8650eb9a482e5bec0b61e5470766ab973d38c00434` |
| `tf2_checkpoints.zip` | 4,6 GB | `89c037e9fc94e04471175cfa59b58a6748d534072b495ddf5e354bb08ae3f3a1` |

El dataset **no se extrae**: no cabe en el disco. Los scripts leen cada `.mha` de
etiquetas directamente del zip, en memoria.

## 4. Lo que se ha medido sobre él

- **Orientación.** Las cabeceras declaran identidad y `RAI`, y sus propias etiquetas lo
  contradicen. Medido sobre siete casos fijos, el array `(z, y, x)` crece hacia caudal,
  posterior y derecha del paciente.
- **Alturas** ([`scripts/mide_toothfairy2_entrenamiento.py`](../../scripts/mide_toothfairy2_entrenamiento.py)):
  - 356 de los 480 volúmenes tienen algún diente superior.
  - Por encima del plano oclusal superior llegan a una mediana de **10,2 mm** (p95 26,7).
  - **Casi no contienen raíces superiores.** Un modelo entrenado con ellos ve poco maxilar.
- **Etiquetas dudosas.** En P_061, P_067 y P_101 hay un diente superior etiquetado muy por
  debajo del resto. En P_061 parece un intercambio entre 26 y 36.

## 5. Checkpoints

El zip trae ocho. Se usó solo
`nnUNetTrainer__onlyMirror01__nnUNetResEncUNetLPlans__3d_fullres`, el entrenado sin espejo
lateral. Pese al nombre, es una **PlainConvUNet de 30,8 M** parada en la época 266 de 1000,
con un solo fold.
