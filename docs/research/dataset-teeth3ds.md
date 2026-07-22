# Dataset — Teeth3DS+ (escaneos intraorales 3D)

> **Estado (2026-07-22):** dataset **evaluado y recomendado**; **licencia resuelta
> a CC-BY 4.0** según el paper (ver §4). Descarga vía **Google Drive** (§6).
> **Dataset completo descargado y verificado** en `data/raw/teeth3ds/` (gitignored):
> **300 pacientes / 600 escaneos** (maxilar+mandíbula), **~70 M vértices
> etiquetados**, 7,3 GiB — todo lo que publican los zips oficiales (base + `_b2`).
> Cada `.obj` tiene su `.json` de labels: **0 huérfanos**. **Issue 1 cerrada.**
>
> *(Hasta el 2026-07-21 se trabajaba sobre un subconjunto de 12 pacientes / 24
> escaneos; el script sigue pudiendo bajar solo ese con `--subset`.)*

Contraparte de código: **PoC MVP 1 hecho** —
[`notebooks/01-vtk-3dgs-poc.ipynb`](../../notebooks/01-vtk-3dgs-poc.ipynb)
(resultados y alcance en [`notebooks/README.md`](../../notebooks/README.md)):
carga la malla, valida el ancla FDI, corre `vtkGaussianSplatter` y serializa al
contrato. Diseño: [`docs/architecture/multi-agent-pipeline.md`](../architecture/multi-agent-pipeline.md).

---

## 1. Identidad

- **Nombre:** Teeth3DS+ (extensión de Teeth3DS).
- **Origen:** retos MICCAI **3DTeethSeg'22** y **3DTeethLand'24**; organizado por
  Udini (Francia), Inria Grenoble (equipo Morpheo) y el Digital Research Center of
  Sfax (Túnez).
- **Contenido:** 1.800 escaneos intraorales (IOS) de **900 pacientes** (Francia y
  Bélgica), maxilar y mandíbula por separado; **23.999 dientes anotados**.
- **Papers:** arXiv:2210.06094 (Teeth3DS+); Ben Hamadou et al., *Teeth3DS: a
  benchmark for teeth segmentation and labeling from intra-oral 3D scans* (2022).

## 2. Formato y estructura (verificado sobre los 600 escaneos)

- **Mallas:** `.obj` (una por arcada: `<ID>_upper.obj` / `<ID>_lower.obj`).
- **Etiquetas:** `.json` **por vértice** (no por diente). Claves:
  `id_patient`, `jaw`, `labels`, `instances`.
  - `labels`: código **FDI** por vértice (p. ej. `31–37, 41–47`), **`0` = encía**.
    Los FDI validan contra el `FDICode` del contrato (`[1-4][1-8]|[5-8][1-5]`).
  - `instances`: id de instancia por vértice (`0` = encía; 1 por diente).
  - Densidad alta: **mediana 116k** vértices etiquetados por malla.
- **Layout en disco — DOS árboles paralelos** (no `.obj`+`.json` juntos):

  ```
  data/raw/teeth3ds/
    3D_scans_per_patient_obj_files/<ID>/<ID>_{upper,lower}.obj
    ground-truth_labels_instances/<ID>/<ID>_{upper,lower}.json
  ```
  El loader del PoC debe cruzar por `<ID>_<jaw>` entre ambos árboles.
- **Splits:** ficheros `.txt` de train/test (listas de IDs, p. ej. `EJWZZZRF_lower`)
  — en el OSF, no necesarios para el PoC.

### 2.1 Qué contiene de verdad (medido sobre los 600 escaneos)

Caracterización recorriendo **todas** las etiquetas (~6 s; los `.json` pesan ~0,7 MB
frente a los ~12 MB de cada malla). Reproducible en
[`notebooks/01-vtk-3dgs-poc.ipynb`](../../notebooks/01-vtk-3dgs-poc.ipynb) §1b, con
las figuras embebidas.

| Magnitud | Valor medido |
|---|---|
| Escaneos / pacientes | 600 / 300 (300 upper + 300 lower) · **0 huérfanos** |
| Vértices por escaneo | mediana **116.475** · rango 13.034 – 219.518 · total **69,9 M** |
| Dientes por arcada | mediana **14** · rango 9 – 16 · **solo 1 de 600** llega a 16 |
| Encía (`label` 0) | **43%** de los vértices de media — clase mayoritaria |

**Desbalance por código FDI** (% de escaneos de su arcada en los que aparece):

| Código | Presencia | Lectura |
|---|---|---|
| `18` | **0%** | **no aparece en ningún escaneo** — clase inaprendible |
| `28`, `38`, `48` | ~1% | cordales, prácticamente ausentes |
| `17`, `27`, `37`, `47` | 63–70% | segundos molares: faltan en ~1 de cada 3 |
| resto | 91–100% | presentes casi siempre |

> **Consecuencia de diseño.** Esto no es color local: (a) obliga a **loss ponderada
> por clase** en el `segmentation-agent` y descarta la *accuracy* global como
> métrica —la encía sola ya da el 43%—; y (b) **acota lo que el sistema puede
> prometer**: un modelo entrenado aquí **no segmenta cordales**, porque casi nunca
> los ha visto. Declararlo es parte del alcance, no una nota al pie.

## 3. Dónde vive cada cosa (realidad de acceso)

| Recurso | Alojamiento | ¿Auto-descargable? |
|---|---|---|
| `license.txt` + splits train/test (`.txt`) | OSF `xctdy` (`osf.io/xctdy`) | Sí (KB; `curl` a veces da 403 por rate-limit → usar navegador) |
| **Mallas `.obj` + labels `.json`** (el dato real) | **Figshare** (release Teeth3DS+) y/o **Google Drive** (zips del reto) | **No limpiamente**: zips grandes / requiere navegador o `gdown` |

> ⚠️ **El OSF `xctdy` NO contiene las mallas** — solo la licencia y los splits.
> Las `.obj` están en Figshare / Google Drive
> (`3D_scans_per_patient_obj_files.zip`, `..._b2.zip`).

## 4. Licencia — RESUELTA: CC-BY 4.0 (con matiz documentado)

Tres fuentes parecían contradecirse; al rastrearlas se resuelve así:

| Fuente | Licencia | Interpretación |
|---|---|---|
| **Paper Teeth3DS+** (Ben-Hamadou et al., 2022, **§5**) | **CC-BY 4.0** | **Vinculante** — es la licencia que declaran los autores para el dataset |
| Web del proyecto (footer) | CC BY-SA 4.0 | Aplica **solo al sitio web**, no al dataset — descartada |
| `license.txt` del bundle OSF | CC BY-NC-ND 4.0 | **Inconsistencia**; el OSF solo aloja splits, no las mallas |

**Veredicto:** se usa **CC-BY 4.0** (paper §5). Permite generar y **publicar modelos
3DGS derivados** con **atribución**. El *"via Figshare"* del paper **no tiene enlace
publicado**: la descarga real es por Google Drive (§6).

**Para el proyecto:**
- Citar **CC-BY 4.0 (Ben-Hamadou et al., 2022)** y atribuir en cualquier derivado.
- Dejar constancia de la inconsistencia del `license.txt` del OSF (BY-NC-ND).
- Si se publica formalmente, confirmar por email a los autores (blinda el matiz).

## 5. Por qué encaja con nuestra arquitectura (si la licencia lo permite)

- **Mallas → VTK nativo**: se cargan directas y se muestrean a nube de puntos para
  `vtkGaussianSplatter` (PoC MVP 1).
- **Labels por diente → `region_id` FDI**: alimentan el `segmentation-agent` y el
  ancla semántica de la fusión (ver [pipeline §3-4](../architecture/multi-agent-pipeline.md)).
- **Truco «solo fotos»**: al tener malla 3D con ground truth, se pueden **renderizar
  fotos multi-vista sintéticas** desde cada malla → alimentar el pipeline
  foto→3DGS **con verdad-terreno para comparar**, resolviendo la inexistencia de un
  dataset público de fotos dentales multi-vista (DentalSplat/Dental3R son cerrados).

### 5.1 Qué HABILITA el dataset (y qué no)

El dataset es de **mallas**, no de fotos. Eso determina qué PoC son posibles:

| Objetivo | ¿Posible con este dataset? | Por qué |
|---|---|---|
| **Gaussian splatting clásico** (`vtkGaussianSplatter`) | ✅ Hecho (PoC MVP 1) | Partimos de la geometría (puntos de la malla) |
| **3DGS moderno** (fotos-con-pose → gaussianas entrenadas) | ✅ Sí, **vía vistas sintéticas** | Renderizamos la malla desde N ángulos → **fotos + poses conocidas** + nube inicial; se salta COLMAP. Sirve de validación con verdad-terreno |
| **Foto→3D REAL** (fotos en crudo de cámara → COLMAP → 3DGS) | ❌ No | No hay **fotos reales** dentales multi-vista; el dataset no las contiene y no existe uno público |

> **Matiz honesto:** el 3DGS moderno vía vistas sintéticas es **circular** (renderizamos
> desde la malla y reconstruimos la misma malla) — vale como **validación del motor
> 3DGS** y para producir un `.splat` para el visor web (Issue 17), **no** como
> solución del caso clínico «solo con fotos del móvil», que requeriría una captura
> real fuera de este dataset.

**Escala alcanzada (2026-07-22).** Con el dataset completo, esa validación del motor
dejó de ser una anécdota: [`03`](../../notebooks/03-synthetic-views-for-3dgs.ipynb)
genera **2.880 vistas** (20 casos × 144) con error de reproyección de pose
**0,0000 px en todas**, y [`04`](../../notebooks/04-train-3dgs-gsplat.ipynb) entrena
8 casos evaluando en **vistas retenidas** (1 de cada 8, nunca vistas): **21,04 ±
0,19 dB** de PSNR, con una brecha train−retenidas de 0,65 dB. Que la desviación
entre anatomías sea de 0,19 dB es lo que convierte esto en insumo del **ADR 002**.
Generar 2.880 imágenes sintéticas en vez de 24 **no cambia** el matiz circular de
arriba: mejora la evidencia sobre el motor, no sobre el pipeline foto→3D clínico.

## 6. Pasos de descarga (Google Drive — ruta real)

> **Vía recomendada (reproducible):** `./scripts/fetch_teeth3ds.sh` — baja los zips
> del Drive oficial y los extrae en `data/raw/teeth3ds/`, verificando el emparejado
> `.obj` ↔ `.json`. Idempotente. Requiere `gdown` (`uv pip install gdown`).
>
> | Invocación | Qué deja en disco |
> |---|---|
> | `./scripts/fetch_teeth3ds.sh` | **dataset completo**: 300 pacientes / 600 escaneos / ~7,3 GiB |
> | `./scripts/fetch_teeth3ds.sh --subset` | solo los 12 pacientes del PoC original (~300 MiB) |
> | `… --force` | re-extrae aunque ya haya casos en destino |
>
> Los pasos manuales de abajo son el *fallback* si el script falla (p. ej. cuota de
> Drive).

Carpeta oficial (usada por la comunidad, p. ej. ToothGroupNetwork):
`https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby`

Ficheros (zips) — **los cuatro juntos son el dataset**, 300 pacientes / 600 escaneos:
- Mallas: `3D_scans_per_patient_obj_files.zip` + `3D_scans_per_patient_obj_files_b2.zip`
- Labels: `ground-truth_labels_instances.zip` + `ground-truth_labels_instances_b2.zip`

Pasos:
1. Descargar los **cuatro** zips (el `_b2` es la continuación, no un duplicado; con
   el par base solo se tiene media biblioteca de pacientes).
2. Descomprimir los dos árboles en `data/raw/teeth3ds/` (carpeta **gitignored**),
   respetando `3D_scans_per_patient_obj_files/` y `ground-truth_labels_instances/`.
3. Verificar: cada caso = `<ID>_<upper|lower>.obj` **+** su `.json` de labels. Un
   `.obj` sin `.json` se descarta (es la regla de ingesta que aplica el código).

> Nota práctica: los zips agrupan **todos** los pacientes (varios GB); no se puede
> bajar «solo 10 casos» — se baja el zip y se extrae lo que interese (eso hace
> `--subset`). Herramienta: `gdown` (`pip install gdown`; `gdown --folder <URL>` o
> por ID de fichero).

## 7. Alternativas si la licencia bloquea

- **MMDental** (PhysioNet): CBCT + informes; requiere credencial + DUA.
- **CTooth** (GitHub): CBCT anotado; acceso por petición.
- (CBCT es volumétrico → también VTK-nativo, pero cambia la modalidad de partida.)

## 8. Referencias

- Teeth3DS+: [web](https://crns-smartvision.github.io/teeth3ds/) ·
  [arXiv:2210.06094](https://arxiv.org/abs/2210.06094) · [OSF `xctdy`](https://osf.io/xctdy/)
- Reto: [3DTeethSeg'22 (Grand Challenge)](https://3dteethseg.grand-challenge.org/) ·
  [repo](https://github.com/abenhamadou/3DTeethSeg22_challenge)
- Índice: [Awesome-Medical-Dataset / Teeth3DS](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/Teeth3DS.md)
