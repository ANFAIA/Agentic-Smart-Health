# Dataset — Teeth3DS+ (escaneos intraorales 3D)

> **Estado (2026-07-15):** dataset **evaluado y recomendado**; **licencia resuelta
> a CC-BY 4.0** según el paper (ver §4). Descarga vía **Google Drive** (§6).
> **Subconjunto descargado y verificado** en `data/raw/teeth3ds/` (gitignored):
> **12 pacientes / 24 escaneos** (maxilar+mandíbula), cada `.obj` con su `.json` de
> labels. **Issue 1 cerrada.**

Contraparte de código: PoC MVP 1 (`notebooks/01-vtk-3dgs-poc.ipynb`, pendiente).
Diseño: [`docs/architecture/multi-agent-pipeline.md`](../architecture/multi-agent-pipeline.md).

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

## 2. Formato y estructura (verificado en el subconjunto)

- **Mallas:** `.obj` (una por arcada: `<ID>_upper.obj` / `<ID>_lower.obj`).
- **Etiquetas:** `.json` **por vértice** (no por diente). Claves:
  `id_patient`, `jaw`, `labels`, `instances`.
  - `labels`: código **FDI** por vértice (p. ej. `31–37, 41–47`), **`0` = encía**.
    Los FDI validan contra el `FDICode` del contrato (`[1-4][1-8]|[5-8][1-5]`).
  - `instances`: id de instancia por vértice (`0` = encía; 1 por diente).
  - Densidad alta: ~110k vértices etiquetados por malla.
- **Layout en disco — DOS árboles paralelos** (no `.obj`+`.json` juntos):

  ```
  data/raw/teeth3ds/
    3D_scans_per_patient_obj_files/<ID>/<ID>_{upper,lower}.obj
    ground-truth_labels_instances/<ID>/<ID>_{upper,lower}.json
  ```
  El loader del PoC debe cruzar por `<ID>_<jaw>` entre ambos árboles.
- **Splits:** ficheros `.txt` de train/test (listas de IDs, p. ej. `EJWZZZRF_lower`)
  — en el OSF, no necesarios para el PoC.

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

## 6. Pasos de descarga (Google Drive — ruta real)

Carpeta oficial (usada por la comunidad, p. ej. ToothGroupNetwork):
`https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby`

Ficheros (zips):
- Mallas: `3D_scans_per_patient_obj_files.zip` + `3D_scans_per_patient_obj_files_b2.zip`
- Labels: `ground-truth_labels_instances.zip` + `ground-truth_labels_instances_b2.zip`

Pasos:
1. Descargar **un** zip de mallas + su zip de labels (el `_b2` es la 2ª mitad; con
   el primero basta para el PoC).
2. Descomprimir y quedarse con un **subconjunto pequeño** (~10 casos maxilar+mandíbula)
   en `data/raw/teeth3ds/` (carpeta **gitignored**).
3. Verificar: cada caso = `<ID>_<upper|lower>.obj` + su `.json` de labels.

> Nota práctica: los zips agrupan **todos** los pacientes (varios GB); no se puede
> bajar «solo 10 casos» — se baja el zip y se extrae el subconjunto. Herramienta:
> `gdown` (`pip install gdown`; `gdown --folder <URL>` o por ID de fichero).

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
