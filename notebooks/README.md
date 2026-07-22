# Notebooks — PoC / MVPs escalonados

Experimentación y pruebas de concepto del pipeline. Cada notebook valida **un
eslabón** de la arquitectura ([`docs/architecture/multi-agent-pipeline.md`](../docs/architecture/multi-agent-pipeline.md))
de forma manual, antes de convertirlo en agente.

## Qué se ha probado (resumen)

Estos notebooks son **spikes de validación técnica** — no son el sistema final ni
resultados clínicos, sino pruebas para de-arriesgar decisiones de arquitectura.
Todos corren sobre **Teeth3DS+ completo** (`data/raw/teeth3ds/`, **300 pacientes /
600 escaneos / ~70 M vértices etiquetados**, 7,3 GiB, gitignored — ver
[nota del dataset](../docs/research/dataset-teeth3ds.md)). Los renders detallados
usan un caso de referencia fijo (`01A6GW4A_lower`) para ser reproducibles, pero
**cada notebook ejercita además el dataset entero**: barridos sobre muestras
aleatorias con semilla fija, en vez de una demo sobre un único escaneo.

| Notebook | Qué se validó exactamente | Escala | GPU |
|---|---|---|---|
| **01** | VTK carga la malla real y hace *splatting clásico* (baseline) → serializa al contrato · caracteriza los 600 escaneos · barrido de robustez | 600 escaneos + 24 casos en la cadena completa | No |
| **02** | Visor 3D interactivo de escritorio (VTK): malla, campo, nube de puntos, sobre cualquier caso | selector de los 600 | No |
| **03** | Generar **vistas sintéticas + poses de cámara exactas** (input del 3DGS), sin COLMAP | **2 880 vistas** · 20 casos | No |
| **04** | **3DGS moderno entrenado** (`gsplat`/GPU) evaluado en **vistas retenidas** → contrato | 8 casos entrenados | **Sí** |
| **exercise** | **Segmentación FDI por punto** (Point Transformer/PyG) sobre Teeth3DS+ completo — prototipo del `segmentation-agent` | 600 mallas | **Sí** |

**No se ha probado (aún):** foto→3D con **fotos reales**, **fusión multimodal**
(CBCT+STL), ni la integración como **agentes / LLM**. El 3DGS moderno se validó por
vía sintética (matiz «circular» documentado en
[`docs/research/dataset-teeth3ds.md` §5.1](../docs/research/dataset-teeth3ds.md)).

> **Cómo ejecutarlos:** para **01–03** (sin GPU), desde la raíz del repo,
> `uv run jupyter notebook`. El prefijo `uv run` hace que el kernel use el `.venv`
> del workspace (donde están `vtk`, `numpy`, `pydantic`); lanzarlo con `jupyter
> notebook` a secas usaría el Python del sistema y fallaría con `ModuleNotFoundError`.
>
> **El `04` (GPU) usa un entorno aparte.** `torch`/`gsplat` son específicos de la
> máquina (cu128/Blackwell) y **no** viven en el `.venv` del workspace —un `uv sync`
> los podaría—: tienen su propio venv y su kernel de Jupyter **"Dental GPU (3DGS)"**
> (montaje en §04). Al abrir el `04`, selecciona ese kernel.

> **Sobre los datos (`data/` está gitignored):** los notebooks **leen** de
> `data/raw/teeth3ds/` (el dataset, que se baja con
> [`scripts/fetch_teeth3ds.sh`](../scripts/fetch_teeth3ds.sh)) y **escriben** sus
> artefactos (`.ply`, vistas, `transforms.json`, campo entrenado) en
> `data/processed/`. Nada de eso se versiona: es **regenerable** re-ejecutando los
> notebooks, **pesa** (varios GB) y **no debe redistribuirse** por el repo (licencia
> y soberanía del dato). El repo guarda la *receta* (código + script), no el dato.

---

## `01-vtk-3dgs-poc.ipynb` — Malla dental → Gaussian Splatting (VTK)

**Issue 2 · PoC MVP 1.** Valida el eslabón mínimo del pipeline
*«malla 3D → representación gaussiana volumétrica → contrato de datos»* con la
librería de entrada (VTK) y el dataset real (Teeth3DS+).

### Qué hace (flujo)

`inventario de los 600 escaneos → caracterización (dientes/arcada, encía, FDI raros) →
caso → malla .obj + labels FDI → render coloreado → nube de puntos →
vtkGaussianSplatter → campo de densidad 3D → render isosuperficie →
artefacto .ply + hash → TwinSnapshot del contrato →
barrido de la cadena sobre 24 casos aleatorios → galería de 6 pacientes`

### Qué se logró (validado)

- **La cadena mínima corre de extremo a extremo** con datos reales: VTK carga las
  mallas de Teeth3DS+ (mediana **116k vértices**), las *splattea* a un campo de
  densidad volumétrico 3D y lo renderiza.
- **Y aguanta el dataset entero**: el barrido (§6) repite carga + labels +
  *splatting* sobre **24 escaneos aleatorios** (semilla fija) sin una sola
  excepción, midiendo el coste — ~0,05 s de carga + ~0,15 s de splat por escaneo,
  que es el presupuesto del futuro `mesh-agent`.
- **Las etiquetas FDI casan con la geometría** (render coloreado por diente): el
  ancla semántica `region_id` está bien alineada → ground truth listo para el
  futuro `segmentation-agent` y para la fusión semántica. La galería (§7) lo
  enseña sobre **seis anatomías distintas**, no solo sobre el caso de la foto.
- **El desbalance del dataset está cuantificado** (§1b, los 600 escaneos en ~6 s):
  la encía es el **43%** de los vértices; la mediana real es de **14 dientes por
  arcada**, no 16; el FDI `18` **no aparece en ningún escaneo**, los cordales
  `28`/`38`/`48` en el **1%**, y los segundos molares (`17`/`27`/`37`/`47`) faltan
  en un tercio. Justifica la *loss* ponderada del `segmentation-agent` y **acota lo
  que puede prometer**: no segmentará cordales, porque casi no los ha visto.
- **VTK es viable** como librería de entrada y renderiza *headless* (offscreen →
  PNG), sirve en servidor/CI sin pantalla.
- **El PoC no queda huérfano de la arquitectura**: la salida se serializa al
  contrato [`core-schemas`](../packages/core-schemas/) (`TwinSnapshot` +
  `gaussian_field_ref` por hash). El patrón «el campo masivo se referencia, no se
  embebe» funciona en la práctica.
- **El emparejamiento es una regla de ingesta**, no un detalle: `list_cases()`
  descarta el `.obj` sin su `.json` — sin ancla semántica no hay ingesta.

### Qué NO es (alcance honesto)

- **No es 3DGS entrenado.** `vtkGaussianSplatter` es *splatting* de densidad
  clásico: gaussianas **isótropas**, sin optimización diferenciable ni armónicos
  esféricos. Es un **baseline / banco de pruebas**, no el motor final.
- **No hace foto→3D todavía**: parte de una malla existente (aunque abre la puerta
  a renderizar vistas sintéticas con verdad-terreno para ese pipeline).
- **No hay fusión multimodal** (CBCT+STL) — es trabajo posterior.

### Qué decide (insumo para los ADR)

| Hallazgo | Alimenta |
|---|---|
| Qué da VTK y qué le falta (isótropas, coste O(n³), sensibilidad a `Radius`) | **D1 · ADR 002** (motor de render) |
| Los `.obj` traen color por vértice sin usar | canal `color_superficie` de la **fusión** |
| Encía aislable (`label`/`instance` 0) | futuro PoC de inflamación/pH |
| Desbalance FDI medido (encía 43%, cordales ~ausentes, mediana 14 dientes) | *loss* ponderada y **alcance declarado** del `segmentation-agent` |
| Coste por escaneo medido y extrapolado a los 600 | dimensionado de la ingesta por lotes (`mesh-agent`) |

### Cómo correrlo

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01-vtk-3dgs-poc.ipynb
# o, interactivo:
uv run jupyter notebook
```

Tarda ~15 s (lo caro es el barrido de 24 casos, no el caso de referencia). Requiere
Teeth3DS+ en `data/raw/teeth3ds/` (gitignored) — ver
[`scripts/fetch_teeth3ds.sh`](../scripts/fetch_teeth3ds.sh) y la
[nota del dataset](../docs/research/dataset-teeth3ds.md). Genera un artefacto
`.ply` en `data/processed/teeth3ds/` (gitignored). Renders, figuras del dataset y
galería quedan **embebidos** en el `.ipynb` (visibles en GitHub sin ejecutar).

**Siguiente:** Issue 3 (visor web three.js / GaussianSplats3D) y, tras el spike de
motores, redactar el **ADR 002** de render.

---

## `02-vtk-interactive-viewer.ipynb` — Visor 3D interactivo (ventana nativa VTK)

Complemento del `01`. Aquel renderiza *offscreen* a PNG (fotos fijas); este abre una
**ventana nativa del sistema** para **rotar / zoom / pan** el modelo con el ratón,
usando `vtkRenderWindowInteractor` con estilo *trackball*. **Cero dependencias
nuevas.**

> ⚠️ **Requiere entorno gráfico (pantalla).** No corre *headless* (servidor/CI) ni
> renderiza dentro del notebook: abre una **ventana aparte**, y la celda que la
> lanza **se bloquea** hasta que la cierras (tecla `q`). Por eso va **sin salidas
> embebidas** — su resultado es la ventana, no una imagen.

Muestra tres cosas rotables: la malla coloreada por FDI, el campo
`vtkGaussianSplatter` y la nube de puntos (vértices). Controles: arrastrar
(rotar), rueda (zoom), Shift+arrastrar (pan), `q` (cerrar).

**Selector de caso.** La §1 inventaría los **600 escaneos** y expone
`CASO = ("<paciente>", "<arcada>")` — cámbialo por cualquiera, o usa
`caso_aleatorio()` / `caso_aleatorio(seed=3)` y re-ejecuta. Es la herramienta para
mirar *de verdad* un escaneo cuando un número del `01` sorprende (por qué esa
arcada tiene 9 dientes, o esa malla 25k vértices en vez de 116k).

### Cómo correrlo

```bash
uv run jupyter notebook   # abrir 02, ejecutar celdas en orden; NO usar nbconvert --execute (bloquea)
```

Mismo dataset de entrada que el `01`.

---

## `03-synthetic-views-for-3dgs.ipynb` — 3DGS moderno · Mitad 1 (vistas + poses)

**Prerequisito del 3DGS moderno.** El 3DGS entrenado necesita **fotos multi-vista
con pose de cámara** + una nube inicial. No hay fotos dentales reales, así que las
**sintetizamos desde la malla** (poses conocidas → se salta COLMAP). Sin GPU.

**Ahora genera un lote, no una demo:** `N_CASOS × AZIMUTS × ELEVACIONES`. Por
defecto **20 casos × 144 vistas = 2 880 imágenes** en ~100 s y ~128 MiB
(configurable en la celda de configuración). Genera en
`data/processed/teeth3ds/<caso>_3dgs/` (gitignored), **uno por caso**:
- `images/r_XXXX.png` — 144 vistas RGB (24 azimuts × 6 elevaciones).
- `transforms.json` — intrínsecos + c2w por vista (formato instant-ngp/Nerfstudio),
  **auto-verificado por reproyección en todas las vistas**.
- `init.ply` — nube de puntos inicial para sembrar la optimización.

**Resultado:** peor error de reproyección **0,0000 px sobre las 2 880 vistas** — las
poses del dataset sintético son exactas, así que si el 04 reconstruye mal, la culpa
no es de las cámaras. La §5 verifica además la integridad de cada paquete
(nº de PNG == nº de poses, `init.ply` legible).

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03-synthetic-views-for-3dgs.ipynb
```

> **Por qué más vistas:** con 24 el 3DGS puede parecer que funciona porque hay poco
> que contradecirle; con ~144 por caso la reconstrucción tiene que ser consistente
> desde muchos más ángulos, y sobran vistas para **retener algunas** y evaluar en
> ellas (lo que hace el `04`).

> **Mitad 2:** ver `04` abajo. Generar 2 880 imágenes sintéticas en vez de 24 mejora
> la validación del **motor** y **no cambia** el matiz «circular» (no es foto→3D
> real) documentado en
> [`docs/research/dataset-teeth3ds.md` §5.1](../docs/research/dataset-teeth3ds.md).

---

## `04-train-3dgs-gsplat.ipynb` — 3DGS moderno · Mitad 2 (entrenamiento)

**El 3DGS moderno de verdad.** Toma los paquetes del `03` (vistas + poses +
`init.ply`) y **entrena** un campo de gaussianas anisótropas optimizándolas contra
las vistas (pérdida fotométrica L1) con `gsplat`. Exporta el `.ply` entrenado y lo
serializa al contrato (`TwinSnapshot`).

**Se evalúa en vistas retenidas.** 1 de cada 8 vistas se aparta y el modelo **no la
ve nunca** (convenio Nerfstudio/instant-ngp, que solo es viable ahora que hay 144
vistas por caso). El PSNR sobre ellas distingue *reconstruir geometría* de
*memorizar fotos* — la L1 de entrenamiento, sola, no.

✅ **Validado end-to-end en RTX 5070 (sm_120)**: `torch 2.11.0+cu128` + `gsplat 1.5.3`.

| Métrica (8 casos entrenados, 2 000 iters c/u) | Valor |
|---|---|
| PSNR en **vistas retenidas** | **21,04 ± 0,19 dB** (rango 20,72–21,30) |
| Brecha PSNR train − retenidas | **0,65 dB** → sin sobreajuste apreciable |
| Coste | ~5 ms/iteración · <400 MiB de VRAM · ~6 s por caso |

Que la desviación entre anatomías sea de **0,19 dB** es el resultado que vale para
el ADR: el motor se comporta igual en arcadas distintas, no solo en el caso bonito.

> ⚠️ **Requiere GPU, en su propio entorno.** `torch`/`gsplat` son específicos de la
> máquina (cu128/Blackwell) y **no** van en `pyproject.toml` (romperían la lock
> compartida / CI, y un `uv sync` los podaría del `.venv`). Viven en un **venv
> dedicado** con su kernel de Jupyter. Montaje **una sola vez**:
> ```bash
> python3.13 -m venv ~/.venvs/dental-gpu
> ~/.venvs/dental-gpu/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128  # Blackwell/sm_120
> ~/.venvs/dental-gpu/bin/pip install gsplat vtk numpy pydantic jupyter ipykernel  # gsplat compila kernels CUDA (~45 s)
> ~/.venvs/dental-gpu/bin/python -m ipykernel install --user --name dental-gpu --display-name "Dental GPU (3DGS)"
> ```
> El `.venv` del workspace queda intacto para contrato/tests; un `uv sync` ya no te
> tumba torch. `core_schemas` se importa por `sys.path` (no hace falta instalarlo).

```bash
# ejecutar con el entorno dedicado (NO 'uv run', que usaría el .venv del workspace sin torch)
~/.venvs/dental-gpu/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04-train-3dgs-gsplat.ipynb
```

Tarda ~1,5 min (caso de referencia con 3 000 iters + barrido de 8 casos). Incluye un
**visor interactivo** (§8, ventana nativa VTK como el `02`) para rotar el campo de
gaussianas entrenado — requiere pantalla; lánzalo con
`~/.venvs/dental-gpu/bin/jupyter notebook` (kernel **"Dental GPU (3DGS)"**).

**Mejoras naturales:** densificación/poda (`gsplat` `DefaultStrategy`), color por
armónicos esféricos (sin ellos el especular del render no se puede reproducir),
SSIM, y export `.splat` para el visor web (**Issue 17**).

---

## `exercise-point-transformer-teeth3ds.ipynb` — Segmentación de dientes (FDI) por punto

Prototipo del **`segmentation-agent`**: nube de puntos del escaneo intraoral →
etiqueta **FDI por punto**. Point Transformer (PyG) sobre **Teeth3DS+ completo**
(300 pacientes / 600 mallas), con **loss ponderada por clase** y diagnóstico `tooth_acc`.

**Hallazgo (ablación de features, `tooth_acc` test):** `pos-only` no generaliza (**0.08**); basta
**un descriptor local por punto** para arreglarlo — **normales 0.84**, gris CBCT sintético 0.79,
y combinarlos **no suma** (redundantes). La palanca no es «más datos» ni «el CBCT», sino la
geometría **local**. El gris real aporta por **fusión** (registro CBCT↔malla, estilo DDMF), no
pintando gris por vértice.

Detalle completo, ablación, referencia DDMF y reproducción:
[`exercise-point-transformer-teeth3ds.md`](exercise-point-transformer-teeth3ds.md). Requiere el
kernel GPU **"Dental GPU (3DGS)"** (ver §04).
