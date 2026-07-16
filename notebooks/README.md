# Notebooks — PoC / MVPs escalonados

Experimentación y pruebas de concepto del pipeline. Cada notebook valida **un
eslabón** de la arquitectura ([`docs/architecture/multi-agent-pipeline.md`](../docs/architecture/multi-agent-pipeline.md))
de forma manual, antes de convertirlo en agente.

## Qué se ha probado (resumen)

Estos notebooks son **spikes de validación técnica** — no son el sistema final ni
resultados clínicos, sino pruebas para de-arriesgar decisiones de arquitectura.
Todos corren sobre el **subconjunto de Teeth3DS+ que descargamos**
(`data/raw/teeth3ds/`, **12 pacientes / 24 escaneos**, gitignored — ver
[nota del dataset](../docs/research/dataset-teeth3ds.md)); los renders usan un caso
representativo (`01A6GW4A_lower`).

| Notebook | Qué se validó exactamente | GPU |
|---|---|---|
| **01** | VTK carga la malla real y hace *splatting clásico* (baseline) → serializa al contrato | No |
| **02** | Visor 3D interactivo de escritorio (VTK): malla, campo, nube de puntos | No |
| **03** | Generar **vistas sintéticas + poses de cámara exactas** (input del 3DGS), sin COLMAP | No |
| **04** | **3DGS moderno entrenado** (`gsplat`/GPU) sobre esas vistas → contrato | **Sí** |

**No se ha probado (aún):** foto→3D con **fotos reales**, **fusión multimodal**
(CBCT+STL), ni la integración como **agentes / LLM**. El 3DGS moderno se validó por
vía sintética (matiz «circular» documentado en
[`docs/research/dataset-teeth3ds.md` §5.1](../docs/research/dataset-teeth3ds.md)).

> **Cómo ejecutarlos:** desde la raíz del repo, `uv run jupyter notebook`. El
> prefijo `uv run` hace que el kernel use el `.venv` del workspace (donde están
> `vtk`, `numpy`, etc.); lanzarlo con `jupyter notebook` a secas usaría el Python
> del sistema y fallaría con `ModuleNotFoundError`.

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

`caso → malla .obj + labels FDI → render coloreado → nube de puntos →
vtkGaussianSplatter → campo de densidad 3D → render isosuperficie →
artefacto .ply + hash → TwinSnapshot del contrato`

### Qué se logró (validado)

- **La cadena mínima corre de extremo a extremo** con datos reales: VTK carga las
  mallas de Teeth3DS+ (~110k vértices), las *splattea* a un campo de densidad
  volumétrico 3D y lo renderiza.
- **Las etiquetas FDI casan con la geometría** (render coloreado por diente): el
  ancla semántica `region_id` está bien alineada → ground truth listo para el
  futuro `segmentation-agent` y para la fusión semántica.
- **VTK es viable** como librería de entrada y renderiza *headless* (offscreen →
  PNG), sirve en servidor/CI sin pantalla.
- **El PoC no queda huérfano de la arquitectura**: la salida se serializa al
  contrato [`core-schemas`](../packages/core-schemas/) (`TwinSnapshot` +
  `gaussian_field_ref` por hash). El patrón «el campo masivo se referencia, no se
  embebe» funciona en la práctica.

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

### Cómo correrlo

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01-vtk-3dgs-poc.ipynb
# o, interactivo:
uv run jupyter notebook
```

Requiere el subconjunto Teeth3DS+ en `data/raw/teeth3ds/` (gitignored) —
ver [`scripts/fetch_teeth3ds.sh`](../scripts/fetch_teeth3ds.sh) y la
[nota del dataset](../docs/research/dataset-teeth3ds.md). Genera un artefacto
`.ply` en `data/processed/teeth3ds/` (gitignored). Los dos renders quedan
**embebidos** en el `.ipynb` (visibles en GitHub sin ejecutar).

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

Genera en `data/processed/teeth3ds/<caso>_3dgs/` (gitignored):
- `images/r_XXX.png` — N vistas RGB (órbita azimut × elevación).
- `transforms.json` — intrínsecos + c2w por vista (formato instant-ngp/Nerfstudio),
  **auto-verificado por reproyección** (~0 px).
- `init.ply` — nube de puntos inicial para sembrar la optimización.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03-synthetic-views-for-3dgs.ipynb
```

> **Mitad 2:** ver `04` abajo. Alcance y matiz «circular» (validación del motor, no
> foto→3D real) en [`docs/research/dataset-teeth3ds.md` §5.1](../docs/research/dataset-teeth3ds.md).

---

## `04-train-3dgs-gsplat.ipynb` — 3DGS moderno · Mitad 2 (entrenamiento)

**El 3DGS moderno de verdad.** Toma el paquete del `03` (vistas + poses + `init.ply`)
y **entrena** un campo de gaussianas anisótropas optimizándolas contra las vistas
(pérdida fotométrica L1) con `gsplat`. Exporta el `.ply` entrenado y lo serializa al
contrato (`TwinSnapshot`).

✅ **Validado end-to-end en RTX 5070 (sm_120)**: `torch 2.11.0+cu128` + `gsplat 1.5.3`.
L1 baja de ~0.16 a ~0.03; la reconstrucción reproduce la arcada dental.

> ⚠️ **Requiere GPU.** `torch`/`gsplat` se instalan **ad-hoc en el venv** (no en
> `pyproject.toml`: son específicos de GPU y romperían la lock compartida / CI):
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell/sm_120
> uv pip install gsplat   # compila kernels CUDA en el primer uso (~45 s)
> ```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04-train-3dgs-gsplat.ipynb
```

Incluye un **visor interactivo** (§7, ventana nativa VTK como el `02`) para rotar el
campo de gaussianas entrenado — requiere pantalla, se lanza con `uv run jupyter notebook`.

**Mejoras naturales:** densificación/poda (`gsplat` `DefaultStrategy`), color por
armónicos esféricos, métricas PSNR/SSIM, export `.splat` para el visor web (Issue 3).
