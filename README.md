# Agentic Smart Health

Sistema multiagente para la integración, análisis y representación de datos clínicos dentales heterogéneos sobre un **Digital Twin** del paciente, basado en Gaussian Splatting con atributos clínicos por punto/zona y soporte de series temporales.

[![tests](https://github.com/ANFAIA/Agentic-Smart-Health/actions/workflows/tests.yml/badge.svg)](https://github.com/ANFAIA/Agentic-Smart-Health/actions/workflows/tests.yml)

> Proyecto open source · Licencia Apache 2.0 · Python ≥ 3.13

---

## Contexto del proyecto

El sector dental maneja datos altamente heterogéneos: escáneres CBCT (DICOM), archivos STL de escaneos intraorales, informes clínicos en PDF e imágenes 2D. Esta información vive fragmentada en silos por proveedor y por clínica, lo que impide un seguimiento longitudinal real del paciente y compromete su soberanía sobre los propios datos de salud.

**Agentic Smart Health** aborda este problema mediante una arquitectura multiagente que organiza, integra y analiza de forma autónoma datos dentales heterogéneos, proyectándolos sobre un gemelo digital del paciente. El proceso es reversible: el sistema puede regenerar ficheros STL e imágenes directamente desde el Digital Twin.

---

## Cómo encaja todo (vista rápida)

Varios **agentes** (trabajadores con una única responsabilidad) traducen ficheros
clínicos heterogéneos (DICOM, STL, PDF, foto) a un **documento común** —el
`TwinSnapshot` de [`core-schemas`](packages/core-schemas/)—, lo enriquecen y lo
materializan para que un **visor** lo muestre; un **orquestador**
([`agent-orchestrator`](apps/agent-orchestrator/)) reparte el trabajo. El
«modelo» (LLM) no es una capa central: es el *cerebro* que razona **dentro** de un
agente concreto (hoy solo `research-agent`), y no todos lo necesitan.

> 📐 **Mapa completo de las 6 capas y el recorrido del dato** (pensado para quien
> llega nuevo): [`docs/architecture/multi-agent-pipeline.md` §0](docs/architecture/multi-agent-pipeline.md#0-vista-de-conjunto-para-quien-llega-nuevo).

## Estado actual (semanas 1–4)

La **capa de ingesta está construida y probada**; la fusión, el análisis y la
exportación son el trabajo de las semanas siguientes. Lo que ya funciona hoy:

**Contrato de datos** — `core-schemas` (Pydantic v2, esquema **`1.2.0`**). El
`TwinSnapshot` es el documento común: `gaussian_field_ref` (campo 3DGS),
`surface_ref` (malla), `image_refs` (fotos, lista), `regional` (observaciones por
diente FDI) y `provenance` por valor (trazabilidad raw→contrato).

**Agentes de ingesta** — `packages/ingestion-agents`: **4 modalidades**, una por
soporte, deterministas y *fail-loud* (nunca lanzan una excepción; devuelven estado +
confianza y dejan la basura en cuarentena):

| Agente | Entrada | Produce |
|---|---|---|
| `mesh-agent` | STL / OBJ (escáner intraoral) | superficie + normales |
| `cbct-agent` | DICOM (CBCT) | volumen → campo de gaussianas |
| `report-agent` | PDF / TXT (informe clínico) | pH por diente FDI (reglas o LLM) |
| `image-agent` | JPG / PNG / HEIC (foto) | píxeles RGB **sin EXIF** |

Diseño transversal: **Provenance** por valor, **ArtifactStore** direccionado por
contenido (SHA-256), **gate de human-in-the-loop** por umbral de confianza (0,7) y
**anonimización** (EXIF fuera, seudonimización HMAC — ver
[`docs/architecture/anonymization-strategy.md`](docs/architecture/anonymization-strategy.md)).

**Orquestador** — `agent-orchestrator` dispara los agentes en paralelo, ensambla el
`TwinSnapshot`, aplica el gate HITL y respeta el presupuesto de <60 s.

**Reconstrucción 3DGS** (en notebooks, ver más abajo): malla real → **Blender**
(vistas con pose exacta, sin COLMAP) → **gsplat** → campo de gaussianas evaluable en
vistas retenidas, servido en un **visor web** ([`dental-3dgs-viewer`](https://github.com/lgarbayo/dental-3dgs-viewer),
repo aparte) con dos casos reales — Teeth3DS+ (con color por armónicos) y Bite2Text
(color de esmalte/encía **muestreado de las fotos** con el `image-agent`).

**Cobertura**: la suite completa en verde, verificada en cada push y cada PR por
el workflow [`tests`](.github/workflows/tests.yml) — el badge de arriba lo publica
esa ejecución. El CI **falla si la cobertura de agentes y pipeline baja del 80 %**,
que es el criterio de éxito del proyecto; el umbral vive en `pyproject.toml`, así
que `uv run pytest --cov` mide en local exactamente lo mismo. Aquí no se escribe
ningún número a mano: los recuentos manuales envejecen solos.

**Todavía no**: fusión multimodal (registrar y combinar CBCT + STL + foto), color
**per-píxel** (registro foto↔malla — probado, no converge barato sin calibración),
agentes de **análisis** (segmentación, patología) y **exportación reversible**. El
paquete `3dgs-engine` es hoy un placeholder: la reconstrucción vive en los notebooks
+ `gsplat`.

---

## Arquitectura del monorepo

El repositorio está organizado como un **monorepo gestionado con [`uv` workspaces`](https://docs.astral.sh/uv/concepts/workspaces/)**. El archivo `pyproject.toml` raíz declara el workspace y agrupa automáticamente todos los miembros bajo `apps/` y `packages/`:

```toml
[tool.uv.workspace]
members = ["apps/*", "packages/*"]
```

Esto permite que cada aplicación y paquete tenga su propio `pyproject.toml` y ciclo de vida independiente, mientras comparten un único entorno virtual (`.venv/`) en la raíz y un lockfile común (`uv.lock`). Las dependencias internas se resuelven mediante referencias de workspace (`workspace = true`), sin pasar por PyPI.

```
agentic-smart-health/          ← workspace root
├── pyproject.toml             ← declaración del workspace uv
├── uv.lock                    ← lockfile unificado
├── Makefile                   ← comandos de desarrollo
├── apps/
│   ├── agent-orchestrator/    ← orquestador del sistema multiagente
│   ├── research-agent/        ← agente de investigación (RAG + literatura científica)
│   └── slicer-mcp-server/     ← servidor MCP para integración con 3D Slicer
├── packages/
│   ├── core-schemas/          ← esquemas Pydantic compartidos (el contrato TwinSnapshot)
│   ├── ingestion-agents/      ← 4 agentes de ingesta (mesh · cbct · report · image)
│   ├── fusion-agents/         ← fusión geométrica y semántica sobre el twin
│   ├── tooth-aggregation/     ← agregación de etiquetas por punto a instancias de diente
│   └── 3dgs-engine/           ← placeholder (la reconstrucción 3DGS vive hoy en notebooks + gsplat)
├── data/
│   └── research-agent/        ← knowledge base del agente de investigación
├── docs/                      ← documentación (ver nota más abajo)
├── notebooks/                 ← experimentación y exploración (01–07)
├── tests/                     ← suite de pruebas global
├── scripts/                   ← utilidades: render Blender, auditor de PRs, fetch de datasets
└── .github/
    └── workflows/             ← CI: agente de revisión de código (ai-code-reviewer)
```

---

## Aplicaciones (`apps/`)

### `agent-orchestrator`

Orquestador central del sistema multiagente. Coordina los agentes de cada fase del pipeline:

- **Ingesta** ✅ *(implementado)*: dispara los 4 agentes de `ingestion-agents` en paralelo sobre una adquisición (STL + CBCT + informe + N fotos), ensambla el `TwinSnapshot` y aplica el gate de revisión humana; presupuesto de <60 s.
- **Fusión** *(pendiente)*: integración multimodal y temporal de los datos en el Digital Twin.
- **Análisis** *(pendiente)*: razonamiento clínico sobre el estado del gemelo digital.
- **Exportación** *(pendiente)*: regeneración reversible de ficheros STL e imágenes desde el Digital Twin.

Depende de `core-schemas` e `ingestion-agents` (vía workspace) para garantizar contratos de datos compartidos con el resto del sistema.

### `slicer-mcp-server`

Servidor **MCP (Model Context Protocol)** que expone una interfaz para la integración con [3D Slicer](https://www.slicer.org/), la plataforma open source de referencia para visualización y análisis de imágenes médicas. Permite que los agentes interactúen con modelos 3D e imágenes DICOM directamente desde el entorno de Slicer.

Depende igualmente de `core-schemas` para mantener la coherencia de los datos a través de la interfaz MCP.

### `research-agent`

Agente de investigación autónomo que busca, ingerir y resume literatura científica sobre 3D Gaussian Splatting, el estándar DICOM y normativas clínicas. Construido con Python, Anthropic Claude / Ollama, Qdrant y embeddings locales.

**Funcionalidades principales:**
- Búsqueda semántica de papers en Semantic Scholar y arXiv
- Ingesta y indexación de documentos mediante RAG (Qdrant + fastembed)
- Generación de reportes estructurados en Markdown
- Soporte para ejecución local con Ollama (gratis, sin API key)

**Modos de ejecución:**
- `uv run python -m src.main` — Claude con tool calling nativo (requiere API key)
- `uv run python -m src.main_local` — Ollama local (gratis, 100% privado)

**Corpus de partida.** Los PDF de referencia **no están en el repositorio**: son
binarios de terceros y la licencia de buena parte de ellos no permite
redistribuirlos. Lo que se versiona es el inventario
([`manifest.yaml`](data/research-agent/knowledge_base/manifest.yaml): título, DOI o
arXiv ID, URL y licencia verificada en origen de cada documento). Para
materializarlos:

```bash
uv run python scripts/fetch_knowledge_base.py          # baja lo que falte
uv run python scripts/fetch_knowledge_base.py --check  # solo comprueba
```

Un par de editores (Wiley, AAAI) no sirven el PDF a un script: esos quedan como
descarga manual y el comando imprime el enlace. El agente funciona sin corpus —
`search_references` descubre literatura nueva—, pero `read_directory` e `index`
no encontrarán nada hasta que se ejecute.

**Estructura:**
- `src/main.py` — Orquestador CLI con Claude
- `src/main_local.py` — Variante local con Ollama
- `src/tools.py` — Herramientas de sistema (sandbox de disco)
- `src/rag.py` — Motor RAG (Qdrant + fastembed)
- `src/references.py` — Descubrimiento de papers

No depende de `core-schemas`; mantiene sus propios modelos internos para RAG.

**Nota:** Este agente es un port de [jeicob](https://github.com/lgarbayo/jeicob), adaptado para integrarse en el monorepo.

---

## Paquetes compartidos (`packages/`)

### `core-schemas`

Biblioteca de **esquemas Pydantic v2** compartidos por todas las aplicaciones del workspace. Define los modelos de datos canónicos del sistema: el `TwinSnapshot`, la `Provenance`, las observaciones regionales por diente FDI y los contratos entre agentes. Actúa como **fuente única de verdad** de los tipos de datos del proyecto (esquema versionado, hoy `1.2.0`).

### `ingestion-agents`

**Capa de ingesta** del pipeline: 4 agentes (`mesh` · `cbct` · `report` · `image`), uno por modalidad/soporte, que traducen los ficheros crudos al contrato. Cada agente es **determinista y fail-loud** (nunca lanza; devuelve estado + confianza y aísla en cuarentena), adjunta **Provenance** por valor y guarda los artefactos pesados (mallas, volúmenes, píxeles) en un **ArtifactStore direccionado por contenido** (SHA-256). El `image-agent` descarta el **EXIF** por construcción (privacidad). Guía para añadir o modificar un agente: skill `add-ingestion-agent`; ficha completa en [`AGENTS.md`](AGENTS.md).

### `3dgs-engine`

**Placeholder.** Reservado para el motor de renderizado/procesamiento 3D Gaussian Splatting como paquete reutilizable. Hoy la reconstrucción 3DGS **no** vive aquí, sino en los [notebooks](notebooks/) (`gsplat` + Blender) y en el visor web. Se promoverá a paquete cuando la receta se estabilice y deje de ser experimental.

---

## Notebooks — pruebas de concepto (spikes)

El directorio [`notebooks/`](notebooks/) contiene **spikes de validación técnica**
(no el sistema final ni resultados clínicos): pruebas manuales que de-arriesgan las
decisiones de arquitectura antes de convertir cada eslabón en agente. Corren sobre
dos datasets reales: **Teeth3DS+** (01–06, escáneres intraorales etiquetados,
CC-BY) y **Bite2Text** (07, escáner + fotos + informes, CC-BY-SA). Ambos gitignored.

| Notebook | Qué valida | Dataset | GPU |
|---|---|---|---|
| `01` | Malla → *splatting clásico* (VTK, baseline) → contrato · caracterización del dataset | Teeth3DS+ | No |
| `02` | Visor 3D interactivo de escritorio (VTK), sobre cualquier caso | Teeth3DS+ | No |
| `03` | Vistas sintéticas + poses de cámara (input del 3DGS, sin COLMAP) | Teeth3DS+ | No |
| `04` | **3DGS moderno entrenado** (`gsplat`) evaluado en vistas retenidas → contrato | Teeth3DS+ | Sí |
| `05` | Vistas sintéticas **densas** (528/caso) — rejilla más fina que `03` | Teeth3DS+ | No |
| `06` | 3DGS **denso** con la receta de referencia (SSIM + densificación/poda, armónicos g2) | Teeth3DS+ | Sí |
| `07` | **Escáner real → Blender (EEVEE) → 3DGS**, con **color de las fotos** (`image-agent`) y pérdida SSIM · 1600 vistas · holdout 31,5 dB | Bite2Text | Sí |

Detalle, alcance y cómo ejecutarlos: [`notebooks/README.md`](notebooks/README.md).
El notebook `07` es el que integra los **agentes de ingesta** (`mesh` + `report` +
`image`) en el flujo de reconstrucción. **No** cubierto todavía: fusión multimodal
real (CBCT + STL + foto en un mismo twin), **color per-píxel** (registro foto↔malla)
y los agentes de **análisis**.

## Revisión de código y CI (`ai-code-reviewer`)

Cada Pull Request pasa por un **agente guardián de revisión estática** ejecutado en GitHub Actions. No usa LLM: combina linters estándar con un auditor de arquitectura propio, y revisa **únicamente los archivos Python que toca el PR** (enfocado en el diff). Publica anotaciones inline sobre las líneas afectadas y un comentario-resumen en el PR.

**Qué comprueba:**

| Chequeo | Herramienta | ¿Bloquea el merge? |
|---|---|---|
| Estilo y formato | `ruff` | No — informativo (anotaciones inline) |
| Tipos | `mypy` | No — informativo (anotaciones inline) |
| **Arquitectura** | `scripts/audit_pr.py` | **Sí** — hace fallar el check |

**Reglas de arquitectura (bloqueantes):**

- **Pydantic v2 estricto** en `packages/core-schemas`: prohíbe el shim `pydantic.v1` y los idiomas de v1 (`@validator`, `@root_validator`, `class Config`, `BaseSettings`).
- **Sin dependencias cruzadas entre `apps/`**: un app no puede importar el paquete de otro; el código compartido debe vivir en `packages/` (p. ej. `core-schemas`).

**Componentes:**

- `.github/workflows/ai-code-review.yml` — orquesta los chequeos, publica comentarios y decide el gate de merge.
- `scripts/audit_pr.py` — auditor de arquitectura (AST, solo librería estándar).

Utilidades del repositorio (esta tabla la genera `docs_sync.py`):

<!-- generado: scripts — no editar a mano -->
| Script | Qué hace |
|---|---|
| [`scripts/ablacion_recetas.py`](scripts/ablacion_recetas.py) | Ablación de la receta de entrenamiento: qué aporta cada pieza. |
| [`scripts/audit_pr.py`](scripts/audit_pr.py) | Agentic Smart Health. |
| [`scripts/blender_render_views.py`](scripts/blender_render_views.py) | Render multivista de una malla intraoral con **Blender** (headless). |
| [`scripts/docs_sync.py`](scripts/docs_sync.py) | Comprueba que la documentación no le mienta al código. |
| [`scripts/fetch_knowledge_base.py`](scripts/fetch_knowledge_base.py) | Materializa la knowledge base del `research-agent`. |
| [`scripts/fetch_teeth3ds.sh`](scripts/fetch_teeth3ds.sh) | Descarga reproducible de Teeth3DS+ desde el Google Drive oficial. |
<!-- /generado: scripts -->

Las herramientas de desarrollo se instalan con `uv sync --group dev` (grupo `dev`: `ruff`, `mypy`). Ficha completa del agente en [`AGENTS.md`](AGENTS.md).

---

## Quickstart

### Requisitos previos

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) instalado en el sistema

### Instalación

Clona el repositorio e instala todas las dependencias del workspace con un único comando:

```bash
git clone https://github.com/anfaia/agentic-smart-health.git
cd agentic-smart-health
make install
```

Esto ejecuta `uv sync`, que resuelve y bloquea todas las dependencias (internas y externas) y crea el entorno virtual en `.venv/`.

### Comandos disponibles

<!-- generado: make — no editar a mano -->
| Comando | Ejecuta |
|---|---|
| `make install` | `uv sync` |
| `make test` | `uv run pytest` |
| `make lint` | `uv run ruff check` |
<!-- /generado: make -->

`make install` activa además los **hooks de git** del repositorio
(`git config core.hooksPath .githooks`). El de `pre-commit` regenera los bloques
generados de la documentación —tablas de variables, scripts, comandos y registro de
agentes— y los **añade al mismo commit**, para que la documentación viaje siempre
con el cambio que la afecta. Solo toca lo que hay entre marcas: la prosa nunca.
Si alguna vez estorba, `git commit --no-verify` se lo salta, y el CI seguirá
avisando en la PR.

### Activar el entorno (opcional)

Si necesitas trabajar directamente en el entorno virtual:

```bash
source .venv/bin/activate
```

O bien, usa el prefijo `uv run` para ejecutar cualquier comando dentro del entorno sin activarlo:

```bash
uv run python -c "import core_schemas; print('workspace OK')"
```

---

## Variables de entorno

Copia el archivo de ejemplo y configura las variables necesarias:

```bash
cp .env.example .env
```

`.env.example` documenta **solo las variables que el código lee de verdad**, con
quién las usa y qué pasa si no se definen. Esta tabla la genera
[`scripts/docs_sync.py`](scripts/docs_sync.py) leyendo el código, y el CI falla si
se desincroniza — por eso no se edita a mano. Un `—` en la última columna significa
que la llamada no lleva valor por defecto (el módulo puede tener su propio respaldo):

<!-- generado: env-vars — no editar a mano -->
| Variable | Se lee en | Por defecto |
|---|---|---|
| `ANTHROPIC_API_KEY` | `apps/research-agent/src/main.py` | — |
| `ASH_PSEUDONYM_SALT` | `packages/ingestion-agents/src/ingestion_agents/cbct_agent.py` | `dev-salt-no-usar-en-produccion` |
| `OLLAMA_HOST` | `apps/research-agent/src/main_local.py` | `http://localhost:11434` |
| `QDRANT_PATH` | `apps/research-agent/src/rag.py` | — |
| `RESEARCH_AGENT_LOCAL_MODEL` | `apps/research-agent/src/main_local.py` | `qwen2.5:7b` |
| `RESEARCH_AGENT_MODEL` | `apps/research-agent/src/main.py` | `claude-opus-4-8` |
<!-- /generado: env-vars -->

Ninguna hace falta para ejecutar `make test`. La sal de seudónimo es la única que
es un **secreto**: sin ella el pipeline funciona, pero los seudónimos que emite no
sirven para datos de pacientes — y si cambia después, dejan de coincidir con los ya
emitidos.

---

## Documentación

> **Nota:** el directorio `docs/` está reservado exclusivamente para documentación de investigación y arquitectura del proyecto. No contiene documentación de usuario ni tutoriales de uso del código.
>
> - `docs/architecture/` — decisiones de diseño, diagramas de arquitectura y ADRs (Architecture Decision Records).
> - `docs/research/` — referencias bibliográficas, notas de investigación sobre Gaussian Splatting, estándares DICOM/STL, interoperabilidad clínica y normativa aplicable (RGPD, HIPAA).

La documentación técnica orientada a desarrolladores y contribuidores se mantendrá en este README y en los `pyproject.toml` de cada componente.

---

## Hitos del proyecto

| Semana | Hito |
|---|---|
| 2 | Revisión de arquitectura multiagente y esquema de atributos clínicos del Digital Twin |
| 4 | Demo PoC: agentes de ingesta + primera versión del Digital Twin con datos sintéticos |
| 6 | Sistema integrado: agentes de fusión y exportación, regeneración STL desde el Digital Twin |
| 8 | MVP testado, validación preliminar con la organización partner, documentación técnica final |

---

## Métricas de éxito

- Cobertura de pruebas automatizadas > 80% del código de agentes y pipeline.
- Fidelidad de reconstrucción STL desde el Digital Twin: error de malla < 0,1 mm.
- Latencia de ingesta de un conjunto completo (STL + CBCT + informe clínico): < 60 segundos.
- Fiabilidad de los agentes de ingesta: > 95% en el dataset de validación.

---

## Licencia

[Apache License 2.0](LICENSE)

---

*Becas de Verano ANFAIA 2026 · Julio – Agosto 2026*
