# Agentic Smart Health

Sistema multiagente para la integración, análisis y representación de datos clínicos dentales heterogéneos sobre un **Digital Twin** del paciente, basado en Gaussian Splatting con atributos clínicos por punto/zona y soporte de series temporales.

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
│   ├── core-schemas/          ← esquemas Pydantic compartidos
│   └── 3dgs-engine/           ← motor de renderizado 3D Gaussian Splatting
├── data/
│   └── research-agent/        ← knowledge base del agente de investigación
├── docs/                      ← documentación (ver nota más abajo)
├── notebooks/                 ← experimentación y exploración
├── tests/                     ← suite de pruebas global
├── scripts/                   ← utilidades de CI (auditor de arquitectura de PRs)
└── .github/
    └── workflows/             ← CI: agente de revisión de código (ai-code-reviewer)
```

---

## Aplicaciones (`apps/`)

### `agent-orchestrator`

Orquestador central del sistema multiagente. Coordina los agentes especializados en las distintas fases del pipeline:

- **Ingesta**: lectura y normalización de STL, CBCT (DICOM) e informes clínicos en PDF.
- **Fusión**: integración multimodal y temporal de los datos en el Digital Twin.
- **Análisis**: razonamiento clínico sobre el estado del gemelo digital.
- **Exportación**: regeneración reversible de ficheros STL e imágenes desde el Digital Twin.

Depende de `core-schemas` (vía workspace) para garantizar contratos de datos compartidos con el resto del sistema.

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

Biblioteca de **esquemas Pydantic v2** compartidos por todas las aplicaciones del workspace. Define los modelos de datos canónicos del sistema: estructuras del Digital Twin, representaciones de datos clínicos dentales, contratos entre agentes y formatos de exportación. Actúa como fuente única de verdad para los tipos de datos del proyecto.

### `3dgs-engine`

Motor de **renderizado y procesamiento 3D Gaussian Splatting** (3DGS). Implementa la representación neuronal del Digital Twin: cada punto/zona del espacio almacena atributos clínicos (color, densidad ósea, datos clínicos asociados, timestamp). Proporciona las primitivas necesarias para construir, actualizar y consultar el gemelo digital, así como para las operaciones de exportación reversible a STL e imágenes.

---

## Notebooks — pruebas de concepto (spikes)

El directorio [`notebooks/`](notebooks/) contiene **spikes de validación técnica**
(no el sistema final ni resultados clínicos): pruebas manuales que de-arriesgan las
decisiones de arquitectura antes de convertir cada eslabón en agente. Corren sobre
**Teeth3DS+ completo** (`data/raw/teeth3ds/`, 300 pacientes / 600 escaneos / ~70 M
vértices etiquetados, gitignored).

| Notebook | Qué valida | Escala | GPU |
|---|---|---|---|
| `01` | Malla → *splatting clásico* (VTK, baseline) → contrato · caracterización del dataset | 600 escaneos · barrido de 24 | No |
| `02` | Visor 3D interactivo de escritorio (VTK), sobre cualquier caso | selector de los 600 | No |
| `03` | Vistas sintéticas + poses de cámara (input del 3DGS, sin COLMAP) | 2 880 vistas · 20 casos | No |
| `04` | **3DGS moderno entrenado** (`gsplat`) evaluado en vistas retenidas → contrato | 8 casos · 21,0 dB PSNR | Sí |

Detalle, alcance y cómo ejecutarlos: [`notebooks/README.md`](notebooks/README.md).
Aún **no** cubierto: foto→3D con fotos reales, fusión multimodal (CBCT+STL) e
integración como agentes.

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

| Comando | Descripción |
|---|---|
| `make install` | Sincroniza el entorno con `uv sync` |
| `make test` | Ejecuta la suite de pruebas con `pytest` |
| `make lint` | Analiza el código con `ruff check` |

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

Consulta `.env.example` para ver las variables requeridas (claves de API para modelos de IA, configuración de servicios, etc.).

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
