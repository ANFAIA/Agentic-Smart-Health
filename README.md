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
| `make hooks` | `git config core.hooksPath .githooks` |
| `make test` | `uv run pytest` |
| `make lint` | `uv run ruff check` |
| `make docs` | `uv run python scripts/docs_sync.py --write` |
<!-- /generado: make -->

`make install` activa además los **hooks de git** del repositorio
(`git config core.hooksPath .githooks`), y el de `pre-commit` hace dos cosas:

- **Detiene el commit** si `data_guard.py` encuentra un dato ajeno en el stage
  (un PDF, una malla, un binario grande). Es el único sitio donde eso sale barato:
  una vez commiteado, sacarlo obliga a reescribir la historia.
- **Regenera los bloques generados** de la documentación —tablas de variables,
  scripts, comandos y registro de agentes— y los **añade al mismo commit**, para
  que la documentación viaje siempre con el cambio que la afecta. Solo toca lo que
  hay entre marcas: la prosa nunca.

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

## Estado actual — MVP cerrado (semana 8)

La **ingesta, la fusión, la segmentación y los cuatro canales de exportación están
construidos y probados**, y el recorrido completo entrada → twin → fichero tiene prueba de
integración. El entregable es un contenedor **`.uos`**: un caso clínico real cierra en 12
entradas y 18 assets, conformidad UOS-Core + UOS-Vol, 0 errores. Lo adquirido no viaja
dentro —se declara por su dirección de contenido, con hash por corte para los 397 del
CBCT— y el [visor de referencia](https://github.com/lgarbayo/uos-viewer) lo abre en el
navegador sin subir nada.

Lo que está **medido**, cada número releyendo lo que se produjo en vez de prometerlo:

| Qué | Medida | Sobre |
|---|---|---|
| Reversibilidad de la malla | **3,8 × 10⁻⁶ mm** de desviación máxima, contra un presupuesto de **0,1 mm** | escaneo real, 110.804 vértices | <!--const:REVERSIBILITY_BUDGET_MM-->
| Registro CBCT ↔ intraoral | **0,452 mm** sobre la población solapada | paciente real |
| Render desde el campo | **PSNR 102 dB · SSIM 0,99999999**, reproducible byte a byte | ciclo twin → PLY → render |
| Arcada imprimible | **0,372 mm (p95) · sesgo −0,02 mm** — *esto no es reversibilidad*: mide el reconstructor de raíces contra la corona escaneada | única banda con dos medidas del mismo tejido |

**Contrato de datos** — `core-schemas` (Pydantic v2, esquema **`1.6.0`**). El <!--const:SCHEMA_VERSION-->
`TwinSnapshot` es el documento común, con `provenance` por valor. Los agentes de ingesta
son deterministas y *fail-loud*: nunca lanzan, devuelven estado y confianza, y hay un
**gate de human-in-the-loop** por umbral (0,7). El orquestador respeta un <!--const:DEFAULT_HITL_THRESHOLD-->
presupuesto de <60 s. <!--const:LATENCY_BUDGET_S-->

**Todavía no**: color **per-píxel** —la señal está medida, falta la pose de cámara— y el
`pathology-agent`. `3dgs-engine` es un placeholder: la reconstrucción vive en los
notebooks con `gsplat`.

> 📋 **El inventario honesto del cierre** —qué está medido, qué no está resuelto, en qué
> orden atacarlo, los hitos y las métricas de éxito— está en
> [`docs/cierre-mvp.md`](docs/cierre-mvp.md), junto con lo que el CI **no** puede
> verificar por no tener GPU.

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
├── packages/
│   ├── core-schemas/          ← esquemas Pydantic compartidos (el contrato TwinSnapshot)
│   ├── ingestion-agents/      ← 4 agentes de ingesta (mesh · cbct · report · image)
│   ├── fusion-agents/         ← fusión geométrica y semántica sobre el twin
│   ├── analysis-agents/       ← segmentación anatómica: region_id (FDI) por gaussiana
│   ├── export-agents/         ← regeneración de malla, campo y render desde el twin, con el error medido
│   ├── gaussian-engine/       ← ajuste de elipsoides anisótropos a la densidad que midió el CBCT
│   ├── uos/                   ← contenedor Unified Oral Scene: el caso entero con sus relaciones declaradas
│   ├── tooth-aggregation/     ← agregación de etiquetas por punto a instancias de diente
│   └── 3dgs-engine/           ← placeholder (la reconstrucción 3DGS vive hoy en notebooks + gsplat)
├── data/
│   └── research-agent/        ← knowledge base del agente de investigación
├── schemas/                   ← JSON Schema publicado del manifiesto UOS, por versión (§12)
├── docs/                      ← documentación (ver nota más abajo)
├── notebooks/                 ← experimentación y exploración (01–09)
├── experiments/               ← bancos de prueba fuera del pipeline (CBCT→Blender→3DGS, capas por HU)
├── tests/                     ← suite de pruebas global
├── scripts/                   ← utilidades: render Blender, auditor de PRs, fetch de datasets
└── .github/
    └── workflows/             ← CI: agente de revisión de código (ai-code-reviewer)
```

---

## Aplicaciones (`apps/`)

### `agent-orchestrator`

Orquestador central del sistema multiagente. Coordina los agentes de cada fase del pipeline:

- **Ingesta** ✅ *(implementado)*: dispara los 4 agentes de `ingestion-agents` en paralelo sobre una adquisición (STL + CBCT + informe + N fotos), ensambla el `TwinSnapshot` y aplica el gate de revisión humana; presupuesto de <60 s. <!--const:LATENCY_BUDGET_S-->
- **Fusión** ✅ *(implementado)*: `IngestionPipeline.fuse()` encadena **dos** `GeometricFusionAgent` —registro escáner↔escáner y el ICP IOS↔CBCT, con su `rms_error_mm` y su estado de verificación— y el `SemanticFusionAgent`, que cuelga los hallazgos del informe de códigos FDI y marca el conflicto cuando informe y geometría discrepan.
- **Análisis** 🟡 *(la parte anatómica, sí; la clínica, no)*: el `segmentation-agent` corre **dentro de `fuse()`**, entre las dos etapas de fusión, y llena `PipelineResult.analysis`. ⚠️ Su calidad está medida y es el hueco principal del MVP: **11 de 14 piezas se descartan por anatomía** ([`docs/research/segmentacion-fdi-escaner.md`](docs/research/segmentacion-fdi-escaner.md)). El razonamiento clínico —`pathology-agent`— sigue `planned`, y va con revisión humana obligatoria por diseño.
- **Exportación** ✅ *(los cuatro canales)*: `export-agents` regenera desde el `TwinSnapshot` la **malla** en STL, el **campo gaussiano** en PLY (en el marco del twin o en mm reales del CBCT) y un **render multivista** en PNG por Beer-Lambert, cada uno con su error medido —desviación máxima y media para la geometría, PSNR/SSIM para la imagen—. Los dispara el orquestador con `IngestionPipeline.exportar(result, destino)`, y el recorrido completo **entrada → twin → fichero** está probado de punta a punta en `tests/test_e2e.py`.

Depende de `core-schemas` e `ingestion-agents` (vía workspace) para garantizar contratos de datos compartidos con el resto del sistema.

### Interoperabilidad con [3D Slicer](https://www.slicer.org/) y otras plataformas

**Por formatos abiertos, no por un servidor.** El pipeline materializa cada caso en STL, PLY
y PNG, más el JSON del propio `TwinSnapshot`, y todos ellos los lee Slicer de forma nativa.
Eso ya es interoperabilidad: no hay protocolo que negociar ni servicio que mantener vivo, y
el fichero sigue abriéndose dentro de diez años sin nosotros.

Hubo aquí un `slicer-mcp-server` y **se ha retirado**. Era un directorio con un `server.py`
de **cero líneas** descrito en este mismo README en presente —«expone una interfaz»,
«permite que los agentes interactúen»— y su desbloqueo dependía de que un tercero
confirmase formato y sentido de la llamada. Una pieza vacía que no podemos desbloquear
nosotros no es arquitectura: es una intención escrita en el sitio donde se documentan los
hechos.

Un servidor MCP tendría sentido para interacción **viva y bidireccional** — que un agente
conduzca la sesión de Slicer, no que lea un fichero. Nadie ha pedido eso todavía, y cuando
se pida se construye. Ver la issue #40, que ahora es una pregunta al partner y no un
componente de este repositorio.

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

Una frase por paquete; la ficha completa de cada agente está en [`AGENTS.md`](AGENTS.md).

### `core-schemas`

**Fuente única de verdad** de los tipos: esquemas Pydantic v2 compartidos por todo el
workspace —`TwinSnapshot`, `Provenance`, observaciones por diente FDI— versionados, hoy
`1.6.0`. <!--const:SCHEMA_VERSION-->

### `ingestion-agents`

Los **4 agentes de ingesta** (`mesh` · `cbct` · `report` · `image`), uno por modalidad.
Deterministas y *fail-loud*, con `Provenance` por valor, `ArtifactStore` direccionado por
contenido y EXIF descartado por construcción. Para añadir uno: skill `add-ingestion-agent`.

### `export-agents`

La única familia que **escribe** ficheros de salida: STL, PLY, PNG multivista y arcada
imprimible. Todos **miden lo que producen releyéndolo**. Dos cosas que sorprenden: el PLY
del campo no es un `.ply` de 3DGS y el render no rasteriza splats —`density` es atenuación
radiológica, no opacidad, así que se compone por Beer-Lambert, que además es independiente
del orden y por eso reproducible byte a byte.

### `uos`

**El contenedor y su manifiesto**, el entregable del proyecto: un ZIP sin comprimir con el
caso entero y **las relaciones entre sus partes declaradas**. La regla que lo sostiene es
que **lo medido y lo inferido no se mezclan**: la inferencia vive solo bajo `derived/`, y
un `.uos` sin `derived/` sigue siendo válido y completo. Esquema en [`schemas/`](schemas/),
formato en [`docs/spec/uos-format-spec-v0.2.tex`](docs/spec/uos-format-spec-v0.2.tex).

### `fusion-agents`

**Fusión** (ADR 004): registro geométrico por ICP, declarando siempre su `rms_error_mm` y
si alguien lo ha verificado, y anclaje de los hallazgos del informe a códigos FDI, con el
**conflicto** marcado cuando informe y geometría discrepan.

### `analysis-agents`

**Análisis anatómico**: `region_id` por gaussiana y el mapa `FDI → confianza`. ⚠️ Su
calidad está medida y es el hueco principal del MVP
([`docs/research/segmentacion-fdi-escaner.md`](docs/research/segmentacion-fdi-escaner.md)).

### `gaussian-engine`

**Ajuste del campo**: de semillas isótropas del tamaño del vóxel a elipsoides medidos. El
único paquete que toca `torch`, y lo importa dentro de la función para instalarse sin CUDA.

### `tooth-aggregation`

**Agregación punto → diente** (instancias + FDI). Sin depender de `torch` a propósito: el
*forward* es cosa de quien llama.

### `3dgs-engine`

**Placeholder.** La reconstrucción 3DGS vive hoy en los [notebooks](notebooks/) con
`gsplat` y Blender. Se promoverá a paquete cuando la receta deje de ser experimental.

---

## Notebooks — pruebas de concepto (spikes)

Nueve **spikes de validación técnica** (no el sistema final ni resultados clínicos) que
de-arriesgan las decisiones de arquitectura antes de convertir cada eslabón en agente.
Corren sobre datasets reales gitignored: **Teeth3DS+** (01–06) y **Bite2Text** (07). El
`07` es el que integra los agentes de ingesta en el flujo de reconstrucción, con color de
las fotos y holdout de 31,5 dB.

Qué valida cada uno, alcance y cómo ejecutarlos:
[`notebooks/README.md`](notebooks/README.md).


## Revisión de código y CI

Cada Pull Request pasa por un **agente guardián de revisión estática**
([`ai-code-review.yml`](.github/workflows/ai-code-review.yml)). No usa LLM: combina Ruff y
MyPy con un auditor de arquitectura propio, y revisa **solo los ficheros Python que toca
el PR**. Publica anotaciones inline y un comentario-resumen. Las violaciones de
arquitectura y la cobertura por debajo del 80 % bloquean el merge.

Además, [`docs_sync.py`](scripts/docs_sync.py) comprueba que esta documentación no se
separe del código —rutas citadas, registro de agentes, constantes, el árbol de aquí
abajo— y un hook de pre-commit aborta el commit si intenta versionar datos clínicos.

**Vigilancia de literatura** — el único trabajo programado
([`literature-watch.yml`](.github/workflows/literature-watch.yml)): cada lunes busca en
arXiv lo publicado esa semana, lee la licencia del OAI-PMH (no la supone) y abre una PR
proponiendo entradas nuevas para el manifiesto. **No mergea.** Ningún PDF se escribe en el
runner: se descargan a memoria para calcular `sha256` y se liberan ahí mismo.


Utilidades del repositorio (esta tabla la genera `docs_sync.py`):

<!-- generado: scripts — no editar a mano -->
| Script | Qué hace |
|---|---|
| [`scripts/ablacion_recetas.py`](scripts/ablacion_recetas.py) | Ablación de la receta de entrenamiento: qué aporta cada pieza. |
| [`scripts/altura_corona.py`](scripts/altura_corona.py) | mide la altura de corona clínica sobre el escáner intraoral. |
| [`scripts/audit_pr.py`](scripts/audit_pr.py) | Guardián de las reglas de arquitectura del monorepo. |
| [`scripts/blender_render_views.py`](scripts/blender_render_views.py) | Render multivista de una malla intraoral con **Blender** (headless). |
| [`scripts/caso_completo.py`](scripts/caso_completo.py) | El pipeline entero sobre un caso clínico real, etapa por etapa. |
| [`scripts/composicion_cbct_ios.py`](scripts/composicion_cbct_ios.py) | Dientes segmentados en el CBCT + encía del IOS, en gaussianas. |
| [`scripts/data_guard.py`](scripts/data_guard.py) | Impide que datos ajenos entren al repositorio sin permiso. |
| [`scripts/desplazamiento_relativo.py`](scripts/desplazamiento_relativo.py) | ¿Se puede decir «esta pieza se desplazó X mm»? Referencia leave-one-out y umbral. |
| [`scripts/docs_sync.py`](scripts/docs_sync.py) | Comprueba que la documentación no le mienta al código. |
| [`scripts/entrena_diente_cbct.py`](scripts/entrena_diente_cbct.py) | Segmentador de diente en CBCT, contra el listón del umbral. |
| [`scripts/entrena_gs_escaner.py`](scripts/entrena_gs_escaner.py) | 3DGS de verdad sobre la superficie del escaner. |
| [`scripts/entrenar_3dgs.py`](scripts/entrenar_3dgs.py) | EXPERIMENTO con resultado NEGATIVO: 3DGS entrenado de una arcada. |
| [`scripts/eval_informes.py`](scripts/eval_informes.py) | ¿Cuánto de lo que dice un informe acaba en el contrato? |
| [`scripts/fetch_knowledge_base.py`](scripts/fetch_knowledge_base.py) | Materializa la knowledge base del `research-agent`. |
| [`scripts/fetch_teeth3ds.sh`](scripts/fetch_teeth3ds.sh) | Descarga reproducible de Teeth3DS+ desde el Google Drive oficial. |
| [`scripts/malla_mejorada.py`](scripts/malla_mejorada.py) | El STL mejorado, sacado del contenedor y de nada más. |
| [`scripts/metricas.py`](scripts/metricas.py) | las cuatro cifras del brief, MEDIDAS y no prometidas. |
| [`scripts/mide_segmentacion.py`](scripts/mide_segmentacion.py) | cuanto se puede DESCARTAR de la segmentacion FDI de un `.uos`. |
| [`scripts/prepara_toothfairy.py`](scripts/prepara_toothfairy.py) | Descarga ToothFairy2 caso a caso y lo deja entrenable. |
| [`scripts/promedio_y_escala.py`](scripts/promedio_y_escala.py) | Dos preguntas de diseño sobre el registro por diente, medidas en vez de argumentadas. |
| [`scripts/refina_3dgs.py`](scripts/refina_3dgs.py) | La fase que faltaba: el campo semilla optimizado como 3DGS. |
| [`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py) | mide si el escáner intraoral y el CBCT se pueden alinear. |
| [`scripts/resolucion_modalidades.py`](scripts/resolucion_modalidades.py) | Simula qué resolución alcanza cada modalidad dental. |
| [`scripts/segmentar_fdi.py`](scripts/segmentar_fdi.py) | etiqueta cada diente de una arcada con su código FDI. |
| [`scripts/seguimiento_histora.py`](scripts/seguimiento_histora.py) | cuánto se ha movido el margen gingival entre dos escaneos. |
| [`scripts/umbral_vs_verdad.py`](scripts/umbral_vs_verdad.py) | ¿Cuánto diente recupera un umbral, contra una verdad conocida? |
| [`scripts/verifica_contenedor.py`](scripts/verifica_contenedor.py) | que el `.uos` diga la verdad SOBRE SI MISMO. |
| [`scripts/watch_literature.py`](scripts/watch_literature.py) | Vigila la literatura y propone entradas del manifiesto. |
<!-- /generado: scripts -->

Las herramientas de desarrollo se instalan con `uv sync --group dev` (grupo `dev`: `ruff`, `mypy`). Ficha completa del agente en [`AGENTS.md`](AGENTS.md).

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
> - `docs/spec/` — la especificación normativa del formato `.uos` y el white paper del proyecto, en LaTeX.

La documentación técnica orientada a desarrolladores y contribuidores se mantendrá en este README y en los `pyproject.toml` de cada componente.

### Por dónde empezar a leer

| Documento | Responde a |
|---|---|
| [`docs/cierre-mvp.md`](docs/cierre-mvp.md) | qué está medido, qué no está resuelto y qué queda para después |
| [`docs/spec/uos-format-spec-v0.2.tex`](docs/spec/uos-format-spec-v0.2.tex) | la especificación del formato: qué lleva un `.uos`, cómo se lee, cómo se amplía |
| [`docs/spec/uos-white-paper.tex`](docs/spec/uos-white-paper.tex) | por qué hace falta un formato nuevo, qué hipótesis se probaron y con qué resultados |
| [`docs/research/segmentacion-fdi-escaner.md`](docs/research/segmentacion-fdi-escaner.md) | por qué la segmentación FDI no está resuelta, con la medida |
| [`docs/research/frontera-encia-desde-foto.md`](docs/research/frontera-encia-desde-foto.md) | dónde sí está la frontera diente-encía, y qué falta para usarla |
| [`docs/research/color-por-pieza-desde-foto.md`](docs/research/color-por-pieza-desde-foto.md) | el tono de cada corona, y cómo se descuenta la caída del flash sin invertirla |
| [`docs/research/segmentacion-diente-cbct.md`](docs/research/segmentacion-diente-cbct.md) | hasta dónde llega un clasificador sobre el CBCT, y dónde deja de llegar |

---

## Licencia

[Apache License 2.0](LICENSE)

---

*Becas de Verano ANFAIA 2026 · Julio – Agosto 2026*
