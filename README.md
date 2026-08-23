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

La **ingesta, la fusión, la segmentación y los tres canales de exportación están
construidos y probados**, y el recorrido completo entrada → twin → fichero tiene prueba de
integración; lo que queda del camino es el análisis clínico. Lo que ya funciona hoy:

**Contrato de datos** — `core-schemas` (Pydantic v2, esquema **`1.6.0`**). El <!--const:SCHEMA_VERSION-->
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
contenido (SHA-256), **gate de human-in-the-loop** por umbral de confianza (0,7) y <!--const:DEFAULT_HITL_THRESHOLD-->
**anonimización** (EXIF fuera, seudonimización HMAC — ver
[`docs/architecture/anonymization-strategy.md`](docs/architecture/anonymization-strategy.md)).

**Orquestador** — `agent-orchestrator` dispara los agentes en paralelo, ensambla el
`TwinSnapshot`, aplica el gate HITL y respeta el presupuesto de <60 s. <!--const:LATENCY_BUDGET_S-->

**Reconstrucción 3DGS** (en notebooks, ver más abajo): malla real → **Blender**
(vistas con pose exacta, sin COLMAP) → **gsplat** → campo de gaussianas evaluable en
vistas retenidas, servido en un **visor web** ([`dental-3dgs-viewer`](https://github.com/lgarbayo/dental-3dgs-viewer),
repo aparte) con dos casos reales — Teeth3DS+ (con color por armónicos) y Bite2Text
(color de esmalte/encía **muestreado de las fotos** con el `image-agent`).

Lo que el contrato promete de **todos** los agentes por igual —los tres caminos
(`OK`/`MISSING`/`FAILED`), no lanzar nunca, emitir `Provenance`, ser reproducible y
no copiar dato clínico a la cuarentena— lo verifica una **suite de conformidad**
([`test_conformidad.py`](packages/ingestion-agents/tests/test_conformidad.py))
parametrizada sobre los cuatro agentes. Un agente nuevo entra en esa lista y queda
sometido a las nueve reglas sin escribir un test.

Y frente al caso que *sale bien* de `synthetic.py` hay un catálogo de **casos
límite** ([`edge_cases.py`](packages/ingestion-agents/src/ingestion_agents/edge_cases.py)):
cabecera DICOM truncada, resonancia etiquetada como CBCT, espaciado cero o negativo,
`NaN` en la malla, PNG a medias, rutas con unicode, enlaces rotos. Cada caso declara
**qué debe pasar y por qué**, porque no todos deben fallar: un pH imposible se
descarta línea a línea y la ingesta sigue siendo válida. Encontró cuatro defectos
reales el día que se escribió.

**Cobertura**: la suite completa en verde, verificada en cada push y cada PR por
el workflow [`tests`](.github/workflows/tests.yml) — el badge de arriba lo publica
esa ejecución. El CI **falla si la cobertura de agentes y pipeline baja del 80 %**,
que es el criterio de éxito del proyecto; el umbral vive en `pyproject.toml`, así
que `uv run pytest --cov` mide en local exactamente lo mismo. Aquí no se escribe
ningún número a mano: los recuentos manuales envejecen solos.

> **Lo que el CI no verifica, dicho antes de que haga falta preguntarlo.** El runner no
> tiene GPU. Eso **no** deja partes del pipeline sin probar: `packages/` y `apps/` no
> importan `torch` en ninguna línea, a propósito — los modelos entran por los `Protocol`
> `Segmenter` y `Registrar`, así que lo que se ejecuta en producción es numpy y se prueba
> entero. Lo que queda fuera son tres scripts de investigación
> ([`entrenar_3dgs.py`](scripts/entrenar_3dgs.py),
> [`segmentar_fdi.py`](scripts/segmentar_fdi.py),
> [`ablacion_recetas.py`](scripts/ablacion_recetas.py)) y los notebooks: se ejecutan a
> mano en una máquina con GPU y su producto es una **medida**, no un servicio. Cuando un
> número de esta página sale de ahí, la sección lo dice y enlaza el script que lo produjo.

**Fusión y segmentación** — `fusion-agents` (registro geométrico + anclaje semántico
al FDI, ADR 004) y `analysis-agents` (`segmentation-agent`: `region_id` por gaussiana
y el mapa `FDI → confianza` que consume la fusión semántica). El registro
CBCT↔intraoral está **medido sobre un paciente real**
([`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py)): 0,452 mm sobre la
población solapada, con la etapa gruesa que el ADR dejaba pendiente ya implementada.

**Exportación reversible** — `export-agents`, **tres canales, y los tres miden lo que
producen releyéndolo** en vez de prometerlo:

| Agente | Materializa | Error medido |
|---|---|---|
| `export-agent` | `surface_ref` → **STL binario** | **3,8·10⁻⁶ mm** de desviación máxima sobre un escaneo real de Teeth3DS+ (110.804 vértices, arcada de 86 mm) en 0,07 s — la que impone el `float32` del formato, cuatro órdenes de magnitud bajo el presupuesto de **0,1 mm** del brief | <!--const:REVERSIBILITY_BUDGET_MM-->
| `field-export-agent` | `gaussian_field_ref` → **PLY binario** | **0,0 mm** exactos sobre el CBCT de un paciente real (498.407 primitivas, 27,9 MB en 0,06 s): las posiciones van en `double` para que la verificación mida *bugs* de formato y no el redondeo |
| `render-export-agent` | `gaussian_field_ref` → **PNG multivista** | **PSNR 102 dB · SSIM 0,99999999** en el ciclo twin → PLY → render, reproducible byte a byte |

El STL sale en el sistema del escáner o en el del twin; el PLY, centrado o en mm reales
del CBCT. Y un snapshot parcial lo declara en `hitl_reasons` **y dentro del propio
fichero**.

**Todavía no**: color **per-píxel** (registro foto↔malla — probado, no converge barato
sin calibración), `pathology-agent`. Sigue pendiente del ADR de motor de render
**de dónde sale el color** de un campo de densidad — el contenedor ya está resuelto, pero
un CBCT no mide color y el PLY no se lo inventa. El paquete `3dgs-engine` es hoy un
placeholder: la reconstrucción vive en los notebooks + `gsplat`.

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

- **Ingesta** ✅ *(implementado)*: dispara los 4 agentes de `ingestion-agents` en paralelo sobre una adquisición (STL + CBCT + informe + N fotos), ensambla el `TwinSnapshot` y aplica el gate de revisión humana; presupuesto de <60 s. <!--const:LATENCY_BUDGET_S-->
- **Fusión** *(pendiente)*: integración multimodal y temporal de los datos en el Digital Twin.
- **Análisis** *(pendiente)*: razonamiento clínico sobre el estado del gemelo digital.
- **Exportación** ✅ *(los tres canales)*: `export-agents` regenera desde el `TwinSnapshot` la **malla** en STL, el **campo gaussiano** en PLY (en el marco del twin o en mm reales del CBCT) y un **render multivista** en PNG por Beer-Lambert, cada uno con su error medido —desviación máxima y media para la geometría, PSNR/SSIM para la imagen—. Los dispara el orquestador con `IngestionPipeline.exportar(result, destino)`, y el recorrido completo **entrada → twin → fichero** está probado de punta a punta en `tests/test_e2e.py`.

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

### `core-schemas`

Biblioteca de **esquemas Pydantic v2** compartidos por todas las aplicaciones del workspace. Define los modelos de datos canónicos del sistema: el `TwinSnapshot`, la `Provenance`, las observaciones regionales por diente FDI y los contratos entre agentes. Actúa como **fuente única de verdad** de los tipos de datos del proyecto (esquema versionado, hoy `1.2.0`).

### `ingestion-agents`

**Capa de ingesta** del pipeline: 4 agentes (`mesh` · `cbct` · `report` · `image`), uno por modalidad/soporte, que traducen los ficheros crudos al contrato. Cada agente es **determinista y fail-loud** (nunca lanza; devuelve estado + confianza y aísla en cuarentena), adjunta **Provenance** por valor y guarda los artefactos pesados (mallas, volúmenes, píxeles) en un **ArtifactStore direccionado por contenido** (SHA-256). El `image-agent` descarta el **EXIF** por construcción (privacidad). Guía para añadir o modificar un agente: skill `add-ingestion-agent`; ficha completa en [`AGENTS.md`](AGENTS.md).

### `export-agents`

**Capa de exportación** (fase 6): la única familia que escribe ficheros de salida, igual que la ingesta es la única que lee ficheros de entrada. Tres canales, y **los tres miden lo que producen releyéndolo**, no estimándolo: `export-agent` → **STL** desde `surface_ref` (desviación máxima y Chamfer), `field-export-agent` → **PLY** desde `gaussian_field_ref`, `render-export-agent` → **PNG multivista** con PSNR/SSIM del ciclo. Es de **solo lectura sobre el gemelo**: no muta el snapshot y su `Protocol` de almacén ni siquiera declara `put`.

Dos decisiones que se ven raras hasta que se leen: el PLY del campo **no es un `.ply` de 3D Gaussian Splatting** y el render **no rasteriza splats**. `density` es atenuación radiológica, no opacidad, y un CBCT no mide color — así que el fichero declara las propiedades que existen y el render compone por **Beer-Lambert**, que además es independiente del orden de las primitivas y por eso reproducible byte a byte. Fichas completas en [`AGENTS.md`](AGENTS.md).

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
| **Coherencia documental** | `scripts/docs_sync.py` | **Sí** — hace fallar el check |
| **Datos y licencias** | `scripts/data_guard.py` | **Sí** — hace fallar el check |

**Reglas de arquitectura (bloqueantes):**

- **Pydantic v2 estricto** en `packages/core-schemas`: prohíbe el shim `pydantic.v1` y los idiomas de v1 (`@validator`, `@root_validator`, `class Config`, `BaseSettings`).
- **Sin dependencias cruzadas entre `apps/`**: un app no puede importar el paquete de otro; el código compartido debe vivir en `packages/` (p. ej. `core-schemas`).

**Componentes:**

- `.github/workflows/ai-code-review.yml` — orquesta los chequeos, publica comentarios y decide el gate de merge.
- `scripts/audit_pr.py` — auditor de arquitectura (AST, solo librería estándar).

Dos de esos chequeos **bloquean el merge** y el resto solo comenta, por un motivo
concreto: son los que producen daño que no se arregla con otro commit. La deriva
documental (`docs_sync.py`) se convierte en verdad publicada en cuanto se mergea, y
un dato ajeno (`data_guard.py`) entra en la historia de git y solo sale
reescribiéndola — que es lo que costó la issue 45.

### Vigilancia de literatura (`literature watch`)

El único trabajo **programado** del repositorio: cada lunes,
[`scripts/watch_literature.py`](scripts/watch_literature.py) busca en arXiv lo
publicado esa semana, descarta lo que ya está en el manifiesto, **lee la licencia del
OAI-PMH de arXiv** (no la supone) y abre una PR proponiendo las entradas nuevas.

Siete consultas en dos ámbitos, con **puerta distinta cada uno**: las cuatro
dentales (3DGS, segmentación CBCT, escaneo intraoral, gemelo digital) exigen un
término del dominio en título o resumen; las tres de estándares (DICOM, FHIR/HL7,
interoperabilidad en imagen médica) lo exigen **en el título**. La distinción está
medida, no supuesta: un artículo de interoperabilidad clínica casi nunca dice
«tooth», y uno que solo menciona DICOM de pasada no va sobre DICOM. El cupo de cada
PR se reparte por turnos entre consultas, para que las de mayor volumen no dejen la
propuesta sin un solo artículo dental.

Reparto de trabajo deliberado: la máquina hace lo repetitivo y verificable —qué hay
nuevo, bajo qué licencia—, y la persona que revisa la PR decide lo único que exige
criterio: si el artículo aporta algo al proyecto. **El agente no mergea nunca.**

Ningún PDF llega a escribirse: se descargan a memoria para calcular `sha256` y
`bytes`, y se liberan ahí mismo. Lo que se propone commitear son diez líneas de
YAML por artículo. Los ficheros se materializan después, en local, con
`uv run python scripts/fetch_knowledge_base.py`.

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
| [`scripts/prepara_toothfairy.py`](scripts/prepara_toothfairy.py) | Descarga ToothFairy2 caso a caso y lo deja entrenable. |
| [`scripts/promedio_y_escala.py`](scripts/promedio_y_escala.py) | Dos preguntas de diseño sobre el registro por diente, medidas en vez de argumentadas. |
| [`scripts/refina_3dgs.py`](scripts/refina_3dgs.py) | La fase que faltaba: el campo semilla optimizado como 3DGS. |
| [`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py) | mide si el escáner intraoral y el CBCT se pueden alinear. |
| [`scripts/resolucion_modalidades.py`](scripts/resolucion_modalidades.py) | Simula qué resolución alcanza cada modalidad dental. |
| [`scripts/segmentar_fdi.py`](scripts/segmentar_fdi.py) | etiqueta cada diente de una arcada con su código FDI. |
| [`scripts/seguimiento_histora.py`](scripts/seguimiento_histora.py) | cuánto se ha movido el margen gingival entre dos escaneos. |
| [`scripts/umbral_vs_verdad.py`](scripts/umbral_vs_verdad.py) | ¿Cuánto diente recupera un umbral, contra una verdad conocida? |
| [`scripts/watch_literature.py`](scripts/watch_literature.py) | Vigila la literatura y propone entradas del manifiesto. |
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
| `REPORT_AGENT_LOCAL_MODEL` | `packages/ingestion-agents/src/ingestion_agents/report_agent.py` | `qwen3:14b` |
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
- Fidelidad de reconstrucción STL desde el Digital Twin: error de malla < 0,1 mm. <!--const:REVERSIBILITY_BUDGET_MM-->
- Latencia de ingesta de un conjunto completo (STL + CBCT + informe clínico): < 60 segundos. <!--const:LATENCY_BUDGET_S-->
- Fiabilidad de los agentes de ingesta: > 95% en el dataset de validación.

---

## Licencia

[Apache License 2.0](LICENSE)

---

*Becas de Verano ANFAIA 2026 · Julio – Agosto 2026*
