# AGENTS.md — Registro Central de Agentes

Este documento es la fuente única de verdad para todos los agentes del sistema **Agentic Smart Health**. Cada agente autónomo o semi-autónomo del sistema multiagente debe registrarse aquí con su rol, herramientas MCP disponibles y reglas de delegación.

Actualiza este archivo siempre que añadas, modifiques o retires un agente. Las decisiones de diseño que afecten a la arquitectura de agentes deben registrarse también en `docs/architecture/`.

---

## Principios de diseño de agentes

- **Responsabilidad única**: cada agente tiene un rol delimitado y no replica la lógica de otro.
- **Contratos de datos**: los agentes se comunican exclusivamente a través de los esquemas definidos en `packages/core-schemas`.
- **Human-in-the-loop**: las decisiones clínicamente sensibles requieren supervisión humana explícita antes de ejecutarse. Esto debe indicarse en las reglas de delegación del agente correspondiente.
- **Trazabilidad**: todo agente debe registrar qué dato ingirió, qué transformación aplicó y qué output generó.
- **Soberanía del dato**: ningún agente puede retener, reenviar ni persistir datos clínicos fuera del almacenamiento autorizado definido en la arquitectura.

---

## Registro de agentes

Resumen generado desde el código — las fichas de abajo, no: inputs, outputs y
herramientas son prosa y se escriben a mano.

<!-- generado: agentes — no editar a mano -->
| Agente | Implementado en |
|---|---|
| `cbct-agent` | [`packages/ingestion-agents/src/ingestion_agents/cbct_agent.py`](packages/ingestion-agents/src/ingestion_agents/cbct_agent.py) |
| `geometric-fusion-agent` | [`packages/fusion-agents/src/fusion_agents/geometric.py`](packages/fusion-agents/src/fusion_agents/geometric.py) |
| `image-agent` | [`packages/ingestion-agents/src/ingestion_agents/image_agent.py`](packages/ingestion-agents/src/ingestion_agents/image_agent.py) |
| `mesh-agent` | [`packages/ingestion-agents/src/ingestion_agents/mesh_agent.py`](packages/ingestion-agents/src/ingestion_agents/mesh_agent.py) |
| `report-agent` | [`packages/ingestion-agents/src/ingestion_agents/report_agent.py`](packages/ingestion-agents/src/ingestion_agents/report_agent.py) |
| `segmentation-agent` | [`packages/analysis-agents/src/analysis_agents/segmentation.py`](packages/analysis-agents/src/analysis_agents/segmentation.py) |
| `semantic-fusion-agent` | [`packages/fusion-agents/src/fusion_agents/semantic.py`](packages/fusion-agents/src/fusion_agents/semantic.py) |
<!-- /generado: agentes -->


### `research-agent` — Agente de investigación

| Campo | Valor |
|---|---|
| **Nombre** | `research-agent` |
| **Versión** | `0.1.0` |
| **Ubicación** | `apps/research-agent/` (`src/main.py` · `src/main_local.py`) |
| **Estado** | `active` |
| **Fase del pipeline** | Ingesta y síntesis de conocimiento (no toca datos clínicos del paciente) |
| **Cerebro (LLM)** | Claude (`claude-opus-4-8` por defecto) vía Tool Runner del SDK de Anthropic. Variante local sin coste con Ollama (`main_local.py`, bucle ReAct manual). |

**Rol / Propósito**

> Agente conversacional (CLI) que **descubre, ingiere, indexa y sintetiza**
> literatura científica sobre 3D Gaussian Splatting, el estándar DICOM y
> normativas clínicas. Recupera papers de fuentes académicas abiertas, los
> vuelca en una base vectorial local (RAG) y produce reportes Markdown
> estructurados (abstract + explicación completa) en `docs_output/`. Actúa en la
> fase de **ingesta de conocimiento**: alimenta al proyecto con contexto
> bibliográfico; **no procesa datos clínicos de pacientes**.

**Herramientas a las que tiene acceso** (tool calling nativo `@beta_tool`, **no** MCP)

| Herramienta | Backend | Permisos | Notas |
|---|---|---|---|
| `read_directory` | Filesystem (`data/research-agent/knowledge_base/`) | read | Lista los documentos disponibles en el corpus. |
| `read_file` | Filesystem (`knowledge_base/`) | read | Lee un PDF/MD/TXT completo (truncado a 100k chars). Sandbox anti *path traversal*. |
| `ingest_corpus` | RAG — Qdrant on-disk + `fastembed` | write (índice) | Trocea (chunk 1000/overlap 150), vectoriza e indexa todo el corpus. Idempotente. |
| `search_corpus` | RAG — Qdrant | read | Búsqueda semántica (coseno, `top_k=5`) multi-documento. |
| `search_references` | HTTP — Semantic Scholar → arXiv (fallback) | read (red) | Descubre papers externos; sin API key. |
| `download_reference` | HTTP + Filesystem (`knowledge_base/`) | write (fichero) | Descarga un PDF (solo http(s), valida `%PDF`, máx. 50 MB) y lo auto-indexa. |
| `write_summary` | Filesystem (`docs_output/`) | write | Persiste el reporte final; valida estructura (abstract + explicación más extensa). Fuerza `.md` y nombre base. |

> **Fronteras de seguridad:** lectura confinada a `knowledge_base/`, escritura
> confinada a `docs_output/`; `../` y symlinks que escapen se bloquean antes de
> tocar disco. Modelo por defecto vía `ANTHROPIC_API_KEY` (`.env`); la variante
> Ollama no envía documentos fuera de la máquina.

**Inputs esperados**

```
Consulta en lenguaje natural del usuario (CLI interactivo), p. ej.:
  "Busca literatura reciente sobre 3DGS en imagen dental y resúmela."
Corpus de partida (opcional): ficheros .pdf/.md/.txt en
  data/research-agent/knowledge_base/
  Los PDF NO estan versionados (licencia de terceros): se materializan con
  `uv run python scripts/fetch_knowledge_base.py` a partir de manifest.yaml
```

**Outputs generados**

```
- Reporte Markdown en apps/research-agent/docs_output/resumen_<tema>.md
  Estructura: # Título · > Fuente · ## Abstract · ## Explicación completa
              · ## Puntos clave (opcional)
- Efecto lateral: base vectorial Qdrant persistida en
  data/research-agent/.qdrant_data/ (colección "papers")
- Respuestas conversacionales citando la fuente (nombre de documento)
```

**Reglas de delegación**

- Flujo autónomo de descubrimiento: `search_references` → `download_reference`
  (auto-indexa) → `search_corpus`/`read_file` → `write_summary`.
- **No requiere aprobación humana**: opera sobre literatura pública y su propio
  sandbox de ficheros; no accede a datos clínicos ni a almacenamiento autorizado
  del paciente.
- No puede delegar en otros agentes del sistema (no hay integración con el
  orquestador todavía); es un agente autocontenido de un solo turno interactivo.
- Política de fallo: las tools **nunca** lanzan al llamador — devuelven el error
  como texto (`ERROR: …`) para que el modelo reaccione/reintente. Semantic
  Scholar cae automáticamente a arXiv ante cualquier fallo (incl. 429).
- Concurrencia: Qdrant on-disk bloquea el directorio para un solo proceso; para
  concurrencia real se migraría a Qdrant en servidor sin cambiar la interfaz.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-07-14 | 0.1.0 | Registro inicial: RAG local (Qdrant + fastembed), tools de filesystem, descubrimiento externo (Semantic Scholar/arXiv) y generación de reportes. Variantes Claude y Ollama. |

---

### `ai-code-reviewer` — Agente guardián de CI/CD

| Campo | Valor |
|---|---|
| **Nombre** | `ai-code-reviewer` |
| **Tipo** | Agente guardián dev-time (no forma parte del sistema en producción) |
| **Ubicación** | `.github/workflows/ai-code-review.yml` + `scripts/audit_pr.py` |
| **Estado** | `active` |
| **Tecnología** | Revisión **estática, sin LLM**: Ruff + MyPy + auditor de arquitectura propio |

**Rol / Propósito**

> Audita automáticamente cada Pull Request antes del merge para garantizar calidad
> de código, tipado y cumplimiento de la arquitectura hexagonal del monorepo.
> Se dispara en los eventos `pull_request` (`opened`, `synchronize`, `reopened`) y
> revisa **únicamente los archivos Python que toca el PR** (enfocado en el diff).

**Chequeos que ejecuta**

| Chequeo | Herramienta | ¿Bloquea el merge? | Cómo reporta |
|---|---|---|---|
| Estilo / lint | `ruff check` | No (informativo) | Anotaciones inline nativas de GitHub |
| Formato | `ruff format --diff` | No (informativo) | Log del job |
| Tipos | `mypy` | No (informativo) | Anotaciones inline (`::error`/`::warning`) |
| **Arquitectura** | `scripts/audit_pr.py` | **Sí** | Comentario de revisión en la línea afectada + resumen |

**Reglas de arquitectura auditadas** (regla → violación que detecta)

1. **Pydantic v2 estricto en `packages/core-schemas`**: prohíbe el shim `pydantic.v1`,
   los decoradores/estilos de v1 (`@validator`, `@root_validator`, `class Config`) y
   `BaseSettings` (movido a `pydantic-settings`).
2. **Sin dependencias cruzadas en `apps/`**: un componente de `apps/` no puede importar
   el paquete de otro app. El código compartido debe vivir en `packages/`
   (p. ej. `core-schemas`) y comunicarse mediante sus contratos de datos.

**Permisos (GitHub Actions token)**

| Permiso | Nivel | Motivo |
|---|---|---|
| `contents` | `read` | Hacer checkout del código del PR |
| `pull-requests` | `write` | Publicar comentarios inline y el resumen de la revisión |
| `checks` | `write` | Marcar el check como fallido cuando hay violaciones de arquitectura |

> El agente **no** tiene permiso de escritura sobre el código (`contents: read`): no
> puede aplicar cambios ni hacer merge; solo comenta y aprueba/bloquea el check. Usa el
> `GITHUB_TOKEN` efímero del workflow, sin secretos externos ni acceso a datos clínicos.

**Política de fallo**

- Violación de arquitectura → el check falla (`core.setFailed`) y bloquea el merge.
- Errores de Ruff/MyPy → se reportan como anotaciones pero **no** bloquean (informativo).
- Cualquier archivo no parseable se omite en el auditor (lo cazan Ruff/MyPy).

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-07-14 | 0.1.0 | Registro inicial del agente guardián de CI |

---

### Agentes de ingesta — `mesh-agent` · `cbct-agent` · `report-agent`

| Campo | Valor |
|---|---|
| **Ubicación** | `packages/ingestion-agents/` (`mesh_agent.py` · `cbct_agent.py` · `report_agent.py`) |
| **Versión** | `0.1.0` |
| **Estado** | `active` |
| **Fase del pipeline** | 1 · Ingesta (frontera raw → contrato) |
| **Contrato común** | `IngestionOutput` + `BaseIngestionAgent` en `ingestion_agents/base.py` |
| **Orquestador** | `apps/agent-orchestrator` (`IngestionPipeline`) |

> **Por qué viven en `packages/` y no en `apps/<modalidad>-agent/`:** el
> orquestador (un `app`) tiene que importarlos, y el `ai-code-reviewer` prohíbe
> las dependencias cruzadas entre `apps/`. La regla del monorepo obliga a que el
> código compartido esté en `packages/` — es la rama «o `packages/` si lo
> comparten varios» de la skill `add-ingestion-agent`.

**Rol / Propósito**

> Traducen **un** fichero clínico crudo a un fragmento del contrato de
> `core-schemas`, declarando su `Provenance`. Regla: **1 modalidad = 1 soporte =
> 1 agente**. Son los **únicos** componentes que tocan ficheros crudos: a partir
> del `TwinSnapshot` nadie vuelve al original.

| Agente | Entrada | `Modality` | `Support` | Produce | Cerebro |
|---|---|---|---|---|---|
| `mesh-agent` | OBJ / STL intraoral | `mesh` | superficial | `surface_ref` (posiciones float64 + caras + normales + color) | determinista |
| `cbct-agent` | directorio de serie DICOM | `cbct` | volumétrico | `gaussian_field_ref` (campo σ semilla) | determinista |
| `report-agent` | PDF / TXT / MD | `report` | regional | `list[RegionalObservation]` (pH por FDI) | determinista (`rules`) · LLM opcional (`llm`) |
| `image-agent` (PoC) | foto JPG / PNG / HEIC | `image` | superficial | `artifact_ref` (píxeles RGB, **sin EXIF**) | determinista |

**Herramientas y permisos** (código tipado, **no** MCP ni tool calling)

| Recurso | Permisos | Notas |
|---|---|---|
| Fichero crudo de la modalidad | read | Única lectura de datos crudos del sistema. |
| `ArtifactStore` (`data/interim/artifacts/`) | write | Blobs pesados por hash SHA-256 del contenido; nunca embebidos en Pydantic. |
| Directorio de cuarentena | write | Solo ruta + traceback del fallo; **nunca** el contenido clínico. |
| API de Anthropic | red (solo `report-agent` con `backend="llm"`, **desactivado por defecto**) | Requiere `ANTHROPIC_API_KEY`; sin ella el agente falla declarando, no lanza. |

**Outputs generados**

```
IngestionOutput
  ├─ ingestion : ModalityIngestion (ok/missing/failed) — SIEMPRE presente
  ├─ provenance: Provenance (source_file, modality, agent, confidence)
  ├─ artifact_ref / n_primitives   (mesh, cbct)
  ├─ regional  : list[RegionalObservation]  (report)
  └─ latency_s, quarantine_ref
```

**Reglas de delegación**

- No se delegan entre sí ni deciden nada: producen fragmentos de contrato. El
  `agent-orchestrator` los dispara **en paralelo** (las tres modalidades son
  independientes) y ensambla el `TwinSnapshot`.
- **Fail-loud, nunca fail-fast**: un fichero corrupto devuelve
  `status=FAILED` + `detail`; jamás propaga la excepción. Un fallo de una
  modalidad no puede llevarse por delante las otras dos.
- **Human-in-the-loop**: el agente **no** decide qué se persiste. Emite
  `Provenance.confidence` y el orquestador aplica el umbral
  (`DEFAULT_HITL_THRESHOLD = 0.7`); por debajo, el snapshot requiere revisión
  humana antes de persistirse.
- **Soberanía del dato**: el `cbct-agent` seudonimiza el `PatientID` del DICOM
  (HMAC-SHA256 con sal de `ASH_PSEUDONYM_SALT`); ningún identificador directo
  llega al contrato.

**Reglas específicas por modalidad**

- 🔒 `mesh-agent` — **guardarraíl de reversibilidad**: conserva la superficie de
  origen sin pérdida (posiciones `float64` + topología completa), no una nube
  remuestreada. Round-trip con error **cero** (test
  `test_round_trip_de_superficie_sin_perdida`). Acepta **OBJ** (color por vértice,
  Teeth3DS+) y **STL** (siempre pelado → `color=None`, Bite2Text); el gris uniforme
  del OBJ de Teeth3DS+ es un *placeholder* del exportador y también se trata como
  **ausencia** de color.
- `cbct-agent` — **envuelve** la reconstrucción tipo RGS, no reimplementa su
  algoritmo residual. Produce la semilla isótropa (σ normalizado, cuaternión
  identidad) que un optimizador refinaría.
- `report-agent` — valida cada valor contra la **ontología clínica mínima**
  (`ingestion_agents/ontology.py`) antes de escribirlo en el contrato.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-07-22 | 0.1.0 | Registro inicial: los tres agentes de ingesta pasan de `planned` a `active`. Contrato común `IngestionOutput`, almacén por contenido, cuarentena, seudonimización, ontología mínima, generador de casos sintéticos y orquestación de la fase 1. |

---

### Agentes de fusión — `geometric-fusion-agent` · `semantic-fusion-agent`

| Campo | Valor |
|---|---|
| **Ubicación** | `packages/fusion-agents/` (`geometric.py` · `semantic.py` · `registration.py` · `twin.py`) |
| **Versión** | `0.1.0` |
| **Estado** | `active` |
| **Fase del pipeline** | 2 · Fusión geométrica **y** 4 · Fusión semántica (separadas por la segmentación) |
| **Contrato común** | `FusionOutput` + `BaseFusionAgent` en `fusion_agents/base.py` |
| **Orquestador** | `apps/agent-orchestrator` (`IngestionPipeline.fuse()`) |
| **Decisiones** | [ADR 004 — Fusión](docs/architecture/004-fusion.md) |

> **Por qué son dos agentes y no uno:** entre ambos corre la **segmentación**, así
> que uno solo tendría que invocarse dos veces con banderas y guardar estado entre
> llamadas. Además tienen material y criterio de aceptación distintos: la semántica
> se valida contra el informe, la geométrica contra pares CBCT+IOS.

**Rol / Propósito**

> Enriquecen un `TwinSnapshot` **ya ensamblado**: no tocan ficheros crudos ni
> vuelven al original. La geométrica alinea dos medidas del mismo objeto físico y
> deja constancia **invertible** de la transformación; la semántica cuelga las
> observaciones del diente correcto y **marca lo que no cuadra** en vez de decidir.

| Agente | Entrada | Qué produce | No hace | Cerebro |
|---|---|---|---|---|
| `geometric-fusion-agent` | `TwinSnapshot` + dos nubes `(N,3)` en mm | `Provenance.transform` (`RigidTransform` invertible) + confianza del residuo + **color por gaussiana** desde la malla | no lee ni escribe `region_id`; no usa las fotos (error no-rígido) | determinista |
| `semantic-fusion-agent` | `TwinSnapshot` + `detected: FDI → confianza` | `RegionalObservation` ancladas con la confianza propagada | no toca geometría ni transforma nada | determinista |

**Herramientas y permisos** (código tipado, **no** MCP ni tool calling)

| Recurso | Permisos | Notas |
|---|---|---|
| `TwinSnapshot` (en memoria) | read | **Nunca** vuelven al fichero crudo: la ingesta es la única frontera con el dato original. |
| Directorio de cuarentena | write | Solo `acquisition_id` + traceback; **nunca** el contenido clínico (ni el pH). |

> No usan `ArtifactStore`: `detected` se pasa **explícitamente** en vez de leer el
> campo gaussiano del almacén. Son ~14 códigos FDI — cargar millones de primitivas
> para obtenerlos sería absurdo, y así los agentes son testeables sin almacén ni GPU.

**Outputs generados**

```
FusionOutput
  ├─ status      : ModalityStatus (ok/missing/failed) — SIEMPRE presente
  ├─ snapshot    : TwinSnapshot | None  — NUEVO, nunca el de entrada mutado
  ├─ hitl_reasons: list[str]            — vacío = no hace falta revisión
  └─ latency_s, quarantine_ref, detail
```

**Reglas de delegación**

- **Fail-loud, nunca fail-fast**: el fallo se devuelve como `status=FAILED` +
  `detail`; jamás propaga la excepción. El orquestador conserva el snapshot de la
  ingesta — que la fusión falle no destruye lo que la ingesta sí consiguió.
- **Human-in-the-loop**: el agente **no** decide qué se persiste. Emite
  `hitl_reasons` y el orquestador aplica el umbral (`DEFAULT_HITL_THRESHOLD = 0.7`).
- **Nunca mutan** el snapshot de entrada y **conservan el `acquisition_id`**: es la
  identidad de visita, y es lo que hace que reejecutar la fusión **reemplace** en vez
  de inflar el historial del paciente con visitas que no ocurrieron
  (`insert_snapshot`, ADR 004 §2.5).

**Reglas específicas por etapa**

- 🔒 `geometric-fusion-agent` — **guardarraíl de reversibilidad**: la transformación
  se guarda como `RigidTransform` (cuaternión + traslación), **no** como matriz 4×4.
  Una 4×4 puede codificar escala y cizalla; si un ICP devolviera una escala espuria,
  la reversibilidad se rompería en silencio. Esta forma la hace *imposible de
  expresar*, y un validador rechaza el cuaternión no unitario. La confianza sale del
  residuo, `clamp(1 − rms/ε, 0, 1)` con **ε = 0.5 mm** — que **no** es la métrica de
  0.1 mm del brief: esa mide reversibilidad de *una* malla, esta el alineamiento
  entre *dos* modalidades.
- `geometric-fusion-agent` — el algoritmo vive tras un `Protocol` (`Registrar`).
  Implementada solo la etapa **fina** (ICP multiescala); la **gruesa** (RANSAC-FPFH)
  queda pendiente, así que **converge solo si la pose inicial ya está cerca**.
  **Transfiere el color desde la malla** (ADR 004 §2.8): cada gaussiana dentro de la
  banda ε toma el de su vértice más cercano. Las **fotos quedan fuera** — el notebook 07
  midió que el error foto↔malla es **no-rígido** (ICP estancado en IoU ≈ 0,55), así que
  no es el mismo problema que el registro rígido. Con malla pelada o gris *placeholder*
  el resultado es **ausencia de color**, que es respuesta válida y no un bug.
- `semantic-fusion-agent` — la confianza es el **eslabón más débil**,
  `min(confianza_observación, confianza_FDI)`: anclar un pH a un diente no puede ser
  más fiable que saber qué diente es.
- ⚠️ `semantic-fusion-agent` — **ante un conflicto FDI no elige ganador**. Si el
  informe referencia un diente que la segmentación no encontró, conserva el FDI *del
  informe* (fuente clínica), pone la confianza a **0.0** y con eso cae al gate y va a
  revisión humana. El motivo está **medido**: el error dominante del modelo es el
  desplazamiento al diente vecino ([experimento Point
  Transformer](notebooks/exercise-point-transformer-teeth3ds.md)), así que ahí es la
  parte menos fiable — pero el informe tampoco es infalible. Resolverlo en silencio
  sería el fallo que el [ADR 003](docs/architecture/003-verification-fault-tolerance.md)
  señala como el peor: silencioso e irreversible, sobre un dato clínico.
- `semantic-fusion-agent` — la `Provenance` de cada observación **conserva la del
  informe** (`source_file`, `modality`, `agent`); solo se reescribe `confidence`. El
  valor sigue viniendo del PDF. Quién fusionó consta en la `Provenance` del snapshot.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-04 | 0.1.0 | Registro inicial. ADR 004; `RigidTransform` en `core-schemas` (contrato 1.3.0); fusión semántica completa; fusión geométrica con ICP multiescala (falta la etapa gruesa); inserción idempotente en la serie temporal; enganche en el orquestador. |

---

### `segmentation-agent` — Agente de segmentación anatómica

| Campo | Valor |
|---|---|
| **Ubicación** | `packages/analysis-agents/` (`segmentation.py` · `base.py`) |
| **Versión** | `0.1.0` |
| **Estado** | `active` |
| **Fase del pipeline** | 3 · Análisis · segmentación (**entre** las dos etapas de fusión) |
| **Contrato común** | `AnalysisOutput` + `BaseAnalysisAgent` en `analysis_agents/base.py` |
| **Orquestador** | `apps/agent-orchestrator` (`IngestionPipeline.fuse()`, etapa 2 de 3) |
| **Agregación punto → diente** | `packages/tooth-aggregation` |

**Rol / Propósito**

> Pone **nombre de diente** a cada gaussiana. Es el **ancla semántica** del
> pipeline: sin `region_id`, la fusión semántica no tiene contra qué validar el
> diente que cita el informe, y la capa regional (pH y demás) queda colgando de un
> código que nadie ha confirmado que exista en esa boca.

| Entrada | Qué produce | No hace | Cerebro |
|---|---|---|---|
| `TwinSnapshot` + el campo gaussiano que referencia | `region_id` (FDI) por gaussiana en un artefacto **nuevo** + `detected: FDI → confianza` + motivos de revisión | no corrige el informe ni al revés (eso lo declara la fusión semántica); no entrena ni ejecuta el modelo | modelo de segmentación de nubes de puntos, **inyectado** |

**Herramientas y permisos** (código tipado, **no** MCP ni tool calling)

| Recurso | Permisos | Notas |
|---|---|---|
| `TwinSnapshot` (en memoria) | read | Nunca vuelve al fichero crudo: la ingesta es la única frontera con el dato original. |
| Almacén de artefactos | read + write | Vía el `Protocol` `GaussianStore`, **no** vía `ArtifactStore`: el almacén es un *seam* que se sustituirá por `3dgs-engine`. |
| Modelo de segmentación | call | Vía el `Protocol` `Segmenter`. **Sin valor por defecto**: un modelo de juguete por omisión produciría etiquetas anatómicas inventadas con toda la pinta de ser buenas. |
| Directorio de cuarentena | write | Solo `acquisition_id` + traceback; **nunca** coordenadas ni contenido clínico. |

**Outputs generados**

```
SegmentationOutput
  ├─ status              : ModalityStatus (ok/missing/failed) — SIEMPRE presente
  ├─ snapshot            : TwinSnapshot | None  — NUEVO, con gaussian_field_ref etiquetado
  ├─ detected            : {FDI: confianza}     — entrada exacta del semantic-fusion-agent
  ├─ n_teeth             : int
  ├─ unassigned_fraction : float                — fragmentación descartada por tamaño
  ├─ hitl_reasons        : list[str]            — vacío = no hace falta revisión
  └─ latency_s, quarantine_ref, detail
```

**Reglas de delegación**

- **Fail-loud, nunca fail-fast**: el fallo se devuelve como `status=FAILED` +
  `detail`. Si la segmentación falla, el orquestador **no ancla** el informe contra
  un ancla que no existe: la fusión semántica sencillamente no corre.
- **Human-in-the-loop**: `detected` pasado a mano al orquestador **manda sobre el
  modelo** y la etapa ni se ejecuta. Es por donde entran las etiquetas revisadas por
  un clínico; sin esa puerta, el gate de revisión no serviría de nada.
- **Nunca muta** el snapshot de entrada y **conserva el `acquisition_id`**.

**Reglas específicas**

- 🔒 **La etiqueta es aditiva.** El artefacto nuevo lleva los mismos
  `centers`/`scales`/`rotations`/`density` **byte a byte** más un array `region_id`
  (`0` = sin asignar). El blob anterior sigue en el almacén, así que segmentar no
  puede degradar la geometría ni romper la reversibilidad. Como el almacén
  direcciona por contenido, volver a segmentar el mismo campo con el mismo modelo
  devuelve **la misma referencia**.
- ⚠️ **Se comprueba que el modelo devuelve log-probabilidades, no logits.** Un logit
  tiene la misma forma y el mismo `argmax`: las etiquetas saldrían bien y las
  **confianzas** serían falsas, sin que nada chille. Es el mismo modo de fallo caro
  que la modalidad DICOM en el `cbct-agent`, y se cierra igual — verificando la
  premisa (`∑ⱼ exp(logprob) = 1`) en vez de confiar en ella.
- **La confianza de un diente es la media geométrica de la probabilidad por punto**
  (`exp` de la log-probabilidad media de la instancia), en `[0, 1]` y comparable con
  el umbral de HITL y con el resto del pipeline.
- **La métrica honesta es el acierto por DIENTE, no por punto** — está medido que el
  acierto por punto *subestima* la identificación por diente ([experimento Point
  Transformer](notebooks/exercise-point-transformer-teeth3ds.md)).
- **`enforce_unique` viene desactivado a propósito.** Imponer «un FDI por arcada»
  sobre instancias fragmentadas *inventa* errores: la restricción presupone «una
  instancia = un diente», y con fragmentos esa premisa es falsa (medido).
- **Cuatro cosas van a revisión humana** en vez de resolverse solas: confianza bajo
  el umbral, el mismo FDI en dos instancias, una clase sin código FDI en el mapeo, y
  más de un 10% de los puntos de diente descartados por fragmentación — que la
  agregación se los coma **en silencio** es el riesgo, no que se los coma.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-06 | 0.1.0 | Registro inicial. Paquete `analysis-agents` con su contrato base; agregación punto → diente sobre `tooth-aggregation`; `region_id` persistido como artefacto aditivo; enganche en `IngestionPipeline.fuse()` entre las dos etapas de fusión. |

---

### Agentes de análisis (stubs `planned`)

Agentes del pipeline clínico **aún no implementados**. Se registran aquí como
*stubs* de diseño (roles y contratos previstos) para cerrar la Tarea 3 de la issue
de arquitectura multiagente. Su diseño de alto nivel vive en
[`docs/architecture/multi-agent-pipeline.md`](docs/architecture/multi-agent-pipeline.md).
Todos **consumen y enriquecen** un `TwinSnapshot` a través de `packages/core-schemas`
—nunca vuelven al fichero crudo— y dejan su propia `Provenance`.

| Agente | Estado | Fase | Rol previsto | Entrada → salida | Human-in-the-loop |
|---|---|---|---|---|---|
| `pathology-agent` | `planned` | Análisis · hallazgos clínicos | Señalar posibles patologías (densidad σ, color, geometría) como **hallazgos candidatos para revisión clínica**. | `TwinSnapshot` → `RegionalObservation` con hallazgos candidatos (no diagnóstico) | **Sí** — clínicamente sensible |
| `clinical-poc-agent` | `planned` (PoC) | Análisis · prueba de concepto | Métrica visual básica: inflamación por color de encía y espacio encía-diente. | `TwinSnapshot` → reporte de texto (log) | Sí |

> **Frontera de diseño:** estos stubs **no** tienen tools MCP, permisos ni reglas de
> delegación definitivos todavía; se detallarán al implementarlos, cada uno con su
> ficha completa (como `research-agent`) y, si toca, su ADR. Registrarlos ahora fija
> su **rol y contrato**, no su implementación.

> **Marco clínico y regulatorio (importante).** Los agentes de análisis con
> Human-in-the-loop (`pathology-agent`, `clinical-poc-agent`, y cualquier medida
> clínicamente sensible como el **fenotipo periodontal** encía↔hueso) producen
> **hallazgos y medidas candidatos para revisión del clínico**, con `Provenance`
> trazable — **no emiten diagnóstico** ni sustituyen la decisión clínica. Es un
> **uso investigacional / demostrador**, y mantenerlo así deja el sistema **fuera**
> de la categoría de producto sanitario (*Medical Device* / SaMD). En el momento en
> que una salida se declare "diagnóstico" o el clínico se apoye en ella para tratar,
> entra en el terreno regulado (UE **MDR** / **FDA**) — fuera del alcance de esta fase.

**Agentes de ingesta:** `cbct-agent`, `mesh-agent` y `report-agent` ya están
`active` — ficha completa en la [sección anterior](#agentes-de-ingesta--mesh-agent--cbct-agent--report-agent).
El `image-agent` (foto 2D) es la **4ª modalidad del `IngestionPipeline`**: ingiere
JPG/PNG/HEIC, **descarta el EXIF** (privacidad) y guarda los píxeles como artefacto.
Es la única modalidad **0..N** (una adquisición trae varias fotos, 5 en Bite2Text),
así que el orquestador ingiere cada foto y el `TwinSnapshot` las recoge en
**`image_refs: list[str]`** (apariencia pre-fusión, como el `surface_ref`). No
reconstruye 3D de una foto —eso es fusión— pero deja la apariencia lista y trazable.
Detalle del contrato de ingesta en el
[pipeline multiagente](docs/architecture/multi-agent-pipeline.md#2-tarea-1--contratos-de-ingesta).

---

### Agentes de desarrollo (dev-time)

Herramientas de IA externas que el equipo usa para asistir el desarrollo. **No forman parte del sistema en producción** ni tienen acceso autónomo al runtime: toda su salida entra al repositorio como código propuesto y pasa por Pull Request + revisión humana (y por el guardián `ai-code-reviewer`) antes de mergearse.

| Herramienta | Rol en el proyecto | Modelo | Notas de gobernanza |
|---|---|---|---|
| OpenCode / Claude Code | Asistentes de codificación interactivos: generación, refactorización, tests y documentación bajo dirección de una persona del equipo | Claude (Opus/Sonnet según sesión) | Conducidos por humano (no autónomos); sin acceso a datos clínicos; todo output vía PR + revisión humana. No se les delega decisiones clínicas ni de arquitectura. |

> Se documentan a nivel de fila (no con la ficha de agente) porque son asistentes **interactivos**, no agentes del sistema: no tienen contrato de datos, fase de pipeline ni reglas de delegación propias. Registra aquí cualquier otra herramienta de IA dev-time que se incorpore.
