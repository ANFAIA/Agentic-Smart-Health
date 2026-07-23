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
| `mesh-agent` | OBJ intraoral | `mesh` | superficial | `surface_ref` (posiciones float64 + caras + normales + color) | determinista |
| `cbct-agent` | directorio de serie DICOM | `cbct` | volumétrico | `gaussian_field_ref` (campo σ semilla) | determinista |
| `report-agent` | PDF / TXT / MD | `report` | regional | `list[RegionalObservation]` (pH por FDI) | determinista (`rules`) · LLM opcional (`llm`) |

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
  `test_round_trip_de_superficie_sin_perdida`). El gris uniforme de Teeth3DS+ es
  un *placeholder* del exportador: se trata como **ausencia** de color.
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

### Agentes de análisis (stubs `planned`)

Agentes del pipeline clínico **aún no implementados**. Se registran aquí como
*stubs* de diseño (roles y contratos previstos) para cerrar la Tarea 3 de la issue
de arquitectura multiagente. Su diseño de alto nivel vive en
[`docs/architecture/multi-agent-pipeline.md`](docs/architecture/multi-agent-pipeline.md).
Todos **consumen y enriquecen** un `TwinSnapshot` a través de `packages/core-schemas`
—nunca vuelven al fichero crudo— y dejan su propia `Provenance`.

| Agente | Estado | Fase | Rol previsto | Entrada → salida | Human-in-the-loop |
|---|---|---|---|---|---|
| `segmentation-agent` | `planned` | Análisis · segmentación | Asignar `region_id` (FDI) a las gaussianas (segmentación anatómica). Prerrequisito del ancla semántica de la fusión. | `TwinSnapshot` sin etiquetas → snapshot con `region_id` poblado | Revisión si afecta al diagnóstico |
| `pathology-agent` | `planned` | Análisis · diagnóstico | Detectar patologías a partir de densidad (σ), color y geometría. | `TwinSnapshot` → `RegionalObservation` con hallazgos | **Sí** — decisión clínicamente sensible |
| `clinical-poc-agent` | `planned` (PoC) | Análisis · prueba de concepto | Métrica visual básica: inflamación por color de encía y espacio encía-diente. | `TwinSnapshot` → reporte de texto (log) | Sí |

> **Frontera de diseño:** estos stubs **no** tienen tools MCP, permisos ni reglas de
> delegación definitivos todavía; se detallarán al implementarlos, cada uno con su
> ficha completa (como `research-agent`) y, si toca, su ADR. Registrarlos ahora fija
> su **rol y contrato**, no su implementación.

**Agentes de ingesta:** `cbct-agent`, `mesh-agent` y `report-agent` ya están
`active` — ficha completa en la [sección anterior](#agentes-de-ingesta--mesh-agent--cbct-agent--report-agent).
Sigue `planned` el `image-agent` (foto 2D → previsualización 3D, PoC), fuera del
hito de Semana 3-4. Detalle del contrato de ingesta en el
[pipeline multiagente](docs/architecture/multi-agent-pipeline.md#2-tarea-1--contratos-de-ingesta).

---

### Agentes de desarrollo (dev-time)

Herramientas de IA externas que el equipo usa para asistir el desarrollo. **No forman parte del sistema en producción** ni tienen acceso autónomo al runtime: toda su salida entra al repositorio como código propuesto y pasa por Pull Request + revisión humana (y por el guardián `ai-code-reviewer`) antes de mergearse.

| Herramienta | Rol en el proyecto | Modelo | Notas de gobernanza |
|---|---|---|---|
| OpenCode / Claude Code | Asistentes de codificación interactivos: generación, refactorización, tests y documentación bajo dirección de una persona del equipo | Claude (Opus/Sonnet según sesión) | Conducidos por humano (no autónomos); sin acceso a datos clínicos; todo output vía PR + revisión humana. No se les delega decisiones clínicas ni de arquitectura. |

> Se documentan a nivel de fila (no con la ficha de agente) porque son asistentes **interactivos**, no agentes del sistema: no tienen contrato de datos, fase de pipeline ni reglas de delegación propias. Registra aquí cualquier otra herramienta de IA dev-time que se incorpore.
