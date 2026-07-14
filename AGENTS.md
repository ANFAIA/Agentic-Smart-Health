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

### Agentes de desarrollo (dev-time)

Herramientas de IA externas que el equipo usa para asistir el desarrollo. **No forman parte del sistema en producción** ni tienen acceso autónomo al runtime: toda su salida entra al repositorio como código propuesto y pasa por Pull Request + revisión humana (y por el guardián `ai-code-reviewer`) antes de mergearse.

| Herramienta | Rol en el proyecto | Modelo | Notas de gobernanza |
|---|---|---|---|
| OpenCode / Claude Code | Asistentes de codificación interactivos: generación, refactorización, tests y documentación bajo dirección de una persona del equipo | Claude (Opus/Sonnet según sesión) | Conducidos por humano (no autónomos); sin acceso a datos clínicos; todo output vía PR + revisión humana. No se les delega decisiones clínicas ni de arquitectura. |

> Se documentan a nivel de fila (no con la ficha de agente) porque son asistentes **interactivos**, no agentes del sistema: no tienen contrato de datos, fase de pipeline ni reglas de delegación propias. Registra aquí cualquier otra herramienta de IA dev-time que se incorpore.
