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

### Agentes activos

| # | Nombre del agente | Rol / Propósito | Herramientas MCP | Reglas de delegación |
|---|---|---|---|---|
| 1 | research-agent | Agente de investigación autónomo que busca, ingerir y resume literatura científica (3DGS, DICOM, normativas) usando RAG | Ninguna (trabaja con filesystem local) | No requiere aprobación humana; puede delegar en el orquestador para escalar problemas de ingesta |

---

## Plantilla de registro

Copia el bloque siguiente para cada nuevo agente y rellena todos los campos.

---

### `<NombreDelAgente>`

| Campo | Valor |
|---|---|
| **Nombre** | `<nombre-tecnico-del-agente>` |
| **Versión** | `0.1.0` |
| **Ubicación** | `apps/agent-orchestrator/src/agents/<nombre>/` |
| **Estado** | `active` / `experimental` / `deprecated` |

**Rol / Propósito**

> Descripción de una o dos frases explicando qué hace el agente, qué problema resuelve y en qué fase del pipeline actúa (ingesta / fusión / análisis / exportación).

**Herramientas MCP a las que tiene acceso**

| Herramienta MCP | Servidor | Permisos | Notas |
|---|---|---|---|
| `<nombre-herramienta>` | `slicer-mcp-server` / `<otro>` | read / write / execute | *(descripción breve)* |

**Inputs esperados**

```
# Esquema de entrada (referencia a core-schemas o descripción inline)
```

**Outputs generados**

```
# Esquema de salida (referencia a core-schemas o descripción inline)
```

**Reglas de delegación**

- El agente puede delegar en: `<NombreDeOtroAgente>` cuando `<condición>`.
- Requiere aprobación humana (human-in-the-loop) para: `<lista de acciones sensibles>`.
- No puede delegar: `<lista de responsabilidades no delegables>`.
- Política de reintentos: `<n>` reintentos antes de escalar al orquestador.
- Política de fallo: `<comportamiento en caso de error no recuperable>`.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| YYYY-MM-DD | 0.1.0 | Registro inicial del agente |

---

## Agentes de desarrollo (dev-time)

Agentes de IA externos utilizados para asistir el desarrollo del proyecto. No forman parte del sistema en producción.

| Herramienta | Rol en el proyecto | Notas |
|---|---|---|
| OpenCode / Claude Code | Codificación, refactorización, generación de tests | Integrado en el flujo de trabajo diario |
| AI Code Reviewer (guardián de CI) | Revisión estática automática de cada PR | Ver ficha detallada abajo |

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
