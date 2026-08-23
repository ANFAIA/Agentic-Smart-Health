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
| `composite-export-agent` | [`packages/export-agents/src/export_agents/compuesto.py`](packages/export-agents/src/export_agents/compuesto.py) |
| `export-agent` | [`packages/export-agents/src/export_agents/stl.py`](packages/export-agents/src/export_agents/stl.py) |
| `field-export-agent` | [`packages/export-agents/src/export_agents/field.py`](packages/export-agents/src/export_agents/field.py) |
| `geometric-fusion-agent` | [`packages/fusion-agents/src/fusion_agents/geometric.py`](packages/fusion-agents/src/fusion_agents/geometric.py) |
| `image-agent` | [`packages/ingestion-agents/src/ingestion_agents/image_agent.py`](packages/ingestion-agents/src/ingestion_agents/image_agent.py) |
| `mesh-agent` | [`packages/ingestion-agents/src/ingestion_agents/mesh_agent.py`](packages/ingestion-agents/src/ingestion_agents/mesh_agent.py) |
| `render-export-agent` | [`packages/export-agents/src/export_agents/render.py`](packages/export-agents/src/export_agents/render.py) |
| `report-agent` | [`packages/ingestion-agents/src/ingestion_agents/report_agent.py`](packages/ingestion-agents/src/ingestion_agents/report_agent.py) |
| `segmentation-agent` | [`packages/analysis-agents/src/analysis_agents/segmentation.py`](packages/analysis-agents/src/analysis_agents/segmentation.py) |
| `semantic-fusion-agent` | [`packages/fusion-agents/src/fusion_agents/semantic.py`](packages/fusion-agents/src/fusion_agents/semantic.py) |
| `uos-export-agent` | [`packages/uos/src/uos/agente.py`](packages/uos/src/uos/agente.py) |
| `viewer-export-agent` | [`packages/export-agents/src/export_agents/visor.py`](packages/export-agents/src/export_agents/visor.py) |
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
| **Estilo / lint** | `ruff check` | **Sí** | Anotaciones inline nativas de GitHub |
| **Tipos** | `mypy` (solo `*/src/`) | **Sí** | Anotaciones inline (`::error`/`::warning`) |
| **Arquitectura** | `scripts/audit_pr.py` | **Sí** | Comentario de revisión en la línea afectada + resumen |

> **`ruff format` no se comprueba**, y es una decisión, no un olvido. El formato de este
> repositorio es deliberado —tuplas agrupadas por significado, comentarios alineados con
> el dato que explican— y `ruff format` lo deshace: 46 de 87 ficheros cambiarían, y
> `_PROPIEDADES` pasaría de tres líneas agrupadas (posición · escala · rotación) a once
> sueltas. Un check permanentemente rojo que nadie puede arreglar enseña a ignorar los
> checks. Lo mecanizable del formato —longitud de línea, orden de imports— ya lo cubre
> `ruff check`.

> **MyPy mira solo los `src/`**, que es el código que se ejecuta en producción y que hoy
> está limpio. Lo que queda fuera, medido antes de decidirlo: `scripts/` (17 errores;
> exploratorio, carga módulos por `importlib` y vive de arrays sin tipar), los tests (49;
> casi todos un `TwinSnapshot | None` que el propio test ya afirma que no es None, y cuya
> puerta es pytest), y `packages/3dgs-engine/`, cuyo guion no es un identificador válido
> de Python y hace abortar a MyPy antes de mirar nada. Meter cualquiera de los tres
> cerraría la puerta en rojo desde el primer día, que es la forma más rápida de que una
> puerta acabe desactivada.

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
- Errores de Ruff/MyPy → el paso falla y bloquea el merge, además de anotar la línea.
- **Cada puerta da su veredicto aunque otra se cierre**: los pasos posteriores llevan
  `!cancelled()`, así que un error de estilo no deja al PR sin pasar por el guardián de
  datos. El job falla igual; lo que no se pierde es el diagnóstico completo.
- El resumen publica el resultado **real** de cada puerta (`limpio` / `bloquea` / `no
  aplica`), no un «ejecutado» que decía lo mismo con el linter limpio y con veinte errores.
- Cualquier archivo no parseable se omite en el auditor (lo cazan Ruff/MyPy).

**Versiones clavadas.** `ruff` y `mypy` van con `==` en `pyproject.toml`, no con `>=`.
Un linter que se actualiza solo estrena reglas sobre código que nadie ha tocado y pone
el CI rojo por un `uv sync` en una rama que no cambió nada — pasó con `ruff>=0.9`, que
acabó instalando la 0.15. Subir de versión es un commit deliberado que arregla lo que la
versión nueva encuentre.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-07-14 | 0.1.0 | Registro inicial del agente guardián de CI |
| 2026-08-17 | 0.1.0 | Ruff y MyPy pasan de informativos a **puerta de merge**; versiones clavadas; `ruff format` retirado; `scripts/` fuera de MyPy; las puertas posteriores ya no se saltan cuando una falla. Deuda saldada para poder cerrarlas: 27 errores de Ruff y 9 de MyPy. |

---

### `data-guardian` — Guardián de datos y licencias

| Campo | Valor |
|---|---|
| **Nombre** | `data-guardian` |
| **Versión** | `0.1.0` |
| **Ubicación** | [`scripts/data_guard.py`](scripts/data_guard.py) |
| **Estado** | `active` |
| **Disparo** | Hook `pre-commit` (**bloquea**) · `ai-code-review.yml` · `literature-watch.yml` |
| **Tecnología** | Código tipado, **sin LLM** |

**Rol / Propósito**

> Impide que un dato ajeno —de terceros, de licencia incompatible o clínico— entre en
> la historia de git. Es el único guardián que **detiene el commit**, y la asimetría es
> deliberada: un fichero que ya está en la historia no se quita sin reescribirla.
> Ocurrió (issue 45: 156 MiB de PDF, `filter-repo` y `force-push`). Aquí es más barato
> que en ningún otro sitio.

**Qué comprueba** (todas nacidas de un fallo real)

| Comprobación | Qué caza |
|---|---|
| `extensiones` | `.pdf`, mallas (`.ply/.stl/.obj/.glb/.gltf`), volúmenes (`.dcm/.nii/.nrrd`…), pesos de ML |
| `tamano` | cualquier fichero versionado por encima de 5 MiB |
| `ignore` | pregunta a `git check-ignore` por **rutas sonda**: verifica el comportamiento, no el texto del `.gitignore`, así sobrevive a que alguien las reordene y detecta que las ha borrado |
| `manifiesto` | que las entradas de `manifest.yaml` estén completas y parseen |
| `procedencia` | un notebook versionado con imágenes embebidas debe citar su ficha de dataset |
| `fichas` | que cada dataset usado tenga la suya |

> Los identificadores no son decorativos: salen del registro `COMPROBACIONES` del
> script, y el `docs-guardian` compara esta tabla con él. Añadir una comprobación y no
> documentarla aquí —o quitarla y dejarla anunciada— falla en el CI.

**Reglas de delegación**

- **Bloquea, no avisa**: sale con código ≠ 0 y el commit no ocurre. Para saltárselo hay
  que escribir `--no-verify` a mano, que deja rastro de intención.
- No decide sobre licencias dudosas: veta por extensión y tamaño, que son criterios
  mecánicos. El juicio sobre si una licencia permite redistribuir es humano.

> ⚠️ **Punto ciego conocido (2026-08-09, sin cerrar y AHORA MÁS ANCHO).** Veta `.stl`,
> `.ply` y `.obj`, pero un `.html` con esa misma geometría embebida en base64 le pasa por
> delante: la única barrera es el límite de tamaño, que no es el criterio correcto. Se
> detectó al construir un visor con geometría real de paciente. El repositorio es público.
>
> Subir el límite de 2 a 5 MiB el 2026-08-18 **ensancha este hueco**: un visor de 4 MiB
> que antes se paraba, ahora pasa. Mitigado a medias porque `notebooks/visores/` está
> ignorado, pero eso protege una ruta concreta, no el patrón. Cerrarlo de verdad pide
> mirar **dentro** del HTML —buscar bloques base64 con firma de malla— y es trabajo
> pendiente.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-05 | 0.1.0 | Registro inicial: seis comprobaciones, rutas sonda para las reglas de ignore, enganche en el hook y en los dos workflows. |
| 2026-08-11 | 0.1.1 | Sin cambio de comportamiento: las comprobaciones pasan a declararse en un registro con identificador estable, y esta ficha a reproducirlo. |

---

### `docs-guardian` — Guardián de coherencia documentación ↔ código

| Campo | Valor |
|---|---|
| **Nombre** | `docs-guardian` |
| **Versión** | `0.3.0` |
| **Ubicación** | [`scripts/docs_sync.py`](scripts/docs_sync.py) |
| **Estado** | `active` |
| **Disparo** | Hook `pre-commit` (**no bloquea**) · `ai-code-review.yml` · `literature-watch.yml` |
| **Tecnología** | Código tipado + `ast`, **sin LLM** |

**Rol / Propósito**

> La documentación se desincroniza en silencio: nadie recibe un error rojo por escribir
> un número que dejó de ser cierto. Este agente lo convierte en un fallo visible.
> Casos reales de este repositorio: un recuento de tests que decía 166 cuando eran 265,
> un `.env.example` con cinco variables que no leía nadie, y el árbol del README
> anunciando un notebook ya eliminado.

**Comprueba, no redacta** — es su regla de diseño y su frontera

> Solo **genera** lo que es copia mecánica de una fuente de verdad (las tablas entre
> marcas `<!-- generado: … -->`). Todo lo demás lo **verifica**. Un agente que
> "arreglase" la documentación por su cuenta haría que el documento se adapte al código
> **incluso cuando el que está mal es el código**: el 166 se habría convertido en 265
> sin que nadie se enterase de que llevaba meses mintiendo.

| Comprobación | Qué caza |
|---|---|
| `env` | variables leídas por el código ↔ `.env.example` ↔ README |
| `rutas` | ficheros citados en la documentación ↔ ficheros versionados |
| `agentes` | atributo `name` de las clases ↔ registro de `AGENTS.md` |
| `versiones` | atributo `version` de la clase ↔ la fila **Versión** de su ficha |
| `guardianes` | scripts que ejecutan los workflows o los hooks ↔ ficha en `AGENTS.md` |
| `comprobaciones` | el registro `COMPROBACIONES` de cada guardián ↔ la tabla de su ficha (esta misma) |
| `constantes` | un número o una cadena escritos en la prosa ↔ el valor de la constante de la que salen |
| `inventario` y `arbol` | que lo que existe esté citado, no solo que lo citado exista |
| `vacios` | un componente con ficha en el README ↔ que tenga código, o que la ficha lo declare placeholder |
| `bloques` | que las tablas generadas coincidan con el código |

> **Por qué los guardianes no se versionan.** Se planteó darles `__version__` y
> comparar, como se hace con las clases `*Agent`. No se hizo, y por tres motivos. El
> `version` de un agente **se usa**: viaja en `qualified` dentro de `provenance`, así
> que una fusión guardada dice con qué fórmula se calculó. En un script no lo leería
> nadie, y la comprobación quedaría comparando dos constantes que nadie tiene motivo
> para tocar: verde permanente, peor que no comprobar. Además obligaría a cada
> guardián a tener ficha propia con fila de versión, cuando `audit_pr.py` vive
> —correctamente— dentro de la del `ai-code-reviewer`. Y SemVer sin nadie que importe
> el módulo no significa nada. Lo que sí cambia y le importa a quien lo lea es **qué
> comprueba**, y eso es lo que se compara.

**Reglas de delegación**

- **No bloquea el commit.** Un hook que impide trabajar por un problema de
  documentación acaba desinstalado; si falla, avisa y el CI lo dice en la PR.
- **Fuente de verdad = `git ls-files`**, no el disco: si no, un fichero ignorado que
  existe en local haría pasar en tu máquina algo que en CI falla.

**Cómo se ata un número a su constante.** `constantes` no adivina: hace falta un
marcador, en un comentario HTML que no se ve al renderizar.

```markdown
...cuatro órdenes de magnitud bajo el presupuesto de **0,1 mm** <!--const:REVERSIBILITY_BUDGET_MM-->
```

A partir de ahí, si alguien cambia `REVERSIBILITY_BUDGET_MM` en el código y no toca esta
frase, el CI lo dice con el fichero y la línea. Hoy hay 13 números marcados.

> **Por qué un marcador y no buscar el nombre de la constante en el texto.** Porque la
> prosa peligrosa **no la nombra**: dice «ε = 0,5 mm» o «< 0,1 mm» a secas. De los 25
> sitios del repositorio donde hay un número que sale del código, solo 3 citaban la
> constante — una comprobación por nombre habría vigilado el 12 % y dado la sensación de
> cubrirlo todo. El marcador invierte la carga: quien escribe el número declara de dónde
> sale, una vez, y el CI lo vigila para siempre.

> ⚠️ **Lo que sigue sin cubrir.** Una afirmación en prosa **sin número** que se vuelve
> falsa. La ficha de fusión decía «la etapa gruesa queda pendiente» mucho después de que
> dejara de estarlo, y eso no hay constante que lo respalde. La otra mitad de aquel caso
> —«ε = 0,5 mm»— sí está cubierta desde la 0.4.0.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-04 | 0.1.0 | Registro inicial: env, rutas, agentes, inventario, árbol y bloques generados. |
| 2026-08-10 | 0.2.0 | Comprobación de **versiones** (la ficha declara la que declara la clase) y de **guardianes** (un script que corre solo necesita ficha) — esta misma. |
| 2026-08-11 | 0.3.0 | Comprobación de **comprobaciones**: el registro `COMPROBACIONES` de cada guardián contra la tabla de su ficha. |
| 2026-08-17 | 0.4.0 | Comprobación de **constantes**: los números citados en la documentación contra el valor real en el código, atados con un marcador `<!--const:NOMBRE-->`. Era la extensión que esta misma ficha declaraba pendiente. |
| 2026-08-18 | 0.5.0 | `constantes` admite **cadenas**, no solo números. Lo pidió `SCHEMA_VERSION`: el README anunciaba el esquema `1.2.0` con el contrato ya en `1.3.0`, y la comprobación pasaba en verde porque «1.2.0» no es un número. |

---

### `literature-watcher` — Vigilante de literatura científica

| Campo | Valor |
|---|---|
| **Nombre** | `literature-watcher` |
| **Versión** | `0.2.0` |
| **Ubicación** | [`scripts/watch_literature.py`](scripts/watch_literature.py) + [`.github/workflows/literature-watch.yml`](.github/workflows/literature-watch.yml) |
| **Estado** | `active` |
| **Disparo** | `cron` lunes 06:00 UTC · `workflow_dispatch` |
| **Tecnología** | Código tipado contra la API y el OAI-PMH de arXiv, **sin LLM** |

**Rol / Propósito**

> El `research-agent` ya sabe buscar literatura, pero es un REPL: solo descubre
> mientras alguien está sentado delante. Este agente hace la parte que se repite —mirar
> qué ha salido, descartar lo ya inventariado y averiguar bajo qué licencia se
> publicó— y deja el juicio *(¿es relevante?)* donde debe estar: en una persona
> revisando una PR.

**Qué hace, y sus fronteras**

- **Siete consultas** con puerta temática por consulta: cuatro dentales (3DGS, CBCT +
  segmentación, escáner intraoral, gemelo digital) y tres de estándares (DICOM,
  FHIR/HL7, interoperabilidad) — estas últimas filtradas **solo por título**, porque
  con el resumen se colaban falsos positivos.
- **Cupo por turnos**, no «los N más nuevos»: las consultas de estándares dan mucho más
  volumen, y ordenar por fecha dejaba PRs sin un solo artículo dental.
- **Ningún PDF toca el disco ni el repositorio.** Se descarga a memoria para calcular
  `sha256` y tamaño, y se libera. Descargar no es redistribuir.
- **La licencia se verifica en origen** (OAI-PMH, `verb=GetRecord`), no se supone. Una
  licencia adivinada por el título es peor que ninguna, porque parece un dato.

**Reglas de delegación**

- **No mergea.** Su salida es una rama y una PR; la decisión es humana, y esa PR pasa
  por los mismos guardianes que cualquier otra.
- **«No hay nada» y «no contestó nadie» son desenlaces distintos**, con código de
  salida distinto (`0` y `2`). Un vigilante que existe para que nadie tenga que
  acordarse de mirar no puede avisar de que está roto callándose.
- Reintenta solo lo transitorio (429, 5xx, cortes de red): un 400 por consulta mal
  formada no se cura esperando, y darle tres pasadas solo retrasa el diagnóstico.
- Si la organización prohíbe que Actions abra PRs, **no falla**: deja la rama subida y
  el cuerpo de la PR en el resumen del run, listo para pegar.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-05 | 0.1.0 | Registro inicial: consultas con puerta temática, verificación de licencia en origen, cupo por turnos y apertura de PR. |
| 2026-08-10 | 0.2.0 | Primer disparo real del cron. Reintentos con espera creciente, distinción entre «sin novedades» y «ninguna consulta respondió», declaración de fallos parciales en la PR, y tolerancia a que la organización bloquee la PR automática. |

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
| `report-agent` | PDF / TXT / MD | `report` | regional | `list[RegionalObservation]` (pH, anatomía radicular y hallazgos por FDI) + `list[Medida]` | determinista (`rules`) · LLM opcional (`llm`), **mismos campos** |
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
| **Versión** | `geometric-fusion-agent` **0.2.0** · `semantic-fusion-agent` `0.1.0` |
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
  expresar*, y un validador rechaza el cuaternión no unitario.
- ⚠️ `geometric-fusion-agent` — **la confianza sale del residuo de la población
  solapada**, no del de la nube completa: `clamp(1 − rms_solapado/ε, 0, 1)`. Medido
  sobre un paciente real con CBCT y escáner ([`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py)):
  **4,98 mm** sobre la nube entera frente a **0,452 mm** sobre los puntos que sí tienen
  contrapartida, para el mismo registro. El primero no mide el registro, mide qué
  fracción del escaneo es paladar. Y por debajo de `min_overlap` (20 %) la confianza
  es **0**: con cuatro puntos emparejados siempre hay una pose que los acerca.
- ⚠️ `geometric-fusion-agent` — **ε es por par de modalidades**, no una constante.
  `clamp(1 − rms/ε) ≥ 0,7` equivale a `rms ≤ 0,3·ε`, o sea 0,15 mm con ε = 0,5 — por <!--const:DEFAULT_HITL_THRESHOLD--> <!--const:DEFAULT_EPSILON_MM-->
  debajo del suelo físico de un CBCT de vóxel 0,30 mm y PSF 425 µm, así que con ese
  valor la fusión intraoral↔CBCT **no podría pasar el gate nunca**. ε se lee como *el
  error a partir del cual el resultado deja de servir*: **0,5 mm** para una malla <!--const:DEFAULT_EPSILON_MM-->
  derivada del propio volumen, **1,5 mm** (`EPSILON_IOS_CBCT_MM`) para intraoral↔CBCT.
  Ninguno de los dos es la métrica de 0,1 mm del brief: esa mide reversibilidad de <!--const:REVERSIBILITY_BUDGET_MM-->
  *una* malla, estas el alineamiento entre *dos* modalidades.
- `geometric-fusion-agent` — el algoritmo vive tras un `Protocol` (`Registrar`), con
  dos implementaciones: `icp` (etapa **fina**, con recorte opcional de atípicos) e
  `icp_global` (etapa **gruesa**, barrido de SO(3)). La gruesa cierra el hueco que el
  ADR 004 dejaba para RANSAC-FPFH, por fuerza bruta y sin arrastrar Open3D. Sin ella
  el ICP fino **no revienta**: converge a un mínimo local de ~0,43 mm —una cifra que
  parece un buen registro— con una pose equivocada. Es un fallo que hay que ir a
  buscar. **Recortar y buscar se pelean**: un recorte agresivo hace que una pose mala
  puntúe bien, así que se criba flojo y se refina fuerte.
- `geometric-fusion-agent` —
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
| 2026-08-10 | 0.2.0 (geométrica) | Medido el registro contra un paciente real con CBCT + escáner ([`scripts/registro_ios_cbct.py`](scripts/registro_ios_cbct.py)). La confianza pasa a salir de la **población solapada**, ε pasa a ser **por par de modalidades** (0,5 mm derivada de volumen · 1,5 mm intraoral↔CBCT), se añade la puerta por **solapamiento mínimo**, y se implementa la **etapa gruesa** (`icp_global`) que el ADR 004 dejaba pendiente. Aditivo: un registrador que no mide solapamiento se comporta igual que antes. |

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

### `export-agent` — Agente de exportación (regeneración de la malla)

| Campo | Valor |
|---|---|
| **Ubicación** | `packages/export-agents/` (`stl.py` · `base.py`) |
| **Versión** | `0.1.0` |
| **Estado** | `active` |
| **Fase del pipeline** | 6 · Exportación (frontera contrato → fichero) |
| **Contrato común** | `ExportOutput` + `BaseExportAgent` en `export_agents/base.py` |
| **Orquestador** | `IngestionPipeline.exportar(result, destino)` dispara los tres canales |
| **Decisiones** | [Pipeline §5](docs/architecture/multi-agent-pipeline.md#5-tarea-4--formato-y-pipeline-de-exportación) · [ADR 001](docs/architecture/001-digital-twin-core-schemas.md) (refs *fail-loud*) · [ADR 004 §2.2](docs/architecture/004-fusion.md) (`transform` invertible) |

> **Por qué es una cuarta familia y no una función suelta.** Es la **única** que
> escribe ficheros de salida, igual que la ingesta es la única que lee ficheros de
> entrada. Y su salida no encaja en las otras: `FusionOutput` y `AnalysisOutput`
> devuelven un `TwinSnapshot` enriquecido, y un exportador **no enriquece nada** —
> devuelve un fichero y la medida de cuánto se parece a lo que entró.

**Rol / Propósito**

> Cierra el círculo de la **reversibilidad**: la ingesta convirtió la malla del
> escáner en contrato y este agente la devuelve a un fichero, con el **error de
> reconstrucción medido** en vez de prometido. Es la fase que convierte la métrica
> del brief («error de malla < 0,1 mm») en algo que se comprueba en cada ejecución.

| Agente | Entrada | Qué produce | No hace | Cerebro |
|---|---|---|---|---|
| `export-agent` | `TwinSnapshot` + ruta de salida | **STL binario** desde `surface_ref` + `max_deviation_mm` y `mean_deviation_mm` medidas releyendo el fichero | no malla el campo gaussiano; no modifica el twin; no escribe en el almacén | determinista |
| `field-export-agent` | `TwinSnapshot` + ruta de salida | **PLY binario** desde `gaussian_field_ref`, en el marco del twin o en mm del CBCT | no inventa color ni opacidad; no convierte a formato de splat de INRIA | determinista |
| `render-export-agent` | `TwinSnapshot` + directorio | **PNG multivista** por Beer-Lambert + `psnr_db` / `ssim` del ciclo | no rasteriza splats; no reproduce las fotos intraorales | determinista |

> El **canal de metadatos** (el `TwinSnapshot` a JSON) no tiene agente a propósito:
> es `model_dump()` de Pydantic. Envolver una llamada de una línea en un contrato de
> fallos que no puede fallar sería ceremonia, no diseño.

**Herramientas y permisos** (código tipado, **no** MCP ni tool calling)

| Recurso | Permisos | Notas |
|---|---|---|
| `TwinSnapshot` (en memoria) | read | Nunca vuelve al fichero crudo. |
| Almacén de artefactos | **read** | Vía el `Protocol` `SurfaceStore`, que **solo declara `load`**: que un exportador no escriba en el almacén no es una promesa en prosa, es algo que el tipo no permite expresar. |
| Fichero de salida (STL) | write | Escritura **atómica** (temporal + `replace`): un STL a medio escribir no debe quedar donde alguien lo confunda con el bueno. |
| Directorio de cuarentena | write | Solo `acquisition_id` + traceback; **nunca** geometría ni contenido clínico. |

**Outputs generados**

```
ExportOutput
  ├─ status          : ModalityStatus (ok/missing/failed) — SIEMPRE presente
  ├─ path            : Path | None      — fichero escrito
  ├─ format · frame  : 'stl' · 'source' | 'twin'
  ├─ n_vertices · n_faces
  ├─ max_deviation_mm: float | None     — MEDIDA, releyendo el fichero (métrica del brief)
  ├─ hitl_reasons    : list[str]        — vacío = no hace falta revisión
  └─ latency_s, quarantine_ref, detail
```

**Reglas de delegación**

- **Fail-loud, nunca fail-fast**: igual que las otras tres familias, no lanza. Una
  referencia colgante, una cara fuera de rango o un destino imposible se devuelven
  como `status=FAILED` + `detail` y van a cuarentena.
- **Human-in-the-loop**: el agente **no** decide si el fichero se entrega. Emite
  `hitl_reasons` (`DEFAULT_HITL_THRESHOLD = 0.7`) y quien llama decide.
- **Solo lectura sobre el gemelo**: no muta el snapshot ni reescribe artefactos. Si
  un export pudiera reescribir el blob que exporta, la copia fiel de la superficie
  —el guardarraíl del `mesh-agent`— dejaría de ser fiel al primer round-trip.

**Reglas específicas**

- 🔒 **La geometría sale de `surface_ref`, nunca del campo gaussiano.** El
  `mesh-agent` guarda la superficie de origen tal cual (`float64` + topología
  completa) para que exista una copia fiel; el único error del round-trip es el que
  impone el **formato** (`float32`). Medido sobre un escaneo real de Teeth3DS+
  (110.804 vértices, arcada de 86 mm): **3,8·10⁻⁶ mm**, cuatro órdenes de magnitud
  bajo el presupuesto, con la exportación en 0,07 s. Sacar la malla del volumen por *marching
  cubes* sería **otra cosa**: está medido
  ([`scripts/resolucion_modalidades.py`](scripts/resolucion_modalidades.py)) que la
  isosuperficie solo está bien definida donde el gradiente es fuerte —esmalte, 364
  HU/vóxel— y que sobre hueso trabecular el área **no existe** como magnitud. Un
  snapshot sin malla se declara `MISSING`; no se «rescata» interpolando.
- ⚠️ **La desviación se mide releyendo el fichero, no estimando.** Una estimación
  del error de `float32` sobrevive intacta a un bug de endianness o de orden de
  vértices; una relectura, no. Y sin medida `within_budget` es `False`: un
  exportador que no verifica no puede afirmar que cumple los 0,1 mm.
- **Dos sistemas de referencia, explícitos.** `frame="source"` (por defecto) escribe
  la malla como entró —es el que mide la reversibilidad—; `frame="twin"` aplica la
  `RigidTransform` que registró la fusión geométrica para superponer con el CBCT.
  Pedir `twin` sobre un snapshot que nunca pasó por fusión **falla declarando**: la
  alternativa sería entregar la malla en el sistema del escáner haciéndola pasar por
  la del CBCT.
- **Un snapshot parcial no llega callado a exportación** (ADR 001). El exportador es
  el último punto donde eso se puede decir, y lo dice dos veces: en `hitl_reasons` y
  **dentro del propio fichero**, estampando `PARCIAL` en la cabecera de 80 bytes del
  STL junto al `acquisition_id`. Es trazabilidad que sobrevive a que alguien copie el
  fichero fuera del sistema, donde ya no hay `Provenance` que consultar.
- **Lo que el formato no puede llevar se declara, no se finge.** El STL es «pelado»:
  sin color por vértice, sin normales por vértice y sin topología compartida. El
  `color_superficie` del twin no cabe en el fichero y sigue vivo en `surface_ref`.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-13 | 0.1.0 | Registro inicial. Paquete `export-agents` con su contrato base (`ExportOutput` · `BaseExportAgent` · `SurfaceStore`), regeneración de la malla a STL binario desde `surface_ref`, verificación por relectura contra el presupuesto de 0,1 mm, exportación en el sistema del escáner o en el del twin, y declaración de snapshot parcial en la cabecera del fichero. |
| 2026-08-17 | 0.1.0 | `mean_deviation_mm` junto al máximo: es el **Chamfer** del round-trip, calculado sobre la correspondencia conocida (vértice *i* releído contra vértice *i*) y no por vecino más próximo, que esconderría una permutación. Y la desviación pasa a ser **distancia euclídea por vértice** en vez de error máximo por coordenada: un desplazamiento diagonal de 1 mm son √3 ≈ 1,73 mm, y medir por coordenada lo reportaba como 1,0. |

---

### `field-export-agent` — Exportación del campo gaussiano

| Campo | Valor |
|---|---|
| **Ubicación** | `packages/export-agents/` (`field.py` · `base.py`) |
| **Versión** | `0.1.0` |
| **Estado** | `active` |
| **Fase del pipeline** | 6 · Exportación (frontera contrato → fichero) |
| **Contrato común** | `ExportOutput` + `BaseExportAgent` |
| **Orquestador** | `IngestionPipeline.exportar(result, destino)` |

**Rol / Propósito**

> Materializa lo que el escáner **no puede ver**: el interior que sembró el CBCT.
> El canal de malla devuelve la superficie medida; éste devuelve el volumen.

**Reglas específicas**

- ⚠️ **El PLY no es un `.ply` de 3D Gaussian Splatting, y es deliberado.** La
  convención de INRIA guarda `opacity` y armónicos esféricos `f_dc_*` — color y
  transparencia. Este campo tiene `density` (σₙ, atenuación Beer-Lambert) y un CBCT
  **no mide color**. Ponerle esa cabecera lo haría abrible en cualquier visor de
  splats, que pintaría un color inventado sobre una magnitud física: peor que no
  abrirlo. De qué color se pinta un campo de densidad es decisión de producto y sigue
  pendiente del ADR de motor de render.
- 🔒 **`origin` y `hu_range` viajan en el artefacto, no en `Provenance`.** El
  `cbct-agent` centra el campo (`centers -= mundo.mean(0)`) y normaliza la densidad a
  `[0, 1]`. Hasta 2026-08-17 el desplazamiento **se descartaba**, y eso hacía el campo
  irreversible: depende del dato, así que ninguna versión del agente lo recomputa. No
  se metió en `Provenance` porque `transform` ya significa el alineamiento de la
  fusión geométrica (ADR 004) y reutilizarlo colisionaría, y porque el esquema declara
  el snapshot **autocontenido**: lo que hace reversible un blob viaja con él.
- **Pedir `frame="cbct"` sobre un artefacto sin `origin` falla declarando** y dice que
  hay que reingerir. Entregarlo centrado haciéndolo pasar por coordenadas del CBCT
  desplazaría todo lo que se midiese encima, y con buen aspecto.
- **Las posiciones se escriben en `double`, no en `float32`.** Así
  `max_deviation_mm` sale **exactamente 0** y la verificación mide *bugs* —endianness,
  orden de propiedades, un stride mal calculado— en vez del redondeo del formato, que
  taparía uno pequeño. Un splat convencional usa float32; aquí interesa la medida.
- **Los dos marcos salen del dato, no de quien llama.** A diferencia de `frame="twin"`
  en el canal de malla —que aplica `Provenance.transform`, escrita registrando contra
  lo que el llamante pase como `target`, y el pipeline no fija ese marco—, aquí ambos
  se definen con arrays del propio artefacto. Cerrar ese hueco es del protocolo entre
  agentes, no de un exportador.
- **`region_id` se escribe si el campo viene segmentado, y solo entonces.** Es la
  etiqueta FDI por gaussiana que produce el `segmentation-agent`, o sea lo único que el
  pipeline sabe de anatomía; tirarla al exportar dejaba el PLY con la geometría bien y
  **mudo sobre el contenido**, obligando a resegmentar algo ya calculado. No se escribe
  una columna de ceros cuando falta: en el convenio del agente `0` significa «sin
  asignar», que es una afirmación distinta de «nadie lo ha segmentado todavía».

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-17 | 0.1.0 | Registro inicial. Sale de `planned`: PLY binario con las propiedades que el campo **tiene**, dos marcos (`twin` centrado / `cbct` en mm reales), reversibilidad medida por relectura y `densidad_a_hu` para deshacer la normalización. Requirió que el `cbct-agent` empezase a guardar `origin` y `hu_range`. |
| 2026-08-17 | 0.1.0 | `region_id` (propiedad opcional `short`) viaja al PLY cuando el campo está segmentado. Lo destapó atar la segmentación al recorrido extremo a extremo: las dos etapas pasaban sus tests por separado y la etiqueta se perdía justo entre ellas. |

---

### `render-export-agent` — Render multivista del campo

| Campo | Valor |
|---|---|
| **Ubicación** | `packages/export-agents/` (`render.py` · `base.py`) |
| **Versión** | `0.1.0` |
| **Estado** | `active` |
| **Fase del pipeline** | 6 · Exportación (frontera contrato → fichero) |
| **Contrato común** | `ExportOutput` + `BaseExportAgent` |
| **Orquestador** | `IngestionPipeline.exportar(result, destino)`; `render=False` lo salta |

**Rol / Propósito**

> Convierte el campo en algo que una persona puede **mirar y aprobar**. Es el canal
> que hace revisable el interior del gemelo, y el que cierra la métrica del brief para
> lo que no son milímetros.

**Reglas específicas**

- ⚠️ **No son las fotos intraorales.** Son renders del gemelo. De las originales solo
  se guardaron muestras de apariencia (`image_refs`), así que no hay contra qué
  compararlas y no se finge que la haya: lo medido es el ciclo *twin → fichero →
  render*, con `psnr_db` y `ssim`.
- **No rasteriza splats: compone por Beer-Lambert.** Un rasterizador de 3DGS mezcla
  color con `alpha blending`, que depende del orden. `density` **no es opacidad**, y la
  integral que le corresponde, `I = exp(−∫σ ds)`, es aditiva en la profundidad óptica y
  por tanto **independiente del orden**. Eso no es eficiencia: hace el render
  determinista sin ordenar por profundidad, y lo vuelve una radiografía sintética en
  vez de una foto inventada.
- ⚠️ **Se deposita masa, no amplitud.** Evaluar τ en el centro del píxel —lo evidente—
  produce aliasing, no imagen: con `s = 0,15 mm` y píxeles de ~0,7 mm el centro cae a
  más de 4σ de casi todas las gaussianas. Y recortar σ a medio píxel para taparlo infla
  la amplitud: medido sobre `histora`, τ máximo pasaba de 34 a 256 px a **226** a
  128 px, o sea la imagen dependía del tamaño del detector. Se deposita la masa con un
  perfil normalizado y se divide por el área del píxel; hay test de regresión.
- **Las vistas se nombran por ángulo, nunca por anatomía** (`az090_el+00`). El
  significado anatómico de un eje depende de cómo el equipo escriba el DICOM, y en este
  proyecto suponerlo en vez de leerlo salió mal tres veces sobre el mismo paciente. Un
  nombre como `az090_el+00` no puede mentir; `oclusal` sí.
- **El encuadre es común a todas las vistas.** Si cada imagen eligiese el suyo, dos
  renders del mismo campo no serían comparables píxel a píxel y el SSIM mediría el
  encuadre; además una vista podría recortar lo que otra muestra.
- **Se devuelven PSNR y SSIM, no uno.** Miden cosas distintas: el PSNR promedia el
  error por píxel y se deja engañar por un desplazamiento global de brillo, el SSIM
  compara estructura local. Medido sobre una imagen suave, un sesgo de 10 niveles y
  ruido de la misma energía dan PSNR casi igual y SSIM de 0,99 frente a 0,57.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-17 | 0.1.0 | Registro inicial. Render multivista por Beer-Lambert, reproducible byte a byte, con PSNR/SSIM del ciclo contra el PLY exportado y presupuestos `RENDER_PSNR_BUDGET_DB = 40` / `RENDER_SSIM_BUDGET = 0,99`. | <!--const:RENDER_PSNR_BUDGET_DB--> <!--const:RENDER_SSIM_BUDGET-->

---

### `uos-export-agent` — El caso entero como escena UOS

| Campo | Valor |
|---|---|
| **Ubicación** | `packages/uos/` (`agente.py` · `manifiesto.py` · `contenedor.py` · `validador.py` · `vistas.py` · `procedencia.py`) |
| **Versión** | `0.1.0` |
| **Estado** | `active` |
| **Fase del pipeline** | 6 · Exportación (frontera contrato → fichero) |
| **Contrato común** | `ExportOutput` + `BaseExportAgent` |
| **Orquestador** | `IngestionPipeline.exportar(...)`; sin seudónimo declara `FAILED`, sin malla `MISSING` |

**Rol / Propósito**

> Los otros cinco canales materializan **una** cosa —la malla, el campo, el compuesto, el
> paquete del visor, el render—. Este empaqueta el caso entero con sus relaciones
> declaradas: qué asset viene de qué visita, en qué marco vive cada uno y con qué
> transformada se alinean. Es la diferencia entre entregar ficheros y entregar una escena.

Implementa el nivel **UOS-Core** del borrador de spec *Unified Oral Scene* v0.2: un ZIP
**sin comprimir** cuya primera entrada física es `manifest.json`, que **referencia los
formatos nativos intactos** en vez de transcodificarlos. Ningún formato existente hace eso
—DICOM no modela gaussianas ni se transmite bien en web, glTF no modela volúmenes ni
metadatos clínicos, OpenUSD es ajeno al ecosistema clínico—, y por eso el contenedor es
propio y el validador puede correr sobre un fichero que escribió otro.

**Reglas específicas**

- ⚠️ **Referencia, no transcodifica.** Los ficheros entran tal cual y su `sha256` va en el
  manifiesto: lo que sale es byte-idéntico a lo que entró. La desviación cero que reporta
  está **medida** —se relee el contenedor y se recomputa el hash de cada asset—, no
  afirmada. Sobre el caso real: malla de 11.004.334 bytes dentro y fuera.
- ⚠️ **Sin seudónimo se declara `FAILED`, y NO se cae al `acquisition_id`.** Ese
  identificador sale del nombre del directorio del caso, que en un sistema real lleva el
  nombre del paciente o su número de historia. Un seudónimo por defecto que resulta ser el
  dato identificable es peor que no tener ninguno, porque `phi_state` diría
  `pseudonymized` mintiendo.
- ⚠️ **Ningún nombre de fichero del proveedor viaja dentro.** Los de verdad llevan
  identificadores —la malla de este caso se llamaba `1574 UpperJawScan.stl` y `1574` es el
  número de caso—. Dentro todo se nombra por su papel: `scene/scan.stl`,
  `scene/appearance.ply`, `images/img_000.jpg`. La trazabilidad la da el `sha256`, que es
  más fuerte que un nombre y no identifica a nadie.
- **El marco canónico es el ESCÁNER, no el CBCT**, y eso invierte lo que hace el pipeline.
  Aquí se trabaja centrado en el CBCT porque es donde vive el campo gaussiano; UOS pone el
  escáner de hub porque es la geometría de referencia de un caso dental —micras frente a
  vóxeles de 0,3 mm—. La inversión se hace en el borde y **se declara** como
  `reg.ct_to_ios`, con su matriz, su método y su error, en vez de reescribir geometría. Es
  coherente con que la fusión anote en lugar de transformar.
- **Los ejes anatómicos de las vistas se MIDEN, no se suponen.** Es la misma regla que rige
  el eje ápico-coronal del CBCT, que se lee del `ImagePositionPatient`. Una malla de
  escáner no trae cabecera que lo diga, así que cada dirección sale de las etiquetas FDI:
  *oclusal* de la encía hacia las coronas, *derecha* del centroide de los cuadrantes 2 y 3
  al de los cuadrantes 1 y 4, *anterior* de los molares a los incisivos. **Sin etiquetas no
  hay vistas y se dice**: bautizar los ejes principales de la nube produce nombres
  plausibles y a veces invertidos, y una vista que se llama «vestibular derecha» y enseña
  la izquierda es peor que no tenerla. Es el mismo motivo por el que el
  `render-export-agent` nombra las suyas por ángulo.
- **Solo tienen vista propia las piezas ANOTADAS.** Una por diente etiquetado serían
  dieciséis entradas equivalentes; lo que hace útil un deep-link es que apunte a donde
  alguien miró. Las piezas que el informe cita y el escáner no trae se agrupan en **un**
  aviso, no en uno por diente: el gate ya lleva uno que lo explica entero, y repetirlo
  entierra los motivos que solo aparecen una vez.
- **Un `.uos` es append-only lógico.** Modificar no es editar: es escribir una versión
  nueva del manifiesto que apunta al `sha256` de la anterior. La autoridad está en
  `prev_manifest_sha256`, dentro del manifiesto; `provenance/chain.json` la materializa
  para poder recorrerla sin abrir todas las versiones. El validador comprueba que la cadena
  y los manifiestos **cuenten la misma historia** — retocar un manifiesto, arrancar un
  eslabón o pegar la cadena de otro caso invalidan.
- **Lo estructurado vive solo en el manifiesto.** `ExportOutput` es `extra="forbid"` y lo
  comparten seis canales: ensancharlo con `n_assets` o `conformidad` daría dos sitios donde
  la misma verdad puede divergir y obligaría a los otros cinco a cargar con campos que no
  usan.
- **Layer 3 vive en `derived/` y solo ahí.** Es lo que permite desmontar el módulo de IA
  borrando un directorio y sus entradas del manifiesto, y distribuir el caso en
  jurisdicciones donde no está habilitado. El validador lo comprueba en los dos sentidos.

**Lo que NO está, y se dice para que nadie lo dé por hecho**

- El **volumen** y las **señales** — son los niveles `UOS-Vol` y `UOS-Sig`. Hoy el
  contenedor lleva la malla y la apariencia, no el CBCT del que salió todo.
- El **`fhir_map`**, que se declara vacío: poblarlo exige decidir a qué recurso FHIR R4
  mapea cada asset.
- Las **firmas Ed25519**. No falta el código: falta decidir qué clave firma —la clínica
  emisora, la plataforma, o ambas— y dónde vive. Firmar con una clave inventada daría un
  `.uos` que *parece* firmado, que es peor que uno que declara no estarlo. El validador
  avisa si encuentra `provenance/signatures/` para no ignorarlas en silencio.

**Historial de cambios**

| Fecha | Versión | Cambio |
|---|---|---|
| 2026-08-24 | 0.1.0 | Registro inicial. Nivel UOS-Core: manifiesto, contenedor ZIP/STORE, validador con niveles de conformidad, vistas con ejes anatómicos medidos y cadena de procedencia entre versiones. Verificado sobre el caso clínico real: `VALIDO`, 10 assets, 19 vistas, malla byte-idéntica. |


---

### La fase 6 en el orquestador

`IngestionPipeline.exportar(result, destino)` dispara los **seis** canales sobre un
`PipelineResult` y devuelve otro **nuevo**, con `exports` y los motivos de revisión
acumulados. Cinco materializan una pieza del gemelo —STL, campo, compuesto, paquete del
visor, render— y el sexto, `uos-export-agent`, empaqueta el caso entero como escena.

Va en un método aparte —no dentro de `run`— por contrato: exportar **escribe
ficheros**, y `run`/`fuse` son puras respecto al disco salvo por el almacén de artefactos.

Tres reglas que conviene tener a mano:

- **Los canales son independientes.** Que no haya malla no impide exportar el campo, y que
  falle el render no borra el STL ya escrito.
- **`PipelineResult.reversible`** exige que cada canal que corrió esté dentro de *su*
  presupuesto —milímetros para geometría, PSNR/SSIM para imagen—, que ninguno esté en
  `FAILED`, y que se haya exportado algo. Sin recorrido no hay reversibilidad que afirmar.
- **`MISSING` no es un motivo de revisión humana.** Que una adquisición sea solo CBCT es
  normal y no dice nada del fichero que sí se escribió; un `FAILED` sí se declara.

Probado de punta a punta en
[`apps/agent-orchestrator/tests/test_e2e.py`](apps/agent-orchestrator/tests/test_e2e.py):
**ingesta → fusión → segmentación → exportación**, con la fusión registrando las nubes
reales del twin —la malla del `mesh-agent` contra el campo del `cbct-agent`— y no un blob
sintético contra sí mismo. Tres comprobaciones cargan el peso:

- **La geometría que sale es la que entró**, comparando las cotas del OBJ original con las
  del STL regenerado. Existe porque las métricas internas de cada agente miden contra lo que
  *ese agente* escribió: un eje perdido o un espejo pasarían todas y esta no.
- **La fusión desbloquea `frame="twin"`**, que necesita la `RigidTransform` que solo ella
  escribe. Ata las fases por una dependencia real y no por el orden de las llamadas: si
  alguien desconectara la fusión, se pondría rojo aunque cada etapa siguiera pasando lo suyo.
- **Las etiquetas de diente llegan al fichero**: el `region_id` del PLY exportado tiene que
  ser el que infirió la segmentación. Esta es la que encontró algo — ver más abajo.

El modelo de segmentación del recorrido es un **doble de test**, porque el de verdad pide
GPU y un checkpoint sin versionar. No debilita lo que se prueba: la integración tiene que
demostrar que la etapa está *atada*, y para eso el origen de las etiquetas da igual. La
calidad del modelo se mide en otro sitio y no es un test.

⚠️ **Atar la segmentación al recorrido destapó una pérdida silenciosa.** Las dos etapas
pasaban sus tests por separado —el `segmentation-agent` guardaba `region_id` en el
artefacto, el `field-export-agent` escribía un PLY correcto— y la etiqueta se perdía justo
entre ellas: el fichero salía con la geometría bien y sin nada que dijera qué gaussiana es
qué diente. Es el mismo modo de fallo que el centroide descartado del `cbct-agent`, y por el
mismo motivo no lo veía nadie: **ningún test de etapa mira lo que la etapa siguiente
necesita**.

⚠️ **Confianza y reversibilidad no son lo mismo**, y el recorrido lo enseña: registrar la
malla del escáner contra los vóxeles de tejido duro del CBCT da 0,605 mm sobre un 52,7 %
solapado, así que el gate humano **salta** — correctamente, no son la misma superficie. Y
aun así la exportación sale con sus métricas en verde, porque el error de reconstrucción
mide si el fichero reproduce el twin, no si el twin está bien registrado.

---

### Agentes de desarrollo (dev-time)

Herramientas de IA externas que el equipo usa para asistir el desarrollo. **No forman parte del sistema en producción** ni tienen acceso autónomo al runtime: toda su salida entra al repositorio como código propuesto y pasa por Pull Request + revisión humana (y por el guardián `ai-code-reviewer`) antes de mergearse.

| Herramienta | Rol en el proyecto | Modelo | Notas de gobernanza |
|---|---|---|---|
| OpenCode / Claude Code | Asistentes de codificación interactivos: generación, refactorización, tests y documentación bajo dirección de una persona del equipo | Claude (Opus/Sonnet según sesión) | Conducidos por humano (no autónomos); sin acceso a datos clínicos; todo output vía PR + revisión humana. No se les delega decisiones clínicas ni de arquitectura. |

> Se documentan a nivel de fila (no con la ficha de agente) porque son asistentes **interactivos**, no agentes del sistema: no tienen contrato de datos, fase de pipeline ni reglas de delegación propias. Registra aquí cualquier otra herramienta de IA dev-time que se incorpore.
