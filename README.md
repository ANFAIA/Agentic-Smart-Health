# Agentic Smart Health

Sistema multiagente para la integración, análisis y representación de datos clínicos dentales heterogéneos sobre un **Digital Twin** del paciente, basado en Gaussian Splatting con atributos clínicos por punto/zona y soporte de series temporales.

> Proyecto open source · Licencia Apache 2.0 · Python ≥ 3.11

---

## Contexto del proyecto

El sector dental maneja datos altamente heterogéneos: escáneres CBCT (DICOM), archivos STL de escaneos intraorales, informes clínicos en PDF e imágenes 2D. Esta información vive fragmentada en silos por proveedor y por clínica, lo que impide un seguimiento longitudinal real del paciente y compromete su soberanía sobre los propios datos de salud.

**Open Dental Data Twin** aborda este problema mediante una arquitectura multiagente que organiza, integra y analiza de forma autónoma datos dentales heterogéneos, proyectándolos sobre un gemelo digital del paciente. El proceso es reversible: el sistema puede regenerar ficheros STL e imágenes directamente desde el Digital Twin.

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
│   └── slicer-mcp-server/     ← servidor MCP para integración con 3D Slicer
├── packages/
│   ├── core-schemas/          ← esquemas Pydantic compartidos
│   └── 3dgs-engine/           ← motor de renderizado 3D Gaussian Splatting
├── data/                      ← datos de trabajo (sintéticos / anonimizados)
├── docs/                      ← documentación (ver nota más abajo)
├── notebooks/                 ← experimentación y exploración
└── tests/                     ← suite de pruebas global
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

---

## Paquetes compartidos (`packages/`)

### `core-schemas`

Biblioteca de **esquemas Pydantic v2** compartidos por todas las aplicaciones del workspace. Define los modelos de datos canónicos del sistema: estructuras del Digital Twin, representaciones de datos clínicos dentales, contratos entre agentes y formatos de exportación. Actúa como fuente única de verdad para los tipos de datos del proyecto.

### `3dgs-engine`

Motor de **renderizado y procesamiento 3D Gaussian Splatting** (3DGS). Implementa la representación neuronal del Digital Twin: cada punto/zona del espacio almacena atributos clínicos (color, densidad ósea, datos clínicos asociados, timestamp). Proporciona las primitivas necesarias para construir, actualizar y consultar el gemelo digital, así como para las operaciones de exportación reversible a STL e imágenes.

---

## Quickstart

### Requisitos previos

- Python ≥ 3.11
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
