"""Agentes de **exportación** del Digital Twin dental (fase 6 del pipeline).

Cuarta familia, la única que escribe ficheros de salida:

```
[INGESTA]    fichero crudo → contrato        (única lectura del original)
   ▼
[FUSIÓN · SEGMENTACIÓN · ANÁLISIS]  enriquecen el TwinSnapshot
   ▼
[EXPORTACIÓN]  contrato → fichero            ← este paquete
```

| Agente | Canal | Qué materializa | Estado |
|---|---|---|---|
| `export-agent` | malla | `surface_ref` → **STL binario** + error medido | implementado |
| `field-export-agent` | campo gaussiano | `gaussian_field_ref` → `.ply` / `.splat` | `planned` |

El formato binario del segundo depende de qué motor de render se adopte, y se
decide en el futuro ADR de motor de render — no aquí.

El **canal de metadatos** —el `TwinSnapshot` serializado a JSON— no necesita agente:
es `model_dump()` de Pydantic, y meterlo aquí sería envolver una llamada de una
línea en un contrato de fallos que no puede fallar
(`docs/architecture/multi-agent-pipeline.md` §5).

Exportar es **solo lectura sobre el gemelo**: ningún agente de esta familia modifica
el snapshot ni escribe en el `ArtifactStore`.
"""

from export_agents.base import (
    DEFAULT_HITL_THRESHOLD,
    REVERSIBILITY_BUDGET_MM,
    BaseExportAgent,
    ExportAgentProtocol,
    ExportOutput,
    SurfaceStore,
)
from export_agents.stl import (
    FRAMES,
    ExportAgent,
    Frame,
    face_normals,
    read_stl_triangles,
    stl_header,
    write_binary_stl,
)

__all__ = [
    "DEFAULT_HITL_THRESHOLD",
    "FRAMES",
    "REVERSIBILITY_BUDGET_MM",
    "BaseExportAgent",
    "ExportAgent",
    "ExportAgentProtocol",
    "ExportOutput",
    "Frame",
    "SurfaceStore",
    "face_normals",
    "read_stl_triangles",
    "stl_header",
    "write_binary_stl",
]
