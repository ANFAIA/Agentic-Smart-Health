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
| `export-agent` | malla | `surface_ref` → **STL binario** + error medido | ✅ |
| `field-export-agent` | campo | `gaussian_field_ref` → **PLY binario** + error medido | ✅ |
| `render-export-agent` | imagen | `gaussian_field_ref` → **PNG multivista** + PSNR/SSIM | ✅ |

El PLY del segundo **no es un `.ply` de 3D Gaussian Splatting**, y la diferencia es
deliberada: el contrato sustituye la opacidad α por `density` (atenuación
Beer-Lambert) y un CBCT no mide color, así que no hay `opacity` ni armónicos
esféricos que escribir. Ponerle la cabecera de INRIA lo haría abrible en cualquier
visor de splats, que pintaría un color inventado sobre una magnitud física. De qué
color se pinta un campo de densidad es una decisión de producto, no una conversión
de formato, y sigue pendiente del ADR de motor de render.

Por lo mismo, el tercero **no rasteriza splats**: compone por Beer-Lambert, que es la
integral que corresponde a una densidad y no a una opacidad. El resultado es una
radiografía sintética, es independiente del orden de las primitivas —y por tanto
determinista sin ordenar por profundidad— y no inventa color.

El **canal de metadatos** —el `TwinSnapshot` serializado a JSON— no necesita agente:
es `model_dump()` de Pydantic, y meterlo aquí sería envolver una llamada de una
línea en un contrato de fallos que no puede fallar
(`docs/architecture/multi-agent-pipeline.md` §5).

Exportar es **solo lectura sobre el gemelo**: ningún agente de esta familia modifica
el snapshot ni escribe en el `ArtifactStore`.
"""

from export_agents.base import (
    DEFAULT_HITL_THRESHOLD,
    RENDER_PSNR_BUDGET_DB,
    RENDER_SSIM_BUDGET,
    REVERSIBILITY_BUDGET_MM,
    BaseExportAgent,
    ExportAgentProtocol,
    ExportOutput,
    SurfaceStore,
)
from export_agents.field import (
    COLUMNAS_DE_ARRAY,
    FIELD_FRAMES,
    FieldExportAgent,
    FieldFrame,
    densidad_a_hu,
    escribe_ply,
    lee_ply,
)
from export_agents.render import (
    VISTAS_POR_DEFECTO,
    RenderExportAgent,
    Vista,
    beer_lambert,
    escribe_png,
    lee_png,
    profundidad_optica,
    psnr,
    ssim,
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
    "COLUMNAS_DE_ARRAY",
    "DEFAULT_HITL_THRESHOLD",
    "FIELD_FRAMES",
    "FRAMES",
    "RENDER_PSNR_BUDGET_DB",
    "RENDER_SSIM_BUDGET",
    "REVERSIBILITY_BUDGET_MM",
    "VISTAS_POR_DEFECTO",
    "BaseExportAgent",
    "ExportAgent",
    "ExportAgentProtocol",
    "ExportOutput",
    "FieldExportAgent",
    "FieldFrame",
    "Frame",
    "RenderExportAgent",
    "SurfaceStore",
    "Vista",
    "beer_lambert",
    "densidad_a_hu",
    "escribe_ply",
    "escribe_png",
    "face_normals",
    "lee_ply",
    "lee_png",
    "profundidad_optica",
    "psnr",
    "read_stl_triangles",
    "ssim",
    "stl_header",
    "write_binary_stl",
]
