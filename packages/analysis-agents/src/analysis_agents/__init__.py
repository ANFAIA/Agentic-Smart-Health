"""Agentes de **análisis** del Digital Twin dental.

Tercera familia del pipeline, después de la ingesta y en medio de la fusión:

```
[FUSIÓN GEOMÉTRICA]  registro malla↔CBCT (banda ε) · NO usa FDI
   ▼
[SEGMENTACIÓN]       puebla region_id (FDI)          ← este paquete
   ▼
[FUSIÓN SEMÁNTICA]   ancla pH/observaciones al FDI
```

| Agente | Rol | Estado |
|---|---|---|
| `SegmentationAgent` | `region_id` (FDI) por gaussiana + mapa `FDI → confianza` | implementado |
| `pathology-agent` | hallazgos candidatos para revisión clínica | `planned` |
| `clinical-poc-agent` | métrica visual básica de encía | `planned` |

Todos consumen y enriquecen un `TwinSnapshot` a través de `core-schemas` —nunca
vuelven al fichero crudo— y dejan su propia `Provenance`.
"""

from analysis_agents.base import (
    DEFAULT_HITL_THRESHOLD,
    AnalysisAgent,
    AnalysisOutput,
    BaseAnalysisAgent,
)
from analysis_agents.dental import (
    RADIO_NOMBRE_MM,
    SegmentadorDental,
    absorbe_islas,
    afina_fronteras,
    quita_motas,
    rellena_etiquetas,
    rellena_huecos_interiores,
)
from analysis_agents.segmentation import (
    DEFAULT_CODES,
    DEFAULT_UNASSIGNED_LIMIT,
    GUM_CLASS,
    GaussianStore,
    SegmentationAgent,
    SegmentationOutput,
    Segmenter,
)

__all__ = [
    "DEFAULT_CODES",
    "RADIO_NOMBRE_MM",
    "SegmentadorDental",
    "absorbe_islas",
    "afina_fronteras",
    "quita_motas",
    "rellena_huecos_interiores",
    "rellena_etiquetas",
    "DEFAULT_HITL_THRESHOLD",
    "DEFAULT_UNASSIGNED_LIMIT",
    "GUM_CLASS",
    "AnalysisAgent",
    "AnalysisOutput",
    "BaseAnalysisAgent",
    "GaussianStore",
    "SegmentationAgent",
    "SegmentationOutput",
    "Segmenter",
]
