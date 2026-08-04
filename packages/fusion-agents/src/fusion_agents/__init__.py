"""Agentes de fusión del Digital Twin dental (ADR 004).

El pipeline separa la fusión en **dos etapas, con la segmentación en medio**:

```
[FUSIÓN GEOMÉTRICA]  registro malla↔CBCT (banda ε) · color · NO usa FDI
   ▼
[SEGMENTACIÓN]       puebla region_id (FDI)
   ▼
[FUSIÓN SEMÁNTICA]   ancla pH/observaciones al FDI · NO usa geometría
```

| Agente | Etapa | Estado |
|---|---|---|
| `SemanticFusionAgent` | anclaje al FDI | implementado |
| `GeometricFusionAgent` | registro malla↔CBCT | implementado (etapa fina) |

Son dos agentes y no uno porque entre ambos corre otra etapa, y porque tienen
material y criterio de aceptación distintos: la semántica se valida contra el
informe, la geométrica contra un lote de pares CBCT+IOS.
"""

from fusion_agents.base import (
    DEFAULT_HITL_THRESHOLD,
    BaseFusionAgent,
    FusionAgent,
    FusionOutput,
)
from fusion_agents.color import transfer_surface_color
from fusion_agents.geometric import GeometricFusionAgent
from fusion_agents.registration import (
    DEFAULT_EPSILON_MM,
    Registrar,
    RegistrationResult,
    icp,
)
from fusion_agents.semantic import SemanticFusionAgent
from fusion_agents.twin import insert_snapshot

__all__ = [
    "DEFAULT_EPSILON_MM",
    "DEFAULT_HITL_THRESHOLD",
    "BaseFusionAgent",
    "FusionAgent",
    "FusionOutput",
    "GeometricFusionAgent",
    "Registrar",
    "RegistrationResult",
    "SemanticFusionAgent",
    "icp",
    "transfer_surface_color",
    "insert_snapshot",
]
