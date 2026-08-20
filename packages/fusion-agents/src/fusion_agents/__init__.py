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
| `GeometricFusionAgent` | registro malla↔CBCT | implementado (fina + gruesa) |

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
from fusion_agents.geometric import DEFAULT_MIN_OVERLAP, GeometricFusionAgent
from fusion_agents.preparacion import (
    HU_CORONA,
    HU_ESMALTE,
    arcada_del_nombre,
    nubes_para_registro,
    plano_oclusal,
    plano_oclusal_del_esmalte,
)
from fusion_agents.registration import (
    DEFAULT_EPSILON_MM,
    DEFAULT_OVERLAP_MM,
    EPSILON_IOS_CBCT_MM,
    Registrar,
    RegistrationResult,
    icp,
    icp_global,
)
from fusion_agents.semantic import SemanticFusionAgent
from fusion_agents.twin import insert_snapshot

__all__ = [
    "plano_oclusal",
    "plano_oclusal_del_esmalte",
    "HU_ESMALTE",
    "nubes_para_registro",
    "arcada_del_nombre",
    "HU_CORONA",
    "DEFAULT_EPSILON_MM",
    "DEFAULT_MIN_OVERLAP",
    "DEFAULT_OVERLAP_MM",
    "EPSILON_IOS_CBCT_MM",
    "DEFAULT_HITL_THRESHOLD",
    "BaseFusionAgent",
    "FusionAgent",
    "FusionOutput",
    "GeometricFusionAgent",
    "Registrar",
    "RegistrationResult",
    "SemanticFusionAgent",
    "icp",
    "icp_global",
    "transfer_surface_color",
    "insert_snapshot",
]
