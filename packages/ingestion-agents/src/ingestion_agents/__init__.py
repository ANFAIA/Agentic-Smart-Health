"""Agentes de ingesta del Digital Twin dental.

Regla del pipeline: **1 modalidad = 1 soporte = 1 agente**. Los agentes de
ingesta son los únicos componentes que tocan ficheros crudos; a partir del
`TwinSnapshot` se opera solo sobre el contrato de `core-schemas`.

| Agente | Modalidad | Soporte | Cerebro |
|---|---|---|---|
| `MeshAgent` | `mesh` (OBJ intraoral) | superficial | determinista |
| `CBCTAgent` | `cbct` (serie DICOM) | volumétrico | determinista |
| `ReportAgent` | `report` (PDF/texto) | regional | determinista (`rules`) o LLM (`llm`) |

Solo el informe justifica un LLM: es la única modalidad cuya entrada no tiene
esquema. Decisiones de diseño y puntos abiertos en el `README.md` del paquete.
"""

from ingestion_agents.base import (
    BaseIngestionAgent,
    IngestionAgent,
    IngestionOutput,
)
from ingestion_agents.cbct_agent import CBCTAgent, pseudonymize
from ingestion_agents.mesh_agent import MeshAgent
from ingestion_agents.report_agent import ReportAgent
from ingestion_agents.store import ArtifactStore

__all__ = [
    "ArtifactStore",
    "BaseIngestionAgent",
    "CBCTAgent",
    "IngestionAgent",
    "IngestionOutput",
    "MeshAgent",
    "ReportAgent",
    "pseudonymize",
]
