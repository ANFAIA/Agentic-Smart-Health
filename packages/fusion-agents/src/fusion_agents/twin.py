"""Inserción del snapshot fusionado en la serie temporal del paciente (ADR 004 §2.5).

Implementa la regla **1 snapshot = 1 visita** de la issue #33: la identidad de una
visita es su `acquisition_id`, así que reejecutar la fusión sobre el mismo material
**reemplaza** en vez de añadir. Sin esto, cada corrida inflaría el historial clínico
del paciente con visitas que nunca ocurrieron.
"""

from __future__ import annotations

from core_schemas import PatientDigitalTwin, TwinSnapshot


def insert_snapshot(twin: PatientDigitalTwin, snapshot: TwinSnapshot) -> PatientDigitalTwin:
    """Devuelve un twin **nuevo** con `snapshot` insertado en su sitio temporal.

    - Si el `acquisition_id` ya está en la serie, se **reemplaza** → idempotente.
    - Si no, se **añade** y la serie se reordena por `timestamp`, de modo que
      reprocesar una visita antigua no la deja al final fuera de orden.

    No muta el twin de entrada: el historial clínico es *append-only* y la
    trazabilidad depende de que reescribir sea un acto explícito.
    """
    resto = [s for s in twin.snapshots if s.acquisition_id != snapshot.acquisition_id]
    serie = sorted([*resto, snapshot], key=lambda s: s.timestamp)
    return twin.model_copy(update={"snapshots": serie})
