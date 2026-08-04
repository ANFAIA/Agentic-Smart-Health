"""`semantic-fusion-agent`: ancla las observaciones regionales al diente (FDI).

Es la segunda etapa de fusión del pipeline, la que va **después** de la
segmentación. No toca geometría: solo comprueba que el diente al que el informe
cuelga una observación es un diente que la segmentación ha encontrado de verdad, y
propaga la confianza en consecuencia (ADR 004 §2.3, §2.4).
"""

from __future__ import annotations

from collections.abc import Mapping

from core_schemas import ModalityStatus, Provenance, RegionalObservation, TwinSnapshot

from fusion_agents.base import BaseFusionAgent, FusionOutput


class SemanticFusionAgent(BaseFusionAgent):
    """Cuelga cada `RegionalObservation` de su diente, y marca lo que no cuadra.

    **Entrada.** El `TwinSnapshot` con las observaciones que produjo el
    `report-agent`, y `detected`: el mapa `FDI → confianza` de los dientes que el
    `segmentation-agent` encontró. Se pasa explícitamente en vez de leer el campo
    gaussiano del almacén porque son ~14 códigos: cargar millones de primitivas
    para obtenerlos sería absurdo, y mantiene al agente testeable sin el almacén.

    **Qué decide.** Para cada observación:

    - Si su FDI está entre los detectados → la confianza es el **eslabón más
      débil**, `min(confianza_observación, confianza_FDI)`. Anclar un pH a un
      diente no puede ser más fiable que saber qué diente es.
    - Si **no** está → conflicto. Se conserva el FDI **del informe**, que es la
      fuente clínica, se pone la confianza a **0.0** y con eso cae por debajo del
      gate y va a revisión humana. No hace falta ningún campo nuevo: *la propia
      confianza es la marca*.

    **Qué NO decide.** No elige ganador entre informe y modelo. El experimento del
    Point Transformer midió que el error dominante del modelo es precisamente el
    desplazamiento al diente vecino, así que ahí es la parte menos fiable — pero el
    informe tampoco es infalible. Resolverlo en silencio produciría el fallo que el
    ADR 003 señala como el peor: silencioso e irreversible, sobre un dato clínico.
    """

    name = "semantic-fusion-agent"
    version = "0.1.0"

    def _fuse(  # type: ignore[override]
        self, snapshot: TwinSnapshot, *, detected: Mapping[str, float]
    ) -> FusionOutput:
        if not detected:
            # Sin segmentación no hay nada contra lo que validar el anclaje. Es
            # ausencia de entrada, no un fallo: se declara y no se inventa nada.
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "No hay dientes detectados: el segmentation-agent no ha corrido "
                    "sobre este snapshot o no encontró ninguno."
                ),
            )

        ancladas: list[RegionalObservation] = []
        motivos: list[str] = []

        for obs in snapshot.regional:
            fdi = obs.region_id
            if fdi in detected:
                confianza = min(obs.provenance.confidence, detected[fdi])
                if confianza < self.hitl_threshold:
                    motivos.append(
                        f"FDI {fdi}: confianza {confianza:.2f} bajo el umbral "
                        f"{self.hitl_threshold:.2f}"
                    )
            else:
                confianza = 0.0
                motivos.append(
                    f"FDI {fdi}: el informe lo referencia pero la segmentación no lo "
                    f"encontró (detectados: {', '.join(sorted(detected))})"
                )
            ancladas.append(self._anclar(obs, confianza))

        return self._outcome(
            ModalityStatus.OK,
            snapshot=self._emitir(snapshot, ancladas),
            hitl_reasons=motivos,
            detail=f"{len(ancladas)} observación(es) ancladas; {len(motivos)} para revisión.",
        )

    # --- piezas ---------------------------------------------------------- #
    @staticmethod
    def _anclar(obs: RegionalObservation, confianza: float) -> RegionalObservation:
        """Reescribe la confianza de la observación, conservando su procedencia.

        Se actualiza **solo** `confidence`: el *valor* (el pH) sigue viniendo del
        informe, y perder `source_file`/`modality` rompería la trazabilidad hasta el
        PDF. Quién hizo la fusión queda registrado en la `Provenance` del snapshot
        emitido, que es el valor que este agente sí deriva.
        """
        prov: Provenance = obs.provenance.model_copy(update={"confidence": confianza})
        return obs.model_copy(update={"provenance": prov})

    def _emitir(self, snapshot: TwinSnapshot, ancladas: list[RegionalObservation]) -> TwinSnapshot:
        """Snapshot **nuevo**, nunca el de entrada mutado (ADR 004 §2.5).

        Conserva el `acquisition_id`: es la identidad de visita, y mantenerlo es lo
        que hace que reejecutar la fusión sea idempotente en vez de inflar la serie
        del paciente con visitas ficticias.
        """
        prov = snapshot.provenance.model_copy(update={"agent": self.qualified})
        return snapshot.model_copy(update={"regional": ancladas, "provenance": prov})
