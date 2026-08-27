"""`semantic-fusion-agent`: ancla las observaciones regionales al diente (FDI).

Es la segunda etapa de fusión del pipeline, la que va **después** de la
segmentación. No toca geometría: solo comprueba que el diente al que el informe
cuelga una observación es un diente que la segmentación ha encontrado de verdad, y
propaga la confianza en consecuencia (ADR 004 §2.3, §2.4).
"""

from __future__ import annotations

from collections.abc import Mapping

from core_schemas import (
    Hallazgo,
    ModalityStatus,
    Provenance,
    RegionalObservation,
    TwinSnapshot,
)
from ingestion_agents.ontology import describe, is_valid_fdi

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
        self, snapshot: TwinSnapshot, *, detected: Mapping[str, float],
        arcada_aportada: str | None = None,
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

        # Arcadas en las que la segmentación encontró ALGO. Una arcada con cero dientes
        # no es que se buscara y no hubiera: casi siempre es que no había con qué mirar
        # —el escáner intraoral es de una sola arcada— y eso es ausencia de entrada, no
        # desacuerdo. Ver `_falta_la_arcada`.
        cubiertas = {_arcada_de(f) for f in detected}

        ancladas: list[RegionalObservation] = []
        motivos: list[str] = []
        sin_arcada: dict[str, list[str]] = {}

        for obs in snapshot.regional:
            fdi = obs.region_id
            # ⚠️ **«El informe lo referencia» no es lo mismo que «el informe dice que
            # existe».** Un informe dental es una ficha de 32 posiciones y habla de todas,
            # incluidas las que declara `ausente`. Cuando el hallazgo es ese, que la
            # segmentación no encuentre el diente es ACUERDO, no desacuerdo — y llamarlo
            # conflicto avisaba justo cuando el pipeline acertaba. Medido sobre un caso
            # real: el informe daba la 28 por ausente con confianza 0,877, el escáner no
            # la traía y el gate lo denunciaba igual.
            declarado_ausente = Hallazgo.AUSENTE in obs.attributes.hallazgos
            if fdi in detected:
                confianza = min(obs.provenance.confidence, detected[fdi])
                if declarado_ausente:
                    # Y el desacuerdo de verdad es el otro: el informe dice que no está y
                    # el modelo lo encuentra. Uno de los dos se equivoca sobre si al
                    # paciente le falta una pieza, que no es un matiz.
                    motivos.append(
                        f"FDI {fdi}: el informe lo da por AUSENTE y la segmentación sí lo "
                        f"encontró (confianza {detected[fdi]:.2f})"
                    )
                elif confianza < self.hitl_threshold:
                    motivos.append(
                        f"FDI {fdi}: confianza {confianza:.2f} bajo el umbral "
                        f"{self.hitl_threshold:.2f}"
                    )
            else:
                confianza = 0.0
                arco = _arcada_de(fdi)
                if declarado_ausente:
                    # Las dos fuentes coinciden en que ese diente no está. No hay nada que
                    # revisar, y la observación se ancla igual con confianza 0: sigue
                    # siendo lo que el informe dijo de esa posición.
                    pass
                elif arco in cubiertas:
                    # Aquí sí se miró y no estaba: es un desacuerdo real entre el informe
                    # y el modelo, y va uno por uno porque cada uno es una decisión.
                    motivos.append(
                        f"FDI {fdi}: el informe lo referencia pero la segmentación no lo "
                        f"encontró (detectados: {', '.join(sorted(detected))})"
                    )
                else:
                    sin_arcada.setdefault(arco, []).append(fdi)
            ancladas.append(self._anclar(obs, confianza))

        motivos += _falta_la_arcada(sin_arcada, arcada_aportada)

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


# --- arcadas sin cubrir ------------------------------------------------------ #
def _arcada_de(fdi: str) -> str:
    """`"superior"` / `"inferior"`, o `"?"` si el código no es un FDI válido.

    No se lanza: un FDI inválido en el informe es problema del `report-agent`, y hacer
    caer la fusión aquí escondería el anclaje del resto de observaciones.
    """
    return describe(fdi).arch if is_valid_fdi(fdi) else "?"


def _falta_la_arcada(sin_arcada: dict[str, list[str]],
                     arcada_aportada: str | None = None) -> list[str]:
    """**Un** motivo por arcada no cubierta, no uno por diente. Y es la diferencia
    entre un gate que se lee y uno que se ignora.

    Medido sobre un caso clínico real: el informe referencia 32 dientes y solo se aportó
    el escaneo maxilar, así que salían **22 motivos, 16 de ellos la misma frase** para los
    dientes mandibulares. Ninguno decía nada que los otros no dijeran, y entre ellos
    quedaban enterrados los que sí —el registro sin converger, la confianza bajo umbral—.
    Un aviso que salta en bloque es como se desactiva un gate.

    **No se ocultan.** La información sigue entera —qué arcada y qué códigos— y la
    confianza de esas observaciones sigue puesta a 0,0, así que tampoco se anclan. Lo que
    cambia es que se dice una vez.

    ⚠️ **La causa se afirma SOLO si se sabe.** Este mensaje decía «puede ser que no se
    aportara escaneo o que la segmentación fallara en ella» — una disyuntiva que el emisor
    sí puede resolver cuando le llega `arcada_aportada`: el orquestador ya recibe qué
    arcada trae el escáner intraoral (`fuse(..., arcada=...)`), así que sabía la respuesta
    y la estaba escondiendo detrás de un «puede ser». Sobre un caso real eran quince
    observaciones del informe declaradas como posible fallo de segmentación cuando lo que
    pasaba es que de esa arcada **no hay escaneo**: no hay nada que arreglar ahí.

    Sin `arcada_aportada` se mantiene la disyuntiva, que sigue siendo lo honesto cuando de
    verdad no se sabe.
    """
    return [
        f"arcada {arco}: la segmentación no cubrió NINGÚN diente, así que las "
        f"{len(codigos)} observación(es) del informe sobre ella se quedan sin anclar "
        f"(FDI {', '.join(sorted(codigos))}). " + (
            f"El escáner intraoral aportado es de la arcada {arcada_aportada}, así que de "
            f"esta NO hay geometría a la que anclarlas: no es un fallo de la segmentación "
            f"ni un desacuerdo diente a diente."
            if arcada_aportada and arcada_aportada != arco else
            "Puede ser que no se aportara escaneo de esa arcada o que la segmentación "
            "fallara en ella: no es un desacuerdo diente a diente."
        )
        for arco, codigos in sorted(sin_arcada.items())
    ]
