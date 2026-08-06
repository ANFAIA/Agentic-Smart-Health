"""`segmentation-agent`: pone nombre de diente (FDI) a cada gaussiana.

Es el **ancla semántica** del pipeline. Sin `region_id`, la fusión semántica no
tiene contra qué validar el diente que cita el informe, y la capa regional (pH y
demás) queda colgando de un código que nadie ha confirmado que exista en la boca.

**Qué hace y qué no.** El *forward* del modelo de segmentación —Point Transformer
u otro, con GPU y `torch`— **no vive aquí**: entra como el `Protocol` `Segmenter`.
Es la misma costura que `Registrar` en la fusión geométrica, y por el mismo motivo:
el paquete se instala y se testea en el workspace normal, sin el venv de GPU, y el
modelo se puede cambiar sin tocar el agente. Lo que el agente sí posee es el salto
de **predicción por punto → dientes con código FDI, confianza y revisión humana**,
que es donde se pierden los proyectos que se quedan en la métrica del modelo.

El salto en sí lo hace `tooth-aggregation` (componentes conexas, fusión de
fragmentos, unicidad húngara opcional). Aquí se le añade lo que hace de eso un
agente del repositorio: leer el campo gaussiano del almacén, persistir las
etiquetas como artefacto nuevo, producir el mapa `FDI → confianza` que consume el
`semantic-fusion-agent`, y **declarar** lo que no se puede dar por bueno solo.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot
from ingestion_agents.ontology import all_fdi_codes
from pydantic import Field
from tooth_aggregation import ToothInstance, aggregate_teeth

from analysis_agents.base import AnalysisOutput, BaseAnalysisAgent

# Índice de la clase «encía»: no genera instancias. Convenio de `tooth-aggregation`.
GUM_CLASS = 0

# Mapeo clase → código FDI por defecto: el índice 0 es la encía y los 32 siguientes
# son la dentición permanente en orden de `all_fdi_codes()` (11…18, 21…28, …). Es un
# **convenio del modelo**, no una verdad clínica: un modelo entrenado con otro orden
# de clases necesita su propio `codes`, y por eso el parámetro existe.
DEFAULT_CODES: dict[int, int] = {i: int(c) for i, c in enumerate(all_fdi_codes(), start=1)}

# Por encima de esta fracción de puntos que el modelo llamó «diente» y que no
# acabaron en ninguna instancia, la segmentación salió fragmentada: la agregación
# los descarta por tamaño y lo haría **en silencio**. No es un fallo del algoritmo,
# es una señal de que el resultado no debería usarse sin mirarlo.
DEFAULT_UNASSIGNED_LIMIT = 0.10

# Tolerancia al comprobar que el segmentador devuelve log-probabilidades (∑ p = 1).
_LOGSUMEXP_TOL = 1e-3
# Filas que se comprueban como mucho. Muestreo con paso fijo, no aleatorio: la
# validación tiene que dar el mismo veredicto en cada corrida.
_LOGSUMEXP_SAMPLE = 1024


@runtime_checkable
class Segmenter(Protocol):
    """Modelo de segmentación de nubes de puntos.

    Recibe `(N, 3)` posiciones en mm y devuelve `(N, C)` **log-probabilidades**
    (salida de un `log_softmax`), con la columna `GUM_CLASS` para la encía.
    """

    def __call__(self, points: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class GaussianStore(Protocol):
    """Lo único que este agente necesita del almacén de artefactos.

    Se declara como `Protocol` en vez de importar `ArtifactStore`: el almacén es un
    *seam* que está previsto sustituir cuando exista `3dgs-engine` de verdad (ver el
    docstring de `ingestion_agents.store`), y el análisis no tiene por qué enterarse.
    """

    def load(self, ref: str) -> dict[str, np.ndarray]: ...

    def put(self, **arrays: np.ndarray) -> str: ...


class SegmentationOutput(AnalysisOutput):
    """`AnalysisOutput` más lo que la segmentación infiere: qué dientes hay."""

    detected: dict[str, float] = Field(
        default_factory=dict,
        description="Mapa `FDI → confianza` de los dientes encontrados. Es exactamente "
        "la entrada `detected` del `semantic-fusion-agent`.",
    )
    n_teeth: int = Field(default=0, ge=0)
    unassigned_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fracción de puntos clasificados como diente que no entraron en "
        "ninguna instancia (fragmentación descartada por tamaño).",
    )


class SegmentationAgent(BaseAnalysisAgent[SegmentationOutput]):
    """Segmenta el campo gaussiano y puebla `region_id` (FDI).

    **Entrada.** El `TwinSnapshot` con su `gaussian_field_ref`. Los puntos se leen
    del almacén, no se pasan por parámetro: el campo son cientos de miles de
    primitivas y el snapshot ya dice dónde están. Una referencia colgante es un
    error, no un campo vacío silencioso (invariante fail-loud del ADR 001) — y como
    el almacén lanza, el envoltorio lo convierte en `FAILED` con motivo.

    **Salida.** Un snapshot nuevo cuyo `gaussian_field_ref` apunta a un artefacto
    que tiene los mismos `centers`/`scales`/`rotations`/`density` **byte a byte**
    más un array `region_id`. La etiqueta es aditiva: no se pierde nada, el blob
    anterior sigue en el almacén, y como el almacén direcciona por contenido,
    volver a segmentar el mismo campo con el mismo modelo devuelve **la misma
    referencia**. En `region_id`, `0` significa *sin asignar* (encía, o diente
    descartado): el código FDI es el entero, y `str(...)` lo devuelve al `FDICode`
    del contrato.

    **De qué está hecha la confianza.** `ToothInstance.confidence` es la
    log-probabilidad **media** de la clase asignada sobre los puntos de la
    instancia; aquí se exponencia, así que la confianza de un diente es la **media
    geométrica de la probabilidad por punto**, en `[0, 1]`, comparable con el umbral
    de human-in-the-loop y con las confianzas del resto del pipeline. Que el
    segmentador devuelva log-probabilidades y no logits se **comprueba**: leer
    logits como log-probabilidades daría confianzas plausibles y falsas, que es el
    modo de fallo caro (el mismo que la modalidad DICOM en el `cbct-agent`).

    **Qué NO decide.** No corrige el informe ni al revés: el conflicto entre el FDI
    que dice el informe y el que dice el modelo lo resuelve —o mejor dicho, lo
    *declara*— el `semantic-fusion-agent`. Y `enforce_unique` viene desactivado a
    propósito: está medido que imponer «un FDI por arcada» sobre instancias
    fragmentadas **inventa** errores (ver `tooth_aggregation`).
    """

    name = "segmentation-agent"
    version = "0.1.0"
    # También en el camino de fallo se devuelve un `SegmentationOutput`: quien llama
    # puede leer `.detected` (vacío) sin preguntar antes si el agente falló.
    output_model = SegmentationOutput

    def __init__(
        self,
        store: GaussianStore,
        *,
        segmenter: Segmenter,
        codes: Mapping[int, int] | None = None,
        min_size: int = 30,
        merge_mult: float = 12.0,
        enforce_unique: bool = False,
        unassigned_limit: float = DEFAULT_UNASSIGNED_LIMIT,
        quarantine_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(quarantine_dir=quarantine_dir, **kwargs)
        self.store = store
        # Sin modelo no hay segmentación. No hay valor por defecto a propósito: un
        # segmentador «de juguete» por omisión produciría etiquetas anatómicas
        # inventadas con toda la pinta de ser buenas.
        self.segmenter = segmenter
        self.codes = dict(DEFAULT_CODES if codes is None else codes)
        self.min_size = min_size
        self.merge_mult = merge_mult
        self.enforce_unique = enforce_unique
        self.unassigned_limit = unassigned_limit

    # --- análisis --------------------------------------------------------- #
    def _analyze(self, snapshot: TwinSnapshot, **inputs: Any) -> SegmentationOutput:
        campo = self.store.load(snapshot.gaussian_field_ref)
        if "centers" not in campo:
            raise ValueError(
                f"El artefacto {snapshot.gaussian_field_ref} no tiene `centers`: "
                "no es un campo gaussiano."
            )
        puntos = np.asarray(campo["centers"], dtype=np.float64)
        logprob = self._logprob(puntos)

        instancias = aggregate_teeth(
            puntos,
            logprob,
            gum_class=GUM_CLASS,
            min_size=self.min_size,
            merge_mult=self.merge_mult,
            enforce_unique=self.enforce_unique,
            codes=self.codes,
        )
        if not instancias:
            # La entrada estaba y el modelo corrió: no es un fallo, es que no hay
            # ancla que producir. Se declara para que la fusión semántica no crea
            # que la segmentación pasó por aquí y no encontró conflicto ninguno.
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    f"El segmentador no encontró ningún diente de al menos "
                    f"{self.min_size} puntos sobre {len(puntos)} primitivas."
                ),
                hitl_reasons=["La segmentación no produjo ningún diente: no hay ancla FDI."],
            )

        etiquetas = logprob.argmax(axis=1)
        region = self._region_array(len(puntos), instancias)
        detected, motivos = self._detectados(instancias)
        sin_asignar = self._fraccion_sin_asignar(etiquetas, instancias)
        motivos += self._motivos_de_cobertura(sin_asignar)

        return self._outcome(
            ModalityStatus.OK,
            snapshot=self._emitir(snapshot, campo, region),
            detected=detected,
            n_teeth=len(instancias),
            unassigned_fraction=sin_asignar,
            hitl_reasons=motivos,
            detail=(
                f"{len(instancias)} diente(s) segmentado(s), {len(detected)} código(s) FDI; "
                f"{sin_asignar:.1%} de los puntos de diente sin asignar."
            ),
        )

    # --- piezas ----------------------------------------------------------- #
    def _logprob(self, puntos: np.ndarray) -> np.ndarray:
        """Llama al modelo y comprueba que devuelve lo que dice devolver."""
        logprob = np.asarray(self.segmenter(puntos), dtype=np.float64)
        if logprob.ndim != 2 or len(logprob) != len(puntos):
            raise ValueError(
                f"El segmentador debe devolver (N, C) con N={len(puntos)}; "
                f"devolvió {logprob.shape}."
            )
        if logprob.shape[1] < 2:
            raise ValueError(
                f"El segmentador devolvió {logprob.shape[1]} clase(s): hacen falta al "
                "menos dos (encía y algún diente)."
            )
        if not np.isfinite(logprob).all():
            raise ValueError("El segmentador devolvió valores no finitos (NaN/inf).")

        # Un logit tiene la misma forma que una log-probabilidad y se comporta igual
        # en el `argmax`, así que las etiquetas saldrían bien y las **confianzas**
        # saldrían mal sin que nada chille. Se comprueba ∑ⱼ exp(logprob) = 1.
        paso = max(1, len(logprob) // _LOGSUMEXP_SAMPLE)
        muestra = logprob[::paso]
        maximo = muestra.max(axis=1, keepdims=True)
        lse = (maximo + np.log(np.exp(muestra - maximo).sum(axis=1, keepdims=True))).ravel()
        if not np.allclose(lse, 0.0, atol=_LOGSUMEXP_TOL):
            peor = float(np.abs(lse).max())
            raise ValueError(
                "El segmentador debe devolver log-probabilidades normalizadas "
                f"(`log_softmax`), no logits: log ∑ exp se desvía hasta {peor:.3f} de 0. "
                "Las etiquetas saldrían bien y las confianzas serían falsas."
            )
        return logprob

    def _region_array(self, n: int, instancias: list[ToothInstance]) -> np.ndarray:
        """`region_id` por gaussiana. `0` = sin asignar (encía o diente descartado)."""
        region = np.zeros(n, dtype=np.int16)
        for inst in instancias:
            if inst.fdi is not None:
                region[inst.vertices] = inst.fdi
        return region

    def _detectados(self, instancias: list[ToothInstance]) -> tuple[dict[str, float], list[str]]:
        """Mapa `FDI → confianza` y los motivos de revisión que salen de él.

        Con un código repetido en dos instancias se conserva **la más fiable** y se
        pide revisión: no se puede elegir en silencio cuál de los dos trozos es el
        diente de verdad, y bajar la confianza a 0 escondería que el modelo sí
        encontró algo ahí.
        """
        detected: dict[str, float] = {}
        motivos: list[str] = []
        repetidos: set[str] = set()

        for inst in instancias:
            if inst.fdi is None:
                motivos.append(
                    f"La clase {inst.label} ({inst.size} puntos) no tiene código FDI en el "
                    "mapeo del modelo: esas gaussianas quedan sin `region_id`."
                )
                continue
            fdi = str(inst.fdi)
            confianza = float(np.exp(inst.confidence))
            if fdi in detected:
                repetidos.add(fdi)
                confianza = max(confianza, detected[fdi])
            detected[fdi] = confianza

        for fdi in sorted(repetidos):
            motivos.append(
                f"FDI {fdi}: el modelo lo asignó a más de una instancia; se conserva la "
                "más fiable, pero una de las dos no es ese diente."
            )
        for fdi, confianza in sorted(detected.items()):
            if confianza < self.hitl_threshold:
                motivos.append(
                    f"FDI {fdi}: confianza {confianza:.2f} bajo el umbral "
                    f"{self.hitl_threshold:.2f}"
                )
        return detected, motivos

    def _fraccion_sin_asignar(
        self, etiquetas: np.ndarray, instancias: list[ToothInstance]
    ) -> float:
        """Cuánto de lo que el modelo llamó «diente» se quedó fuera de toda instancia.

        No lleva guarda de división por cero: solo se llama con `instancias` no
        vacía, y una instancia existe únicamente si al menos `min_size` puntos no
        son encía. El denominador es por tanto ≥ `min_size`.
        """
        predichos = int(np.count_nonzero(etiquetas != GUM_CLASS))
        asignados = sum(inst.size for inst in instancias)
        return max(0.0, 1.0 - asignados / predichos)

    def _motivos_de_cobertura(self, sin_asignar: float) -> list[str]:
        if sin_asignar <= self.unassigned_limit:
            return []
        return [
            f"{sin_asignar:.1%} de los puntos clasificados como diente no entraron en "
            f"ninguna instancia (límite {self.unassigned_limit:.0%}): la segmentación "
            "salió fragmentada."
        ]

    def _emitir(
        self, snapshot: TwinSnapshot, campo: dict[str, np.ndarray], region: np.ndarray
    ) -> TwinSnapshot:
        """Snapshot **nuevo** apuntando al campo etiquetado (ADR 004 §2.5).

        En la `Provenance` solo se reescribe `agent`. La confianza **por diente**
        vive en `detected`, y machacar la del snapshot con un único número —la del
        peor diente, o una media— perdería precisamente la información que el gate
        de human-in-the-loop necesita para saber *qué* revisar.
        """
        ref = self.store.put(**{**campo, "region_id": region})
        prov = snapshot.provenance.model_copy(update={"agent": self.qualified})
        return snapshot.model_copy(update={"gaussian_field_ref": ref, "provenance": prov})
