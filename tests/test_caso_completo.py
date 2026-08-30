"""El color medido, desde que se mide hasta que entra en el snapshot.

Esto existe por un fallo concreto: `_con_color` usaba `dataclasses.replace` sobre un
`TwinSnapshot`, que es un modelo **pydantic**. La excepción caía dentro del `except` de la
apariencia, se anunciaba como «Error entrenando apariencia» y el contenedor salía completo
—con su PSNR, su cabecera y sus 13 coronas medidas en el PLY— y sin una sola pieza con
color en `clinical/observations.json`.

⚠️ **La prueba de `capa_clinica` no lo cogía y no podía cogerlo**: se le pasa a mano un
snapshot que ya trae el color. Lo que faltaba probar es el paso ANTERIOR, el que mete el
color en el snapshot.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core_schemas import (
    ClinicalAttributes,
    Derivation,
    Hallazgo,
    Modality,
    Provenance,
    RegionalObservation,
    TwinSnapshot,
)

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "caso_completo.py"
_spec = importlib.util.spec_from_file_location("caso_completo", _RUTA)
assert _spec and _spec.loader
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


@dataclass
class _Tono:
    """Lo que `tono_foto` devuelve por corona, con lo justo para esta prueba."""

    fdi: int
    lab: tuple
    n_pixeles: int
    foto_sha256: str
    correccion: tuple | None = None
    """La pendiente con la que se descontó la caída del flash. Ver
    `gaussian_engine.tono_foto.ajuste_de_iluminacion`."""


def _tono(fdi: int) -> _Tono:
    return _Tono(
        fdi=fdi,
        lab=((55.4, 9.6, 22.0), (58.1, 8.9, 21.0), (57.2, 9.0, 26.1)),
        n_pixeles=10205,
        foto_sha256="a" * 64,
        correccion=(0.65, 0.58, 0.68),
    )


def _snapshot(*observaciones: RegionalObservation) -> TwinSnapshot:
    return TwinSnapshot(
        acquisition_id="acq-1",
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
        regional=list(observaciones),
    )


def _del_informe(fdi: str) -> RegionalObservation:
    return RegionalObservation(
        region_id=fdi,
        attributes=ClinicalAttributes(hallazgos=[Hallazgo.RESTAURACION]),
        timestamp=datetime.now(UTC),
        provenance=Provenance(
            source_file="informe.pdf", modality=Modality.REPORT,
            agent="report-agent@0.1.0", confidence=0.9,
            derivation=Derivation.DETERMINISTIC,
        ),
    )


def test_el_color_llega_al_snapshot() -> None:
    """⚠️ **La que faltaba.** Un snapshot pydantic no se copia con `dataclasses.replace`."""
    fus = cc._con_color(_snapshot(), [_tono(26), _tono(11)])
    con_color = {o.region_id for o in fus.regional if o.attributes.color is not None}
    assert con_color == {"26", "11"}


def test_el_color_se_anade_a_la_observacion_del_informe_en_vez_de_duplicarla() -> None:
    """Son dos afirmaciones sobre el mismo diente; `capa_clinica` agrupa por `region_id`.

    Si se creara una entrada aparte, la segunda pisaría a la primera y el contenedor
    perdería o el hallazgo o el color, según el orden.
    """
    fus = cc._con_color(_snapshot(_del_informe("26")), [_tono(26)])
    assert len(fus.regional) == 1
    obs = fus.regional[0]
    assert obs.attributes.hallazgos == [Hallazgo.RESTAURACION]
    assert obs.attributes.color.n_pixeles == 10205
    # La procedencia sigue siendo la del informe: el color no la reescribe.
    assert obs.provenance.modality is Modality.REPORT


def test_una_pieza_que_el_informe_no_menciona_estrena_observacion() -> None:
    """Con lo único que se sabe de ella, y declarando de qué foto salió."""
    fus = cc._con_color(_snapshot(_del_informe("26")), [_tono(17)])
    nueva = next(o for o in fus.regional if o.region_id == "17")
    assert nueva.provenance.modality is Modality.IMAGE
    assert nueva.provenance.source_file == "sha256:" + "a" * 64
    assert nueva.provenance.derivation is Derivation.DETERMINISTIC


def test_sin_tonos_el_snapshot_sale_intacto() -> None:
    """No medir color no es medirlo vacío: el snapshot es el mismo objeto."""
    antes = _snapshot(_del_informe("26"))
    assert cc._con_color(antes, []) is antes
    assert cc._con_color(None, [_tono(26)]) is None


def test_la_correccion_de_iluminacion_viaja_con_el_color() -> None:
    """⚠️ **Sin este campo, dos tonos que no son comparables lo parecen.**

    Una pieza corregida y una cruda llevan escalas distintas: la cruda tiene dentro lo
    lejos que le llegó el flash. En el caso real eso valía 22,7 puntos de `L*` — el 21
    salía blanco y el 27 marrón siendo la misma boca. Si el campo no viajase, quien lea el
    contenedor no podría saber cuál de las dos cosas está mirando.
    """
    tono = _tono(11)
    snap = cc._con_color(_snapshot(), [tono])
    color = snap.regional[0].attributes.color
    assert color.correccion_iluminacion == tono.correccion

    crudo = _Tono(fdi=21, lab=tono.lab, n_pixeles=99, foto_sha256="b" * 64)
    snap2 = cc._con_color(_snapshot(), [crudo])
    # Ausente, NO cero: «no se corrigió» y «se corrigió con pendiente 0» son cosas
    # distintas, y sólo la primera hace el dato incomparable con el resto.
    assert snap2.regional[0].attributes.color.correccion_iluminacion is None
