"""`clinical/observations.json`: lo que el contenedor dice de cada pieza.

Ver `uos.clinico`. Es EXTENSION nuestra: el spec v0.2 no da sitio a atributos clínicos
por diente y los manda a un servidor FHIR externo, que es justo lo que impide que un
`.uos` suelto responda «qué dice el informe del 24».
"""

from __future__ import annotations

from datetime import UTC, datetime

from core_schemas import (
    ClinicalAttributes,
    ColorCorona,
    Derivation,
    Hallazgo,
    Modality,
    Provenance,
    RegionalObservation,
    TwinSnapshot,
)
from uos.clinico import capa_clinica


def _snapshot(*observaciones: RegionalObservation) -> TwinSnapshot:
    return TwinSnapshot(
        acquisition_id="acq-1",
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
        regional=list(observaciones),
    )


def _obs(fdi: str, **attrs) -> RegionalObservation:
    return RegionalObservation(
        region_id=fdi,
        attributes=ClinicalAttributes(**attrs),
        timestamp=datetime.now(UTC),
        provenance=Provenance(
            source_file="informe.pdf", modality=Modality.REPORT,
            agent="report-agent@0.1.0", confidence=0.9,
            derivation=Derivation.DETERMINISTIC,
        ),
    )


_COLOR = ColorCorona(
    cervical=(55.4, 9.6, 22.0), medio=(58.1, 8.9, 21.0), incisal=(57.2, 9.0, 26.1),
    foto_sha256="a" * 64, n_pixeles=10205,
    correccion_iluminacion=(0.65, 0.58, 0.68),
)


def test_el_color_medido_sale_en_la_ficha_de_la_pieza() -> None:
    """⚠️ **El color tiene que ser consultable, no solo pintable.**

    El campo gaussiano lo lleva como píxeles y eso basta para que el visor lo enseñe, pero
    no para responder «de qué color es el 26» sin abrir un PLY de 8 MB y buscar sus
    gaussianas. Aquí es un dato con su soporte y su origen.
    """
    ficha = capa_clinica(_snapshot(_obs("26", color=_COLOR)), [])
    pieza = ficha["teeth"][0]
    assert pieza["fdi"] == "26"
    color = pieza["color"]
    assert color["space"] == "CIELAB"
    assert color["cervical"] == [55.4, 9.6, 22.0]
    assert color["n_pixels"] == 10205
    # ⚠️ El nombre del fichero NO viaja: los de una clínica llevan datos del paciente.
    assert color["from_photo"] == "sha256:" + "a" * 64
    # ⚠️ Y se dice lo que NO es: es color medido, no un tono de guía certificado.
    assert "NO es un tono" in color["note"]


def test_la_nota_del_color_dice_si_se_corrigio_la_iluminacion() -> None:
    """⚠️ **La nota se DERIVA de la medida; escrita a mano se queda mintiendo.**

    La que había aquí advertía de que «el flash cayendo hacia el fondo de la boca entra en
    el número», y siguió diciéndolo después de que el emisor empezara a descontar esa caída
    con la encía del propio paciente como referencia. Quien leyera el contenedor
    descartaría por artefacto una diferencia entre piezas que ya era real — y el caso donde
    se vio: `L*` recorría 22,7 puntos entre el 21 y el 27, y tras corregir recorre 5,6.

    Es la tercera vez en este proyecto que un descriptor escrito a mano se desincroniza de
    lo que describe. Por eso el test mira que la nota CAMBIE con el dato, no que contenga
    una frase concreta.
    """
    corregido = capa_clinica(_snapshot(_obs("26", color=_COLOR)), [])
    color_ok = corregido["teeth"][0]["color"]
    nota_ok = color_ok["note"]
    assert "0.65/0.58/0.68" in nota_ok
    # Y el número también viaja: leer una frase no debería ser la única forma de saber si
    # dos tonos se pueden poner uno al lado del otro.
    assert color_ok["illumination_slope"] == [0.65, 0.58, 0.68]
    assert "comparables entre si" in nota_ok

    crudo = _COLOR.model_copy(update={"correccion_iluminacion": None})
    sin = capa_clinica(_snapshot(_obs("26", color=crudo)), [])
    color_no = sin["teeth"][0]["color"]
    nota_no = color_no["note"]
    assert nota_no != nota_ok
    assert "illumination_slope" not in color_no
    # ⚠️ Ausente no es cero: una pieza sin corregir NO es comparable con una corregida, y
    # la nota tiene que decirlo, no callarse.
    assert "no es comparable" in nota_no


def test_una_pieza_sin_color_no_declara_el_campo() -> None:
    """Ausente no es lo mismo que nulo: si no se midió, no aparece."""
    ficha = capa_clinica(_snapshot(_obs("24", ph=6.2)), [])
    assert "color" not in ficha["teeth"][0]
    assert ficha["teeth"][0]["ph"] == 6.2


def test_el_color_y_el_informe_conviven_en_la_misma_pieza() -> None:
    """Son dos afirmaciones sobre el mismo diente y van juntas.

    Partirlas en dos entradas obligaría a quien lee a reunirlas, y `capa_clinica` agrupa
    por `region_id` — la segunda pisaría a la primera.
    """
    ficha = capa_clinica(
        _snapshot(_obs("26", hallazgos=[Hallazgo.RESTAURACION], color=_COLOR)), []
    )
    pieza = ficha["teeth"][0]
    assert pieza["findings"] == ["restauracion"]
    assert pieza["color"]["n_pixels"] == 10205


def test_la_capa_declara_su_estatuto_regulatorio() -> None:
    """Layer 1: transcripción de un informe firmado y medidas sobre fotos del paciente.

    NO es `derived/`: borrar lo inferido no puede llevarse por delante lo que dice el
    informe ni lo que se midió sobre una fotografía.
    """
    ficha = capa_clinica(_snapshot(_obs("11", ph=6.8)), ["revisar"])
    assert ficha["regulatory"]["layer"] == 1
    assert ficha["review"]["reasons"] == ["revisar"]
