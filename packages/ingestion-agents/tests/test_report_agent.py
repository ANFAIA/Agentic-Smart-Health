"""`report-agent`: extracción regional (pH por FDI) con el backend determinista."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from core_schemas import (
    ClinicalAttributes,
    Derivation,
    Hallazgo,
    Modality,
    ModalityStatus,
    Support,
)
from ingestion_agents import ArtifactStore, ReportAgent
from ingestion_agents.report_agent import (
    _EXTRACTION_TOOL,
    _limite,
    extract_ph_by_rules,
    extract_text,
    report_date,
    valida_propuestas,
)


# --- extracción por reglas ------------------------------------------------- #
def test_formato_tabulado() -> None:
    assert extract_ph_by_rules("Diente 16: pH 5.4").findings == {"16": 5.4}


def test_coma_decimal() -> None:
    """Los informes en español escriben `5,4`."""
    assert extract_ph_by_rules("Diente 26 — pH 5,9").findings == {"26": 5.9}


def test_notacion_con_punto() -> None:
    """FDI también se escribe `1.6`; no debe confundirse con un decimal."""
    assert extract_ph_by_rules("Diente 1.6 presenta pH 6.2").findings == {"16": 6.2}


def test_orden_invertido() -> None:
    assert extract_ph_by_rules("pH 6.1 en el diente 36").findings == {"36": 6.1}


@pytest.mark.parametrize("sep", [":", "=", ""])
def test_separadores(sep: str) -> None:
    assert extract_ph_by_rules(f"Diente 21 pH{sep} 6.8").findings == {"21": 6.8}


def test_varias_lineas() -> None:
    texto = "Diente 16: pH 5.2\nDiente 21: pH 6.8\nDiente 26: pH 5.9\n"
    assert extract_ph_by_rules(texto).findings == {"16": 5.2, "21": 6.8, "26": 5.9}


def test_cada_ph_se_empareja_con_el_diente_de_su_linea() -> None:
    """El fallo silencioso a evitar: colgar un pH del diente equivocado."""
    texto = "Diente 16: pH 5.2\nDiente 47: pH 7.0\n"
    assert extract_ph_by_rules(texto).findings == {"16": 5.2, "47": 7.0}


def test_linea_sin_diente_se_descarta() -> None:
    """Mejor no ingerir que ingerir mal."""
    assert extract_ph_by_rules("pH medio de la arcada: 6.4").findings == {}


def test_linea_sin_ph_se_descarta() -> None:
    assert extract_ph_by_rules("Diente 16 con restauración de composite.").findings == {}


def test_valor_fuera_de_rango_se_descarta() -> None:
    """`7.4` mal leído como `74`: lo caza la ontología antes del contrato."""
    assert extract_ph_by_rules("Diente 16: pH 74").findings == {}
    assert extract_ph_by_rules("Diente 16: pH 1.2").findings == {}


def test_codigo_fdi_inexistente_se_descarta() -> None:
    assert extract_ph_by_rules("Diente 19: pH 6.0").findings == {}


def test_los_digitos_del_ph_no_se_leen_como_diente() -> None:
    """`pH 5.4` no debe interpretarse como el diente 54."""
    assert extract_ph_by_rules("pH 5.4").findings == {}


# --- descartes: lo que se cae NO puede caerse en silencio ------------------- #
def test_un_ph_fuera_de_rango_se_registra_como_descarte() -> None:
    """El caso real: «pH 74» es un 7.4 mal tecleado. Rechazarlo está bien;
    perderlo sin dejar rastro, no."""
    discards = extract_ph_by_rules("Diente 47: pH 74").discards
    assert len(discards) == 1
    assert "fuera del rango plausible" in discards[0].reason
    assert "Diente 47" in discards[0].line


def test_un_fdi_inexistente_se_registra_como_descarte() -> None:
    discards = extract_ph_by_rules("Diente 19: pH 6.0").discards
    assert len(discards) == 1
    assert "FDI inexistente" in discards[0].reason


def test_un_ph_sin_diente_se_registra_como_descarte() -> None:
    """Distinto motivo que el FDI inválido: aquí no había ningún diente."""
    discards = extract_ph_by_rules("pH 6.4 de media en la arcada superior").discards
    assert len(discards) == 1
    assert "sin diente asociado" in discards[0].reason


def test_un_ph_separado_de_su_etiqueta_no_se_detecta_siquiera() -> None:
    """Límite conocido del backend `rules`: el patrón exige el valor pegado a
    «pH». En «pH medio de la arcada: 6.4» el número queda demasiado lejos, así
    que la línea no llega ni a considerarse candidata — no hay hallazgo, pero
    tampoco descarte que declarar. Es justo el tipo de prosa que motiva el
    backend LLM."""
    extraction = extract_ph_by_rules("pH medio de la arcada: 6.4")
    assert extraction.findings == {}
    assert extraction.discards == []


def test_un_diente_repetido_se_registra_como_descarte() -> None:
    """Dos pH para el mismo diente: se conserva el primero y se declara el otro."""
    extraction = extract_ph_by_rules("Diente 16: pH 5.4\nDiente 16: pH 6.9")
    assert extraction.findings == {"16": 5.4}
    assert len(extraction.discards) == 1
    assert "ya tenía un pH" in extraction.discards[0].reason


def test_una_linea_sin_ph_no_es_un_descarte() -> None:
    """Solo es descarte lo que *parecía* un hallazgo; el texto normal, no."""
    texto = "Revisión rutinaria. Diente 16 con restauración de composite."
    assert extract_ph_by_rules(texto).discards == []


def test_un_informe_limpio_no_descarta_nada() -> None:
    extraction = extract_ph_by_rules("Diente 16: pH 5.2\nDiente 21: pH 6.8")
    assert len(extraction.findings) == 2
    assert extraction.discards == []


# --- metadatos ------------------------------------------------------------- #
def test_fecha_del_informe() -> None:
    assert report_date("Fecha: 2026-03-14") == datetime(2026, 3, 14, tzinfo=UTC)


def test_sin_fecha() -> None:
    assert report_date("Report sin fecha") is None


def test_fecha_imposible_se_ignora() -> None:
    assert report_date("Fecha: 2026-13-45") is None


def test_extract_text_rechaza_formatos_desconocidos(tmp_path: Path) -> None:
    path = tmp_path / "informe.docx"
    path.write_bytes(b"PK")
    with pytest.raises(ValueError, match="no sabe leer"):
        extract_text(path)


# --- PDF (el informe clínico llega como PDF: entregable Semana 3-4) ---------- #
def _make_pdf(text: str) -> bytes:
    """PDF de una página con una línea de texto, sin dependencias de generación.

    Construye los objetos mínimos (catálogo, páginas, página, contenido, fuente)
    con la tabla xref de offsets correcta, para tener un PDF real y extraíble en
    los tests sin arrastrar reportlab/fpdf.
    """
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        None,  # contenido (stream), se rellena abajo
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    stream = b"BT /F1 12 Tf 20 150 Td (" + text.encode("latin-1") + b") Tj ET"
    objs[3] = b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream"
    out = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj" + obj + b"endobj\n"  # type: ignore[operator]
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer<</Size " + str(len(objs) + 1).encode() + b"/Root 1 0 R>>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    )
    return out


def test_extract_text_lee_pdf(tmp_path: Path) -> None:
    path = tmp_path / "informe.pdf"
    path.write_bytes(_make_pdf("Diente 16: pH 5.4 - riesgo"))
    assert "pH 5.4" in extract_text(path)


def test_ingesta_de_informe_pdf_extrae_ph(tmp_path: Path) -> None:
    """El camino completo del entregable: informe clínico en PDF → RegionalObservation."""
    path = tmp_path / "informe.pdf"
    path.write_bytes(_make_pdf("Fecha: 2026-05-09 Diente 16: pH 5.4"))
    outcome = ReportAgent().ingest(path)
    assert outcome.status is ModalityStatus.OK
    assert {o.region_id: o.attributes.ph for o in outcome.regional} == {"16": 5.4}
    assert outcome.regional[0].timestamp == datetime(2026, 5, 9, tzinfo=UTC)


def test_pdf_ilegible_falla_sin_lanzar(tmp_path: Path) -> None:
    """Un PDF corrupto es fail-loud como cualquier otro fallo de ingesta."""
    path = tmp_path / "roto.pdf"
    path.write_bytes(b"%PDF-1.4 basura no es un pdf valido")
    outcome = ReportAgent().ingest(path)
    assert outcome.status is ModalityStatus.FAILED


# --- agente ---------------------------------------------------------------- #
def test_ingesta_del_informe_sintetico(report_path: Path) -> None:
    outcome = ReportAgent().ingest(report_path)

    assert outcome.status is ModalityStatus.OK
    assert outcome.modality is Modality.REPORT
    assert outcome.support is Support.REGIONAL
    assert {o.region_id for o in outcome.regional} == {"11", "16", "21", "26"}
    assert outcome.artifact_ref is None  # el soporte regional no tiene blob pesado


def test_cada_observacion_lleva_su_provenance(report_path: Path) -> None:
    """Trazabilidad por valor: cada pH sabe de qué fichero y qué agente vino."""
    for obs in ReportAgent().ingest(report_path).regional:
        assert obs.provenance.agent == "report-agent@0.1.0"
        assert obs.provenance.modality is Modality.REPORT
        assert obs.provenance.source_file == str(report_path)


def test_la_fecha_del_informe_data_las_observaciones(tmp_path: Path) -> None:
    """Fechar mal una observación desordena la serie temporal del paciente."""
    path = tmp_path / "informe.txt"
    path.write_text("Fecha: 2026-03-14\nDiente 16: pH 5.4\n", encoding="utf-8")
    obs = ReportAgent().ingest(path).regional[0]
    assert obs.timestamp == datetime(2026, 3, 14, tzinfo=UTC)


def test_sin_fecha_usa_la_de_respaldo(tmp_path: Path) -> None:
    path = tmp_path / "informe.txt"
    path.write_text("Diente 16: pH 5.4\n", encoding="utf-8")
    fallback = datetime(2020, 1, 1, tzinfo=UTC)
    obs = ReportAgent(default_timestamp=fallback).ingest(path).regional[0]
    assert obs.timestamp == fallback


def test_un_descarte_parcial_baja_la_confianza_y_lo_declara(tmp_path: Path) -> None:
    """El fallo que esto corrige: antes el informe se ingería `ok` con confianza
    0.9 y los hallazgos descartados desaparecían sin dejar rastro."""
    path = tmp_path / "informe.txt"
    path.write_text(
        "Diente 16: pH 5.1\nDiente 19: pH 6.0\nDiente 47: pH 74\n", encoding="utf-8"
    )

    outcome = ReportAgent().ingest(path)

    assert outcome.ok
    assert {o.region_id for o in outcome.regional} == {"16"}
    assert outcome.provenance is not None
    # Por debajo del umbral del gate humano del orquestador (0.7).
    assert outcome.provenance.confidence == 0.6
    assert "2 hallazgos descartados" in (outcome.detail or "")
    assert "Diente 19" in (outcome.detail or "")
    assert "Diente 47" in (outcome.detail or "")


def test_un_informe_limpio_conserva_la_confianza_alta(tmp_path: Path) -> None:
    path = tmp_path / "informe.txt"
    path.write_text("Diente 16: pH 5.1\nDiente 21: pH 6.8\n", encoding="utf-8")
    outcome = ReportAgent().ingest(path)
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.9
    assert outcome.detail is None


def test_informe_sin_hallazgos_baja_la_confianza_a_cero(tmp_path: Path) -> None:
    """Un PDF escaneado sin OCR no es un éxito vacío: debe parar en el gate humano."""
    path = tmp_path / "informe.txt"
    path.write_text("Revisión rutinaria sin incidencias.", encoding="utf-8")
    outcome = ReportAgent().ingest(path)
    assert outcome.ok
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.0
    assert "No se extrajo" in (outcome.detail or "")


def test_informe_vacio_falla(tmp_path: Path) -> None:
    path = tmp_path / "informe.txt"
    path.write_text("   \n", encoding="utf-8")
    outcome = ReportAgent().ingest(path)
    assert outcome.status is ModalityStatus.FAILED
    assert "texto extraíble" in (outcome.detail or "")


def test_backend_desconocido_se_rechaza_al_construir() -> None:
    with pytest.raises(ValueError, match="Backend desconocido"):
        ReportAgent(backend="magia")


def test_el_backend_llm_sin_clave_falla_declarando(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aun con LLM, un fallo es un dato: no escapa como excepción."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / "informe.txt"
    path.write_text("Diente 16: pH 5.4\n", encoding="utf-8")
    outcome = ReportAgent(backend="llm").ingest(path)
    assert outcome.status is ModalityStatus.FAILED


# --- backend LLM: la política, sin red ------------------------------------- #
# `valida_propuestas` es pura a propósito, así que todo lo que decide qué entra al
# contrato se prueba sin `anthropic` instalado y sin clave. Lo único que queda sin
# cubrir es la llamada HTTP, que no toma ninguna decisión.
def test_un_diente_completo_pasa_entero() -> None:
    extraccion = valida_propuestas([
        {"fdi": "16", "ph": 5.2, "n_raices": 3, "n_conductos": 4,
         "hallazgos": ["caries"], "confianza": 0.8},
    ])
    assert extraccion.findings == {"16": (5.2, 0.8)}
    assert extraccion.clinicos == {
        "16": {"n_raices": 3, "n_conductos": 4, "hallazgos": [Hallazgo.CARIES]}
    }
    assert extraccion.discards == []


def test_un_campo_malo_no_se_lleva_los_buenos_del_mismo_diente() -> None:
    """El motivo de validar campo a campo: «3 raíces, pH 74» son dos datos buenos."""
    extraccion = valida_propuestas([
        {"fdi": "16", "ph": 74, "n_raices": 3, "n_conductos": 4, "confianza": 0.9},
    ])
    assert extraccion.findings == {}
    assert extraccion.clinicos == {"16": {"n_raices": 3, "n_conductos": 4}}
    assert len(extraccion.discards) == 1
    assert "fuera del rango plausible" in extraccion.discards[0].reason


def test_un_fdi_inexistente_tumba_el_diente_y_lo_registra() -> None:
    extraccion = valida_propuestas([{"fdi": "19", "ph": 6.0, "confianza": 0.9}])
    assert extraccion.findings == {} and extraccion.clinicos == {}
    assert "FDI inexistente" in extraccion.discards[0].reason


@pytest.mark.parametrize(
    ("campo", "valor"), [("n_raices", 9), ("n_conductos", 0), ("n_raices", -1)]
)
def test_un_entero_fuera_del_rango_del_contrato_se_descarta(campo: str, valor: int) -> None:
    """Y se descarta AQUÍ: dejarlo pasar lo convertiría en un `ValidationError` que
    tumbaría el informe entero a `FAILED` por un solo valor."""
    extraccion = valida_propuestas([{"fdi": "16", campo: valor, "confianza": 0.9}])
    assert extraccion.clinicos == {}
    assert "fuera del rango del contrato" in extraccion.discards[0].reason


def test_un_valor_implausible_no_impide_construir_el_contrato() -> None:
    """La consecuencia de lo anterior, comprobada de verdad."""
    extraccion = valida_propuestas([
        {"fdi": "16", "n_raices": 9, "n_conductos": 4, "confianza": 0.9},
    ])
    assert ClinicalAttributes(**extraccion.clinicos["16"]).n_conductos == 4


def test_un_hallazgo_fuera_del_vocabulario_cae_y_los_validos_siguen() -> None:
    extraccion = valida_propuestas([
        {"fdi": "21", "hallazgos": ["caries", "fractura_radicular"], "confianza": 0.7},
    ])
    assert extraccion.clinicos == {"21": {"hallazgos": [Hallazgo.CARIES]}}
    assert "fuera del vocabulario controlado" in extraccion.discards[0].reason


def test_un_diente_sin_nada_declarado_no_crea_entrada() -> None:
    """Devolver el campo vacío es correcto; inventarlo no. Tampoco lo es apuntarlo."""
    extraccion = valida_propuestas([{"fdi": "16", "hallazgos": [], "confianza": 0.9}])
    assert extraccion.findings == {} and extraccion.clinicos == {}
    assert extraccion.discards == []


def test_un_ph_repetido_se_registra_como_descarte() -> None:
    extraccion = valida_propuestas([
        {"fdi": "16", "ph": 5.2, "confianza": 0.9},
        {"fdi": "16", "ph": 6.4, "confianza": 0.9},
    ])
    assert extraccion.findings == {"16": (5.2, 0.9)}
    assert "ya tenía un pH" in extraccion.discards[0].reason


@pytest.mark.parametrize(("declarada", "esperada"), [(None, 0.5), (1.4, 1.0), (-2, 0.0)])
def test_la_confianza_se_acota_y_tiene_respaldo(declarada: float | None, esperada: float) -> None:
    """Sin confianza declarada no hay extracción segura: el respaldo es bajo aposta."""
    propuesta = {"fdi": "16", "ph": 5.2}
    if declarada is not None:
        propuesta["confianza"] = declarada
    assert valida_propuestas([propuesta]).findings["16"][1] == esperada


def test_los_limites_los_pone_el_contrato_y_no_una_copia() -> None:
    """Si alguien cambia el rango en `core-schemas`, la barrera del agente le sigue."""
    assert _limite("n_raices") == (1, 5)
    assert _limite("n_conductos") == (1, 8)


def test_el_esquema_de_la_tool_declara_el_vocabulario_completo() -> None:
    """El `enum` se genera desde `Hallazgo`: no puede quedarse atrás del contrato."""
    diente = _EXTRACTION_TOOL["input_schema"]["properties"]["dientes"]["items"]  # type: ignore[index]
    assert set(diente["properties"]["hallazgos"]["items"]["enum"]) == {
        h.value for h in Hallazgo
    }
    # El pH dejó de ser obligatorio: un informe de CBCT con IA no lo mide.
    assert diente["required"] == ["fdi", "confianza"]


# --- procedencia: qué cerebro produjo cada valor --------------------------- #
def test_el_backend_de_reglas_declara_determinismo(tmp_path: Path) -> None:
    path = tmp_path / "informe.txt"
    path.write_text("Fecha: 2026-03-14\nDiente 16: pH 5.4\n", encoding="utf-8")
    prov = ReportAgent().ingest(path).regional[0].provenance
    assert prov.derivation is Derivation.DETERMINISTIC
    assert prov.model is None


def test_el_backend_de_modelo_declara_cuál() -> None:
    """No basta con «inferido»: dentro de un año importará saber con qué modelo.

    El twin persistido es lo único que quedará para distinguir un `claude-sonnet-5`
    de cualquier otro cerebro que se enchufe aquí más adelante.
    """
    assert ReportAgent(backend="llm")._derivation() == (
        Derivation.INFERRED,
        "llm:claude-sonnet-5",
    )


def test_los_indices_no_son_inferidos_ni_con_backend_de_modelo(tmp_path: Path) -> None:
    """El caso que obliga a que `derivation` sea por valor y no por agente.

    `extract_medidas_by_rules` corre con los dos backends. Si la procedencia se
    tomase del agente, un informe ingerido con modelo llevaría sus índices de
    oclusión marcados como inferidos — mintiendo en el sentido contrario al hueco
    que este campo vino a tapar.
    """
    agente = ReportAgent(backend="llm")
    prov = agente._provenance(
        tmp_path / "informe.txt",
        confidence=0.9,
        derivation=Derivation.DETERMINISTIC,
    )
    assert prov.derivation is Derivation.DETERMINISTIC
    assert prov.model is None
    # …mientras que lo que sí produce el modelo se declara inferido.
    assert agente._provenance(tmp_path / "informe.txt").model == "llm:claude-sonnet-5"


def test_los_agentes_deterministas_lo_declaran(mesh_path: Path, store: ArtifactStore) -> None:
    """El defecto afirma determinismo porque para estos agentes es verdad."""
    from ingestion_agents import MeshAgent

    prov = MeshAgent(store).ingest(mesh_path).provenance
    assert prov is not None
    assert prov.derivation is Derivation.DETERMINISTIC
