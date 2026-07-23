"""`report-agent`: extracción regional (pH por FDI) con el backend determinista."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from core_schemas import Modality, ModalityStatus, Support
from ingestion_agents import ReportAgent
from ingestion_agents.report_agent import (
    extract_ph_by_rules,
    extract_text,
    report_date,
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
    assert report_date("Informe sin fecha") is None


def test_fecha_imposible_se_ignora() -> None:
    assert report_date("Fecha: 2026-13-45") is None


def test_extract_text_rechaza_formatos_desconocidos(tmp_path: Path) -> None:
    path = tmp_path / "informe.docx"
    path.write_bytes(b"PK")
    with pytest.raises(ValueError, match="no sabe leer"):
        extract_text(path)


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
