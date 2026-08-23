"""El backend determinista contra el corpus de evaluación (`report_corpus.py`).

`test_report_agent.py` comprueba patrones sueltos —una línea, un resultado—; esto
recorre **informes enteros** y fija dos cosas distintas:

1. **El suelo, exactamente donde está.** Cada caso declara lo que el regex saca hoy;
   si mañana saca otra cosa, este test lo dice. Es lo que convierte «el determinista
   es el suelo medible» de un comentario en una cifra que se puede citar.
2. **Que el suelo no baje por el lado caro.** Los falsos positivos del corpus están
   contados uno a uno. Añadir un patrón que mejore la cobertura y de paso invente un
   dato más rompe la suite, que es justo lo que debe pasar: un valor falso dentro del
   contrato es peor que un hueco, porque el hueco se ve.

Que un caso tenga `limite` documentado **no** lo convierte en un fallo tolerado: es
una deuda escrita, con su motivo, contra la que se medirá el backend `llm`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core_schemas import ModalityStatus
from ingestion_agents import ArtifactStore, report_corpus
from ingestion_agents.report_agent import (
    ReportAgent,
    extract_hallazgos_by_rules,
    extract_medidas_by_rules,
    extract_ph_by_rules,
)
from ingestion_agents.report_corpus import CASES, ReportCase, normaliza, puntua

CASOS = [pytest.param(caso, id=caso.name) for caso in CASES]

# Falsos positivos que el backend determinista produce HOY sobre el corpus: valores
# que llegan al contrato sin que el informe los diga, o diciéndolos de otra forma.
# El número está aquí y no calculado sobre la marcha a propósito — es una cifra que
# solo debe moverse hacia abajo, y moverla exige tocar esta línea y explicar por qué.
FALSOS_POSITIVOS_RULES = 3


# --- el suelo, caso a caso -------------------------------------------------- #
@pytest.mark.parametrize("caso", CASOS)
def test_el_determinista_saca_exactamente_lo_declarado(caso: ReportCase) -> None:
    esperado_ph, esperado_clinicos = caso.esperado_rules()
    assert extract_ph_by_rules(caso.text).findings == esperado_ph
    # Por semántica y no por forma: el extractor rellena `n_raices=None` donde el
    # catálogo simplemente no declara el campo, y son la misma afirmación.
    assert normaliza(extract_hallazgos_by_rules(caso.text)) == normaliza(esperado_clinicos)


@pytest.mark.parametrize("caso", CASOS)
def test_las_medidas_declaradas_se_capturan(caso: ReportCase) -> None:
    assert len(extract_medidas_by_rules(caso.text)) == caso.medidas


# --- coherencia del propio catálogo ----------------------------------------- #
@pytest.mark.parametrize("caso", CASOS)
def test_un_limite_documentado_equivale_a_una_diferencia_real(caso: ReportCase) -> None:
    """El catálogo no puede mentir en ninguna de las dos direcciones.

    Un `limite` escrito sobre un caso que el regex resuelve entero es ruido que
    envejece mal; una diferencia sin `limite` es una deuda que nadie anotó.
    """
    esperado_ph, esperado_clinicos = caso.esperado_rules()
    difiere = esperado_ph != caso.ph or normaliza(esperado_clinicos) != normaliza(caso.clinicos)
    assert difiere == bool(caso.limite), (
        f"{caso.name}: difiere={difiere} pero limite={caso.limite!r}"
    )


# --- lo que ningún cambio puede empeorar ------------------------------------ #
def test_los_falsos_positivos_del_determinista_estan_contados() -> None:
    """Lo que el twin recibiría siendo falso. La cifra solo puede bajar."""
    total = 0
    for caso in CASES:
        esperado_ph, esperado_clinicos = caso.esperado_rules()
        ph = puntua(esperado_ph, caso.ph)
        clinicos = puntua(normaliza(esperado_clinicos), normaliza(caso.clinicos))
        total += ph.errores + ph.sobran + clinicos.errores + clinicos.sobran
    assert total == FALSOS_POSITIVOS_RULES


@pytest.mark.parametrize("caso", report_corpus.by_familia(report_corpus.ABSTENCION))
def test_abstenerse_es_no_extraer_nada_de_mas(caso: ReportCase) -> None:
    """En estos casos lo correcto es callarse; extraer de más es el fallo."""
    esperado_ph, _ = caso.esperado_rules()
    assert set(esperado_ph) - set(caso.ph) == (
        {"14"} if caso.name == "notacion-universal" else set()
    ), "un caso de abstención inventó un diente que no estaba documentado"


# --- el agente completo, extremo a extremo ---------------------------------- #
@pytest.mark.parametrize("caso", CASOS)
def test_ningun_informe_del_corpus_lanza(caso: ReportCase, tmp_path: Path) -> None:
    """Fail-loud, no fail-silent: puede fallar declarando, nunca reventar."""
    ruta = caso.write(tmp_path / "informes")
    agente = ReportAgent(quarantine_dir=tmp_path / "cuarentena")
    salida = agente.ingest(ruta)

    assert salida.status in {ModalityStatus.OK, ModalityStatus.FAILED}
    if salida.status is ModalityStatus.OK:
        esperado_ph, esperado_clinicos = caso.esperado_rules()
        assert {obs.region_id for obs in salida.regional} == (
            set(esperado_ph) | set(esperado_clinicos)
        )


def test_un_informe_sin_nada_regional_no_pasa_el_gate(tmp_path: Path) -> None:
    """La confianza a 0 es lo que lo para: un informe ilegible no es un éxito vacío."""
    caso = next(c for c in CASES if c.name == "sin-hallazgos-regionales")
    salida = ReportAgent().ingest(caso.write(tmp_path))
    assert salida.provenance is not None
    assert salida.provenance.confidence == 0.0


def test_el_corpus_declara_bastante_verdad_para_medir() -> None:
    """Un corpus de tres valores no sostiene un ">95%". Este es el mínimo útil."""
    valores_ph, valores_clinicos = report_corpus.total_verdad()
    assert len(CASES) >= 20
    assert valores_ph + valores_clinicos >= 60


def test_el_corpus_se_materializa_en_disco(tmp_path: Path) -> None:
    """Se escribe como `.txt` para poder correr el pipeline entero sobre él."""
    rutas = report_corpus.write_all(tmp_path / "corpus")
    assert len(rutas) == len(CASES)
    assert all(ruta.read_text(encoding="utf-8").strip() for ruta in rutas)


def test_el_almacen_no_hace_falta_para_el_informe(tmp_path: Path) -> None:
    """El `report-agent` no persiste arrays: no depende del `ArtifactStore`."""
    ArtifactStore(tmp_path / "artifacts")  # existe, pero el agente no lo recibe
    assert ReportAgent().ingest(CASES[0].write(tmp_path)).status is ModalityStatus.OK
