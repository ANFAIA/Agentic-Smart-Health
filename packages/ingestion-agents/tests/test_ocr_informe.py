"""El informe ESCANEADO: de fichero tirado a fichero ingerido y declarado.

El caso real que motiva esto es un **pasaporte de implantes** de una clínica: tres
implantes con su REF y su número de lote en pegatinas impresas, escaneado sin capa de
texto y **girado 90 grados**. El agente lo declaraba `FAILED`, que era correcto pero
tiraba trazabilidad regulatoria de primer orden.

⚠️ **Lo que se prueba aquí no es que el OCR acierte** — eso depende de `tesseract` y no de
este repo. Se prueba lo que sí es nuestro: que se intente, que el texto leído de píxeles
**no se confunda** con una capa de texto, y que cuando no hay con qué hacer OCR se diga
qué falta en vez de «falló el OCR».
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from ingestion_agents.report_agent import (
    OcrNoDisponible,
    ReportAgent,
    ocr_disponible,
    ocr_pdf,
)


def _binario(destino: Path, nombre: str, guion: str) -> None:
    """Un ejecutable de mentira en el PATH, para ejercitar el `subprocess` de verdad.

    Se prefiere esto a parchear `subprocess.run`: lo que hay que comprobar es que se
    invocan los binarios que son con los argumentos que son, y un `monkeypatch` de la
    llamada probaría el parche.
    """
    ruta = destino / nombre
    ruta.write_text(guion)
    ruta.chmod(ruta.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def ocr_de_mentira(tmp_path, monkeypatch) -> Path:
    binarios = tmp_path / "bin"
    binarios.mkdir()
    # `pdftoppm -r DPI -png ENTRADA PREFIJO` — escribe una página.
    _binario(binarios, "pdftoppm", '#!/bin/sh\ntouch "${5}-1.png"\n')
    # `tesseract PNG stdout -l IDIOMAS --psm 1`
    _binario(
        binarios,
        "tesseract",
        '#!/bin/sh\necho "IMPLANTE N 01 REF 300340 LOT 12251592 Localizacion 11"\n',
    )
    monkeypatch.setenv("PATH", str(binarios) + os.pathsep + os.environ["PATH"])
    return binarios


def _pdf_escaneado(destino: Path) -> Path:
    """Un PDF **válido** de una página que es sólo imagen, como sale de un escáner.

    ⚠️ Escribirlo a mano no vale, y el intento lo demostró: un PDF sin `startxref` revienta
    en `pypdf` con `PdfReadError` **antes** de llegar al OCR, que es otra cosa distinta y
    debe seguir siendo otra cosa distinta. Un PDF roto no es un PDF escaneado. `PIL` guarda
    una imagen como PDF de una página y eso es exactamente el caso real.
    """
    from PIL import Image

    ruta = destino / "pasaporte.pdf"
    Image.new("RGB", (1600, 2293), (255, 255, 255)).save(ruta, "PDF")
    return ruta


def test_sin_los_binarios_se_dice_CUAL_falta(monkeypatch, tmp_path) -> None:
    """«Falló el OCR» no es accionable; «falta tesseract, instálalo así» sí."""
    monkeypatch.setenv("PATH", str(tmp_path))
    falta = ocr_disponible()
    assert "tesseract" in falta
    assert "pacman" in falta
    with pytest.raises(OcrNoDisponible, match="tesseract"):
        ocr_pdf(tmp_path / "loquesea.pdf")


def test_con_los_binarios_se_rasteriza_y_se_lee(ocr_de_mentira, tmp_path) -> None:
    assert ocr_disponible() == ""
    texto = ocr_pdf(_pdf_escaneado(tmp_path))
    assert "12251592" in texto


def test_el_agente_ingiere_el_escaneado_en_vez_de_tirarlo(ocr_de_mentira, tmp_path) -> None:
    salida = ReportAgent().ingest(_pdf_escaneado(tmp_path))
    assert salida.ok, salida.detail


def test_el_texto_de_OCR_no_pasa_por_capa_de_texto(ocr_de_mentira, tmp_path) -> None:
    """⚠️ **El corazón de esto.** Un número de lote mal leído —un 8 por un 6— es un fallo de
    trazabilidad que nadie ve leyendo el `.uos`, porque el dato está ahí y parece bueno.

    Así que el agente sale por debajo del umbral del gate (0,70) y lo declara. Ingerido no
    es verificado.
    """
    salida = ReportAgent().ingest(_pdf_escaneado(tmp_path))
    assert salida.provenance is not None
    assert salida.provenance.confidence < 0.7
    assert "OCR" in (salida.detail or "")
    assert "cotejarlo con el original" in (salida.detail or "")


def test_sin_OCR_el_fallo_sigue_diciendo_que_clase_de_fichero_es(monkeypatch, tmp_path):
    """Y ahora además por qué no se intentó, que antes no se decía."""
    monkeypatch.setenv("PATH", str(tmp_path))
    salida = ReportAgent().ingest(_pdf_escaneado(tmp_path))
    assert not salida.ok
    assert "sha256:" in (salida.detail or "")
    assert "No se ha intentado OCR" in (salida.detail or "")


def test_un_informe_CON_capa_de_texto_no_pasa_por_el_OCR(ocr_de_mentira, tmp_path) -> None:
    """Si el PDF trae texto, el OCR ni se toca: su confianza no debe bajar por nada."""
    txt = tmp_path / "informe.txt"
    txt.write_text("Pieza 24: pH 6.2\n", encoding="utf-8")
    salida = ReportAgent().ingest(txt)
    assert "OCR" not in (salida.detail or "")
    assert salida.provenance is not None
    assert salida.provenance.confidence >= 0.7
