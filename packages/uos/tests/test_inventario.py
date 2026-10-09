"""El inventario de series para la clínica: describe sin identificar.

Fija las propiedades por las que existe: no publica UID ni valores identificables, detecta
cortes e instancias ausentes, recorta las fechas y se niega a incluir texto libre cuando
alguna cabecera identifica al paciente.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pydicom
import pytest
from ingestion_agents import synthetic
from uos.inventario import (
    clave,
    inventaria,
    lee_cabeceras_de_directorio,
    lee_cabeceras_de_zip,
)


def _serie(directorio: Path, cortes: int = 10) -> Path:
    return synthetic.write_dicom_series(
        directorio, np.zeros((cortes, 4, 4)), (0.5, 0.5, 0.5),
        study_date=datetime(2026, 3, 14, tzinfo=UTC))


def _edita(directorio: Path, **tags: object) -> None:
    for f in sorted(directorio.glob("*.dcm")):
        ds = pydicom.dcmread(f)
        for k, v in tags.items():
            setattr(ds, k, v)
        ds.save_as(f)


def test_serie_completa_es_uniforme(tmp_path: Path) -> None:
    inv = inventaria(lee_cabeceras_de_directorio(_serie(tmp_path / "s")))
    serie = inv["series"][0]
    assert inv["single_identity"] and serie["instances"] == 10
    assert serie["positions"]["uniform"] is True
    assert serie["positions"]["gaps"] == []
    assert serie["positions"]["estimated_missing_slices"] == 0
    assert serie["instance_numbers"]["missing_count"] == 0


def test_cortes_retirados_aparecen_como_hueco_y_numeros_ausentes(tmp_path: Path) -> None:
    directorio = _serie(tmp_path / "s", cortes=12)
    ficheros = sorted(directorio.glob("*.dcm"),
                      key=lambda f: int(pydicom.dcmread(f).InstanceNumber))
    for f in ficheros[4:7]:
        f.unlink()
    serie = inventaria(lee_cabeceras_de_directorio(directorio))["series"][0]
    pos, num = serie["positions"], serie["instance_numbers"]
    assert len(pos["gaps"]) == 1 and pos["gaps"][0]["estimated_missing"] == 3
    assert pos["uniform"] is False
    assert num["missing_count"] == 3
    assert pos["instance_order_matches_position"] is True


def test_no_publica_uid_ni_identificador_del_paciente(tmp_path: Path) -> None:
    directorio = _serie(tmp_path / "s")
    cabeceras = lee_cabeceras_de_directorio(directorio)
    texto = json.dumps(inventaria(cabeceras))
    for ds in cabeceras[:1]:
        for uid in (ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID):
            assert str(uid) not in texto
        assert str(ds.PatientID) not in texto
    assert clave(cabeceras[0].SeriesInstanceUID) in texto


def test_fechas_recortadas_a_mes_salvo_que_se_pidan(tmp_path: Path) -> None:
    cabeceras = lee_cabeceras_de_directorio(_serie(tmp_path / "s"))
    assert inventaria(cabeceras)["series"][0]["dates"]["StudyDate"] == ["2026-03"]
    completo = inventaria(cabeceras, fechas_completas=True)
    assert completo["series"][0]["dates"]["StudyDate"] == ["2026-03-14"]


def test_texto_libre_solo_sin_tags_identificables(tmp_path: Path) -> None:
    directorio = _serie(tmp_path / "s")
    # La serie sintética trae un nombre ficticio, y se trata como cualquier otro.
    _edita(directorio, PatientName="", SeriesDescription="Maxilar 8x8")
    limpio = inventaria(lee_cabeceras_de_directorio(directorio))
    assert limpio["series"][0]["free_text"]["SeriesDescription"] == ["Maxilar 8x8"]

    _edita(directorio, PatientName="Garcia^Ana")
    sucio = inventaria(lee_cabeceras_de_directorio(directorio))
    assert sucio["identifying_tags_present"] == ["PatientName"]
    assert sucio["free_text_included"] is False
    assert sucio["series"][0]["free_text"] is None
    assert "Garcia" not in json.dumps(sucio)


def test_marcador_de_anonimizador_no_cuenta_como_identificable(tmp_path: Path) -> None:
    directorio = _serie(tmp_path / "s")
    _edita(directorio, PatientName="ANONYMIZED")
    assert inventaria(lee_cabeceras_de_directorio(directorio))[
        "identifying_tags_present"] == []


@pytest.mark.filterwarnings("ignore:Duplicate name")
def test_zip_con_nombres_repetidos_conserva_las_dos_series(tmp_path: Path) -> None:
    a, b = _serie(tmp_path / "a", 3), _serie(tmp_path / "b", 4)
    archivo = tmp_path / "export.zip"
    with ZipFile(archivo, "w") as zf:
        for origen in (a, b):
            for f in origen.glob("*.dcm"):
                zf.write(f, f.name)   # mismo nombre en las dos series, a propósito
    inv = inventaria(lee_cabeceras_de_zip(archivo))
    assert inv["instances"] == 7
    assert sorted(s["instances"] for s in inv["series"]) == [3, 4]
    assert inv["identity_counts"]["FrameOfReferenceUID"] == 2


def test_sin_cabeceras_falla(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        inventaria([])
