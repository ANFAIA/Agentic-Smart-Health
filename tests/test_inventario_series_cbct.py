"""El script del paquete para la clínica no escribe nada si hay datos identificativos."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom
from ingestion_agents import synthetic

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inventario_series_cbct.py"


def _corre(serie: Path, salida: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--serie", str(serie),
                           "--salida", str(salida), *extra], capture_output=True, text=True)


def test_se_niega_con_identificables_y_escribe_sin_ellos(tmp_path: Path) -> None:
    serie = synthetic.write_dicom_series(tmp_path / "s", np.zeros((6, 4, 4)), (1, 1, 1))
    hecho = _corre(serie, tmp_path / "con")
    assert hecho.returncode == 2 and "PatientName" in hecho.stderr
    assert not (tmp_path / "con").exists()

    for f in serie.glob("*.dcm"):
        ds = pydicom.dcmread(f)
        ds.PatientName = ""
        ds.save_as(f)
    hecho = _corre(serie, tmp_path / "sin")
    assert hecho.returncode == 0, hecho.stderr
    nombres = {p.name for p in (tmp_path / "sin").iterdir()}
    assert nombres == {"inventario.json", "inventario.md", "cobertura.svg"}
    informe = (tmp_path / "sin" / "inventario.md").read_text()
    assert "Preguntas para la clínica" in informe
    assert "SINTETICO" not in informe and "SYNTH-0001" not in informe


def test_con_permiso_declara_los_tags_en_el_informe(tmp_path: Path) -> None:
    serie = synthetic.write_dicom_series(tmp_path / "s", np.zeros((6, 4, 4)), (1, 1, 1))
    hecho = _corre(serie, tmp_path / "out", "--permitir-identificables")
    assert hecho.returncode == 0, hecho.stderr
    informe = (tmp_path / "out" / "inventario.md").read_text()
    assert "datos identificativos" in informe and "PatientName" in informe
    assert "SINTETICO" not in informe
