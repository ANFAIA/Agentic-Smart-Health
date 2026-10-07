"""ZIP entries with shared basenames must not overwrite another acquisition."""
import importlib.util
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from ingestion_agents import synthetic

_path = Path(__file__).resolve().parents[1] / "scripts/prepara_series_cbct.py"
_spec = importlib.util.spec_from_file_location("prepara_series_cbct", _path)
assert _spec and _spec.loader
module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = module
_spec.loader.exec_module(module)


def test_nombres_repetidos_se_conservan_por_serie(tmp_path):
    first = synthetic.write_dicom_series(tmp_path / "a", np.zeros((3, 2, 2)), (1, 1, 1))
    second = synthetic.write_dicom_series(tmp_path / "b", np.ones((3, 2, 2)), (1, 1, 1))
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as z:
        for index, source in enumerate((first, second)):
            for file in source.glob("*.dcm"):
                z.write(file, f"export-{index}/{file.name}")
    result = module.prepara(archive, first, tmp_path / "output")
    assert len(result["series"]) == 2
    assert sum(g["count"] for g in result["series"].values()) == 6
    assert len(result["changed_in_previous_extraction"]) == 3
    for group in result["series"].values():
        directory = Path(group["directory"])
        assert len(list(directory.glob("*.dcm"))) == 3
        for item in group["instances"]:
            content = (directory / item["file"]).read_bytes()
            assert content in ((first / item["file"]).read_bytes(),
                               (second / item["file"]).read_bytes())
