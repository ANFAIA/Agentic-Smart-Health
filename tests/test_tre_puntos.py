"""El script de TRE elige el registro desde el volumen DICOM, no el del campo centrado."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fusion_agents.tre import tre_de_registro

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "tre_puntos.py"
_spec = importlib.util.spec_from_file_location("tre_puntos", _RUTA)
assert _spec and _spec.loader
tp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tp
_spec.loader.exec_module(tp)


def _manifiesto(con_volumen: bool = True):
    def reg(ident, origen, tx):
        m = np.eye(4)
        m[0, 3] = tx
        return SimpleNamespace(id=ident, source_frame=origen, target_frame="frame.ios_master",
                               transform_4x4_row_major=m.ravel().tolist())
    assets = [SimpleNamespace(kind=SimpleNamespace(value="volume"), frame="frame.ct_001")]
    return SimpleNamespace(
        assets=assets if con_volumen else [],
        registrations=[reg("reg.ct_centered_to_ios", "frame.ct_001.centered", 9.0),
                       reg("reg.ct_to_ios", "frame.ct_001", 1.0)])


def test_elige_el_registro_desde_el_volumen() -> None:
    ident, matriz, origen, _ = tp.registro_del_volumen(_manifiesto(), None)
    assert ident == "reg.ct_to_ios" and origen == "frame.ct_001"
    assert matriz.reshape(4, 4)[0, 3] == 1.0


def test_sin_volumen_hay_que_nombrar_el_registro() -> None:
    with pytest.raises(SystemExit):
        tp.registro_del_volumen(_manifiesto(con_volumen=False), None)
    ident, *_ = tp.registro_del_volumen(_manifiesto(con_volumen=False), "reg.ct_to_ios")
    assert ident == "reg.ct_to_ios"


def test_informe_declara_alcance_y_observador_unico() -> None:
    puntos = {"11-i": np.zeros(3), "21-i": np.ones(3)}
    resultado = {
        "registration": {"id": "reg.ct_to_ios", "source_frame": "frame.ct_001",
                         "target_frame": "frame.ios_master"},
        "stl_loaded_as": "LPS",
        "observers": {"cbct": {"observers": 1, "not_marked_by_all": []},
                      "ios": {"observers": 1, "not_marked_by_all": []}},
        "existing_registration": tre_de_registro(puntos, puntos, np.eye(4)),
    }
    texto = tp.informe_md(resultado)
    assert "No es exactitud general" in texto
    assert "no se puede medir el acuerdo" in texto
