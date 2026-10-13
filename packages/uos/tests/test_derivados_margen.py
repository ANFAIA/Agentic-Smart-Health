"""El registro del margen gingival viaja en el sidecar solo si se calculó."""

from __future__ import annotations

import numpy as np
from uos.derivados import meta_segmentacion


def _meta(**extra):
    return meta_segmentacion(np.array([0, 11, 11, 0]), asset_origen="asset.scene",
                             modelo="m", version=None, **extra)


def test_ausente_significa_que_no_se_ejecuto() -> None:
    assert "gingival_margin" not in _meta()


def test_presente_lleva_metodo_estado_y_piezas() -> None:
    margen = {"method": "analysis_agents.frontera.ancla_al_pliegue", "applied": False,
              "teeth": {"11": {"moved": 0, "declined": True, "reason": "weak_crease"}}}
    meta = _meta(margen=margen)
    assert meta["gingival_margin"] == margen
    assert meta["regulatory"]["layer"] == 3
