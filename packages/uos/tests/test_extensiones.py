"""El mecanismo de extensiones — propuesta nuestra, no UOS v0.2.

Lo que se prueba no es que los campos existan, sino que **hagan de puerta**: que usar algo
sin declararlo invalide, que exigir algo que no se usa invalide, y sobre todo que nada de
lo nuestro sea obligatorio. Una extensión que sólo añade información y se marca `required`
haría que un visor conforme se negara a abrir un caso que podría enseñar perfectamente.
"""

from __future__ import annotations

import struct
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from core_schemas import Modality, Provenance, RigidTransform, TwinSnapshot
from uos import UOSExportAgent, escribe_uos, lee_manifiesto, valida
from uos.manifiesto import (
    Asset,
    Clase,
    EstadoPHI,
    Extension,
    Frame,
    Manifiesto,
    Sujeto,
    Visita,
)


def _stl(triangulos: int = 4) -> bytes:
    rng = np.random.default_rng(0)
    crudo = b"ASH fixture" + bytes(69) + struct.pack("<I", triangulos)
    for _ in range(triangulos):
        v = rng.normal(0, 10, (3, 3)).astype("<f4")
        crudo += np.zeros(3, dtype="<f4").tobytes() + v.tobytes() + b"\x00\x00"
    return crudo


@pytest.fixture
def malla(tmp_path) -> Path:
    p = tmp_path / "scan.stl"
    p.write_bytes(_stl())
    return p


def _asset(ruta: Path) -> Asset:
    import hashlib

    crudo = ruta.read_bytes()
    return Asset(
        id="asset.doc", kind=Clase.DOCUMENT, visit="v1", uri="scene/scan.stl",
        media_type="model/stl", sha256=hashlib.sha256(crudo).hexdigest(),
        bytes=len(crudo), frame="frame.ios_master",
    )


def _manifiesto(assets, **kw) -> Manifiesto:
    base = dict(
        case_id="urn:uuid:0", generator={"name": "test", "version": "0"},
        phi_state=EstadoPHI.PSEUDONYMIZED, subject=Sujeto(pseudonym="P-1"),
        canonical_frame=Frame(id="frame.ios_master"),
        visits=[Visita(id="v1", date="2026-08-24")], assets=assets,
    )
    return Manifiesto(**{**base, **kw})


def _snapshot() -> TwinSnapshot:
    t = RigidTransform(rotation=(1.0, 0.0, 0.0, 0.0), translation=(1.0, 2.0, 3.0))
    return TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.CBCT, agent="a@0",
                              transform=t),
    )


# --- el mecanismo hace de puerta --------------------------------------------- #
def test_usar_una_extension_sin_declararla_INVALIDA(tmp_path, malla):
    """Un lector que ve un nombre en `used` y no lo encuentra en `extensions` no tiene
    forma de saber qué es: ni leerla, ni saltarla a sabiendas."""
    m = _manifiesto([_asset(malla)], extensions_used=["ash_clinical"])
    salida = escribe_uos(tmp_path / "c.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)

    assert not inf.valido
    assert any("no las declara" in e for e in inf.errores)


def test_exigir_algo_que_no_se_usa_INVALIDA(tmp_path, malla):
    """Dejaría el caso sin abrir para nada."""
    m = _manifiesto(
        [_asset(malla)],
        extensions={"x": Extension(name="x", version="1.0")},
        extensions_used=[],
        extensions_required=["x"],
    )
    salida = escribe_uos(tmp_path / "c.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)

    assert not inf.valido
    assert any("EXIGE" in e for e in inf.errores)


def test_una_extension_que_apunta_a_lo_que_no_esta_INVALIDA(tmp_path, malla):
    m = _manifiesto(
        [_asset(malla)],
        extensions={"x": Extension(name="x", version="1.0", uri="clinical/no_existe.json")},
        extensions_used=["x"],
    )
    salida = escribe_uos(tmp_path / "c.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)

    assert not inf.valido
    assert any("no es ningun asset declarado" in e for e in inf.errores)


def test_una_extension_OBLIGATORIA_se_avisa_aunque_el_contenedor_sea_valido(tmp_path, malla):
    """El emisor está diciendo «sin entender esto no abras el fichero», y un validador que
    lo callara dejaría a un lector abrirlo creyendo que lo entiende entero."""
    m = _manifiesto(
        [_asset(malla)],
        extensions={"x": Extension(name="x", version="1.0")},
        extensions_used=["x"],
        extensions_required=["x"],
    )
    salida = escribe_uos(tmp_path / "c.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)

    assert inf.valido
    assert any("OBLIGATORIA" in a for a in inf.avisos)


def test_declarar_una_extension_y_no_usarla_es_un_AVISO(tmp_path, malla):
    m = _manifiesto(
        [_asset(malla)],
        extensions={"x": Extension(name="x", version="1.0")},
    )
    salida = escribe_uos(tmp_path / "c.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)

    assert inf.valido
    assert any("sobran" in a for a in inf.avisos)


# --- y lo que emitimos nosotros ---------------------------------------------- #
def test_el_agente_declara_lo_que_anade(tmp_path, malla):
    from core_schemas import ClinicalAttributes, RegionalObservation

    obs = RegionalObservation(
        region_id="16", attributes=ClinicalAttributes(ph=5.4),
        timestamp=datetime.now(UTC),
        provenance=Provenance(source_file="x", modality=Modality.REPORT, agent="r@0"),
    )
    salida = UOSExportAgent(None).export(
        _snapshot().model_copy(update={"regional": [obs]}),
        tmp_path / "caso", pseudonimo="P-1", malla=malla,
    )

    assert salida.ok, salida.detail
    m = lee_manifiesto(salida.path)
    assert "ash_clinical" in m.extensions_used
    assert m.extensions["ash_clinical"].uri == "clinical/observations.json"
    assert "FHIR" in m.extensions["ash_clinical"].description


def test_NADA_de_lo_nuestro_es_obligatorio(tmp_path, malla):
    """La regla que hace que esto no rompa la interoperabilidad: todo lo que añadimos SUMA.

    Un visor conforme tiene que poder abrir un caso nuestro sin entender ni una de nuestras
    extensiones, y enseñar la escena, el volumen y las fotos. Si algún día algo entrara en
    `required`, sería una decisión deliberada y este test la haría visible.
    """
    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla
    )

    m = lee_manifiesto(salida.path)
    assert m.extensions_required == []
    assert valida(salida.path).valido
