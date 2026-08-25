"""Tests del contrato de datos (`core-schemas`).

Cubren las garantías que ahora tienen lógica clínica, no solo tipos:
  · versión de contrato serializada  (schema_version)
  · manejo explícito de fallos de ingesta  (ModalityIngestion / status)
  · retrocompatibilidad con la forma que usan los notebooks
  · validación de formato ISO-FDI
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest
from core_schemas import (
    SCHEMA_VERSION,
    ClinicalAttributes,
    Derivation,
    Modality,
    ModalityIngestion,
    ModalityStatus,
    Provenance,
    RegionalObservation,
    RigidTransform,
    TwinSnapshot,
)
from core_schemas.models import _rotar
from pydantic import ValidationError


def _prov(**kw) -> Provenance:
    base = dict(source_file="scan.obj", modality=Modality.MESH, agent="mesh-agent")
    base.update(kw)
    return Provenance(**base)


def _snap(**kw) -> TwinSnapshot:
    base = dict(
        acquisition_id="A1",
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:abc",
        provenance=_prov(),
    )
    base.update(kw)
    return TwinSnapshot(**base)


# --- #8 schema_version ------------------------------------------------------ #
def test_schema_version_default_y_en_dump():
    snap = _snap()
    assert snap.schema_version == SCHEMA_VERSION
    assert f'"schema_version":"{SCHEMA_VERSION}"' in snap.model_dump_json()


def test_round_trip_json_identico():
    snap = _snap(
        ingestion=[ModalityIngestion(modality=Modality.MESH, status=ModalityStatus.OK)]
    )
    assert TwinSnapshot.model_validate_json(snap.model_dump_json()) == snap


# --- #4 manejo explícito de fallos de ingesta ------------------------------- #
def test_snapshot_parcial_se_declara():
    snap = _snap(
        modalities=[Modality.MESH],
        ingestion=[
            ModalityIngestion(modality=Modality.MESH, status=ModalityStatus.OK),
            ModalityIngestion(
                modality=Modality.CBCT, status=ModalityStatus.FAILED, detail="DICOM corrupto"
            ),
            ModalityIngestion(modality=Modality.REPORT, status=ModalityStatus.MISSING),
        ],
    )
    estados = {r.modality: r.status for r in snap.ingestion}
    assert estados[Modality.CBCT] is ModalityStatus.FAILED
    assert estados[Modality.REPORT] is ModalityStatus.MISSING


def test_retrocompatibilidad_forma_notebook():
    """`TwinSnapshot(modalities=[...])` sin campos nuevos → defaults, sin romper."""
    snap = _snap(modalities=[Modality.MESH])
    assert snap.ingestion == []
    assert snap.schema_version == SCHEMA_VERSION


# --- validación de formato ISO-FDI (regresión) ------------------------------ #
@pytest.mark.parametrize("code", ["16", "48", "51", "85"])
def test_fdi_valido(code):
    obs = RegionalObservation(
        region_id=code,
        attributes=ClinicalAttributes(ph=7.0),
        timestamp=datetime.now(UTC),
        provenance=_prov(modality=Modality.REPORT, agent="report-agent"),
    )
    assert obs.region_id == code


@pytest.mark.parametrize("code", ["00", "49", "86", "9", "160"])
def test_fdi_invalido_se_rechaza(code):
    with pytest.raises(ValidationError):
        RegionalObservation(
            region_id=code,
            attributes=ClinicalAttributes(ph=7.0),
            timestamp=datetime.now(UTC),
            provenance=_prov(modality=Modality.REPORT, agent="report-agent"),
        )


# --- ADR 004 · transformación rígida en Provenance -------------------------- #
def _rot_z(grados: float) -> tuple[float, float, float, float]:
    """Cuaternión (w, x, y, z) de una rotación de `grados` sobre el eje Z."""
    mitad = math.radians(grados) / 2
    return (math.cos(mitad), 0.0, 0.0, math.sin(mitad))


def test_provenance_sin_transform_por_defecto():
    """Los agentes de ingesta no transforman: el campo es opcional y no estorba."""
    assert _prov().transform is None


def test_json_sin_transform_sigue_validando():
    """Garantía aditiva: un JSON escrito antes de 1.3.0 no queda huérfano."""
    sin_campo = json.loads(_snap().model_dump_json())
    del sin_campo["provenance"]["transform"]
    assert TwinSnapshot.model_validate(sin_campo).provenance.transform is None


def test_cuaternion_sin_normalizar_se_rechaza():
    """Un cuaternión no unitario codifica una escala encubierta: rompe reversibilidad."""
    with pytest.raises(ValidationError, match="normalizado"):
        RigidTransform(rotation=(1.0, 1.0, 1.0, 1.0), translation=(0.0, 0.0, 0.0))


def test_inversa_deshace_la_traslacion():
    t = RigidTransform(rotation=_rot_z(90), translation=(10.0, 0.0, 5.0))
    # R⁻¹ lleva (10,0,5) a (0,-10,5); la traslación inversa es su opuesta.
    assert t.inverse().translation == pytest.approx((0.0, 10.0, -5.0), abs=1e-9)


def test_doble_inversa_es_la_identidad():
    t = RigidTransform(rotation=_rot_z(37), translation=(1.5, -2.0, 0.25), rms_mm=0.08)
    ida_vuelta = t.inverse().inverse()
    assert ida_vuelta.rotation == pytest.approx(t.rotation, abs=1e-12)
    assert ida_vuelta.translation == pytest.approx(t.translation, abs=1e-12)


@pytest.mark.parametrize("punto", [(0.0, 0.0, 0.0), (3.0, -1.0, 2.5), (-7.25, 0.5, 11.0)])
def test_aplicar_y_deshacer_devuelve_el_punto(punto):
    """La garantía que sostiene la reversibilidad del ADR 004: T⁻¹(T(p)) == p."""
    t = RigidTransform(rotation=_rot_z(115), translation=(4.0, -3.0, 8.0))
    inv = t.inverse()

    def aplicar(tr: RigidTransform, p):
        rx, ry, rz = _rotar(tr.rotation, p)
        return (rx + tr.translation[0], ry + tr.translation[1], rz + tr.translation[2])

    assert aplicar(inv, aplicar(t, punto)) == pytest.approx(punto, abs=1e-9)


def test_transform_sobrevive_al_round_trip_json():
    t = RigidTransform(rotation=_rot_z(45), translation=(1.0, 2.0, 3.0), rms_mm=0.17)
    snap = _snap(provenance=_prov(transform=t, agent="geometric-fusion-agent"))
    revivido = TwinSnapshot.model_validate_json(snap.model_dump_json())
    assert revivido.provenance.transform == t
    assert revivido.provenance.transform.rms_mm == 0.17


def test_rms_negativo_se_rechaza():
    with pytest.raises(ValidationError):
        RigidTransform(rotation=_rot_z(0), translation=(0.0, 0.0, 0.0), rms_mm=-0.1)


# --- `Derivation`: leído vs inferido --------------------------------------- #
# El hueco que estos tests tapan: `Provenance` decía qué agente produjo un valor,
# pero no si lo había *leído* o *inferido*. Con un `report-agent` que tiene un
# backend por reglas y dos por modelo, «lo produjo report-agent@0.1.0» dejó de ser
# una respuesta: el mismo agente y la misma versión dan las dos cosas.
def test_sin_declarar_no_es_lo_mismo_que_determinista() -> None:
    """`None` significa «nadie lo dijo», no «es reproducible».

    Es la misma distinción que `ModalityStatus` hace entre `MISSING` y `FAILED`. Si
    el defecto afirmase determinismo, un agente que infiere y olvida declararlo
    produciría un valor que **miente** en vez de uno que calla.
    """
    assert _prov().derivation is None
    assert _prov().derivation is not Derivation.DETERMINISTIC


def test_una_inferencia_declara_su_modelo() -> None:
    prov = _prov(derivation=Derivation.INFERRED, model="llm:claude-sonnet-5")
    assert prov.derivation is Derivation.INFERRED
    assert prov.model == "llm:claude-sonnet-5"


def test_una_inferencia_sin_modelo_no_se_construye() -> None:
    """«Inferido por no se sabe quién» no es auditable ni reproducible."""
    with pytest.raises(ValidationError, match="exige declarar `model`"):
        _prov(derivation=Derivation.INFERRED)


@pytest.mark.parametrize("derivation", [None, Derivation.DETERMINISTIC])
def test_un_modelo_sin_inferencia_es_una_contradiccion(derivation: object) -> None:
    """Declarar modelo y a la vez determinismo es una afirmación que no puede ser cierta."""
    with pytest.raises(ValidationError, match="solo tiene sentido"):
        _prov(derivation=derivation, model="llm:claude-sonnet-5")


def test_el_determinismo_no_arrastra_modelo() -> None:
    prov = _prov(derivation=Derivation.DETERMINISTIC)
    assert prov.model is None


def test_la_derivacion_sobrevive_al_contrato_serializado() -> None:
    """Si no viajase en el JSON, el twin persistido perdería la distinción."""
    prov = _prov(derivation=Derivation.INFERRED, model="llm:claude-sonnet-5")
    assert Provenance.model_validate_json(prov.model_dump_json()) == prov
