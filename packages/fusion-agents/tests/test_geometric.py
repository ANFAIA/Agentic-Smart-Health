"""`geometric-fusion-agent` y el registro (ADR 004 §2.6, §2.7).

Nivel **unitario** del §2.7: se aplica a una nube una transformación **conocida** y
se exige que el algoritmo la recupere. La verdad de referencia es exacta porque la
fabricamos aquí, así que un fallo es del algoritmo y no del dato — que es
precisamente lo que el nivel de integración, contra pares reales, no puede aislar.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest
from core_schemas import Modality, ModalityStatus, Provenance, TwinSnapshot
from fusion_agents import GeometricFusionAgent, RegistrationResult, icp, icp_global
from fusion_agents.registration import apply, kabsch, matrix_to_quaternion


def _rot_z(grados: float) -> np.ndarray:
    a = math.radians(grados)
    return np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


def _nube(n: int = 400, seed: int = 0) -> np.ndarray:
    """Nube asimétrica: una esfera perfecta no fija la rotación (es degenerada)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 10.0, (n, 3)) * np.array([1.0, 0.6, 0.3])


def _snap(**kw) -> TwinSnapshot:
    base = dict(
        acquisition_id="A1",
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:abc",
        provenance=Provenance(
            source_file="caso/", modality=Modality.CBCT, agent="cbct-agent@1.0.0"
        ),
    )
    base.update(kw)
    return TwinSnapshot(**base)


# --- piezas del algoritmo --------------------------------------------------- #
def test_kabsch_recupera_la_pose_de_puntos_emparejados():
    origen = _nube()
    rot, trans = _rot_z(30), np.array([2.0, -1.0, 0.5])
    destino = apply(rot, trans, origen)
    r, t = kabsch(origen, destino)
    assert r == pytest.approx(rot, abs=1e-9)
    assert t == pytest.approx(trans, abs=1e-9)


def test_kabsch_no_devuelve_una_reflexion():
    """Sin la corrección del determinante, la SVD puede dar una reflexión."""
    origen = _nube()
    r, _ = kabsch(origen, apply(_rot_z(120), np.zeros(3), origen))
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)  # rotación pura, no reflexión


@pytest.mark.parametrize("grados", [0.0, 15.0, 90.0, 179.0])
def test_matrix_to_quaternion_cubre_las_cuatro_ramas(grados):
    q = matrix_to_quaternion(_rot_z(grados))
    assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0, abs=1e-9)


# --- ICP · nivel unitario del ADR 004 §2.7 ---------------------------------- #
@pytest.mark.parametrize(
    ("grados", "traslacion"),
    [(0.0, (0.0, 0.0, 0.0)), (5.0, (1.0, 0.0, 0.0)), (12.0, (1.5, -0.8, 0.3))],
)
def test_icp_recupera_una_transformacion_conocida(grados, traslacion):
    origen = _nube()
    destino = apply(_rot_z(grados), np.array(traslacion), origen)
    res = icp(origen, destino)
    assert res.rms_mm < 1e-6
    assert res.translation == pytest.approx(traslacion, abs=1e-6)


def test_icp_es_determinista():
    """Dos corridas sobre el mismo dato deben dar lo mismo, o la confianza no significa nada."""
    origen = _nube()
    destino = apply(_rot_z(8), np.array([1.0, 1.0, 1.0]), origen)
    assert icp(origen, destino).rms_mm == icp(origen, destino).rms_mm


def test_icp_sobre_nubes_identicas_no_mueve_nada():
    origen = _nube()
    res = icp(origen, origen)
    assert res.rms_mm == pytest.approx(0.0, abs=1e-9)
    assert res.translation == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_el_residuo_se_mide_sobre_la_nube_completa():
    """No sobre el submuestreo: es el número del que sale la confianza."""
    origen = _nube(n=600)
    destino = apply(_rot_z(3), np.array([0.5, 0.0, 0.0]), origen) + np.array([0.0, 0.0, 2.0])
    assert icp(origen, destino).rms_mm > 0  # el desajuste sistemático no se esconde


@pytest.mark.parametrize("mala", [np.zeros((2, 3)), np.zeros((10, 2)), np.zeros(10)])
def test_icp_rechaza_entradas_invalidas(mala):
    with pytest.raises(ValueError):
        icp(mala, _nube())


def test_resultado_se_convierte_al_contrato():
    res = icp(_nube(), apply(_rot_z(10), np.array([1.0, 0.0, 0.0]), _nube()))
    rt = res.to_rigid_transform()
    assert rt.rms_mm == pytest.approx(res.rms_mm)
    assert math.sqrt(sum(c * c for c in rt.rotation)) == pytest.approx(1.0, abs=1e-6)


# --- el agente -------------------------------------------------------------- #
@pytest.fixture
def agente() -> GeometricFusionAgent:
    return GeometricFusionAgent()


def test_registro_perfecto_da_confianza_maxima(agente):
    nube = _nube()
    out = agente.fuse(_snap(), source=nube, target=nube)
    assert out.ok
    assert out.snapshot.provenance.confidence == pytest.approx(1.0)
    assert not out.hitl_required


def test_la_transformacion_queda_en_la_procedencia(agente):
    """Es lo que hace el cambio reversible (ADR 004 §2.2)."""
    nube = _nube()
    out = agente.fuse(
        _snap(), source=nube, target=apply(_rot_z(9), np.array([2.0, 0.0, 0.0]), nube)
    )
    tr = out.snapshot.provenance.transform
    assert tr is not None
    assert tr.translation == pytest.approx((2.0, 0.0, 0.0), abs=1e-5)
    assert tr.inverse().inverse().translation == pytest.approx(tr.translation, abs=1e-9)


def test_la_confianza_sale_del_residuo():
    """confianza = 1 − rms/ε, con un registrador de residuo fijado."""

    def registrador_falso(source, target):
        return RegistrationResult((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.25, 1, True)

    agente = GeometricFusionAgent(epsilon_mm=0.5, registrar=registrador_falso)
    out = agente.fuse(_snap(), source=_nube(), target=_nube())
    assert out.snapshot.provenance.confidence == pytest.approx(0.5)  # 1 − 0.25/0.5


def test_residuo_fuera_de_banda_no_falla_pero_pide_revision():
    """Un mal registro se entrega declarado, no se lanza ni se esconde."""

    def registrador_malo(source, target):
        return RegistrationResult((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2.0, 60, False)

    out = GeometricFusionAgent(registrar=registrador_malo).fuse(
        _snap(), source=_nube(), target=_nube()
    )
    assert out.ok
    assert out.snapshot.provenance.confidence == 0.0
    assert out.hitl_required
    assert any("sin converger" in m for m in out.hitl_reasons)


def test_sin_una_de_las_nubes_se_declara_missing(agente):
    out = agente.fuse(_snap(), source=np.zeros((0, 3)), target=_nube())
    assert out.status is ModalityStatus.MISSING
    assert out.snapshot is None


def test_no_muta_el_snapshot_de_entrada(agente):
    entrada = _snap()
    antes = entrada.model_dump_json()
    agente.fuse(entrada, source=_nube(), target=_nube())
    assert entrada.model_dump_json() == antes


def test_epsilon_invalido_se_rechaza_al_construir():
    with pytest.raises(ValueError, match="epsilon_mm"):
        GeometricFusionAgent(epsilon_mm=0.0)


def test_el_fallo_del_registrador_no_se_propaga(tmp_path):
    def registrador_roto(source, target):
        raise RuntimeError("SVD no converge")

    out = GeometricFusionAgent(registrar=registrador_roto, quarantine_dir=tmp_path).fuse(
        _snap(), source=_nube(), target=_nube()
    )
    assert out.status is ModalityStatus.FAILED
    assert "SVD no converge" in out.detail
    assert out.quarantine_ref is not None


# --- solapamiento parcial y etapa gruesa (medido en histora, 2026-08-10) ---- #
def _superficie(n: int = 800, semilla: int = 0) -> np.ndarray:
    """Nube con curvatura: una silla de montar no tiene simetrías que confundan al ICP."""
    rng = np.random.default_rng(semilla)
    xy = rng.uniform(-10, 10, (n, 2))
    z = 0.04 * (xy[:, 0] ** 2 - xy[:, 1] ** 2)
    return np.column_stack([xy, z])


def _girar(p: np.ndarray, grados: tuple[float, float, float], t: np.ndarray) -> np.ndarray:
    a, b, c = np.radians(grados)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return p @ (rz @ ry @ rx).T + t


def test_icp_mide_la_fraccion_solapada():
    destino = _superficie()
    # La mitad de `origen` son puntos lejanos sin contrapartida posible.
    origen = np.vstack([destino[:400], destino[:400] + np.array([0.0, 0.0, 50.0])])
    res = icp(origen, destino, trim=0.5)
    assert res.overlap_fraction == pytest.approx(0.5, abs=0.05)


def test_el_rms_completo_no_describe_un_registro_con_solapamiento_parcial():
    """El hallazgo de histora: 4,98 mm de nube completa frente a 0,452 mm reales."""
    destino = _superficie()
    origen = np.vstack([destino[:400], destino[:400] + np.array([0.0, 0.0, 50.0])])
    res = icp(origen, destino, trim=0.5)
    assert res.rms_mm > 10.0            # lo que hoy alimentaba la confianza
    assert res.rms_overlap_mm < 0.1     # lo que de verdad mide el registro
    assert res.rms_efectivo_mm == res.rms_overlap_mm


def test_sin_medir_solapamiento_el_rms_efectivo_es_el_de_siempre():
    """Retrocompatible: un registrador que no mide solapamiento se comporta igual."""
    res = RegistrationResult((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.25, 1, True)
    assert res.rms_efectivo_mm == 0.25
    assert res.to_rigid_transform().rms_mm == pytest.approx(0.25)


def test_la_procedencia_guarda_el_rms_efectivo():
    """Es lo que alguien leerá dentro de un año para juzgar ese registro."""
    res = RegistrationResult(
        (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 9.9, 1, True,
        rms_overlap_mm=0.4, overlap_fraction=0.3,
    )
    assert res.to_rigid_transform().rms_mm == pytest.approx(0.4)


def test_solapamiento_escaso_anula_la_confianza():
    """Con cuatro puntos emparejados siempre hay una pose que los acerca."""
    def registrador(source, target):
        return RegistrationResult(
            (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 8.0, 1, True,
            rms_overlap_mm=0.01, overlap_fraction=0.02,  # rms buenísimo, casi sin medir
        )

    agente = GeometricFusionAgent(registrar=registrador, min_overlap=0.20)
    out = agente.fuse(_snap(), source=_nube(), target=_nube())
    assert out.snapshot.provenance.confidence == 0.0
    assert any("tiene contrapartida" in m for m in out.hitl_reasons)


def test_min_overlap_invalido_se_rechaza_al_construir():
    with pytest.raises(ValueError, match="min_overlap"):
        GeometricFusionAgent(min_overlap=1.5)


@pytest.mark.parametrize(
    "giro",
    [(70.0, -50.0, 110.0), (180.0, 0.0, 0.0), (150.0, 120.0, 90.0), (120.0, 160.0, 200.0)],
    ids=["oblicuo", "medio-giro", "compuesto", "invertido"],
)
def test_la_etapa_gruesa_encuentra_la_pose_que_el_icp_fino_solo_aproxima(giro):
    """El hueco declarado del ADR 004, y por qué su ausencia era peligrosa.

    Ojo al modo de fallo: el ICP fino **no revienta**. Se queda en un mínimo local a
    ~0,4 mm —una cifra que parece un buen registro— y devuelve una pose que no es la
    correcta. Un fallo ruidoso se ve; éste hay que ir a buscarlo.
    """
    destino = _superficie()
    origen = _girar(destino, giro, np.array([40.0, -25.0, 15.0]))

    fino = icp(origen, destino)
    grueso = icp_global(
        origen, destino, poses=120, criba=300, fino=800, trim=1.0, trim_criba=1.0
    )

    assert fino.rms_efectivo_mm > 0.2    # plausible, y equivocado
    assert grueso.rms_efectivo_mm < 0.01  # la transformación exacta
