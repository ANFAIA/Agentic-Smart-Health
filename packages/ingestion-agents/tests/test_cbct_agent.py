"""`cbct-agent`: serie DICOM → campo gaussiano semilla, con seudonimización."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from core_schemas import Modality, ModalityStatus, Support
from ingestion_agents import ArtifactStore, CBCTAgent, synthetic
from ingestion_agents.cbct_agent import HU_SATURATION, _read_series, pseudonymize


# --- seudonimización ------------------------------------------------------- #
def test_el_seudonimo_es_estable() -> None:
    """El mismo paciente debe dar el mismo seudónimo entre adquisiciones:
    es lo que permite montar su serie temporal sin conocer su identidad."""
    assert pseudonymize("PAC-001", salt="s") == pseudonymize("PAC-001", salt="s")


def test_pacientes_distintos_seudonimos_distintos() -> None:
    assert pseudonymize("PAC-001", salt="s") != pseudonymize("PAC-002", salt="s")


def test_el_seudonimo_depende_de_la_sal() -> None:
    """Sin la sal no se puede reidentificar: la sal es el dato a proteger (RGPD)."""
    assert pseudonymize("PAC-001", salt="a") != pseudonymize("PAC-001", salt="b")


def test_el_seudonimo_no_contiene_el_identificador() -> None:
    assert "PAC-001" not in pseudonymize("PAC-001", salt="s")


def test_sal_desde_el_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASH_PSEUDONYM_SALT", "sal-de-test")
    assert pseudonymize("PAC-001") == pseudonymize("PAC-001", salt="sal-de-test")


# --- ingesta --------------------------------------------------------------- #
def test_ingesta_de_la_serie_sintetica(cbct_dir: Path, store: ArtifactStore) -> None:
    outcome = CBCTAgent(store).ingest(cbct_dir)

    assert outcome.status is ModalityStatus.OK
    assert outcome.modality is Modality.CBCT
    assert outcome.support is Support.VOLUMETRIC

    arrays = store.load(outcome.artifact_ref or "")
    assert set(arrays) == {
        "centers", "scales", "rotations", "density", "origin", "hu_range",
        "paso", "n_origen",
    }
    n = arrays["centers"].shape[0]
    assert (arrays["scales"].shape, arrays["rotations"].shape) == ((n, 3), (n, 4))
    assert outcome.n_primitives == n


def test_el_campo_vuelve_a_coordenadas_reales(cbct_dir: Path, store: ArtifactStore) -> None:
    """El centrado en el origen tiene que ser DESHACIBLE, o el campo es irreversible.

    Restar el centroide y olvidarlo es la clase de pérdida que ninguna versión del agente
    recupera: depende del dato, no del código. Con `origin` guardado, la exportación puede
    devolver el campo a las coordenadas del CBCT y medir la reversibilidad de verdad.
    """
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    mundo = arrays["centers"].astype(np.float64) + arrays["origin"]

    serie = _read_series(cbct_dir)
    sx, sy, _ = serie.spacing
    ocupados = np.argwhere(serie.volume >= CBCTAgent(store).hu_threshold)
    esperado = np.column_stack(
        [ocupados[:, 2] * sx, ocupados[:, 1] * sy, serie.z[ocupados[:, 0]]]
    )
    # Mismas cotas del volumen original, holgura por el float32 de `centers`.
    assert np.abs(mundo.min(axis=0) - esperado.min(axis=0)).max() < 1e-3
    assert np.abs(mundo.max(axis=0) - esperado.max(axis=0)).max() < 1e-3
    assert arrays["origin"].shape == (3,)


def test_la_densidad_vuelve_a_hounsfield(cbct_dir: Path, store: ArtifactStore) -> None:
    """`hu_range` es lo que permite deshacer la normalización a [0, 1].

    Sin él, un σ de 0,5 no significa nada legible: los dos extremos son configuración del
    agente, y reconstruirlos «por la versión» ataría el fichero exportado al código.
    """
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    bajo, alto = arrays["hu_range"]
    assert (bajo, alto) == (CBCTAgent(store).hu_threshold, HU_SATURATION)
    # La densidad no saturada es invertible; la saturada solo dice «≥ alto».
    hu = arrays["density"].astype(np.float64) * (alto - bajo) + bajo
    assert hu.min() >= bajo
    assert float(hu.max()) == pytest.approx(alto)


def test_la_densidad_esta_normalizada(cbct_dir: Path, store: ArtifactStore) -> None:
    """σ ≥ 0 es un invariante del contrato (`GaussianPrimitive.density`)."""
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    density = arrays["density"]
    assert density.min() >= 0.0
    assert density.max() <= 1.0


def test_el_esmalte_satura_la_densidad(cbct_dir: Path, store: ArtifactStore) -> None:
    """El esmalte sintético está a 2200 HU, por encima de la saturación: debe dar σ=1."""
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    assert synthetic._HU_ENAMEL > HU_SATURATION
    assert float(arrays["density"].max()) == pytest.approx(1.0)


def test_la_semilla_no_inventa_rotacion(cbct_dir: Path, store: ArtifactStore) -> None:
    """Cuaternión identidad (w,x,y,z): la anisotropía la aprende el optimizador
    RGS, no la ingesta."""
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    rotations = arrays["rotations"]
    np.testing.assert_allclose(rotations, np.tile([1.0, 0, 0, 0], (len(rotations), 1)))


def test_el_campo_queda_centrado(cbct_dir: Path, store: ArtifactStore) -> None:
    arrays = store.load(CBCTAgent(store).ingest(cbct_dir).artifact_ref or "")
    np.testing.assert_allclose(arrays["centers"].mean(axis=0), 0.0, atol=1e-3)


def test_seudonimiza_el_patient_id_del_dicom(cbct_dir: Path, store: ArtifactStore) -> None:
    """El identificador directo del DICOM nunca debe salir del agente."""
    agent = CBCTAgent(store)
    agent.ingest(cbct_dir)
    assert agent.patient_pseudonym is not None
    assert "SYNTH-0001" not in agent.patient_pseudonym


def test_el_submuestreo_respeta_el_tope(cbct_dir: Path, store: ArtifactStore) -> None:
    outcome = CBCTAgent(store, max_primitives=500).ingest(cbct_dir)
    assert outcome.n_primitives is not None and outcome.n_primitives <= 500
    # Se declara que el campo es una submuestra bajando la confianza.
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.9


def test_umbral_inalcanzable_falla_en_vez_de_devolver_vacio(
    cbct_dir: Path, store: ArtifactStore
) -> None:
    """Un campo vacío exportado en silencio sería peor que un fallo declarado."""
    outcome = CBCTAgent(store, hu_threshold=1e6).ingest(cbct_dir)
    assert outcome.status is ModalityStatus.FAILED
    assert "umbral" in (outcome.detail or "")


def test_un_corte_suelto_no_es_una_serie(cbct_dir: Path, store: ArtifactStore) -> None:
    outcome = CBCTAgent(store).ingest(next(cbct_dir.glob("*.dcm")))
    assert outcome.status is ModalityStatus.FAILED
    assert "directorio" in (outcome.detail or "")


def test_directorio_sin_dicom(tmp_path: Path, store: ArtifactStore) -> None:
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    outcome = CBCTAgent(store).ingest(vacio)
    assert outcome.status is ModalityStatus.FAILED
    assert "DICOM" in (outcome.detail or "")


def test_ingesta_reproducible(cbct_dir: Path, store: ArtifactStore) -> None:
    """Submuestreo por paso uniforme, no aleatorio: la misma serie da la misma referencia."""
    a = CBCTAgent(store, max_primitives=1000).ingest(cbct_dir)
    b = CBCTAgent(store, max_primitives=1000).ingest(cbct_dir)
    assert a.artifact_ref == b.artifact_ref


def test_el_orden_de_los_cortes_no_depende_del_nombre(
    tmp_path: Path, store: ArtifactStore
) -> None:
    """Se ordena por posición física: un nombre de fichero engañoso no debe deformar el volumen."""
    volume, spacing = synthetic.build_volume(spacing=2.5)
    serie = tmp_path / "serie"
    synthetic.write_dicom_series(serie, volume, spacing)

    referencia = CBCTAgent(store).ingest(serie).artifact_ref

    # Se renombran los cortes en orden inverso; el contenido (y su ImagePositionPatient)
    # no cambia, así que el volumen reconstruido debe ser idéntico.
    revuelta = tmp_path / "revuelta"
    revuelta.mkdir()
    slices = sorted(serie.glob("*.dcm"))
    for i, src in enumerate(reversed(slices)):
        (revuelta / f"z{i:04d}.dcm").write_bytes(src.read_bytes())

    assert CBCTAgent(store).ingest(revuelta).artifact_ref == referencia


@pytest.mark.skipif(
    os.environ.get("ASH_PSEUDONYM_SALT") is not None, reason="sal ya definida en el entorno"
)
def test_sal_por_defecto_es_de_desarrollo() -> None:
    """Debe ser evidente que no vale para datos de paciente."""
    assert pseudonymize("PAC-001") == pseudonymize(
        "PAC-001", salt="dev-salt-no-usar-en-produccion"
    )


def _salto_dentro_de_una_fila(centros: np.ndarray) -> float:
    """Mediana del salto entre gaussianas consecutivas DE LA MISMA FILA, en mm.

    ⚠️ Medirlo con `unique` por eje NO sirve, y se comprobó: colapsa todas las filas
    juntas, así que un peine —filas densas muy separadas entre sí— sale igual que una
    rejilla sana. El defecto vive **dentro** de una fila y hay que mirarlo ahí.
    """
    c = np.asarray(centros, dtype=np.float64)
    fila = np.round(c[:, 1:], 4)                       # (y, z) fija
    orden = np.lexsort((c[:, 0], fila[:, 1], fila[:, 0]))
    q = c[orden]
    f = fila[orden]
    misma = (f[1:] == f[:-1]).all(axis=1)
    d = np.diff(q[:, 0])[misma]
    d = d[d > 1e-6]
    return float(np.median(d)) if d.size else 0.0


def test_el_submuestreo_no_deja_la_nube_en_PEINE(
    cbct_dir: Path, store: ArtifactStore
) -> None:
    """El diezmado se hace en la rejilla, no sobre el array de `argwhere`.

    ⚠️ Este es el test que faltaba. `occupied[::step]` recorre un array en orden C —z
    lento, x rápido— así que se comía `step-1` de cada `step` vóxeles **a lo largo de una
    sola fila**. Medido sobre un caso real con `step = 9`: 1,35 mm entre gaussianas
    consecutivas de una fila frente a 0,15 en los otros ejes, con σ de 0,075. El campo
    salía como puntos aislados que no llegaban a tocarse nunca, y nada fallaba: el tope se
    respetaba, la ingesta era reproducible y el número de primitivas era el pedido.

    El listón es la propia σ: si el salto dentro de una fila es mayor que **cuatro veces**
    la σ de ese eje, las gaussianas vecinas ya no se solapan de forma apreciable
    (a 2σ la contribución es 0,14) y el campo deja de reconstruir nada entre ellas.
    """
    salida = CBCTAgent(store, max_primitives=400).ingest(cbct_dir)
    assert salida.artifact_ref is not None
    campo = store.load(salida.artifact_ref)
    salto = _salto_dentro_de_una_fila(campo["centers"])
    sigma_x = float(np.asarray(campo["scales"])[0][0])
    assert salto > 0.0, "no hay dos gaussianas en la misma fila: la nube está deshecha"
    assert salto <= 4.0 * sigma_x, (
        f"la nube está peinada: {salto:.3f} mm entre gaussianas de una misma fila con "
        f"sigma {sigma_x:.3f} mm en ese eje — no llegan a tocarse"
    )


def test_sigma_crece_con_el_diezmado(cbct_dir: Path, store: ArtifactStore) -> None:
    """Media arista de la celda QUE HAY, no de la del vóxel original.

    La otra mitad del mismo fallo: con el campo diezmado, sembrar σ = medio vóxel deja las
    gaussianas muy por debajo de su separación real y el campo no reconstruye nada entre
    ellas. σ tiene que escalar con el paso.
    """
    entero = CBCTAgent(store, max_primitives=10**9).ingest(cbct_dir)
    diezmado = CBCTAgent(store, max_primitives=400).ingest(cbct_dir)
    assert entero.artifact_ref and diezmado.artifact_ref
    s_entero = np.asarray(store.load(entero.artifact_ref)["scales"])[0]
    s_diezmado = np.asarray(store.load(diezmado.artifact_ref)["scales"])[0]
    assert (s_diezmado >= s_entero).all(), (s_entero, s_diezmado)
    assert (s_diezmado > s_entero).any(), (
        "el campo se diezmó y ninguna sigma creció: las gaussianas no se tocan"
    )


def test_el_artefacto_DECLARA_que_es_una_submuestra(
    cbct_dir: Path, store: ArtifactStore
) -> None:
    """Un campo diezmado que no dice que lo está es una medida con menos resolución de la
    que aparenta, y desde fuera es indistinguible de una completa.

    ⚠️ El agente ya bajaba su `confidence` a 0,9 al submuestrear, pero eso vive en la
    procedencia del snapshot y no llega al `.uos`. `paso` y `n_origen` viajan **con el
    artefacto**, que es lo que hace que el sidecar del contenedor pueda declararlo.
    """
    diezmado = CBCTAgent(store, max_primitives=400).ingest(cbct_dir)
    campo = store.load(diezmado.artifact_ref)
    assert "paso" in campo and "n_origen" in campo
    paso = np.asarray(campo["paso"])
    assert paso.shape == (3,) and (paso >= 1).all()
    assert int(paso.prod()) > 1, "se diezmó y el paso declarado es (1,1,1)"
    assert int(campo["n_origen"]) > len(campo["centers"])


def test_sin_diezmar_el_paso_declarado_es_UNO(
    cbct_dir: Path, store: ArtifactStore
) -> None:
    """El contrario, para que el test de arriba pueda fallar: si `paso` fuera siempre > 1
    los dos pasarían y ninguno probaría nada."""
    entero = CBCTAgent(store, max_primitives=10**9).ingest(cbct_dir)
    campo = store.load(entero.artifact_ref)
    assert (np.asarray(campo["paso"]) == 1).all()
    assert int(campo["n_origen"]) == len(campo["centers"])
    assert entero.provenance is not None and entero.provenance.confidence >= 0.9
