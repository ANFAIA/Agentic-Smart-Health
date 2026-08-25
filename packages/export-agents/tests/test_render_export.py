"""`render-export-agent` — render multivista del campo y métricas del canal de imagen.

El test que manda es el **ciclo completo**: serie DICOM sintética → `cbct-agent` →
artefacto → `field-export-agent` → PLY → render, contra el render directo del twin. Si el
ciclo pierde algo, PSNR y SSIM lo dicen.

Lo demás son tres familias: que el render sea **reproducible** (la aceptación de la
issue), que las métricas **detecten un bug de verdad** (un eje intercambiado, que ninguna
estimación vería), y que la profundidad óptica no dependa de la resolución — que fue un bug
real de este módulo y por eso tiene test de regresión.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from core_schemas import (
    Modality,
    ModalityIngestion,
    ModalityStatus,
    Provenance,
    TwinSnapshot,
)
from export_agents import (
    RENDER_PSNR_BUDGET_DB,
    RENDER_SSIM_BUDGET,
    VISTAS_POR_DEFECTO,
    FieldExportAgent,
    RenderExportAgent,
    Vista,
    beer_lambert,
    escribe_ply,
    lee_ply,
    lee_png,
    profundidad_optica,
    psnr,
    ssim,
)
from ingestion_agents import ArtifactStore, CBCTAgent, synthetic


@pytest.fixture(scope="session")
def cbct_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raiz = tmp_path_factory.mktemp("caso-render")
    codigos = synthetic.upper_arch_codes()
    volumen, sp = synthetic.build_volume(codigos, spacing=1.2)
    return synthetic.write_dicom_series(raiz / "cbct", volumen, sp, patient_id="SYNTH-0001")


def _snapshot(field_ref: str, **kw) -> TwinSnapshot:
    base: dict = dict(
        acquisition_id="ACQ-001",
        timestamp=datetime.now(UTC),
        gaussian_field_ref=field_ref,
        provenance=Provenance(
            source_file="caso/", modality=Modality.CBCT, agent="agent-orchestrator@0.1.0"
        ),
    )
    base.update(kw)
    return TwinSnapshot(**base)


@pytest.fixture
def ingerido(cbct_dir: Path, tmp_path: Path) -> tuple[ArtifactStore, TwinSnapshot, Path]:
    store = ArtifactStore(tmp_path / "artifacts")
    salida = CBCTAgent(store).ingest(cbct_dir)
    assert salida.ok and salida.artifact_ref
    return store, _snapshot(salida.artifact_ref), tmp_path


@pytest.fixture
def con_ply(ingerido) -> tuple[ArtifactStore, TwinSnapshot, Path, Path]:
    """El campo ya exportado a PLY: es el otro lado del ciclo que se mide."""
    store, snapshot, tmp_path = ingerido
    ply = tmp_path / "campo.ply"
    assert FieldExportAgent(store).export(snapshot, ply).ok
    return store, snapshot, tmp_path, ply


# --- el ciclo: la métrica de la issue --------------------------------------- #
def test_el_ciclo_twin_ply_render_no_pierde_nada(con_ply) -> None:
    store, snapshot, tmp_path, ply = con_ply
    salida = RenderExportAgent(store, resolucion=64).export(
        snapshot, tmp_path / "vistas", ply=ply
    )

    assert salida.ok, salida.detail
    assert salida.format == "png"
    assert len(salida.paths) == len(VISTAS_POR_DEFECTO)
    assert all(p.exists() for p in salida.paths)
    assert salida.path == tmp_path / "vistas"
    # El PLY guarda posiciones en `double`, así que el ciclo es exacto: no «casi».
    assert salida.psnr_db == float("inf")
    assert salida.ssim == pytest.approx(1.0)
    assert salida.image_within_budget
    assert not salida.hitl_required


def test_el_ciclo_es_exacto_aunque_el_campo_NO_este_centrado(ingerido, tmp_path) -> None:
    """El caso que el test de arriba no cubre, y que rompió el ciclo en producción.

    El campo del `cbct-agent` sale con el centroide en el origen —`origin` ES la media—,
    así que un verificador que recentrara restando el centroide daba exacto por
    casualidad. El `gaussian-engine` ajusta elipsoides a la densidad y **mueve** el
    centroide: medido sobre un caso real, 4,4 mm sobre una escena de 87 mm. Aquella resta
    pasó a ser una traslación y el ciclo se desplomó a 14,1 dB contra un presupuesto de
    40, sin que reventara nada.

    Aquí se reproduce a mano: se desplaza el campo y se exige que el ciclo siga siendo
    exacto. El marco lo declara la cabecera y se lee de ahí.
    """
    store, snapshot, _ = ingerido
    arrays = dict(store.load(snapshot.gaussian_field_ref))
    arrays["centers"] = np.asarray(arrays["centers"], dtype=np.float64) + [0.04, -1.84, 3.97]
    desplazado = _snapshot(store.put(**arrays), acquisition_id=snapshot.acquisition_id)

    ply = tmp_path / "descentrado.ply"
    assert FieldExportAgent(store).export(desplazado, ply).ok
    salida = RenderExportAgent(store, resolucion=64).export(
        desplazado, tmp_path / "vistas-descentrado", ply=ply
    )

    assert salida.ok, salida.detail
    assert salida.psnr_db == float("inf")
    assert salida.ssim == pytest.approx(1.0)
    assert salida.image_within_budget


def test_el_marco_del_ply_se_LEE_de_la_cabecera_y_no_se_deduce(ingerido, tmp_path) -> None:
    """Un PLY en coordenadas del CBCT lleva `centers + origin` y lo dice. El verificador
    resta el `origin_mm` que declara, no el centroide que le parezca."""
    from export_agents.field import metadatos_ply

    store, snapshot, _ = ingerido
    ply = tmp_path / "en-cbct.ply"
    assert FieldExportAgent(store).export(snapshot, ply, frame="cbct").ok

    meta = metadatos_ply(ply)
    assert meta["frame"] == "cbct"
    origen = np.asarray([float(v) for v in meta["origin_mm"].split()])
    assert np.allclose(origen, store.load(snapshot.gaussian_field_ref)["origin"])

    salida = RenderExportAgent(store, resolucion=64).export(
        snapshot, tmp_path / "vistas-cbct", ply=ply
    )
    assert salida.psnr_db == float("inf")


def test_un_ply_que_no_declara_su_marco_es_un_ERROR_y_no_un_caso_por_defecto(
    ingerido, tmp_path
) -> None:
    """Adivinar el marco es exactamente lo que hubo que arreglar: se falla ruidosamente."""
    from export_agents.render import _desplazamiento

    store, snapshot, _ = ingerido
    ply = tmp_path / "mudo.ply"
    assert FieldExportAgent(store).export(snapshot, ply).ok
    crudo = ply.read_bytes().replace(b"comment frame twin\n", b"", 1)
    ply.write_bytes(crudo)

    with pytest.raises(ValueError, match="marco conocido"):
        _desplazamiento(ply)


def test_las_metricas_cazan_un_eje_intercambiado(con_ply) -> None:
    """El bug que una estimación del error de formato NO vería.

    Es la misma razón por la que el canal de malla relee el STL en vez de estimar: un
    fichero con los ejes cambiados tiene todos los bytes «bien» y describe otra cosa.
    """
    store, snapshot, tmp_path, ply = con_ply
    leido = lee_ply(ply)
    torcido = tmp_path / "torcido.ply"
    columnas = dict(leido)
    columnas["x"], columnas["y"] = leido["y"], leido["x"]   # ← el bug
    # El `frame` va en la cabecera como en cualquier PLY que salga del exportador: lo
    # que este test finge es un BUG de geometria, no un fichero sin declarar su marco.
    escribe_ply(torcido, columnas,
                comentarios=["frame twin", "ejes intercambiados a proposito"])

    salida = RenderExportAgent(store, resolucion=64).export(
        snapshot, tmp_path / "vistas", ply=torcido
    )
    assert salida.ok, "el render se escribe: el fallo es del ciclo, no de la escritura"
    assert salida.psnr_db is not None and salida.psnr_db < RENDER_PSNR_BUDGET_DB
    assert salida.ssim is not None and salida.ssim < RENDER_SSIM_BUDGET
    assert not salida.image_within_budget
    assert salida.hitl_required
    assert any("no reproduce" in m for m in salida.hitl_reasons)


def test_sin_ply_no_hay_metrica_ni_presupuesto_cumplido(ingerido) -> None:
    """Sin el otro lado del ciclo no se puede afirmar nada: se declara, no se inventa."""
    store, snapshot, tmp_path = ingerido
    salida = RenderExportAgent(store, resolucion=64).export(snapshot, tmp_path / "v")
    assert salida.ok and salida.paths
    assert salida.psnr_db is None and salida.ssim is None
    assert not salida.image_within_budget


# --- reproducibilidad: la aceptación literal de la issue -------------------- #
def test_el_render_es_reproducible_byte_a_byte(ingerido) -> None:
    """«Render multivista reproducible desde el snapshot», y reproducible es byte a byte.

    Nada de muestreo aleatorio ni de orden por profundidad: Beer-Lambert es aditivo en la
    profundidad óptica, así que el resultado no depende del orden de las primitivas.
    """
    store, snapshot, tmp_path = ingerido
    a = RenderExportAgent(store, resolucion=64).export(snapshot, tmp_path / "a")
    b = RenderExportAgent(store, resolucion=64).export(snapshot, tmp_path / "b")
    assert a.ok and b.ok
    for x, y in zip(a.paths, b.paths, strict=True):
        assert x.name == y.name
        assert x.read_bytes() == y.read_bytes()


def test_el_orden_de_las_primitivas_no_cambia_la_imagen(ingerido) -> None:
    """Consecuencia de Beer-Lambert, y lo que permite no ordenar por profundidad."""
    store, snapshot, _ = ingerido
    a = store.load(snapshot.gaussian_field_ref)
    c, s, d = (a["centers"].astype(np.float64), a["scales"].astype(np.float64),
               a["density"].astype(np.float64))
    orden = np.random.default_rng(0).permutation(len(c))
    v = Vista(37.0, 12.0)
    tau_a = profundidad_optica(c, s, d, vista=v, resolucion=64)
    tau_b = profundidad_optica(c[orden], s[orden], d[orden], vista=v, resolucion=64)
    assert np.abs(tau_a - tau_b).max() < 1e-9


# --- el bug que tuvo este módulo: τ dependía de la resolución -------------- #
def test_la_profundidad_optica_converge_al_refinar(ingerido) -> None:
    """Regresión. Una integral de línea NO puede depender del tamaño del píxel.

    La primera versión evaluaba τ en el centro del píxel y recortaba σ a medio píxel para
    no perder las gaussianas subpíxel. Eso inflaba la amplitud con el píxel grande: medido
    sobre el CBCT de `histora`, τ máximo pasaba de 34 a 256 px a **226** a 128 px. Se
    arregló depositando masa y dividiendo por el área del píxel.

    El encuadre se fija a mano: si cada resolución eligiera el suyo, la comparación no
    mediría la convergencia sino el encuadre.
    """
    store, snapshot, _ = ingerido
    a = store.load(snapshot.gaussian_field_ref)
    c, s, d = (a["centers"].astype(np.float64), a["scales"].astype(np.float64),
               a["density"].astype(np.float64))
    centro = c.mean(axis=0)
    radio = float(np.linalg.norm(c - centro, axis=1).max()) + 2.0
    encuadre = (np.array([-radio, -radio]), np.array([radio, radio]))

    medias = []
    for res in (64, 128, 256):
        tau = profundidad_optica(
            c, s, d, vista=VISTAS_POR_DEFECTO[0], resolucion=res, encuadre=encuadre
        )
        medias.append(float(tau[tau > 1e-9].mean()))
    # Cuadruplicar el número de píxeles no puede mover la media más de un 50 %.
    assert max(medias) / min(medias) < 1.5, f"tau no converge: {medias}"


def test_la_masa_total_se_conserva_al_refinar(ingerido) -> None:
    """La suma de τ·área es la masa depositada, y no depende de la rejilla."""
    store, snapshot, _ = ingerido
    a = store.load(snapshot.gaussian_field_ref)
    c, s, d = (a["centers"].astype(np.float64), a["scales"].astype(np.float64),
               a["density"].astype(np.float64))
    centro = c.mean(axis=0)
    radio = float(np.linalg.norm(c - centro, axis=1).max()) + 2.0
    encuadre = (np.array([-radio, -radio]), np.array([radio, radio]))

    masas = []
    for res in (64, 128, 256):
        tau = profundidad_optica(
            c, s, d, vista=VISTAS_POR_DEFECTO[0], resolucion=res, encuadre=encuadre
        )
        area_pixel = (2 * radio / res) ** 2
        masas.append(float(tau.sum() * area_pixel))
    assert max(masas) / min(masas) < 1.01, f"la masa no se conserva: {masas}"


# --- física del render ------------------------------------------------------ #
def test_lo_denso_sale_oscuro(ingerido) -> None:
    """Beer-Lambert: más densidad, menos intensidad. Una radiografía, no una foto."""
    store, snapshot, _ = ingerido
    a = store.load(snapshot.gaussian_field_ref)
    c, s = a["centers"].astype(np.float64), a["scales"].astype(np.float64)
    d = a["density"].astype(np.float64)
    v = VISTAS_POR_DEFECTO[0]
    flojo = beer_lambert(profundidad_optica(c, s, d * 0.25, vista=v, resolucion=64))
    fuerte = beer_lambert(profundidad_optica(c, s, d, vista=v, resolucion=64))
    assert fuerte.mean() < flojo.mean()
    assert flojo.max() <= 1.0 and fuerte.min() >= 0.0


def test_un_campo_vacio_da_una_imagen_en_blanco() -> None:
    """Sin materia no hay atenuación: la imagen es blanca, no negra ni un fallo."""
    vacio = np.zeros((0, 3))
    tau = profundidad_optica(vacio, np.zeros((0, 3)), np.zeros(0),
                             vista=VISTAS_POR_DEFECTO[0], resolucion=32)
    assert tau.shape == (32, 32)
    assert np.all(beer_lambert(tau) == 1.0)


# --- las vistas ------------------------------------------------------------- #
def test_las_vistas_se_nombran_por_angulo_y_no_por_anatomia() -> None:
    """Un nombre anatómico mentiría: el eje depende de cómo el equipo escriba el DICOM.

    En este proyecto suponer el significado de un eje en vez de leerlo salió mal tres veces
    sobre el mismo paciente. `az000_el+00` no puede mentir.
    """
    assert Vista(0.0, 0.0).nombre == "az000_el+00"
    assert Vista(90.0, -30.0).nombre == "az090_el-30"
    assert Vista(360.0, 90.0).nombre == "az000_el+90"
    prohibidos = ("oclusal", "vestibular", "lingual", "frontal", "lateral", "axial")
    for v in VISTAS_POR_DEFECTO:
        assert not any(p in v.nombre for p in prohibidos)


def test_la_base_de_la_vista_es_ortonormal_y_dextrogira() -> None:
    """Elegir tres vectores a mano es como se cuela un marco especular."""
    for vista in (*VISTAS_POR_DEFECTO, Vista(37.0, -63.0), Vista(0.0, 89.5)):
        base = vista.base
        assert np.abs(base @ base.T - np.eye(3)).max() < 1e-12, vista.nombre
        assert np.linalg.det(base) == pytest.approx(1.0), vista.nombre


def test_el_encuadre_es_comun_a_todas_las_vistas(ingerido) -> None:
    """Si cada vista se encuadrase sola, dos renders no serían comparables.

    Y una vista podría recortar lo que otra muestra, que en una prueba multivista es
    exactamente lo que no puede pasar.
    """
    store, snapshot, tmp_path = ingerido
    agente = RenderExportAgent(store, resolucion=64)
    a = store.load(snapshot.gaussian_field_ref)
    centers = a["centers"].astype(np.float64)
    bajo, alto = agente._encuadre(centers)
    assert np.allclose(bajo, -alto), "el encuadre está centrado"
    # Cubre el campo entero desde cualquier ángulo: el radio máximo desde el centroide.
    radio = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())
    assert float(alto[0]) >= radio


# --- las métricas como funciones -------------------------------------------- #
def test_psnr_y_ssim_de_dos_imagenes_identicas() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    assert psnr(img, img) == float("inf")
    assert ssim(img, img) == pytest.approx(1.0)


def test_ssim_penaliza_la_estructura_y_psnr_el_error_medio() -> None:
    """Se devuelven las dos porque no miden lo mismo.

    Un desplazamiento global de brillo cambia poco la estructura; ruido de la misma energía
    la destroza. Con una sola cifra los dos casos serían indistinguibles.

    La imagen base tiene que ser **suave**, y eso costó una iteración: con una base de ruido
    uniforme el SSIM también los separa, pero por 0,03 en vez de por 0,7, porque su varianza
    local ya es enorme y añadirle ruido casi no la cambia. Un render de un campo de densidad
    es suave a escala de píxel, así que la base suave es además el caso realista.
    """
    y, x = np.mgrid[0:64, 0:64].astype(np.float64)
    base = 128.0 + 60.0 * np.sin(x / 12.0) * np.cos(y / 15.0)
    rng = np.random.default_rng(0)
    sesgo = base + 10.0
    ruido = base + rng.normal(0.0, 10.0, base.shape)
    # Error cuadrático medio parecido → PSNR parecido…
    assert abs(psnr(base, sesgo) - psnr(base, ruido)) < 3.0
    # …pero el SSIM sí los distingue, y por mucho: ~0,99 contra ~0,57.
    assert ssim(base, sesgo) > 0.9
    assert ssim(base, sesgo) - ssim(base, ruido) > 0.3


def test_las_metricas_no_comparan_formas_distintas() -> None:
    a, b = np.zeros((8, 8)), np.zeros((8, 9))
    with pytest.raises(ValueError, match="No se puede comparar"):
        psnr(a, b)
    with pytest.raises(ValueError, match="No se puede comparar"):
        ssim(a, b)


def test_ssim_exige_una_imagen_mayor_que_la_ventana() -> None:
    with pytest.raises(ValueError, match="más pequeña que la ventana"):
        ssim(np.zeros((4, 4)), np.zeros((4, 4)))


# --- configuración y fallos ------------------------------------------------- #
def test_un_render_sin_vistas_no_existe(ingerido) -> None:
    store, _, _ = ingerido
    with pytest.raises(ValueError, match="Hacen falta vistas"):
        RenderExportAgent(store, vistas=())


def test_una_resolucion_que_no_permite_medir_se_rechaza(ingerido) -> None:
    store, _, _ = ingerido
    with pytest.raises(ValueError, match="demasiado baja"):
        RenderExportAgent(store, resolucion=8)


def test_un_artefacto_que_no_es_campo_es_un_fallo(ingerido) -> None:
    store, _, tmp_path = ingerido
    ref = store.put(positions=np.zeros((3, 3)), faces=np.zeros((1, 3), dtype=np.int32))
    salida = RenderExportAgent(store, resolucion=64).export(_snapshot(ref), tmp_path / "v")
    assert salida.status is ModalityStatus.FAILED
    assert salida.detail is not None and "centers" in salida.detail


def test_referencia_colgante_es_un_fallo(ingerido) -> None:
    store, _, tmp_path = ingerido
    salida = RenderExportAgent(store, resolucion=64).export(
        _snapshot("sha256:" + "0" * 64), tmp_path / "v"
    )
    assert salida.status is ModalityStatus.FAILED
    assert salida.detail is not None and "colgante" in salida.detail.lower()


def test_un_twin_parcial_llega_marcado(ingerido) -> None:
    store, snapshot, tmp_path = ingerido
    roto = snapshot.model_copy(
        update={
            "ingestion": [
                ModalityIngestion(
                    modality=Modality.MESH,
                    status=ModalityStatus.FAILED,
                    detail="el STL no se pudo leer",
                )
            ]
        }
    )
    salida = RenderExportAgent(store, resolucion=64).export(roto, tmp_path / "v")
    assert salida.ok and salida.hitl_required
    assert any("mesh" in m for m in salida.hitl_reasons)


def test_el_png_se_relee_en_escala_de_grises(ingerido) -> None:
    store, snapshot, tmp_path = ingerido
    salida = RenderExportAgent(store, resolucion=64).export(snapshot, tmp_path / "v")
    img = lee_png(salida.paths[0])
    assert img.shape == (64, 64) and img.dtype == np.uint8
