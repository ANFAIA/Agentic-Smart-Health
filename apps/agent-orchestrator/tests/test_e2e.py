"""Integración extremo a extremo: **ingesta → fusión → segmentación → exportación**.

Lo que prueba este fichero no es ninguna etapa por separado —para eso están
`test_pipeline.py` y los tests de cada paquete— sino que **el recorrido completo existe**:
un caso entra como ficheros crudos, se convierte en `TwinSnapshot`, se enriquece, y vuelve
a salir como ficheros con el error de reconstrucción **medido**.

Es la prueba que cierra la promesa del brief («el proceso es reversible: el sistema puede
regenerar ficheros directamente desde el Digital Twin»), y la que habría destapado que el
campo gaussiano no era reversible mucho antes de que se notara: el `cbct-agent` restaba el
centroide y lo tiraba, y no había ningún recorrido que volviese a pedirlo.

**Dos cosas son de mentira aquí, y las dos a propósito.**

El *caso* es el sintético de `ingestion_agents.synthetic`, no dato clínico: una arcada
paramétrica con su serie DICOM y su informe. Sirve porque lo que se comprueba aquí es el
**recorrido**, no la anatomía.

El *modelo* de segmentación es `_segmentador`, un doble de test, porque el Point
Transformer necesita GPU y un checkpoint sin versionar. Eso no debilita lo que se prueba:
lo que la integración tiene que demostrar es que la etapa está **atada** —que sus
etiquetas llegan al fichero de salida—, y para eso el origen de las etiquetas da igual.
La calidad del modelo se mide en otro sitio, y no es un test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from agent_orchestrator import CaseInput, IngestionPipeline
from core_schemas import ModalityStatus
from export_agents import (
    RENDER_PSNR_BUDGET_DB,
    RENDER_SSIM_BUDGET,
    REVERSIBILITY_BUDGET_MM,
    lee_ply,
)
from ingestion_agents import ArtifactStore, synthetic
from ingestion_agents.mesh_agent import parse_obj, parse_stl
from ingestion_agents.ontology import all_fdi_codes


@pytest.fixture(scope="session")
def case_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("caso-e2e") / "acq-001"
    synthetic.write_case(root, spacing=1.2)
    return root


@pytest.fixture
def pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        ArtifactStore(tmp_path / "artifacts"), quarantine_dir=tmp_path / "quarantine"
    )


def _segmentador(fdis: list[str], *, prob: float = 0.99):
    """Doble de test del `Segmenter`: parte el campo en tantas zonas como FDI se pidan.

    Se corta por `x` ordenado para que cada zona salga **conexa**: la agregación por
    componentes descartaría los trozos pequeños y el agente tendría razón al quejarse. El
    índice de clase sale del convenio de `DEFAULT_CODES` (0 = encía, 1..32 = permanentes).
    """
    codigos = [all_fdi_codes().index(f) + 1 for f in fdis]
    n_clases = len(all_fdi_codes()) + 1

    def segmentar(points: np.ndarray) -> np.ndarray:
        clases = np.zeros(len(points), dtype=int)
        for codigo, zona in zip(
            codigos, np.array_split(np.argsort(points[:, 0]), len(codigos)), strict=True
        ):
            clases[zona] = codigo
        p = np.full((len(points), n_clases), (1.0 - prob) / (n_clases - 1))
        p[np.arange(len(points)), clases] = prob
        return np.log(p)

    return segmentar


def _nubes_del_twin(pipeline: IngestionPipeline, resultado) -> tuple[np.ndarray, np.ndarray]:
    """Las dos nubes que la fusión geométrica registra, sacadas del propio twin.

    No un blob sintético registrado contra sí mismo: la malla que ingirió el `mesh-agent`
    contra el campo que sembró el `cbct-agent`. Es el registro que el pipeline existe para
    hacer, y usar cualquier otra cosa dejaría la etapa probada pero no *integrada*.

    El campo vive centrado en el origen, así que se le suma `origin` para llevarlo al
    sistema en el que está la malla — que es exactamente para lo que se guarda.
    """
    snapshot = resultado.snapshot
    malla = pipeline.store.load(snapshot.surface_ref)["positions"].astype(np.float64)
    campo = pipeline.store.load(snapshot.gaussian_field_ref)
    centros = campo["centers"].astype(np.float64) + campo["origin"]
    paso = max(1, len(centros) // 4000)
    return malla[:: max(1, len(malla) // 4000)], centros[::paso]


@pytest.fixture
def con_modelo(pipeline: IngestionPipeline, case_dir: Path, tmp_path: Path):
    """El pipeline entero, con el modelo de segmentación puesto, y el caso ya ingerido.

    Hacen falta dos objetos `IngestionPipeline` porque el doble de test necesita saber qué
    dientes buscar y eso solo se sabe **después** de leer el informe. Comparten el almacén,
    así que es un único recorrido: el segundo continúa el snapshot del primero, no lo
    reingiere. Con el modelo de verdad la vuelta no haría falta —encuentra lo que hay sin
    que se lo digan—, y es la única costura que este doble introduce.
    """
    ingerido = pipeline.run(CaseInput.from_case_dir(case_dir))
    fdis = sorted({o.region_id for o in ingerido.snapshot.regional})
    completo = IngestionPipeline(
        pipeline.store,
        quarantine_dir=tmp_path / "quarantine",
        segmenter=_segmentador(fdis),
    )
    return completo, ingerido, fdis


@pytest.fixture
def recorrido(con_modelo, pipeline: IngestionPipeline, tmp_path: Path):
    """El caso completo: **ingesta → fusión → segmentación → export**.

    Es el recorrido de la issue, con las cuatro fases encadenadas por sus datos y no por
    el orden de las llamadas: la fusión geométrica necesita el campo que sembró la
    ingesta, la segmentación necesita ese mismo campo, y el export necesita las dos.

    Devuelve `(resultado, directorio de salida)`.
    """
    completo, ingerido, _ = con_modelo
    salida = tmp_path / "export"
    fusionado = completo.fuse(ingerido, registration=_nubes_del_twin(pipeline, ingerido))
    return completo.exportar(fusionado, salida), salida


# --- el recorrido completo -------------------------------------------------- #
def test_un_caso_entra_como_ficheros_y_sale_como_ficheros(recorrido) -> None:
    """El criterio de aceptación: entrada → export con las métricas en verde."""
    resultado, salida = recorrido

    assert resultado.snapshot is not None
    # Las CUATRO fases, en orden: la ingesta ensambló, la fusión registró, la segmentación
    # etiquetó y ancló el informe, y el export escribió.
    assert [o.modality for o in resultado.outcomes]
    assert any(o.agent.startswith("geometric-fusion-agent") for o in resultado.fusion)
    assert resultado.snapshot.provenance.transform is not None
    assert any(a.agent.startswith("segmentation-agent") for a in resultado.analysis)
    assert any(o.agent.startswith("semantic-fusion-agent") for o in resultado.fusion)
    assert len(resultado.exports) == 4, "malla, campo, compuesto y render"
    assert all(e.ok for e in resultado.exports), [e.detail for e in resultado.exports]
    assert resultado.reversible, [
        (e.agent, e.max_deviation_mm, e.psnr_db, e.ssim) for e in resultado.exports
    ]

    stl = resultado.export("export-agent")
    ply = resultado.export("field-export-agent")
    png = resultado.export("render-export-agent")
    assert stl.path.exists() and stl.path.suffix == ".stl"
    assert ply.path.exists() and ply.path.suffix == ".ply"
    assert png.paths and all(p.exists() for p in png.paths)
    # Todo cuelga del directorio que se pidió: un exportador no escribe fuera.
    for p in (stl.path, ply.path, *png.paths):
        assert salida in p.parents


def test_el_gate_humano_avisa_pero_no_bloquea_la_materializacion(recorrido) -> None:
    """Dos cosas distintas que es fácil confundir en un pipeline: **confianza** y
    **reversibilidad**.

    Registrar la malla del escáner contra los vóxeles de tejido duro del CBCT no es
    registrar dos medidas de la misma superficie: sale 0,605 mm sobre un 52,7 % solapado, y
    la fusión lo marca por debajo de su umbral de confianza. Eso es correcto y tiene que
    verse — pero **no impide exportar**, porque el error de reconstrucción es otra cosa: mide
    si el fichero que sale reproduce el twin, no si el twin está bien registrado.

    Si algún día el gate bloqueara la exportación, este test lo diría.
    """
    resultado, _ = recorrido
    assert resultado.hitl_required
    assert any("confianza" in m for m in resultado.hitl_reasons)
    assert resultado.reversible
    assert all(e.ok for e in resultado.exports)


def test_las_metricas_de_reversibilidad_estan_en_verde(recorrido) -> None:
    """Cada canal con su presupuesto: milímetros para geometría, PSNR/SSIM para imagen."""
    resultado, _ = recorrido
    stl = resultado.export("export-agent")
    ply = resultado.export("field-export-agent")
    png = resultado.export("render-export-agent")

    assert stl.max_deviation_mm is not None and stl.max_deviation_mm < REVERSIBILITY_BUDGET_MM
    assert stl.mean_deviation_mm is not None and stl.mean_deviation_mm <= stl.max_deviation_mm
    # El PLY escribe posiciones en `double`: el ciclo es exacto, no «casi».
    assert ply.max_deviation_mm == 0.0
    assert png.psnr_db is not None and png.psnr_db >= RENDER_PSNR_BUDGET_DB
    assert png.ssim is not None and png.ssim >= RENDER_SSIM_BUDGET


def test_la_geometria_que_sale_es_la_que_entro(recorrido, case_dir: Path) -> None:
    """Sin esto, «reversible» sería una propiedad de los números y no de los ficheros.

    Se comparan las cotas del OBJ original con las del STL regenerado: si el recorrido
    perdiese un eje, lo espejase o lo escalase, las métricas internas podrían seguir en
    verde —cada agente mide contra lo que él mismo escribió— y esto no.
    """
    resultado, _ = recorrido
    original = np.asarray(parse_obj(case_dir / "scan_upper.obj")["positions"], dtype=np.float64)
    regenerado = np.asarray(
        parse_stl(resultado.export("export-agent").path)["positions"], dtype=np.float64
    )
    assert np.abs(np.ptp(regenerado, axis=0) - np.ptp(original, axis=0)).max() < 1e-3
    assert np.abs(regenerado.mean(0) - original.mean(0)).max() < 1e-3


def test_el_campo_exportado_vuelve_a_coordenadas_del_cbct(recorrido, pipeline) -> None:
    """El `origin` del artefacto tiene que sobrevivir al recorrido entero.

    Es lo que faltaba: el `cbct-agent` centraba el campo y **tiraba** el desplazamiento, así
    que ningún fichero exportado podía volver a las coordenadas del CBCT. Aquí se comprueba
    sobre el recorrido, no sobre el agente aislado.
    """
    resultado, _ = recorrido
    arrays = pipeline.store.load(resultado.snapshot.gaussian_field_ref)
    assert "origin" in arrays and "hu_range" in arrays

    leido = lee_ply(resultado.export("field-export-agent").path)
    centrado = np.column_stack([leido["x"], leido["y"], leido["z"]])
    mundo = centrado + arrays["origin"]
    assert np.abs(centrado.mean(0)).max() < 1e-3, "el PLY sale en el marco del twin"
    assert np.abs(mundo.mean(0) - arrays["origin"]).max() < 1e-3


def test_las_etiquetas_de_diente_llegan_hasta_el_fichero(recorrido, con_modelo, pipeline) -> None:
    """La segmentación no es una etapa suelta: lo que infiere tiene que **salir**.

    Es el mismo fallo que el centroide tirado, en otra magnitud. Cada etapa por su cuenta
    pasaba: el `segmentation-agent` etiquetaba y guardaba `region_id` en el artefacto, y el
    `field-export-agent` escribía un PLY correcto —sin la columna—. Solo el recorrido
    entero enseña que el único conocimiento anatómico del pipeline se perdía al escribir, y
    que quien abriese el fichero tendría que volver a segmentar para recuperarlo.
    """
    resultado, _ = recorrido
    _, _, fdis = con_modelo

    # 1. La etapa corrió y encontró los dientes que cita el informe.
    seg = next(a for a in resultado.analysis if a.agent.startswith("segmentation-agent"))
    assert sorted(seg.detected) == fdis

    # 2. El campo del twin quedó etiquetado. Es el artefacto **nuevo** que emitió la
    #    segmentación, no el que sembró la ingesta: el almacén es el mismo.
    campo = pipeline.store.load(resultado.snapshot.gaussian_field_ref)
    assert "region_id" in campo

    # 3. Y esas etiquetas están en el PLY, no solo en el almacén.
    leido = lee_ply(resultado.export("field-export-agent").path)
    assert "region_id" in leido
    assert np.array_equal(leido["region_id"], campo["region_id"])
    assert sorted(str(c) for c in np.unique(leido["region_id"]) if c > 0) == fdis


def test_la_segmentacion_no_rompe_la_reversibilidad(recorrido) -> None:
    """Etiquetar es aditivo: el campo que sale sigue siendo el que entró.

    La etapa reescribe `gaussian_field_ref` apuntando a un artefacto nuevo, así que lo que
    se exporta ya **no** es el blob que sembró la ingesta. Si por el camino se reordenaran
    las gaussianas o se perdiera `origin`, las métricas de geometría lo dirían aquí.
    """
    resultado, _ = recorrido
    ply = resultado.export("field-export-agent")
    assert ply.max_deviation_mm == 0.0
    assert resultado.reversible


# --- lo que el recorrido tiene que declarar en vez de esconder -------------- #
def test_sin_snapshot_no_se_exporta_y_se_dice(pipeline, case_dir: Path, tmp_path: Path) -> None:
    """Sin CBCT no hay twin, y sin twin no hay nada que materializar."""
    caso = CaseInput(acquisition_id="acq-sin-cbct", mesh=case_dir / "scan_upper.obj")
    resultado = pipeline.run(caso)
    assert resultado.snapshot is None

    fuera = pipeline.exportar(resultado, tmp_path / "export")
    assert fuera.exports == []
    assert not fuera.reversible, "no se puede afirmar reversible sin haber exportado"
    assert any("no hay snapshot" in m for m in fuera.hitl_reasons)


def test_un_caso_sin_malla_exporta_el_campo_igualmente(
    pipeline, case_dir: Path, tmp_path: Path
) -> None:
    """Los cuatro canales son independientes: que falte uno no tumba a los otros.

    Y la ausencia de malla es `MISSING`, no `FAILED`: una adquisición puede ser solo CBCT y
    eso no dice nada malo del fichero que sí se escribió.
    """
    resultado = pipeline.run(CaseInput(acquisition_id="acq-solo-cbct", cbct=case_dir / "cbct"))
    fuera = pipeline.exportar(resultado, tmp_path / "export")

    stl = fuera.export("export-agent")
    assert stl.status is ModalityStatus.MISSING and stl.path is None
    assert fuera.export("field-export-agent").ok
    assert fuera.export("render-export-agent").ok
    assert fuera.reversible, "los canales que sí corrieron cumplen su presupuesto"
    assert not any("export-agent" in m for m in fuera.hitl_reasons), (
        "que no haya malla no es motivo de revisión humana"
    )


def test_un_twin_parcial_llega_marcado_hasta_los_ficheros(
    pipeline, case_dir: Path, tmp_path: Path
) -> None:
    """El invariante del ADR 001, comprobado de punta a punta y no por agente.

    La foto tiene que estar **corrupta**, no ausente: un fichero que no existe es `MISSING`
    y eso es normal —una adquisición puede no traer fotos—. Lo que hace parcial a un twin es
    que una modalidad se intentara y reventara.
    """
    rota = tmp_path / "foto-corrupta.png"
    rota.write_bytes(b"\x89PNG\r\n\x1a\n" + b"esto no es un PNG")
    caso = CaseInput(
        acquisition_id="acq-parcial",
        cbct=case_dir / "cbct",
        mesh=case_dir / "scan_upper.obj",
        images=[rota],
    )
    resultado = pipeline.run(caso)
    assert any(i.status is ModalityStatus.FAILED for i in resultado.snapshot.ingestion)

    fuera = pipeline.exportar(resultado, tmp_path / "export")
    assert fuera.hitl_required
    assert any("parcial" in m for m in fuera.hitl_reasons)
    # Y lo dice dentro del fichero, que es lo que sobrevive a sacarlo del sistema.
    assert b"PARCIAL" in fuera.export("export-agent").path.read_bytes()[:80]


def test_exportar_no_muta_el_resultado_de_entrada(recorrido, pipeline, tmp_path: Path) -> None:
    """Mismo contrato que `fuse`: devuelve uno nuevo."""
    resultado, _ = recorrido
    antes = len(resultado.exports)
    otro = pipeline.exportar(resultado, tmp_path / "otra-salida")
    assert len(resultado.exports) == antes
    assert len(otro.exports) == antes + 4


def test_el_render_se_puede_desactivar(pipeline, case_dir: Path, tmp_path: Path) -> None:
    """Es el canal más caro; quien solo quiera geometría no debería pagarlo."""
    resultado = pipeline.run(CaseInput.from_case_dir(case_dir))
    fuera = pipeline.exportar(resultado, tmp_path / "export", render=False)
    assert len(fuera.exports) == 3
    assert fuera.export("render-export-agent") is None
    assert fuera.reversible


def test_el_marco_de_la_malla_se_puede_elegir(pipeline, case_dir: Path, tmp_path: Path) -> None:
    """`frame="twin"` sin fusión previa falla declarando, no escribe en el marco equivocado."""
    resultado = pipeline.run(CaseInput.from_case_dir(case_dir))
    fuera = pipeline.exportar(resultado, tmp_path / "export", marco_malla="twin")

    stl = fuera.export("export-agent")
    assert stl.status is ModalityStatus.FAILED
    assert not fuera.reversible
    assert any("falló al exportar" in m for m in fuera.hitl_reasons)
    # Y los otros dos canales siguen adelante.
    assert fuera.export("field-export-agent").ok


def test_la_fusion_desbloquea_el_marco_del_twin(pipeline, case_dir: Path, tmp_path: Path) -> None:
    """Las tres fases atadas por una dependencia real, no por el orden de las llamadas.

    Es el test que de verdad prueba la *integración*: exportar en el sistema del CBCT
    necesita la `RigidTransform` que **solo** escribe la fusión geométrica. Sin esa etapa el
    canal de malla falla declarando (test de arriba); con ella, sale. Si alguien
    desconectara la fusión del recorrido, esto se pondría rojo aunque cada etapa siguiera
    pasando sus propios tests.
    """
    ingerido = pipeline.run(CaseInput.from_case_dir(case_dir))
    fusionado = pipeline.fuse(ingerido, registration=_nubes_del_twin(pipeline, ingerido))
    assert fusionado.snapshot.provenance.transform is not None

    fuera = pipeline.exportar(fusionado, tmp_path / "export", marco_malla="twin")
    stl = fuera.export("export-agent")
    assert stl.ok, stl.detail
    assert stl.frame == "twin"
    assert stl.within_budget, stl.max_deviation_mm
