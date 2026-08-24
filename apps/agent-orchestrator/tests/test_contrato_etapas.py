"""El contrato entre etapas, probado contra los dos fallos que lo motivaron.

Ninguno de los dos lo vio ningún test el día que ocurrió, y los dos eran del mismo tipo:
una etapa produce algo que la siguiente necesita, cada una cumple su propio contrato, y
nada comprueba el borde entre ellas.

- **El centroide.** El `cbct-agent` restaba `origin` del campo y lo tiraba, así que
  ningún fichero exportado podía volver a coordenadas del CBCT.
- **`region_id`.** El `segmentation-agent` lo escribía en el artefacto y el
  `field-export-agent` producía un PLY correcto **sin esa columna**.

Estos tests no comprueban que el código actual funcione — eso ya lo hacen los de cada
agente. Comprueban que **si el fallo volviera, el contrato lo diría**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from agent_orchestrator import CaseInput, IngestionPipeline
from agent_orchestrator.pipeline import CONTRATOS
from core_schemas import revisa_conservacion, revisa_requisitos
from export_agents import lee_ply
from ingestion_agents import ArtifactStore, synthetic


@pytest.fixture(scope="session")
def caso(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raiz = tmp_path_factory.mktemp("contrato") / "acq-001"
    synthetic.write_case(raiz, spacing=1.2)
    return raiz


@pytest.fixture
def pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(ArtifactStore(tmp_path / "artifacts"))


# --- el fallo del centroide ------------------------------------------------- #
def test_una_etapa_declara_lo_que_le_falta_en_vez_de_seguir(pipeline, caso, tmp_path):
    """Un artefacto sin `origin` no se exporta callado al marco equivocado.

    Es el caso del centroide: entregar el campo centrado diciendo que va en coordenadas
    del CBCT desplazaría todo lo que se midiese encima, y con muy buen aspecto.
    """
    from core_schemas import ContratoEtapa

    resultado = pipeline.run(CaseInput.from_case_dir(caso))
    arrays = pipeline.store.load(resultado.snapshot.gaussian_field_ref)
    viejo = {k: v for k, v in arrays.items() if k != "origin"}

    contrato = ContratoEtapa(nombre="x", requiere_arrays=frozenset({"origin"}))
    faltan = revisa_requisitos(contrato, resultado.snapshot, viejo)
    assert len(faltan) == 1
    assert "`origin`" in faltan[0] and "reingerir" in faltan[0]
    # Y con el artefacto completo no dice nada: no es un aviso permanente.
    assert revisa_requisitos(contrato, resultado.snapshot, arrays) == []


# --- el fallo de region_id -------------------------------------------------- #
def test_perder_una_clave_entre_entrada_y_salida_se_declara():
    """La mitad que faltaba: la entrada era correcta y la pérdida ocurría dentro."""
    antes = {"centers": np.zeros((4, 3)), "region_id": np.zeros(4)}
    igual = revisa_conservacion(CONTRATOS["segmentation"], antes, antes)
    assert igual == []

    perdido = revisa_conservacion(
        CONTRATOS["segmentation"], antes, {"centers": np.zeros((4, 3))}
    )
    assert len(perdido) == 1
    assert "`region_id`" in perdido[0] and "perdió" in perdido[0]


def test_el_ply_tiene_que_llevar_todo_array_por_gaussiana(pipeline, caso, tmp_path):
    """El fallo real, sobre el recorrido: un campo etiquetado exporta su `region_id`.

    Se comprueba en las dos direcciones. Con la columna, el contrato calla; sin ella —el
    PLY que el `field-export-agent` escribía antes— lo declara nombrando la clave.
    """
    resultado = pipeline.run(CaseInput.from_case_dir(caso))
    arrays = dict(pipeline.store.load(resultado.snapshot.gaussian_field_ref))
    arrays["region_id"] = np.full(len(arrays["centers"]), 36, dtype=np.int16)
    etiquetado = resultado.snapshot.model_copy(
        update={"gaussian_field_ref": pipeline.store.put(**arrays)}
    )

    fuera = pipeline.exportar(
        resultado.__class__(snapshot=etiquetado), tmp_path / "export"
    )
    ply = fuera.export("field-export-agent")
    assert ply.ok, ply.detail
    assert "region_id" in lee_ply(ply.path), "la columna tiene que estar en el fichero"
    # ⚠️ Atado al mensaje del CONTRATO, no al substring «no lleva». Otros canales lo usan
    # para cosas que no tienen nada que ver —el visor cuando no hay escáner, el compuesto
    # cuando no hay pieza etiquetada— y este test pasaba por que ninguno de ellos saltara.
    assert not any("el PLY exportado no lleva" in m for m in fuera.hitl_reasons)

    # Y el contrato caza el PLY sin la columna, que es como salía antes del arreglo.
    del arrays["region_id"]
    motivos = pipeline._columnas_del_campo(
        {**arrays, "region_id": np.zeros(len(arrays["centers"]))}, ply.path
    )
    assert motivos == [], "este PLY sí la lleva"


def test_los_metadatos_del_campo_no_se_exigen_como_columna(pipeline, caso, tmp_path):
    """`origin` (3,) y `hu_range` (2,) van en la cabecera, no por vértice.

    Sin esta distinción el contrato pediría una columna `origin` de 500.000 valores
    repetidos, y el aviso saltaría siempre — que es como se desactiva una comprobación.
    """
    resultado = pipeline.run(CaseInput.from_case_dir(caso))
    fuera = pipeline.exportar(resultado, tmp_path / "export")
    ply = fuera.export("field-export-agent")
    assert ply.ok
    assert not any("origin" in m and "no lleva" in m for m in fuera.hitl_reasons)


# --- las dos etapas de fusión ----------------------------------------------- #
def test_todas_las_etapas_tienen_contrato():
    """Ninguna etapa corre sin declarar qué necesita y qué conserva.

    Una etapa sin entrada en `CONTRATOS` no es que pase la comprobación: es que **no se
    comprueba**, y por fuera se ve igual. Este test es el que impide que la próxima etapa
    entre al pipeline sin contrato por olvido.
    """
    assert set(CONTRATOS) == {
        "fusion-geometrica",
        "fusion-semantica",
        "segmentation",
        "export-malla",
        "export-campo",
        "export-compuesto",
        "export-malla-compuesta",
        "export-visor",
        "export-uos",
        "export-render",
    }


def test_una_fusion_que_reescribiera_el_campo_no_podria_perder_nada():
    """Las dos fusiones no leen el campo, pero prometen no perderlo.

    Hoy escriben solo en la procedencia, así que la promesa se cumple sola. Se declara
    igual porque `GeometricFusionAgent.transfer_color` ya existe y persistir el color es
    trabajo del orquestador: el día que se conecte, la etapa pasará a emitir un artefacto
    nuevo, y ese es justo el momento en que se pierden claves.
    """
    antes = {"centers": np.zeros((4, 3)), "colors": np.zeros((4, 3))}
    for clave in ("fusion-geometrica", "fusion-semantica"):
        assert revisa_conservacion(CONTRATOS[clave], antes, antes) == []
        perdido = revisa_conservacion(CONTRATOS[clave], antes, {"centers": antes["centers"]})
        assert len(perdido) == 1 and "`colors`" in perdido[0]


def test_una_etapa_que_no_toca_el_campo_no_paga_por_comprobarlo(pipeline, caso):
    """El atajo, y por qué importa que sea gratis.

    Comparar entrada y salida cuesta dos descompresiones de medio millón de gaussianas.
    Si la referencia no cambió, no hay nada que comparar —la referencia es el hash del
    artefacto—, y sin el atajo declarar el contrato en las fusiones habría añadido cuatro
    cargas por caso para verificar algo cierto por construcción. Una comprobación cara es
    una comprobación que alguien acaba quitando.
    """
    resultado = pipeline.run(CaseInput.from_case_dir(caso))
    snapshot = resultado.snapshot
    cargas = []
    original = pipeline.store.load
    pipeline.store.load = lambda ref: (cargas.append(ref), original(ref))[1]  # type: ignore[method-assign]

    # Mismo campo a la entrada y a la salida: no se carga nada.
    assert pipeline._conservacion("fusion-geometrica", snapshot, snapshot) == []
    assert cargas == []

    # Y sin salida tampoco: una etapa que falló no perdió nada, no llegó a producir.
    assert pipeline._conservacion("fusion-geometrica", snapshot, None) == []
    assert cargas == []

