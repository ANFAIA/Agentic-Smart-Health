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
    assert not any("no lleva" in m for m in fuera.hitl_reasons)

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
