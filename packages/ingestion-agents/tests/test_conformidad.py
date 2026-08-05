"""Suite de conformidad: lo que el contrato exige a **todos** los agentes de ingesta.

Los tests de cada agente comprueban lo suyo —que el OBJ triangula los quads, que el
seudónimo del DICOM es estable, que el pH se empareja con su diente—. Ninguno
comprueba lo que `base.py` promete de todos por igual, y por eso estaba desigual:
antes de esta suite, `mesh-agent` e `image-agent` no probaban el camino MISSING, y
`report-agent` no probaba que su ingesta fuera reproducible.

**Un fichero, todos los agentes.** Cada prueba está parametrizada sobre la lista
`CASOS`, así que añadir un agente a esa lista lo somete a las nueve reglas de golpe.
Es la diferencia con generar un esqueleto correcto: el esqueleto se escribe una vez
y no vuelve a mirar; esto se ejecuta en cada PR y también sobre lo ya escrito.

Lo que se exige aquí sale del contrato, no del gusto de nadie:

- **Nunca lanzar** (`base.py`: "no hay canal de excepción hacia el orquestador").
  Un DICOM corrupto que tumbara el proceso se llevaría por delante las otras
  modalidades, y el brief exige >95 % de fiabilidad de la ingesta.
- **Los tres caminos** (`OK` / `MISSING` / `FAILED`) con la carga que corresponde a
  cada uno: `provenance` solo si hubo dato, `detail` solo si algo salió mal.
- **Reproducibilidad**: dos ingestas del mismo fichero producen el mismo artefacto.
  Sin esto, el `sha256` del ArtifactStore no direcciona nada.
- **Soberanía del dato**: la cuarentena registra la *ruta* y el traceback, nunca el
  contenido del fichero clínico (AGENTS.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from core_schemas import ModalityStatus
from ingestion_agents import ArtifactStore
from ingestion_agents.base import BaseIngestionAgent, IngestionAgent
from ingestion_agents.cbct_agent import CBCTAgent
from ingestion_agents.image_agent import ImageAgent
from ingestion_agents.mesh_agent import MeshAgent
from ingestion_agents.report_agent import ReportAgent


# --------------------------------------------------------------------------- #
# Qué agentes se someten y con qué material
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Caso:
    """Un agente y las tres entradas que hacen falta para probar su contrato."""

    nombre: str
    construir: Callable[[ArtifactStore, Path], BaseIngestionAgent]
    valida: Callable[[Path, Path], Path]
    """(caso_sintetico, tmp) -> fuente que el agente ingiere bien."""
    corrupta: Callable[[Path], Path]
    """(tmp) -> fuente de su propio tipo que hace fallar `_ingest`."""


def _png(tmp: Path, datos: bytes | None = None) -> Path:
    """Una foto de verdad, o basura con extensión de foto si se pasan `datos`."""
    destino = tmp / "foto.png"
    if datos is not None:
        destino.write_bytes(datos)
        return destino
    from PIL import Image

    pixeles = (np.random.default_rng(0).random((32, 24, 3)) * 255).astype(np.uint8)
    Image.fromarray(pixeles).save(destino)
    return destino


def _obj_roto(tmp: Path) -> Path:
    destino = tmp / "roto.obj"
    destino.write_text("esto no es una malla\n", encoding="utf-8")
    return destino


def _dicom_roto(tmp: Path) -> Path:
    carpeta = tmp / "cbct-roto"
    carpeta.mkdir()
    (carpeta / "corte.dcm").write_bytes(b"no soy un DICOM")
    return carpeta


def _pdf_roto(tmp: Path) -> Path:
    destino = tmp / "informe.pdf"
    destino.write_bytes(b"%PDF-1.4 pero por dentro no lo es")
    return destino


CASOS: tuple[Caso, ...] = (
    Caso(
        nombre="mesh-agent",
        construir=lambda store, q: MeshAgent(store, quarantine_dir=q),
        valida=lambda caso, tmp: caso / "scan_upper.obj",
        corrupta=_obj_roto,
    ),
    Caso(
        nombre="cbct-agent",
        construir=lambda store, q: CBCTAgent(store, quarantine_dir=q),
        valida=lambda caso, tmp: caso / "cbct",
        corrupta=_dicom_roto,
    ),
    Caso(
        nombre="image-agent",
        construir=lambda store, q: ImageAgent(store, quarantine_dir=q),
        valida=lambda caso, tmp: _png(tmp),
        corrupta=lambda tmp: _png(tmp, b"esto no es un PNG"),
    ),
    Caso(
        nombre="report-agent",
        construir=lambda store, q: ReportAgent(quarantine_dir=q),
        valida=lambda caso, tmp: caso / "informe.txt",
        corrupta=_pdf_roto,
    ),
)

_ids = [c.nombre for c in CASOS]
conforme = pytest.mark.parametrize("caso", CASOS, ids=_ids)


@pytest.fixture
def cuarentena(tmp_path: Path) -> Path:
    return tmp_path / "cuarentena"


def _agente(caso: Caso, tmp_path: Path, cuarentena: Path) -> BaseIngestionAgent:
    return caso.construir(ArtifactStore(tmp_path / "artifacts"), cuarentena)


def _sin_tiempos(valor: Any) -> Any:
    """Quita las marcas temporales para poder comparar dos ingestas.

    `Provenance.ingested_at` es `datetime.now()`: cambia entre ejecuciones sin que
    el resultado sea distinto. Lo que tiene que coincidir es la carga útil.
    """
    if isinstance(valor, dict):
        return {k: _sin_tiempos(v) for k, v in valor.items() if k not in ("ingested_at",)}
    if isinstance(valor, list):
        return [_sin_tiempos(v) for v in valor]
    return valor


def _carga(salida: Any) -> Any:
    """Lo que dos ingestas del mismo fichero deben producir idéntico."""
    return (
        salida.status,
        salida.artifact_ref,
        salida.n_primitives,
        _sin_tiempos([o.model_dump(mode="json") for o in salida.regional]),
    )


# --------------------------------------------------------------------------- #
# 1. Identidad: el orquestador tiene que poder preguntar quién es
# --------------------------------------------------------------------------- #
@conforme
def test_declara_su_identidad(caso: Caso, tmp_path: Path, cuarentena: Path) -> None:
    agente = _agente(caso, tmp_path, cuarentena)
    assert agente.name == caso.nombre
    assert agente.version, "un agente sin versión no se puede trazar en la provenance"
    assert agente.modality is not None
    assert agente.support is not None
    # El orquestador depende del Protocol, no de la clase base (base.py).
    assert isinstance(agente, IngestionAgent)


# --------------------------------------------------------------------------- #
# 2. Los tres caminos del contrato
# --------------------------------------------------------------------------- #
@conforme
def test_camino_ok(caso: Caso, case_dir: Path, tmp_path: Path, cuarentena: Path) -> None:
    agente = _agente(caso, tmp_path, cuarentena)
    salida = agente.ingest(caso.valida(case_dir, tmp_path))

    assert salida.status is ModalityStatus.OK, salida.detail
    assert salida.ok
    assert salida.provenance is not None, "una ingesta correcta siempre declara procedencia"
    assert salida.provenance.agent == f"{agente.name}@{agente.version}"
    assert salida.provenance.modality is agente.modality
    assert salida.detail is None, "no se explica un fallo que no ha ocurrido"
    assert salida.quarantine_ref is None
    assert salida.latency_s >= 0.0
    # La proyección que viaja dentro del TwinSnapshot tiene que cuadrar.
    assert salida.ingestion.modality is agente.modality
    assert salida.ingestion.status is ModalityStatus.OK


@conforme
def test_camino_missing(caso: Caso, tmp_path: Path, cuarentena: Path) -> None:
    """Modalidad no aportada: se declara, no se inventa ni se rompe."""
    agente = _agente(caso, tmp_path, cuarentena)
    salida = agente.ingest(tmp_path / "no-existe" / "nada.dat")

    assert salida.status is ModalityStatus.MISSING
    assert salida.detail, "MISSING sin motivo deja al orquestador adivinando"
    assert salida.provenance is None, "no hay procedencia de un fichero que no existe"
    assert salida.artifact_ref is None
    assert salida.regional == []
    assert not cuarentena.exists(), "una modalidad ausente no es un fallo que aislar"


@conforme
def test_camino_failed_deja_rastro(caso: Caso, tmp_path: Path, cuarentena: Path) -> None:
    agente = _agente(caso, tmp_path, cuarentena)
    salida = agente.ingest(caso.corrupta(tmp_path))

    assert salida.status is ModalityStatus.FAILED
    assert salida.detail, "un fallo sin motivo no es diagnosticable"
    assert salida.quarantine_ref, "un fallo sin cuarentena no se puede inspeccionar"
    assert Path(salida.quarantine_ref).exists()


# --------------------------------------------------------------------------- #
# 3. Fail-loud: pase lo que pase, devuelve un resultado
# --------------------------------------------------------------------------- #
@conforme
def test_nunca_lanza(caso: Caso, tmp_path: Path, cuarentena: Path) -> None:
    """La regla que sostiene el pipeline entero: el fallo es un dato, no una excepción.

    Si una modalidad pudiera lanzar, un DICOM corrupto se llevaría por delante la
    malla y el informe del mismo paciente.
    """
    agente = _agente(caso, tmp_path, cuarentena)

    hostiles = tmp_path / "hostiles"
    hostiles.mkdir()
    (hostiles / "vacio.obj").write_bytes(b"")
    (hostiles / "binario.dcm").write_bytes(bytes(range(256)) * 8)
    (hostiles / "ñandú — informe (1).txt").write_text("¿pH?", encoding="utf-8")
    (hostiles / "sin_extension").write_bytes(b"nada")
    (hostiles / "carpeta_vacia").mkdir()

    for fuente in sorted(hostiles.iterdir()) + [hostiles]:
        salida = agente.ingest(fuente)  # si esto lanza, el test falla solo
        assert salida.status in tuple(ModalityStatus), f"estado inválido con {fuente.name}"
        assert salida.agent == f"{agente.name}@{agente.version}"
        if salida.status is not ModalityStatus.OK:
            assert salida.detail, f"{fuente.name}: fallo mudo"


@conforme
def test_sin_cuarentena_configurada_sigue_devolviendo(caso: Caso, tmp_path: Path) -> None:
    """La cuarentena es diagnóstico opcional: su ausencia no puede ser el fallo."""
    agente = caso.construir(ArtifactStore(tmp_path / "artifacts"), None)  # type: ignore[arg-type]
    salida = agente.ingest(caso.corrupta(tmp_path))

    assert salida.status is ModalityStatus.FAILED
    assert salida.quarantine_ref is None
    assert salida.detail


# --------------------------------------------------------------------------- #
# 4. Reproducibilidad y soberanía del dato
# --------------------------------------------------------------------------- #
@conforme
def test_ingesta_reproducible(caso: Caso, case_dir: Path, tmp_path: Path, cuarentena: Path) -> None:
    """Mismo fichero, mismo artefacto.

    El ArtifactStore está direccionado por contenido: si dos ingestas del mismo
    fichero dieran `sha256` distinto, la referencia dejaría de identificar el dato.
    """
    fuente = caso.valida(case_dir, tmp_path)
    primera = _agente(caso, tmp_path / "a", cuarentena).ingest(fuente)
    segunda = _agente(caso, tmp_path / "b", cuarentena).ingest(fuente)

    assert primera.status is ModalityStatus.OK
    assert _carga(primera) == _carga(segunda)


@conforme
def test_la_cuarentena_no_copia_el_dato_clinico(
    caso: Caso, tmp_path: Path, cuarentena: Path
) -> None:
    """Se registra la ruta y el traceback, nunca el contenido (AGENTS.md).

    Copiar un DICOM a un directorio de cuarentena duplicaría dato de paciente fuera
    del almacenamiento autorizado.
    """
    agente = _agente(caso, tmp_path, cuarentena)
    fuente = caso.corrupta(tmp_path)
    marca = "SECRETO-DEL-PACIENTE-42"
    objetivo = fuente if fuente.is_file() else next(f for f in fuente.iterdir() if f.is_file())
    objetivo.write_bytes(objetivo.read_bytes() + marca.encode())

    salida = agente.ingest(fuente)
    assert salida.quarantine_ref

    registro = Path(salida.quarantine_ref)
    assert marca not in registro.read_text(encoding="utf-8"), "el contenido clínico se filtró"
    for suelto in cuarentena.rglob("*"):
        if suelto.is_file():
            assert suelto.suffix == ".json", f"la cuarentena guardó un {suelto.suffix}"
