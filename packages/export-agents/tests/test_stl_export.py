"""`export-agent` — regeneración de la malla desde el twin (fase 6).

El test que manda es el **round-trip completo**: OBJ → `mesh-agent` → artefacto →
`export-agent` → STL → `parse_stl`. Por eso estas pruebas importan
`ingestion_agents` aunque el paquete no dependa de él: medir la reversibilidad
contra un almacén de mentira no mediría nada: la garantía del brief («error de malla
< 0,1 mm») es sobre el camino real, y el camino real empieza en la ingesta.

Lo demás son los fallos que un exportador tiene que declarar en vez de escribir:
referencia colgante, snapshot sin malla, índice de cara inventado y sistema de
referencia equivocado.
"""

from __future__ import annotations

import json
import struct
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from core_schemas import (
    Modality,
    ModalityIngestion,
    ModalityStatus,
    Provenance,
    RigidTransform,
    TwinSnapshot,
)
from export_agents import (
    REVERSIBILITY_BUDGET_MM,
    ExportAgent,
    ExportOutput,
    SurfaceStore,
    read_stl_triangles,
    stl_header,
    write_binary_stl,
)
from fusion_agents.registration import apply, quaternion_to_matrix
from ingestion_agents import ArtifactStore, MeshAgent
from ingestion_agents.mesh_agent import parse_stl

# Coordenadas del tamaño de una arcada real (~50 mm) y con los ~6 decimales que
# escribe un escáner. Las dos cosas importan para que la medida signifique algo: el
# error de `float32` es **relativo**, así que un cubo unidad daría un número
# optimista; y valores redondos como 12.5 o 30.25 son exactos en binario, así que
# el round-trip saldría con error cero por casualidad y no por ser reversible.
_TETRAEDRO = np.array(
    [
        [10.123456, 12.987654, -30.246813],
        [45.135791, 11.024680, -28.531975],
        [22.753197, 40.864209, -31.129753],
        [25.591357, 20.246802, 8.753197],
    ]
)
_CARAS = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int32)


def _obj(tmp_path: Path, *, color: bool = False) -> Path:
    """Escribe el tetraedro como OBJ (con color por vértice si se pide)."""
    lineas = []
    for i, (x, y, z) in enumerate(_TETRAEDRO):
        extra = f" {0.1 * i:.3f} {0.2 * i:.3f} {0.3 * i:.3f}" if color else ""
        lineas.append(f"v {x:.6f} {y:.6f} {z:.6f}{extra}")
    lineas += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in _CARAS]
    ruta = tmp_path / "arcada.obj"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def _snapshot(surface_ref: str | None, **kw) -> TwinSnapshot:
    base: dict = dict(
        acquisition_id="ACQ-001",
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:campo",
        surface_ref=surface_ref,
        provenance=Provenance(
            source_file="caso/", modality=Modality.CBCT, agent="agent-orchestrator@0.1.0"
        ),
    )
    base.update(kw)
    return TwinSnapshot(**base)


@pytest.fixture
def ingerido(tmp_path: Path) -> tuple[ArtifactStore, TwinSnapshot, Path]:
    """El camino real: un OBJ ingerido de verdad, con su snapshot apuntando al blob."""
    store = ArtifactStore(tmp_path / "artifacts")
    salida = MeshAgent(store).ingest(_obj(tmp_path))
    assert salida.ok and salida.artifact_ref
    return store, _snapshot(salida.artifact_ref), tmp_path


# --- el round-trip: lo que mide la métrica del brief ------------------------ #
def test_round_trip_obj_twin_stl_por_debajo_del_presupuesto(ingerido):
    """OBJ → twin → STL: la geometría vuelve dentro de los 0,1 mm del brief."""
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "salida" / "arcada.stl"

    salida = ExportAgent(store).export(snapshot, destino)

    assert salida.ok
    assert salida.path == destino and destino.exists()
    assert (salida.n_vertices, salida.n_faces) == (4, 4)
    # La medida existe, es la del formato (no cero) y cabe de sobra en el presupuesto.
    assert salida.max_deviation_mm is not None
    assert 0.0 < salida.max_deviation_mm < 1e-4
    assert salida.within_budget
    assert salida.max_deviation_mm < REVERSIBILITY_BUDGET_MM / 1000

    # Y el fichero es la misma malla, releída por el parser de la ingesta.
    leida = parse_stl(destino)
    assert np.allclose(
        np.unique(leida["positions"], axis=0), np.unique(_TETRAEDRO, axis=0), atol=1e-4
    )
    assert len(leida["faces"]) == len(_CARAS)


def test_lo_escrito_lo_lee_el_parser_de_la_ingesta_como_binario(ingerido):
    """Compatibilidad de formato: `parse_stl` detecta binario por el tamaño exacto."""
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "arcada.stl"
    ExportAgent(store).export(snapshot, destino)

    datos = destino.read_bytes()
    assert not datos.startswith(b"solid")  # arruinaría la detección binario/ASCII
    assert len(datos) == 84 + struct.unpack_from("<I", datos, 80)[0] * 50


def test_la_desviacion_se_mide_releyendo_no_estimando(ingerido, monkeypatch):
    """Si el fichero escrito no coincide, la desviación tiene que notarlo.

    La desviación es una **distancia euclídea por vértice**, no el error máximo por
    coordenada: «esta malla se desvía X mm» significa una distancia, y el presupuesto de
    0,1 mm del brief también. La diferencia se ve justo aquí — un desplazamiento de 1 mm en
    los tres ejes está a √3 ≈ 1,73 mm, y medir por coordenada lo reportaría como 1,0,
    quedándose corto en el factor peor: el de un error diagonal.
    """
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "arcada.stl"

    original = write_binary_stl

    def desplazado(path, positions, faces, **kw):
        # Un bug plausible: escribir la malla movida 1 mm. Una estimación del error
        # de `float32` no lo vería; releer el fichero, sí.
        return original(path, positions + 1.0, faces, **kw)

    monkeypatch.setattr("export_agents.stl.write_binary_stl", desplazado)
    salida = ExportAgent(store).export(snapshot, destino)

    assert salida.ok  # el fichero se escribió: el fallo no es de escritura
    assert salida.max_deviation_mm == pytest.approx(np.sqrt(3.0), abs=1e-5)
    # Desplazamiento rígido: todos los vértices se van lo mismo, así que la media
    # (el Chamfer) coincide con el máximo salvo el redondeo de `float32`, que hace que
    # cada vértice se desvíe un pelo distinto. Es lo que distingue un sesgo de un pico.
    assert salida.mean_deviation_mm == pytest.approx(salida.max_deviation_mm, abs=1e-5)
    assert not salida.within_budget
    assert any("presupuesto" in m for m in salida.hitl_reasons)


def test_el_chamfer_distingue_un_pico_de_un_sesgo(ingerido, monkeypatch):
    """Un solo vértice roto sube el máximo pero apenas la media. Y al revés.

    Es la razón de devolver las dos: con una sola cifra, un pico de 1 mm en un vértice de
    100.000 y una malla entera desplazada 1 mm son indistinguibles, y no son el mismo
    problema — el primero es un dato corrupto, el segundo un marco equivocado.
    """
    store, snapshot, tmp_path = ingerido
    original = write_binary_stl

    def un_vertice_roto(path, positions, faces, **kw):
        movido = positions.copy()
        movido[0] += 1.0
        return original(path, movido, faces, **kw)

    monkeypatch.setattr("export_agents.stl.write_binary_stl", un_vertice_roto)
    salida = ExportAgent(store).export(snapshot, tmp_path / "pico.stl")

    assert salida.max_deviation_mm == pytest.approx(np.sqrt(3.0), abs=1e-5)
    assert salida.mean_deviation_mm is not None
    assert salida.mean_deviation_mm < salida.max_deviation_mm / 2


def test_sin_verificar_no_hay_medida_ni_presupuesto_cumplido(ingerido):
    store, snapshot, tmp_path = ingerido
    salida = ExportAgent(store, verify=False).export(snapshot, tmp_path / "arcada.stl")

    assert salida.ok and salida.max_deviation_mm is None
    # Sin medida no se puede afirmar que se cumple: `within_budget` no da el beneficio
    # de la duda.
    assert not salida.within_budget
    assert "sin verificar" in (salida.detail or "")


# --- sistemas de referencia -------------------------------------------------- #
def test_frame_twin_aplica_la_transformacion_registrada(ingerido):
    """`frame='twin'` mueve la malla al sistema del CBCT, no la deja donde estaba."""
    store, _, tmp_path = ingerido
    # Rotación de 90° sobre Z (cuaternión (√2/2, 0, 0, √2/2)) más traslación.
    q = (2**-0.5, 0.0, 0.0, 2**-0.5)
    transform = RigidTransform(rotation=q, translation=(1.0, -2.0, 0.5), rms_mm=0.05)
    ingesta = MeshAgent(store).ingest(_obj(tmp_path))
    snapshot = _snapshot(
        ingesta.artifact_ref,
        provenance=Provenance(
            source_file="caso/",
            modality=Modality.MESH,
            agent="geometric-fusion-agent@0.2.0",
            transform=transform,
        ),
    )
    destino = tmp_path / "twin.stl"

    salida = ExportAgent(store, frame="twin").export(snapshot, destino)

    assert salida.ok and salida.frame == "twin"
    esperado = apply(quaternion_to_matrix(q), np.array(transform.translation), _TETRAEDRO)
    assert np.allclose(
        np.unique(parse_stl(destino)["positions"], axis=0), np.unique(esperado, axis=0), atol=1e-4
    )


def test_frame_twin_sin_transformacion_falla_en_vez_de_entregar_el_sistema_equivocado(ingerido):
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "twin.stl"

    salida = ExportAgent(store, frame="twin").export(snapshot, destino)

    assert salida.status is ModalityStatus.FAILED
    assert "fusión geométrica" in (salida.detail or "")
    assert not destino.exists()


def test_el_frame_del_constructor_se_puede_pisar_por_llamada(ingerido):
    store, snapshot, tmp_path = ingerido
    salida = ExportAgent(store, frame="twin").export(
        snapshot, tmp_path / "fuente.stl", frame="source"
    )
    assert salida.ok and salida.frame == "source"


def test_un_frame_inventado_se_rechaza(ingerido):
    store, snapshot, tmp_path = ingerido
    with pytest.raises(ValueError, match="frame"):
        ExportAgent(store, frame="cbct")  # type: ignore[arg-type]

    salida = ExportAgent(store).export(snapshot, tmp_path / "x.stl", frame="cbct")
    assert salida.status is ModalityStatus.FAILED


# --- fallos que hay que declarar, no escribir -------------------------------- #
def test_snapshot_sin_malla_es_missing_y_no_escribe_nada(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    destino = tmp_path / "nada.stl"

    salida = ExportAgent(store).export(_snapshot(None), destino)

    assert salida.status is ModalityStatus.MISSING
    assert not destino.exists()
    # Y se dice por qué no se malla el campo gaussiano en su lugar.
    assert "no malla el campo gaussiano" in (salida.detail or "")


def test_referencia_colgante_falla_y_deja_cuarentena(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    cuarentena = tmp_path / "quarantine"
    agente = ExportAgent(store, quarantine_dir=cuarentena)

    salida = agente.export(_snapshot("sha256:noexiste"), tmp_path / "x.stl")

    assert salida.status is ModalityStatus.FAILED
    assert "Referencia colgante" in (salida.detail or "")
    registros = list(cuarentena.glob("export-agent-*.json"))
    assert len(registros) == 1
    # Cuarentena = diagnóstico, nunca dato clínico: solo el id de adquisición.
    registro = json.loads(registros[0].read_text(encoding="utf-8"))
    assert registro["acquisition_id"] == "ACQ-001"
    assert registro["agent"] == "export-agent@0.1.0"


def test_artefacto_sin_geometria_no_es_una_malla_exportable(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put(colors_rgb8=np.zeros((3, 3), dtype=np.uint8))

    salida = ExportAgent(store).export(_snapshot(ref), tmp_path / "x.stl")

    assert salida.status is ModalityStatus.FAILED
    assert "positions" in (salida.detail or "")


def test_cara_fuera_de_rango_no_produce_un_stl_con_geometria_inventada(tmp_path):
    """El peor fallo posible sería un fichero válido con geometría falsa."""
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put(positions=_TETRAEDRO, faces=np.array([[0, 1, 99]], dtype=np.int32))
    destino = tmp_path / "x.stl"

    salida = ExportAgent(store).export(_snapshot(ref), destino)

    assert salida.status is ModalityStatus.FAILED
    assert "fuera de rango" in (salida.detail or "")
    assert not destino.exists() and not list(tmp_path.glob("*.tmp"))


def test_malla_sin_caras_no_se_exporta(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put(positions=_TETRAEDRO, faces=np.zeros((0, 3), dtype=np.int32))

    salida = ExportAgent(store).export(_snapshot(ref), tmp_path / "x.stl")

    assert salida.status is ModalityStatus.FAILED
    assert "no tiene caras" in (salida.detail or "")


def test_coordenadas_no_finitas_no_llegan_al_fichero(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    posiciones = _TETRAEDRO.copy()
    posiciones[2, 1] = np.nan
    ref = store.put(positions=posiciones, faces=_CARAS)

    salida = ExportAgent(store).export(_snapshot(ref), tmp_path / "x.stl")

    assert salida.status is ModalityStatus.FAILED
    assert "no finitas" in (salida.detail or "")


def test_el_destino_imposible_se_declara_no_se_lanza(ingerido):
    """El contrato de la familia: **nunca lanza**, también con el disco en contra."""
    store, snapshot, tmp_path = ingerido
    directorio = tmp_path / "soy-un-directorio.stl"
    directorio.mkdir()

    salida = ExportAgent(store).export(snapshot, directorio)

    assert salida.status is ModalityStatus.FAILED
    assert salida.path is None


def test_crea_el_directorio_de_salida_si_no_existe(ingerido):
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "a" / "b" / "c" / "arcada.stl"

    assert ExportAgent(store).export(snapshot, destino).ok
    assert destino.exists()


# --- lo que el fichero declara de sí mismo ---------------------------------- #
def test_la_cabecera_lleva_la_procedencia_y_no_empieza_por_solid(ingerido):
    store, snapshot, tmp_path = ingerido
    destino = tmp_path / "arcada.stl"
    ExportAgent(store).export(snapshot, destino)

    cabecera = destino.read_bytes()[:80].rstrip(b"\0").decode("ascii")
    assert cabecera.startswith("ASH export-agent@0.1.0")
    assert "ACQ-001" in cabecera and "frame=source" in cabecera
    assert "PARCIAL" not in cabecera


def test_un_snapshot_parcial_no_llega_callado_a_exportacion(ingerido):
    """ADR 001: un snapshot parcial debe declararse parcial. Aquí, dos veces."""
    store, snapshot, tmp_path = ingerido
    parcial = snapshot.model_copy(
        update={
            "ingestion": [
                ModalityIngestion(
                    modality=Modality.CBCT,
                    status=ModalityStatus.FAILED,
                    detail="DICOM corrupto",
                ),
                ModalityIngestion(modality=Modality.MESH, status=ModalityStatus.OK),
                # Que falte una modalidad no es un fallo: no se declara como tal.
                ModalityIngestion(modality=Modality.IMAGE, status=ModalityStatus.MISSING),
            ]
        }
    )
    destino = tmp_path / "parcial.stl"

    salida = ExportAgent(store).export(parcial, destino)

    assert salida.ok and salida.hitl_required
    assert sum("parcial" in m for m in salida.hitl_reasons) == 1
    assert "DICOM corrupto" in salida.hitl_reasons[0]
    assert "PARCIAL" in destino.read_bytes()[:80].decode("ascii")


def test_confianza_baja_pide_revision_antes_de_entregar(ingerido):
    store, snapshot, tmp_path = ingerido
    dudoso = snapshot.model_copy(
        update={"provenance": snapshot.provenance.model_copy(update={"confidence": 0.4})}
    )

    salida = ExportAgent(store).export(dudoso, tmp_path / "dudoso.stl")

    assert salida.ok and salida.hitl_required
    assert any("confianza 0.40" in m for m in salida.hitl_reasons)


def test_el_color_del_twin_se_declara_perdido_no_se_finge(tmp_path):
    """El STL es pelado: el color existe en el twin y no cabe en el fichero."""
    store = ArtifactStore(tmp_path / "artifacts")
    ingesta = MeshAgent(store).ingest(_obj(tmp_path, color=True))
    assert "colors_rgb8" in store.load(ingesta.artifact_ref)

    salida = ExportAgent(store).export(_snapshot(ingesta.artifact_ref), tmp_path / "c.stl")

    assert salida.ok and "color por vértice" in (salida.detail or "")


# --- invariantes de la familia ---------------------------------------------- #
def test_exportar_no_toca_el_twin_ni_el_almacen(ingerido):
    """Solo lectura sobre el gemelo: ni muta el snapshot ni necesita `put`."""
    store, snapshot, tmp_path = ingerido
    antes = snapshot.model_dump_json()

    class SoloLectura:
        """Un almacén sin `put`. Si el exportador escribiera, no compilaría el uso."""

        def load(self, ref: str) -> dict[str, np.ndarray]:
            return store.load(ref)

    solo_lectura = SoloLectura()
    assert isinstance(solo_lectura, SurfaceStore)

    salida = ExportAgent(solo_lectura).export(snapshot, tmp_path / "x.stl")

    assert salida.ok
    assert snapshot.model_dump_json() == antes


def test_el_agente_cumple_el_contrato_de_salida(ingerido):
    store, snapshot, tmp_path = ingerido
    salida = ExportAgent(store).export(snapshot, tmp_path / "x.stl")

    assert isinstance(salida, ExportOutput)
    assert salida.agent == "export-agent@0.1.0"
    assert salida.latency_s > 0.0
    assert salida.format == "stl"


# --- piezas sueltas ---------------------------------------------------------- #
def test_la_cabecera_se_trunca_a_ochenta_bytes():
    assert len(stl_header("x" * 200)) == 80
    assert len(stl_header("")) == 80


def test_releer_un_stl_truncado_lo_dice(tmp_path):
    destino = tmp_path / "roto.stl"
    write_binary_stl(destino, _TETRAEDRO, _CARAS)
    destino.write_bytes(destino.read_bytes()[:-10])

    with pytest.raises(ValueError, match="triángulos exigen"):
        read_stl_triangles(destino)

    destino.write_bytes(b"corto")
    with pytest.raises(ValueError, match="cabecera"):
        read_stl_triangles(destino)


def test_un_triangulo_degenerado_no_inventa_una_normal(tmp_path):
    destino = tmp_path / "degenerado.stl"
    posiciones = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    write_binary_stl(destino, posiciones, np.array([[0, 1, 2]], dtype=np.int32))

    datos = destino.read_bytes()
    normal = np.frombuffer(datos, dtype="<3f4", count=1, offset=84)[0]
    assert np.array_equal(normal, np.zeros(3))
