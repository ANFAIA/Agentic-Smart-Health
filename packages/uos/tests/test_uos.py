"""El contenedor y el manifiesto tienen que cumplir el spec, no parecerse a el."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from uos import Asset, Frame, Manifest, Registration, Subject, Visit, validate, write_uos
from uos.contenedor import MANIFIESTO, read_manifest
from uos.manifiesto import AssetKind, Deidentification, PHIState, Regulatory
from uos.validador import Conformance


def _asset(ruta: Path, **kw) -> Asset:
    crudo = ruta.read_bytes()
    base = dict(
        id="asset.ios", kind=AssetKind.MESH_GS_SCENE, visit="v1", uri="scene/scan.stl",
        media_type="model/stl", sha256=hashlib.sha256(crudo).hexdigest(),
        bytes=len(crudo), frame="frame.ios_master",
    )
    return Asset(**{**base, **kw})


def _manifiesto(assets: list[Asset], **kw) -> Manifest:
    base = dict(
        case_id="urn:uuid:0",
        generator={"name": "test", "version": "0"},
        phi_state=PHIState.PSEUDONYMIZED,
        # B-3: declarar el estado sin decir que medidas lo produjeron es una afirmacion
        # que nadie puede comprobar, asi que el bloque es obligatorio fuera de
        # `identified`. Aqui va el minimo: perfil y herramienta, sin opciones aplicadas.
        deidentification=Deidentification(
            profile="DICOM PS3.15 E.1 Basic Application Level Confidentiality Profile",
        ),
        subject=Subject(pseudonym="P-1"),
        canonical_frame=Frame(id="frame.ios_master"),
        visits=[Visit(id="v1", date="2026-08-23")],
        assets=assets,
    )
    return Manifest(**{**base, **kw})


@pytest.fixture
def malla(tmp_path) -> Path:
    # Nombre con identificador, como los de verdad: es lo que NO debe viajar.
    p = tmp_path / "1574 UpperJawScan.stl"
    p.write_bytes(_stl_binario())
    return p


# --- el contenedor ----------------------------------------------------------- #
def test_el_manifiesto_es_la_PRIMERA_entrada_del_zip(tmp_path, malla):
    """Es la identificacion positiva del formato: un lector abre los primeros bytes y ya
    sabe que tiene delante, sin adivinar por la extension."""
    salida = write_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    with zipfile.ZipFile(salida) as z:
        assert z.namelist()[0] == MANIFIESTO


def test_nada_va_comprimido(tmp_path, malla):
    """STORE porque los payloads ya vienen comprimidos y comprimir el ZIP solo rompe el
    acceso aleatorio por rangos, que es lo que permite bajar un asset suelto."""
    salida = write_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    with zipfile.ZipFile(salida) as z:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


def test_lo_que_sale_es_BYTE_IDENTICO_a_lo_que_entro(tmp_path, malla):
    """Referencia, no transcodificacion (§2.1). Es lo que permite afirmar que el
    contenedor no degrada nada, y lo que hace la trazabilidad forense posible."""
    original = malla.read_bytes()
    salida = write_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    with zipfile.ZipFile(salida) as z:
        assert z.read("scene/scan.stl") == original


def test_un_asset_declarado_y_no_aportado_se_declara(tmp_path, malla):
    """Una referencia colgante es un error, no un hueco — igual que en el resto del
    sistema. Un manifiesto que promete algo que no esta es peor que uno que no lo promete."""
    with pytest.raises(ValueError, match="no se aportaron"):
        write_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]), [])


def test_un_fichero_aportado_y_no_declarado_se_declara(tmp_path, malla):
    """Al reves tambien: contenido que el manifiesto no menciona es contenido que nadie
    puede verificar ni saber que hace ahi."""
    otro = tmp_path / "suelto.bin"
    otro.write_bytes(b"x")
    with pytest.raises(ValueError, match="no declara"):
        write_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                    [("scene/scan.stl", malla), ("suelto.bin", otro)])


def test_las_rutas_internas_no_pueden_escapar():
    """Un `..` en una ruta de ZIP es la travesia de directorios clasica: un lector que la
    resuelva ingenuamente escribe FUERA del destino."""
    for mala in ("../fuera.stl", "/absoluta.stl", "scene/../../x.stl"):
        with pytest.raises(ValueError, match="relativas"):
            Asset(id="a", kind=AssetKind.MESH_GS_SCENE, visit="v1", uri=mala,
                  media_type="model/stl", sha256="0" * 64, bytes=1,
                  frame="frame.ios_master")


# --- el validador ------------------------------------------------------------ #
def test_un_hash_que_no_cuadra_invalida(tmp_path, malla):
    """Verificar es la politica en ingesta (§8): si el sha256 no cuadra, el asset no es el
    que el manifiesto dice, y eso invalida el caso entero."""
    a = _asset(malla, sha256="f" * 64)
    salida = write_uos(tmp_path / "caso.uos", _manifiesto([a]),
                         [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("sha256" in e for e in inf.errors)


def test_un_frame_desconectado_del_canonico_invalida(tmp_path, malla):
    """§6: el grafo DEBE ser conexo hacia el canonico. Si no, un asset queda sin forma de
    alinearse y el visor lo colocaria en el sitio equivocado sin poder detectarlo."""
    a = _asset(malla, frame="frame.huerfano")
    salida = write_uos(tmp_path / "caso.uos", _manifiesto([a]),
                         [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("no conecta con el canonico" in e for e in inf.errors)


def test_una_registracion_lo_conecta(tmp_path, malla):
    """Y con la registracion declarada, el mismo caso es valido."""
    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], registrations=[Registration(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="icp_surface", verified_by="user:pedro",
    )])
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert inf.valid, inf.errors


def test_un_registro_automatico_sin_verificar_es_un_AVISO_no_un_error(tmp_path, malla):
    """No invalida el fichero, pero el visor tiene que presentarlo como PROVISIONAL: un
    alineamiento que nadie ha mirado no es lo mismo que uno firmado."""
    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], registrations=[Registration(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="auto_dl", operator="auto:un-agente@0.1.0",
        regulatory=Regulatory(layer=2),
    )])
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert inf.valid
    assert any("PROVISIONAL" in a_ for a_ in inf.warnings)


def test_provisional_mira_QUIEN_lo_calculo_y_no_con_que_algoritmo(tmp_path, malla):
    """El registro que de verdad emitimos, que la regla anterior NO cazaba.

    ⚠️ `provisional` exigía `method == "auto_dl"`. Nuestra única registración declara
    `method: "icp_surface"` —porque eso es lo que usa— con `operator` de máquina y sin
    verificar: automática, sin revisar, 0,666 mm de residuo, y no disparaba el aviso que
    existe justo para ella. Una salvaguarda escrita contra el nombre de UN algoritmo deja
    de funcionar en cuanto alguien usa otro, que es siempre.

    Lo que decide si una alineación es provisional es si la miró una persona. `method`
    describe la técnica y es otro dato.
    """
    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], registrations=[Registration(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="icp_surface", rms_error_mm=0.666,
        operator="auto:geometric-fusion-agent@0.2.0",
        regulatory=Regulatory(layer=2),
    )])
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert inf.valid, inf.errors
    assert any("PROVISIONAL" in a_ and "reg.ct_to_ios" in a_ for a_ in inf.warnings), inf.warnings


def test_una_registracion_automatica_TIENE_que_declarar_su_capa(tmp_path, malla):
    """B-5: `regulatory` no tiene defecto, y una maquina que calcula tiene que decirlo.

    Antes el campo llevaba `default_factory`, asi que toda registracion salia con
    `layer: 1` puesto sin que nadie lo escribiera: una transformada calculada por un ICP
    quedaba declarada tan adquirida como el CBCT del que salio, y no habia forma de
    distinguir «se declaro capa 1» de «nadie lo declaro». Un ICP es computo determinista
    sobre dos nubes: capa 2.
    """
    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], registrations=[Registration(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="icp_surface", operator="auto:geometric-fusion-agent@0.2.0",
    )])
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("no declara `regulatory`" in e for e in inf.errors), inf.errors


def test_un_registro_de_una_PERSONA_sin_verificar_no_es_provisional(tmp_path, malla):
    """El contrario: lo que separa no es que falte `verified_by`, es que no hubo nadie.

    Un registro que alineó una persona a mano y que nadie más ha contrafirmado no es un
    resultado sin supervisar. Avisar de él gastaría la misma palabra en dos cosas distintas
    y le quitaría valor a la que importa.
    """
    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], registrations=[Registration(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="manual", operator="user:pedro",
    )])
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    assert not any("PROVISIONAL" in a_ for a_ in validate(salida).warnings)


def test_layer_3_tiene_que_vivir_en_derived(tmp_path, malla):
    """La regla que hace que `derived/` sea DESMONTABLE (§5.5): si un asset de inferencia
    vive fuera, borrar el directorio no lo quita y el caso deja de ser distribuible donde
    el modulo no esta habilitado."""
    a = _asset(malla, regulatory=Regulatory(layer=3))
    salida = write_uos(tmp_path / "caso.uos", _manifiesto([a]),
                         [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("desmontar" in e for e in inf.errors)


def test_el_nivel_de_conformidad_sale_de_lo_que_hay(tmp_path, malla):
    """UOS-Core es manifiesto + mesh_gs_scene + image2d (§12). Sin volumen no se puede
    declarar UOS-Vol, y decirlo mal haria que un implementador no pudiera fiarse."""
    salida = write_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert Conformance.CORE in inf.levels
    assert Conformance.VOL not in inf.levels


def test_el_manifiesto_se_relee_igual(tmp_path, malla):
    """Ida y vuelta exacta: el manifiesto es el contrato, y un contrato que cambia al
    releerse no sirve para encadenar hashes entre versiones (§8)."""
    m = _manifiesto([_asset(malla)], created=datetime(2026, 8, 23, tzinfo=UTC))
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    assert read_manifest(salida).json_canonico() == m.json_canonico()


def test_un_zip_sin_manifiesto_primero_se_rechaza(tmp_path):
    """Sin identificacion positiva no hay formato: cualquier ZIP se haria pasar por .uos."""
    falso = tmp_path / "falso.uos"
    with zipfile.ZipFile(falso, "w") as z:
        z.writestr("otra_cosa.txt", "x")
        z.writestr(MANIFIESTO, json.dumps({"uos_version": "0.2"}))

    with pytest.raises(ValueError, match="primera entrada"):
        read_manifest(falso)
    assert not validate(falso).valid


# --- el agente --------------------------------------------------------------- #
def test_sin_seudonimo_NO_se_cae_al_identificador_del_caso(tmp_path, malla):
    """El `acquisition_id` sale del nombre del directorio del caso, que en un sistema real
    lleva el nombre del paciente o su numero de historia.

    Un seudonimo por defecto que resulta ser el dato identificable es PEOR que no tener
    seudonimo, porque el `phi_state` diria `pseudonymized` mintiendo. Se declara FAILED.
    """
    from datetime import datetime

    from core_schemas import Modality, Provenance, TwinSnapshot
    from uos import UOSExportAgent

    snap = TwinSnapshot(
        acquisition_id="perez_garcia_juan",
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
    )
    salida = UOSExportAgent(None).export(snap, tmp_path / "caso", malla=malla)

    assert not salida.ok
    assert "seudonimo" in (salida.detail or "")
    assert "perez_garcia_juan" not in (salida.detail or "")


def test_sin_malla_no_hay_frame_canonico_y_se_declara(tmp_path):
    """En UOS el hub geometrico ES el escaner (§2.2). Sin el, todo lo demas queda sin
    ancla y no hay escena que empaquetar — se dice, no se empaqueta a medias."""
    from datetime import datetime

    from core_schemas import Modality, ModalityStatus, Provenance, TwinSnapshot
    from uos import UOSExportAgent

    snap = TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.CBCT, agent="a@0"),
    )
    salida = UOSExportAgent(None).export(
        snap, tmp_path / "caso", pseudonimo="P-1", malla=None
    )

    assert salida.status is ModalityStatus.MISSING
    assert "canonico" in (salida.detail or "")


def test_ningun_nombre_de_fichero_del_proveedor_viaja(tmp_path, malla):
    """Los nombres de fichero de un proveedor llevan identificadores. El de este caso real
    se llamaba `1574 UpperJawScan.stl` y `1574` es el numero de caso.

    Dentro del contenedor todo se nombra por su PAPEL en la escena, y la trazabilidad la
    da el sha256 — que es mas fuerte que un nombre y no identifica a nadie.
    """
    from datetime import datetime

    from core_schemas import Modality, Provenance, TwinSnapshot
    from uos import UOSExportAgent, read_manifest

    foto = tmp_path / "0000144500014386_PEREZ.jpg"
    foto.write_bytes(b"\xff\xd8\xff" + bytes(50))
    snap = TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
    )
    salida = UOSExportAgent(None).export(
        snap, tmp_path / "caso", pseudonimo="P-1", malla=malla, imagenes=[foto]
    )

    assert salida.ok, salida.detail
    m = read_manifest(salida.path)
    uris = " ".join(a.uri for a in m.assets)
    assert "PEREZ" not in uris and "0000144500014386" not in uris
    assert any(a.uri.startswith("sha256:") for a in m.assets if a.id == "asset.img_000")
    # Y la malla igual: su nombre tampoco entra.
    assert malla.stem not in uris
    # ⚠️ Y ahora es MÁS fuerte que antes: el escáner ni siquiera viaja, así que su `uri`
    # es su dirección de contenido. Un `sha256` no puede llevar el nombre de nadie.
    assert any(a.uri.startswith("sha256:") for a in m.assets if a.id == "asset.ios")



def test_un_informe_ilegible_QUEDA_DECLARADO_en_el_manifiesto(tmp_path, malla):
    """Un PDF sin capa de texto **no se pierde**: queda declarado por su `sha256`.

    Antes solo viajaba lo que el `report-agent` conseguia transcribir, asi que un informe
    escaneado desaparecia entero — el gate decia «hay un PDF que nadie pudo leer» y el
    PDF no estaba en ninguna parte para que alguien lo leyera. En el caso real que lo
    destapo ese fichero era el **pasaporte de implantes**: tres implantes con su posicion
    FDI, su fecha, su marca y su lote, y ningun otro documento del caso los mencionaba.

    Obedece la misma regla que el escaner y las fotos: **ningun original viaja dentro**, se
    declara por su direccion de contenido. Lo que se arreglo no es que viaje, es que EXISTA
    en el manifiesto — antes no habia ni asset, y el documento se caia entero del caso.
    """
    from datetime import datetime

    from core_schemas import Modality, Provenance, TwinSnapshot
    from uos import UOSExportAgent, read_manifest

    doc = tmp_path / "APELLIDOS_NOMBRE_Informe.pdf"
    doc.write_bytes(b"%PDF-1.4 escaneado sin texto")
    snap = TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
    )
    salida = UOSExportAgent(None).export(
        snap, tmp_path / "caso", pseudonimo="P-1", malla=malla, informes=[doc],
    )

    assert salida.ok, salida.detail
    m = read_manifest(salida.path)
    docs = [a for a in m.assets if a.id.startswith("asset.doc_")]
    assert len(docs) == 1, "el informe ilegible no existe en el manifiesto"
    # Declarado por su contenido, no por una ruta: no viaja y no lleva el nombre de nadie.
    assert docs[0].uri.startswith("sha256:")
    assert "APELLIDOS" not in docs[0].uri and "NOMBRE" not in docs[0].uri


def _snapshot(**kw):
    """Un snapshot minimo. `surface_ref` y `regional` son lo que alimenta las vistas."""
    from datetime import datetime

    from core_schemas import Modality, Provenance, TwinSnapshot

    base = dict(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
    )
    return TwinSnapshot(**{**base, **kw})


class _Almacen:
    """Devuelve la malla SIN transformar, que es lo que la deja en el frame canonico.

    Con `faces`, como la guarda el `mesh-agent`: desde que la escena se construye de aquí
    (§5.1), una malla sin topología no es una malla — el agente cae al respaldo de convertir
    el STL y pierde las etiquetas, que es exactamente lo que este doble existe para evitar.
    """

    def __init__(self, posiciones):
        import numpy as np

        self.posiciones = posiciones
        n = len(posiciones) - (len(posiciones) % 3)
        self.caras = np.arange(n, dtype="int32").reshape(-1, 3)

    def load(self, _ref):
        return {"positions": self.posiciones, "faces": self.caras}


def _arcada_de_juguete():
    import numpy as np

    pos, etq = [], []
    for cuadrante, signo in ((1, +1.0), (2, -1.0)):
        for pieza in range(1, 9):
            ang = np.deg2rad(10 + (pieza - 1) * 11)
            centro = [signo * 25.0 * np.sin(ang), 25.0 * np.cos(ang), 4.0]
            pos.append(np.random.default_rng(cuadrante * 10 + pieza).normal(
                centro, [1.5, 1.5, 3.0], size=(60, 3)))
            etq += [cuadrante * 10 + pieza] * 60
    t = np.linspace(0, np.pi, 300)
    pos.append(np.stack([28 * np.cos(t), 28 * np.sin(t), np.full_like(t, -3.0)], axis=1))
    etq += [0] * 300
    return np.vstack(pos), np.array(etq, dtype=np.int64)


def test_el_agente_escribe_vistas_y_solo_de_las_piezas_ANOTADAS(tmp_path, malla):
    """Una vista por diente etiquetado serian dieciseis entradas equivalentes. Lo que hace
    util un deep-link es que apunte a donde alguien miro."""
    from datetime import datetime

    from core_schemas import (
        ClinicalAttributes,
        Modality,
        Provenance,
        RegionalObservation,
    )
    from uos import UOSExportAgent
    from uos.vistas import VIEWS

    pos, etq = _arcada_de_juguete()
    obs = RegionalObservation(
        region_id="16", attributes=ClinicalAttributes(ph=5.4),
        timestamp=datetime.now(UTC),
        provenance=Provenance(source_file="x", modality=Modality.REPORT, agent="r@0"),
    )
    snap = _snapshot(surface_ref="sha256:malla", regional=[obs])

    salida = UOSExportAgent(_Almacen(pos)).export(
        snap, tmp_path / "caso", pseudonimo="P-1", malla=malla, etiquetas_ios=etq
    )

    assert salida.ok, salida.detail
    with zipfile.ZipFile(salida.path) as z:
        vistas = json.loads(z.read(VIEWS))["views"]
    ids = {v["id"] for v in vistas}
    assert "view.oclusal" in ids and "view.vestibular_derecha" in ids
    assert "view.pieza_16" in ids
    # El 17 esta etiquetado en la malla y NO lo anota nadie: no tiene vista propia.
    assert "view.pieza_17" not in ids
    assert all(v["visit"] == "v1" for v in vistas)


def test_sin_etiquetas_el_agente_no_inventa_vistas_y_lo_dice_en_los_motivos(tmp_path, malla):
    """El aviso llega al gate, no se queda en el fichero: es un hueco que alguien decide."""
    from uos import UOSExportAgent
    from uos.vistas import VIEWS

    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla
    )

    assert salida.ok, salida.detail
    with zipfile.ZipFile(salida.path) as z:
        assert json.loads(z.read(VIEWS))["views"] == []
    assert any("no lleva vistas" in m for m in salida.hitl_reasons)


def test_reexportar_encima_produce_la_version_2_y_no_un_borrado(tmp_path, malla):
    """Un `.uos` es append-only logico: modificar es encadenar, no sobrescribir."""
    from uos import UOSExportAgent
    from uos.procedencia import CHAIN, Chain

    destino = tmp_path / "caso"
    agente = UOSExportAgent(None)
    primera = agente.export(_snapshot(), destino, pseudonimo="P-1", malla=malla)
    segunda = agente.export(_snapshot(), destino, pseudonimo="P-1", malla=malla)

    assert primera.path == segunda.path
    with zipfile.ZipFile(segunda.path) as z:
        cadena = Chain.model_validate_json(z.read(CHAIN))
    assert [e.version for e in cadena.links] == [1, 2]
    assert cadena.links[1].prev_manifest_sha256 == cadena.links[0].manifest_sha256
    assert validate(segunda.path).version == 2


def test_una_vista_que_apunta_a_una_visita_inexistente_invalida(tmp_path, malla):
    """Un deep-link a una visita que el manifiesto no declara abre en ninguna parte."""
    from uos.vistas import VIEWS

    m = _manifiesto([_asset(malla)])
    salida = write_uos(
        tmp_path / "caso.uos", m, [("scene/scan.stl", malla)],
        extras={VIEWS: json.dumps({"views": [{
            "id": "view.x", "label": "X", "visit": "v9",
            "camera": {"position": [0, 0, 1], "target": [0, 0, 0], "up": [0, 1, 0]},
        }]})},
    )

    inf = validate(salida)

    assert not inf.valid
    assert any("que el manifiesto no declara" in e for e in inf.errors)


def test_dos_vistas_con_el_mismo_id_invalidan(tmp_path, malla):
    """El id es la ancla del deep-link: repetido, `#view=…` es ambiguo."""
    from uos.vistas import VIEWS

    vista = {
        "id": "view.x", "label": "X", "visit": "v1",
        "camera": {"position": [0, 0, 1], "target": [0, 0, 0], "up": [0, 1, 0]},
    }
    salida = write_uos(
        tmp_path / "caso.uos", _manifiesto([_asset(malla)]), [("scene/scan.stl", malla)],
        extras={VIEWS: json.dumps({"views": [vista, vista]})},
    )

    inf = validate(salida)

    assert not inf.valid
    assert any("repetido" in e for e in inf.errors)


def _stl_binario(triangulos: int = 4) -> bytes:
    """Un STL binario mínimo y VÁLIDO.

    Antes las fixtures escribían bytes sueltos y colaban porque el fichero sólo se hasheaba.
    Desde que la escena se construye convirtiéndolo (§5.1), tiene que ser un STL de verdad.
    """
    import struct

    import numpy as np

    rng = np.random.default_rng(0)
    crudo = b"ASH fixture" + bytes(69) + struct.pack("<I", triangulos)
    for _ in range(triangulos):
        v = rng.normal(0, 10, (3, 3)).astype("<f4")
        crudo += np.zeros(3, dtype="<f4").tobytes() + v.tobytes() + b"\x00\x00"
    return crudo


# --- §6: la registracion, que llevaba dos fallos callados --------------------- #
def _registro_de(ruta) -> dict:
    """La unica registracion del contenedor escrito."""
    with zipfile.ZipFile(ruta) as z:
        return json.loads(z.read("manifest.json"))["registrations"][0]



def _con_registro(rms=0.666):
    from datetime import datetime

    from core_schemas import Modality, Provenance, RigidTransform, TwinSnapshot

    return TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0", surface_ref="sha256:1",
        provenance=Provenance(
            source_file="x", modality=Modality.MESH,
            # ⚠️ El ULTIMO agente que toca la procedencia es la fusion SEMANTICA, no la
            # que calculo la ICP. Es justo la confusion que producia el fallo.
            agent="semantic-fusion-agent@0.1.0",
            transform=RigidTransform(
                rotation=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0), rms_mm=rms
            ),
        ),
    )


def test_el_rms_del_registro_LLEGA_al_manifiesto(tmp_path, malla):
    """El fallo que esto guarda: el campo del contrato se llama `rms_mm` y aqui se leia
    `rms_error_mm` con un `getattr(..., None)`, asi que TODOS los contenedores salian con
    `rms_error_mm: null` teniendo el residuo medido.

    §6 lo pide porque una registracion automatica sin `verified_by` es provisional, y el
    residuo es lo unico que dice cuanto de provisional: 0,666 mm y 6 mm no permiten lo
    mismo. Un `null` ahi no es un hueco, es tirar una medida que ya existe.
    """
    import numpy as np
    from uos.agente import UOSExportAgent

    almacen = _Almacen(np.random.default_rng(0).normal(0, 5, (30, 3)))
    salida = UOSExportAgent(almacen).export(
        _con_registro(), tmp_path / "c", malla=malla, pseudonimo="P-1"
    )
    assert salida.ok, salida.detail

    assert _registro_de(salida.path)["rms_error_mm"] == pytest.approx(0.666)


def test_el_operator_acredita_a_QUIEN_registro(tmp_path, malla):
    """El otro fallo del mismo objeto: se escribia `auto:{provenance.agent}`, que nombra
    al ultimo agente que toco la procedencia y no al que calculo la ICP. El contenedor
    acreditaba una medida a quien no la hizo, en el unico campo que existe para eso."""
    import numpy as np
    from uos.agente import UOSExportAgent

    almacen = _Almacen(np.random.default_rng(0).normal(0, 5, (30, 3)))
    salida = UOSExportAgent(almacen).export(
        _con_registro(), tmp_path / "c", malla=malla, pseudonimo="P-1",
        registrador="auto:geometric-fusion-agent@0.2.0",
    )
    assert salida.ok, salida.detail
    reg = _registro_de(salida.path)

    assert reg["operator"] == "auto:geometric-fusion-agent@0.2.0"
    assert "semantic" not in (reg["operator"] or "")


def test_sin_saber_quien_registro_se_deja_NULL_y_no_se_inventa(tmp_path, malla):
    """§6 admite no saber quien registro. No admite escribir a alguien que no fue."""
    import numpy as np
    from uos.agente import UOSExportAgent

    almacen = _Almacen(np.random.default_rng(0).normal(0, 5, (30, 3)))
    salida = UOSExportAgent(almacen).export(
        _con_registro(), tmp_path / "c", malla=malla, pseudonimo="P-1"
    )

    assert _registro_de(salida.path)["operator"] is None


# --- §5.1: metadata odontologica por sub-mesh -------------------------------- #
def _gltf_de(ruta) -> dict:
    """El chunk JSON del `scene.glb` del contenedor."""
    import struct

    with zipfile.ZipFile(ruta) as z:
        glb = z.read("scene/scene.glb")
    return json.loads(glb[20 : 20 + struct.unpack_from("<I", glb, 12)[0]])



def test_la_escena_NO_lleva_el_FDI_ni_por_sub_mesh_ni_por_gaussiana(tmp_path, malla):
    """B-1: `scene/scene.glb` es Layer 1 y el codigo FDI sale de un segmentador.

    Se emitia (0.4.0) para que el picking del §11.3 funcionase en un visor ajeno, y el
    precio era que quitar `derived/` dejaba de quitar la inferencia: la malla seguia
    partida en catorce trozos con su codigo. La revision externa lo declaro bloqueante.

    Un solo primitive, sin `extras`, y la primitiva de gaussianas sin `_REGION_ID`. Las
    etiquetas siguen viajando enteras en `derived/seg_teeth`, que es lo que el segundo
    assert comprueba: la regla no es «se pierde el dato», es «el dato vive en su plano».
    """
    import zipfile

    from uos.agente import UOSExportAgent
    from uos.derivados import SEGMENTACION

    pos, etq = _arcada_de_juguete()
    salida = UOSExportAgent(_Almacen(pos)).export(
        _snapshot(surface_ref="sha256:malla"), tmp_path / "c",
        pseudonimo="P-1", malla=malla, etiquetas_ios=etq,
    )
    assert salida.ok, salida.detail

    g = _gltf_de(salida.path)
    prims = g["meshes"][0]["primitives"]
    assert len(prims) == 1, "la malla viaja partida: eso es Layer 3 horneada en Layer 1"
    assert "extras" not in prims[0]
    for m in g["meshes"]:
        for pr in m["primitives"]:
            assert "uos_fdi" not in (pr.get("extras") or {})
            assert "_REGION_ID" not in pr["attributes"]

    with zipfile.ZipFile(salida.path) as z:
        assert SEGMENTACION in z.namelist(), "las etiquetas tienen que seguir viajando"


def test_el_validador_CAZA_el_FDI_horneado_en_una_escena_de_layer_1(tmp_path, malla):
    """B-1: el check de `derived/` mira donde se DECLARA la capa 3, no donde esta.

    Durante la 0.4.0 el manifiesto declaraba `asset.scene` como Layer 1 y el validador
    pasaba, mientras la escena viajaba partida por diente con `extras.uos_fdi` dentro.
    Este test reconstruye ese contenedor —un GLB parcheado a mano— y exige que ahora
    falle. Sin el, revertir la particion es una decision que nada protege de volver.
    """
    import struct

    from uos.agente import UOSExportAgent
    from uos.validador import validate

    pos, etq = _arcada_de_juguete()
    salida = UOSExportAgent(_Almacen(pos)).export(
        _snapshot(surface_ref="sha256:malla"), tmp_path / "c",
        pseudonimo="P-1", malla=malla, etiquetas_ios=etq,
    )
    assert validate(salida.path).valid, "el contenedor limpio tiene que pasar"

    # El mismo contenedor con el FDI horneado en la escena, como lo emitia la 0.4.0.
    with zipfile.ZipFile(salida.path) as z:
        entradas = [(i, z.read(i.filename)) for i in z.infolist()]
    parcheado = tmp_path / "con-fdi.uos"
    with zipfile.ZipFile(parcheado, "w", zipfile.ZIP_STORED) as z:
        for info, crudo in entradas:
            if info.filename == "scene/scene.glb":
                largo = struct.unpack_from("<I", crudo, 12)[0]
                doc = json.loads(crudo[20:20 + largo])
                doc["meshes"][0]["primitives"][0]["extras"] = {"uos_fdi": "16"}
                cab = json.dumps(doc).encode()
                cab += b" " * (-len(cab) % 4)
                binario = crudo[20 + largo:]
                crudo = (
                    struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(cab) + len(binario))
                    + struct.pack("<II", len(cab), 0x4E4F534A) + cab + binario
                )
            z.writestr(info, crudo)

    inf = validate(parcheado)
    assert not inf.valid
    assert any("uos_fdi" in e for e in inf.errors), inf.errors


def test_la_escena_conserva_el_orden_de_los_vertices(tmp_path, malla):
    """La union entre `derived/seg_teeth` y la escena es POSICIONAL: el codigo `i` es del
    vertice `i`. Con la escena sin partir (B-1) esa union es lo UNICO que queda para
    reconstruir el picking, asi que importa mas que antes: si el orden cambia, la
    segmentacion se pinta sobre los dientes equivocados y nada protesta."""

    from uos.agente import UOSExportAgent

    pos, etq = _arcada_de_juguete()
    salida = UOSExportAgent(_Almacen(pos)).export(
        _snapshot(surface_ref="sha256:malla"), tmp_path / "c",
        pseudonimo="P-1", malla=malla, etiquetas_ios=etq,
    )
    g = _gltf_de(salida.path)
    prims = g["meshes"][0]["primitives"]

    assert len({p["attributes"]["POSITION"] for p in prims}) == 1, "POSITION se ha duplicado"
    assert g["accessors"][prims[0]["attributes"]["POSITION"]]["count"] == len(pos)


# --- §12: el esquema publicado ------------------------------------------------ #
def test_el_esquema_publicado_valida_un_contenedor_de_verdad(tmp_path, malla):
    """§12 pide que el validador contraste contra un «JSON Schema publicado por version».

    Validabamos con Pydantic, que es correcto y es NUESTRO: alguien ajeno no tenia contra
    que comprobar un `.uos`. Un formato cuya unica definicion ejecutable vive dentro de la
    implementacion de referencia no es un formato.
    """
    import json as _json

    import jsonschema
    from uos.agente import UOSExportAgent
    from uos.esquema import esquema_del_manifiesto

    pos, etq = _arcada_de_juguete()
    salida = UOSExportAgent(_Almacen(pos)).export(
        _snapshot(surface_ref="sha256:malla"), tmp_path / "c",
        pseudonimo="P-1", malla=malla, etiquetas_ios=etq,
    )
    assert salida.ok, salida.detail
    with zipfile.ZipFile(salida.path) as z:
        manifiesto = _json.loads(z.read("manifest.json"))

    jsonschema.validate(manifiesto, esquema_del_manifiesto())


def test_el_esquema_del_repositorio_NO_se_queda_atras(tmp_path):
    """El esquema se DERIVA del contrato, pero el fichero publicado es una copia en disco
    y una copia se separa. Este test es lo que obliga a regenerarlo cuando cambia un
    campo, en vez de descubrirlo cuando a alguien de fuera no le validate un contenedor."""
    import json as _json

    from uos.esquema import RUTA, esquema_del_manifiesto

    publicado = Path(__file__).resolve().parents[3] / RUTA
    assert publicado.exists(), f"falta {RUTA}: regenera con `uv run python -m uos.esquema`"
    assert _json.loads(publicado.read_text()) == esquema_del_manifiesto(), (
        f"{RUTA} se ha quedado atras respecto al contrato: regeneralo"
    )


# --- perfil ligero: los originales se declaran y no viajan -------------------- #
def _snap_min():
    from datetime import datetime

    from core_schemas import Modality, Provenance, TwinSnapshot

    return TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
    )


def test_sin_originales_el_STL_se_DECLARA_y_no_viaja(tmp_path, malla):
    """El perfil ligero: identidad del original dentro, fichero fuera.

    Lo que se comprueba es lo que hace que siga siendo auditable: el asset sigue en el
    manifiesto, con su `sha256` y su tamaño REALES —no ceros, no `null`—, y lo único que
    cambia es que sus bytes no están en el ZIP.
    """
    import hashlib
    import zipfile

    from uos import UOSExportAgent, read_manifest

    salida = UOSExportAgent(None).export(
        _snap_min(), tmp_path / "caso", pseudonimo="P-1", malla=malla,
    )
    assert salida.ok, salida.detail
    m = read_manifest(salida.path)
    ios = next(a for a in m.assets if a.id == "asset.ios")

    assert ios.external is True
    assert ios.sha256 == hashlib.sha256(malla.read_bytes()).hexdigest()
    assert ios.bytes == malla.stat().st_size
    with zipfile.ZipFile(salida.path) as z:
        assert ios.uri not in set(z.namelist()), "el original ha viajado igualmente"


def test_sin_originales_el_validador_AVISA_y_no_falla(tmp_path, malla):
    """Un asset que no se puede verificar no es un error, pero tampoco se calla.

    ⚠️ Es la diferencia entre las dos garantías, y por eso hay un test: con el original
    dentro el contenedor afirma «lo que sale es lo que entró» y el validador lo comprueba;
    sin él afirma «sé el hash de lo que debería haber ahí». Si el validador no lo dijera,
    las dos afirmaciones serían indistinguibles desde fuera.

    ⚠️ **Y se dice UNA vez, no una por asset.** Que los originales no viajen es el formato,
    así que un aviso por cada uno repetiría la definición tantas veces como assets tenga el
    caso —y un aviso que sale siempre deja de leerse, enterrando los que sí distinguen algo.
    Lo comprueba `test_los_originales_referenciados_producen_UN_solo_aviso`.
    """
    from uos import UOSExportAgent
    from uos.validador import validate

    salida = UOSExportAgent(None).export(
        _snap_min(), tmp_path / "caso", pseudonimo="P-1", malla=malla,
    )
    inf = validate(salida.path)
    assert inf.errors == [], inf.errors
    referencias = [a for a in inf.warnings if "REFERENCIA" in a]
    assert len(referencias) == 1, inf.warnings
    assert "asset.ios" in referencias[0], referencias
    assert inf.external_count >= 1


def test_sin_assets_externos_NO_se_dice_nada(tmp_path, malla):
    """El contrario, para que el test de arriba pueda fallar.

    Sin esto, un validador que avisara SIEMPRE pasaría los dos y no probaría nada.

    ⚠️ El contenedor lo escribe el test, no el exportador. El exportador ya no sabe emitir
    un original dentro —ninguno viaja— y devolverle esa capacidad sólo para tener un
    contrario sería mantener en producción un camino que la especificación prohíbe. Lo que
    se prueba aquí es el VALIDADOR, y un validador corre sobre lo que escribió otro.
    """
    from uos.validador import validate

    salida = write_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert inf.errors == [], inf.errors
    assert not any("REFERENCIA" in a for a in inf.warnings), inf.warnings
    assert inf.external_count == 0


def test_los_originales_referenciados_producen_UN_solo_aviso(tmp_path, malla):
    """Un contenedor con varios originales referenciados avisa UNA vez, no una por asset.

    ⚠️ **Es la regresión que hace falta cazar, y antes no se cazaba.** El validador emitía
    un aviso por cada asset externo. Mientras el caso de prueba tenía un solo original eso
    parecía razonable; con un caso real —el escáner, la serie de CBCT, nueve fotos y tres
    informes— son trece líneas diciendo lo mismo, y ninguna dice nada: que los originales
    adquiridos no viajen es la DEFINICIÓN del formato, no una excepción de este contenedor.
    Un aviso que sale siempre deja de leerse y entierra a los que sí distinguen algo —el
    registro sin verificar, la extensión obligatoria—, que son justo los que hay que ver.

    Lo que sí merece decirse una vez, y por eso el aviso no desaparece del todo, es que de
    esos assets el validador no puede comprobar nada.
    """
    from uos.validador import validate

    externos = [
        _asset(malla, id=f"asset.ext_{i}", external=True,
               uri=f"sha256:{hashlib.sha256(malla.read_bytes()).hexdigest()}")
        for i in range(4)
    ]
    salida = write_uos(
        tmp_path / "caso.uos",
        _manifiesto([_asset(malla), *externos]), [("scene/scan.stl", malla)],
    )

    inf = validate(salida)
    assert inf.errors == [], inf.errors
    referencias = [a for a in inf.warnings if "REFERENCIA" in a]
    assert len(referencias) == 1, f"un aviso por asset otra vez: {referencias}"
    assert inf.external_count == 4
    # Y los nombra, porque «hay cuatro» sin decir cuáles no se puede accionar.
    assert all(f"asset.ext_{i}" in referencias[0] for i in range(4)), referencias[0]


def test_un_asset_externo_se_nombra_por_su_CONTENIDO(tmp_path, malla):
    """La `uri` de un asset que no viaja es `sha256:<hex>`, no una ruta.

    ⚠️ Dos razones, y la segunda es de dato. Un fichero que no está dentro **no tiene sitio
    dentro**, así que una ruta sería una promesa sobre un ZIP en el que no está; lo único
    que sigue siendo cierto de él es qué fichero es. Y una dirección de contenido **no puede
    llevar dato de paciente**: la ruta local de un caso clínico lleva el directorio del
    paciente, un hash no lleva nada. Referenciar saca ficheros del contenedor, no
    identidades.
    """
    import hashlib

    from uos import UOSExportAgent, read_manifest

    salida = UOSExportAgent(None).export(
        _snap_min(), tmp_path / "caso", pseudonimo="P-1", malla=malla,
    )
    m = read_manifest(salida.path)
    ios = next(a for a in m.assets if a.id == "asset.ios")

    esperado = hashlib.sha256(malla.read_bytes()).hexdigest()
    assert ios.uri == f"sha256:{esperado}"
    assert ios.sha256 == esperado, "la dirección y el campo del contrato tienen que cuadrar"
    # Y nada del disco ni del proveedor viaja en ella.
    assert str(tmp_path) not in ios.uri and malla.stem not in ios.uri


def test_una_direccion_de_contenido_en_un_asset_que_SI_viaja_se_rechaza():
    """El sentido contrario, que es el que hace que la regla signifique algo.

    Un asset con dirección de contenido que además viaja dentro sería imposible de
    localizar en el ZIP: el lector buscaría una entrada llamada `sha256:…`.
    """
    from uos import Asset
    from uos.manifiesto import AssetKind

    with pytest.raises(ValueError, match="viaja dentro"):
        Asset(id="a", kind=AssetKind.DOCUMENT, visit="v1", uri="sha256:" + "a" * 64,
              media_type="model/stl", sha256="a" * 64, bytes=1, frame="frame.ios_master")


def test_un_asset_externo_con_RUTA_se_rechaza():
    """Y el otro sentido: externo obliga a dirección de contenido."""
    from uos import Asset
    from uos.manifiesto import AssetKind

    with pytest.raises(ValueError, match="es una ruta"):
        Asset(id="a", kind=AssetKind.DOCUMENT, visit="v1", uri="scene/scan.stl",
              media_type="model/stl", sha256="a" * 64, bytes=1,
              frame="frame.ios_master", external=True)


def test_la_direccion_tiene_que_ser_el_MISMO_hash_que_el_campo_del_contrato():
    """Dos sitios con el mismo dato se separan. Que no puedan es el punto."""
    from uos import Asset
    from uos.manifiesto import AssetKind

    with pytest.raises(ValueError, match="no.*son el mismo hash"):
        Asset(id="a", kind=AssetKind.DOCUMENT, visit="v1", uri="sha256:" + "a" * 64,
              media_type="model/stl", sha256="b" * 64, bytes=1,
              frame="frame.ios_master", external=True)


# --- el descriptor describe el FICHERO, no lo que alguien cree ---------------- #
def _ply_apariencia(ruta, *, unidades: str | None, n: int = 7) -> None:
    """Un PLY de apariencia mínimo, con o sin la línea de unidades."""
    import struct

    cab = ["ply", "format binary_little_endian 1.0",
           "comment perfil INRIA 3DGS grado 0 - APARIENCIA, no medida"]
    if unidades is not None:
        cab.append(f"comment unidades {unidades}")
    cab += [f"element vertex {n}", "property float x", "end_header"]
    ruta.write_bytes(("\n".join(cab) + "\n").encode("ascii") + struct.pack(f"<{n}f", *range(n)))


def test_el_descriptor_lee_las_unidades_DEL_FICHERO(tmp_path):
    """⚠️ El fallo que esto guarda: el sidecar afirmaba `units: "mm"` con un literal y el
    PLY de apariencia estaba en el espacio normalizado de Blender —32 veces más pequeño—.
    Nada contrastaba una cosa con la otra, así que la contradicción no podía fallar: solo
    verse, y solo comparando la nube con la malla.
    """
    from uos import UOSExportAgent

    p = tmp_path / "a.ply"
    _ply_apariencia(p, unidades="normalizado", n=11)
    assert UOSExportAgent(None)._cabecera_ply(p) == ("normalizado", 11, ["x"])

    _ply_apariencia(p, unidades="mm", n=11)
    assert UOSExportAgent(None)._cabecera_ply(p) == ("mm", 11, ["x"])


def test_un_ply_que_no_declara_unidades_devuelve_None_y_no_un_defecto(tmp_path):
    """Suponer es lo que falló. `None` deja que se vea; `"mm"` lo taparía otra vez."""
    from uos import UOSExportAgent

    p = tmp_path / "b.ply"
    _ply_apariencia(p, unidades=None, n=5)
    assert UOSExportAgent(None)._cabecera_ply(p) == (None, 5, ["x"])


def test_la_cabecera_se_lee_sin_cargar_el_fichero_entero(tmp_path):
    """Un PLY de apariencia pesa decenas de megas; la cabecera son cientos de bytes."""
    from uos import UOSExportAgent

    p = tmp_path / "c.ply"
    _ply_apariencia(p, unidades="mm", n=3)
    p.write_bytes(p.read_bytes() + b"\x00" * (5 * 1024 * 1024))
    assert UOSExportAgent(None)._cabecera_ply(p) == ("mm", 3, ["x"])


def test_la_cabecera_devuelve_las_propiedades_en_el_orden_del_fichero(tmp_path):
    """⚠️ Lo que faltaba: `esquema_apariencia()` enumeraba catorce columnas mientras el
    escritor emitía dieciocho propiedades, y `region_id` —el código FDI por gaussiana—
    viajaba en los bytes sin aparecer en el sidecar. El orden importa: `columns` es la
    receta para montar el registro binario."""
    import struct

    from uos import UOSExportAgent

    p = tmp_path / "e.ply"
    cab = ["ply", "format binary_little_endian 1.0", "comment unidades mm",
           "element vertex 2", "property float x", "property float y",
           "property short region_id", "end_header"]
    p.write_bytes(("\n".join(cab) + "\n").encode("ascii")
                  + struct.pack("<ffhffh", 0, 0, 11, 1, 1, 21))
    assert UOSExportAgent(None)._cabecera_ply(p) == ("mm", 2, ["x", "y", "region_id"])


def test_un_fichero_que_no_es_un_ply_no_revienta(tmp_path):
    """El descriptor no puede tumbar una exportación por no entender un asset."""
    from uos import UOSExportAgent

    p = tmp_path / "d.bin"
    p.write_bytes(b"\x00\x01\x02")
    assert UOSExportAgent(None)._cabecera_ply(p) == (None, None, [])
    assert UOSExportAgent(None)._cabecera_ply(tmp_path / "no-existe.ply") == (None, None, [])


def test_el_sidecar_de_segmentacion_dice_de_que_pieza_fiarse() -> None:
    """⚠️ **Sin esto, seleccionar una pieza en el visor puede mentir en silencio.**

    El contenedor dice la verdad sobre la pieza —su color, su pH, sus hallazgos— y enciende
    una superficie que arrastra medio diente vecino. Lo que se lee al lado es correcto y lo
    que se ve, no, que es la peor combinación posible. El veredicto por pieza convierte eso
    en un dato consultable.

    El criterio no es una opinión: sale del `p95` de `|ancho medido - tabla|` sobre 188
    coronas etiquetadas por experto. Contar coronas «demasiado anchas» no vale — las de
    experto lo fallan en el 77 %.
    """
    import numpy as np
    from uos.derivados import meta_segmentacion

    etq = np.array([11, 11, 12, 0], np.int16)
    calidad = {11: {"mesiodistal_mm": 8.9, "table_mm": 8.5, "excess_mm": 0.4,
                    "within_expert_range": True},
               12: {"mesiodistal_mm": 13.1, "table_mm": 6.5, "excess_mm": 6.6,
                    "within_expert_range": False}}
    meta = meta_segmentacion(etq, asset_origen="asset.scene", modelo="m",
                             version=None, calidad=calidad)
    bloque = meta["per_tooth_boundary"]
    assert bloque["teeth"]["11"]["within_expert_range"] is True
    assert bloque["teeth"]["12"]["within_expert_range"] is False
    assert "77 %" in bloque["note"]

    # Y sin el dato el bloque NO aparece: ausente no es «todas mal».
    assert "per_tooth_boundary" not in meta_segmentacion(
        etq, asset_origen="asset.scene", modelo="m", version=None)


# --- KHR_gaussian_splatting: la apariencia dentro del glTF -------------------- #
def _campo_apariencia(n=64, con_sh1=True, con_region=True):
    """Un PLY INRIA de juguete, con las convenciones REALES: logit, log y (w,x,y,z)."""
    import numpy as np

    rng = np.random.default_rng(7)
    props = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    if con_sh1:
        props += [(f"f_rest_{k}", "f4") for k in range(9)]
    props += [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
              ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    if con_region:
        props += [("region_id", "i2")]
    fila = np.zeros(n, dtype=np.dtype([(k, v) for k, v in props]))
    for k, v in props:
        fila[k] = (rng.integers(11, 28, n) if k == "region_id"
                   else rng.normal(0, 1, n).astype(v))
    cab = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    tipo = {"f4": "float", "i2": "short"}
    cab += [f"property {tipo[v]} {k}" for k, v in props]
    cab += ["end_header", ""]
    # ⚠️ **El esquema es el de VERDAD, no uno escrito aquí.** Dos exportaciones del caso
    # real murieron porque estos tests inventaban la forma del descriptor: primero un
    # `{"columns": [...]}` donde el emisor pasa la lista pelada, y después dicts con clave
    # `name` donde `esquema_apariencia` devuelve objetos `ColumnaCampo` con `.nombre`. Los
    # tres tests en verde las dos veces. Un esquema de juguete prueba el juguete.
    from gaussian_engine.agente_apariencia import esquema_apariencia

    return ("\n".join(cab).encode("ascii") + fila.tobytes(),
            esquema_apariencia([k for k, _ in props]))


def test_la_apariencia_viaja_DENTRO_del_gltf_con_la_extension_de_Khronos(tmp_path):
    """La capa 3DGS como primitiva `KHR_gaussian_splatting`, no como puntero a un `.ply`.

    ⚠️ **Es la diferencia entre un formato abierto y uno nuestro.** Con el fallback
    `extras.uos_gs_uri` que el borrador admite, un visor glTF cualquiera abre la escena,
    encuentra un puntero a un fichero opaco y **no dibuja la apariencia**: sólo la lee quien
    conozca nuestra convención. Con la extensión la dibuja cualquier implementación
    conforme sin saber nada de UOS.

    Lo que se comprueba aquí es el CONTRATO de la extensión, atributo a atributo, porque
    inventárselo sería exactamente lo contrario de interoperar.
    """
    import json as _json

    import numpy as np
    from uos.agente import _splats_khr
    from uos.escena import build_glb

    crudo, columnas = _campo_apariencia()
    ply = tmp_path / "appearance.ply"
    ply.write_bytes(crudo)
    gs = _splats_khr(ply, columnas)

    pos, etq = _arcada_de_juguete()
    glb = build_glb(pos, np.array([[0, 1, 2]]), splats=gs)
    largo = int.from_bytes(glb[12:16], "little")
    doc = _json.loads(glb[20:20 + largo])

    assert doc["extensionsUsed"] == ["KHR_gaussian_splatting"]
    # Nunca en `required`: un lector que no la entienda tiene que poder ver la malla.
    assert "extensionsRequired" not in doc

    prim = doc["meshes"][1]["primitives"][0]
    assert prim["mode"] == 0, "la extensión exige modo POINTS"
    ext = prim["extensions"]["KHR_gaussian_splatting"]
    assert ext["kernel"] == "ellipse"
    assert ext["colorSpace"] == "srgb_rec709_display"

    a = prim["attributes"]
    for nombre, tipo in [
        ("POSITION", "VEC3"),
        ("KHR_gaussian_splatting:ROTATION", "VEC4"),
        ("KHR_gaussian_splatting:SCALE", "VEC3"),
        ("KHR_gaussian_splatting:OPACITY", "SCALAR"),
        ("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", "VEC3"),
    ]:
        assert nombre in a, f"falta el atributo obligatorio {nombre}"
        assert doc["accessors"][a[nombre]]["type"] == tipo

    # Grado 1 entero o nada: la extensión exige que si va un grado superior estén todos.
    assert all(f"KHR_gaussian_splatting:SH_DEGREE_1_COEF_{k}" in a for k in range(3))
    # El FDI por gaussiana NO viaja: es Layer 3 y esta escena es Layer 1 (B-1).
    assert "_REGION_ID" not in a

    # Y la apariencia cuelga del nodo de la malla, que ES el marco canónico (§5.1).
    assert doc["nodes"][0]["children"] == [1]
    assert doc["nodes"][1]["mesh"] == 1


def test_las_unidades_de_la_extension_NO_son_las_del_PLY(tmp_path):
    """Opacidad lineal en [0,1], escala lineal no negativa y cuaternión en orden glTF.

    ⚠️ **Y se le pasa la LISTA de columnas, que es lo que le pasa el emisor.** Estos tests
    pasaban un `{"columns": [...]}` y el conversor lo pedía así, mientras
    `esquema_apariencia` devuelve la lista pelada: los tres tests en verde y la exportación
    del caso real muerta con un `AttributeError` que el gate resumía como «uos-export-agent
    falló». Un test que llama distinto que el código de verdad no prueba el código de verdad.

    ⚠️ **Son tres conversiones reales, no tres renombrados.** El PLY INRIA guarda la
    opacidad en logit, las escalas en logaritmo y el cuaternión como `(w,x,y,z)`; la
    extensión pide lineal, lineal y `(x,y,z,w)`. Copiar los arrays tal cual produce un
    fichero que validate y se dibuja mal: opacidades fuera de rango, elipses del tamaño
    equivocado y cada una girada. Es el peor tipo de fallo — no revienta.
    """
    import numpy as np
    from uos.agente import _splats_khr

    crudo, columnas = _campo_apariencia(n=32)
    ply = tmp_path / "appearance.ply"
    ply.write_bytes(crudo)
    gs = _splats_khr(ply, columnas)

    assert gs.opacidad.min() >= 0.0 and gs.opacidad.max() <= 1.0
    assert gs.escala.min() > 0.0, "la escala llega en logaritmo: hay que exponenciarla"
    # Cuaterniones unitarios, como exige la extensión.
    assert np.allclose(np.linalg.norm(gs.rotacion, axis=1), 1.0, atol=1e-5)

    # El orden: el último componente de la extensión es el `w`, que en el PLY es `rot_0`.
    import numpy as _np
    fila = _np.frombuffer(
        crudo, _np.dtype([(c.nombre, "<i2" if c.nombre == "region_id" else "<f4")
                          for c in columnas]),
        count=32, offset=crudo.index(b"end_header\n") + len(b"end_header\n"),
    )
    esperado = _np.stack([fila["rot_1"], fila["rot_2"], fila["rot_3"], fila["rot_0"]], 1)
    esperado /= _np.linalg.norm(esperado, axis=1, keepdims=True)
    assert _np.allclose(gs.rotacion, esperado, atol=1e-6)


def test_sin_grado_1_la_apariencia_sigue_siendo_valida(tmp_path):
    """El grado 1 es opcional. Sin él no se emite ningún `SH_DEGREE_1_*`, y eso es correcto.

    ⚠️ Emitir un grado 1 a cero «por si acaso» no sería neutro: la extensión obliga a que
    si va un grado superior estén todos los inferiores, y un lector que los encuentre
    asumirá que hay dependencia de la vista donde no la hay.
    """
    import json as _json

    import numpy as np
    from uos.agente import _splats_khr
    from uos.escena import build_glb

    crudo, columnas = _campo_apariencia(n=16, con_sh1=False, con_region=False)
    ply = tmp_path / "appearance.ply"
    ply.write_bytes(crudo)
    gs = _splats_khr(ply, columnas)
    assert gs.sh1 is None

    pos, etq = _arcada_de_juguete()
    glb = build_glb(pos, np.array([[0, 1, 2]]), splats=gs)
    largo = int.from_bytes(glb[12:16], "little")
    a = _json.loads(glb[20:20 + largo])["meshes"][1]["primitives"][0]["attributes"]
    assert not any(k.startswith("KHR_gaussian_splatting:SH_DEGREE_1") for k in a)
    assert "_REGION_ID" not in a


# --- B-3 y B-4: PHI y proposito de uso ---------------------------------------- #
def test_el_proposito_TIENE_que_caber_en_lo_que_se_consintio(tmp_path, malla) -> None:
    """B-4: salir hacia un laboratorio, una segunda opinion o un entrenamiento son tres
    actos juridicos distintos, y el contenedor no distinguia ninguno.

    Emitir para `model_training` un caso cuyo consentimiento solo cubre `treatment` no es
    un matiz administrativo: es el uso para el que el paciente NO dio permiso, y es la
    primera pregunta de cualquier revision de proteccion de datos.
    """
    from uos.manifiesto import Consent, PurposeOfUse

    a = _asset(malla)
    m = _manifiesto(
        [a], purpose_of_use=PurposeOfUse.ENTRENAMIENTO,
        subject=Subject(pseudonym="P-1",
                       consent=Consent(scope=[PurposeOfUse.TRATAMIENTO])),
    )
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("model_training" in e for e in inf.errors), inf.errors


def test_el_desplazamiento_de_fechas_NO_viaja_si_no_esta_identificado(tmp_path, malla) -> None:
    """B-3: `date_shift_days` es la CLAVE de re-identificacion.

    Desplazar todas las fechas del caso por igual conserva la longitudinalidad y es la
    opcion recomendada de PS3.15. Publicar cuantos dias se desplazaron deshace exactamente
    la medida que se dice haber aplicado: con el numero, cualquiera vuelve a las fechas
    reales. Solo puede viajar en un contenedor que ya se declara `identified`, donde no
    protege nada porque no hay nada que proteger.
    """
    from uos.manifiesto import Deidentification as D

    a = _asset(malla)
    m = _manifiesto([a], deidentification=D(
        profile="DICOM PS3.15 E.1 Basic Application Level Confidentiality Profile",
        options=["RetainLongitudinalTemporalInformationModifiedDates"],
        date_shift_days=137,
    ))
    salida = write_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("date_shift_days" in e for e in inf.errors), inf.errors


def test_el_esquema_publicado_esta_ENTERO_en_ingles() -> None:
    """G-1: el esquema es el unico artefacto que un ajeno usa sin leernos el codigo.

    ⚠️ **Salia bilingue sin que nadie lo decidiera.** Los nombres de campo y los valores de
    enumeracion son ingleses porque son el formato de cable; `title` y `description` los
    generaba pydantic del nombre de la clase y del docstring, que estan en castellano
    porque el contrato lo leemos nosotros. El resultado era un esquema cuya mitad legible
    no la puede leer su destinatario — y este fichero existe justo para que alguien
    compruebe su lector contra algo que no sea su propia salida.

    Se comprueba sobre el fichero PUBLICADO y no sobre el generado: lo que viaja es aquel.
    """
    import json as _json
    import re

    from uos.esquema import RUTA

    publicado = Path(__file__).resolve().parents[3] / RUTA
    crudo = publicado.read_text(encoding="utf-8")
    castellano = re.compile(
        r"[áéíóúñ¿¡]|\b(que|para|los|las|con|una|del|por|como|cada|sobre|este|segun)\b"
    )
    culpables = [
        f"{ruta}: {texto}"
        for ruta, texto in _recorre_textos(_json.loads(crudo))
        if castellano.search(texto)
    ]
    assert culpables == [], "el esquema publicado lleva castellano:\n" + "\n".join(culpables)


def _recorre_textos(nodo, ruta=""):
    """Los `title` y `description` del esquema, con su ruta, para poder senalar cual."""
    if isinstance(nodo, dict):
        for k, v in nodo.items():
            if k in ("title", "description") and isinstance(v, str):
                yield f"{ruta}.{k}", v
            else:
                yield from _recorre_textos(v, f"{ruta}.{k}")
    elif isinstance(nodo, list):
        for i, v in enumerate(nodo):
            yield from _recorre_textos(v, f"{ruta}[{i}]")


def test_UOS_Distributable_separa_abrible_de_enviable(tmp_path, malla) -> None:
    """B-6: los niveles dicen si un lector puede ABRIR el contenedor, no si puede SALIR.

    Core/Vol/Sig/Full describen que tipos de asset hay dentro. Ninguno responde a la
    pregunta que se hace justo antes de adjuntar un caso a un correo. Son las condiciones
    de B-1, B-3 y B-4 a la vez, y a la vez porque de una en una no deciden: un contenedor
    con el proposito declarado y la cara dentro no se puede mandar igual.

    Que NO sea distribuible no es un error — es lo normal mientras el caso vive dentro de
    la clinica. Lo que no puede pasar es que nadie lo sepa hasta despues.
    """
    from uos.manifiesto import Consent, PurposeOfUse
    from uos.manifiesto import Deidentification as D

    a = _asset(malla)
    # Le falta el proposito: valido, abrible, y no enviable.
    m = _manifiesto([a])
    salida = write_uos(tmp_path / "sin-proposito.uos", m, [("scene/scan.stl", malla)])
    inf = validate(salida)
    assert inf.valid, inf.errors
    assert not inf.distributable
    assert any("purpose_of_use" in r for r in inf.not_distributable_because)

    # Con todo declarado, si.
    completo = _manifiesto(
        [a], purpose_of_use=PurposeOfUse.FABRICACION,
        subject=Subject(pseudonym="P-1",
                       consent=Consent(scope=[PurposeOfUse.FABRICACION])),
        deidentification=D(
            profile="DICOM PS3.15 E.1 Basic Application Level Confidentiality Profile",
            options=["CleanDescriptors", "CleanRecognizableVisualFeatures"],
        ),
    )
    salida2 = write_uos(tmp_path / "listo.uos", completo, [("scene/scan.stl", malla)])
    inf2 = validate(salida2)
    assert inf2.distributable, inf2.not_distributable_because


def test_un_contenedor_identificado_NUNCA_es_distribuible(tmp_path, malla) -> None:
    """El caso que importa: todo lo demas declarado y dato identificable dentro.

    Es exactamente el estado en el que sale hoy nuestro pipeline —lleva densidad medida
    del CBCT sin limpiar rasgos reconocibles (B-3)— y el perfil tiene que decirlo aunque
    el consentimiento, el proposito y la de-identificacion esten todos rellenos.
    """
    from uos.manifiesto import Consent, PurposeOfUse
    from uos.manifiesto import Deidentification as D

    a = _asset(malla)
    m = _manifiesto(
        [a], phi_state=PHIState.IDENTIFIED, purpose_of_use=PurposeOfUse.TRATAMIENTO,
        subject=Subject(pseudonym="P-1",
                       consent=Consent(scope=[PurposeOfUse.TRATAMIENTO])),
        deidentification=D(profile="DICOM PS3.15 E.1"),
    )
    salida = write_uos(tmp_path / "identificado.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert inf.valid, inf.errors
    assert not inf.distributable
    assert any("identified" in r for r in inf.not_distributable_because)


def test_el_frame_de_un_volumen_se_ancla_al_UID_de_DICOM(tmp_path, malla) -> None:
    """D-1 y D-2: `frame.ct_001` es una cadena que se invento el escritor.

    DICOM ya identifica un sistema de coordenadas de forma global y unica —el Frame of
    Reference UID, `(0020,0052)`, que toda serie CBCT lleva—. Un lector que reciba la serie
    por otro canal no tiene forma de saber que `frame.ct_001` es ESA serie salvo por
    confianza. Y «diestro» fija la quiralidad, no la orientacion: sin declarar LPS, nadie
    sabe cual de las direcciones es anterior o superior del paciente, que es lo que hace
    falta para medir un angulo o una distancia a una estructura.
    """
    from uos.manifiesto import AnatomicalConvention

    a = _asset(malla, frame="frame.ct_001")
    a = a.model_copy(update={"kind": AssetKind.VOLUME})
    m = _manifiesto([a], frames=[Frame(id="frame.ct_001")], registrations=[Registration(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="manual", operator="user:pedro",
    )])
    salida = write_uos(tmp_path / "sin-uid.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("dicom_frame_of_reference_uid" in e for e in inf.errors), inf.errors
    assert any("LPS" in e for e in inf.errors), inf.errors

    # Con las dos cosas declaradas, el mismo caso vale.
    bien = _manifiesto([a], registrations=m.registrations, frames=[Frame(
        id="frame.ct_001", anatomical=AnatomicalConvention.LPS,
        dicom_frame_of_reference_uid="1.2.826.0.1.3680043.8.498.1",
    )])
    salida2 = write_uos(tmp_path / "con-uid.uos", bien, [("scene/scan.stl", malla)])
    assert validate(salida2).valid, validate(salida2).errors


def test_una_registracion_no_se_declara_apta_para_lo_que_no_se_ha_medido(tmp_path, malla):
    """D-9: `rms_error_mm` es un PROMEDIO y no decide un uso clinico.

    Para cirugia guiada de implantes el error que importa es el maximo local en la zona de
    interes: 0,666 mm de RMS es aceptable para visualizar y no para planificar. Un lector
    que solo vea el promedio no puede distinguirlo, asi que supondra — y suponer aptitud
    es exactamente lo que `fit_for` existe para impedir.
    """
    from uos.manifiesto import RegistrationFitness

    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], occlusion="single_arch", registrations=[Registration(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="icp_surface", rms_error_mm=0.666, operator="user:pedro",
        fit_for=[RegistrationFitness.CIRUGIA_GUIADA],
    )])
    salida = write_uos(tmp_path / "apto.uos", m, [("scene/scan.stl", malla)])

    inf = validate(salida)
    assert not inf.valid
    assert any("cirugia guiada" in e for e in inf.errors), inf.errors


def test_el_contenedor_dice_si_hubo_registro_de_MORDIDA(tmp_path, malla) -> None:
    """D-9: mandibula<->maxila es la registracion clinicamente mas importante.

    El caso de referencia es solo maxilar y por eso no aparecia — que es razon para
    reservarla, no para omitirla. Y el silencio no es «no hay»: un caso de una arcada
    responde `single_arch`, pero responde.
    """
    a = _asset(malla)
    inf = validate(write_uos(tmp_path / "muda.uos", _manifiesto([a]),
                             [("scene/scan.stl", malla)]))
    assert any("occlusion" in av for av in inf.warnings), inf.warnings

    dicha = _manifiesto([a], occlusion="single_arch")
    inf2 = validate(write_uos(tmp_path / "dicha.uos", dicha,
                              [("scene/scan.stl", malla)]))
    assert not any("occlusion" in av for av in inf2.warnings)


# --- T-3: los checks que el texto declaraba y el algoritmo no hacia ----------- #
def test_un_fichero_que_el_manifiesto_no_declara_INVALIDA(tmp_path, malla) -> None:
    """T-3: se comprobaba que todo lo declarado estuviera, no que todo lo que esta lo este.

    Un fichero de mas en el ZIP viaja sin hash que lo acredite, sin capa regulatoria y sin
    que nadie lo nombre. Es exactamente la forma que tendria una fuga, y el §14.6 lo
    prohibe desde el principio — solo que nadie lo comprobaba.
    """
    import zipfile

    a = _asset(malla)
    salida = write_uos(tmp_path / "c.uos", _manifiesto([a], occlusion="single_arch"),
                         [("scene/scan.stl", malla)])
    assert validate(salida).valid

    with zipfile.ZipFile(salida, "a") as z:
        z.writestr("colado.txt", "esto no lo declara nadie")

    inf = validate(salida)
    assert not inf.valid
    assert any("colado.txt" in e for e in inf.errors), inf.errors


def test_un_asset_externo_cuyo_uri_no_es_su_hash_NI_SE_PARSEA() -> None:
    """T-3 pedia este check en el algoritmo, y ya existe una capa antes.

    El §3.4.3 dice que un externo se nombra por su direccion de contenido y que la relacion
    se verifica «en las dos direcciones». La revision no lo vio porque miro el algoritmo del
    validador, y la regla vive en el CONTRATO: `Asset._direccion_y_custodia` es un validador
    de modelo, asi que un manifiesto incoherente ni llega a parsearse.

    Este test fija donde vive la regla. Anadirla tambien al algoritmo habria sido codigo
    inalcanzable, que aparenta una cobertura que en realidad viene de otro sitio.
    """
    import pytest
    from pydantic import ValidationError
    from uos.manifiesto import Asset, AssetKind

    with pytest.raises(ValidationError, match="no son el mismo hash"):
        Asset(
            id="asset.ios", kind=AssetKind.DOCUMENT, visit="v1", frame="frame.ios_master",
            external=True,
            uri="sha256:" + "a" * 64, media_type="model/stl", sha256="b" * 64, bytes=1,
        )
