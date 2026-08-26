"""El contenedor y el manifiesto tienen que cumplir el spec, no parecerse a el."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from uos import Asset, Frame, Manifiesto, Registro, Sujeto, Visita, escribe_uos, valida
from uos.contenedor import MANIFIESTO, lee_manifiesto
from uos.manifiesto import Clase, EstadoPHI, Regulatorio
from uos.validador import Conformidad


def _asset(ruta: Path, **kw) -> Asset:
    crudo = ruta.read_bytes()
    base = dict(
        id="asset.ios", kind=Clase.MESH_GS_SCENE, visit="v1", uri="scene/scan.stl",
        media_type="model/stl", sha256=hashlib.sha256(crudo).hexdigest(),
        bytes=len(crudo), frame="frame.ios_master",
    )
    return Asset(**{**base, **kw})


def _manifiesto(assets: list[Asset], **kw) -> Manifiesto:
    base = dict(
        case_id="urn:uuid:0",
        generator={"name": "test", "version": "0"},
        phi_state=EstadoPHI.PSEUDONYMIZED,
        subject=Sujeto(pseudonym="P-1"),
        canonical_frame=Frame(id="frame.ios_master"),
        visits=[Visita(id="v1", date="2026-08-23")],
        assets=assets,
    )
    return Manifiesto(**{**base, **kw})


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
    salida = escribe_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    with zipfile.ZipFile(salida) as z:
        assert z.namelist()[0] == MANIFIESTO


def test_nada_va_comprimido(tmp_path, malla):
    """STORE porque los payloads ya vienen comprimidos y comprimir el ZIP solo rompe el
    acceso aleatorio por rangos, que es lo que permite bajar un asset suelto."""
    salida = escribe_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    with zipfile.ZipFile(salida) as z:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


def test_lo_que_sale_es_BYTE_IDENTICO_a_lo_que_entro(tmp_path, malla):
    """Referencia, no transcodificacion (§2.1). Es lo que permite afirmar que el
    contenedor no degrada nada, y lo que hace la trazabilidad forense posible."""
    original = malla.read_bytes()
    salida = escribe_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    with zipfile.ZipFile(salida) as z:
        assert z.read("scene/scan.stl") == original


def test_un_asset_declarado_y_no_aportado_se_declara(tmp_path, malla):
    """Una referencia colgante es un error, no un hueco — igual que en el resto del
    sistema. Un manifiesto que promete algo que no esta es peor que uno que no lo promete."""
    with pytest.raises(ValueError, match="no se aportaron"):
        escribe_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]), [])


def test_un_fichero_aportado_y_no_declarado_se_declara(tmp_path, malla):
    """Al reves tambien: contenido que el manifiesto no menciona es contenido que nadie
    puede verificar ni saber que hace ahi."""
    otro = tmp_path / "suelto.bin"
    otro.write_bytes(b"x")
    with pytest.raises(ValueError, match="no declara"):
        escribe_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                    [("scene/scan.stl", malla), ("suelto.bin", otro)])


def test_las_rutas_internas_no_pueden_escapar():
    """Un `..` en una ruta de ZIP es la travesia de directorios clasica: un lector que la
    resuelva ingenuamente escribe FUERA del destino."""
    for mala in ("../fuera.stl", "/absoluta.stl", "scene/../../x.stl"):
        with pytest.raises(ValueError, match="relativas"):
            Asset(id="a", kind=Clase.MESH_GS_SCENE, visit="v1", uri=mala,
                  media_type="model/stl", sha256="0" * 64, bytes=1,
                  frame="frame.ios_master")


# --- el validador ------------------------------------------------------------ #
def test_un_hash_que_no_cuadra_invalida(tmp_path, malla):
    """Verificar es la politica en ingesta (§8): si el sha256 no cuadra, el asset no es el
    que el manifiesto dice, y eso invalida el caso entero."""
    a = _asset(malla, sha256="f" * 64)
    salida = escribe_uos(tmp_path / "caso.uos", _manifiesto([a]),
                         [("scene/scan.stl", malla)])

    inf = valida(salida)
    assert not inf.valido
    assert any("sha256" in e for e in inf.errores)


def test_un_frame_desconectado_del_canonico_invalida(tmp_path, malla):
    """§6: el grafo DEBE ser conexo hacia el canonico. Si no, un asset queda sin forma de
    alinearse y el visor lo colocaria en el sitio equivocado sin poder detectarlo."""
    a = _asset(malla, frame="frame.huerfano")
    salida = escribe_uos(tmp_path / "caso.uos", _manifiesto([a]),
                         [("scene/scan.stl", malla)])

    inf = valida(salida)
    assert not inf.valido
    assert any("no conecta con el canonico" in e for e in inf.errores)


def test_una_registracion_lo_conecta(tmp_path, malla):
    """Y con la registracion declarada, el mismo caso es valido."""
    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], registrations=[Registro(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="icp_surface", verified_by="user:pedro",
    )])
    salida = escribe_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)
    assert inf.valido, inf.errores


def test_un_registro_automatico_sin_verificar_es_un_AVISO_no_un_error(tmp_path, malla):
    """No invalida el fichero, pero el visor tiene que presentarlo como PROVISIONAL: un
    alineamiento que nadie ha mirado no es lo mismo que uno firmado."""
    a = _asset(malla, frame="frame.ct_001")
    m = _manifiesto([a], registrations=[Registro(
        id="reg.ct_to_ios", source_frame="frame.ct_001", target_frame="frame.ios_master",
        transform_4x4_row_major=[1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        method="auto_dl",
    )])
    salida = escribe_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)
    assert inf.valido
    assert any("PROVISIONAL" in a_ for a_ in inf.avisos)


def test_layer_3_tiene_que_vivir_en_derived(tmp_path, malla):
    """La regla que hace que `derived/` sea DESMONTABLE (§5.5): si un asset de inferencia
    vive fuera, borrar el directorio no lo quita y el caso deja de ser distribuible donde
    el modulo no esta habilitado."""
    a = _asset(malla, regulatory=Regulatorio(layer=3))
    salida = escribe_uos(tmp_path / "caso.uos", _manifiesto([a]),
                         [("scene/scan.stl", malla)])

    inf = valida(salida)
    assert not inf.valido
    assert any("desmontar" in e for e in inf.errores)


def test_el_nivel_de_conformidad_sale_de_lo_que_hay(tmp_path, malla):
    """UOS-Core es manifiesto + mesh_gs_scene + image2d (§12). Sin volumen no se puede
    declarar UOS-Vol, y decirlo mal haria que un implementador no pudiera fiarse."""
    salida = escribe_uos(tmp_path / "caso.uos", _manifiesto([_asset(malla)]),
                         [("scene/scan.stl", malla)])

    inf = valida(salida)
    assert Conformidad.CORE in inf.niveles
    assert Conformidad.VOL not in inf.niveles


def test_el_manifiesto_se_relee_igual(tmp_path, malla):
    """Ida y vuelta exacta: el manifiesto es el contrato, y un contrato que cambia al
    releerse no sirve para encadenar hashes entre versiones (§8)."""
    m = _manifiesto([_asset(malla)], created=datetime(2026, 8, 23, tzinfo=UTC))
    salida = escribe_uos(tmp_path / "caso.uos", m, [("scene/scan.stl", malla)])

    assert lee_manifiesto(salida).json_canonico() == m.json_canonico()


def test_un_zip_sin_manifiesto_primero_se_rechaza(tmp_path):
    """Sin identificacion positiva no hay formato: cualquier ZIP se haria pasar por .uos."""
    falso = tmp_path / "falso.uos"
    with zipfile.ZipFile(falso, "w") as z:
        z.writestr("otra_cosa.txt", "x")
        z.writestr(MANIFIESTO, json.dumps({"uos_version": "0.2"}))

    with pytest.raises(ValueError, match="primera entrada"):
        lee_manifiesto(falso)
    assert not valida(falso).valido


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
    from uos import UOSExportAgent, lee_manifiesto

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
    m = lee_manifiesto(salida.path)
    uris = " ".join(a.uri for a in m.assets)
    assert "PEREZ" not in uris and "0000144500014386" not in uris
    assert "images/img_000.jpg" in uris
    # Y la malla igual: su nombre tampoco entra.
    assert malla.stem not in uris
    assert "scene/scan" in uris



def test_un_informe_ilegible_viaja_dentro_del_contenedor(tmp_path, malla):
    """Un PDF sin capa de texto **no se pierde**: viaja byte-identico y sin su nombre.

    Antes solo viajaba lo que el `report-agent` conseguia transcribir, asi que un informe
    escaneado desaparecia entero — el gate decia «hay un PDF que nadie pudo leer» y el
    PDF no estaba en ninguna parte para que alguien lo leyera. En el caso real que lo
    destapo ese fichero era el **pasaporte de implantes**: tres implantes con su posicion
    FDI, su fecha, su marca y su lote, y ningun otro documento del caso los mencionaba.

    Con `sin_originales` obedece la misma regla que el STL y las fotos: se declara por su
    direccion de contenido y no viaja dentro. Lo que se arregla no es que viaje SIEMPRE, es
    que exista en el manifiesto — antes no habia ni asset.
    """
    import zipfile
    from datetime import datetime

    from core_schemas import Modality, Provenance, TwinSnapshot
    from uos import UOSExportAgent, lee_manifiesto

    doc = tmp_path / "APELLIDOS_NOMBRE_Informe.pdf"
    doc.write_bytes(b"%PDF-1.4 escaneado sin texto")
    snap = TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.MESH, agent="a@0"),
    )
    salida = UOSExportAgent(None).export(
        snap, tmp_path / "caso", pseudonimo="P-1", malla=malla, informes=[doc]
    )

    assert salida.ok, salida.detail
    m = lee_manifiesto(salida.path)
    docs = [a for a in m.assets if a.uri.startswith("clinical/documents/")]
    assert len(docs) == 1
    assert docs[0].media_type == "application/pdf"
    # Layer 1: es registro clinico, no salida de un modelo. Borrar `derived/` no se lo
    # lleva por delante.
    assert docs[0].regulatory.layer == 1
    # El apellido del paciente esta en el NOMBRE del fichero y no puede entrar.
    assert "APELLIDOS" not in " ".join(a.uri for a in m.assets)
    with zipfile.ZipFile(salida.path) as z:
        assert z.read(docs[0].uri) == doc.read_bytes()

    # Y en el perfil ligero se declara por su hash, como el STL y las fotos.
    ligero = UOSExportAgent(None).export(
        snap, tmp_path / "ligero", pseudonimo="P-1", malla=malla, informes=[doc],
        sin_originales=True,
    )
    assert ligero.ok, ligero.detail
    fuera = [a for a in lee_manifiesto(ligero.path).assets if a.id == "asset.doc_000"]
    assert len(fuera) == 1 and fuera[0].external
    assert fuera[0].uri == f"sha256:{docs[0].sha256}"
    with zipfile.ZipFile(ligero.path) as z:
        assert "clinical/documents/doc_000.pdf" not in z.namelist()


# --- el agente, vistas y cadena ---------------------------------------------- #
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
    from uos.vistas import VISTAS

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
        vistas = json.loads(z.read(VISTAS))["views"]
    ids = {v["id"] for v in vistas}
    assert "view.oclusal" in ids and "view.vestibular_derecha" in ids
    assert "view.pieza_16" in ids
    # El 17 esta etiquetado en la malla y NO lo anota nadie: no tiene vista propia.
    assert "view.pieza_17" not in ids
    assert all(v["visit"] == "v1" for v in vistas)


def test_sin_etiquetas_el_agente_no_inventa_vistas_y_lo_dice_en_los_motivos(tmp_path, malla):
    """El aviso llega al gate, no se queda en el fichero: es un hueco que alguien decide."""
    from uos import UOSExportAgent
    from uos.vistas import VISTAS

    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla
    )

    assert salida.ok, salida.detail
    with zipfile.ZipFile(salida.path) as z:
        assert json.loads(z.read(VISTAS))["views"] == []
    assert any("no lleva vistas" in m for m in salida.hitl_reasons)


def test_reexportar_encima_produce_la_version_2_y_no_un_borrado(tmp_path, malla):
    """Un `.uos` es append-only logico: modificar es encadenar, no sobrescribir."""
    from uos import UOSExportAgent
    from uos.procedencia import CADENA, Cadena

    destino = tmp_path / "caso"
    agente = UOSExportAgent(None)
    primera = agente.export(_snapshot(), destino, pseudonimo="P-1", malla=malla)
    segunda = agente.export(_snapshot(), destino, pseudonimo="P-1", malla=malla)

    assert primera.path == segunda.path
    with zipfile.ZipFile(segunda.path) as z:
        cadena = Cadena.model_validate_json(z.read(CADENA))
    assert [e.version for e in cadena.links] == [1, 2]
    assert cadena.links[1].prev_manifest_sha256 == cadena.links[0].manifest_sha256
    assert valida(segunda.path).version == 2


def test_una_vista_que_apunta_a_una_visita_inexistente_invalida(tmp_path, malla):
    """Un deep-link a una visita que el manifiesto no declara abre en ninguna parte."""
    from uos.vistas import VISTAS

    m = _manifiesto([_asset(malla)])
    salida = escribe_uos(
        tmp_path / "caso.uos", m, [("scene/scan.stl", malla)],
        extras={VISTAS: json.dumps({"views": [{
            "id": "view.x", "label": "X", "visit": "v9",
            "camera": {"position": [0, 0, 1], "target": [0, 0, 0], "up": [0, 1, 0]},
        }]})},
    )

    inf = valida(salida)

    assert not inf.valido
    assert any("que el manifiesto no declara" in e for e in inf.errores)


def test_dos_vistas_con_el_mismo_id_invalidan(tmp_path, malla):
    """El id es la ancla del deep-link: repetido, `#view=…` es ambiguo."""
    from uos.vistas import VISTAS

    vista = {
        "id": "view.x", "label": "X", "visit": "v1",
        "camera": {"position": [0, 0, 1], "target": [0, 0, 0], "up": [0, 1, 0]},
    }
    salida = escribe_uos(
        tmp_path / "caso.uos", _manifiesto([_asset(malla)]), [("scene/scan.stl", malla)],
        extras={VISTAS: json.dumps({"views": [vista, vista]})},
    )

    inf = valida(salida)

    assert not inf.valido
    assert any("repetido" in e for e in inf.errores)


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



def test_la_escena_lleva_el_FDI_por_sub_mesh(tmp_path, malla):
    """§5.1 pide `extras.uos_fdi` por sub-mesh, y no es decoracion: el picking semantico
    del §11.3 esta definido SOBRE ese campo.

    Sin el, un visor ajeno abre nuestro contenedor y no puede seleccionar un diente por
    mucho que las etiquetas viajen en `derived/seg_teeth` — eso lo lee el NUESTRO porque
    sabe que existe, no un lector cualquiera.
    """
    import numpy as np
    from uos.agente import UOSExportAgent

    pos, etq = _arcada_de_juguete()
    salida = UOSExportAgent(_Almacen(pos)).export(
        _snapshot(surface_ref="sha256:malla"), tmp_path / "c",
        pseudonimo="P-1", malla=malla, etiquetas_ios=etq,
    )
    assert salida.ok, salida.detail

    prims = _gltf_de(salida.path)["meshes"][0]["primitives"]
    con_fdi = {p["extras"]["uos_fdi"] for p in prims if "extras" in p}
    assert con_fdi == {str(int(f)) for f in np.unique(etq) if f > 0}
    # Y queda un primitive SIN codigo: la encia y las caras que cruzan de un diente a
    # otro. Ni se reparten ni se descartan — descartarlas dejaria agujeros en la malla.
    assert any("extras" not in p for p in prims)


def test_partir_por_FDI_no_toca_el_orden_de_los_vertices(tmp_path, malla):
    """La union entre `derived/seg_teeth` y la escena es POSICIONAL: el codigo `i` es del
    vertice `i`. Lo que se parte es el indice, nunca las posiciones, y todos los
    primitives comparten el mismo accesor de POSITION. Si eso cambia, la segmentacion se
    pinta sobre los dientes equivocados y nada protesta."""

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
    campo, en vez de descubrirlo cuando a alguien de fuera no le valida un contenedor."""
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

    from uos import UOSExportAgent, lee_manifiesto

    salida = UOSExportAgent(None).export(
        _snap_min(), tmp_path / "caso", pseudonimo="P-1", malla=malla,
        sin_originales=True,
    )
    assert salida.ok, salida.detail
    m = lee_manifiesto(salida.path)
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
    """
    from uos import UOSExportAgent
    from uos.validador import valida

    salida = UOSExportAgent(None).export(
        _snap_min(), tmp_path / "caso", pseudonimo="P-1", malla=malla,
        sin_originales=True,
    )
    inf = valida(salida.path)
    assert inf.errores == [], inf.errores
    assert any("EXTERNO" in a and "asset.ios" in a for a in inf.avisos), inf.avisos


def test_con_originales_no_hay_ningun_aviso_de_externo(tmp_path, malla):
    """El contrario, para que el test de arriba pueda fallar.

    Sin esto, un validador que avisara SIEMPRE pasaría los dos y no probaría nada.
    """
    from uos import UOSExportAgent
    from uos.validador import valida

    salida = UOSExportAgent(None).export(
        _snap_min(), tmp_path / "caso", pseudonimo="P-1", malla=malla,
    )
    inf = valida(salida.path)
    assert inf.errores == [], inf.errores
    assert not any("EXTERNO" in a for a in inf.avisos), inf.avisos


def test_un_asset_externo_se_nombra_por_su_CONTENIDO(tmp_path, malla):
    """La `uri` de un asset que no viaja es `sha256:<hex>`, no una ruta.

    ⚠️ Dos razones, y la segunda es de dato. Un fichero que no está dentro **no tiene sitio
    dentro**, así que una ruta sería una promesa sobre un ZIP en el que no está; lo único
    que sigue siendo cierto de él es qué fichero es. Y una dirección de contenido **no puede
    llevar dato de paciente**: la ruta local de un caso clínico lleva el directorio del
    paciente, un hash no lleva nada. El perfil ligero saca ficheros del contenedor, no
    identidades.
    """
    import hashlib

    from uos import UOSExportAgent, lee_manifiesto

    salida = UOSExportAgent(None).export(
        _snap_min(), tmp_path / "caso", pseudonimo="P-1", malla=malla,
        sin_originales=True,
    )
    m = lee_manifiesto(salida.path)
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
    from uos.manifiesto import Clase

    with pytest.raises(ValueError, match="viaja dentro"):
        Asset(id="a", kind=Clase.DOCUMENT, visit="v1", uri="sha256:" + "a" * 64,
              media_type="model/stl", sha256="a" * 64, bytes=1, frame="frame.ios_master")


def test_un_asset_externo_con_RUTA_se_rechaza():
    """Y el otro sentido: externo obliga a dirección de contenido."""
    from uos import Asset
    from uos.manifiesto import Clase

    with pytest.raises(ValueError, match="es una ruta"):
        Asset(id="a", kind=Clase.DOCUMENT, visit="v1", uri="scene/scan.stl",
              media_type="model/stl", sha256="a" * 64, bytes=1,
              frame="frame.ios_master", external=True)


def test_la_direccion_tiene_que_ser_el_MISMO_hash_que_el_campo_del_contrato():
    """Dos sitios con el mismo dato se separan. Que no puedan es el punto."""
    from uos import Asset
    from uos.manifiesto import Clase

    with pytest.raises(ValueError, match="no.*son el mismo hash"):
        Asset(id="a", kind=Clase.DOCUMENT, visit="v1", uri="sha256:" + "a" * 64,
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
