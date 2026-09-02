"""UOS-Vol: la serie DICOM viaja ENTERA y sin tocar, con su sidecar (§5.2).

⚠️ **El exportador ya NO puede emitir una serie dentro, y estos tests siguen aquí a
propósito.** Ningún original adquirido viaja en nuestro `.uos` —sólo su dirección de
contenido—, así que la bandera que lo permitía se ha eliminado. Pero el VALIDADOR sí tiene
que saber verificar una serie que viaje: un validador corre sobre ficheros que escribió
otro, y el §5.2 del borrador sigue siendo parte del formato aunque nosotros no lo emitamos.

Por eso los tests que necesitan la serie dentro **construyen el contenedor ellos mismos**
con `escribe_uos`, en vez de pedirle al exportador que produzca algo que el producto no
produce. Los demás usan el exportador tal cual: un volumen referenciado sigue siendo un
asset declarado, con su frame, su sidecar, su registro y su nivel de conformidad.

Lo que se prueba no es que quepa en el ZIP, sino que **siga siendo la misma serie**: cada
corte byte-idéntico y verificable por separado, y el manifiesto declarando exactamente los
que hay. Un contenedor que dice llevar un CBCT y lleva 396 de 397 cortes es peor que uno
que no lo lleva, porque el hueco no se ve.
"""

from __future__ import annotations

import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from core_schemas import Modality, Provenance, RigidTransform, TwinSnapshot
from ingestion_agents import synthetic
from uos import UOSExportAgent, lee_manifiesto, valida
from uos.contenedor import asset_de_directorio
from uos.manifiesto import Clase, Desidentificacion, digesto_de_partes
from uos.validador import Conformidad
from uos.volumen import SIDECAR, describe_serie


@pytest.fixture(scope="module")
def serie(tmp_path_factory) -> Path:
    """Una serie DICOM pequeña pero de verdad: cabeceras completas y píxeles.

    ⚠️ Se **anonimiza** después de generarla, porque `write_dicom_series` escribe
    `PatientName = "SINTETICO^CASO"` y UOS-Vol rechaza —correctamente— una serie con el
    nombre poblado. La fixture representa lo que el nivel admite: un export ya anonimizado,
    con el identificador opaco que sobrevive a la seudonimización. Que el control saltara
    sobre esta fixture al escribirla es la razón de que exista `TAGS_IDENTIFICABLES`.
    """
    import pydicom

    raiz = tmp_path_factory.mktemp("cbct")
    volumen, sp = synthetic.build_volume(synthetic.upper_arch_codes(), spacing=3.0)
    cruda = synthetic.write_dicom_series(raiz / "ct", volumen, sp, patient_id="SYNTH-1")
    for f in sorted(cruda.glob("*.dcm")):
        ds = pydicom.dcmread(str(f))
        ds.PatientName = "Anonymized"
        ds.save_as(f)
    return cruda


@pytest.fixture
def malla(tmp_path) -> Path:
    p = tmp_path / "scan.stl"
    p.write_bytes(_stl_binario())
    return p


def _snapshot(con_registro: bool = True) -> TwinSnapshot:
    """Con transformada por defecto: sin ella el frame del CBCT no conecta con el canónico
    y el volumen NO puede viajar — que es lo que prueba su propio test."""
    t = RigidTransform(rotation=(1.0, 0.0, 0.0, 0.0), translation=(1.0, 2.0, 3.0))
    return TwinSnapshot(
        acquisition_id="acq-1", timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:0",
        provenance=Provenance(source_file="x", modality=Modality.CBCT, agent="a@0",
                              transform=t if con_registro else None),
    )


# --- el sidecar se LEE del DICOM que viaja ----------------------------------- #
def test_el_sidecar_describe_la_serie_sin_abrir_un_solo_pixel(serie):
    """Un visor web no debería necesitar un parser DICOM completo para saber qué le llega."""
    sidecar, avisos = describe_serie(serie, frame="frame.ct_001")

    n = len(list(serie.glob("*.dcm")))
    assert sidecar["dimensions"][2] == n
    assert sidecar["frame"] == "frame.ct_001"
    assert all(s > 0 for s in sidecar["spacing_mm"])
    assert sidecar["modality"] == "CT"
    assert sidecar["origin_mm"] is not None
    assert "registrations" in sidecar["nota"]
    assert isinstance(avisos, list)


def test_la_orientacion_se_LEE_y_suponerla_se_declara_como_aviso(serie, tmp_path):
    """Un volumen al que se le supone la orientación se renderiza igual de bien y espejado."""
    import pydicom

    copia = tmp_path / "sin-iop"
    copia.mkdir()
    for f in sorted(serie.glob("*.dcm")):
        ds = pydicom.dcmread(str(f))
        del ds.ImageOrientationPatient
        ds.save_as(copia / f.name)

    _, avisos = describe_serie(copia, frame="frame.ct_001")

    assert any("SUPOSICION" in a for a in avisos)


def test_el_sidecar_NO_duplica_la_transformada_al_canonico(serie):
    """Dos sitios donde vive la misma transformada acaban siendo dos transformadas."""
    sidecar, _ = describe_serie(serie, frame="frame.ct_001")

    assert "transform" not in sidecar
    assert "transform_4x4_row_major" not in sidecar


# --- el digesto de un directorio --------------------------------------------- #
def test_el_digesto_de_la_serie_no_depende_del_orden_de_lectura(serie):
    a = asset_de_directorio(serie, "volume/ct_001/", id_="asset.ct_001",
                            kind=Clase.VOLUME, visit="v1", frame="frame.ct_001",
                            media_type="application/dicom")

    assert a.sha256 == digesto_de_partes(list(reversed(a.parts)))
    assert a.bytes == sum(p.bytes for p in a.parts)
    assert len(a.parts) == len(list(serie.glob("*.dcm")))


def test_renombrar_un_corte_NO_cambia_la_identidad_de_la_serie(serie, tmp_path):
    """D-3: el digesto identifica la SERIE CLÍNICA, no una serialización suya.

    ⚠️ **Antes iba sobre el nombre y el hash del fichero, y los dos cambian al
    de-identificar.** Reescribir cabeceras —el paso que da todo flujo clínico, y que B-3
    además exige— cambiaba el hash de cada corte y con él el digesto, así que la
    trazabilidad que el §3.4.2 promete se rompía justo en el paso más común.

    Ahora va sobre el `SOPInstanceUID` y el hash de `PixelData`: lo que DICOM define como
    identidad y lo que la de-identificación no toca. Renombrar deja de mover el digesto —
    lo cual es correcto: es la misma serie— y el nombre sigue viajando en `parts[]` como
    dato de ordenación, donde la verificación byte a byte lo sigue viendo.
    """
    original = asset_de_directorio(serie, "volume/ct_001/", id_="asset.ct_001",
                                   kind=Clase.VOLUME, visit="v1", frame="frame.ct_001",
                                   media_type="application/dicom")
    copia = tmp_path / "renombrada"
    copia.mkdir()
    for i, f in enumerate(sorted(serie.glob("*.dcm"))):
        (copia / f"otro_{i:04d}.dcm").write_bytes(f.read_bytes())
    renombrada = asset_de_directorio(copia, "volume/ct_001/", id_="asset.ct_001",
                                     kind=Clase.VOLUME, visit="v1", frame="frame.ct_001",
                                     media_type="application/dicom")

    assert renombrada.bytes == original.bytes
    assert renombrada.sha256 == original.sha256, (
        "renombrar cambio la identidad de la serie, y es la misma serie"
    )
    # Pero el nombre NO se pierde: sigue en `parts[]`, que es donde vive el orden.
    assert [p.name for p in renombrada.parts] != [p.name for p in original.parts]
    # Y la identidad es la que DICOM define, no una invencion nuestra.
    assert all(p.sop_instance_uid and p.pixel_data_sha256 for p in original.parts)


def _uos_con_la_serie_dentro(destino: Path, serie: Path) -> Path:
    """Un contenedor que SÍ lleva la serie corte a corte. Lo escribe el test.

    ⚠️ No lo produce `UOSExportAgent` porque el exportador ya no sabe hacerlo, y esa es la
    decisión correcta: nuestro producto no custodia originales. Lo que se prueba aquí no es
    el exportador sino `_valida_serie`, y un validador tiene que funcionar sobre lo que
    escribió otro emisor —incluido uno que sí decida llevar el DICOM dentro—.
    """
    import json

    from uos.contenedor import escribe_uos
    from uos.manifiesto import EstadoPHI, Frame, Manifiesto, Registro, Sujeto, Visita

    sidecar_uri = SIDECAR.format(id="ct_001")
    sidecar, _ = describe_serie(serie, frame="frame.ct_001")
    volumen = asset_de_directorio(
        serie, "volume/ct_001/", id_="asset.ct_001", kind=Clase.VOLUME, visit="v1",
        frame="frame.ct_001", media_type="application/dicom", sidecar_uri=sidecar_uri,
    )
    m = Manifiesto(
        case_id="urn:uuid:0", generator={"name": "test", "version": "0"},
        phi_state=EstadoPHI.PSEUDONYMIZED, subject=Sujeto(pseudonym="P-1"),
        # B-3: fuera de `identified`, el bloque es obligatorio.
        deidentification=Desidentificacion(
            profile="DICOM PS3.15 E.1 Basic Application Level Confidentiality Profile",
        ),
        canonical_frame=Frame(id="frame.ios_master"),
        # D-1/D-2 · el frame de un volumen se ancla al UID de DICOM y declara LPS. Los
        # dos se LEEN del sidecar, que a su vez los leyo de la serie: aqui no se inventan.
        frames=[Frame(
            id="frame.ct_001",
            dicom_frame_of_reference_uid=sidecar["dicom_frame_of_reference_uid"],
            anatomical=sidecar["anatomical"],
        )],
        visits=[Visita(id="v1", date="2026-08-28")],
        assets=[volumen],
        registrations=[Registro(
            id="reg.ct_to_ios", source_frame="frame.ct_001",
            target_frame="frame.ios_master", method="icp_surface",
            transform_4x4_row_major=[1.0, 0, 0, 0, 0, 1.0, 0, 0,
                                     0, 0, 1.0, 0, 0, 0, 0, 1.0],
        )],
    )
    return escribe_uos(
        destino, m, [], directorios={"volume/ct_001/": serie},
        extras={sidecar_uri: json.dumps(sidecar)},
    )


# --- el ciclo completo -------------------------------------------------------- #
def test_la_serie_sale_BYTE_IDENTICA_corte_a_corte(serie, tmp_path):
    caso = _uos_con_la_serie_dentro(tmp_path / "caso.uos", serie)

    assert valida(caso).valido
    with zipfile.ZipFile(caso) as z:
        for f in sorted(serie.glob("*.dcm")):
            assert z.read(f"volume/ct_001/{f.name}") == f.read_bytes(), f.name


def test_con_volumen_la_conformidad_sube_a_UOS_Vol(serie, malla, tmp_path):
    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla, cbct=serie,
    )
    informe = valida(salida.path)

    assert informe.valido, informe.errores
    assert Conformidad.VOL in informe.niveles
    assert Conformidad.CORE in informe.niveles


def test_sin_volumen_se_queda_en_UOS_Core_y_no_finge(malla, tmp_path):
    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla
    )
    informe = valida(salida.path)

    assert informe.niveles == [Conformidad.CORE]
    assert not any(a.kind is Clase.VOLUME for a in lee_manifiesto(salida.path).assets)


def test_el_volumen_vive_en_el_frame_del_CBCT_y_conecta_por_la_registracion(
    serie, malla, tmp_path
):
    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla, cbct=serie,
    )
    m = lee_manifiesto(salida.path)

    volumen = next(a for a in m.assets if a.kind is Clase.VOLUME)
    assert volumen.frame == "frame.ct_001"
    assert volumen.sidecar_uri == SIDECAR.format(id="ct_001")
    assert [r.id for r in m.registrations] == ["reg.ct_to_ios"]
    assert valida(salida.path).valido


def test_sin_registro_el_volumen_NO_viaja_y_el_resto_del_caso_SI(serie, malla, tmp_path):
    """Un volumen que no conecta con el canónico lo colocaría un visor en el sitio
    equivocado sin poder detectarlo, que es peor que no llevarlo. Y tirar la exportación
    entera por eso sería desproporcionado: la malla y las vistas están bien."""
    salida = UOSExportAgent(None).export(
        _snapshot(con_registro=False), tmp_path / "caso",
        pseudonimo="P-1", malla=malla, cbct=serie,
    )

    assert salida.ok, salida.detail
    assert valida(salida.path).niveles == [Conformidad.CORE]
    assert not any(a.kind is Clase.VOLUME for a in lee_manifiesto(salida.path).assets)
    assert any("el CBCT NO viaja" in m for m in salida.hitl_reasons)


def test_un_corte_que_falta_INVALIDA_aunque_el_resto_cuadre(serie, tmp_path):
    """El fallo que este nivel existe para cazar: una serie a la que le falta un corte se
    abre, se renderiza y tiene un hueco que nadie ve."""
    caso = _uos_con_la_serie_dentro(tmp_path / "caso.uos", serie)
    mutilado = tmp_path / "mutilado.uos"
    with zipfile.ZipFile(caso) as z, zipfile.ZipFile(
        mutilado, "w", compression=zipfile.ZIP_STORED
    ) as s:
        quitado = None
        for n in z.namelist():
            if quitado is None and n.startswith("volume/ct_001/"):
                quitado = n
                continue
            s.writestr(n, z.read(n))

    informe = valida(mutilado)

    assert not informe.valido
    assert any("falta" in e and quitado.split("/")[-1] in e for e in informe.errores)


def test_un_corte_ALTERADO_invalida_y_se_dice_CUAL(serie, tmp_path):
    """Un hash del conjunto diría «la serie no cuadra». Estos dicen qué corte."""
    caso = _uos_con_la_serie_dentro(tmp_path / "caso.uos", serie)
    tocado = tmp_path / "tocado.uos"
    victima = None
    with zipfile.ZipFile(caso) as z, zipfile.ZipFile(
        tocado, "w", compression=zipfile.ZIP_STORED
    ) as s:
        for n in z.namelist():
            crudo = z.read(n)
            if victima is None and n.startswith("volume/ct_001/"):
                victima = n
                crudo = crudo[:-1] + bytes([crudo[-1] ^ 0xFF])
            s.writestr(n, crudo)

    informe = valida(tocado)

    assert not informe.valido
    assert any(victima.split("/")[-1] in e and "sha256" in e for e in informe.errores)


def test_un_corte_de_MAS_tambien_invalida(serie, tmp_path):
    """Que la serie que sale no sea la que se dice que entró es igual de grave por exceso."""
    caso = _uos_con_la_serie_dentro(tmp_path / "caso.uos", serie)
    colado = tmp_path / "colado.uos"
    with zipfile.ZipFile(caso) as z, zipfile.ZipFile(
        colado, "w", compression=zipfile.ZIP_STORED
    ) as s:
        for n in z.namelist():
            s.writestr(n, z.read(n))
        s.writestr("volume/ct_001/intruso.dcm", b"no es un corte")

    informe = valida(colado)

    assert not informe.valido
    assert any("no declara" in e for e in informe.errores)


def test_el_sidecar_del_volumen_viaja_y_el_manifiesto_lo_declara(serie, malla, tmp_path):
    import json

    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla, cbct=serie,
    )

    with zipfile.ZipFile(salida.path) as z:
        sidecar = json.loads(z.read(SIDECAR.format(id="ct_001")))
    assert sidecar["dimensions"][2] == len(list(serie.glob("*.dcm")))
    assert sidecar["frame"] == "frame.ct_001"


def test_el_volumen_esta_en_el_fhir_map_como_ImagingStudy(serie, malla, tmp_path):
    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla, cbct=serie,
    )

    m = lee_manifiesto(salida.path)
    assert m.fhir_map["asset.ct_001"].resource_type == "ImagingStudy"
    # Y sin referencia concreta: no hay servidor FHIR donde ese recurso exista.
    assert m.fhir_map["asset.ct_001"].resource is None


# --- y el DICOM no puede desmentir al `phi_state` ---------------------------- #
def test_una_serie_CON_nombre_de_paciente_no_viaja_y_se_dice_que_tag(serie, malla, tmp_path):
    """El fallo que este control existe para cazar.

    El DICOM viaja intacto —es el punto del formato— así que sus cabeceras viajan con él, y
    el manifiesto afirma `phi_state: pseudonymized`. Un contenedor que dice estar
    seudonimizado y lleva el nombre dentro es PEOR que uno que declara `identified`: quien
    lo reciba se fía del campo y no abre 397 cabeceras a comprobarlo.
    """
    import pydicom

    con_phi = tmp_path / "con-phi"
    con_phi.mkdir()
    for f in sorted(serie.glob("*.dcm")):
        ds = pydicom.dcmread(str(f))
        ds.PatientName = "APELLIDO^NOMBRE"
        ds.InstitutionName = "Clinica Ejemplo"
        ds.save_as(con_phi / f.name)

    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla, cbct=con_phi
    )

    assert salida.ok, salida.detail
    assert valida(salida.path).niveles == [Conformidad.CORE]
    assert not any(a.kind is Clase.VOLUME for a in lee_manifiesto(salida.path).assets)
    motivo = next(m for m in salida.hitl_reasons if "el CBCT NO viaja" in m)
    assert "PatientName" in motivo and "InstitutionName" in motivo
    # Y el motivo dice QUÉ tag, no el valor: repetir el nombre lo sacaría igualmente.
    assert "APELLIDO" not in motivo and "Ejemplo" not in motivo


def test_un_identificador_OPACO_no_cuenta_como_dato_identificable(serie, malla, tmp_path):
    """Un export anonimizado rellena `PatientID` con un UUID y `PatientName` con un
    marcador. Exigir que estén vacíos rechazaría series perfectamente anónimas."""
    import pydicom

    anonima = tmp_path / "anonima"
    anonima.mkdir()
    for f in sorted(serie.glob("*.dcm")):
        ds = pydicom.dcmread(str(f))
        ds.PatientName = "Anonymized3"
        ds.PatientID = "05a06426-d1e9-4d98-9a90-ff6841ad92b6"
        ds.save_as(anonima / f.name)

    salida = UOSExportAgent(None).export(
        _snapshot(), tmp_path / "caso", pseudonimo="P-1", malla=malla, cbct=anonima
    )

    assert Conformidad.VOL in valida(salida.path).niveles
    assert not any("el CBCT NO viaja" in m for m in salida.hitl_reasons)


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
