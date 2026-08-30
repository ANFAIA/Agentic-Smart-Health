"""La cadena de hashes entre versiones (§8).

Lo que se prueba no es que la cadena se escriba, sino que **no se pueda mentir con ella**:
un `.uos` que declara historial y no lo tiene, o que dice venir de una version que no es,
tiene que invalidar. Declarar procedencia y que no cuadre es peor que no declararla,
porque quien lo abra da por hecho que puede recorrerla.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from uos import Asset, Frame, Manifiesto, Sujeto, Visita, escribe_uos, valida
from uos.contenedor import MANIFIESTO
from uos.manifiesto import Clase, EstadoPHI, Procedencia
from uos.procedencia import (
    CADENA,
    FIRMAS,
    Cadena,
    Eslabon,
    encadena,
    lee_version_previa,
)

CASO = "urn:uuid:0"


def _asset(ruta: Path) -> Asset:
    crudo = ruta.read_bytes()
    return Asset(
        id="asset.ios", kind=Clase.MESH_GS_SCENE, visit="v1", uri="scene/scan.stl",
        media_type="model/stl", sha256=hashlib.sha256(crudo).hexdigest(),
        bytes=len(crudo), frame="frame.ios_master",
    )


def _manifiesto(assets, **kw) -> Manifiesto:
    base = dict(
        case_id=CASO, generator={"name": "test", "version": "0"},
        phi_state=EstadoPHI.PSEUDONYMIZED, subject=Sujeto(pseudonym="P-1"),
        canonical_frame=Frame(id="frame.ios_master"),
        visits=[Visita(id="v1", date="2026-08-23")], assets=assets,
    )
    return Manifiesto(**{**base, **kw})


@pytest.fixture
def malla(tmp_path) -> Path:
    p = tmp_path / "scan.stl"
    p.write_bytes(_stl_binario())
    return p


def _escribe(destino: Path, malla: Path, previo: Path | None = None) -> Path:
    """Una version del caso, encadenada a la anterior si se da."""
    previo_sha, cadena_previa, _ = (lee_version_previa(previo) if previo
                                    else (None, None, None))
    m = _manifiesto(
        [_asset(malla)],
        provenance=Procedencia(prev_manifest_sha256=previo_sha, chain=CADENA),
    )
    crudo = m.json_canonico()
    cadena = encadena(
        case_id=CASO, manifiesto_json=crudo, previo_sha256=previo_sha,
        cadena_previa=cadena_previa, generator=m.generator, assets=1,
    )
    return escribe_uos(destino, m, [("scene/scan.stl", malla)],
                       json_manifiesto=crudo, extras={CADENA: cadena.json_canonico()})


def _reescribe(origen: Path, destino: Path, **reemplazos: str) -> Path:
    """Copia un `.uos` cambiando entradas de texto. Es como se falsifica uno."""
    with zipfile.ZipFile(origen) as z, zipfile.ZipFile(
        destino, "w", compression=zipfile.ZIP_STORED
    ) as s:
        for n in z.namelist():
            s.writestr(n, reemplazos.get(n, z.read(n)))
    return destino


# --- la cadena se construye ------------------------------------------------- #
def test_la_primera_version_no_viene_de_ninguna_parte(tmp_path, malla):
    v1 = _escribe(tmp_path / "v1.uos", malla)

    inf = valida(v1)
    with zipfile.ZipFile(v1) as z:
        cadena = Cadena.model_validate_json(z.read(CADENA))

    assert inf.valido, inf.errores
    assert inf.version == 1
    assert len(cadena.links) == 1
    assert cadena.links[0].prev_manifest_sha256 is None


def test_la_segunda_version_apunta_al_HASH_de_la_primera(tmp_path, malla):
    """`prev_manifest_sha256` es la autoridad: la cadena solo lo materializa."""
    v1 = _escribe(tmp_path / "v1.uos", malla)
    v2 = _escribe(tmp_path / "v2.uos", malla, previo=v1)

    with zipfile.ZipFile(v1) as z:
        hash_v1 = hashlib.sha256(z.read(MANIFIESTO)).hexdigest()
    with zipfile.ZipFile(v2) as z:
        m2 = Manifiesto.model_validate_json(z.read(MANIFIESTO))
        cadena = Cadena.model_validate_json(z.read(CADENA))

    assert valida(v2).valido
    assert m2.provenance.prev_manifest_sha256 == hash_v1
    assert [e.version for e in cadena.links] == [1, 2]
    assert cadena.links[1].prev_manifest_sha256 == hash_v1
    # El eslabon de v1 sobrevive intacto: la cadena crece, no se reescribe.
    assert cadena.links[0].manifest_sha256 == hash_v1


def test_la_cadena_crece_sin_perder_eslabones(tmp_path, malla):
    v = _escribe(tmp_path / "v1.uos", malla)
    for i in (2, 3, 4):
        v = _escribe(tmp_path / f"v{i}.uos", malla, previo=v)

    inf = valida(v)
    with zipfile.ZipFile(v) as z:
        cadena = Cadena.model_validate_json(z.read(CADENA))

    assert inf.valido, inf.errores
    assert inf.version == 4
    assert [e.version for e in cadena.links] == [1, 2, 3, 4]
    for anterior, actual in zip(cadena.links, cadena.links[1:], strict=False):
        assert actual.prev_manifest_sha256 == anterior.manifest_sha256


def test_un_case_id_distinto_EMPIEZA_cadena_en_vez_de_mezclarse(tmp_path, malla):
    """Dos casos distintos no son la misma historia aunque se exporten al mismo sitio."""
    v1 = _escribe(tmp_path / "v1.uos", malla)
    _, cadena_previa, _ = lee_version_previa(v1)

    otra = encadena(
        case_id="urn:uuid:otro", manifiesto_json="{}", previo_sha256=None,
        cadena_previa=cadena_previa, generator={}, assets=1,
    )

    assert otra.case_id == "urn:uuid:otro"
    assert [e.version for e in otra.links] == [1]


# --- y no se puede mentir con ella ------------------------------------------ #
def test_un_manifiesto_retocado_deja_de_cuadrar_con_su_cadena(tmp_path, malla):
    """El fallo que la cadena existe para cazar: alguien edita el manifiesto de la version
    que tiene en la mano y lo reparte como si fuera el que se emitio."""
    v1 = _escribe(tmp_path / "v1.uos", malla)
    with zipfile.ZipFile(v1) as z:
        m = json.loads(z.read(MANIFIESTO))
    m["subject"]["pseudonym"] = "OTRO"
    falso = _reescribe(v1, tmp_path / "falso.uos", **{MANIFIESTO: json.dumps(m, indent=1)})

    inf = valida(falso)

    assert not inf.valido
    assert any("no termina aqui" in e for e in inf.errores)


def test_un_eslabon_arrancado_del_medio_rompe_la_cadena(tmp_path, malla):
    v1 = _escribe(tmp_path / "v1.uos", malla)
    v2 = _escribe(tmp_path / "v2.uos", malla, previo=v1)
    v3 = _escribe(tmp_path / "v3.uos", malla, previo=v2)
    with zipfile.ZipFile(v3) as z:
        cadena = Cadena.model_validate_json(z.read(CADENA))
    sin_medio = Cadena(case_id=cadena.case_id, links=[cadena.links[0], cadena.links[2]])
    roto = _reescribe(v3, tmp_path / "roto.uos", **{CADENA: sin_medio.json_canonico()})

    inf = valida(roto)

    assert not inf.valido
    assert any("la cadena esta rota" in e for e in inf.errores)


def test_una_cadena_de_otro_caso_no_cuela(tmp_path, malla):
    v1 = _escribe(tmp_path / "v1.uos", malla)
    with zipfile.ZipFile(v1) as z:
        cadena = Cadena.model_validate_json(z.read(CADENA))
    ajena = Cadena(case_id="urn:uuid:otro", links=cadena.links)
    mezclado = _reescribe(v1, tmp_path / "mezcla.uos", **{CADENA: ajena.json_canonico()})

    inf = valida(mezclado)

    assert not inf.valido
    assert any("dos historias distintas" in e for e in inf.errores)


def test_declarar_cadena_y_no_llevarla_invalida(tmp_path, malla):
    """Un lector que ve `provenance.chain` da por hecho que puede recorrer el historial."""
    m = _manifiesto([_asset(malla)], provenance=Procedencia(chain=CADENA))
    salida = escribe_uos(tmp_path / "sin.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)

    assert not inf.valido
    assert any("no esta en el contenedor" in e for e in inf.errores)


def test_llevar_cadena_sin_declararla_tambien_invalida(tmp_path, malla):
    """Una cadena que nadie referencia no se puede verificar, y aparenta procedencia."""
    m = _manifiesto([_asset(malla)])
    salida = escribe_uos(
        tmp_path / "huerfana.uos", m, [("scene/scan.stl", malla)],
        extras={CADENA: Cadena(case_id=CASO).json_canonico()},
    )

    inf = valida(salida)

    assert not inf.valido
    assert any("no lo declara" in e for e in inf.errores)


def test_venir_de_una_version_anterior_sin_cadena_es_un_AVISO_no_un_error(tmp_path, malla):
    """El historial existe —lo dice el hash— pero no hay por donde recorrerlo."""
    m = _manifiesto([_asset(malla)], provenance=Procedencia(prev_manifest_sha256="a" * 64))
    salida = escribe_uos(tmp_path / "corta.uos", m, [("scene/scan.stl", malla)])

    inf = valida(salida)

    assert inf.valido
    assert any("no hay" in a and CADENA in a for a in inf.avisos)


def test_las_firmas_no_se_ignoran_en_silencio(tmp_path, malla):
    """No se verifican —falta decidir que clave firma—, y por eso se dicen. Un `.uos` que
    parece firmado y que nadie ha comprobado es el fallo callado de siempre."""
    v1 = _escribe(tmp_path / "v1.uos", malla)
    firmado = tmp_path / "firmado.uos"
    with zipfile.ZipFile(v1) as z, zipfile.ZipFile(
        firmado, "w", compression=zipfile.ZIP_STORED
    ) as s:
        for n in z.namelist():
            s.writestr(n, z.read(n))
        s.writestr(f"{FIRMAS}clinica.sig", b"no es una firma")

    inf = valida(firmado)

    assert inf.valido
    assert any("NO comprueba" in a for a in inf.avisos)


def test_un_uos_previo_ilegible_empieza_cadena_en_vez_de_reventar(tmp_path, malla):
    """Un contenedor anterior corrupto no debe impedir escribir el actual; lo que no puede
    pasar es que se declare una continuidad que no existe."""
    roto = tmp_path / "roto.uos"
    roto.write_bytes(b"esto no es un zip")

    assert lee_version_previa(roto) == (None, None, None)

    v = _escribe(tmp_path / "v1.uos", malla, previo=roto)
    with zipfile.ZipFile(v) as z:
        m = Manifiesto.model_validate_json(z.read(MANIFIESTO))
    assert m.provenance.prev_manifest_sha256 is None
    assert valida(v).valido


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


def test_una_cadena_ROTA_no_se_continua_y_se_dice(tmp_path):
    """Una cadena cuyo último eslabón no cuadra con su manifiesto no se encadena encima.

    ⚠️ El caso real: cuatro versiones donde el eslabón 3 declaraba `38f0a225…` y su
    manifiesto hasheaba `0ec380e4…`. La exportación siguiente construía un `prev` correcto
    —el hash real— que no cuadraba con lo declarado, el validador la rechazaba entera, y el
    caso quedaba **inexportable para siempre** por una corrupción heredada.

    Repararla en silencio sería peor: una cadena de procedencia que se auto-repara no sirve
    para nada. Lo correcto es no reclamar una continuidad que no existe, empezar cadena
    nueva y decir por qué.
    """
    import json
    import zipfile

    v1 = tmp_path / "caso.uos"
    manifiesto = json.dumps({"uos_version": "0.2"})
    cadena = Cadena(case_id="urn:uuid:0", links=[Eslabon(
        version=1,
        manifest_sha256="a" * 64,          # NO es el hash de `manifiesto`
        prev_manifest_sha256=None,
        created=datetime.now(UTC),
        generator={"name": "t", "version": "0"},
        assets=1,
    )])
    with zipfile.ZipFile(v1, "w") as z:
        z.writestr("manifest.json", manifiesto)
        z.writestr(CADENA, cadena.model_dump_json())

    sha, prev, aviso = lee_version_previa(v1)
    assert (sha, prev) == (None, None), "no puede reclamar continuidad con una cadena rota"
    assert aviso and "ROTA" in aviso and "cadena nueva" in aviso


def test_una_cadena_SANA_si_se_continua(tmp_path):
    """El contrario, para que el de arriba pueda fallar."""
    import hashlib
    import json
    import zipfile

    v1 = tmp_path / "caso.uos"
    manifiesto = json.dumps({"uos_version": "0.2"})
    real = hashlib.sha256(manifiesto.encode("utf-8")).hexdigest()
    cadena = Cadena(case_id="urn:uuid:0", links=[Eslabon(
        version=1, manifest_sha256=real, prev_manifest_sha256=None,
        created=datetime.now(UTC), generator={"name": "t", "version": "0"}, assets=1,
    )])
    with zipfile.ZipFile(v1, "w") as z:
        z.writestr("manifest.json", manifiesto)
        z.writestr(CADENA, cadena.model_dump_json())

    sha, prev, aviso = lee_version_previa(v1)
    assert sha == real and prev is not None and aviso is None


def test_una_rotura_EN_MEDIO_de_la_cadena_tambien_corta_la_continuidad(tmp_path):
    """No basta con mirar el último eslabón: una cadena vale lo que su eslabón más débil.

    ⚠️ Este es el caso que mi primer arreglo dejó pasar, y por eso está escrito aparte. Tras
    una exportación fallida el **último** eslabón cuadraba con su manifiesto —lo había
    escrito bien— y la rotura estaba entre la v3 y la v4. La comprobación superficial lo
    daba por bueno, se encadenaba encima, y el validador seguía rechazando el contenedor
    entero: el caso quedaba igual de inexportable, pero ahora en silencio.
    """
    import zipfile

    manifiesto = json.dumps({"uos_version": "0.2"})
    real = hashlib.sha256(manifiesto.encode("utf-8")).hexdigest()
    ahora = datetime.now(UTC)
    gen = {"name": "t", "version": "0"}
    cadena = Cadena(case_id="urn:uuid:0", links=[
        Eslabon(version=1, manifest_sha256="1" * 64, prev_manifest_sha256=None,
                created=ahora, generator=gen, assets=1),
        # ⚠️ la rotura: apunta a "9"*64 y el eslabón 1 declara "1"*64
        Eslabon(version=2, manifest_sha256=real, prev_manifest_sha256="9" * 64,
                created=ahora, generator=gen, assets=1),
    ])
    v = tmp_path / "caso.uos"
    with zipfile.ZipFile(v, "w") as z:
        z.writestr("manifest.json", manifiesto)
        z.writestr(CADENA, cadena.model_dump_json())

    sha, prev, aviso = lee_version_previa(v)
    # El último eslabón SÍ cuadra con el manifiesto, que es lo que engañaba antes.
    assert cadena.links[-1].manifest_sha256 == real
    assert (sha, prev) == (None, None), "una rotura interna también corta la continuidad"
    assert aviso and "ROTA" in aviso
