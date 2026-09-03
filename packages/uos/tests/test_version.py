"""§15 — leer un contenedor que declara OTRA version.

⚠️ **El campo se escribia y no lo miraba nadie.** `uos_version` iba en el manifiesto desde
el principio y ni el lector ni el validador lo comparaban con la suya, asi que la garantia
de las versiones menores —«un lector v0.2 abre un contenedor v0.3»— no la podia cumplir
nuestro propio lector: todos los modelos llevan `extra="forbid"`, y un campo opcional nuevo
reventaba el parseo en vez de ignorarse.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from uos.contenedor import MANIFIESTO, read_manifest, read_manifest_from
from uos.version import Lectura, como_leer, partes, puede_reemitir


def _manifiesto_crudo(**cambios) -> bytes:
    base = {
        "uos_version": "0.2", "case_id": "urn:uuid:0",
        "generator": {"name": "test", "version": "0"},
        "phi_state": "pseudonymized", "subject": {"pseudonym": "P-1"},
        "canonical_frame": {"id": "frame.ios_master"},
    }
    return json.dumps({**base, **cambios}).encode()


def test_una_menor_superior_se_lee_IGNORANDO_lo_que_no_se_conoce():
    """La garantia de §15: lo que este lector conoce sigue significando lo mismo."""
    crudo = _manifiesto_crudo(uos_version="0.3", campo_de_la_03="lo que sea")

    m, ignorados = read_manifest_from(crudo)

    assert m.case_id == "urn:uuid:0"
    assert ignorados == ["campo_de_la_03"], "y se dice CUAL se ignoro, no se calla"


def test_se_ignora_tambien_dentro_de_un_asset_anidado():
    """`extra="forbid"` esta en cada modelo, no solo en la raiz."""
    asset = {
        "id": "asset.x", "kind": "mesh_gs_scene", "visit": "v1", "uri": "scene/x.glb",
        "media_type": "model/gltf-binary", "sha256": "0" * 64, "bytes": 1,
        "frame": "frame.ios_master", "novedad_de_la_03": 7,
    }
    m, ignorados = read_manifest_from(_manifiesto_crudo(uos_version="0.3", assets=[asset]))

    assert m.assets[0].id == "asset.x"
    assert ignorados == ["assets.0.novedad_de_la_03"]


def test_una_MAYOR_superior_se_RECHAZA_aunque_no_traiga_campos_nuevos():
    """⚠️ El peligro no son los campos desconocidos, son los CONOCIDOS.

    Este manifiesto no trae ni un campo que el lector no sepa leer, y aun asi se rechaza:
    en una mayor, un campo que se reconoce puede haber cambiado de significado —si `density`
    dejara de ser sigma normalizada, se leeria mal sin que nada fallara—. Por eso «ignoro lo
    que no entiendo» no basta aqui.
    """
    with pytest.raises(ValueError, match="mayor"):
        read_manifest_from(_manifiesto_crudo(uos_version="1.0"))


def test_la_propia_version_y_las_anteriores_se_leen_ESTRICTO():
    """Dentro de una version se conoce el conjunto completo: un campo raro es una errata."""
    with pytest.raises(Exception, match="extra"):
        read_manifest_from(_manifiesto_crudo(sha_256="typo"))

    m, ignorados = read_manifest_from(_manifiesto_crudo(uos_version="0.1"))
    assert ignorados == []


def test_leer_permisivo_y_REEMITIR_no_se_combinan():
    """El fallo callado que esto evita: la cadena diria sucesora legitima y no lo seria.

    Un lector v0.2 que abre una v0.3, ignora lo que no entiende y escribe la version N+1
    BORRARIA esos campos. La procedencia seguiria cuadrando criptograficamente mientras el
    contenido se pierde, que es peor que negarse.
    """
    assert puede_reemitir("0.2") and puede_reemitir("0.1")
    assert not puede_reemitir("0.3"), "leerla si, sucederla no"
    assert not puede_reemitir("1.0")


def test_una_version_ILEGIBLE_no_se_da_por_compatible():
    """Tratarla como la mas antigua la leeria estricto y en silencio: eso es adivinar."""
    with pytest.raises(ValueError, match="mayor.*menor|no se puede decidir"):
        partes("cero coma dos")


@pytest.mark.parametrize(("version", "rama"), [
    ("0.1", Lectura.ESTRICTA), ("0.2", Lectura.ESTRICTA),
    ("0.3", Lectura.PERMISIVA), ("0.9", Lectura.PERMISIVA),
    ("1.0", Lectura.RECHAZO), ("2.1", Lectura.RECHAZO),
])
def test_las_tres_ramas(version, rama):
    assert como_leer(version) is rama


def test_el_lector_de_un_uos_real_tambien_comprueba_la_version(tmp_path):
    """No basta con la funcion: `read_manifest` tiene que usarla."""
    caso = tmp_path / "futuro.uos"
    with zipfile.ZipFile(caso, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(MANIFIESTO, _manifiesto_crudo(uos_version="9.0"))

    with pytest.raises(ValueError, match="mayor"):
        read_manifest(Path(caso))
