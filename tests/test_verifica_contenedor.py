"""El verificador de coherencia, probado sobre contenedores fabricados a mano.

Un comprobador que solo se ejecuta sobre el caso bueno no comprueba nada: aquí se le dan
contenedores con cada incoherencia metida a propósito, y tiene que encontrarlas. Y uno
limpio, para que los otros puedan fallar.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import zipfile
from pathlib import Path

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "verifica_contenedor.py"
_spec = importlib.util.spec_from_file_location("verifica_contenedor", _RUTA)
assert _spec and _spec.loader
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def _ply(n: int, *, unidades: str | None = "mm", tilde: bool = False,
         propiedades: tuple[str, ...] = ("x",)) -> bytes:
    cab = ["ply", "format binary_little_endian 1.0"]
    if tilde:
        cab.append("comment el optimizador las movió")
    if unidades:
        cab.append(f"comment unidades {unidades}")
    cab += [f"element vertex {n}"]
    cab += [f"property float {p}" for p in propiedades]
    cab += ["end_header"]
    cuerpo = struct.pack(f"<{n * len(propiedades)}f", *range(n * len(propiedades)))
    return ("\n".join(cab) + "\n").encode("utf-8") + cuerpo


def _caso(tmp: Path, *, n_ply: int = 10, n_declara: int = 10,
          unidades_ply: str | None = "mm", unidades_sc: str = "mm",
          perfil: str = "ash-twin/1.0", tilde: bool = False,
          propiedades: tuple[str, ...] = ("x",),
          columnas: tuple[str, ...] | None = None) -> Path:
    crudo = _ply(n_ply, unidades=unidades_ply, tilde=tilde, propiedades=propiedades)
    manifiesto = {
        "uos_version": "0.2",
        "assets": [{
            "id": "asset.field", "uri": "scene/field.ply",
            "media_type": "application/octet-stream",
            "sha256": hashlib.sha256(crudo).hexdigest(), "bytes": len(crudo),
            "sidecar_uri": "scene/field.gs.json",
        }],
    }
    sidecar = {"profile": perfil, "units": unidades_sc,
               "n_primitives": n_declara, "measured": True,
               "columns": [{"name": c}
                           for c in (propiedades if columnas is None else columnas)]}
    destino = tmp / "caso.uos"
    with zipfile.ZipFile(destino, "w") as z:
        z.writestr("manifest.json", json.dumps(manifiesto))
        z.writestr("scene/field.ply", crudo)
        z.writestr("scene/field.gs.json", json.dumps(sidecar))
    return destino


def test_un_contenedor_coherente_no_da_fallos(tmp_path):
    """Para que los demás puedan fallar."""
    assert vc.verifica(_caso(tmp_path)) == []


def test_caza_que_n_primitives_no_es_el_del_fichero(tmp_path):
    """⚠️ Pasó de verdad: el sidecar del compuesto declaraba 1.341.990 —el número del campo
    semilla, copiado del snapshot— sobre un fichero de 1.454.057 gaussianas."""
    f = vc.verifica(_caso(tmp_path, n_ply=1454, n_declara=1341))
    assert any("1,341" in x and "1,454" in x for x in f), f


def test_caza_que_las_unidades_declaradas_no_son_las_del_fichero(tmp_path):
    """⚠️ El fallo caro: `units: "mm"` sobre un PLY en el espacio normalizado de Blender.
    La nube salía 32 veces más pequeña y el visor pintaba una mota en el origen."""
    f = vc.verifica(_caso(tmp_path, unidades_ply="normalizado", unidades_sc="mm"))
    assert any("unidades" in x and "normalizado" in x for x in f), f


def test_caza_los_bytes_no_ASCII_de_la_cabecera(tmp_path):
    """El formato PLY define la cabecera como ASCII; la nuestra llevaba tildes."""
    f = vc.verifica(_caso(tmp_path, tilde=True))
    assert any("no-ASCII" in x for x in f), f


def test_caza_un_perfil_desconocido(tmp_path):
    """Las columnas se llaman igual que las del 3DGS de facto y significan otra cosa: un
    perfil que no reconocemos es un fichero que alguien va a pintar mal."""
    f = vc.verifica(_caso(tmp_path, perfil="inria/1.0"))
    assert any("perfil desconocido" in x for x in f), f


def test_caza_la_segmentacion_que_no_indexa_nada(tmp_path):
    """`derived/seg_teeth` indexa los vértices de `scene.glb`. Sin la escena es una lista
    de códigos que no indexa nada — y con `--solo-gaussianas` eso puede pasar."""
    destino = tmp_path / "sinescena.uos"
    crudo = _ply(4)
    m = {"uos_version": "0.2", "assets": [
        {"id": "asset.field", "uri": "scene/field.ply",
         "media_type": "application/octet-stream",
         "sha256": hashlib.sha256(crudo).hexdigest(), "bytes": len(crudo)},
        {"id": "asset.seg", "uri": "derived/seg_teeth.bin",
         "media_type": "application/octet-stream",
         "sha256": "0" * 64, "bytes": 8},
    ]}
    with zipfile.ZipFile(destino, "w") as z:
        z.writestr("manifest.json", json.dumps(m))
        z.writestr("scene/field.ply", crudo)
        z.writestr("derived/seg_teeth.bin", b"\x00" * 8)
    f = vc.verifica(destino)
    assert any("no indexan nada" in x for x in f), f


def test_caza_un_asset_externo_con_ruta_en_vez_de_hash(tmp_path):
    """Un fichero que no viaja se nombra por su contenido: una ruta sería una promesa
    sobre un ZIP en el que no está."""
    destino = tmp_path / "ext.uos"
    m = {"uos_version": "0.2", "assets": [
        {"id": "asset.ios", "uri": "scene/scan.stl", "external": True,
         "media_type": "model/stl", "sha256": "a" * 64, "bytes": 1},
    ]}
    with zipfile.ZipFile(destino, "w") as z:
        z.writestr("manifest.json", json.dumps(m))
    f = vc.verifica(destino)
    assert any("es externo y su uri" in x for x in f), f


def test_caza_una_propiedad_del_PLY_que_el_sidecar_no_declara(tmp_path):
    """⚠️ Pasó de verdad, dos veces en el mismo contenedor: el PLY de apariencia traía
    `region_id` —el código FDI por gaussiana, que es lo que permite encender una pieza sin
    malla— y el sidecar declaraba catorce columnas de dieciocho; y el compuesto traía
    `origen` —de qué modalidad viene cada gaussiana— sin declararlo tampoco.

    El fichero no se rompe: el dato simplemente no existe para ningún lector que no sea el
    nuestro, que es lo contrario de un formato abierto."""
    f = vc.verifica(_caso(tmp_path, propiedades=("x", "region_id"), columnas=("x",)))
    assert any("no declara" in x and "region_id" in x for x in f), f


def test_caza_una_columna_declarada_que_el_PLY_no_trae(tmp_path):
    """La mentira simétrica: promete un dato que no está y descoloca a quien monte el
    registro desde `columns`."""
    f = vc.verifica(_caso(tmp_path, propiedades=("x",), columnas=("x", "density")))
    assert any("el PLY no trae" in x and "density" in x for x in f), f


def test_caza_las_mismas_columnas_en_otro_orden(tmp_path):
    """`columns` es la receta para montar el registro binario: en otro orden se lee todo
    cruzado sin que nada falle."""
    f = vc.verifica(_caso(tmp_path, propiedades=("x", "y"), columnas=("y", "x")))
    assert any("otro orden" in x for x in f), f


def test_caza_una_pieza_que_el_informe_da_por_AUSENTE_y_la_segmentacion_emite(tmp_path):
    """El informe dice «Diente 28 Ausente» y el modelo emite un FDI 28. Las dos
    afirmaciones viajan en el mismo `.uos` y nadie las cruzaba.

    ⚠️ No son dos opiniones al mismo nivel: la transcripción del informe es **Layer 1**
    —un documento que firmó una persona— y la segmentación **Layer 3**, inferencia de un
    modelo. Un contenedor que afirma las dos se contradice, y el formato promete que eso
    lo puede comprobar **quien lo recibe**, sin acceso al pipeline que lo escribió.
    """
    import numpy as np

    crudo = _ply(4)
    seg = np.array([27, 27, 28, 0], dtype="<i2").tobytes()
    glb_json = json.dumps({
        "accessors": [{"count": 4}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }).encode()
    glb = b"glTF" + struct.pack("<II", 2, 0) + struct.pack("<I", len(glb_json)) + b"JSON" + glb_json
    clinico = {"teeth": [
        {"fdi": "27", "findings": ["calculo_pulpar"]},
        {"fdi": "28", "findings": ["ausente"]},
    ]}
    m = {"uos_version": "0.2", "assets": [
        {"id": "asset.field", "uri": "scene/field.ply",
         "media_type": "application/octet-stream",
         "sha256": hashlib.sha256(crudo).hexdigest(), "bytes": len(crudo)},
        {"id": "asset.scene", "uri": "scene/scene.glb",
         "media_type": "model/gltf-binary",
         "sha256": hashlib.sha256(glb).hexdigest(), "bytes": len(glb)},
        {"id": "asset.seg", "uri": "derived/seg_teeth.bin",
         "media_type": "application/octet-stream",
         "sha256": hashlib.sha256(seg).hexdigest(), "bytes": len(seg)},
        {"id": "asset.clinical", "uri": "clinical/observations.json",
         "media_type": "application/json", "sha256": "0" * 64, "bytes": 1},
    ]}
    destino = tmp_path / "contradictorio.uos"
    with zipfile.ZipFile(destino, "w") as z:
        z.writestr("manifest.json", json.dumps(m))
        z.writestr("scene/field.ply", crudo)
        z.writestr("scene/scene.glb", glb)
        z.writestr("derived/seg_teeth.bin", seg)
        z.writestr("clinical/observations.json", json.dumps(clinico))
    f = vc.verifica(destino)
    assert any("AUSENTE" in x and "FDI 28" in x for x in f), f

    # Y sin la contradicción no dice nada: el 27 lo emite el modelo y el informe no lo
    # da por ausente, así que ahí no hay nada que reprochar.
    sin_28 = np.array([27, 27, 27, 0], dtype="<i2").tobytes()
    m["assets"][2]["sha256"] = hashlib.sha256(sin_28).hexdigest()
    m["assets"][2]["bytes"] = len(sin_28)
    limpio = tmp_path / "coherente.uos"
    with zipfile.ZipFile(limpio, "w") as z:
        z.writestr("manifest.json", json.dumps(m))
        z.writestr("scene/field.ply", crudo)
        z.writestr("scene/scene.glb", glb)
        z.writestr("derived/seg_teeth.bin", sin_28)
        z.writestr("clinical/observations.json", json.dumps(clinico))
    assert not [x for x in vc.verifica(limpio) if "AUSENTE" in x]
