"""El STL mejorado sacado del `.uos`. Ver `scripts/malla_mejorada.py`.

Lo que se prueba aquí son los dos LECTORES, que es donde este proyecto ya se ha roto: una
convención supuesta en vez de preguntada se rompe en silencio el día que el emisor cambia,
y el fichero sale con colores plausibles y mal.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "malla_mejorada.py"
_spec = importlib.util.spec_from_file_location("malla_mejorada_script", _RUTA)
assert _spec and _spec.loader
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def _stl(triangulos: np.ndarray) -> bytes:
    tri = np.zeros(len(triangulos), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                              ("a", "<u2")]))
    tri["v"] = triangulos
    return b"\0" * 80 + struct.pack("<I", len(triangulos)) + tri.tobytes()


def test_el_stl_se_lee_uniendo_los_vertices_repetidos() -> None:
    """⚠️ **Un STL repite cada punto una vez por triángulo que lo toca.**

    Sin unirlos, el color se calcularía varias veces para el mismo punto y el PLY saldría
    con seis veces más vértices de los que tiene la superficie — y el 3MF, ilegible.
    """
    a, b, c, d = [0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [1.0, 1, 0]
    datos = _stl(np.array([[a, b, c], [b, d, c]]))  # dos triángulos, arista compartida
    pos, caras = mm.lee_stl(datos)
    assert len(pos) == 4, "los dos vértices de la arista compartida no se han unido"
    assert caras.shape == (2, 3)
    # Y la geometría sobrevive: cada triángulo sigue siendo el mismo en el espacio.
    assert np.allclose(sorted(pos[caras[0]].tolist()), sorted([a, b, c]))


def _ply(propiedades: list[tuple[str, str]], filas: dict[str, list[float]]) -> bytes:
    n = len(next(iter(filas.values())))
    cab = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
           *(f"property {t} {nombre}" for t, nombre in propiedades), "end_header"]
    tipos = {"float": "<f4", "uchar": "u1"}
    dt = np.dtype([(nombre, tipos[t]) for t, nombre in propiedades])
    v = np.empty(n, dt)
    for _, nombre in propiedades:
        v[nombre] = filas[nombre]
    return ("\n".join(cab) + "\n").encode("ascii") + v.tobytes()


def test_la_convencion_del_campo_se_PREGUNTA_al_descriptor() -> None:
    """⚠️ **La opacidad va en logit y las escalas en logaritmo — y no se supone.**

    El emisor lo declara columna a columna en `scene/appearance.gs.json`. Un lector que
    escriba la convención como constante no falla: devuelve números plausibles y mal, y
    nadie se entera hasta que mira una corona y la ve translúcida.
    """
    datos = _ply([("float", "opacity"), ("float", "scale_0")],
                 {"opacity": [0.0, 2.0], "scale_0": [0.0, np.log(3.0)]})
    esquema = {"columns": [{"name": "opacity", "unit": "logit"},
                           {"name": "scale_0", "unit": "log(mm)"}]}
    col = mm.lee_apariencia(datos, esquema)
    assert np.allclose(col["opacity"], [0.5, 1 / (1 + np.exp(-2.0))])
    assert np.allclose(col["scale_0"], [1.0, 3.0])


def test_un_emisor_que_declare_lineal_no_se_transforma() -> None:
    """La otra mitad de lo mismo: si el descriptor dice lineal, se deja como está."""
    datos = _ply([("float", "opacity"), ("float", "scale_0")],
                 {"opacity": [0.25], "scale_0": [2.0]})
    esquema = {"columns": [{"name": "opacity", "unit": ""},
                           {"name": "scale_0", "unit": "mm"}]}
    col = mm.lee_apariencia(datos, esquema)
    assert col["opacity"][0] == 0.25 and col["scale_0"][0] == 2.0


def test_una_columna_nueva_no_rompe_el_lector() -> None:
    """⚠️ **Este lector ya se rompió dos veces por llevar la lista de propiedades a mano.**

    Una al añadirse `f_rest_*` y otra al añadirse `ao`. El `dtype` sale de lo DECLARADO en
    la cabecera, así que una columna que no conoce la ignora en vez de descolocarse.
    """
    datos = _ply([("float", "x"), ("float", "ao"), ("uchar", "inventada")],
                 {"x": [1.0, 2.0], "ao": [0.5, 0.9], "inventada": [7, 8]})
    col = mm.lee_apariencia(datos, {"columns": []})
    assert np.allclose(col["x"], [1.0, 2.0])
    assert np.allclose(col["ao"], [0.5, 0.9])
    assert np.allclose(col["inventada"], [7, 8])


def test_el_sha256_del_escaner_se_busca_por_id_y_no_por_posicion() -> None:
    """⚠️ **La forma del manifiesto es del emisor, no nuestra.**

    Este script tiene que seguir leyendo contenedores de otras versiones, así que busca
    `asset.ios` recorriendo la estructura en vez de indexar una ruta fija.
    """
    manifiesto = {"visits": [{"assets": [
        {"id": "asset.campo", "sha256": "b" * 64},
        {"id": "asset.ios", "sha256": "a" * 64, "media_type": "model/stl"},
    ]}]}
    assert mm._sha256_declarado(manifiesto) == "a" * 64
    assert mm._sha256_declarado({"visits": []}) is None


def test_una_malla_que_no_es_la_del_caso_se_rechaza(tmp_path, capsys) -> None:
    """⚠️ **Pasó de verdad, con dos ficheros del mismo paciente.**

    La carpeta traía `UpperJawScan.stl` y `Visualization_DigitalModelUnsectioned_18-28.stl`.
    La segunda es la misma boca y es la equivocada: mejorar una malla que no es la que se
    ingirió produce un fichero con colores plausibles puestos sobre otra geometría, y nada
    en el resultado lo delataría. El `sha256` que el manifiesto declara sí.
    """
    import sys
    import zipfile

    uos = tmp_path / "caso.uos"
    with zipfile.ZipFile(uos, "w") as z:
        z.writestr("manifest.json", '{"assets":[{"id":"asset.ios","sha256":"' + "a" * 64 + '"}]}')
        z.writestr("scene/appearance.ply", b"ply\n")
    otra = tmp_path / "otra.stl"
    otra.write_bytes(_stl(np.zeros((1, 3, 3))))

    argv = sys.argv
    sys.argv = ["malla_mejorada.py", "--uos", str(uos), "--malla", str(otra),
                "--salida", str(tmp_path / "out")]
    try:
        assert mm.main() == 1
    finally:
        sys.argv = argv
    assert "no es la malla de este caso" in capsys.readouterr().out


def _malla_de_dos_dientes(n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Una rejilla `n x n` partida por la MITAD, como dos piezas vecinas.

    ⚠️ La frontera tiene que ser corta comparada con el interior, que es lo que hace que un
    diente sea contiguo. El primer intento partía la rejilla a lo largo de toda su
    longitud: entonces casi toda arista cruzaba de una etiqueta a la otra y la prueba de
    contigüidad rechazaba unas etiquetas que estaban bien.
    """
    idx = np.arange(n * n).reshape(n, n)
    caras = np.concatenate([
        np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
    ])
    etq = np.where(np.arange(n * n) % n < n // 2, 11, 12).astype(np.int16)
    return caras, etq


def test_las_etiquetas_solo_se_usan_si_son_de_ESTA_malla() -> None:
    """⚠️ **Un reordenado de vértices no se ve, y ya invalidó una medición entera.**

    `derived/seg_teeth` indexa los vértices «en el mismo orden en que la escena los
    declara», y este script deduplica el STL por su cuenta. Si los dos órdenes no
    coinciden, cada vértice recibe la etiqueta de otro y el fichero sale con los códigos FDI
    barajados **sin que nada falle**. Un diente es contiguo, así que la prueba es barata:
    sobre el caso real el orden correcto da 0,959 de aristas con la misma etiqueta y el
    barajado 0,117.
    """
    caras, etq = _malla_de_dos_dientes()
    meta = {"encoding": {"dtype": "int16-le"}}

    ok = mm.etiquetas_alineadas(etq.tobytes(), meta, len(etq), caras)
    assert ok is not None and (ok > 0).all()

    barajadas = np.random.default_rng(0).permutation(etq)
    assert mm.etiquetas_alineadas(barajadas.tobytes(), meta, len(etq), caras) is None


def test_unas_etiquetas_que_no_cuentan_lo_mismo_se_rechazan() -> None:
    """Un número de etiquetas distinto del de vértices no es un desajuste menor."""
    caras, etq = _malla_de_dos_dientes()
    meta = {"encoding": {"dtype": "int16-le"}}
    assert mm.etiquetas_alineadas(etq[:-3].tobytes(), meta, len(etq), caras) is None
    # Y una codificación que no es la declarada tampoco se interpreta a la ligera.
    assert mm.etiquetas_alineadas(etq.tobytes(), {"encoding": {"dtype": "int32-le"}},
                                  len(etq), caras) is None


def _glb(pos: np.ndarray, prims: list[tuple[np.ndarray, str | None]]) -> bytes:
    """Un GLB minimo: un accessor de POSITION compartido y un `indices` por primitiva."""
    import json as _json

    buf = bytearray()
    vistas, accs, lista = [], [], []
    vistas.append({"buffer": 0, "byteOffset": 0, "byteLength": pos.astype("<f4").nbytes})
    buf += pos.astype("<f4").tobytes()
    accs.append({"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3"})
    for tri, fdi in prims:
        vistas.append({"buffer": 0, "byteOffset": len(buf),
                       "byteLength": tri.astype("<u4").nbytes})
        buf += tri.astype("<u4").tobytes()
        accs.append({"bufferView": len(vistas) - 1, "componentType": 5125,
                     "count": tri.size, "type": "SCALAR"})
        p = {"attributes": {"POSITION": 0}, "indices": len(accs) - 1, "mode": 4}
        if fdi is not None:
            p["extras"] = {"uos_fdi": fdi}
        lista.append(p)
    cab = _json.dumps({"asset": {"version": "2.0"}, "meshes": [{"primitives": lista}],
                       "accessors": accs, "bufferViews": vistas,
                       "buffers": [{"byteLength": len(buf)}]}).encode()
    cab += b" " * (-len(cab) % 4)
    buf += b"\0" * (-len(buf) % 4)
    total = 12 + 8 + len(cab) + 8 + len(buf)
    return (b"glTF" + struct.pack("<II", 2, total)
            + struct.pack("<I", len(cab)) + b"JSON" + cab
            + struct.pack("<I", len(buf)) + b"BIN\0" + bytes(buf))


def test_la_geometria_sale_del_GEMELO_y_no_del_escaner_original() -> None:
    """⚠️ **El encargo dice «regenerar a partir del Digital Twin», no transportar.**

    Un contenedor de perfil ligero NO lleva el escáner original —se declara por su `sha256`
    y se queda fuera— y aun así tiene que poder devolver la malla mejorada.
    `scene/scene.glb` **es** la escena del gemelo, no una copia del fichero de entrada.
    """
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    glb = _glb(pos, [(np.array([[0, 1, 2]]), None), (np.array([[1, 3, 2]]), "26")])
    v, caras, fdi = mm.lee_glb(glb)
    assert len(v) == 4 and caras.shape == (2, 3)
    assert np.allclose(v, pos)
    # El código FDI viaja en la primitiva, que es lo que el borrador define para el picking.
    assert fdi[3] == 26 and fdi[0] == 0


def test_una_primitiva_sin_fdi_no_inventa_codigo() -> None:
    """La encía es una primitiva más y no tiene código: se queda en 0, «sin asignar»."""
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    v, caras, fdi = mm.lee_glb(_glb(pos, [(np.array([[0, 1, 2]]), None)]))
    assert (fdi == 0).all()
