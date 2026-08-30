"""La prueba de plausibilidad anatomica, probada a su vez sobre piezas fabricadas.

Una prueba que solo descarta no vale nada si no se comprueba que **puede** descartar y
que **puede** no hacerlo. Aqui se fabrican tres bocas de juguete: una con coronas del
tamano que les toca, otra con una corona inflada, y otra con dos contralaterales de areas
descaradamente distintas. Si la primera saltara, la prueba seria un adorno que dice que
todo esta mal; si las otras dos no saltaran, seria un adorno que dice que todo esta bien.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import numpy as np

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "mide_segmentacion.py"
_spec = importlib.util.spec_from_file_location("mide_segmentacion", _RUTA)
assert _spec and _spec.loader
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


def _caja(centro, lados, n=8):
    """Una rejilla de puntos dentro de una caja, con su triangulacion."""
    ejes = [np.linspace(-lado / 2, lado / 2, n) for lado in lados]
    g = np.stack(np.meshgrid(*ejes, indexing="ij"), axis=-1).reshape(-1, 3)
    return g + np.asarray(centro, dtype=np.float64)


def _glb(pos: np.ndarray, tri: np.ndarray) -> bytes:
    """Un GLB minimo con una malla de una primitiva: POSITION float32 e indices uint32."""
    p = np.ascontiguousarray(pos, dtype="<f4").tobytes()
    i = np.ascontiguousarray(tri.ravel(), dtype="<u4").tobytes()
    cuerpo = p + i
    cuerpo += b"\x00" * (-len(cuerpo) % 4)
    cab = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(cuerpo)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(p)},
            {"buffer": 0, "byteOffset": len(p), "byteLength": len(i)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": tri.size, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    }
    j = json.dumps(cab).encode("utf-8")
    j += b" " * (-len(j) % 4)
    total = 12 + 8 + len(j) + 8 + len(cuerpo)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(j), 0x4E4F534A) + j
        + struct.pack("<II", len(cuerpo), 0x004E4942) + cuerpo
    )


def _boca(tmp: Path, piezas: dict[int, tuple[tuple, tuple]]) -> Path:
    """Un `.uos` de juguete: `{fdi: (centro, lados)}`, cada pieza una caja etiquetada."""
    pos, etq = [], []
    for fdi, (centro, lados) in piezas.items():
        p = _caja(centro, lados)
        pos.append(p)
        etq.append(np.full(len(p), fdi, dtype="<i2"))
    pos = np.concatenate(pos)
    etq = np.concatenate(etq)
    # Triangulos sueltos sobre tripletas consecutivas: el area no importa en si, importa
    # que cada cara tenga las tres etiquetas iguales para que cuente en su pieza.
    n = (len(pos) // 3) * 3
    tri = np.arange(n, dtype=np.int64).reshape(-1, 3)

    destino = tmp / "juguete.uos"
    with zipfile.ZipFile(destino, "w") as z:
        z.writestr("scene/scene.glb", _glb(pos, tri))
        z.writestr("derived/seg_teeth.bin", etq.tobytes())
    return destino


# Cajas del tamano que Wheeler da para cada tipo: tienen que pasar.
SANAS = {
    11: ((0, 0, 0), (8.5, 7.0, 10.5)),
    21: ((12, 0, 0), (8.5, 7.0, 10.5)),
    16: ((30, 0, 0), (10.0, 11.0, 7.5)),
    26: ((45, 0, 0), (10.0, 11.0, 7.5)),
}


def test_una_boca_con_coronas_del_tamano_que_les_toca_no_descarta_ninguna(tmp_path):
    r = ms.mide(_boca(tmp_path, SANAS))
    assert r["n_descartadas"] == 0, r["descartadas"]
    assert r["cota_superior_correctas_pct"] == 100.0


def test_una_corona_del_DOBLE_de_su_tipo_se_descarta(tmp_path):
    piezas = dict(SANAS)
    piezas[16] = ((30, 0, 0), (20.0, 22.0, 15.0))
    r = ms.mide(_boca(tmp_path, piezas))
    assert 16 in r["descartadas"]
    assert next(p for p in r["piezas"] if p["fdi"] == 16)["razon"] > ms.TOLERANCIA


def test_dos_contralaterales_de_areas_muy_distintas_se_descartan_LAS_DOS(tmp_path):
    """Cuando el espejo falla no se sabe cual de las dos sobra, asi que caen las dos.

    Es deliberado: elegir una seria inventarse de que lado esta el error.
    """
    piezas = dict(SANAS)
    # Misma caja, mucha menos superficie triangulada: el area cae sin tocar el tamano.
    piezas[26] = ((45, 0, 0), (10.0, 11.0, 0.4))
    r = ms.mide(_boca(tmp_path, piezas))
    espejo = next(e for e in r["espejos"] if e["par"] == [16, 26])
    assert espejo["descartado"], espejo
    assert {16, 26} <= set(r["descartadas"])


def test_la_cota_es_SUPERIOR_no_una_nota(tmp_path):
    """Una corona del tamano correcto con el nombre del vecino pasa la prueba.

    Se fija por escrito porque es el limite de lo que esto puede decir: sin verdad de
    campo, un nombre equivocado sobre una geometria plausible es indistinguible de un
    acierto, y el numero que sale de aqui **no** es acierto.
    """
    piezas = dict(SANAS)
    # El 11 y el 21 intercambiados: los dos siguen midiendo lo que debe un incisivo.
    piezas[11], piezas[21] = SANAS[21], SANAS[11]
    r = ms.mide(_boca(tmp_path, piezas))
    assert r["n_descartadas"] == 0
    assert r["cota_superior_correctas_pct"] == 100.0


def _valle(tooth_desde: float, nx: int = 64, paso: float = 0.5):
    """Una loseta plana con UN pliegue recto en `y = 0`, y las etiquetas donde se pida.

    `z = 0.4·|y|` es un valle: la unica concavidad de toda la malla esta en `y = 0`. Con
    `tooth_desde = 0` la frontera cae dentro del pliegue —que es lo que hace un anotador—;
    con `tooth_desde = 3` cae sobre la parte lisa, que es lo que hace nuestro modelo. La
    misma geometria y dos fronteras: lo unico que cambia es lo que se quiere medir.
    """
    ys = np.arange(-10.0, 10.0 + paso, paso)
    xs = np.arange(nx) * paso
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X, Y, 0.4 * np.abs(Y)], axis=-1).reshape(-1, 3)
    ny = len(ys)
    tri = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = i * ny + j, i * ny + j + 1
            c, d = (i + 1) * ny + j, (i + 1) * ny + j + 1
            tri += [(a, c, b), (b, c, d)]
    etq = np.where(pos[:, 1] >= tooth_desde, 11, 0).astype(np.int64)
    return pos, np.asarray(tri, dtype=np.int64), etq


def test_una_frontera_EN_el_pliegue_da_una_razon_alta():
    """El caso del anotador: el borde cae donde la superficie se dobla."""
    pos, tri, etq = _valle(0.0)
    razon = ms.razon_de_concavidad(pos, tri, etq)
    assert razon is not None
    assert razon > ms.RAZON_CONCAVIDAD_MINIMA


def test_la_misma_malla_con_la_frontera_CORRIDA_da_una_razon_de_cero():
    """⚠️ **Y esto es lo que ninguna de las otras dos pruebas ve.**

    La geometria es identica y las piezas siguen midiendo lo que median; lo unico que ha
    cambiado es que el borde se ha ido tres milimetros del pliegue, a un sitio liso. La
    diagonal contra el tipo y la simetria contralateral dan exactamente lo mismo.
    """
    pos, tri, etq = _valle(3.0)
    razon = ms.razon_de_concavidad(pos, tri, etq)
    assert razon is not None
    assert abs(razon) < ms.RAZON_CONCAVIDAD_MINIMA


def test_sin_frontera_diente_encia_no_se_inventa_una_cifra():
    """`None` y no cero: no medir y medir cero son cosas distintas."""
    pos, tri, _ = _valle(0.0)
    todo_diente = np.full(len(pos), 11, dtype=np.int64)
    assert ms.razon_de_concavidad(pos, tri, todo_diente) is None


def test_la_boca_de_cubos_no_declara_frontera(tmp_path):
    """Las bocas sinteticas de arriba no traen encia, asi que la prueba se abstiene."""
    caso = _boca(tmp_path, {11: ((0, 0, 0), (8.5, 7.0, 10.5))})
    r = ms.mide(caso)
    assert r["razon_concavidad"] is None
    assert r["frontera_en_el_pliegue"] is None
