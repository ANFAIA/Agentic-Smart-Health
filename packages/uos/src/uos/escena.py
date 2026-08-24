"""`scene/scene.glb` — la malla como escena glTF binaria (§3.1, §5.1).

**Por que convertir y no meter el STL tal cual.** Un STL es una sopa de triangulos: no
tiene indices, ni atributos, ni un solo campo de metadatos. El §11.3 hace el picking por
`raycast al mesh -> extras.uos_fdi`, o sea que la escena tiene que poder **llevar cosas
colgadas**, y un STL no puede llevar ninguna. glTF si.

⚠️ **La conversion es con perdida, y por eso el STL original SIGUE viajando.** glTF exige
`float32` y nuestras posiciones son `float64`: a ±100 mm eso deja unos 10 nanometros de
resolucion, irrelevante para mirar y no para afirmar reversibilidad. El §1.1 dice que UOS
no re-encodea datos fuente y el §3.1 dibuja la malla como «STL convertido»; las dos cosas
solo son compatibles si el convertido es **presentacion** y el original se queda. Asi que
el contenedor lleva los dos: `asset.ios` byte-identico y `asset.scene` para mirar.

**Se construye desde la malla INGERIDA, no desde el fichero STL.** El `mesh-agent` guarda
vertices deduplicados con sus caras, y las etiquetas FDI indexan ESE orden. Convertir el
STL cara a cara daria una nube tres veces mas grande y sin forma de casar las etiquetas.

**Los nodos GS cuelgan de la malla, y su `matrix` ES la registracion** (§5.1). El nodo raiz
de la malla es el marco canonico; cada capa de gaussianas va debajo con la transformada que
la lleva alli. Asi un `GLTFLoader` cualquiera coloca las gaussianas en su sitio sin leer el
manifiesto — y el manifiesto sigue siendo la version auditable de esa misma relacion.

⚠️ **`extras.uos_gs_uri` es el fallback declarado**, no un invento: `KHR_gaussian_splatting`
es release candidate de Khronos y el §13 dice que en v1.0, tras ratificarse, este camino se
retira. Hasta entonces el payload va como fichero aparte DENTRO del contenedor y el nodo
apunta a el.

⚠️ **`extras.uos_fdi` NO se emite**, aunque el §5.1 lo contemple «si el mesh viene
segmentado». El nuestro no viene segmentado: lo segmentamos con un modelo, y eso es Layer 3.
Horneado aqui, quitar `derived/` dejaria de quitar la inferencia y se rompe la regla dura
del §5.5. Las etiquetas van en `derived/`, indexadas por vertice.
"""

from __future__ import annotations

import json
import struct
from typing import NamedTuple

import numpy as np

MEDIA_GLB = "model/gltf-binary"

_MAGIA = 0x46546C67          # 'glTF'
_JSON = 0x4E4F534A           # 'JSON'
_BIN = 0x004E4942            # 'BIN\0'

_FLOAT = 5126
_UINT32 = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


def _alinea(b: bytes, relleno: bytes = b"\x00") -> bytes:
    """glTF exige que cada trozo y cada `bufferView` empiecen en multiplo de 4."""
    return b + relleno * (-len(b) % 4)


class NodoGS(NamedTuple):
    """Una capa de gaussianas colgada de la malla.

    `matriz_fila` es la transformada 4x4 en orden de FILAS —como la escribe el manifiesto—
    y aqui se traspone: glTF guarda las matrices por COLUMNAS. Confundirlas no revienta,
    coloca la nube girada y espejada, que es peor.
    """

    uri: str
    nombre: str
    matriz_fila: list[float] | None = None
    extras: dict | None = None


def construye_glb(
    posiciones: np.ndarray,
    caras: np.ndarray,
    normales: np.ndarray | None = None,
    *,
    nombre: str = "scan",
    generador: str = "agentic-smart-health",
    extras: dict | None = None,
    nodos_gs: list[NodoGS] | None = None,
) -> bytes:
    """La escena en un solo `bytes`. Indexada: el orden de vertices se conserva.

    Que se conserve no es un detalle de eficiencia: es lo que permite que
    `derived/seg_teeth` sea una lista de codigos indexada por vertice y que el visor los
    case sin nada mas. Reordenar aqui romperia esa union en silencio.
    """
    pos = np.ascontiguousarray(posiciones, dtype=np.float32)
    idx = np.ascontiguousarray(caras, dtype=np.uint32).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"posiciones tiene forma {pos.shape}; se esperaba (n, 3)")
    if idx.size and int(idx.max()) >= len(pos):
        raise ValueError(
            f"una cara referencia el vertice {int(idx.max())} y solo hay {len(pos)}: "
            "las caras y las posiciones no son de la misma malla"
        )

    trozos: list[bytes] = []
    vistas: list[dict] = []
    accesos: list[dict] = []
    desplazamiento = 0

    def _anade(datos: np.ndarray, objetivo: int) -> int:
        nonlocal desplazamiento
        crudo = _alinea(datos.tobytes())
        vistas.append({
            "buffer": 0, "byteOffset": desplazamiento,
            "byteLength": int(datos.nbytes), "target": objetivo,
        })
        trozos.append(crudo)
        desplazamiento += len(crudo)
        return len(vistas) - 1

    # POSITION. `min`/`max` son OBLIGATORIOS en glTF para este accesor: es lo que deja a un
    # visor calcular el encuadre sin leer el buffer entero.
    accesos.append({
        "bufferView": _anade(pos, _ARRAY_BUFFER), "componentType": _FLOAT,
        "count": int(len(pos)), "type": "VEC3",
        "min": [float(x) for x in pos.min(axis=0)],
        "max": [float(x) for x in pos.max(axis=0)],
    })
    atributos = {"POSITION": 0}

    if normales is not None and len(normales) == len(pos):
        nor = np.ascontiguousarray(normales, dtype=np.float32)
        # glTF exige normales UNITARIAS. Las del escaner lo son, pero normalizar aqui es
        # barato y evita que una malla de otro emisor se renderice con la luz al reves.
        largo = np.linalg.norm(nor, axis=1, keepdims=True)
        nor = np.divide(nor, largo, out=np.zeros_like(nor), where=largo > 0)
        accesos.append({
            "bufferView": _anade(nor, _ARRAY_BUFFER), "componentType": _FLOAT,
            "count": int(len(nor)), "type": "VEC3",
        })
        atributos["NORMAL"] = len(accesos) - 1

    accesos.append({
        "bufferView": _anade(idx, _ELEMENT_ARRAY_BUFFER), "componentType": _UINT32,
        "count": int(idx.size), "type": "SCALAR",
    })

    # El nodo 0 es la malla y ES el marco canonico (§5.1). Los GS van de hijos suyos.
    nodos: list[dict] = [{"mesh": 0, "name": nombre}]
    for n in nodos_gs or []:
        nodo: dict = {
            "name": n.nombre,
            "extras": {"uos_gs_uri": n.uri, **(n.extras or {})},
        }
        if n.matriz_fila is not None:
            if len(n.matriz_fila) != 16:
                raise ValueError(
                    f"la matriz del nodo {n.nombre} trae {len(n.matriz_fila)} numeros y "
                    "una transformada 4x4 son 16"
                )
            m = np.asarray(n.matriz_fila, dtype=np.float64).reshape(4, 4)
            # Fila -> columna. glTF lo exige y es el error clasico de este sitio.
            nodo["matrix"] = [float(x) for x in m.T.ravel()]
        nodos.append(nodo)
        nodos[0].setdefault("children", []).append(len(nodos) - 1)

    gltf = {
        "asset": {"version": "2.0", "generator": generador},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodos,
        "meshes": [{
            "name": nombre,
            "primitives": [{
                "attributes": atributos,
                "indices": len(accesos) - 1,
                "mode": 4,   # TRIANGLES
            }],
        }],
        "accessors": accesos,
        "bufferViews": vistas,
        "buffers": [{"byteLength": desplazamiento}],
    }
    if extras:
        gltf["extras"] = extras

    binario = b"".join(trozos)
    cabecera_json = _alinea(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(cabecera_json) + 8 + len(binario)
    return b"".join([
        struct.pack("<III", _MAGIA, 2, total),
        struct.pack("<II", len(cabecera_json), _JSON), cabecera_json,
        struct.pack("<II", len(binario), _BIN), binario,
    ])


def lee_stl_binario(crudo: bytes) -> tuple[np.ndarray, np.ndarray]:
    """`(posiciones, caras)` de un STL binario. Sopa de triangulos, sin deduplicar.

    Es el camino de RESPALDO: se usa cuando el llamante aporta el fichero del escaner pero
    no la malla ingerida —por ejemplo alguien que use este agente fuera del pipeline—. Da
    tres veces mas vertices, y sobre todo **no se le pueden colgar las etiquetas FDI**,
    porque estas indexan el orden deduplicado que guarda el `mesh-agent`. Cuando esa malla
    esta, se usa aquella.
    """
    if len(crudo) < 84:
        raise ValueError("el fichero es mas corto que una cabecera de STL binario")
    caras = int.from_bytes(crudo[80:84], "little")
    if 84 + caras * 50 != len(crudo):
        raise ValueError(
            f"el STL declara {caras} caras, que son {84 + caras * 50} bytes, y trae "
            f"{len(crudo)}. O esta truncado o es un STL de texto, que esto no lee."
        )
    tri = np.frombuffer(crudo, dtype=np.dtype([
        ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2"),
    ]), count=caras, offset=84)
    pos = np.asarray(tri["v"], dtype=np.float64).reshape(-1, 3)
    return pos, np.arange(len(pos), dtype=np.int32).reshape(-1, 3)
