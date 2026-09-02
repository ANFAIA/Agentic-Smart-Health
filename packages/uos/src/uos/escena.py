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

⚠️ **`extras.uos_fdi` NO se emite, y `_REGION_ID` tampoco.** Las dos cosas son el codigo
FDI, y el codigo FDI sale de un segmentador: es Layer 3. Metido aqui quedaria horneado en
una escena de Layer 1, y entonces quitar `derived/` dejaria de quitar la inferencia — que
es justo lo que el §3.1 promete que se puede hacer para distribuir el caso donde el modulo
de IA no esta autorizado. Un contenedor asi sigue llevando salida de modelo, y ningun
«compromiso documentado» sostiene esa afirmacion delante de un auditor.

Esto fue un compromiso durante la 0.4.0 —la escena viajaba partida en un *primitive* por
diente— y la revision externa de la spec (B-1) lo declaro bloqueante. Se revierte: la
escena es Layer 1 pura, un solo *primitive* y un solo indice.

**Lo que se pierde, dicho claro.** Un visor glTF ajeno abre el contenedor, dibuja la arcada
y NO puede seleccionar un diente. El picking del §11.3 exige `derived/`: las etiquetas
viajan ahi, por vertice, y quien las quiera las cruza por indice — que es exacto porque
esta escena conserva el orden de vertices. Para las gaussianas el cruce es el mismo que
las produjo: vertice de corona mas cercano. Es mas trabajo para el lector y es la unica
forma de que la separacion en planos sea verdad.
"""

from __future__ import annotations

import json
import struct
from typing import Any, NamedTuple

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


class SplatsKHR(NamedTuple):
    """Una capa 3DGS lista para `KHR_gaussian_splatting`, con sus unidades YA convertidas.

    ⚠️ **Los arrays llegan aqui en la convencion de la EXTENSION, no en la nuestra.** El
    PLY INRIA guarda la opacidad en logit, las escalas en logaritmo y el cuaternion en
    orden `(w,x,y,z)`; la extension pide opacidad lineal en `[0,1]`, escala lineal no
    negativa y el cuaternion en orden glTF `(x,y,z,w)`. Convertir aqui dentro seria
    esconder tres transformaciones en un constructor: quien las hace es quien ha leido el
    descriptor que las declara, y este modulo solo escribe bytes.

    ⚠️ **No lleva `_REGION_ID`.** El codigo FDI por gaussiana sale del segmentador (por
    vecino mas cercano, ver `apariencia.py`), asi que es Layer 3 y no puede viajar en un
    asset de Layer 1 — misma regla que `extras.uos_fdi`, ver el docstring del modulo. Un
    visor conforme dibuja la arcada y no puede seleccionar un diente sin `derived/`.
    """

    posiciones: np.ndarray          # (n,3) float32, mm, en el marco canonico
    rotacion: np.ndarray            # (n,4) float32, cuaternion unitario, orden glTF
    escala: np.ndarray              # (n,3) float32, LINEAL y no negativa
    opacidad: np.ndarray            # (n,)  float32, lineal en [0,1]
    sh0: np.ndarray                 # (n,3) float32, coeficiente DC
    sh1: np.ndarray | None = None   # (n,3,3) float32, grado 1 (tres coeficientes VEC3)
    # ⚠️ **La oclusion ambiental NO es de la extension y sin ella la arcada se ve PLANA.**
    # Es un factor de visualizacion en [0,1] que quien dibuja multiplica por el color;
    # esta fuera de `f_dc` a proposito, porque una lectura de tono no debe oscurecerse
    # porque la pieza tenga una fisura al lado. Pero si no viaja, el visor no puede
    # aplicarla — y la primera version de esta primitiva la dejo fuera: la apariencia se
    # dibujaba sin sombreado y no habia nada en el fichero que dijera que faltaba.
    ao: np.ndarray | None = None          # (n,) float32 en [0,1]
    # Las normales del vertice mas cercano. Van con el semantico ESTANDAR `NORMAL`, no con
    # guion bajo: glTF ya lo define y una primitiva de puntos puede llevarlo.
    normales: np.ndarray | None = None    # (n,3) float32, unitarias
    nombre: str = "apariencia"
    # ⚠️ Obligatorio en la extension y NO tiene defecto razonable: dice si el color esta
    # en sRGB o en lineal, y equivocarse cambia el tono de toda la arcada. El nuestro sale
    # del entrenamiento contra renders sRGB.
    color_space: str = "srgb_rec709_display"
    extras: dict | None = None


def _mesh_splats(
    gs: SplatsKHR, _anade: Any, accesos: list[dict],
) -> dict:
    """El *mesh* con la primitiva `KHR_gaussian_splatting`. Modo POINTS, como exige."""
    n = len(gs.posiciones)
    for nombre, arr in (
        ("posiciones", gs.posiciones), ("rotacion", gs.rotacion),
        ("escala", gs.escala), ("opacidad", gs.opacidad), ("sh0", gs.sh0),
    ):
        if len(arr) != n:
            raise ValueError(
                f"la capa de apariencia trae {n} posiciones y {len(arr)} en `{nombre}`"
            )
    if float(np.min(gs.escala)) < 0:
        raise ValueError(
            "`KHR_gaussian_splatting:SCALE` DEBE ser no negativa y llega con valores "
            "negativos: parece una escala en logaritmo sin exponenciar"
        )
    if float(np.max(gs.opacidad)) > 1.0 or float(np.min(gs.opacidad)) < 0.0:
        raise ValueError(
            "`KHR_gaussian_splatting:OPACITY` DEBE ir en [0,1] y llega fuera de rango: "
            "parece una opacidad en logit sin pasar por la sigmoide"
        )

    def _vec(datos: np.ndarray, tipo: str, con_extremos: bool = False) -> int:
        a = np.ascontiguousarray(datos, dtype=np.float32)
        acc: dict = {
            "bufferView": _anade(a, _ARRAY_BUFFER), "componentType": _FLOAT,
            "count": n, "type": tipo,
        }
        if con_extremos:
            acc["min"] = [float(x) for x in a.reshape(n, -1).min(axis=0)]
            acc["max"] = [float(x) for x in a.reshape(n, -1).max(axis=0)]
        accesos.append(acc)
        return len(accesos) - 1

    atributos = {
        "POSITION": _vec(gs.posiciones, "VEC3", con_extremos=True),
        "KHR_gaussian_splatting:ROTATION": _vec(gs.rotacion, "VEC4"),
        "KHR_gaussian_splatting:SCALE": _vec(gs.escala, "VEC3"),
        "KHR_gaussian_splatting:OPACITY": _vec(gs.opacidad.reshape(n), "SCALAR"),
        "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": _vec(gs.sh0, "VEC3"),
    }
    # ⚠️ El grado 1 es OPCIONAL, y si va tiene que ir entero: la extension exige que si se
    # usa un grado superior esten todos los inferiores. Son tres coeficientes VEC3.
    if gs.sh1 is not None:
        for k in range(3):
            atributos[f"KHR_gaussian_splatting:SH_DEGREE_1_COEF_{k}"] = _vec(
                gs.sh1[:, k, :], "VEC3"
            )
    if gs.normales is not None:
        atributos["NORMAL"] = _vec(gs.normales, "VEC3")
    # ⚠️ Atributos de APLICACION, con guion bajo, que es lo que glTF reserva para lo que su
    # especificacion no define. Un visor conforme los ignora y dibuja igual; el nuestro los
    # usa para el sombreado y para seleccionar una pieza. Que no sean estandar es
    # exactamente por lo que el sidecar `ash_gs_measured` sigue haciendo falta.
    if gs.ao is not None:
        atributos["_AO"] = _vec(np.asarray(gs.ao).reshape(n), "SCALAR")
    return {
        "name": gs.nombre,
        "primitives": [{
            "attributes": atributos,
            # POINTS. La extension lo exige: cada gaussiana es un punto, y la elipse la
            # levanta el rasterizador a partir de la escala y la rotacion.
            "mode": 0,
            "extensions": {
                "KHR_gaussian_splatting": {
                    "kernel": "ellipse",
                    "colorSpace": gs.color_space,
                },
            },
            **({"extras": gs.extras} if gs.extras else {}),
        }],
    }


def construye_glb(
    posiciones: np.ndarray,
    caras: np.ndarray,
    normales: np.ndarray | None = None,
    *,
    nombre: str = "scan",
    generador: str = "agentic-smart-health",
    extras: dict | None = None,
    nodos_gs: list[NodoGS] | None = None,
    splats: SplatsKHR | None = None,
) -> bytes:
    """La escena en un solo `bytes`. Indexada: el orden de vertices se conserva.

    Que se conserve no es un detalle de eficiencia: es lo que permite que
    `derived/seg_teeth` sea una lista de codigos indexada por vertice y que el visor los
    case sin nada mas. Reordenar aqui romperia esa union en silencio.

    ⚠️ **Un solo *primitive*, y NO acepta etiquetas.** La malla no se parte por diente:
    partirla exige el codigo FDI, que sale del segmentador, y hornear Layer 3 en una
    escena de Layer 1 rompe la removibilidad de `derived/` (B-1, ver el docstring del
    modulo). Quien quiera geometria por pieza cruza `derived/seg_teeth` por indice.
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

    # Un unico primitive y un unico index buffer: Layer 1 pura (B-1).
    accesos.append({
        "bufferView": _anade(idx, _ELEMENT_ARRAY_BUFFER), "componentType": _UINT32,
        "count": int(idx.size), "type": "SCALAR",
    })
    mallas: list[dict] = [{
        "name": nombre,
        "primitives": [{"attributes": atributos, "indices": len(accesos) - 1, "mode": 4}],
    }]

    # El nodo 0 es la malla y ES el marco canonico (§5.1). Los GS van de hijos suyos.
    nodos: list[dict] = [{"mesh": 0, "name": nombre}]

    # ⚠️ **La apariencia va DENTRO del glTF, que es lo que el §5.1 pedia desde el
    # principio.** Iba como `.ply` aparte con un `extras.uos_gs_uri` apuntandolo — el
    # fallback que el borrador admite mientras `KHR_gaussian_splatting` no este ratificada.
    # El precio de ese fallback es que solo lo lee quien conozca la convencion: un visor
    # glTF cualquiera abria la escena, veia un puntero a un fichero opaco y no dibujaba
    # nada. Con la extension, la capa la renderiza cualquier implementacion conforme sin
    # saber nada de UOS, que es la razon de existir de un formato abierto.
    #
    # Y va como HIJO del nodo de la malla, no suelto: el §5.1 dice que la registracion
    # GS->malla se codifica como transformada de ese nodo. Aqui la apariencia se entreno
    # en el marco del escaner, asi que la transformada es la identidad y se omite — pero
    # la JERARQUIA queda dicha, que es lo que permite que otro emisor ponga ahi la suya.
    if splats is not None:
        mallas.append(_mesh_splats(splats, _anade, accesos))
        nodos.append({"mesh": len(mallas) - 1, "name": splats.nombre})
        nodos[0].setdefault("children", []).append(len(nodos) - 1)
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
        "meshes": mallas,
        "accessors": accesos,
        "bufferViews": vistas,
        "buffers": [{"byteLength": desplazamiento}],
    }
    # ⚠️ `extensionsUsed` y NUNCA `extensionsRequired`. Es la misma regla que aplicamos al
    # manifiesto: un lector que no entienda la extension tiene que poder abrir la escena y
    # ver la malla. Ponerla en `required` haria que un visor conforme se negara a abrir un
    # caso cuya geometria puede enseñar perfectamente.
    if splats is not None:
        gltf["extensionsUsed"] = ["KHR_gaussian_splatting"]
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
