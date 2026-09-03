"""Reconstruccion de apariencia real desde fotos intraorales con 3D Gaussian Splatting.

Este modulo implementa el flujo completo demostrado en `notebooks/07`:

    mesh-agent (STL) + image-agent (fotos)
        → Blender (EEVEE, N vistas con pose exacta)
        → gsplat (optimizacion contra renders, 0.8*L1 + 0.2*(1-SSIM))
        → PLY en formato INRIA con un degradado de dos tonos tomados de las fotos

**Que NO es.** No es el ajuste de densidad del CBCT (ese es `ajuste.py`). Aqui se
optimiza contra **imagenes 2D** (renders de Blender) porque el objetivo es la
**apariencia visual**, no la densidad fisica. La representacion es la misma —gaussianas
anisotropas— pero la funcion de perdida, los datos de entrada y la semantica del
resultado son distintos.

**Perfil `'histora-gs-apariencia/1.0'`.** Declara que el campo fue entrenado con gsplat
sobre renders EEVEE de una malla pintada con dos tonos de las fotos. Un lector que
vea este perfil sabe que:

  - Las posiciones NO corresponden 1:1 con vertices del escaneo (el optimizador las mueve).
  - El color es REAL (del paciente), no falso por codigo FDI.
  - La escala esta en **logaritmo** (convencion INRIA), no en mm lineales.
  - La opacidad es de **visualizacion**, no atenuacion radiologica.
  - Las normales son cero (un campo no es una malla).

**Dependencias.** `torch` y `gsplat` se importan dentro de las funciones, no al nivel
del modulo: este paquete se puede importar sin CUDA. `blender` se ejecuta como
subprocesso —debe estar en el PATH— con el script `scripts/blender_render_views.py`.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constantes medidas (ver notebook 07 y entrena_gs_escaner.py)
# ---------------------------------------------------------------------------

# Perfil de este modulo. Distingue este campo del semilla (`histora-twin/1.0`) y del
# ajustado (`histora-twin-ajustado/1.0`).
PERFIL = "histora-gs-apariencia/1.0"

#: A mas de esto de cualquier vertice del escaneo, una gaussiana NO hereda etiqueta. Dos
#: milimetros es el orden del grosor de la encia adherida: mas alla, el vecino mas cercano
#: ya no dice nada sobre donde esta esa gaussiana.
LIMITE_VECINO = 2.0

#: Cuantos vertices vota la etiqueta de cada gaussiana. Ver `_fdi_por_gaussiana`.
#:
#: ⚠️ **Uno solo no basta, y este proyecto ya lo sabia.** Sobre la malla corre
#: `afina_fronteras`, que reasigna por mayoria justo porque «el contacto interproximal no
#: tiene borde geometrico»: en un caso real movio 3.559 vertices. Las gaussianas heredaban
#: con 1-NN y se saltaban esa correccion, reintroduciendo el ruido que la malla ya habia
#: quitado.
#:
#: Medido sobre el caso real: la pureza del vecindario de cada gaussiana —cuantos de sus
#: 16 vecinos llevan su misma etiqueta— pasa de **85,1 % con 1-NN a 91,4 % con 16**, y 32
#: solo aniade 0,6 puntos mas. Cambia el 3,6 % de las etiquetas.
VOTOS_FDI = 16

#: Lo que una `uri` de unidades puede valer en la cabecera de un PLY de apariencia.
UNIDADES_MM = "mm"
UNIDADES_NORMALIZADO = "normalizado"


@dataclass(frozen=True)
class CampoEscrito:
    """Lo que `escribe_inria` ACABA de escribir, para que nadie tenga que suponerlo.

    ⚠️ Existe por un fallo concreto: el descriptor `.gs.json` del contenedor afirmaba
    `units: "mm"` con un literal, y el PLY estaba en el espacio normalizado de Blender.
    Las dos cosas se escriben en sitios distintos y una creia por la otra, asi que no habia
    forma de que la contradiccion fallara — solo de que se viera, y solo si alguien
    comparaba la nube con la malla.

    Es el mismo patron que ya mordio dos veces en este repositorio: un campo de contrato
    relleno con un literal en vez de con el dato acaba mintiendo. La regla que sale de ahi
    es que **quien escribe declara, y quien describe pregunta**.
    """

    n_primitivas: int
    unidades: str
    des_normalizado: bool

# Numero por defecto de vistas Blender. 1600 da ~31.5 dB en 6000 iteraciones sobre
# un escaneo intraoral tipico (medido en notebook 07, caso F1980).
N_VISTAS = 1_600

# Resolucion en pixeles por lado de cada vista. 1024 da buena calidad sin explotar RAM
# (1600 vistas a 1024 son ~20 GB en PNG; se cargan bajo demanda).
RESOLUCION = 1_024

# Iteraciones de entrenamiento gsplat. 6000 es el optimo medido: el PSNR se estanca
# mas alla (notebook 7: 31.5 dB @ 6000 vs 31.9 dB @ 6000 con 1600 vistas).
ITERACIONES = 6_000

# Semillas iniciales: gaussianas sembradas desde la superficie del mesh. 150K es un
# buen compromiso calidad/tiempo para un escaneo intraoral (notebook 07).
SEMILLAS = 150_000

# Muestras de validacion para PSNR holdout.
MUESTRAS_VALIDACION = 24


# ---------------------------------------------------------------------------
# Resultado del entrenamiento
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntrenamientoApariencia:
    """Metricas del entrenamiento 3DGS (medidas, no inferidas).

    `psnr_db` y `ssim` se miden sobre vistas de validacion que el modelo NUNCA
    vio durante el entrenamiento. Son la metrica honesta de calidad.
    """

    psnr_db: float
    ssim: float
    n_gaussianas: int
    iteraciones: int
    loss_curva: list[tuple[int, float, float, float]]
    tiempo_s: float
    n_vistas_train: int
    n_vistas_holdout: int
    resolucion: int
    perfil: str = PERFIL
    # ⚠️ **El color medido por pieza sale de aqui para poder viajar como DATO.** El campo
    # gaussiano lo lleva como pixeles y eso basta para pintar, pero no para preguntar «de
    # que color es el 26» sin abrir un PLY de 8 MB y buscar sus gaussianas. Con soporte
    # REGIONAL cabe en `ClinicalAttributes.color` y sale en `clinical/observations.json`,
    # que es donde un clinico lo busca.
    tonos: list = field(default_factory=list)
    """`TonoPieza` por corona medida. Vacio si no se pudo medir color por pieza."""

    # ⚠️ **Salen de aqui porque un aviso impreso no es un aviso.** Estos motivos —una foto
    # cuya lateralidad nadie resolvio, una pieza que ninguna foto ve— se escribian en el
    # log del pipeline y se quedaban ahi: el `.uos` se exportaba sin ellos, y quien lo
    # abriera no tenia forma de saber que dos fotos se descartaron por ambiguas. El gate
    # de revision humana es donde vive esa clase de cosa.
    motivos: list[str] = field(default_factory=list)
    """Motivos de revision del color por pieza. Vacios si no hubo ninguno."""

    def como_artefacto(self) -> dict[str, np.ndarray]:
        """Convierte a formato store: arrays float32 + metadatos."""
        return {
            "psnr_db": np.asarray(self.psnr_db),
            "ssim": np.asarray(self.ssim),
            "n_gaussianas": np.asarray(self.n_gaussianas),
            "iteraciones": np.asarray(self.iteraciones),
            "tiempo_s": np.asarray(self.tiempo_s),
            "n_vistas_train": np.asarray(self.n_vistas_train),
            "n_vistas_holdout": np.asarray(self.n_vistas_holdout),
            "resolucion": np.asarray(self.resolucion),
        }


# ---------------------------------------------------------------------------
# Color desde fotos (patron del notebook 07, celda 1c)
# ---------------------------------------------------------------------------

def _muestra_color_fotos(rutas_fotos: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Extrae el color medio de esmalte y encia de las fotos intraorales.

    Devuelve (enamel_rgb, gingiva_rgb) como arrays float64 en [0, 255].
    El patron es el del notebook 07: pixels claros-calidos = esmalte, rosa/rojo = encia.
    """
    from PIL import Image

    dientes, encias = [], []
    for ruta in rutas_fotos:
        img = np.asarray(Image.open(ruta), dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        val = img.max(-1)
        red = R - 0.5 * (G + B)
        # Esmalte: claro, calido, R >= G >= B
        dientes.append(img[(val > 160) & (red > 4) & (red < 28) & (R >= G) & (G >= B - 3)])
        # Encia: rosa/rojo, brillo medio
        encias.append(img[(red > 40) & (val > 70) & (val < 230)])

    if not dientes or not encias:
        raise ValueError(
            "No se pudo muestrear color de las fotos: asegurate de que hay fotos "
            "intraorales con esmalte visible (claro, calido) y encia (rosa/rojo)."
        )

    enamel = np.median(np.concatenate(dientes), axis=0)
    gingiva = np.median(np.concatenate(encias), axis=0)
    return enamel, gingiva


def _colorea_malla(
    posiciones: np.ndarray, normales: np.ndarray,
    enamel_rgb: np.ndarray, gingiva_rgb: np.ndarray,
    etiquetas: np.ndarray | None = None,
) -> np.ndarray:
    """Esmalte donde la segmentacion dice diente, encia donde dice que no.

    ⚠️ **El limite sale de las etiquetas FDI, no de la altura.** Antes era un degradado
    entre los percentiles 30 y 70 de `z`, y eso **afirmaba el margen gingival por un numero
    inventado** — justo la frontera que este proyecto tiene medido que no sabe determinar.

    Con las etiquetas el color sigue sin ser medido —son dos tonos muestreados de las
    fotos— pero su frontera **tiene procedencia**: es la salida del segmentador, que viaja
    en `derived/` marcada como Layer 3, con el hash de sus pesos y su fiabilidad publicada
    (11 de 14 piezas se descartan por anatomia; cota superior 21 %). Cambia «el limite lo
    pusimos por altura» por «el limite lo puso un modelo, y dice cual y cuanto acierta».

    Y es la MISMA etiqueta que `region_id` pone en cada gaussiana para poder encender una
    pieza: una sola fuente de verdad en vez de dos criterios que pueden discrepar.

    Sin etiquetas se cae al degradado por altura, declarado como el apano que es.

    Devuelve (N, 3) uint8.
    """
    if etiquetas is not None and len(etiquetas) == len(posiciones):
        diente = np.asarray(etiquetas).reshape(-1) > 0
        vcol = np.where(diente[:, None], enamel_rgb, gingiva_rgb)
        return vcol.astype(np.uint8)

    z = posiciones[:, 2]
    p30, p70 = np.percentile(z, [30, 70])
    w = np.clip((z - p30) / (p70 - p30), 0, 1)[:, None]
    vcol = (w * enamel_rgb + (1 - w) * gingiva_rgb).astype(np.uint8)
    return vcol


def _normales_de_malla(pos: np.ndarray, caras: np.ndarray) -> np.ndarray:
    """Normal por vertice, ponderada por area (el area sale del modulo del producto)."""
    a, b, c = (pos[caras[:, i]] for i in range(3))
    n_cara = np.cross(b - a, c - a)
    n = np.zeros_like(pos)
    for i in range(3):
        np.add.at(n, caras[:, i], n_cara)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _rota(n: np.ndarray, eje: np.ndarray, grados: float) -> np.ndarray:
    """Rodrigues: gira `n` alrededor de `eje` los grados pedidos."""
    k = eje / max(float(np.linalg.norm(eje)), 1e-12)
    c, s = np.cos(np.radians(grados)), np.sin(np.radians(grados))
    return n * c + np.cross(k, n) * s + k * ((n @ k) * (1.0 - c))[:, None]


def _oclusion_ambiental(malla: np.ndarray, normales: np.ndarray) -> np.ndarray:
    """Cuanto tapa la propia geometria a cada punto, en `[OCLUSION_MINIMA, 1]`.

    Es una medida de cavidad, no un trazado de rayos: para cada punto se mira **cuanto** se
    salen sus vecinos de su plano tangente, dentro de `RADIO_OCLUSION_MM`. En una superficie
    lisa los vecinos estan EN el plano y el seno medio es ~0; en el fondo de una tronera
    estan por encima y sube.

    ⚠️ **Contar cuantos vecinos quedan por delante NO vale, y es lo que se probo primero.**
    Sobre una superficie el plano tangente parte el vecindario por la mitad se este donde
    se este, asi que la fraccion salia ~0,5 en todas partes: medido sobre el caso real, el
    **83 %** de las gaussianas se iba al suelo del rango y la oclusion dejaba de distinguir
    un surco de una cuspide — oscurecia la arcada entera por igual. Lo que separa las dos
    cosas es la MAGNITUD del seno, no su signo.

    ⚠️ **No es la oclusion ambiental de un renderizador y no se declara como tal.** Aquella
    integra visibilidad sobre el hemisferio; esta cuenta vecinos. Coinciden en lo que
    importa aqui —que las hendiduras se oscurezcan y las cuspides no— y no coinciden en el
    valor absoluto, que por eso no se publica como magnitud fisica sino como un factor de
    visualizacion entre 0 y 1.

    ⚠️ **Se calcula SOBRE LA MALLA y se transfiere despues, y la firma lo impone.** Medirla
    en los centros de las gaussianas no vale y esta medido: el optimizador las mueve fuera
    de la superficie, asi que una que quede por dentro tiene a TODOS sus vecinos por delante
    de su plano tangente y sale oscura siempre. Sobre el caso real eso mandaba el **78 %**
    de las gaussianas al suelo del rango, contra el **5 %** que sale calculandolo en los
    vertices. La oclusion es una propiedad de la superficie, no de como se muestree.
    """
    from scipy.spatial import cKDTree

    d, i = cKDTree(malla).query(malla, k=VECINOS_OCLUSION,
                                distance_upper_bound=RADIO_OCLUSION_MM)
    valido = np.isfinite(d)
    # El vecino que falta viene con indice fuera de rango: se apunta al 0 y se descarta
    # con `valido`, que es mas barato que recortar cada fila por separado.
    i = np.where(valido, i, 0)
    hacia = malla[i] - malla[:, None, :]
    largo = np.maximum(np.linalg.norm(hacia, axis=2), 1e-9)
    seno = np.einsum("ijk,ik->ij", hacia, normales) / largo
    # Solo lo que se sale POR DELANTE tapa; lo que queda por detras del plano no ve.
    tapado = np.where(valido, np.maximum(seno, 0.0), 0.0).sum(1) / np.maximum(
        valido.sum(1), 1
    )
    return np.clip(1.0 - GANANCIA_OCLUSION * tapado, OCLUSION_MINIMA, 1.0)


def _relieve_sh1(
    centros: np.ndarray, malla: np.ndarray, normales: np.ndarray,
    albedo: np.ndarray | None, eje_rasante: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """`(normal por gaussiana, coeficientes SH1 (N,9))`, o `(None, None)` sin albedo.

    El rasterizador evalua  `c = C0*sh0 + C1*(-y*sh1 + z*sh2 - x*sh3) + 0.5`  con `(x,y,z)`
    la direccion de vista. Poniendo `sh1 = -k*n_y/C1`, `sh2 = k*n_z/C1`, `sh3 = -k*n_x/C1`
    el segundo sumando vale **exactamente** `k*(n·v)`. No es una aproximacion: el grado 1
    es lineal en la direccion, que es justo la forma de un termino difuso.

    El orden de salida es canal-mayor —R,R,R,G,G,G,B,B,B— que es el del PLY de INRIA.
    """
    from scipy.spatial import cKDTree

    if albedo is None or len(albedo) != len(centros):
        return None, None
    n = normales[cKDTree(malla).query(centros, k=1)[1]]
    # La normal que se codifica es la ROTADA: eso convierte la luz frontal —plana— en una
    # rasante, que es la que alarga los gradientes. `nx,ny,nz` siguen llevando la normal
    # DE VERDAD, sin rotar: lo que se declara medido tiene que ser lo medido.
    if eje_rasante is None:
        n_luz = n
    else:
        n_luz = (
            _rota(n, eje_rasante, RASANTE_GRADOS)
            + RELLENO_RELIEVE * _rota(n, eje_rasante, -0.6 * RASANTE_GRADOS)
        )
    k = FUERZA_RELIEVE * albedo
    sh = np.empty((len(centros), 3, 3), dtype=np.float64)
    sh[:, :, 0] = -k * n_luz[:, None, 1] / C1
    sh[:, :, 1] = k * n_luz[:, None, 2] / C1
    sh[:, :, 2] = -k * n_luz[:, None, 0] / C1
    return n, sh.reshape(len(centros), 9)


def escribe_ply_coloreado(
    destino: Path, posiciones: np.ndarray, caras: np.ndarray, rgb: np.ndarray
) -> None:
    """PLY binario little-endian con color de vértice (RGB uint8).

    Es el formato que Blender espera para pintar con `ShaderNodeVertexColor`.
    """
    V = posiciones.astype(np.float32)
    with destino.open("wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(V)}\n".encode())
        f.write(
            b"property float x\nproperty float y\nproperty float z\n"
            b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        f.write(f"element face {len(caras)}\n".encode())
        f.write(b"property list uchar int vertex_indices\nend_header\n")
        vd = np.empty(
            len(V),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")],
        )
        vd["x"], vd["y"], vd["z"] = V[:, 0], V[:, 1], V[:, 2]
        vd["r"], vd["g"], vd["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(vd.tobytes())
        fd = np.empty(
            len(caras),
            dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")],
        )
        fd["n"] = 3
        fd["a"], fd["b"], fd["c"] = caras[:, 0], caras[:, 1], caras[:, 2]
        f.write(fd.tobytes())


def _lee_ply_coloreado(ruta: Path) -> np.ndarray:
    """El color por vertice `(N, 3)` float de un PLY escrito por `escribe_ply_coloreado`.

    Es el inverso del escritor: sirve para REUSAR el color ya calculado de otra corrida
    del mismo caso sin repetir la proyeccion de fotos. El dtype sale de las propiedades
    DECLARADAS en la cabecera, no de una lista escrita aqui.
    """
    datos = ruta.read_bytes()
    fin = datos.index(b"end_header\n") + len(b"end_header\n")
    campos, n = [], 0
    for linea in datos[:fin].decode("ascii").splitlines():
        if linea.startswith("element vertex"):
            n = int(linea.split()[-1])
        elif linea.startswith("property ") and "list" not in linea:
            _, tipo, nombre = linea.split()
            campos.append((nombre, {"float": "<f4", "uchar": "u1"}[tipo]))
    v = np.frombuffer(datos, np.dtype(campos), count=n, offset=fin)
    return np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Blender: render multivista
# ---------------------------------------------------------------------------

def _render_blender(
    scan_ply: Path,
    destino: Path,
    *,
    n_vistas: int = N_VISTAS,
    resolucion: int = RESOLUCION,
    script_blender: Path | None = None,
    cerca: int = 0,
) -> dict:
    """Llama a Blender headless para renderizar N vistas con pose exacta.

    Devuelve el diccionario `transforms.json` con las poses de camara.
    Lanza `RuntimeError` si Blender falla o no produce el `transforms.json`.
    """
    if script_blender is None:
        # Buscar en el repo raíz: ../../../../scripts/blender_render_views.py
        # __file__ = .../packages/gaussian-engine/src/gaussian_engine/apariencia.py
        script_blender = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "scripts" / "blender_render_views.py"
        )
    if not script_blender.exists():
        raise FileNotFoundError(
            f"Script de Blender no encontrado: {script_blender}. "
            "Instala el proyecto o pasa --script-blender."
        )

    (destino / "images").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    r = subprocess.run(
        [
            "blender", "--background", "--python", str(script_blender), "--",
            "--scan", str(scan_ply),
            "--out", str(destino),
            "--views", str(n_vistas),
            "--res", str(resolucion),
            "--samples", "16",
            "--elevations", "8",
            *(["--cerca", str(cerca)] if cerca else []),
        ],
        capture_output=True,
        text=True,
    )
    tiempo_blender = time.perf_counter() - t0

    transforms_path = destino / "transforms.json"
    if not transforms_path.exists():
        stderr_recortado = (r.stderr or "")[-2000:]
        raise RuntimeError(
            f"Blender no produjo transforms.json ({tiempo_blender:.0f}s). "
            f"Stderr:\n{stderr_recortado}"
        )

    T = json.loads(transforms_path.read_text())
    print(
        f"Blender: {tiempo_blender:.0f}s · {len(T['frames'])} vistas · "
        f"fov_x {T['camera_angle_x']:.3f} · escala {T['scan_scale']:.4f}"
    )
    return T


def _requisitos_profundidad(T: dict, destino: Path) -> None:
    """Comprueba que la supervision de profundidad tiene su mapa por CADA vista, o falla alto.

    ⚠️ El fallo es ALTO a proposito. Un entrenamiento con `peso_profundidad > 0` sin los
    mapas fallaria con un `FileNotFoundError` dentro del bucle, 500 iteraciones tarde.
    Se comprueba el CONTEO contra las vistas del render: un `depth/` de otra corrida con
    distinto numero de vistas se detecta aqui y no a mitad de entrenamiento.
    """
    if not (destino / "depth").is_dir():
        raise FileNotFoundError(
            f"peso_profundidad > 0 pero {destino / 'depth'} no existe: "
            "genera los mapas con _profundidad_raycast."
        )
    n = len(list((destino / "depth").glob("z_*.npy")))
    if n != len(T["frames"]):
        raise ValueError(
            f"peso_profundidad > 0 pero {destino / 'depth'} trae {n} mapa(s) y el render "
            f"{len(T['frames'])} vista(s): no son de la misma corrida."
        )


def _profundidad_raycast(
    T: dict,
    destino: Path,
    posiciones: np.ndarray,
    caras: np.ndarray,
    *,
    traza: bool = False,
) -> None:
    """Escribe `depth/z_%05d.npy` por vista: la distancia al plano de camara del PRIMER
    impacto, por raycast de la malla con las poses EXACTAS del render.

    ⚠️ Esto antes salia de la pasada Z de Blender, y ya no puede: Blender 5.x quito el
    buffer Z del sistema de imagenes y el compositor nuevo no lo expone. El dato es el
    MISMO —el Z de la primera superficie visible desde la camara, que es lo unico que el
    render pudo dibujar— y aqui la malla y las poses son las mismas del render, asi que el
    mapa es exacto salvo el antialias del borde, que la mascara de alfa recorta igual que
    en la perdida.

    El raycast usa trimesh+embree (unos 0,1-0,3 s por vista a 1.024 px). Los ficheros van
    en `.npy` float32, en las MISMAS unidades normalizadas en que entrenan las camaras:
    el lector no aplica ninguna escala.
    """
    import trimesh

    P = np.asarray(posiciones, dtype=np.float64)
    Pn = (P - np.asarray(T["scan_offset"], dtype=np.float64)) * float(T["scan_scale"])
    malla = trimesh.Trimesh(
        vertices=Pn.astype(np.float64), faces=np.asarray(caras, dtype=np.int64),
        process=False,
    )
    if not hasattr(malla.ray, "intersects_location"):
        raise ImportError(
            "trimesh sin backend de rayos: instala `embreex` en el entorno de GPU."
        )

    W = H = T["w"]
    fx = 0.5 * W / math.tan(0.5 * T["camera_angle_x"])
    cx, cy = W / 2, H / 2
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dir_cam = np.stack([(us - cx) / fx, (vs - cy) / fx, -np.ones_like(us, float)], -1)
    dir_cam /= np.linalg.norm(dir_cam, axis=-1, keepdims=True)  # (H, W, 3)

    (destino / "depth").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, f in enumerate(T["frames"]):
        from PIL import Image

        alfa = np.asarray(
            Image.open(destino / f["file_path"]).convert("RGBA"), np.float32
        )[..., 3]
        m = (alfa > 127.0).ravel()
        if not m.any():
            np.save(destino / "depth" / f"z_{k:05d}.npy", np.zeros((H, W), np.float32))
            continue
        M = np.asarray(f["transform_matrix"], dtype=np.float64)
        R, t = M[:3, :3], M[:3, 3]
        # ⚠️ El rayo en MARCO DE CAMARA se gira al mundo antes del raycast: sin esto el
        # trazado apuntaria siempre a -z del mundo y no a donde mira la camara.
        rayos = np.flatnonzero(m)
        dirs = (dir_cam.reshape(-1, 3)[m]) @ R.T  # mundo = R @ camara
        ori = np.broadcast_to(t, dirs.shape)
        locs, iray, _tri = malla.ray.intersects_location(
            ori, dirs, multiple_hits=False
        )
        z = np.zeros((H, W), np.float32)
        if len(locs):
            p_cam = (locs - t) @ R          # R^T·(loc-t) con R ortogonal
            # `iray` indexa DENTRO de los rayos enmascarados: los rayos sin impacto no
            # aparecen, asi que se dispersa por el indice y no por la mascara entera.
            z.ravel()[rayos[iray]] = -p_cam[:, 2]  # positivo delante (Blender mira -z)
        np.save(destino / "depth" / f"z_{k:05d}.npy", z)
        if traza and (k % 200 == 0 or k == len(T["frames"]) - 1):
            dt = time.perf_counter() - t0
            print(f"  profundidad: {k + 1}/{len(T['frames'])} vistas en {dt:.0f}s "
                  f"({dt / (k + 1):.3f}s/vista)")
    print(f"profundidad por raycast: {len(T['frames'])} vistas en "
          f"{time.perf_counter() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Entrenamiento gsplat
# ---------------------------------------------------------------------------

def _siembra_superficie(
    P: np.ndarray, caras: np.ndarray, colores_rgb: np.ndarray,
    n: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Semillas repartidas por AREA sobre los triangulos, no una por vertice.

    ⚠️ **Es lo unico que separa la resolucion del campo de la de la malla.** Sembrando por
    vertice —`rng.choice(len(P), min(semillas, len(P)))`— el numero de gaussianas queda
    acotado por arriba al numero de vertices, y el campo hereda su densidad: 5,11 muestras
    por mm sobre una arcada de 4.366 mm2, o sea 0,11 Mpx para la boca entera. Las vistas se
    renderizan a 15,8 px/mm, tres veces mas finas: ese detalle se captura y no se guarda en
    ningun sitio.

    Muestreando puntos baricentricos con probabilidad proporcional al area, la apariencia
    puede ser mas fina que la geometria —que es lo que hace una textura sobre una malla de
    pocos poligonos— y `semillas` vuelve a significar lo que dice.

    El color se interpola con los mismos pesos baricentricos: un punto dentro de un
    triangulo no tiene color propio, lo hereda de sus tres vertices.
    """
    v = P[caras]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    idx = rng.choice(len(caras), n, p=area / area.sum())
    # Reflejar el cuadrado unidad sobre la diagonal da un uniforme en el triangulo.
    u, w = rng.random(n), rng.random(n)
    fuera = u + w > 1
    u[fuera], w[fuera] = 1 - u[fuera], 1 - w[fuera]
    bar = np.stack([1 - u - w, u, w], 1)[:, :, None]
    tri = caras[idx]
    pts = (bar * P[tri]).sum(1)
    col = (bar * colores_rgb[tri].astype(np.float64)).sum(1)
    return pts.astype(np.float32), col.astype(np.float32)


def _entrena_gsplat(
    T: dict,
    posiciones: np.ndarray,
    colores_rgb: np.ndarray,
    *,
    destino: Path,
    iteraciones: int = ITERACIONES,
    semillas: int = SEMILLAS,
    caras: np.ndarray | None = None,
    siembra: str = "vertice",
    dispositivo: str = "cuda",
    traza: bool = False,
    # ⚠️ Estrategia de densificacion/poda. "mcmc" (Kheradmand et al. 2024) REUBICA las
    # gaussianas muertas en vez de podarlas: la cuenta se mantiene cerca de `cap_max` y la
    # niebla sale mucho menor. MEDIDO en este proyecto: con la estrategia clasica, la
    # siembra por area de 1,09 M cayo a 120 k por las podas de opacidad, y con eso se fue
    # la capacidad. Con "mcmc" no hay poda masiva: hay reubicacion.
    estrategia: str = "default",
    # ⚠️ Render ANTI-ALIASED (Mip-Splatting, CVPR 2024): filtro de huella de pixel que
    # evita el halo por dilatacion y los primitivos degenerados. Es el modo pensado para
    # que la recreacion se vea nitida a resoluciones mayores que la de entrenamiento.
    antialiased: bool = False,
    # ⚠️ Supervision de PROFUNDIDAD: compara el Z esperado del campo contra el Z por
    # raycast de la malla (`depth/z_*.npy`), para que las gaussianas no floten.
    # `_requisitos_profundidad` comprueba que los mapas estan y falla alto si no.
    peso_profundidad: float = 0.0,
    # ⚠️ El peso de profundidad entra EN RAMPA, y el porqué está MEDIDO: a peso pleno desde
    # la iteracion 0, la E[z] de la niebla inicial es caotica y el anclaje pelea contra la
    # imagen (holdout 26 dB contra los 36 del mismo caso sin profundidad). Tras la
    # convergencia de imagen, el campo ya cumple E[z]≈superficie (mediana de error 0,005
    # unidades sobre el campo de 36 dB, medida contra el raycast), asi que la profundidad
    # PULE en vez de pelear.
    profundidad_desde: int = 6_000,
    profundidad_rampa: int = 3_000,
    # ⚠️ Regularizador de APLANAMIENTO: penaliza el eje MENOR en absoluto (media de
    # exp(scale_min)), empujando el elipsoide hacia el disco pegado a la superficie.
    # ⚠️ La version anterior penalizaba la RAZON menor/mayor y era DEGENERADA, y esta
    # MEDIDO: el gradiente del cociente tambien EXPANDE el eje mayor, y el resultado fue
    # discos de espesor 3,4 µm con huella de 1.344 µm — peor que no regularizar. En
    # absoluto el gradiente decrece con el propio eje (auto-limitante, no colapsa a cero)
    # y el eje mayor queda solo a cargo de la perdida de imagen. 0 = desactivado.
    peso_aplanado: float = 0.0,
    aplanado_desde: int = 500,
) -> tuple[dict[str, Any], list[tuple[int, float, float, float]]]:
    """Bucle de entrenamiento 3DGS con gsplat.

    Parametros:
        T: transforms.json de Blender (poses de camara).
        posiciones: (N, 3) vertices de la malla en mm.
        colores_rgb: (N, 3) color RGB [0,255] por vertice.
        destino: directorio donde estan las imagenes renderizadas.

    Devuelve:
        params_dict: diccionario con los arrays entrenados (means, scales, etc.)
        curva: lista de (iter, loss, train_psnr, holdout_psnr)
    """
    import torch
    from gsplat import DefaultStrategy, rasterization
    from PIL import Image
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    dev = torch.device(dispositivo if torch.cuda.is_available() else "cpu")

    # Normalizacion IDENTICA a la de Blender (centrar + escalar a caja unidad).
    P = posiciones.astype(np.float32)
    centro = (P.min(0) + P.max(0)) / 2
    P = (P - centro) * (2.0 / (P.max(0) - P.min(0)).max())

    rng = np.random.default_rng(0)
    if siembra == "area" and caras is not None:
        pts, col_np = _siembra_superficie(P, caras, colores_rgb, semillas, rng)
    else:
        # ⚠️ El `min()` ACOTA las semillas al numero de vertices: pedir mas no da mas. Es
        # el comportamiento historico y se conserva por defecto; `siembra="area"` es el
        # que deja que `semillas` signifique lo que dice.
        sel = rng.choice(len(P), min(semillas, len(P)), replace=False)
        pts, col_np = P[sel], colores_rgb[sel].astype(np.float32)
    means0 = torch.tensor(pts, device=dev)
    n0 = len(pts)

    # Sigma inicial: distancia media a los 4 vecinos mas cercanos.
    d = torch.cdist(means0[:2000], means0[:2000]).topk(4, largest=False).values[:, 1:].mean()

    # Color inicial: el muestreado de las fotos, no un gris constante.
    col0 = torch.tensor(col_np / 255.0, device=dev).clamp(0.01, 0.99)

    # ParameterDict con un Adam POR clave: lo que DefaultStrategy necesita para
    # anadir/quitar filas (densificar y podar) sin descuadrar el estado de Adam.
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means0),
        "scales": torch.nn.Parameter(
            torch.log(torch.full((n0, 3), float(d), device=dev))
        ),
        "quats": torch.nn.Parameter(
            torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n0, 1)
        ),
        "opacities": torch.nn.Parameter(
            torch.logit(torch.full((n0,), 0.1, device=dev))
        ),
        "colors": torch.nn.Parameter(torch.logit(col0)),
    }).to(dev)

    # Camaras: la pose de Blender (OpenGL) se convierte a la convencion OpenCV de gsplat.
    W = H = T["w"]
    fx = 0.5 * W / math.tan(0.5 * T["camera_angle_x"])
    K = torch.tensor(
        [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
        dtype=torch.float32, device=dev,
    )
    flip = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    viewmats = torch.stack([
        torch.linalg.inv(
            torch.tensor(f["transform_matrix"], dtype=torch.float32) @ flip
        )
        for f in T["frames"]
    ]).to(dev)

    usa_profundidad = peso_profundidad > 0
    if usa_profundidad:
        _requisitos_profundidad(T, destino)

    def carga_vista(k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """RGB premultiplicado y alfa de la vista k, bajo demanda (las 1600 son ~20 GB)."""
        rgba = np.asarray(
            Image.open(destino / T["frames"][k]["file_path"]).convert("RGBA"),
            np.float32,
        ) / 255.0
        alfa = torch.from_numpy(rgba[..., 3]).to(dev)
        return torch.from_numpy(rgba[..., :3] * rgba[..., 3:4]).to(dev), alfa

    def carga_imagen(k: int) -> torch.Tensor:
        return carga_vista(k)[0]

    def carga_profundidad(k: int) -> torch.Tensor:
        """Z por raycast de la malla (`depth/z_*.npy`), en las mismas unidades de escena
        que las camaras. 0 = pixel sin superficie. Ver `_profundidad_raycast`."""
        gris = np.load(destino / "depth" / f"z_{k:05d}.npy")
        return torch.from_numpy(gris.astype(np.float32)).to(dev)

    # Split train/holdout: 1 de cada 4 para validacion.
    idx = np.arange(len(T["frames"]))
    test = idx[::4]
    train = np.setdiff1d(idx, test)
    rng2 = np.random.default_rng(1)
    eval_test = rng2.choice(test, min(MUESTRAS_VALIDACION, len(test)), replace=False)

    def render(vm: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out, _, info = rasterization(
            params["means"],
            torch.nn.functional.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            torch.sigmoid(params["colors"]),
            vm[None], K[None], W, H, packed=False,
            # ⚠️ Con profundidad el render trae 4 canales: RGB + la z ESPERADA del campo
            # ("RGB+ED" = Σw·z/Σw, a lo largo del eje de vista). "RGB+D" no vale: es la z
            # ACUMULADA sin normalizar, que se hunde donde la transmitancia no llega a 1 y
            # anclaria el campo mas cerca de la camara. Medido sobre semillas en superficie.
            render_mode="RGB+ED" if usa_profundidad else "RGB",
            rasterize_mode="antialiased" if antialiased else "classic",
        )
        return out[0], info

    def psnr_en(js: np.ndarray) -> float:
        with torch.no_grad():
            return float(np.mean([
                -10 * math.log10(
                    ((render(viewmats[j])[0][..., :3] - carga_imagen(j)) ** 2).mean().item()
                    + 1e-10
                )
                for j in js
            ]))

    # Optimizadores: uno por parametro (lo que DefaultStrategy necesita).
    lr = {
        "means": 1e-3, "scales": 5e-3, "quats": 1e-3,
        "opacities": 5e-2, "colors": 1e-2,
    }
    opt = {k: torch.optim.Adam([v], lr=lr[k]) for k, v in params.items()}

    # Control adaptativo de densidad. La clasica (Kerbl et al. 2023) densifica suave, poda
    # casi-transparentes y resetea opacidad; la MCMC (Kheradmand et al. 2024) reubica las
    # muertas y mantiene la cuenta cerca del tope. Ver `estrategia` en la firma.
    from gsplat import MCMCStrategy

    if estrategia == "mcmc":
        strat = MCMCStrategy(
            cap_max=1_000_000,
            refine_start_iter=500,
            # ⚠️ La densificacion tardia es la causa principal de floaters (medido por los
            # autores del MCMC): se corta en el primer tercio y la cuenta ya no se mueve.
            refine_stop_iter=6_000,
            refine_every=100,
            # ⚠️ El ruido Langevin por defecto (5e5) esta pensado para lr~1e-2 y escenas
            # de gaussianas pequenas. MEDIDO aqui: con lr 1e-3 el desplazamiento es
            # 500·s², que para una gaussiana grande de la niebla son saltos de 0,45
            # unidades cada 100 pasos — el campo no se asienta nunca (12 dB a la 5.000 y
            # E[z] aleatoria). Con 1e4 el salto es ~1 % del tamano de la gaussiana: la
            # reubicacion sigue viva y el ruido deja de destrozar el ajuste.
            noise_lr=1e4,
            verbose=False,
        )
    else:
        strat = DefaultStrategy(
            refine_start_iter=500,
            refine_stop_iter=iteraciones // 2,
            reset_every=1500,
            refine_every=100,
            verbose=False,
        )
    strat.check_sanity(params, opt)
    # ⚠️ La MCMC no usa la escala de escena: su estado es la tabla de binomios.
    estado = (strat.initialize_state(scene_scale=1.0) if estrategia == "default"
              else strat.initialize_state())

    torch.manual_seed(0)
    curva: list[tuple[int, float, float, float]] = []
    t0 = time.perf_counter()

    for it in range(iteraciones):
        i = int(np.random.choice(train))
        out, info = render(viewmats[i])
        strat.step_pre_backward(params, opt, estado, it, info)

        tgt, alfa = carga_vista(i)
        rgb = out[..., :3]
        a = rgb.permute(2, 0, 1).unsqueeze(0)
        b = tgt.permute(2, 0, 1).unsqueeze(0)
        perdida = 0.8 * (rgb - tgt).abs().mean() + 0.2 * (1.0 - ssim(a, b, data_range=1.0))
        perdida_d = perdida_a = None

        # ⚠️ Supervision de profundidad, enmascarada a los pixeles que el render de veras
        # dibujo (alfa) y con Z de Blender valido. Impide que las gaussianas floten fuera
        # de la superficie: el Z del campo se compara CONTRA la pasada Z del mismo render.
        if usa_profundidad:
            z_gt = carga_profundidad(i)
            z_pr = out[..., 3]
            # La mascara tambien exige Z renderizado > 0: donde el campo todavia no dibuja
            # nada, gsplat devuelve z=0 y anclaria contra un valor que no es el campo.
            m = (alfa > 0.5) & (z_gt > 0) & (z_pr > 0)
            if m.any():
                perdida_d = (z_pr[m] - z_gt[m]).abs().mean()
                w = min(1.0, max(0.0, (it - profundidad_desde) / max(profundidad_rampa, 1)))
                perdida = perdida + peso_profundidad * w * perdida_d

        # ⚠️ Regularizador de aplanamiento: eje MENOR en absoluto, hacia disco pegado a la
        # superficie. Ver la nota de la firma: la razon menor/mayor estaba descartada por
        # degenerada (medido). Arranca en `aplanado_desde` para no aplastar antes de que
        # la superficie se oriente.
        if peso_aplanado > 0 and it >= aplanado_desde:
            perdida_a = torch.exp(params["scales"]).min(dim=-1).values.mean()
            perdida = perdida + peso_aplanado * perdida_a

        for o in opt.values():
            o.zero_grad()
        perdida.backward()
        for o in opt.values():
            o.step()
        if estrategia == "mcmc":
            # ⚠️ La MCMC no recibe `packed` y SI recibe el lr de `means`, que escala el
            # ruido de Langevin de la reubicacion.
            strat.step_post_backward(params, opt, estado, it, info, lr["means"])
        else:
            strat.step_post_backward(params, opt, estado, it, info, packed=False)

        if it % 500 == 0 or it == iteraciones - 1:
            ptr = psnr_en(train) if len(train) > 0 else 0.0
            pt = psnr_en(eval_test)
            curva.append((it, perdida.item(), ptr, pt))
            if traza:
                extra = ""
                if perdida_d is not None:
                    extra += f"  z {perdida_d.item():.4f}"
                if perdida_a is not None:
                    extra += f"  plano {perdida_a.item():.4f}"
                print(
                    f"  it {it:5d}  N={params['means'].shape[0]:>7,}  "
                    f"loss {perdida.item():.4f}{extra}  train {ptr:.2f}  holdout {pt:.2f} dB"
                )

    tiempo_total = time.perf_counter() - t0
    n_final = params["means"].shape[0]
    print(
        f"entrenamiento: {tiempo_total:.0f}s · {n_final:,} gaussianas finales · "
        f"PSNR holdout {curva[-1][3]:.2f} dB"
    )

    # Convertir a dict de arrays numpy para el store.
    with torch.no_grad():
        resultado = {
            "means": params["means"].cpu().numpy().astype(np.float64),
            "scales": params["scales"].cpu().numpy().astype(np.float64),
            "quats": torch.nn.functional.normalize(
                params["quats"], dim=-1
            ).cpu().numpy().astype(np.float64),
            "opacities": params["opacities"].cpu().numpy().astype(np.float64),
            "colors": torch.sigmoid(params["colors"]).cpu().numpy().astype(np.float64),
        }

    return resultado, curva


# ---------------------------------------------------------------------------
# Exportacion PLY INRIA
# ---------------------------------------------------------------------------

# Coeficiente DC de los armonicos esfericos (grado 0).
C0 = 0.28209479177387814
# Base de los armonicos de grado 1. El rasterizador evalua
#   c = C0*sh0 + C1*(-y*sh1 + z*sh2 - x*sh3) + 0.5,  con (x,y,z) la direccion de vista.
C1 = 0.4886025119029199

# ⚠️ **Cuanto modula el relieve, en fraccion del albedo, y por que existe.**
#
# El campo se entrena contra renders de ALBEDO plano —sin eso las gaussianas guardaban el
# diente bajo un sol que nos inventabamos nosotros, y recuperar el color medido costaba
# ΔE 28 en vez de 0,35 por pieza—. El precio es que un albedo puro se dibuja PLANO: sin
# variacion con la vista una arcada pierde el volumen y no se lee.
#
# El grado 1 de los armonicos es exactamente una funcion lineal de la direccion de vista,
# asi que un termino `n·v` cabe ahi EXACTO y se puede escribir en vez de aprenderlo. El
# grado 0 sigue siendo el albedo intacto: quien lea `f_dc` se lleva el color del paciente
# sin nada horneado, y quien mire la escena ve el relieve.
#
# El signo es negativo porque el rasterizador usa `dir = normalize(splat - camara)`: una
# cara que mira a la camara tiene `n·v < 0` y es la que hay que iluminar.
FUERZA_RELIEVE = -0.35

# ⚠️ **Y la luz va RASANTE, no en el eje de la camara.** Esto no es un adorno: es lo que
# `scripts/blender_render_views.py` ya tenia escrito sobre su propio montaje —
#
#   «una luz EN el eje de la camara (frontal puro) es plana: n·l ~ n·v, asi que toda
#    superficie que te mira brilla igual y el relieve se pierde. La luz oblicua alarga los
#    gradientes y hace visible el microrrelieve de la anatomia.»
#
# — y la primera version de este relieve codificaba justo `n·v`, o sea el caso frontal puro
# que ese comentario identifica como el malo. Daba volumen y NO daba resolucion.
#
# Cabe igual en el grado 1 porque rotar es lineal:  n·(R v) = (R^T n)·v.  Basta codificar
# la normal ROTADA y sale una luz oblicua que sigue a la camara, que es lo que hacia el
# montaje de Blender al emparentar la luz. El grado 0 no se toca.
#
# El eje de giro es el SUPERIOR de la arcada y no uno cualquiera: las hendiduras que hay
# que revelar —troneras interproximales, surco gingival— corren verticales, y una luz
# desplazada de lado es la que las alarga.
RASANTE_GRADOS = 35.0

# El RELLENO del montaje de Blender, por el lado contrario y flojo: `--relleno 0.8` contra
# los 3.0 de la principal, y menos inclinado (`-_rk * 0.6`). Existe por lo mismo que alli —
# «para que la cara en sombra no se cierre a negro: lo que ahi se pierda no lo puede
# aprender ningun entrenamiento» — y aqui ademas es gratis: la suma de dos terminos
# lineales en la direccion de vista **sigue siendo lineal**, asi que las dos luces caben en
# los mismos nueve coeficientes del grado 1. Basta sumar las normales giradas.
RELLENO_RELIEVE = 0.8 / 3.0

# ⚠️ **La oclusion ambiental NO cabe en los armonicos, y por eso va en columna aparte.**
# Un surco esta oscuro lo mires desde donde lo mires: es independiente de la vista, asi que
# dentro de los SH solo cabria en el grado 0 — que ES el albedo. Meterla ahi es volver a
# contaminar el color que `clinical/observations.json` declara al lado, que es justo el
# fallo que costo llegar a ΔE 0,35 por pieza.
#
# En una columna propia no hay ese problema: el emisor la calcula, la declara como
# CALCULADA, y quien dibuja la multiplica. Quien lee `f_dc` para tomar un tono no se lleva
# ninguna sombra — y eso importa, porque una lectura de color no debe oscurecerse porque la
# pieza tenga una fisura al lado.
#
# Radio en mm de la cavidad que se mide. 2,0 es la escala de una tronera interproximal y
# del surco gingival, que son las hendiduras que el ojo busca. Mas pequeno mide rugosidad
# del mallado; mas grande oscurece la arcada entera por estar dentro de una boca.
RADIO_OCLUSION_MM = 2.0
# Vecinos que se consultan por punto. Con 32 la cuenta satura en las cavidades cerradas,
# que es donde el valor tiene que saturar.
VECINOS_OCLUSION = 32
# Suelo de la oclusion: por debajo de esto no se oscurece mas. Sin suelo, el fondo de un
# surco se va a negro y ahi se pierde el detalle que se queria ver.
OCLUSION_MINIMA = 0.45
# Cuanto oscurece un seno medio dado. Con 2,5 una superficie lisa —seno ~0— se queda en 1 y
# el fondo de una tronera llega al suelo, que es el rango util. Es un juicio declarado.
GANANCIA_OCLUSION = 2.5

# Propiedades INRIA 3DGS grado 0 (las mismas que usa el visor web).
PROPIEDADES_INRIA = (
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "f_rest_0", "f_rest_1", "f_rest_2",
    "f_rest_3", "f_rest_4", "f_rest_5",
    "f_rest_6", "f_rest_7", "f_rest_8",
    "ao",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


def _comentarios_color(params: dict) -> list[str]:
    """Que es el `f_dc_*` de ESTE fichero, leido de lo que el entrenamiento guardo.

    ⚠️ **Se lee, no se afirma.** Estas cuatro lineas eran un literal que decia «DOS tonos
    muestreados de las fotos», y antes de eso decian «color real del paciente». Las dos
    veces era una frase escrita a mano en un sitio distinto de donde se decidia el color, y
    las dos veces envejecio. Ahora el entrenamiento guarda cuantos vertices llevan pixel
    medido y la cabecera lo cuenta.
    """
    n = int(np.asarray(params.get("n_vertices_malla", 0)).reshape(-1)[0] or 0)
    med = int(np.asarray(params.get("n_vertices_medidos", 0)).reshape(-1)[0] or 0)
    interp = int(np.asarray(params.get("n_vertices_interpolados", 0)).reshape(-1)[0] or 0)
    pieza = int(np.asarray(params.get("n_vertices_por_pieza", 0)).reshape(-1)[0] or 0)
    piezas = int(np.asarray(params.get("n_piezas_con_tono", 0)).reshape(-1)[0] or 0)
    if int(np.asarray(params.get("color_reutilizado", 0)).reshape(-1)[0] or 0):
        return [
            "comment f_dc_* = color REUTILIZADO de otra corrida del mismo caso (su",
            "comment scan_colored.ply), para comparar experimentos con identico color de",
            "comment entrada. La procedencia del color la declara la corrida origen.",
        ]
    # ⚠️ **Dos mecanismos distintos no caben en un contador.** Antes esta cabecera
    # atribuia TODO a la proyeccion por vertice con PnP; con el color por pieza mandando,
    # eso decia una procedencia falsa del 97 % de los vertices. Y los interpolados que
    # declaraba eran los del camino viejo, que la pieza habia sobrescrito.
    if n and pieza:
        return [
            f"comment f_dc_* = color MEDIDO POR PIEZA: {piezas} corona(s) con su",
            "comment mediana de pixeles por tercio (cervical, medio, incisal) tomada de la",
            "comment foto que mejor ve cada una, mas la encia medida aparte. NO hace falta",
            "comment pose: la foto se parte en coronas y se alinea con el arco.",
            f"comment {pieza} de {n} vertices reciben color de SU pieza. De los "
            f"{n - pieza} restantes,",
            f"comment {med} llevan pixel proyectado, {interp} lo heredan del medido mas "
            f"cercano y {n - pieza - med - interp}",
            "comment se pintan con el degradado de respaldo, que NO es color del paciente.",
            "comment NO es measured=true: el optimizador movio, dividio y podo las",
            "comment gaussianas, asi que no hay correspondencia 1:1 con lo proyectado",
        ]
    if n and med:
        return [
            "comment f_dc_* = color MEDIDO: el pixel que una foto intraoral ve en cada",
            "comment vertice, con la pose resuelta por PnP sobre correspondencias por",
            f"comment diente. {med} de {n} vertices medidos, {interp} interpolados del",
            "comment medido mas cercano y el resto con el degradado de respaldo.",
            "comment NO es measured=true: el optimizador movio, dividio y podo las",
            "comment gaussianas, asi que no hay correspondencia 1:1 con lo proyectado",
        ]
    return [
        "comment f_dc_* = DOS tonos muestreados de las fotos (mediana de esmalte y de",
        "comment encia). NO es color medido por vertice. El limite entre los dos sale",
        "comment de la segmentacion FDI (Layer 3, ver derived/) si viaja, y si no de un",
        "comment percentil de altura, que es un apano y no una medida",
    ]


def _fdi_por_gaussiana(
    centros: np.ndarray,
    vertices: np.ndarray,
    etiquetas: np.ndarray,
    *,
    limite: float,
    votos: int = VOTOS_FDI,
) -> tuple[np.ndarray, np.ndarray, int]:
    """El codigo FDI de cada gaussiana, por MAYORIA de los vertices mas cercanos.

    Devuelve `(region_id, lejos, cuantas_cambiaron)`.

    ⚠️ **Por mayoria y no por el vecino mas cercano, y el porque esta medido.** El contacto
    interproximal no tiene borde geometrico: dos coronas contiguas se tocan y el vertice
    mas proximo a una gaussiana del contacto puede ser el del diente de al lado. Sobre la
    malla eso ya se corrige —`afina_fronteras` reasigno 3.559 vertices en un caso real— y
    las gaussianas se saltaban la correccion heredando con 1-NN.

    Medido sobre el caso real: la pureza del vecindario de cada gaussiana pasa de **85,1 %
    con 1-NN a 91,4 % con dieciseis votos**, cambiando el 3,6 % de las etiquetas.

    ⚠️ **Los empates caen del lado de la encia.** `np.bincount` sobre los codigos devuelve
    el menor de los empatados, y el 0 —encia o sin asignar— es el menor de todos. Es la
    direccion correcta en la que equivocarse: marcar encia como diente pinta esmalte sobre
    encia y enciende trozos ajenos al seleccionar una pieza; marcar diente como encia solo
    deja un borde sin asignar.

    ⚠️ **Y la cota de distancia se mide contra el vecino MAS CERCANO, no contra el voto.**
    El optimizador mueve, divide y poda, y algunas gaussianas acaban lejos de cualquier
    vertice: heredar la etiqueta de un vecino a cinco milimetros es inventarsela. Si el mas
    proximo ya esta fuera de la cota, los otros quince tambien.
    """
    from scipy.spatial import cKDTree

    k = max(1, min(int(votos), len(vertices)))
    d, idx = cKDTree(vertices).query(centros, k=k)
    if k == 1:
        d = d[:, None]
        idx = idx[:, None]
    etq = np.asarray(etiquetas).reshape(-1).astype(np.int64)
    vecinas = etq[idx]

    # Voto por mayoria fila a fila. Se compactan los codigos a indices densos porque
    # `bincount` reserva hasta el maximo, y un FDI es 48: barato, pero explicito.
    codigos, comprimido = np.unique(vecinas, return_inverse=True)
    comprimido = comprimido.reshape(vecinas.shape)
    cuentas = np.zeros((len(centros), len(codigos)), dtype=np.int32)
    np.add.at(cuentas, (np.arange(len(centros))[:, None], comprimido), 1)
    reg = codigos[cuentas.argmax(axis=1)].astype(np.int16)

    cercano = etq[idx[:, 0]].astype(np.int16)
    lejos = d[:, 0] > limite
    reg[lejos] = 0
    cercano[lejos] = 0
    return reg, lejos, int((reg != cercano).sum())


def escribe_inria(
    destino: Path,
    params: dict[str, np.ndarray],
    *,
    perfil: str = PERFIL,
    n_vistas: int | None = None,
    iteraciones: int | None = None,
    acquisition_id: str = "",
    scan_scale: float | None = None,
    scan_offset: np.ndarray | list[float] | None = None,
    region_id: np.ndarray | None = None,
) -> CampoEscrito:
    """PLY binario en formato INRIA 3DGS (grado 0 de armonicos esfericos).

    ⚠️ **`scan_scale` y `scan_offset` deshacen la normalizacion de Blender, y sin ellos el
    fichero MIENTE.** Blender normaliza la malla para renderizar —la mete en una caja de
    lado ~2 centrada en el origen— y gsplat entrena en ESE espacio, asi que los parametros
    que salen del optimizador no estan en milimetros. Escribirlos tal cual produce un PLY
    que declara `units: mm` sobre un dato normalizado.

    Medido sobre un caso real: `scan_scale` 0,0308, o sea que la nube salia **32 veces mas
    pequena** que la malla —caja de 2,3 x 1,0 x 2,2 contra 65,0 x 23,2 x 60,5 mm— y con una
    sigma mediana de 0,022 en vez de 0,70 mm. En el visor eso es una mota en el origen, y
    el descriptor afirmando milimetros hacia que nadie pudiera sospecharlo.

    Los dos salen del `transforms.json` que escribe el paso de Blender. **Las escalas van en
    LOGARITMO**, asi que su correccion es una suma —`-log(scan_scale)`— y no un producto.

    Propiedades:
    - x, y, z: float (float32) - centro de la gaussiana en mm
    - nx, ny, nz: float (float32) - normales (0,0,0 para campos)
    - f_dc_0..2: float (float32) - color RGB (coeficiente DC de SH)
    - opacity: float (float32) - opacidad (logit)
    - scale_0..2: float (float32) - escala (logaritmo de mm)
    - rot_0..3: float (float32) - cuaternion (w, x, y, z)

    Diferencia con `escribe_ply` del field-export-agent:
    - density → opacity (con transformacion logit)
    - sin region_id (la apariencia no segmenta)
    - scale en log (convencion INRIA, no lineal)
    - color por gaussiana (f_dc_*)
    """
    n = len(params["means"])
    pos = params["means"].astype(np.float64)
    col = params["colors"].astype(np.float64)  # ya en [0,1]
    esc = params["scales"].astype(np.float64)  # ya en log
    # ⚠️ **Si no se pasan, se buscan en los propios parametros.** No es comodidad: es que
    # olvidarlos escribe un fichero que miente sobre sus unidades, y ya paso dos veces —una
    # en el pipeline y otra en el agente de UOS, que reescribe el PLY desde el almacen—.
    # Un argumento que hay que acordarse de pasar para que el resultado sea correcto es un
    # argumento mal puesto; el defecto tiene que ser lo correcto.
    if scan_scale is None and "scan_scale" in params:
        scan_scale = float(np.asarray(params["scan_scale"]).reshape(-1)[0])
    if scan_offset is None and "scan_offset" in params:
        scan_offset = np.asarray(params["scan_offset"], dtype=np.float64).reshape(3)
    if region_id is None and "region_id" in params:
        region_id = np.asarray(params["region_id"])
    # Normales y relieve: si el entrenamiento los calculo viajan en `params`, igual que el
    # `region_id`. Si no, van a cero — que es un grado 1 nulo, o sea el plano de antes, que
    # es la degradacion correcta y no una invencion.
    nrm = np.asarray(params["normales"], dtype=np.float64) if "normales" in params else None
    sh1 = np.asarray(params["sh1"], dtype=np.float64) if "sh1" in params else None
    # 1 = nada tapado. Sin malla no se puede calcular, y 1 es la respuesta correcta: no
    # oscurecer. Cero seria afirmar que todo esta ocluido.
    ao = np.asarray(params["ao"], dtype=np.float64) if "ao" in params else None

    # Des-normalizacion, en la forma que `scripts/blender_render_views.py` deja escrita
    # junto a los dos valores: `mundo = normalizado / scan_scale + scan_offset`.
    #
    # ⚠️ El signo del offset se comprueba, no se supone: con el contrario el tamano sale
    # bien y el centro se va 7,2 mm, que es un error que no se ve mirando la nube sola —
    # solo aparece comparandola con la malla.
    if scan_scale is not None:
        if scan_scale <= 0:
            raise ValueError(f"scan_scale tiene que ser positivo y vale {scan_scale}")
        pos = pos / scan_scale
        esc = esc - float(np.log(scan_scale))
    if scan_offset is not None:
        pos = pos + np.asarray(scan_offset, dtype=np.float64)
    pos = pos.astype(np.float32)
    unidades = UNIDADES_MM if scan_scale is not None else UNIDADES_NORMALIZADO
    rot = params["quats"].astype(np.float64)   # ya normalizado
    opa = params["opacities"].astype(np.float64)  # ya en logit

    # Leer metadatos del dict si no se pasaron como argumento.
    if n_vistas is None:
        n_vistas = int(params["n_vistas"]) if "n_vistas" in params else 0
    if iteraciones is None:
        iteraciones = int(params["iteraciones"]) if "iteraciones" in params else 0

    # Color RGB → SH DC coefficient: f_dc = (color - 0.5) / C0
    f_dc = (col - 0.5) / C0

    # ⚠️ **El FDI por gaussiana, si se sabe.** Sin el, seleccionar una pieza obliga a tener
    # la malla delante —el picking del §11.3 esta definido sobre `extras.uos_fdi` de un
    # primitive— y un contenedor de solo gaussianas no la lleva. Con el, encender una pieza
    # es filtrar por codigo: no hace falta ninguna superficie.
    #
    # NO sale del optimizador: sale de preguntar, por cada gaussiana, la etiqueta del
    # vertice de corona MAS CERCANO. El optimizador movio, dividio y podo, asi que no hay
    # correspondencia 1:1 que heredar — y decir «vecino mas cercano» es exacto y auditable,
    # mientras que decir «etiqueta entrenada» seria falso.
    reg = None if region_id is None else np.asarray(region_id, dtype=np.int16)
    if reg is not None and len(reg) != n:
        raise ValueError(f"region_id trae {len(reg)} codigos y hay {n} gaussianas")

    # ⚠️ **La cabecera va sin tildes, a proposito.** El formato PLY la define como ASCII y
    # nuestro propio lector la decodifica asi (`Ply.ts`); un lector estricto ajeno puede
    # rechazar el fichero. Salieron 7 bytes no-ASCII en un contenedor real —"movio",
    # "visualizacion", "radiologica"— y un formato que solo lee su emisor no es un formato.
    cabecera = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment generado por gaussian-engine@{perfil}",
        f"comment acquisition_id {acquisition_id}",
        "comment perfil INRIA 3DGS grado 0 - APARIENCIA, no medida",
        "comment las gaussianas NO son los vertices del escaner: el optimizador las",
        "comment movio, dividio y podo. No hay correspondencia 1:1 con lo medido.",
        # ⚠ NO es el color del paciente, y decirlo asi era el fallo. De las fotos salen
        # exactamente DOS numeros —la mediana de los pixeles claros-calidos y la de los
        # rosados— y la malla se pinta interpolandolos por altura z antes de renderizar.
        # El 3DGS aprende ESO. Ademas ese degradado afirma el margen gingival por percentil
        # de altura, que es justo la frontera que este proyecto tiene medido que no sabe.
        *_comentarios_color(params),
        *(("comment nx,ny,nz = normal del vertice de malla MAS CERCANO (antes iban a cero",
           "comment por convencion INRIA). De ahi sale el relieve de abajo.",
           "comment f_rest_* = SH grado 1 CALCULADO, no entrenado: vale exactamente",
           f"comment {abs(FUERZA_RELIEVE)}*albedo*(n' . v) con n' la normal girada",
           f"comment {RASANTE_GRADOS:.0f} grados sobre el eje superior mas un relleno",
           "comment opuesto: es el mismo rig direccional que iluminaba los renders, que",
           "comment cabe entero en el grado 1 porque es lineal en la direccion de vista.",
           "comment Es un realce de forma que este emisor anade;",
           "comment para que la pieza se lea con volumen. NO es medida y NO toca el color:",
           "comment f_dc sigue siendo el albedo. Leyendo solo el grado 0 se recupera el",
           "comment color medido sin nada horneado.")
          if sh1 is not None else
          ("comment f_rest_* van a cero: sin malla no hay normales con que calcular el",
           "comment relieve, asi que el grado 1 es nulo y la escena se dibuja plana.",)),
        *(("comment ao = oclusion ambiental CALCULADA por el emisor: fraccion de vecinos",
           f"comment a menos de {RADIO_OCLUSION_MM:.0f} mm que caen por delante del plano",
           "comment tangente, o sea cuanto se tapa el punto a si mismo. Es un factor de",
           "comment VISUALIZACION en [0,1] que quien dibuja multiplica por el color; NO",
           "comment esta metido en f_dc_* a proposito, porque una lectura de tono no debe",
           "comment oscurecerse porque la pieza tenga una fisura al lado.")
          if ao is not None else
          ("comment ao va a 1: sin malla no hay con que calcular la oclusion, y 1 es no",
           "comment oscurecer. Cero afirmaria que todo esta tapado.")),
        "comment opacity = opacidad de visualizacion (logit), NO es atenuacion radiologica",
        "comment scale en logaritmo (convencion INRIA), NO en mm lineales",
        "comment rot es cuaternion (w,x,y,z) normalizado",
        f"comment entrenado contra {n_vistas} renders EEVEE, {iteraciones} iteraciones",
        # ⚠ Las unidades van EN EL FICHERO. Es lo que permite que el descriptor del
        # contenedor las lea en vez de afirmarlas: si algun dia alguien vuelve a escribir
        # sin des-normalizar, el `.gs.json` dira `normalizado` y no `mm`.
        f"comment unidades {unidades}",
        f"element vertex {n}",
        *(f"property float {p}" for p in PROPIEDADES_INRIA),
        *(["comment region_id es el codigo FDI de la corona MAS CERCANA, no una etiqueta",
           "comment aprendida: el optimizador no conserva correspondencia con los vertices",
           "property short region_id"] if reg is not None else []),
        "end_header",
    ]

    # Construir array estructurado
    columnas_extra = [("region_id", "<i2")] if reg is not None else []
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        *((f"f_rest_{k}", "<f4") for k in range(9)),
        ("ao", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        *columnas_extra,
    ])
    filas = np.empty(n, dtype=dtype)
    if reg is not None:
        filas["region_id"] = reg
    filas["x"] = pos[:, 0]
    filas["y"] = pos[:, 1]
    filas["z"] = pos[:, 2]
    # ⚠️ `nx,ny,nz` iban a CERO por convencion INRIA — campos declarados y vacios. Van
    # rellenos cuando se sabe la normal, porque de ahi sale el relieve de `f_rest_*` y un
    # lector tiene que poder recalcularlo o contradecirlo.
    for nombre_eje, k in zip(("nx", "ny", "nz"), range(3), strict=True):
        filas[nombre_eje] = 0.0 if nrm is None else nrm[:, k].astype(np.float32)
    filas["f_dc_0"] = f_dc[:, 0].astype(np.float32)
    filas["f_dc_1"] = f_dc[:, 1].astype(np.float32)
    filas["f_dc_2"] = f_dc[:, 2].astype(np.float32)
    for k in range(9):
        filas[f"f_rest_{k}"] = 0.0 if sh1 is None else sh1[:, k].astype(np.float32)
    filas["ao"] = 1.0 if ao is None else ao.astype(np.float32)
    filas["opacity"] = opa.astype(np.float32)
    filas["scale_0"] = esc[:, 0].astype(np.float32)
    filas["scale_1"] = esc[:, 1].astype(np.float32)
    filas["scale_2"] = esc[:, 2].astype(np.float32)
    filas["rot_0"] = rot[:, 0].astype(np.float32)
    filas["rot_1"] = rot[:, 1].astype(np.float32)
    filas["rot_2"] = rot[:, 2].astype(np.float32)
    filas["rot_3"] = rot[:, 3].astype(np.float32)

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("wb") as f:
        # PLY es ASCII, pero los comentarios pueden contener acentos (nombre del caso).
        # latin-1 cubre todo el rango europeo sin romper la compatibilidad.
        f.write(("\n".join(cabecera) + "\n").encode("latin-1"))
        f.write(filas.tobytes())

    print(f"PLY INRIA: {destino.name} ({n:,} gaussianas, {destino.stat().st_size / 1e6:.1f} MB)")
    return CampoEscrito(
        n_primitivas=n, unidades=unidades, des_normalizado=scan_scale is not None
    )


# ---------------------------------------------------------------------------
# Funcion principal: snapshot-in, snapshot-out
# ---------------------------------------------------------------------------

def _eje_oclusal(posiciones: np.ndarray, etiquetas: np.ndarray) -> np.ndarray:
    """El eje apico-oclusal de la arcada, MEDIDO. Ver `export_agents.anatomia`.

    Se usa para repartir los tres tercios a lo largo de la corona. Si no se puede medir
    —hace falta encia y coronas etiquetadas— se cae al eje menor de la nube, que es la
    misma regla que `marco_anatomico` aplica antes de mirar las etiquetas.
    """
    from export_agents.anatomia import marco_anatomico

    base, _ = marco_anatomico(posiciones, etiquetas)
    if base is not None:
        return np.asarray(base.oclusal, dtype=np.float64)
    pos = np.asarray(posiciones, dtype=np.float64)
    return np.linalg.svd(pos - pos.mean(axis=0), full_matrices=False)[2][-1]


def entrena_apariencia(
    posiciones: np.ndarray,
    caras: np.ndarray,
    rutas_fotos: list[Path],
    *,
    destino: Path,
    n_vistas: int = N_VISTAS,
    resolucion: int = RESOLUCION,
    iteraciones: int = ITERACIONES,
    semillas: int = SEMILLAS,
    # "vertice" acota las semillas al numero de vertices de la malla; "area" las reparte
    # por los triangulos y deja que el campo sea mas fino que la geometria.
    siembra: str = "vertice",
    dispositivo: str = "cuda",
    script_blender: Path | None = None,
    traza: bool = False,
    # Estrategia de densidad ("default" | "mcmc") y render antialiased: ver `_entrena_gsplat`.
    estrategia: str = "default",
    antialiased: bool = False,
    # FDI por vertice del escaneo. Si viene, cada gaussiana sale con el codigo de la corona
    # mas cercana, y entonces se puede seleccionar una pieza SIN tener la malla delante.
    etiquetas: np.ndarray | None = None,
    # ⚠️ **El bit que ninguna medida puede dar.** Para cada foto, el codigo FDI de su
    # PRIMERA corona. El arco es un espejo exacto y una foto intraoral puede estar tomada
    # con espejo, asi que ni la huella de anchuras ni la imagen dicen de que lado es una
    # tira (ver `tono_foto.alinea_con_el_arco`). Sin esto no se usa color por pieza y se
    # cae al de por vertice; con esto, el resto sigue siendo medido.
    lado_fotos: dict[Path, int] | None = None,
    # Vistas de CERCA que se añaden a la orbita. Ver `--cerca` en el script de Blender:
    # con 1.024 px sobre 65 mm de arcada un detalle de 0,2 mm son 3 pixeles, y a tres
    # pixeles la densificacion no tiene de que agarrarse.
    cerca: int = 0,
    # Supervision de PROFUNDIDAD y regularizador de APLANAMIENTO: ver `_entrena_gsplat`.
    # Van por defecto a 0 para que el pipeline emita exactamente igual que antes.
    peso_profundidad: float = 0.0,
    profundidad_desde: int = 6_000,
    profundidad_rampa: int = 3_000,
    peso_aplanado: float = 0.0,
    aplanado_desde: int = 500,
    # REUSAR el color y el render de otra corrida del mismo caso (directorio con
    # `scan_colored.ply`, `transforms.json`, `images/`). Para comparar experimentos con
    # IDENTICO color de entrada y IDENTICOS pixeles, sin que el RANSAC de la pose de foto
    # ni el render metan ruido entre corridas. La cabecera del PLY lo declara.
    reusa: Path | None = None,
) -> tuple[dict[str, np.ndarray], EntrenamientoApariencia]:
    """Entrena un campo de gaussianas sobre una malla pintada con dos tonos de las fotos.

    ⚠️ **No es color medido.** Ver `_colorea_malla` y `_muestra_color_fotos`: de las fotos
    salen dos RGB y la malla se pinta interpolandolos por altura. El color por vertice de
    verdad exige proyectar los pixeles sobre la geometria, que necesita pose de camara.

    Flujo completo:
        1. Muestrear color de fotos (esmalte/encía por altura)
        2. Escribir PLY coloreado para Blender
        3. Blender renderiza N vistas con pose exacta
        4. Sembrar gaussianas desde la malla
        5. Entrenar con gsplat (0.8*L1 + 0.2*(1-SSIM))
        6. Exportar PLY INRIA con ese degradado (no color medido)
        7. Devolver arrays + metricas

    Parametros:
        posiciones: (N, 3) vertices de la malla en mm
        caras: (F, 3) indices de caras
        rutas_fotos: rutas a las fotos intraorales (JPG/PNG)
        destino: directorio de salida para renders y PLY entrenado

    Devuelve:
        (arrays_dict, metricas): arrays del campo entrenado + metricas de calidad
    """
    if len(rutas_fotos) == 0:
        raise ValueError(
            "Se necesita al menos 1 foto intraoral para entrenar apariencia."
        )
    if len(posiciones) == 0:
        raise ValueError("La malla esta vacia: no hay vertices que sembrar.")

    destino.mkdir(parents=True, exist_ok=True)

    if reusa is not None:
        # ⚠️ REUSA: el color y el render de otra corrida del MISMO caso, tal cual. Existe
        # para comparar experimentos con identicos pixeles de entrada: el color por vertice
        # se relee del `scan_colored.ply` de la corrida origen y las vistas salen del mismo
        # `images/`, asi que la unica diferencia entre corridas es la configuracion del
        # entrenamiento. La cabecera del PLY declara el color como REUTILIZADO.
        scan_coloreado = Path(reusa) / "scan_colored.ply"
        if not (scan_coloreado.exists() and (Path(reusa) / "transforms.json").exists()):
            raise FileNotFoundError(
                f"reusa={reusa} no trae scan_colored.ply ni transforms.json"
            )
        T = json.loads((Path(reusa) / "transforms.json").read_text())
        vcol = _lee_ply_coloreado(scan_coloreado)
        if len(vcol) != len(posiciones):
            raise ValueError(
                f"el scan_colored.ply reutilizado trae {len(vcol):,} vertices y esta "
                f"malla tiene {len(posiciones):,}: no son del mismo escaneo"
            )
        por_pieza: list = []
        motivos_tono: list[str] = []
        medido = interpolado = n_por_pieza = 0
        print(f"Paso 1-2/4: color y render REUTILIZADOS de {reusa}")
    else:
        # Paso 1: Color desde fotos
        print("Paso 1/4: color desde las fotos...")
        enamel_rgb, gingiva_rgb = _muestra_color_fotos(rutas_fotos)
        print(f"  dos tonos de respaldo: esmalte {enamel_rgb.round().astype(int)} · "
              f"encía {gingiva_rgb.round().astype(int)}")
        respaldo = _colorea_malla(
            posiciones, np.zeros_like(posiciones), enamel_rgb, gingiva_rgb, etiquetas
        )

        # ⚠️ **Se intenta color MEDIDO antes que los dos tonos.** Proyectar los pixeles de la
        # foto sobre la malla necesita la pose de camara, y este proyecto tenia medido que no
        # sabia sacarla: COLMAP da CERO pares geometricos entre las fotos clinicas. `pose_foto`
        # la saca por PnP sobre correspondencias por diente —la Etapa 1 de DentalGS— y sobre un
        # caso real da 0,74 mm de error, con 4,31 sigma de separacion esmalte/encia sobre la
        # malla y dos piezas menos fuera de su caja anatomica a cobertura constante.
        #
        # Si no hay pose sostenible se cae al degradado de dos tonos, que es lo que habia, y se
        # dice. `medido` viaja para que el descriptor no tenga que suponerlo.
        # ⚠️ **Primero el color por PIEZA, que no necesita pose.** Proyectar por vertice deja
        # sin medida el 35 % de la superficie —solo una de seis fotos resuelve pose— y ese hueco
        # se rellena interpolando del vertice medido mas cercano, que en la cara vestibular es
        # una sombra interdental: es el color raro que se ve en el visor. Por pieza no hay hueco
        # que rellenar, y un vertice mal segmentado recibe el color de un diente vecino en vez
        # de una sombra.
        medido = interpolado = 0
        por_pieza, motivos_tono = [], []
        if etiquetas is not None and lado_fotos:
            try:
                from analysis_agents.dental import ancho_admitido

                from gaussian_engine.tono_foto import (
                    color_de_encia,
                    pinta_malla,
                    tonos_de_fotos,
                )

                arco = [f for f in (17, 16, 15, 14, 13, 12, 11,
                                    21, 22, 23, 24, 25, 26, 27)
                        if ancho_admitido(f) is not None]
                anchos = np.array([ancho_admitido(f) for f in arco], dtype=np.float64)
                por_pieza, motivos_tono = tonos_de_fotos(
                    list(rutas_fotos), arco, anchos, lado_conocido=lado_fotos
                )
                for m in motivos_tono:
                    print(f"    ⚠ {m}")
            except ImportError as e:
                print(f"  ⚠ sin color por pieza ({e})")

        if por_pieza and etiquetas is not None:
            eje = _eje_oclusal(posiciones, etiquetas)
            vcol_pieza, med_pieza = pinta_malla(
                posiciones, etiquetas, por_pieza, color_de_encia(list(rutas_fotos)), eje
            )
            print(f"  color por PIEZA: {len(por_pieza)} corona(s) medida(s), "
                  f"{100 * med_pieza.mean():.1f} % de los vertices sin interpolar nada")
            for t in por_pieza:
                print(f"    FDI {t.fdi}: {t.n_pixeles:,} px  "
                      + " ".join(f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in t.rgb))

        cm = None
        try:
            from gaussian_engine.pose_foto import color_por_vertice

            diag_por_foto: dict[str, dict] = {}
            if etiquetas is None:
                raise ValueError("el color por vertice necesita etiquetas FDI")
            cm = color_por_vertice(list(rutas_fotos), posiciones, caras, etiquetas,
                                   respaldo_rgb=respaldo, traza=traza,
                                   diag_por_foto=diag_por_foto)
            vcol = cm.rgb
            medido, interpolado = int(cm.medido.sum()), int(cm.interpolado.sum())
            print(f"  color MEDIDO: {cm.resumen()}")
            for pose in cm.poses:
                print(f"    {pose.ruta.name}: pose {pose.error_mm:.2f} mm · "
                      f"apoyo {100*pose.apoyo:.0f} % · "
                      f"{pose.inliers}/{pose.correspondencias}")
            for ruta, razon in cm.descartadas:
                print(f"    ✗ {ruta.name}: {razon}")
            # ⚠️ **El resultado de pose por foto SOLO existia en stdout**, y auditar un
            # fallo de pose sin los candidatos es conjetura. Aqui se persiste: por foto, el
            # motivo exacto del fallo (o la pose ganadora) y los candidatos parciales, que
            # ademas son las semillas del refinamiento de `pose_refina`.
            from PIL import Image

            for ruta in rutas_fotos:
                d = diag_por_foto.get(ruta.name, {"archivo": ruta.name,
                                                  "motivo": "sin diagnostico"})
                with Image.open(ruta) as im:
                    d["pixeles"] = list(im.size)
            (destino / "pose_diagnostico.json").write_text(
                json.dumps({"fotos": list(diag_por_foto.values())},
                           indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        except ImportError as e:
            vcol = respaldo
            print(f"  ⚠ sin color medido ({e}): se pinta el degradado de dos tonos, que NO "
                  "es color del paciente. Instala el extra `appearance`.")
        n_por_pieza = 0
        if por_pieza:
            # El color por pieza MANDA donde lo hay; el de por vertice se queda para el
            # resto, que es sobre todo lo que ninguna corona reclama.
            vcol = np.where(med_pieza[:, None], vcol_pieza, vcol)
            n_por_pieza = int(med_pieza.sum())
            # ⚠️ **Los interpolados hay que RECONTARLOS.** Los que la pieza reclamo estan
            # sobrescritos por color medido; seguir declarandolos exagera lo que se ha
            # inventado, y esta cabecera existe justo para no exagerar.
            interpolado = 0 if cm is None else int((~med_pieza & cm.interpolado).sum())
            medido = 0 if cm is None else int((~med_pieza & cm.medido).sum())
        elif medido == 0:
            print("  ⚠ ninguna foto ha dado una pose sostenible: el color son DOS TONOS y "
                  "su frontera sale de las etiquetas FDI, que es inferencia (Layer 3).")

        # Paso 2: Render Blender
        print("Paso 2/4: renderizando vistas Blender...")
        scan_coloreado = destino / "scan_colored.ply"
        escribe_ply_coloreado(scan_coloreado, posiciones, caras, vcol)

        T = _render_blender(
            scan_coloreado, destino,
            n_vistas=n_vistas, resolucion=resolucion,
            script_blender=script_blender, cerca=cerca,
        )

    # Paso 3: Entrenar gsplat
    print("Paso 3/4: entrenando 3DGS...")
    # ⚠️ Con `reusa`, las vistas se leen del directorio de la corrida origen: el `destino`
    # de esta solo recibe el PLY nuevo, y las imagenes no se tocan ni se copian.
    render_dir = destino if reusa is None else Path(reusa)

    # Paso 2b: mapas de profundidad por raycast, si la supervision los pide y el render no
    # los trae ya. Se generan UNA vez por render y se reusan entre experimentos, igual que
    # las propias vistas. Ver `_profundidad_raycast`: la pasada Z de Blender 5.x ya no es
    # legible, y el dato es el mismo —la primera superficie visible desde cada camara—.
    if peso_profundidad > 0 and not (render_dir / "depth").exists():
        _profundidad_raycast(T, render_dir, posiciones, caras, traza=traza)
    params, curva = _entrena_gsplat(
        T, posiciones, vcol,
        destino=render_dir,
        iteraciones=iteraciones, semillas=semillas,
        caras=caras, siembra=siembra,
        dispositivo=dispositivo, traza=traza or traza,
        estrategia=estrategia, antialiased=antialiased,
        peso_profundidad=peso_profundidad,
        profundidad_desde=profundidad_desde, profundidad_rampa=profundidad_rampa,
        peso_aplanado=peso_aplanado, aplanado_desde=aplanado_desde,
    )
    if reusa is not None:
        # La cabecera del PLY declara que este color NO se proyecto en esta corrida: se
        # reutilizo el de otra. Ver `_comentarios_color`.
        params["color_reutilizado"] = np.asarray(1, dtype=np.int64)

    # Paso 4: Exportar PLY INRIA
    print("Paso 4/4: exportando PLY...")
    ply_path = destino / "apariencia.ply"
    # ⚠️ **El FDI por gaussiana, por vecino mas cercano.** El optimizador movio, dividio y
    # podo, asi que no hay correspondencia con los vertices que heredar. Se pregunta la
    # etiqueta de la corona mas cercana y se dice asi en la cabecera del PLY: es exacto y
    # auditable, y no finge una etiqueta aprendida que no existe.
    #
    # Las gaussianas estan en el espacio NORMALIZADO y los vertices en mm, asi que la
    # consulta se hace llevando los vertices al mismo espacio — no al reves, para no
    # depender de que la des-normalizacion ya se haya aplicado.
    if etiquetas is not None and posiciones is not None:
        etq = np.asarray(etiquetas)
        if len(etq) == len(posiciones):
            # ⚠️ **Contra TODOS los vertices, no solo contra las coronas.** Consultando solo
            # las coronas, cada gaussiana hereda el FDI de la corona mas proxima **este
            # donde este**: salio el 100 % de las gaussianas marcadas como diente y ni una
            # como encia. Y para encender una pieza eso es peor que no tener etiqueta,
            # porque encenderia tambien el trozo de encia mas cercano a ella.
            #
            # Preguntando a todos, una gaussiana sobre la encia hereda el 0 de la encia,
            # que es la respuesta correcta.
            V = np.asarray(posiciones, dtype=np.float64)
            Vn = (V - np.asarray(T["scan_offset"], dtype=np.float64)) * float(T["scan_scale"])
            reg, lejos, cambiadas = _fdi_por_gaussiana(
                np.asarray(params["means"], dtype=np.float64), Vn, etq,
                limite=LIMITE_VECINO * float(T["scan_scale"]),
            )
            params["region_id"] = reg
            # ⚠️ **El relieve se calcula aqui, con la malla delante, y viaja en `params`.**
            # Mismo criterio que el `region_id`: la normal sale del vertice MAS CERCANO,
            # porque el optimizador movio, dividio y podo y no hay correspondencia 1:1 que
            # heredar. Decir «vecino mas cercano» es exacto y auditable.
            nrm, sh1 = _relieve_sh1(
                np.asarray(params["means"], dtype=np.float64), Vn,
                _normales_de_malla(Vn, np.asarray(caras, dtype=np.int64)),
                np.clip(np.asarray(params["colors"], dtype=np.float64), 0.0, 1.0)
                if "colors" in params else None,
                eje_rasante=_eje_oclusal(posiciones, etiquetas),
            )
            if sh1 is not None:
                params["normales"], params["sh1"] = nrm, sh1
                # La oclusion se mide en MILIMETROS, asi que se calcula sobre la nube sin
                # normalizar: `RADIO_OCLUSION_MM` no significa nada en el espacio de Blender.
                escala = float(T["scan_scale"])
                malla_mm = Vn / escala
                normales_malla = _normales_de_malla(Vn, np.asarray(caras, dtype=np.int64))
                # Se calcula en la malla y se lleva a cada gaussiana por el vertice mas
                # cercano, igual que la normal y el `region_id`.
                ao_malla = _oclusion_ambiental(malla_mm, normales_malla)
                from scipy.spatial import cKDTree as _KD
                ao = ao_malla[_KD(malla_mm).query(
                    np.asarray(params["means"], dtype=np.float64) / escala, k=1
                )[1]]
                params["ao"] = ao
                print(f"  relieve SH1 calculado sobre {len(sh1):,} gaussianas "
                      f"(fuerza {abs(FUERZA_RELIEVE)}, rasante {RASANTE_GRADOS:.0f}°) "
                      f"+ oclusion (mediana {np.median(ao):.2f}, "
                      f"{(ao <= OCLUSION_MINIMA + 1e-6).mean() * 100:.0f}% al suelo): "
                      "el color no se toca")
            print(f"  region_id por gaussiana: "
                  f"{len([c for c in set(reg.tolist()) if c > 0])} codigo(s) FDI · "
                  f"{(reg > 0).mean() * 100:.0f}% diente · "
                  f"{lejos.mean() * 100:.1f}% sin vecino a menos de {LIMITE_VECINO} mm · "
                  f"voto de {VOTOS_FDI}: {cambiadas:,} distinta(s) del vecino mas cercano")

    # ⚠️ **Todo lo que hace falta para reescribir el PLY viaja en `params`.** El PLY lo
    # escriben DOS sitios —este y el agente de UOS, desde el almacen— y el segundo solo ve
    # lo que el almacen guarde. La des-normalizacion y el FDI por gaussiana se perdieron
    # ahi, cada uno una vez, porque `store.put()` enumeraba claves a mano. Metiendolos en
    # el dict, guardar `**params` los lleva sin que nadie tenga que acordarse.
    params["scan_scale"] = np.asarray(T["scan_scale"], dtype=np.float64)
    params["scan_offset"] = np.asarray(T["scan_offset"], dtype=np.float64)
    # ⚠️ **De donde salio el color, en numeros y dentro del artefacto.** Sin esto el
    # descriptor tendria que SUPONER si el `f_dc_*` es color medido o los dos tonos de
    # respaldo, y suponer es lo que ya hizo que el contenedor declarase «color real del
    # paciente» sobre un degradado inventado. Con esto lo lee.
    params["n_vertices_medidos"] = np.asarray(medido, dtype=np.int64)
    params["n_vertices_por_pieza"] = np.asarray(n_por_pieza, dtype=np.int64)
    params["n_piezas_con_tono"] = np.asarray(len(por_pieza), dtype=np.int64)
    params["n_vertices_interpolados"] = np.asarray(interpolado, dtype=np.int64)
    params["n_vertices_malla"] = np.asarray(len(posiciones), dtype=np.int64)

    # Inyectar metadatos en el dict para que `escribe_inria` y el store los tengan.
    n_vistas_reales = len(T["frames"])
    params["n_vistas"] = np.array(n_vistas_reales, dtype=np.int32)
    params["iteraciones"] = np.array(iteraciones, dtype=np.int32)
    # ⚠️ La des-normalizacion viaja AQUI y no dentro del entrenamiento: gsplat optimiza en
    # el espacio normalizado de Blender y tiene que seguir haciendolo —es donde estan las
    # camaras—. Lo que no puede es salir de esta funcion sin deshacerse, porque el PLY
    # declara milimetros. `escribe_inria` sin estos dos argumentos escribe un fichero que
    # miente sobre sus unidades; ver su docstring.
    escribe_inria(
        ply_path, params,
        n_vistas=n_vistas_reales,
        iteraciones=iteraciones,
        scan_scale=float(T["scan_scale"]),
        scan_offset=T.get("scan_offset"),
    )

    # Metricas finales
    psnr_final = curva[-1][3] if curva else 0.0
    # SSIM final: calcular sobre las vistas de holdout
    ssim_final = _calcula_ssim_final(params, T, destino, curva)

    metricas = EntrenamientoApariencia(
        psnr_db=psnr_final,
        ssim=ssim_final,
        n_gaussianas=params["means"].shape[0],
        iteraciones=iteraciones,
        loss_curva=curva,
        tiempo_s=sum(c[1] for c in curva) if curva else 0.0,
        n_vistas_train=len(T["frames"]) * 3 // 4,
        n_vistas_holdout=len(T["frames"]) // 4,
        resolucion=resolucion,
        tonos=list(por_pieza),
        motivos=list(motivos_tono),
    )

    return params, metricas


def _calcula_ssim_final(
    params: dict[str, np.ndarray],
    T: dict,
    destino: Path,
    curva: list[tuple[int, float, float, float]],
) -> float:
    """Calcula SSIM final sobre vistas de validacion."""
    try:
        import torch
        from gsplat import rasterization
        from PIL import Image
        from torchmetrics.functional import structural_similarity_index_measure as ssim

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        W = H = T["w"]
        fx = 0.5 * W / math.tan(0.5 * T["camera_angle_x"])
        K = torch.tensor(
            [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
            dtype=torch.float32, device=dev,
        )
        flip = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        viewmats = torch.stack([
            torch.linalg.inv(
                torch.tensor(f["transform_matrix"], dtype=torch.float32) @ flip
            )
            for f in T["frames"]
        ]).to(dev)

        means = torch.tensor(params["means"], dtype=torch.float32, device=dev)
        quats = torch.tensor(params["quats"], dtype=torch.float32, device=dev)
        scales = torch.tensor(params["scales"], dtype=torch.float32, device=dev)
        opacities = torch.tensor(params["opacities"], dtype=torch.float32, device=dev)
        colors = torch.tensor(params["colors"], dtype=torch.float32, device=dev)

        idx = np.arange(len(T["frames"]))
        test = idx[::4]
        rng = np.random.default_rng(1)
        eval_test = rng.choice(test, min(MUESTRAS_VALIDACION, len(test)), replace=False)

        ssims = []
        for j in eval_test:
            rgba = np.asarray(
                Image.open(destino / T["frames"][j]["file_path"]).convert("RGBA"),
                np.float32,
            ) / 255.0
            tgt = torch.from_numpy(rgba[..., :3] * rgba[..., 3:4]).to(dev)
            with torch.no_grad():
                out, _, _ = rasterization(
                    means,
                    torch.nn.functional.normalize(quats, dim=-1),
                    torch.exp(scales),
                    torch.sigmoid(opacities),
                    torch.sigmoid(colors),
                    viewmats[j : j + 1], K[None], W, H, packed=False,
                )
            a = out[0].permute(2, 0, 1).unsqueeze(0)
            b = tgt.permute(2, 0, 1).unsqueeze(0)
            ssims.append(float(ssim(a, b, data_range=1.0)))

        return float(np.mean(ssims))
    except Exception:
        # Si falla (gsplat no disponible, etc.), devolver un valor conservador
        return 0.0
