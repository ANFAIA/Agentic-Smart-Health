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

**Perfil `'ash-gs-apariencia/1.0'`.** Declara que el campo fue entrenado con gsplat
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constantes medidas (ver notebook 07 y entrena_gs_escaner.py)
# ---------------------------------------------------------------------------

# Perfil de este modulo. Distingue este campo del semilla (`ash-twin/1.0`) y del
# ajustado (`ash-twin-ajustado/1.0`).
PERFIL = "ash-gs-apariencia/1.0"

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


# ---------------------------------------------------------------------------
# Entrenamiento gsplat
# ---------------------------------------------------------------------------

def _entrena_gsplat(
    T: dict,
    posiciones: np.ndarray,
    colores_rgb: np.ndarray,
    *,
    destino: Path,
    iteraciones: int = ITERACIONES,
    semillas: int = SEMILLAS,
    dispositivo: str = "cuda",
    traza: bool = False,
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
    sel = rng.choice(len(P), min(semillas, len(P)), replace=False)
    means0 = torch.tensor(P[sel], device=dev)
    n0 = len(sel)

    # Sigma inicial: distancia media a los 4 vecinos mas cercanos.
    d = torch.cdist(means0[:2000], means0[:2000]).topk(4, largest=False).values[:, 1:].mean()

    # Color inicial: el muestreado de las fotos, no un gris constante.
    col0 = torch.tensor(
        colores_rgb[sel].astype(np.float32) / 255.0, device=dev
    ).clamp(0.01, 0.99)

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

    def carga_imagen(k: int) -> torch.Tensor:
        """Carga imagen bajo demanda (a 1024px las 1600 son ~20 GB)."""
        rgba = np.asarray(
            Image.open(destino / T["frames"][k]["file_path"]).convert("RGBA"),
            np.float32,
        ) / 255.0
        return torch.from_numpy(rgba[..., :3] * rgba[..., 3:4]).to(dev)

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
        )
        return out[0], info

    def psnr_en(js: np.ndarray) -> float:
        with torch.no_grad():
            return float(np.mean([
                -10 * math.log10(
                    ((render(viewmats[j])[0] - carga_imagen(j)) ** 2).mean().item()
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

    # Control adaptativo de densidad (Kerbl et al. 2023):
    # - densificacion suave (la siembra desde la malla ya es densa)
    # - poda de gaussianas casi transparentes
    # - reset periodico de opacidad
    strat = DefaultStrategy(
        refine_start_iter=500,
        refine_stop_iter=iteraciones // 2,
        reset_every=1500,
        refine_every=100,
        verbose=False,
    )
    strat.check_sanity(params, opt)
    estado = strat.initialize_state(scene_scale=1.0)

    torch.manual_seed(0)
    curva: list[tuple[int, float, float, float]] = []
    t0 = time.perf_counter()

    for it in range(iteraciones):
        i = int(np.random.choice(train))
        out, info = render(viewmats[i])
        strat.step_pre_backward(params, opt, estado, it, info)

        tgt = carga_imagen(i)
        a = out.permute(2, 0, 1).unsqueeze(0)
        b = tgt.permute(2, 0, 1).unsqueeze(0)
        perdida = 0.8 * (out - tgt).abs().mean() + 0.2 * (1.0 - ssim(a, b, data_range=1.0))

        for o in opt.values():
            o.zero_grad()
        perdida.backward()
        for o in opt.values():
            o.step()
        strat.step_post_backward(params, opt, estado, it, info, packed=False)

        if it % 500 == 0 or it == iteraciones - 1:
            ptr = psnr_en(train) if len(train) > 0 else 0.0
            pt = psnr_en(eval_test)
            curva.append((it, perdida.item(), ptr, pt))
            if traza:
                print(
                    f"  it {it:5d}  N={params['means'].shape[0]:>7,}  "
                    f"loss {perdida.item():.4f}  train {ptr:.2f}  holdout {pt:.2f} dB"
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

# Propiedades INRIA 3DGS grado 0 (las mismas que usa el visor web).
PROPIEDADES_INRIA = (
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
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
    filas["nx"] = 0.0
    filas["ny"] = 0.0
    filas["nz"] = 0.0
    filas["f_dc_0"] = f_dc[:, 0].astype(np.float32)
    filas["f_dc_1"] = f_dc[:, 1].astype(np.float32)
    filas["f_dc_2"] = f_dc[:, 2].astype(np.float32)
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
    dispositivo: str = "cuda",
    script_blender: Path | None = None,
    traza: bool = False,
    # FDI por vertice del escaneo. Si viene, cada gaussiana sale con el codigo de la corona
    # mas cercana, y entonces se puede seleccionar una pieza SIN tener la malla delante.
    etiquetas: np.ndarray | None = None,
    # ⚠️ **El bit que ninguna medida puede dar.** Para cada foto, el codigo FDI de su
    # PRIMERA corona. El arco es un espejo exacto y una foto intraoral puede estar tomada
    # con espejo, asi que ni la huella de anchuras ni la imagen dicen de que lado es una
    # tira (ver `tono_foto.alinea_con_el_arco`). Sin esto no se usa color por pieza y se
    # cae al de por vertice; con esto, el resto sigue siendo medido.
    lado_fotos: dict[Path, int] | None = None,
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
    por_pieza: list = []
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

    if por_pieza:
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

        cm = color_por_vertice(list(rutas_fotos), posiciones, caras, etiquetas,
                               respaldo_rgb=respaldo, traza=traza)
        vcol = cm.rgb
        medido, interpolado = int(cm.medido.sum()), int(cm.interpolado.sum())
        print(f"  color MEDIDO: {cm.resumen()}")
        for pose in cm.poses:
            print(f"    {pose.ruta.name}: pose {pose.error_mm:.2f} mm · "
                  f"apoyo {100*pose.apoyo:.0f} % · {pose.inliers}/{pose.correspondencias}")
        for ruta, razon in cm.descartadas:
            print(f"    ✗ {ruta.name}: {razon}")
    except ImportError as e:
        vcol = respaldo
        print(f"  ⚠ sin color medido ({e}): se pinta el degradado de dos tonos, que NO es "
              "color del paciente. Instala el extra `appearance`.")
    n_por_pieza = 0
    if por_pieza:
        # El color por pieza MANDA donde lo hay; el de por vertice se queda para el resto,
        # que es sobre todo lo que ninguna corona reclama.
        vcol = np.where(med_pieza[:, None], vcol_pieza, vcol)
        n_por_pieza = int(med_pieza.sum())
        # ⚠️ **Los interpolados hay que RECONTARLOS.** Los que la pieza reclamo estan
        # sobrescritos por color medido; seguir declarandolos exagera lo que se ha
        # inventado, y esta cabecera existe justo para no exagerar.
        interpolado = 0 if cm is None else int((~med_pieza & cm.interpolado).sum())
        medido = 0 if cm is None else int((~med_pieza & cm.medido).sum())
    elif medido == 0:
        print("  ⚠ ninguna foto ha dado una pose sostenible: el color son DOS TONOS y su "
              "frontera sale de las etiquetas FDI, que es inferencia (Layer 3).")

    # Paso 2: Render Blender
    print("Paso 2/4: renderizando vistas Blender...")
    scan_coloreado = destino / "scan_colored.ply"
    escribe_ply_coloreado(scan_coloreado, posiciones, caras, vcol)

    T = _render_blender(
        scan_coloreado, destino,
        n_vistas=n_vistas, resolucion=resolucion,
        script_blender=script_blender,
    )

    # Paso 3: Entrenar gsplat
    print("Paso 3/4: entrenando 3DGS...")
    params, curva = _entrena_gsplat(
        T, posiciones, vcol,
        destino=destino,
        iteraciones=iteraciones, semillas=semillas,
        dispositivo=dispositivo, traza=traza or traza,
    )

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
