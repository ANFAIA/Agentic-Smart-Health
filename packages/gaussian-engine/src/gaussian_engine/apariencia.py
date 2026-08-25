"""Reconstruccion de apariencia real desde fotos intraorales con 3D Gaussian Splatting.

Este modulo implementa el flujo completo demostrado en `notebooks/07`:

    mesh-agent (STL) + image-agent (fotos)
        → Blender (EEVEE, N vistas con pose exacta)
        → gsplat (optimizacion contra renders, 0.8*L1 + 0.2*(1-SSIM))
        → PLY en formato INRIA con color real del paciente

**Que NO es.** No es el ajuste de densidad del CBCT (ese es `ajuste.py`). Aqui se
optimiza contra **imagenes 2D** (renders de Blender) porque el objetivo es la
**apariencia visual**, no la densidad fisica. La representacion es la misma —gaussianas
anisotropas— pero la funcion de perdida, los datos de entrada y la semantica del
resultado son distintos.

**Perfil `'ash-gs-apariencia/1.0'`.** Declara que el campo fue entrenado con gsplat
sobre renders EEVEE, con color real de fotos. Un lector que vea este perfil sabe que:

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
) -> np.ndarray:
    """Aplica color por altura z: coronas (z alto) = esmalte, margen (z bajo) = encia.

    Devuelve (N, 3) uint8.
    """
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


def escribe_inria(
    destino: Path,
    params: dict[str, np.ndarray],
    *,
    perfil: str = PERFIL,
    n_vistas: int | None = None,
    iteraciones: int | None = None,
    acquisition_id: str = "",
) -> None:
    """PLY binario en formato INRIA 3DGS (grado 0 de armonicos esfericos).

    Propiedades:
    - x, y, z: double (float64) - centro de la gaussiana en mm
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
    rot = params["quats"].astype(np.float64)   # ya normalizado
    opa = params["opacities"].astype(np.float64)  # ya en logit

    # Leer metadatos del dict si no se pasaron como argumento.
    if n_vistas is None:
        n_vistas = int(params["n_vistas"]) if "n_vistas" in params else 0
    if iteraciones is None:
        iteraciones = int(params["iteraciones"]) if "iteraciones" in params else 0

    # Color RGB → SH DC coefficient: f_dc = (color - 0.5) / C0
    f_dc = (col - 0.5) / C0

    cabecera = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment generado por gaussian-engine@{perfil}",
        f"comment acquisition_id {acquisition_id}",
        "comment perfil INRIA 3DGS grado 0 - APARIENCIA, no medida",
        "comment las gaussianas NO son los vertices del escaner: el optimizador las",
        "comment movió, dividió y podó. No hay correspondencia 1:1 con lo medido.",
        "comment f_dc_* = color RGB real del paciente (coeficiente DC de SH)",
        "comment opacity = opacidad de visualización (logit), NO es atenuación radiológica",
        "comment scale en logaritmo (convención INRIA), NO en mm lineales",
        "comment rot es cuaternion (w,x,y,z) normalizado",
        f"comment entrenado contra {n_vistas} renders EEVEE, {iteraciones} iteraciones",
        f"element vertex {n}",
        *(f"property float {p}" for p in PROPIEDADES_INRIA),
        "end_header",
    ]

    # Construir array estructurado
    dtype = np.dtype([
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    filas = np.empty(n, dtype=dtype)
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


# ---------------------------------------------------------------------------
# Funcion principal: snapshot-in, snapshot-out
# ---------------------------------------------------------------------------

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
) -> tuple[dict[str, np.ndarray], EntrenamientoApariencia]:
    """Entrena un campo de gaussianas con color real desde fotos intraorales.

    Flujo completo:
        1. Muestrear color de fotos (esmalte/encía por altura)
        2. Escribir PLY coloreado para Blender
        3. Blender renderiza N vistas con pose exacta
        4. Sembrar gaussianas desde la malla
        5. Entrenar con gsplat (0.8*L1 + 0.2*(1-SSIM))
        6. Exportar PLY INRIA con color real
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
    print("Paso 1/4: muestreando color de fotos...")
    enamel_rgb, gingiva_rgb = _muestra_color_fotos(rutas_fotos)
    print(f"  esmalte {enamel_rgb.round().astype(int)} · "
          f"encía {gingiva_rgb.round().astype(int)} · "
          f"{len(rutas_fotos)} fotos")

    # Paso 2: Colorear malla y render Blender
    print("Paso 2/4: renderizando vistas Blender...")
    vcol = _colorea_malla(posiciones, np.zeros_like(posiciones), enamel_rgb, gingiva_rgb)
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
    # Inyectar metadatos en el dict para que `escribe_inria` y el store los tengan.
    n_vistas_reales = len(T["frames"])
    params["n_vistas"] = np.array(n_vistas_reales, dtype=np.int32)
    params["iteraciones"] = np.array(iteraciones, dtype=np.int32)
    escribe_inria(
        ply_path, params,
        n_vistas=n_vistas_reales,
        iteraciones=iteraciones,
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
