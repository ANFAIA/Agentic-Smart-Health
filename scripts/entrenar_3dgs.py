#!/usr/bin/env python
"""entrenar_3dgs.py — EXPERIMENTO con resultado NEGATIVO: 3DGS entrenado de una arcada.

    ~/.venvs/dental-gpu/bin/python scripts/entrenar_3dgs.py \
        --escaneo A.stl --salida DIR --ply campo.ply

⚠️ **Esto NO es parte de ningún pipeline y no debe cablearse a ninguno.** Se conserva
porque un resultado negativo sin código que lo reproduzca es una opinión y no una
medida, y porque de camino dejó dos cosas que sí valen: la ablación de la luz rasante
y el descarte de la resolución de render como cuello de botella.

**El veredicto, arriba del todo:** para una imagen que un clínico va a mirar, entrenar
sale **peor** que no entrenar. La alternativa que gana son los discos tangentes de
`seguimiento_histora.py`, que cierran la superficie sin mover un solo vértice, en
segundos en vez de treinta y nueve minutos.

⚠️ **No se ejecuta con `uv run`.** Necesita torch con CUDA y `gsplat`, que viven en el
entorno dedicado `~/.venvs/dental-gpu` (RTX 5070, sm_120, torch cu128) y no en el del
proyecto — un `uv sync` los arrasaría. Los paquetes del monorepo se cargan por ruta,
así que ese entorno los ve sin instalarlos.

## Qué gana esto y qué NO

El camino es: malla → Blender renderiza N vistas con pose exacta → 3DGS aprende de esas
vistas. Así que **el techo lo pone la malla**: el campo aprende a reproducir un render
de nuestra geometría y no puede contener un detalle que el escáner no midió. Si lo que
se busca es resolución anatómica, el dinero está en el escaneo, no aquí. Está medido:
el espaciado mediano de `histora` inferior es 0,138 mm y ninguna iluminación lo baja.

Lo que gana de verdad, después de medirlo, es **solo esto**:

- **Sombreado dependiente de la vista**, en los armónicos esféricos, en vez del Lambert
  envolvente que `seguimiento_histora.py` hornea a mano y declara falso. Es la única
  ventaja que sobrevive, y es estética: no mide nada.
- **La ablación de la luz rasante**, que era la técnica que se quería probar y que
  quedó cuantificada (ver abajo). El conocimiento se quedó en
  `blender_render_views.py`, que es donde se reutiliza.

Y lo que **creía que ganaba y no gana**, porque conviene que quede escrito:

- ❌ **«Superficie continua».** Era el motivo principal de montar esto. El campo naïve
  de una gaussiana isótropa por vértice enseñaba el punteado, y el entrenamiento lo
  cerraba estirando y transparentando las gaussianas (anisotropía mediana 1,0 → 3,9;
  opacidad 0,98 → 0,047). Pero el punteado no era un problema de *representación*: era
  una σ mal elegida. Con σ sacada del espaciado local y la gaussiana aplanada contra la
  superficie, los huecos pasan del 20,45 % al 0,34 % **sin entrenar nada**. Y el
  entrenamiento cobra un precio que los discos no: reparte gaussianas anchas a lo largo
  de la superficie —σ mayor 216 µm— y con eso lava la anatomía. En un molar de cerca se
  ve superficie continua y detalle borrado.
- ❌ **Nitidez.** Medida como es debido —energía de gradiente **solo donde alfa > 0,99**,
  para que los propios agujeros no la inflen— el campo entrenado no gana nada.

## La luz, que es la decisión de fondo

La intuición de partida era pasear una luz por la superficie para sacar detalle, como
en un render de Blender. Eso, tal cual, **rompe el 3DGS**: el modelo representa el color
como función de la *dirección de vista*, así que si la luz se mueve por su cuenta el
mismo punto visto desde el mismo sitio cambia de brillo entre fotogramas, el modelo no
puede representarlo y lo promedia — sale lavado. La técnica que sí recupera relieve
moviendo la luz es la **estereofotometría**, con la cámara quieta, y no devuelve un
campo de radiancia sino normales: es otro producto.

La versión que conserva el efecto sin romper nada: la luz **solidaria con la cámara**,
pero **separada de su eje** (`--raking-deg`). Solidaria, la iluminación es función de la
vista y el modelo la absorbe; separada del eje, no es un frontal plano —un frontal puro
tiene n·l ≈ n·v y borra el relieve— sino una rasante que alarga los gradientes. Va
montada en `blender_render_views.py`.

## Subir la resolución del render NO sirve — medido, y en contra de lo que parecía

Razonamiento tentador: a 800 px las vistas muestrean a 126 µm/px y hacen falta dos
píxeles por rasgo, o sea 252 µm, mientras que la malla tiene 138 µm de espaciado; luego
el render sería el cuello de botella. **Es falso.** Mismo modo, misma receta, mismas
9000 iteraciones, cambiando solo la resolución:

    800 px    188.219 gaussianas   σ menor 36,1 µm   σ mayor 208,1 µm   nitidez 8,10
    2048 px   163.132 gaussianas   σ menor 42,6 µm   σ mayor 216,5 µm   nitidez 8,27

A 2048 px las gaussianas salen **más gordas** y hay menos, por 11 veces más tiempo (39
minutos frente a 3,5) y un 2 % de nitidez, que es ruido. Dos razones:

1. **Son 1200 vistas, no una.** La información agregada desde muchos ángulos recupera
   detalle muy por debajo del píxel de una sola imagen: es triangulación multivista, el
   principio por el que la fotogrametría saca precisión submilimétrica de fotos
   normales. El límite de una imagen suelta no es el límite del conjunto.
2. **El eje que emborrona no cambia.** El σ *mayor* —gaussianas anchas a lo largo de la
   superficie— se queda en ~210 µm en los dos casos. Es una propiedad de cómo el 3DGS
   reparte gaussianas, no de lo que se le dé de comer. Subir la resolución afina el eje
   que ya estaba bien y deja intacto el que estropea.

Por eso `--res` sigue en 800 por defecto. Y por eso, para una imagen que un clínico va a
mirar, la vía buena no era entrenar más sino no entrenar: los discos tangentes de
`seguimiento_histora.py` dejan cada gaussiana donde la puso el escáner.

## Reversibilidad

Blender normaliza la malla a una caja unidad para que el encuadre no dependa de los mm
del caso. El campo se devuelve a milímetros al exportarlo, con el `scan_scale` y el
`scan_offset` que `transforms.json` guarda para eso. Importa: las cotas de
`seguimiento_histora.py` están en mm, y un campo en otra escala no se puede anotar.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RESUMEN_EN = "Negative experiment: trains 3DGS from a dental arch."

RAIZ = Path(__file__).resolve().parent.parent
for _paquete in ("ingestion-agents", "fusion-agents", "analysis-agents", "core-schemas"):
    sys.path.insert(0, str(RAIZ / f"packages/{_paquete}/src"))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))

from ingestion_agents import ArtifactStore, MeshAgent  # noqa: E402

_spec = importlib.util.spec_from_file_location("sh", RAIZ / "scripts/seguimiento_histora.py")
sh = importlib.util.module_from_spec(_spec)
# Registrarlo ANTES de ejecutarlo. `module_from_spec` no lo mete en `sys.modules`, y
# `@dataclass` resuelve sus anotaciones mirando ahí: sin esto, el `Margen` de
# `seguimiento_histora` revienta con un AttributeError sobre None que no dice nada.
sys.modules["sh"] = sh
_spec.loader.exec_module(sh)

_C0 = 0.28209479177387814  # armónico esférico de grado 0

# Grado 2 = 9 coeficientes por canal. Es lo máximo que carga el visor (`shGrado: 0|1|2`)
# y es donde vive el efecto de la rasante: con la luz solidaria a la cámara, el brillo
# ES función de la dirección de vista, así que hay algo real que estos coeficientes
# pueden aprender. Con grado 0 el entrenamiento cerraría los huecos pero devolvería una
# superficie mate, y la mitad del motivo de entrenar se perdería.
GRADO_SH = 2
SUBIR_SH_CADA = 1000  # el grado sube por etapas: empezar por todo desestabiliza

# Receta de Kerbl et al. 2023, la misma de los notebooks 04/06/07/08.
PESO_L1 = 0.8  # el resto va a (1 − SSIM)

# Supervisión del canal alfa. Sin esto salían gaussianas NEGRAS FLOTANDO sobre la
# arcada —medido: 235 por encima del punto más alto de la malla, con color mediano
# (0,0,0)— que desde algunas vistas se cruzan delante y parecen una sombra colgada.
# El motivo: las vistas van compuestas sobre negro, así que una gaussiana que se va al
# vacío y se pone negra no paga nada en la pérdida y la densificación la conserva. El
# fondo acaba modelado como si fuera geometría.
#
# Blender renderiza con `film_transparent`, o sea que el alfa dice EXACTAMENTE dónde no
# hay nada. Comparar el alfa renderizado contra ese es decírselo al modelo directamente,
# que es mejor que borrar los flotantes después: no llegan a formarse.
PESO_ALFA = 0.5

# Poda geométrica, como red de seguridad de lo anterior. Aquí la malla ES la verdad
# sobre dónde está la superficie, así que una gaussiana a más de esta distancia no
# representa nada real. Se declara porque es una intervención sobre el resultado.
UMBRAL_FLOTANTE_MM = 3.0
TASAS = {"means": 1e-3, "scales": 5e-3, "quats": 1e-3,
         "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}


# --------------------------------------------------------------------------- #
# 1 · La malla con su color, para que Blender la ilumine
# --------------------------------------------------------------------------- #
def escribe_malla_color(V: np.ndarray, F: np.ndarray, color: np.ndarray, destino: Path) -> None:
    """Malla + color por vértice en PLY binario, que es lo que importa Blender.

    ⚠️ El color va **sin sombrear**. `seguimiento_histora.py` hornea un Lambert
    envolvente porque allí no hay nadie que ilumine; aquí ilumina Blender, y hornear
    además el sombreado propio lo aplicaría dos veces: las zonas en sombra se cerrarían
    a negro y lo que ahí se pierda no lo aprende ningún entrenamiento.
    """
    v = np.zeros(len(V), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = V[:, 0], V[:, 1], V[:, 2]
    rgb = np.clip(color * 255.0, 0, 255).astype(np.uint8)
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    caras = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4"), ("j", "<i4"), ("k", "<i4")])
    caras["n"] = 3
    caras["i"], caras["j"], caras["k"] = F[:, 0], F[:, 1], F[:, 2]
    cabecera = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(V)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(F)}\n"
        "property list uchar int vertex_index\n"
        "end_header\n"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("wb") as f:
        f.write(cabecera.encode("ascii"))
        f.write(v.tobytes())
        f.write(caras.tobytes())


# --------------------------------------------------------------------------- #
# 2 · Las vistas
# --------------------------------------------------------------------------- #
def estirado_global(carpeta: Path, T: dict, muestras: int = 48) -> tuple[float, float]:
    """Percentiles 1 y 99 de la luminancia DENTRO de la máscara, sobre toda la tanda.

    Estira el rango a blanco y negro para que el relieve se lea al máximo. La clave está
    en que el estirado es **uno solo para todas las vistas**: normalizar cada imagen por
    sus propios percentiles haría que el mismo punto de la superficie saliera con
    distinto brillo según desde dónde se mire, que es exactamente el fallo que se evita
    atando la luz a la cámara — el modelo no puede representar eso y lo promedia.

    Se toman percentiles y no el mínimo y el máximo porque un solo píxel extremo
    (un brillo especular, un pico de ruido) fijaría el rango por sí solo.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    js = rng.choice(len(T["frames"]), min(muestras, len(T["frames"])), replace=False)
    valores = []
    for j in js:
        a = np.asarray(Image.open(carpeta / T["frames"][j]["file_path"]).convert("RGBA"),
                       np.float32) / 255.0
        m = a[..., 3] > 0.5
        if m.sum() > 200:
            valores.append(a[..., :3].mean(-1)[m])
    v = np.concatenate(valores)
    return float(np.percentile(v, 1)), float(np.percentile(v, 99))


def render_vistas(malla: Path, destino: Path, vistas: int, res: int,
                  raking: float, anillos: int, gris: float | None = None,
                  ambiente: float = 0.4) -> dict:
    """Llama a Blender headless. Las poses salen de su grafo de escena: son exactas.

    Por eso este camino no necesita COLMAP ni structure-from-motion, que es donde un
    pipeline 3DGS normal acumula el error que luego no se puede separar del modelo.
    """
    orden = ["blender", "--background", "--python",
             str(RAIZ / "scripts/blender_render_views.py"), "--",
             "--scan", str(malla), "--out", str(destino),
             "--views", str(vistas), "--res", str(res), "--samples", "16",
             "--elevations", str(anillos), "--raking-deg", str(raking),
             "--ambiente", str(ambiente)]
    if gris is not None:
        orden += ["--gris", str(gris)]
    t = time.perf_counter()
    r = subprocess.run(orden, capture_output=True, text=True)
    if not (destino / "transforms.json").exists():
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("Blender no produjo transforms.json")
    T = json.loads((destino / "transforms.json").read_text())
    print(f"blender  {len(T['frames'])} vistas @ {res}px · rasante {raking}° · "
          f"{time.perf_counter() - t:.0f} s")
    return T


# --------------------------------------------------------------------------- #
# 3 · El entrenamiento
# --------------------------------------------------------------------------- #
def entrena(T: dict, carpeta: Path, P0: np.ndarray, C0: np.ndarray,
            iteraciones: int, estirado: tuple[float, float] | None = None) -> tuple[dict, dict]:
    import torch
    from gsplat import DefaultStrategy, rasterization
    from PIL import Image
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    dev = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("Sin GPU. Este script necesita ~/.venvs/dental-gpu.")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # --- siembra: los propios vértices, con su color ------------------------ #
    means0 = torch.tensor(P0, dtype=torch.float32, device=dev)
    n = len(P0)
    # σ inicial = distancia típica al vecino, para que la superficie arranque cerrada.
    d = torch.cdist(means0[:2000], means0[:2000]).topk(4, largest=False).values[:, 1:].mean()
    k = (GRADO_SH + 1) ** 2
    sh0 = torch.tensor((C0 - 0.5) / _C0, dtype=torch.float32, device=dev)[:, None, :]
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means0),
        "scales": torch.nn.Parameter(torch.log(torch.full((n, 3), float(d), device=dev))),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1)),
        # Opacidad inicial baja (0,1) a propósito: el control de densidad PODA lo que
        # no gana opacidad, y arrancar opaco deja neblina que ningún visor sabe quitar.
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(torch.zeros((n, k - 1, 3), device=dev)),
    }).to(dev)
    # Un Adam POR clave: es lo que el DefaultStrategy necesita para añadir y quitar
    # filas (densificar y podar) sin descuadrar el estado del optimizador.
    opt = {c: torch.optim.Adam([v], lr=TASAS[c]) for c, v in params.items()}

    # --- cámaras ------------------------------------------------------------ #
    W = H = T["w"]
    fx = 0.5 * W / math.tan(0.5 * T["camera_angle_x"])
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
                     dtype=torch.float32, device=dev)
    flip = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                        dtype=torch.float32)  # OpenGL (Blender) → OpenCV (gsplat)
    viewmats = torch.stack([
        torch.linalg.inv(torch.tensor(f["transform_matrix"], dtype=torch.float32) @ flip)
        for f in T["frames"]]).to(dev)

    # A resolución alta las imágenes no caben en RAM: se cargan bajo demanda.
    def imagen(j: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = np.asarray(Image.open(carpeta / T["frames"][j]["file_path"]).convert("RGBA"),
                       np.float32) / 255.0
        rgb = a[..., :3]
        if estirado is not None:
            lo, hi = estirado
            rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        # El color va compuesto sobre negro; el alfa se devuelve aparte porque es la
        # supervisión de "aquí no hay nada", no un cuarto canal de color. El estirado se
        # aplica ANTES de componer: si no, el fondo se levantaría del negro.
        return (torch.from_numpy(rgb * a[..., 3:4]).to(dev),
                torch.from_numpy(a[..., 3]).to(dev))

    def dibuja(vm: torch.Tensor, grado: int):
        return rasterization(
            params["means"], torch.nn.functional.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.cat([params["sh0"], params["shN"]], 1),
            vm[None], K[None], W, H, sh_degree=grado, packed=False)

    # Una de cada cuatro vistas se RETIENE y no se entrena con ella: sin holdout el
    # PSNR solo dice cuánto ha memorizado el modelo, que es exactamente lo que no
    # interesa saber.
    idx = np.arange(len(T["frames"]))
    test = idx[::4]
    train = np.setdiff1d(idx, test)
    ev_test = rng.choice(test, min(24, len(test)), replace=False)
    ev_train = rng.choice(train, min(24, len(train)), replace=False)

    def psnr(js, grado) -> float:
        with torch.no_grad():
            return float(np.mean([
                -10 * math.log10(((dibuja(viewmats[j], grado)[0][0] - imagen(j)[0]) ** 2)
                                 .mean().item() + 1e-10) for j in js]))

    estr = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(iteraciones * 0.5),
                           reset_every=1500, refine_every=100, verbose=False)
    estr.check_sanity(params, opt)
    estado = estr.initialize_state(scene_scale=1.0)

    print(f"entreno  {n:,} gaussianas sembradas · {len(train)} vistas · "
          f"{len(test)} retenidas · SH grado {GRADO_SH} por etapas")
    t = time.perf_counter()
    curva = []
    for it in range(iteraciones):
        grado = min(GRADO_SH, it // SUBIR_SH_CADA)
        j = int(rng.choice(train))
        salida, alfa, info = dibuja(viewmats[j], grado)
        estr.step_pre_backward(params, opt, estado, it, info)
        objetivo, alfa_obj = imagen(j)
        a = salida[0].permute(2, 0, 1).unsqueeze(0)
        b = objetivo.permute(2, 0, 1).unsqueeze(0)
        perdida = (PESO_L1 * (salida[0] - objetivo).abs().mean()
                   + (1 - PESO_L1) * (1.0 - ssim(a, b, data_range=1.0))
                   + PESO_ALFA * (alfa[0, ..., 0] - alfa_obj).abs().mean())
        for o in opt.values():
            o.zero_grad()
        perdida.backward()
        for o in opt.values():
            o.step()
        estr.step_post_backward(params, opt, estado, it, info, packed=False)
        if it % 500 == 0 or it == iteraciones - 1:
            pt, pe = psnr(ev_test, grado), psnr(ev_train, grado)
            curva.append({"iteracion": it, "psnr_retenidas": pt, "psnr_entrenamiento": pe,
                          "gaussianas": int(params["means"].shape[0])})
            print(f"  it {it:5d}  SH {grado}  pérdida {perdida.item():.4f}  "
                  f"PSNR retenidas {pt:5.2f} / entrenamiento {pe:5.2f}  "
                  f"{params['means'].shape[0]:>7,} gaussianas")
    print(f"entreno  {time.perf_counter() - t:.0f} s")

    salida = {c: v.detach().cpu().numpy() for c, v in params.items()}
    informe = {"curva": curva, "vistas_entrenamiento": len(train), "vistas_retenidas": len(test),
               "psnr_retenidas": curva[-1]["psnr_retenidas"],
               "psnr_entrenamiento": curva[-1]["psnr_entrenamiento"],
               "gaussianas": curva[-1]["gaussianas"], "iteraciones": iteraciones,
               "sh_grado": GRADO_SH}
    return salida, informe


# --------------------------------------------------------------------------- #
# 4 · Exportación al visor, de vuelta en milímetros
# --------------------------------------------------------------------------- #
def poda_flotantes(p: dict, superficie: np.ndarray, umbral: float) -> tuple[dict, int]:
    """Quita las gaussianas que se han ido lejos de la superficie medida.

    Red de seguridad de `PESO_ALFA`, no sustituto: lo que hay que evitar es que se
    formen. Aquí la malla es la verdad sobre dónde está la superficie —a diferencia de
    un 3DGS fotográfico, donde no hay ninguna—, así que una gaussiana a más de `umbral`
    del escaneo no representa tejido: representa el fondo. Se declara en el informe
    porque es una intervención sobre el resultado, no una consecuencia del ajuste.
    """
    from scipy.spatial import cKDTree

    d, _ = cKDTree(superficie).query(p["means"])
    se_queda = d <= umbral
    return {c: v[se_queda] for c, v in p.items()}, int((~se_queda).sum())


def exporta(p: dict, destino: Path, escala: float, desplazamiento: np.ndarray) -> None:
    """Campo entrenado → PLY en la convención INRIA, en el espacio en mm del escaneo."""
    n = len(p["means"])
    k = p["shN"].shape[1]
    campos = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    campos += [f"f_rest_{i}" for i in range(3 * k)]
    campos += ["opacity", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3"]
    reg = np.zeros(n, dtype=[(c, "<f4") for c in campos])

    # Deshacer la normalización de Blender: es una semejanza, así que las posiciones
    # llevan escala y traslación y las escalas (guardadas en log) solo la escala.
    xyz = p["means"] / escala + desplazamiento
    reg["x"], reg["y"], reg["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for i in range(3):
        reg[f"f_dc_{i}"] = p["sh0"][:, 0, i]
    # ⚠️ INRIA guarda f_rest por CANAL, no por coeficiente: primero los k del rojo,
    # luego los del verde, luego los del azul. Escribirlo en el otro orden compila y
    # carga igual, y el campo sale con los colores girados según se mueve la cámara.
    resto = p["shN"].transpose(0, 2, 1).reshape(n, 3 * k)
    for i in range(3 * k):
        reg[f"f_rest_{i}"] = resto[:, i]
    reg["opacity"] = p["opacities"]
    for i in range(3):
        reg[f"scale_{i}"] = p["scales"][:, i] - math.log(escala)
    for i in range(4):
        reg[f"rot_{i}"] = p["quats"][:, i]

    cabecera = ("ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                + "".join(f"property float {c}\n" for c in campos)
                + "end_header\n")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("wb") as f:
        f.write(cabecera.encode("ascii"))
        f.write(reg.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--escaneo", required=True, type=Path, help="Malla intraoral (STL).")
    ap.add_argument("--salida", required=True, type=Path, help="Directorio. Fuera del repo.")
    ap.add_argument("--ply", type=Path, help="Campo entrenado para el visor.")
    ap.add_argument("--vistas", type=int, default=1200)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--anillos", type=int, default=8, help="Anillos de elevación de la órbita.")
    ap.add_argument("--iteraciones", type=int, default=7000)
    ap.add_argument("--raking-deg", type=float, default=35.0,
                    help="Separación de la luz respecto al eje de la cámara. 0 = frontal plano.")
    ap.add_argument("--gris", type=float, metavar="V",
                    help="Gris plano en vez del color por región. Mira RELIEVE, no anatomía.")
    ap.add_argument("--ambiente", type=float, default=0.4,
                    help="Fuerza del mundo. Bajarlo profundiza las sombras.")
    ap.add_argument("--sin-estirado", action="store_true",
                    help="No estirar el contraste a blanco y negro.")
    ap.add_argument("--saltar-render", action="store_true",
                    help="Reutiliza las vistas ya renderizadas en --salida.")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    # --- 1 · ingesta y color por región, con los agentes de siempre ---------- #
    store = ArtifactStore(args.salida / "_store")
    res = MeshAgent(store).ingest(args.escaneo)
    print(f"malla    {args.escaneo.name[:38]:40} {res.status.value:9} "
          f"{res.n_primitives or 0:>7,} primitivas · {res.latency_s:.2f} s")
    if not res.ok or res.artifact_ref is None:
        print(f"  {res.detail}", file=sys.stderr)
        return 1
    arte = store.load(res.artifact_ref)
    V, F = arte["positions"].astype(np.float64), arte["faces"]

    # Marco del arco, el mismo que usa la medida: así el campo entrenado y las cotas
    # de `seguimiento_histora.py` viven en el mismo sistema de coordenadas.
    centro, ejes, _, razon = sh.ac.marco_arcada(V)
    P = (V - centro) @ ejes.T
    if razon > 0.6:
        print(f"⚠ orientación poco clara ({razon:.2f}): revisa el render antes de creerte nada.",
              file=sys.stderr)
    logp = sh.SegmentadorGingival()(P)
    if args.gris is not None:
        # Gris plano: sin tono que distraiga, todo el rango dinámico va al relieve. El
        # campo se siembra con el mismo gris, así que arranca neutro y aprende la forma.
        color = np.full((len(P), 3), float(args.gris))
        print(f"color    gris plano {args.gris:.2f} · ambiente {args.ambiente:.2f} · "
              "el relieve manda, no la anatomía")
    else:
        color = sh.color_por_region(np.exp(logp[:, 1]))
        print(f"color    corona {(logp.argmax(1) == 1).mean():.0%} · esmalte "
              f"{tuple(round(c * 255) for c in sh.COLOR_ESMALTE)} · encía "
              f"{tuple(round(c * 255) for c in sh.COLOR_ENCIA)}")

    malla_color = args.salida / "malla_color.ply"
    escribe_malla_color(P, F, color, malla_color)

    # --- 2 · vistas ---------------------------------------------------------- #
    vistas_dir = args.salida / "vistas"
    if args.saltar_render and (vistas_dir / "transforms.json").exists():
        T = json.loads((vistas_dir / "transforms.json").read_text())
        print(f"blender  reutilizadas {len(T['frames'])} vistas de {vistas_dir}")
    else:
        T = render_vistas(malla_color, vistas_dir, args.vistas, args.res,
                          args.raking_deg, args.anillos, args.gris, args.ambiente)

    # --- 3 · entrenamiento --------------------------------------------------- #
    escala = float(T["scan_scale"])
    desplazamiento = np.array(T.get("scan_offset", [0.0, 0.0, 0.0]))
    P_norm = ((P - desplazamiento) * escala).astype(np.float32)

    estirado = None
    if not args.sin_estirado:
        estirado = estirado_global(vistas_dir, T)
        lo, hi = estirado
        print(f"contraste  rango util [{lo:.3f}, {hi:.3f}] → [0, 1] · "
              f"ganancia x{1 / max(hi - lo, 1e-6):.2f}")
        color = np.clip((color - lo) / max(hi - lo, 1e-6), 0.0, 1.0)  # la siembra, igual

    entrenado, informe = entrena(T, vistas_dir, P_norm, color, args.iteraciones, estirado)

    # La poda va en el espacio normalizado, que es donde vive el campo hasta exportarlo.
    entrenado, podadas = poda_flotantes(entrenado, P_norm, UMBRAL_FLOTANTE_MM * escala)
    informe["podadas_por_distancia"] = podadas
    informe["umbral_flotante_mm"] = UMBRAL_FLOTANTE_MM
    informe["gaussianas"] = len(entrenado["means"])
    print(f"poda     {podadas:,} gaussianas a más de {UMBRAL_FLOTANTE_MM} mm de la "
          f"superficie · quedan {informe['gaussianas']:,}")

    informe["raking_deg"] = args.raking_deg
    informe["resolucion"] = args.res
    informe["gris"] = args.gris
    informe["ambiente"] = args.ambiente
    informe["estirado_contraste"] = estirado
    (args.salida / "entrenamiento.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {args.salida / 'entrenamiento.json'}")

    # --- 4 · exportación ------------------------------------------------------ #
    if args.ply:
        exporta(entrenado, args.ply, escala, desplazamiento)
        print(f"→ {args.ply}  ({informe['gaussianas']:,} gaussianas · "
              f"PSNR retenidas {informe['psnr_retenidas']:.2f} dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
