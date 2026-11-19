#!/usr/bin/env python
"""segmenta_cbct_toothfairy2.py — Numeración FDI en el CBCT con el modelo oficial de ToothFairy2.

    TF2=~/anfaia/toothfairy2-oficial
    SIN_ESPEJO=nnUNetTrainer__onlyMirror01__nnUNetResEncUNetLPlans__3d_fullres
    ~/.venvs/nnunet-tf2/bin/python scripts/segmenta_cbct_toothfairy2.py \\
        --cbct ~/anfaia/histora/CASE-64A77C_files \\
        --modelo $TF2/checkpoints/Checkpoints/$SIN_ESPEJO \\
        --escaneo data/processed/caso-agosto/entrada/escaneo-superior.stl \\
        --fdi data/processed/fdi-votada/agosto_upper/etiquetas_upper.npy \\
        --uos data/processed/caso-agosto/run-5-modelo/export/entrada.uos \\
        --verifica-orientacion $TF2/ToothFairy2_Dataset.zip \\
        --salida data/processed/caso-agosto/toothfairy2-run-1

**Qué resuelve.** El segmentador propio del CBCT (`entrena_diente_cbct.py`) es binario:
diente o no diente. Hoy la separación entre piezas en el CBCT sale de las semillas del
escáner, y por eso sus errores se heredan (el «28» fantasma, la isla del tercer molar).
ToothFairy2 etiqueta cada diente por su código FDI, así que un modelo entrenado sobre él
separa las piezas **desde el propio volumen**. Este script solo produce esa segmentación
y la describe; compararla con el escáner es otra medida, con sus criterios fijados aparte.

**Decisiones tomadas antes de ver ninguna salida sobre el caso clínico:**

- **Qué checkpoint.** El entrenado SIN espejo izquierda-derecha (`onlyMirror01`). El
  espejo lateral de nnU-Net no intercambia etiquetas: enseña que un 16 reflejado sigue
  siendo un 16, que es exactamente la confusión 1x/2x que no podemos permitirnos. ⚠️ La
  carpeta se llama `…ResEncUNetL…`, pero sus pesos y sus planes son una **PlainConvUNet**
  de 88 M de parámetros. Se construye la red que dicen los planes y se carga en modo
  estricto: si no casa, falla.
- **Orientación.** Las cabeceras `.mha` de ToothFairy2 declaran identidad y origen cero,
  lo que no informa de nada. La orientación se **midió sobre sus etiquetas**: en el array
  `(z, y, x)`, el maxilar queda a menor `z` que la mandíbula, los incisivos a menor `y`
  que los molares y los dientes 2x/3x a menor `x` que los 1x/4x (y los canales igual).
  Es decir, los índices crecen hacia caudal, posterior y derecha del paciente. Nuestro
  lector exige IOP axial estándar y ordena por IPP, así que crece hacia craneal,
  posterior e izquierda. Se invierten los ejes 0 y 2. `--verifica-orientacion` repite la
  medida sobre siete casos fijos del zip y aborta si no sale lo mismo.
- **Recorte.** El volumen entero (542×535×535 con 49 clases) no cabe en memoria. La caja
  se centra en las coronas del escáner llevadas al CBCT con el registro de la ejecución
  de referencia: en el plano, 400×400 vóxeles (120 mm) centrados en su centroide; en
  altura, de 45 mm por debajo a 40 mm por encima del plano oclusal superior (la cota más
  caudal de esas coronas). Tamaño parecido al set F de ToothFairy2. El registro tiene
  1–2 mm de error en incisivos: irrelevante frente a esos márgenes.
- **Sin interpolar.** Si la serie tiene huecos, se usa el tramo regular más largo, como
  en `caso_completo.py`; y el espaciado debe ser el del modelo (0,3 mm) o se aborta.
- **Inferencia oficial.** Preprocesado de nnU-Net (normalización z-score sobre la imagen
  entera), ventana deslizante con paso 0,5, gaussiana y espejo SOLO en los ejes que el
  checkpoint declara permitidos.

**Lo que NO significa la salida.** Es una segmentación automática de un modelo entrenado
sobre otro escáner y otra población (CC BY-NC-SA 4.0). No es una medida, no está validada
en este caso y sus etiquetas son Capa 3. Los grises del CS 9600 no están acreditados como
HU; la normalización z-score atenúa eso, no lo resuelve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))

RESUMEN_EN = "Runs the official ToothFairy2 nnU-Net checkpoint on a CBCT series."

ESPACIADO_MODELO = 0.3
"""mm/vóxel con el que se entrenó el checkpoint (planes `3d_fullres`)."""

LADO_PLANO = 400
"""Vóxeles por lado de la caja en el plano axial (120 mm a 0,3 mm)."""

BAJO_OCLUSAL_MM = 45.0
SOBRE_OCLUSAL_MM = 40.0
"""Extensión de la caja en altura respecto al plano oclusal superior."""

CASOS_ORIENTACION = ("ToothFairy2F_001", "ToothFairy2F_027", "ToothFairy2F_050",
                     "ToothFairy2P_001", "ToothFairy2P_100", "ToothFairy2P_250",
                     "ToothFairy2P_400")
"""Casos del zip oficial sobre los que se midió la orientación, fijados de antemano."""

DERECHA = [*range(11, 19), *range(41, 49)]
IZQUIERDA = [*range(21, 29), *range(31, 39)]
DIENTES = DERECHA + IZQUIERDA


# ----------------------------------------------------------------------------------------
# Trazabilidad


def _git(*a: str) -> str | None:
    try:
        return subprocess.run(["git", *a], cwd=RAIZ, capture_output=True, timeout=30,
                              check=True).stdout.decode()
    except Exception:  # noqa: BLE001 - sin git se declara desconocido
        return None


def revision_de_codigo() -> dict:
    head = _git("rev-parse", "HEAD")
    diff = _git("diff", "HEAD", "--", "scripts", "packages", "apps") or ""
    sueltos = _git("ls-files", "--others", "--exclude-standard", "--",
                   "scripts", "packages", "apps") or ""
    untracked = sorted(x for x in sueltos.splitlines() if x)
    return {"git_head": head.strip() if head else None,
            "git_dirty": bool(diff) or bool(untracked),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "untracked_code": untracked}


def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 22), b""):
            h.update(bloque)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------
# Orientación de ToothFairy2, medida sobre sus etiquetas


def _lee_mha_uint8(z: zipfile.ZipFile, nombre: str) -> np.ndarray:
    datos = z.read(nombre)
    marca = b"ElementDataFile = LOCAL\n"
    i = datos.index(marca) + len(marca)
    cab = dict(lin.split(" = ", 1) for lin in datos[:i].decode().strip().splitlines())
    if cab.get("ElementType") != "MET_UCHAR":
        raise ValueError(f"{nombre}: se esperaba MET_UCHAR, no {cab.get('ElementType')}")
    nx, ny, nz = map(int, cab["DimSize"].split())
    return np.frombuffer(datos[i:], dtype=np.uint8).reshape(nz, ny, nx)


def _centroide(a: np.ndarray, etiquetas: list[int]) -> np.ndarray | None:
    m = np.isin(a, etiquetas)
    return np.array(np.nonzero(m)).mean(1) if m.any() else None


def verifica_orientacion_tf2(zip_ruta: Path) -> list[dict]:
    """Signo de cada eje anatómico en el array `(z, y, x)` de ToothFairy2.

    Esperado en todos los casos: maxilar−mandíbula < 0 en z (z crece a caudal),
    incisivos−molares < 0 en y (y crece a posterior), izquierda−derecha < 0 en x
    (x crece hacia la derecha del paciente). Se exige en TODOS los casos medibles.
    """
    filas = []
    with zipfile.ZipFile(zip_ruta) as z:
        for caso in CASOS_ORIENTACION:
            a = _lee_mha_uint8(z, f"Dataset112_ToothFairy2/labelsTr/{caso}.mha")
            fila: dict = {"caso": caso}
            sup, inf = _centroide(a, [2]), _centroide(a, [1])
            der, izq = _centroide(a, DERECHA), _centroide(a, IZQUIERDA)
            inc, mol = _centroide(a, [11, 21, 31, 41]), _centroide(a, [16, 26, 36, 46])
            if sup is not None and inf is not None:
                fila["z_maxilar_menos_mandibula"] = float(sup[0] - inf[0])
            if izq is not None and der is not None:
                fila["x_izquierda_menos_derecha"] = float(izq[2] - der[2])
            if inc is not None and mol is not None:
                fila["y_incisivos_menos_molares"] = float(inc[1] - mol[1])
            filas.append(fila)
    for clave in ("z_maxilar_menos_mandibula", "x_izquierda_menos_derecha",
                  "y_incisivos_menos_molares"):
        valores = [f[clave] for f in filas if clave in f]
        if len(valores) < 3 or not all(v < 0 for v in valores):
            raise SystemExit(f"Orientación de ToothFairy2 distinta de la medida: {clave} = "
                             f"{valores}. No se infiere nada con una orientación supuesta.")
    return filas


# ----------------------------------------------------------------------------------------
# Caja de recorte desde las coronas registradas


def coronas_en_cbct(escaneo: Path, fdi: Path, uos: Path) -> np.ndarray:
    """Vértices de corona del escáner en milímetros DICOM (LPS) del CBCT."""
    from ingestion_agents.mesh_agent import parse_stl

    V = np.asarray(parse_stl(escaneo)["positions"], dtype=np.float64)
    etiquetas = np.load(fdi)
    if len(etiquetas) != len(V):
        raise SystemExit(f"{fdi.name}: {len(etiquetas)} etiquetas para {len(V)} vértices.")
    with zipfile.ZipFile(uos) as z:
        manifiesto = json.loads(z.read("manifest.json"))
    marcos = {f["id"]: f for f in manifiesto.get("frames", [])}
    reg = next(r for r in manifiesto["registrations"] if r["id"] == "reg.ct_to_ios")
    if marcos.get(reg["source_frame"], {}).get("anatomical") != "LPS":
        raise SystemExit("El registro no parte de un marco LPS declarado: no se usa.")
    T = np.asarray(reg["transform_4x4_row_major"], dtype=np.float64).reshape(4, 4)
    corona = V[etiquetas > 0]
    h = np.c_[corona, np.ones(len(corona))]
    return (np.linalg.inv(T) @ h.T).T[:, :3]


def caja_de_recorte(serie, puntos_mm: np.ndarray) -> dict:
    sx, sy, _sz = serie.spacing
    col = (puntos_mm[:, 0] - serie.origin_xy[0]) / sx
    fil = (puntos_mm[:, 1] - serie.origin_xy[1]) / sy
    z = np.asarray(serie.z, dtype=np.float64)
    oclusal = float(puntos_mm[:, 2].min())       # z crece a craneal: lo más caudal
    c0, f0 = float(np.median(col)), float(np.median(fil))
    n_cortes, n_filas, n_cols = serie.volume.shape

    def rango(centro: float, n: int) -> tuple[int, int]:
        a = int(round(centro - LADO_PLANO / 2))
        a = max(0, min(a, n - LADO_PLANO)) if n >= LADO_PLANO else 0
        return a, min(n, a + LADO_PLANO)

    k_lo = int(np.searchsorted(z, oclusal - BAJO_OCLUSAL_MM, side="left"))
    k_hi = int(np.searchsorted(z, oclusal + SOBRE_OCLUSAL_MM, side="right"))
    return {"cortes": (k_lo, min(k_hi, n_cortes)), "filas": rango(f0, n_filas),
            "columnas": rango(c0, n_cols), "plano_oclusal_z_mm": oclusal,
            "recortada_por_el_volumen": bool(z[0] > oclusal - BAJO_OCLUSAL_MM
                                             or z[-1] < oclusal + SOBRE_OCLUSAL_MM)}


# ----------------------------------------------------------------------------------------
# Red e inferencia


def construye_predictor(carpeta: Path):
    import torch
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    ckpt = torch.load(carpeta / "checkpoint_best.pth", map_location="cpu",
                      weights_only=False)
    planes = ckpt["init_args"]["plans"]
    dataset_json = ckpt["init_args"]["dataset_json"]
    cfg = planes["configurations"][ckpt["init_args"]["configuration"]]
    if cfg.get("UNet_class_name") != "PlainConvUNet":
        raise SystemExit(f"Planes con red {cfg.get('UNet_class_name')}: este script solo "
                         "construye la PlainConvUNet que declara el checkpoint sin espejo.")
    if [float(s) for s in cfg["spacing"]] != [ESPACIADO_MODELO] * 3:
        raise SystemExit(f"Espaciado del modelo {cfg['spacing']}, se esperaba 0,3 mm.")
    n_etapas = len(cfg["conv_kernel_sizes"])
    red = PlainConvUNet(
        input_channels=1, n_stages=n_etapas,
        features_per_stage=[min(cfg["UNet_base_num_features"] * 2 ** i,
                                cfg["unet_max_num_features"]) for i in range(n_etapas)],
        conv_op=torch.nn.Conv3d, kernel_sizes=cfg["conv_kernel_sizes"],
        strides=cfg["pool_op_kernel_sizes"], n_conv_per_stage=cfg["n_conv_per_stage_encoder"],
        num_classes=len(dataset_json["labels"]),
        n_conv_per_stage_decoder=cfg["n_conv_per_stage_decoder"], conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=False)
    red.load_state_dict(ckpt["network_weights"], strict=True)

    pm = PlansManager(planes)
    cm = pm.get_configuration(ckpt["init_args"]["configuration"])
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           perform_everything_on_device=True,
                           device=torch.device("cuda"), allow_tqdm=True)
    pred.manual_initialization(red, pm, cm, None, dataset_json, ckpt["trainer_name"],
                               tuple(ckpt["inference_allowed_mirroring_axes"]))
    info = {"trainer": ckpt["trainer_name"], "epoch": int(ckpt["current_epoch"]),
            "mirroring_axes": list(ckpt["inference_allowed_mirroring_axes"]),
            "network": "PlainConvUNet", "parameters_M": round(
                sum(p.numel() for p in red.parameters()) / 1e6, 3),
            "best_ema_fg_dice_internal": float(ckpt["_best_ema"]),
            "dataset_name": planes.get("dataset_name"), "fold": ckpt["init_args"]["fold"]}
    return pred, pm, cm, dataset_json, info


def infiere(pred, pm, cm, dataset_json, volumen_tf2: np.ndarray):
    """Etiqueta y probabilidad máxima por vóxel, en el array tal como entra."""
    import torch
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    props = {"spacing": [ESPACIADO_MODELO] * 3}
    datos, _seg, props = DefaultPreprocessor(verbose=False).run_case_npy(
        volumen_tf2[None].astype(np.float32), None, props, pm, cm, dataset_json)
    if tuple(datos.shape[1:]) != volumen_tf2.shape:
        raise SystemExit(f"El preprocesado cambió la forma {volumen_tf2.shape} → "
                         f"{datos.shape[1:]}: no se esperaba remuestreo ni recorte.")
    with torch.no_grad():
        logits = pred.predict_sliding_window_return_logits(torch.from_numpy(datos))
    logits = logits.cpu() if hasattr(logits, "cpu") else torch.from_numpy(logits)
    etiqueta = np.empty(volumen_tf2.shape, dtype=np.uint8)
    pmax = np.empty(volumen_tf2.shape, dtype=np.float16)
    for k in range(0, logits.shape[1], 16):            # por tandas de cortes: memoria
        trozo = logits[:, k:k + 16].float()
        prob = torch.softmax(trozo, dim=0)
        m, i = prob.max(dim=0)
        etiqueta[k:k + 16] = i.numpy().astype(np.uint8)
        pmax[k:k + 16] = m.numpy().astype(np.float16)
    return etiqueta, pmax


# ----------------------------------------------------------------------------------------
# Descripción de la salida


def describe(etiqueta: np.ndarray, pmax: np.ndarray, nombres: dict[int, str],
             a_mm) -> list[dict]:
    from scipy.ndimage import label as componentes

    filas = []
    for v in sorted(int(x) for x in np.unique(etiqueta) if x != 0):
        m = etiqueta == v
        n = int(m.sum())
        comp, nc = componentes(m)
        tam = np.bincount(comp.ravel())[1:] if nc else np.array([0])
        idx = np.array(np.nonzero(m)).mean(1)
        filas.append({
            "label": v, "name": nombres.get(v, "?"), "voxels": n,
            "volume_mm3": round(n * ESPACIADO_MODELO ** 3, 1),
            "components": int(nc),
            "largest_component_fraction": round(float(tam.max() / n), 4),
            "mean_max_probability": round(float(pmax[m].astype(np.float32).mean()), 4),
            "centroid_lps_mm": [round(float(c), 2) for c in a_mm(idx)],
        })
    return filas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cbct", type=Path, required=True)
    ap.add_argument("--modelo", type=Path, required=True,
                    help="Carpeta con checkpoint_best.pth del modelo oficial.")
    ap.add_argument("--escaneo", type=Path, required=True)
    ap.add_argument("--fdi", type=Path, required=True,
                    help="Etiquetas FDI por vértice del escaneo (solo para la caja).")
    ap.add_argument("--uos", type=Path, required=True,
                    help="Contenedor de la ejecución de referencia, con reg.ct_to_ios.")
    ap.add_argument("--verifica-orientacion", type=Path, required=True,
                    help="ToothFairy2_Dataset.zip oficial.")
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()

    from ingestion_agents.cbct_agent import _read_series, recorta_serie, tramo_regular

    if args.salida.exists() and any(args.salida.iterdir()):
        raise SystemExit(f"{args.salida} ya tiene contenido: una ejecución no se pisa.")
    args.salida.mkdir(parents=True, exist_ok=True)
    registro: dict = {
        "schema": "histora-run-record/1.0", "argv": sys.argv, "seed": None,
        "code": revision_de_codigo(),
        "started_at": datetime.now(UTC).isoformat(), "finished_at": None,
        "scope": "Identifies code and environment of one run; not physical calibration",
    }

    print("· orientación de ToothFairy2, medida sobre sus etiquetas")
    registro["toothfairy2_orientation_check"] = verifica_orientacion_tf2(
        args.verifica_orientacion)

    print("· serie CBCT")
    serie = _read_series(args.cbct)
    if not serie.z_es_superior:
        raise SystemExit("La serie no ordena por IPP hacia craneal: orientación no leída.")
    total = len(serie.z)
    motivos = []
    if serie.huecos:
        ini, fin = tramo_regular(serie.z)
        serie = recorta_serie(serie, ini, fin)
        motivos.append(f"tramo regular de {fin - ini} de {total} cortes; no se interpola")
    if not np.allclose(serie.spacing, ESPACIADO_MODELO, atol=0.005):
        raise SystemExit(f"Espaciado {serie.spacing}: distinto del del modelo, no se remuestrea.")

    puntos = coronas_en_cbct(args.escaneo, args.fdi, args.uos)
    caja = caja_de_recorte(serie, puntos)
    (k0, k1), (f0, f1), (c0, c1) = caja["cortes"], caja["filas"], caja["columnas"]
    sub = serie.volume[k0:k1, f0:f1, c0:c1]
    # Premisa del registro: las coronas llevadas al CBCT caen sobre tejido denso.
    ci = np.clip(np.rint((puntos[:, 0] - serie.origin_xy[0]) / serie.spacing[0]).astype(int),
                 0, serie.volume.shape[2] - 1)
    fi = np.clip(np.rint((puntos[:, 1] - serie.origin_xy[1]) / serie.spacing[1]).astype(int),
                 0, serie.volume.shape[1] - 1)
    ki = np.clip(np.searchsorted(np.asarray(serie.z), puntos[:, 2]), 0, len(serie.z) - 1)
    gris_coronas = float(np.median(serie.volume[ki, fi, ci]))
    gris_volumen = float(np.median(serie.volume[::4, ::4, ::4]))
    print(f"  caja cortes {k0}:{k1} filas {f0}:{f1} columnas {c0}:{c1} → {sub.shape}; "
          f"gris mediano en coronas {gris_coronas:.0f} frente a {gris_volumen:.0f} en el volumen")
    if gris_coronas <= gris_volumen:
        raise SystemExit("Las coronas registradas no caen sobre tejido denso: el marco del "
                         "registro no es el del volumen. No se recorta a ciegas.")

    # Nuestro array crece a craneal, posterior, izquierda; ToothFairy2 a caudal,
    # posterior, derecha.
    volumen_tf2 = np.ascontiguousarray(sub[::-1, :, ::-1])

    print("· modelo")
    pred, pm, cm, dataset_json, info = construye_predictor(args.modelo)
    print(f"  {info}")
    t0 = time.time()
    etiqueta_tf2, pmax_tf2 = infiere(pred, pm, cm, dataset_json, volumen_tf2)
    latencia = time.time() - t0
    etiqueta = np.ascontiguousarray(etiqueta_tf2[::-1, :, ::-1])
    pmax = np.ascontiguousarray(pmax_tf2[::-1, :, ::-1])

    z_caja = np.asarray(serie.z[k0:k1], dtype=np.float64)

    def a_mm(idx):
        k, f, c = idx
        return (serie.origin_xy[0] + (c0 + c) * serie.spacing[0],
                serie.origin_xy[1] + (f0 + f) * serie.spacing[1],
                float(np.interp(k, np.arange(len(z_caja)), z_caja)))

    nombres = {int(v): k for k, v in dataset_json["labels"].items()}
    filas = describe(etiqueta, pmax, nombres, a_mm)

    # Premisas que la orientación tiene que cumplir sobre ESTE caso. No ajustan nada: si
    # fallan, la ejecución se declara inválida.
    premisas = {}
    der = [f for f in filas if f["label"] in DERECHA]
    izq = [f for f in filas if f["label"] in IZQUIERDA]
    if der and izq:
        xd = np.average([f["centroid_lps_mm"][0] for f in der], weights=[f["voxels"] for f in der])
        xi = np.average([f["centroid_lps_mm"][0] for f in izq], weights=[f["voxels"] for f in izq])
        premisas["teeth_left_minus_right_x_lps_mm"] = round(float(xi - xd), 2)
    sup = [f for f in filas if f["label"] == 2]
    inf = [f for f in filas if f["label"] == 1]
    if sup and inf:
        premisas["upper_minus_lower_jaw_z_lps_mm"] = round(
            sup[0]["centroid_lps_mm"][2] - inf[0]["centroid_lps_mm"][2], 2)
    premisas["holds"] = (premisas.get("teeth_left_minus_right_x_lps_mm", 0) > 0
                         and premisas.get("upper_minus_lower_jaw_z_lps_mm", 0) > 0)

    np.savez_compressed(
        args.salida / "etiquetas-tf2.npz", etiqueta=etiqueta, prob_max=pmax,
        z_mm=z_caja, origin_xy_mm=np.asarray(serie.origin_xy),
        spacing_xyz_mm=np.asarray(serie.spacing), offset_kfc=np.asarray([k0, f0, c0]),
        orden_ejes=np.asarray(["corte(craneal+)", "fila(posterior+)", "columna(izquierda+)"]))
    resumen = {
        "scope": "Automatic segmentation by an external model (Layer 3). Not a measurement; "
                 "not validated on this case.",
        "model": info, "model_licence": "CC BY-NC-SA 4.0 (ToothFairy2 website)",
        "series_slices_total": total, "notes": motivos, "crop": caja,
        "crop_shape": list(sub.shape),
        "registered_crowns_median_grey": gris_coronas, "volume_median_grey": gris_volumen,
        "latency_s": round(latencia, 1), "orientation_premises": premisas, "labels": filas,
    }
    (args.salida / "resumen.json").write_text(json.dumps(resumen, indent=2,
                                                         ensure_ascii=False) + "\n")

    import torch
    registro["environment"] = {"python": platform.python_version(),
                               "platform": platform.platform(), "numpy": np.__version__,
                               "torch": torch.__version__, "cuda_build": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0)}
    from importlib.metadata import version
    for paquete in ("nnunetv2", "dynamic_network_architectures", "scipy"):
        registro["environment"][paquete] = version(paquete)
    registro["inputs"] = {"checkpoint_sha256": sha256_de(args.modelo / "checkpoint_best.pth"),
                          "scan_sha256": sha256_de(args.escaneo),
                          "fdi_sha256": sha256_de(args.fdi), "uos_sha256": sha256_de(args.uos)}
    registro["finished_at"] = datetime.now(UTC).isoformat()
    (args.salida / "run-record.json").write_text(json.dumps(registro, indent=2,
                                                            ensure_ascii=False) + "\n")

    print(f"· {len([f for f in filas if f['label'] in DIENTES])} dientes etiquetados en "
          f"{latencia:.0f} s; premisas de orientación: {premisas}")
    for f in filas:
        print(f"  {f['label']:>2} {f['name'][:38]:38} {f['volume_mm3']:>10.1f} mm³  "
              f"comp {f['components']:>3}  mayor {f['largest_component_fraction']:.3f}  "
              f"p̄ {f['mean_max_probability']:.3f}")
    if not premisas["holds"]:
        print("⚠️ Las premisas de orientación NO se cumplen: la ejecución no es válida.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
