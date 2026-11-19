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
        --salida data/processed/caso-agosto/toothfairy2-run-N

**Qué resuelve.** El segmentador propio del CBCT (`entrena_diente_cbct.py`) es binario:
diente o no diente. Hoy la separación entre piezas en el CBCT sale de las semillas del
escáner, y por eso sus errores se heredan (el «28» fantasma, la isla del tercer molar).
ToothFairy2 etiqueta cada diente por su código FDI, así que un modelo entrenado sobre él
separa las piezas **desde el propio volumen**. Este script solo produce esa segmentación
y la describe; compararla con el escáner es otra medida, con sus criterios fijados aparte.
Protocolo y resultados: `docs/research/toothfairy2-caso-b.md`.

**El checkpoint.** El entrenado SIN espejo izquierda-derecha (`onlyMirror01`): espejo solo
en los ejes 0 y 1, también al entrenar. El espejo lateral de nnU-Net no intercambia
etiquetas y enseñaría que un 16 reflejado sigue siendo un 16. ⚠️ La carpeta se llama
`…ResEncUNetL…`, pero sus planes y sus pesos son una **PlainConvUNet de 30,8 M de
parámetros** (el `state_dict` suma 88 M porque repite tensores con alias). Se construye la
red que dicen los planes y se carga en modo estricto; se verificó que es idéntica a la que
reconstruye el propio nnU-Net 2.8.1. ⚠️ Es una foto **a mitad de entrenamiento**: época 266
de un programa de 1000 (lo dice su curva de tasa de aprendizaje), un solo fold (0), y su
mejor Dice interno coincide con la última época registrada. No es el modelo final del reto.

**Orientación.** Las cabeceras `.mha` declaran identidad, origen cero y
`AnatomicalOrientation = RAI`, y sus propias etiquetas lo contradicen en x y en z. La
orientación se **midió sobre las etiquetas**: en el array `(z, y, x)` crecen hacia caudal,
posterior y derecha del paciente. Nuestro lector exige IOP axial estándar y ordena por IPP,
así que crece hacia craneal, posterior e izquierda: se invierten los ejes 0 y 2.
`--verifica-orientacion` repite la medida sobre siete casos fijos del zip (`Dataset112`) y
aborta si no sale lo mismo. El checkpoint se entrenó sobre `Dataset666`: que sean los
mismos volúmenes es una inferencia (mismo número de casos, mismas etiquetas, misma mediana
de tamaño), no un dato. Como el modelo se entrenó con espejo en z e y, solo el eje x
(izquierda-derecha) pone a prueba la inversión; z e y son comprobaciones de cordura.

**Recorte.** El volumen entero no cabe en memoria con 49 clases. La caja se centra en la
**mediana** de las coronas del escáner llevadas al CBCT con el registro de la ejecución de
referencia (`reg.ct_to_ios`, rms 0,68 mm, apto solo para visualización): en el plano,
400×400 vóxeles (120 mm); en altura, `--bajo-oclusal-mm` por debajo y `--sobre-oclusal-mm`
por encima del plano oclusal superior (la cota más caudal de esas coronas). Si la serie
tiene huecos se usa el tramo regular que contiene la caja, sin interpolar; si ninguno la
contiene, se aborta. Antes de recortar se comprueba que el registro cae sobre tejido denso
y que lo hace claramente mejor que si se hubiera leído el registro del marco centrado.

**Normalización.** El preprocesado oficial de nnU-Net (z-score sin máscara) se calcula
sobre **la caja**, no sobre la serie: otra caja cambia lo que ve la red, y por eso la media
y la desviación se registran. `--recorta-grises entrenamiento` lleva antes los grises al
rango de primer plano visto en entrenamiento; es una **desviación** del preprocesado
oficial y se declara como tal.

**Lo que NO significa la salida.** Es una segmentación automática de un modelo entrenado
sobre otro escáner y otra población. No es una medida, no está validada en este caso y sus
etiquetas son Capa 3. Licencia: la web de ToothFairy2 dice CC BY-NC-SA 4.0 y los
`dataset.json` del zip y del checkpoint dicen CC-BY-SA 4.0; se aplica la más restrictiva.
Los grises del CS 9600 no están acreditados como HU.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
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

TOLERANCIA_ESPACIADO = 1e-3
"""mm. Más holgura escondería un remuestreo que el nnU-Net oficial sí haría."""

LADO_PLANO = 400
"""Vóxeles por lado de la caja en el plano axial (120 mm a 0,3 mm)."""

BAJO_OCLUSAL_MM = 45.0
SOBRE_OCLUSAL_MM = 40.0
"""Altura por defecto de la caja respecto al plano oclusal superior (la de la ejecución 1)."""

ALTURA_FUERA_DE_ENTRENAMIENTO_MM = 30.0
"""Por encima de esta altura sobre el plano oclusal, la muestra de entrenamiento no llega."""

PIEZA_PEQUENA_MM3 = 50.0
"""Por debajo, una etiqueta de diente se informa aparte: no es una pieza, es un resto."""

FRACCION_DENSA_MINIMA = 0.5
"""Coronas registradas cuyo entorno de 0,9 mm alcanza el p99 de grises del volumen."""

CASOS_ORIENTACION = ("ToothFairy2F_001", "ToothFairy2F_027", "ToothFairy2F_050",
                     "ToothFairy2P_001", "ToothFairy2P_100", "ToothFairy2P_250",
                     "ToothFairy2P_400")
"""Casos del zip oficial sobre los que se midió la orientación, fijados de antemano."""

DERECHA = [*range(11, 19), *range(41, 49)]
IZQUIERDA = [*range(21, 29), *range(31, 39)]
DIENTES = DERECHA + IZQUIERDA
SUPERIOR = [2, *range(11, 19), *range(21, 29)]
INFERIOR = [1, *range(31, 39), *range(41, 49)]
INCISIVOS = [11, 21, 31, 41]
PRIMEROS_MOLARES = [16, 26, 36, 46]


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


def identidad_de_serie(directorio: Path) -> dict:
    """Qué serie se leyó, sin que viaje ningún identificador en claro.

    `files_sha256` sigue la definición de `uos.manifiesto.digesto_de_partes` para cortes
    sin identidad declarada: `nombre\\0sha256\\n` ordenado por nombre. `uids_sha256` resume
    estudio, serie y marco de referencia, que es la identidad que DICOM define.
    """
    import pydicom

    h = hashlib.sha256()
    ficheros = sorted(p for p in directorio.iterdir() if p.is_file())
    for p in ficheros:
        h.update(f"{p.name}\0{sha256_de(p)}\n".encode())
    uids = set()
    for p in ficheros:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception:  # noqa: BLE001 - un fichero que no es DICOM no aporta UID
            continue
        uids.add("\0".join(str(getattr(ds, k, "")) for k in
                           ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID")))
    return {"n_files": len(ficheros), "files_sha256": h.hexdigest(),
            "uid_sets": len(uids),
            "uids_sha256": hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()}


# ----------------------------------------------------------------------------------------
# Orientación de ToothFairy2, medida sobre sus etiquetas


_TIPOS_MHA = {"MET_UCHAR": np.uint8, "MET_CHAR": np.int8, "MET_USHORT": np.uint16,
              "MET_SHORT": np.int16, "MET_UINT": np.uint32, "MET_INT": np.int32}


def lee_mha_etiquetas(z: zipfile.ZipFile, nombre: str) -> np.ndarray:
    """Un `.mha` de etiquetas leído del zip sin extraerlo. `(z, y, x)` como SimpleITK."""
    datos = z.read(nombre)
    marca = b"ElementDataFile = LOCAL\n"
    i = datos.index(marca) + len(marca)
    cab = dict(lin.split(" = ", 1) for lin in datos[:i].decode().strip().splitlines())
    if cab.get("CompressedData", "False") != "False":
        raise ValueError(f"{nombre}: datos comprimidos, no previsto")
    tipo = _TIPOS_MHA.get(cab.get("ElementType", ""))
    if tipo is None:
        raise ValueError(f"{nombre}: tipo {cab.get('ElementType')} no previsto")
    nx, ny, nz = map(int, cab["DimSize"].split())
    return np.frombuffer(datos[i:], dtype=tipo).reshape(nz, ny, nx)


def _centroide(a: np.ndarray, etiquetas: list[int]) -> np.ndarray | None:
    m = np.isin(a, etiquetas)
    return np.array(np.nonzero(m)).mean(1) if m.any() else None


def verifica_orientacion_tf2(zip_ruta: Path) -> list[dict]:
    """Signo de cada eje anatómico en el array `(z, y, x)` de ToothFairy2.

    Esperado en todos los casos medibles: maxilar−mandíbula < 0 en z (z crece a caudal),
    incisivos−molares < 0 en y (y crece a posterior), dientes izquierdos−derechos < 0 en x
    y canal izquierdo−derecho < 0 en x (x crece hacia la derecha del paciente).
    """
    filas = []
    with zipfile.ZipFile(zip_ruta) as z:
        for caso in CASOS_ORIENTACION:
            a = lee_mha_etiquetas(z, f"Dataset112_ToothFairy2/labelsTr/{caso}.mha")
            fila: dict = {"caso": caso}
            pares = {"z_maxilar_menos_mandibula": ([2], [1], 0),
                     "x_izquierda_menos_derecha": (IZQUIERDA, DERECHA, 2),
                     "x_canal_izq_menos_der": ([3], [4], 2),
                     "y_incisivos_menos_molares": (INCISIVOS, PRIMEROS_MOLARES, 1)}
            for clave, (a_, b_, eje) in pares.items():
                ca, cb = _centroide(a, a_), _centroide(a, b_)
                if ca is not None and cb is not None:
                    fila[clave] = round(float(ca[eje] - cb[eje]), 2)
            filas.append(fila)
    for clave in ("z_maxilar_menos_mandibula", "x_izquierda_menos_derecha",
                  "x_canal_izq_menos_der", "y_incisivos_menos_molares"):
        valores = [f[clave] for f in filas if clave in f]
        if len(valores) < 3 or not all(v < 0 for v in valores):
            raise SystemExit(f"Orientación de ToothFairy2 distinta de la medida: {clave} = "
                             f"{valores}. No se infiere nada con una orientación supuesta.")
    return filas


# ----------------------------------------------------------------------------------------
# Registro, tramo y caja


def coronas_en_cbct(escaneo: Path, fdi: Path, uos: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Vértices de corona del escáner en mm DICOM (LPS) del CBCT.

    Devuelve también dónde caerían si se hubiera leído por error el registro del marco
    centrado (`reg.ct_centered_to_ios`): es la hipótesis alternativa contra la que se
    compara la densidad, porque la descripción de `frame.ct_001` dice «centrado».
    """
    from ingestion_agents.mesh_agent import parse_stl

    V = np.asarray(parse_stl(escaneo)["positions"], dtype=np.float64)
    etiquetas = np.load(fdi)
    if len(etiquetas) != len(V):
        raise SystemExit(f"{fdi.name}: {len(etiquetas)} etiquetas para {len(V)} vértices.")
    with zipfile.ZipFile(uos) as z:
        manifiesto = json.loads(z.read("manifest.json"))
    marcos = {f["id"]: f for f in manifiesto.get("frames", [])}
    regs = {r["id"]: r for r in manifiesto["registrations"]}
    reg = regs["reg.ct_to_ios"]
    if marcos.get(reg["source_frame"], {}).get("anatomical") != "LPS":
        raise SystemExit("El registro no parte de un marco LPS declarado: no se usa.")
    corona = np.c_[V[etiquetas > 0], np.ones(int((etiquetas > 0).sum()))]

    def lleva(r: dict) -> np.ndarray:
        T = np.asarray(r["transform_4x4_row_major"], dtype=np.float64).reshape(4, 4)
        return (np.linalg.inv(T) @ corona.T).T[:, :3]

    alternativa = (lleva(regs["reg.ct_centered_to_ios"])
                   if "reg.ct_centered_to_ios" in regs else None)
    info = {"id": reg["id"], "method": reg.get("method"),
            "rms_error_mm": reg.get("rms_error_mm"), "fit_for": reg.get("fit_for"),
            "n_crown_vertices": int(len(corona))}
    return lleva(reg), alternativa, info


def tramos_regulares(z: np.ndarray) -> list[tuple[int, int]]:
    """`[inicio, fin)` de cada tramo de cortes seguidos, partiendo donde falta alguno."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) < 3:
        return [(0, len(z))]
    pasos = np.diff(z)
    paso = float(np.median(pasos))
    cortes = [0, *(int(i) + 1 for i in np.nonzero(pasos > 1.5 * paso)[0]), len(z)]
    return list(zip(cortes[:-1], cortes[1:], strict=True))


def fraccion_densa(serie, puntos: np.ndarray, p99: float) -> float:
    """Puntos cuyo entorno 3×3×3 (~0,9 mm) alcanza el p99 de grises del volumen."""
    from scipy.ndimage import maximum_filter

    sx, sy, _ = serie.spacing
    n_k, n_f, n_c = serie.volume.shape
    c = np.rint((puntos[:, 0] - serie.origin_xy[0]) / sx).astype(int)
    f = np.rint((puntos[:, 1] - serie.origin_xy[1]) / sy).astype(int)
    k = np.searchsorted(np.asarray(serie.z), puntos[:, 2])
    dentro = (c >= 1) & (c < n_c - 1) & (f >= 1) & (f < n_f - 1) & (k >= 1) & (k < n_k - 1)
    if not dentro.any():
        return 0.0
    k0, k1 = int(k[dentro].min()) - 1, int(k[dentro].max()) + 2
    f0, f1 = int(f[dentro].min()) - 1, int(f[dentro].max()) + 2
    c0, c1 = int(c[dentro].min()) - 1, int(c[dentro].max()) + 2
    local = maximum_filter(serie.volume[k0:k1, f0:f1, c0:c1], size=3)
    valores = local[k[dentro] - k0, f[dentro] - f0, c[dentro] - c0]
    return float((valores >= p99).sum() / len(puntos))


def caja_de_recorte(serie, puntos_mm: np.ndarray, bajo: float, sobre: float) -> dict:
    sx, sy, _sz = serie.spacing
    col = (puntos_mm[:, 0] - serie.origin_xy[0]) / sx
    fil = (puntos_mm[:, 1] - serie.origin_xy[1]) / sy
    z = np.asarray(serie.z, dtype=np.float64)
    oclusal = float(puntos_mm[:, 2].min())       # z crece a craneal: lo más caudal
    n_cortes, n_filas, n_cols = serie.volume.shape

    def rango(centro: float, n: int) -> tuple[int, int, bool]:
        a = int(round(centro - LADO_PLANO / 2))
        b = max(0, min(a, n - LADO_PLANO)) if n >= LADO_PLANO else 0
        return b, min(n, b + LADO_PLANO), b != a

    k_lo = int(np.searchsorted(z, oclusal - bajo, side="left"))
    k_hi = min(int(np.searchsorted(z, oclusal + sobre, side="right")), n_cortes)
    f0, f1, f_mov = rango(float(np.median(fil)), n_filas)
    c0, c1, c_mov = rango(float(np.median(col)), n_cols)
    return {"cortes": (k_lo, k_hi), "filas": (f0, f1), "columnas": (c0, c1),
            "plano_oclusal_z_mm": oclusal,
            "altura_real_mm": (round(float(z[k_lo] - oclusal), 2),
                               round(float(z[k_hi - 1] - oclusal), 2)),
            "limitada_por_el_volumen": {"z": bool(z[0] > oclusal - bajo
                                                  or z[-1] < oclusal + sobre),
                                        "filas": f_mov, "columnas": c_mov}}


# ----------------------------------------------------------------------------------------
# Red e inferencia


def construye_predictor(carpeta: Path, dispositivo: str):
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

    # Etapa del entrenamiento, deducida de la curva poly (lr = lr0·(1 − e/N)^0,9).
    lrs = [float(x) for x in ckpt["logging"]["lrs"]]
    ema = [float(x) for x in ckpt["logging"]["ema_fg_dice"]]
    e = len(lrs) - 1
    planificadas = (round(e / (1 - (lrs[-1] / lrs[0]) ** (1 / 0.9)))
                    if 0 < lrs[-1] < lrs[0] else None)

    pm = PlansManager(planes)
    cm = pm.get_configuration(ckpt["init_args"]["configuration"])
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           perform_everything_on_device=(dispositivo == "cuda"),
                           device=torch.device(dispositivo), allow_tqdm=True)
    pred.manual_initialization(red, pm, cm, [ckpt["network_weights"]], dataset_json,
                               ckpt["trainer_name"],
                               tuple(ckpt["inference_allowed_mirroring_axes"]))
    intensidad = planes["foreground_intensity_properties_per_channel"]["0"]
    info = {"trainer": ckpt["trainer_name"], "network": "PlainConvUNet",
            "parameters_M": round(sum(p.numel() for p in red.parameters()) / 1e6, 3),
            "logged_epochs": len(lrs), "planned_epochs": planificadas,
            "best_is_last_logged": bool(int(np.argmax(ema)) == len(ema) - 1),
            "best_ema_fg_dice_internal": float(ckpt["_best_ema"]),
            "dataset_name": planes.get("dataset_name"), "fold": ckpt["init_args"]["fold"],
            "mirroring_axes": list(ckpt["inference_allowed_mirroring_axes"]),
            "training_foreground_intensity": {k: float(intensidad[k]) for k in
                                              ("min", "max", "percentile_00_5",
                                               "percentile_99_5", "mean", "std")},
            "licence_sources": {"website": "CC BY-NC-SA 4.0",
                                "checkpoint_dataset_json": dataset_json.get("license"),
                                "applied": "CC BY-NC-SA 4.0 (most restrictive)"}}
    return pred, pm, cm, dataset_json, info


def infiere(pred, pm, cm, dataset_json, volumen_tf2: np.ndarray,
            espaciado_zyx: list[float]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Etiqueta y probabilidad máxima por vóxel, en el array tal como entra."""
    import torch
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    props = {"spacing": list(espaciado_zyx)}
    datos, _seg, props = DefaultPreprocessor(verbose=False).run_case_npy(
        volumen_tf2[None].astype(np.float32), None, props, pm, cm, dataset_json)
    if tuple(datos.shape[1:]) != volumen_tf2.shape:
        raise SystemExit(f"El preprocesado cambió la forma {volumen_tf2.shape} → "
                         f"{datos.shape[1:]}: no se esperaba remuestreo ni recorte.")
    salida = io.StringIO()
    with torch.no_grad(), contextlib.redirect_stdout(salida):
        logits = pred.predict_logits_from_preprocessed_data(torch.from_numpy(datos))
    texto = salida.getvalue()
    print(texto, end="")
    etiqueta = np.empty(volumen_tf2.shape, dtype=np.uint8)
    pmax = np.empty(volumen_tf2.shape, dtype=np.float16)
    for k in range(0, logits.shape[1], 16):            # por tandas de cortes: memoria
        prob = torch.softmax(logits[:, k:k + 16].float(), dim=0)
        m, i = prob.max(dim=0)
        etiqueta[k:k + 16] = i.numpy().astype(np.uint8)
        pmax[k:k + 16] = m.numpy().astype(np.float16)
    ajustes = {"device": str(pred.device), "tile_step_size": pred.tile_step_size,
               "use_gaussian": pred.use_gaussian, "use_mirroring": pred.use_mirroring,
               "autocast_fp16": pred.device.type == "cuda",
               "results_moved_to_cpu": "unsuccessful" in texto}
    return etiqueta, pmax, ajustes


# ----------------------------------------------------------------------------------------
# Descripción de la salida


def contactos(etiqueta: np.ndarray, a: list[int], b: list[int]) -> int:
    """Caras de vóxel compartidas entre el grupo `a` y el grupo `b` (vecindad 6)."""
    ma, mb = np.isin(etiqueta, a), np.isin(etiqueta, b)
    total = 0
    for eje in range(3):
        n = etiqueta.shape[eje]
        x0 = [slice(None)] * 3
        x1 = [slice(None)] * 3
        x0[eje], x1[eje] = slice(0, n - 1), slice(1, n)
        total += int((ma[tuple(x0)] & mb[tuple(x1)]).sum()
                     + (mb[tuple(x0)] & ma[tuple(x1)]).sum())
    return total


def describe(etiqueta, pmax, gris, altura_mm, nombres, a_mm, max_entrenamiento) -> list[dict]:
    from scipy.ndimage import label as componentes

    filas = []
    for v in sorted(int(x) for x in np.unique(etiqueta) if x != 0):
        m = etiqueta == v
        n = int(m.sum())
        comp, nc = componentes(m)
        tam = np.bincount(comp.ravel())[1:] if nc else np.array([0])
        idx = np.array(np.nonzero(m)).mean(1)
        alturas = altura_mm[np.nonzero(m)[0]]
        volumen = n * ESPACIADO_MODELO ** 3
        filas.append({
            "label": v, "name": nombres.get(v, "?"), "voxels": n,
            "volume_mm3": round(volumen, 1), "components": int(nc),
            "largest_component_fraction": round(float(tam.max() / n), 4),
            "mean_max_probability": round(float(pmax[m].astype(np.float32).mean()), 4),
            "median_max_probability": round(float(np.median(pmax[m].astype(np.float32))), 4),
            "fraction_above_training_max_grey": round(
                float((gris[m] > max_entrenamiento).mean()), 4),
            "fraction_above_30mm_over_occlusal": round(float(
                (alturas > ALTURA_FUERA_DE_ENTRENAMIENTO_MM).mean()), 4),
            "small_fragment": bool(v in DIENTES and volumen < PIEZA_PEQUENA_MM3),
            "centroid_lps_mm": [round(float(c), 2) for c in a_mm(idx)],
        })
    return filas


def premisas_de_orientacion(filas: list[dict]) -> dict:
    """Signos que la orientación tiene que cumplir sobre ESTE caso. No ajustan nada.

    Solo x pone a prueba la inversión (el modelo se entrenó sin espejo lateral); z e y
    son cordura, porque el modelo vio esos ejes reflejados al entrenar.
    """
    def media(labels: list[int], eje: int) -> float | None:
        sel = [f for f in filas if f["label"] in labels and not f["small_fragment"]]
        if not sel:
            return None
        return float(np.average([f["centroid_lps_mm"][eje] for f in sel],
                                weights=[f["voxels"] for f in sel]))

    out: dict = {}
    pares = {"teeth_left_minus_right_x_lps_mm": (IZQUIERDA, DERECHA, 0, 1),
             "incisors_minus_first_molars_y_lps_mm": (INCISIVOS, PRIMEROS_MOLARES, 1, -1),
             "upper_minus_lower_teeth_z_lps_mm": ([*range(11, 19), *range(21, 29)],
                                                  [*range(31, 39), *range(41, 49)], 2, 1)}
    cumple = True
    for clave, (a, b, eje, signo) in pares.items():
        ma, mb = media(a, eje), media(b, eje)
        if ma is None or mb is None:
            out[clave] = None
            cumple = False
            continue
        out[clave] = round(ma - mb, 2)
        cumple &= (ma - mb) * signo > 0
    out["holds"] = bool(cumple)
    out["tested_by_flip"] = "teeth_left_minus_right_x_lps_mm"
    return out


# ----------------------------------------------------------------------------------------


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
                    help="ToothFairy2_Dataset.zip oficial (se lee, no se extrae).")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--bajo-oclusal-mm", type=float, default=BAJO_OCLUSAL_MM)
    ap.add_argument("--sobre-oclusal-mm", type=float, default=SOBRE_OCLUSAL_MM)
    ap.add_argument("--recorta-grises", choices=("no", "entrenamiento"), default="no",
                    help="'entrenamiento' lleva los grises al rango de primer plano del "
                         "entrenamiento antes del preprocesado. Desviación declarada.")
    ap.add_argument("--dispositivo", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    if args.salida.exists():
        raise SystemExit(f"{args.salida} ya existe: una ejecución no se pisa.")
    temporal = args.salida.with_name(f".{args.salida.name}.en-curso-{os.getpid()}")
    temporal.mkdir(parents=True)
    registro: dict = {
        "schema": "histora-run-record/1.0", "argv": sys.argv, "cwd": os.getcwd(),
        "seed": None, "code": revision_de_codigo(),
        "started_at": datetime.now(UTC).isoformat(), "finished_at": None,
        "exit_code": None, "error": None,
        "scope": "Identifies code and environment of one run; not physical calibration",
    }
    codigo = 1
    try:
        codigo = ejecuta(args, temporal, registro)
    except SystemExit as e:
        registro["error"] = str(e.code)
        print(f"✗ {e.code}", file=sys.stderr)
        codigo = 1
    except Exception as e:
        registro["error"] = repr(e)
        raise
    finally:
        registro["exit_code"] = codigo
        registro["environment"] = entorno()
        registro["finished_at"] = datetime.now(UTC).isoformat()
        (temporal / "run-record.json").write_text(
            json.dumps(registro, indent=2, ensure_ascii=False) + "\n")
        temporal.rename(args.salida)
    return codigo


def entorno() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    import torch

    datos: dict = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": np.__version__, "torch": torch.__version__,
                   "cuda_build": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(),
                   "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                           else None)}
    for paquete in ("nnunetv2", "dynamic_network_architectures", "scipy", "pydicom",
                    "SimpleITK"):
        try:
            datos[paquete] = version(paquete)
        except PackageNotFoundError:
            datos[paquete] = None
    return datos


def ejecuta(args, salida: Path, registro: dict) -> int:
    from ingestion_agents.cbct_agent import CBCTAgent, _read_series, recorta_serie

    registro["agents"] = {CBCTAgent.name: CBCTAgent.version}

    print("· orientación de ToothFairy2, medida sobre sus etiquetas")
    registro["toothfairy2_orientation_check"] = verifica_orientacion_tf2(
        args.verifica_orientacion)

    print("· serie CBCT")
    registro["series"] = identidad_de_serie(args.cbct)
    serie = _read_series(args.cbct)
    if not serie.z_es_superior:
        raise SystemExit("La serie no ordena por IPP hacia craneal: orientación no leída.")
    total = len(serie.z)

    puntos, alternativa, info_reg = coronas_en_cbct(args.escaneo, args.fdi, args.uos)
    oclusal = float(puntos[:, 2].min())
    z_lo, z_hi = oclusal - args.bajo_oclusal_mm, oclusal + args.sobre_oclusal_mm
    tramos = tramos_regulares(serie.z)
    validos = [(i, f) for i, f in tramos
               if serie.z[i] <= max(z_lo, serie.z[0]) and serie.z[f - 1] >= min(z_hi, serie.z[-1])]
    if not validos:
        raise SystemExit(f"Ningún tramo de cortes seguidos contiene la caja (z {z_lo:.1f} a "
                         f"{z_hi:.1f} mm). Tramos: {tramos}. No se interpola.")
    ini, fin = validos[0]
    if (ini, fin) != (0, total):
        serie = recorta_serie(serie, ini, fin)
    registro["series"]["regular_run"] = [ini, fin]
    registro["series"]["slices_total"] = total
    espaciado = np.asarray(serie.spacing, dtype=np.float64)
    if not np.allclose(espaciado, ESPACIADO_MODELO, atol=TOLERANCIA_ESPACIADO):
        raise SystemExit(f"Espaciado {serie.spacing}: distinto del del modelo, no se remuestrea.")

    p99 = float(np.percentile(serie.volume[::4, ::4, ::4], 99))
    densa = fraccion_densa(serie, puntos, p99)
    densa_alt = fraccion_densa(serie, alternativa, p99) if alternativa is not None else None
    info_reg.update({"crowns_reaching_p99_within_0_9mm": round(densa, 4),
                     "same_if_centred_registration_were_read": (
                         None if densa_alt is None else round(densa_alt, 4)),
                     "volume_p99_grey": p99})
    otra = "—" if densa_alt is None else f"{densa_alt:.0%}"
    print(f"  registro: {densa:.0%} de vértices de corona junto a tejido denso "
          f"(con el marco centrado serían {otra})")
    if densa < FRACCION_DENSA_MINIMA or (densa_alt is not None and densa < 2 * densa_alt):
        raise SystemExit("Las coronas registradas no caen sobre tejido denso, o no claramente "
                         "mejor que con el marco centrado. No se recorta a ciegas.")

    caja = caja_de_recorte(serie, puntos, args.bajo_oclusal_mm, args.sobre_oclusal_mm)
    (k0, k1), (f0, f1), (c0, c1) = caja["cortes"], caja["filas"], caja["columnas"]
    sub = serie.volume[k0:k1, f0:f1, c0:c1]
    print(f"  caja cortes {k0}:{k1} filas {f0}:{f1} columnas {c0}:{c1} → {sub.shape}; "
          f"altura {caja['altura_real_mm']} mm sobre el plano oclusal")

    print("· modelo")
    pred, pm, cm, dataset_json, info = construye_predictor(args.modelo, args.dispositivo)
    print(f"  {info}")
    entrada = sub
    recorte = None
    if args.recorta_grises == "entrenamiento":
        t = info["training_foreground_intensity"]
        recorte = (t["min"], t["max"])
        entrada = np.clip(sub, *recorte)
    normalizacion = {"over": "crop box", "mean": float(entrada.astype(np.float64).mean()),
                     "std": float(entrada.astype(np.float64).std()),
                     "grey_clip": recorte,
                     "crop_max_grey": float(sub.max()),
                     "crop_voxels_above_training_max": int(
                         (sub > info["training_foreground_intensity"]["max"]).sum())}

    # Nuestro array crece a craneal, posterior, izquierda; ToothFairy2 a caudal,
    # posterior, derecha.
    volumen_tf2 = np.ascontiguousarray(entrada[::-1, :, ::-1])
    t0 = time.time()
    etiqueta_tf2, pmax_tf2, ajustes = infiere(
        pred, pm, cm, dataset_json, volumen_tf2,
        [float(espaciado[2]), float(espaciado[1]), float(espaciado[0])])
    latencia = time.time() - t0
    etiqueta = np.ascontiguousarray(etiqueta_tf2[::-1, :, ::-1])
    pmax = np.ascontiguousarray(pmax_tf2[::-1, :, ::-1])
    registro["inference"] = ajustes

    z_caja = np.asarray(serie.z[k0:k1], dtype=np.float64)
    altura = z_caja - oclusal

    def a_mm(idx):
        k, f, c = idx
        return (serie.origin_xy[0] + (c0 + c) * espaciado[0],
                serie.origin_xy[1] + (f0 + f) * espaciado[1],
                float(np.interp(k, np.arange(len(z_caja)), z_caja)))

    nombres = {int(v): k for k, v in dataset_json["labels"].items()}
    filas = describe(etiqueta, pmax, sub, altura, nombres, a_mm,
                     info["training_foreground_intensity"]["max"])
    premisas = premisas_de_orientacion(filas)
    mandibula_sobre = int(((etiqueta == 1) & (altura[:, None, None] > 0)).sum())
    arcadas = {
        "upper_lower_jawbone_contact_faces": contactos(etiqueta, [2], [1]),
        "upper_group_lower_group_contact_faces": contactos(etiqueta, SUPERIOR, INFERIOR),
        "upper_teeth_lower_teeth_contact_faces": contactos(
            etiqueta, [*range(11, 19), *range(21, 29)], [*range(31, 39), *range(41, 49)]),
        "lower_jawbone_above_occlusal_mm3": round(mandibula_sobre * ESPACIADO_MODELO ** 3, 1),
    }

    np.savez_compressed(
        salida / "etiquetas-tf2.npz", etiqueta=etiqueta, prob_max=pmax,
        z_mm=z_caja, origin_xy_mm=np.asarray(serie.origin_xy),
        spacing_xyz_mm=espaciado, offset_kfc=np.asarray([k0, f0, c0]),
        orden_ejes=np.asarray(["corte(craneal+)", "fila(posterior+)", "columna(izquierda+)"]))
    resumen = {
        "scope": "Automatic segmentation by an external model (Layer 3). Not a measurement; "
                 "not validated on this case.",
        "model": info, "registration": info_reg, "crop": caja,
        "crop_shape": list(sub.shape), "normalisation": normalizacion,
        "latency_s": round(latencia, 1), "orientation_premises": premisas,
        "arch_separation": arcadas,
        "tooth_labels": sorted(f["label"] for f in filas
                               if f["label"] in DIENTES and not f["small_fragment"]),
        "small_tooth_fragments": sorted(f["label"] for f in filas if f["small_fragment"]),
        "labels": filas,
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2,
                                                    ensure_ascii=False) + "\n")
    registro["inputs"] = {
        "checkpoint_sha256": sha256_de(args.modelo / "checkpoint_best.pth"),
        "scan_sha256": sha256_de(args.escaneo), "fdi_sha256": sha256_de(args.fdi),
        "uos_sha256": sha256_de(args.uos),
        "orientation_zip_sha256": sha256_de(args.verifica_orientacion)}
    registro["outputs"] = {p.name: sha256_de(p) for p in sorted(salida.iterdir())
                           if p.name != "run-record.json"}
    registro["orientation_premises_hold"] = premisas["holds"]

    print(f"· {len(resumen['tooth_labels'])} dientes etiquetados (más "
          f"{len(resumen['small_tooth_fragments'])} restos) en {latencia:.0f} s")
    print(f"  premisas de orientación: {premisas}")
    print(f"  separación de arcadas: {arcadas}")
    for f in filas:
        print(f"  {f['label']:>2} {f['name'][:38]:38} {f['volume_mm3']:>10.1f} mm³  "
              f"comp {f['components']:>3}  p̄ {f['mean_max_probability']:.3f}  "
              f">máx {f['fraction_above_training_max_grey']:.2f}  "
              f">30mm {f['fraction_above_30mm_over_occlusal']:.2f}")
    if not premisas["holds"]:
        print("⚠️ Las premisas de orientación NO se cumplen: la ejecución no es válida.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
