#!/usr/bin/env python
"""entrena_diente_cbct.py — Segmentador de diente en CBCT, contra el listón del umbral.

    ~/.venvs/dental-gpu/bin/python scripts/entrena_diente_cbct.py --datos ~/anfaia/toothfairy2

**Qué problema resuelve, y por qué hace falta un modelo.** Está medido en
una medida previa —hoy fuera del repositorio— que **ningún umbral de HU separa diente de
hueso**: el mejor compromiso posible da F1 0,530 (recall 52,6 %, precisión 53,3 %),
porque las dos clases **comparten rango de intensidad** — el p95 del hueso (1716 HU) cae
por encima de la mediana del diente (1735). La información que los distingue no está en
el vóxel: está en la **forma**. Eso es lo que un modelo aporta y un umbral no puede.

> **El listón es F1 0,530.** Un modelo que no lo supere no está aportando nada, y esa
> comparación es la única razón por la que este script existe. Se imprime en cada
> evaluación al lado del resultado, para que no se pueda leer el número sin el control.

**La tarea es binaria: diente sí o no.** El mapa de 6 clases distingue esmalte de dentina
y tres tipos de hueso, pero la pregunta del proyecto es dónde acaba el diente y empieza
el hueso — que es lo que bloquea la composición con la encía del IOS. Separar esmalte de
dentina es otro problema y ya salió negativo por resolución.

**Por parches y no por volumen entero.** Un CBCT recortado son ~270×300×350 vóxeles; una
U-Net 3D sobre eso no cabe en 12 GB. Se entrena sobre parches de 96³, con la mitad
centrados en diente — sin ese sesgo el diente es el 0,5 % de los vóxeles y la red aprende
a decir «todo hueso», que acierta el 99,5 % y no sirve para nada.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESUMEN_EN = "Trains the CBCT tooth segmenter against the threshold baseline."

# Clases del mapa de 6 del espejo de ToothFairy2 (ver `tf_pipeline/bands.py`).
CLASES_DIENTE = (2, 3, 4, 5)

# El listón medido por umbral de HU contra verdad anotada sobre F_001. No se toca
# sin re-medirlo, y el script que lo midió ya no vive aqui (licencia del banco).
F1_UMBRAL = 0.530

PARCHE = 96
# Ventana de HU. Por debajo de -1000 no hay tejido y por encima de 3000 solo metal, que
# satura y arrastra el rango: recortar ahí evita que la normalización la fije el artefacto.
HU_MIN, HU_MAX = -1000.0, 3000.0


def normaliza(hu: np.ndarray) -> np.ndarray:
    return ((np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


class Caso:
    """Un CBCT preparado, **mapeado en memoria**: solo se lee el parche que se pide.

    ⚠️ La version anterior guardaba los casos en `.npz` comprimido y cacheaba 16
    descomprimidos en RAM. Medido: con 135 casos de entrenamiento y lotes de 2 tomados al
    azar, casi cada paso expulsaba y volvia a descomprimir ~140 MB. El proceso leia
    1,24 GB cada 10 s **sin tocar disco** —todo era zlib— y la GPU se quedaba al **2 %**.
    26 minutos sin llegar al paso 250.

    Con `.npy` crudo y `mmap_mode="r"`, leer un parche de 96 lee 96 voxeles y no el
    volumen entero. Cuesta 10,3 GB de disco en vez de 4,4, y es la diferencia entre
    entrenar en horas o en dias.
    """

    def __init__(self, ruta_hu: Path) -> None:
        self.hu = np.load(ruta_hu, mmap_mode="r")
        self.lab = np.load(ruta_hu.with_name(ruta_hu.name.replace("_hu.npy", "_lab.npy")),
                           mmap_mode="r")

    def recorta(self, ini: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """El parche en `ini`, normalizado. Solo aqui se toca la memoria de verdad."""
        s = tuple(slice(i, i + PARCHE) for i in ini)
        return normaliza(np.asarray(self.hu[s])), np.asarray(self.lab[s])


def parche(caso: Caso, rng: np.random.Generator, *, en_diente: bool) -> tuple:
    """Un parche de `PARCHE³`. `en_diente` lo centra en un voxel de diente.

    El centro de un parche de diente se busca sobre el memmap de etiquetas, que es uint8
    y ~28 MB: barato de recorrer y no hay que traer el volumen de HU para decidir donde
    mirar.
    """
    forma = caso.hu.shape
    if any(s < PARCHE for s in forma):
        hu = normaliza(np.asarray(caso.hu))
        lab = np.asarray(caso.lab)
        pad = [(0, max(0, PARCHE - s)) for s in forma]
        return np.pad(hu, pad), np.pad(lab, pad)

    if en_diente:
        cand = np.argwhere(np.asarray(caso.lab) > 0)
        c = cand[rng.integers(len(cand))] if len(cand) else np.array(forma) // 2
        ini = [int(np.clip(c[i] - PARCHE // 2, 0, forma[i] - PARCHE)) for i in range(3)]
    else:
        ini = [int(rng.integers(0, forma[i] - PARCHE + 1)) for i in range(3)]
    return caso.recorta(ini)


def lote(casos: list[Caso], n: int, rng: np.random.Generator, dev, *,
         frac_diente: float = 0.5) -> tuple:
    """Parte de los parches centrados en diente. `frac_diente` gobierna cuántos.

    El diente es ~0,5 % de los vóxeles. Muestreando al azar, casi ningún parche lo
    contiene y la red converge a predecir «fondo» siempre — 99,5 % de acierto y F1 cero.
    Por eso hace falta sesgo.

    ⚠️ Pero el sesgo se paga en precisión, y está medido: con 0,5 el modelo ve ~30 % de
    vóxeles positivos en entrenamiento y ~1 % en inferencia, y ese salto de prior le hace
    sobre-predecir — recall 0,948 con precisión 0,588. En el compuesto de `histora` eso
    sale como hueso etiquetado de diente y piezas de 37 mm donde el máximo anatómico es
    25. Bajarlo acerca el prior al real a cambio de que la clase positiva sea más rara.
    """
    xs, ys = [], []
    for _ in range(n):
        c = casos[rng.integers(len(casos))]
        x, y = parche(c, rng, en_diente=(rng.random() < frac_diente))
        xs.append(x)
        ys.append(y)
    x = torch.from_numpy(np.stack(xs)).unsqueeze(1).to(dev)
    y = torch.from_numpy(np.stack(ys)).unsqueeze(1).float().to(dev)
    return x, y


def bloque(ent: int, sal: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(ent, sal, 3, padding=1), nn.InstanceNorm3d(sal), nn.LeakyReLU(0.01, True),
        nn.Conv3d(sal, sal, 3, padding=1), nn.InstanceNorm3d(sal), nn.LeakyReLU(0.01, True),
    )


class UNet3D(nn.Module):
    """U-Net 3D pequeña. Nada exótico a propósito: el interés está en el listón.

    Con `InstanceNorm` y no `BatchNorm` porque los lotes son de 2-4 parches: con lotes
    tan pequeños las estadísticas de BatchNorm son ruido, y el modelo entrena peor de lo
    que debería por un motivo que no tiene nada que ver con la tarea.
    """

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        b = base
        self.e1, self.e2, self.e3 = bloque(1, b), bloque(b, b * 2), bloque(b * 2, b * 4)
        self.fondo = bloque(b * 4, b * 8)
        self.u3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.d3 = bloque(b * 8, b * 4)
        self.u2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.d2 = bloque(b * 4, b * 2)
        self.u1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.d1 = bloque(b * 2, b)
        self.salida = nn.Conv3d(b, 1, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        f = self.fondo(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(f), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.salida(d1)


def perdida(logits, y):
    """BCE + Dice. La Dice es la que impide la solución degenerada de «todo fondo».

    Con BCE sola, y el diente al 0,5 %, predecir cero en todas partes ya da una pérdida
    baja. La Dice se mide sobre el solape, así que esa solución le da 0.
    """
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum() + 1.0
    den = p.sum() + y.sum() + 1.0
    return bce + (1 - num / den)


@torch.no_grad()
def evalua_volumen(modelo, casos: list[Caso], dev) -> dict:
    """F1 sobre el VOLUMEN ENTERO, que es lo único comparable con el listón del umbral.

    ⚠️ La evaluación por parches NO vale para esto, y es un error fácil de cometer: si la
    mitad de los parches se centran en diente, la clase positiva pasa del 0,5 % real a
    cerca del 30 %, y la precisión sube sola. Un F1 de 0,59 medido así frente al 0,530 del
    umbral —medido sobre el volumen completo— compara dos cosas distintas y hace parecer
    que el modelo gana cuando puede estar perdiendo.

    Aquí se recorre el caso con una ventana deslizante sin solape y se acumulan los
    conteos sobre todos los vóxeles, igual que la medida del umbral de HU.
    """
    modelo.eval()
    tp = fp = fn = 0
    for caso in casos:
        hu, lab = normaliza(np.asarray(caso.hu)), np.asarray(caso.lab)
        pad = [(0, (-s) % PARCHE) for s in hu.shape]
        h, lb = np.pad(hu, pad), np.pad(lab, pad)
        for z in range(0, h.shape[0], PARCHE):
            for y in range(0, h.shape[1], PARCHE):
                for x in range(0, h.shape[2], PARCHE):
                    s = (slice(z, z + PARCHE), slice(y, y + PARCHE), slice(x, x + PARCHE))
                    xt = torch.from_numpy(h[s]).unsqueeze(0).unsqueeze(0).to(dev)
                    pred = torch.sigmoid(modelo(xt))[0, 0].cpu().numpy() > 0.5
                    yb = lb[s].astype(bool)
                    tp += int((pred & yb).sum())
                    fp += int((pred & ~yb).sum())
                    fn += int((~pred & yb).sum())
    modelo.train()
    rec = tp / (tp + fn) if tp + fn else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    return {"f1": 0.0 if rec + pre == 0 else 2 * rec * pre / (rec + pre),
            "recall": rec, "precision": pre}


@torch.no_grad()
def evalua(modelo, casos: list[Caso], dev, *, n_parches: int = 64, semilla: int = 1) -> dict:
    """F1 sobre parches. Barato, para ver la curva — NO comparable con el umbral."""
    modelo.eval()
    rng = np.random.default_rng(semilla)
    tp = fp = fn = 0
    for i in range(n_parches):
        c = casos[rng.integers(len(casos))]
        x, y = parche(c, rng, en_diente=(i % 2 == 0))
        xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(dev)
        pred = (torch.sigmoid(modelo(xt))[0, 0].cpu().numpy() > 0.5)
        yb = y.astype(bool)
        tp += int((pred & yb).sum())
        fp += int((pred & ~yb).sum())
        fn += int((~pred & yb).sum())
    modelo.train()
    rec = tp / (tp + fn) if tp + fn else 0.0
    pre = tp / (tp + fp) if tp + fp else 0.0
    return {"f1": 0.0 if rec + pre == 0 else 2 * rec * pre / (rec + pre),
            "recall": rec, "precision": pre}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--datos", type=Path, default=Path.home() / "anfaia/toothfairy2")
    ap.add_argument("--pasos", type=int, default=3000)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--frac-diente", type=float, default=0.5,
                    help="Fracción de parches centrados en diente. Bajarlo sube precisión.")
    ap.add_argument("--val", type=int, default=8, help="Casos reservados para validar.")
    ap.add_argument("--salida", type=Path,
                    default=Path("data/processed/cbct-diente/modelo.pt"))
    args = ap.parse_args()

    ficheros = sorted(args.datos.glob("memmap/*_hu.npy"))
    if len(ficheros) < args.val + 2:
        print(f"✗ solo hay {len(ficheros)} caso(s) en {args.datos}; hacen falta más.")
        return 1

    # Partición por CASO, nunca por parche: dos parches del mismo paciente comparten
    # anatomía, así que validar con ellos mide memorización, no generalización.
    rng = np.random.default_rng(0)
    orden = rng.permutation(len(ficheros))
    val = [Caso(ficheros[i]) for i in orden[: args.val]]
    ent = [Caso(ficheros[i]) for i in orden[args.val :]]
    print(f"{len(ent)} caso(s) de entrenamiento · {len(val)} de validación")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = UNet3D().to(dev)
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"UNet3D · {n_par / 1e6:.1f} M parámetros · {dev}")
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pasos)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    rng_ent = np.random.default_rng(0)
    mejor, t0 = 0.0, time.time()
    for paso in range(1, args.pasos + 1):
        x, y = lote(ent, args.lote, rng_ent, dev, frac_diente=args.frac_diente)
        opt.zero_grad(set_to_none=True)
        p = perdida(modelo(x), y)
        p.backward()
        opt.step()
        sched.step()

        if paso % 250 == 0 or paso == args.pasos:
            m = evalua(modelo, val, dev)
            marca = "★" if m["f1"] > mejor else " "
            print(f"[{paso:>5}/{args.pasos}] pérdida {p.item():.4f} · "
                  f"F1(parche) {m['f1']:.3f} "
                  f"(rec {m['recall']:.3f} pre {m['precision']:.3f}) {marca} · "
                  f"{(time.time() - t0) / 60:.0f} min")
            if m["f1"] > mejor:
                mejor = m["f1"]
                torch.save({"modelo": modelo.state_dict(), "f1_parche": mejor,
                            "parche": PARCHE, "hu": (HU_MIN, HU_MAX)}, args.salida)

    # El veredicto se da SOLO sobre volumen completo: es lo que mide el listón.
    print("\nevaluando sobre el volumen entero (lo comparable con el umbral)...")
    modelo.load_state_dict(torch.load(args.salida)["modelo"])
    v = evalua_volumen(modelo, val, dev)
    print(f"F1 volumen {v['f1']:.3f} (rec {v['recall']:.3f} pre {v['precision']:.3f}) "
          f"· listón del umbral {F1_UMBRAL:.3f}")
    print("→ " + ("SUPERA el umbral" if v["f1"] > F1_UMBRAL
                  else "NO supera: el modelo no aporta todavía"))
    print(f"(F1 por parches {mejor:.3f} — más alto por construcción, NO comparable)")
    print(f"guardado en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
