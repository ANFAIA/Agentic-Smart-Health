#!/usr/bin/env python
"""refina_3dgs.py — La fase que faltaba: el campo semilla optimizado como 3DGS.

    ~/.venvs/dental-gpu/bin/python scripts/refina_3dgs.py --cbct <dir-dicom>

**Qué hace y por qué no es un 3DGS normal.** El `cbct-agent` siembra un campo gaussiano
desde el CBCT, y el ADR 001 lo describe como «la inicialización que un optimizador RGS
refinaría». Ese optimizador no existía: el pipeline llegaba hasta la semilla y de ahí
saltaba a exportación, así que **el gemelo digital nunca se ha entrenado como 3DGS**.

Un 3DGS convencional optimiza **color** contra fotografías. Aquí no hay ni una cosa ni la
otra: los valores son **densidad radiológica** (σ, atenuación Beer-Lambert) y no existen
fotos del interior. Optimizar contra renders del propio campo semilla sería circular — el
modelo solo aprendería a reproducir lo que la semilla ya dice.

Lo que sí es verdad de referencia es **el volumen CBCT**. Así que se optimiza contra sus
proyecciones: la profundidad óptica acumulada a lo largo de un rayo, que es exactamente lo
que mide un radiograma reconstruido (DRR) y lo que el campo debe reproducir. Es la
formulación de RGS (Lin et al., arXiv:2604.27552), y la que hace que el residuo signifique
algo físico y no un parecido visual.

**El criterio.** El campo semilla ya reproduce el volumen hasta cierto punto — nace de él.
La pregunta medible es si refinar **mejora** ese punto, y cuánto. Se mide en PSNR sobre
vistas RETENIDAS, que no participan en la optimización.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent.parent
for paquete in ("core-schemas", "ingestion-agents", "fusion-agents", "export-agents"):
    sys.path.insert(0, str(RAIZ / f"packages/{paquete}/src"))

from export_agents.render import Vista  # noqa: E402
from ingestion_agents import ArtifactStore, CBCTAgent  # noqa: E402
from ingestion_agents.cbct_agent import HU_SATURATION, _read_series  # noqa: E402

# Soporte del kernel en sigmas. A 3σ queda fuera el 0,3 % de la masa, y el coste crece
# con el cuadrado: ir a 4σ multiplica por 1,8 el trabajo para recuperar un 0,27 %.
SIGMAS = 3.0
MM_POR_PIXEL = 0.5


def vistas(n: int) -> list[Vista]:
    """`n` direcciones repartidas en azimut, con dos elevaciones.

    No se usan las cuatro ortogonales de `VISTAS_POR_DEFECTO`: con tan pocas y todas en
    ejes principales, un campo puede acertar las proyecciones y estar mal en diagonal.
    """
    fuera = []
    for i in range(n):
        fuera.append(Vista(azimut_deg=360.0 * i / n, elevacion_deg=(-25.0 if i % 2 else 25.0)))
    return fuera


def proyecta_volumen(vol: torch.Tensor, spacing, base: np.ndarray, lado: int, mm: float):
    """DRR del volumen: profundidad óptica integrada a lo largo de `base[2]`.

    Se muestrea el volumen con `grid_sample` sobre un haz de rayos paralelos, que es la
    misma geometría ortográfica que usa el render del campo. Comparar una proyección
    ortográfica contra una perspectiva mediría el modelo de cámara, no el campo.
    """
    dev = vol.device
    sx, sy, sz = spacing
    nz, ny, nx = vol.shape
    # Centro del volumen en mm y semidiagonal, para cubrirlo entero sea cual sea la vista.
    tam = torch.tensor([nx * sx, ny * sy, nz * sz], device=dev, dtype=torch.float32)
    radio = float(torch.linalg.norm(tam) / 2)

    u = torch.linspace(-lado * mm / 2, lado * mm / 2, lado, device=dev)
    t = torch.linspace(-radio, radio, int(2 * radio / mm), device=dev)
    B = torch.tensor(base, device=dev, dtype=torch.float32)

    # (lado, lado, n_t, 3) en mm respecto al centro del volumen.
    gu, gv, gt = torch.meshgrid(u, u, t, indexing="ij")
    p = gu[..., None] * B[0] + gv[..., None] * B[1] + gt[..., None] * B[2]

    # mm -> coordenadas normalizadas de `grid_sample`, que espera (x, y, z) en [-1, 1].
    norm = torch.stack([
        p[..., 0] / (nx * sx / 2), p[..., 1] / (ny * sy / 2), p[..., 2] / (nz * sz / 2)
    ], dim=-1)
    muestras = torch.nn.functional.grid_sample(
        vol[None, None], norm[None], align_corners=False, padding_mode="zeros"
    )[0, 0]
    return muestras.sum(-1) * mm


def splat(centros, sigmas, densidad, base, lado, mm):
    """Profundidad óptica del campo gaussiano, **diferenciable**.

    Reproduce la física de `export_agents.render.profundidad_optica`: se deposita la masa
    `σ·(2π)^{3/2}·s³` con un perfil normalizado y se divide por el área del píxel, para
    que τ no dependa de la resolución. Aquí además tiene que derivar respecto a posición,
    escala y densidad, así que se hace con `index_put_` acumulativo en vez de bucles.
    """
    dev = centros.device
    B = torch.as_tensor(base, device=dev, dtype=torch.float32)
    uv = centros @ B[:2].T                      # (N, 2) coordenadas en el plano imagen
    pix = uv / mm + lado / 2

    masa = densidad * (2 * np.pi) ** 1.5 * sigmas**3
    s_pix = torch.clamp(sigmas / mm, min=0.5)
    radio = int(np.ceil(SIGMAS * float(s_pix.detach().max().clamp(max=8.0))))
    radio = max(1, min(radio, 8))

    tau = torch.zeros(lado * lado, device=dev)
    base_i = torch.round(pix).long()
    dx = torch.arange(-radio, radio + 1, device=dev)
    oy, ox = torch.meshgrid(dx, dx, indexing="ij")
    peso_total = torch.zeros_like(masa)
    aportes = []
    for j in range(oy.numel()):
        iy = base_i[:, 1] + int(oy.flatten()[j])
        ix = base_i[:, 0] + int(ox.flatten()[j])
        d2 = ((ix.float() - pix[:, 0]) ** 2 + (iy.float() - pix[:, 1]) ** 2)
        w = torch.exp(-0.5 * d2 / s_pix**2)
        dentro = (ix >= 0) & (ix < lado) & (iy >= 0) & (iy < lado)
        peso_total = peso_total + w * dentro
        aportes.append((iy.clamp(0, lado - 1) * lado + ix.clamp(0, lado - 1), w * dentro))

    peso_total = torch.clamp(peso_total, min=1e-8)
    for idx, w in aportes:
        tau = tau.index_add(0, idx, masa * w / peso_total)
    return tau.view(lado, lado) / (mm * mm)


def psnr_t(a: torch.Tensor, b: torch.Tensor) -> float:
    pico = float(b.max())
    if pico <= 0:
        return float("nan")
    mse = float(torch.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(pico**2 / mse)


def _refina(
    campo: dict[str, np.ndarray],
    mu: np.ndarray,
    spacing,
    *,
    n_vistas: int = 12,
    pasos: int = 400,
    vistas_por_paso: int = 4,
    lado: int = 192,
    registro=print,
) -> tuple[dict[str, np.ndarray], dict]:
    """Optimiza UN campo contra el volumen `mu` ya normalizado. `(arrays, informe)`.

    `mu` es la densidad objetivo en la MISMA escala que `campo["density"]` (el `cbct-agent`
    la normaliza y cada banda de `siembra_por_banda` la suya). Quien llama decide qué es
    `mu`: el volumen entero, o un solo tramo de HU con el resto a cero.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vol = torch.from_numpy(mu).to(dev)
    registro(f"volumen {tuple(vol.shape)} · campo {len(campo['centers']):,} gaussianas · {dev}")

    centros = torch.tensor(campo["centers"], dtype=torch.float32, device=dev)
    sig = torch.tensor(np.cbrt(np.prod(campo["scales"], axis=1)), dtype=torch.float32, device=dev)
    den = torch.tensor(campo["density"], dtype=torch.float32, device=dev)

    todas = vistas(n_vistas)
    # Una de cada cuatro se RETIENE: sin vistas fuera del ajuste, «mejora» solo dice que
    # el optimizador sabe memorizar las que ve.
    ent = [v for i, v in enumerate(todas) if i % 4 != 3]
    ret = [v for i, v in enumerate(todas) if i % 4 == 3]
    registro(f"{len(ent)} vistas de ajuste · {len(ret)} retenidas")

    objetivo = {}
    with torch.no_grad():
        for v in todas:
            objetivo[v.nombre] = proyecta_volumen(vol, spacing, v.base, lado,
                                                  MM_POR_PIXEL)
    del vol
    torch.cuda.empty_cache()

    def evalua(cs, ss, ds, cuales) -> float:
        with torch.no_grad():
            return float(np.mean([
                psnr_t(splat(cs, ss, ds, v.base, lado, MM_POR_PIXEL), objetivo[v.nombre])
                for v in cuales
            ]))

    base_ret = evalua(centros, sig, den, ret)
    registro(f"semilla (sin refinar): PSNR retenidas {base_ret:.2f} dB")

    d_c = centros.clone().requires_grad_(True)
    d_s = torch.log(sig.clamp(min=1e-4)).clone().requires_grad_(True)
    d_d = den.clone().requires_grad_(True)
    # ⚠️ Un paso por vista NO converge, y esta medido: con una sola proyeccion por paso la
    # perdida rebotaba (21 → 6,5 → 16) y la PSNR en retenidas subia +1,3 dB hacia el paso
    # 200 y luego CAIA por debajo de la semilla. Cada paso tiraba del campo hacia una
    # proyeccion distinta, y con 498.407 posiciones libres eso es un paseo aleatorio.
    #
    # La prueba de que era eso y no sobreajuste: al pasar de 8 vistas a 24 fue a PEOR. Si
    # sobrase capacidad, mas vistas habrian ayudado.
    opt = torch.optim.Adam([
        {"params": [d_c], "lr": 5e-4},
        {"params": [d_s], "lr": 1e-3},
        {"params": [d_d], "lr": 1e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pasos)
    rng = np.random.default_rng(0)

    for paso in range(1, pasos + 1):
        # El gradiente se promedia sobre VARIAS vistas: es la direccion que las satisface
        # a la vez, no la de la ultima que toco.
        elegidas = rng.choice(len(ent), min(vistas_por_paso, len(ent)), replace=False)
        perdida = sum(
            torch.nn.functional.mse_loss(
                splat(d_c, torch.exp(d_s), torch.clamp(d_d, min=0.0), ent[i].base,
                      lado, MM_POR_PIXEL),
                objetivo[ent[i].nombre],
            )
            for i in elegidas
        ) / len(elegidas)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        opt.step()
        sched.step()
        if paso % 100 == 0 or paso == pasos:
            p = evalua(d_c.detach(), torch.exp(d_s.detach()),
                       torch.clamp(d_d.detach(), min=0.0), ret)
            registro(f"[{paso:>4}/{pasos}] perdida {float(perdida.detach()):.5f} · "
                     f"PSNR retenidas {p:.2f} dB  ({p - base_ret:+.2f} sobre la semilla)")

    final = evalua(d_c.detach(), torch.exp(d_s.detach()),
                   torch.clamp(d_d.detach(), min=0.0), ret)
    arrays = {
        "centers": d_c.detach().cpu().numpy(),
        "scales": np.repeat(torch.exp(d_s.detach()).cpu().numpy()[:, None], 3, axis=1),
        "rotations": campo["rotations"],
        "density": torch.clamp(d_d.detach(), min=0.0).cpu().numpy(),
        "origin": campo["origin"],
        "hu_range": campo["hu_range"],
    }
    informe = {
        "psnr_semilla_db": base_ret,
        "psnr_refinado_db": final,
        "delta_db": final - base_ret,
        "aporta": final > base_ret + 0.1,
        "vistas_retenidas": len(ret),
        "pasos": pasos,
    }
    return arrays, informe


def refina(
    campo: dict[str, np.ndarray],
    serie,
    *,
    n_vistas: int = 12,
    pasos: int = 400,
    vistas_por_paso: int = 4,
    lado: int = 192,
    registro=print,
) -> tuple[dict[str, np.ndarray], dict]:
    """El campo único optimizado contra los DRR del volumen completo.

    Firma original: `caso_completo.py` la llama con `--refina-3dgs`. Normaliza el
    volumen igual que el `cbct-agent` (una sola ventana de σ) y delega en `_refina`.
    """
    hu = np.clip(serie.volume.astype(np.float32), 300.0, HU_SATURATION)
    mu = (hu - 300.0) / (HU_SATURATION - 300.0)
    return _refina(campo, mu, serie.spacing, n_vistas=n_vistas, pasos=pasos,
                   vistas_por_paso=vistas_por_paso, lado=lado, registro=registro)


def refina_por_banda(
    campos,
    serie,
    *,
    n_vistas: int = 12,
    pasos: int = 400,
    vistas_por_paso: int = 4,
    lado: int = 192,
    registro=print,
) -> list[tuple[str, dict[str, np.ndarray], dict]]:
    """Refina N capas por separado, cada una contra la DRR de SU tramo de HU.

    `campos` son los `CampoBanda` de `siembra_por_banda` (cada uno trae `banda`,
    `arrays` con `hu_particion` y `hu_range`). La DRR objetivo de cada capa es el
    volumen con SOLO sus vóxeles, en su propia escala de σ: es la descomposición que
    `docs/research/3dgs-volumetrico-cbct.md` midió en +2,48 dB.
    """
    hu = serie.volume.astype(np.float32)
    resultados = []
    for c in campos:
        banda = c.banda
        arrays = c.arrays
        p_lo, p_hi = arrays["hu_particion"]
        n_lo, n_hi = arrays["hu_range"]
        dentro = (hu >= p_lo) & (hu < p_hi)
        mu = np.where(dentro, np.clip((hu - n_lo) / (n_hi - n_lo), 0.0, 1.0),
                      0.0).astype(np.float32)
        registro(f"— banda {banda} ({p_lo:.0f}..{p_hi:.0f} HU, σ {n_lo:.0f}..{n_hi:.0f})")
        arr, informe = _refina(arrays, mu, serie.spacing, n_vistas=n_vistas, pasos=pasos,
                               vistas_por_paso=vistas_por_paso, lado=lado,
                               registro=registro)
        resultados.append((banda, arr, informe))
    return resultados


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cbct", type=Path, required=True)
    ap.add_argument("--vistas", type=int, default=12)
    ap.add_argument("--pasos", type=int, default=400)
    ap.add_argument("--vistas-por-paso", type=int, default=4)
    ap.add_argument("--lado", type=int, default=192)
    ap.add_argument("--salida", type=Path, default=RAIZ / "data/processed/rgs")
    args = ap.parse_args()

    store = ArtifactStore(args.salida / "artifacts")
    salida = CBCTAgent(store).ingest(args.cbct)
    if not salida.ok:
        print(f"✗ ingesta: {salida.detail}")
        return 1
    campo = store.load(salida.artifact_ref)

    arrays, informe = refina(
        campo, _read_series(args.cbct), n_vistas=args.vistas, pasos=args.pasos,
        vistas_por_paso=args.vistas_por_paso, lado=args.lado,
    )
    print(f"\nsemilla {informe['psnr_semilla_db']:.2f} dB → refinado "
          f"{informe['psnr_refinado_db']:.2f} dB ({informe['delta_db']:+.2f})")
    print("→ " + ("el refinado APORTA" if informe["aporta"]
                  else "el refinado NO aporta sobre la semilla"))

    args.salida.mkdir(parents=True, exist_ok=True)
    ref = store.put(**arrays)
    print(f"campo refinado: {ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
