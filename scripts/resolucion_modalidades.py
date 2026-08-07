#!/usr/bin/env python
"""resolucion_modalidades.py — Simula qué resolución alcanza cada modalidad dental.

    uv run python scripts/resolucion_modalidades.py

Regenera las figuras de `docs/research/resolucion-modalidades.md`: una misma
superficie oclusal con tres hallazgos, pasada por la cadena física de CBCT, IOS,
3DGS y aleta de mordida.

**Por qué existe.** La pregunta «¿cuánta resolución da 3DGS sobre un CBCT?» se
contesta con un experimento, no con una intuición — y la respuesta importa porque
condiciona qué puede prometer el gemelo digital. Aquí la cadena de adquisición se
modela explícitamente (PSF gaussiana → muestreo al paso de la modalidad → ruido),
de modo que el panel de 3DGS salga de **los mismos datos** que el de CBCT y se vea
que la suavidad no añade información.

**Qué es supuesto y qué es derivado.** La PSF (FWHM 0,35 mm ≈ 1,5 pl/mm) y el ruido
(200 HU) son supuestos plausibles de un CBCT dental, **no calibrados** contra un
equipo concreto: cámbialos aquí si tienes medidas de fantoma. El contraste que
sobrevive, en cambio, es forma cerrada — el valor central de una esfera
convolucionada con una gaussiana— y no depende de ningún ajuste.

El muestreo de 0,23 mm del IOS sale de las mallas reales de Teeth3DS+ (mediana de
116 k vértices sobre ~6000 mm² de arcada), no de la ficha del fabricante.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402
from scipy.special import erf  # noqa: E402

DESTINO = Path(__file__).resolve().parent.parent / "docs" / "research"

FINO, LADO = 0.01, 9.0  # mm/px de la verdad-terreno · lado del parche
PSF_CBCT, RUIDO_CBCT = 0.35, 200.0  # FWHM en mm · desviación típica en HU
RUIDO_ALETA = 60.0

# (nombre, FWHM mm, color, paso de muestreo mm)
MODALIDADES = [
    ("CBCT / 3DGS desde CBCT", PSF_CBCT, "#b9482e", 0.30),
    ("IOS como se entrega (0,23 mm)", 0.23, "#a06a12", 0.23),
    ("IOS al límite (0,05 mm)", 0.05, "#2c7a4b", 0.05),
    ("Aleta de mordida (0,033 mm)", 0.033, "#2c5f8a", 0.033),
]

# (etiqueta, diámetro mm, ΔHU real) — los tres hallazgos, cada uno en otro soporte
HALLAZGOS = [
    ("fisura oclusal (geometría)", 0.15, 300),
    ("mancha blanca (color)", 0.60, 500),
    ("lesión dentinaria (densidad)", 1.50, 700),
]


def contraste_retenido(d_mm: float | np.ndarray, fwhm: float) -> float | np.ndarray:
    """Fracción del contraste real que sobrevive en el centro de una lesión esférica.

    Forma cerrada del valor central de una esfera de diámetro `d_mm` convolucionada
    con una gaussiana de anchura `fwhm`. No es un ajuste: es la integral.
    """
    a, s = np.asarray(d_mm) / 2, fwhm / 2.3548
    return erf(a / (s * np.sqrt(2))) - np.sqrt(2 / np.pi) * (a / s) * np.exp(-(a**2) / (2 * s**2))


def verdad_terreno() -> tuple[np.ndarray, np.ndarray]:
    """Superficie oclusal con los tres hallazgos. Devuelve `(densidad HU, color RGB)`."""
    n = int(LADO / FINO)
    y, x = np.mgrid[0:n, 0:n] * FINO
    dens = np.full((n, n), 2800.0)  # esmalte
    color = np.ones((n, n, 3)) * np.array([0.93, 0.89, 0.79])

    # 1 · fisura: existe como forma, no como densidad ni color propios
    fisura = (np.abs(x - 4.5 - 0.9 * np.sin(y * 1.1)) < 0.075) | (np.abs(y - 4.5) < 0.075)
    dens[fisura], color[fisura] = -900, [0.33, 0.28, 0.22]

    # 2 · mancha blanca: color en superficie + algo menos de densidad
    mancha = ((x - 2.4) ** 2 + (y - 6.6) ** 2) < 0.30**2
    dens[mancha], color[mancha] = 2400, [1.0, 1.0, 1.0]

    # 3 · lesión dentinaria bajo esmalte intacto: SOLO densidad
    lesion = ((x - 6.6) ** 2 + (y - 2.2) ** 2) < 0.75**2
    dens[lesion] = 2100

    return dens, color


def muestrear(
    campo: np.ndarray, paso: float, *, fwhm: float = 0.0, ruido: float = 0.0, semilla: int = 0
) -> tuple[np.ndarray, int]:
    """Cadena de adquisición: PSF → muestreo al paso de la modalidad → ruido."""
    c = campo.astype(float)
    if fwhm:
        c = gaussian_filter(c, sigma=fwhm / 2.3548 / FINO, axes=(0, 1))
    k = max(1, int(round(paso / FINO)))
    c = c[k // 2 :: k, k // 2 :: k]
    if ruido:
        c = c + np.random.default_rng(semilla).normal(0, ruido, c.shape)
    return c, k


def _gris(d: np.ndarray) -> np.ndarray:
    return np.clip((d + 1000) / 4000, 0, 1)


def _expandir(a: np.ndarray, k: int, orden: int = 0) -> np.ndarray:
    """`orden=0` deja ver el píxel de la modalidad; `orden=1` interpola (el caso 3DGS)."""
    return np.clip(
        zoom(a, (k, k) + (1,) * (a.ndim - 2), order=orden, grid_mode=True, mode="nearest"), 0, 1
    )


def figura_paneles(destino: Path) -> None:
    """Los seis paneles: la misma superficie por seis cadenas de adquisición."""
    dens, color = verdad_terreno()
    n = dens.shape[0]

    c30, k30 = muestrear(dens, 0.30, fwhm=PSF_CBCT, ruido=RUIDO_CBCT, semilla=1)
    ci, ki = muestrear(color, 0.23)
    ct, kt = muestrear(color, 0.05)
    ca, ka = muestrear(dens, 0.033, fwhm=0.033, ruido=RUIDO_ALETA, semilla=2)

    paneles = [
        ("CBCT", "0,30 mm · PSF 0,35 · ruido 200 HU", "11 pt/mm²",
         _expandir(_gris(c30), k30), "#b9482e"),
        # Mismos datos que el panel de arriba, solo que interpolados: es la
        # demostración de que suavizar no añade información.
        ("3DGS desde ese CBCT", "continuo — MISMA informacion", "11 pt/mm²",
         _expandir(_gris(c30), k30, 1), "#b9482e"),
        ("IOS como se entrega", "malla 0,23 mm · color, sin densidad", "19 pt/mm²",
         _expandir(ci, ki), "#a06a12"),
        ("IOS al limite del escaner", "0,05 mm · color, sin densidad", "400 pt/mm²",
         _expandir(ct, kt), "#2c7a4b"),
        ("Aleta de mordida (referencia)", "0,033 mm · ~15 pl/mm", "918 pt/mm²",
         _expandir(_gris(ca), ka), "#2c5f8a"),
    ]
    marcas = [(2.4, 6.6, 0.45, "2"), (6.6, 2.2, 0.95, "3"), (4.5, 4.5, 0.5, "1")]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10.2), facecolor="white")
    axes = axes.ravel()
    sombreada = color * np.clip(0.5 + 0.5 * _gris(dens), 0, 1)[..., None]
    axes[0].imshow(sombreada, interpolation="nearest")
    axes[0].set_title("Verdad-terreno · 0,01 mm", fontsize=12, weight="bold", pad=14)

    for eje, (titulo, sub, dens_pt, img, col) in zip(axes[1:], paneles, strict=True):
        eje.imshow(img, cmap=None if img.ndim == 3 else "gray", vmin=0, vmax=1,
                   interpolation="nearest")
        eje.set_title(f"{titulo}\n{sub}", fontsize=10.5, pad=14)
        eje.text(
            0.98, 0.02, dens_pt, transform=eje.transAxes, ha="right", va="bottom",
            fontsize=10, weight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.9),
        )

    for i, eje in enumerate(axes):
        esc = n if i == 0 else paneles[i - 1][3].shape[0]
        for mx, my, r, lab in marcas:
            eje.add_patch(
                Circle((mx / LADO * esc, my / LADO * esc), r / LADO * esc,
                       fill=False, ec="#c0392b", lw=1.3, ls=(0, (4, 3)), alpha=0.85)
            )
            eje.text(mx / LADO * esc, (my - r) / LADO * esc - esc * 0.022, lab,
                     color="#c0392b", fontsize=9, weight="bold", ha="center")
        eje.set_xticks([])
        eje.set_yticks([])
        for lado in eje.spines.values():
            lado.set_edgecolor("#d5d5d5")

    fig.text(
        0.5, 0.017,
        "1  fisura oclusal 0,15 mm (geometria)     ·     "
        "2  mancha blanca 0,6 mm (color en superficie)     ·     "
        "3  lesion dentinaria 1,5 mm bajo esmalte intacto (solo densidad)",
        ha="center", fontsize=10.5, color="#c0392b", weight="bold",
    )
    plt.subplots_adjust(left=0.012, right=0.988, top=0.915, bottom=0.075, wspace=0.045, hspace=0.20)
    plt.savefig(destino, dpi=112, facecolor="white")
    plt.close(fig)


def figura_curva(destino: Path) -> None:
    """Contraste que sobrevive frente al tamaño del hallazgo, por modalidad."""
    d = np.logspace(np.log10(0.03), np.log10(4), 400)
    fig, eje = plt.subplots(figsize=(11, 6.2), facecolor="white")

    for nombre, fwhm, col, _ in MODALIDADES:
        eje.plot(d, contraste_retenido(d, fwhm) * 100, color=col, lw=2.4, label=nombre)

    eje.axhline(50, color="#888", lw=1, ls=":")
    eje.text(0.032, 52, "50 % del contraste", fontsize=9, color="#666")
    for etiqueta, tam, _ in HALLAZGOS:
        eje.axvline(tam, color="#c0392b", lw=1, ls=(0, (4, 3)), alpha=0.6)
        corta = etiqueta.split(" (")[0].replace(" ", "\n")
        eje.text(tam, 88, corta, ha="center", va="top",
                 fontsize=8.5, color="#c0392b", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#e8b4ab"))

    eje.set_xscale("log")
    eje.set_xlim(0.03, 4)
    eje.set_ylim(0, 100)
    eje.set_xlabel("Tamano del detalle (mm)", fontsize=11)
    eje.set_ylabel("Contraste que sobrevive (%)", fontsize=11)
    eje.set_xticks([0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 4])
    eje.set_xticklabels(["0,05", "0,1", "0,2", "0,3", "0,5", "1", "2", "4"])
    eje.grid(alpha=0.22)
    eje.legend(loc="lower right", frameon=True, fontsize=10)
    eje.set_title("Cuanto contraste llega al dato, segun el tamano del hallazgo",
                  fontsize=12.5, weight="bold", pad=12)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    plt.tight_layout()
    plt.savefig(destino, dpi=112, facecolor="white")
    plt.close(fig)


def tabla() -> None:
    """La misma cuenta en texto, para pegar en la ficha."""
    print(f"\n{'modalidad':<32}" + "".join(f"{t:>10.2f} mm" for _, t, _ in HALLAZGOS))
    for nombre, fwhm, _, _ in MODALIDADES:
        fila = "".join(f"{contraste_retenido(t, fwhm) * 100:>12.0f}%" for _, t, _ in HALLAZGOS)
        print(f"{nombre:<32}{fila}")

    print("\nDetectabilidad en CBCT (contraste retenido / ruido = CNR; Rose exige >= 4):")
    for etiqueta, tam, dhu in HALLAZGOS:
        visto = dhu * contraste_retenido(tam, PSF_CBCT)
        cnr = visto / RUIDO_CBCT
        print(f"  {etiqueta:<32} {tam:>5.2f} mm  {visto:>6.0f} HU  CNR {cnr:>5.1f}"
              f"  {'detectable' if cnr >= 4 else 'NO detectable'}")


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    figura_paneles(DESTINO / "resolucion-modalidades-paneles.png")
    figura_curva(DESTINO / "resolucion-modalidades-curva.png")
    print(f"Figuras escritas en {DESTINO}")
    tabla()


if __name__ == "__main__":
    main()
