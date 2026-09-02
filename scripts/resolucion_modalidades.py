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

**Qué es medido y qué es derivado.** La PSF (FWHM 0,425 mm ≈ 1,18 pl/mm) y el ruido
(47 HU) están **medidos sobre el CBCT real** de `data/raw/histora` — un Carestream
CS 9600 — así que la simulación describe un equipo concreto y no un CBCT genérico.
El contraste que sobrevive es forma cerrada: el valor central de una esfera
convolucionada con una gaussiana, sin ningún ajuste.

El muestreo de 0,23 mm del IOS sale de las mallas reales de Teeth3DS+ (mediana de
116 k vértices sobre ~6000 mm² de arcada), no de la ficha del fabricante.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

RESUMEN_EN = "Simulates the effective resolution of each dental modality."

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402
from scipy.special import erf  # noqa: E402

DESTINO = Path(__file__).resolve().parent.parent / "docs" / "research"

FINO, LADO = 0.01, 9.0  # mm/px de la verdad-terreno · lado del parche

# PSF y ruido **medidos** sobre el CBCT real de `data/raw/histora` (Carestream
# CS 9600, 578 cortes, vóxel 0,30 mm) — ya no son supuestos. La PSF sale de ajustar
# una erf a ~800 perfiles perpendiculares a bordes de alto contraste; el ruido, de
# la desviación robusta entre vóxeles vecinos dentro de tejido homogéneo.
#
# Se toma la ventana de **bajo contraste** (tejido↔hueso) como PSF del sistema: las
# de esmalte y metal salen peores (0,567 y 0,611 mm) porque llevan endurecimiento de
# haz dentro, que es artefacto y no óptica. Y el ruido de hueso trabecular en vez del
# de tejido blando (14,7 HU) porque es el tejido que rodea a los dientes.
PSF_CBCT, RUIDO_CBCT = 0.425, 47.0  # FWHM en mm · desviación típica en HU
PSF_MEDIDAS = {"tejido↔hueso": 0.425, "aire↔esmalte": 0.567, "metal": 0.611}
RUIDO_MEDIDO = {"tejido blando": 14.7, "hueso trabecular": 47.2}
RUIDO_ALETA = 60.0

# Superficies de referencia para traducir pt/mm² a cifras que signifiquen algo:
# la cara oclusal simulada (9 × 9 mm) y una arcada completa.
OCLUSAL_MM2, ARCADA_MM2 = LADO * LADO, 6000.0
# Alto de una arcada con hueso alveolar: da cuántas capas tiene el CBCT hacia dentro,
# que es justo lo que ninguna malla de superficie posee.
PROFUNDIDAD_MM = 40.0

# (nombre, FWHM mm, color, paso de muestreo mm)
#
# El paso del IOS comercial NO es su *trueness*: son magnitudes distintas y
# confundirlas es fácil. La trueness (32-98 um) dice cuánto se desvía la superficie
# de la verdad; el paso dice cada cuánto hay un punto. Medido sobre arcada completa,
# los escáneres entregan 34-80 pt/mm² -> 0,112-0,171 mm. Se toma Trios (41,21 pt/mm²)
# como representativo. Ref.: Nedelcu et al., PMC5937957.
MODALIDADES = [
    ("CBCT · y STL por marching cubes", PSF_CBCT, "#b9482e", 0.30),
    ("IOS Teeth3DS+ como lo tenemos", 0.23, "#a06a12", 0.23),
    ("IOS comercial medido (Trios)", 0.156, "#2c7a4b", 0.156),
    ("Aleta de mordida", 0.033, "#2c5f8a", 0.033),
]

# Resolución medida de escáneres reales sobre arcada completa (pt/mm²).
IOS_MEDIDOS = {"Omnicam": 79.82, "True Definition": 54.68, "Trios": 41.21, "iTero": 34.20}

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
    ct, kt = muestrear(color, 0.156)
    ca, ka = muestrear(dens, 0.033, fwhm=0.033, ruido=RUIDO_ALETA, semilla=2)

    paneles = [
        ("CBCT · y el STL que sale de él",
         "0,30 mm · PSF 425 µm · ruido 47 HU (medidos)", "11 pt/mm²",
         _expandir(_gris(c30), k30), "#b9482e"),
        # Mismos datos que el panel de arriba, solo que interpolados: es la
        # demostración de que suavizar no añade información.
        ("3DGS desde ese CBCT", "continuo — MISMA informacion", "11 pt/mm²",
         _expandir(_gris(c30), k30, 1), "#b9482e"),
        ("IOS como se entrega", "malla 0,23 mm · color, sin densidad", "19 pt/mm²",
         _expandir(ci, ki), "#a06a12"),
        ("IOS comercial (medido)", "0,156 mm · Trios, 41 pt/mm²", "41 pt/mm²",
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


def _lado(puntos: float) -> str:
    """Los mismos puntos, expresados como imagen cuadrada: la unidad que sí se intuye."""
    n = int(round(puntos**0.5))
    return f"{n}×{n}"


def tabla_puntos() -> None:
    """Densidad de muestreo **de las modalidades**, en puntos y en píxeles.

    3DGS queda deliberadamente fuera de esta tabla y se trata aparte: no es una
    modalidad, es un contenedor **sin resolución propia** — hereda la de aquello con
    lo que se entrene. Mezclarlo aquí hace leer «11 pt/mm²» como un techo suyo cuando
    es el de la fuente que hoy le damos.

    `pt/mm²` y `resuelve` dicen cosas distintas: cuántos datos hay, y qué detalle
    sobrevive. Un render de más píxeles no mueve ninguna de las dos.
    """
    base = 1 / MODALIDADES[0][3] ** 2
    print(
        f"\n{'modalidad':<26}{'paso':>7}{'pt/mm²':>8}{'×CBCT':>7}"
        f"{'oclusal':>12}{'arcada':>13}{'MP':>7}{'resuelve':>11}"
    )
    print("-" * 91)
    for nombre, fwhm, _, paso in MODALIDADES:
        d = 1 / paso**2
        etiqueta = nombre.split(" (")[0]
        print(
            f"{etiqueta:<26}{paso:>6.3f} {d:>7.0f} {d / base:>6.0f}× "
            f"{_lado(d * OCLUSAL_MM2):>11} {_lado(d * ARCADA_MM2):>12} "
            f"{d * ARCADA_MM2 / 1e6:>6.2f} {fwhm:>9.2f} mm"
        )

    print("\nSaltos entre escalones consecutivos:")
    for (n1, f1, _, h1), (n2, f2, _, h2) in zip(MODALIDADES, MODALIDADES[1:], strict=False):
        e1, e2 = (n.split(" (")[0] for n in (n1, n2))
        print(f"  {e1:<26} → {e2:<26} {(h1 / h2) ** 2:>5.1f}× puntos  {f1 / f2:>4.1f}× resolución")

    # --- 3DGS aparte: lo que hereda hoy y lo que debería heredar tras fusionar --- #
    capas = int(PROFUNDIDAD_MM / MODALIDADES[0][3])
    print("\nEscáneres intraorales medidos sobre arcada completa (Nedelcu et al.):")
    for nombre, pt in sorted(IOS_MEDIDOS.items(), key=lambda kv: -kv[1]):
        print(f"  {nombre:<20} {pt:>6.2f} pt/mm²  paso {1 / pt**0.5:>6.3f} mm")

    print(f"\n3DGS no tiene resolución propia. En la cáscara de una arcada ({ARCADA_MM2:.0f} mm²):")
    for etiqueta, paso, nota in (
        ("hoy, sembrado del CBCT", MODALIDADES[0][3], "una gaussiana por vóxel ocupado"),
        ("tras fusionar la malla", MODALIDADES[2][3], "la banda ε toma el detalle del IOS"),
    ):
        d = 1 / paso**2
        print(f"  {etiqueta:<26} {d:>7.0f} pt/mm²  {_lado(d * ARCADA_MM2):>11} px"
              f"  {d * ARCADA_MM2 / 1e6:>6.2f} MP   ({nota})")
    print(f"  y en ambos casos, {capas} capas más hacia dentro que ninguna malla tiene.")


def tabla_resolucion_cbct() -> None:
    """Lo que el CBCT muestrea frente a lo que de verdad resuelve.

    En mm y micras, que es como se mide la resolución — no en puntos. Son dos
    magnitudes distintas: el **vóxel** es el paso de muestreo, elegido al
    reconstruir; la **FWHM** es lo que la óptica del equipo puede separar. El
    CS 9600 muestrea más fino de lo que resuelve, que es buena práctica contra el
    *aliasing*, pero significa que el vóxel sobrestima lo que el equipo mide.
    """
    voxel = MODALIDADES[0][3]
    print(f"\n{'':<34}{'mm':>8}{'µm':>8}{'pl/mm':>9}")
    print("-" * 59)
    print(f"{'MUESTREA (vóxel del CS 9600)':<34}{voxel:>8.3f}{voxel * 1000:>8.0f}"
          f"{1 / (2 * voxel):>9.2f}")
    for nombre, fwhm in PSF_MEDIDAS.items():
        print(f"{'RESUELVE · ' + nombre:<34}{fwhm:>8.3f}{fwhm * 1000:>8.0f}"
              f"{1 / (2 * fwhm):>9.2f}")
    print(f"\n  la resolución real es {PSF_CBCT / voxel:.2f}× más gruesa que el vóxel")
    print("  ruido medido: " + " · ".join(f"{k} {v} HU" for k, v in RUIDO_MEDIDO.items()))


def tabla_marching_cubes(voxeles: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10)) -> None:
    """Qué espaciado de puntos da un STL extraído del CBCT con marching cubes.

    La respuesta es que la arista mediana **es** el vóxel, exactamente, porque MC
    coloca los vértices sobre las aristas de la rejilla. Eso significa que un STL
    denso sacado de un CBCT describe interpolación, no medida: bajar el vóxel
    multiplica los vértices sin añadir un solo dato nuevo.
    """
    import vtk
    from ingestion_agents import synthetic
    from vtk.util import numpy_support as ns

    print(f"\n{'vóxel':>7}{'vértices':>11}{'área mm²':>11}{'arista mediana':>16}")
    print("-" * 45)
    for v in voxeles:
        vol, esp = synthetic.build_volume(spacing=v)
        img = vtk.vtkImageData()
        img.SetDimensions(vol.shape[2], vol.shape[1], vol.shape[0])
        img.SetSpacing(*esp)
        img.GetPointData().SetScalars(
            ns.numpy_to_vtk(vol.ravel(order="C").astype(np.float32), deep=True)
        )
        mc = vtk.vtkFlyingEdges3D()
        mc.SetInputData(img)
        mc.SetValue(0, 300.0)
        mc.Update()
        poly = mc.GetOutput()
        V = ns.vtk_to_numpy(poly.GetPoints().GetData())
        F = ns.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        t = V[F]
        cruz = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        area = float(np.linalg.norm(cruz, axis=1).sum() / 2)
        e = np.vstack([V[F[:, 0]] - V[F[:, 1]], V[F[:, 1]] - V[F[:, 2]], V[F[:, 2]] - V[F[:, 0]]])
        L = np.linalg.norm(e, axis=1)
        print(f"{v:>7.2f}{len(V):>11,}{area:>11,.0f}{np.median(L[L > 0]):>15.3f} mm")


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    figura_paneles(DESTINO / "resolucion-modalidades-paneles.png")
    figura_curva(DESTINO / "resolucion-modalidades-curva.png")
    print(f"Figuras escritas en {DESTINO}")
    tabla_puntos()
    tabla_resolucion_cbct()
    tabla_marching_cubes()
    tabla()


if __name__ == "__main__":
    main()
