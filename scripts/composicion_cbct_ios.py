#!/usr/bin/env python
"""composicion_cbct_ios.py — Dientes segmentados en el CBCT + encía del IOS, en gaussianas.

    uv run python scripts/composicion_cbct_ios.py --cbct <dir-dicom> --ios <malla.stl>

**Qué monta.** El modelo compuesto que el proyecto persigue: los dientes salen del CBCT
—que es lo único que ve por debajo del margen— y la encía sale del escáner intraoral, que
es lo único que la mide bien. Se compone en **el campo gaussiano**, que es la
representación del twin: aquí no se extrae ninguna malla del volumen, se **etiquetan
gaussianas** (`region_id`), que es lo que consume el resto del pipeline.

**Qué NO hace, y es el punto.** No transfiere las etiquetas del IOS al CBCT. Eso es lo
que se hizo antes y solo puede etiquetar lo que el escáner ve —la corona—, así que la
raíz quedaba fuera por construcción. Aquí la segmentación ocurre **dentro del CBCT**; del
IOS solo se toma el **nombre** (el código FDI) por vecindad con la corona, que es una
asignación de identidad, no una segmentación.

**El muro conocido, y por eso la fase 1 es una medida.** La única frontera real entre
raíz y hueso alveolar es el ligamento periodontal: 0,15–0,38 mm frente a un vóxel de
0,30 mm en esta serie. Por encima de la cresta ósea el diente da contra aire o tejido
blando y ese borde sí está resuelto (1050 HU/mm medido, ~1 vóxel); por debajo, no. Así
que antes de componer nada se **barre el umbral** y se mide a qué HU el tejido duro se
parte en tantas componentes como dientes. Si no se parte a ningún umbral, el compuesto no
se puede construir y el script lo dice en vez de entregar un modelo con el hueso pegado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parent.parent
for paquete in ("ingestion-agents", "fusion-agents", "core-schemas", "export-agents"):
    sys.path.insert(0, str(RAIZ / f"packages/{paquete}/src"))

from export_agents.field import densidad_a_hu, escribe_ply  # noqa: E402
from fusion_agents.registration import apply, icp, quaternion_to_matrix  # noqa: E402
from ingestion_agents import ArtifactStore, CBCTAgent  # noqa: E402
from ingestion_agents.mesh_agent import parse_stl  # noqa: E402

# Umbrales del barrido. El extremo bajo es el del `cbct-agent` (300 HU, tejido duro
# cualquiera) y el alto es la saturación del esmalte: por encima no queda dentina.
BARRIDO_HU = (700, 900, 1100, 1300, 1500, 1700, 1900)

# Radio de vecindad para las componentes conexas, en múltiplos del espaciado del vóxel.
# 1,8 conecta un vóxel con sus vecinos en diagonal de cara y arista pero no en esquina:
# más laxo une dientes por el punto de contacto, más estricto parte un diente sano.
RADIO_VECINDAD = 1.8

# Por encima de esto `query_pairs` materializa demasiados pares y no termina. El umbral
# se salta declarandolo, en vez de dejar el script colgado sin decir por que.
MAX_PARA_COMPONENTES = 700_000

# Una componente por debajo de esto es ruido de umbral, no una pieza.
MIN_GAUSSIANAS = 150

# Cotas de tamaño de un diente, en mm. Un incisivo inferior mide ~4 mm de ancho y un
# molar con raíz ~25 de largo. SIN esta cota el barrido da por buena una componente de
# 88 mm —la mandíbula entera— y el compuesto sale con el hueso pegado y buen aspecto.
DIENTE_MIN_MM = 4.0
DIENTE_MAX_MM = 25.0


def paso_de_rejilla(centros: np.ndarray, *, muestra: int = 20_000, semilla: int = 0) -> float:
    """Espaciado real entre gaussianas vecinas, medido.

    ⚠️ NO es `median(scales)`: esa es la σ de la gaussiana, que el `cbct-agent` fija en
    torno a MEDIO vóxel. Usarla como paso da un radio de vecindad **menor que la
    distancia entre vecinos**, así que el grafo sale sin una sola arista, cada punto es
    su propia componente y el barrido concluye —con toda tranquilidad— que los dientes no
    se separan. Es el modo de fallo caro: un resultado negativo plausible y falso.
    """
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(centros), min(muestra, len(centros)), replace=False)
    d, _ = cKDTree(centros).query(centros[idx], k=2)
    return float(np.median(d[:, 1]))


def componentes(puntos: np.ndarray, radio: float) -> np.ndarray:
    """Etiqueta de componente conexa por punto, con un grafo de vecindad por radio."""
    arbol = cKDTree(puntos)
    pares = arbol.query_pairs(radio, output_type="ndarray")
    if len(pares) == 0:
        return np.arange(len(puntos))
    g = coo_matrix(
        (np.ones(len(pares)), (pares[:, 0], pares[:, 1])),
        shape=(len(puntos), len(puntos)),
    )
    _, etiquetas = connected_components(g, directed=False)
    return etiquetas


def piezas_de(etiquetas: np.ndarray, minimo: int = MIN_GAUSSIANAS) -> list[np.ndarray]:
    """Índices de cada componente que supera el tamaño mínimo, de mayor a menor."""
    valores, cuentas = np.unique(etiquetas, return_counts=True)
    grandes = valores[cuentas >= minimo]
    piezas = [np.flatnonzero(etiquetas == v) for v in grandes]
    return sorted(piezas, key=len, reverse=True)


def parece_diente(puntos: np.ndarray) -> bool:
    """¿Tiene esta componente el tamaño de un diente, o es un trozo de mandíbula?

    Es la comprobación que faltaba. Contar componentes no basta: a 700 HU salen 51, que
    suena a «los dientes se separaron», y la mayor mide 88 mm. Sin mirar la extensión,
    el compuesto se monta igual y parece correcto.
    """
    extension = float(np.ptp(puntos, axis=0).max())
    return DIENTE_MIN_MM <= extension <= DIENTE_MAX_MM


def barrido_de_separacion(
    centros: np.ndarray, hu: np.ndarray, paso_mm: float, *, esperados: int
) -> list[dict]:
    """A qué umbral de HU el tejido duro se parte en tantas piezas como dientes.

    Es **la** medida que decide si el compuesto se puede construir. Un umbral bajo deja
    todo pegado (dientes + hueso alveolar = una sola componente); uno alto deja solo las
    cúspides de esmalte, que ya no son un diente. Si no hay ningún umbral intermedio con
    ~`esperados` piezas de tamaño de diente, el recorte no es posible con este dato.
    """
    radio = RADIO_VECINDAD * paso_mm
    filas = []
    for umbral in BARRIDO_HU:
        sel = hu >= umbral
        n = int(sel.sum())
        if n < MIN_GAUSSIANAS or n > MAX_PARA_COMPONENTES:
            filas.append({"hu": umbral, "n": n, "componentes": 0, "dientes": 0,
                          "mayor": 0, "extension_mayor": 0.0,
                          "saltado": n > MAX_PARA_COMPONENTES})
            continue
        todas = piezas_de(componentes(centros[sel], radio))
        dientes = [p for p in todas if parece_diente(centros[sel][p])]
        mayor = len(todas[0]) if todas else 0
        extension = float(np.ptp(centros[sel][todas[0]], axis=0).max()) if todas else 0.0
        filas.append({
            "hu": umbral,
            "n": n,
            "componentes": len(todas),
            "dientes": len(dientes),
            "mayor": mayor,
            "extension_mayor": extension,
            "_piezas": dientes,
            "_sel": sel,
        })
    return filas


def nombra_por_vecindad(
    centros_pieza: list[np.ndarray], coronas: np.ndarray, fdi_coronas: np.ndarray
) -> list[int]:
    """El código FDI de cada pieza, por voto de la corona del IOS más cercana.

    Del escáner se toma **el nombre, no la forma**: la segmentación ya está hecha en el
    CBCT. Una pieza cuyo voto no llega a mayoría se queda sin nombre (0) en vez de
    heredar el del vecino más próximo, que es como se cuelan identidades inventadas.
    """
    arbol = cKDTree(coronas)
    nombres = []
    for idx in centros_pieza:
        _, vecinos = arbol.query(idx)
        votos = fdi_coronas[vecinos]
        votos = votos[votos > 0]
        if len(votos) == 0:
            nombres.append(0)
            continue
        valores, cuentas = np.unique(votos, return_counts=True)
        ganador, cuenta = valores[cuentas.argmax()], cuentas.max()
        nombres.append(int(ganador) if cuenta > 0.5 * len(votos) else 0)
    return nombres


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cbct", type=Path, required=True, help="Directorio de la serie DICOM.")
    ap.add_argument("--ios", type=Path, required=True, help="Malla intraoral (STL).")
    ap.add_argument("--fdi", type=Path, help="`region_id` por vértice del IOS (.npy).")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data/processed/compuesto.ply")
    ap.add_argument("--artefactos", type=Path, default=RAIZ / "data/processed/artifacts")
    ap.add_argument("--esperados", type=int, default=14, help="Dientes que se esperan.")
    ap.add_argument(
        "--max-primitivas", type=int, default=500_000,
        help="Tope del `cbct-agent`. Con el valor por defecto una serie fina se DECIMA, "
             "y el paso de rejilla efectivo empeora justo en el dato que se queria medir.",
    )
    args = ap.parse_args()

    # --- 1 · el CBCT entra como campo gaussiano, no como malla ---------------- #
    store = ArtifactStore(args.artefactos)
    salida = CBCTAgent(store, max_primitives=args.max_primitivas).ingest(args.cbct)
    if not salida.ok or not salida.artifact_ref:
        print(f"✗ el `cbct-agent` no pudo ingerir: {salida.detail}")
        return 1
    campo = store.load(salida.artifact_ref)
    centros = np.asarray(campo["centers"], dtype=np.float64)
    hu = densidad_a_hu(campo["density"], campo["hu_range"])
    paso = paso_de_rejilla(centros)
    sigma = float(np.median(np.asarray(campo["scales"], dtype=np.float64)))
    print(f"campo gaussiano: {len(centros):,} primitivas · HU {hu.min():.0f}–{hu.max():.0f} "
          f"· paso de rejilla {paso:.3f} mm (sigma {sigma:.3f})")

    # --- 2 · ¿se separan los dientes del hueso? ------------------------------- #
    print("\n--- barrido de separación (la medida que decide si esto se puede montar) ---")
    print(f"{'HU':>6} {'gaussianas':>11} {'componentes':>12} {'tam. diente':>12} "
          f"{'mayor':>8} {'ext. mayor':>11}")
    filas = barrido_de_separacion(centros, hu, paso, esperados=args.esperados)
    for f in filas:
        if f.get("saltado"):
            print(f"{f['hu']:>6} {f['n']:>11,} {'— demasiadas para componentes conexas':>48}")
            continue
        print(f"{f['hu']:>6} {f['n']:>11,} {f['componentes']:>12} {f['dientes']:>12} "
              f"{f['mayor']:>8,} {f['extension_mayor']:>10.1f} mm")

    # El umbral bueno es el que da más piezas de tamaño de diente sin que una sola se
    # lleve la mayoría del tejido duro (eso sería el bloque diente+hueso sin partir).
    # El umbral bueno es el MAS BAJO en el que ya no queda bloque: la componente mayor
    # tiene tamaño de diente. Elegir "el que da más piezas" es la trampa — a 700 HU salen
    # 41 piezas de tamaño de diente Y un bloque de 112 mm con la mandíbula entera; las 41
    # son los restos de alrededor. Mientras la mayor mida 100 mm, no hay separación.
    candidatas = [
        f for f in filas
        if f.get("_piezas") and 0 < f["extension_mayor"] <= DIENTE_MAX_MM
    ]
    if not candidatas:
        print(
            "\n✗ A ningún umbral queda una componente mayor con tamaño de diente: lo que\n"
            f"  hay es el bloque diente + hueso alveolar. Con un paso de {paso:.3f} mm y un\n"
            "  ligamento periodontal de 0,15–0,38 mm, la frontera no está muestreada y el\n"
            "  compuesto NO se monta.\n\n"
            "  ⚠ Esto es un negativo sobre ESTE método —umbral de HU + componentes\n"
            "  conexas—, no sobre el dato. Un segmentador entrenado puede separar raíz de\n"
            "  hueso con prior de forma donde el contraste no llega; lo que queda\n"
            "  demostrado es que umbralizar no basta."
        )
        return 2

    mejor = min(candidatas, key=lambda f: f["hu"])
    piezas, sel = mejor["_piezas"], mejor["_sel"]
    print(f"\n✓ umbral elegido {mejor['hu']} HU → {len(piezas)} pieza(s) "
          f"(se esperaban ~{args.esperados})")

    # --- 3 · el nombre (FDI) sale del IOS; la forma, del CBCT ----------------- #
    malla = parse_stl(args.ios)
    V_ios = np.asarray(malla["positions"], dtype=np.float64)
    r = icp(V_ios, centros[sel], trim=0.8)
    V_ios = apply(quaternion_to_matrix(r.rotation), np.asarray(r.translation), V_ios)
    print(f"registro IOS → campo del CBCT: {r.rms_efectivo_mm:.3f} mm")

    fdi_ios = (
        np.load(args.fdi).astype(np.int64)
        if args.fdi
        else np.ones(len(V_ios), dtype=np.int64)
    )
    if len(fdi_ios) != len(V_ios):
        print(f"✗ `--fdi` trae {len(fdi_ios)} etiquetas y la malla {len(V_ios)} vértices.")
        return 1

    centros_sel = centros[sel]
    nombres = nombra_por_vecindad([centros_sel[p] for p in piezas], V_ios, fdi_ios)
    con_nombre = sum(1 for n in nombres if n)
    print(f"piezas con código FDI asignado: {con_nombre}/{len(piezas)}")

    # --- 4 · el compuesto: dientes del CBCT + encía del IOS ------------------- #
    region = np.zeros(len(centros), dtype=np.int16)
    indices_globales = np.flatnonzero(sel)
    for idx, nombre in zip(piezas, nombres, strict=True):
        region[indices_globales[idx]] = nombre

    encia = fdi_ios == 0
    print(f"\ncompuesto: {int((region > 0).sum()):,} gaussianas de diente (CBCT) + "
          f"{int(encia.sum()):,} vértices de encía (IOS)")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    escribe_ply(
        args.salida,
        {
            "x": centros[:, 0], "y": centros[:, 1], "z": centros[:, 2],
            "scale_0": campo["scales"][:, 0], "scale_1": campo["scales"][:, 1],
            "scale_2": campo["scales"][:, 2],
            "rot_0": campo["rotations"][:, 0], "rot_1": campo["rotations"][:, 1],
            "rot_2": campo["rotations"][:, 2], "rot_3": campo["rotations"][:, 3],
            "density": campo["density"],
            "region_id": region,
        },
        comentarios=[
            "compuesto dientes-CBCT + encia-IOS",
            f"umbral de separacion {mejor['hu']} HU",
            f"registro IOS->CBCT {r.rms_efectivo_mm:.3f} mm",
            "region_id es el codigo FDI por gaussiana, 0 = sin asignar",
        ],
    )
    print(f"escrito {args.salida}")

    # --- 5 · hasta dónde llega el recorte ------------------------------------ #
    print("\n--- hasta dónde baja cada pieza antes de fundirse con el hueso ---")
    print(f"{'FDI':>5} {'gaussianas':>11} {'altura mm':>10}")
    for idx, nombre in sorted(zip(piezas, nombres, strict=True), key=lambda p: -p[1]):
        alto = float(np.ptp(centros_sel[idx][:, 2]))
        print(f"{nombre or '—':>5} {len(idx):>11,} {alto:>10.1f}")
    print(
        "\n⚠ La altura es lo que se recorta HOY. Una corona mide 7-9 mm y un diente\n"
        "  entero 20-25: lo que pase de ~10 mm ya está arrastrando hueso, no raíz."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
