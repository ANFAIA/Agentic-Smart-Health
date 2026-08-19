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

sys.path.insert(0, str(RAIZ / "scripts"))
from registro_ios_cbct import arcada_del_escaneo, plano_oclusal  # noqa: E402

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

# A más de esto de cualquier corona etiquetada, una gaussiana se queda SIN nombre. Un
# diente entero mide ~22 mm y la corona ocupa los 8 superiores, así que el ápice de una
# raíz queda a ~15 mm de la corona más cercana. Con menos, las raíces se quedarían mudas;
# con mucho más, el hueso de alrededor heredaría el FDI del diente vecino.
RADIO_NOMBRE_MM = 16.0

# HU mínima del objetivo con el que se registra el escáner intraoral.
#
# ⚠️ El objetivo NO es la máscara de diente entera. El escáner ve la **corona**, así que
# registrar corona contra corona es la correspondencia correcta; meterle la raíz y el
# hueso que el modelo marca de más le da al ICP cosas equivocadas con las que encajar.
# Medido sobre el maxilar de `histora`, mirando de corona a gaussiana (la dirección que
# no contamina el sobre-marcado):
#
#     objetivo = todo el diente     p50 5,48 mm ·  18 % por debajo de 2 mm
#     objetivo = diente + HU>=1200  p50 0,81 mm ·  78 % por debajo de 2 mm
#
# Y el aviso que va con ello: el rms del propio ICP pasó de 0,622 a 0,642 — **empeoró**
# mientras la calidad real mejoraba 7×. El residuo del registro no discrimina, y es el
# mismo modo de fallo que documenta `registro_ios_cbct.separa_arcadas`.
HU_CORONA = 1200.0

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


def probabilidad_por_modelo(volumen: np.ndarray, checkpoint: Path, *, lado: int = 96):
    """Probabilidad de diente por vóxel, según el segmentador entrenado.

    Devuelve la **probabilidad** y no la máscara: umbralizarla es barato y la inferencia
    no, así que barrer el umbral de decisión sobre un solo pase cuesta segundos en vez de
    repetir el volumen entero por cada valor.

    Ventana deslizante SIN solape, la misma con la que se midió el F1 del modelo: si aquí
    se solapara y allí no, el número publicado no describiría a este código.

    `torch` se importa dentro a propósito. Este script corre en el venv normal cuando usa
    el umbral, y solo necesita GPU en esta rama — importarlo arriba obligaría a todo el
    script a vivir en el entorno de GPU sin motivo.
    """
    import torch

    sys.path.insert(0, str(RAIZ / "scripts"))
    from entrena_diente_cbct import HU_MAX, HU_MIN, UNet3D, normaliza

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=dev)
    modelo = UNet3D().to(dev)
    modelo.load_state_dict(ckpt["modelo"])
    modelo.eval()
    print(f"modelo: {checkpoint.name} · F1 de validacion {ckpt.get('f1_parche', float('nan')):.3f} "
          f"(por parches) · HU {HU_MIN:g}..{HU_MAX:g} · {dev}")

    pad = [(0, (-s) % lado) for s in volumen.shape]
    hu = normaliza(np.pad(volumen, pad))
    fuera = np.zeros(hu.shape, dtype=np.float32)
    with torch.no_grad():
        for z in range(0, hu.shape[0], lado):
            for y in range(0, hu.shape[1], lado):
                for x in range(0, hu.shape[2], lado):
                    s = (slice(z, z + lado), slice(y, y + lado), slice(x, x + lado))
                    xt = torch.from_numpy(hu[s]).unsqueeze(0).unsqueeze(0).to(dev)
                    fuera[s] = torch.sigmoid(modelo(xt))[0, 0].cpu().numpy()
    recorte = tuple(slice(0, s) for s in volumen.shape)
    return fuera[recorte]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cbct", type=Path, required=True, help="Directorio de la serie DICOM.")
    ap.add_argument("--ios", type=Path, nargs="+", required=True,
                    help="Mallas intraorales (STL). Se pueden dar LAS DOS arcadas.")
    ap.add_argument("--fdi", type=Path, nargs="+",
                    help="`region_id` por vértice, uno por escaneo y en el mismo orden.")
    ap.add_argument("--salida", type=Path, default=RAIZ / "data/processed/compuesto.ply")
    ap.add_argument("--artefactos", type=Path, default=RAIZ / "data/processed/artifacts")
    ap.add_argument("--esperados", type=int, default=14, help="Dientes que se esperan.")
    ap.add_argument(
        "--modelo", type=Path, default=None,
        help="Checkpoint del segmentador de CBCT. Sin el, se cae al barrido de umbral — "
             "que esta medido que NO separa diente de hueso (F1 0,530 contra verdad "
             "anotada). El modelo es la via, el umbral es el control.",
    )
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

    # --- 2 · quién dice qué es diente: el modelo, o el umbral como control ---- #
    if args.modelo is not None:
        from ingestion_agents.cbct_agent import _read_series

        serie = _read_series(args.cbct)
        prob = probabilidad_por_modelo(serie.volume, args.modelo)

        # De vóxel a gaussiana. El `cbct-agent` construye los centros como
        # `(col*sx, fila*sy, serie.z[corte])` y luego resta el centroide, así que se
        # deshace: mundo = centros + origin, y de ahí a índices. `serie.z` NO es
        # `indice*espaciado` —un corte ausente desplazaría todo lo de encima— así que la
        # coordenada z se busca por vecino más próximo en el eje real.
        sx, sy, _ = serie.spacing
        mundo = centros + np.asarray(campo["origin"], dtype=np.float64)
        ix = np.clip(np.rint(mundo[:, 0] / sx).astype(int), 0, serie.volume.shape[2] - 1)
        iy = np.clip(np.rint(mundo[:, 1] / sy).astype(int), 0, serie.volume.shape[1] - 1)
        iz = np.abs(mundo[:, 2, None] - np.asarray(serie.z)[None, :]).argmin(axis=1)
        p_gauss = prob[iz, iy, ix]

        # ⚠️ El CBCT trae LAS DOS ARCADAS y el IOS es solo una. Sin separar, el modelo
        # marca —correctamente— los dientes del maxilar, que luego quedan a 40 mm de
        # cualquier corona mandibular: medido, 43.786 de 58.533 gaussianas se quedaban
        # sin nombre y una «pieza» abarcaba 36,9 mm cosiendo dientes de las dos arcadas.
        # Cuál es cuál lo dice el nombre del fichero del escaneo, nunca el residuo.
        # Con las DOS arcadas escaneadas no se aisla nada: el CBCT las tiene y ahora
        # ambas tienen quien las nombre. Aislar era el parche para cuando solo había una.
        arcadas = {arcada_del_escaneo(m) for m in args.ios} - {None}
        quiere = next(iter(arcadas)) if len(arcadas) == 1 else None
        if quiere is not None:
            corte = plano_oclusal(centros[p_gauss > 0.5][:, 2])
            lado = (centros[:, 2] >= corte) if quiere == "maxilar" else (centros[:, 2] < corte)
            p_gauss = np.where(lado, p_gauss, 0.0)
            print(f"arcada del escaneo: {quiere} · plano oclusal z={corte:.1f} → "
                  f"{int((p_gauss > 0.5).sum()):,} gaussianas de diente en ese lóbulo")
        else:
            print(f"escaneos de {len(arcadas) or 'arcada indeterminada'} arcada(s): "
                  "no se aísla ningún lóbulo")

        # Barrido del umbral de DECISIÓN (no de HU: esto es la probabilidad del modelo).
        #
        # A 0,5 el modelo troceaba: 26 piezas para 14 dientes, con el `43` en seis
        # fragmentos y la mayor de 40,5 mm — dos piezas fusionadas. Encaja con su
        # precisión medida (0,588 frente a recall 0,948): sobre-predice, y las componentes
        # conexas parten lo que queda mal conectado. Subir el umbral recorta lo dudoso.
        print("\n--- barrido del umbral de decisión del modelo ---")
        print(f"{'umbral':>7} {'gaussianas':>11} {'comp.':>6} {'tam.diente':>11} "
              f"{'mayor mm':>9} {'FDI':>5}")
        opciones = []
        for u in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            s = p_gauss > u
            if s.sum() < MIN_GAUSSIANAS:
                continue
            todas = piezas_de(componentes(centros[s], RADIO_VECINDAD * paso))
            if not todas:
                continue
            pz = [q for q in todas if parece_diente(centros[s][q])]
            mayor = float(np.ptp(centros[s][todas[0]], axis=0).max())
            opciones.append({"u": u, "sel": s, "piezas": pz, "mayor": mayor,
                             "comp": len(todas)})
            print(f"{u:>7.2f} {int(s.sum()):>11,} {len(todas):>6} {len(pz):>11} "
                  f"{mayor:>8.1f} {'':>5}")

        # El bueno: la componente mayor con tamaño anatómico —sin bloques fusionados— y,
        # entre los que lo cumplen, el MÁS BAJO, que es el que conserva más raíz.
        validos = [o for o in opciones if o["mayor"] <= DIENTE_MAX_MM]
        elegido = min(validos, key=lambda o: o["u"]) if validos else max(
            opciones, key=lambda o: len(o["piezas"]))
        sel, piezas = elegido["sel"], elegido["piezas"]
        print(f"\n✓ umbral de decisión {elegido['u']:.2f} → {len(piezas)} pieza(s), "
              f"mayor {elegido['mayor']:.1f} mm"
              + ("" if validos else "  ⚠ NINGUNO deja la mayor bajo 25 mm"))
        if not piezas:
            print("✗ ninguna componente con tamaño de pieza. El compuesto NO se monta.")
            return 2
        mejor = {"hu": f"modelo:{args.modelo.name} @ {elegido['u']:.2f}"}
    else:
        mejor, piezas, sel = _por_umbral(centros, hu, paso, args)
        if mejor is None:
            return 2

    return _compone(args, store, campo, centros, hu, sel, piezas, mejor)


def _por_umbral(centros, hu, paso, args):
    """El control: el barrido de umbral, que está medido que NO separa (F1 0,530)."""
    print("\n--- barrido de separación (el CONTROL: umbral de HU) ---")
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
        return None, None, None

    mejor = min(candidatas, key=lambda f: f["hu"])
    piezas = mejor["_piezas"]
    print(f"\n✓ umbral elegido {mejor['hu']} HU → {len(piezas)} pieza(s) "
          f"(se esperaban ~{args.esperados})")
    return mejor, mejor["_piezas"], mejor["_sel"]


def _compone(args, store, campo, centros, hu, sel, piezas, mejor) -> int:
    """El compuesto: nombre desde el IOS, y el PLY con `region_id` por gaussiana."""
    # --- 3 · el nombre (FDI) sale del IOS; la forma, del CBCT ----------------- #
    #
    # Cada arcada se registra POR SEPARADO contra el campo. Registrarlas juntas exigiría
    # que la relación entre maxilar y mandíbula fuese la misma en el escáner que en el
    # CBCT, y no lo es: son adquisiciones distintas y la boca se abre y se cierra entre
    # ellas. Con dos rígidas independientes cada arcada cae donde le corresponde, que es
    # además como el paciente las llevaba en cada momento.
    etiquetas = args.fdi or [None] * len(args.ios)
    if len(etiquetas) != len(args.ios):
        print(f"✗ {len(args.ios)} escaneo(s) y {len(etiquetas)} fichero(s) de etiquetas.")
        return 1

    centros_sel = centros[sel]
    # Con las dos arcadas, cada una se registra y se nombra **contra su propio lóbulo**.
    #
    # ⚠️ Registrar un escaneo mandibular contra la nube de las DOS arcadas no funciona, y
    # está medido: 41.133 de 58.533 gaussianas se quedaban sin nombre y salían piezas de
    # 35 mm. El ICP encuentra una pose con buen residuo (0,604 mm) porque una arcada se
    # parece a la otra, y a partir de ahí cada gaussiana toma el FDI de la corona más
    # cercana, que puede ser la de enfrente. Dos adquisiciones, dos rígidas, dos lóbulos.
    corte_oclusal = plano_oclusal(centros_sel[:, 2])
    encia_total, rms = 0, []
    coronas_todas, etiquetas_todas = [], []

    for malla_p, fdi_p in zip(args.ios, etiquetas, strict=True):
        V = np.asarray(parse_stl(malla_p)["positions"], dtype=np.float64)
        f = np.load(fdi_p).astype(np.int64) if fdi_p else np.ones(len(V), dtype=np.int64)
        if len(f) != len(V):
            print(f"✗ {fdi_p} trae {len(f)} etiquetas y la malla {len(V)} vértices.")
            return 1

        # El lóbulo se usa SOLO para registrar: dar al ICP la mitad que le corresponde
        # evita que encaje una arcada sobre la otra, que se parecen lo bastante como para
        # puntuar bien (0,604 mm) estando mal.
        arc = arcada_del_escaneo(malla_p)
        if arc == "maxilar":
            lobulo = centros_sel[:, 2] >= corte_oclusal
        elif arc == "mandibular":
            lobulo = centros_sel[:, 2] < corte_oclusal
        else:
            lobulo = np.ones(len(centros_sel), dtype=bool)

        objetivo_reg = lobulo & (hu[sel] >= HU_CORONA)
        if objetivo_reg.sum() < 500:  # sin corona suficiente, mejor todo el lóbulo
            objetivo_reg = lobulo
        r = icp(V, centros_sel[objetivo_reg], trim=0.8)
        V = apply(quaternion_to_matrix(r.rotation), np.asarray(r.translation), V)
        rms.append(r.rms_efectivo_mm)
        coronas_todas.append(V[f > 0])
        etiquetas_todas.append(f[f > 0])
        encia_total += int((f == 0).sum())
        d_ctrl, _ = cKDTree(centros_sel).query(V[f > 0])
        print(f"{arc or 'arcada ?':<12} rms {r.rms_efectivo_mm:.3f} mm · objetivo "
              f"{int(objetivo_reg.sum()):,} gaussianas de corona · corona→gaussiana p50 "
              f"{np.median(d_ctrl):.2f} mm, {(d_ctrl < 2).mean():.0%} bajo 2 mm")

    # Pero NOMBRAR no se hace por lóbulo, sino contra las coronas de LAS DOS arcadas.
    #
    # ⚠️ Nombrar dentro del lóbulo estaba mal, y está medido: los incisivos superiores
    # cuelgan POR DEBAJO del plano oclusal, así que un corte horizontal los mete en el
    # lóbulo inferior y se quedan mudos. Se veía en el resultado — 16, 17, 26 y 27 salían
    # con raíz y el 12, 13, 15, 22, 23 con menos de 20 gaussianas cada uno — y en la
    # distancia mediana a corona del lóbulo superior: 36,9 mm frente a 4,2 en el inferior.
    #
    # Con el árbol conjunto no hace falta suponer dónde está la frontera: cada gaussiana
    # toma el FDI de la corona más cercana, sea de la arcada que sea. La anatomía decide.
    coronas = np.concatenate(coronas_todas)
    etq = np.concatenate(etiquetas_todas)
    d, vecino = cKDTree(coronas).query(centros_sel)
    nombres_gauss = np.where(d <= RADIO_NOMBRE_MM, etq[vecino], 0)

    # El nombre se asigna POR GAUSSIANA, no por componente conexa.
    #
    # ⚠️ Nombrar componentes enteras no funciona, y está medido: los dientes se tocan en
    # el punto de contacto interproximal, así que una componente puede abarcar dos piezas
    # —salió una de 40,5 mm, con 25 de máximo anatómico— y darle un solo FDI seria
    # inventar. Subir el umbral de decisión del modelo de 0,50 a 0,95 NO la rompe: el
    # modelo está seguro de esos vóxeles, porque ahí hay diente de verdad. Es el mismo
    # muro que ya tumbó la separación por islas conexas sobre la malla del escáner.
    #
    # La salida es que el IOS **ya trae los dientes separados**: cada gaussiana toma el
    # FDI de la corona etiquetada más cercana. La separación diente-hueso la pone el
    # modelo del CBCT; la separación diente-diente, el escáner. Cada modalidad aporta lo
    # que sabe.
    print(f"\nnombrado total: {int((nombres_gauss > 0).sum()):,} de "
          f"{len(centros_sel):,} gaussianas · "
          f"{len(set(nombres_gauss[nombres_gauss > 0].tolist()))} código(s) FDI")

    # --- 4 · el compuesto: dientes del CBCT + encía del IOS ------------------- #
    region = np.zeros(len(centros), dtype=np.int16)
    region[np.flatnonzero(sel)] = nombres_gauss.astype(np.int16)

    print(f"\ncompuesto: {int((region > 0).sum()):,} gaussianas de diente (CBCT) + "
          f"{encia_total:,} vértices de encía (IOS)")

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
            f"segmentacion: {mejor['hu']}",
            "registro IOS->CBCT " + " / ".join(f"{x:.3f} mm" for x in rms),
            "region_id es el codigo FDI por gaussiana, 0 = sin asignar",
        ],
    )
    print(f"escrito {args.salida}")

    # --- 5 · hasta dónde llega el recorte ------------------------------------ #
    print("\n--- cada diente compuesto, y hasta dónde llega ---")
    print(f"{'FDI':>5} {'gaussianas':>11} {'altura mm':>10}  veredicto")
    for fdi in sorted(set(nombres_gauss[nombres_gauss > 0].tolist())):
        pts = centros_sel[nombres_gauss == fdi]
        alto = float(np.ptp(pts[:, 2]))
        # Una corona sola mide 7-9 mm; con raíz, 18-25. La altura dice cual de las dos
        # cosas se recorto, que es LA pregunta que este experimento venia arrastrando.
        v = "corona sola" if alto < 12 else ("con raiz" if alto <= 26 else "⚠ desbordada")
        print(f"{fdi:>5} {len(pts):>11,} {alto:>10.1f}  {v}")
    print(
        "\n⚠ La altura es lo que se recorta HOY. Una corona mide 7-9 mm y un diente\n"
        "  entero 20-25: lo que pase de ~10 mm ya está arrastrando hueso, no raíz."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
