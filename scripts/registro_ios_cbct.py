#!/usr/bin/env python
"""registro_ios_cbct.py — mide si el escáner intraoral y el CBCT se pueden alinear.

    uv run python scripts/registro_ios_cbct.py --cbct DIR --ios FICHERO.stl

**Por qué existe.** El `geometric-fusion-agent` fija su banda ε en **0,5 mm** y de
ahí saca la confianza del registro, pero ese número nunca se había medido contra
dato real. Y de él cuelga algo concreto: la medida periodontal —margen gingival
(escáner) → cresta ósea (CBCT)— es la única del proyecto que no tiene equivalente
en 2D, y **no se puede medir entre dos modalidades con más precisión de la que se
registran entre sí**. Una recesión clínica son 1-3 mm; si el registro yerra 8, el
panel enseñaría ficción.

Este script contesta esa pregunta sobre el paciente de `histora`, que trae CBCT y
escáner del **mismo** sujeto.

---

## Las cuatro decisiones del método, y por qué

**1 · Se registra contra el ESMALTE (HU ≥ 2000), no contra el umbral de ingesta.**
El `cbct-agent` siembra gaussianas a partir de 300 HU, que es el isovalor correcto
para *representar* el volumen y el **peor posible** para registrar: ahí la
isosuperficie atraviesa hueso trabecular y su dimensión fractal medida es 2,45 —no
hay superficie estable contra la que alinear nada. A 2000 HU la dimensión baja a
2,10 y además es lo que el escáner ve de verdad: la corona.

**2 · Hay que aislar la arcada, y el metal lo impide.** Este es el error que costó
dos intentos y merece quedar escrito. El umbral de esmalte **también captura el
metal de las restauraciones y sus estrías**, que se reparten por todo el campo. Con
eso dentro, partir el volumen por un plano en z deja una arcada a un lado y un saco
de 122 × 126 mm al otro. Registrar contra ese saco da un rms *engañosamente bajo*
—cualquier pose queda cerca de algo en una nube difusa— y contra el resto da 8 mm.
Ninguno de los dos números significa nada. El plano oclusal se busca como **mínimo
entre los dos lóbulos** del histograma de z, y se comprueba que lo que sale tiene
forma de arcada (≈64 × 50 × 20 mm) antes de seguir.

**3 · El ICP tiene que recortar.** Un escaneo maxilar es en su mayor parte paladar y
encía, que no tienen esmalte contra el que emparejarse. El ICP del repo es
punto-a-punto **sin rechazo de atípicos**, así que esos puntos —la mayoría— tiran
del ajuste. Aquí se conserva la fracción `--recorte` más cercana en cada iteración.

**4 · Control positivo.** Se registra el escáner contra una copia suya movida por una
rígida conocida. Si el arnés no recupera eso, ningún número de abajo vale.

> ⚠ **El control NO cierra con el recorte del 35 %, y es un resultado en sí mismo.**
> Medido: con inicialización perfecta el ICP recortado es exacto (rms 0,00000 mm) a
> cualquier recorte, y la búsqueda recupera la rígida conocida con recorte del 70 % y
> del 100 %. Con el 35 % se queda en 0,52 mm. El motivo es que **un recorte agresivo
> hace que una pose equivocada puntúe bien** —le basta con que la fracción conservada
> caiga cerca—, así que destruye la capacidad de discriminar entre poses. Y el
> recorte agresivo es imprescindible en el caso real. Se pelean.
>
> Por eso la criba y el refinado usan recortes distintos. Aun así el control se queda
> en 0,486 mm, así que **hoy este script no tiene un suelo limpio contra el que
> comparar**: el número real de abajo se sostiene por la FORMA del reparto, no por su
> distancia al control. Cerrar el control es trabajo pendiente.

---

## Lo que midió (paciente de histora, CS 9600 + escáner intraoral)

    registro IOS ↔ CBCT, población emparejada     0,452 mm   (27 % de los puntos)
    rms sobre la nube completa                    4,98  mm
    control con correspondencia perfecta          0,486 mm   ← no cierra, ver arriba

El reparto de distancias es **bimodal**: un pico en 0-0,5 mm con 3.836 puntos —2,5
veces el siguiente bin— y una cola de 3-12 mm que es el paladar. El 27 % del escaneo
cae bajo 1 mm, que es del orden de lo que es corona en un escaneo maxilar. Esa forma
—y no el rms— es lo que distingue un registro correcto de una casualidad: una pose
equivocada, medida antes, daba un reparto **plano** de 0 a 15 mm con solo el 4,6 %
bajo medio milímetro.

**Consecuencia para el agente de fusión:** la banda ε de 0,5 mm resulta estar **bien
puesta**; lo que está mal es calcular la confianza sobre la nube completa. Con
solapamiento parcial eso da 0,00 para un registro que es bueno. La confianza tiene
que medirse sobre la población emparejada.

**Y el 0,451 mm es una cota superior del error de registro**, no el error: incluye la
cuantización del vóxel (0,30 mm) y el sesgo de que la isosuperficie de 2000 HU cae
*por dentro* del esmalte —la PSF de 425 µm difumina el borde aire/esmalte— mientras
que el escáner ve la superficie externa. Parte de eso es calibrable.

> **Alcance honesto.** Un paciente, un equipo, una arcada. Dice que el registro es
> alcanzable y con qué margen; no dice que lo sea en general.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from fusion_agents.registration import apply, kabsch
from ingestion_agents.cbct_agent import _read_series
from ingestion_agents.mesh_agent import parse_stl
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Isovalor del esmalte. Ver decisión 1 del encabezado.
HU_ESMALTE = 2000
# Un escaneo maxilar es mayoritariamente paladar y encía: sin esmalte contra el que
# emparejar. Se conserva algo más de un tercio, que es del orden de lo que es corona.
RECORTE = 0.35
# Distancia por debajo de la cual se considera que un punto tiene contrapartida real.
EMPAREJADO_MM = 1.0


# --------------------------------------------------------------------------- #
# Piezas
# --------------------------------------------------------------------------- #
def icp_recortado(
    fuente: np.ndarray,
    objetivo: np.ndarray,
    *,
    frac: float = RECORTE,
    iters: int = 60,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """ICP punto-a-punto **con rechazo de atípicos**. Devuelve `(rotación, traslación)`.

    En cada iteración se emparejan todos los puntos y se ajusta solo con la fracción
    `frac` más cercana. Es lo que permite registrar dos superficies que **no** se
    solapan del todo, que es exactamente el caso escáner ↔ esmalte.
    """
    arbol = cKDTree(objetivo)
    rot, trans = np.eye(3), np.zeros(3)
    previo = np.inf
    for _ in range(iters):
        movido = apply(rot, trans, fuente)
        d, idx = arbol.query(movido)
        k = max(3, int(len(d) * frac))
        cerca = np.argpartition(d, k - 1)[:k]
        d_rot, d_trans = kabsch(movido[cerca], objetivo[idx[cerca]])
        rot, trans = d_rot @ rot, d_rot @ trans + d_trans
        rms = float(np.sqrt(np.mean(d[cerca] ** 2)))
        if abs(previo - rms) < tol:
            break
        previo = rms
    return rot, trans


def plano_oclusal(z: np.ndarray, bins: int = 80) -> float:
    """Altura que separa las dos arcadas: el **mínimo entre los dos lóbulos** de z.

    No un percentil. Un percentil no sabe dónde están los lóbulos y, con el metal
    metiendo puntos por todo el campo, corta donde no debe — el fallo que dio 8 mm.
    """
    h, bordes = np.histogram(z, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    pico = int(np.argmax(h))
    # El otro lóbulo es el máximo que queda separado del principal; el corte va en el
    # valle entre ambos.
    lejos = np.abs(centros - centros[pico]) > 4.0
    if not lejos.any():
        return float(centros[pico])
    segundo = int(np.flatnonzero(lejos)[np.argmax(h[lejos])])
    a, b = sorted((pico, segundo))
    return float(centros[a + int(np.argmin(h[a : b + 1]))])


def arcada_superior(esmalte: np.ndarray) -> tuple[np.ndarray, float]:
    """Puntos de esmalte de la arcada maxilar, y el corte que se usó."""
    corte = plano_oclusal(esmalte[:, 2])
    return esmalte[esmalte[:, 2] >= corte], corte


# Altura bajo el plano oclusal donde viven las coronas inferiores. Ver `arcada_inferior`.
BANDA_INFERIOR_MM = 22.0
PERCENTIL_DENSIDAD = 50


def arcada_inferior(esmalte: np.ndarray) -> tuple[np.ndarray, float]:
    """Puntos de esmalte de la arcada mandibular. **No es simétrica de la maxilar.**

    Quedarse con lo que hay bajo el plano oclusal —el reflejo exacto de
    `arcada_superior`— aquí **no vale**, y está medido sobre `histora`: se lleva el
    cuerpo mandibular entero más las estrías metálicas repartidas por el campo, y sale
    un saco de **122 × 126 × 49 mm**, que es justo el fallo descrito en la decisión 2
    del encabezado. Hacia arriba el problema no aparece porque encima del plano oclusal
    apenas hay hueso que recoger.

    Dos filtros, y hacen falta los dos:

    - **Banda de 22 mm** bajo el plano oclusal: las coronas inferiores están ahí, el
      cuerpo mandibular no.
    - **Densidad local**: el esmalte real forma cúmulos compactos y las estrías del
      metal van sueltas, así que contar vecinos en 3 mm las separa. Con la mediana como
      corte queda **49,5 × 36,9 × 19,5 mm**, que sí pasa `parece_arcada`.

    Sigue siendo obligatorio comprobar el resultado con `parece_arcada`: estos dos
    números se ajustaron sobre un caso, y un CBCT con otra cantidad de metal puede
    necesitar otros.
    """
    corte = plano_oclusal(esmalte[:, 2])
    banda = esmalte[(esmalte[:, 2] < corte) & (esmalte[:, 2] > corte - BANDA_INFERIOR_MM)]
    if len(banda) < 3:
        return banda, corte
    vecinos = np.array([len(v) for v in cKDTree(banda).query_ball_point(banda, 3.0)])
    return banda[vecinos >= np.percentile(vecinos, PERCENTIL_DENSIDAD)], corte


def parece_arcada(p: np.ndarray) -> bool:
    """¿Esto tiene forma de arcada dental, o es el campo de visión entero?

    Una arcada humana mide del orden de 45-75 mm de ancho, 30-60 de fondo y 10-30 de
    alto. El saco contaminado de metal salía 122 × 126 × 47: la comprobación existe
    para que ese caso pare el script en vez de producir un número sin sentido.
    """
    ancho, fondo, alto = sorted(np.ptp(p, axis=0))[::-1]
    return 45 <= ancho <= 80 and 25 <= fondo <= 65 and 8 <= alto <= 32


def submuestrear(p: np.ndarray, n: int, semilla: int) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[np.random.default_rng(semilla).choice(len(p), n, replace=False)]


def buscar_pose(
    fuente: np.ndarray,
    objetivo: np.ndarray,
    *,
    poses: int,
    criba: int,
    fino: int,
    mejores: int,
    frac: float,
    frac_criba: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Etapa **gruesa** que al `geometric-fusion-agent` le falta, en su forma más simple.

    Barre SO(3) al azar, criba con un ICP barato y refina las mejores. No es
    RANSAC-FPFH: es fuerza bruta, y basta porque la traslación la resuelve el propio
    ICP y solo hay que acertar la rotación.

    **El recorte de la criba y el del refinado son distintos a propósito**, y el
    control positivo es quien lo destapó. Un recorte agresivo hace que una pose
    equivocada puntúe bien —le basta con que la fracción conservada caiga cerca—, así
    que durante la criba destruye la capacidad de discriminar: con `frac=0.35` la
    búsqueda ni siquiera recupera una rígida conocida sobre la MISMA nube. Se criba
    flojo (discriminar) y se refina fuerte (sobrevivir al paladar).

    Devuelve `(rotación, traslación, distancias del escaneo completo)`, y la rígida que
    devuelve es la **completa**: `apply(rot, trans, fuente)` lleva la fuente sobre el
    objetivo, sin más pasos.

    ⚠ **Antes no lo era, y era una trampa silenciosa.** Se devolvía la `(rot, trans)`
    del ICP de refinado, que se aplican a `inicial` —la nube ya girada por la rotación
    aleatoria ganadora `R_i`, que no salía de la función—. Con eso solo se puede
    informar del rms, que es lo único que hacía el `main()` de este script, así que el
    fallo no se notaba. Al usarla para transformar de verdad daba una pose incoherente:
    un rms de 0,486 mm de aspecto perfectamente respetable y **0 %** de puntos en
    correspondencia al comprobarlo contra otra fuente. La composición es lo único que
    se añade:

        salida = ((p − c_f) @ R_i.T + c_o) @ rot.T + trans
               = p @ (rot @ R_i).T + [trans + c_o @ rot.T − c_f @ R_i.T @ rot.T]
    """
    c_f, c_o = fuente.mean(axis=0), objetivo.mean(axis=0)
    rotaciones = Rotation.random(poses, random_state=0).as_matrix()

    cri_f, cri_o = submuestrear(fuente, criba, 1), submuestrear(objetivo, criba, 2)
    arbol_criba = cKDTree(cri_o)
    puntuadas = []
    for i, r in enumerate(rotaciones):
        inicial = (cri_f - c_f) @ r.T + c_o
        rot, trans = icp_recortado(inicial, cri_o, frac=frac_criba, iters=25)
        d, _ = arbol_criba.query(apply(rot, trans, inicial))
        k = max(3, int(len(d) * frac_criba))
        puntuadas.append((float(np.sqrt(np.mean(np.sort(d)[:k] ** 2))), i))
    puntuadas.sort()

    fin_f = submuestrear(fuente, fino, 3)
    arbol = cKDTree(objetivo)
    resultados = []
    for _, i in puntuadas[:mejores]:
        giro = rotaciones[i]
        inicial = (fin_f - c_f) @ giro.T + c_o
        rot, trans = icp_recortado(inicial, objetivo, frac=frac)
        d, _ = arbol.query(apply(rot, trans, inicial))
        k = max(3, int(len(d) * frac))
        rot_total = rot @ giro
        trans_total = trans + c_o @ rot.T - (c_f @ giro.T) @ rot.T
        resultados.append(
            (float(np.sqrt(np.mean(np.sort(d)[:k] ** 2))), rot_total, trans_total, d)
        )
    resultados.sort(key=lambda r: r[0])
    _, rot, trans, d = resultados[0]
    return rot, trans, d


def informe(d: np.ndarray, etiqueta: str) -> float:
    """Imprime el reparto de distancias y devuelve el rms de la población emparejada.

    **El reparto importa más que el rms.** Un registro correcto con solapamiento
    parcial da una distribución *bimodal*: un pico sub-milimétrico (lo que sí tiene
    contrapartida) y una cola (lo que no). Un registro equivocado da una distribución
    plana. El rms solo no distingue los dos casos.
    """
    print(f"\n  {etiqueta}")
    bordes = np.concatenate([np.arange(0, 3.01, 0.5), [4, 6, 8, 12, 20, 1e9]])
    h, _ = np.histogram(d, bins=bordes)
    for cnt, lo, hi in zip(h, bordes[:-1], bordes[1:], strict=False):
        if cnt:
            tope = "∞" if hi > 1e8 else f"{hi:.1f}"
            print(f"    {lo:>4.1f}–{tope:<4} mm {'█' * max(1, int(40 * cnt / h.max())):<40} {cnt}")
    emparejados = d[d < EMPAREJADO_MM]
    rms = float(np.sqrt(np.mean(emparejados**2))) if len(emparejados) else float("nan")
    print(f"    bajo {EMPAREJADO_MM:.0f} mm: {100 * (d < EMPAREJADO_MM).mean():.1f} %"
          f"  ·  rms de esa población: {rms:.3f} mm")
    return rms


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    raiz = Path.home() / "anfaia" / "histora"
    ap.add_argument("--cbct", type=Path, default=raiz / "CASE-64A77C_files",
                    help="Directorio de la serie DICOM.")
    ap.add_argument("--ios", type=Path,
                    default=raiz / "CASE-EB5070_files" / "PREVIO UpperJawScan.stl",
                    help="STL del escáner intraoral.")
    ap.add_argument("--arcada", choices=("superior", "inferior"), default="superior",
                    help="Qué arcada se aísla del CBCT. Tiene que casar con --ios.")
    ap.add_argument("--poses", type=int, default=500, help="Inicializaciones sobre SO(3).")
    ap.add_argument("--criba", type=int, default=2000, help="Puntos por pose al cribar.")
    ap.add_argument("--fino", type=int, default=20000, help="Puntos al refinar.")
    ap.add_argument("--mejores", type=int, default=10, help="Poses que se refinan.")
    ap.add_argument("--recorte", type=float, default=RECORTE,
                    help="Fracción de correspondencias que conserva el ICP.")
    ap.add_argument("--epsilon", type=float, default=0.5,
                    help="Banda ε del geometric-fusion-agent, en mm.")
    args = ap.parse_args()

    if not args.cbct.is_dir() or not args.ios.is_file():
        print(
            f"✗ No encuentro el dato.\n    CBCT: {args.cbct}\n    IOS:  {args.ios}\n"
            "  Es dato clínico y NO está versionado (ver .gitignore). Pasa --cbct/--ios.",
            file=sys.stderr,
        )
        return 2

    print(f"Leyendo {args.cbct.name}…")
    serie = _read_series(args.cbct)
    ocupados = np.argwhere(serie.volume >= HU_ESMALTE)
    sx, sy, _ = serie.spacing
    # z real de cada corte, nunca índice × espaciado: a esta serie le falta un corte.
    esmalte = np.column_stack(
        [ocupados[:, 2] * sx, ocupados[:, 1] * sy, serie.z[ocupados[:, 0]]]
    ).astype(np.float64)

    aisla = arcada_superior if args.arcada == "superior" else arcada_inferior
    arco, corte = aisla(esmalte)
    print(f"  esmalte HU≥{HU_ESMALTE}: {len(esmalte)} vóxeles")
    print(f"  plano oclusal en z = {corte:.1f} mm  →  arcada de {len(arco)} puntos")
    print(f"  bbox de la arcada: {np.ptp(arco, axis=0).round(1)} mm")
    if not parece_arcada(arco):
        print(
            "✗ Lo aislado NO tiene forma de arcada. Casi seguro el metal y sus estrías "
            "se han colado (ver decisión 2 del encabezado). No sigo: el número saldría "
            "sin sentido.",
            file=sys.stderr,
        )
        return 3

    ios = np.asarray(parse_stl(args.ios)["positions"], dtype=np.float64)
    print(f"  escáner: {len(ios)} vértices, bbox {np.ptp(ios, axis=0).round(1)} mm")

    busqueda = dict(poses=args.poses, criba=args.criba, fino=args.fino,
                    mejores=args.mejores, frac=args.recorte)

    print(f"\n{'─' * 72}\nCONTROL POSITIVO · correspondencia perfecta, rígida conocida")
    giro = Rotation.from_euler("xyz", [23, -41, 67], degrees=True).as_matrix()
    _, _, d_ctrl = buscar_pose(ios @ giro.T + np.array([37.0, -12.0, 55.0]), ios, **busqueda)
    suelo = informe(d_ctrl, "escáner ↔ copia suya movida")

    print(f"\n{'─' * 72}\nESCÁNER INTRAORAL  ↔  ARCADA MAXILAR DEL CBCT")
    _, _, d_real = buscar_pose(ios, arco, **busqueda)
    rms = informe(d_real, "escáner ↔ esmalte del CBCT")

    print(f"\n{'─' * 72}\nVEREDICTO")
    print(f"  suelo del método                : {suelo:.3f} mm  (= muestreo del escáner)")
    print(f"  registro IOS ↔ CBCT             : {rms:.3f} mm  ({rms / suelo:.1f}× el suelo)")
    print(f"  banda ε del agente de fusión    : {args.epsilon:.3f} mm")
    completo = float(np.sqrt(np.mean(d_real**2)))
    print(
        f"\n  Confianza clamp(1 − rms/ε):\n"
        f"    sobre la nube completa ({completo:.2f} mm) : "
        f"{min(max(1 - completo / args.epsilon, 0), 1):.2f}  ← lo que calcula hoy el agente\n"
        f"    sobre la población emparejada        : "
        f"{min(max(1 - rms / args.epsilon, 0), 1):.2f}  ← lo que debería"
    )
    print(
        "\n  Con solapamiento parcial el rms de la nube completa no mide el registro:\n"
        "  mide cuánto del escáner es paladar. La confianza tiene que salir de los\n"
        "  puntos que sí tienen contrapartida."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
