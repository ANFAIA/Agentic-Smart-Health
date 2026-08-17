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
    """Puntos de esmalte del lóbulo de **z alta** —el maxilar—, y el corte que se usó.

    El nombre es correcto, y conviene saber **por qué** lo es en vez de darlo por hecho.
    La `z` de estas nubes sale de `Serie.z`, que es `ImagePositionPatient[2]`, y `IPP`
    viene ya en el sistema del paciente (LPS: +z craneal). No es una convención del
    fichero ni depende de `PatientPosition`: el equipo la resuelve al escribirla. Así que
    z alta es craneal es **maxilar**, y `Serie.z_es_superior` dice cuándo esto vale.

    Comprobado sobre `histora` por dos vías independientes: el DICOM (`IPP[2]` de −11,7 a
    +161,4 mm en orden de corte) y la anatomía (subiendo desde el plano oclusal aparece
    aire dentro del hueso —seno maxilar y fosa nasal— hasta el 51,5 % a 33 mm, mientras
    que bajando hay **0,0 %** de aire y hueso continuo: cuerpo mandibular).

    ⚠ Lo que **no** puede usarse para decidir esto es el residuo del registro: ver
    `separa_arcadas`, donde está medido que no discrimina.
    """
    corte = plano_oclusal(esmalte[:, 2])
    return esmalte[esmalte[:, 2] >= corte], corte


BANDA_ARCADA_MM = 22.0     # altura de coronas alrededor del plano oclusal
PERCENTIL_DENSIDAD = 50


def _limpia_lobulo(puntos: np.ndarray) -> np.ndarray:
    """Quita del lóbulo lo que no es corona: hueso de fondo y estrías del metal.

    Sin esto, un lóbulo entero mide **122 × 126 × 49 mm** —el saco descrito en la
    decisión 2 del encabezado, el fallo que dio 8 mm—. Con la banda de altura y el
    filtro de densidad queda del orden de **50 × 37 × 20 mm**, que sí es una arcada.

    La densidad separa bien porque el esmalte forma cúmulos compactos y las estrías del
    metal van sueltas por todo el campo.
    """
    if len(puntos) < 3:
        return puntos
    centro = np.median(puntos[:, 2])
    banda = puntos[np.abs(puntos[:, 2] - centro) < BANDA_ARCADA_MM / 2]
    if len(banda) < 3:
        return banda
    vecinos = np.array([len(v) for v in cKDTree(banda).query_ball_point(banda, 3.0)])
    return banda[vecinos >= np.percentile(vecinos, PERCENTIL_DENSIDAD)]


def separa_arcadas(esmalte: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Devuelve los **dos lóbulos limpios** y el corte: `(z_alta, z_baja, corte)`.

    Quién es quién lo dice el DICOM, no esta función: con `Serie.z_es_superior`, `z_alta`
    es el **maxilar** y `z_baja` la **mandíbula** (ver `arcada_superior`). Se devuelven
    los dos porque hay pipelines que necesitan ambos, y porque tener el otro a mano
    permite el aviso de más abajo.

    ⚠ **Lo que está medido que NO sirve para decidir la arcada es el residuo.** Se probó
    registrar contra los dos lóbulos y quedarse con el que ajusta: sobre el escaneo
    mandibular de `histora` salió **0,490 mm contra z_baja y 0,509 contra z_alta**, un
    3,8 % de diferencia. Una arcada dental se parece bastante a otra arcada dental, y con
    recorte agresivo una pose equivocada puntúa bien — es el modo de fallo que advierte el
    encabezado. Por eso el llamante compara los dos ajustes solo para **avisar**, nunca
    para elegir.

    ⚠ **Los criterios geométricos tampoco valen.** Se probaron dos y los dos fallan:

    - **anchura del arco** (el maxilar es más ancho porque resalta sobre el inferior):
      da respuestas OPUESTAS según se mida el `ptp` sobre el lóbulo completo o sobre su
      cresta, porque los ejes salen de la PCA del lóbulo y el subconjunto de cresta no
      tiene por qué extenderse más en el mismo eje.
    - **masa ósea a cada lado**: el cuerpo mandibular es denso, pero el macizo facial
      también, y los números no separan (361k/455k frente a 511k/270k).

    Lo que **sí** discrimina, además del `IPP`, es el **aire intraóseo**: subiendo desde
    el plano oclusal aparecen el seno maxilar y la fosa nasal (0 → 51,5 % de vóxeles bajo
    −500 HU a 33 mm) y bajando no hay ninguno (0,0 % en 33 mm, con hueso continuo). Es
    anatomía y coincide con la metadata.
    """
    corte = plano_oclusal(esmalte[:, 2])
    return (
        _limpia_lobulo(esmalte[esmalte[:, 2] >= corte]),
        _limpia_lobulo(esmalte[esmalte[:, 2] < corte]),
        corte,
    )


def arcada_del_escaneo(ruta: Path) -> str | None:
    """`"maxilar"`, `"mandibular"` o `None` si el nombre del fichero no lo dice.

    Es una **etiqueta del operador**, no una medida: la pone el software del escáner al
    exportar. Se usa porque la alternativa —deducir la arcada de la geometría— se probó y
    falla (ver `separa_arcadas`), y porque una etiqueta equivocada se destapa: el aviso de
    `main()` salta si el otro lóbulo ajusta mejor que el que la etiqueta predice.

    No se acepta un fichero que diga las dos cosas, ni uno que no diga ninguna. Adivinar
    aquí es lo que dejó un registro de la mandíbula contra el maxilar puntuando 0,452 mm.
    """
    n = ruta.name.lower()
    arriba = any(s in n for s in ("upper", "maxilar", "superior"))
    abajo = any(s in n for s in ("lower", "mandibular", "inferior"))
    if arriba == abajo:            # ninguna, o las dos: no hay etiqueta utilizable
        return None
    return "maxilar" if arriba else "mandibular"


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
    # No hay `--arcada`: cuál de los dos lóbulos del CBCT corresponde al escaneo lo
    # decide el AJUSTE, no un nombre. Ver `separa_arcadas`.
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

    z_alta, z_baja, corte = separa_arcadas(esmalte)
    lobulos = {"z_alta": z_alta, "z_baja": z_baja}
    print(f"  esmalte HU≥{HU_ESMALTE}: {len(esmalte)} vóxeles")
    print(f"  plano oclusal en z = {corte:.1f} mm")
    for nombre, p in lobulos.items():
        print(f"  lóbulo {nombre}: {len(p):>7,} pts · bbox {np.ptp(p, axis=0).round(1)}"
              f" mm · {'arcada' if parece_arcada(p) else 'NO parece arcada'}")
    utiles = {n: p for n, p in lobulos.items() if parece_arcada(p)}
    if not utiles:
        print(
            "✗ Ningún lóbulo tiene forma de arcada. Casi seguro el metal y sus estrías "
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

    # Se registra contra LOS DOS lóbulos, pero la elección NO sale del ajuste: está
    # medido que no discrimina (0,490 frente a 0,509 mm). Sale del DICOM, y el segundo
    # ajuste queda solo como aviso. Ver `separa_arcadas`.
    ajustes = {}
    for nombre, p in utiles.items():
        print(f"\n{'─' * 72}\nESCÁNER INTRAORAL  ↔  LÓBULO {nombre.upper()} DEL CBCT")
        _, _, d = buscar_pose(ios, p, **busqueda)
        ajustes[nombre] = (informe(d, f"escáner ↔ esmalte del CBCT ({nombre})"), d)

    esperado = arcada_del_escaneo(args.ios)
    if esperado is None or not serie.z_es_superior:
        motivo = ("el nombre del STL no dice la arcada"
                  if esperado is None else "la serie no trae ImagePositionPatient")
        elegido = min(ajustes, key=lambda n: ajustes[n][0])
        print(f"\n  ⚠ {motivo}: cayendo al AJUSTE, que está medido que apenas "
              f"discrimina. Trata la elección como no verificada.")
    else:
        elegido = "z_alta" if esperado == "maxilar" else "z_baja"
        print(f"\n  arcada del escaneo: {esperado} (nombre del STL) → lóbulo {elegido} "
              f"(IPP[2] crece hacia craneal)")
        if elegido not in ajustes:
            print(f"✗ El lóbulo {elegido} no tiene forma de arcada, así que el escaneo "
                  f"{esperado} no tiene contra qué registrarse.", file=sys.stderr)
            return 3
    rms, d_real = ajustes[elegido]
    if len(ajustes) == 2:
        otro = next(n for n in ajustes if n != elegido)
        print(f"  ajuste: {elegido} {rms:.3f} mm · {otro} {ajustes[otro][0]:.3f} mm")
        if ajustes[otro][0] < rms * 1.3:
            print("  ⚠ los dos lóbulos ajustan casi igual. Es lo ESPERADO —una arcada se "
                  "parece a otra—, y la razón de no elegir por ajuste.")
        if ajustes[otro][0] < rms:
            print("  ⚠ el otro lóbulo ajusta MEJOR que el que dice la anatomía. O el "
                  "nombre del STL miente, o el registro falló: revísalo a mano.")

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
