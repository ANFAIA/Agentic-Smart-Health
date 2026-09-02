#!/usr/bin/env python
"""mide_segmentacion.py — cuanto se puede DESCARTAR de la segmentacion FDI de un `.uos`.

    uv run python scripts/mide_segmentacion.py CASO.uos

## Que mide, y sobre todo que NO mide

No existe verdad de campo para estos pacientes: nadie ha etiquetado diente a diente sus
escaneos, y el checkpoint que produce las etiquetas se entreno sobre Teeth3DS+, que es
otro escaner y otras bocas. Asi que **no se puede medir el acierto**. Lo unico que se
puede medir sin etiquetas es la **plausibilidad anatomica**, y eso es asimetrico:

- una pieza que mide mas que la corona mas grande de su tipo esta **mal**, seguro;
- una pieza que cabe dentro de su tipo **no esta por eso bien** — puede ser la corona del
  vecino, del tamano correcto y con el nombre equivocado.

Es decir: esto cuenta errores, no aciertos. El porcentaje que imprime es una **cota
superior** de lo que funciona, nunca una nota.

## Las dos pruebas

**1 · Tamano contra el tipo.** Cada corona tiene una caja anatomica conocida
(mesiodistal x vestibulolingual x altura clinica, valores de Wheeler). La diagonal de esa
caja es la mayor distancia que puede haber entre dos puntos de la corona. Se compara con
la diagonal de la caja propia de la pieza etiquetada —en sus ejes principales, no en los
del fichero— y se marca la que se pasa de `TOLERANCIA`.

⚠️ `TOLERANCIA` es un **juicio declarado, no una medida**: el escaner entra un poco en el
surco y los valores de Wheeler son medias de poblacion, asi que hace falta margen. A
1,30 una corona tendria que ser un 30 % mayor que la mayor de su tipo para que salte.

**2 · Simetria contralateral.** El 16 y el 26 son la misma pieza de la misma boca. Sus
areas tienen que parecerse, y cuando una es el triple de la otra es que una de las dos se
comio a un vecino o perdio la mitad. Esta prueba no necesita ninguna constante de
poblacion: la referencia es el propio paciente.

**3 · La frontera, ¿cae en el pliegue?** Las dos anteriores miran el TAMANO de cada pieza
y las dos son ciegas a donde esta puesto el borde: una corona puede tener la diagonal
correcta y la simetria correcta con la frontera corrida dos milimetros sobre la encia. Esta
mira otra cosa — donde el diente sale de la encia hay un pliegue, y la frontera buena cae
dentro de el.

Se mide la concavidad media de los vertices que tocan encia dividida por la concavidad
tipica de cualquier punto de la malla. El denominador la hace inmune a la escala del
escaner y a la densidad del mallado, asi que la cifra se puede comparar entre casos.

⚠️ **Y aqui el umbral SI tiene calibracion detras**, a diferencia de `TOLERANCIA`. Sobre 12
maxilares de Teeth3DS+ con anotacion de experto la razon sale **+1,82**, con un rango de
1,50 a 2,25 — el margen gingival es un pliegue y se mide. Una segmentacion que salga cerca
de **cero** no tiene la frontera mal puesta: la tiene puesta donde la superficie no hace
nada, o sea al azar respecto de la anatomia. Y eso es un diagnostico distinto de «se pasa
de ancho», porque explica por que afinar el modelo no la mueve.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

RESUMEN_EN = "Measures how much FDI segmentation can be discarded from a `.uos` file."

# Wheeler, dientes permanentes: (mesiodistal, vestibulolingual, altura de corona) en mm.
# La clave es el ultimo digito del FDI; los dos primeros cuadrantes son maxilar.
CORONA_MAXILAR = {
    1: (8.5, 7.0, 10.5), 2: (6.5, 6.0, 9.0), 3: (7.5, 8.0, 10.0), 4: (7.0, 9.0, 8.5),
    5: (7.0, 9.0, 8.5), 6: (10.0, 11.0, 7.5), 7: (9.0, 11.0, 7.0), 8: (8.5, 10.0, 6.5),
}
CORONA_MANDIBULAR = {
    1: (5.0, 6.0, 9.0), 2: (5.5, 6.5, 9.5), 3: (7.0, 7.5, 11.0), 4: (7.0, 7.5, 8.5),
    5: (7.0, 8.0, 8.0), 6: (11.0, 10.5, 7.5), 7: (10.5, 10.0, 7.0), 8: (10.0, 9.5, 7.0),
}

TOLERANCIA = 1.30
# Dos coronas contralaterales de la misma boca por debajo de esto son la misma pieza vista
# dos veces; por encima, una de las dos tiene material que no es suyo. 1,50 deja sitio a
# la asimetria real de una boca con patologia sin dejar pasar un factor 2.
TOLERANCIA_ESPEJO = 1.50

# Calibrado, no declarado: 12 maxilares de Teeth3DS+ con anotacion de experto dan 1,82
# (rango 1,50-2,25). El umbral se pone en 1,00 —bien por debajo del peor experto— para que
# solo salte una frontera que de verdad no esta en el pliegue, no una anotacion algo floja.
RAZON_CONCAVIDAD_MINIMA = 1.00
# Por debajo de esto la razon es ruido: no hay bastante frontera diente/encia que medir.
MINIMO_VERTICES_MARGEN = 50

SEGMENTACION = "derived/seg_teeth.bin"
ESCENA = "scene/scene.glb"


def _caja_anatomica(fdi: int) -> tuple[float, float, float]:
    return (CORONA_MAXILAR if fdi // 10 in (1, 2, 5, 6) else CORONA_MANDIBULAR)[fdi % 10]


def lee_escena(glb: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones y triangulos de un GLB, uniendo TODAS las primitivas.

    ⚠️ Las primitivas comparten el accesor de POSITION, asi que unirlas es concatenar
    indices y **no** re-indexar: el orden de vertices se conserva, que es justo lo que
    hace que el cruce por indice con `seg_teeth.bin` sea exacto.
    """
    desp, trozos = 12, []
    while desp < len(glb):
        largo, _ = struct.unpack("<II", glb[desp:desp + 8])
        desp += 8
        trozos.append(glb[desp:desp + largo])
        desp += largo
    cab = json.loads(trozos[0].decode("utf-8"))
    cuerpo = trozos[1]

    def accesor(i: int) -> np.ndarray:
        a = cab["accessors"][i]
        bv = cab["bufferViews"][a["bufferView"]]
        o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        tipo = {5125: "<u4", 5123: "<u2", 5126: "<f4"}[a["componentType"]]
        n = {"SCALAR": 1, "VEC3": 3}[a["type"]]
        v = np.frombuffer(cuerpo, dtype=tipo, count=a["count"] * n, offset=o)
        return v.reshape(a["count"], n) if n > 1 else v

    prims = cab["meshes"][0]["primitives"]
    pos = accesor(prims[0]["attributes"]["POSITION"]).astype(np.float64)
    tri = np.concatenate([accesor(p["indices"]).astype(np.int64) for p in prims])
    return pos, tri.reshape(-1, 3)


# Los extremos se recortan al 1 % por lado antes de medir la caja. Una sola mota mal
# etiquetada a 10 mm agranda la caja entera, y entonces esto mediria la mota y no la pieza.
# ⚠️ Comprobado que aqui NO cambia la conclusion —los brutos son 1,5-3 mm mayores y las
# mismas piezas saltan—, lo que dice que lo que sobra es cuerpo y no motas.
RECORTE_PCT = 1.0


from analysis_agents.frontera import _aristas, concavidad  # noqa: E402


def razon_de_concavidad(pos: np.ndarray, tri: np.ndarray, etq: np.ndarray) -> float | None:
    """Cuanto mas concavo es el margen gingival que la malla entera. `None` si no lo hay.

    ⚠️ El contacto interproximal —diente contra diente— queda FUERA a proposito: ahi no hay
    margen gingival que medir, y meterlo mezclaria dos fronteras distintas en una cifra.
    """
    i, j = _aristas(tri)
    marca = np.zeros(len(pos), dtype=bool)
    marca[i[(etq[i] != 0) & (etq[j] == 0)]] = True
    if int(marca.sum()) < MINIMO_VERTICES_MARGEN:
        return None
    c = concavidad(pos, tri)
    return float(c[marca].mean() / max(float(np.abs(c).mean()), 1e-12))


def _diagonal_propia(puntos: np.ndarray) -> float:
    """Diagonal de la caja de la pieza EN SUS EJES, no en los del fichero.

    En los ejes del fichero un molar sale mas grande solo por estar girado respecto al
    escaner, y entonces la comparacion con la caja anatomica mediria la pose.
    """
    c = puntos - puntos.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    p = c @ vt.T
    alto = np.percentile(p, 100.0 - RECORTE_PCT, axis=0)
    bajo = np.percentile(p, RECORTE_PCT, axis=0)
    return float(np.linalg.norm(alto - bajo))


def mide(ruta: Path) -> dict:
    with zipfile.ZipFile(ruta) as z:
        nombres = set(z.namelist())
        if SEGMENTACION not in nombres:
            raise SystemExit(f"{ruta.name} no trae `{SEGMENTACION}`: no hay nada que medir.")
        etq = np.frombuffer(z.read(SEGMENTACION), dtype="<i2").astype(np.int64)
        pos, tri = lee_escena(z.read(ESCENA))

    if len(etq) != len(pos):
        raise SystemExit(
            f"`{SEGMENTACION}` indexa {len(etq):,} vertices y la escena declara "
            f"{len(pos):,}: el cruce por indice no es valido."
        )

    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    porcara = etq[tri]

    piezas = []
    for fdi in sorted({int(x) for x in np.unique(etq) if x > 0}):
        suya = (porcara == fdi).all(axis=1)
        md, vl, alt = _caja_anatomica(fdi)
        limite = float(np.linalg.norm((md, vl, alt)))
        propia = _diagonal_propia(pos[etq == fdi])
        piezas.append({
            "fdi": fdi,
            "area_mm2": round(float(area[suya].sum()), 1),
            "vertices": int((etq == fdi).sum()),
            "diagonal_mm": round(propia, 1),
            "diagonal_anatomica_mm": round(limite, 1),
            "razon": round(propia / limite, 2),
            "descartada": propia / limite > TOLERANCIA,
        })

    por_fdi = {p["fdi"]: p for p in piezas}
    espejos = []
    for p in piezas:
        gemelo = p["fdi"] + 10 if p["fdi"] // 10 in (1, 3, 5, 7) else p["fdi"] - 10
        if gemelo not in por_fdi or gemelo < p["fdi"]:
            continue
        q = por_fdi[gemelo]
        r = max(p["area_mm2"], q["area_mm2"]) / max(min(p["area_mm2"], q["area_mm2"]), 1e-9)
        espejos.append({
            "par": [p["fdi"], gemelo],
            "areas_mm2": [p["area_mm2"], q["area_mm2"]],
            "razon": round(r, 2),
            "descartado": r > TOLERANCIA_ESPEJO,
        })

    malas = {p["fdi"] for p in piezas if p["descartada"]}
    for e in espejos:
        if e["descartado"]:
            malas.update(e["par"])

    total_area = float(area.sum())
    con_fdi = float(area[(porcara > 0).all(axis=1)].sum())
    razon_frontera = razon_de_concavidad(pos, tri, etq)
    return {
        "caso": ruta.stem,
        "piezas": piezas,
        "espejos": espejos,
        "n_piezas": len(piezas),
        "n_descartadas": len(malas),
        "descartadas": sorted(malas),
        "cota_superior_correctas_pct": round(100 * (len(piezas) - len(malas)) / len(piezas), 1),
        "area_total_mm2": round(total_area, 0),
        "area_etiquetada_pct": round(100 * con_fdi / total_area, 1),
        "razon_concavidad": None if razon_frontera is None else round(razon_frontera, 2),
        "frontera_en_el_pliegue": (
            None if razon_frontera is None else razon_frontera >= RAZON_CONCAVIDAD_MINIMA
        ),
        "tolerancia": TOLERANCIA,
        "tolerancia_espejo": TOLERANCIA_ESPEJO,
        "razon_concavidad_minima": RAZON_CONCAVIDAD_MINIMA,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("caso", type=Path, help="contenedor `.uos` con `derived/seg_teeth.bin`")
    ap.add_argument("--json", type=Path, default=None, help="escribe el resultado ahi")
    args = ap.parse_args()

    r = mide(args.caso)
    print(f"segmentacion FDI · {r['n_piezas']} piezas · "
          f"{r['area_etiquetada_pct']} % del area con nombre\n")
    print(f"{'FDI':>4} {'area':>9} {'diag':>7} {'anatom':>8} {'razon':>6}")
    for p in r["piezas"]:
        marca = "  <-- se pasa" if p["descartada"] else ""
        print(f"{p['fdi']:>4} {p['area_mm2']:8.1f}² {p['diagonal_mm']:6.1f} "
              f"{p['diagonal_anatomica_mm']:7.1f} {p['razon']:6.2f}{marca}")
    print(f"\n{'par':>9} {'areas':>19} {'razon':>6}")
    for e in r["espejos"]:
        marca = "  <-- asimetrico" if e["descartado"] else ""
        print(f"{e['par'][0]:>4}/{e['par'][1]:<4} "
              f"{e['areas_mm2'][0]:8.1f} {e['areas_mm2'][1]:8.1f} {e['razon']:6.2f}{marca}")
    print(f"\ndescartadas por anatomia: {r['n_descartadas']} de {r['n_piezas']} "
          f"({', '.join(str(x) for x in r['descartadas'])})")
    print(f"cota SUPERIOR de piezas correctas: {r['cota_superior_correctas_pct']} % "
          "— sin verdad de campo no se puede decir cuantas aciertan, solo cuantas fallan.")

    rc = r["razon_concavidad"]
    if rc is None:
        print("\nfrontera diente/encia: sin margen que medir en esta escena.")
    elif r["frontera_en_el_pliegue"]:
        print(f"\nfrontera diente/encia: razon de concavidad {rc:.2f} "
              f"(experto 1,82; minimo {RAZON_CONCAVIDAD_MINIMA:.2f}) — cae en el pliegue.")
    else:
        print(f"\nfrontera diente/encia: razon de concavidad {rc:.2f} contra 1,82 de "
              "anotacion de experto. La frontera NO esta en el pliegue donde el diente sale "
              "de la encia: esta puesta donde la superficie no hace nada. Las dos pruebas de "
              "arriba son ciegas a esto — una corona puede tener la diagonal correcta con el "
              "borde corrido milimetros sobre la encia.")

    if args.json:
        args.json.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nescrito: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
