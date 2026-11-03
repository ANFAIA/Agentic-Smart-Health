"""Puntuar el margen corona-encía de UN caso contra líneas que marcó una persona.

**Por qué existe.** El banco de `margen_metricas` puntúa contra Teeth3DS+, que trae la
etiqueta de experto por vértice. Un caso propio no la trae, así que hasta ahora solo se
podía falsar: decir que algo parece mal, nunca que está bien. Lo más barato que una persona
puede aportar es **una línea cerrada por pieza**, trazada sobre el escaneo en 3D Slicer.
Este módulo lee esas líneas y las convierte en lo que el banco ya sabe medir.

**Dos medidas, y por qué no una.** La distancia de nuestro margen a la línea es la cifra
directa: no depende de nada más que de la línea. La etiqueta por vértice que se deriva de
ella —lo que queda dentro del lazo es esa pieza— permite reutilizar las medidas del banco,
en particular cuánta corona conserva cada pieza, pero pasa por un lazo de vértices y hereda
su resolución: una arista de la malla.

⚠️ **La línea se sigue como polilínea entre puntos de control.** Slicer dibuja una curva
suave; entre dos puntos a 2 mm sobre una corona, la cuerda se separa de ella del orden de
una décima. Por eso se declara la separación máxima entre puntos de cada línea.

⚠️ **El contacto entre dientes no es margen gingival.** La línea de una pieza pasa por el
punto de contacto con la vecina, donde nuestras etiquetas no tienen encía. Esos tramos se
reconocen por la propia anotación —están pegados a la línea de otra pieza— y no cuentan en
la distancia de la línea a nuestro margen.

⚠️ **Los sistemas de coordenadas, otra vez.** El `.mrk.json` declara LPS o RAS; el STL no
declara nada y quien llama dice cómo se cargó. Si las líneas no caen sobre la malla, se
rechaza: no se prueba el convenio contrario en silencio.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from analysis_agents.frontera import _aristas, margen_gingival
from analysis_agents.margen_metricas import area_por_vertice

# Paso con el que se muestrea la polilínea. Muy por debajo de la arista mediana de un
# escaneo intraoral (~0,2 mm), para que el muestreo no sea lo que limita la medida.
PASO_MUESTREO_MM = 0.1
# Un punto de control a más de esto de la malla no se marcó sobre ella: convenio de
# coordenadas equivocado u otro fichero. Holgado frente a la arista (p95 ~0,45 mm).
DISTANCIA_MAXIMA_A_MALLA_MM = 1.0
# Un tramo de línea a menos de esto de la línea de otra pieza es contacto interproximal.
# Es un convenio, no una medida: se declara en el resultado.
CONTACTO_MM = 0.5
# Con menos puntos no hay lazo que cerrar.
MINIMO_PUNTOS_DE_CONTROL = 4

_INVIERTE_XY = np.array([-1.0, -1.0, 1.0])
_TIPOS_CERRADOS = ("ClosedCurve",)
_FDI = re.compile(r"(?<!\d)([1-4][1-8])(?!\d)")


def lee_curva(ruta: Path) -> np.ndarray:
    """Puntos de control de una curva cerrada de Slicer, **en LPS** y en su orden."""
    ruta = Path(ruta)
    marcas = json.loads(ruta.read_text(encoding="utf-8")).get("markups") or []
    if len(marcas) != 1:
        raise ValueError(f"{ruta.name}: se esperaba una curva, hay {len(marcas)}")
    tipo = marcas[0].get("type")
    if tipo not in _TIPOS_CERRADOS:
        raise ValueError(f"{ruta.name}: es `{tipo}` y hace falta una curva cerrada "
                         "(Closed Curve)")
    sistema = str(marcas[0].get("coordinateSystem") or "").strip().upper()
    if sistema not in ("LPS", "RAS"):
        raise ValueError(f"{ruta.name} no declara su sistema de coordenadas")
    puntos = np.array([[float(v) for v in p["position"]]
                       for p in marcas[0].get("controlPoints", [])
                       if p.get("positionStatus", "defined") == "defined"], dtype=np.float64)
    if len(puntos) < MINIMO_PUNTOS_DE_CONTROL:
        raise ValueError(f"{ruta.name}: {len(puntos)} puntos de control no cierran un lazo")
    return puntos * _INVIERTE_XY if sistema == "RAS" else puntos


def fdi_de_fichero(ruta: Path) -> int:
    """El código FDI que nombra el fichero: `16.mrk.json`, `diente_16.mrk.json`.

    Si el nombre no lo trae o trae dos, no se adivina: Slicer no guarda el nombre de la
    curva dentro del fichero, así que el nombre es lo único que dice de qué pieza es.
    """
    nombre = Path(ruta).name.split(".")[0]
    hallados = sorted({int(x) for x in _FDI.findall(nombre)})
    if len(hallados) != 1:
        raise ValueError(f"{Path(ruta).name}: el nombre debe llevar un único código FDI")
    return hallados[0]


def lee_anotacion(carpeta: Path, stl_cargado_como: str) -> dict[int, np.ndarray]:
    """Las curvas de una carpeta, por FDI, en las coordenadas crudas del STL."""
    modo = stl_cargado_como.strip().upper()
    if modo not in ("LPS", "RAS"):
        raise ValueError(f"Sistema de coordenadas desconocido: {stl_cargado_como!r}")
    curvas: dict[int, np.ndarray] = {}
    for ruta in sorted(Path(carpeta).glob("*.mrk.json")):
        fdi = fdi_de_fichero(ruta)
        if fdi in curvas:
            raise ValueError(f"{carpeta}: dos ficheros para la pieza {fdi}")
        puntos = lee_curva(ruta)
        curvas[fdi] = puntos if modo == "LPS" else puntos * _INVIERTE_XY
    if not curvas:
        raise ValueError(f"{carpeta}: no hay ningún `.mrk.json`")
    return curvas


def muestrea(control: np.ndarray, paso: float = PASO_MUESTREO_MM) -> np.ndarray:
    """La polilínea CERRADA por los puntos de control, muestreada cada `paso` mm."""
    cerrada = np.vstack([control, control[:1]])
    trozos = []
    for a, b in zip(cerrada[:-1], cerrada[1:], strict=True):
        n = max(int(np.ceil(np.linalg.norm(b - a) / paso)), 1)
        trozos.append(a + (b - a) * (np.arange(n) / n)[:, None])
    return np.concatenate(trozos)


def comprueba_sobre_malla(pos: np.ndarray, curvas: dict[int, np.ndarray]) -> dict[str, float]:
    """Que las líneas se marcaron sobre ESTA malla; si no, error con el motivo probable."""
    from scipy.spatial import cKDTree

    arbol = cKDTree(pos)
    control = np.concatenate(list(curvas.values()))
    d = arbol.query(control)[0]
    if float(np.median(d)) > DISTANCIA_MAXIMA_A_MALLA_MM:
        volteada = float(np.median(arbol.query(control * _INVIERTE_XY)[0]))
        pista = ("Con el convenio contrario (LPS/RAS) sí caen sobre ella: revisa cómo se "
                 "cargó el STL." if volteada <= DISTANCIA_MAXIMA_A_MALLA_MM
                 else "Tampoco con el convenio contrario: ¿es otro escaneo?")
        raise ValueError(f"Las líneas quedan a {float(np.median(d)):.1f} mm de la malla "
                         f"(mediana). {pista}")
    return {"control_to_mesh_p50_mm": float(np.median(d)),
            "control_to_mesh_max_mm": float(d.max())}


def _grafo(pos: np.ndarray, tri: np.ndarray):
    from scipy.sparse import csr_matrix

    i, j = _aristas(tri)
    largo = np.linalg.norm(pos[i] - pos[j], axis=1)
    return csr_matrix((largo, (i, j)), shape=(len(pos), len(pos)))


def _camino(grafo: Any, a: int, b: int, limite: float) -> list[int] | None:
    """Vértices intermedios del camino más corto de `a` a `b`; `None` si no se alcanza."""
    from scipy.sparse.csgraph import dijkstra

    _, previo = dijkstra(grafo, indices=a, return_predecessors=True, limit=limite)
    if previo[b] < 0:
        return None
    paso, camino = int(previo[b]), []
    while paso != a:
        camino.append(paso)
        paso = int(previo[paso])
    return camino


def lazo_de_vertices(pos: np.ndarray, grafo: Any, control: np.ndarray) -> np.ndarray:
    """Los vértices que sigue la línea, como camino cerrado por aristas de la malla.

    Cada muestra se lleva a su vértice más cercano; donde dos consecutivos no son vecinos
    se unen por el camino más corto. Un camino de vértices unidos por aristas separa la
    superficie: ninguna arista lo cruza sin tocar uno de ellos.
    """
    from scipy.spatial import cKDTree

    cerca = cKDTree(pos).query(muestrea(control))[1]
    orden = cerca[np.concatenate([[True], cerca[1:] != cerca[:-1]])]
    lazo: list[int] = []
    for a, b in zip(orden, np.roll(orden, -1), strict=True):
        lazo.append(int(a))
        if a == b or grafo[a, b] != 0:
            continue
        # Primero cerca: casi todos los saltos son de un par de aristas. Solo si no llega
        # —la cuerda cruzó de una cara a otra de la corona— se busca en toda la malla.
        salto = float(np.linalg.norm(pos[a] - pos[b]))
        camino: list[int] | None = None
        for limite in (4.0 * salto + 1.0, np.inf):
            camino = _camino(grafo, int(a), int(b), limite)
            if camino is not None:
                break
        if camino is None:
            raise ValueError("la línea salta entre trozos de malla que no se tocan")
        lazo.extend(camino)
    return np.unique(np.asarray(lazo, dtype=np.int64))


def interior_del_lazo(grafo: Any, lazo: np.ndarray) -> np.ndarray | None:
    """Máscara de lo que UN lazo encierra por sí solo, lazo incluido; `None` si no encierra nada.

    Quitado el lazo, lo de fuera es la componente mayor: el resto de la arcada. Dentro
    queda toda otra componente **que toque el lazo**; un fragmento suelto del escaneo en la
    otra punta de la malla no lo toca y no cuenta.
    """
    regiones = _regiones_entre_lazos(grafo, {0: lazo})
    return None if regiones[0] is None else regiones[0]


def _regiones_entre_lazos(grafo: Any,
                          lazos: dict[int, np.ndarray]) -> dict[int, np.ndarray | None]:
    """Lo que encierra cada lazo, quitando TODOS a la vez.

    ⚠️ **Por qué todos a la vez y no uno a uno.** Donde dos coronas se tocan, el escáner
    las une en una sola superficie por encima del punto de contacto. Una línea que pasa por
    el fondo del surco interproximal, por debajo de ese puente, deja su corona conectada con
    la vecina, y la vecina con la encía: medido sobre un caso real, dos de catorce líneas no
    encerraban nada por separado. Quitando los lazos de las dos, las dos coronas quedan
    juntas en una región cerrada, y esa región se reparte entre ellas por cercanía
    geodésica a cada lazo. Donde no hay puente, cada región toca un solo lazo y el reparto
    no hace nada.
    """
    from scipy.sparse.csgraph import connected_components, dijkstra

    n = grafo.shape[0]
    barrera = np.zeros(n, dtype=bool)
    for lazo in lazos.values():
        barrera[lazo] = True
    quedan = np.flatnonzero(~barrera)
    _, comp_local = connected_components(grafo[quedan][:, quedan], directed=False)
    comp = np.full(n, -1, dtype=np.int64)
    comp[quedan] = comp_local
    tamano = np.bincount(comp_local)
    # Qué componentes toca cada lazo, y lo de fuera: la mayor de todas las que tocan alguno.
    toca = {f: {int(c) for c in np.unique(comp[grafo[lazo].indices]) if c >= 0}
            for f, lazo in lazos.items()}
    todas = set().union(*toca.values()) if toca else set()
    if not todas:
        return dict.fromkeys(lazos)
    fuera = max(todas, key=lambda c: tamano[c])
    dentro: dict[int, np.ndarray] = {f: np.zeros(n, dtype=bool) for f in lazos}
    for c in todas - {fuera}:
        suyos = [f for f in lazos if c in toca[f]]
        miembros = np.flatnonzero(comp == c)
        if len(suyos) == 1:
            dentro[suyos[0]][miembros] = True
            continue
        # Varios lazos tocan la misma región: cada vértice va al lazo más cercano por la
        # superficie, sin cruzar otros lazos ni salir de la región.
        sub = np.concatenate([miembros] + [lazos[f] for f in suyos])
        local = grafo[sub][:, sub]
        distancias = []
        for f in suyos:
            fuentes = np.flatnonzero(np.isin(sub, lazos[f]))
            distancias.append(dijkstra(local, indices=fuentes, min_only=True)[:len(miembros)])
        gana = np.argmin(np.vstack(distancias), axis=0)
        for k, f in enumerate(suyos):
            dentro[f][miembros[gana == k]] = True
    salida: dict[int, np.ndarray | None] = {}
    for f, lazo in lazos.items():
        if dentro[f].any():
            dentro[f][lazo] = True
            salida[f] = dentro[f]
        else:
            salida[f] = None
    return salida


def etiquetas_de_curvas(pos: np.ndarray, tri: np.ndarray,
                        curvas: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Etiqueta FDI por vértice a partir de las líneas, y qué pasó con cada una.

    Una línea que no encierra nada —abierta sobre el borde del escaneo, o que salta entre
    trozos de malla— se declara y su pieza queda sin etiquetar: no se rellena a ojo. Una
    región que encierran entre varias líneas se reparte entre ellas y se declara en
    `shared_with`. Un vértice que reclaman dos piezas va a la de centro más cercano y se
    cuenta.
    """
    grafo = _grafo(pos, tri)
    area = area_por_vertice(pos, tri)
    lazos: dict[int, np.ndarray] = {}
    registros: dict[str, Any] = {}
    for fdi, control in sorted(curvas.items()):
        cerrada = np.vstack([control, control[:1]])
        registros[str(fdi)] = {
            "control_points": int(len(control)),
            "max_control_gap_mm": float(np.linalg.norm(np.diff(cerrada, axis=0), axis=1).max()),
            "usable": False, "reason": None, "shared_with": [],
        }
        try:
            lazos[fdi] = lazo_de_vertices(pos, grafo, control)
        except ValueError as e:
            registros[str(fdi)]["reason"] = str(e)
    regiones = _regiones_entre_lazos(grafo, lazos) if lazos else {}
    solas = {f: interior_del_lazo(grafo, lazo) for f, lazo in lazos.items()}
    dentro: dict[int, np.ndarray] = {}
    for fdi, mascara in regiones.items():
        registro = registros[str(fdi)]
        if mascara is None:
            registro["reason"] = "la línea no encierra ninguna región"
            continue
        dentro[fdi] = mascara
        registro.update(usable=True, vertices=int(mascara.sum()),
                        area_mm2=float(area[mascara].sum()))
        if solas[fdi] is None:
            # Sola no cerraba: su región es la que comparte con las vecinas.
            registro["shared_with"] = sorted(
                int(f) for f, m in solas.items()
                if f != fdi and m is None and fdi in regiones and regiones[f] is not None)

    etq = np.zeros(len(pos), dtype=np.int64)
    veces = np.zeros(len(pos), dtype=np.int64)
    for mascara in dentro.values():
        veces += mascara
    disputados = veces > 1
    for fdi, mascara in dentro.items():
        etq[mascara & ~disputados] = fdi
    if disputados.any():
        piezas = sorted(dentro)
        centros = np.array([curvas[f].mean(axis=0) for f in piezas])
        for v in np.flatnonzero(disputados):
            suyas = [k for k, f in enumerate(piezas) if dentro[f][v]]
            lejos = np.linalg.norm(centros[suyas] - pos[v], axis=1)
            etq[v] = piezas[suyas[int(np.argmin(lejos))]]
    return etq, {"per_tooth": registros, "disputed_vertices": int(disputados.sum())}


def _resumen(d: np.ndarray) -> dict[str, Any]:
    if not len(d):
        return {"n": 0, "p50_mm": None, "p95_mm": None, "max_mm": None}
    return {"n": int(len(d)), "p50_mm": float(np.percentile(d, 50)),
            "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max())}


def _gingival(curvas: dict[int, np.ndarray], fdi: int) -> np.ndarray:
    """Las muestras de la línea de `fdi` que NO están pegadas a la línea de otra pieza."""
    from scipy.spatial import cKDTree

    propias = muestrea(curvas[fdi])
    otras = [muestrea(c) for f, c in curvas.items() if f != fdi]
    if not otras:
        return propias
    return propias[cKDTree(np.concatenate(otras)).query(propias)[0] > CONTACTO_MM]


def empareja(pred: np.ndarray, anotado: np.ndarray,
             curvas: dict[int, np.ndarray]) -> dict[int, dict[str, Any]]:
    """Qué pieza de `pred` corresponde a cada línea, por SOLAPE y no por número.

    ⚠️ Quien anota numera según su criterio y el programa según el suyo. Si discrepan —un
    molar que uno llama 16 y el otro 15—, emparejar por código mide la distancia entre dos
    dientes distintos: sobre un caso real, cuatro de catorce líneas caían sobre un vecino de
    otro número. Aquí cada línea se empareja con la pieza que más vértices tiene dentro de
    su región, y la discrepancia de número se declara aparte. Sin región, se cae al número.
    """
    pred = np.asarray(pred, dtype=np.int64)
    en_pred = {int(x) for x in np.unique(pred) if x > 0}
    parejas: dict[int, dict[str, Any]] = {}
    for fdi in sorted(curvas):
        dentro = pred[(anotado == fdi) & (pred > 0)]
        if len(dentro):
            codigos, cuenta = np.unique(dentro, return_counts=True)
            parejas[fdi] = {"pred": int(codigos[np.argmax(cuenta)]), "by": "overlap"}
        else:
            parejas[fdi] = {"pred": fdi if fdi in en_pred else None, "by": "number"}
    return parejas


def distancia_a_las_curvas(pos: np.ndarray, tri: np.ndarray, pred: np.ndarray,
                           curvas: dict[int, np.ndarray], anotado: np.ndarray,
                           con_region: set[int] | None = None) -> dict[str, Any]:
    """Distancia, por pieza, entre el margen de `pred` y la línea de esa misma pieza.

    Se emparejan con `empareja`, por solape. `pred_to_curve` va de nuestros vértices de margen a la
    línea; `curve_to_pred`, de la línea —sin sus tramos de contacto— a nuestro margen. El
    signo usa la región anotada: positivo si nuestro margen cae sobre lo que la persona
    dejó fuera de toda corona (desborde), negativo si cae dentro (sobre-recorte). Una pieza
    cuya línea no dio región (`con_region`) se mide igual, pero sin signo.
    """
    from scipy.spatial import cKDTree

    pred = np.asarray(pred, dtype=np.int64)
    margen = margen_gingival(tri, pred)
    parejas = empareja(pred, anotado, curvas)
    por_pieza: dict[str, Any] = {}
    todas_pc, todas_cp = [], []
    for fdi in sorted(curvas):
        suyo = margen & (pred == parejas[fdi]["pred"])
        if parejas[fdi]["pred"] is None or not suyo.any():
            continue
        d_pc = cKDTree(muestrea(curvas[fdi])).query(pos[suyo])[0]
        gingival = _gingival(curvas, fdi)
        d_cp = (cKDTree(pos[suyo]).query(gingival)[0] if len(gingival)
                else np.zeros(0))
        signo = np.where(anotado[suyo] == 0, 1.0, -1.0)
        medible = con_region is None or fdi in con_region
        por_pieza[str(fdi)] = {
            "pred_to_curve": _resumen(d_pc), "curve_to_pred": _resumen(d_cp),
            "signed_mean_mm": float((d_pc * signo).mean()) if medible else None,
            "paired_with": parejas[fdi]["pred"], "paired_by": parejas[fdi]["by"]}
        todas_pc.append(d_pc)
        todas_cp.append(d_cp)
    en_pred = {int(x) for x in np.unique(pred) if x > 0}
    emparejadas = {v["pred"] for v in parejas.values() if v["pred"] is not None}
    return {
        "per_tooth": por_pieza,
        "renumbered": {str(f): v["pred"] for f, v in parejas.items()
                       if v["pred"] is not None and v["pred"] != f},
        "pred_to_curve": _resumen(np.concatenate(todas_pc) if todas_pc else np.zeros(0)),
        "curve_to_pred": _resumen(np.concatenate(todas_cp) if todas_cp else np.zeros(0)),
        "only_annotated": sorted(f for f, v in parejas.items() if v["pred"] is None),
        "only_predicted": sorted(en_pred - emparejadas),
    }


def acuerdo_entre_anotaciones(a: dict[int, np.ndarray],
                              b: dict[int, np.ndarray]) -> dict[str, Any]:
    """Distancia simétrica entre las líneas de dos anotaciones, por pieza y en conjunto.

    Es el suelo de la medida: un margen nuestro que dista de la línea menos de lo que dos
    personas —o la misma en dos días— distan entre sí no se puede distinguir de correcto.
    """
    from scipy.spatial import cKDTree

    por_pieza: dict[str, Any] = {}
    todas = []
    for fdi in sorted(set(a) & set(b)):
        ma, mb = muestrea(a[fdi]), muestrea(b[fdi])
        d = np.concatenate([cKDTree(mb).query(ma)[0], cKDTree(ma).query(mb)[0]])
        por_pieza[str(fdi)] = _resumen(d)
        todas.append(d)
    return {"per_tooth": por_pieza,
            "overall": _resumen(np.concatenate(todas) if todas else np.zeros(0)),
            "only_first": sorted(set(a) - set(b)), "only_second": sorted(set(b) - set(a))}
