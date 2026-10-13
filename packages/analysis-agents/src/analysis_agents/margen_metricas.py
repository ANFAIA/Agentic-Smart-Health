"""Medir una frontera corona-encía contra etiquetas de experto, sin dejarse engañar.

**Por qué existe.** Las cifras con las que se decidió no aplicar `recorta_al_cuello` —0,891
→ 0,906 de acuerdo, 9/14 coronas anchas— salieron de notebooks y de ejecuciones sueltas.
Ningún script versionado puntuaba un método de frontera contra la verdad de campo, así que
cada intento nuevo se juzgaba con una regla distinta. Esto es esa regla, escrita una vez.

⚠️ **Los dos errores van SEPARADOS, y es lo que más importa de este fichero.** Una
etiqueta puede fallar por desbordarse sobre la encía (*falso diente*) o por comerse la
corona (*falsa encía*). El acuerdo global los suma, y entonces un método que recorta de más
puede subirlo mientras estropea lo que se ve: es exactamente lo que pasó con el recorte al
cuello sobre el caso clínico. Cada uno tiene aquí su tasa y su signo.

⚠️ **Ninguna cifra sola decide.** La razón de concavidad llega a la del experto con el 87 %
de las etiquetas mal; contar coronas anchas da 77 % sobre etiquetas de experto. Por eso se
informan varias y los criterios de aceptación exigen que mejoren a la vez.

Todo es numpy y scipy: se evalúa sin GPU a partir de predicciones cacheadas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis_agents.frontera import margen_gingival, razon_de_concavidad

# Por debajo de esta fracción de su corona de experto, a una pieza se la considera
# recortada de más. Es el detector de bimodalidad: una media puede mejorar mientras un
# puñado de piezas pierde media corona.
CORONA_MINIMA_CONSERVADA = 0.85


def area_por_vertice(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Área que le toca a cada vértice: un tercio de la de sus caras."""
    a, b, c = (pos[tri[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    por_vertice = np.zeros(len(pos))
    for k in range(3):
        np.add.at(por_vertice, tri[:, k], area / 3.0)
    return por_vertice


def _percentil(valores: np.ndarray, q: float) -> float | None:
    return float(np.percentile(valores, q)) if len(valores) else None


def distancia_de_margen(pos: np.ndarray, tri: np.ndarray, pred: np.ndarray,
                        experto: np.ndarray) -> dict[str, Any]:
    """Cuánto dista el margen predicho del margen del experto, en milímetros.

    Simétrica —del predicho al experto y al revés— porque una sola dirección perdona al
    método que no pone margen donde debería haber uno. El **signo** se mide aparte: un
    vértice del margen predicho que el experto llama encía es desborde (positivo); uno que
    el experto llama diente es sobre-recorte (negativo).
    """
    from scipy.spatial import cKDTree

    mp, me = margen_gingival(tri, pred), margen_gingival(tri, experto)
    if not mp.any() or not me.any():
        return {"symmetric_p50_mm": None, "symmetric_p95_mm": None,
                "signed_mean_mm": None, "n_pred": int(mp.sum()), "n_expert": int(me.sum())}
    d_pe = cKDTree(pos[me]).query(pos[mp])[0]
    d_ep = cKDTree(pos[mp]).query(pos[me])[0]
    ambas = np.concatenate([d_pe, d_ep])
    signo = np.where(experto[mp] == 0, 1.0, -1.0)
    return {
        "symmetric_p50_mm": _percentil(ambas, 50), "symmetric_p95_mm": _percentil(ambas, 95),
        "signed_mean_mm": float((d_pe * signo).mean()),
        "n_pred": int(mp.sum()), "n_expert": int(me.sum()),
    }


def por_pieza(pos: np.ndarray, tri: np.ndarray, pred: np.ndarray,
              experto: np.ndarray) -> dict[int, dict[str, Any]]:
    """Por pieza del EXPERTO: cuánta corona conserva la predicción y a qué dista su margen.

    `retained` no mira el código FDI: pregunta si el vértice sigue siendo diente. Un
    desplazamiento al vecino es otro error y lo mide el acuerdo FDI.
    """
    from scipy.spatial import cKDTree

    mp, me = margen_gingival(tri, pred), margen_gingival(tri, experto)
    arbol = cKDTree(pos[mp]) if mp.any() else None
    salida: dict[int, dict[str, Any]] = {}
    for t in (int(x) for x in np.unique(experto) if x > 0):
        suyos = experto == t
        margen = suyos & me
        dist = arbol.query(pos[margen])[0] if arbol is not None and margen.any() else None
        salida[t] = {
            "expert_vertices": int(suyos.sum()),
            "retained": float((pred[suyos] > 0).mean()),
            "fdi_agreement": float((pred[suyos] == t).mean()),
            "margin_p50_mm": None if dist is None else float(np.median(dist)),
        }
    return salida


def anchos(pos: np.ndarray, tri: np.ndarray, etq: np.ndarray) -> list[float]:
    """`|ancho medido − tabla|` por pieza, en mm. Lista vacía si no se puede orientar."""
    from export_agents.anatomia import anchos_de_corona

    return [abs(float(m) - float(t)) for m, t in anchos_de_corona(pos, tri, etq).values()]


def metricas_de_arcada(pos: np.ndarray, tri: np.ndarray, pred: np.ndarray,
                       experto: np.ndarray) -> dict[str, Any]:
    """Todas las medidas de una arcada. `pred` y `experto`: código FDI por vértice, 0 = encía."""
    pred = np.asarray(pred, dtype=np.int64)
    experto = np.asarray(experto, dtype=np.int64)
    if len(pred) != len(experto) or len(pred) != len(pos):
        raise ValueError("pred, experto y posiciones no indexan los mismos vértices")
    p, e = pred > 0, experto > 0
    union = int((p | e).sum())
    area = area_por_vertice(pos, tri)
    total = float(area.sum()) or 1.0
    piezas = por_pieza(pos, tri, pred, experto)
    conservadas = np.array([v["retained"] for v in piezas.values()])
    ancho = np.array(anchos(pos, tri, pred))
    mayoria = np.array([v["fdi_agreement"] > 0.5 for v in piezas.values()])
    return {
        "vertices": int(len(pos)),
        "binary_agreement": float((p == e).mean()),
        "binary_iou": float((p & e).sum() / union) if union else None,
        # Desborde: de lo que el experto llama encía, cuánto se predice diente.
        "false_tooth_rate": float((p & ~e).sum() / max(int((~e).sum()), 1)),
        # Sobre-recorte: de lo que el experto llama diente, cuánto se predice encía.
        "false_gum_rate": float((~p & e).sum() / max(int(e.sum()), 1)),
        "fdi_agreement": float((pred == experto).mean()),
        # ⚠️ Solo sobre vértices de DIENTE: es el `tooth_acc` del notebook de entrenamiento
        # y la cifra con la que se comprueba que la partición no tiene fuga. No dice nada
        # del margen: un modelo que lo etiqueta todo diente la tiene perfecta.
        "tooth_fdi_agreement": float((pred[e] == experto[e]).mean()) if e.any() else None,
        "margin": distancia_de_margen(pos, tri, pred, experto),
        "labelled_fraction": {
            "pred_vertices": float(p.mean()), "expert_vertices": float(e.mean()),
            "pred_area": float(area[p].sum() / total),
            "expert_area": float(area[e].sum() / total),
            "paired_diff_area": float((area[p].sum() - area[e].sum()) / total),
        },
        "concavity_ratio": {"pred": razon_de_concavidad(pos, tri, pred),
                            "expert": razon_de_concavidad(pos, tri, experto)},
        "width_error_mm": {"p50": _percentil(ancho, 50), "p90": _percentil(ancho, 90),
                           "p95": _percentil(ancho, 95), "n": int(len(ancho))},
        "teeth": {
            "expert": int(len(piezas)),
            "pred": int(len([t for t in np.unique(pred) if t > 0])),
            "majority_correct": int(mayoria.sum()),
            "under_retained": int((conservadas < CORONA_MINIMA_CONSERVADA).sum()),
            "min_retained": float(conservadas.min()) if len(conservadas) else None,
        },
        "per_tooth": {str(t): v for t, v in piezas.items()},
    }


# Qué se resume de cada arcada y con qué nombre. La ruta va por claves anidadas.
RESUMIBLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("binary_agreement", ("binary_agreement",)),
    ("binary_iou", ("binary_iou",)),
    ("false_tooth_rate", ("false_tooth_rate",)),
    ("false_gum_rate", ("false_gum_rate",)),
    ("fdi_agreement", ("fdi_agreement",)),
    ("tooth_fdi_agreement", ("tooth_fdi_agreement",)),
    ("margin_p50_mm", ("margin", "symmetric_p50_mm")),
    ("margin_p95_mm", ("margin", "symmetric_p95_mm")),
    ("margin_signed_mm", ("margin", "signed_mean_mm")),
    ("labelled_area_pred", ("labelled_fraction", "pred_area")),
    ("labelled_area_paired_diff", ("labelled_fraction", "paired_diff_area")),
    ("concavity_ratio", ("concavity_ratio", "pred")),
    ("width_error_p50_mm", ("width_error_mm", "p50")),
    ("width_error_p90_mm", ("width_error_mm", "p90")),
)


def _extrae(arcada: dict[str, Any], ruta: tuple[str, ...]) -> float | None:
    valor: Any = arcada
    for clave in ruta:
        valor = valor.get(clave) if isinstance(valor, dict) else None
        if valor is None:
            return None
    return float(valor)


def agrega(arcadas: list[dict[str, Any]]) -> dict[str, Any]:
    """Media, p50 y p90 por medida sobre las arcadas, y el recuento de piezas dañadas.

    Una medida que falta en una arcada —sin margen, sin piezas con las que orientar— no
    entra en su resumen y se cuenta en `n`: un `None` promediado como cero mentiría.
    """
    resumen: dict[str, Any] = {"arches": len(arcadas)}
    for nombre, ruta in RESUMIBLES:
        v = np.array([x for x in (_extrae(a, ruta) for a in arcadas) if x is not None])
        resumen[nombre] = ({"n": 0} if not len(v) else {
            "n": int(len(v)), "mean": float(v.mean()),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90))})
    piezas = sum(a["teeth"]["expert"] for a in arcadas)
    danadas = sum(a["teeth"]["under_retained"] for a in arcadas)
    resumen["teeth_expert"] = int(piezas)
    resumen["teeth_under_retained"] = int(danadas)
    correctas = sum(a["teeth"]["majority_correct"] for a in arcadas)
    resumen["teeth_majority_correct_fraction"] = float(correctas / piezas) if piezas else None
    resumen["teeth_under_retained_fraction"] = float(danadas / piezas) if piezas else None
    return resumen


def compara(candidato: list[dict[str, Any]], base: list[dict[str, Any]],
            nombre: str, ruta: tuple[str, ...], *, menor_es_mejor: bool) -> dict[str, Any]:
    """Diferencia PAREADA arcada a arcada de una medida entre dos métodos.

    Parear importa: la fracción etiquetada depende de cuánta encía trae cada escaneo, así
    que comparar medias de dos conjuntos mezcla el efecto del método con el del encuadre.
    """
    pares = [(c, b) for c, b in ((_extrae(x, ruta), _extrae(y, ruta))
                                 for x, y in zip(candidato, base, strict=True))
             if c is not None and b is not None]
    if not pares:
        return {"measure": nombre, "n": 0}
    d = np.array([c - b for c, b in pares])
    mejora = d < 0 if menor_es_mejor else d > 0
    return {"measure": nombre, "n": int(len(d)), "mean_diff": float(d.mean()),
            "median_diff": float(np.median(d)), "improved_fraction": float(mejora.mean()),
            "worst_diff": float(d.max() if menor_es_mejor else d.min())}
