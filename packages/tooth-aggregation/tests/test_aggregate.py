"""Agregación punto → diente: instancias, fusión de fragmentos y unicidad FDI."""

from __future__ import annotations

import numpy as np
import pytest
from tooth_aggregation import (
    ToothInstance,
    aggregate_teeth,
    assign_unique,
    connected_labels,
    majority_label,
    merge_fragments,
    typical_spacing,
)

STEP = 0.01


def cubo(centro: tuple[float, float, float], lado: int = 4, step: float = STEP) -> np.ndarray:
    """Rejilla regular de `lado³` puntos centrada en `centro`.

    Rejilla y no nube aleatoria: así el espaciado es exactamente `step` y los
    umbrales de fusión se pueden razonar a mano en los tests.
    """
    g = np.arange(lado) * step
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return pts - pts.mean(axis=0) + np.asarray(centro, dtype=float)


def escena(*bloques: tuple[np.ndarray, int]) -> tuple[np.ndarray, np.ndarray]:
    """Concatena bloques `(puntos, etiqueta)` en (points, labels)."""
    pts = np.concatenate([b for b, _ in bloques])
    lab = np.concatenate([np.full(len(b), c) for b, c in bloques])
    return pts, lab


def logprob_de(labels: np.ndarray, n_clases: int, alto: float = -0.1) -> np.ndarray:
    """Log-probs coherentes con `labels`: la clase correcta domina."""
    lp = np.full((len(labels), n_clases), -9.0)
    lp[np.arange(len(labels)), labels] = alto
    return lp


# --- espaciado -------------------------------------------------------------- #
def test_typical_spacing_recupera_el_paso_de_la_rejilla() -> None:
    assert typical_spacing(cubo((0, 0, 0)), k=8) == pytest.approx(STEP, rel=0.35)


def test_typical_spacing_de_nube_degenerada_es_cero() -> None:
    assert typical_spacing(np.zeros((1, 3))) == 0.0


# --- instancias ------------------------------------------------------------- #
def test_dos_dientes_separados_dan_dos_instancias() -> None:
    pts, lab = escena((cubo((0, 0, 0)), 1), (cubo((1, 0, 0)), 2))
    assert len(connected_labels(pts, lab, min_size=30)) == 2


def test_mismo_codigo_en_dos_sitios_son_instancias_distintas() -> None:
    # Dos manchas lejanas con la MISMA etiqueta: la conectividad las separa.
    pts, lab = escena((cubo((0, 0, 0)), 1), (cubo((1, 0, 0)), 1))
    assert len(connected_labels(pts, lab, min_size=30)) == 2


def test_la_encia_no_genera_instancias() -> None:
    pts, lab = escena((cubo((0, 0, 0)), 0), (cubo((1, 0, 0)), 1))
    comps = connected_labels(pts, lab, min_size=30, gum_class=0)
    assert len(comps) == 1
    assert set(lab[comps[0]]) == {1}


def test_componentes_pequenas_se_descartan_por_ruido() -> None:
    pts, lab = escena((cubo((0, 0, 0)), 1), (cubo((1, 0, 0), lado=2), 2))  # 64 vs 8
    comps = connected_labels(pts, lab, min_size=30)
    assert len(comps) == 1


def test_sin_puntos_de_diente_no_hay_instancias() -> None:
    pts, lab = escena((cubo((0, 0, 0)), 0))
    assert connected_labels(pts, lab) == []


# --- fusión ----------------------------------------------------------------- #
def _fusiona(sep: float, etiquetas: tuple[int, int], mult: float = 12.0) -> int:
    """Nº de instancias tras fusionar dos bloques separados `sep`."""
    pts, lab = escena((cubo((0, 0, 0)), etiquetas[0]), (cubo((sep, 0, 0)), etiquetas[1]))
    comps = connected_labels(pts, lab, min_size=30)
    return len(merge_fragments(pts, comps, lab, spacing=STEP, mult=mult))


def test_fusiona_fragmentos_pegados_de_la_misma_clase() -> None:
    # Separación 0.05 = 5x el espaciado, por debajo del umbral de 12x.
    assert _fusiona(0.05, (1, 1)) == 1


def test_no_fusiona_manchas_lejanas_aunque_compartan_clase() -> None:
    assert _fusiona(1.0, (1, 1)) == 2


def test_no_fusiona_dientes_vecinos_de_clases_distintas() -> None:
    # Pegados en el espacio, pero de distinta clase: no deben unirse nunca.
    assert _fusiona(0.05, (1, 2)) == 2


def test_mult_cero_desactiva_la_fusion() -> None:
    assert _fusiona(0.05, (1, 1), mult=0.0) == 2


def test_la_fusion_es_transitiva() -> None:
    # A-B pegados y B-C pegados, pero A-C lejos: los tres en una instancia.
    pts, lab = escena((cubo((0, 0, 0)), 1), (cubo((0.05, 0, 0)), 1), (cubo((0.10, 0, 0)), 1))
    comps = connected_labels(pts, lab, min_size=30)
    assert len(comps) == 3
    assert len(merge_fragments(pts, comps, lab, spacing=STEP, mult=6.0)) == 1


# --- unicidad (húngara) ----------------------------------------------------- #
def test_assign_unique_no_repite_etiqueta() -> None:
    lp = np.full((8, 3), -9.0)
    lp[:, 1] = -0.1  # las dos instancias prefieren la clase 1
    inst = [ToothInstance(1, np.arange(4), -0.1), ToothInstance(1, np.arange(4, 8), -0.1)]
    assert {i.label for i in assign_unique(inst, lp)} == {1, 2}


def test_assign_unique_es_optimo_global_no_voraz() -> None:
    """El caso donde el voraz falla: A debe CEDER su favorita para que gane el total.

    A: clase1=-0.1, clase2=-0.2   ·   B: clase1=-0.2, clase2=-5.0
    Voraz  -> A se queda clase1 (la celda mejor), B carga con clase2: total -5.1
    Óptimo -> A cede a clase2, B toma clase1:                        total -0.4
    """
    lp = np.full((8, 3), -9.0)
    a, b = np.arange(4), np.arange(4, 8)
    lp[np.ix_(a, [1, 2])] = [-0.1, -0.2]
    lp[np.ix_(b, [1, 2])] = [-0.2, -5.0]
    out = assign_unique([ToothInstance(1, a, -0.1), ToothInstance(1, b, -0.2)], lp)
    assert (out[0].label, out[1].label) == (2, 1)


def test_assign_unique_nunca_asigna_la_encia() -> None:
    lp = np.full((4, 3), -9.0)
    lp[:, 0] = 0.0  # la encía es con diferencia la más probable...
    out = assign_unique([ToothInstance(1, np.arange(4), -9.0)], lp, gum_class=0)
    assert out[0].label != 0  # ...y aun así no se asigna


def test_assign_unique_sin_instancias() -> None:
    assert assign_unique([], np.zeros((4, 3))) == []


# --- pipeline completo ------------------------------------------------------- #
def test_aggregate_teeth_devuelve_dientes_con_codigo_fdi() -> None:
    pts, lab = escena((cubo((0, 0, 0)), 0), (cubo((1, 0, 0)), 1), (cubo((2, 0, 0)), 2))
    out = aggregate_teeth(pts, logprob_de(lab, 3), codes={1: 11, 2: 12}, min_size=30)
    assert len(out) == 2
    assert {i.fdi for i in out} == {11, 12}
    assert all(i.size == 64 for i in out)
    assert all(i.confidence == pytest.approx(-0.1) for i in out)


def test_aggregate_teeth_ordena_por_tamano() -> None:
    pts, lab = escena((cubo((0, 0, 0), lado=4), 1), (cubo((1, 0, 0), lado=5), 2))
    out = aggregate_teeth(pts, logprob_de(lab, 3), min_size=30)
    assert [i.size for i in out] == [125, 64]


def test_aggregate_teeth_fusiona_antes_de_imponer_unicidad() -> None:
    """Un diente partido en dos trozos pegados no debe consumir dos códigos."""
    pts, lab = escena((cubo((0, 0, 0)), 1), (cubo((0.05, 0, 0)), 1))
    out = aggregate_teeth(pts, logprob_de(lab, 3), min_size=30, enforce_unique=True)
    assert len(out) == 1
    assert out[0].label == 1


def test_aggregate_teeth_sin_dientes_devuelve_lista_vacia() -> None:
    pts, lab = escena((cubo((0, 0, 0)), 0))
    assert aggregate_teeth(pts, logprob_de(lab, 3)) == []


@pytest.mark.parametrize(
    ("pts", "lp"),
    [
        (np.zeros((10, 2)), np.zeros((10, 3))),  # points no es (N,3)
        (np.zeros((10, 3)), np.zeros((9, 3))),  # logprob descuadra con points
        (np.zeros((10, 3)), np.zeros(10)),  # logprob no es 2D
    ],
)
def test_aggregate_teeth_rechaza_formas_invalidas(pts: np.ndarray, lp: np.ndarray) -> None:
    with pytest.raises(ValueError):
        aggregate_teeth(pts, lp)


def test_majority_label_es_determinista_en_empate() -> None:
    assert majority_label(np.array([2, 2, 5, 5])) == 2
