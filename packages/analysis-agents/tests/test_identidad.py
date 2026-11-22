"""La etapa de identidad por pieza (§6.11): verificador, orden y renumeración."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from analysis_agents.identidad import (
    asigna_posiciones,
    componente_principal,
    distancia_con_signo,
    instancias,
    orden_desde_linea_media,
    p1_por_instancia,
    recorte,
    renumera,
    verifica,
)


def _fuerza_bruta(S: np.ndarray) -> float:
    m = len(S)
    return max(sum(S[i, p - 1] for i, p in enumerate(c))
               for c in itertools.combinations(range(1, 9), m))


@pytest.mark.parametrize("m", [1, 2, 5, 8])
def test_la_asignacion_es_la_optima_y_estrictamente_creciente(m: int) -> None:
    rng = np.random.default_rng(m)
    for _ in range(20):
        S = rng.normal(size=(m, 8))
        p = asigna_posiciones(S)
        assert p is not None and len(p) == m
        assert all(a < b for a, b in itertools.pairwise(p))
        assert sum(S[i, q - 1] for i, q in enumerate(p)) == pytest.approx(_fuerza_bruta(S))


def test_mas_de_ocho_instancias_no_se_asignan() -> None:
    assert asigna_posiciones(np.zeros((9, 8))) is None
    assert asigna_posiciones(np.zeros((0, 8))) == []


def _arco(n: int = 7, lado: float = 1.0) -> np.ndarray:
    """Centroides de una hemiarcada en U, de la línea media hacia atrás."""
    t = np.linspace(0.15, 1.4, n)
    return np.stack([lado * 25 * np.sin(t), 25 * np.cos(t), np.zeros(n)], axis=1)


def test_el_orden_va_de_la_linea_media_hacia_atras_en_cada_lado() -> None:
    R = np.eye(3)  # filas: lateral = x, anterior = y, oclusal = z
    for lado in (1.0, -1.0):
        c = _arco(lado=lado)
        barajado = np.random.default_rng(0).permutation(len(c))
        orden = orden_desde_linea_media(c[barajado], np.zeros(3), R)
        assert list(barajado[orden]) == list(range(len(c)))


def test_el_sentido_lo_da_el_cuadrante_contrario_y_no_el_signo_del_eje() -> None:
    """El fallo de `dev-privado`: con el eje anterior al revés, el orden salía invertido."""
    al_reves = np.diag([1.0, -1.0, 1.0])
    derecha, izquierda = _arco(lado=1.0), _arco(lado=-1.0)
    barajado = np.random.default_rng(1).permutation(len(derecha))
    sin_contrario = orden_desde_linea_media(derecha[barajado], np.zeros(3), al_reves)
    assert list(barajado[sin_contrario]) == list(range(len(derecha)))[::-1]
    con_contrario = orden_desde_linea_media(derecha[barajado], np.zeros(3), al_reves,
                                            izquierda)
    assert list(barajado[con_contrario]) == list(range(len(derecha)))


def test_el_verificador_solo_quita_lo_que_baja_del_umbral() -> None:
    e = np.array([0, 11, 11, 12, 18, 18])
    nuevas, quitadas = verifica(e, {11: 0.9, 12: 0.5, 18: 0.1}, tau=0.5)
    assert quitadas == [18]
    assert list(nuevas) == [0, 11, 11, 12, 0, 0]
    assert list(e) == [0, 11, 11, 12, 18, 18]  # no muta la entrada


def _cuadrante_corrido() -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Seis piezas reales 1…6 que C2 llama 12…17: corrido un puesto, falta la 11."""
    c = _arco(6)
    pos = np.repeat(c, 3, axis=0)
    etiquetas = np.repeat(np.arange(12, 18), 3)
    p2 = {}
    for k, codigo in enumerate(range(12, 18)):
        lp = np.full(8, np.log(0.02))
        lp[k] = np.log(0.86)  # el modelo de pieza ve la posición real, k + 1
        p2[codigo] = lp
    return pos, etiquetas, p2


def test_la_renumeracion_endereza_una_hemiarcada_corrida() -> None:
    pos, etiquetas, p2 = _cuadrante_corrido()
    p1 = {c: np.zeros(8) for c in p2}
    nuevas, reg = renumera(etiquetas, pos, np.zeros(3), np.eye(3), p2, p1, w=0.0)
    assert list(np.unique(nuevas)) == [11, 12, 13, 14, 15, 16]
    assert reg["renumbered"]["12"] == 11 and reg["quadrants_untouched"] == []


def test_con_peso_alto_manda_c2() -> None:
    pos, etiquetas, p2 = _cuadrante_corrido()
    # C2 muy seguro de su numeración corrida.
    p1 = {c: np.where(np.arange(1, 9) == c % 10, 0.0, -50.0) for c in p2}
    nuevas, reg = renumera(etiquetas, pos, np.zeros(3), np.eye(3), p2, p1, w=1.0)
    assert list(nuevas) == list(etiquetas) and reg["renumbered"] == {}


def test_p1_normaliza_entre_las_ocho_posiciones_del_cuadrante() -> None:
    codigos = [0, *range(11, 19), *range(21, 29)]
    logp = np.log(np.full((4, len(codigos)), 1 / len(codigos)))
    logp[:, codigos.index(13)] = np.log(0.5)
    p1 = p1_por_instancia([logp, logp], codigos, np.array([0, 13, 13, 0]))
    assert set(p1) == {13}
    assert np.logaddexp.reduce(p1[13]) == pytest.approx(0.0)
    assert int(np.argmax(p1[13])) == 2


def test_la_distancia_con_signo_separa_dentro_y_fuera() -> None:
    x = np.linspace(0, 10, 101)
    pos = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    dentro = np.flatnonzero(x <= 5)
    idx, d = distancia_con_signo(pos, dentro, radio=2.0)
    assert set(idx[d <= 0]) == set(dentro)
    assert d[idx == 65][0] == pytest.approx(1.5, abs=1e-9)
    assert d[idx == 30][0] == pytest.approx(-2.0, abs=0.11)  # a 2 mm del borde, dentro
    assert not (idx == 90).any()  # a 4 mm: fuera del radio
    rng = np.random.default_rng(0)
    nrm = np.tile([0.0, 0.0, 1.0], (len(pos), 1))
    P, X = recorte(pos, nrm, idx, d, rng, desplaza=0.0, margen=1.0)
    assert X[:, 3].sum() == len(dentro)  # la bandera marca la pieza
    assert len(P) == len(dentro) + 10  # y el margen añade 1 mm por fuera
    assert recorte(pos, nrm, idx, d, rng, desplaza=-6.0) is None


def test_instancias_ignora_la_encia() -> None:
    assert set(instancias(np.array([0, 0, 21, 22, 21]))) == {21, 22}


def test_la_componente_principal_suelta_el_trozo_lejano() -> None:
    x = np.arange(0, 5, 0.2)
    cerca = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    lejos = cerca[:5] + [40.0, 0.0, 0.0]
    pos = np.concatenate([lejos, cerca])
    principal = componente_principal(pos, np.arange(len(pos)))
    assert set(principal) == set(range(5, len(pos)))
    assert list(componente_principal(pos, np.array([3]))) == [3]
