"""La frontera de cada pieza. Ver `analysis_agents.frontera`.

Esto existe por lo que se ve en el visor: seleccionas una pieza y viene con medio diente
vecino. El contenedor decía la verdad sobre esa pieza y enseñaba otra cosa.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis_agents.frontera import (
    EXCESO_ADMITIDO_MM,
    nucleos,
    reparte_por_geodesica,
)


def _rejilla(n: int = 24, m: int = 12, paso: float = 0.4):
    """Una lámina `n x m` triangulada, como un trozo de superficie."""
    idx = np.arange(n * m).reshape(n, m)
    x, y = np.meshgrid(np.arange(n) * paso, np.arange(m) * paso, indexing="ij")
    pos = np.stack([x.ravel(), y.ravel(), np.zeros(n * m)], 1)
    caras = np.concatenate([
        np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1]], -1).reshape(-1, 3),
        np.stack([idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3),
    ])
    return pos, caras, idx


def test_una_isla_suelta_vuelve_a_su_pieza() -> None:
    """Un maxilar real trae decenas de islas por código, lejos de la pieza que las declara.

    ⚠️ **Lo que NO se afirma:** que la competencia arregle el desborde. Se escribió un test
    que lo daba por hecho y es falso — ver `test_un_desborde_UNIFORME…` y el aviso de la
    propia función. Lo que sí hace, y es comprobable, es devolver a su sitio un trozo que
    está más cerca de otra pieza que de la que lo reclama.
    """
    pos, caras, idx = _rejilla()
    verdad = np.where(np.arange(24 * 12) // 12 < 12, 11, 12).astype(np.int64)
    sucia = verdad.copy()
    sucia[idx[20, 3]] = 11                    # una isla del 11 en pleno territorio del 12

    limpia = reparte_por_geodesica(pos, caras, sucia)
    assert limpia[idx[20, 3]] == 12
    # La frontera entre las dos piezas se mueve como mucho un anillo: en una rejilla el
    # punto medio geodesico no cae exactamente sobre la fila de la verdad.
    assert (limpia == verdad).mean() > 0.95


def test_un_desborde_UNIFORME_no_lo_arregla_la_competencia_sola() -> None:
    """⚠️ **El límite del método, y hay que tenerlo escrito.**

    La competencia deja la frontera a medio camino ENTRE LOS NÚCLEOS, así que si la
    etiqueta invade a la vecina de forma pareja, el núcleo se desplaza con ella y el
    resultado no se mueve. Reconstruir la frontera diente-diente de Teeth3DS+ desde los
    núcleos da 0,995 de acuerdo — pero eso es partiendo de etiquetas de experto, donde los
    núcleos están bien. Sobre el caso real la competencia sola no cambia ni una pieza de
    veredicto (4/14) y sólo baja el |exceso| mediano de 2,60 a 2,38 mm.

    Lo que sí funciona es combinarla con el recorte al cuello: 4/14 -> 7/14.
    """
    pos, caras, idx = _rejilla()
    verdad = np.where(np.arange(24 * 12) // 12 < 12, 11, 12).astype(np.int64)
    sucia = verdad.copy()
    sucia[idx[12:18].ravel()] = 11            # seis columnas enteras, parejo

    limpia = reparte_por_geodesica(pos, caras, sucia)
    assert (limpia == verdad).mean() == (sucia == verdad).mean()


def test_el_nucleo_borra_las_islas_sueltas() -> None:
    """⚠️ **Un maxilar real trae decenas de islas por código**, y bastan para inflar un
    ancho medido hasta 41 mm. El núcleo las borra porque una isla es todo borde."""
    pos, caras, idx = _rejilla()
    etq = np.full(24 * 12, 11, np.int64)
    etq[idx[20:, :].ravel()] = 0
    etq[idx[22, 5]] = 12                      # una isla de un vértice
    nuc = nucleos(caras, etq)
    assert (nuc == 12).sum() == 0
    assert (nuc == 11).sum() > 0


def test_la_competencia_no_toca_la_encia() -> None:
    """Aquí se decide de qué DIENTE es cada vértice de diente, no dónde acaba el diente."""
    pos, caras, idx = _rejilla()
    etq = np.where(np.arange(24 * 12) // 12 < 12, 11, 12).astype(np.int64)
    etq[idx[:, 10:].ravel()] = 0
    salida = reparte_por_geodesica(pos, caras, etq)
    assert ((etq == 0) == (salida == 0)).all()


def test_el_umbral_de_ancho_sale_de_etiquetas_de_experto() -> None:
    """⚠️ **Contar coronas anchas no mide un defecto: las de experto también lo fallan.**

    El gate declaraba «11 de 14 coronas más anchas de lo admitido» — y las etiquetas de
    experto de Teeth3DS+ dan **86 de 111 (77 %)** con ese mismo recuento. Lo que separa es
    la magnitud: `|medido - tabla|` da p95 **1,92 mm** sobre 188 coronas de experto, y
    nuestro caso daba p50 2,38 y p90 7,35.
    """
    assert 1.5 <= EXCESO_ADMITIDO_MM <= 2.0


# --------------------------------------------------------------------------- #
# Las tres funciones que no tenían prueba, aunque un comentario decía que sí.

def _valle(tooth_desde: float, nx: int = 64, paso: float = 0.5):
    """Loseta con un pliegue recto en `y = 0`; diente en `y >= tooth_desde`."""
    ys = np.arange(-10.0, 10.0 + paso, paso)
    xs = np.arange(nx) * paso
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X, Y, 0.4 * np.abs(Y)], axis=-1).reshape(-1, 3)
    ny = len(ys)
    tri = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = i * ny + j, i * ny + j + 1
            c, d = (i + 1) * ny + j, (i + 1) * ny + j + 1
            tri += [(a, c, b), (b, c, d)]
    etq = np.where(pos[:, 1] >= tooth_desde, 11, 0).astype(np.int64)
    return pos, np.asarray(tri, dtype=np.int64), etq


def test_la_concavidad_es_positiva_en_el_pliegue_y_nula_en_lo_liso() -> None:
    from analysis_agents.frontera import concavidad

    pos, tri, _ = _valle(0.0)
    c = concavidad(pos, tri)
    interior = (pos[:, 0] > 2.0) & (pos[:, 0] < 28.0)
    pliegue = interior & (np.abs(pos[:, 1]) < 1e-9)
    liso = interior & (np.abs(pos[:, 1]) > 2.0) & (np.abs(pos[:, 1]) < 8.0)
    assert c[pliegue].min() > 0.05
    assert np.abs(c[liso]).max() < 1e-9


def test_el_eje_oclusal_va_de_la_encia_a_los_dientes() -> None:
    from analysis_agents.frontera import eje_oclusal

    pos, _, etq = _valle(0.0)
    eje = eje_oclusal(pos, etq)
    assert np.linalg.norm(eje) == pytest.approx(1.0)
    assert eje[1] > 0.9                          # los dientes están en +y


def test_el_recorte_al_cuello_CORTA_AUNQUE_NO_HAYA_PLIEGUE() -> None:
    """⚠️ Fija el defecto por el que no se aplica: no sabe declinar.

    Una lámina completamente plana no tiene cuello que encontrar. Aun así el recorte
    devuelve a la encía todo lo que queda por debajo de la altura de tabla, porque cuando
    no encuentra anillo cóncavo corta donde dice la anatomía. Sobre el caso clínico eso
    fue «la encía bajando sobre las coronas». Un método nuevo tiene que hacer lo
    contrario aquí: dejar la etiqueta como está.
    """
    from analysis_agents.frontera import recorta_al_cuello

    pos, caras, _ = _rejilla(n=80, m=30, paso=0.4)
    etq = np.where(pos[:, 0] >= 4.0, 11, 0).astype(np.int64)   # 28 mm de «corona» plana
    recortada = recorta_al_cuello(pos, caras, etq)
    assert int((recortada > 0).sum()) < int((etq > 0).sum())
    assert not (recortada[etq == 0] > 0).any()                 # y nunca hace crecer


# --------------------------------------------------------------------------- #
# `ancla_al_pliegue`: lleva el margen al surco, o no toca nada.

def test_una_etiqueta_desbordada_vuelve_al_surco() -> None:
    """El caso del modelo: la etiqueta baja 3 mm por la encía y el surco está en y = 0."""
    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(-3.0)
    nueva, registros = ancla_al_pliegue(pos, tri, etq)
    assert pos[nueva > 0, 1].min() == pytest.approx(0.0)
    r = registros[11]
    assert r["moved"] > 0 and not r["declined"]
    assert r["evidence"] > 1.5 and r["coverage"] == pytest.approx(1.0)
    assert r["concavity_before"] == pytest.approx(0.0, abs=1e-6)


def test_una_etiqueta_que_ya_esta_en_el_surco_no_se_mueve() -> None:
    """Lo que hace un anotador. Sin esto, cada pasada se comería un anillo de corona."""
    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(0.0)
    nueva, registros = ancla_al_pliegue(pos, tri, etq)
    assert (nueva == etq).all()
    assert registros[11]["moved"] == 0 and not registros[11]["declined"]


def test_sin_pliegue_no_se_toca_nada() -> None:
    """La diferencia con `recorta_al_cuello`, que sobre esta misma lámina corta."""
    from analysis_agents.frontera import ancla_al_pliegue

    pos, caras, _ = _rejilla(n=80, m=30, paso=0.4)
    etq = np.where(pos[:, 0] >= 4.0, 11, 0).astype(np.int64)
    nueva, registros = ancla_al_pliegue(pos, caras, etq)
    assert (nueva == etq).all()
    assert registros[11]["moved"] == 0


def test_con_evidencia_insuficiente_declina_y_lo_dice() -> None:
    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(-3.0)
    nueva, registros = ancla_al_pliegue(pos, tri, etq, fuerza_minima=1e9)
    assert (nueva == etq).all()
    assert registros[11]["declined"] and registros[11]["reason"] == "weak_crease"
    assert registros[11]["evidence"] is not None      # declina enseñando lo que vio


def test_solo_cede_y_ninguna_pieza_desaparece() -> None:
    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(-3.0)
    dos = etq.copy()
    dos[(etq > 0) & (pos[:, 0] > 16.0)] = 12
    nueva, registros = ancla_al_pliegue(pos, tri, dos)
    assert not ((nueva > 0) & (dos == 0)).any()        # la encía nunca pasa a diente
    assert ((nueva == dos) | (nueva == 0)).all()       # y ningún vértice cambia de pieza
    assert set(np.unique(nueva)) == {0, 11, 12}
    assert registros[11]["moved"] > 0 and registros[12]["moved"] > 0


def test_el_contacto_entre_dos_piezas_no_es_un_sitio_por_donde_cortar() -> None:
    """⚠️ El fallo del corte por pieza: una etiqueta que invade a la vecina.

    El «11» cubre aquí también media pieza 12. Cortando cada pieza por separado, el corte
    más barato del 11 era su contacto con el 12 y media corona ajena acababa en la encía.
    Con toda la arcada en el mismo grafo, las dos piezas llegan al mismo surco.
    """
    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(-3.0)
    invadida = etq.copy()
    invadida[(etq > 0) & (pos[:, 0] > 24.0)] = 12      # el 11 ocupa tres cuartos del ancho
    nueva, _ = ancla_al_pliegue(pos, tri, invadida)
    corona = pos[:, 1] >= 0.0
    assert (nueva[corona] == invadida[corona]).all()   # nada de la corona se cede
    assert not (nueva[~corona] > 0).any()              # y todo el desborde, sí


def test_una_esquirla_no_se_toca() -> None:
    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(-3.0)
    con_esquirla = etq.copy()
    esquirla = (pos[:, 0] < 1.0) & (pos[:, 1] < -2.0) & (etq > 0)
    con_esquirla[esquirla] = 28
    nueva, registros = ancla_al_pliegue(pos, tri, con_esquirla)
    assert registros[28]["reason"] == "fragment" and registros[28]["declined"]
    assert (nueva[esquirla] == 28).all()


def test_es_idempotente_y_sus_registros_se_pueden_guardar() -> None:
    import json

    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(-3.0)
    una, registros = ancla_al_pliegue(pos, tri, etq)
    otra, _ = ancla_al_pliegue(pos, tri, una)
    assert (una == otra).all()
    assert json.loads(json.dumps({str(k): v for k, v in registros.items()}))


def test_sin_encia_o_sin_dientes_devuelve_lo_que_recibe() -> None:
    from analysis_agents.frontera import ancla_al_pliegue

    pos, tri, etq = _valle(0.0)
    todo = np.full(len(pos), 11, dtype=np.int64)
    nada = np.zeros(len(pos), dtype=np.int64)
    assert (ancla_al_pliegue(pos, tri, todo)[0] == todo).all()
    assert ancla_al_pliegue(pos, tri, nada) == (pytest.approx(nada), {})
