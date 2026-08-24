"""Cerrar una cáscara abierta en un sólido imprimible.

Estos tests existen porque «cerrado» es una propiedad que se **cuenta**, no que se
afirma, y porque la forma que hay que cerrar aquí no es la que uno supondría. El
contorno de un escaneo intraoral proyectado sobre el plano de la base se autointersecta
—136 veces sobre 3.978 segmentos, medido— y da 0,001 vueltas alrededor de su propio
centroide en vez de una: la arcada es una banda en herradura y su centro cae en el
paladar, que es aire. Cualquier método que suponga un contorno simple o estrellado deja
la malla abierta, y una malla abierta no se imprime.
"""

from __future__ import annotations

import numpy as np
from export_agents.solido import (
    _tapa_plana,
    cierra_en_solido,
    es_estanca,
    lazos_de_borde,
    volumen_con_signo,
)


def _casquete(n: int = 40, radio: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Media esfera abierta por abajo: la forma de un escaneo, sin fondo ni interior."""
    theta = np.linspace(0.0, np.pi / 2, n)
    phi = np.linspace(0.0, 2 * np.pi, n)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    x = radio * np.sin(t) * np.cos(p)
    y = radio * np.sin(t) * np.sin(p)
    z = radio * np.cos(t)
    pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    f, c = x.shape
    i, j = np.meshgrid(np.arange(f - 1), np.arange(c - 1), indexing="ij")
    a, b = (i * c + j).ravel(), (i * c + j + 1).ravel()
    d, e = ((i + 1) * c + j).ravel(), ((i + 1) * c + j + 1).ravel()
    caras = np.concatenate([np.column_stack([a, b, d]), np.column_stack([b, e, d])])
    return pos, caras.astype(np.int64)


ARRIBA = np.array([0.0, 0.0, 1.0])


def test_una_cascara_abierta_sale_ESTANCA():
    """El criterio, y el único que importa: cada arista compartida por dos caras. Un
    laminador necesita saber qué es interior, y una superficie sin volumen no lo dice."""
    pos, caras = _casquete()
    assert not es_estanca(caras), "la fixture ya venía cerrada: no prueba nada"

    _, cerradas, informe = cierra_en_solido(pos, caras, arriba=ARRIBA)

    assert informe["estanca"] is True
    assert es_estanca(cerradas)
    assert not lazos_de_borde(cerradas), "quedan bordes abiertos"


def test_el_volumen_sale_POSITIVO_aunque_la_malla_venga_del_reves():
    """Una malla cerrada con las normales hacia dentro se imprime del revés. El signo del
    volumen lo detecta de una vez, y sólo tiene sentido preguntarlo si es estanca: sobre
    un bobinado inconsistente el número no significa nada."""
    pos, caras = _casquete()
    pos2, cerradas, informe = cierra_en_solido(pos, caras[:, ::-1].copy(), arriba=ARRIBA)

    assert informe["estanca"] is True
    assert volumen_con_signo(pos2, cerradas) > 0.0
    assert float(informe["volumen_mm3"]) > 0.0


def test_la_base_es_perpendicular_al_eje_que_se_le_pasa():
    """El eje Z de un escaneo es el que tenía la máquina y no significa nada. Una base
    construida sobre él sale inclinada y el modelo se cae de la bandeja."""
    pos, caras = _casquete()
    eje = np.array([0.0, 1.0, 0.0])
    pos2, cerradas, _ = cierra_en_solido(pos, caras, arriba=eje)

    nuevos = pos2[len(pos) :]
    # Los vértices de la base tienen que caer todos en el mismo plano perpendicular
    # al eje pedido, no al Z.
    altura = nuevos @ eje
    assert float(np.ptp(altura)) < 1e-9
    assert float(np.ptp(nuevos @ np.array([0.0, 0.0, 1.0]))) > 1.0


def test_un_agujero_pequeno_se_tapa_DONDE_ESTA_y_no_baja_al_plano():
    """Bajar todos los lazos al plano produciría chimeneas atravesando el modelo. El
    grande es la apertura; los pequeños son huecos que la cámara no llegó a ver."""
    pos, caras = _casquete()
    # Se abre un agujero quitando una cara CLARAMENTE interior. El índice del medio no
    # vale: cae en la costura entre los dos bloques de triángulos de la rejilla, junto al
    # polo, donde la parametrización es degenerada y no llega a abrir nada.
    interior = 20 * 39 + 20
    caras = np.delete(caras, interior, axis=0)
    assert len(lazos_de_borde(caras)) == 2, "la fixture no ha abierto ningún agujero"

    _, cerradas, informe = cierra_en_solido(pos, caras, arriba=ARRIBA)

    assert informe["huecos_tapados"] == 1
    assert informe["estanca"] is True


def test_una_malla_YA_cerrada_se_devuelve_intacta():
    pos, caras = _casquete()
    _, cerradas, _ = cierra_en_solido(pos, caras, arriba=ARRIBA)
    pos2, otra, informe = cierra_en_solido(pos, cerradas, arriba=ARRIBA)

    assert informe.get("ya_cerrada") is True
    assert np.array_equal(otra, cerradas)


# --- la tapa, aparte ---------------------------------------------------------- #


def _herradura(n: int = 60) -> np.ndarray:
    """Un contorno en herradura: no es convexo ni estrellado desde su centroide.

    Es la forma real del problema. Una arcada es una banda en U y el centroide de su
    borde cae en el hueco, que es aire — por eso el contorno da cero vueltas alrededor
    de él y cualquier abanico desde el centro fabrica triángulos sobre el vacío.
    """
    a = np.linspace(np.pi, 0.0, n)
    fuera = np.column_stack([10 * np.cos(a), 10 * np.sin(a), np.zeros(n)])
    dentro = np.column_stack([6 * np.cos(a[::-1]), 6 * np.sin(a[::-1]), np.zeros(n)])
    return np.concatenate([fuera, dentro])


def test_la_tapa_tiene_por_frontera_EXACTAMENTE_el_contorno():
    """La propiedad que hace falta, y la razón de recortar orejas en vez de triangular
    por Delaunay: si la frontera de la tapa no es el anillo, la base no cierra contra la
    falda y el sólido queda abierto por dentro."""
    contorno = _herradura()
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    caras = _tapa_plana(contorno, u, v)

    assert len(caras) == len(contorno) - 2, "no es una triangulación del polígono"
    aristas = np.sort(
        np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [0, 2]]]), axis=1
    )
    unicas, cuenta = np.unique(aristas, axis=0, return_counts=True)
    frontera = {tuple(x) for x in unicas[cuenta == 1].tolist()}

    n = len(contorno)
    esperada = {
        tuple(sorted((i, (i + 1) % n))) for i in range(n)
    }
    assert frontera == esperada


def test_la_tapa_no_se_sale_del_contorno():
    """Un abanico desde el centroide cerraría igual de bien y pondría material sobre el
    hueco de la herradura. Los baricentros tienen que caer dentro del polígono."""
    contorno = _herradura()
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    caras = _tapa_plana(contorno, u, v)

    xy = contorno[:, :2]
    b = xy[caras].mean(axis=1)
    radio = np.linalg.norm(b, axis=1)
    # La herradura tiene material entre r=6 y r=10 en el semiplano y>0. Se tolera un
    # puñado de triángulos degenerados en los extremos rectos, no un abanico entero.
    fuera = (radio < 5.5) | (radio > 10.5)
    assert fuera.mean() < 0.05, f"{fuera.sum()} de {len(caras)} triángulos fuera"
