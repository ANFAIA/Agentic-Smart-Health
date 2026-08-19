"""Las tres decisiones de la preparación, cada una fijada por lo que costó averiguarla.

Este módulo existe porque esa lógica vivía en un script: para procesar un caso clínico
real había que escribir a mano el aislamiento de arcada, la elección del lóbulo y el
submuestreo. Los tests no comprueban que las funciones «hagan algo», comprueban que
sigan tomando las decisiones que se midieron.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion_agents import HU_CORONA, arcada_del_nombre, nubes_para_registro, plano_oclusal


def _dos_lobulos(sep: float = 20.0, n: int = 2000, semilla: int = 0) -> np.ndarray:
    """Dos nubes separadas en z, como las dos arcadas de un CBCT."""
    rng = np.random.default_rng(semilla)
    bajo = rng.normal([0, 0, 0], [8, 6, 2], (n, 3))
    alto = rng.normal([0, 0, sep], [8, 6, 2], (n, 3))
    return np.vstack([bajo, alto])


# --- el plano oclusal -------------------------------------------------------- #
def test_el_corte_cae_en_el_valle_entre_los_dos_lobulos():
    z = _dos_lobulos(sep=20.0)[:, 2]
    corte = plano_oclusal(z)
    assert 5.0 < corte < 15.0, f"el corte {corte:.1f} no está entre los lóbulos"


def test_no_es_un_percentil_y_esa_es_la_diferencia():
    """Con lóbulos de tamaños muy distintos, la mediana corta dentro del grande.

    Es el fallo que dio 8 mm: el metal mete puntos por todo el campo y desequilibra los
    lóbulos, así que un percentil corta donde hay más puntos, no donde hay un valle.
    """
    rng = np.random.default_rng(0)
    grande = rng.normal([0, 0, 0], [8, 6, 2], (4000, 3))
    pequeno = rng.normal([0, 0, 20], [8, 6, 2], (300, 3))
    z = np.vstack([grande, pequeno])[:, 2]

    corte = plano_oclusal(z)
    assert corte > z[:4000].max() - 5, "el corte tiene que dejar atrás el lóbulo grande"
    assert abs(corte - np.median(z)) > 3.0, "la mediana caería dentro del lóbulo grande"


def test_con_un_solo_lobulo_lo_declara_en_vez_de_partir():
    """El fallo que este test encontró: sobre una nube unimodal la búsqueda del segundo
    pico encuentra la cola de la propia nube y devuelve un corte creíble. Quien lo usara
    para quedarse con un lóbulo tiraría media arcada **en silencio**."""
    from fusion_agents.preparacion import VALLE_MAXIMO, separacion_de_arcadas

    rng = np.random.default_rng(0)
    _, valle_uno = separacion_de_arcadas(rng.normal(0, 2, 2000))
    assert valle_uno > VALLE_MAXIMO, "una sola nube no puede pasar por dos lóbulos"

    _, valle_dos = separacion_de_arcadas(_dos_lobulos(sep=20.0)[:, 2])
    assert valle_dos <= VALLE_MAXIMO, "dos lóbulos claros sí se separan"


def test_con_una_sola_arcada_no_se_parte_y_se_dice():
    campo, hu = _campo()
    campo = {"centers": np.asarray(campo["centers"])[:2000], "origin": campo["origin"]}
    _, destino, informe = nubes_para_registro(
        campo, np.zeros((10, 3)), arcada="mandibular", hu=hu[:2000]
    )
    assert "no_se_parte" in informe and "plano_oclusal_mm" not in informe
    assert len(destino) > 0


# --- la etiqueta de arcada --------------------------------------------------- #
@pytest.mark.parametrize(
    ("nombre", "espera"),
    [
        ("PREVIO LowerJawScan.stl", "mandibular"),
        ("caso UpperJawScan.stl", "maxilar"),
        ("modelo_maxilar_01.ply", "maxilar"),
        ("arcada inferior.obj", "mandibular"),
    ],
)
def test_la_arcada_sale_del_nombre(nombre, espera):
    assert arcada_del_nombre(nombre) == espera


@pytest.mark.parametrize("nombre", ["upper_lower_bite.stl", "escaneo.stl", "modelo.obj"])
def test_un_nombre_ambiguo_devuelve_none_en_vez_de_adivinar(nombre):
    """Adivinar aquí es lo que dejó un registro de la mandíbula contra el maxilar
    puntuando 0,452 mm, que es un número perfectamente creíble."""
    assert arcada_del_nombre(nombre) is None


# --- las nubes --------------------------------------------------------------- #
def _campo(n: int = 4000, semilla: int = 0) -> tuple[dict, np.ndarray]:
    puntos = _dos_lobulos(n=n // 2, semilla=semilla)
    rng = np.random.default_rng(semilla)
    # Un tercio con HU de corona; el resto, raíz y hueso.
    hu = np.where(rng.random(len(puntos)) < 0.33, 1800.0, 600.0)
    return {"centers": puntos, "origin": np.array([5.0, -3.0, 2.0])}, hu


def test_el_destino_va_en_el_marco_del_cbct():
    """Se le suma `origin`: es exactamente para lo que el `cbct-agent` lo guarda."""
    campo, hu = _campo()
    _, destino, _ = nubes_para_registro(campo, np.zeros((10, 3)), arcada=None, hu=hu)
    sin_origen = np.asarray(campo["centers"])
    assert destino.min(0).min() > sin_origen.min(0).min() - 50  # sanidad
    campo_sin = {"centers": campo["centers"]}
    _, d2, _ = nubes_para_registro(campo_sin, np.zeros((10, 3)), arcada=None, hu=hu)
    assert not np.allclose(destino.mean(0), d2.mean(0)), "sin `origin` el marco es otro"


def test_solo_entra_la_corona_en_el_objetivo():
    """El escáner ve corona: registrar contra raíz y hueso da correspondencias falsas."""
    campo, hu = _campo()
    _, destino, informe = nubes_para_registro(
        campo, np.zeros((10, 3)), arcada=None, hu=hu
    )
    assert informe["n_corona"] == int((hu >= HU_CORONA).sum())
    assert informe["n_objetivo"] <= informe["n_corona"]


def test_se_queda_con_un_solo_lobulo_y_dice_por_donde_corto():
    campo, hu = _campo()
    _, arriba, inf_a = nubes_para_registro(campo, np.zeros((10, 3)), arcada="maxilar", hu=hu)
    _, abajo, inf_b = nubes_para_registro(
        campo, np.zeros((10, 3)), arcada="mandibular", hu=hu
    )
    assert "plano_oclusal_mm" in inf_a
    assert arriba[:, 2].mean() > abajo[:, 2].mean(), "los lóbulos están intercambiados"
    # Y sin etiqueta no se parte: no se adivina.
    _, todo, inf_c = nubes_para_registro(campo, np.zeros((10, 3)), arcada=None, hu=hu)
    assert "plano_oclusal_mm" not in inf_c
    assert len(todo) >= len(arriba)


def test_el_submuestreo_no_deja_una_nube_vacia_ni_desbordada():
    campo, hu = _campo(n=20_000)
    origen, destino, _ = nubes_para_registro(
        campo, np.zeros((50_000, 3)), arcada="maxilar", hu=hu, muestra=1000
    )
    assert len(origen) == 1000
    assert 0 < len(destino) <= 1000


def test_sin_hu_no_filtra_por_corona_en_vez_de_fallar():
    """Un campo sin densidad legible sigue siendo registrable: se usa entero."""
    campo, _ = _campo()
    _, destino, informe = nubes_para_registro(campo, np.zeros((10, 3)), arcada=None)
    assert "n_corona" not in informe
    assert len(destino) > 0
