"""Las tres decisiones de la preparación, cada una fijada por lo que costó averiguarla.

Este módulo existe porque esa lógica vivía en un script: para procesar un caso clínico
real había que escribir a mano el aislamiento de arcada, la elección del lóbulo y el
submuestreo. Los tests no comprueban que las funciones «hagan algo», comprueban que
sigan tomando las decisiones que se midieron.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion_agents import HU_CORONA, arcada_del_nombre, icp, nubes_para_registro, plano_oclusal
from fusion_agents.preparacion import (
    BARRIDO_OBJETIVO,
    VALLE_MAXIMO,
    plano_oclusal_del_esmalte,
    puntua_contra_esmalte,
    separacion_de_arcadas,
)


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


# --- el plano oclusal por el esmalte ---------------------------------------- #
def test_el_plano_sale_del_pico_no_de_un_hueco():
    """En oclusión las coronas se tocan: hay UN pico, y ahí está el plano.

    Es la corrección del criterio anterior. `separacion_de_arcadas` busca un valle, y la
    postura de la adquisición garantiza que no lo hay — medido sobre el caso real: valle
    0,65 a HU ≥ 1200 y 1,00 a HU ≥ 1500, unimodal las dos veces. Un plano no necesita un
    hueco; solo necesita dejar cada diente del lado correcto.
    """
    rng = np.random.default_rng(0)
    # Esmalte apilado en el plano oclusal, con las raíces repartiéndose arriba y abajo.
    esmalte = rng.normal(40.0, 2.0, 4000)
    raices = np.concatenate([rng.normal(55.0, 6.0, 800), rng.normal(25.0, 6.0, 800)])
    z = np.concatenate([esmalte, raices])

    plano = plano_oclusal_del_esmalte(z)
    assert abs(plano - 40.0) < 2.0

    # Y el criterio del valle NO lo encuentra: por eso hacía falta otra función.
    assert separacion_de_arcadas(z)[1] > VALLE_MAXIMO


def test_el_plano_no_depende_de_acertar_con_el_umbral():
    """Medido sobre el caso real: 40,9 · 40,8 · 40,7 · 40,7 · 39,6 · 40,9 · 40,7 mm para
    HU ≥ 1200 … 1900. Siete umbrales dentro de 1,3 mm — el modo es estable porque el
    esmalte de las dos arcadas se apila en la misma altura, no porque el umbral acierte.
    """
    rng = np.random.default_rng(1)
    denso = np.concatenate([rng.normal(40.0, 2.0, 4000), rng.uniform(0.0, 90.0, 3000)])
    escaso = np.concatenate([rng.normal(40.0, 2.0, 900), rng.uniform(0.0, 90.0, 200)])
    assert abs(plano_oclusal_del_esmalte(denso) - plano_oclusal_del_esmalte(escaso)) < 3.0


def test_un_plano_dado_manda_sobre_el_veto_del_valle():
    """`nubes_para_registro` acepta el plano de fuera y parte, valle o no valle.

    Sin esta puerta el módulo se negaba a partir en el único caso que importa —dos arcadas
    en oclusión— y registraba el escaneo de una arcada contra las dos.
    """
    rng = np.random.default_rng(2)
    centros = np.column_stack([rng.normal(0, 10, 3000), rng.normal(0, 10, 3000),
                               rng.normal(40.0, 8.0, 3000)])
    campo = {"centers": centros}
    vertices = rng.normal(0, 10, (500, 3))

    _, sin_plano, inf_sin = nubes_para_registro(campo, vertices, arcada="maxilar")
    assert "no_se_parte" in inf_sin  # unimodal: se niega, y lo dice

    _, con_plano, inf_con = nubes_para_registro(
        campo, vertices, arcada="maxilar", plano=40.0
    )
    assert inf_con["plano_dado"] is True
    assert inf_con["plano_oclusal_mm"] == 40.0
    assert len(con_plano) < len(sin_plano)  # partió de verdad


# --- el árbitro que elige el objetivo --------------------------------------- #
def test_la_encia_no_puede_cambiar_el_orden_entre_poses():
    """El árbitro es una FRACCIÓN por esto: la encía no tiene esmalte debajo.

    Con una mediana —lo que usaba el script, que sí tenía etiquetas— el número mediría
    sobre todo cuánta encía trae el escaneo. Es el fallo que invalidó una ejecución entera.
    """
    rng = np.random.default_rng(0)
    esmalte = rng.normal(0, 3, (2000, 3))
    coronas = esmalte[:400] + rng.normal(0, 0.3, (400, 3))
    encia = rng.normal([0, 0, 40], 3, (1600, 3))  # lejos, sin contrapartida posible

    solo_coronas = puntua_contra_esmalte(coronas, esmalte)
    con_encia = puntua_contra_esmalte(np.vstack([coronas, encia]), esmalte)
    assert solo_coronas > 0.9
    # Diluye el valor, pero no lo anula: la señal sigue siendo la de las coronas.
    assert 0.15 < con_encia < solo_coronas

    # Y con la máscara de corona se recupera el número afilado, sin necesitarla.
    mascara = np.zeros(2000, dtype=bool)
    mascara[:400] = True
    assert puntua_contra_esmalte(
        np.vstack([coronas, encia]), esmalte, corona=mascara
    ) == pytest.approx(solo_coronas)


def test_el_objetivo_se_elige_midiendo_y_no_por_el_residuo():
    """`registrar` activa el barrido: gana el que más vértices acerca al esmalte.

    Sin esto se usaba `hu_corona` a secas, que es **el peor de los cinco umbrales** sobre
    el caso medido (6,84 mm en el maxilar, 7,35 en la mandíbula).
    """
    rng = np.random.default_rng(1)
    n = 6000
    centros = rng.normal(0, 12, (n, 3))
    hu = rng.uniform(300, 1300, n)
    # Un núcleo denso —el «esmalte»— que es lo único con lo que la malla puede casar.
    centros[:1200] = rng.normal(0, 2.5, (1200, 3))
    hu[:1200] = rng.uniform(1850, 2000, 1200)
    campo = {"centers": centros}
    vertices = centros[:600] + rng.normal(0, 0.2, (600, 3))

    _, destino, informe = nubes_para_registro(
        campo, vertices, arcada=None, hu=hu, registrar=icp
    )
    assert informe["hu_objetivo"] in BARRIDO_OBJETIVO
    assert informe["puntuacion_arbitro"] > 0.5
    assert len(informe["puntuaciones"]) >= 2, "tiene que haber comparado candidatos"
    assert len(destino) > 0


def test_sin_registrar_no_hay_barrido_y_se_comporta_como_antes():
    """El árbitro es opt-in: quien no lo pide conserva el camino de siempre."""
    rng = np.random.default_rng(2)
    campo = {"centers": rng.normal(0, 10, (3000, 3))}
    hu = rng.uniform(300, 2000, 3000)
    _, _, informe = nubes_para_registro(campo, rng.normal(0, 10, (200, 3)),
                                        arcada=None, hu=hu)
    assert "hu_objetivo" not in informe and "puntuaciones" not in informe


def test_partir_las_dos_arcadas_hay_que_declararlo():
    """`dos_arcadas` no se deduce del dato, y se intentó: ni valle ni extensión sirven."""
    rng = np.random.default_rng(3)
    centros = np.column_stack([rng.normal(0, 10, 4000), rng.normal(0, 10, 4000),
                               rng.normal(40.0, 9.0, 4000)])
    hu = np.full(4000, 1500.0)
    campo = {"centers": centros}
    v = rng.normal(0, 10, (300, 3))

    _, sin_declarar, inf_sin = nubes_para_registro(campo, v, arcada="maxilar", hu=hu)
    assert "no_se_parte" in inf_sin  # por defecto NO parte: tirar media arcada es peor

    _, partido, inf_con = nubes_para_registro(
        campo, v, arcada="maxilar", hu=hu, dos_arcadas=True
    )
    assert inf_con["plano_por_esmalte"] is True
    assert len(partido) < len(sin_declarar)

