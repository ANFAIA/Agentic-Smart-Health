"""Las vistas (§7). Lo que se prueba es que los NOMBRES anatomicos sean ciertos.

Una vista que se llama «vestibular derecha» y encuadra la izquierda es el fallo caro: no
revienta, se ve bien, y quien la abra no tiene forma de notarlo. Por eso los tests se
construyen sobre una arcada sintetica cuyos ejes se conocen de antemano, y se comprueba
que las direcciones MEDIDAS coincidan con los que se pusieron.
"""

from __future__ import annotations

import numpy as np
import pytest
from uos.vistas import FOV_GRADOS, anatomical_frame, build_views

# Arcada de juguete, con ejes elegidos a proposito para que NO sean los de los indices:
#   +X = derecha del paciente · +Y = anterior · +Z = oclusal
# Si el codigo se apoyara en el orden de las columnas en vez de en las etiquetas, cualquier
# permutacion de estos ejes rompería el test, que es justo lo que se quiere.
_EJES = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])


def _arcada(rot: np.ndarray = _EJES) -> tuple[np.ndarray, np.ndarray]:
    """Una herradura de dientes sobre una encia, con sus codigos FDI.

    Cuadrante 1 (derecha del paciente) en +X, cuadrante 2 (izquierda) en -X, y las piezas
    1-8 recorriendo la arcada del frente al fondo. La encia queda por DEBAJO en oclusal.
    """
    pos, etq = [], []
    for cuadrante, signo in ((1, +1.0), (2, -1.0)):
        for pieza in range(1, 9):
            # La pieza 1 va delante (anterior alto) y la 8 al fondo.
            angulo = np.deg2rad(10 + (pieza - 1) * 11)
            x = signo * 25.0 * np.sin(angulo)
            y = 25.0 * np.cos(angulo)
            nube = np.random.default_rng(cuadrante * 10 + pieza).normal(
                [x, y, 4.0], [1.5, 1.5, 3.0], size=(80, 3)
            )
            pos.append(nube)
            etq += [cuadrante * 10 + pieza] * len(nube)
    # Encia: un anillo mas ancho y por debajo, sin etiqueta.
    t = np.linspace(0, np.pi, 400)
    encia = np.stack([28 * np.cos(t), 28 * np.sin(t), np.zeros_like(t) - 3.0], axis=1)
    pos.append(encia)
    etq += [0] * len(encia)
    return np.vstack(pos) @ rot.T, np.array(etq, dtype=np.int64)


def test_los_tres_ejes_se_miden_y_apuntan_a_donde_dicen():
    """Oclusal hacia las coronas, derecha hacia el cuadrante 1, anterior hacia los incisivos."""
    pos, etq = _arcada()

    marco, motivo = anatomical_frame(pos, etq)

    assert motivo == ""
    assert marco is not None
    assert marco.oclusal @ [0, 0, 1] > 0.9
    assert marco.derecha @ [1, 0, 0] > 0.9
    assert marco.anterior @ [0, 1, 0] > 0.9


def test_los_ejes_siguen_a_la_arcada_cuando_la_arcada_gira():
    """La base sale de la geometria y las etiquetas, no del orden de las columnas.

    Un escaner no entrega la boca alineada con los ejes del fichero. Si el codigo se
    apoyara en que +Z es lo oclusal, aqui daria vistas invertidas sin avisar.
    """
    # Giro de 90 grados en torno a X: el que era +Z (oclusal) pasa a ser +Y.
    giro = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    pos, etq = _arcada(giro)

    marco, _ = anatomical_frame(pos, etq)

    assert marco is not None
    assert marco.oclusal @ (giro @ [0, 0, 1]) > 0.9
    assert marco.derecha @ (giro @ [1, 0, 0]) > 0.9
    assert marco.anterior @ (giro @ [0, 1, 0]) > 0.9


def test_la_base_es_ortonormal():
    """Si no lo fuera, `up` no seria perpendicular a la direccion y el visor la reproyecta."""
    pos, etq = _arcada()

    marco, _ = anatomical_frame(pos, etq)

    assert marco is not None
    m = np.stack([marco.oclusal, marco.derecha, marco.anterior])
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)


def test_sin_etiquetas_NO_se_inventan_vistas_y_se_dice_por_que():
    """Bautizar los ejes principales de la nube da nombres plausibles y a veces invertidos."""
    pos, _ = _arcada()

    vistas, avisos = build_views(pos, np.zeros(len(pos), dtype=np.int64), visita="v1")

    assert vistas == []
    assert len(avisos) == 1
    assert "ni una pieza etiquetada" in avisos[0]


def test_sin_encia_no_hay_signo_del_eje_oclusal():
    """El eje lo da la geometria; el SIGNO solo lo da que las coronas caigan de un lado."""
    pos, etq = _arcada()
    m = etq > 0

    marco, motivo = anatomical_frame(pos[m], etq[m])

    assert marco is None
    assert "sin encia" in motivo


def test_un_solo_lado_de_la_arcada_no_permite_decir_cual_es_la_derecha():
    pos, etq = _arcada()
    m = (etq == 0) | (etq // 10 == 1)

    marco, motivo = anatomical_frame(pos[m], etq[m])

    assert marco is None
    assert "los dos lados" in motivo


def test_las_camaras_miran_al_centro_desde_la_direccion_que_nombran():
    pos, etq = _arcada()

    vistas, _ = build_views(pos, etq, visita="v1")

    por_id = {v.id: v for v in vistas}
    assert set(por_id) == {
        "view.oclusal", "view.frontal", "view.vestibular_derecha",
        "view.vestibular_izquierda",
    }
    for id_, eje in (("view.oclusal", [0, 0, 1]), ("view.frontal", [0, 1, 0]),
                     ("view.vestibular_derecha", [1, 0, 0]),
                     ("view.vestibular_izquierda", [-1, 0, 0])):
        c = por_id[id_]
        d = np.array(c.camera.position) - np.array(c.camera.target)
        assert d @ eje / np.linalg.norm(d) > 0.9, id_
        # `up` perpendicular a la direccion de vista: si no, la camara es degenerada.
        # La tolerancia es la del redondeo a la micra con el que se escribe el JSON.
        assert abs(np.array(c.camera.up) @ (d / np.linalg.norm(d))) < 1e-3
        assert c.camera.fov == FOV_GRADOS


def test_la_camara_se_aleja_lo_bastante_para_que_la_arcada_quepa():
    """El encuadre se calcula, no se fija: una arcada al doble de tamano se ve igual."""
    pos, etq = _arcada()

    cerca = {v.id: v for v in build_views(pos, etq, visita="v1")[0]}
    lejos = {v.id: v for v in build_views(pos * 2.0, etq, visita="v1")[0]}

    d_cerca = np.linalg.norm(np.array(cerca["view.oclusal"].camera.position)
                             - np.array(cerca["view.oclusal"].camera.target))
    d_lejos = np.linalg.norm(np.array(lejos["view.oclusal"].camera.position)
                             - np.array(lejos["view.oclusal"].camera.target))
    assert d_lejos == pytest.approx(2 * d_cerca, rel=0.02)


def test_una_pieza_con_contenido_clinico_tiene_su_vista_y_mira_desde_vestibular():
    pos, etq = _arcada()

    vistas, avisos = build_views(pos, etq, visita="v1", piezas=["16"])

    assert avisos == []
    v = next(v for v in vistas if v.id == "view.pieza_16")
    objetivo = np.array(v.camera.target)
    centro_pieza = pos[etq == 16].mean(axis=0)
    assert np.allclose(objetivo, centro_pieza, atol=1e-3)
    # La camara queda por FUERA de la arcada, no dentro de la boca.
    fuera = objetivo - pos.mean(axis=0)
    assert (np.array(v.camera.position) - objetivo) @ fuera > 0


def test_una_pieza_anotada_que_el_escaner_no_trae_se_AVISA_en_vez_de_omitirse():
    """El informe habla del 47 y el escaner es del maxilar: el hueco tiene que verse."""
    pos, etq = _arcada()

    vistas, avisos = build_views(pos, etq, visita="v1", piezas=["47"])

    assert not any(v.id == "view.pieza_47" for v in vistas)
    assert len(avisos) == 1
    assert "FDI 47" in avisos[0]


def test_las_piezas_que_faltan_van_en_UN_aviso_y_no_en_uno_por_diente():
    """Un maxilar escaneado con un informe de las dos arcadas da dieciseis huecos. En
    dieciseis lineas entierran los motivos que solo aparecen una vez."""
    pos, etq = _arcada()
    inferior = [str(c) for c in (31, 32, 33, 34, 41, 42, 43, 44)]

    vistas, avisos = build_views(pos, etq, visita="v1", piezas=[*inferior, "16"])

    assert any(v.id == "view.pieza_16" for v in vistas)
    assert len(avisos) == 1
    assert "8 pieza(s)" in avisos[0]
    assert all(f"FDI {c}" in avisos[0] for c in inferior)


def test_la_capa_de_apariencia_solo_aparece_si_hay_apariencia():
    """Declarar `gs` sin escena de apariencia haria que el visor buscara una capa que no esta."""
    pos, etq = _arcada()

    sin_gs, _ = build_views(pos, etq, visita="v1")
    con_gs, _ = build_views(pos, etq, visita="v1", con_apariencia=True)

    assert set(sin_gs[0].layers) == {"mesh"}
    assert set(con_gs[0].layers) == {"mesh", "gs"}
    # Por debajo de 1.0: es reconstruida contra renders, no medida.
    assert con_gs[0].layers["gs"].opacity < 1.0
