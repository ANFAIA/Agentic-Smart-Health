"""La pose por PnP y la proyeccion del color, sobre datos con la respuesta conocida.

⚠️ **Lo que estos tests guardan no es que la pose sea buena: es que una MALA se descarte.**
Medido sobre un caso real, meter una vista con 1,15 mm de error anula por completo la
aportacion de las buenas — de dos piezas menos fuera de su caja anatomica a ninguna. Una
vista torcida no aporta poco: estropea.
"""

from __future__ import annotations

import numpy as np
import pytest
from gaussian_engine.pose_foto import (
    ARCO_MANDIBULAR,
    ARCO_MAXILAR,
    ColorMedido,
    centros_oclusales,
    es_radiografia,
    normales,
)


#: `pose_foto` importa opencv y scikit-image DENTRO de las funciones, igual que
#: `apariencia.ajusta()` hace con torch: asi importar el paquete no arrastra el extra. Por
#: eso el salto va por test y no por modulo — la mitad de estas pruebas no los necesitan.
def _necesita_extra():
    pytest.importorskip("cv2", reason="necesita el extra `appearance`")
    pytest.importorskip("skimage", reason="necesita el extra `appearance`")


def test_el_arco_no_se_ordena_con_sorted():
    """⚠️ El fallo que esto fija: `sorted()` da 11, 12, ... 17, 21, ... — empieza por el
    incisivo, se va al molar y salta de cuadrante. No es el recorrido del arco, y con ese
    orden cada blob de la foto se empareja con un diente que no le toca. Medido: la pose
    pasaba de 0,49 mm a 0,67 y la lateral que la tenia dejaba de tenerla."""
    piezas = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27]
    en_arco = [c for c in ARCO_MAXILAR if c in piezas]
    assert en_arco[0] == 17 and en_arco[-1] == 27
    assert en_arco != sorted(piezas)
    # Y el 11 y el 21 son vecinos en el arco, aunque sorted() los separe por seis piezas.
    assert abs(en_arco.index(11) - en_arco.index(21)) == 1


def test_los_dos_arcos_no_comparten_ningun_codigo():
    assert not set(ARCO_MAXILAR) & set(ARCO_MANDIBULAR)


def test_una_radiografia_se_reconoce_por_no_traer_color(tmp_path):
    """⚠️ De nueve «fotos intraorales» de un caso real, TRES eran periapicales. Se
    triplicaban a RGB y entraban en la mediana del esmalte."""
    from PIL import Image

    gris = tmp_path / "rx.jpg"
    Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (64, 1))).save(gris)
    assert es_radiografia(gris)

    color = tmp_path / "foto.jpg"
    a = np.zeros((64, 64, 3), np.uint8)
    a[..., 0] = 200
    a[..., 1] = 90
    Image.fromarray(a).save(color)
    assert not es_radiografia(color)


def test_el_centroide_es_del_TERCIO_OCLUSAL_y_no_de_la_pieza_entera():
    """⚠️ Deliberado: las etiquetas invaden la encia —medido, el 26 llega a 2,98x su
    altura de corona— asi que el centroide de la pieza entera esta desplazado hacia
    gingival, y con el la pose."""
    # Una columna de puntos: la mitad de arriba es corona, la de abajo encia etiquetada
    # como corona por error. El centroide del tercio superior tiene que estar arriba.
    z = np.linspace(0, 10, 400)
    V = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    etq = np.zeros(len(V), np.int64)
    etq[:] = 11
    # unos cuantos vertices de encia de verdad, para que el eje apunte bien
    V = np.vstack([V, np.stack([np.full(50, 5.0), np.zeros(50), np.linspace(-5, 0, 50)], 1)])
    etq = np.concatenate([etq, np.zeros(50, np.int64)])
    c = centros_oclusales(V, etq, [11])[0]
    entero = V[etq == 11].mean(0)
    assert c[2] > entero[2], "el centroide del tercio oclusal no esta por encima"


def test_las_normales_salen_unitarias():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    F = np.array([[0, 1, 2], [1, 3, 2]])
    n = normales(V, F)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_el_resumen_reparte_los_vertices_sin_perder_ninguno():
    """El contenedor tiene que poder decir cuantos vertices llevan pixel medido: sin ese
    numero, el descriptor tendria que SUPONER si el color es medido o el respaldo."""
    n = 100
    med = np.zeros(n, bool)
    med[:60] = True
    interp = np.zeros(n, bool)
    interp[60:85] = True
    cm = ColorMedido(rgb=np.zeros((n, 3), np.uint8), medido=med, interpolado=interp,
                     poses=[], descartadas=[])
    assert cm.cobertura == pytest.approx(0.60)
    assert "60 medido(s)" in cm.resumen()
    assert "25 interpolado(s)" in cm.resumen()
    assert "15 sin color" in cm.resumen()


def test_sin_fotos_utiles_se_devuelve_el_respaldo_intacto(tmp_path):
    """Y sin reventar: un caso cuyas unicas imagenes son radiografias es legitimo."""
    _necesita_extra()
    from gaussian_engine.pose_foto import color_por_vertice
    from PIL import Image

    rx = tmp_path / "rx.jpg"
    Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (64, 1))).save(rx)
    V = np.random.default_rng(0).normal(size=(300, 3))
    F = np.array([[0, 1, 2]])
    etq = np.zeros(300, np.int64)
    respaldo = np.full((300, 3), 77, np.uint8)
    cm = color_por_vertice([rx], V, F, etq, respaldo_rgb=respaldo)
    assert cm.medido.sum() == 0
    assert np.array_equal(cm.rgb, respaldo)
    assert len(cm.descartadas) == 1 and "radiografia" in cm.descartadas[0][1]


def test_una_pose_por_encima_del_tope_se_descarta_entera(monkeypatch, tmp_path):
    """⚠️ **El test que de verdad importa.** Medido sobre un caso real: con una lateral de
    1,15 mm en la mezcla, la aportacion del color pasa de +3 piezas a +0. Por eso el
    criterio es DESCARTAR y no ponderar — una vista torcida no aporta menos, resta."""
    _necesita_extra()
    from gaussian_engine import pose_foto
    from PIL import Image

    foto = tmp_path / "foto.jpg"
    a = np.zeros((64, 64, 3), np.uint8)
    a[..., 0] = 200
    a[..., 1] = 90
    Image.fromarray(a).save(foto)

    mala = pose_foto.PoseFoto(
        ruta=foto, rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 50.0]), focal_px=100.0,
        ancho=64, alto=64, error_px=10.0,
        error_mm=pose_foto.ERROR_MAXIMO_MM + 0.01,
        inliers=9, correspondencias=14, umbral_a=17.0, apoyo=0.99)
    monkeypatch.setattr(pose_foto, "estima_pose", lambda *a, **k: mala)

    V = np.random.default_rng(1).normal(size=(300, 3))
    F = np.array([[0, 1, 2]])
    etq = np.zeros(300, np.int64)
    respaldo = np.full((300, 3), 42, np.uint8)
    cm = pose_foto.color_por_vertice([foto], V, F, etq, respaldo_rgb=respaldo)
    assert cm.medido.sum() == 0, "una pose por encima del tope no puede pintar nada"
    assert np.array_equal(cm.rgb, respaldo)
    assert any("por encima" in r for _, r in cm.descartadas)


def test_una_pose_dentro_del_tope_si_pinta(monkeypatch, tmp_path):
    """El simetrico: el tope tiene que dejar pasar lo bueno, o no seria un tope."""
    _necesita_extra()
    from gaussian_engine import pose_foto
    from PIL import Image

    foto = tmp_path / "foto.jpg"
    a = np.zeros((64, 64, 3), np.uint8)
    a[..., 0] = 200
    a[..., 1] = 90
    Image.fromarray(a).save(foto)
    buena = pose_foto.PoseFoto(
        ruta=foto, rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 50.0]), focal_px=200.0,
        ancho=64, alto=64, error_px=5.0, error_mm=0.5, inliers=9, correspondencias=14,
        umbral_a=17.0, apoyo=0.98)
    monkeypatch.setattr(pose_foto, "estima_pose", lambda *a, **k: buena)
    # Una rejilla plana delante de la camara, con normales hacia ella.
    g = np.linspace(-5, 5, 20)
    xx, yy = np.meshgrid(g, g)
    V = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1)
    F = np.array([[0, 1, 20], [1, 21, 20]])
    etq = np.zeros(len(V), np.int64)
    respaldo = np.full((len(V), 3), 42, np.uint8)
    cm = pose_foto.color_por_vertice([foto], V, F, etq, respaldo_rgb=respaldo)
    assert cm.medido.sum() > 0, "una pose buena tiene que pintar"
    assert not cm.descartadas


def test_sin_etiquetas_la_pose_dice_que_le_falta_el_dato(tmp_path):
    """⚠️ **Un dato que falta no puede salir como un fallo de entrenamiento.**

    Corriendo el pipeline sin `--fdi`, `etiquetas` llegaba aquí como `None` y la primera
    comparación reventaba con «'>' not supported between instances of 'NoneType' and
    'int'» desde dentro de una comprensión. Doscientas líneas más arriba el `except` de la
    etapa lo imprimía como «Error entrenando apariencia» — y el entrenamiento no tenía
    nada que ver: lo que faltaba era la segmentación.
    """
    import numpy as np
    import pytest
    from gaussian_engine.pose_foto import estima_pose

    with pytest.raises(ValueError, match="region_id"):
        estima_pose(tmp_path / "no-se-abre.jpg", np.zeros((4, 3)), None)


def test_el_diagnostico_dice_por_que_no_hay_pose(tmp_path):
    """Un fallo de pose tiene que decir POR QUE fallo, no solo que fallo.

    «No se ha podido resolver una pose» no es auditable: el `diag` existe para que el
    motivo quede escrito y la fase de refinamiento sepa de que candidatos partir.
    """
    _necesita_extra()
    from gaussian_engine.pose_foto import estima_pose

    V = np.random.default_rng(3).normal(size=(400, 3))
    etq = np.zeros(400, np.int64)  # ninguna pieza etiquetada
    d = {}
    assert estima_pose(tmp_path / "no-se-abre.jpg", V, etq, diag=d) is None
    assert "ninguna pieza etiquetada" in d["motivo"]
    assert d["archivo"] == "no-se-abre.jpg"


def test_el_diagnostico_se_rellena_para_las_descartadas(monkeypatch, tmp_path):
    """El JSON de diagnostico cubre TODAS las fotos, no solo las que dan pose."""
    _necesita_extra()
    from gaussian_engine import pose_foto
    from PIL import Image

    foto = tmp_path / "foto.jpg"
    a = np.zeros((64, 64, 3), np.uint8)
    a[..., 0] = 200
    a[..., 1] = 90
    Image.fromarray(a).save(foto)
    mala = pose_foto.PoseFoto(
        ruta=foto, rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 50.0]), focal_px=100.0,
        ancho=64, alto=64, error_px=10.0,
        error_mm=pose_foto.ERROR_MAXIMO_MM + 0.01,
        inliers=9, correspondencias=14, umbral_a=17.0, apoyo=0.99)
    monkeypatch.setattr(pose_foto, "estima_pose", lambda *a, **k: mala)

    V = np.random.default_rng(1).normal(size=(300, 3))
    F = np.array([[0, 1, 2]])
    etq = np.zeros(300, np.int64)
    respaldo = np.full((300, 3), 42, np.uint8)
    diags: dict[str, dict] = {}
    cm = pose_foto.color_por_vertice([foto], V, F, etq, respaldo_rgb=respaldo,
                                     diag_por_foto=diags)
    assert cm.medido.sum() == 0
    assert foto.name in diags
    assert "por encima" in diags[foto.name]["motivo"]
