"""El refinado de pose por silueta, sobre una arcada SINTETICA con la respuesta conocida.

Se construye una arcada de 8 cajas etiquetadas, se "fotografia" desde una pose conocida
pintando la mascara con el casco convexo de cada corona, y se comprueba que el refinado
recupera la pose cuando la semilla llega perturbada — y que el gate descarta lo que no
pasa, igual que con una pose estimada.

⚠️ **Lo que estos tests guardan es el CONTRATO**: recuperacion desde perturbacion, gate
intacto (error, apoyo, inliers) y que el sentido de espejo se decide por el gate y no por
un apoyo redondeado.
"""

from __future__ import annotations

import numpy as np
import pytest


def _necesita_extra():
    pytest.importorskip("cv2", reason="necesita el extra `appearance`")
    pytest.importorskip("skimage", reason="necesita el extra `appearance`")


CODS = [11, 12, 13, 14, 15, 16, 17, 18]  # de izquierda a derecha


def _arcada_sintetica():
    """Vertices etiquetados de 8 coronas (cajas) + encia, y su mascara proyectada.

    Devuelve `(V, etq, arco, xy, p3_todos, P3, corona, K, rvec_gt, tvec_gt)`.

    ⚠️ **Las coronas son FINAS (4 mm de profundidad), y es a proposito.** La metrica
    `error_mm` compara el centroide 3D del tercio oclusal con la media de las
    proyecciones 2D, y la proyeccion no es afin: la mitad cercana de una corona pesa
    mas en pixeles. Con 30 mm de profundidad ese sesgo es tal que hasta la pose
    VERDADERA puntua 2 mm y el gate la descartaria; con 4 mm queda ~0,4 mm, que es el
    orden del sesgo que el gate real ya absorbe (medido: la foto buena puntua 0,845).
    """
    rng = np.random.default_rng(7)
    trozos, etiquetas = [], []
    for i, cod in enumerate(CODS):
        c = np.array([(i - 3.5) * 22.0, 0.0, 0.0])
        # ⚠️ Anchuras DISTINTAS por pieza (4..11 de media caja), como una denticion real:
        # con cajas identicas la arcada es simetrica y el sentido de espejo encaja igual
        # en los dos — la ambiguedad que el test quiere medir no existiria.
        p = rng.uniform(-1, 1, size=(600, 3)) * np.array([4.0 + i, 12.0, 2.0]) + c
        trozos.append(p)
        etiquetas.append(np.full(len(p), cod))
    # La encia existe solo para definir el eje oclusal (diente - encia).
    trozos.append(rng.uniform(-1, 1, size=(800, 3)) * np.array([90.0, 40.0, 1.0])
                  + np.array([0.0, 0.0, -35.0]))
    etiquetas.append(np.zeros(800))
    V = np.vstack(trozos)
    etq = np.concatenate(etiquetas)

    ancho, alto, focal = 900, 700, 900.0
    K = np.array([[focal, 0, ancho / 2], [0, focal, alto / 2], [0, 0, 1]])
    rvec_gt = np.zeros(3)
    tvec_gt = np.array([0.0, -10.0, 220.0])

    import cv2

    R, _ = cv2.Rodrigues(rvec_gt)
    Vc = V @ R.T + tvec_gt
    uv = (K @ Vc.T / Vc[:, 2]).T[:, :2]
    arco = np.zeros((alto, ancho), np.uint8)
    xy = []
    for cod in CODS:
        p = uv[etq == cod].astype(np.float32)
        hull = cv2.convexHull(p).reshape(-1, 2)
        cv2.fillConvexPoly(arco, hull.astype(np.int32), 1)
        xy.append(p.mean(0))
    xy = np.asarray(xy, float)

    from gaussian_engine.pose_foto import centros_oclusales, corona_oclusal
    from gaussian_engine.pose_refina import superficie_oclusal

    return (V, etq, arco, xy, centros_oclusales(V, etq, CODS),
            superficie_oclusal(V, etq, CODS), corona_oclusal(V, etq, CODS),
            K, rvec_gt, tvec_gt)


def _semilla(ventana=(0, 8), sentido=1, rvec=None, tvec=None, idx=None):
    """Una semilla con la forma del candidato del diagnostico de `estima_pose`."""
    return {
        "focal": 900.0, "ventana": list(ventana), "sentido": sentido,
        "inliers_idx": list(range(8)) if idx is None else idx,
        "rvec": [float(x) for x in np.zeros(3) if rvec is None] if rvec is None
        else [float(x) for x in np.asarray(rvec).ravel()],
        "tvec": [0.0, -10.0, 220.0] if tvec is None
        else [float(x) for x in np.asarray(tvec).ravel()],
    }


def test_el_refinado_recupera_la_pose_desde_una_perturbacion(tmp_path):
    """La semilla llega 3° girada y 5 mm desplazada; el refinado la devuelve al sitio.

    La semilla perturbada puntua ~6 mm (la puerta la descarta); la refinada tiene que
    pasar el gate —que es lo que la fase mide— y la rotacion tiene que recuperarse: sin
    el anclaje de la primera ronda la silueta no ve girar una arcada de blobs convexos.
    """
    _necesita_extra()
    from gaussian_engine.pose_refina import refina_silueta

    V, etq, arco, xy, p3, P3, corona, K, rvec_gt, tvec_gt = _arcada_sintetica()
    import cv2

    # Perturbacion: 3 grados alrededor de y + 5 mm de translacion.
    r_pert = cv2.Rodrigues(np.array([0.0, np.deg2rad(3.0), 0.0]))[0]
    t_pert = tvec_gt + np.array([5.0, 0.0, -5.0])
    rv_pert, _ = cv2.Rodrigues(r_pert)

    semilla = _semilla(rvec=rv_pert.ravel(), tvec=t_pert)
    pr, _ = cv2.projectPoints(np.ascontiguousarray(p3), rv_pert.ravel(), t_pert, K, None)
    e_semilla = np.linalg.norm(pr.reshape(-1, 2) - xy, axis=1).mean()

    pose = refina_silueta(arco, xy, p3, P3, corona, 17.0, semilla,
                          tmp_path / "sintetica.jpg")
    assert pose is not None, "el refinado tiene que recuperar una semilla perturbada"
    assert pose.error_mm < 0.9, f"la refinada tiene que pasar el gate: {pose.error_mm:.3f}"
    assert pose.error_px < e_semilla * 0.3, (
        f"mejora esperada de {e_semilla:.1f} px a menos de un tercio: {pose.error_px:.2f}")
    assert pose.apoyo >= 0.80
    assert pose.inliers == 8 and pose.correspondencias == 8


def test_el_sentido_de_espejo_lo_decide_el_gate_no_un_apoyo_redondeado(tmp_path):
    """La semilla con el sentido invertido empareja mal y el gate la descarta."""
    _necesita_extra()
    from gaussian_engine.pose_refina import refina_silueta

    V, etq, arco, xy, p3, P3, corona, _K, _rv, _tv = _arcada_sintetica()
    invertida = refina_silueta(arco, xy, p3, P3, corona, 17.0,
                               _semilla(sentido=-1), tmp_path / "sintetica.jpg")
    assert invertida is None, "correspondencias cruzadas no pueden pasar el gate"


def test_el_gate_sigue_siendo_el_mismo_para_una_refinada(tmp_path):
    """Una refinada con 5 inliers se descarta IGUAL que una estimada: el criterio no
    cambia porque la pose sea bonita."""
    _necesita_extra()
    from gaussian_engine.pose_refina import refina_silueta

    V, etq, arco, xy, p3, P3, corona, _K, _rv, _tv = _arcada_sintetica()
    cinco = refina_silueta(arco, xy, p3, P3, corona, 17.0,
                           _semilla(idx=list(range(5))), tmp_path / "sintetica.jpg")
    assert cinco is None


def test_el_driver_elige_la_semilla_que_gana(monkeypatch, tmp_path):
    """`refina_desde_candidatos` con una semilla buena y una invertida devuelve la buena."""
    _necesita_extra()
    from gaussian_engine import pose_refina

    V, etq, arco, xy, p3, P3, corona, K, rvec_gt, tvec_gt = _arcada_sintetica()
    import cv2

    r_pert = cv2.Rodrigues(np.array([0.0, np.deg2rad(3.0), 0.0]))[0]
    rv_pert, _ = cv2.Rodrigues(r_pert)
    t_pert = tvec_gt + np.array([5.0, 0.0, -5.0])

    monkeypatch.setattr(pose_refina, "_blobs_para_pose",
                        lambda ruta, cuantas: (arco, xy, 17.0, 0))
    candidatos = [
        _semilla(rvec=rv_pert.ravel(), tvec=t_pert),
        _semilla(sentido=-1, rvec=rv_pert.ravel(), tvec=t_pert),
    ]
    mejor = pose_refina.refina_desde_candidatos(tmp_path / "sintetica.jpg",
                                                V, etq, CODS, candidatos)
    assert mejor is not None
    assert mejor.error_mm < 0.9
    assert mejor.apoyo >= 0.80
