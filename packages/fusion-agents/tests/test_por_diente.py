"""Registration por diente: que recupere movimiento real y que declare cuándo no puede."""

from __future__ import annotations

import numpy as np
from fusion_agents.por_diente import (
    condicionamiento,
    desplazamientos_relativos,
    registra_dientes,
    transfiere_etiquetas,
)
from scipy.spatial.transform import Rotation


def dientes_sinteticos(
    n_dientes: int = 6, por_diente: int = 1600, semilla: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cúpulas ELIPSOIDALES a lo largo de un arco. `(puntos, etiquetas, normales)`.

    Ni planos ni esferas, y las dos exclusiones importan:

    - un parche **plano** no determina la pose: desliza sobre sí mismo;
    - una **esfera** tampoco, y es menos obvio. Con normales radiales `p × n = 0` en
      todo punto, así que las tres filas de rotación del jacobiano punto-a-plano se
      anulan y la rotación queda libre. Es correcto: girar una esfera sobre su centro
      la deja igual. Un diente no es una esfera, así que la cúpula va escalada por ejes
      distintos y sus normales dejan de ser radiales.
    """
    rng = np.random.default_rng(semilla)
    ejes = np.array([4.0, 2.6, 5.2])
    puntos, etiquetas, normales = [], [], []
    for i in range(n_dientes):
        u = rng.random(por_diente) * 2 * np.pi
        v = np.arccos(1 - rng.random(por_diente))  # media esfera
        dir_ = np.column_stack([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)])
        cupula = dir_ * ejes
        # Normal de un elipsoide: proporcional a (x/a², y/b², z/c²), no al radio.
        n = cupula / ejes**2
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        puntos.append(cupula + np.array([i * 9.0, 0.0, 0.0]))
        etiquetas.append(np.full(por_diente, 31 + i))
        normales.append(n)
    return np.vstack(puntos), np.concatenate(etiquetas), np.vstack(normales)


def test_transfiere_etiquetas_respeta_el_radio() -> None:
    destino = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    etiquetas = np.array([31, 32])
    origen = np.array([[0.1, 0, 0], [10.1, 0, 0], [50.0, 0, 0]])
    salida = transfiere_etiquetas(origen, destino, etiquetas, radio_mm=1.0)
    assert salida.tolist() == [31, 32, 0], "lo que no tiene contrapartida queda sin diente"


def test_recupera_el_movimiento_de_un_solo_diente() -> None:
    """Un diente se mueve y los demás no: la rígida de ese diente lo tiene que recoger."""
    destino, etiquetas, normales = dientes_sinteticos()
    origen = destino.copy()
    movido = etiquetas == 33
    giro = Rotation.from_euler("xyz", [0.0, 0.0, 3.0], degrees=True).as_matrix()
    centro = destino[movido].mean(0)
    origen[movido] = (destino[movido] - centro) @ giro.T + centro + np.array([0.35, 0.0, 0.0])

    fuera = registra_dientes(
        origen, destino, etiquetas, normales,
        rot_global=np.eye(3), trans_global=np.zeros(3),
    )
    por_fdi = {d.fdi: d for d in fuera}
    assert len(por_fdi) == 6

    # El diente movido tiene que declarar traslación clara; los quietos, casi nada.
    assert por_fdi[33].traslacion_mm > 0.2
    quietos = [d.traslacion_mm for f, d in por_fdi.items() if f != 33]
    assert max(quietos) < 0.1, f"un diente quieto no puede moverse: {quietos}"


def test_el_residuo_se_mide_en_la_mitad_retenida() -> None:
    """La mejora tiene que sobrevivir a datos que la transformación no vio."""
    destino, etiquetas, normales = dientes_sinteticos()
    giro = Rotation.from_euler("xyz", [1.5, -1.0, 2.0], degrees=True).as_matrix()
    origen = destino @ giro.T + np.array([0.4, -0.2, 0.15])

    fuera = registra_dientes(
        origen, destino, etiquetas, normales,
        rot_global=np.eye(3), trans_global=np.zeros(3),
    )
    assert fuera, "no se registró ningún diente"
    for d in fuera:
        assert d.rms_diente_mm < d.rms_global_mm, f"{d.fdi} no mejora en retenido"


def test_la_matriz_compuesta_lleva_el_origen_sobre_el_destino() -> None:
    """La 4×4 devuelta incluye la global; usarla sola tiene que bastar."""
    destino, etiquetas, normales = dientes_sinteticos(n_dientes=3)
    giro = Rotation.from_euler("xyz", [10.0, 4.0, -7.0], degrees=True).as_matrix()
    desplaza = np.array([12.0, -3.0, 5.0])
    # Se construye para que `apply(giro.T, desplaza, origen) == destino`:
    #   origen @ (giro.T).T + desplaza = origen @ giro + desplaza
    origen = (destino - desplaza) @ giro.T

    fuera = registra_dientes(
        origen, destino, etiquetas, normales,
        rot_global=giro.T, trans_global=desplaza,
    )
    assert fuera
    for d in fuera:
        sel = etiquetas == d.fdi
        homog = np.hstack([origen[sel], np.ones((sel.sum(), 1))])
        llevado = (d.matriz @ homog.T).T[:, :3]
        assert np.abs(llevado - destino[sel]).max() < 0.5


def test_condicionamiento_separa_lo_determinado_de_lo_que_desliza() -> None:
    """Plano y esfera son libres; un cubo no. Eso tiene que leerse en la salida.

    La esfera está aquí a propósito: es el caso que engaña. Parece la forma «más 3D»
    posible, pero con normales radiales `p × n = 0` y su rotación es tan libre como la
    de un plano. Un cubo, con normales por cara, sí fija los seis grados de libertad.
    """
    rng = np.random.default_rng(0)
    plano = np.column_stack([rng.random(2000) * 10, rng.random(2000) * 10, np.zeros(2000)])
    normales_plano = np.tile([0.0, 0.0, 1.0], (2000, 1))

    esfera = rng.normal(size=(2000, 3))
    esfera /= np.linalg.norm(esfera, axis=1, keepdims=True)

    cara = rng.integers(0, 6, 3000)
    uv = rng.random((3000, 2)) * 2 - 1
    cubo = np.zeros((3000, 3))
    normales_cubo = np.zeros((3000, 3))
    for c in range(6):
        sel, eje, signo = cara == c, c // 2, 1.0 if c % 2 else -1.0
        otros = [k for k in range(3) if k != eje]
        cubo[sel, eje] = signo
        cubo[np.ix_(sel, otros)] = uv[sel]
        normales_cubo[sel, eje] = signo

    bien = condicionamiento(cubo, normales_cubo)
    assert bien < 100, f"un cubo determina la pose, y salió {bien:.0f}"
    assert condicionamiento(plano, normales_plano) > bien * 100
    assert condicionamiento(esfera, esfera) > bien * 100


def test_relativo_recupera_el_diente_movido_y_deja_quietos_los_demas() -> None:
    """El caso que la función existe para resolver, con una traslación conocida."""
    destino, etiquetas, normales = dientes_sinteticos(n_dientes=7)
    origen = destino.copy()
    movido = etiquetas == 34
    origen[movido] = destino[movido] + np.array([0.8, 0.0, 0.0])

    fuera = desplazamientos_relativos(
        origen, destino, etiquetas, normales,
        rot_global=np.eye(3), trans_global=np.zeros(3),
    )
    por_fdi = {d.fdi: d for d in fuera}
    assert len(por_fdi) == 7
    assert por_fdi[34].dientes_referencia == 6, "la referencia excluye el diente medido"
    assert abs(por_fdi[34].traslacion_mm - 0.8) < 0.15
    quietos = {f: d.traslacion_mm for f, d in por_fdi.items() if f != 34}
    assert max(quietos.values()) < 0.1, f"un diente quieto no se mueve: {quietos}"


def test_relativo_no_hereda_el_marco_global_equivocado() -> None:
    """La razón de ser: una referencia global mala NO debe cambiar el desplazamiento.

    En el uso real la global se ajusta sobre el arco entero, encía incluida, y la encía
    cambia de forma entre dos momentos sin que ningún diente se haya movido. Eso desplaza
    la referencia. Aquí se simula pasando una global deliberadamente sesgada: como la
    referencia se **reajusta** con los demás dientes, la cifra tiene que salir igual.
    """
    destino, etiquetas, normales = dientes_sinteticos(n_dientes=7)
    origen = destino.copy()
    origen[etiquetas == 34] = destino[etiquetas == 34] + np.array([0.8, 0.0, 0.0])

    sesgo = Rotation.from_euler("xyz", [0.8, -0.6, 1.2], degrees=True).as_matrix()
    limpia, sucia = (
        desplazamientos_relativos(
            origen, destino, etiquetas, normales,
            rot_global=rot, trans_global=trans,
        )
        for rot, trans in ((np.eye(3), np.zeros(3)), (sesgo, np.array([0.5, -0.4, 0.3])))
    )
    a = {d.fdi: d.traslacion_mm for d in limpia}
    b = {d.fdi: d.traslacion_mm for d in sucia}
    assert a.keys() == b.keys() and a, "las dos pasadas tienen que ver los mismos dientes"
    peor = max(abs(a[f] - b[f]) for f in a)
    assert peor < 0.1, f"la global sesgada movió los desplazamientos {peor:.3f} mm"


def test_relativo_no_deja_que_el_diente_medido_arrastre_su_referencia() -> None:
    """Un movimiento GRANDE de una pieza no puede contagiar a las quietas.

    Es lo que sí pasa midiendo contra el registro global: el diente que se mueve entra en
    el ajuste de la referencia, tira de ella, y reparte su propio movimiento entre todos.
    """
    destino, etiquetas, normales = dientes_sinteticos(n_dientes=6)
    origen = destino.copy()
    origen[etiquetas == 33] = destino[etiquetas == 33] + np.array([0.0, 0.0, 2.0])

    por_fdi = {
        d.fdi: d
        for d in desplazamientos_relativos(
            origen, destino, etiquetas, normales,
            rot_global=np.eye(3), trans_global=np.zeros(3),
        )
    }
    assert abs(por_fdi[33].traslacion_mm - 2.0) < 0.2
    contagiados = {f: d.traslacion_mm for f, d in por_fdi.items() if f != 33}
    assert max(contagiados.values()) < 0.1, f"contagio a los quietos: {contagiados}"


def test_relativo_exige_dientes_suficientes_para_una_referencia() -> None:
    """Con pocas piezas el marco lo fija tan poca cosa que no se devuelve nada."""
    destino, etiquetas, normales = dientes_sinteticos(n_dientes=3)
    fuera = desplazamientos_relativos(
        destino, destino, etiquetas, normales,
        rot_global=np.eye(3), trans_global=np.zeros(3),
    )
    assert fuera == [], "3 dientes no bastan para una referencia leave-one-out"


def test_dientes_pequenos_se_omiten_en_vez_de_inventarse() -> None:
    destino, etiquetas, normales = dientes_sinteticos(n_dientes=2, por_diente=200)
    fuera = registra_dientes(
        destino, destino, etiquetas, normales,
        rot_global=np.eye(3), trans_global=np.zeros(3),
    )
    assert fuera == [], "con menos del mínimo de puntos no se devuelve una rígida"
