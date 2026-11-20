"""Registro anatómico por banda de escáner (`fusion_agents.anatomico`), sobre el fantasma.

Todo sintético (`fantasma_arcada`): corre en CI sin datos clínicos. La verdad es exacta
porque el escaneo se fabrica en el mundo del CBCT y se lleva a su propio marco con un
movimiento conocido; registrar tiene que deshacerlo.

Cada registro completo cuesta del orden de 15–25 s de CPU (90 arranques + ajuste fino +
incertidumbre), así que se hacen pocos y se reutilizan: un registro de referencia por
módulo, y solo los tests que necesitan otra entrada (otro movimiento, otro teselado, otra
serie) registran de nuevo, y sin la incertidumbre cuando no la miran.

Lo que se comprueba es lo que P0 del protocolo (`docs/research/registro-anatomico-
protocolo.md` §5) exige al software: sin azar, byte a byte entre procesos, equivariante
ante el movimiento y el orden del fichero, estable ante el teselado; y que las vueltas, la
cobertura, el espejo y la escala se declaran en vez de pasar en silencio.
"""

from __future__ import annotations

import contextlib
import math
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from core_schemas import Modality, Provenance, TwinSnapshot
from export_agents.anatomia import anatomical_frame
from fantasma_arcada import campo_cbct, escaneo_superior, mueve, rotacion
from fusion_agents import GeometricFusionAgent, anatomico
from fusion_agents.anatomico import RegistroAnatomico, marco_cbct, motivos, prepara_anatomico
from fusion_agents.registration import quaternion_to_matrix

AQUI = Path(__file__).resolve().parent

# El escáner en su propio marco: fichero = MOVIMIENTO(mundo). Registrar es deshacerlo.
ROT_FICHERO = rotacion([0.3, 1.0, 0.2], 97.0)
TRAS_FICHERO = np.array([40.0, -12.0, 80.0])
PASO = 0.4


# --------------------------------------------------------------------------- #
# Sin azar
# --------------------------------------------------------------------------- #
_ALEATORIAS = ("default_rng", "seed", "choice", "rand", "random", "permutation")


@contextlib.contextmanager
def sin_azar() -> Iterator[list[str]]:
    """Mientras dura, cualquier generador aleatorio de NumPy lanza y queda anotado.

    Se anota además de lanzar por si algún `try/except` del código se tragara la
    excepción: el test mira la lista, no solo que no haya volado nada.
    """
    llamadas: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        for nombre in _ALEATORIAS:
            def prohibida(*_a: Any, _n: str = nombre, **_k: Any) -> None:
                llamadas.append(_n)
                raise AssertionError(f"np.random.{_n} llamado durante el registro")

            mp.setattr(np.random, nombre, prohibida)
        yield llamadas


def registra(
    V: np.ndarray, F: np.ndarray, etq: np.ndarray, campo: dict, hu: np.ndarray, *,
    incertidumbre: bool = False, arranques: str | None = None,
) -> tuple[np.ndarray, np.ndarray, RegistroAnatomico, dict, list[str]]:
    """`prepara_anatomico` como lo llama el orquestador, sin azar.

    `incertidumbre=False` la sustituye por un diccionario vacío —la mitad del coste— en
    los tests que solo miran la pose. `arranques` deja solo los arranques cuyo nombre
    empieza así, para los tests que miran un diagnóstico y no la búsqueda.
    """
    with pytest.MonkeyPatch.context() as mp:
        if not incertidumbre:
            mp.setattr(RegistroAnatomico, "incertidumbre", lambda self: {})
        if arranques is not None:
            todos = RegistroAnatomico.arranques

            def pocos(self: RegistroAnatomico) -> list:
                elegidos = [a for a in todos(self) if a[0].startswith(arranques)]
                assert elegidos, f"ningún arranque se llama {arranques!r}"
                return elegidos

            mp.setattr(RegistroAnatomico, "arranques", pocos)
        with sin_azar() as llamadas:
            ejes, motivo = anatomical_frame(V, etq)
            assert ejes is not None, motivo
            origen, destino, registrador, informe = prepara_anatomico(
                campo, hu, V, F, etq, ejes
            )
    return origen, destino, registrador, informe, llamadas


def pose(registrador: RegistroAnatomico, origen: np.ndarray, destino: np.ndarray) -> tuple:
    """`(R, t)` del registrador; la búsqueda está memorizada, así que no se repite."""
    r = registrador(origen, destino)
    return quaternion_to_matrix(r.rotation), np.asarray(r.translation), r


def subdivide(
    V: np.ndarray, F: np.ndarray, etq: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1→4 por punto medio. Los vértices originales conservan su índice; un punto medio
    hereda la etiqueta si sus dos extremos coinciden, y si no es 0 (protocolo P0)."""
    aristas = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    unicas, inversa = np.unique(aristas, axis=0, return_inverse=True)
    inversa = np.asarray(inversa).ravel()
    n, m = len(V), len(F)
    medio = n + inversa
    ab, bc, ca = medio[:m], medio[m : 2 * m], medio[2 * m :]
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    F4 = np.concatenate([
        np.c_[a, ab, ca], np.c_[ab, b, bc], np.c_[ca, bc, c], np.c_[ab, bc, ca],
    ])
    V4 = np.vstack([V, 0.5 * (V[unicas[:, 0]] + V[unicas[:, 1]])])
    e = etq[unicas]
    etq4 = np.r_[etq, np.where(e[:, 0] == e[:, 1], e[:, 0], 0)]
    return V4, F4, etq4


# --------------------------------------------------------------------------- #
# Fixtures del módulo
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mundo() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """El escaneo superior en el mundo del CBCT (la verdad)."""
    return escaneo_superior()


@pytest.fixture(scope="module")
def fichero(mundo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Vw, F, etq = mundo
    return mueve(Vw, ROT_FICHERO, TRAS_FICHERO), F, etq


@pytest.fixture(scope="module")
def cbct() -> tuple[dict, np.ndarray]:
    return campo_cbct(paso=PASO)


@pytest.fixture(scope="module")
def referencia(fichero, cbct) -> dict:
    """EL registro del módulo: completo, con incertidumbre, y con el azar prohibido."""
    V, F, etq = fichero
    origen, destino, registrador, informe, llamadas = registra(
        V, F, etq, *cbct, incertidumbre=True
    )
    R, t, resultado = pose(registrador, origen, destino)
    return {
        "origen": origen, "destino": destino, "registrador": registrador,
        "informe": informe, "llamadas": llamadas, "R": R, "t": t, "resultado": resultado,
    }


def _en_cbct(V: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return V @ R.T + t


# --------------------------------------------------------------------------- #
# 1 · Recupera la pose conocida
# --------------------------------------------------------------------------- #
def test_recupera_la_pose_conocida_en_planta_y_deja_el_error_en_altura(
    referencia, fichero, mundo
) -> None:
    """En planta casi exacto; lo que queda de error es vertical, la dirección peor medida.

    Hoy son ~0,17 mm hacia la raíz con este teselado. ⚠️ Medido al escribir este test: no
    es el medio paso del contorno (que `contornos` ya corrige), sino el recorte del ajuste
    por distancia al VÉRTICE más próximo; con la malla subdividida baja a 0,03 mm y
    recortando por residuo punto a plano a 0,015 (ver el test de teselado). Por eso la
    cota vertical es holgada y sin signo: lo que se exige es que el error no salga de la
    altura.
    """
    V, _, etq = fichero
    Vw = mundo[0]
    corona = etq > 0
    error = _en_cbct(V[corona], referencia["R"], referencia["t"]) - Vw[corona]
    norma = np.linalg.norm(error, axis=1)
    assert referencia["resultado"].veto is None
    assert float(np.median(norma)) < 0.3
    media = error.mean(axis=0)
    assert np.linalg.norm(media[:2]) < 0.03          # en planta, nada
    assert abs(media[2]) < 0.3                       # en altura, acotado
    assert float(np.median(np.linalg.norm(error[:, :2], axis=1))) < 0.1
    # La rotación es la del movimiento, a menos de medio grado.
    giro = referencia["R"] @ ROT_FICHERO
    assert anatomico.angulo_rotacion_grados(giro) < 0.5


def test_recupera_la_pose_en_una_serie_dispersa(fichero, mundo) -> None:
    """Cortes cada 0,8 mm y un hueco de 2 mm en mitad de la banda: más laxo, mismo método."""
    campo, hu = campo_cbct(paso=PASO, paso_z=0.8, quita=(2.0, 4.0))
    V, F, etq = fichero
    origen, destino, registrador, informe, _ = registra(V, F, etq, campo, hu)
    assert informe["contour"]["sparse"]
    R, t, r = pose(registrador, origen, destino)
    assert r.veto is None
    corona = etq > 0
    error = _en_cbct(V[corona], R, t) - mundo[0][corona]
    assert float(np.median(np.linalg.norm(error, axis=1))) < 0.5
    assert float(np.median(np.linalg.norm(error[:, :2], axis=1))) < 0.15


def test_raices_tan_anchas_como_el_cuello_no_arrastran_la_banda(fichero, mundo) -> None:
    """Fija lo que HOY hace el registro con raíces realistas (`raices="realistas"`).

    En el caso A (`docs/research/registro-anatomico.md` §7) el ajuste se desliza hacia las
    raíces desde la pose del registro por cortes, que no es verdad. La hipótesis era que el
    fantasma no lo mostraba por sus raíces cónicas finas. Medido al escribir este test, con
    raíces del ancho del cuello de su corona y 13 mm de largo: **no lo reproduce**. Desde la
    pose verdadera el ajuste fino se queda a 0,08 mm (lo mismo que con conos), la búsqueda
    elige la cuenca correcta a 0,08 mm de mediana y sin veto, y C en la verdad es 0,97.
    Si un cambio del método rompe esto, ha empeorado el caso que sí tiene verdad.
    """
    campo, hu = campo_cbct(paso=PASO, raices="realistas")
    V, F, etq = fichero
    origen, destino, registrador, _, _ = registra(V, F, etq, campo, hu)
    R, t, r = pose(registrador, origen, destino)
    assert r.veto is None
    corona = etq > 0
    error = _en_cbct(V[corona], R, t) - mundo[0][corona]
    assert float(np.median(np.linalg.norm(error, axis=1))) < 0.3

    # Desde la verdad, el ajuste fino no se va hacia la raíz.
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    ra, ta, _, convergido = registrador.afina(*verdad, destino)
    assert convergido
    assert registrador.rms_desplazamiento((ra, ta), verdad) < 0.2
    puntuacion = registrador.puntua_secciones(*verdad, anatomico._por_corte(destino, None))
    assert puntuacion["C"] >= anatomico.COBERTURA_MIN


def _registrador_sin_busqueda(
    campo: dict, hu: np.ndarray, V: np.ndarray, F: np.ndarray, etq: np.ndarray
) -> tuple[RegistroAnatomico, np.ndarray]:
    """El registrador y su contorno a 1400, sin búsqueda ni incertidumbre (segundos)."""
    ejes, motivo = anatomical_frame(V, etq)
    assert ejes is not None, motivo
    escaner = anatomico.Escaner.construye(V, F, etq, ejes)
    centros = campo["centers"] + campo["origin"]
    base, plano, _ = marco_cbct(centros, hu, escaner.arcada)
    u = float(anatomico.UMBRAL_CONTORNO)
    contornos = {u: anatomico.contornos_losa(centros, hu, u, plano, escaner.arcada)}
    registrador = RegistroAnatomico(
        escaner=escaner, base_cbct=base, plano=plano, contornos=contornos
    )
    return registrador, contornos[u][0]


@pytest.mark.parametrize("mm", [2.0, 4.0, 8.0])
def test_deslizar_hacia_la_raiz_cuesta_en_s_aunque_la_raiz_sea_tan_ancha_como_la_banda(
    fichero, mundo, mm
) -> None:
    """Lo que una pose deslizada hacia la raíz deja sin pareja es la parte OCLUSAL de la
    corona del escáner, metida dentro de la raíz. S lo cobra porque se mide sobre la corona
    entera, no solo sobre la banda.

    Medido al escribirlo, con raíces realistas: con la banda y sin desfase, 8 mm hacia la
    raíz subían S de 0,148 a 0,295 mm y la cobertura seguía en 1,0; con la corona y el
    desfase del isonivel, de 0,115 a 0,362, y 2 mm a 0,205. El desfase absorbe parte de un
    deslizamiento pequeño (por el cono de la corona, subir se parece a encoger), así que la
    cota es el margen con que se declara una vuelta: un deslizamiento tiene que perder por
    más de lo que se considera «casi igual».
    """
    campo, hu = campo_cbct(paso=PASO, raices="realistas")
    registrador, destino = _registrador_sin_busqueda(campo, hu, *fichero)
    por_corte = anatomico._por_corte(destino, registrador._normales_de(destino))
    R, t = ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO
    raiz = registrador.base_cbct[2]  # arcada superior: la raíz está hacia +S
    buena = registrador.puntua_secciones(R, t, por_corte)
    deslizada = registrador.puntua_secciones(R, t + mm * raiz, por_corte)
    assert buena["C"] >= anatomico.COBERTURA_MIN
    assert deslizada["S"] > buena["S"] + anatomico.MARGEN_VUELTA_S_MM


@pytest.mark.parametrize("desde_mm", [0.0, 8.0], ids=["verdad", "8mm-raiz"])
def test_con_el_borde_desenfocado_s_prefiere_la_verdad_y_el_ajuste_se_desliza(
    fichero, mundo, desde_mm, monkeypatch
) -> None:
    """El mecanismo del caso A, reproducido en el fantasma: con la PSF el borde a 1400 cae
    DENTRO de la superficie, más en la raíz (dentina) y en las cúspides que en las paredes,
    y un ajuste que lo toma por la superficie sube la arcada por el cono de la corona.

    Fija lo que hay, no lo que se querría. Medido al escribirlo, σ = 0,4 mm, raíces
    realistas: el ajuste por defecto se va 0,70 mm hacia la raíz desde la verdad, y 19 mm
    desde 8 mm. S con el desfase del isonivel puntúa la verdad mejor que donde acaba (sin
    el desfase, al revés: 0,156 frente a 0,193). Con el desfase también en el ajuste
    (`DESFASE_EN_AJUSTE`, apagado: rompe P0-teselado) la pose queda a 0,24 y 0,33 mm.
    """
    campo, hu = campo_cbct(paso=PASO, raices="realistas", desenfoque_mm=0.4)
    V, F, etq = fichero
    registrador, destino = _registrador_sin_busqueda(campo, hu, V, F, etq)
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    desde = verdad[1] + desde_mm * registrador.base_cbct[2]
    corona = etq > 0

    def error(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return _en_cbct(V[corona], R, t) - mundo[0][corona]

    R0, t0, _, _ = registrador.afina(verdad[0], desde, destino)
    assert float(np.median(error(R0, t0)[:, 2])) > 0.5  # hacia la raíz
    por_corte = anatomico._por_corte(destino, registrador._normales_de(destino))
    assert (registrador.puntua_secciones(*verdad, por_corte)["S"]
            < registrador.puntua_secciones(R0, t0, por_corte)["S"])

    monkeypatch.setattr(anatomico, "DESFASE_EN_AJUSTE", True)
    R, t, _, _ = registrador.afina(verdad[0], desde, destino)
    e = error(R, t)
    assert float(np.median(np.linalg.norm(e, axis=1))) < 0.4
    assert abs(float(np.median(e[:, 2]))) < 0.4


def test_el_sentido_inverso_del_ajuste_sigue_disponible_y_vuelve_sin_desenfoque(
    fichero, mundo, monkeypatch
) -> None:
    """`BILATERAL` está apagado (ver la constante); encendido, sin desenfoque, vuelve
    desde 8 mm hacia la raíz. Medido al escribirlo: 0,07 mm de mediana sin desfase."""
    monkeypatch.setattr(anatomico, "BILATERAL", True)
    campo, hu = campo_cbct(paso=PASO, raices="realistas")
    V, F, etq = fichero
    registrador, destino = _registrador_sin_busqueda(campo, hu, V, F, etq)
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    R, t, _, convergido = registrador.afina(
        verdad[0], verdad[1] + 8.0 * registrador.base_cbct[2], destino
    )
    assert convergido
    corona = etq > 0
    error = np.linalg.norm(_en_cbct(V[corona], R, t) - mundo[0][corona], axis=1)
    assert float(np.median(error)) < 0.15


def _perfil(densidad: Any) -> dict:
    z = np.round(np.arange(-10.0, 25.0, 0.4), 3)
    return anatomico._perfil_z(np.repeat(z, np.round(densidad(z)).astype(int)))


def _campana(media: float, sigma: float) -> Any:
    return lambda z: np.exp(-0.5 * ((z - media) / sigma) ** 2)


def test_el_valle_solo_es_ancla_si_cae_entre_las_dos_coronas() -> None:
    """Dos coronas en oclusión dejan el valle cerca del punto medio entre sus picos; un
    valle pegado a uno de ellos es un hombro del perfil (dentina o hueso denso por encima
    de las coronas), no un contacto, y no entra como ancla."""
    dos_arcadas = _perfil(lambda z: 100 * _campana(0, 2)(z) + 100 * _campana(12, 2)(z))
    assert dos_arcadas["bimodal"]
    assert dos_arcadas["valley_offcentre"] <= anatomico.VALLE_CENTRAL

    hombro = _perfil(
        lambda z: 100 * _campana(0, 2)(z) + 30 * ((z > 0) & (z < 10.5))
        + 15 * ((z >= 10.5) & (z < 12)) + 60 * _campana(13.6, 0.7)(z)
    )
    assert hombro["valley_offcentre"] > anatomico.VALLE_CENTRAL
    assert not hombro["bimodal"]
    assert "valley_rejected" in hombro


# --------------------------------------------------------------------------- #
# 2 · Sin azar, y 3 · byte a byte entre procesos
# --------------------------------------------------------------------------- #
def test_ni_el_registro_ni_la_incertidumbre_tocan_el_azar(referencia) -> None:
    """La referencia se calculó con todos los generadores de NumPy lanzando."""
    assert referencia["llamadas"] == []
    assert referencia["informe"]["uncertainty"]  # la incertidumbre corrió de verdad
    with sin_azar() as llamadas, pytest.raises(AssertionError):
        np.random.default_rng(0)
    assert llamadas == ["default_rng"]  # y la trampa funciona


_PROCESO = textwrap.dedent(
    """
    import sys
    import numpy as np
    np.random.seed(int(sys.argv[1]))
    from export_agents.anatomia import anatomical_frame
    from fusion_agents.anatomico import RegistroAnatomico, prepara_anatomico
    from fantasma_arcada import campo_cbct, escaneo_superior, mueve, rotacion

    RegistroAnatomico.incertidumbre = lambda self: {}
    Vw, F, etq = escaneo_superior()
    V = mueve(Vw, rotacion(%r, %r), np.array(%r))
    campo, hu = campo_cbct(paso=%r)
    ejes, _ = anatomical_frame(V, etq)
    origen, destino, reg, _ = prepara_anatomico(campo, hu, V, F, etq, ejes)
    r = reg(origen, destino)
    print(np.asarray(r.rotation, dtype=np.float64).tobytes().hex(),
          np.asarray(r.translation, dtype=np.float64).tobytes().hex())
    """
) % ([0.3, 1.0, 0.2], 97.0, TRAS_FICHERO.tolist(), PASO)


def test_otro_proceso_con_otras_semillas_da_los_mismos_bytes(referencia) -> None:
    """P0-semilla: lo antiguo daba la vuelta en ~30 % de semillas; esto no puede cambiar
    ni un bit con `PYTHONHASHSEED` ni con `np.random.seed`.

    El segundo proceso es el de pytest: la referencia del módulo, sin semilla fijada.
    """
    np.testing.assert_array_equal(rotacion([0.3, 1.0, 0.2], 97.0), ROT_FICHERO)
    semilla_hash = "1" if os.environ.get("PYTHONHASHSEED") == "4242" else "4242"
    env = {**os.environ, "PYTHONHASHSEED": semilla_hash,
           "PYTHONPATH": os.pathsep.join([str(AQUI), os.environ.get("PYTHONPATH", "")])}
    hecho = subprocess.run(
        [sys.executable, "-c", _PROCESO, "987654"], env=env, capture_output=True,
        text=True, timeout=900,
    )
    assert hecho.returncode == 0, hecho.stderr
    r = referencia["resultado"]
    aqui = (np.asarray(r.rotation, dtype=np.float64).tobytes().hex() + " "
            + np.asarray(r.translation, dtype=np.float64).tobytes().hex())
    assert hecho.stdout.strip() == aqui


# --------------------------------------------------------------------------- #
# 4 · Equivariancia ante movimiento y orden, 5 · teselado
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("eje", "grados", "traslacion", "permuta"),
    [
        ([1.0, 0.0, 0.0], 180.0, [100.0, -100.0, 100.0], True),
        ([0.0, 0.0, 1.0], 90.0, [-100.0, 100.0, -100.0], False),
    ],
    ids=["180x+permutado", "90z"],
)
def test_mover_o_reordenar_el_fichero_no_mueve_las_coronas(
    referencia, fichero, cbct, eje, grados, traslacion, permuta
) -> None:
    """P0-movimiento y P0-orden: las coronas en mm del DICOM no dependen del fichero.

    Las etiquetas viajan con sus vértices. La permutación reordena vértices y caras y rota
    el orden dentro de cada cara (sin cambiar su sentido).
    """
    V, F, etq = fichero
    W = mueve(V, rotacion(eje, grados), np.asarray(traslacion))
    F2, etq2, orden = F, etq, np.arange(len(V))
    if permuta:
        rng = np.random.default_rng(2001)
        orden = rng.permutation(len(V))
        inversa = np.empty_like(orden)
        inversa[orden] = np.arange(len(V))
        W, etq2 = W[orden], etq[orden]
        F2 = inversa[F][rng.permutation(len(F))]
        giro = rng.integers(0, 3, len(F2))
        F2 = np.stack([np.roll(f, k) for f, k in zip(F2, giro, strict=True)])
    origen, destino, registrador, _, _ = registra(W, F2, etq2, *cbct)
    R, t, r = pose(registrador, origen, destino)
    assert r.veto is None

    corona = etq > 0
    ref = _en_cbct(V[corona], referencia["R"], referencia["t"])
    ahora = np.empty_like(V)
    ahora[orden] = _en_cbct(W, R, t)  # de vuelta al orden original
    d = np.linalg.norm(ahora[corona] - ref, axis=1)
    assert float(d.max()) < 1e-6


def test_subdividir_la_malla_no_mueve_las_coronas(referencia, fichero, cbct) -> None:
    """P0-teselado: el mismo escaneo con cuatro veces más triángulos.

    Un solo arranque (el anatómico) para no pagar 90 sobre una malla cuatro veces mayor:
    lo que depende del teselado es el ajuste, no la búsqueda.
    """
    V, F, etq = fichero
    V4, F4, etq4 = subdivide(V, F, etq)
    origen, destino, registrador, _, _ = registra(
        V4, F4, etq4, *cbct, arranques="rejilla(+0,+0,+0)"
    )
    R, t, r = pose(registrador, origen, destino)
    assert r.veto is None
    corona = etq > 0
    d = np.linalg.norm(
        _en_cbct(V[corona], R, t) - _en_cbct(V[corona], referencia["R"], referencia["t"]),
        axis=1,
    )
    assert float(np.median(d)) <= 0.05
    assert float(np.percentile(d, 95)) <= 0.10


# --------------------------------------------------------------------------- #
# 6 · Vueltas
# --------------------------------------------------------------------------- #
def test_las_hipotesis_de_vuelta_se_ajustan_y_nunca_ganan(referencia) -> None:
    busqueda = referencia["registrador"].cache["ultima"]
    elegida = busqueda["chosen"]
    assert elegida["check"]["eligible"]
    assert not elegida["flip_start"]
    assert elegida["check"]["angle_to_start_deg"] < anatomico.ANGULO_ANATOMICO_MAX
    # Las vueltas se ajustaron —hay cuencas a ~180° del arranque— y ninguna es elegible.
    vueltas = [
        c for c in busqueda["basins"]
        if c["check"]["angle_to_start_deg"] > anatomico.ANGULO_VUELTA
    ]
    assert vueltas
    assert not any(c["check"]["eligible"] for c in vueltas)
    assert all(not c["check"]["eligible"] for c in busqueda["basins"] if c["flip_start"])
    assert referencia["informe"]["search"]["veto"] is None


def test_el_margen_de_vuelta_llega_al_gate_cuando_los_datos_no_la_excluyen(referencia) -> None:
    """En el fantasma —simétrico de izquierda a derecha— una vuelta encaja casi igual.

    Lo que se exige es coherencia: si el margen medido no la excluye, el gate lo dice.
    """
    informe = referencia["informe"]
    margen = informe["search"]["flip_margin"]
    assert margen is not None and margen["angle_deg"] > anatomico.ANGULO_VUELTA
    no_excluida = (
        margen["S_difference_mm"] <= anatomico.MARGEN_VUELTA_S_MM
        or margen["C_difference"] < anatomico.MARGEN_VUELTA_C
    )
    texto = " ".join(motivos(informe))
    assert ("los datos no la excluyen" in texto) == no_excluida
    assert "no un TRE" in texto


# --------------------------------------------------------------------------- #
# 7 · Cabeza inclinada
# --------------------------------------------------------------------------- #
INCLINACION = rotacion([1.0, 0.0, 0.0], 10.0)


@pytest.fixture(scope="module")
def cbct_inclinado() -> tuple[dict, np.ndarray]:
    return campo_cbct(paso=PASO, inclinacion=INCLINACION)


def test_el_marco_del_cbct_no_inventa_inclinacion(cbct_inclinado) -> None:
    """`marco_cbct` no estima la inclinación: da los ejes del DICOM tal cual. Un plano por
    PCA al esmalte leía entre 0,2° y 40° según la ventana para 10° reales."""
    campo, hu = cbct_inclinado
    base, plano, marco = marco_cbct(campo["centers"] + campo["origin"], hu)
    np.testing.assert_array_equal(base, anatomico.EJES_LPS)
    np.testing.assert_array_equal(plano.normal, anatomico.EJES_LPS[2])
    assert marco["anchors_mm"][0] == 0.0


def test_con_la_boca_inclinada_se_registra_y_se_mide_la_inclinacion(
    fichero, mundo, cbct_inclinado
) -> None:
    """La rejilla de arranques, la banda por diente y la losa ancha absorben 10° de
    cabeceo: la pose se recupera, y la inclinación la mide el escáner registrado."""
    campo, hu = cbct_inclinado
    V, F, etq = fichero
    origen, destino, registrador, informe, _ = registra(V, F, etq, campo, hu)
    R, t, r = pose(registrador, origen, destino)
    assert r.veto is None
    corona = etq > 0
    verdad = mueve(mundo[0], INCLINACION, np.zeros(3))[corona]
    error = np.linalg.norm(_en_cbct(V[corona], R, t) - verdad, axis=1)
    assert float(np.median(error)) < 0.3
    # Con la losa horizontal, un extremo de la arcada inclinada aporta menos contorno: la
    # rotación sale a 0,57°, no a las centésimas de la boca sin inclinar.
    assert anatomico.angulo_rotacion_grados(R @ ROT_FICHERO @ INCLINACION.T) < 1.0
    # La inclinación que se declara es la del eje oclusal que sale de las etiquetas FDI, y
    # ese eje no es exactamente el vertical del fantasma: va 0,99° hacia anterior. Así que
    # la verdad no son 10°, sino lo que inclina ESE eje; la tolerancia es la de la rotación.
    assert informe["tilt_from_registration_deg"] == pytest.approx(
        _inclinacion_del_eje_fdi(mundo, INCLINACION), abs=1.0
    )


def _inclinacion_del_eje_fdi(mundo, inclinacion: np.ndarray) -> float:
    Vw, _, etq = mundo
    ejes, motivo = anatomical_frame(Vw, etq)
    assert ejes is not None, motivo
    return anatomico.angulo_grados(inclinacion @ ejes.superior, anatomico.EJES_LPS[2])


def test_sin_inclinar_la_inclinacion_registrada_es_la_del_eje_fdi(mundo, referencia) -> None:
    assert referencia["informe"]["tilt_from_registration_deg"] == pytest.approx(
        _inclinacion_del_eje_fdi(mundo, np.eye(3)), abs=0.3
    )


# --------------------------------------------------------------------------- #
# 8 · Espejo
# --------------------------------------------------------------------------- #
def _cuadrantes_cambiados(V, F, etq):
    cambia = np.where(etq // 10 == 1, etq + 10, np.where(etq // 10 == 2, etq - 10, etq))
    return V, F, cambia


def _escaneo_en_espejo(V, F, etq):
    return V * np.array([-1.0, 1.0, 1.0]), F[:, ::-1].copy(), etq


@pytest.mark.parametrize(
    "espejo", [_cuadrantes_cambiados, _escaneo_en_espejo], ids=["etiquetas", "malla"]
)
def test_un_escaneo_o_unas_etiquetas_en_espejo_no_se_registran(fichero, cbct, espejo) -> None:
    """Ninguna rotación corrige una reflexión: se dice, no se registra algo reflejado."""
    V, F, etq = espejo(*fichero)
    ejes, motivo = anatomical_frame(V, etq)
    assert ejes is not None, motivo
    base = np.stack([ejes.derecha, ejes.anterior, ejes.superior])
    assert np.linalg.det(base) < 0
    with pytest.raises(ValueError, match="espejo"):
        prepara_anatomico(*cbct, V, F, etq, ejes)


# --------------------------------------------------------------------------- #
# 9 · Puntuación y 10 · cobertura
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mm", [1.0, -1.0], ids=["hundida", "flotando"])
def test_una_corona_desplazada_en_altura_puntua_peor(referencia, mm) -> None:
    """S es lo que elige: 1 mm hacia la raíz (o hacia la cúspide) tiene que perder.

    Y el residuo con signo dice hacia dónde: − hundida en el diente del CBCT, + flotando.

    ⚠️ Medido sobre el fantasma, con paredes de corona casi verticales en la banda: hundir
    1 mm sube S solo de 0,149 a 0,160 mm (+8 %, dentro de `MARGEN_AMBIGUEDAD_S`) y lleva
    el residuo con signo de −0,02 a −0,09 mm, muy lejos de `RESIDUO_FIRMADO_MAX_MM`;
    flotar 1 mm, a 0,237 y +0,14. Las cotas siguen eso: el signo y el orden, no el tamaño.
    """
    registrador = referencia["registrador"]
    busqueda = registrador.cache["ultima"]
    rot, tras = busqueda["chosen"]["pose"]
    raiz = registrador.base_cbct[2]  # arcada superior: la raíz está hacia +S
    buena = registrador.puntua_secciones(rot, tras, busqueda["por_corte"])
    movida = registrador.puntua_secciones(rot, tras + mm * raiz, busqueda["por_corte"])
    assert movida["S"] > buena["S"] + 0.005

    def firmado(p: dict) -> float:
        return float(np.median([v["median"] for v in p["signed_residual"].values()]))

    if mm > 0:
        assert firmado(movida) < firmado(buena) - 0.03
    else:
        assert firmado(movida) > firmado(buena) + 0.03


def test_sin_cobertura_una_pose_no_es_elegible(referencia) -> None:
    registrador = referencia["registrador"]
    rot, tras = registrador.cache["ultima"]["chosen"]["pose"]
    justo = registrador.coherencia(rot, tras, {"C": anatomico.COBERTURA_MIN})
    assert justo["eligible"]
    poca = registrador.coherencia(rot, tras, {"C": anatomico.COBERTURA_MIN - 0.01})
    assert not poca["eligible"]
    assert any("cobertura" in m for m in poca["why_not"])


def _snap() -> TwinSnapshot:
    return TwinSnapshot(
        acquisition_id="A1",
        timestamp=datetime.now(UTC),
        gaussian_field_ref="sha256:abc",
        provenance=Provenance(
            source_file="caso/", modality=Modality.CBCT, agent="cbct-agent@1.0.0"
        ),
    )


def test_sin_ninguna_cuenca_elegible_hay_veto_y_el_agente_da_confianza_cero(
    fichero, cbct, monkeypatch
) -> None:
    """Se fuerza una cobertura imposible: ninguna cuenca pasa, y eso no se esconde.

    Con un solo arranque, porque lo que se prueba es qué se hace sin elegibles, no la
    búsqueda.
    """
    monkeypatch.setattr(anatomico, "COBERTURA_MIN", 1.01)
    V, F, etq = fichero
    origen, destino, registrador, informe, _ = registra(
        V, F, etq, *cbct, arranques="rejilla(+0,+0,+0)"
    )
    resultado = registrador(origen, destino)
    assert resultado.veto is not None and "cobertura" in resultado.veto
    assert informe["search"]["veto"] == resultado.veto
    assert resultado.veto in motivos(informe)

    out = GeometricFusionAgent(registrar=registrador, epsilon_mm=1.5).fuse(
        _snap(), source=origen, target=destino
    )
    assert out.ok
    assert out.snapshot.provenance.confidence == 0.0
    assert any(m.startswith("registro vetado") for m in out.hitl_reasons)


# --------------------------------------------------------------------------- #
# 11 · Escala
# --------------------------------------------------------------------------- #
def test_un_escaneo_a_otra_escala_se_declara(referencia, fichero, cbct) -> None:
    """Un 2 % de escala no lo absorbe ninguna pose rígida, y el gate lo dice."""
    assert not any("modelo rígido" in m for m in motivos(referencia["informe"]))
    assert referencia["informe"]["uncertainty"]["scale_scanner_to_cbct"] == pytest.approx(
        1.0, abs=anatomico.ESCALA_MAX
    )

    V, F, etq = fichero
    centro = V[etq > 0].mean(axis=0)
    grande = centro + 1.02 * (V - centro)
    _, _, _, informe, _ = registra(
        grande, F, etq, *cbct, incertidumbre=True, arranques="rejilla(+0,+0,+0)"
    )
    escala = informe["uncertainty"]["scale_scanner_to_cbct"]
    assert escala == pytest.approx(1 / 1.02, abs=0.005)
    assert any("ningún modelo rígido" in m for m in motivos(informe))


# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
def test_la_huella_de_las_constantes_va_al_informe_y_cambia_con_ellas(
    referencia, monkeypatch
) -> None:
    huella = referencia["informe"]["constants_sha256"]
    assert huella == anatomico.huella_constantes()
    monkeypatch.setattr(anatomico, "COBERTURA_MIN", 0.51)
    assert anatomico.huella_constantes() != huella
    assert math.isfinite(referencia["informe"]["search"]["chosen"]["S"])
