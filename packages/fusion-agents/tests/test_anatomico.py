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

import fantasma_arcada
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

    Desde la máscara de esmalte (§8 de la nota) la raíz ni siquiera entra en el contorno
    (`test_solo_cuenta_el_contorno_de_corona_y_la_losa_deja_de_importar`); este test sigue
    fijando que, con eso, la pose en el fantasma no empeora.
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


def test_solo_cuenta_el_contorno_de_corona_y_la_losa_deja_de_importar(monkeypatch) -> None:
    """La corona es esmalte y la raíz dentina: el contorno que se ajusta es solo el de corona.

    Con raíces realistas y una losa que llega a todo lo alto de la raíz —como la de dos
    anclas del caso A, §7 de la nota—, sin la máscara de esmalte el contorno trae paredes de
    raíz; con ella no queda ningún punto por encima del cuello, y el contorno es el MISMO
    que con la losa normal: ni la altura de la losa ni una raíz del ancho del cuello pueden
    ya tirar de la banda.
    """
    campo, hu = campo_cbct(paso=PASO, raices="realistas")
    centros = campo["centers"] + campo["origin"]
    _, plano, _ = marco_cbct(centros, hu)
    normal = anatomico.contornos_losa(centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper")[0]
    monkeypatch.setattr(anatomico, "LOSA_CBCT_MM", (-4.0, 25.6))
    alta = anatomico.contornos_losa(centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper")[0]
    sin_mascara = anatomico.contornos_losa(
        centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper", solo_esmalte=False
    )[0]
    # Cuello de las coronas: centro a ±6 mm y corte a 3 mm hacia la raíz (`fantasma_arcada`).
    cuello = 6.0 + 3.0 + PASO
    assert (np.abs(sin_mascara[:, 2]) > cuello).sum() > 1000
    assert not (np.abs(alta[:, 2]) > cuello).any()
    np.testing.assert_array_equal(normal, alta)


def test_con_esmalte_fino_hacia_el_cuello_la_mascara_no_mueve_la_pose(fichero) -> None:
    """El riesgo de la máscara: que el esmalte cervical, fino y desenfocado, no llegue a
    `HU_ESMALTE` y la banda se quede sin su contorno de arriba.

    Con una capa de esmalte de 1,5 mm que se adelgaza hasta cero en el cuello y un
    desenfoque de σ = 0,3 mm (más que la PSF de 425 µm de un CBCT dental), el ajuste desde
    la verdad acaba en el mismo sitio con la máscara que sin ella. El desenfoque sí deja un
    sesgo vertical de ~0,3 mm hacia la raíz —el isonivel 1400 se mete en el diente—, pero
    es del umbral, no de la máscara, y lo mide la sensibilidad a 1100/1900 de U.
    """
    campo, hu = campo_cbct(paso=PASO, raices="realistas", esmalte="capa", desenfoque_mm=0.3)
    centros = campo["centers"] + campo["origin"]
    V, F, etq = fichero
    ejes, _ = anatomical_frame(V, etq)
    escaner = anatomico.Escaner.construye(V, F, etq, ejes)
    base, plano, _ = marco_cbct(centros, hu)
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    poses = []
    for mascara in (True, False):
        P, N2 = anatomico.contornos_losa(centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper",
                                         solo_esmalte=mascara)
        reg = RegistroAnatomico(escaner=escaner, base_cbct=base, plano=plano,
                                contornos={anatomico.UMBRAL_CONTORNO: (P, N2)})
        r, t, _, convergido = reg.afina(*verdad, P)
        assert convergido
        assert reg.rms_desplazamiento((r, t), verdad) < 0.5
        poses.append((r, t))
    assert reg.rms_desplazamiento(*poses) < 0.02


def test_el_radio_de_la_mascara_de_corona_entra_en_la_incertidumbre(referencia) -> None:
    """Qué cuenta como corona es una elección declarada, y U dice cuánto mueve la pose. En
    el fantasma —corona maciza de esmalte— no mueve nada; en el caso A sí (§8)."""
    inc = referencia["informe"]["uncertainty"]
    radios = inc["crown_mask_radius"]
    assert set(radios) == {str(r) for r in anatomico.RADIOS_ESMALTE_INCERTIDUMBRE}
    assert all(v["converged"] and v["median_mm"] < 0.05 for v in radios.values())
    assert all("crown_mask_median_mm" in v for v in inc["per_tooth"].values())


def test_un_pico_de_ruido_en_el_perfil_no_es_la_otra_arcada() -> None:
    """El valle solo entra como ancla si el segundo pico es de verdad otra arcada.

    En el caso A el perfil tenía un «pico» 13,6 mm por encima con 3·10⁻⁴ de la altura del
    primero —unas pocas gaussianas sobre un perfil casi vacío— y su valle ensanchaba la
    losa hasta la raíz. Dos arcadas comparables sí dan dos anclas.
    """
    cortes_z = np.round(np.arange(-10.0, 20.0, 0.4), 3)

    def perfil(alturas: dict[float, int]) -> dict:
        z = [np.full(n, c) for c, n in alturas.items()]
        return anatomico._perfil_z(np.concatenate([cortes_z, *z]))

    una = {float(c): 400 for c in cortes_z if abs(c) <= 3}
    ruido = perfil({**una, **{float(c): 2 for c in cortes_z if abs(c - 13) <= 0.5}})
    assert not ruido["bimodal"]
    assert ruido["second_peak_ignored_mm"] == pytest.approx(13.0, abs=0.6)
    dos = perfil({**una, **{float(c): 300 for c in cortes_z if abs(c - 12) <= 3}})
    assert dos["bimodal"]
    assert dos["second_peak_ratio"] >= anatomico.PICO_SECUNDARIO_RELATIVO


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


def test_la_sobremordida_del_fantasma_es_la_de_una_mordida_normal() -> None:
    """La opción `oclusion="sobremordida"` es anatomía de libro, no un valor buscado.

    Incisivos inferiores 2–3 mm por lingual de los superiores y solapados 2–4 mm en altura;
    cúspides inferiores posteriores 1–2 mm por encima de las puntas superiores. Por defecto
    la arcada inferior es la de siempre, así que el resto de los tests no cambia.
    """
    from fantasma_arcada import centros_coronas, semiejes

    sup, f_sup = centros_coronas("upper")
    plana, f_inf = centros_coronas("lower")
    np.testing.assert_array_equal(plana, centros_coronas("lower", "plana")[0])
    inf, _ = centros_coronas("lower", "sobremordida")
    alto = np.array([semiejes(int(f))[2] for f in f_inf])
    punta_inf = inf[:, 2] + alto
    punta_sup = sup[:, 2] - np.array([semiejes(int(f))[2] for f in f_sup])
    for i, f in enumerate(f_inf):
        j = int(np.argmin(np.linalg.norm(sup[:, :2] - inf[i, :2], axis=1)))
        solape = punta_inf[i] - punta_sup[j]
        if f % 10 in (1, 2):
            assert 2.0 <= solape <= 4.0
            assert 2.0 <= float(np.linalg.norm(inf[i, :2] - sup[j, :2])) <= 3.0
            assert inf[i, 1] > sup[j, 1]  # lingual: hacia +y, que en LPS es posterior
        elif f % 10 >= 4:
            assert 1.0 <= solape <= 2.0
    with pytest.raises(ValueError, match="oclusión"):
        centros_coronas("lower", "abierta")


def test_la_sobremordida_no_baja_la_cobertura_en_la_verdad(fichero) -> None:
    """Hipótesis descartada: la sobremordida no explica la C ≈ 0,43 del caso A.

    Con sobremordida y resalte, la otra arcada tiene esmalte a las alturas de la banda y la
    máscara no la quita, así que podía contar contra el lado del contorno de C. Medido
    (`docs/research/registro-anatomico.md` §10): en la pose verdadera C baja de 0,975 a
    0,972, y a 0,94–0,95 en el peor caso plausible con capa de esmalte, desenfoque y serie
    dispersa; lejos de 0,4. El ajuste fino desde la verdad no se mueve. Si un fantasma
    con sobremordida algún día baja C en la verdad, este test lo dice.
    """
    campo, hu = campo_cbct(paso=PASO, raices="realistas", oclusion="sobremordida")
    centros = campo["centers"] + campo["origin"]
    V, F, etq = fichero
    ejes, _ = anatomical_frame(V, etq)
    escaner = anatomico.Escaner.construye(V, F, etq, ejes)
    base, plano, _ = marco_cbct(centros, hu)
    P, N2 = anatomico.contornos_losa(centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper")
    reg = RegistroAnatomico(escaner=escaner, base_cbct=base, plano=plano,
                            contornos={anatomico.UMBRAL_CONTORNO: (P, N2)})
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    puntuacion = reg.puntua_secciones(*verdad, anatomico._por_corte(P, None))
    assert puntuacion["C"] >= 0.9
    assert reg.coherencia(*verdad, puntuacion)["eligible"]
    r, t, _, convergido = reg.afina(*verdad, P)
    assert convergido
    assert reg.rms_desplazamiento((r, t), verdad) < 0.2


def test_la_encia_al_cuello_solo_entra_en_la_banda_con_coronas_cortas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`encia="cuello"`: el desborde de etiquetas solo llega a la banda si la corona es corta.

    Con la banda fija a 6 mm (`BANDA_AL_PLIEGUE` apagado); con el pliegue, ni así.

    Por defecto el escaneo es el de siempre. Con la encía naciendo en el margen y las
    etiquetas FDI desbordadas 3 mm sobre ella, la banda (1,5–6 mm bajo la cúspide) no cambia
    con coronas clínicas de 9 mm: la encía etiquetada empieza más arriba. Con coronas de
    6,5 mm, la papila (2,5 mm más oclusal en interproximal) sí mete encía en la banda.
    """
    V0, F0, e0 = escaneo_superior()
    V1, F1, e1 = escaneo_superior(encia="cinta")
    np.testing.assert_array_equal(V0, V1)
    np.testing.assert_array_equal(F0, F1)
    np.testing.assert_array_equal(e0, e1)
    with pytest.raises(ValueError, match="encia='cuello'"):
        escaneo_superior(desborde_mm=1.0)
    with pytest.raises(ValueError, match="encía"):
        escaneo_superior(encia="otra")

    monkeypatch.setattr(anatomico, "BANDA_AL_PLIEGUE", False)

    def banda(corona: str, desborde: float) -> tuple[np.ndarray, np.ndarray]:
        V, F, etq = escaneo_superior(encia="cuello", corona=corona, desborde_mm=desborde)
        ejes, _ = anatomical_frame(V, etq)
        return anatomico.Escaner.construye(V, F, etq, ejes).banda, etq

    sin, e_sin = banda("normal", 0.0)
    con, e_con = banda("normal", 3.0)
    assert (e_con > 0).sum() > (e_sin > 0).sum()  # el desborde etiqueta encía
    np.testing.assert_array_equal(sin, con)
    corta_sin, e_corta = banda("corta", 0.0)
    corta_con, _ = banda("corta", 3.0)
    encia_en_banda = corta_con & (e_corta == 0)
    assert not (corta_sin & (e_corta == 0)).any()
    assert encia_en_banda.sum() > 0.1 * corta_con.sum()
    monkeypatch.setattr(anatomico, "BANDA_AL_PLIEGUE", True)
    corta_pliegue, _ = banda("corta", 3.0)
    assert not (corta_pliegue & (e_corta == 0)).any()
    np.testing.assert_array_equal(banda("normal", 3.0)[0], con)


def test_el_desborde_de_etiquetas_sobre_la_encia_no_baja_la_cobertura_en_la_verdad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hipótesis descartada: el desborde de etiquetas no explica la C ≈ 0,43 del caso A.

    Medido con la banda fija a 6 mm (`BANDA_AL_PLIEGUE` apagado), que es la que deja
    entrar la encía; con el pliegue la encía ya no está en la banda.

    Con coronas clínicas de 6,5 mm y las etiquetas FDI 3 mm sobre la encía, un 15 % de la
    banda es encía. Medido (`docs/research/registro-anatomico.md` §11): en la pose
    verdadera C queda en 0,976 y el lado del escáner en 0,996, porque la encía de la banda
    es papila entre dos coronas y queda a menos de 1 mm del contorno de alguna. Afinar
    desde la verdad se mueve 0,10 mm hacia la raíz, frente a 0,08 sin desborde.
    """
    monkeypatch.setattr(anatomico, "BANDA_AL_PLIEGUE", False)
    Vw, F, etq = escaneo_superior(encia="cuello", corona="corta", desborde_mm=3.0)
    V = mueve(Vw, ROT_FICHERO, TRAS_FICHERO)
    campo, hu = campo_cbct(paso=PASO, raices="realistas")
    centros = campo["centers"] + campo["origin"]
    ejes, _ = anatomical_frame(V, etq)
    escaner = anatomico.Escaner.construye(V, F, etq, ejes)
    base, plano, _ = marco_cbct(centros, hu)
    P, N2 = anatomico.contornos_losa(centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper")
    reg = RegistroAnatomico(escaner=escaner, base_cbct=base, plano=plano,
                            contornos={anatomico.UMBRAL_CONTORNO: (P, N2)})
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    puntuacion = reg.puntua_secciones(*verdad, anatomico._por_corte(P, None))
    assert puntuacion["C"] >= 0.9
    assert puntuacion["C_scanner_side"] >= 0.95
    assert reg.coherencia(*verdad, puntuacion)["eligible"]
    r, t, _, convergido = reg.afina(*verdad, P)
    assert convergido
    assert reg.rms_desplazamiento((r, t), verdad) < 0.2


def test_el_desborde_sobre_la_encia_lleva_el_ajuste_hacia_la_raiz_sin_bajar_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debilidad medida con la banda fija a 6 mm: la encía etiquetada tira de la altura.

    Con `BANDA_AL_PLIEGUE` apagado (el valor por defecto, la banda fija), explícito para que
    no cambie si un día se enciende. El remedio lo fija el test siguiente.

    Con capa de esmalte, desenfoque de 0,18 mm y serie dispersa, la máscara deja poco
    contorno cervical, y la encía etiquetada como diente (inclinada, a 1 mm de la corona)
    se lleva un 17 % de las parejas del ajuste. Afinar desde la verdad desliza la arcada
    0,41 mm hacia la raíz con 3 mm de desborde, frente a 0,04 mm sin él, y C sigue por
    encima de 0,9: el veto no lo detecta (`docs/research/registro-anatomico.md` §11). Si
    algún día el ajuste deja de deslizar, o C empieza a verlo, este test lo dice.
    """
    monkeypatch.setattr(anatomico, "BANDA_AL_PLIEGUE", False)
    campo, hu = campo_cbct(paso=PASO, paso_z=0.45, quita=(2.2, 3.9), raices="realistas",
                           esmalte="capa", desenfoque_mm=0.18)
    centros = campo["centers"] + campo["origin"]
    base, plano, _ = marco_cbct(centros, hu)
    P, N2 = anatomico.contornos_losa(centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper")
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    desliza = {}
    for desborde in (0.0, 3.0):
        Vw, F, etq = escaneo_superior(encia="cuello", corona="corta", desborde_mm=desborde)
        V = mueve(Vw, ROT_FICHERO, TRAS_FICHERO)
        ejes, _ = anatomical_frame(V, etq)
        reg = RegistroAnatomico(escaner=anatomico.Escaner.construye(V, F, etq, ejes),
                                base_cbct=base, plano=plano,
                                contornos={anatomico.UMBRAL_CONTORNO: (P, N2)})
        r, t, _, convergido = reg.afina(*verdad, P)
        assert convergido
        desliza[desborde] = reg.rms_desplazamiento((r, t), verdad)
        assert reg.puntua_secciones(r, t, anatomico._por_corte(P, None))["C"] >= 0.9
    assert desliza[0.0] < 0.1
    assert desliza[3.0] > 0.3


@pytest.mark.parametrize(("grosor", "tope"), [(1.0, 0.10), (2.0, 0.15)])
def test_la_banda_al_pliegue_cervical_deja_fuera_la_encia_desbordada(
    monkeypatch: pytest.MonkeyPatch, grosor: float, tope: float
) -> None:
    """Remedio de §12: la banda de cada diente acaba en su pliegue cervical, no a 6 mm.

    Serie dispersa, capa de esmalte con σ = 0,18 mm, coronas de 6,5 mm y 3 mm de encía
    etiquetada como diente. Con la banda fija, afinar desde la verdad desliza 0,41 mm
    (encía de 1 mm) y 1,1 mm (2 mm). Criterio fijado antes de medir: < 0,10 y < 0,15 mm.
    Ningún vértice de encía queda en la banda y ningún diente se declara sin pliegue.
    """
    monkeypatch.setattr(anatomico, "BANDA_AL_PLIEGUE", True)
    monkeypatch.setattr(fantasma_arcada, "GROSOR_ENCIA_MM", grosor)
    campo, hu = campo_cbct(paso=PASO, paso_z=0.45, quita=(2.2, 3.9), raices="realistas",
                           esmalte="capa", desenfoque_mm=0.18)
    centros = campo["centers"] + campo["origin"]
    base, plano, _ = marco_cbct(centros, hu)
    P, N2 = anatomico.contornos_losa(centros, hu, anatomico.UMBRAL_CONTORNO, plano, "upper")
    verdad = (ROT_FICHERO.T, -ROT_FICHERO.T @ TRAS_FICHERO)
    Vw, F, etq = escaneo_superior(encia="cuello", corona="corta", desborde_mm=3.0)
    _, _, corona = escaneo_superior(encia="cuello", corona="corta", desborde_mm=0.0)
    V = mueve(Vw, ROT_FICHERO, TRAS_FICHERO)
    ejes, _ = anatomical_frame(V, etq)
    esc = anatomico.Escaner.construye(V, F, etq, ejes)
    assert esc.pliegue["sin_pliegue"] == []
    assert not (esc.banda & (corona == 0)).any()
    reg = RegistroAnatomico(escaner=esc, base_cbct=base, plano=plano,
                            contornos={anatomico.UMBRAL_CONTORNO: (P, N2)})
    r, t, _, convergido = reg.afina(*verdad, P)
    assert convergido
    assert reg.rms_desplazamiento((r, t), verdad) < tope


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
