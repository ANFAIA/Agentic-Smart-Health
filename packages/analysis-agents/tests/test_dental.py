"""El `Segmenter` que junta las dos medidas, probado por lo que cada una NO puede sola.

El modelo del CBCT sabe *qué* es diente y no sabe *cuál*; el escáner intraoral sabe cuál y
solo ve corona. Estos tests comprueban que ninguna de las dos se usa para lo que no sirve.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis_agents import DEFAULT_CODES, GUM_CLASS, SegmentadorDental
from analysis_agents.segmentation import SegmentationAgent


def _coronas() -> tuple[np.ndarray, np.ndarray]:
    """Dos coronas separadas 10 mm, con códigos FDI distintos."""
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 1.0, (200, 3))
    b = rng.normal([10, 0, 0], 1.0, (200, 3))
    return np.vstack([a, b]), np.array([36] * 200 + [37] * 200)


def test_devuelve_log_probabilidades_de_verdad():
    """El agente lo comprueba, y con razón: leer logits como log-probabilidades daría
    confianzas plausibles y falsas — el modo de fallo caro."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.9), coronas, etq)
    logp = seg(coronas[:50])

    assert np.allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-3)
    assert (logp <= 0).all()


def test_el_fdi_lo_pone_el_escaner_no_el_modelo():
    """La separación entre dientes viene del IOS. El modelo del CBCT es binario y no
    puede darla: los dientes se tocan en el punto de contacto interproximal."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.95), coronas, etq)
    logp = seg(np.array([[0.0, 0, 0], [10.0, 0, 0]]))

    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    assert int(np.argmax(logp[0])) == col[36]
    assert int(np.argmax(logp[1])) == col[37]


def test_un_punto_de_diente_sin_nombre_cuenta_como_encia():
    """Lo importante, y es una decisión, no un descuido.

    Si el modelo dice «diente» pero ninguna corona lo reclama, no se puede declarar
    diente: no habría a qué colgarle un hallazgo clínico. Declarar «diente sin saber cuál»
    sería inventar la mitad que falta — y el sitio donde pasa es el hueso alveolar que el
    modelo marca de más, justo lo que NO debe heredar el FDI del vecino.
    """
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.99), coronas, etq)
    lejos = np.array([[0.0, 0.0, 500.0]])  # muy fuera del radio de nombre

    logp = seg(lejos)
    assert int(np.argmax(logp[0])) == GUM_CLASS
    assert np.exp(logp[0, GUM_CLASS]) == pytest.approx(1.0)


def test_la_probabilidad_del_modelo_decide_diente_contra_encia():
    """El IOS dice cuál, pero no dice si hay diente ahí: eso es del modelo del CBCT."""
    coronas, etq = _coronas()
    col = {fdi: c for c, fdi in DEFAULT_CODES.items()}
    punto = np.array([[0.0, 0, 0]])

    alto = SegmentadorDental(lambda p: np.full(len(p), 0.9), coronas, etq)(punto)
    bajo = SegmentadorDental(lambda p: np.full(len(p), 0.1), coronas, etq)(punto)

    assert int(np.argmax(alto[0])) == col[36]
    assert int(np.argmax(bajo[0])) == GUM_CLASS
    assert np.exp(alto[0, col[36]]) == pytest.approx(0.9, abs=1e-6)


def test_sin_ningun_codigo_falla_en_vez_de_segmentar_a_ciegas():
    """Correr sin etiquetas produciría un `region_id` con un solo código para todo, que es
    exactamente lo que pasó al ejecutar la composición sin `--fdi`: 30.592 gaussianas bajo
    un único FDI y una «pieza» de 66 mm."""
    coronas, _ = _coronas()
    with pytest.raises(ValueError, match="código FDI"):
        SegmentadorDental(lambda p: np.ones(len(p)), coronas, np.zeros(len(coronas)))


def test_encaja_con_el_agente_que_lo_va_a_usar():
    """El contrato real: `SegmentationAgent` lo acepta como `Segmenter` y saca `detected`."""
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.full(len(p), 0.9), coronas, etq)
    agente = SegmentationAgent(_AlmacenFalso({"centers": coronas}), segmenter=seg)

    assert isinstance(agente.segmenter(coronas[:10]), np.ndarray)


class _AlmacenFalso:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    def load(self, ref: str) -> dict[str, np.ndarray]:
        return self._arrays

    def put(self, **arrays: np.ndarray) -> str:
        return "sha256:" + "0" * 64


def test_probabilidad_1_no_produce_menos_infinito():
    """`SegmentationAgent` rechaza no finitos, y `log(1 - 1.0)` es `-inf`.

    No es teórico: el modelo devuelve 1,0 exacto para puntos de esmalte, y sin el recorte
    la etapa entera salía `FAILED` con «el segmentador devolvió valores no finitos».
    """
    coronas, etq = _coronas()
    seg = SegmentadorDental(lambda p: np.ones(len(p)), coronas, etq)
    logp = seg(coronas[:20])
    assert np.isfinite(logp).all()
    assert np.allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-3)

