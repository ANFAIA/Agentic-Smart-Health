"""El almacén direccionado por contenido: reproducibilidad y fail-loud."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ingestion_agents.store import REF_PREFIX, ArtifactStore, content_digest


def test_misma_entrada_misma_referencia(store: ArtifactStore) -> None:
    """La referencia es una huella del contenido: la trazabilidad depende de ello."""
    arrays = {"a": np.arange(10, dtype=np.float32)}
    assert store.put(**arrays) == store.put(**arrays)


def test_contenido_distinto_referencia_distinta(store: ArtifactStore) -> None:
    ref_a = store.put(a=np.zeros(4, dtype=np.float32))
    ref_b = store.put(a=np.ones(4, dtype=np.float32))
    assert ref_a != ref_b


def test_el_dtype_forma_parte_de_la_huella() -> None:
    """Mismos valores con distinto dtype son artefactos distintos, no el mismo."""
    assert content_digest({"a": np.zeros(4, dtype=np.float32)}) != content_digest(
        {"a": np.zeros(4, dtype=np.float64)}
    )


def test_reingerir_no_duplica_en_disco(store: ArtifactStore) -> None:
    arrays = {"a": np.arange(100, dtype=np.float32)}
    store.put(**arrays)
    store.put(**arrays)
    assert len(list(store.root.iterdir())) == 1


def test_roundtrip(store: ArtifactStore) -> None:
    original = {"pos": np.random.rand(5, 3).astype(np.float32), "n": np.arange(5)}
    loaded = store.load(store.put(**original))
    assert set(loaded) == set(original)
    for key, value in original.items():
        np.testing.assert_array_equal(loaded[key], value)


def test_referencia_colgante_es_error(store: ArtifactStore) -> None:
    """Invariante del ADR 001: una referencia que no resuelve es un error, no un vacío."""
    fake = f"{REF_PREFIX}{'0' * 64}"
    assert store.exists(fake) is False
    with pytest.raises(FileNotFoundError):
        store.load(fake)


def test_referencia_malformada(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="mal formada"):
        store.load("no-es-una-referencia")


def test_artefacto_vacio_rechazado(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="vacío"):
        store.put()


def test_no_deja_temporales(store: ArtifactStore) -> None:
    """La escritura es atómica: no queda ningún `.tmp` a medias."""
    store.put(a=np.arange(50, dtype=np.float32))
    assert not list(Path(store.root).glob("*.tmp"))
