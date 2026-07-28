"""`image-agent`: decodifica la foto, descarta EXIF, acota tamaño, fail-loud."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from core_schemas import Modality, ModalityStatus, Support
from ingestion_agents import ArtifactStore, ImageAgent

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _write_img(path: Path, size=(64, 48), color=(200, 120, 90)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_ingesta_de_jpg(tmp_path: Path, store: ArtifactStore) -> None:
    src = _write_img(tmp_path / "foto.jpg")
    outcome = ImageAgent(store).ingest(src)

    assert outcome.status is ModalityStatus.OK
    assert outcome.modality is Modality.IMAGE
    assert outcome.support is Support.SURFACE
    arrays = store.load(outcome.artifact_ref or "")
    assert set(arrays) == {"pixels"}
    px = arrays["pixels"]
    assert px.dtype == np.uint8 and px.shape == (48, 64, 3)  # (H, W, 3)


def test_png_y_conversion_a_rgb(tmp_path: Path, store: ArtifactStore) -> None:
    """Un PNG con alfa (RGBA) se ingiere como RGB de 3 canales."""
    src = tmp_path / "foto.png"
    Image.new("RGBA", (32, 32), (10, 20, 30, 128)).save(src)
    px = store.load(ImageAgent(store).ingest(src).artifact_ref or "")["pixels"]
    assert px.shape[2] == 3  # el alfa se descarta


def test_gris_a_rgb(tmp_path: Path, store: ArtifactStore) -> None:
    src = tmp_path / "gris.png"
    Image.new("L", (16, 16), 128).save(src)
    px = store.load(ImageAgent(store).ingest(src).artifact_ref or "")["pixels"]
    assert px.shape == (16, 16, 3)


def test_el_exif_no_viaja_al_artefacto(tmp_path: Path, store: ArtifactStore) -> None:
    """Privacidad: el artefacto es solo píxeles, sin metadatos EXIF."""
    src = tmp_path / "con_exif.jpg"
    exif = Image.Exif()
    exif[0x0110] = "SecretPhone X"    # Model
    exif[0x9003] = "2026:01:01 12:00:00"  # DateTimeOriginal
    Image.new("RGB", (32, 32), (1, 2, 3)).save(src, exif=exif)

    # sanity: el fichero original SÍ tiene el EXIF
    assert Image.open(src).getexif().get(0x0110) == "SecretPhone X"

    outcome = ImageAgent(store).ingest(src)
    arrays = store.load(outcome.artifact_ref or "")
    # el artefacto es un ndarray de píxeles: no hay ningún canal donde meter EXIF
    assert set(arrays) == {"pixels"}
    assert "SecretPhone X" not in str(arrays["pixels"].tobytes()[:64])


def test_reescala_si_supera_el_lado_maximo(tmp_path: Path, store: ArtifactStore) -> None:
    src = _write_img(tmp_path / "grande.jpg", size=(4000, 3000))
    outcome = ImageAgent(store, max_edge=1024).ingest(src)
    px = store.load(outcome.artifact_ref or "")["pixels"]
    assert max(px.shape[:2]) == 1024
    # se declara que es una copia reescalada, no la original
    assert outcome.provenance is not None and outcome.provenance.confidence == 0.9


def test_pequena_no_se_reescala(tmp_path: Path, store: ArtifactStore) -> None:
    src = _write_img(tmp_path / "chica.jpg", size=(200, 150))
    outcome = ImageAgent(store, max_edge=1024).ingest(src)
    assert outcome.provenance is not None and outcome.provenance.confidence == 1.0


def test_orientacion_exif_se_aplica(tmp_path: Path, store: ArtifactStore) -> None:
    """La orientación EXIF se aplica (imagen derecha) antes de descartar el metadato."""
    src = tmp_path / "girada.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotar 90°
    Image.new("RGB", (40, 20), (5, 5, 5)).save(src, exif=exif)
    px = store.load(ImageAgent(store).ingest(src).artifact_ref or "")["pixels"]
    # 40×20 con orientación 6 (rota 90°) -> queda 20 ancho × 40 alto
    assert px.shape[:2] == (40, 20)


def test_rechaza_otras_extensiones(tmp_path: Path, store: ArtifactStore) -> None:
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF")
    outcome = ImageAgent(store).ingest(src)
    assert outcome.status is ModalityStatus.FAILED
    assert "JPG/PNG/HEIC" in (outcome.detail or "")


def test_imagen_corrupta_falla_sin_lanzar(tmp_path: Path, store: ArtifactStore) -> None:
    src = tmp_path / "rota.jpg"
    src.write_bytes(b"esto no es un jpeg")
    outcome = ImageAgent(store).ingest(src)
    assert outcome.status is ModalityStatus.FAILED


def test_ingesta_reproducible(tmp_path: Path, store: ArtifactStore) -> None:
    src = _write_img(tmp_path / "foto.png")
    a = ImageAgent(store).ingest(src)
    b = ImageAgent(store).ingest(src)
    assert a.artifact_ref == b.artifact_ref


def test_foto_real_de_bite2text(store: ArtifactStore) -> None:
    real = Path.home() / "anfaia" / "Bite2Text" / "F1980" / "intraoral-photo"
    fotos = sorted(real.glob("*.jpg")) if real.is_dir() else []
    if not fotos:
        pytest.skip("Bite2Text no disponible")
    outcome = ImageAgent(store).ingest(fotos[0])
    assert outcome.ok
    px = store.load(outcome.artifact_ref or "")["pixels"]
    assert px.ndim == 3 and px.shape[2] == 3 and max(px.shape[:2]) <= 2048
