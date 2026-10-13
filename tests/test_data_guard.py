"""`incrustados`: una malla en base64 dentro de un HTML es el mismo dato que un `.stl`."""

from __future__ import annotations

import base64
import importlib.util
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "data_guard.py"
_spec = importlib.util.spec_from_file_location("data_guard", _RUTA)
assert _spec and _spec.loader
dg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dg
_spec.loader.exec_module(dg)


def _png(lado: int = 200) -> bytes:
    """PNG válido y poco comprimible, para que su base64 supere el mínimo."""
    rng = np.random.default_rng(0)
    filas = b"".join(b"\x00" + rng.integers(0, 256, lado * 3, dtype=np.uint8).tobytes()
                     for _ in range(lado))

    def trozo(tipo: bytes, datos: bytes) -> bytes:
        return (struct.pack(">I", len(datos)) + tipo + datos
                + struct.pack(">I", zlib.crc32(tipo + datos)))

    cabecera = struct.pack(">IIBBBBB", lado, lado, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + trozo(b"IHDR", cabecera)
            + trozo(b"IDAT", zlib.compress(filas)) + trozo(b"IEND", b""))


def _html(binario: bytes) -> bytes:
    return (b'<html><script>const d="' + base64.b64encode(binario)
            + b'";</script></html>')


def test_png_pasa() -> None:
    assert dg.bloques_no_imagen(_html(_png())) == []


def test_ply_se_veta() -> None:
    ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 9000\n" + bytes(108000)
    [(tam, parece)] = dg.bloques_no_imagen(_html(ply))
    assert parece == "PLY" and tam > 100_000


def test_gltf_se_veta() -> None:
    [(_, parece)] = dg.bloques_no_imagen(_html(b"glTF" + bytes(20000)))
    assert parece == "glTF binario"


def test_array_float_crudo_se_veta() -> None:
    vertices = np.random.default_rng(1).normal(size=(5000, 3)).astype(np.float32)
    [(_, parece)] = dg.bloques_no_imagen(_html(vertices.tobytes()))
    assert "sin firma conocida" in parece


def test_bloque_pequeno_se_ignora() -> None:
    assert dg.bloques_no_imagen(_html(b"ply\n" + bytes(1000))) == []


def test_notebook_con_lineas_partidas(tmp_path: Path, monkeypatch) -> None:
    codificado = base64.b64encode(b"ply\n" + bytes(30000)).decode()
    trozos = [codificado[i:i + 76] + "\n" for i in range(0, len(codificado), 76)]
    nb = {"cells": [{"outputs": [{"data": {"application/octet-stream": trozos}}]}]}
    (tmp_path / "n.ipynb").write_text(json.dumps(nb))
    (tmp_path / "foto.ipynb").write_text(json.dumps(
        {"cells": [{"outputs": [{"data": {"image/png": base64.b64encode(_png()).decode()}}]}]}))
    monkeypatch.setattr(dg, "REPO", tmp_path)
    problemas = dg.revisar_incrustados({"n.ipynb": 1, "foto.ipynb": 1, "otro.py": 1})
    assert len(problemas) == 1 and "`n.ipynb`" in problemas[0] and "PLY" in problemas[0]
