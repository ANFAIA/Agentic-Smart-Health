"""Almacén *content-addressed* para los artefactos pesados de la ingesta.

El contrato (`core-schemas`) **no embebe** los arrays masivos: los referencia por
hash (`gaussian_field_ref`, `surface_ref`). Este módulo implementa ese almacén de
la forma más simple que cumple el invariante: un fichero `.npz` por artefacto,
nombrado por el **SHA-256 de su contenido**.

Dos propiedades que el proyecto necesita, y que salen gratis del direccionamiento
por contenido:

* **Reproducibilidad / trazabilidad.** El mismo fichero de entrada produce
  siempre la misma referencia, así que la referencia *es* una huella verificable
  de qué se ingirió (requisito de trazabilidad del brief).
* **Deduplicación.** Reingerir el mismo escaneo no duplica gigabytes en disco.

> **Seam.** Esta implementación es deliberadamente mínima y vive aquí, no en
> `packages/3dgs-engine/`, porque ese paquete se llama `3dgs-engine` y su módulo
> (`3dgs_engine`) **no es un identificador Python válido** — no se puede importar
> hasta renombrarlo. Cuando el motor exista de verdad, `ArtifactStore` es la
> interfaz que se sustituye; los agentes no cambian.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

REF_PREFIX = "sha256:"


def content_digest(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 del contenido de un conjunto de arrays.

    Se hashea nombre + dtype + shape + bytes de cada array en orden alfabético.
    No se hashea el `.npz` serializado a propósito: el formato ZIP incluye
    marcas de tiempo, así que el mismo contenido daría hashes distintos.
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


class ArtifactStore:
    """Almacén de blobs direccionados por contenido, sobre el sistema de ficheros."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- escritura ------------------------------------------------------- #
    def put(self, **arrays: np.ndarray) -> str:
        """Persiste los arrays y devuelve su referencia ``sha256:<hex>``.

        Idempotente: si el contenido ya está almacenado no se reescribe.
        """
        if not arrays:
            raise ValueError("No se puede almacenar un artefacto vacío.")
        digest = content_digest(arrays)
        path = self._path(digest)
        if not path.exists():
            tmp = path.with_suffix(".npz.tmp")
            # Se escribe sobre un handle abierto, no sobre la ruta: `savez` añade
            # un `.npz` al nombre si no lo lleva ya, y el temporal no lo lleva.
            with tmp.open("wb") as fh:
                # `**arrays` con claves dinámicas: mypy no puede descartar que una
                # colisione con un keyword del stub de numpy (`allow_pickle`).
                np.savez_compressed(fh, **arrays)  # type: ignore[arg-type]
            tmp.replace(path)  # atómico: nunca deja un .npz a medio escribir
        return f"{REF_PREFIX}{digest}"

    # --- lectura --------------------------------------------------------- #
    def exists(self, ref: str) -> bool:
        """¿Está el blob referenciado realmente en disco?

        Es el chequeo *fail-loud* del ADR 001: una referencia colgante es un
        error, no un modelo vacío silencioso.
        """
        return self._path(self._digest_of(ref)).exists()

    def load(self, ref: str) -> dict[str, np.ndarray]:
        path = self._path(self._digest_of(ref))
        if not path.exists():
            raise FileNotFoundError(f"Referencia colgante: {ref} no está en {self.root}")
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    def path_of(self, ref: str) -> Path:
        return self._path(self._digest_of(ref))

    # --- internos -------------------------------------------------------- #
    def _path(self, digest: str) -> Path:
        return self.root / f"{digest}.npz"

    @staticmethod
    def _digest_of(ref: str) -> str:
        if not ref.startswith(REF_PREFIX):
            raise ValueError(f"Referencia mal formada (esperado '{REF_PREFIX}<hex>'): {ref!r}")
        return ref[len(REF_PREFIX) :]
