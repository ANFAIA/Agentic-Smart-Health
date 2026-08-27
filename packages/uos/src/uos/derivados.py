"""`derived/` — lo que sale de un modelo, con su etiqueta regulatoria (§5.5).

**La regla dura del spec:** vive SOLO bajo `derived/`, siempre con `regulatory.layer: 3` y
un sidecar `meta.json` que declara qué modelo lo produjo, con qué pesos y sobre qué assets.
Y un `.uos` sin `derived/` sigue siendo válido y completo: borrar el directorio y sus
entradas del manifiesto es una operación soportada, para distribuir el caso en
jurisdicciones donde el módulo de IA no está habilitado.

⚠️ **Por eso las etiquetas FDI NO se hornean dentro de `scene.glb`.** El §11.3 sugiere el
picking por `extras.uos_fdi` en la malla, pero nuestras etiquetas salen de un segmentador:
son Layer 3. Metidas en la escena, quitar `derived/` dejaría de quitar la inferencia y la
regla de arriba se rompería en silencio. Van aquí, indexadas por vértice, y quien quiera
pintarlas las cruza por índice — que es exacto porque la escena conserva el orden.

**Formato.** El spec nombra DICOM SEG y labelmap NRRD, los dos pensados para segmentación
sobre una rejilla. La nuestra no es sobre rejilla: es una etiqueta **por primitiva**, que
es una tercera forma legítima y la única que no pierde nada de lo que tenemos. Se escribe
como `int16` little-endian y el `meta.json` declara qué indexa, cuántos hay y en qué orden.
Inventar una rejilla para caber en un formato conocido sería re-muestrear el dato para que
encaje en el envoltorio.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import numpy as np

SEGMENTACION = "derived/seg_teeth.bin"
SEGMENTACION_META = "derived/seg_teeth.meta.json"


def codifica_etiquetas(etiquetas: np.ndarray) -> bytes:
    """Los códigos FDI como `int16` little-endian, en el orden de la escena."""
    return np.ascontiguousarray(etiquetas, dtype="<i2").tobytes()


def meta_segmentacion(
    etiquetas: np.ndarray,
    *,
    asset_origen: str,
    modelo: str,
    version: str | None,
    pesos_sha256: str | None = None,
    estado: str = "investigational",
    jurisdicciones: list[str] | None = None,
    calidad: dict[int, dict] | None = None,
) -> dict[str, Any]:
    """El sidecar del §5.5. Todo lo que hace falta para saber **de dónde salió esto**.

    ⚠️ `weights_sha256` va a `null` cuando no se conoce, y NO se rellena con el hash de
    otra cosa. El campo existe para poder reproducir la inferencia; un hash que no es el de
    los pesos convierte «no lo sé» en «sí lo sé» y es indistinguible desde fuera.
    """
    etq = np.asarray(etiquetas)
    codigos = sorted(int(c) for c in np.unique(etq) if c > 0)
    return {
        "model": {
            "name": modelo,
            "version": version,
            "weights_sha256": pesos_sha256,
        },
        "source_assets": [asset_origen],
        "regulatory": {
            "layer": 3,
            "status": estado,
            "jurisdictions": jurisdicciones or [],
        },
        "generated": datetime.now(UTC).isoformat(),
        # Qué indexa esto. Sin esto, un `int16` suelto es un montón de números.
        "encoding": {
            "dtype": "int16-le",
            "count": int(etq.size),
            "indexes": (
                f"un código por vértice de `{asset_origen}`, en el mismo orden en que la "
                "escena los declara. 0 = sin asignar."
            ),
            "vocabulary": "ISO-3950 (FDI)",
        },
        # ⚠️ **Qué pieza se puede seleccionar de verdad, y cuál no.** Sin esto, el visor
        # enciende una corona que arrastra medio diente vecino y no hay nada en el fichero
        # que lo diga: lo que se enseña al lado es correcto y lo que se ve, no. El umbral
        # no es una opinión — sale del `p95` de `|ancho medido - tabla|` sobre 188 coronas
        # etiquetadas por experto de Teeth3DS+. Ver `analysis_agents.frontera`.
        **({} if calidad is None else {"per_tooth_boundary": {
            "criterion": "|mesiodistal medido - tabla anatomica| <= p95 de experto",
            "note": ("contar coronas «demasiado anchas» NO mide un defecto: las etiquetas "
                     "de experto lo fallan en el 77 %. Lo que separa es la magnitud"),
            "teeth": {str(k): v for k, v in sorted(calidad.items())},
        }}),
        "labels": {
            "present": codigos,
            "n_labelled": int((etq > 0).sum()),
            "n_total": int(etq.size),
        },
    }


def sha256_de_fichero(ruta) -> str | None:
    """Hash de los pesos, si el fichero está. `None` si no, sin inventar nada."""
    try:
        h = hashlib.sha256()
        with open(ruta, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
        return h.hexdigest()
    except OSError:
        return None
