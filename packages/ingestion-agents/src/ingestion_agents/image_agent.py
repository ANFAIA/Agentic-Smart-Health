"""`image-agent` — foto intraoral 2D (JPG / PNG / HEIC) → soporte **superficial**.

Modalidad `image`, soporte `SURFACE`. Worker **determinista**: una imagen es un
formato con esquema, se decodifica, no se razona.

**Alcance honesto (PoC).** Este agente **no** reconstruye geometría 3D de una foto
suelta — eso no es posible sin más vistas. Lo que hace es **traer la apariencia al
contrato**: decodifica la foto, la deja lista (RGB, tamaño acotado) y la guarda como
artefacto. El **valor real** de la foto —pintar el color sobre la malla— se realiza
en la **fusión geométrica** (registro foto↔malla + proyección), que es una fase
posterior, no la ingesta. Aquí solo se ingiere la fuente de color, con trazabilidad.

**Privacidad — lo propio de esta modalidad.** Una foto arrastra **EXIF**: GPS,
número de serie del dispositivo, fecha/hora, a veces miniatura. El agente **descarta
el EXIF por construcción**: aplica la orientación EXIF (para que la imagen quede
derecha) y luego guarda **solo los píxeles** — el metadato **no** viaja al artefacto.
Es el eslabón de anonimización de imagen que el `docs/architecture/anonymization-strategy.md`
dejaba pendiente para fases con datos reales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from core_schemas import Modality, Support

from ingestion_agents.base import BaseIngestionAgent, IngestionOutput
from ingestion_agents.store import ArtifactStore

# Formatos que se decodifican sin dependencias extra. HEIC necesita `pillow-heif`.
_NATIVE = {".jpg", ".jpeg", ".png"}
_HEIC = {".heic", ".heif"}


class ImageAgent(BaseIngestionAgent):
    """Ingiere una foto 2D como fuente de apariencia (sin EXIF)."""

    name = "image-agent"
    version = "0.1.0"
    modality = Modality.IMAGE
    support = Support.SURFACE

    def __init__(
        self,
        store: ArtifactStore,
        *,
        max_edge: int = 2048,
        quarantine_dir: str | Path | None = None,
    ) -> None:
        super().__init__(quarantine_dir=quarantine_dir)
        self.store = store
        # Lado largo máximo del artefacto (la foto original de 24MP pesa demasiado cruda).
        self.max_edge = max_edge

    def _ingest(self, source: Path) -> IngestionOutput:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:  # pragma: no cover - entorno sin la dependencia
            raise RuntimeError(
                "El `image-agent` necesita `Pillow` (dependencia de `ingestion-agents`)."
            ) from exc

        ext = source.suffix.lower()
        if ext in _HEIC:
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
            except ImportError as exc:
                raise RuntimeError(
                    "Leer HEIC requiere el extra `heic` de `ingestion-agents` (pillow-heif)."
                ) from exc
        elif ext not in _NATIVE:
            raise ValueError(
                f"`image-agent` ingiere foto JPG/PNG/HEIC, no {ext!r}."
            )

        raw = Image.open(source)
        # Aplica la orientación declarada en EXIF (deja la imagen derecha) y a partir
        # de aquí se trabaja solo con píxeles: el EXIF NO llega al artefacto.
        img = ImageOps.exif_transpose(raw).convert("RGB")

        # Acota el tamaño: la original (p. ej. 6016×4016) como array cruda son ~70 MB.
        # El artefacto es una copia de trabajo; el fichero original sigue en disco.
        w, h = img.size
        confidence = 1.0
        if max(w, h) > self.max_edge:
            scale = self.max_edge / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
            confidence = 0.9  # es una copia reescalada, no la foto original a plena resolución

        pixels = np.asarray(img, dtype=np.uint8)  # (H, W, 3), sin canal alfa ni EXIF
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("La imagen no quedó como RGB de 3 canales tras decodificar.")

        return self._success(
            source,
            confidence=confidence,
            artifact_ref=self.store.put(pixels=pixels),
        )
