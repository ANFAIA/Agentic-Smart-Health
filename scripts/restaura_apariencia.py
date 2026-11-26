"""Reuse an archived appearance only when its originals and carried data match."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RESUMEN_EN = "Restores archived appearance after verifying its source hashes and carried data."


class ColorDeOtraFrontera(ValueError):
    """La apariencia archivada se entrenó con otra frontera diente/encía que la de ahora."""


def reetiqueta(params: dict, vertices: np.ndarray, etiquetas: np.ndarray) -> tuple[int, int]:
    """Vuelve a numerar cada gaussiana de la apariencia con las etiquetas de ESTA ejecución.

    La apariencia archivada trae el `region_id` del modelo que segmentó cuando se entrenó.
    Reutilizarla tal cual con otras etiquetas del escáner deja dos numeraciones en el mismo
    contenedor: `derived/seg_teeth` con la nueva y la selección del visor con la vieja. Se
    recalcula con la MISMA función y la misma cota que usa el entrenamiento
    (`_fdi_por_gaussiana`, voto de `VOTOS_FDI` vértices, `LIMITE_VECINO` mm), en el marco
    normalizado del artefacto.

    Devuelve `(cambian, cruzan)`: cuántas gaussianas cambian de código y cuántas de ellas
    cruzan la frontera diente/encía (de un código FDI a 0, o al revés). Ver `restaura`.
    """
    from gaussian_engine.apariencia import LIMITE_VECINO, _fdi_por_gaussiana

    escala = float(params["scan_scale"])
    vn = (np.asarray(vertices, dtype=np.float64)
          - np.asarray(params["scan_offset"], dtype=np.float64)) * escala
    reg, _lejos, _ = _fdi_por_gaussiana(
        np.asarray(params["means"], dtype=np.float64), vn, np.asarray(etiquetas),
        limite=LIMITE_VECINO * escala,
    )
    antes = np.asarray(params["region_id"]).astype(np.int16)
    params["region_id"] = reg
    cruzan = (np.asarray(reg) > 0) != (antes > 0)
    return int((reg != antes).sum()), int(cruzan.sum())


def restaura(uos: Path, artefacto: Path, malla: Path, fotos: list[Path],
             etiquetas: tuple[np.ndarray, np.ndarray] | None = None):
    """`etiquetas` = `(vértices, códigos)` de esta ejecución; con ellas se renumera."""
    from malla_mejorada import lee_apariencia_glb
    from uos import validate
    from uos.fidelidad import Process

    report = validate(uos)
    # Legacy manifests can contain fields retired by the current strict contract.
    # They are read as migration inputs, never re-emitted or declared conformant.
    if any(e.code != "UOS-E-004" for e in report.errors):
        raise ValueError("El UOS anterior tiene errores fuera del contrato legado")
    with zipfile.ZipFile(uos) as z:
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("uos_version") not in {"0.2", "0.3"}:
            raise ValueError("Versión de origen no soportada para reutilización")
        for asset in manifest["assets"]:
            if not asset.get("external") and not asset["uri"].endswith("/"):
                raw = z.read(asset["uri"])
                if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                    raise ValueError("Un payload anterior no coincide con su hash")
        ios = next(a for a in manifest["assets"] if a["id"] == "asset.ios")
        if hashlib.sha256(malla.read_bytes()).hexdigest() != ios["sha256"]:
            raise ValueError("La apariencia anterior pertenece a otro escaneo")
        scene = next(a for a in manifest["assets"] if a["id"] == "asset.scene")
        carried = lee_apariencia_glb(z.read(scene["uri"]))
        clinical = json.loads(z.read("clinical/observations.json"))
    with np.load(artefacto, allow_pickle=False) as data:
        params = {k: np.array(data[k], copy=True) for k in data.files}
    positions = (params["means"].astype(np.float64) / float(params["scan_scale"])
                 + params["scan_offset"]).astype(np.float32)
    expected = np.column_stack([carried[k] for k in ("x", "y", "z")])
    coefficients = ((params["colors"].astype(np.float64) - 0.5)
                    / 0.28209479177387814).astype(np.float32)
    dc = np.column_stack([carried[f"f_dc_{i}"] for i in range(3)])
    if not np.array_equal(positions, expected) or not np.allclose(coefficients, dc, atol=1e-6):
        raise ValueError("El artefacto no coincide con la apariencia del UOS anterior")
    photo_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in fotos}
    tones = []
    for tooth in clinical["teeth"]:
        color = tooth.get("color")
        if not color:
            continue
        photo = color["from_photo"].removeprefix("sha256:")
        if photo not in photo_hashes:
            raise ValueError("Falta una foto fuente del color anterior")
        tones.append(SimpleNamespace(
            fdi=int(tooth["fdi"]), lab=tuple(tuple(color[k]) for k in
                                           ("cervical", "middle", "incisal")),
            foto_sha256=photo, n_pixeles=color["n_pixels"],
            correccion=color.get("illumination_slope"),
        ))
    if not tones or "region_id" not in params:
        raise ValueError("La apariencia anterior no conserva tonos o etiquetas FDI")
    renumeradas = cruzan = None
    if etiquetas is not None:
        renumeradas, cruzan = reetiqueta(params, *etiquetas)
        # ⚠️ **El color de la apariencia está CONDICIONADO por las etiquetas con que se
        # entrenó** (`_colorea_malla` y `pinta_malla`: esmalte donde dice diente, encía donde
        # no). Renumerar cambia la selección, no el color: una gaussiana que pasa de diente a
        # encía conserva el color del diente. En el caso A, con C3, era una franja de encía
        # de unos 2 mm alrededor de cada corona con el color del diente, en la vista
        # frontal. Eso no se arregla renumerando: hay que volver a entrenar.
        if cruzan:
            raise ColorDeOtraFrontera(
                f"{cruzan:,} de {len(params['region_id']):,} gaussianas de la apariencia "
                "archivada cruzan la frontera diente/encía con las etiquetas de esta "
                "ejecución. Su color se entrenó con la frontera vieja y saldría cambiado (encía "
                "con color de diente, o al revés). Entrena la apariencia con estas etiquetas "
                "(`--entrena-apariencia`) en vez de reutilizarla.")
    process = Process(
        operation="reuse-archived-appearance", agent="caso_completo.restore",
        reproducibility="unknown", parameters={
            "source_uos_sha256": hashlib.sha256(uos.read_bytes()).hexdigest(),
            "source_asset_sha256": scene["sha256"],
            "source_artifact_sha256": hashlib.sha256(artefacto.read_bytes()).hexdigest(),
            "same_ios_sha256": ios["sha256"],
            "training_executed_this_run": False,
            # `region_id` recalculado con las etiquetas del escáner de esta ejecución, o
            # `None` si se conserva el del entrenamiento archivado.
            "fdi_labels_recomputed_changed": renumeradas,
            # Cuántas cruzan diente/encía: siempre 0 aquí, porque si no la reutilización se
            # rechaza. Viaja para que la procedencia lo diga y no haya que suponerlo.
            "fdi_labels_tooth_gingiva_crossed": cruzan,
            "source_conforms_to_current_contract": report.valid,
        },
    )
    params["processing_json"] = np.frombuffer(process.model_dump_json().encode(), dtype=np.uint8)
    params["uses_inferred_labels"] = np.asarray(True)
    return params, tones
