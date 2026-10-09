"""Inventario de series DICOM sin identificadores, para que una clínica lo interprete.

**Para qué existe.** El ZIP del caso real trajo un estudio con dos series y dos marcos, con
huecos de hasta 12,3 mm entre cortes. Qué representa cada serie y si esos huecos son
deliberados solo lo puede decir quien hizo la adquisición. Para preguntárselo hace falta
describir las series con precisión **sin enviar nada que identifique al paciente**.

Lo que sale de aquí:

- Los UID solo como `sha256[:12]`, el mismo esquema que `scripts/prepara_series_cbct.py`.
  Sirve para referirse a «la serie 4528d89111be» en una conversación, no para buscarla en
  un PACS.
- Los tags de `TAGS_IDENTIFICABLES` se buscan en **todas** las cabeceras, no en la muestra
  de `identificables_en`, y se informan por NOMBRE, nunca por valor.
- Las fechas se recortan a mes salvo que se pidan completas: bastan para cotejar con el
  escáner y las fotos, y el paquete puede circular más allá de la clínica.
- El texto libre del proveedor (descripción de serie y estudio, protocolo) solo entra si
  ninguna cabecera tiene un tag identificable poblado. ⚠️ Ese texto **no se inspecciona**
  en busca de nombres: es la mejor pista de qué es cada serie, y también un sitio donde un
  operador puede escribir cualquier cosa. Se declara en `scope`.

No se leen píxeles. Nada aquí acredita calibración: son declaraciones del proveedor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np

from uos.volumen import tags_identificables_de

ESQUEMA = "histora-cbct-inventory/1.0"

IDENTIDADES = ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID")

# Declaraciones de adquisición y de equipo. Ninguna identifica a una persona; el número de
# serie del equipo (`DeviceSerialNumber`) y el nombre de la estación (`StationName`) sí
# identifican a una instalación, y por eso NO están.
DECLARACIONES = (
    "Modality", "ImageType", "Rows", "Columns", "PixelSpacing", "SliceThickness",
    "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept", "RescaleType",
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "KVP",
    "XRayTubeCurrent", "Exposure", "ExposureTime", "ReconstructionDiameter",
    "NumberOfFrames", "PixelSpacingCalibrationType", "PixelSpacingCalibrationDescription",
    "DateOfLastCalibration",
)

FECHAS = ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate")

TEXTO_LIBRE = ("StudyDescription", "SeriesDescription", "ProtocolName")

# Un salto mayor que 1,5 veces la mediana es un hueco: la misma regla que el `cbct-agent`.
FACTOR_HUECO = 1.5


def clave(uid: object) -> str | None:
    """Referencia corta y no reversible a un UID. Vacío → `None`, no el hash de ''."""
    texto = str(uid or "")
    return hashlib.sha256(texto.encode()).hexdigest()[:12] if texto else None


def _valores(cabeceras: list[Any], tag: str) -> list[str]:
    return sorted({str(getattr(ds, tag, "") or "") for ds in cabeceras})


def _fecha(valor: str, completas: bool) -> str | None:
    """`YYYYMMDD` → `YYYY-MM` (o `YYYY-MM-DD`). Lo que no tenga ese formato no sale."""
    if len(valor) != 8 or not valor.isdigit():
        return None
    return f"{valor[:4]}-{valor[4:6]}" + (f"-{valor[6:]}" if completas else "")


def _numeros_de_instancia(cabeceras: list[Any]) -> dict:
    numeros = [int(ds.InstanceNumber) for ds in cabeceras
               if getattr(ds, "InstanceNumber", None) not in (None, "")]
    if not numeros:
        return {"declared": 0}
    unicos = sorted(set(numeros))
    faltan = sorted(set(range(unicos[0], unicos[-1] + 1)) - set(unicos))
    return {
        "declared": len(numeros), "min": unicos[0], "max": unicos[-1],
        "missing_count": len(faltan), "missing": faltan,
        "duplicates": sorted({n for n in numeros if numeros.count(n) > 1}),
    }


def _posiciones(cabeceras: list[Any]) -> dict | None:
    """Cobertura en z, huecos y si el orden por instancia coincide con el de posición."""
    if not all(getattr(ds, "ImagePositionPatient", None) is not None for ds in cabeceras):
        return None
    z = np.array([float(ds.ImagePositionPatient[2]) for ds in cabeceras])
    orden = np.argsort(z, kind="stable")
    zs = z[orden]
    resultado: dict = {"z_min_mm": float(zs[0]), "z_max_mm": float(zs[-1]),
                       "extent_mm": float(zs[-1] - zs[0])}
    if len(zs) < 2:
        return resultado
    delta = np.diff(zs)
    paso = float(np.median(delta))
    huecos = []
    for i in np.flatnonzero(delta > FACTOR_HUECO * paso) if paso > 0 else []:
        huecos.append({"from_mm": float(zs[i]), "to_mm": float(zs[i + 1]),
                       "step_mm": float(delta[i]),
                       "estimated_missing": int(max(round(delta[i] / paso) - 1, 0))})
    estimados = (int(np.maximum(np.rint(delta / paso).astype(int) - 1, 0).sum())
                 if paso > 0 else None)
    resultado.update(
        median_step_mm=paso, min_step_mm=float(delta.min()), max_step_mm=float(delta.max()),
        uniform=bool(not huecos and np.allclose(delta, paso, rtol=0.01, atol=1e-4)),
        estimated_missing_slices=estimados, gaps=huecos,
        repeated_positions=int((delta == 0).sum()),
    )
    numeros: list[Any] = [getattr(ds, "InstanceNumber", None) for ds in cabeceras]
    if all(n not in (None, "") for n in numeros):
        enteros = np.array([int(n) for n in numeros])
        por_instancia = z[np.argsort(enteros, kind="stable")]
        d = np.diff(por_instancia)
        resultado["instance_order_matches_position"] = bool(np.all(d >= 0) or np.all(d <= 0))
    return resultado


def inventaria(cabeceras: Iterable[Any], *, fechas_completas: bool = False) -> dict:
    """Describe las series de un conjunto de cabeceras DICOM sin publicar identificadores.

    Las cabeceras se agrupan por `SeriesInstanceUID`; nada se fusiona ni se reescribe.
    """
    todas = list(cabeceras)
    if not todas:
        raise ValueError("No hay cabeceras DICOM que inventariar")
    identificables = sorted(set().union(*(tags_identificables_de(ds) for ds in todas)))
    con_texto = not identificables
    grupos: dict[str, list[Any]] = {}
    for ds in todas:
        grupos.setdefault(str(getattr(ds, "SeriesInstanceUID", "") or ""), []).append(ds)
    series = []
    for uid, grupo in sorted(grupos.items(), key=lambda kv: clave(kv[0]) or ""):
        fechas = {t: sorted({f for v in _valores(grupo, t)
                             if (f := _fecha(v, fechas_completas))}) for t in FECHAS}
        series.append({
            "series_key": clave(uid),
            "study_keys": sorted({k for ds in grupo
                                  if (k := clave(getattr(ds, "StudyInstanceUID", "")))}),
            "frame_keys": sorted({k for ds in grupo
                                  if (k := clave(getattr(ds, "FrameOfReferenceUID", "")))}),
            "instances": len(grupo),
            "instance_numbers": _numeros_de_instancia(grupo),
            "positions": _posiciones(grupo),
            "declarations": {t: _valores(grupo, t) for t in DECLARACIONES},
            "dates": fechas,
            "free_text": ({t: [v for v in _valores(grupo, t) if v] for t in TEXTO_LIBRE}
                          if con_texto else None),
        })
    conteos = {k: len({str(getattr(ds, k, "") or "") for ds in todas} - {""})
               for k in IDENTIDADES}
    return {
        "schema": ESQUEMA,
        "instances": len(todas),
        "identity_counts": conteos,
        "single_identity": all(n == 1 for n in conteos.values()),
        "identifying_tags_present": identificables,
        "free_text_included": con_texto,
        "dates_resolution": "day" if fechas_completas else "month",
        "series": series,
        "scope": ("Header declarations of the provider; no pixels read, no independent "
                  "calibration. UIDs as sha256[:12]. Identifying tags reported by name "
                  "only. Free text is included only when no identifying tag is populated "
                  "and is NOT itself inspected for names."),
    }


def lee_cabeceras_de_directorio(directorio: Any) -> list[Any]:
    """Cabeceras de los `.dcm` de un directorio, sin píxeles."""
    from pathlib import Path

    import pydicom

    return [pydicom.dcmread(p, stop_before_pixels=True)
            for p in sorted(Path(directorio).glob("*.dcm"))]


def lee_cabeceras_de_zip(archivo: Any) -> list[Any]:
    """Cabeceras de CADA entrada `.dcm` de un ZIP, leídas por `ZipInfo` y no por nombre.

    ⚠️ Leer por nombre devuelve una sola de las entradas que lo comparten: es exactamente
    lo que hizo que una extracción plana perdiera cuatro instancias del caso real.
    """
    import io
    from zipfile import ZipFile

    import pydicom

    with ZipFile(archivo) as zf:
        return [pydicom.dcmread(io.BytesIO(zf.read(info)), stop_before_pixels=True)
                for info in zf.infolist() if info.filename.lower().endswith(".dcm")]
