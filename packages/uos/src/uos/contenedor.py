"""El contenedor fisico: un `.uos` es un ZIP **sin comprimir** (§3).

**Por que STORE.** Los payloads ya vienen comprimidos —DICOM JPEG-LS/J2K, SPZ, GLB con
Draco— asi que comprimir el ZIP no ahorra y si rompe el acceso aleatorio. Con STORE y el
directorio central al final, un cliente HTTP con *range requests* lee el indice y baja un
asset suelto sin traerse el caso entero. Es el mismo precedente que `.usdz`.

**Y por que `manifest.json` va primero.** Es la identificacion positiva del formato: un
lector abre los primeros bytes, ve la entrada y su `uos_version`, y ya sabe que tiene
delante sin adivinar por la extension.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

from uos.manifiesto import Asset, Manifiesto

MANIFIESTO = "manifest.json"


def sha256(ruta: Path) -> str:
    """Hash del fichero, por bloques: un DICOM de 400 MB no se carga en memoria."""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def escribe_uos(
    destino: Path,
    manifiesto: Manifiesto,
    ficheros: Iterable[tuple[str, Path]],
    *,
    extras: dict[str, str] | None = None,
    json_manifiesto: str | None = None,
) -> Path:
    """Escribe el `.uos`. `ficheros` es `(uri interna, ruta en disco)`.

    Se escribe de forma **atomica** —temporal y `replace`, como el `ArtifactStore`—: un
    contenedor a medio escribir por un disco lleno no debe quedar donde alguien lo confunda
    con el bueno.

    ⚠️ Se verifica que toda `uri` del manifiesto tenga fichero y al reves. Un manifiesto
    que referencia algo que no esta es exactamente la referencia colgante que el resto del
    sistema trata como error y no como hueco.

    `json_manifiesto` deja al llamante aportar la serializacion que ya hizo. Lo necesita
    la cadena de procedencia (§8): su eslabon lleva el hash de estos bytes, y volver a
    serializar aqui dejaria el hash apuntando a una segunda copia que solo se le parece.
    """
    mapa = dict(ficheros)
    declaradas = {a.uri for a in manifiesto.assets}
    # Un asset puede ser un DIRECTORIO (una serie DICOM): su uri acaba en "/" y agrupa.
    sueltas = {u for u in declaradas if not u.endswith("/")}
    if faltan := sueltas - set(mapa):
        raise ValueError(
            f"el manifiesto referencia {len(faltan)} asset(s) que no se aportaron: "
            + ", ".join(sorted(faltan)[:3])
        )
    if sobran := set(mapa) - declaradas - set(extras or {}):
        raise ValueError(
            f"se aportaron {len(sobran)} fichero(s) que el manifiesto no declara: "
            + ", ".join(sorted(sobran)[:3])
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        # PRIMERA entrada, siempre. Es la identificacion positiva del formato.
        z.writestr(MANIFIESTO, json_manifiesto or manifiesto.json_canonico())
        for uri, ruta in sorted(mapa.items()):
            z.write(ruta, uri)
        for uri, texto in (extras or {}).items():
            z.writestr(uri, texto)
    tmp.replace(destino)
    return destino


def lee_manifiesto(ruta: Path) -> Manifiesto:
    """Lee el manifiesto de un `.uos`, comprobando que sea la primera entrada."""
    with zipfile.ZipFile(ruta) as z:
        nombres = z.namelist()
        if not nombres or nombres[0] != MANIFIESTO:
            raise ValueError(
                f"{ruta.name}: la primera entrada es {nombres[0] if nombres else 'ninguna'!r} "
                f"y el spec exige {MANIFIESTO!r} — sin eso no hay identificacion positiva."
            )
        return Manifiesto.model_validate_json(z.read(MANIFIESTO))


def asset_de(
    ruta: Path, uri: str, *, id_: str, kind, visit: str, frame: str,
    media_type: str, **extra,
) -> Asset:
    """Construye el sobre de un asset midiendo el fichero: hash y tamano reales."""
    from uos.manifiesto import PRIORIDAD

    return Asset(
        id=id_, kind=kind, visit=visit, uri=uri, media_type=media_type,
        sha256=sha256(ruta), bytes=ruta.stat().st_size, frame=frame,
        load_priority=extra.pop("load_priority", PRIORIDAD[kind]), **extra,
    )


def json_de(obj: object) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)
