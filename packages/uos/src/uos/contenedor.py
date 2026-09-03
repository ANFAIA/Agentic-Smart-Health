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

from uos.manifiesto import UOS_VERSION, Asset, Manifest, Part, digesto_de_partes

MANIFIESTO = "manifest.json"


def sha256(ruta: Path) -> str:
    """Hash del fichero, por bloques: un DICOM de 400 MB no se carga en memoria."""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def write_uos(
    destino: Path,
    manifiesto: Manifest,
    ficheros: Iterable[tuple[str, Path]],
    *,
    directorios: dict[str, Path] | None = None,
    extras: dict[str, str | bytes] | None = None,
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
    dirs = dict(directorios or {})
    # ⚠️ Los assets EXTERNOS se declaran y no se aportan: el manifiesto lleva su identidad
    # —`uri` logica, `sha256`, `bytes`— y el fichero vive fuera. Es el perfil ligero, y por
    # eso quedan fuera de las dos comprobaciones de abajo: exigir su fichero convertiria en
    # error justo lo que el perfil hace a proposito. Ver `Asset.external`.
    declaradas = {a.uri for a in manifiesto.assets if not a.external}
    # Un asset puede ser un DIRECTORIO (una serie DICOM): su uri acaba en "/" y agrupa.
    sueltas = {u for u in declaradas if not u.endswith("/")}
    if faltan := {u for u in declaradas if u.endswith("/")} - set(dirs):
        raise ValueError(
            f"el manifiesto referencia {len(faltan)} directorio(s) que no se aportaron: "
            + ", ".join(sorted(faltan))
        )
    # Un asset puede venir de un fichero del caso o generarse aqui —la escena convertida,
    # la segmentacion—, y para el contenedor son lo mismo: bytes con su hash. Lo que no
    # puede es faltar.
    aportadas = set(mapa) | set(extras or {})
    if faltan := sueltas - aportadas:
        raise ValueError(
            f"el manifiesto referencia {len(faltan)} asset(s) que no se aportaron: "
            + ", ".join(sorted(faltan)[:3])
        )
    if sobran := set(mapa) - declaradas:
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
        for prefijo, carpeta in sorted(dirs.items()):
            for hijo in sorted(p for p in carpeta.rglob("*") if p.is_file()):
                z.write(hijo, prefijo + hijo.relative_to(carpeta).as_posix())
        for uri, texto in (extras or {}).items():
            z.writestr(uri, texto)
    tmp.replace(destino)
    return destino


def read_manifest(ruta: Path) -> Manifest:
    """Lee el manifiesto de un `.uos`, comprobando que sea la primera entrada.

    ⚠️ **Y comprobando la VERSION, que es lo primero que el spec pide mirar y no se
    miraba.** El campo se escribia, se declaraba en el modelo y no lo leia nadie: un
    contenedor de una version posterior se parseaba con el contrato de esta, y como todos
    los modelos llevan `extra="forbid"`, un campo opcional nuevo —justo lo que una version
    menor tiene permitido anadir— no daba un aviso, reventaba el parseo. Ver `uos.version`.
    """
    with zipfile.ZipFile(ruta) as z:
        nombres = z.namelist()
        if not nombres or nombres[0] != MANIFIESTO:
            raise ValueError(
                f"{ruta.name}: la primera entrada es {nombres[0] if nombres else 'ninguna'!r} "
                f"y el spec exige {MANIFIESTO!r} — sin eso no hay identificacion positiva."
            )
        return read_manifest_from(z.read(MANIFIESTO), nombre=ruta.name)[0]


def read_manifest_from(
    crudo: bytes, *, nombre: str = "manifest.json"
) -> tuple[Manifest, list[str]]:
    """`(manifiesto, campos ignorados)` aplicando la rama de version que toque (§15)."""
    import json

    from uos.version import Lectura, como_leer, lee_permisivo

    declarada = str(json.loads(crudo).get("uos_version", ""))
    rama = como_leer(declarada)
    if rama is Lectura.RECHAZO:
        raise ValueError(
            f"{nombre}: declara uos_version {declarada!r} y este lector implementa "
            f"{UOS_VERSION!r}. Una version mayor no promete compatibilidad, y el riesgo no "
            "son los campos que no conozco sino los que SI conozco y pueden haber cambiado "
            "de significado: abrirlo seria adivinar."
        )
    if rama is Lectura.PERMISIVA:
        m, ignorados = lee_permisivo(crudo)
        return m, ignorados
    return Manifest.model_validate_json(crudo), []


def asset_de(
    ruta: Path, uri: str, *, id_: str, kind, visit: str, frame: str,
    media_type: str, **extra,
) -> Asset:
    """Construye el sobre de un asset midiendo el fichero: hash y tamano reales.

    Si el asset es `external`, la `uri` que se pase se descarta y se nombra por su
    **direccion de contenido**: un fichero que no viaja no tiene sitio dentro del
    contenedor, y una ruta seria una promesa sobre un ZIP en el que no esta. Ver
    `Asset._direccion_y_custodia`.
    """
    from uos.manifiesto import PRIORIDAD, direccion_de_contenido

    h = sha256(ruta)
    return Asset(
        id=id_, kind=kind, visit=visit,
        uri=direccion_de_contenido(h) if extra.get("external") else uri,
        media_type=media_type,
        sha256=h, bytes=ruta.stat().st_size, frame=frame,
        load_priority=extra.pop("load_priority", PRIORIDAD[kind]), **extra,
    )


def json_de(obj: object) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


def identidad_dicom_de(crudo: bytes) -> tuple[str | None, str | None, float | None, float | None]:
    """Como `_identidad_dicom` pero sobre BYTES, para quien lee de un ZIP (D-3)."""
    import io

    return _identidad_dicom(io.BytesIO(crudo))


def _identidad_dicom(ruta: Path) -> tuple[str | None, str | None, float | None, float | None]:
    """`(sop_instance_uid, sha256 de PixelData)` de un corte, o `(None, None)` (D-3).

    ⚠️ **Se hashea el VALOR de `(7FE0,0010)`, no el fichero.** Es la unica parte que la
    de-identificacion no toca salvo que se limpie a proposito, asi que sobrevive al paso
    que rompia la trazabilidad basada en el hash del fichero. Un fichero que no sea DICOM
    legible devuelve `(None, None)` y el corte se queda con lo que ya tenia: la identidad
    es un anadido, no un requisito para empaquetar.
    """
    try:
        import numpy as np
        import pydicom

        ds = pydicom.dcmread(ruta, stop_before_pixels=False, force=True)
        uid = getattr(ds, "SOPInstanceUID", None)
        px = ds.get(0x7FE00010)
        if uid is None or px is None or px.value is None:
            return None, None, None, None
        # ⚠️ **El rango sale de ESTA lectura y no de otra (T-2).** Los pixeles ya estan
        # descomprimidos aqui para hashearlos; sacar el minimo y el maximo es aritmetica
        # sobre un array que ya esta en memoria. Hacerlo en una pasada aparte —que es como
        # estaba— pagaba una tercera lectura del volumen entero por un dato gratuito.
        try:
            arr = ds.pixel_array
            bajo, alto = float(np.min(arr)), float(np.max(arr))
        except Exception:  # noqa: BLE001 - hay cortes con pixeles ilegibles
            bajo = alto = None
        return str(uid), hashlib.sha256(bytes(px.value)).hexdigest(), bajo, alto
    except Exception:  # noqa: BLE001 - un corte ilegible no debe tumbar el empaquetado
        return None, None, None, None


def partes_y_rango(carpeta: Path) -> tuple[list[Part], list[float] | None]:
    """Las partes de la serie **y el rango de sus pixeles, en la misma pasada** (T-2).

    ⚠️ **Por que van juntos.** El §6.1 dejaba `value_range` nulo alegando que barrer los
    pixeles en cada exportacion es caro, y la consecuencia era real: un visor sin ventana
    de visualizacion. Pero el escritor **ya lee cada byte de cada corte** aqui —para el
    `sha256` del fichero y para el hash de `PixelData`— asi que el minimo y el maximo
    salen de una lectura que ya estaba pagada. Calcularlos en otro sitio, como se hacia,
    anadia una tercera pasada sobre el volumen entero por un dato gratuito.

    El rango va SIN reescalar: `slope`/`intercept` los aplica quien describe la serie,
    que es quien ha leido esas etiquetas.
    """
    partes: list[Part] = []
    bajo = alto = None
    for hijo in sorted(p for p in carpeta.rglob("*") if p.is_file()):
        uid, px, b, a = _identidad_dicom(hijo)
        if b is not None and a is not None:
            bajo = b if bajo is None else min(bajo, b)
            alto = a if alto is None else max(alto, a)
        partes.append(Part(
            name=hijo.relative_to(carpeta).as_posix(),
            sha256=sha256(hijo),
            bytes=hijo.stat().st_size,
            sop_instance_uid=uid,
            pixel_data_sha256=px,
        ))
    return partes, (None if bajo is None else [bajo, alto])


def partes_de(carpeta: Path) -> list[Part]:
    """Solo las partes. Para quien no necesita el rango."""
    return partes_y_rango(carpeta)[0]


def asset_de_directorio(
    carpeta: Path, uri: str, *, id_: str, kind, visit: str, frame: str,
    media_type: str, partes: list[Part] | None = None, **extra,
) -> Asset:
    """El sobre de un asset que es una SERIE entera, midiendo fichero a fichero.

    ⚠️ Los nombres de los cortes viajan tal cual, y aqui eso es correcto aunque en el resto
    del contenedor no lo sea: el orden de una serie DICOM es dato clinico, y renombrarlos a
    `IM0001.dcm…` seria reescribir el fichero que decimos entregar intacto. Los nombres que
    trae un export de CBCT son del equipo (`3DSlice100.dcm`), no del paciente — si algun
    proveedor los emitiera con identificador dentro, eso hay que cazarlo en la ingesta y no
    aqui, porque para entonces el DICOM ya lo lleva en sus tags.
    """
    from uos.manifiesto import PRIORIDAD, direccion_de_contenido

    # ⚠️ Si el llamante ya recorrio la serie —para sacar tambien el rango de
    # pixeles (T-2)— se reusan sus partes en vez de volver a leerla entera.
    partes = partes_de(carpeta) if partes is None else partes
    if not partes:
        raise ValueError(f"{carpeta} no tiene ni un fichero: no hay serie que empaquetar.")
    h = digesto_de_partes(partes)
    # ⚠️ Externa, la serie sigue llevando sus `parts` con el hash de CADA corte. Es lo que
    # permite que quien la custodie demuestre que no le falta ninguno — la garantia que se
    # pierde es la custodia, no la trazabilidad.
    return Asset(
        id=id_, kind=kind, visit=visit,
        uri=(direccion_de_contenido(h) if extra.get("external")
             else (uri if uri.endswith("/") else uri + "/")),
        media_type=media_type, sha256=h,
        bytes=sum(p.bytes for p in partes), frame=frame, parts=partes,
        load_priority=extra.pop("load_priority", PRIORIDAD[kind]), **extra,
    )


def asset_de_bytes(
    crudo: bytes, uri: str, *, id_: str, kind, visit: str, frame: str,
    media_type: str, **extra,
) -> Asset:
    """El sobre de un asset que se GENERA aqui y no sale de un fichero del caso.

    La escena convertida y la segmentacion no existen en disco: se construyen al exportar.
    Su `sha256` es el de los bytes que van a acabar en el ZIP, igual que el de los demas, y
    por eso el validador los comprueba sin saber que unos vinieron de un fichero y otros de
    memoria.
    """
    from uos.manifiesto import PRIORIDAD

    return Asset(
        id=id_, kind=kind, visit=visit, uri=uri, media_type=media_type,
        sha256=hashlib.sha256(crudo).hexdigest(), bytes=len(crudo), frame=frame,
        load_priority=extra.pop("load_priority", PRIORIDAD[kind]), **extra,
    )
