#!/usr/bin/env python
"""genera_fixture.py — El banco de pruebas de conformidad del formato UOS (G-4).

    uv run python scripts/genera_fixture.py --destino fixtures/uos-0.2

**Por que existe.** Un formato cuya unica definicion ejecutable vive dentro de su
implementacion de referencia obliga a un segundo implementador a comprobar su lector
contra su propia salida, que es comprobar nada. La §16 lo listaba como pendiente y la
revision externa lo declaro bloqueante antes de anunciar el formato (G-4).

**Que produce.** Un contenedor VALIDO y varios rotos a proposito, cada uno con un solo
defecto y con la clase de error que debe producir escrita al lado, en `expected.json`. Un
lector ajeno corre su validador sobre el directorio y compara: si acepta uno de los rotos,
o rechaza el valido, sabe exactamente que le falta sin haber leido nuestro codigo.

⚠️ **Se construye sobre dato SINTETICO, y esa es la condicion para que pueda publicarse.**
Los `.uos` con dato clinico no se versionan y no salen de la clinica (ver B-3): un banco de
pruebas hecho con un caso real seria util una vez y no se podria distribuir, que es lo
contrario de lo que hace falta.

**Cada caso roto sale del valido**, parcheandolo. Fabricarlos por separado dejaria que
divergieran en cosas que no son el defecto que quieren probar, y entonces un lector que
rechace uno no sabria por cual de las dos razones.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "packages" / "uos" / "src"))

RESUMEN_EN = "Generates the format's conformance fixture: one valid container and several broken."


def _reescribe(origen: Path, destino: Path, cambia) -> None:
    """Copia un `.uos` aplicando `cambia(nombre, bytes) -> bytes | None`.

    Devolver `None` borra la entrada. Se conserva STORE y el orden, que es lo que hace que
    `manifest.json` siga siendo la primera entrada fisica.
    """
    with zipfile.ZipFile(origen) as z:
        entradas = [(i, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_STORED) as z:
        for info, crudo in entradas:
            nuevo = cambia(info.filename, crudo)
            if nuevo is not None:
                z.writestr(info, nuevo)


def _manifiesto(crudo: bytes, toca) -> bytes:
    """Aplica `toca(dict)` al manifiesto y lo vuelve a serializar.

    ⚠️ **Y suelta la cadena de procedencia.** El §8 encadena el `sha256` de los BYTES del
    manifiesto, asi que tocarlo —para lo que sea— invalida la cadena. Un caso que quiere
    probar UNA regla y falla por dos no prueba ninguna: quien corra el banco no sabria cual
    de las dos rechazo su lector. Un contenedor sin cadena es valido (es la primera version
    de un caso), asi que soltarla deja el defecto que se queria probar como el unico.
    """
    doc = json.loads(crudo)
    toca(doc)
    doc["provenance"] = {"chain": None, "prev_manifest_sha256": None}
    return json.dumps(doc, indent=1, ensure_ascii=False).encode()


def _sin_cadena(nombre: str, crudo: bytes, toca) -> bytes | None:
    """El parche del manifiesto y el borrado del fichero de cadena, que van juntos."""
    if nombre == "manifest.json":
        return _manifiesto(crudo, toca)
    return None if nombre.startswith("provenance/") else crudo



def _con_la_serie_dentro(destino: Path, serie: Path) -> Path:
    """Un contenedor que CUSTODIA la serie corte a corte.

    ⚠️ **Nuestro escritor no produce esto y no debe.** Referencia los originales y no los
    custodia (§3.4.3), asi que sin este contenedor el banco no puede ejercitar la
    verificacion por corte —el check 7— y se queda en UOS-Core: justo el nivel donde esa
    maquinaria no existe. Un validador tiene que funcionar sobre lo que escribio OTRO
    emisor, incluido uno que si decida llevar el DICOM dentro, y eso es lo que se prueba.
    """
    from uos.contenedor import asset_de_directorio, write_uos
    from uos.manifiesto import (
        AssetKind,
        Deidentification,
        Frame,
        Manifest,
        PHIState,
        Registration,
        Subject,
        Visit,
    )
    from uos.volumen import SIDECAR, describe_series

    sidecar_uri = SIDECAR.format(id="ct_001")
    sidecar, _ = describe_series(serie, frame="frame.ct_001")
    m = Manifest(
        case_id="urn:uuid:0", generator={"name": "fixture", "version": "0.2"},
        phi_state=PHIState.PSEUDONYMIZED, subject=Subject(pseudonym="FIXTURE-0001"),
        deidentification=Deidentification(
            profile="DICOM PS3.15 E.1 Basic Application Level Confidentiality Profile",
        ),
        canonical_frame=Frame(id="frame.ios_master"),
        frames=[Frame(
            id="frame.ct_001",
            dicom_frame_of_reference_uid=sidecar["dicom_frame_of_reference_uid"],
            anatomical=sidecar["anatomical"],
        )],
        visits=[Visit(id="v1", date="2026-09-03")],
        assets=[asset_de_directorio(
            serie, "volume/ct_001/", id_="asset.ct_001", kind=AssetKind.VOLUME,
            visit="v1", frame="frame.ct_001", media_type="application/dicom",
            sidecar_uri=sidecar_uri,
        )],
        registrations=[Registration(
            id="reg.ct_to_ios", source_frame="frame.ct_001",
            target_frame="frame.ios_master", method="manual", operator="user:fixture",
            transform_4x4_row_major=[1.0, 0, 0, 0, 0, 1.0, 0, 0,
                                     0, 0, 1.0, 0, 0, 0, 0, 1.0],
        )],
        occlusion="single_arch",
    )
    return write_uos(
        destino, m, [], directorios={"volume/ct_001/": serie},
        extras={sidecar_uri: json.dumps(sidecar)},
    )


def _rompe_hash_interno(doc: dict) -> None:
    """Falsea el `sha256` del primer asset que viaja dentro del contenedor."""
    for a in doc["assets"]:
        if not a.get("external"):
            a["sha256"] = "0" * 64
            return
    raise SystemExit("el contenedor base no lleva ningun asset interno que romper")


def genera(destino: Path) -> list[dict]:
    """Escribe el banco y devuelve su indice."""
    from agent_orchestrator import CaseInput, IngestionPipeline
    from ingestion_agents import ArtifactStore, synthetic
    from uos import UOSExportAgent

    destino.mkdir(parents=True, exist_ok=True)
    trabajo = destino / "_trabajo"
    trabajo.mkdir(exist_ok=True)

    # ⚠️ **Por el pipeline de verdad, no fabricando un manifiesto a mano.** Un banco
    # construido a mano prueba lo que su autor cree que emite el escritor; este prueba lo
    # que el escritor emite. Si los dos se separan, es el banco el que tiene que enterarse.
    synthetic.write_case(trabajo / "entrada", patient_id="FIXTURE-0001")
    almacen = ArtifactStore(trabajo / "artifacts")
    resultado = IngestionPipeline(almacen).run(
        CaseInput.from_case_dir(trabajo / "entrada")
    )
    if resultado.snapshot is None:
        raise SystemExit("la ingesta sintetica no produjo snapshot")
    salida = UOSExportAgent(almacen).export(
        resultado.snapshot, trabajo / "export", pseudonimo="FIXTURE-0001",
        malla=trabajo / "entrada" / "scan_upper.obj",
    )
    if not salida.ok:
        raise SystemExit(f"no se pudo generar el contenedor base: {salida.detail}")

    valido = destino / "valid.uos"
    valido.write_bytes(salida.path.read_bytes())
    indice = [{
        "file": "valid.uos",
        "expects": "valid",
        "why": "the reference container. A conformant reader accepts it with no errors.",
    }]

    def caso_roto(nombre: str, espera: str, porque: str, cambia) -> None:
        _reescribe(valido, destino / nombre, cambia)
        indice.append({"file": nombre, "expects": espera, "why": porque})

    caso_roto(
        "undeclared-entry.uos", "error",
        "carries a file the manifest does not name: no hash vouching for it and no regulatory "
        "layer. This is the shape a leak would have (§14.6)",
        lambda n, c: c,
    )
    with zipfile.ZipFile(destino / "undeclared-entry.uos", "a",
                         zipfile.ZIP_STORED) as z:
        z.writestr("colado.txt", "nadie declara esto")

    caso_roto(
        "hash-mismatch.uos", "error",
        "an asset declares a sha256 that is not the one of its bytes",
        # El de un asset que VIAJA: el de uno externo lo caza el contrato antes de llegar
        # al algoritmo, y este caso existe para probar el check de hashes.
        lambda n, c: _sin_cadena(n, c, _rompe_hash_interno),
    )
    caso_roto(
        "manifest-not-first.uos", "error",
        "`manifest.json` is not the first physical entry of the ZIP, so a reader cannot read "
        "it without walking the whole container (§3)",
        lambda n, c: None if n == "manifest.json" else c,
    )
    with zipfile.ZipFile(valido) as z:
        man = z.read("manifest.json")
    with zipfile.ZipFile(destino / "manifest-not-first.uos", "a",
                         zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", man)

    caso_roto(
        "layer-3-outside-derived.uos", "error",
        "an asset declares layer 3 and does not live under `derived/`, so deleting that "
        "directory would not remove the inference (§9)",
        lambda n, c: _sin_cadena(n, c, lambda d: d["assets"][0].update(
            {"regulatory": {"layer": 3, "clearances": []}}
        )),
    )
    caso_roto(
        "provisional-registration.uos", "warning",
        "an automatic registration with no `verified_by`: the container is VALID and the viewer "
        "must present it as provisional (§6). A reader that does not warn hides it",
        lambda n, c: c,
    )
    caso_roto(
        "higher-minor-new-field.uos", "valid-with-warning",
        "declares a higher minor version and carries a field this reader does not know: it "
        "must be ignored and NAMED, not grounds for rejecting the container (§15.2)",
        lambda n, c: _sin_cadena(n, c, lambda d: d.update(
            {"uos_version": "0.99", "field_from_the_future": 1}
        )),
    )
    caso_roto(
        "compressed.uos", "error",
        "the ZIP uses DEFLATE instead of STORE, so range access --- the whole point of the "
        "wrapper --- does not work (§2)",
        lambda n, c: c,
    )
    with zipfile.ZipFile(valido) as z:
        entradas = [(i.filename, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(destino / "compressed.uos", "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, crudo in entradas:
            z.writestr(nombre, crudo)

    # ── check 7 · la verificacion CORTE A CORTE (§6) ──────────────────────────
    # Sin un contenedor que custodie la serie, el banco no ejercita nada de esto y se
    # queda en UOS-Core. Son los tres desenlaces que el §6 distingue, y distinguirlos es
    # la diferencia entre «esta serie no cuadra» y «el corte 3 esta corrupto».
    con_serie = _con_la_serie_dentro(destino / "series-complete.uos",
                                     trabajo / "entrada" / "cbct")
    indice.append({
        "file": "series-complete.uos", "expects": "valid",
        "why": "it holds the whole DICOM series and every slice matches its hash (§6). "
                  "Our writer does not emit this way --- it references the originals --- "
                  "and a validator has to accept a writer that does. Note it reaches NO "
                  "conformance level, and that is correct: UOS-Core requires a renderable "
                  "scene and this container carries only the volume",
    })

    def _de_la_serie(nombre: str, espera: str, porque: str, cambia) -> None:
        _reescribe(con_serie, destino / nombre, cambia)
        indice.append({"file": nombre, "expects": espera, "why": porque})

    cortes = sorted(n for n in zipfile.ZipFile(con_serie).namelist()
                    if n.startswith("volume/ct_001/"))
    _de_la_serie(
        "series-slice-missing.uos", "error",
        f"slice {cortes[len(cortes) // 2]!r} is missing. A digest over the set would "
        "only say that something changed; §6 requires saying WHICH",
        lambda n, c: None if n == cortes[len(cortes) // 2] else c,
    )
    _de_la_serie(
        "series-slice-altered.uos", "error",
        f"slice {cortes[0]!r} has one byte changed: same name, same size, different "
        "content",
        lambda n, c: (c[:-1] + bytes([c[-1] ^ 0xFF])) if n == cortes[0] else c,
    )
    _de_la_serie(
        "series-slice-extra.uos", "error",
        "there is one slice more than the manifest declares in `parts[]`. An extra is as "
        "serious as a missing one: nobody knows where it came from",
        lambda n, c: c,
    )
    with zipfile.ZipFile(destino / "series-slice-extra.uos", "a",
                         zipfile.ZIP_STORED) as z:
        z.writestr(cortes[0].rsplit("/", 1)[0] + "/colado.dcm",
                   zipfile.ZipFile(con_serie).read(cortes[0]))

    # ⚠️ **El caso que da sentido a los dos niveles (D-3).** Un corte de-identificado
    # conserva su identidad clinica —SOP Instance UID y pixeles— y pierde sus bytes,
    # porque limpiar etiquetas reescribe la cabecera. Un validador que solo compare hashes
    # de fichero dice «esta serie no es la de este caso», que es falso y es la conclusion
    # mas cara posible. Con los dos niveles dice lo que pasa: es este corte, limpiado.
    def _deidentifica(nombre: str, crudo: bytes) -> bytes:
        if not nombre.startswith("volume/ct_001/") or nombre != cortes[0]:
            return crudo
        import io

        import pydicom

        ds = pydicom.dcmread(io.BytesIO(crudo))
        ds.PatientName = ""
        ds.InstitutionName = ""
        salida = io.BytesIO()
        ds.save_as(salida)
        return salida.getvalue()

    _de_la_serie(
        "series-slice-deidentified.uos", "warning",
        f"slice {cortes[0]!r} has had tags cleaned: it keeps its SOP Instance UID and "
        "its pixels, and its bytes are no longer the declared ones. This is a WARNING and "
        "not an error --- it is the same slice de-identified, not a different slice (§6)",
        _deidentifica,
    )

    (destino / "expected.json").write_text(
        json.dumps({
            "format": "UOS",
            "version": "0.2",
            "note": (
                "Conformance bench. Run your validator over each file and compare with "
                "`expects`. `error` means the container is NOT valid; `warning` that it is "
                "valid and there is something to say; `valid-with-warning` that it is "
                "accepted while ignoring what it does not understand. Every byte is "
                "synthetic: there is no patient behind it."
            ),
            "cases": indice,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for f in sorted((destino / "_trabajo").rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    (destino / "_trabajo").rmdir()
    return indice


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--destino", type=Path, default=RAIZ / "fixtures" / "uos-0.2")
    args = p.parse_args()
    indice = genera(args.destino)
    print(f"banco escrito en {args.destino}", file=sys.stderr)
    for c in indice:
        tam = (args.destino / c["file"]).stat().st_size
        print(f"  {c['expects']:18} {c['file']:34} {tam / 1024:7.1f} KB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
