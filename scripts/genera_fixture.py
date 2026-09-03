#!/usr/bin/env python
"""genera_fixture.py — El banco de pruebas de conformidad del formato UOS (G-4).

    uv run python scripts/genera_fixture.py --destino fixtures/uos-0.2

**Por que existe.** Un formato cuya unica definicion ejecutable vive dentro de su
implementacion de referencia obliga a un segundo implementador a comprobar su lector
contra su propia salida, que es comprobar nada. La §16 lo listaba como pendiente y la
revision externa lo declaro bloqueante antes de anunciar el formato (G-4).

**Que produce.** Un contenedor VALIDO y varios rotos a proposito, cada uno con un solo
defecto y con la clase de error que debe producir escrita al lado, en `esperado.json`. Un
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

    valido = destino / "valido.uos"
    valido.write_bytes(salida.path.read_bytes())
    indice = [{
        "fichero": "valido.uos",
        "espera": "valido",
        "porque": "el contenedor de referencia. Un lector conforme lo acepta sin errores.",
    }]

    def caso_roto(nombre: str, espera: str, porque: str, cambia) -> None:
        _reescribe(valido, destino / nombre, cambia)
        indice.append({"fichero": nombre, "espera": espera, "porque": porque})

    caso_roto(
        "entrada-no-declarada.uos", "error",
        "lleva un fichero que el manifiesto no nombra: sin hash que lo acredite y sin capa "
        "regulatoria. Es la forma que tendria una fuga (§14.6)",
        lambda n, c: c,
    )
    with zipfile.ZipFile(destino / "entrada-no-declarada.uos", "a",
                         zipfile.ZIP_STORED) as z:
        z.writestr("colado.txt", "nadie declara esto")

    caso_roto(
        "hash-que-no-cuadra.uos", "error",
        "un asset declara un sha256 que no es el de sus bytes",
        # El de un asset que VIAJA: el de uno externo lo caza el contrato antes de llegar
        # al algoritmo, y este caso existe para probar el check de hashes.
        lambda n, c: _sin_cadena(n, c, _rompe_hash_interno),
    )
    caso_roto(
        "manifiesto-no-es-primero.uos", "error",
        "`manifest.json` no es la primera entrada fisica del ZIP, asi que un lector no "
        "puede leerlo sin recorrer el contenedor entero (§3)",
        lambda n, c: None if n == "manifest.json" else c,
    )
    with zipfile.ZipFile(valido) as z:
        man = z.read("manifest.json")
    with zipfile.ZipFile(destino / "manifiesto-no-es-primero.uos", "a",
                         zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", man)

    caso_roto(
        "capa-3-fuera-de-derived.uos", "error",
        "un asset declara layer 3 y no vive bajo `derived/`, asi que borrar ese directorio "
        "no quitaria la inferencia (§9)",
        lambda n, c: _sin_cadena(n, c, lambda d: d["assets"][0].update(
            {"regulatory": {"layer": 3, "clearances": []}}
        )),
    )
    caso_roto(
        "registro-provisional.uos", "aviso",
        "una registracion automatica sin `verified_by`: el contenedor es VALIDO y el visor "
        "tiene que presentarla como provisional (§6). Un lector que no avise se lo calla",
        lambda n, c: c,
    )
    caso_roto(
        "minor-superior-con-campo-nuevo.uos", "valido-con-aviso",
        "declara una version menor superior y trae un campo que este lector no conoce: "
        "hay que ignorarlo y NOMBRARLO, no rechazar el contenedor (§15.2)",
        lambda n, c: _sin_cadena(n, c, lambda d: d.update(
            {"uos_version": "0.99", "campo_del_futuro": 1}
        )),
    )
    caso_roto(
        "comprimido.uos", "error",
        "el ZIP usa DEFLATE en vez de STORE, asi que el acceso por rangos —la razon de ser "
        "del envoltorio— no funciona (§2)",
        lambda n, c: c,
    )
    with zipfile.ZipFile(valido) as z:
        entradas = [(i.filename, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(destino / "comprimido.uos", "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, crudo in entradas:
            z.writestr(nombre, crudo)

    (destino / "esperado.json").write_text(
        json.dumps({
            "formato": "UOS",
            "version": "0.2",
            "nota": (
                "Banco de conformidad. Corre tu validador sobre cada fichero y compara con "
                "`espera`. `error` significa que el contenedor NO es valido; `aviso` que si "
                "lo es y hay algo que decir; `valido-con-aviso` que se acepta ignorando lo "
                "que no se entiende. Todo el dato es sintetico: no hay paciente detras."
            ),
            "casos": indice,
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
        tam = (args.destino / c["fichero"]).stat().st_size
        print(f"  {c['espera']:18} {c['fichero']:34} {tam / 1024:7.1f} KB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
