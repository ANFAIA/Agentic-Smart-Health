#!/usr/bin/env python
"""verifica_contenedor.py — que el `.uos` diga la verdad SOBRE SI MISMO.

    uv run python scripts/verifica_contenedor.py CASO.uos

## Por que existe, y por que no basta con el validador

`uos.validador` comprueba el **contrato**: que el ZIP este bien formado, que los hashes
cuadren, que el grafo de marcos sea conexo, que `derived/` sea Layer 3. Todo eso puede
estar perfecto mientras un descriptor afirma algo que su fichero no sostiene.

Y paso, ocho veces en un dia, sobre codigo que llevaba semanas escrito:

- el sidecar de la apariencia declaraba `units: "mm"` sobre un PLY en el espacio
  normalizado de Blender: la nube salia **32 veces mas pequena** y el visor pintaba una
  mota en el origen;
- ese mismo sidecar declaraba `n_primitives: 1.341.990` y el fichero traia **118.325**;
- la cabecera del PLY llevaba **7 bytes no-ASCII** —el formato la define ASCII— y un
  lector estricto ajeno la habria rechazado;
- el campo semilla viajaba **diezmado 9 a 1 sobre un solo eje** sin declararlo;
- el sidecar de la apariencia declaraba **catorce columnas sobre un PLY de dieciocho
  propiedades**: `region_id` —el codigo FDI por gaussiana— viajaba en los bytes y no en el
  descriptor, asi que para cualquier lector que no fuera el nuestro no existia.

Ninguno lo detecta el validador, porque ninguno es una violacion del contrato: son
**afirmaciones falsas dentro de un contenedor valido**. Y todos se ven en un minuto si
alguien compara lo que el manifiesto dice con lo que los bytes traen — que es lo unico
que hace este script.

⚠️ **Comprueba coherencia, no correccion.** Que las unidades declaradas sean las del
fichero no dice que el fichero este bien: dice que no miente sobre lo que es. Es un liston
bajo, y aun asi es el que se cayo ocho veces.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

RESUMEN_EN = "Verifies that a `.uos` container tells the truth about itself."

RAIZ = Path(__file__).resolve().parent.parent
for _src in sorted(RAIZ.glob("packages/*/src")):
    sys.path.insert(0, str(_src))

#: Perfiles de gaussianas que este formato emite. Uno desconocido es un fallo, no un aviso:
#: las columnas se llaman igual que las del 3DGS de facto y significan otra cosa.
PERFILES = {"ash-twin/1.0", "ash-twin-ajustado/1.0", "ash-gs-apariencia/1.0"}


def _cabecera_ply(crudo: bytes) -> tuple[dict[str, str], int, list[str], list[str]]:
    """`(comentarios, n_vertices, no_ascii, propiedades)` de un PLY, solo la cabecera."""
    fin = crudo.find(b"end_header")
    if fin < 0:
        return {}, -1, [], []
    cab = crudo[:fin]
    malos = [f"pos {i}: {hex(c)}" for i, c in enumerate(cab) if c > 127]
    texto = cab.decode("ascii", "replace")
    comentarios: dict[str, str] = {}
    propiedades: list[str] = []
    n = -1
    for linea in texto.splitlines():
        if linea.startswith("comment "):
            resto = linea[8:].strip()
            esp = resto.find(" ")
            if esp > 0:
                comentarios.setdefault(resto[:esp], resto[esp + 1:])
        elif linea.startswith("element vertex "):
            try:
                n = int(linea.split()[2])
            except (IndexError, ValueError):
                n = -1
        elif linea.startswith("property ") and not linea.startswith("property list"):
            partes = linea.split()
            if len(partes) == 3:
                propiedades.append(partes[2])
    return comentarios, n, malos, propiedades


def verifica(ruta: Path) -> list[str]:
    """Las incoherencias encontradas. Vacia si el contenedor no se contradice."""
    fallos: list[str] = []
    with zipfile.ZipFile(ruta) as z:
        dentro = set(z.namelist())
        m = json.loads(z.read("manifest.json"))

        for a in m["assets"]:
            uri, sidecar = a["uri"], a.get("sidecar_uri")
            if a.get("external"):
                # Un asset externo se nombra por su contenido, no por una ruta.
                if not uri.startswith("sha256:"):
                    fallos.append(f"{a['id']}: es externo y su uri {uri!r} es una ruta")
                elif uri.split(":", 1)[1] != a["sha256"]:
                    fallos.append(f"{a['id']}: la direccion de contenido no es su sha256")
                continue
            if sidecar and sidecar not in dentro:
                fallos.append(f"{a['id']}: declara el sidecar {sidecar} y no viaja")
            if not uri.endswith("/") and uri not in dentro:
                fallos.append(f"{a['id']}: declara {uri} y no viaja")

        # --- lo que ningun validador de contrato mira -----------------------
        for a in m["assets"]:
            sc = a.get("sidecar_uri")
            if not sc or not sc.endswith(".gs.json") or sc not in dentro:
                continue
            d = json.loads(z.read(sc))
            if d.get("profile") not in PERFILES:
                fallos.append(f"{a['id']}: perfil desconocido {d.get('profile')!r}")
            if a["uri"] not in dentro:
                continue
            crudo = z.read(a["uri"])
            com, n, no_ascii, props = _cabecera_ply(crudo[:65536])

            if n >= 0 and d.get("n_primitives") != n:
                fallos.append(
                    f"{a['id']}: el sidecar dice {d.get('n_primitives'):,} primitivas y "
                    f"el fichero trae {n:,}"
                )
            if no_ascii:
                fallos.append(
                    f"{a['id']}: la cabecera del PLY trae {len(no_ascii)} byte(s) "
                    f"no-ASCII ({no_ascii[0]}); el formato PLY la define ASCII"
                )
            # Las unidades: si el fichero las declara, tienen que ser las del sidecar.
            if (u := com.get("unidades")) and u != d.get("units"):
                fallos.append(
                    f"{a['id']}: el fichero dice unidades {u!r} y el sidecar {d.get('units')!r}"
                )
            # Un campo MEDIDO que viaja submuestreado tiene que decirlo.
            if d.get("measured") and "diezmado" in json.dumps(d, ensure_ascii=False):
                pass
            if com.get("submuestreo") and "submuestreo" not in d:
                fallos.append(f"{a['id']}: el fichero declara submuestreo y el sidecar no")

            # ⚠️ **Las columnas declaradas tienen que ser las propiedades del fichero.**
            # El sidecar de la apariencia declaraba catorce columnas sobre un PLY de
            # dieciocho propiedades: `region_id` —el codigo FDI por gaussiana, que es lo
            # que permite encender una pieza sin malla— viajaba en los bytes y no en el
            # descriptor. Ningun lector ajeno podia saber que estaba ahi, y el fichero
            # tampoco se rompia: simplemente el dato no existia para nadie mas.
            #
            # Se mira en las dos direcciones porque las dos son mentiras distintas:
            # declarar de menos esconde un dato, declarar de mas promete uno que no esta
            # y descoloca a quien monte el registro desde `columns`.
            if props and isinstance(d.get("columns"), list):
                declaradas = [c.get("name") for c in d["columns"]]
                sobran = [c for c in declaradas if c not in props]
                faltan = [p for p in props if p not in declaradas]
                if faltan:
                    fallos.append(
                        f"{a['id']}: el PLY trae {len(faltan)} propiedad(es) que el "
                        f"sidecar no declara ({', '.join(faltan)})"
                    )
                if sobran:
                    fallos.append(
                        f"{a['id']}: el sidecar declara {len(sobran)} columna(s) que el "
                        f"PLY no trae ({', '.join(str(c) for c in sobran)})"
                    )
                elif not faltan and declaradas != props:
                    fallos.append(
                        f"{a['id']}: el sidecar declara las mismas columnas en otro orden "
                        f"que el PLY; quien monte el registro desde `columns` lee cruzado"
                    )

        seg = next((a for a in m["assets"] if a["uri"].startswith("derived/seg_teeth")), None)

        # --- el informe frente a lo que el modelo emitio ---------------------
        # ⚠️ **Una pieza que el informe declara AUSENTE no puede estar segmentada.** Las dos
        # afirmaciones viajan hoy en el mismo contenedor y nadie las cruzaba: en un caso
        # real el informe decia «Diente 28 Ausente» —transcripcion determinista de un
        # documento que firmo una persona, Layer 1— y la segmentacion emitia un FDI 28 de
        # 275 vertices, que ademas obligaba al watershed de la pose a partir la foto en una
        # pieza de mas. El contenedor se contradecia a si mismo y se declaraba coherente.
        #
        # Va aqui, en el verificador, y no solo aguas arriba: el formato promete que un
        # `.uos` no se contradice, y esa promesa la tiene que poder comprobar quien lo
        # RECIBE, sin acceso al pipeline que lo escribio.
        obs = next((a for a in m["assets"]
                    if a["uri"].endswith("clinical/observations.json")
                    and a["uri"] in dentro), None)
        if obs and seg is not None and seg["uri"] in dentro:
            clinico = json.loads(z.read(obs["uri"]))
            ausentes = {
                int(t["fdi"]) for t in clinico.get("teeth", [])
                if "ausente" in t.get("findings", []) and str(t.get("fdi", "")).isdigit()
            }
            codigos = np.frombuffer(z.read(seg["uri"]), dtype="<i2")
            emitidos = {int(c) for c in np.unique(codigos) if c > 0}
            choque = sorted(ausentes & emitidos)
            if choque:
                cuantos = {c: int((codigos == c).sum()) for c in choque}
                fallos.append(
                    "el informe declara AUSENTE(S) la(s) pieza(s) "
                    + ", ".join(f"FDI {c} y la segmentacion emite {n:,} vertice(s) suyos"
                                for c, n in cuantos.items())
                    + ". La transcripcion del informe es Layer 1 y la segmentacion Layer 3: "
                    "el contenedor afirma dos cosas incompatibles sobre la misma pieza"
                )

        # --- la escena glTF, si viaja ---------------------------------------
        glb = next((a for a in m["assets"]
                    if a.get("media_type") == "model/gltf-binary"
                    and a["uri"] in dentro), None)
        if seg and not glb:
            fallos.append(
                "derived/seg_teeth viaja y la escena glTF no: esas etiquetas indexan sus "
                "vertices, asi que no indexan nada"
            )
        if glb and seg and seg["uri"] in dentro:
            b = z.read(glb["uri"])
            largo = struct.unpack("<I", b[12:16])[0]
            cab = json.loads(b[20:20 + largo].decode("utf-8"))
            acc = cab["accessors"][
                cab["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
            ]["count"]
            codigos = len(z.read(seg["uri"])) // 2
            if acc != codigos:
                fallos.append(
                    f"seg_teeth trae {codigos:,} codigos y la escena {acc:,} vertices: "
                    "no se pueden cruzar por indice"
                )
    return fallos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caso", type=Path)
    args = ap.parse_args()

    fallos = verifica(args.caso)
    if not fallos:
        print(f"✓ {args.caso.name}: no se contradice a si mismo")
        return 0
    print(f"✗ {args.caso.name}: {len(fallos)} incoherencia(s)")
    for f in fallos:
        print(f"  · {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
