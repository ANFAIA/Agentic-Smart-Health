#!/usr/bin/env python
"""Pacientes de Teeth3DS+ que no están en el Google Drive: la reserva y más entrenamiento.

    uv run python scripts/fetch_teeth3ds_osf.py reserva
    uv run python scripts/fetch_teeth3ds_osf.py entrenamiento

**Por qué existe.** `fetch_teeth3ds.sh` baja los zips del Google Drive del reto, que son
300 pacientes: exactamente el `public-training-set-1` de 3DTeethSeg'22. Con ellos se
entrenó el modelo FDI (240) y se hicieron `dev` y `eval` (60), así que no quedaba ningún
paciente sin usar. El proyecto OSF `xctdy` aloja el dataset entero en siete zips: 900
pacientes etiquetados. Los otros 600 son el `public-training-set-2` y el
`private-testing-set` del reto, y ninguno ha pasado por este repositorio.

- `reserva`: 60 pacientes del `private-testing-set`, elegidos con semilla fija. **Nadie
  los ha visto**: ni nuestro modelo ni ningún modelo entrenado con los datos públicos del
  reto. Es el conjunto donde se mide una sola vez lo que se elija en `dev`.
- `entrenamiento`: los 300 del `public-training-set-2`, convertidos a npz (posiciones,
  normales y etiquetas) para no ocupar 10 GB de OBJ.

**Sin bajar los zips.** Pesan 1,3-1,5 GB cada uno y el disco no da. El índice central de
un zip está al final y cada fichero se puede leer por separado, así que se piden por HTTP
Range solo los bytes de los pacientes que hacen falta. `zipfile` comprueba el CRC de cada
uno al leerlo.

⚠️ La reserva va a un árbol APARTE (`data/raw/teeth3ds-osf/`). Meterla en
`data/raw/teeth3ds/` cambiaría la lista de pacientes de la que `benchmark_margen.particion`
reproduce la partición del entrenamiento, y metería pacientes de entrenamiento en `dev`
sin que nada avisara.

Licencia: CC-BY 4.0 (Ben-Hamadou et al., 2022), ver `docs/research/dataset-teeth3ds.md`.
Nada de esto se versiona salvo la lista de IDs de la reserva.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

RESUMEN_EN = "Fetches unused Teeth3DS+ patients from OSF by HTTP range, without full zips."
RAIZ = Path(__file__).resolve().parent.parent

# Los siete zips del proyecto OSF `xctdy` (componentes data_part_1 … data_part_7).
PARTES = {
    "data_part_1": "https://osf.io/download/qhprs/",
    "data_part_2": "https://osf.io/download/4pwnr/",
    "data_part_3": "https://osf.io/download/frwdp/",
    "data_part_4": "https://osf.io/download/2arn4/",
    "data_part_5": "https://osf.io/download/xrz5f/",
    "data_part_6": "https://osf.io/download/23hgq/",
    "data_part_7": "https://osf.io/download/u83ad/",
}
# Las listas oficiales del reto, con su sha256 fijado: si OSF las cambiara, la reserva
# saldría distinta y este script se niega a seguir en vez de elegir otra en silencio.
LISTAS = {
    "private-testing-set": ("https://osf.io/download/x7rfj/",
                            "7abce7d8c51c6c3f394efd5376ad5602d2fd4a61ce873e20ec91924e78b268b7"),
    "public-training-set-2": ("https://osf.io/download/jd2fc/",
                              "476323bb0da66577906b4f8c352438f301d8e8f8c168a4ec7d96fc8feb6aac60"),
}
ARCADAS = ("upper", "lower")

# ── Reserva, fijada el 2026-11-20 antes de bajar ni mirar ninguno de sus pacientes ──
SEMILLA_RESERVA = 20261120
PACIENTES_RESERVA = 60
LISTA_RESERVA = RAIZ / "docs/research/teeth3ds-reserva.txt"

DESTINO = RAIZ / "data/raw/teeth3ds-osf"
MALLAS = DESTINO / "3D_scans_per_patient_obj_files"
ETIQUETAS = DESTINO / "ground-truth_labels_instances"
NPZ = RAIZ / "data/processed/teeth3ds-osf-npz"


# ── Lectura por rangos ──────────────────────────────────────────────────────────

class FicheroRemoto(io.RawIOBase):
    """Un fichero HTTP que se lee por rangos, para que `zipfile` no lo baje entero."""

    def __init__(self, url: str, sesion=None) -> None:
        import requests

        self.sesion = sesion or requests.Session()
        r = self.sesion.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.tamano, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, desplazamiento: int, desde: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.tamano}[desde]
        self.pos = base + desplazamiento
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.tamano - self.pos
        n = min(n, self.tamano - self.pos)
        if n <= 0:
            return b""
        for intento in range(5):
            try:
                r = self.sesion.get(self.url, timeout=300,
                                    headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                if r.status_code == 206:
                    break
            except Exception:  # noqa: BLE001 — red: se reintenta y, si no, se dice
                if intento == 4:
                    raise
        else:
            raise OSError(f"el servidor no respeta Range ({r.status_code})")
        self.pos += len(r.content)
        return r.content

    def readinto(self, b) -> int:  # type: ignore[no-untyped-def]
        datos = self.read(len(b))
        b[: len(datos)] = datos
        return len(datos)


def indice_de(zf: zipfile.ZipFile, parte: str) -> dict[str, dict[str, tuple[str, str]]]:
    """`{paciente: {nombre_de_fichero: (parte, ruta_en_el_zip)}}` de un zip."""
    salida: dict[str, dict[str, tuple[str, str]]] = {}
    for info in zf.infolist():
        nombre = Path(info.filename).name
        if info.is_dir() or not nombre.endswith((".obj", ".json")):
            continue
        tallo = nombre.rsplit(".", 1)[0]
        if "_" not in tallo or tallo.rsplit("_", 1)[1] not in ARCADAS:
            continue
        salida.setdefault(tallo.rsplit("_", 1)[0], {})[nombre] = (parte, info.filename)
    return salida


def indice(cache: Path = DESTINO / "indice-osf.json") -> dict[str, dict[str, list[str]]]:
    if cache.is_file():
        return json.loads(cache.read_text())
    total: dict[str, dict[str, list[str]]] = {}
    for parte, url in PARTES.items():
        with zipfile.ZipFile(FicheroRemoto(url)) as zf:
            for paciente, ficheros in indice_de(zf, parte).items():
                total.setdefault(paciente, {}).update(
                    {k: list(v) for k, v in ficheros.items()})
        print(f"  índice {parte}: {len(total)} pacientes acumulados", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(total, indent=1, sort_keys=True) + "\n")
    return total


# ── Elección ────────────────────────────────────────────────────────────────────

def ids_de_fichero(ruta: Path) -> list[str]:
    """Los IDs de una lista versionada: una por línea; `#` empieza un comentario."""
    lineas = (x.split("#", 1)[0].strip() for x in ruta.read_text().splitlines())
    return [x for x in lineas if x]


def ids_reserva() -> list[str]:
    return ids_de_fichero(LISTA_RESERVA)


def pacientes_de_lista(texto: str) -> list[str]:
    """Pacientes con sus dos arcadas en una lista oficial (`ID_upper`, `ID_lower`)."""
    arcadas: dict[str, set[str]] = {}
    for linea in texto.split():
        if "_" in linea:
            p, a = linea.rsplit("_", 1)
            arcadas.setdefault(p, set()).add(a)
    return sorted(p for p, a in arcadas.items() if a == set(ARCADAS))


def elige_reserva(candidatos: list[str], n: int = PACIENTES_RESERVA,
                  semilla: int = SEMILLA_RESERVA) -> list[str]:
    """`n` pacientes sin reemplazo, de la lista ORDENADA, con semilla fija; salen ordenados."""
    ordenados = sorted(set(candidatos))
    if n > len(ordenados):
        raise ValueError(f"se piden {n} pacientes y hay {len(ordenados)}")
    idx = np.random.default_rng(semilla).choice(len(ordenados), size=n, replace=False)
    return sorted(ordenados[i] for i in idx)


def lista_oficial(nombre: str) -> str:
    import requests

    url, sha = LISTAS[nombre]
    datos = requests.get(url, timeout=60).content
    if hashlib.sha256(datos).hexdigest() != sha:
        raise SystemExit(f"⚠ la lista {nombre} de OSF ya no es la que se fijó; no sigo.")
    return datos.decode()


# ── Extracción ──────────────────────────────────────────────────────────────────

def extrae(zf: zipfile.ZipFile, ruta_zip: str, destino: Path) -> str:
    """Escribe un miembro del zip y devuelve su sha256. `zipfile` comprueba el CRC."""
    datos = zf.read(ruta_zip)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(datos)
    return hashlib.sha256(datos).hexdigest()


class Partes:
    """Un `ZipFile` remoto por parte, abierto la primera vez que hace falta."""

    def __init__(self) -> None:
        self.abiertas: dict[str, zipfile.ZipFile] = {}

    def __getitem__(self, parte: str) -> zipfile.ZipFile:
        if parte not in self.abiertas:
            self.abiertas[parte] = zipfile.ZipFile(FicheroRemoto(PARTES[parte]))
        return self.abiertas[parte]


def ficheros_del_paciente(idx: dict, paciente: str) -> dict[str, tuple[str, str]]:
    esperados = [f"{paciente}_{a}.{e}" for a in ARCADAS for e in ("obj", "json")]
    tiene = idx.get(paciente, {})
    faltan = [f for f in esperados if f not in tiene]
    if faltan:
        raise SystemExit(f"⚠ {paciente}: faltan {faltan} en OSF")
    return {f: tuple(tiene[f]) for f in esperados}  # type: ignore[misc]


def reserva(args: argparse.Namespace) -> int:
    elegidos = elige_reserva(pacientes_de_lista(lista_oficial("private-testing-set")))
    versionada = ids_reserva() if LISTA_RESERVA.is_file() else None
    if args.solo_lista or versionada is None:
        LISTA_RESERVA.write_text(
            "# Reserva de Teeth3DS+: 60 pacientes del private-testing-set de 3DTeethSeg'22,\n"
            "# elegidos por scripts/fetch_teeth3ds_osf.py (semilla 20261120). Ver §6.10 de\n"
            "# docs/research/segmentacion-fdi-escaner.md. Se evalúa sobre ellos UNA vez.\n"
            + "\n".join(elegidos) + "\n")
        print(f"→ {LISTA_RESERVA} ({len(elegidos)} pacientes)")
        if args.solo_lista:
            return 0
    elif versionada != elegidos:
        raise SystemExit("⚠ la reserva derivada no coincide con la versionada; no sigo.")
    idx, partes, huellas = indice(), Partes(), {}
    for k, paciente in enumerate(elegidos, 1):
        for nombre, (parte, ruta) in ficheros_del_paciente(idx, paciente).items():
            arbol = MALLAS if nombre.endswith(".obj") else ETIQUETAS
            destino = arbol / paciente / nombre
            if not destino.is_file():
                huellas[nombre] = extrae(partes[parte], ruta, destino)
            else:
                huellas[nombre] = hashlib.sha256(destino.read_bytes()).hexdigest()
        print(f"  [{k}/{len(elegidos)}] {paciente}", flush=True)
    (DESTINO / "reserva-sha256.json").write_text(json.dumps(huellas, indent=1) + "\n")
    return 0


def a_npz(obj: Path, etiquetas: Path, destino: Path) -> int:
    """Posiciones y normales como las lee el ENTRENAMIENTO (vtk, sin split), y etiquetas."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    vtk.vtkObject.GlobalWarningDisplayOff()
    r = vtk.vtkOBJReader()
    r.SetFileName(str(obj))
    r.Update()
    poly = r.GetOutput()
    pos = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(poly)
    nf.SetSplitting(0)
    nf.SetConsistency(1)
    nf.SetComputePointNormals(1)
    nf.SetComputeCellNormals(0)
    nf.Update()
    nrm = vtk_to_numpy(nf.GetOutput().GetPointData().GetNormals()).astype(np.float32)
    lab = np.asarray(json.loads(etiquetas.read_text())["labels"], dtype=np.int16)
    if not len(lab) == len(pos) == len(nrm):
        raise ValueError(f"{obj.name}: {len(lab)} etiquetas, {len(pos)} vértices")
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino, pos=pos, nrm=nrm, labels=lab)
    return len(pos)


def entrenamiento(args: argparse.Namespace) -> int:
    pacientes = pacientes_de_lista(lista_oficial("public-training-set-2"))
    if args.limite:
        pacientes = pacientes[: args.limite]
    reservados = set(ids_reserva()) if LISTA_RESERVA.is_file() else set()
    if reservados & set(pacientes):
        raise SystemExit("⚠ un paciente de la reserva está en el entrenamiento; no sigo.")
    idx, partes = indice(), Partes()
    with tempfile.TemporaryDirectory() as tmp:
        for k, paciente in enumerate(pacientes, 1):
            ficheros = ficheros_del_paciente(idx, paciente)
            for arcada in ARCADAS:
                destino = NPZ / f"{paciente}_{arcada}.npz"
                if destino.is_file():
                    continue
                rutas = {}
                for ext in ("obj", "json"):
                    parte, ruta = ficheros[f"{paciente}_{arcada}.{ext}"]
                    rutas[ext] = Path(tmp) / f"{paciente}_{arcada}.{ext}"
                    extrae(partes[parte], ruta, rutas[ext])
                a_npz(rutas["obj"], rutas["json"], destino)
                for r in rutas.values():
                    r.unlink()
            print(f"  [{k}/{len(pacientes)}] {paciente}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    r = ordenes.add_parser("reserva")
    r.add_argument("--solo-lista", action="store_true",
                   help="Escribe la lista versionada y no baja nada.")
    r.set_defaults(funcion=reserva)
    e = ordenes.add_parser("entrenamiento")
    e.add_argument("--limite", type=int)
    e.set_defaults(funcion=entrenamiento)
    args = ap.parse_args()
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
