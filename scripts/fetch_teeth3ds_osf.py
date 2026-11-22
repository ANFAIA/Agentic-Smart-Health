#!/usr/bin/env python
"""Pacientes de Teeth3DS+ que no están en el Google Drive: la reserva y más entrenamiento.

    uv run python scripts/fetch_teeth3ds_osf.py reserva
    uv run python scripts/fetch_teeth3ds_osf.py entrenamiento
    uv run python scripts/fetch_teeth3ds_osf.py reserva-2

**Por qué existe.** `fetch_teeth3ds.sh` baja los zips del Google Drive del reto, que son
300 pacientes: exactamente el `public-training-set-1` de 3DTeethSeg'22. Con ellos se
entrenó el modelo FDI (240) y se hicieron `dev` y `eval` (60), así que no quedaba ningún
paciente sin usar. El proyecto OSF `xctdy` aloja el dataset entero en siete zips: 900
pacientes etiquetados. Los otros 600 son el `public-training-set-2` y el
`private-testing-set` del reto, y ninguno ha pasado por este repositorio.

- `reserva`: 60 pacientes del `private-testing-set`, elegidos con semilla fija. **Nadie
  los ha visto**: ni nuestro modelo ni ningún modelo entrenado con los datos públicos del
  reto. Es el conjunto donde se mide una sola vez lo que se elija en `dev`.
- `reserva-2`: otros 60 del `private-testing-set`, elegidos con otra semilla entre los 240
  que la primera dejó. La primera reserva se gastó el 2026-11-20 (§6.10 de
  `segmentacion-fdi-escaner.md`) y pasa a ser desarrollo; esta es la que la sustituye.
- `entrenamiento`: los 300 del `public-training-set-2`, convertidos a npz (posiciones,
  normales y etiquetas) para no ocupar 10 GB de OBJ.

**Un zip cada vez.** El índice central de cada zip se lee por HTTP Range, sin bajarlo,
para saber en qué parte está cada paciente. Para extraer, en cambio, se baja el zip entero
de una sola petición, se saca lo que hace falta y se borra: medido, OSF limita el número de
peticiones (429 a los pocos minutos de leer paciente a paciente), no el ancho de banda. En
disco nunca hay más de un zip (1,5 GB como mucho). `zipfile` comprueba el CRC de cada
fichero al leerlo.

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

# ── Segunda reserva, fijada el 2026-11-20, después de gastar la primera y antes de bajar
# ni mirar ninguno de sus pacientes. Sale de los que la primera dejó en el private-testing-set.
SEMILLA_RESERVA_2 = 20261121
LISTA_RESERVA_2 = RAIZ / "docs/research/teeth3ds-reserva-2.txt"

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
        self.origen = url
        self._resuelve()
        self.pos = 0

    def _resuelve(self) -> None:
        """La URL de OSF redirige a una firmada que caduca: se vuelve a pedir si hace falta."""
        r = self.sesion.head(self.origen, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.tamano = r.url, int(r.headers["Content-Length"])

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
        import time

        # OSF limita la tasa (429) y a veces corta: se espera cada vez más y se reintenta.
        # Un 403 es la URL firmada caducada, y se resuelve de nuevo.
        estado = None
        for intento in range(12):
            try:
                r = self.sesion.get(self.url, timeout=300,
                                    headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                estado = r.status_code
                if estado == 206 and len(r.content) == n:
                    break
                if estado in (401, 403):
                    self._resuelve()
            except Exception:  # noqa: BLE001 — red: se reintenta y, si no, se dice
                pass
            time.sleep(min(2 ** intento, 120))
        else:
            raise OSError(f"OSF no sirvió el rango tras 12 intentos (último estado {estado})")
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


def ids_reserva_2() -> list[str]:
    return ids_de_fichero(LISTA_RESERVA_2)


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
    """La lista oficial, comprobada contra su sha256. Se guarda junto a los datos.

    OSF limita la tasa y, cuando lo hace, responde 429 con una página HTML: sin la
    comprobación del hash esa página pasaría por una lista vacía.
    """
    import time

    import requests

    url, sha = LISTAS[nombre]
    cache = DESTINO / "listas" / f"{nombre}.txt"
    if cache.is_file() and hashlib.sha256(cache.read_bytes()).hexdigest() == sha:
        return cache.read_text()
    for intento in range(10):
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            break
        time.sleep(min(2 ** intento, 120))
    if hashlib.sha256(r.content).hexdigest() != sha:
        raise SystemExit(f"⚠ la lista {nombre} de OSF ya no es la que se fijó "
                         f"(HTTP {r.status_code}); no sigo.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return r.content.decode()


# ── Extracción ──────────────────────────────────────────────────────────────────

def extrae(zf: zipfile.ZipFile, ruta_zip: str, destino: Path) -> str:
    """Escribe un miembro del zip y devuelve su sha256. `zipfile` comprueba el CRC."""
    datos = zf.read(ruta_zip)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(datos)
    return hashlib.sha256(datos).hexdigest()


def ficheros_del_paciente(idx: dict, paciente: str) -> dict[str, tuple[str, str]]:
    esperados = [f"{paciente}_{a}.{e}" for a in ARCADAS for e in ("obj", "json")]
    tiene = idx.get(paciente, {})
    faltan = [f for f in esperados if f not in tiene]
    if faltan:
        raise SystemExit(f"⚠ {paciente}: faltan {faltan} en OSF")
    return {f: tuple(tiene[f]) for f in esperados}  # type: ignore[misc]


def descarga_parte(parte: str, destino: Path) -> Path:
    """El zip entero de una parte, en UNA petición (reanudable).

    ⚠️ OSF limita las peticiones, no el ancho de banda: leer paciente a paciente por Range
    son unas diez por paciente y a los pocos minutos todo es 429. Un zip entero son 1,5 GB
    como mucho y se borra en cuanto se ha extraído lo que hacía falta.
    """
    import time

    import requests

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(".parcial")
    for intento in range(12):
        hecho = parcial.stat().st_size if parcial.is_file() else 0
        try:
            cabecera = {"Range": f"bytes={hecho}-"} if hecho else {}
            with requests.get(PARTES[parte], headers=cabecera, stream=True,
                              timeout=300) as r:
                if r.status_code == 416:
                    break
                if r.status_code not in (200, 206):
                    raise OSError(f"HTTP {r.status_code}")
                if r.status_code == 200:
                    hecho = 0
                with parcial.open("ab" if hecho else "wb") as f:
                    for trozo in r.iter_content(1 << 22):
                        f.write(trozo)
            break
        except Exception as e:  # noqa: BLE001 — red: se reintenta y, si no, se dice
            print(f"  {parte}: {e}; reintento {intento + 1}", flush=True)
            time.sleep(min(2 ** intento, 300))
    else:
        raise OSError(f"no se pudo bajar {parte}")
    parcial.replace(destino)
    return destino


def trae(reserva: list[str], entrenamiento: list[str],
         huellas_de: str = "reserva-sha256.json") -> None:
    """Baja cada parte que haga falta, extrae la reserva y convierte el entrenamiento."""
    if set(reserva) & set(entrenamiento):
        raise SystemExit("⚠ un paciente de la reserva está en el entrenamiento; no sigo.")
    idx = indice()
    pendiente: dict[str, list[tuple[str, str, str]]] = {}
    for paciente in reserva:
        for nombre, (parte, _ruta) in ficheros_del_paciente(idx, paciente).items():
            arbol = MALLAS if nombre.endswith(".obj") else ETIQUETAS
            if not (arbol / paciente / nombre).is_file():
                pendiente.setdefault(parte, []).append(("reserva", paciente, nombre))
    for paciente in entrenamiento:
        ficheros = ficheros_del_paciente(idx, paciente)
        for arcada in ARCADAS:
            if not (NPZ / f"{paciente}_{arcada}.npz").is_file():
                parte = ficheros[f"{paciente}_{arcada}.obj"][0]
                if ficheros[f"{paciente}_{arcada}.json"][0] != parte:
                    raise SystemExit(f"⚠ {paciente}_{arcada}: OBJ y JSON en partes distintas")
                pendiente.setdefault(parte, []).append(("entrenamiento", paciente, arcada))
    for parte in sorted(pendiente):
        tareas = pendiente[parte]
        print(f"{parte}: {len(tareas)} tareas; bajando el zip", flush=True)
        local = descarga_parte(parte, DESTINO / "zips" / f"{parte}.zip")
        with zipfile.ZipFile(local) as zf, tempfile.TemporaryDirectory() as tmp:
            for tipo, paciente, cosa in tareas:
                if tipo == "reserva":
                    arbol = MALLAS if cosa.endswith(".obj") else ETIQUETAS
                    extrae(zf, idx[paciente][cosa][1], arbol / paciente / cosa)
                    continue
                rutas = {}
                for ext in ("obj", "json"):
                    nombre = f"{paciente}_{cosa}.{ext}"
                    rutas[ext] = Path(tmp) / nombre
                    extrae(zf, idx[paciente][nombre][1], rutas[ext])
                a_npz(rutas["obj"], rutas["json"], NPZ / f"{paciente}_{cosa}.npz")
                for r in rutas.values():
                    r.unlink()
        local.unlink()
        print(f"{parte}: hecho", flush=True)
    if reserva:
        huellas = {f"{p}/{n}": hashlib.sha256(
            ((MALLAS if n.endswith(".obj") else ETIQUETAS) / p / n).read_bytes()).hexdigest()
            for p in reserva for n in ficheros_del_paciente(idx, p)}
        (DESTINO / huellas_de).write_text(json.dumps(huellas, indent=1) + "\n")


def reserva_derivada(solo_lista: bool = False) -> list[str]:
    """La reserva derivada de la lista oficial; tiene que coincidir con la versionada."""
    elegidos = elige_reserva(pacientes_de_lista(lista_oficial("private-testing-set")))
    if solo_lista or not LISTA_RESERVA.is_file():
        LISTA_RESERVA.write_text(
            "# Reserva de Teeth3DS+: 60 pacientes del private-testing-set de 3DTeethSeg'22,\n"
            "# elegidos por scripts/fetch_teeth3ds_osf.py (semilla 20261120). Ver §6.10 de\n"
            "# docs/research/segmentacion-fdi-escaner.md. Se evalúa sobre ellos UNA vez.\n"
            + "\n".join(elegidos) + "\n")
        print(f"→ {LISTA_RESERVA} ({len(elegidos)} pacientes)")
    elif ids_reserva() != elegidos:
        raise SystemExit("⚠ la reserva derivada no coincide con la versionada; no sigo.")
    return elegidos


def elige_reserva_2(privados: list[str], primera: list[str]) -> list[str]:
    """La segunda reserva: de los privados que NO están en la primera, con otra semilla."""
    if not set(primera) <= set(privados):
        raise ValueError("la primera reserva no sale de esta lista de privados")
    return elige_reserva(sorted(set(privados) - set(primera)), semilla=SEMILLA_RESERVA_2)


def reserva_2_derivada(solo_lista: bool = False) -> list[str]:
    """La segunda reserva derivada de la lista oficial; tiene que coincidir con la versionada."""
    elegidos = elige_reserva_2(pacientes_de_lista(lista_oficial("private-testing-set")),
                               reserva_derivada())
    if solo_lista or not LISTA_RESERVA_2.is_file():
        LISTA_RESERVA_2.write_text(
            "# Segunda reserva de Teeth3DS+: 60 pacientes del private-testing-set de\n"
            "# 3DTeethSeg'22 que no están en teeth3ds-reserva.txt, elegidos por\n"
            "# scripts/fetch_teeth3ds_osf.py (semilla 20261121). Ver §6.11 de\n"
            "# docs/research/segmentacion-fdi-escaner.md. Se evalúa sobre ellos UNA vez.\n"
            + "\n".join(elegidos) + "\n")
        print(f"→ {LISTA_RESERVA_2} ({len(elegidos)} pacientes)")
    elif ids_reserva_2() != elegidos:
        raise SystemExit("⚠ la segunda reserva derivada no coincide con la versionada; no sigo.")
    return elegidos


def entrenamiento_osf() -> list[str]:
    return pacientes_de_lista(lista_oficial("public-training-set-2"))


def baja(args: argparse.Namespace) -> int:
    if args.orden == "reserva-2":
        elegidos = reserva_2_derivada(solo_lista=args.solo_lista)
        if not args.solo_lista:
            trae(elegidos, [], huellas_de="reserva-2-sha256.json")
        return 0
    if args.orden == "reserva" and args.solo_lista:
        reserva_derivada(solo_lista=True)
        return 0
    reserva = reserva_derivada() if args.orden in ("reserva", "todo") else []
    entrena = entrenamiento_osf() if args.orden in ("entrenamiento", "todo") else []
    trae(reserva, entrena)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = ap.add_subparsers(dest="orden", required=True)
    r = ordenes.add_parser("reserva")
    r.add_argument("--solo-lista", action="store_true",
                   help="Escribe la lista versionada y no baja nada.")
    r2 = ordenes.add_parser("reserva-2")
    r2.add_argument("--solo-lista", action="store_true",
                    help="Escribe la lista versionada y no baja nada.")
    ordenes.add_parser("entrenamiento")
    ordenes.add_parser("todo", help="Las dos cosas, bajando cada zip una sola vez.")
    args = ap.parse_args()
    return baja(args)


if __name__ == "__main__":
    sys.exit(main())
