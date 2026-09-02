#!/usr/bin/env python
"""fetch_knowledge_base.py — Materializa la knowledge base del `research-agent`.

Los PDF de referencia **no viven en el repositorio**: son 156 MiB de binarios y,
en 9 de los 17 casos, su licencia no permite redistribuirlos (arXiv perpetual,
actas de AAAI, CC BY-NC-ND, informes sin licencia). Lo que sí se versiona es el
**inventario** —`data/research-agent/knowledge_base/manifest.yaml`—, y este script
lo convierte en ficheros bajando cada documento **de su fuente original**.

    uv run python scripts/fetch_knowledge_base.py            # baja lo que falte
    uv run python scripts/fetch_knowledge_base.py --check    # solo comprueba, no baja
    uv run python scripts/fetch_knowledge_base.py --force    # vuelve a bajar todo

**Qué se puede automatizar y qué no.** arXiv y las URL directas se bajan solas.
Wiley devuelve 403 a cualquier cliente que no sea un navegador, y las actas de
AAAI no tienen URL estable de PDF: esos quedan como **descarga manual**, y el
script imprime el enlace en vez de fingir que funciona.

**El `sha256` avisa, no bloquea.** Es el del ejemplar con el que se trabajó. Si el
editor resube el PDF —arXiv lo hace al publicar una versión nueva— el hash cambia
sin que el documento sea otro. Un aviso es información; abortar sería ruido.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

RESUMEN_EN = "Materialises the research agent knowledge base."

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "research-agent" / "knowledge_base" / "manifest.yaml"

# Wiley y AAAI no sirven el PDF a un cliente automatizado: se declaran manuales
# en vez de dejar que el script falle con un 403 que no dice nada.
FUENTES_MANUALES = {"doi", "aaai"}
_AGENTE = "Mozilla/5.0 (compatible; agentic-smart-health/1.0; +fetch_knowledge_base.py)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url: str, destino: Path) -> None:
    """Baja a un fichero temporal y renombra: nunca deja un PDF a medias."""
    peticion = urllib.request.Request(url, headers={"User-Agent": _AGENTE})
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(peticion, timeout=120) as respuesta:
        tipo = respuesta.headers.get("Content-Type", "")
        datos = respuesta.read()
    if "pdf" not in tipo.lower() and not datos.startswith(b"%PDF"):
        raise ValueError(f"la respuesta no es un PDF (Content-Type: {tipo!r})")
    parcial.write_bytes(datos)
    parcial.rename(destino)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Solo informa de qué falta y qué no cuadra; no descarga.")
    ap.add_argument("--force", action="store_true",
                    help="Vuelve a descargar aunque el fichero ya exista.")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()

    manifiesto = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    destino_dir = REPO / manifiesto["directorio"]
    destino_dir.mkdir(parents=True, exist_ok=True)
    documentos = manifiesto["documentos"]

    ok = bajados = manuales = fallos = distintos = 0
    pendientes_manuales: list[tuple[str, str]] = []

    for doc in documentos:
        destino = destino_dir / doc["fichero"]
        etiqueta = doc["titulo"][:58]

        if destino.exists() and not args.force:
            actual = sha256(destino)
            if actual == doc["sha256"]:
                ok += 1
                continue
            distintos += 1
            print(f"[≠] {etiqueta}\n    el fichero difiere del inventariado "
                  f"(¿versión nueva del editor?): {actual[:12]}… vs {doc['sha256'][:12]}…")
            continue

        if doc["fuente"] in FUENTES_MANUALES:
            manuales += 1
            pendientes_manuales.append((doc["fichero"], doc["url"]))
            continue

        if args.check:
            print(f"[·] falta: {etiqueta}")
            continue

        try:
            descargar(doc["url"], destino)
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            fallos += 1
            print(f"[✗] {etiqueta}\n    {doc['url']}\n    {e}")
            continue

        obtenido = sha256(destino)
        bajados += 1
        aviso = "" if obtenido == doc["sha256"] else "  (sha256 distinto del inventariado)"
        print(f"[✓] {etiqueta}{aviso}")

    print(f"\n{len(documentos)} documentos · {ok} ya estaban · {bajados} descargados · "
          f"{distintos} con hash distinto · {manuales} manuales · {fallos} fallos")

    if pendientes_manuales:
        print("\nDescarga manual (el editor no la sirve a un script):")
        for fichero, url in pendientes_manuales:
            print(f"  {url}\n     guardar como: {fichero}")

    # Recordatorio de licencia: lo que se baja no se puede volver a publicar.
    no_redis = [d for d in documentos if not d.get("redistribuible", False)]
    if no_redis:
        print(f"\n{len(no_redis)} de estos documentos NO son redistribuibles "
              "(arXiv perpetual, AAAI, CC BY-NC-ND, sin licencia declarada).")
        print("Quedan en un directorio ignorado por git a propósito: no los subas.")

    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
