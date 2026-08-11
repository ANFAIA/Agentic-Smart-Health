#!/usr/bin/env python
"""data_guard.py — Impide que datos ajenos entren al repositorio sin permiso.

    uv run python scripts/data_guard.py            # informe completo
    uv run python scripts/data_guard.py --quiet    # solo los problemas (CI)

**Por qué existe.** La issue 45 se abrió porque 156 MiB de PDF de terceros —nueve
de ellos con licencia que prohíbe redistribuirlos— llevaban meses versionados sin
que nadie lo notase. Limpiarlo costó reescribir la historia con `git filter-repo`
y un `force-push`: una operación cara e irreversible que ninguna revisión humana
habría hecho falta si un `grep` hubiese fallado el día del primer `git add`.

**El `.gitignore` no basta, y ese es el punto.** Ignorar un patrón evita el
descuido, no el `git add -f`, ni el fichero que ya estaba dentro antes de escribir
la regla, ni la regla que alguien borra en un merge. Este guardián comprueba el
resultado —lo que git *sigue de verdad*— y además verifica que las reglas siguen
vivas, preguntándoselo a git en vez de leyendo el fichero.

**Las comprobaciones, todas nacidas de un fallo real de este repositorio** (el
registro vivo es `COMPROBACIONES`, ahí abajo; `docs_sync` compara sus identificadores
con la tabla de la ficha, así que añadir una y olvidar documentarla se cae en el CI):

- `extensiones`: ningún binario de datos versionado. Es la que habría cazado la 45.
- `tamano`: nada por encima del límite. Un dataset entra por peso antes que por
  extensión, y la historia de git no olvida.
- `ignore`: las rutas protegidas siguen ignorándose. Si alguien borra una
  línea del `.gitignore`, salta aquí y no tres meses después.
- `manifiesto`: el inventario de la knowledge base está completo, con licencia
  declarada y sin apuntar a ficheros que estén versionados.
- `procedencia`: un notebook versionado que lleva imágenes dentro tiene que citar
  un dataset con ficha. Los renders de una cohorte son un derivado del dato, y
  viajan escondidos en base64 donde ninguna regla de extensión los ve.
- `fichas`: toda ficha de dataset declara su licencia. Sin eso no se puede saber
  si un derivado se puede publicar — que fue exactamente la duda de Bite2Text.

**Determinista y sin LLM**: no hay nada que interpretar. O el fichero está
versionado o no lo está.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
MANIFIESTO = REPO / "data" / "research-agent" / "knowledge_base" / "manifest.yaml"

# Formatos de dato clínico, malla, volumen o modelo. Ninguno se versiona: se
# regeneran desde su origen o se descargan con su script.
_VETADAS = (
    ".pdf "  # documentos de terceros
    ".ply .splat .stl .obj .glb .gltf "  # mallas y nubes de puntos
    ".dcm .nii .gz .nrrd .mha .mhd .mnc .img .hdr "  # volúmenes clínicos
    ".npz .npy .ckpt .pt .pth .safetensors"  # pesos y artefactos de ML
)
EXTENSIONES_VETADAS = frozenset(_VETADAS.split())

# 2 MiB. El fichero versionado más grande hoy es un notebook de 1,4 MiB, así que
# el límite deja margen para trabajo normal y corta un dataset en seco.
LIMITE_BYTES = 2 * 1024 * 1024

# Rutas sonda: no existen: se le pregunta a git si *las ignoraría*. Comprobar el
# comportamiento y no el texto del `.gitignore` sobrevive a que alguien reordene
# las reglas, y detecta que las ha borrado.
SONDAS_IGNORADAS = (
    "data/raw/cohorte.npy",
    "data/processed/nube.ply",
    "data/external/tercero.zip",
    "data/interim/salida.npz",
    "data/research-agent/knowledge_base/paper.pdf",
    "notebooks/malla.stl",
    "estudio.dcm",
    "volumen.nii.gz",
)

CAMPOS_MANIFIESTO = ("fichero", "titulo", "fuente", "url", "licencia", "redistribuible", "sha256")


def versionados() -> dict[str, int]:
    """Fichero versionado -> tamaño en bytes. Fuente de verdad: lo que git sigue."""
    salida = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    rutas = [r for r in salida.split("\0") if r]
    tamanos = {}
    for ruta in rutas:
        fichero = REPO / ruta
        tamanos[ruta] = fichero.stat().st_size if fichero.exists() else 0
    return tamanos


def revisar_extensiones(ficheros: dict[str, int]) -> list[str]:
    problemas = []
    for ruta in sorted(ficheros):
        sufijos = Path(ruta).suffixes
        # `.nii.gz` y compañía: se mira el último sufijo y también el penúltimo.
        vetada = next((s for s in reversed(sufijos[-2:]) if s.lower() in EXTENSIONES_VETADAS), None)
        if vetada:
            problemas.append(
                f"`{ruta}` está versionado y es un `{vetada}`. Ese formato no se sube: "
                "si es dato, va a `data/` (ignorado) y se regenera; si es un PDF de "
                "referencia, va al `manifest.yaml` y lo baja `fetch_knowledge_base.py`."
            )
    return problemas


def revisar_tamano(ficheros: dict[str, int]) -> list[str]:
    problemas = []
    for ruta, tam in sorted(ficheros.items(), key=lambda f: -f[1]):
        if tam > LIMITE_BYTES:
            problemas.append(
                f"`{ruta}` pesa {tam / 1048576:.1f} MiB y el límite son "
                f"{LIMITE_BYTES / 1048576:.0f} MiB. Un binario entra en la historia de git "
                "para siempre: sacarlo obliga a reescribirla."
            )
    return problemas


def revisar_reglas_ignore() -> list[str]:
    problemas = []
    for sonda in SONDAS_IGNORADAS:
        ignorada = subprocess.run(["git", "check-ignore", "-q", sonda], cwd=REPO).returncode == 0
        if not ignorada:
            problemas.append(
                f"`{sonda}` ya NO está ignorada: alguien quitó o rompió su regla en "
                "`.gitignore`. Mientras siga así, un `git add .` mete ese dato en el repo."
            )
    return problemas


def revisar_manifiesto(ficheros: dict[str, int]) -> list[str]:
    if not MANIFIESTO.exists():
        return [
            f"Falta `{MANIFIESTO.relative_to(REPO)}`: sin inventario no se sabe qué "
            "documentos usa el research-agent ni bajo qué licencia."
        ]

    problemas = []
    datos = yaml.safe_load(MANIFIESTO.read_text(encoding="utf-8"))
    documentos = datos.get("documentos") or []
    directorio = datos.get("directorio", "")
    vistos: set[str] = set()

    for i, doc in enumerate(documentos, 1):
        etiqueta = doc.get("fichero") or doc.get("titulo") or f"entrada {i}"
        for campo in CAMPOS_MANIFIESTO:
            if doc.get(campo) in (None, ""):
                problemas.append(f"`{etiqueta}`: le falta el campo `{campo}` en el manifiesto.")
        if not isinstance(doc.get("redistribuible"), bool):
            problemas.append(
                f"`{etiqueta}`: `redistribuible` tiene que ser `true` o `false`. "
                "Es el campo que decide si el documento se puede volver a publicar."
            )
        sha = str(doc.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            problemas.append(f"`{etiqueta}`: `sha256` no es un hash hexadecimal de 64 caracteres.")
        if not str(doc.get("url", "")).startswith(("http://", "https://")):
            problemas.append(f"`{etiqueta}`: `url` no apunta a un origen descargable.")
        if doc.get("fichero") in vistos:
            problemas.append(f"`{etiqueta}`: aparece dos veces en el manifiesto.")
        vistos.add(doc.get("fichero"))

        # Lo que el manifiesto inventaría NO puede estar además versionado: el
        # inventario existe precisamente para no guardar el binario.
        ruta = f"{directorio}/{doc.get('fichero')}"
        if ruta in ficheros:
            problemas.append(
                f"`{ruta}` está versionado y a la vez inventariado. El manifiesto sustituye "
                "al fichero, no lo acompaña."
            )
    return problemas


def _fichas_dataset() -> dict[str, Path]:
    """Slug -> ficha, leyendo `docs/research/dataset-*.md`."""
    return {
        p.name[len("dataset-") : -len(".md")]: p
        for p in sorted((REPO / "docs" / "research").glob("dataset-*.md"))
    }


def revisar_procedencia(ficheros: dict[str, int]) -> list[str]:
    """Notebook versionado con imágenes dentro ↔ dataset con ficha.

    Las salidas de un notebook llevan PNG en base64. Ahí puede viajar el render de
    una cohorte con acuerdo de uso sin que ninguna regla de extensión lo vea, que
    es como estuvimos a un `git add` de publicar imágenes de un CBCT ajeno.
    """
    problemas = []
    slugs = _fichas_dataset()
    for ruta in sorted(r for r in ficheros if r.endswith(".ipynb")):
        try:
            cuaderno = json.loads((REPO / ruta).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            problemas.append(f"`{ruta}` no se puede leer como notebook: {e}")
            continue

        imagenes, texto = 0, []
        for celda in cuaderno.get("cells", []):
            texto.append("".join(celda.get("source", [])))
            for salida in celda.get("outputs", []):
                imagenes += sum(1 for k in (salida.get("data") or {}) if k.startswith("image/"))
        if not imagenes:
            continue

        cuerpo = " ".join(texto).lower()
        if not any(slug.lower() in cuerpo for slug in slugs):
            problemas.append(
                f"`{ruta}` lleva {imagenes} imagen(es) embebida(s) y no cita ningún dataset "
                f"con ficha en `docs/research/` (hay: {', '.join(sorted(slugs)) or 'ninguna'}). "
                "Un render es un derivado del dato: sin ficha no se sabe si se puede publicar."
            )
    return problemas


def revisar_fichas() -> list[str]:
    problemas = []
    for slug, ficha in _fichas_dataset().items():
        texto = ficha.read_text(encoding="utf-8").lower()
        if "licencia" not in texto:
            problemas.append(
                f"`{ficha.relative_to(REPO)}` no declara licencia. Es el dato que decide "
                f"si se puede publicar un derivado del dataset `{slug}`."
            )
    return problemas


# --------------------------------------------------------------------------- #
# El registro: identificador, título para la persona que lo lee, y la función.
#
# El identificador va aparte del título a propósito. El título se escribe para quien
# mira la salida y puede reformularse sin consecuencias; el identificador es el que
# aparece en la ficha de `AGENTS.md` y el que compara `docs_sync`, así que es un
# nombre estable y sin acentos, no una frase.
COMPROBACIONES: tuple[tuple[str, str, Callable[[dict[str, int]], list[str]]], ...] = (
    ("extensiones", "extensiones vetadas", revisar_extensiones),
    ("tamano", "tamaño de los ficheros", revisar_tamano),
    ("ignore", "reglas de ignore vivas", lambda _: revisar_reglas_ignore()),
    ("manifiesto", "manifiesto de la knowledge base", revisar_manifiesto),
    ("procedencia", "procedencia de los notebooks", revisar_procedencia),
    ("fichas", "fichas de dataset", lambda _: revisar_fichas()),
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quiet", action="store_true", help="Imprime solo los problemas (para CI).")
    args = ap.parse_args()

    ficheros = versionados()
    problemas: list[str] = []
    for _, nombre, revision in COMPROBACIONES:
        fallos = revision(ficheros)
        if not args.quiet or fallos:
            estado = "✗" if fallos else "✓"
            detalle = f"{len(fallos)} problema(s)" if fallos else "limpio"
            print(f"{estado} {nombre}: {detalle}")
        problemas += fallos

    if not problemas:
        if not args.quiet:
            print(f"\n{len(ficheros)} ficheros versionados, ninguno es un dato ajeno.")
        return 0
    print("\n" + "\n\n".join(f"  - {p}" for p in problemas))
    print(
        "\nSi alguno es un falso positivo, la regla se cambia aquí y se razona en el "
        "commit: no se sortea con `git add -f`."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
