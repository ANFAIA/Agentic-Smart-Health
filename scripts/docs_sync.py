#!/usr/bin/env python
"""docs_sync.py — Comprueba que la documentación no le mienta al código.

    uv run python scripts/docs_sync.py --check    # falla si hay deriva (CI)
    uv run python scripts/docs_sync.py --write    # regenera los bloques generados

**Por qué existe.** Este repositorio documenta tanto como programa: README, AGENTS.md
y los ADR son producto, no adorno. Y la documentación se desincroniza en silencio —
nadie recibe un error rojo por escribir un número que dejó de ser cierto. Casos
reales encontrados a mano en este mismo repositorio: un recuento de tests que decía
166 cuando eran 265, un `.env.example` con cinco variables que no leía nadie, y el
árbol del README anunciando un notebook ya eliminado.

**Comprueba, no redacta.** Un agente que "arregle" la documentación por su cuenta
haría que el documento se adapte al código incluso cuando el que está mal es el
código: el 166 se habría convertido en 265 sin que nadie se enterase de que llevaba
meses mintiendo. Aquí solo se genera lo que es **copia mecánica** de una fuente de
verdad (la tabla de variables de entorno) y todo lo demás se **verifica**, con un
mensaje que dice qué línea miente y cómo arreglarla.

**Fuente de verdad = lo que git sigue.** Las comprobaciones se hacen contra
`git ls-files`, no contra el disco: si no, un fichero ignorado que existe en local
haría pasar en tu máquina algo que en CI falla.

Tres comprobaciones, las tres derivadas de fallos que ya ocurrieron:

| # | Comprueba | Fallo que habría cazado |
|---|---|---|
| `env` | variables leídas por el código ↔ `.env.example` ↔ README | `OPENAI_API_KEY` declarada y sin leer |
| `rutas` | ficheros citados en la documentación ↔ ficheros versionados | el README citando el notebook 08 eliminado |
| `agentes` | atributos `name` de las clases ↔ registro de `AGENTS.md` | un agente implementado y sin ficha |
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Variables que pone el entorno de ejecución, no la configuración del proyecto.
ENV_IGNORADAS = {"GITHUB_BASE_REF", "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "CI", "PATH", "HOME"}

# Prefijos cuyos ficheros DEBEN estar versionados: citar algo de aquí que no exista
# es una referencia rota. `data/` queda fuera a propósito (es contenido ignorado).
PREFIJOS_VERSIONADOS = ("packages/", "apps/", "scripts/", "docs/", "notebooks/", ".github/")

MARCA_INICIO = "<!-- generado: env-vars — no editar a mano -->"
MARCA_FIN = "<!-- /generado: env-vars -->"


def versionados() -> set[str]:
    salida = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout
    return set(salida.splitlines())


# --------------------------------------------------------------------------- #
# 1 · Variables de entorno
# --------------------------------------------------------------------------- #
def _constantes(arbol: ast.Module) -> dict[str, str]:
    """`_PSEUDONYM_SALT_ENV = "ASH_PSEUDONYM_SALT"` — para resolver la indirección."""
    fuera = {}
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and isinstance(nodo.value, ast.Constant) \
                and isinstance(nodo.value.value, str):
            for destino in nodo.targets:
                if isinstance(destino, ast.Name):
                    fuera[destino.id] = nodo.value.value
    return fuera


def _nombre_var(nodo: ast.expr, constantes: dict[str, str]) -> str | None:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
        return nodo.value
    if isinstance(nodo, ast.Name):
        return constantes.get(nodo.id)
    return None


def leidas_por_el_codigo(ficheros: set[str]) -> dict[str, tuple[str, str | None]]:
    """{VARIABLE: (fichero donde se lee, valor por defecto o None)}."""
    encontradas: dict[str, tuple[str, str | None]] = {}
    for ruta in sorted(f for f in ficheros if f.endswith(".py") and "/tests/" not in f):
        try:
            arbol = ast.parse((REPO / ruta).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        constantes = _constantes(arbol)
        for nodo in ast.walk(arbol):
            var = defecto = None
            if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute):
                atributo = nodo.func.attr
                objetivo = ast.unparse(nodo.func.value)
                if (atributo == "getenv" and objetivo == "os") or \
                        (atributo == "get" and objetivo == "os.environ"):
                    if nodo.args:
                        var = _nombre_var(nodo.args[0], constantes)
                    if len(nodo.args) > 1 and isinstance(nodo.args[1], ast.Constant):
                        defecto = str(nodo.args[1].value)
            elif isinstance(nodo, ast.Subscript) and ast.unparse(nodo.value) == "os.environ":
                var = _nombre_var(nodo.slice, constantes)
            if var and var not in ENV_IGNORADAS and var not in encontradas:
                encontradas[var] = (ruta, defecto)
    return encontradas


def declaradas_en_ejemplo() -> set[str]:
    texto = (REPO / ".env.example").read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^([A-Z_][A-Z0-9_]*)=", texto, re.M)}


def tabla_env(leidas: dict[str, tuple[str, str | None]]) -> str:
    filas = ["| Variable | Se lee en | Por defecto |", "|---|---|---|"]
    for var, (fichero, defecto) in sorted(leidas.items()):
        valor = f"`{defecto}`" if defecto else "—"
        filas.append(f"| `{var}` | `{fichero}` | {valor} |")
    return "\n".join(filas)


def revisar_env(ficheros: set[str], escribir: bool) -> list[str]:
    leidas = leidas_por_el_codigo(ficheros)
    declaradas = declaradas_en_ejemplo()
    problemas = []

    for var in sorted(declaradas - set(leidas)):
        problemas.append(
            f".env.example declara `{var}` y no la lee nadie. "
            "O la lee alguien y no lo hemos visto, o sobra en el fichero de ejemplo.")
    for var, (fichero, _) in sorted(leidas.items()):
        if var not in declaradas:
            problemas.append(
                f"`{fichero}` lee `{var}` y `.env.example` no la documenta. "
                "Quien clone el repositorio no puede saber que existe.")

    readme = REPO / "README.md"
    texto = readme.read_text(encoding="utf-8")
    if MARCA_INICIO in texto:
        patron = re.compile(re.escape(MARCA_INICIO) + r".*?" + re.escape(MARCA_FIN), re.S)
        esperado = f"{MARCA_INICIO}\n{tabla_env(leidas)}\n{MARCA_FIN}"
        if patron.search(texto).group(0) != esperado:
            if escribir:
                readme.write_text(patron.sub(lambda _: esperado, texto), encoding="utf-8")
                print("  · README: tabla de variables regenerada")
            else:
                problemas.append(
                    "La tabla de variables del README no coincide con el código. "
                    "Regenérala: `uv run python scripts/docs_sync.py --write`")
    return problemas


# --------------------------------------------------------------------------- #
# 2 · Rutas citadas en la documentación
# --------------------------------------------------------------------------- #
_ENLACE = re.compile(r"\]\(([^)\s#]+\.[a-zA-Z0-9]{1,6})\)")
_CODIGO = re.compile(r"`([^`\s]+\.[a-zA-Z0-9]{1,6})`")
# Marcadores de posición y comodines: `images/r_XXXX.png` no es un fichero.
_COMODIN = re.compile(r"[*{}]|XXX")


def revisar_rutas(ficheros: set[str]) -> list[str]:
    """Dos varas de medir, porque no toda cita es un enlace.

    Un **enlace** de Markdown promete que se puede pinchar: se resuelve contra el
    documento y tiene que existir. Una ruta **entre comillas** es prosa, y solo se
    verifica si está escrita desde la raíz del repositorio (`packages/…`): un
    `ingestion_agents/ontology.py` suelto habla de un módulo, no de una ruta, y
    exigirle que resuelva sería inventarse una convención que el repositorio no usa.
    """
    problemas = []
    for doc in sorted(f for f in ficheros if f.endswith(".md")):
        base = Path(doc).parent
        for numero, linea in enumerate((REPO / doc).read_text(encoding="utf-8").splitlines(), 1):
            citas = [(m.group(1), True) for m in _ENLACE.finditer(linea)]
            citas += [(m.group(1), False) for m in _CODIGO.finditer(linea)]
            for cita, es_enlace in citas:
                if cita.startswith(("http", "#", "mailto:")) or _COMODIN.search(cita):
                    continue
                if es_enlace:
                    destino = os.path.normpath(base / cita)
                else:
                    if not cita.startswith(PREFIJOS_VERSIONADOS):
                        continue
                    destino = os.path.normpath(cita)
                if destino.startswith("..") or destino.startswith("data/"):
                    continue
                if not destino.startswith(PREFIJOS_VERSIONADOS) and "/" in destino:
                    continue
                if destino not in ficheros:
                    problemas.append(
                        f"{doc}:{numero} cita `{cita}`, que no está versionado. "
                        "O se borró y la referencia quedó colgando, o falta añadirlo.")
    return problemas


# --------------------------------------------------------------------------- #
# 3 · Agentes implementados frente a los registrados
# --------------------------------------------------------------------------- #
def agentes_implementados(ficheros: set[str]) -> dict[str, str]:
    """{nombre: fichero} leyendo el atributo `name` de las clases *Agent."""
    fuera: dict[str, str] = {}
    for ruta in sorted(f for f in ficheros if f.endswith(".py") and "/src/" in f):
        try:
            arbol = ast.parse((REPO / ruta).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for nodo in ast.walk(arbol):
            if not (isinstance(nodo, ast.ClassDef) and nodo.name.endswith("Agent")):
                continue
            for cuerpo in nodo.body:
                if isinstance(cuerpo, ast.Assign) and isinstance(cuerpo.value, ast.Constant) \
                        and any(isinstance(t, ast.Name) and t.id == "name" for t in cuerpo.targets) \
                        and isinstance(cuerpo.value.value, str):
                    fuera[cuerpo.value.value] = ruta
    return fuera


def revisar_agentes(ficheros: set[str]) -> list[str]:
    texto = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    registrados = set(re.findall(r"`([a-z0-9]+(?:-[a-z0-9]+)+)`", texto))
    problemas = []
    for nombre, ruta in sorted(agentes_implementados(ficheros).items()):
        if nombre not in registrados:
            problemas.append(
                f"`{ruta}` implementa el agente `{nombre}` y AGENTS.md no lo registra. "
                "El registro es el contrato: un agente sin ficha es invisible para el resto.")
    return problemas


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--check", action="store_true", help="Falla si hay deriva (para CI).")
    modo.add_argument("--write", action="store_true", help="Regenera los bloques generados.")
    args = ap.parse_args()

    ficheros = versionados()
    problemas: list[str] = []
    for nombre, revision in (("variables de entorno", lambda: revisar_env(ficheros, args.write)),
                             ("rutas citadas", lambda: revisar_rutas(ficheros)),
                             ("registro de agentes", lambda: revisar_agentes(ficheros))):
        fallos = revision()
        estado = "✗" if fallos else "✓"
        print(f"{estado} {nombre}: {len(fallos) or 'sin deriva'}"
              f"{' problema(s)' if fallos else ''}")
        problemas += fallos

    if not problemas:
        print("\nLa documentación cuadra con el código.")
        return 0
    print("\n" + "\n".join(f"  - {p}" for p in problemas))
    if args.write:
        print("\nLo anterior no se puede regenerar solo: son decisiones, no copias.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
