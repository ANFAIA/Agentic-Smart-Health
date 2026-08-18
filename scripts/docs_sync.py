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

Las comprobaciones, todas derivadas de fallos que ya ocurrieron:

- `env`: variables leídas por el código ↔ `.env.example` ↔ README.
  Habría cazado el `OPENAI_API_KEY` declarado que no leía nadie.
- `rutas`: ficheros citados en la documentación ↔ ficheros versionados.
  Habría cazado el README citando un notebook ya eliminado.
- `agentes`: atributos `name` de las clases ↔ registro de `AGENTS.md`.
  Habría cazado un agente implementado y sin ficha.
- `versiones`: atributo `version` de la clase ↔ la fila **Versión** de su ficha.
  Habría cazado el `geometric-fusion-agent` subido a 0.2.0 el 2026-08-10 con la
  ficha describiendo todavía el comportamiento de la 0.1.0.
- `guardianes`: scripts que ejecutan los workflows o los hooks ↔ ficha en `AGENTS.md`.
  Habría cazado los tres guardianes que bloqueaban commits y abrían PRs sin ficha.
- `comprobaciones`: el registro `COMPROBACIONES` de cada guardián ↔ la tabla de su
  ficha. Es lo que sustituye a versionar los guardianes: de un guardián no importa su
  número de versión —que nadie lee— sino qué vigila, y eso sí se desincroniza.
- `constantes`: un número **o una cadena** escritos en la prosa ↔ el valor de la constante
  de la que salen, atados por un marcador `<!--const:NOMBRE-->`. Habría cazado la ficha de
  fusión diciendo «ε = 0,5 mm» después de que el código dijera otra cosa, y el README
  anunciando el esquema `1.2.0` con el contrato ya en `1.3.0` — que llevaba meses así.
- `inventario` y `arbol`: que lo que existe esté citado, no solo que lo citado exista.

**Lo que sigue sin cubrirse, y conviene saberlo.** Una afirmación en prosa **sin
número** que se vuelve falsa. La ficha de fusión decía también «la etapa gruesa queda
pendiente» mucho después de que dejara de estarlo, y eso no hay constante que lo
respalde: no se puede mecanizar sin decidir qué significa que una frase siga siendo
cierta. La mitad numérica de aquel caso sí está cubierta desde la 0.4.0.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
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
    salida = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return set(salida.splitlines())


# --------------------------------------------------------------------------- #
# Bloques generados: lo que es copia mecanica de una fuente de verdad
# --------------------------------------------------------------------------- #
def _sincronizar(doc: Path, ident: str, contenido: str, escribir: bool) -> str | None:
    """Deja el bloque `ident` de `doc` igual a `contenido`, o explica que no lo esta.

    El bloque es opcional: si el documento no lo declara, no se inventa. Poner las
    marcas es una decision editorial de quien escribe la pagina.
    """
    inicio, fin = f"<!-- generado: {ident} — no editar a mano -->", f"<!-- /generado: {ident} -->"
    texto = doc.read_text(encoding="utf-8")
    if inicio not in texto:
        return None
    patron = re.compile(re.escape(inicio) + r".*?" + re.escape(fin), re.S)
    esperado = f"{inicio}\n{contenido}\n{fin}"
    if patron.search(texto).group(0) == esperado:
        return None
    if escribir:
        doc.write_text(patron.sub(lambda _: esperado, texto), encoding="utf-8")
        print(f"  · {doc.name}: bloque `{ident}` regenerado")
        return None
    return (
        f"El bloque `{ident}` de {doc.relative_to(REPO)} no coincide con el codigo. "
        "Regeneralo: `uv run python scripts/docs_sync.py --write`"
    )


def _primera_linea_doc(ruta: str) -> str:
    """Docstring del modulo, o primer comentario util de un script de shell."""
    texto = (REPO / ruta).read_text(encoding="utf-8")
    if ruta.endswith(".py"):
        try:
            doc = ast.get_docstring(ast.parse(texto)) or ""
        except SyntaxError:
            doc = ""
        primera = doc.strip().splitlines()[0] if doc.strip() else ""
        return primera.split("—", 1)[-1].strip() if "—" in primera else primera
    for linea in texto.splitlines():
        if linea.startswith("#") and not linea.startswith("#!"):
            limpia = linea.lstrip("# ").strip()
            if limpia:
                return limpia.split("—", 1)[-1].strip() if "—" in limpia else limpia
    return ""


def tabla_scripts(ficheros: set[str]) -> str:
    filas = ["| Script | Qué hace |", "|---|---|"]
    for ruta in sorted(
        f for f in ficheros if f.startswith("scripts/") and f.endswith((".py", ".sh"))
    ):
        filas.append(f"| [`{ruta}`]({ruta}) | {_primera_linea_doc(ruta) or '—'} |")
    return "\n".join(filas)


def tabla_make() -> str:
    """Objetivo -> receta real. Mas honesto que una descripcion a mano."""
    filas = ["| Comando | Ejecuta |", "|---|---|"]
    objetivo = None
    recetas: dict[str, list[str]] = {}
    for linea in (REPO / "Makefile").read_text(encoding="utf-8").splitlines():
        if re.match(r"^[a-zA-Z][\w-]*:", linea):
            objetivo = linea.split(":", 1)[0]
            if objetivo != ".PHONY":
                recetas.setdefault(objetivo, [])
        elif linea.startswith("\t") and objetivo in recetas:
            recetas[objetivo].append(linea.strip())
    for objetivo, ordenes in recetas.items():
        filas.append(f"| `make {objetivo}` | `{' && '.join(ordenes) or '—'}` |")
    return "\n".join(filas)


def tabla_agentes(ficheros: set[str]) -> str:
    """Resumen, NO las fichas: inputs, outputs y herramientas son prosa."""
    filas = ["| Agente | Implementado en |", "|---|---|"]
    for nombre, (ruta, _) in sorted(agentes_implementados(ficheros).items()):
        filas.append(f"| `{nombre}` | [`{ruta}`]({ruta}) |")
    return "\n".join(filas)


def sincronizar_bloques(ficheros: set[str], escribir: bool) -> list[str]:
    leidas = leidas_por_el_codigo(ficheros)
    bloques = (
        (REPO / "README.md", "env-vars", tabla_env(leidas)),
        (REPO / "README.md", "scripts", tabla_scripts(ficheros)),
        (REPO / "README.md", "make", tabla_make()),
        (REPO / "AGENTS.md", "agentes", tabla_agentes(ficheros)),
    )
    return [p for p in (_sincronizar(*b, escribir) for b in bloques) if p]


# --------------------------------------------------------------------------- #
# 1 · Variables de entorno
# --------------------------------------------------------------------------- #
def _constantes(arbol: ast.Module) -> dict[str, str]:
    """`_PSEUDONYM_SALT_ENV = "ASH_PSEUDONYM_SALT"` — para resolver la indirección."""
    fuera = {}
    for nodo in arbol.body:
        if (
            isinstance(nodo, ast.Assign)
            and isinstance(nodo.value, ast.Constant)
            and isinstance(nodo.value.value, str)
        ):
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
                if (atributo == "getenv" and objetivo == "os") or (
                    atributo == "get" and objetivo == "os.environ"
                ):
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


def revisar_env(ficheros: set[str]) -> list[str]:
    leidas = leidas_por_el_codigo(ficheros)
    declaradas = declaradas_en_ejemplo()
    problemas = []

    for var in sorted(declaradas - set(leidas)):
        problemas.append(
            f".env.example declara `{var}` y no la lee nadie. "
            "O la lee alguien y no lo hemos visto, o sobra en el fichero de ejemplo."
        )
    for var, (fichero, _) in sorted(leidas.items()):
        if var not in declaradas:
            problemas.append(
                f"`{fichero}` lee `{var}` y `.env.example` no la documenta. "
                "Quien clone el repositorio no puede saber que existe."
            )

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
                if destino.startswith(".."):
                    if es_enlace:
                        problemas.append(
                            f"{doc}:{numero} enlaza a `{cita}`, que sale del repositorio. "
                            "Un enlace relativo de mas es un enlace roto."
                        )
                    continue
                if destino.startswith("data/"):
                    continue
                if not destino.startswith(PREFIJOS_VERSIONADOS) and "/" in destino:
                    continue
                if destino not in ficheros:
                    problemas.append(
                        f"{doc}:{numero} cita `{cita}`, que no está versionado. "
                        "O se borró y la referencia quedó colgando, o falta añadirlo."
                    )
    return problemas


# --------------------------------------------------------------------------- #
# 3 · Agentes implementados frente a los registrados
# --------------------------------------------------------------------------- #
def agentes_implementados(ficheros: set[str]) -> dict[str, tuple[str, str]]:
    """{nombre: (fichero, versión)} leyendo `name` y `version` de las clases *Agent."""
    fuera: dict[str, tuple[str, str]] = {}
    for ruta in sorted(f for f in ficheros if f.endswith(".py") and "/src/" in f):
        try:
            arbol = ast.parse((REPO / ruta).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for nodo in ast.walk(arbol):
            if not (isinstance(nodo, ast.ClassDef) and nodo.name.endswith("Agent")):
                continue
            atributos: dict[str, str] = {}
            for cuerpo in nodo.body:
                if not (
                    isinstance(cuerpo, ast.Assign)
                    and isinstance(cuerpo.value, ast.Constant)
                    and isinstance(cuerpo.value.value, str)
                ):
                    continue
                for objetivo in cuerpo.targets:
                    if isinstance(objetivo, ast.Name) and objetivo.id in ("name", "version"):
                        atributos[objetivo.id] = cuerpo.value.value
            if "name" in atributos:
                fuera[atributos["name"]] = (ruta, atributos.get("version", ""))
    return fuera


def _ficha_de(texto: str, nombre: str) -> str | None:
    """La ficha de `nombre`: el bloque `###` que lo menciona y declara versión."""
    for seccion in re.split(r"\n(?=### )", texto):
        if f"`{nombre}`" in seccion and "**Versión**" in seccion:
            return seccion
    return None


def _version_declarada(ficha: str, nombre: str) -> str | None:
    """El texto de la fila `**Versión**`, acotado a este agente si la comparte.

    Una ficha puede documentar varios agentes con versiones distintas (los de fusión).
    Cuando la fila los nombra, se recorta desde la mención de este agente hasta la del
    siguiente, para no dar por buena la versión del vecino.
    """
    fila = re.search(r"\|\s*\*\*Versión\*\*\s*\|(.+?)\|\s*$", ficha, re.M)
    if fila is None:
        return None
    celda = fila.group(1)
    marca = f"`{nombre}`"
    if marca in celda:
        resto = celda.split(marca, 1)[1]
        siguiente = re.search(r"`[a-z0-9-]+`", resto)
        celda = resto[: siguiente.start()] if siguiente else resto
    return celda


def guardianes_autonomos(ficheros: set[str]) -> dict[str, list[str]]:
    """{script: [quién lo dispara]} para lo que corre **sin que nadie lo pida**.

    La definición operativa de «agente» que se usa aquí: un script que ejecutan los
    workflows o los hooks corre sin que una persona lo teclee, así que actúa solo y
    necesita ficha. Uno que solo se lanza a mano —`resolucion_modalidades.py`,
    `blender_render_views.py`— es una herramienta, y no la necesita.
    """
    disparadores = sorted(
        f for f in ficheros if f.startswith((".github/workflows/", ".githooks/"))
    )
    fuera: dict[str, list[str]] = {}
    for disparador in disparadores:
        try:
            texto = (REPO / disparador).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for script in sorted(set(re.findall(r"scripts/[\w-]+\.(?:py|sh)", texto))):
            if script in ficheros:
                fuera.setdefault(script, []).append(disparador)
    return fuera


def revisar_guardianes(ficheros: set[str], texto: str) -> list[str]:
    """Que todo script que corre solo esté registrado en `AGENTS.md`.

    El hueco que cerró: `data_guard.py`, `docs_sync.py` y `watch_literature.py` llevaban
    semanas bloqueando commits, verificando documentación y abriendo pull requests **sin
    ficha**, mientras la cabecera de `AGENTS.md` pedía registrar «cada agente autónomo o
    semi-autónomo». No lo cazaba nadie porque `revisar_agentes` solo mira clases con
    atributo `name`, y a un script suelto se le escapa.

    Basta con que la ruta aparezca en el documento: un agente puede compartir ficha con
    otro —`audit_pr.py` vive dentro de la del `ai-code-reviewer`— y exigirle sección
    propia sería imponer un formato, no comprobar un hecho.
    """
    problemas = []
    for script, disparadores in sorted(guardianes_autonomos(ficheros).items()):
        if script in texto:
            continue
        quien = ", ".join(disparadores)
        problemas.append(
            f"`{script}` lo ejecuta {quien} —o sea que corre sin que nadie lo pida— y "
            "AGENTS.md no lo registra. Un agente sin ficha es invisible para el resto."
        )
    return problemas


def comprobaciones_del_codigo(ruta: str) -> list[str]:
    """Los identificadores del registro `COMPROBACIONES` de un guardián.

    Se lee con `ast` y no importando el módulo: `scripts/` no es un paquete instalado,
    y aunque lo fuera, importar un script para inspeccionarlo ejecuta su cuerpo.

    Declarar el registro es voluntario. Un guardián que no lo tenga —`audit_pr.py`, que
    delega el criterio en un modelo— devuelve la lista vacía y queda fuera de la
    comprobación, en vez de obligar a todos a la misma forma.
    """
    try:
        arbol = ast.parse((REPO / ruta).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    for nodo in arbol.body:
        if isinstance(nodo, ast.AnnAssign) and isinstance(nodo.target, ast.Name):
            nombre, valor = nodo.target.id, nodo.value
        elif isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name):
            nombre, valor = nodo.targets[0].id, nodo.value
        else:
            continue
        if nombre != "COMPROBACIONES" or not isinstance(valor, ast.Tuple):
            continue
        return [
            entrada.elts[0].value
            for entrada in valor.elts
            if isinstance(entrada, ast.Tuple)
            and entrada.elts
            and isinstance(entrada.elts[0], ast.Constant)
            and isinstance(entrada.elts[0].value, str)
        ]
    return []


_ENCABEZADO_COMPROBACIONES = re.compile(r"^\|\s*Comprobación\s*\|")


def comprobaciones_de_la_ficha(ficha: str) -> set[str]:
    """Los identificadores entre comillas de la primera columna de la tabla.

    Se leen todos los de la celda, no el primero: una fila puede documentar dos
    comprobaciones juntas cuando contar por separado no aporta nada.
    """
    lineas = ficha.splitlines()
    for i, linea in enumerate(lineas):
        if not _ENCABEZADO_COMPROBACIONES.match(linea):
            continue
        fuera: set[str] = set()
        for fila in lineas[i + 2 :]:  # +2: el encabezado y su separador `|---|---|`
            if not fila.startswith("|"):
                break
            fuera |= set(re.findall(r"`([^`]+)`", fila.split("|")[1]))
        return fuera
    return set()


def _ficha_por_ruta(texto: str, ruta: str) -> str | None:
    """La ficha —bloque `###`— de un guardián, localizada por su ruta."""
    for seccion in re.split(r"\n(?=### )", texto):
        if seccion.startswith("### ") and ruta in seccion:
            return seccion
    return None


def revisar_comprobaciones(ficheros: set[str], texto: str) -> list[str]:
    """Que las comprobaciones que hace un guardián sean las que dice su ficha.

    Es la alternativa a versionar los guardianes con SemVer, que se descartó: un
    `__version__` en un script no lo lee nadie —el `version` de un agente sí viaja
    dentro de `provenance`— y comparar dos constantes que nadie tiene motivo para
    tocar da un verde permanente, que es peor que no comprobar nada. Lo que sí cambia
    de verdad en un guardián, y le importa a quien lo lea, es **qué comprueba**.

    Falla por un motivo real: añadir una comprobación al registro y olvidar la tabla.
    """
    problemas = []
    for script in sorted(guardianes_autonomos(ficheros)):
        registradas = comprobaciones_del_codigo(script)
        if not registradas:
            continue  # no declara registro: no hay nada que comparar
        ficha = _ficha_por_ruta(texto, script)
        if ficha is None:
            continue  # sin ficha: ya lo canta `revisar_guardianes`
        documentadas = comprobaciones_de_la_ficha(ficha)
        if not documentadas:
            problemas.append(
                f"`{script}` declara {len(registradas)} comprobaciones en `COMPROBACIONES` "
                "y su ficha no las enumera. Hace falta una tabla con la columna "
                "«Comprobación» y el identificador de cada una entre comillas."
            )
            continue
        for falta in sorted(set(registradas) - documentadas):
            problemas.append(
                f"`{script}` hace la comprobación `{falta}` y su ficha de AGENTS.md no la "
                "documenta. Quien lea la ficha creerá que ese caso no se vigila."
            )
        for sobra in sorted(documentadas - set(registradas)):
            problemas.append(
                f"La ficha de `{script}` anuncia la comprobación `{sobra}`, que el script ya "
                "no hace. Prometer una vigilancia que no existe es peor que no prometerla."
            )
    return problemas


def revisar_versiones(agentes: dict[str, tuple[str, str]], texto: str) -> list[str]:
    """Que la versión que declara la clase sea la que dice su ficha.

    Es la misma clase de deriva que el recuento de tests que decía 166 cuando eran
    265: un número que fue verdad, dejó de serlo, y nadie recibe un error rojo por
    ello. Subir `version` en el código y olvidar la ficha deja el historial de
    cambios describiendo un agente que ya no existe.

    **Limitación honesta:** solo comprueba fichas que declaran versión, y solo que la
    cadena aparezca en su celda. No detecta una ficha sin fila de versión —eso sería
    exigir un formato que hoy no todas siguen— ni que el *historial de cambios* se
    haya quedado corto.
    """
    problemas = []
    for nombre, (ruta, version) in sorted(agentes.items()):
        if not version:
            continue
        ficha = _ficha_de(texto, nombre)
        if ficha is None:
            continue  # sin ficha: ya lo canta `revisar_agentes`
        celda = _version_declarada(ficha, nombre)
        if celda is None or version in celda:
            continue
        problemas.append(
            f"`{ruta}` declara `{nombre}` en la version {version} y su ficha de "
            f"AGENTS.md dice «{celda.strip()}». Si el cambio de version es real, la "
            "ficha necesita ademas una fila en su historial de cambios."
        )
    return problemas


def revisar_agentes(ficheros: set[str]) -> list[str]:
    texto = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    registrados = set(re.findall(r"`([a-z0-9]+(?:-[a-z0-9]+)+)`", texto))
    problemas = []
    for nombre, (ruta, _) in sorted(agentes_implementados(ficheros).items()):
        if nombre not in registrados:
            problemas.append(
                f"`{ruta}` implementa el agente `{nombre}` y AGENTS.md no lo registra. "
                "El registro es el contrato: un agente sin ficha es invisible para el resto."
            )
    return problemas


# --------------------------------------------------------------------------- #
# 4 · Lo que existe y nadie ha documentado (el sentido inverso)
# --------------------------------------------------------------------------- #
def revisar_inventario(ficheros: set[str]) -> list[str]:
    """Que lo citado exista es media verdad; la otra media es que lo que existe se cite.

    Un notebook nuevo que nadie añade al índice es invisible: quien llega al
    repositorio no sabe que está ahí. Es la deriva contraria a la del enlace roto y
    no la detecta ninguna de las comprobaciones anteriores.
    """
    problemas = []
    indice = (REPO / "notebooks" / "README.md").read_text(encoding="utf-8")
    for ruta in sorted(f for f in ficheros if f.startswith("notebooks/") and f.endswith(".ipynb")):
        if Path(ruta).name not in indice:
            problemas.append(
                f"`{ruta}` existe y el índice de notebooks no lo menciona. "
                "Un notebook sin entrada en el índice es un notebook que nadie encuentra."
            )

    # Vale con que esté documentado en ALGUNA página: `fetch_teeth3ds.sh` se explica
    # en la nota de su dataset y no en el README, y eso es correcto.
    documentacion = "\n".join(
        (REPO / f).read_text(encoding="utf-8") for f in ficheros if f.endswith(".md")
    )
    for ruta in sorted(
        f for f in ficheros if f.startswith("scripts/") and f.endswith((".py", ".sh"))
    ):
        if ruta not in documentacion:
            problemas.append(
                f"`{ruta}` existe y no se menciona en ninguna página de documentación. "
                "Una utilidad que nadie documenta la acaba reescribiendo otro."
            )
    return problemas


def revisar_arbol(ficheros: set[str]) -> list[str]:
    """El arbol del README frente a los directorios que existen de verdad.

    No se genera: cada linea lleva una anotacion escrita a mano que explica para
    que sirve la carpeta, y eso es prosa. Pero si se puede comprobar que no anuncia
    carpetas que ya no estan ni se calla las nuevas.
    """
    texto = (REPO / "README.md").read_text(encoding="utf-8")
    # Las carpetas ocultas (.github, .claude) quedan fuera por los dos lados: el
    # arbol las menciona como contexto, no como inventario.
    anunciados = {
        d for d in re.findall(r"[│├└]──\s+([a-z0-9._-]+)/", texto) if not d.startswith(".")
    }
    # Dos conjuntos distintos, porque las dos direcciones no piden lo mismo:
    #  - `esperados`: lo que el arbol DEBE enumerar (primer nivel, y los paquetes
    #    y apps, que es donde vive el trabajo).
    #  - `existentes`: cualquier directorio del repositorio, a cualquier
    #    profundidad. Anunciar algo que no existe siempre es mentira; no enumerar
    #    un subdirectorio de tercer nivel, no.
    esperados = {f.split("/")[0] for f in ficheros if "/" in f and not f.startswith(".")}
    for padre in ("apps", "packages"):
        esperados |= {f.split("/")[1] for f in ficheros if f.startswith(f"{padre}/")}
    existentes = {
        parte for f in ficheros for parte in Path(f).parent.parts if not parte.startswith(".")
    }
    problemas = []
    for falta in sorted(esperados - anunciados):
        problemas.append(
            f"El arbol del README no menciona `{falta}/`, que existe. "
            "Quien lea la estructura no sabra que esta ahi."
        )
    for sobra in sorted(anunciados - existentes):
        problemas.append(
            f"El arbol del README anuncia `{sobra}/`, que ya no existe en el repositorio."
        )
    return problemas


# --------------------------------------------------------------------------- #
# Constantes citadas en la documentacion
# --------------------------------------------------------------------------- #
# El marcador que ata un numero de la prosa a la constante de la que sale. Va en un
# comentario HTML, asi que no se ve en el Markdown renderizado:
#
#     el error de malla se mide contra 0,1 mm <!--const:REVERSIBILITY_BUDGET_MM-->
#
# Hace falta un marcador explicito, y esto es lo que costo entenderlo: la prosa
# peligrosa NO nombra la constante. Dice «ε = 0,5 mm» o «< 0,1 mm» a secas, asi que una
# comprobacion que buscara el nombre de la constante en el texto cazaria 3 de los 25
# sitios donde hay un numero del codigo escrito a mano. El marcador invierte la carga:
# quien escribe el numero declara de donde sale, una vez, y a partir de ahi el CI lo
# vigila para siempre.
MARCADOR_CONST = re.compile(r"<!--\s*const:\s*([A-Z][A-Z0-9_]*)\s*-->")

# Numeros de la prosa: admite coma decimal (se escribe en espanol) y signo.
_NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?")


def constantes_del_codigo(ficheros: set[str]) -> dict[str, set[float | str]]:
    """`NOMBRE → {valores}` de las constantes numericas de modulo del codigo fuente.

    Por AST y no importando: importar ejecuta, y un guardian que ejecuta el codigo que
    vigila puede fallar por un `ImportError` que no tiene nada que ver con la deriva.

    Se recogen numeros **y cadenas**. Las cadenas hacen falta por `SCHEMA_VERSION`, que es
    el caso que motivo ampliarlo: el README anunciaba el esquema `1.2.0` cuando el codigo
    iba por `1.3.0`, y la comprobacion pasaba en verde porque solo miraba numeros. Una
    version de contrato desincronizada es justo el tipo de mentira que este guardian existe
    para impedir.

    El valor es un **conjunto** porque un mismo nombre puede vivir en dos modulos. Si los
    dos valen lo mismo da igual; si no, la cita es ambigua y eso se declara.
    """
    encontradas: dict[str, set[float | str]] = {}
    fuentes = [
        f for f in ficheros
        if f.endswith(".py") and "/src/" in f and "3dgs-engine" not in f
    ]
    for fichero in sorted(fuentes):
        try:
            arbol = ast.parse((REPO / fichero).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for nodo in arbol.body:
            objetivos: list[ast.expr] = []
            if isinstance(nodo, ast.Assign):
                objetivos = list(nodo.targets)
            elif isinstance(nodo, ast.AnnAssign) and nodo.value is not None:
                objetivos = [nodo.target]
            else:
                continue
            valor = _valor_de(nodo.value)
            if valor is None:
                continue
            for objetivo in objetivos:
                if isinstance(objetivo, ast.Name) and objetivo.id.isupper():
                    encontradas.setdefault(objetivo.id, set()).add(valor)
    return encontradas


def _valor_de(nodo: ast.expr | None) -> float | str | None:
    """El valor de un literal numerico o de cadena. El negativo es un unario, no literal."""
    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, bool):
            return None
        if isinstance(nodo.value, int | float):
            return float(nodo.value)
        if isinstance(nodo.value, str):
            return nodo.value
    if isinstance(nodo, ast.UnaryOp) and isinstance(nodo.op, ast.USub):
        interior = _valor_de(nodo.operand)
        return None if isinstance(interior, str) or interior is None else -interior
    return None


def revisar_constantes(ficheros: set[str]) -> list[str]:
    """Cada numero marcado en la documentacion contra el valor real de su constante.

    Es la comprobacion que le faltaba a este guardian y que su propio docstring anunciaba
    como pendiente: la ficha de fusion decia «ε = 0,5 mm» mucho despues de que el codigo
    dijera otra cosa, y aqui salia todo en verde.
    """
    constantes = constantes_del_codigo(ficheros)
    problemas: list[str] = []

    for doc in sorted(f for f in ficheros if f.endswith(".md")):
        for n, linea, sin_codigo in _lineas_de_prosa((REPO / doc).read_text(encoding="utf-8")):
            for nombre in MARCADOR_CONST.findall(sin_codigo):
                valores = constantes.get(nombre)
                if not valores:
                    problemas.append(
                        f"{doc}:{n} cita la constante `{nombre}`, que no existe en el "
                        "codigo fuente. O se renombro, o se borro, y el numero de al lado "
                        "ya no lo respalda nadie."
                    )
                    continue
                if len(valores) > 1:
                    problemas.append(
                        f"{doc}:{n} cita `{nombre}`, que existe con {len(valores)} valores "
                        f"distintos ({', '.join(sorted(map(str, valores)))}). La cita "
                        "es ambigua: renombra una de las dos constantes."
                    )
                    continue
                esperado = next(iter(valores))
                if isinstance(esperado, str):
                    # Una cadena se busca literal: `SCHEMA_VERSION = "1.4.0"` tiene que
                    # aparecer tal cual, no como tres numeros sueltos.
                    if esperado not in linea:
                        problemas.append(
                            f"{doc}:{n} no dice `{esperado}` en ninguna parte, y "
                            f"`{nombre}` vale eso. Uno de los dos miente, y el que se "
                            "publica es el documento."
                        )
                    continue
                escritos = _numeros_de(MARCADOR_CONST.sub("", linea))
                if not any(math.isclose(x, esperado, rel_tol=1e-9) for x in escritos):
                    problemas.append(
                        f"{doc}:{n} no dice {esperado:g} en ninguna parte, y `{nombre}` "
                        f"vale eso. La linea trae {_lista(escritos)}. Uno de los dos "
                        "miente, y el que se publica es el documento."
                    )
    return problemas


def _lineas_de_prosa(texto: str):
    """`(numero, linea, linea sin codigo en linea)` de lo que es prosa.

    Los dos textos hacen falta, y confundirlos rompe la comprobacion de dos maneras
    distintas — las dos ocurrieron al escribir esto:

    * **El marcador** se busca en la version *sin* comillas invertidas. Uno que aparece
      dentro de un bloque cercado o de un `codigo` es un **ejemplo** —la ficha de este
      guardian enseña cómo se escribe uno—, no una afirmacion sobre un numero.
    * **El numero** se busca en la linea *entera*, comillas incluidas: una fila que dice
      `RENDER_PSNR_BUDGET_DB = 40` cita el valor dentro del codigo en linea, que es donde
      normalmente se escribe. Quitarlo dejaba la linea sin numeros y la daba por mentirosa.
    """
    dentro = False
    for n, linea in enumerate(texto.splitlines(), 1):
        if linea.lstrip().startswith("```"):
            dentro = not dentro
            continue
        if not dentro:
            yield n, linea, re.sub(r"`[^`]*`", "", linea)


def _numeros_de(texto: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in _NUMERO.findall(texto)]


def _lista(numeros: list[float], tope: int = 5) -> str:
    """Los numeros de la linea, recortados: una fila de tabla puede traer quince."""
    if not numeros:
        return "ningun numero"
    mostrados = ", ".join(f"{x:g}" for x in numeros[:tope])
    return mostrados if len(numeros) <= tope else f"{mostrados}… ({len(numeros)} en total)"


# --------------------------------------------------------------------------- #
def _agentes_md() -> str:
    return (REPO / "AGENTS.md").read_text(encoding="utf-8")


# El registro: identificador, título para quien lee la salida, y la función.
#
# El identificador va aparte del título a propósito. El título se reformula sin
# consecuencias; el identificador es el que aparece en la ficha de `AGENTS.md`, así que
# es un nombre estable y sin acentos. Este script se compara a sí mismo con su ficha.
COMPROBACIONES: tuple[tuple[str, str, Callable[[set[str], bool], list[str]]], ...] = (
    ("env", "variables de entorno", lambda f, _: revisar_env(f)),
    ("rutas", "rutas citadas", lambda f, _: revisar_rutas(f)),
    ("agentes", "registro de agentes", lambda f, _: revisar_agentes(f)),
    ("versiones", "versiones de los agentes",
     lambda f, _: revisar_versiones(agentes_implementados(f), _agentes_md())),
    ("guardianes", "guardianes registrados", lambda f, _: revisar_guardianes(f, _agentes_md())),
    ("comprobaciones", "comprobaciones documentadas",
     lambda f, _: revisar_comprobaciones(f, _agentes_md())),
    ("constantes", "constantes citadas", lambda f, _: revisar_constantes(f)),
    ("inventario", "inventario documentado", lambda f, _: revisar_inventario(f)),
    ("arbol", "arbol del README", lambda f, _: revisar_arbol(f)),
    ("bloques", "bloques generados", sincronizar_bloques),
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--check", action="store_true", help="Falla si hay deriva (para CI).")
    modo.add_argument("--write", action="store_true", help="Regenera los bloques generados.")
    args = ap.parse_args()

    ficheros = versionados()
    problemas: list[str] = []
    for _, nombre, revision in COMPROBACIONES:
        fallos = revision(ficheros, args.write)
        estado = "✗" if fallos else "✓"
        print(f"{estado} {nombre}: {len(fallos) or 'sin deriva'}{' problema(s)' if fallos else ''}")
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
