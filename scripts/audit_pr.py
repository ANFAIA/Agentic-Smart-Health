#!/usr/bin/env python3
"""Agente guardián de arquitectura — Agentic Smart Health.

Auditor estático (sin LLM) que se ejecuta en CI sobre cada Pull Request.
Solo revisa los archivos Python *modificados* por el PR y aplica dos reglas
específicas del monorepo:

  1. Pydantic v2 estricto en ``packages/core-schemas``: prohíbe el shim v1, los
     decoradores/estilos de v1 (``@validator``, ``@root_validator``,
     ``class Config``) y ``BaseSettings`` (movido a ``pydantic-settings``).

  2. Sin dependencias cruzadas entre componentes de ``apps/``: un app no puede
     importar el paquete de otro app. El código compartido debe vivir en
     ``packages/`` (p. ej. ``core-schemas``).

Salidas (en el directorio de trabajo):
  - ``audit_report.txt``   : resumen Markdown legible (vacío si todo está limpio).
  - ``line_comments.json`` : ``[{path, line, body}]`` para comentarios inline.

El código de salida es siempre 0; es el workflow quien decide si las violaciones
bloquean el merge, según si ``line_comments.json`` queda no vacío.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

APPS_DIR = "apps"
CORE_SCHEMAS_PREFIX = "packages/core-schemas/"

# Decoradores/símbolos propios de Pydantic v1 que ya no deben usarse en v2.
PYDANTIC_V1_NAMES = {"validator", "root_validator", "BaseSettings"}


def run_git(*args: str) -> str:
    """Ejecuta git y devuelve stdout (vacío si falla)."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - en CI preferimos degradar a "sin cambios"
        return ""


def changed_python_files() -> list[str]:
    """Lista los .py añadidos/modificados en el PR respecto a la rama base.

    En GitHub Actions se usa ``GITHUB_BASE_REF``. En local se cae a un diff
    contra ``origin/main`` o, si no existe, contra ``HEAD~1``.
    """
    base = os.environ.get("GITHUB_BASE_REF") or "main"
    base_ref = f"origin/{base}"

    merge_base = run_git("merge-base", base_ref, "HEAD")
    if not merge_base:
        # Fallback local: intenta contra el commit anterior.
        merge_base = run_git("rev-parse", "HEAD~1")
    if not merge_base:
        return []

    diff = run_git(
        "diff", "--name-only", "--diff-filter=ACM", merge_base, "HEAD"
    )
    files = [f for f in diff.splitlines() if f.endswith(".py")]
    return [f for f in files if Path(f).exists()]


def app_identifiers() -> dict[str, set[str]]:
    """Mapea cada carpeta de app a sus identificadores de módulo importables.

    ``agent-orchestrator`` -> {"agent-orchestrator", "agent_orchestrator"}
    """
    result: dict[str, set[str]] = {}
    apps_path = Path(APPS_DIR)
    if not apps_path.is_dir():
        return result
    for entry in sorted(apps_path.iterdir()):
        if entry.is_dir():
            name = entry.name
            result[name] = {name, name.replace("-", "_")}
    return result


def app_of(path: str) -> str | None:
    """Devuelve el nombre de carpeta de app al que pertenece ``path``, si aplica."""
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == APPS_DIR:
        return parts[1]
    return None


def iter_imports(tree: ast.AST):
    """Genera (modulo_top_level, ruta_completa, lineno) por cada import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0] if module else ""
            yield top, module, node.lineno


def check_cross_app_imports(path: str, tree: ast.AST) -> list[dict]:
    """Regla 2: detecta imports de un app hacia el paquete de otro app."""
    this_app = app_of(path)
    if this_app is None:
        return []

    all_apps = app_identifiers()
    other_ids: set[str] = set()
    for name, ids in all_apps.items():
        if name != this_app:
            other_ids |= ids

    findings: list[dict] = []
    for top, full, lineno in iter_imports(tree):
        # Caso `import apps.<otro>...` / `from apps.<otro>...`
        offending = None
        parts = full.split(".")
        if top == "apps" and len(parts) >= 2 and parts[1] in other_ids:
            offending = parts[1]
        elif top in other_ids:
            offending = top

        if offending:
            findings.append(
                {
                    "path": path,
                    "line": lineno,
                    "body": (
                        "🚫 **Dependencia cruzada entre apps** "
                        f"(`{this_app}` → `{offending}`).\n\n"
                        f"El app `{this_app}` no puede importar `{full}`. "
                        "Los componentes de `apps/` deben ser independientes; "
                        "extrae el código compartido a `packages/` "
                        "(p. ej. `packages/core-schemas`) y comunícate mediante "
                        "los contratos de datos definidos allí."
                    ),
                }
            )
    return findings


def check_pydantic_v2(path: str, tree: ast.AST) -> list[dict]:
    """Regla 1: exige idiomas de Pydantic v2 en core-schemas."""
    if not path.replace("\\", "/").startswith(CORE_SCHEMAS_PREFIX):
        return []

    findings: list[dict] = []

    def flag(line: int, msg: str) -> None:
        findings.append(
            {
                "path": path,
                "line": line,
                "body": f"🚫 **Pydantic v2 requerido en `core-schemas`**.\n\n{msg}",
            }
        )

    for node in ast.walk(tree):
        # Shim de compatibilidad v1.
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pydantic.v1"):
                    flag(
                        node.lineno,
                        "Se importa `pydantic.v1` (shim de compatibilidad). "
                        "Usa la API nativa de Pydantic v2.",
                    )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("pydantic.v1"):
                flag(
                    node.lineno,
                    "Se importa desde `pydantic.v1`. Usa la API nativa de v2.",
                )
            elif module == "pydantic":
                for alias in node.names:
                    if alias.name in PYDANTIC_V1_NAMES:
                        flag(
                            node.lineno,
                            f"`{alias.name}` es API de Pydantic v1. Reemplázalo por "
                            "`field_validator` / `model_validator` "
                            "(o `pydantic-settings` para `BaseSettings`).",
                        )

        # Decoradores de validación v1.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_name = _decorator_name(dec)
                if dec_name in {"validator", "root_validator"}:
                    flag(
                        getattr(dec, "lineno", node.lineno),
                        f"`@{dec_name}` es de Pydantic v1. Usa `@field_validator` "
                        "o `@model_validator`.",
                    )

        # `class Config:` anidada (estilo v1) dentro de un modelo.
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.ClassDef) and child.name == "Config":
                    flag(
                        child.lineno,
                        "`class Config:` es estilo v1. Usa "
                        "`model_config = ConfigDict(...)`.",
                    )

    return findings


def _decorator_name(dec: ast.expr) -> str | None:
    """Extrae el nombre base de un decorador (`x`, `x.y`, `x()`, `x.y()`)."""
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return None


def audit(files: list[str]) -> list[dict]:
    findings: list[dict] = []
    for path in files:
        try:
            source = Path(path).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=path)
        except (OSError, SyntaxError) as exc:
            # Un archivo con error de sintaxis lo caza ruff/mypy; aquí lo saltamos.
            print(f"::warning file={path}::audit_pr no pudo parsear: {exc}")
            continue
        findings.extend(check_cross_app_imports(path, tree))
        findings.extend(check_pydantic_v2(path, tree))
    return findings


def write_outputs(findings: list[dict]) -> None:
    Path("line_comments.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not findings:
        Path("audit_report.txt").write_text("", encoding="utf-8")
        return

    lines = ["Se han detectado violaciones de arquitectura:\n"]
    for f in findings:
        first_line = f["body"].splitlines()[0]
        lines.append(f"- `{f['path']}:{f['line']}` — {first_line}")
    Path("audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    # Modo auxiliar: lista los .py del PR (uno por línea) para acotar ruff/mypy.
    if "--list-changed" in sys.argv:
        for f in changed_python_files():
            print(f)
        return 0

    files = changed_python_files()
    print(f"Archivos Python modificados en el PR: {len(files)}")
    for f in files:
        print(f"  - {f}")

    findings = audit(files)
    write_outputs(findings)

    if findings:
        print(f"\n❌ {len(findings)} violación(es) de arquitectura:")
        for f in findings:
            print(f"  {f['path']}:{f['line']} — {f['body'].splitlines()[0]}")
    else:
        print("\n✅ Sin violaciones de arquitectura.")

    # Siempre 0: el workflow decide el bloqueo según line_comments.json.
    return 0


if __name__ == "__main__":
    sys.exit(main())
