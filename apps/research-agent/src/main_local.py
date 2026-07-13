"""Variante LOCAL y gratuita del orquestador (Ollama en vez de Claude).

Idéntico propósito que `src/main.py`, pero el "cerebro" es un modelo local
servido por Ollama: 0 € por token, sin API key y sin que los documentos salgan
de tu máquina. Reutiliza tal cual las tools (Fase 2) y el motor RAG (Fase 3).

Como Ollama no ofrece un Tool Runner, el bucle ReAct (llamada -> ejecutar tool
-> devolver resultado -> repetir) se implementa a mano aquí.

Requisitos:
    - Ollama instalado y el servidor corriendo:  ollama serve
    - Un modelo con soporte de tools descargado:  ollama pull qwen2.5:7b

Ejecución:
    uv run python -m src.main_local
"""

from __future__ import annotations

import os
import sys

from ollama import Client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import references, tools
from .rag import RAGEngine

console = Console()

# Modelo local (con buen soporte de tool calling). Configurable por entorno.
MODEL = os.getenv("RESEARCH_AGENT_LOCAL_MODEL", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_TOOL_ITERS = 12  # tope de vueltas de tool por turno (evita bucles infinitos)

SYSTEM_PROMPT = """\
Eres research-agent, un agente de investigación autónomo especializado en literatura \
científica sobre 3D Gaussian Splatting, el estándar DICOM y normativas clínicas.
Tu objetivo es buscar, ingerir y sintetizar papers para producir reportes en Markdown.

Herramientas:
- read_directory: ver qué documentos hay en knowledge_base/.
- ingest_corpus: indexar todos los documentos en la base vectorial. Hazlo una \
vez antes de consultar el corpus.
- search_corpus: búsqueda semántica sobre el corpus indexado. Úsala para \
preguntas que cruzan varios documentos.
- read_file: leer un documento concreto entero.
- search_references: buscar papers en fuentes académicas externas (Semantic \
Scholar / arXiv) para DESCUBRIR literatura que aún no tienes en local.
- download_reference: descargar a knowledge_base/ un PDF encontrado con \
search_references. Lo indexa automáticamente (no necesitas ingest_corpus después).
- write_summary: escribir el reporte final en docs_output/. Un reporte SIEMPRE \
lleva un abstract conciso seguido de una explicación completa del paper \
(problema, metodología, resultados, limitaciones y relevancia clínica).

Directrices:
- Cuando tengas información suficiente para actuar, actúa.
- Para recopilar literatura nueva: search_references → download_reference \
(ya la indexa) → search_corpus/read_file → write_summary.
- Cita las fuentes (nombre del documento) en reportes y respuestas.
- Responde en español.
"""

# --- Motor RAG perezoso -------------------------------------------------------

_rag: RAGEngine | None = None


def _get_rag() -> RAGEngine:
    global _rag
    if _rag is None:
        _rag = RAGEngine()
    return _rag


def _download_and_index(url: str, filename: str) -> str:
    """Descarga un PDF y, si va bien, lo indexa en la base vectorial."""
    resultado = references.download_reference(url, filename)
    if resultado.startswith("ERROR"):
        return resultado
    indexado = _get_rag().ingest_file(references.normalized_pdf_name(filename))
    return f"{resultado}\n{indexado}"


# --- Definición de tools para Ollama (JSON Schema + despacho) ------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_directory",
            "description": "Lista los documentos disponibles en knowledge_base/.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido completo de un documento de "
            "knowledge_base/ (PDF o texto/Markdown).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Ruta del documento relativa a knowledge_base/.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_corpus",
            "description": "Indexa todos los documentos de knowledge_base/ en la "
            "base vectorial. Ejecútala una vez antes de usar search_corpus.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Búsqueda semántica en el corpus indexado; devuelve los "
            "fragmentos más relevantes con su fuente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Pregunta o tema a buscar en la literatura.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_references",
            "description": "Busca papers en fuentes académicas externas (Semantic "
            "Scholar / arXiv) y devuelve candidatos con su URL de PDF. Para "
            "descubrir literatura que aún no está en knowledge_base/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Términos de búsqueda (tema, título, autores).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Número máximo de resultados (por defecto 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_reference",
            "description": "Descarga a knowledge_base/ un PDF encontrado con "
            "search_references y lo indexa automáticamente (no necesitas "
            "ingest_corpus después).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL http(s) directa al PDF.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Nombre del .pdf a guardar.",
                    },
                },
                "required": ["url", "filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_summary",
            "description": "Escribe el reporte final en docs_output/. Debe llevar un "
            "abstract conciso seguido de una explicación completa del paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Nombre del .md de salida.",
                    },
                    "title": {"type": "string", "description": "Título del reporte."},
                    "abstract": {
                        "type": "string",
                        "description": "Resumen conciso (3-6 frases).",
                    },
                    "full_explanation": {
                        "type": "string",
                        "description": "Explicación completa y detallada (más extensa "
                        "que el abstract): problema, método, resultados, limitaciones "
                        "y relevancia.",
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Puntos clave opcionales.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Documento fuente (opcional).",
                    },
                },
                "required": ["filename", "title", "abstract", "full_explanation"],
            },
        },
    },
]

# Mapa nombre-de-tool -> función real (Fases 2 y 3).
DISPATCH = {
    "read_directory": lambda: tools.read_directory(),
    "read_file": lambda path: tools.read_file(path),
    "ingest_corpus": lambda: _get_rag().ingest_directory(),
    "search_corpus": lambda question: _get_rag().query(question),
    "search_references": lambda query, limit=5: references.search_references(
        query, limit
    ),
    "download_reference": lambda url, filename: _download_and_index(url, filename),
    "write_summary": lambda **kw: tools.write_summary(**kw),
}


def _run_tool(name: str, args: dict) -> str:
    """Ejecuta una tool por nombre; nunca lanza, devuelve el error como texto."""
    fn = DISPATCH.get(name)
    if fn is None:
        return f"ERROR: tool desconocida '{name}'."
    try:
        return str(fn(**args))
    except TypeError as exc:
        return f"ERROR: argumentos inválidos para '{name}': {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR al ejecutar '{name}': {exc}"


# --- Bucle del agente (ReAct manual) -----------------------------------------


def _run_turn(client: Client, messages: list) -> None:
    """Ejecuta un turno: alterna llamadas al modelo y ejecución de tools hasta
    que el modelo responde sin pedir más herramientas."""
    for _ in range(MAX_TOOL_ITERS):
        respuesta = client.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = respuesta.message
        messages.append(msg)  # historial: turno del asistente (con posibles tool_calls)

        if not msg.tool_calls:
            if msg.content and msg.content.strip():
                console.print(Markdown(msg.content))
            return

        for llamada in msg.tool_calls:
            nombre = llamada.function.name
            args = dict(llamada.function.arguments or {})
            console.print(f"[dim]· usando {nombre}…[/dim]")
            resultado = _run_tool(nombre, args)
            messages.append({"role": "tool", "content": resultado, "tool_name": nombre})

    console.print("[yellow]Se alcanzó el límite de llamadas a tools en este turno.[/]")


def main() -> None:
    client = Client(host=OLLAMA_HOST)

    # Comprobación de conectividad y de que el modelo está disponible.
    try:
        disponibles = [m.model for m in client.list().models]
    except Exception:  # noqa: BLE001
        console.print(
            "[bold red]No hay un servidor Ollama accesible.[/] "
            "Arráncalo con [bold]ollama serve[/] (o comprueba OLLAMA_HOST)."
        )
        sys.exit(1)

    if MODEL not in disponibles and f"{MODEL}:latest" not in disponibles:
        console.print(
            f"[bold red]El modelo '{MODEL}' no está descargado.[/] "
            f"Descárgalo con [bold]ollama pull {MODEL}[/]."
        )
        console.print(
            f"[dim]Modelos disponibles: {', '.join(disponibles) or '(ninguno)'}[/]"
        )
        sys.exit(1)

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    console.print(
        Panel.fit(
            "[bold]research-agent[/] (local) — agente de investigación · [green]0 € · sin API key[/]\n"
            f"Modelo Ollama: [cyan]{MODEL}[/]  ·  Escribe [bold]salir[/] para terminar.",
            border_style="green",
        )
    )

    try:
        while True:
            try:
                user_input = console.input("\n[bold green]tú[/] › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in {"salir", "exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            console.print("\n[bold cyan]research-agent[/]:")
            try:
                _run_turn(client, messages)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[bold red]Error durante el turno:[/] {exc}")
    finally:
        if _rag is not None:
            _rag.close()
        console.print("\n[dim]Hasta luego.[/]")


if __name__ == "__main__":
    main()
