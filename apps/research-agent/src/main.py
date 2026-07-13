"""Orquestador del agente research-agent (CLI interactivo).

Conecta las tools de sistema (Fase 2) y el motor RAG (Fase 3) a Claude
mediante *tool calling* nativo. Usa el Tool Runner del SDK de Anthropic, que
ejecuta el bucle ReAct automáticamente: el modelo decide qué tool llamar, el
runner la ejecuta, le devuelve el resultado y repite hasta terminar el turno.

Ejecución:
    uv run python -m src.main
"""

from __future__ import annotations

import os
import sys

from anthropic import Anthropic, beta_tool
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import references, tools
from .rag import RAGEngine

console = Console()

MODEL = os.getenv("RESEARCH_AGENT_MODEL", "claude-opus-4-8")
MAX_TOKENS = 16000

SYSTEM_PROMPT = """\
Eres research-agent, un agente de investigación autónomo especializado en literatura \
científica sobre 3D Gaussian Splatting, el estándar DICOM y normativas clínicas.
Tu objetivo es buscar, ingerir y sintetizar papers para producir reportes en Markdown.

Herramientas:
- read_directory: ver qué documentos hay en knowledge_base/.
- ingest_corpus: indexar todos los documentos en la base vectorial. Hazlo una \
vez antes de consultar el corpus.
- search_corpus: búsqueda semántica sobre el corpus indexado. Úsala para \
preguntas que cruzan varios documentos o cuando no sabes en qué fichero está la \
respuesta.
- read_file: leer un documento concreto entero. Úsala cuando necesites el \
detalle completo de un paper específico.
- search_references: buscar papers en fuentes académicas externas (Semantic \
Scholar / arXiv) para DESCUBRIR literatura que aún no tienes en local.
- download_reference: descargar a knowledge_base/ un PDF encontrado con \
search_references. Lo indexa automáticamente (no necesitas ingest_corpus después).
- write_summary: escribir el reporte final en docs_output/. Un reporte SIEMPRE \
lleva un abstract conciso seguido de una explicación completa del paper \
(problema, metodología, resultados, limitaciones y relevancia clínica).

Directrices:
- Cuando tengas información suficiente para actuar, actúa. No repreguntes lo ya \
establecido ni enumeres opciones que no vas a seguir.
- Para recopilar literatura nueva sigue este flujo: search_references → \
download_reference (ya la indexa) → search_corpus/read_file → write_summary.
- Cita las fuentes (nombre del documento) en reportes y respuestas.
- Responde en español.
"""

# Motor RAG perezoso: no abre Qdrant hasta que una tool lo necesita, para que
# importar este módulo no bloquee el directorio on-disk.
_rag: RAGEngine | None = None


def _get_rag() -> RAGEngine:
    global _rag
    if _rag is None:
        _rag = RAGEngine()
    return _rag


# --- Tools expuestas al modelo (envuelven Fases 2 y 3) ------------------------


@beta_tool
def read_directory() -> str:
    """Lista los documentos de referencia disponibles en knowledge_base/.

    Úsala para descubrir qué material hay antes de leer, indexar o consultar.
    """
    return tools.read_directory()


@beta_tool
def read_file(path: str) -> str:
    """Lee el contenido completo de un documento de knowledge_base/.

    Soporta PDF y texto/Markdown. Úsala cuando necesites el detalle íntegro de
    un documento concreto.

    Args:
        path: Ruta del documento relativa a knowledge_base/.
    """
    return tools.read_file(path)


@beta_tool
def ingest_corpus() -> str:
    """Indexa todos los documentos de knowledge_base/ en la base vectorial.

    Ejecútala una vez antes de usar search_corpus. Es idempotente para nuevos
    documentos.
    """
    return _get_rag().ingest_directory()


@beta_tool
def search_corpus(question: str) -> str:
    """Busca semánticamente en el corpus indexado y devuelve los fragmentos
    más relevantes con su fuente.

    Úsala para responder preguntas que cruzan varios documentos.

    Args:
        question: Pregunta o tema a buscar en la literatura indexada.
    """
    return _get_rag().query(question)


@beta_tool
def write_summary(
    filename: str,
    title: str,
    abstract: str,
    full_explanation: str,
    key_points: list[str] | None = None,
    source: str | None = None,
) -> str:
    """Escribe el reporte final en Markdown dentro de docs_output/.

    El reporte SIEMPRE debe llevar, en este orden, un abstract conciso y una
    explicación completa del paper (problema, metodología, resultados,
    limitaciones y relevancia clínica).

    Args:
        filename: Nombre del fichero de salida, p. ej. "resumen_dicom.md".
        title: Título del reporte (normalmente el título del paper).
        abstract: Resumen conciso del paper, de 3 a 6 frases.
        full_explanation: Explicación completa y detallada del paper. Debe ser
            sustancialmente más extensa que el abstract.
        key_points: Lista opcional de puntos clave o hallazgos.
        source: Referencia opcional al documento fuente.
    """
    return tools.write_summary(
        filename=filename,
        title=title,
        abstract=abstract,
        full_explanation=full_explanation,
        key_points=key_points,
        source=source,
    )


@beta_tool
def search_references(query: str, limit: int = 5) -> str:
    """Busca papers en fuentes académicas externas (Semantic Scholar / arXiv) y
    devuelve candidatos con su URL de PDF descargable.

    Úsala para descubrir literatura que aún no está en knowledge_base/.

    Args:
        query: Términos de búsqueda (tema, título, autores).
        limit: Número máximo de resultados (por defecto 5).
    """
    return references.search_references(query, limit)


@beta_tool
def download_reference(url: str, filename: str) -> str:
    """Descarga un PDF (de search_references) a knowledge_base/ y lo indexa
    automáticamente en la base vectorial. No necesitas llamar a ingest_corpus
    después: ya queda consultable con search_corpus.

    Args:
        url: URL http(s) directa al PDF.
        filename: Nombre con el que guardarlo, p. ej. "kerbl2023.pdf".
    """
    resultado = references.download_reference(url, filename)
    if resultado.startswith("ERROR"):
        return resultado
    indexado = _get_rag().ingest_file(references.normalized_pdf_name(filename))
    return f"{resultado}\n{indexado}"


AGENT_TOOLS = [
    read_directory,
    read_file,
    ingest_corpus,
    search_corpus,
    search_references,
    download_reference,
    write_summary,
]


# --- Bucle del agente ---------------------------------------------------------


def _run_turn(client: Anthropic, messages: list) -> None:
    """Ejecuta un turno completo (con posibles llamadas a tools) e imprime la
    respuesta del agente. Mantiene `messages` sincronizado con el historial."""
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        tools=AGENT_TOOLS,
        messages=messages,
    )

    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                console.print(Markdown(block.text))
            elif block.type == "tool_use":
                console.print(f"[dim]· usando {block.name}…[/dim]")

        # Espejar el turno en el historial (incluye bloques thinking/tool_use).
        messages.append({"role": "assistant", "content": message.content})
        tool_response = runner.generate_tool_call_response()
        if tool_response is not None:
            messages.append(tool_response)


def main() -> None:
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]Falta ANTHROPIC_API_KEY.[/] "
            "Copia .env.example a .env y añade tu clave, o expórtala en el entorno."
        )
        sys.exit(1)

    client = Anthropic()
    messages: list = []

    console.print(
        Panel.fit(
            "[bold]research-agent[/] — agente de investigación (3DGS · DICOM · normativas)\n"
            f"Modelo: [cyan]{MODEL}[/]  ·  Escribe [bold]salir[/] para terminar.",
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
