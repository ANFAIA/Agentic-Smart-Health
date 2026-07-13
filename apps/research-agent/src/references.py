"""Recopilación de referencias externas (búsqueda + descarga).

Cierra el ciclo de la misión: además de ingerir y resumir documentos locales,
el agente puede **descubrir** literatura en fuentes académicas y **descargarla**
a `knowledge_base/`, desde donde entra en el pipeline RAG normal.

Fuentes (sin API key):
  - **Semantic Scholar** (primaria): cubre CS (3D Gaussian Splatting) y
    biomédico/clínico (DICOM, interoperabilidad), con enlaces a PDF abierto.
  - **arXiv** (fallback): muy fiable y sin límites; se usa cuando Semantic
    Scholar falla o va saturada (su pool anónimo es limitado).

Solo depende de la librería estándar (urllib + json + xml): sin dependencias
nuevas. La descarga se confina a `knowledge_base/` (reutiliza el sandbox de
tools.py), acepta solo http(s), valida que sea un PDF y limita el tamaño.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from . import tools

_S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,externalIds,openAccessPdf"
_ARXIV_SEARCH = "http://export.arxiv.org/api/query"
_USER_AGENT = "research-agent/0.1 (+https://github.com/lgarbayo/agentic-smart-health)"
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB
_TIMEOUT = 45

# Namespaces del Atom que devuelve arXiv.
_ATOM = "{http://www.w3.org/2005/Atom}"


def normalized_pdf_name(filename: str) -> str:
    """Nombre base con extensión .pdf forzada (el mismo con el que se guarda)."""
    base = Path(filename).name
    return base if base.lower().endswith(".pdf") else base + ".pdf"


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _search_semantic_scholar(query: str, limit: int) -> list[dict]:
    """Devuelve referencias normalizadas de Semantic Scholar (o lanza excepción)."""
    params = urllib.parse.urlencode(
        {"query": query, "limit": limit, "fields": _S2_FIELDS}
    )
    data = json.loads(_http_get(f"{_S2_SEARCH}?{params}"))
    referencias = []
    for p in data.get("data") or []:
        abierto = (p.get("openAccessPdf") or {}).get("url")
        arxiv = (p.get("externalIds") or {}).get("ArXiv")
        pdf = abierto or (f"https://arxiv.org/pdf/{arxiv}.pdf" if arxiv else None)
        referencias.append(
            {
                "title": p.get("title") or "(sin título)",
                "authors": [a.get("name", "") for a in (p.get("authors") or [])],
                "year": p.get("year"),
                "pdf_url": pdf,
                "source": "Semantic Scholar",
            }
        )
    return referencias


def _search_arxiv(query: str, limit: int) -> list[dict]:
    """Devuelve referencias normalizadas de arXiv (fallback fiable)."""
    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": limit}
    )
    raiz = ET.fromstring(_http_get(f"{_ARXIV_SEARCH}?{params}"))
    referencias = []
    for entry in raiz.findall(f"{_ATOM}entry"):
        titulo = (entry.findtext(f"{_ATOM}title") or "(sin título)").strip()
        autores = [
            (a.findtext(f"{_ATOM}name") or "").strip()
            for a in entry.findall(f"{_ATOM}author")
        ]
        publicado = entry.findtext(f"{_ATOM}published") or ""
        year = int(publicado[:4]) if publicado[:4].isdigit() else None
        pdf = None
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf":
                pdf = link.get("href")
                break
        referencias.append(
            {
                "title": titulo,
                "authors": autores,
                "year": year,
                "pdf_url": pdf,
                "source": "arXiv",
            }
        )
    return referencias


def _formatear(query: str, referencias: list[dict]) -> str:
    if not referencias:
        return f"Sin resultados para '{query}'."
    fuente = referencias[0]["source"]
    lineas: list[str] = []
    for i, r in enumerate(referencias, 1):
        autores = ", ".join(a for a in r["authors"][:3] if a)
        if len(r["authors"]) > 3:
            autores += " et al."
        year = r["year"] or "s.f."
        disp = f"PDF: {r['pdf_url']}" if r["pdf_url"] else "sin PDF de acceso abierto"
        lineas.append(
            f"[{i}] {r['title']} ({year}) — {autores or 'autor desconocido'}\n    {disp}"
        )
    return (
        f"Referencias para '{query}' (vía {fuente}):\n"
        + "\n".join(lineas)
        + "\n\nPara añadir una al corpus: download_reference(url_del_PDF, nombre.pdf), "
        "luego ingest_corpus."
    )


def search_references(query: str, limit: int = 5) -> str:
    """Busca papers en fuentes académicas y devuelve candidatos con su URL de PDF.

    Intenta Semantic Scholar y, si falla o no devuelve nada, recurre a arXiv.

    Args:
        query: Términos de búsqueda (tema, título, autores…).
        limit: Número máximo de resultados (por defecto 5).

    Returns:
        Un listado con título, autores, año y —cuando existe— la URL de PDF
        descargable (pasable a `download_reference`), o un mensaje de error.
    """
    limit = max(1, min(limit, 20))
    try:
        referencias = _search_semantic_scholar(query, limit)
        if referencias:
            return _formatear(query, referencias)
    except Exception:  # noqa: BLE001 — cae a arXiv ante cualquier fallo (incl. 429)
        pass
    try:
        return _formatear(query, _search_arxiv(query, limit))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR al buscar referencias (Semantic Scholar y arXiv): {exc}"


def download_reference(url: str, filename: str) -> str:
    """Descarga un PDF a `knowledge_base/` para poder ingerirlo después.

    Args:
        url: URL http(s) directa a un PDF (la que devuelve `search_references`).
        filename: Nombre con el que guardar el PDF en knowledge_base/.

    Returns:
        Confirmación con el tamaño descargado, o un mensaje de error. Tras esto,
        usa `ingest_corpus` para indexarlo.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return "ERROR: la URL debe empezar por http:// o https://."

    if not Path(filename).name or Path(filename).name in {".", ".."}:
        return f"ERROR: nombre de fichero inválido: '{filename}'."
    nombre = normalized_pdf_name(filename)
    try:
        destino = tools._resolve_within(tools.KNOWLEDGE_BASE_DIR, nombre)
    except ValueError as exc:
        return f"ERROR: {exc}"

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            datos = resp.read(_MAX_PDF_BYTES + 1)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR al descargar '{url}': {exc}"

    if len(datos) > _MAX_PDF_BYTES:
        return f"ERROR: el fichero supera el límite de {_MAX_PDF_BYTES // (1024 * 1024)} MB."
    if not datos.startswith(b"%PDF") and "pdf" not in content_type.lower():
        return "ERROR: la URL no devolvió un PDF (¿es una página web en vez del PDF?)."

    try:
        tools.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(datos)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR al guardar '{nombre}': {exc}"

    return f"Descargado '{nombre}' en knowledge_base/ ({len(datos) // 1024} KB)."


# --- Prueba de humo -----------------------------------------------------------

if __name__ == "__main__":
    print(search_references("3D gaussian splatting medical imaging", limit=3))
