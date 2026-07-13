"""Motor RAG: ingesta de documentos locales + búsqueda semántica.

Encapsula la conexión a una base vectorial Qdrant (en local, persistente
on-disk) y la generación de embeddings con `fastembed`. Se usa en dos fases:

  1. Ingesta  -> `ingest_directory()` / `ingest_file()`: parte cada documento
     en fragmentos (*chunks*), los vectoriza y los indexa en Qdrant.
  2. Consulta -> `query()`: dada una pregunta, recupera los fragmentos más
     relevantes de TODO el corpus por significado (no por palabra clave).

El modelo de embedding por defecto es multilingüe (español + inglés) para
cubrir tanto papers técnicos como normativas clínicas.

Usa la API de inferencia local de qdrant-client (`models.Document`), que
vectoriza el texto dentro del cliente. Qdrant en modo `path=` bloquea el
directorio para un único proceso a la vez (suficiente para el CLI de un solo
agente); para concurrencia real se migraría a Qdrant en servidor (Docker) sin
cambiar esta interfaz.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pypdf
from qdrant_client import QdrantClient, models

from . import tools

# --- Configuración por defecto ------------------------------------------------

DEFAULT_QDRANT_PATH = tools.PROJECT_ROOT / "data" / "research-agent" / ".qdrant_data"
DEFAULT_COLLECTION = "papers"
# Modelo pequeño y multilingüe (ES + EN), 384 dim, ~0.22 GB.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Troceado por caracteres con solape (mantiene contexto entre fragmentos).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Formatos que sabemos ingerir.
_INGESTIBLE_SUFFIXES = {".pdf", *tools._TEXT_SUFFIXES}


class RAGEngine:
    """Base vectorial local para ingerir y consultar literatura científica."""

    def __init__(
        self,
        qdrant_path: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        path = str(qdrant_path or os.getenv("QDRANT_PATH") or DEFAULT_QDRANT_PATH)
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    # --- Utilidades internas --------------------------------------------------

    def _ensure_collection(self) -> None:
        """Crea la colección con la dimensión del modelo si aún no existe."""
        if self.client.collection_exists(self.collection_name):
            return
        dim = self.client.get_embedding_size(self.embedding_model)
        self.client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )

    def _extract_text(self, ruta: Path) -> str:
        """Extrae el texto completo (sin truncar) de un PDF o fichero de texto."""
        sufijo = ruta.suffix.lower()
        if sufijo == ".pdf":
            lector = pypdf.PdfReader(str(ruta))
            return "\n".join(pagina.extract_text() or "" for pagina in lector.pages)
        if sufijo in tools._TEXT_SUFFIXES:
            return ruta.read_text(encoding="utf-8", errors="replace")
        raise ValueError(f"formato no soportado: '{sufijo}'")

    def _chunk(self, texto: str) -> list[str]:
        """Trocea el texto en ventanas solapadas de tamaño fijo."""
        texto = " ".join(texto.split())  # normaliza espacios y saltos de línea
        if not texto:
            return []
        paso = CHUNK_SIZE - CHUNK_OVERLAP
        fragmentos = [texto[i : i + CHUNK_SIZE] for i in range(0, len(texto), paso)]
        return [f for f in fragmentos if f.strip()]

    # --- Ingesta --------------------------------------------------------------

    def ingest_file(self, filename: str) -> str:
        """Ingiere un documento de `knowledge_base/` en la base vectorial.

        Args:
            filename: Ruta del documento relativa a `knowledge_base/`.

        Returns:
            Mensaje con el número de fragmentos indexados, o un error.
        """
        try:
            ruta = tools._resolve_within(tools.KNOWLEDGE_BASE_DIR, filename)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if not ruta.is_file():
            return f"ERROR: '{filename}' no existe en knowledge_base/."

        try:
            texto = self._extract_text(ruta)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR al leer '{filename}': {exc}"

        fragmentos = self._chunk(texto)
        if not fragmentos:
            return f"'{filename}' no contiene texto extraíble; nada que ingerir."

        rel = str(ruta.relative_to(tools.KNOWLEDGE_BASE_DIR))
        self._ensure_collection()
        puntos = [
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector=models.Document(text=frag, model=self.embedding_model),
                payload={"source": rel, "chunk": i, "document": frag},
            )
            for i, frag in enumerate(fragmentos)
        ]
        self.client.upsert(self.collection_name, points=puntos)
        return f"Ingerido '{rel}': {len(fragmentos)} fragmentos indexados."

    def ingest_directory(self) -> str:
        """Ingiere todos los documentos soportados de `knowledge_base/`."""
        resultados: list[str] = []
        for ruta in sorted(tools.KNOWLEDGE_BASE_DIR.rglob("*")):
            if ruta.is_dir() or ruta.name.startswith("."):
                continue
            if ruta.suffix.lower() not in _INGESTIBLE_SUFFIXES:
                continue
            rel = str(ruta.relative_to(tools.KNOWLEDGE_BASE_DIR))
            resultados.append(self.ingest_file(rel))

        if not resultados:
            return "No hay documentos ingeribles (.pdf/.md/.txt) en knowledge_base/."
        return "\n".join(resultados)

    # --- Consulta -------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> str:
        """Recupera los fragmentos más relevantes del corpus para una pregunta.

        Args:
            question: Pregunta o tema a buscar en la literatura indexada.
            top_k: Número de fragmentos a devolver (por defecto 5).

        Returns:
            Los fragmentos relevantes con su fuente y puntuación, o un aviso
            si aún no hay nada indexado.
        """
        if not self.client.collection_exists(self.collection_name):
            return (
                "Aún no hay nada indexado. Ejecuta ingest_directory() antes de "
                "consultar."
            )

        resultado = self.client.query_points(
            self.collection_name,
            query=models.Document(text=question, model=self.embedding_model),
            limit=top_k,
        )
        if not resultado.points:
            return "Sin resultados relevantes para esa consulta."

        bloques: list[str] = []
        for i, punto in enumerate(resultado.points, 1):
            payload = punto.payload or {}
            fuente = payload.get("source", "desconocida")
            texto = payload.get("document", "")
            bloques.append(
                f"[{i}] fuente: {fuente} | score: {punto.score:.3f}\n{texto}"
            )
        return "\n\n".join(bloques)

    def count(self) -> int:
        """Número de fragmentos indexados en la colección (0 si no existe)."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(self.collection_name).count

    def close(self) -> None:
        """Cierra el cliente Qdrant y libera el lock del directorio on-disk."""
        self.client.close()


# --- Prueba de humo end-to-end ------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # Usa una BD Qdrant temporal y aislada para no tocar la real.
    with tempfile.TemporaryDirectory() as tmp:
        rag = RAGEngine(qdrant_path=tmp, collection_name="_smoke")

        # Documento de prueba dentro de knowledge_base/ (respeta el sandbox).
        muestra = tools.KNOWLEDGE_BASE_DIR / "_smoke_rag.md"
        muestra.write_text(
            "3D Gaussian Splatting es una técnica de renderizado en tiempo real "
            "que representa escenas mediante gaussianas 3D. El estándar DICOM "
            "define el formato de imágenes médicas y sus metadatos clínicos.",
            encoding="utf-8",
        )
        try:
            print(rag.ingest_directory())
            print("Fragmentos indexados:", rag.count())
            print("\n--- Consulta: '¿qué es el estándar DICOM?' ---")
            print(rag.query("¿qué es el estándar DICOM?", top_k=2))
        finally:
            muestra.unlink(missing_ok=True)
            rag.close()
