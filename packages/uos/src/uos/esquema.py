"""El **JSON Schema publicado** del manifiesto, que es lo que §12 pide del validador.

Hasta ahora validábamos con Pydantic, que es correcto y es **nuestro**: alguien ajeno no
tenía contra qué comprobar un `.uos`, ni el suyo ni el nuestro. Un formato cuya única
definición ejecutable vive dentro de la implementación de referencia no es un formato, es
una biblioteca con documentación.

El esquema se **deriva** de los modelos en vez de escribirse a mano, y ahí está la
decisión: un esquema copiado a mano empieza igual al contrato y se separa de él en el
primer campo que alguien añade, sin que nada avise. Derivándolo, la única forma de que
mientan a la vez es que el contrato ya estuviera mal.

⚠️ **Lo que este esquema NO comprueba.** Un JSON Schema valida forma, no verdad: no sabe
si un `sha256` coincide con su asset, si el grafo de frames llega al canónico, ni si un
`derived/` trae su `meta.json`. Eso lo sigue haciendo `validador.py`, y por eso el §12
pide las dos cosas y no una. Publicar el esquema añade un lector externo; no sustituye al
validador.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uos.manifiesto import UOS_VERSION, Manifiesto

# Dónde se publica dentro del repositorio. Va versionado por `uos_version` porque el §12
# lo pide así —«JSON Schema publicado por versión»— y porque un esquema sin versión no
# sirve para validar un contenedor antiguo, que es justo cuando hace falta.
RUTA = Path("schemas") / f"uos-manifest-{UOS_VERSION}.schema.json"

# El identificador del esquema. Es una URI, no una URL que resuelva: mientras el spec no
# se publique no hay dónde alojarlo, y poner una que devuelva 404 sería peor que ninguna.
ID = f"https://histora.dev/schemas/uos-manifest-{UOS_VERSION}.schema.json"


def esquema_del_manifiesto() -> dict[str, Any]:
    """El JSON Schema del manifiesto, derivado del contrato."""
    esquema = Manifiesto.model_json_schema(mode="serialization")
    esquema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    esquema["$id"] = ID
    esquema["title"] = f"UOS manifest v{UOS_VERSION}"
    esquema["description"] = (
        "Manifiesto de un contenedor Unified Oral Scene. Derivado del contrato de la "
        "implementacion de referencia, no escrito a mano: un esquema copiado se separa "
        "del codigo en el primer campo que alguien anade."
    )
    return esquema


def escribe(raiz: Path) -> Path:
    """Escribe el esquema bajo `raiz` y devuelve la ruta. Idempotente."""
    destino = raiz / RUTA
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(esquema_del_manifiesto(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destino


if __name__ == "__main__":  # pragma: no cover - utilidad de linea de comandos
    import sys

    raiz = Path(__file__).resolve().parents[4]
    print(escribe(raiz), file=sys.stderr)
