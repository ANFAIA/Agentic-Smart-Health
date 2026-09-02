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

# El identificador del esquema, y **resuelve**. Antes apuntaba a `histora.dev`, un dominio
# que nadie ha registrado: un `$id` no está obligado a descargarse —JSON Schema pide una URI,
# no una URL— pero escribir `https://` le dice a quien lo lee que haga `curl`, y fallaba. El
# esquema ya vivía en un repositorio público, así que la promesa se cumple sin comprar nada.
#
# ⚠️ Fijado a una ETIQUETA y no a `main`. Un identificador que devuelve contenidos distintos
# según el día no identifica: si esto apuntara a la rama, el mismo `$id` describiría un
# contrato u otro según cuándo se resolviera, que es justo lo que un `$id` existe para evitar.
# La etiqueta se crea al publicar cada versión del formato.
ID = (
    "https://raw.githubusercontent.com/ANFAIA/Agentic-Smart-Health/"
    f"uos-spec-v{UOS_VERSION}/schemas/uos-manifest-{UOS_VERSION}.schema.json"
)


# ── El esquema se publica EN INGLES, y el contrato se lee en castellano ────────────
#
# ⚠️ **Este fichero es el unico artefacto que un implementador ajeno usa sin leernos el
# codigo**, y salia bilingue sin que nadie lo decidiera: los nombres de campo y los valores
# de enumeracion son ingleses porque son el formato de cable, y `title` y `description` los
# generaba pydantic solo, a partir del nombre de la clase y del docstring — que estan en
# castellano porque el contrato lo leemos nosotros. El resultado era un esquema cuya mitad
# legible no la puede leer su destinatario (G-1 de la revision externa).
#
# La traduccion vive aqui y no en `manifiesto.py` a proposito: `title`/`description` son
# **decisiones de publicacion**, no del contrato, y los `StrEnum` no admiten un atributo de
# clase que no acabe siendo un miembro mas de la enumeracion. Lo que evita que se quede
# atras es la comprobacion de abajo, que revienta en cuanto aparece un modelo sin entrada.
#
# La clave del `$defs` pasa a ser el titulo ingles y las referencias se reescriben con ella,
# porque un generador de documentacion usa esa clave como encabezado.
PUBLICACION_EN: dict[str, tuple[str, str]] = {
    "Asset": ("Asset", "One file or directory the container carries or references."),
    "Parte": ("Part", "One file inside an asset that is a directory, such as a DICOM slice."),
    "Frame": ("Frame", "A coordinate system in which asset positions are expressed."),
    "Registro": ("Registration", "A rigid transform taking points from one frame to another."),
    "Visita": ("Visit", "One clinical encounter the assets belong to."),
    "Sujeto": ("Subject", "The patient, always by pseudonym."),
    "Adquisicion": ("Acquisition", "When an asset was captured, and on what equipment."),
    "Dispositivo": ("Device", "The equipment, with fixed keys and no serial number."),
    "Proyeccion": ("Projection", "What kind of 2D image an asset is, and which teeth it targets."),
    "Regulatorio": ("Regulatory", "The regulatory layer of an asset and its clearances."),
    "Autorizacion": ("Clearance", "What one jurisdiction says about this asset."),
    "Desidentificacion": (
        "Deidentification",
        "What was done to de-identify, in the vocabulary of DICOM PS3.15 Annex E.",
    ),
    "Herramienta": ("Tool", "The program that applied the de-identification."),
    "Consentimiento": ("Consent", "What the patient consented to, which bounds purpose of use."),
    "RecursoFHIR": ("FHIRResource", "Which FHIR R4 resource an asset corresponds to."),
    "Extension": ("Extension", "A format extension the container declares and may require."),
    "Procedencia": ("Provenance", "The append-only hash chain linking versions of this case."),
    "EstadoPHI": (
        "PHIState",
        "Explicit PHI state. No value of it makes a container non-personal data.",
    ),
    "EstadoRegulatorio": ("ClearanceStatus", "Closed vocabulary for a clearance status."),
    "Proposito": ("PurposeOfUse", "Closed vocabulary for what a container was issued for."),
    "Clase": ("AssetKind", "What kind of thing an asset is."),
}


def _a_ingles(esquema: dict[str, Any]) -> dict[str, Any]:
    """Reescribe `$defs`, `title` y `description` del esquema al ingles.

    Revienta si algun modelo no declara su traduccion: un esquema publicado a medias en
    castellano es peor que uno entero en castellano, porque parece traducido.
    """
    defs = esquema.get("$defs", {})
    faltan = sorted(set(defs) - set(PUBLICACION_EN))
    if faltan:
        raise SystemExit(
            "Estos modelos no declaran su titulo y descripcion en ingles para el esquema "
            "publicado:\n  " + "\n  ".join(faltan)
            + "\n\nAnadelos a `PUBLICACION_EN` en `uos/esquema.py`. El esquema es el "
            "unico artefacto\nque un implementador ajeno usa sin leer nuestro codigo."
        )
    nombres = {viejo: PUBLICACION_EN[viejo][0] for viejo in defs}

    def _traduce(nodo: Any) -> Any:
        if isinstance(nodo, dict):
            salida = {}
            for k, v in nodo.items():
                if k == "$ref" and isinstance(v, str) and v.startswith("#/$defs/"):
                    salida[k] = "#/$defs/" + nombres.get(v.removeprefix("#/$defs/"), v)
                else:
                    salida[k] = _traduce(v)
            return salida
        return [_traduce(x) for x in nodo] if isinstance(nodo, list) else nodo

    esquema = _traduce(esquema)
    esquema["$defs"] = {}
    for viejo, cuerpo in defs.items():
        titulo, descripcion = PUBLICACION_EN[viejo]
        esquema["$defs"][titulo] = {
            **_traduce(cuerpo), "title": titulo, "description": descripcion,
        }
    return esquema


def esquema_del_manifiesto() -> dict[str, Any]:
    """El JSON Schema del manifiesto, derivado del contrato."""
    esquema = _a_ingles(Manifiesto.model_json_schema(mode="serialization"))
    esquema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    esquema["$id"] = ID
    esquema["title"] = f"UOS manifest v{UOS_VERSION}"
    esquema["description"] = (
        "The manifest of a Unified Oral Scene container. Derived from the reference "
        "implementation's contract rather than written by hand: a schema copied by hand "
        "diverges from the code at the first field somebody adds."
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
