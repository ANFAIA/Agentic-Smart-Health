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

from uos.manifiesto import UOS_VERSION, Manifest

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


# ── Las descripciones del esquema publicado, en ingles ─────────────────────────────
#
# ⚠️ **Este fichero es el unico artefacto que un implementador ajeno usa sin leernos el
# codigo**, y salia bilingue sin que nadie lo decidiera: pydicom —perdon, pydantic— genera
# `title` y `description` del nombre de la clase y del docstring, y el docstring esta en
# castellano porque el contrato lo leemos nosotros (G-1).
#
# Los `title` ya no hacen falta: las clases del contrato se llaman en ingles desde que se
# renombro la API publica, asi que el titulo que pydantic genera solo ya es el correcto.
# Lo que sigue viniendo del docstring es la `description`, y eso es lo que se sustituye
# aqui — una frase, no las doce lineas de razonamiento que un docstring nuestro tiene.
#
# Vive aqui y no en los modelos porque una `description` de esquema es una decision de
# PUBLICACION, no del contrato, y porque un `StrEnum` no admite un atributo de clase que no
# acabe siendo otro miembro de la enumeracion. Lo que evita que se quede atras es la
# comprobacion de abajo, que revienta en cuanto aparece un modelo sin entrada.
DESCRIPCIONES_EN: dict[str, str] = {
    "Asset": "One file or directory the container carries or references.",
    "Part": "One file inside an asset that is a directory, such as a DICOM slice.",
    "Frame": "A coordinate system in which asset positions are expressed.",
    "Registration": "A rigid transform taking points from one frame to another.",
    "Visit": "One clinical encounter the assets belong to.",
    "Subject": "The patient, always by pseudonym.",
    "Acquisition": "When an asset was captured, and on what equipment.",
    "Device": "The equipment, with fixed keys and no serial number.",
    "Projection": "What kind of 2D image an asset is, and which teeth it targets.",
    "Regulatory": "The regulatory layer of an asset and its clearances.",
    "Clearance": "What one jurisdiction says about this asset.",
    "Deidentification": "What was done to de-identify, in the vocabulary of DICOM PS3.15 Annex E.",
    "Tool": "The program that applied the de-identification.",
    "Consent": "What the patient consented to, which bounds purpose of use.",
    "FHIRResource": "Which FHIR R4 resource an asset corresponds to.",
    "Extension": "A format extension the container declares and may require.",
    "Provenance": "The append-only hash chain linking versions of this case.",
    "PHIState": "Explicit PHI state. No value of it makes a container non-personal data.",
    "ClearanceStatus": "Closed vocabulary for a clearance status.",
    "PurposeOfUse": "Closed vocabulary for what a container was issued for.",
    "AssetKind": "What kind of thing an asset is.",
    "OcclusionRecord": "How the mandible-to-maxilla relation was recorded.",
    "RegistrationFitness": "What clinical use a registration was measured fit for.",
    "SiteKind": "A labelled site that is not a tooth.",
    "AnatomicalConvention": (
        "Axis convention of a frame. Handedness fixes chirality, not orientation."
    ),
}


def _a_ingles(esquema: dict[str, Any]) -> dict[str, Any]:
    """Sustituye las `description` en castellano que pydantic saca de los docstrings.

    Revienta si algun modelo no declara la suya: un esquema publicado a medias en
    castellano es peor que uno entero en castellano, porque parece traducido.
    """
    defs = esquema.get("$defs", {})
    faltan = sorted(set(defs) - set(DESCRIPCIONES_EN))
    if faltan:
        raise SystemExit(
            "Estos modelos no declaran su descripcion en ingles para el esquema "
            "publicado:\n  " + "\n  ".join(faltan)
            + "\n\nAnadelos a `DESCRIPCIONES_EN` en `uos/esquema.py`. El esquema es el "
            "unico artefacto\nque un implementador ajeno usa sin leer nuestro codigo."
        )
    for nombre, cuerpo in defs.items():
        cuerpo["description"] = DESCRIPCIONES_EN[nombre]
    return esquema


def esquema_del_manifiesto() -> dict[str, Any]:
    """El JSON Schema del manifiesto, derivado del contrato."""
    esquema = _a_ingles(Manifest.model_json_schema(mode="serialization"))
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
