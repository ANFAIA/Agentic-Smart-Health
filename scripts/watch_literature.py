#!/usr/bin/env python
"""watch_literature.py — Vigila la literatura y propone entradas del manifiesto.

    uv run python scripts/watch_literature.py                  # informe, no toca nada
    uv run python scripts/watch_literature.py --write          # añade al manifest.yaml
    uv run python scripts/watch_literature.py --resumen pr.md  # cuerpo para la PR

**Qué automatiza.** El `research-agent` ya sabe buscar en arXiv y Semantic Scholar,
pero es un REPL: descubre literatura solo mientras alguien está sentado delante.
Este script hace la parte que se repite —mirar qué ha salido esta semana, descartar
lo que ya está inventariado y averiguar bajo qué licencia se publicó— y deja el
juicio (¿es relevante para el proyecto?) donde tiene que estar: en una persona
revisando una PR.

**Ningún PDF toca el disco, ni el repositorio.** El binario se descarga a memoria
solo para calcular `sha256` y `bytes` —sin eso la entrada del manifiesto estaría
incompleta— y se libera sin escribirse en ningún sitio. Lo que se propone commitear
son ~10 líneas de YAML. Descargar no es redistribuir: lo primero lo permite
cualquier licencia de las que aparecen aquí; lo segundo es lo que costó la issue 45.

**La licencia se verifica en origen, no se supone.** Se lee del OAI-PMH de arXiv
(`verb=GetRecord`, `metadataPrefix=arXiv`), que devuelve el URI exacto que declaró
quien subió el artículo. Una licencia adivinada por el título es peor que no tener
ninguna, porque parece un dato.

**El agente no mergea.** Su salida es una rama y una PR. La decisión de si un
artículo entra en la base de conocimiento sigue siendo humana, y la PR pasa por los
mismos guardianes que cualquier otra (`data_guard.py` incluido).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
MANIFIESTO = REPO / "data" / "research-agent" / "knowledge_base" / "manifest.yaml"

_API = "http://export.arxiv.org/api/query"
_OAI = "http://export.arxiv.org/oai2"
_ATOM = "{http://www.w3.org/2005/Atom}"
_UA = "agentic-smart-health/1.0 (+https://github.com/ANFAIA/Agentic-Smart-Health)"
_TIMEOUT = 60
# Tope de descarga. Un PDF de 40 MiB existe (la primera ejecución trajo uno) y no
# tiene sentido gastarlo en calcular un hash de algo que aún nadie ha aprobado.
_MAX_PDF_BYTES = 30 * 1024 * 1024
# arXiv pide un mínimo de 3 s entre peticiones. Respetarlo es la diferencia entre
# un cliente educado y uno bloqueado.
_ESPERA = 3.0

# --------------------------------------------------------------------------- #
# Las dos puertas
# --------------------------------------------------------------------------- #
# Filtrar solo por la consulta no basta: además, título o resumen tienen que nombrar
# el ámbito. Sin puerta, «digital twin AND patient» propone modelos de amígdala y
# gemelos cardiovasculares —medido, fue el resultado de la primera ejecución— y
# «Gaussian Splatting» trae conducción autónoma cada semana.
#
# Con límites de palabra a propósito: buscar la subcadena "oral" acepta "temporal" y
# "behavioral", que es la mitad de los abstracts de medicina.
_DENTAL = re.compile(
    r"\b(dental|dentist\w*|dentition|tooth|teeth|oral|intraoral|cbct|"
    r"maxillofacial|maxilla\w*|mandib\w*|orthodont\w*|periodont\w*|endodont\w*|"
    # `occlusal` sí, `occlusion` no: en visión por computador "occlusion-aware" está
    # en un artículo de cada tres, y colaba reconstrucción de manos como si fuera
    # oclusión dental. Medido: así entró OASIS (arXiv 2607.29633) en una prueba.
    r"caries|gingiv\w*|occlusal|craniofacial|jaw|jaws)\b",
    re.IGNORECASE,
)

# La puerta de los estándares y la interoperabilidad, que NO puede ser la dental: un
# artículo sobre perfiles FHIR o sobre preparación de datos DICOM casi nunca dice
# «tooth». Medido el 2026-08-05 sobre 85 resultados de las tres consultas de abajo:
# **cero** pasaban la puerta dental. Lo que se exige aquí es que el artículo hable del
# formato o del estándar en sí, no de una patología que use ese formato.
#
# Se aplica **solo al título** (`ambito: "titulo"`), y eso no es un matiz: buscando
# también en el resumen entraban un dataset de mieloma espinal, una herramienta de
# planificación de válvula aórtica y un benchmark de MRI cerebral —todos mencionan
# DICOM de pasada—. Un artículo que *va sobre* el estándar lo dice en el título.
# Medido el 2026-08-05: con la puerta en el resumen entraban 5 de 12 falsos
# positivos; con la puerta en el título, ninguno.
#
# `metadata` se dejó fuera a propósito: en un título de ML es demasiado común
# («Metadata Supervised MRI Representations») y no implica estándar.
_INTEROPERABILIDAD = re.compile(
    r"\b(dicom|fhir|hl7|ihe|pacs|snomed|loinc|nifti|"
    r"interoperab\w*|standard\w*|ontolog\w*|terminolog\w*|"
    r"schema\w*|provenance|de-?identif\w*)\b",
    re.IGNORECASE,
)

# Las consultas van aquí, en el código, y no en un fichero de configuración: son
# siete entradas que se revisan en una PR como cualquier otro cambio de criterio.
#
# `puerta` es por consulta y no global. Lo fue, se simplificó a una puerta dental
# única, y esa simplificación es exactamente lo que impedía cubrir los estándares:
# el brief pide vigilar «Gaussian Splatting, DICOM, STL, interoperabilidad clínica»
# y solo el primero es un tema dental.
CONSULTAS: tuple[dict, ...] = (
    {
        "consulta": 'all:"Gaussian Splatting"',
        "puerta": _DENTAL,
        "ambito": "todo",
        "por_que": "El núcleo técnico del proyecto: 3DGS aplicado a la boca.",
    },
    {
        "consulta": 'all:"CBCT" AND all:"segmentation"',
        "puerta": _DENTAL,
        "ambito": "todo",
        "por_que": "Segmentación de CBCT: alimenta el cbct-agent y la fusión.",
    },
    {
        "consulta": 'all:"intraoral scan"',
        "puerta": _DENTAL,
        "ambito": "todo",
        "por_que": "Escaneo intraoral: entrada del mesh-agent y del registro "
        "malla↔volumen. Es también la vía por la que llegan los "
        "artículos de mallas STL del dominio (ver nota al pie).",
    },
    {
        "consulta": 'all:"digital twin" AND all:"patient"',
        "puerta": _DENTAL,
        "ambito": "todo",
        "por_que": "Gemelo digital del paciente: el marco del proyecto entero.",
    },
    {
        "consulta": 'all:"DICOM"',
        "puerta": _INTEROPERABILIDAD,
        "ambito": "titulo",
        "por_que": "El formato de entrada del cbct-agent. Interesa lo que se "
        "publica sobre el estándar y sobre preparación de datos, no "
        "cada estudio clínico que lo use.",
    },
    {
        "consulta": 'all:"FHIR" OR all:"HL7"',
        "puerta": _INTEROPERABILIDAD,
        "ambito": "titulo",
        "por_que": "Interoperabilidad clínica: cómo viaja el dato del paciente "
        "entre sistemas, que es el problema de silos que ataca el proyecto.",
    },
    {
        "consulta": 'all:"medical imaging" AND all:"interoperability"',
        "puerta": _INTEROPERABILIDAD,
        "ambito": "titulo",
        "por_que": "Interoperabilidad específica de imagen médica.",
    },
)

# **Por qué NO hay una consulta de STL**, aunque el brief lo nombre. Se midió el
# 2026-08-05: `all:"STL" AND all:"mesh"` devuelve 11 resultados y solo 1 es dental —
# el resto es automoción y Lattice Boltzmann, porque STL es un acrónimo polisémico
# (*Spatio-Temporal Learning*, *Standard Template Library*, estereolitografía
# industrial). Y `all:"intraoral scan" AND all:"STL"` devuelve **0**. Los artículos
# de mallas dentales llegan por la consulta de escaneo intraoral, que sí funciona.
# Una consulta de STL metería ruido de otras disciplinas sin aportar un solo
# artículo del dominio.

# URI de licencia -> (nombre legible, se puede redistribuir). Los nombres coinciden
# con los que ya usa el manifiesto para no inventar un vocabulario paralelo.
LICENCIAS: dict[str, tuple[str, bool]] = {
    "creativecommons.org/publicdomain/zero/1.0": ("CC0 1.0", True),
    "creativecommons.org/licenses/by/4.0": ("CC BY 4.0", True),
    "creativecommons.org/licenses/by-sa/4.0": ("CC BY-SA 4.0", True),
    "creativecommons.org/licenses/by-nc-sa/4.0": ("CC BY-NC-SA 4.0", False),
    "creativecommons.org/licenses/by-nc-nd/4.0": ("CC BY-NC-ND 4.0", False),
    "arxiv.org/licenses/nonexclusive-distrib/1.0": ("arXiv perpetual non-exclusive 1.0", False),
}


def _get(url: str) -> bytes:
    peticion = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(peticion, timeout=_TIMEOUT) as respuesta:
        return respuesta.read()


def _texto(nodo, etiqueta: str) -> str:
    return (nodo.findtext(f"{_ATOM}{etiqueta}") or "").strip()


def buscar(consulta: str, limite: int) -> list[dict]:
    """Artículos recientes de arXiv, del más nuevo al más viejo."""
    params = urllib.parse.urlencode(
        {
            "search_query": consulta,
            "start": 0,
            "max_results": limite,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    raiz = ET.fromstring(_get(f"{_API}?{params}"))
    articulos = []
    for entrada in raiz.findall(f"{_ATOM}entry"):
        url_id = _texto(entrada, "id")  # http://arxiv.org/abs/2508.07407v2
        identificador = url_id.rsplit("/", 1)[-1]
        articulos.append(
            {
                "id": identificador,
                "base": re.sub(r"v\d+$", "", identificador),
                "titulo": " ".join(_texto(entrada, "title").split()),
                "resumen": " ".join(_texto(entrada, "summary").split()),
                "publicado": _texto(entrada, "published"),
                "url": f"https://arxiv.org/pdf/{identificador}",
            }
        )
    return articulos


def licencia_en_origen(base_id: str) -> tuple[str, bool]:
    """Licencia declarada por el autor, leída del OAI-PMH de arXiv."""
    params = urllib.parse.urlencode(
        {
            "verb": "GetRecord",
            "identifier": f"oai:arXiv.org:{base_id}",
            "metadataPrefix": "arXiv",
        }
    )
    try:
        cuerpo = _get(f"{_OAI}?{params}").decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as e:
        return (f"no verificada ({e})", False)

    encontrada = re.search(r"<license>(.*?)</license>", cuerpo, re.S)
    if not encontrada:
        # arXiv sin licencia explícita = licencia perpetua por defecto.
        return ("arXiv perpetual non-exclusive 1.0", False)
    uri = encontrada.group(1).strip().rstrip("/").replace("https://", "").replace("http://", "")
    return LICENCIAS.get(uri, (f"sin mapear: {uri}", False))


def huella(url: str) -> tuple[str, int]:
    """`sha256` y tamaño del PDF **sin escribirlo en disco**.

    El hash se calcula sobre los bytes de la respuesta, así que no hay ningún
    motivo para materializar el fichero: el PDF vive en memoria mientras se mide y
    se libera después. Es una garantía más fuerte que un temporal que se borra —no
    hay nada que borrar, ni que se quede si matan el proceso a mitad—, y explica
    el tope: lo que no cabe cómodamente en memoria, no se descarga.
    """
    peticion = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(peticion, timeout=_TIMEOUT) as respuesta:
        anunciado = int(respuesta.headers.get("Content-Length") or 0)
        if anunciado > _MAX_PDF_BYTES:
            raise ValueError(f"{anunciado / 1048576:.0f} MiB, por encima del tope")
        # Un byte más que el tope: así se detecta el que miente en Content-Length.
        datos = respuesta.read(_MAX_PDF_BYTES + 1)
    if len(datos) > _MAX_PDF_BYTES:
        raise ValueError("por encima del tope de descarga")
    if not datos.startswith(b"%PDF"):
        raise ValueError("la respuesta no es un PDF")
    return hashlib.sha256(datos).hexdigest(), len(datos)


def nombre_fichero(titulo: str, identificador: str) -> str:
    limpio = re.sub(r"[^\w\s.-]", "_", titulo).strip()[:110].rstrip(" ._-")
    return f"{limpio} - {identificador}.pdf"


def _escalar(valor: str) -> str:
    """Cadena como escalar YAML válido.

    `json.dumps` escapa comillas y barras, y YAML 1.2 acepta la sintaxis de JSON.
    Concatenar comillas a mano no vale: los títulos y las consultas llevan `"`
    dentro (`all:"digital twin"`) y producían un manifiesto que no parseaba.
    """
    return json.dumps(valor, ensure_ascii=False)


def entrada_yaml(art: dict) -> str:
    """Bloque YAML con el mismo formato y orden de campos que el resto."""
    nota = (
        f"Descubierto por watch_literature.py el {art['visto']} "
        f"(consulta: {art['consulta']}). Relevancia sin revisar."
    )
    return (
        f"  - fichero: {_escalar(art['fichero'])}\n"
        f"    titulo: {_escalar(art['titulo'])}\n"
        f"    anio: {art['anio']}\n"
        f"    fuente: arxiv\n"
        f"    id: {_escalar(art['id'])}\n"
        f"    url: {_escalar(art['url'])}\n"
        f"    licencia: {_escalar(art['licencia'])}\n"
        f"    redistribuible: {str(art['redistribuible']).lower()}\n"
        f"    sha256: {art['sha256']}\n"
        f"    bytes: {art['bytes']}\n"
        f"    nota: {_escalar(nota)}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true", help="Añade las propuestas al manifiesto.")
    ap.add_argument("--resumen", type=Path, help="Escribe el cuerpo de la PR en este fichero.")
    ap.add_argument("--dias", type=int, default=90, help="Ventana de novedad (por defecto 90).")
    ap.add_argument("--limite", type=int, default=25, help="Resultados por consulta.")
    ap.add_argument(
        "--max-nuevos",
        type=int,
        default=5,
        help="Tope de propuestas por ejecución: una PR que nadie lee no sirve.",
    )
    args = ap.parse_args()

    datos = yaml.safe_load(MANIFIESTO.read_text(encoding="utf-8"))
    conocidos = {re.sub(r"v\d+$", "", str(d.get("id", ""))) for d in datos["documentos"]}
    corte = datetime.now(UTC) - timedelta(days=args.dias)

    candidatos: list[dict] = []
    vistos: set[str] = set()
    for i, busqueda in enumerate(CONSULTAS):
        if i:
            time.sleep(_ESPERA)
        try:
            resultados = buscar(busqueda["consulta"], args.limite)
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as e:
            print(f"✗ consulta {busqueda['consulta']}: {e}", file=sys.stderr)
            continue

        pasan = 0
        for art in resultados:
            if art["base"] in conocidos or art["base"] in vistos:
                continue
            ambito = (
                art["titulo"]
                if busqueda["ambito"] == "titulo"
                else f"{art['titulo']} {art['resumen']}"
            )
            if not busqueda["puerta"].search(ambito):
                continue
            try:
                fecha = datetime.fromisoformat(art["publicado"].replace("Z", "+00:00"))
            except ValueError:
                continue
            if fecha < corte:
                continue
            art["consulta"] = busqueda["consulta"]
            art["anio"] = fecha.year
            art["visto"] = datetime.now(UTC).date().isoformat()
            vistos.add(art["base"])
            candidatos.append(art)
            pasan += 1
        print(f"· {busqueda['consulta']}: {len(resultados)} resultados, {pasan} nuevos y del tema")

    # Reparto por consulta, no «los N más nuevos». Las consultas de estándares dan
    # mucho más volumen que las dentales (medido: 13 y 8 frente a 5 y 1), así que
    # ordenar por fecha y cortar dejaba una PR entera sin un solo artículo dental —
    # justo el tema principal del proyecto. Se recorre por turnos: cada consulta
    # aporta su más reciente, luego el segundo, hasta llenar el cupo.
    por_consulta: dict[str, list[dict]] = {}
    for art in sorted(candidatos, key=lambda a: a["publicado"], reverse=True):
        por_consulta.setdefault(art["consulta"], []).append(art)

    turnos, candidatos = list(por_consulta.values()), []
    while turnos and len(candidatos) < args.max_nuevos:
        for cola in list(turnos):
            if not cola:
                turnos.remove(cola)
                continue
            candidatos.append(cola.pop(0))
            if len(candidatos) >= args.max_nuevos:
                break
    candidatos.sort(key=lambda a: a["publicado"], reverse=True)

    if not candidatos:
        print("\nNada nuevo que proponer.")
        if args.resumen:
            args.resumen.write_text("", encoding="utf-8")
        return 0

    print(f"\nVerificando licencia en origen y calculando huella de {len(candidatos)}:")
    aceptados = []
    for art in candidatos:
        time.sleep(_ESPERA)
        art["licencia"], art["redistribuible"] = licencia_en_origen(art["base"])
        try:
            art["sha256"], art["bytes"] = huella(art["url"])
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            print(f"  ✗ {art['id']}: no se pudo leer el PDF ({e}); se omite")
            continue
        art["fichero"] = nombre_fichero(art["titulo"], art["id"])
        aceptados.append(art)
        marca = "redistribuible" if art["redistribuible"] else "NO redistribuible"
        print(f"  ✓ {art['id']}  {art['licencia']}  ({marca})  {art['bytes'] / 1048576:.1f} MiB")

    if not aceptados:
        print("\nNinguno superó la verificación.")
        return 0

    bloque = "".join(entrada_yaml(a) for a in aceptados)
    if args.write:
        # Se añade al final (`documentos` es la última clave) en vez de volcar el
        # YAML entero: un `yaml.dump` se llevaría por delante la cabecera comentada
        # del manifiesto, que es donde está explicado por qué no hay PDF aquí.
        original = MANIFIESTO.read_text(encoding="utf-8")
        MANIFIESTO.write_text(original + bloque, encoding="utf-8")
        try:
            yaml.safe_load(MANIFIESTO.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            MANIFIESTO.write_text(original, encoding="utf-8")
            print(f"\n✗ el bloque generado no parsea; manifiesto restaurado: {e}", file=sys.stderr)
            return 1
        print(f"\n{len(aceptados)} entradas añadidas a {MANIFIESTO.relative_to(REPO)}.")
    else:
        print("\n--- YAML propuesto (usa --write para añadirlo) ---")
        print(bloque, end="")

    if args.resumen:
        filas = "\n".join(
            f"| [{a['id']}]({a['url']}) | {a['titulo'][:70]} | {a['licencia']} | "
            f"{'sí' if a['redistribuible'] else '**no**'} |"
            for a in aceptados
        )
        args.resumen.write_text(
            f"{len(aceptados)} artículo(s) de arXiv publicados en los últimos "
            f"{args.dias} días que encajan con el proyecto y no estaban inventariados.\n\n"
            "| arXiv | Título | Licencia | ¿Redistribuible? |\n|---|---|---|---|\n"
            f"{filas}\n\n"
            "La licencia está **leída del OAI-PMH de arXiv**, no supuesta. Ningún PDF "
            "entra en el repositorio: se bajaron a un temporal para calcular `sha256` "
            "y se borraron.\n\n"
            "**Qué decidir al revisar:** si el artículo aporta al proyecto. Si sí, "
            "mergea; los PDF se materializan luego con "
            "`uv run python scripts/fetch_knowledge_base.py`. Si no, cierra la PR: "
            "el manifiesto es una base de conocimiento curada, no un buzón.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
