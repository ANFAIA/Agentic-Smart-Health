"""`report-agent` — informe clínico (PDF/texto) → soporte **regional** (pH por FDI).

Modalidad `report`, soporte `REGIONAL`. Es el **único agente de ingesta con un LLM
de verdad**, y solo cuando hace falta: es la única modalidad cuya entrada **no
tiene esquema**. Un DICOM y un OBJ son formatos; un informe es prosa.

Dos backends, misma salida (`list[RegionalObservation]`):

| backend | cómo extrae | cuándo usarlo |
|---|---|---|
| `rules` (por defecto) | regex sobre el texto, línea a línea | informes tabulados;
  **determinista**, sin red, sin coste — es el que corre en CI |
| `llm` | Claude con *structured output* (tool use) | prosa libre, sinónimos, negaciones |
| `ollama` | modelo local con esquema forzado por gramática | lo mismo, **sin coste y sin
  que el informe salga de la máquina** — que en la modalidad de prosa libre es lo que
  manda (ADR de anonimización §3) |

Los dos cubren los mismos campos —pH, anatomía radicular y hallazgos— y devuelven la
misma forma. Lo único que cambia es quién lee el texto.

**Por qué `rules` es el defecto.** No sobre-agentificar: el LLM entra donde la
entrada es ambigua, no por costumbre. Además el backend determinista da el
suelo medible contra el que comparar al LLM, y ya está medido: **97,8% en informes
tabulados, 43,5% en prosa** sobre el corpus de `report_corpus.py`. Ese reparto es el
que decide dónde vale la pena gastar una llamada al modelo.

**Human-in-the-loop.** El agente **no** decide qué se persiste: emite cada
observación con su `Provenance.confidence` y es el orquestador quien aplica el
umbral y para el flujo si hace falta revisión humana. Separar extracción de
decisión mantiene la responsabilidad única.

**Descartar no es perder.** Un valor que no supera la validación de la ontología
—un `pH 74` que era un `7.4`, un `Diente 19` que no existe— se rechaza, pero
**nunca en silencio**: se registra como `Discard` con su motivo, entra en el
`detail` del resultado y baja la confianza por debajo del umbral del gate. Si no,
el informe diría una cosa y el twin otra sin que nadie pudiera notarlo — el fallo
clínico silencioso que el ADR 003 nombra como riesgo.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from core_schemas import (
    ClinicalAttributes,
    Hallazgo,
    Medida,
    Modality,
    RegionalObservation,
    Support,
)

from ingestion_agents import ontology
from ingestion_agents.base import BaseIngestionAgent, IngestionOutput

# Token con pinta de código dental: dos dígitos juntos ("16") o separados por
# punto ("1.6"), sin formar parte de un número más largo. Acepta CUALQUIER par de
# dígitos y delega la validez en la ontología, en vez de filtrarla en el patrón.
# Es lo que permite distinguir «aquí no había ningún diente» de «había un 19, que
# no existe» — y por tanto reportar el descarte con el motivo correcto.
_TOOTH_TOKEN_RE = re.compile(r"(?<![\d.,])(\d)\.?(\d)(?![\d.,])")
# Valor de pH: "pH 5.4", "pH: 5,4", "pH = 5.4".
_PH_RE = re.compile(r"\bpH\b\s*[:=]?\s*(\d{1,2}(?:[.,]\d+)?)", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_LLM_MODEL = "claude-sonnet-5"
# Modelo local por defecto. Qwen3 14B a Q4_K_M ocupa ~9 GB —entra en una GPU de 12 GB—
# y es el mejor en español de su clase de tamaño, que es el eje real de esta tarea:
# los informes son de 300 caracteres, así que ni el contexto ni la velocidad mandan.
_LOCAL_MODEL = os.getenv("REPORT_AGENT_LOCAL_MODEL", "qwen3:14b")
_OLLAMA_HOST_DEFAULT = "http://localhost:11434"
# Ventana de contexto del backend local. Un informe cabe de sobra; se fija explícita
# porque el defecto de Ollama puede truncar el esquema + el informe sin avisar.
_LOCAL_NUM_CTX = 8192
_BACKENDS = ("rules", "llm", "ollama")
# Confianza que se asigna a una extracción por regex: alta pero no 1.0 — el patrón
# acertó, pero nadie ha verificado que el informe dijera lo que parece decir.
_RULES_CONFIDENCE = 0.9
# Confianza cuando se descartó algún hallazgo. Está POR DEBAJO del umbral de
# human-in-the-loop del orquestador (0.7) a propósito: un descarte significa que
# el informe decía algo que el twin no recoge, y eso lo decide una persona.
_DISCARD_CONFIDENCE = 0.6
# Cuántos descartes se detallan en el `detail` antes de resumir el resto.
_MAX_DISCARDS_IN_DETAIL = 5
# Confianza que se asume si el modelo no la declara. Deliberadamente baja: una
# extracción sin confianza declarada no es una extracción segura.
_LLM_CONFIDENCE_FALLBACK = 0.5
# Campos enteros de `ClinicalAttributes` que el backend LLM puede proponer. Sus
# límites NO se copian aquí: se leen del contrato (`_limite`).
_CAMPOS_ENTEROS = ("n_raices", "n_conductos")


@dataclass(frozen=True)
class Discard:
    """Un hallazgo que el informe declaraba y la ingesta **no** escribió al contrato.

    Existe para que un descarte no sea silencioso. El caso que lo motiva: un
    informe dice «Diente 47: pH 74» (un 7.4 mal tecleado o mal OCReado); el valor
    se rechaza —correctamente— pero, sin este registro, el twin acabaría sin ese
    diente y **nadie sabría que se perdió algo**. Es el fallo clínico silencioso
    que el ADR 003 nombra como riesgo.
    """

    line: str
    reason: str

    def __str__(self) -> str:
        recorte = self.line if len(self.line) <= 70 else self.line[:67] + "…"
        return f"«{recorte}» → {self.reason}"


class RuleExtraction(NamedTuple):
    """Resultado del backend determinista: lo que entra **y lo que se cae**."""

    findings: dict[str, float]
    discards: list[Discard]


# --------------------------------------------------------------------------- #
# Extracción de texto
# --------------------------------------------------------------------------- #
def extract_text(path: Path) -> str:
    """Texto plano del informe. `.txt`/`.md` directo; `.pdf` vía `pypdf`."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "Leer PDF requiere el extra `pdf` de `ingestion-agents` (pypdf)."
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"`report-agent` no sabe leer {suffix!r} (usa .pdf, .txt o .md).")


def report_date(text: str) -> datetime | None:
    """Fecha del informe (`YYYY-MM-DD`), si la declara."""
    match = _DATE_RE.search(text)
    if not match:
        return None
    year, month, day = (int(group) for group in match.groups())
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None  # p. ej. "2026-13-45": el patrón encaja pero la fecha no existe


# --------------------------------------------------------------------------- #
# Backend determinista
# --------------------------------------------------------------------------- #
def extract_ph_by_rules(text: str) -> RuleExtraction:
    """Extrae `{código FDI: pH}` línea a línea, **y registra lo que descarta**.

    Se procesa por líneas porque un informe dental enumera un hallazgo por línea:
    emparejar un pH con el diente de *su* línea evita el fallo silencioso de
    colgar un valor del diente equivocado.

    Descartar es correcto —mejor no ingerir que ingerir mal—, pero **descartar en
    silencio no lo es**: cada línea que menciona un pH y no acaba en el contrato
    se devuelve como `Discard` con su motivo, para que el orquestador pueda
    pararlo en el gate humano en vez de perder el dato sin dejar rastro.
    """
    found: dict[str, float] = {}
    discards: list[Discard] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        ph_match = _PH_RE.search(line)
        if not ph_match:
            continue  # línea sin pH: no es un hallazgo candidato, no es un descarte

        value = float(ph_match.group(1).replace(",", "."))
        if not ontology.PH.accepts(value):
            discards.append(
                Discard(
                    line,
                    f"pH {value:g} fuera del rango plausible "
                    f"({ontology.PH.minimum:g}–{ontology.PH.maximum:g})",
                )
            )
            continue

        # Se ignoran los dígitos del propio valor de pH al buscar el diente.
        rest = line[: ph_match.start()] + " " + line[ph_match.end() :]
        candidates = _TOOTH_TOKEN_RE.findall(rest)
        code = next(
            (f"{q}{p}" for q, p in candidates if ontology.is_valid_fdi(f"{q}{p}")), None
        )
        if code is None:
            discards.append(
                Discard(
                    line,
                    "código FDI inexistente en la línea"
                    if candidates
                    else "pH sin diente asociado en la línea",
                )
            )
            continue

        if code in found:
            discards.append(Discard(line, f"el diente {code} ya tenía un pH en el informe"))
            continue
        found[code] = value

    return RuleExtraction(found, discards)


# --------------------------------------------------------------------------- #
# Informes de CBCT con IA: anatomía radicular y hallazgos por pieza
# --------------------------------------------------------------------------- #
# Estos informes NO traen pH — que era lo único que la capa regional sabía recoger — y sí
# traen, línea a línea, «Diente 16 Diente, 3 raíces, 4 canales, Signos de caries». Es
# exactamente el soporte regional del contrato, en un formato que el extractor de pH no
# ve. Sin esto, un informe con hallazgos de 28 piezas se ingería con 0 observaciones.
_DIENTE_RE = re.compile(r"Diente\s+(\d{2})\s+(.*?)(?=Diente\s+\d{2}\s|\Z)", re.S)
_RAICES_RE = re.compile(r"(\d+)\s+ra[íi](?:z|ces)\b", re.IGNORECASE)
# El informe alterna «conducto» y «canal» para lo mismo. Un vocabulario controlado existe
# precisamente para que esa variación no llegue al twin.
_CONDUCTOS_RE = re.compile(r"(\d+)\s+(?:conducto|canal)e?s?\b", re.IGNORECASE)

_HALLAZGOS_RE: tuple[tuple[re.Pattern[str], Hallazgo], ...] = (
    (re.compile(r"\bausente\b", re.I), Hallazgo.AUSENTE),
    (re.compile(r"\bcaries\b", re.I), Hallazgo.CARIES),
    (re.compile(r"\brestauraci[óo]n\b", re.I), Hallazgo.RESTAURACION),
    (re.compile(r"c[áa]lculo\s+pulpar", re.I), Hallazgo.CALCULO_PULPAR),
    (re.compile(r"p[ée]rdida\s+de\s+hueso\s+periodontal", re.I),
     Hallazgo.PERDIDA_OSEA_PERIODONTAL),
    (re.compile(r"aparato\s+de\s+ortodoncia", re.I), Hallazgo.APARATO_ORTODONCICO),
)


# Un índice de instrumentación con su rango de referencia al lado. La forma la fija el
# informe, no nosotros:
#
#     POC TA      88.09%  I        83≤(%)≤100
#     ASIM         4.58%          -10≤(%)≤10
#
# El nombre puede llevar espacios (`POC TA`, `POC ECM`), la lateralidad es opcional y el
# rango puede venir con cualquiera de los dos extremos negativos.
_INDICE_RE = re.compile(
    r"^\s*(?P<nombre>[A-Z][A-Z ]{1,20}?)\s+"
    r"(?P<valor>-?\d+(?:[.,]\d+)?)\s*(?P<unidad>%|mm|N)?\s*"
    r"(?P<lado>[IDA])?\s+"
    r"(?P<lo>-?\d+(?:[.,]\d+)?)\s*≤\s*\(?(?P=unidad)?\)?\s*≤\s*(?P<hi>-?\d+(?:[.,]\d+)?)\s*$",
    re.MULTILINE,
)


def extract_medidas_by_rules(texto: str) -> list[dict]:
    """Índices del informe con el rango de referencia que el propio informe declara.

    **No se interpreta ninguno.** No hay tabla de qué significa `TORS` ni `POC ECM`, y no
    debería haberla: inventarle semántica clínica a la sigla de un fabricante es
    exactamente el tipo de suposición que este pipeline no hace. Lo que sí se puede hacer
    —y es mucho— es capturar el valor **junto al intervalo que el documento imprime al
    lado**, porque eso permite señalar lo anómalo sin entender la magnitud.

    Medido sobre un estudio real de oclusión: ocho índices, y **dos fuera de su propio
    rango normal** (`TORS` 89,34 con normal 90-100; `POC ECM` 81,20 con normal 83-100).
    Antes de esto el informe entero salía con cero hallazgos y confianza 0,00.

    Solo se recogen las líneas que traen rango. Un número suelto sin intervalo no se puede
    ni validar ni señalar, y meterlo aquí solo añadiría ruido con aspecto de dato.
    """
    fuera = []
    for m in _INDICE_RE.finditer(texto):
        nombre = " ".join(m.group("nombre").split())
        if len(nombre) < 2:
            continue
        fuera.append({
            "nombre": nombre,
            "valor": float(m.group("valor").replace(",", ".")),
            "unidad": m.group("unidad") or "",
            "normal_min": float(m.group("lo").replace(",", ".")),
            "normal_max": float(m.group("hi").replace(",", ".")),
            "lado": m.group("lado"),
            "texto": " ".join(m.group(0).split()),
        })
    return fuera


def extract_hallazgos_by_rules(text: str) -> dict[str, dict]:
    r"""`FDI → {n_raices, n_conductos, hallazgos}` de un informe tabulado por diente.

    El texto se parte en bloques por el marcador `Diente NN`, no línea a línea: el PDF
    corta descripciones largas a mitad («…Signos de caries (Dentina,» + salto), y un
    extractor por líneas perdería el resto del hallazgo sin avisar.

    Solo se acepta un código FDI válido: el patrón `\d{2}` casaría con cualquier par de
    dígitos, y `ontology.is_valid_fdi` es la misma puerta que usa el extractor de pH.
    """
    fuera: dict[str, dict] = {}
    for codigo, cuerpo in _DIENTE_RE.findall(text):
        if not ontology.is_valid_fdi(codigo):
            continue
        raices = _RAICES_RE.search(cuerpo)
        conductos = _CONDUCTOS_RE.search(cuerpo)
        hallazgos = [h for patron, h in _HALLAZGOS_RE if patron.search(cuerpo)]
        if not (raices or conductos or hallazgos):
            continue
        fuera[codigo] = {
            "n_raices": int(raices.group(1)) if raices else None,
            "n_conductos": int(conductos.group(1)) if conductos else None,
            "hallazgos": hallazgos,
        }
    return fuera


def _describe_discards(discards: list[Discard]) -> str:
    """Resumen legible de los descartes para el `detail` del resultado."""
    cabeza = "; ".join(str(d) for d in discards[:_MAX_DISCARDS_IN_DETAIL])
    resto = len(discards) - _MAX_DISCARDS_IN_DETAIL
    sufijo = f" (+{resto} más)" if resto > 0 else ""
    plural = "s" if len(discards) != 1 else ""
    return f"{len(discards)} hallazgo{plural} descartado{plural}: {cabeza}{sufijo}"


# --------------------------------------------------------------------------- #
# Backend LLM (structured output)
# --------------------------------------------------------------------------- #
_EXTRACTION_TOOL: dict[str, Any] = {
    "name": "registrar_dientes",
    "description": (
        "Registra, diente a diente, lo que el informe afirma explícitamente: el pH "
        "medido, la anatomía radicular y los hallazgos del vocabulario controlado. "
        "Incluye un diente solo si el informe dice algo de él, y omite los campos que "
        "ese diente no declare."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dientes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fdi": {
                            "type": "string",
                            # El patrón no sustituye a `ontology.is_valid_fdi` —admite el
                            # 56, que no existe— pero impide lo que la validación no
                            # puede recuperar: la cadena vacía. Medido con un modelo
                            # local, que rellenaba `fdi=""` y hacía perder el diente
                            # entero; con el patrón, la decodificación restringida ya no
                            # deja construir ese token.
                            "pattern": "^[1-8][1-8]$",
                            "description": "Código ISO-FDI de dos dígitos, p. ej. '16'.",
                        },
                        "ph": {"type": "number", "minimum": 3.0, "maximum": 9.0},
                        "n_raices": {"type": "integer", "minimum": 1, "maximum": 5},
                        "n_conductos": {"type": "integer", "minimum": 1, "maximum": 8},
                        "hallazgos": {
                            "type": "array",
                            "items": {"type": "string", "enum": [h.value for h in Hallazgo]},
                            "description": (
                                "Vocabulario CERRADO. Lo que no esté en la lista se omite; "
                                "no se aproxima al término más parecido."
                            ),
                        },
                        "confianza": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["fdi", "confianza"],
                },
            }
        },
        "required": ["dientes"],
    },
}
"""Esquema de la tool: la barrera que impide que el modelo devuelva prosa.

`ph` dejó de ser obligatorio cuando el backend pasó a cubrir la anatomía: un informe
de CBCT con IA no mide pH, y exigirlo obligaba al modelo a inventarlo o a callarse el
diente entero. El vocabulario de `hallazgos` va como `enum` **generado desde
`Hallazgo`**, no copiado: añadir un término al contrato lo añade aquí sin tocar nada.
"""

_SYSTEM_PROMPT = (
    "Eres un extractor de datos de informes odontológicos. Devuelves únicamente lo "
    "que el informe afirma explícitamente. No infieres, no completas dientes que no "
    "aparecen y no conviertes notaciones dudosas: si un código dental es ambiguo, "
    "baja la confianza en vez de adivinar. Numeración ISO-FDI de dos dígitos.\n"
    "\n"
    "Reglas que se incumplen con más frecuencia:\n"
    "- Una negación no es un hallazgo. «Sin signos de caries» significa que NO hay "
    "caries: no la registres.\n"
    "- Un antecedente no es el estado actual. «Caries tratada en 2024, hoy "
    "restaurada» registra la restauración, no la caries.\n"
    "- El vocabulario de hallazgos es cerrado. Si el informe describe algo que no "
    "está en la lista —una fractura radicular, por ejemplo—, omítelo.\n"
    "- Omite el campo que el informe no declare. Un informe de CBCT no trae pH y uno "
    "de pH no trae anatomía radicular: devolverlos vacíos es correcto.\n"
    "- Un rango son varias piezas. «Dientes 16-18» son el 16, el 17 y el 18: "
    "enumera cada una por separado.\n"
    "- Si el informe usa otra numeración (Universal, Palmer) y no declara cuál, no "
    "la conviertas: omite el diente."
)
"""Las reglas del prompt salen de lo que el corpus mide que falla, no de la intuición.

Cada viñeta corresponde a una familia de `report_corpus.py` (negación, antecedente,
vocabulario, rango, notación ajena). Si mañana el corpus mide un fallo nuevo, la
viñeta se añade aquí y `scripts/eval_informes.py` dice si sirvió de algo.
"""


class LLMExtraction(NamedTuple):
    """Resultado del backend LLM: pH, campos clínicos **y lo que se cayó**."""

    findings: dict[str, tuple[float, float]]
    """`FDI → (pH, confianza)`."""
    clinicos: dict[str, dict[str, Any]]
    """`FDI → {n_raices, n_conductos, hallazgos}`, con los campos no declarados ausentes."""
    discards: list[Discard]


def _limite(campo: str) -> tuple[int, int]:
    """Rango que **el contrato** impone a un campo entero de `ClinicalAttributes`.

    Se lee de `model_fields` en vez de copiarse aquí: duplicar el 1–5 de las raíces
    crearía dos verdades que se separan en cuanto alguien toque una. La barrera del
    agente y la del contrato tienen que ser la misma o el agente deja pasar cosas que
    después revientan en `RegionalObservation`.
    """
    minimo = maximo = None
    for restriccion in ClinicalAttributes.model_fields[campo].metadata:
        minimo = getattr(restriccion, "ge", minimo)
        maximo = getattr(restriccion, "le", maximo)
    if minimo is None or maximo is None:  # pragma: no cover - el contrato los declara
        raise RuntimeError(f"`ClinicalAttributes.{campo}` no declara rango.")
    return int(minimo), int(maximo)


def _propuesta(item: Mapping[str, Any]) -> str:
    """La propuesta del modelo tal cual, para que el descarte diga qué se cayó."""
    return " ".join(f"{clave}={valor}" for clave, valor in item.items())


def _confianza(valor: object) -> float:
    return (
        min(1.0, max(0.0, float(valor)))  # type: ignore[arg-type]
        if isinstance(valor, int | float)
        else _LLM_CONFIDENCE_FALLBACK
    )


def valida_propuestas(items: Iterable[Mapping[str, Any]]) -> LLMExtraction:
    """Filtra lo que el modelo propone contra la ontología y el contrato.

    **Pura y sin red.** Toda la política del backend `llm` vive aquí, así que se
    prueba entera sin `anthropic` instalado y sin clave. Lo único que queda fuera es
    la llamada HTTP, que no tiene decisiones dentro.

    **Se valida campo a campo, no diente a diente.** Un informe que dice «Diente 16,
    3 raíces, pH 74» trae un valor malo y dos buenos; tumbar la pieza entera perdería
    los dos buenos. Cae solo el que falla, y cae **registrado**: que lo haya propuesto
    un LLM no es motivo para perderlo en silencio, igual que no lo es en el regex.

    **Por qué la validación no puede delegarse en Pydantic.** `ClinicalAttributes`
    también rechazaría `n_raices=9`, pero lanzando — y el envoltorio *fail-loud* del
    agente convertiría eso en un informe **entero** en `FAILED` y en cuarentena. Un
    valor implausible tiene que costar ese valor, no el documento.
    """
    findings: dict[str, tuple[float, float]] = {}
    clinicos: dict[str, dict[str, Any]] = {}
    discards: list[Discard] = []

    for item in items:
        code = str(item.get("fdi", "")).replace(".", "")
        if not ontology.is_valid_fdi(code):
            discards.append(
                Discard(_propuesta(item), "código FDI inexistente (propuesto por el LLM)")
            )
            continue

        confianza = _confianza(item.get("confianza"))
        campos: dict[str, Any] = {}

        ph = item.get("ph")
        if isinstance(ph, int | float):
            valor = float(ph)
            if not ontology.PH.accepts(valor):
                discards.append(
                    Discard(
                        f"fdi={code} pH={valor:g}",
                        f"pH fuera del rango plausible "
                        f"({ontology.PH.minimum:g}–{ontology.PH.maximum:g})",
                    )
                )
            elif code in findings:
                discards.append(
                    Discard(f"fdi={code} pH={valor:g}", f"el diente {code} ya tenía un pH")
                )
            else:
                findings[code] = (valor, confianza)
        elif ph is not None:
            discards.append(Discard(f"fdi={code} pH={ph!r}", "el pH no es un número"))

        for campo in _CAMPOS_ENTEROS:
            propuesto = item.get(campo)
            if propuesto is None:
                continue
            minimo, maximo = _limite(campo)
            if not isinstance(propuesto, int) or isinstance(propuesto, bool):
                discards.append(
                    Discard(f"fdi={code} {campo}={propuesto!r}", f"{campo} no es un entero")
                )
            elif not minimo <= propuesto <= maximo:
                discards.append(
                    Discard(
                        f"fdi={code} {campo}={propuesto}",
                        f"{campo} fuera del rango del contrato ({minimo}–{maximo})",
                    )
                )
            else:
                campos[campo] = propuesto

        hallazgos: list[Hallazgo] = []
        for propuesto in item.get("hallazgos") or []:
            try:
                hallazgos.append(Hallazgo(str(propuesto)))
            except ValueError:
                discards.append(
                    Discard(
                        f"fdi={code} hallazgo={propuesto!r}",
                        "hallazgo fuera del vocabulario controlado",
                    )
                )
        if hallazgos:
            campos["hallazgos"] = hallazgos

        if campos:
            clinicos.setdefault(code, {}).update(campos)

    return LLMExtraction(findings, clinicos, discards)


def extract_by_llm(text: str, *, model: str = _LLM_MODEL) -> LLMExtraction:
    """Extrae con Claude (salida forzada por esquema) y valida lo que devuelve.

    Aquí no hay política: se llama al modelo y se delega en `valida_propuestas`, que
    es lo que decide qué entra. La separación es lo que permite probar el backend
    completo sin red — y lo que deja claro que el esquema de la tool no basta como
    barrera: el modelo puede respetarlo y aun así proponer un diente que no existe.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - depende del extra `llm`
        raise RuntimeError(
            "El backend `llm` requiere el extra `llm` de `ingestion-agents` (anthropic)."
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Falta ANTHROPIC_API_KEY para el backend `llm` del report-agent.")

    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "registrar_dientes"},
        messages=[{"role": "user", "content": text}],
    )
    propuestas = [
        item
        for block in response.content
        if getattr(block, "type", None) == "tool_use"
        for item in block.input.get("dientes", [])  # type: ignore[union-attr]
    ]
    return valida_propuestas(propuestas)


def extract_by_local_llm(
    text: str, *, model: str = _LOCAL_MODEL, host: str | None = None
) -> LLMExtraction:
    """Igual que `extract_by_llm`, con un modelo local servido por Ollama.

    **Gratis en las dos monedas.** Cero coste por token y —lo que aquí pesa más— el
    informe no sale de la máquina. El informe es la única modalidad cuya entrada es
    prosa libre: el `cbct-agent` puede quitar identificadores porque el DICOM los
    tiene en campos con nombre, pero un nombre o una fecha de nacimiento pueden estar
    en cualquier frase de un informe. Mandarlo a un tercero es justo lo que minimiza
    el §3 del ADR de anonimización.

    **Y es reproducible.** `temperature=0` con semilla fija da el mismo resultado hoy
    y dentro de un año; una API no lo garantiza. En un repo cuyo eje es la
    reversibilidad, eso no es un detalle de coste.

    **El esquema lo impone el runtime, no el modelo.** Ollama compila el JSON Schema a
    una gramática y anula los tokens que la romperían, así que la salida *estructura*
    válida está garantizada con cualquier modelo. Lo que sigue dependiendo del modelo
    —y lo que mide `scripts/eval_informes.py`— son los **valores**: si lee bien una
    negación, si distingue un antecedente del estado actual, si respeta el vocabulario.
    Por eso se reutiliza `valida_propuestas` sin cambiar una línea: la barrera semántica
    es la misma venga de donde venga la propuesta.
    """
    try:
        from ollama import Client
    except ImportError as exc:  # pragma: no cover - depende del extra `local`
        raise RuntimeError(
            "El backend `ollama` requiere el extra `local` de `ingestion-agents` (ollama)."
        ) from exc

    # `OLLAMA_HOST` se lee **al llamar**, no al importar: si se fijase como constante
    # de módulo, apuntar a otro servidor exigiría reimportar el paquete.
    servidor = host or os.getenv("OLLAMA_HOST", _OLLAMA_HOST_DEFAULT)
    respuesta = Client(host=servidor).chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        # El esquema de la tool vale tal cual como esquema de respuesta: es el mismo
        # objeto `{dientes: [...]}`. Una sola definición para los dos backends.
        format=_EXTRACTION_TOOL["input_schema"],
        # Determinismo: mismo informe, mismo resultado. Y `think=False` porque el
        # razonamiento en voz alta de los modelos híbridos no cabe en un esquema
        # cerrado — aquí no se pide criterio, se pide leer lo que pone.
        options={"temperature": 0, "seed": 0, "num_ctx": _LOCAL_NUM_CTX},
        think=False,
    )
    contenido = respuesta.message.content or ""
    try:
        propuestas = json.loads(contenido).get("dientes", [])
    except json.JSONDecodeError as exc:
        # No debería pasar con decodificación restringida, pero si pasa es un fallo
        # del backend, no un descarte: que lo vea el envoltorio fail-loud.
        raise RuntimeError(
            f"El modelo local devolvió algo que no es JSON: {contenido[:200]}"
        ) from exc
    return valida_propuestas(propuestas)


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
class ReportAgent(BaseIngestionAgent):
    """Ingiere el informe clínico y produce la capa dispersa de atributos por FDI."""

    name = "report-agent"
    version = "0.1.0"
    modality = Modality.REPORT
    support = Support.REGIONAL

    def __init__(
        self,
        *,
        backend: str = "rules",
        model: str | None = None,
        default_timestamp: datetime | None = None,
        quarantine_dir: str | Path | None = None,
    ) -> None:
        super().__init__(quarantine_dir=quarantine_dir)
        if backend not in _BACKENDS:
            raise ValueError(
                f"Backend desconocido: {backend!r} (usa {' o '.join(map(repr, _BACKENDS))})."
            )
        self.backend = backend
        # El modelo por defecto depende del backend: no hay uno que sirva a los tres.
        self.model = model or {"llm": _LLM_MODEL, "ollama": _LOCAL_MODEL}.get(backend, "")
        self.default_timestamp = default_timestamp

    def _ingest(self, source: Path) -> IngestionOutput:
        text = extract_text(source)
        if not text.strip():
            raise ValueError(f"El informe no contiene texto extraíble: {source}")

        # La fecha del informe manda: una observación regional es un punto de la
        # serie temporal del paciente, y fecharla mal desordena la evolución.
        timestamp = (
            report_date(text)
            or self.default_timestamp
            or datetime.fromtimestamp(source.stat().st_mtime, tz=UTC)
        )

        # Los informes de CBCT con IA no traen pH: traen anatomía radicular y hallazgos.
        # No compiten con el pH —son otro campo del mismo `ClinicalAttributes`— y un
        # informe puede traer los dos, uno o ninguno. Lo que cambia con el backend es
        # **quién los lee**. Antes salían siempre del regex, también con `backend="llm"`:
        # el modelo quedaba enchufado al campo tabulado (el pH, que el patrón ya cubre al
        # 98%) y ausente del que viene en prosa, que es donde el regex baja al 43%.
        # Medido en `scripts/eval_informes.py` sobre `report_corpus.py`.
        if self.backend == "rules":
            extraction = extract_ph_by_rules(text)
            findings = {
                code: (value, _RULES_CONFIDENCE)
                for code, value in extraction.findings.items()
            }
            clinicos = extract_hallazgos_by_rules(text)
            discards = extraction.discards
        elif self.backend == "llm":
            findings, clinicos, discards = extract_by_llm(text, model=self.model)
        else:
            findings, clinicos, discards = extract_by_local_llm(text, model=self.model)

        # Y lo que el contrato NO interpreta. Va aparte de `findings`/`clinicos` porque no
        # es por diente y porque no está validado contra ningún rango clínico nuestro: lo
        # único que lo acota es el intervalo que el propio informe imprime. Ver `Medida`.
        medidas = [
            Medida(**m, provenance=self._provenance(source, confidence=_RULES_CONFIDENCE))
            for m in extract_medidas_by_rules(text)
        ]

        observations = [
            RegionalObservation(
                region_id=code,
                attributes=ClinicalAttributes(
                    ph=findings.get(code, (None, 0.0))[0],
                    **clinicos.get(code, {}),
                ),
                timestamp=timestamp,
                provenance=self._provenance(
                    source, confidence=findings.get(code, (None, _RULES_CONFIDENCE))[1]
                ),
            )
            for code in sorted(set(findings) | set(clinicos))
        ]

        # Un informe del que no se extrae nada no es un éxito vacío: puede ser un
        # PDF escaneado sin OCR o un formato inesperado. Se declara con confianza
        # baja para que el gate de human-in-the-loop lo pare.
        agent_confidence = min(
            (c for _, c in findings.values()),
            default=_RULES_CONFIDENCE if (clinicos or medidas) else 0.0,
        )

        # Y un informe del que se extrae *parte* tampoco lo es. Si algo se
        # descartó, el informe decía algo que el twin no recoge: la confianza baja
        # por debajo del umbral del gate para que lo mire una persona y decida si
        # era un error de tecleo del clínico o un dato que hay que recuperar.
        motivos = (
            []
            if (findings or clinicos or medidas)
            else ["No se extrajo ningún hallazgo regional del informe."]
        )
        # Un índice fuera del rango que el propio informe declara es lo más accionable que
        # sale de aquí, y el agente **no lo interpreta**: lo repite. Decir «TORS 89,34 con
        # normal 90-100» no es un diagnóstico, es leer el documento en voz alta.
        anomalas = [m for m in medidas if m.fuera_de_rango]
        if anomalas:
            motivos.append(
                f"{len(anomalas)} de {len(medidas)} índice(s) fuera del rango que el "
                "propio informe declara: "
                + ", ".join(
                    f"{m.nombre} {m.valor:g}{m.unidad} (normal "
                    f"{m.normal_min:g}-{m.normal_max:g})" for m in anomalas
                )
            )
        if discards:
            agent_confidence = min(agent_confidence, _DISCARD_CONFIDENCE)
            motivos.append(_describe_discards(discards))

        return self._success(
            source,
            confidence=agent_confidence,
            regional=observations,
            medidas=medidas,
            detail=" ".join(motivos) if motivos else None,
        )
