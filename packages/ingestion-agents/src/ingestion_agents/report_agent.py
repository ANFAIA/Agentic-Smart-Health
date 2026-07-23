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

**Por qué `rules` es el defecto.** No sobre-agentificar: el LLM entra donde la
entrada es ambigua, no por costumbre. Además el backend determinista da el
suelo medible contra el que comparar al LLM (fiabilidad >95%).

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

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from core_schemas import (
    ClinicalAttributes,
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
# Confianza que se asigna a una extracción por regex: alta pero no 1.0 — el patrón
# acertó, pero nadie ha verificado que el informe dijera lo que parece decir.
_RULES_CONFIDENCE = 0.9
# Confianza cuando se descartó algún hallazgo. Está POR DEBAJO del umbral de
# human-in-the-loop del orquestador (0.7) a propósito: un descarte significa que
# el informe decía algo que el twin no recoge, y eso lo decide una persona.
_DISCARD_CONFIDENCE = 0.6
# Cuántos descartes se detallan en el `detail` antes de resumir el resto.
_MAX_DISCARDS_IN_DETAIL = 5


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
_EXTRACTION_TOOL = {
    "name": "registrar_hallazgos",
    "description": (
        "Registra los valores de pH medidos por diente que aparecen en el informe. "
        "Incluye únicamente dientes con un pH explícito en el texto."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hallazgos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fdi": {
                            "type": "string",
                            "description": "Código ISO-FDI de dos dígitos, p. ej. '16'.",
                        },
                        "ph": {"type": "number", "minimum": 3.0, "maximum": 9.0},
                        "confianza": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["fdi", "ph", "confianza"],
                },
            }
        },
        "required": ["hallazgos"],
    },
}

_SYSTEM_PROMPT = (
    "Eres un extractor de datos de informes odontológicos. Devuelves únicamente lo "
    "que el informe afirma explícitamente. No infieres, no completas dientes que no "
    "aparecen y no conviertes notaciones dudosas: si un código dental es ambiguo, "
    "baja la confianza en vez de adivinar. Numeración ISO-FDI de dos dígitos."
)


class LLMExtraction(NamedTuple):
    """Resultado del backend LLM: `{FDI: (pH, confianza)}` **y lo que se cayó**."""

    findings: dict[str, tuple[float, float]]
    discards: list[Discard]


def extract_ph_by_llm(text: str, *, model: str = _LLM_MODEL) -> LLMExtraction:
    """Extrae con Claude (salida forzada por esquema) y registra los descartes.

    El esquema de la tool es la barrera: el modelo no puede devolver prosa ni un
    campo inventado, solo instancias del esquema. Los valores siguen pasando por
    la validación de la ontología igual que en el backend determinista — la
    confianza del modelo no exime de validar. Y, como allí, lo que la validación
    tumba se **registra**: que lo haya propuesto un LLM no es motivo para perderlo
    en silencio; al contrario, es cuando más interesa saberlo.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "El backend `llm` requiere el extra `llm` de `ingestion-agents` (anthropic)."
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Falta ANTHROPIC_API_KEY para el backend `llm` del report-agent.")

    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        tools=[_EXTRACTION_TOOL],  # type: ignore[list-item]
        tool_choice={"type": "tool", "name": "registrar_hallazgos"},
        messages=[{"role": "user", "content": text}],
    )

    out: dict[str, tuple[float, float]] = {}
    discards: list[Discard] = []
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        for item in block.input.get("hallazgos", []):  # type: ignore[union-attr]
            code = str(item.get("fdi", "")).replace(".", "")
            value = float(item.get("ph"))
            propuesta = f"fdi={code} pH={value:g}"
            if not ontology.is_valid_fdi(code):
                discards.append(Discard(propuesta, "código FDI inexistente (propuesto por el LLM)"))
            elif not ontology.PH.accepts(value):
                discards.append(
                    Discard(
                        propuesta,
                        f"pH fuera del rango plausible "
                        f"({ontology.PH.minimum:g}–{ontology.PH.maximum:g})",
                    )
                )
            else:
                out[code] = (value, float(item.get("confianza", 0.5)))
    return LLMExtraction(out, discards)


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
        model: str = _LLM_MODEL,
        default_timestamp: datetime | None = None,
        quarantine_dir: str | Path | None = None,
    ) -> None:
        super().__init__(quarantine_dir=quarantine_dir)
        if backend not in {"rules", "llm"}:
            raise ValueError(f"Backend desconocido: {backend!r} (usa 'rules' o 'llm').")
        self.backend = backend
        self.model = model
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

        if self.backend == "rules":
            extraction = extract_ph_by_rules(text)
            findings = {
                code: (value, _RULES_CONFIDENCE)
                for code, value in extraction.findings.items()
            }
            discards = extraction.discards
        else:
            findings, discards = extract_ph_by_llm(text, model=self.model)

        observations = [
            RegionalObservation(
                region_id=code,
                attributes=ClinicalAttributes(ph=value),
                timestamp=timestamp,
                provenance=self._provenance(source, confidence=confidence),
            )
            for code, (value, confidence) in sorted(findings.items())
        ]

        # Un informe del que no se extrae nada no es un éxito vacío: puede ser un
        # PDF escaneado sin OCR o un formato inesperado. Se declara con confianza
        # baja para que el gate de human-in-the-loop lo pare.
        agent_confidence = min((c for _, c in findings.values()), default=0.0)

        # Y un informe del que se extrae *parte* tampoco lo es. Si algo se
        # descartó, el informe decía algo que el twin no recoge: la confianza baja
        # por debajo del umbral del gate para que lo mire una persona y decida si
        # era un error de tecleo del clínico o un dato que hay que recuperar.
        motivos = [] if findings else ["No se extrajo ningún hallazgo regional del informe."]
        if discards:
            agent_confidence = min(agent_confidence, _DISCARD_CONFIDENCE)
            motivos.append(_describe_discards(discards))

        return self._success(
            source,
            confidence=agent_confidence,
            regional=observations,
            detail=" ".join(motivos) if motivos else None,
        )
