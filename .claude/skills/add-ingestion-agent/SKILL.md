---
name: add-ingestion-agent
description: >-
  Crear un agente de ingesta nuevo del pipeline dental (mesh/cbct/report/image) o
  modificar uno existente. Úsala siempre que se añada una modalidad de entrada o
  se implemente un `*-agent` de ingesta: fija la interfaz común, Provenance,
  ModalityIngestion (fail-loud), el guardarraíl de reversibilidad del mesh, los
  tests (>80%) y la ficha en AGENTS.md. NO aplica a los agentes de análisis
  (segmentation/pathology) ni al research-agent.
---

# Añadir un agente de ingesta

Los agentes de ingesta traducen **un** fichero clínico crudo a fragmentos del
contrato (`packages/core-schemas`). Regla del proyecto: **1 modalidad = 1 soporte
= 1 agente**. Son **parsers deterministas** (no necesitan LLM). Este es el patrón
que TODOS deben cumplir.

## 0. Antes de escribir

- Lee el contrato: [`packages/core-schemas/src/core_schemas/models.py`](../../../packages/core-schemas/src/core_schemas/models.py).
- Lee la ficha `planned` del agente en [`AGENTS.md`](../../../AGENTS.md) y su fila en
  [`docs/architecture/multi-agent-pipeline.md`](../../../docs/architecture/multi-agent-pipeline.md) §2.
- Confirma la modalidad en el enum `Modality` (`cbct` · `mesh` · `report` · `image`).
  Si es una modalidad nueva, primero añádela al enum (es vocabulario controlado).

## 1. Estructura

- Módulo del agente en `apps/<nombre>-agent/` (o `packages/` si lo comparten varios).
- Punto de entrada: una función `ingest(source_file: Path) -> IngestionOutput`.
- Plantillas de partida en esta skill: [`templates/agent_template.py`](templates/agent_template.py)
  y [`templates/test_agent_template.py`](templates/test_agent_template.py). Cópialas y rellena.

## 2. Reglas duras (no negociables)

1. **Frontera raw → contrato.** El agente es el ÚNICO que toca el fichero crudo.
   A partir del contrato, nadie vuelve al original.
2. **Fail-loud, sin excepciones al llamador.** Un fichero corrupto/no parseable NO
   lanza: devuelve `ModalityIngestion(status=FAILED, detail=...)`. Un snapshot
   parcial debe **declararse** parcial, no llegar callado a exportación.
3. **Provenance obligatoria** en cada valor producido: `source_file`, `modality`,
   `agent="<nombre>-agent"`, `confidence`. Es requisito RGPD/HIPAA del proyecto.
4. **Campo gaussiano por hash**, nunca embebido: va a `3dgs-engine`, se referencia
   por `gaussian_field_ref` (ADR 001 §4.2, invariante de referencia colgante).
5. **`patient_id` siempre seudónimo**, nunca identificador directo.

## 3. Regla específica por modalidad

- **mesh-agent (STL/OBJ/PLY):** 🔒 **guardarraíl de reversibilidad (ADR 004).**
  PRESERVA la superficie de origen SIN pérdida (no solo la nube splatteada, que es
  lossy: ~1 mm de error medido, falla la métrica de <0,1 mm). Distingue STL «pelado»
  (`color_superficie=None`) de OBJ/PLY (color por vértice — ojo: en Teeth3DS+ el
  color es un gris placeholder, así que ahí también es `None`).
- **cbct-agent (DICOM):** produce el campo de densidad `σ≥0` (envuelve una
  reconstrucción tipo RGS; NO reimplementes su algoritmo). Reversibilidad = a
  imágenes/proyecciones, no a STL.
- **report-agent (PDF):** produce `RegionalObservation` (pH por FDI) con `timestamp`.
  El pH ya es lossless al guardarse; sin guardarraíl de superficie.

## 4. Definition of Done (criterios de aceptación)

- [ ] Implementa `ingest(source_file) -> IngestionOutput` con la interfaz común.
- [ ] Ingiere un caso real → fragmento de contrato válido (`extra="forbid"` pasa).
- [ ] **Caso corrupto → `status=FAILED`** sin excepción al llamador.
- [ ] `Provenance` completa; `ModalityIngestion` con el status correcto.
- [ ] (mesh) round-trip de superficie con error nanométrico; regla específica cumplida.
- [ ] Tests con caso OK + caso corrupto. **Cobertura >80%.**
- [ ] Fiabilidad >95% en el subset de validación (cuando aplique).
- [ ] Ficha en `AGENTS.md` actualizada de `planned` → `active` (rol, tools, permisos,
      inputs/outputs, reglas de delegación, historial de cambios — como `research-agent`).
- [ ] Pasa el `ai-code-reviewer` (Pydantic v2 estricto en core-schemas; sin imports
      cruzados entre apps/).

## 5. Qué NO hacer

- No metas lógica de LLM: la ingesta es determinista.
- No embebas el campo gaussiano en el modelo Pydantic (millones de puntos).
- No añadas campos «por si acaso» al contrato: solo si hay fuente/requisito real.
- No dejes la ficha en `AGENTS.md` desincronizada del código.
