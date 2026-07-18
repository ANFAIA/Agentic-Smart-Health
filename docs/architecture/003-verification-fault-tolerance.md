# ADR 003 — Verificación y tolerancia a fallos

| | |
|---|---|
| **Estado** | **Borrador especulativo** (explora, no cierra; sin validar contra el pipeline real) |
| **Fecha** | 2026-07-17 |
| **Decisor** | — (ninguno; no es una decisión, es exploración) |
| **Ámbito** | Transversal · cómo el sistema *podría* detectar, acotar y sobrevivir a sus propios errores |
| **Relacionado** | Contrato: [ADR 001](001-digital-twin-core-schemas.md) · Pipeline: [multi-agent-pipeline](multi-agent-pipeline.md) · Registro de agentes: [`AGENTS.md`](../../AGENTS.md) |

> **⚠ Borrador especulativo — léelo como exploración, no como diseño acordado.**
> Este documento razona sobre **fallos hipotéticos** de un pipeline de fusión que
> **aún no está asentado** (fuente de color, registro malla↔CBCT y el propio orden de
> fases siguen *abiertos* — ver pipeline §6). Nace de una cadena de auditorías cuyas
> premisas no se han verificado contra el dato real. **No implementa nada, no cierra
> ninguna decisión abierta, y no debe enlazarse como "la solución".** Ya se **podó** lo
> que dependía de la fusión geométrica (verificador de QA de registro, provenance del
> canal denso, versión de etiquetado, consenso multi-verificador); queda solo lo que
> **sobrevive independientemente de esa premisa** —verificador `fdi-consistency-agent`,
> event sourcing y HIL con timeout—. Se conserva como material para discutir,
> no como parte del diseño vigente.

---

## 1. Contexto y problema

Una auditoría de arquitectura sobre el ADR 001 y el pipeline reveló varias debilidades
que, en apariencia, eran independientes. No lo son: **casi todas se reducen al mismo
patrón — un fallo que es a la vez *silencioso* e *irreversible*.** (Se listan solo las
que sobreviven sin depender de la fusión geométrica —hoy sin asentar—; los temas del
canal denso y de `series()` retroactivo se retiraron por premisa-dependientes.)

| Hallazgo | Dónde se nombró | Síntoma |
|---|---|---|
| Ancla FDI sin validación cruzada | pipeline D4 · ADR 001 §4.6 | swap OCR "36/46" cuelga pH del diente equivocado, sin señal |
| Inmutabilidad vs. «enriquecimiento» | pipeline D5 · ADR 001 §6 | «enriquecer» un snapshot inmutable no está definido |
| Human-in-the-loop ambiguo | pipeline §4 (nota) | «revisión si afecta a diagnóstico» no es accionable; sin timeout ni historial de corrección |
| Concurrencia sobre el snapshot | (diagrama pipeline, ya linealizado) | dos agentes «enriquecen» el mismo snapshot sin estrategia de escritura |

**Principio rector de este ADR:**

> **Nunca pongas a un agente en un camino donde su fallo sea *silencioso* Y
> *irreversible* a la vez.** Basta con romper *uno* de los dos: hacerlo ruidoso
> (verificación, gates) o hacerlo reversible (eventos, no sobrescritura).

---

## 2. Decisiones propuestas

### 2.1 Verificadores independientes (rompe el silencio)

Ningún agente valida su propia salida. Un dato clínicamente sensible lo confirma un
**verificador distinto**, por un **método ortogonal** al que lo produjo.

Esto introduce una **tercera clase de agente**, además de ingesta y análisis —y
encaja con la distinción pipeline-determinista vs. agente-con-razonamiento
([`AGENTS.md`](../../AGENTS.md)): el verificador es razonamiento, no un parser. Los
roles propuestos (fichas `planned` a registrar en `AGENTS.md`):

| Verificador | Verifica | Método **ortogonal** al productor |
|---|---|---|
| `fdi-consistency-agent` | el ancla `region_id` (D4) | cruza FDI del **texto** (informe PDF) contra FDI de la **geometría** (segmentación); marca discrepancia si no coinciden |

> Un solo verificador concreto porque es el único **independiente de la fusión
> geométrica**. Si esa rama vuelve al camino crítico aparecerían más (p. ej. uno de QA
> de registro), pero eso queda fuera mientras la fusión siga sin asentar.

> **Regla dura:** un verificador **no** puede usar el mismo método que el productor
> —heredaría su punto ciego. Por eso `fdi-consistency-agent` **no** vuelve a
> segmentar: cuenta regiones y las cruza con «32 dientes ± ausentes declarados» y
> con los FDI del texto.

- Ejemplo canónico (ataca D4): el `segmentation-agent` cuenta *N* dientes con FDI;
  un verificador contrasta ese conjunto con los dientes **mencionados en el informe**
  (`report-agent`). Si no coinciden → `disputed`, no se propaga en silencio.
- El resultado de verificar se adjunta al dato, no lo reemplaza:

```python
# ESBOZO ilustrativo — no es código a mergear
class VerificationResult(BaseModel):
    verifier_agent: str
    target: str                              # qué se verificó (p. ej. "region_id=16")
    status: Literal["confirmed", "disputed", "unverifiable"]
    confidence: float                        # 0..1
    evidence: str                            # por qué (traza auditable)
    timestamp: datetime

class RegionalObservation(BaseModel):
    ...
    verifications: list[VerificationResult] = []
```

### 2.2 Enriquecimiento como eventos *append-only* (rompe la irreversibilidad)

Aborda **D5** (inmutabilidad vs. enriquecer). La base del snapshot sigue siendo
**inmutable** (reversibilidad del ADR 001 §4.3); los análisis **no mutan** el
snapshot ni fabrican visitas nuevas: **añaden eventos**.

```python
# ESBOZO ilustrativo
class EnrichmentEvent(BaseModel):
    event_id: str
    snapshot_id: str                         # a qué snapshot enriquece
    author_agent: str
    kind: Literal["segmentation", "pathology", "correction", ...]
    payload: dict                            # el enriquecimiento (con su Provenance)
    supersedes: str | None = None            # event_id que corrige, si aplica
    created_at: datetime
```

- «Enriquecer» = *append* de un `EnrichmentEvent`. El estado vigente se **proyecta**
  reduciendo los eventos sobre la base inmutable.
- Una **corrección** es un evento nuevo con `supersedes`, no una sobrescritura → el
  historial se conserva (requisito RGPD/HIPAA) y la concurrencia (§1, «dos agentes
  escriben») deja de pisar: son dos *appends*, no dos writes al mismo campo.

> **Matiz sobre el orden segmentación/fusión.** El event sourcing hace flexible
> *cuándo* corre la segmentación (es un evento, no una fase rígida), pero **no
> disuelve la dependencia de datos**: la fusión **semántica** sigue necesitando el
> `region_id` *antes* (el pH se cuelga del FDI). Esa dependencia —segmentación antes
> que fusión semántica— es lo **único firme**; el resto del orden de fases (si la
> fusión geométrica precede o siquiera está en el camino crítico) queda **provisional**
> (pipeline §6). Así que event sourcing relaja el *cómo* se adjunta, sin comprar el
> «ya no importa el orden» ni fijar más orden del que el pipeline deja abierto.

### 2.3 Puertas de calidad (gates) en los pasos con juicio

Cada paso que pueda salir mal de forma plausible pero incorrecta emite un veredicto
explícito, no un «ok» tácito:

```
proceed              → sigue el pipeline
proceed_with_flag    → sigue, pero marca el dato para revisión posterior
halt_for_review      → detiene y escala a human-in-the-loop
```

- Ejemplo vigente (sin depender de la fusión): si `fdi-consistency-agent` (§2.1)
  devuelve `disputed` sobre un `region_id`, el gate no deja pasar ese dato como bueno
  → `halt_for_review` y escala a HIL (§2.4), en vez de propagar el FDI equivocado en
  silencio. El **mecanismo** (proceed/flag/halt) es lo que fija este ADR; los
  **umbrales** son política.
- *(Si la fusión geométrica volviera al camino crítico, tendría su propio gate —
  fitness del ICP, cobertura de la banda ε—, pero eso queda fuera mientras esa rama
  siga sin asentar.)*

### 2.4 Human-in-the-loop con timeout y corrección trazable

Aborda el HIL ambiguo (§1). La petición de revisión es un objeto con ciclo de vida,
no un criterio subjetivo:

```python
# ESBOZO ilustrativo
class ReviewRequest(BaseModel):
    request_id: str
    target_event: str                        # qué enriquecimiento se revisa
    status: Literal["pending", "approved", "rejected", "expired_unreviewed"]
    created_at: datetime
    expires_at: datetime
    escalation_path: str                     # a quién se escala si expira
```

- **Timeout explícito:** si nadie revisa antes de `expires_at` → `expired_unreviewed`
  (estado nombrado, no bloqueo indefinido ni descarte silencioso; el dato queda
  marcado y sin exportar como clínico).
- **La corrección humana es un `EnrichmentEvent` de tipo `correction`** (§2.2): se
  persiste con su Provenance, no sobrescribe. Así se conserva el historial de
  corrección — justo el dato que un sistema con trazabilidad debe guardar.
- **El disparo deja de ser subjetivo.** «¿Afecta a diagnóstico?» era un criterio
  sin dueño; aquí salta **automáticamente** cuando una verificación (§2.1) da
  `status="disputed"` **o** `confidence` cae bajo umbral. Ningún dato `disputed`
  llega a export sin marcar. (El umbral es política a fijar con criterio clínico;
  el *mecanismo* deja de depender de un juicio implícito.)

---

## 3. Alternativas consideradas

| Alternativa | Idea | Veredicto |
|---|---|---|
| **A · Confiar en «mejor modelo»** | subir la calidad de ICP/segmentación y asumir que acierta | ❌ No rompe el silencio: un fallo mejor-condicionado sigue siendo silencioso e irreversible |
| **B · Auto-verificación** | que cada agente valide su propia salida | ❌ El productor comparte los sesgos del error; no es verificación real |
| **C · Snapshots mutables** | dejar que los análisis reescriban el snapshot | ❌ Rompe la reversibilidad, argumento central del ADR 001 §4.3 |
| **D · Un snapshot nuevo por enriquecimiento** | cada análisis crea una «visita» | ❌ Rompe «1 snapshot = 1 visita»; infla la serie temporal con visitas ficticias |
| **E · Eventos append-only + verificadores + gates (propuesto)** | separar base inmutable de enriquecimientos; verificación ortogonal; gates explícitos | ✅ Rompe *silencio* (verificación/gates) **y** *irreversibilidad* (eventos), sin sacrificar reversibilidad |

---

## 4. Consecuencias

*(Condicionales — este borrador no está adoptado.)*

**Positivas**
- Cada hallazgo dejaría de ser «silencioso e irreversible» a la vez.
- El ancla FDI pasaría de *confiada* a *verificada* (swap OCR "36/46" detectable).
- Historial de corrección conservado (RGPD/HIPAA) sin sacrificar la inmutabilidad.

**Negativas / costes**
- Más superficie de contrato (`VerificationResult`, `EnrichmentEvent`, `ReviewRequest`)
  y una **proyección** de eventos que hoy no existe.
- El **verificador ortogonal** (`fdi-consistency-agent`) y los **umbrales de los
  gates** son trabajo real, aunque no dependen del spike de registro (D2).
- Introduce complejidad operativa (cola de revisión, timeouts, escalado) que solo se
  justifica en los pasos con **juicio clínico**, no en el pipeline determinista de
  ingesta (ver la distinción pipeline-vs-agente).

---

## 5. Qué NO cambia respecto al ADR 001

- `TwinSnapshot` por visita, reversibilidad y seudonimización siguen siendo la base.
- `Provenance` por observación y la validación *fail-loud* (`extra="forbid"`, rangos)
  se mantienen; este ADR **añade** capas, no las sustituye.
- El ancla FDI sigue siendo el pegamento semántico — solo que ahora **verificado**,
  no confiado.

---

## 6. Referencias

- [ADR 001 — Contrato de datos del Digital Twin](001-digital-twin-core-schemas.md)
  (decisión abierta D5; los temas del canal denso y `series()` se retiraron de §4.5/§6
  y aquí solo se exploran).
- [Arquitectura del pipeline multiagente](multi-agent-pipeline.md) (§7 · D4–D5).
- [`AGENTS.md`](../../AGENTS.md) — principios de human-in-the-loop y responsabilidad única.
