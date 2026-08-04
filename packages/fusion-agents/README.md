# `fusion-agents` — fusión del Digital Twin dental

Implementa la capa de fusión que decide el
[ADR 004](../../docs/architecture/004-fusion.md). El pipeline la parte en **dos
etapas separadas por la segmentación**:

```
[FUSIÓN GEOMÉTRICA]  registro malla↔CBCT (banda ε) · color · NO usa FDI
   ▼
[SEGMENTACIÓN]       puebla region_id (FDI)
   ▼
[FUSIÓN SEMÁNTICA]   ancla pH/observaciones al FDI · NO usa geometría
```

| Agente | Etapa | Estado |
|---|---|---|
| `SemanticFusionAgent` | anclaje al FDI | **implementado** |
| `GeometricFusionAgent` | registro malla↔CBCT | **implementado** (etapa fina) |

Son **dos agentes y no uno** porque entre ambos corre otra etapa, y porque su
material y su criterio de aceptación son distintos.

## Uso

```python
from fusion_agents import GeometricFusionAgent, SemanticFusionAgent, insert_snapshot

# 1 · registro geométrico (antes de la segmentación)
geo = GeometricFusionAgent(epsilon_mm=0.5)
out = geo.fuse(snapshot, source=puntos_malla, target=puntos_cbct)
out.snapshot.provenance.transform.inverse()      # el registro es reversible

# 2 · anclaje semántico (después)

agente = SemanticFusionAgent()                      # umbral HITL 0.7 por defecto
out = agente.fuse(snapshot, detected={"46": 0.95, "47": 0.91})

if out.hitl_required:
    print(out.hitl_reasons)                         # qué mirar y por qué
twin = insert_snapshot(twin, out.snapshot)          # idempotente por acquisition_id
```

`detected` es el mapa `FDI → confianza` que produce el `segmentation-agent`. Se pasa
explícitamente en vez de leer el campo gaussiano del almacén: son ~14 códigos, y
cargar millones de primitivas para obtenerlos sería absurdo. De paso, el agente
queda testeable sin almacén y sin GPU.

## El registro

El algoritmo vive **detrás de un `Protocol`** (`Registrar`), no dentro del agente:
así se sustituye sin tocar el contrato y el agente se testea con un registrador
trivial. Por defecto es **ICP multiescala** en numpy + scipy.

**Lo que falta, dicho claro.** La etapa **gruesa** del ADR —RANSAC sobre
descriptores FPFH— no está: es la que da una pose inicial cuando las nubes están muy
desalineadas, y depende de Open3D. Consecuencia: **el ICP de aquí converge solo si la
pose inicial ya está razonablemente cerca**. Con una malla derivada del propio volumen
eso se cumple por construcción; con un intraoral y un CBCT capturados por separado,
hay que medirlo antes de dar el registro por bueno.

**La confianza sale del residuo**, `clamp(1 − rms/ε, 0, 1)` con ε = 0.5 mm. El gate
por defecto (0.7) equivale a exigir `rms ≤ 0.3·ε`. Un registro fuera de banda **no
falla**: entrega con confianza baja y pide revisión. Es la diferencia entre un fallo
declarado y uno silencioso.

**Lo que este agente NO hace: transferir el color.** El pipeline se lo atribuye, pero
su §6 declara que *«de dónde sale el color está sin asentar»*. Es una decisión
abierta, no una pieza pendiente de teclear.

## Las tres decisiones del anclaje semántico

**La confianza es el eslabón más débil**, `min(...)` y no el producto. Encadenar
productos hunde todo bajo el umbral por aritmética, no por desconfianza real.
Anclar un pH a un diente no puede ser más fiable que saber qué diente es.

**Ante un conflicto, el agente no elige ganador.** Si el informe referencia un
diente que la segmentación no encontró, se conserva el FDI **del informe** —es la
fuente clínica—, se pone la confianza a **0.0** y con eso cae al gate y va a
revisión humana. No hay campo nuevo: *la confianza es la marca*.

El motivo está medido: el error dominante del modelo es el
[desplazamiento al diente vecino](../../notebooks/exercise-point-transformer-teeth3ds.md),
así que en un desacuerdo es la parte menos fiable — pero el informe tampoco es
infalible. Resolverlo en silencio sería el fallo que el
[ADR 003](../../docs/architecture/003-verification-fault-tolerance.md) señala como el
peor: silencioso e irreversible, sobre un dato clínico.

**Nunca muta el snapshot de entrada** y conserva el `acquisition_id`, que es la
identidad de visita. Por eso reejecutar la fusión **reemplaza** en vez de añadir, y
no infla el historial del paciente con visitas que no ocurrieron.

## Trazabilidad

La `Provenance` de cada observación **conserva la del informe** (`source_file`,
`modality`, `agent`): el valor sigue viniendo del PDF, y perderlo rompería la
trazabilidad hasta el original. Solo se reescribe `confidence`.

Quién hizo la fusión consta en la `Provenance` **del snapshot**, que es el valor que
este agente sí deriva.

## Tests

```bash
uv run pytest packages/fusion-agents
```

44 tests, cobertura 96%. Del registro cubren el **nivel unitario del ADR §2.7**:
se aplica a una nube una transformación **conocida** y se exige recuperarla — la
verdad de referencia es exacta porque se fabrica en el test, así que un fallo es del
algoritmo y no del dato. Del anclaje semántico: la regla del eslabón más débil, el
conflicto FDI, el gate configurable, la no-mutación, la idempotencia de la serie
temporal y el contrato *fail-loud* — incluido que **la cuarentena no filtre dato
clínico**: guarda el `acquisition_id` y el traceback, nunca el pH.
