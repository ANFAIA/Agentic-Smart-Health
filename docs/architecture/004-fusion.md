# ADR 004 — Fusión: registro geométrico y anclaje semántico

| | |
|---|---|
| **Estado** | **Aceptado** |
| **Fecha** | 2026-08-04 |
| **Decisor** | Equipo de desarrollo — Agentic Smart Health |
| **Ámbito** | Semana 5 · La capa de fusión del pipeline: qué agentes son, qué deciden y qué NO deciden |
| **Relacionado** | Contrato: [ADR 001](001-digital-twin-core-schemas.md) · Pipeline: [multi-agent-pipeline](multi-agent-pipeline.md) · Fallos: [ADR 003](003-verification-fault-tolerance.md) · Evidencia medida: [`exercise-point-transformer-teeth3ds`](../../notebooks/exercise-point-transformer-teeth3ds.md) |

---

## 1. Contexto y problema

El [pipeline](multi-agent-pipeline.md) sitúa la fusión en **dos etapas separadas por la
segmentación**:

```
[FUSIÓN GEOMÉTRICA]  registro malla↔CBCT (banda ε) · asigna color · NO usa FDI
   ▼
[SEGMENTACIÓN]       puebla region_id (FDI) sobre las gaussianas
   ▼
[FUSIÓN SEMÁNTICA]   ancla pH/observaciones al FDI · NO usa geometría
```

La issue #32 las agrupa bajo un único `fusion-agent`, pero **no comparten nada**: ni
entrada, ni dependencias, ni criterio de validación. La geométrica exige CBCT e intraoral
**del mismo paciente** —hay un lote de pares disponible, que pasa a ser el material de
referencia para validarla—; la semántica solo necesita el FDI que produce el
`segmentation-agent`.

Además hay tres huecos que la issue da por resueltos y no lo están: `Provenance` **no
tiene dónde guardar la transformación** que la reversibilidad exige; no está definido
**de qué se calcula la confianza** que dispara el gate HITL a 0.7; y no está decidido
**qué hacer cuando el informe y la segmentación discrepan** sobre qué diente es cual —
un caso que, según el experimento del Point Transformer, es el **más frecuente**, no un
borde raro.

---

## 2. Decisiones

### 2.1 Dos agentes, no uno

| agente | entrada | qué hace | qué NO hace |
|---|---|---|---|
| `geometric-fusion-agent` | `TwinSnapshot` con `mesh` + `cbct` | registra malla↔CBCT, asigna `color_superficie` | no lee ni escribe `region_id` |
| `semantic-fusion-agent` | `TwinSnapshot` con `region_id` poblado | ancla `RegionalObservation` al FDI | no toca geometría ni transforma nada |

**Por qué separados.** La regla del repo es *1 responsabilidad = 1 agente*, y aquí la
separación no es estética: entre ambos se ejecuta la segmentación, así que un agente
único tendría que invocarse dos veces con banderas y guardar estado entre llamadas.
Separarlos también los desacopla en el tiempo: la semántica se valida con Teeth3DS+ y la
geométrica con el lote de pares, que son materiales y criterios de aceptación distintos.

Ambos heredan el contrato de los agentes de ingesta: **no lanzan excepciones**, devuelven
estado + confianza, y mandan a cuarentena lo que no pueden procesar.

### 2.2 `Provenance` gana la transformación (cambio aditivo en `core-schemas`)

La reversibilidad exige poder deshacer lo que la fusión aplicó. `Provenance` hoy tiene
`source_file`, `modality`, `agent`, `confidence`, `ingested_at` — **nada donde guardar
una transformación**.

Se añade **un campo opcional**, con su propio modelo:

```python
class RigidTransform(BaseModel):
    rotation: tuple[float, float, float, float]   # cuaternión (w, x, y, z), normalizado
    translation: tuple[float, float, float]       # mm
    rms_mm: float | None                          # residuo del registro

class Provenance(BaseModel):
    ...
    transform: RigidTransform | None = None
```

- **Rígida, no una matriz 4×4 general.** Una 4×4 puede codificar **escala y cizalla**;
  el registro alinea dos medidas del **mismo objeto físico**, ambas en milímetros reales,
  así que no hay ninguna que representar. Si un ICP devolviera una escala espuria, la 4×4
  la guardaría sin protestar y la reversibilidad se rompería **en silencio**. Esta forma
  la hace *imposible de expresar*, y un validador rechaza el cuaternión no unitario —
  que es como se colaría una escala encubierta.
- **Invertirla es exacto**: conjugado del cuaternión y `-R⁻¹·t`, sin el mal
  condicionamiento de invertir una 4×4 cualquiera. Importa porque la reversibilidad
  exigida es < 0.1 mm, y el `.inverse()` del modelo la hace auditable.
- **Cuaternión `(w, x, y, z)`**: misma convención que `GaussianPrimitive.rotation`, ya
  establecida en el ADR 001.
- **Opcional y aditivo** → los agentes de ingesta lo dejan a `None` y nada se rompe.
  `core-schemas` ya testea esta forma de evolución (`test_models.py`). Contrato en
  **1.3.0**.
- **Va en `Provenance` y no en un modelo colgado aparte** porque la transformación tiene
  que viajar **con el valor que transformó**: si vive en otra tabla, invertirla exige un
  join, y en un fallo es justo lo que no quieres tener que reconstruir.

### 2.3 La confianza se deriva, y cada rama tiene su fórmula

El gate HITL está en **0.7**. De qué sale ese número:

**Geométrica** — del residuo del registro contra la banda de tolerancia:

```
confidence = clamp(1 − rms_mm / ε, 0, 1)
```

Con `rms = 0` da 1.0; con `rms = ε` da 0.0. El gate en 0.7 equivale a exigir
`rms ≤ 0.3·ε`.

**Semántica** — el **eslabón más débil** de la cadena:

```
confidence = min(confianza_observación, confianza_FDI)
```

Se usa `min` y no el producto: encadenando pasos, el producto decae tan rápido que todo
acabaría bajo 0.7 por acumulación aritmética y no por desconfianza real. Anclar un pH a
un diente no puede ser más fiable que saber qué diente es.

### 2.4 Ante un conflicto FDI, la fusión NO decide: marca

Si el informe dice «caries en el 46» y la segmentación dice que ahí hay un 45, el
`semantic-fusion-agent`:

1. **Ancla la observación al FDI del informe** — es la fuente clínica.
2. **Fija `confidence = 0.0`** en la `Provenance` de esa observación.
3. Con eso **cae por debajo del gate y va a HITL**. No hace falta ningún campo nuevo:
   *la propia confianza es la marca*.

**Por qué no se resuelve automáticamente.** El experimento del Point Transformer midió
que el error dominante del modelo es exactamente el **desplazamiento al diente vecino**
(62-83% de los fallos), agravado en arcadas con piezas ausentes. Es decir: en un
desacuerdo de este tipo, el modelo es la parte **menos** fiable — pero el informe
tampoco es infalible. Elegir un ganador en silencio produciría el patrón que el
[ADR 003](003-verification-fault-tolerance.md) señala como el peor: un fallo **silencioso
e irreversible** sobre un dato clínico.

Detectar y escalar es barato. Equivocarse callando, no.

### 2.5 La fusión emite snapshot nuevo; la identidad es `acquisition_id`

- La fusión **nunca muta** el snapshot de entrada: emite uno nuevo con los valores
  derivados y su `Provenance`.
- La identidad de visita es **`acquisition_id`**.
- Al insertar en `PatientDigitalTwin`: si el `acquisition_id` ya está, **se reemplaza**;
  si no, se **añade preservando el orden temporal**.

Así, reejecutar la fusión sobre el mismo material es **idempotente** — cumple el «1
snapshot = 1 visita» de la issue #33 sin inflar la serie con visitas ficticias, y deja
la cadena de derivaciones auditable.

### 2.6 Registro: RANSAC-FPFH + ICP multiescala, con ε = 0.5 mm

- **Grueso**: RANSAC sobre descriptores **FPFH** (descriptor local, invariante a la pose).
- **Fino**: **ICP multiescala**.
- Es la receta de **DDMF** ([arXiv:2203.05784](https://arxiv.org/abs/2203.05784)), validada
  sobre **503 pacientes CBCT+IOS emparejados**, con 0.17 mm de ASSD.
- **Banda ε = 0.5 mm** como tolerancia de aceptación.

> **⚠ ε NO es la métrica de 0.1 mm del brief.** Son cantidades distintas y confundirlas
> sería un error de bulto:
>
> - **< 0.1 mm (brief)** = error de **reconstrucción de malla**: exportar el STL desde el
>   twin y compararlo con el que entró. Mide **reversibilidad** de una sola modalidad.
> - **ε = 0.5 mm (aquí)** = error de **alineamiento entre dos modalidades distintas**,
>   limitado por la resolución del CBCT (vóxel de 0,15–0,4 mm) y sus artefactos.
>
> El estado del arte publicado (0.17 mm) está por encima del 0.1 mm del brief, lo que
> confirma que no pueden ser la misma métrica.

### 2.7 El registro se valida en dos niveles

Separar **corrección del algoritmo** de **calidad sobre dato clínico**, porque fallan por
motivos distintos y confundirlos deja agujeros:

| nivel | material | criterio |
|---|---|---|
| **unitario** | una malla y **una copia suya** a la que se aplica una transformación **conocida** | el algoritmo debe **recuperar esa transformación** dentro de ε |
| **integración** | el lote de pares CBCT+IOS reales | ASSD sobre puntos de control, contra la banda ε |

El nivel unitario es el que hace testeable el agente **sin datos clínicos y sin GPU**: la
verdad de referencia es exacta porque la fabricas tú, así que un fallo ahí es del
algoritmo, no del dato. El de integración es el que dice si el registro sirve en la
realidad, con la resolución y los artefactos del CBCT de por medio.

Un agente que solo pasa el segundo no es auditable; uno que solo pasa el primero no está
validado.

### 2.8 El color sale de la malla; la foto es otro problema

**Fuente canónica: el color por vértice de la malla intraoral.** No hay que registrarlo
contra nada — *está* en la malla. Lo único que hace falta es lo que la fusión geométrica
ya resuelve: llevar malla y campo al mismo sistema. Después, cada gaussiana **dentro de
la banda ε** toma el color de su vértice más cercano. Es literalmente lo que el
[ADR 001](001-digital-twin-core-schemas.md) describe: *«None si la gaussiana no cae en la
banda ε de la superficie»*.

**La ausencia es una respuesta válida.** Si la malla viene pelada (STL) o su color es un
*placeholder* —el gris plano y uniforme de Teeth3DS+—, `color_superficie = None`.
Declarado, no inventado. Hay precedente: el `mesh-agent` ya trata ese gris como
**ausencia**.

**El color desde fotos queda FUERA de este agente.** La razón está medida, no es una
preferencia: el notebook 07 proyectó la foto oclusal sobre las coronas y el **ICP no baja
de IoU ≈ 0,55**, porque el error residual es **no-rígido** —perspectiva de una foto
intraoral sin calibrar, con retractor— y una transformación rígida no lo corrige.

Es decir: **foto↔malla y CBCT↔malla no son el mismo problema.** Uno es rígido y el otro
no. Meterlos en el mismo agente sería juntar dos cosas distintas bajo un nombre. La Vía B
multi-vista del notebook 07 será el primer caso de quien lo aborde, con su propia decisión.

> **Consecuencia práctica que conviene declarar.** Con los datos de hoy el resultado será
> **`None` casi siempre**: Teeth3DS+ es gris plano y Bite2Text es STL pelado. La
> transferencia existirá y estará testeada, pero solo se ejercitará de verdad cuando entre
> un escáner que exporte color real por vértice. Ese `None` **no es un bug**.

> **Dónde se persiste, que la issue #32 daba por hecho.** «Poblar `color_superficie` en el
> snapshot» no es posible tal cual: ese campo vive en `GaussianPrimitive`, y el
> `TwinSnapshot` solo guarda una **referencia por hash** al campo gaussiano. El agente
> **calcula** los colores y quién sea dueño del `ArtifactStore` los materializa. Mantener
> esa frontera es lo que evita que un agente de fusión acabe reescribiendo blobs pesados.

---

## 3. Alternativas consideradas

| Alternativa | Por qué se descarta |
|---|---|
| Un solo `fusion-agent` con dos modos | Obliga a estado entre invocaciones y acopla dos ramas con viabilidad distinta |
| Transformación en un modelo `Registration` aparte | Semánticamente más limpio, pero invertir un valor exigiría un join justo cuando algo ha fallado |
| Confianza como producto de las etapas | Decae por aritmética, no por desconfianza: saturaría el gate de falsos positivos |
| Resolver el conflicto FDI por confianza | Es exactamente el fallo silencioso que el ADR 003 quiere evitar, sobre un dato clínico |
| Mutar el snapshot en sitio | Rompe reversibilidad y event sourcing |
| Validar el registro solo con mallas derivadas del propio volumen | Quedan alineadas por construcción: no hay nada que registrar, así que no validan el algoritmo |

---

## 4. Consecuencias

**A favor**

- Las dos ramas se pueden construir y testear en paralelo, cada una con su material y su
  criterio de aceptación.
- La reversibilidad deja de ser una intención: la transformación viaja con el dato.
- El desacuerdo FDI se convierte en el **primer caso de uso real de HITL**, con criterio
  medido detrás en vez de intuición.
- Idempotencia por `acquisition_id`: la fusión se puede reejecutar sin ensuciar la serie.

**En contra**

- **Toca `core-schemas`**, que es el contrato compartido. Aditivo y opcional, pero obliga
  a versionar y a repasar los tests del paquete.
- La rama geométrica queda **atada a un lote de pares concreto**: hasta validarla contra
  más de una fuente, ε y las cifras de calidad valen para ese material, no en general.
- La regla de conflicto **generará trabajo humano** proporcional al error del
  `segmentation-agent`. Es el precio consciente de no corromper datos en silencio.

---

## 5. Qué NO cambia respecto al ADR 001

- Los tres soportes geométricos, el campo gaussiano referenciado por hash y el enfoque
  snapshot-céntrico.
- `region_id` (FDI) sigue siendo el ancla semántica: la fusión semántica **la usa**, no
  la redefine.
- `Provenance` por observación: se **extiende**, no se sustituye.
- La validación *fail-loud* del contrato.

---

## 6. Referencias

- [ADR 001 — Contrato de datos](001-digital-twin-core-schemas.md)
- [ADR 003 — Verificación y tolerancia a fallos](003-verification-fault-tolerance.md)
- [Arquitectura del pipeline multiagente](multi-agent-pipeline.md)
- [Experimento Point Transformer — perfil de error de la numeración FDI](../../notebooks/exercise-point-transformer-teeth3ds.md)
- DDMF — Deep Dental Multimodal Fusion, [arXiv:2203.05784](https://arxiv.org/abs/2203.05784), *Patterns* 2023
