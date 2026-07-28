# Estrategia de anonimización — datos clínicos del Digital Twin

| | |
|---|---|
| **Estado** | Aceptado (implementado en la capa de ingesta) |
| **Fecha** | 2026-07-28 |
| **Ámbito** | Semana 3-4 · entregable «estrategia de anonimización para datos de fases posteriores» |
| **Relacionado** | [ADR 001](001-digital-twin-core-schemas.md) (contrato) · [`AGENTS.md`](../../AGENTS.md) · [`ingestion-agents`](../../packages/ingestion-agents/) |

> Este documento **consolida** la estrategia que ya aplica el código de ingesta.
> No introduce mecanismos nuevos: nombra, justifica y fija la política que
> implementan `cbct_agent.pseudonymize`, `BaseIngestionAgent` (cuarentena) y el
> contrato de `core-schemas`. Es el artefacto de «definición» que pedía el hito.

## 0. Postura en una frase

**Datos sintéticos primero; seudonimización —no anonimización— en el borde de
ingesta cuando lleguen datos reales.** Ningún identificador directo cruza la
frontera raw→contrato: a partir del `TwinSnapshot`, el paciente es un seudónimo.

## 1. Principio: sintético primero

El brief fija generar y validar con **datos sintéticos** antes de tocar datos de
paciente (`ingestion_agents/synthetic.py`, `main.py --demo`). Los datos reales
anonimizados del partner entran «cuando proceda y esté autorizado», ya sobre un
pipeline probado. Esto acota el riesgo: el sistema se construye y se mide sin
dato sensible.

## 2. Seudonimización, no anonimización

El identificador de paciente se sustituye por un **seudónimo estable y no
reversible** (`cbct_agent.pseudonymize`):

- **HMAC-SHA256** del `PatientID` con una **sal secreta**, truncado a 16 hex.
- **Estable**: el mismo paciente da el mismo seudónimo entre adquisiciones — es lo
  que permite reconstruir su serie temporal (`PatientDigitalTwin`) sin conocer su
  identidad.
- **No reversible sin la sal**: no se puede volver del seudónimo al identificador.

> **Es seudonimización (RGPD art. 4.5), no anonimización.** Con la sal, la
> reidentificación es posible; sin ella, no. Por eso **la sal es el dato a
> proteger**, y el vínculo seudónimo↔identidad, si se conserva, vive fuera de este
> sistema y bajo control del responsable del dato.

### Gestión de la sal

- Se lee de la variable de entorno **`ASH_PSEUDONYM_SALT`**.
- Si no está definida se usa una sal **de desarrollo** cuyo nombre lo advierte
  (`dev-salt-no-usar-en-produccion`): sirve para datos sintéticos, **nunca** para
  datos de paciente.
- En producción: sal larga y aleatoria, provista por secreto del entorno (no en
  el repo, no en el `.env` versionado), rotable por el responsable del dato.

## 3. Minimización: qué cruza la frontera y qué no

El `cbct-agent` extrae del DICOM **solo geometría e intensidades** (vóxeles → campo
gaussiano). **No propaga** al contrato ninguno de los identificadores directos que
el DICOM arrastra: nombre del paciente, fecha de nacimiento, `PatientID` crudo,
fechas de estudio con identidad, etc. El único rastro del paciente en el contrato
es el **seudónimo** derivado.

El contrato refuerza esto estructuralmente: `PatientDigitalTwin.patient_id` está
documentado como **seudónimo**, nunca identificador directo
([`core_schemas/models.py`](../../packages/core-schemas/src/core_schemas/models.py)).

## 4. Soberanía del dato en el manejo de fallos

Un fichero que falla la ingesta va a **cuarentena** (`BaseIngestionAgent._quarantine`),
pero **no se copia el contenido clínico**: el registro guarda la **ruta** del
fichero y el **traceback**, nada más. Mover un DICOM a un directorio de cuarentena
duplicaría dato de paciente fuera del almacenamiento autorizado; se evita a
propósito.

## 5. Trazabilidad compatible con RGPD

Cada valor lleva `Provenance` (fichero de origen, agente, confianza, instante).
Es la explicabilidad que exige el proyecto — «qué dato se ingirió, qué
transformación» — **sin** que ello reintroduzca identidad: el `source_file` es una
ruta, no un identificador de persona, y sobre datos reales apunta al fichero ya
seudonimizado en el almacenamiento autorizado.

## 6. Alcance y límites (honestos)

- **Implementado hoy** sobre la ingesta: seudonimización del `PatientID` de DICOM,
  minimización a geometría/intensidad, cuarentena sin contenido, seudónimo en el
  contrato. Verificado con tests (`test_cbct_agent.py`: seudónimo estable,
  dependiente de la sal, no contiene el identificador).
- **Fuera de alcance de esta versión** (fases posteriores, con datos reales):
  - Anonimización de **texto libre** en informes (los PDF pueden arrastrar
    identidad incidental: nombres, fechas, centro). El `report-agent` **no**
    depura el texto todavía — es trabajo pendiente antes de ingerir informes
    reales.
  - Anonimización de **imágenes** 2D (rostro/metadatos EXIF).
  - Gestión del **mapa de reidentificación** (custodia de la tabla seudónimo↔identidad),
    que es responsabilidad organizativa del partner, no de este código.
- **HIPAA/RGPD**: la seudonimización cubre el identificador estructurado; el texto
  libre y la imagen quedan como riesgo declarado a cerrar antes de datos reales.

## 7. Referencias de código

- `packages/ingestion-agents/src/ingestion_agents/cbct_agent.py` — `pseudonymize`, minimización.
- `packages/ingestion-agents/src/ingestion_agents/base.py` — cuarentena sin contenido.
- `packages/core-schemas/src/core_schemas/models.py` — `patient_id` seudónimo, `Provenance`.
- `packages/ingestion-agents/tests/test_cbct_agent.py` — verificación de la seudonimización.
