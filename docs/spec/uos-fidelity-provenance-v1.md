# Fidelidad y procedencia UOS: extensión 1.0

Fecha: 2026-10-02. Estado: implementación local comprobada con el
`complete case`, con pendientes documentados, y pendiente de publicación. Describe código y límites; no
acredita aptitud clínica ni reproduce los resultados históricos del white paper.

## Contrato y compatibilidad

La implementación usa una extensión opcional versionada sobre UOS 0.3:
`uos_fidelity_provenance`, versión `1.0`, identificador de contrato
`uos-fidelity-provenance/1.0`. El manifiesto referencia el asset incluido
`asset.fidelity_provenance`, cuyo payload es
`metadata/fidelity-provenance.json`. Sus bytes tienen hash y tamaño en el
manifiesto. El asset de metadatos se excluye de su propio inventario para evitar
una referencia circular.

El [JSON Schema](../../schemas/uos-fidelity-provenance-1.0.schema.json) se deriva de
[fidelidad.py](../../packages/uos/src/uos/fidelidad.py). El escritor y los controles
adicionales están en [auditoria.py](../../packages/uos/src/uos/auditoria.py).
La ampliación no incorpora campos nuevos al sobre `Asset` de v0.3. Sí corrige la
descripción de la capa 2: procedencia computada y reproducibilidad son propiedades
distintas. El esquema del manifiesto se regenera por ese cambio documental.

Un contenedor v0.3 anterior sigue siendo legible. `read_fidelity(path)` devuelve
registros de origen y fidelidad desconocidos cuando falta la extensión; no escribe
ni completa garantías en el archivo antiguo. Un lector que omite la extensión
puede interpretar la escena base, pero no debe afirmar que ha comprobado estas
propiedades adicionales.

## Qué declara cada asset

| Propiedad | Significado |
|---|---|
| `origin` | Adquirido, transcrito, calculado, inferido, mixto o desconocido. |
| `sources`, `source_status` | Fuentes y estado resuelto, no resuelto o no aplicable a un original. |
| `process` | Operación, agente, versiones de software, parámetros JSON, semillas y modelo, cuando constan. |
| `current_encoding` | Codificación actual declarada o comprobada, distinta del historial de pérdida. |
| `prior_history_unknown`, `history` | Incertidumbre sobre la historia anterior y declaración/evidencia disponible. |
| `introduced_losses`, `inherited_losses` | Pérdidas registradas en este asset y recibidas por su linaje. |
| `metrics` | Valor, unidades, método, referencia y alcance de una comparación. |
| `calibration`, `sampling` | Calibración acreditada y muestreo declarado, sin confundirlo con resolución efectiva. |
| `registrations_used`, `geometric_uncertainty` | Dependencias geométricas y evidencia disponible de incertidumbre. |
| `clinical_assessments` | Conclusiones para una tarea explícita, únicamente con evidencia de esa tarea. |

Las afirmaciones tienen estados `unknown`, `declared` o `verified`. `unknown`
no puede contener una garantía. `verified` exige una referencia a evidencia,
método, alcance, resultado y verificador. Las métricas no pueden ser NaN o
infinito. Las pérdidas tienen identificadores estables dentro del registro,
asset de origen, clase, método y ratio/dimensiones cuando se conocen.

El validador comprueba la estructura y las referencias de esa evidencia, y avisa
que no autentica al verificador ni certifica la conclusión. Una declaración de
revisión conserva modelo y origen inferido; no transforma una inferencia en
adquisición. Una evaluación clínica sin tarea o evidencia se rechaza. El escritor
no genera automáticamente ninguna conclusión `eligible`.

## Pérdidas e incertidumbre heredadas

El escritor recorre el grafo de fuentes antes de serializar. Cada derivado recibe
las pérdidas documentadas de sus antecesores. Una historia desconocida en una
fuente continúa siendo desconocida en los derivados. Convertir a un formato sin
pérdida no elimina una compresión irreversible previa.

El validador repite ese recorrido. Rechaza ciclos, identificadores de eventos
duplicados, cambios de origen, pérdidas heredadas eliminadas o modificadas,
inventarios discordantes y eliminación de incertidumbre heredada. Una referencia
no resuelta no permite afirmar historia conocida. Las fuentes del registro deben
coincidir con `derived_from` del manifiesto.

La captura automática actual incluye:

- DICOM: declaraciones de compresión irreversible de todos los archivos con
  cabeceras legibles, incluidos método y ratios multivaluados. La sintaxis de
  transferencia actual no sustituye el historial. Se preservan los datos ausentes
  como desconocidos. Se leen los atributos de historia de
  [DICOM PS3.3 C.7.6](https://dicom.nema.org/medical/Dicom/2024e/output/chtml/part03/sect_C.7.6.html).
- JPEG: detección de los marcadores DCT soportados desde los bytes, sin asumir
  compresión ni submuestreo cromático por la extensión del archivo. El ratio y la
  historia anterior permanecen desconocidos cuando no constan.
- Malla glTF: máximo desplazamiento de coordenadas al convertir float64 a float32,
  comparado con las coordenadas de entrada. Es una medida numérica local; no es
  precisión del escáner ni evidencia anatómica.
- Campos gaussianos: submuestreo declarado, tamaños de entrada/salida cuando
  existen, reconstrucción y residual de ajuste con referencia y alcance. Los
  valores de CBCT no se presentan automáticamente como HU calibrados.

El contrato permite registrar otras pérdidas y métricas; su cálculo no se inventa
para formatos o procesos no inspeccionados. No incluye todavía la huella canónica
DICOM, un resolvedor de originales, perfiles completos multiframe ni calibración
colorimétrica o clínica.

## Separación de inferencias y cálculos

La escena base conserva la malla convertida. Los campos semilla y compuestos se
declaran como representaciones calculadas, con sus fuentes y parámetros
disponibles; no como nuevas adquisiciones.

La apariencia ajustada se exporta en `derived/appearance.glb`, con la primitiva
`KHR_gaussian_splatting` y su sidecar. No duplica la malla dentro de ese GLB.
La escena base no contiene esa apariencia ni apunta a ella. La apariencia de
un proceso importado sin caracterizar se conserva de forma conservadora en
capa 3, con las fuentes incompletas explícitas. La geometría ajustada condicionada
por etiquetas se exporta igualmente bajo `derived/`. Quitar una columna FDI no
elimina por sí mismo su influencia anterior sobre colores o geometría.

Las fuentes de informes se identifican por SHA-256 desde la ingesta. Renombrar
un informe mantiene su identidad; sustituir sus bytes deja la referencia sin
resolver. Se mantiene compatibilidad de exportación con snapshots antiguos que
usaban rutas, sin publicar esas rutas dentro de los valores clínicos.

El OCR se declara inferido tanto en observaciones regionales como en medidas
globales y en el resultado de ingesta. Las medidas inferidas se exportan en
`derived/`, conservando modelo y fuente. Las medidas transcritas conservan su
procedencia individual. Cuando dos observaciones proporcionan distintos valores
para el mismo campo dental, se conservan alternativas y procedencias y se señala
la selección no revisada; ya no se pierde silenciosamente el valor anterior.

`remove_inference(source, destination)` crea un sucesor separado, comprueba el
contenedor de origen, retira assets/sidecars inferidos, actualiza el registro de
fidelidad y el mapeo FHIR, y encadena el nuevo manifiesto. Vacía las vistas guardadas
porque su encuadre podía depender de etiquetas. Comprueba el resultado antes de
publicar el archivo de destino. Conserva la escena base y no promete borrar
otras versiones o copias. Las firmas/revisiones no están autenticadas.

Los STL compuestos conservan el mapa por intervalos de caras implementado en la
corrección anterior: superficie del escáner, cierre sintético y raíz reconstruida.
Ese mapa está ligado al hash del STL; no acredita la validez anatómica ni viaja
con una impresión física.

## Reproducibilidad y geometría

El modo del algoritmo (`deterministic`, `stochastic`, `unknown`) no acredita que
se hayan repetido ejecuciones. `process.repeatability` permanece desconocido
hasta registrar la comprobación correspondiente. `process_records` permite
aportar configuración/evidencia explícita al exportador, sin cambiar la identidad
de un modelo ya declarada.

El entrenamiento de apariencia usa ahora un generador local con semilla para
elegir vistas, además de la semilla Torch. Guarda semillas, parámetros y versiones
en el artefacto y los traslada al UOS. No se promete reproducibilidad exacta de
GPU, Blender o de procesos previos por el mero hecho de fijar semillas.

Las dependencias de registros aparecen en los metadatos geométricos. El residual
de ajuste del registro permanece en su contrato original; no se convierte en TRE
ni en incertidumbre propagada. La incertidumbre y la calibración no acreditadas
quedan desconocidas. El cálculo de covarianzas, su propagación y las comprobaciones
con referencias independientes requieren trabajo y evidencia adicionales.

## Validación y publicación

Los códigos nuevos son `UOS-E/W-020` para fidelidad/evidencia y `UOS-E/W-021`
para fuentes/procesamiento. Se prueban herencia transitiva, alteraciones del
historial, ciclos, evidencia incompleta, referencias inexistentes, compatibilidad
anterior, fuentes renombradas/modificadas, OCR, alternativas contradictorias,
separación de apariencia y retirada de inferencias como sucesor válido.

Comprobación local: 1168 tests pasan, 2 se omiten y aparecen 2 advertencias
numéricas en la prueba existente de apariencia. Ruff pasa en los archivos Python
afectados, MyPy en los 12 archivos fuente afectados y el guardián de datos no
detecta infracciones. La especificación LaTeX compila. `docs_sync --check` todavía
señala las dos referencias a este documento sin añadir a Git y la discrepancia
del tag de esquema descrita abajo; no se considera cerrado ese control.

El [complete case](uos-complete-case-2026-10-02.md) se ejecutó después: el UOS y el
sucesor sin inferencias validan. Quedan advertencias, una fuente CBCT sin resolver
y limitaciones de color/revisión humana. Los resultados históricos del paper,
la repetibilidad medida y la validez clínica siguen pendientes de comprobación.
La implementación del contrato no sustituye esos experimentos.

Antes de publicar, deben versionarse los cambios y actualizarse de forma explícita
los artefactos publicados. El tag móvil `uos-spec-v0.3-draft` todavía debe pasar a
una revisión que incluya la descripción corregida del esquema del manifiesto;
no se mueve durante esta implementación. Los commits y la publicación quedan
para ejecución manual del autor.
