# Procedencia, calibración y repetibilidad: resultados del 7 de octubre de 2026

Se completaron la inspección de los originales, las correcciones de ingesta y coordenadas,
seis ejecuciones independientes y la actualización del paper. La procedencia y la
repetibilidad técnica se han comprobado en el alcance descrito. La calibración física y
la alineación clínica no se han acreditado. La revisión de segmentación continúa aplazada.

## 1. Originales y separación de series

El ZIP descargado contiene **401 instancias DICOM**, en dos series y dos marcos de
referencia: **100 y 301 cortes**. Cuatro nombres de fichero se repiten entre series con
contenidos distintos. La extracción anterior a un directorio plano sobrescribió cuatro
instancias y dejó 397 ficheros, distribuidos en grupos de 97 y 300. No faltan nombres
respecto del ZIP: faltaban las versiones de esos cuatro nombres pertenecientes a la otra
serie. Esta comprobación se hizo leyendo cada entrada del ZIP por separado.

Se prepararon copias de trabajo independientes por serie, con hashes por instancia y del
archivo descargado. No se modificaron los originales, no se reescribieron UIDs y no se
fusionaron marcos. Cada serie pertenece a un único estudio, serie y marco. Ambas tienen
muestreo irregular: intervalos máximos de 12,3 y 6,15 mm respectivamente. Estos intervalos
no demuestran si el proveedor seleccionó cortes o si faltan datos de adquisición.

El ejecutor permite seleccionar explícitamente una serie mediante `--cbct`. En el ensayo
se utilizaron ambas por separado; no se eligió automáticamente la mayor para uso clínico.
El campo referencia ahora su DICOM fuente y la auditoría local comprueba los hashes.
Los originales DICOM, STL, fotos e informes siguen fuera del UOS.

## 2. Correcciones de geometría e ingesta

- Se rechazan estudios, series y marcos mezclados antes de decodificar píxeles.
- Se conserva el origen XY de ImagePositionPatient, antes omitido.
- El espaciado Z nominal usa la mediana de separaciones, evitando que un intervalo inicial
  sin cortes determine toda la escala. Las posiciones reales se mantienen.
- El lector axial rechaza orientaciones no soportadas y orígenes XY variables.
- El campo gaussiano centrado y el volumen DICOM tienen marcos distintos en el UOS.
  El registro del campo al escáner compone su origen registrado con la inversa de ICP.
- Los sidecars transportan posiciones de corte y el aviso de muestreo no uniforme.
- La inferencia volumétrica no se ejecuta sobre estas entradas irregulares. No se interpolan
  intervalos sin datos para fabricar una rejilla regular.

La conservación de las posiciones del campo semilla se comprobó contra las cabeceras
fuente, deshaciendo el centrado declarado en el manifiesto:

| Serie | Primitivas semilla | Error XY máximo | Error Z máximo |
|---|---:|---:|---:|
| 100 cortes | 109.148 | 7,98 × 10⁻⁷ mm | 2,68 × 10⁻⁸ mm |
| 301 cortes | 808.761 | 1,81 × 10⁻⁶ mm | 7,89 × 10⁻⁷ mm |

Son errores de conservación numérica respecto de los metadatos originales, no errores
medidos contra un objeto físico independiente.

## 3. Seis ejecuciones y criterio de repetibilidad

Se hicieron **tres ejecuciones independientes por serie**, seis en total, con directorios
de artefactos y exportaciones separados. El protocolo comprende ingesta, registro
geométrico, ajuste de densidad, exportación UOS y regeneración de la malla.

Se usó semilla 0, las mismas entradas y el mismo entorno. Se reutilizaron las etiquetas
IOS existentes y la apariencia archivada de agosto. No se entrenó apariencia nueva ni
se ejecutó el modelo de segmentación CBCT sobre una rejilla irregular. La revisión de
segmentación permanece fuera de alcance.

| Comprobación | Resultado |
|---|---|
| Auditoría contra los ficheros DICOM fuente | Las seis pasan; hashes por instancia y serie coinciden |
| Payloads binarios UOS | Los ocho payloads son idénticos en las tres ejecuciones de cada serie |
| Registros de proceso y matrices de registro | Idénticos dentro de cada serie |
| Comparaciones entre parejas | Las tres parejas de cada serie pasan, seis comparaciones en total |
| STL y PLY regenerados | Idénticos en las seis ejecuciones, también entre series |
| Bytes del contenedor | No se afirma identidad del ZIP; hay metadatos temporales |
| JSON | Se informa igualdad de bytes por separado; no se afirma equivalencia semántica |

La repetibilidad observada se limita a estos datos, configuración y entorno. No se
extrapola a otros equipos, versiones, entrenamientos ni resultados clínicos. Los estados
de repetibilidad del contenedor no se modificaron retrospectivamente: la evidencia se
conserva en un informe externo ligado a los hashes de los seis UOS.

Entorno: Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0, PyTorch 2.11.0+cu128,
CUDA 12.8, NVIDIA RTX 5070 y controlador 610.57.04. Agente exportador **0.18.0**,
agente CBCT **0.2.0**, formato **UOS 0.3**, extensión de fidelidad **1.0**.

## 4. Calibración y registro: qué permanece desconocido

Las cabeceras declaran rescale con pendiente 1, intercepto −1000 y RescaleType=HU.
No se encontró evidencia independiente de calibración; el usuario no conoce referencias
adicionales disponibles. Se conserva la declaración del proveedor, con `calibrated_hu=false`
y `calibration_status=unknown`.

- **Geometría:** se comprueba conservación de coordenadas; no exactitud del escáner frente
  a un patrón físico ni resolución efectiva del CBCT.
- **Intensidad:** se aplica el rescale declarado; no se acreditan HU calibrados.
- **Color:** se recupera la apariencia archivada; no se acredita calibración de color.

Los registros de las series separadas tampoco acreditan alineación clínica:

| Serie | RMS en correspondencias | Solapamiento | RMS sobre la nube completa |
|---|---:|---:|---:|
| 100 cortes | 16,830 mm | 0 % | 16,830 mm |
| 301 cortes | 0,680 mm | 3,6 % | 8,341 mm |

El primer registro no es aceptable para vincular la serie al escáner superior. El residuo
menor del segundo se obtiene sobre una fracción pequeña; no demuestra que toda la arcada
esté correctamente alineada. Repetir ambos resultados no valida su exactitud.

Para acreditar calibración harían falta referencias independientes de geometría,
intensidad y color. Para acreditar el registro harían falta controles independientes y
revisión de correspondencias. Esas capacidades no se declaran cerradas por software.

## 5. Paper y comprobaciones

El executive summary y los apartados de fidelidad, experimento y conclusiones se actualizaron.
La sección histórica conserva resultados anteriores, pero advierte que procedían de una
extracción mixta y que no validan una reconstrucción coherente de una única serie. El
nuevo experimento distingue los resultados repetibles de la exactitud no demostrada.

La suite completa pasa: **1175 pruebas, 2 omitidas**, con dos avisos numéricos preexistentes.
Ruff y MyPy de los módulos modificados pasan. El paper compila a PDF. La protección de
datos verifica que no se incorporan originales clínicos al historial Git.

Las evidencias se guardan localmente en `data/processed/verification-20261007/`:
preparación de series, y bajo `final/`, seis ejecuciones, registro de entorno, auditorías
contra las fuentes, comparaciones entre parejas y conservación de coordenadas.
Los ensayos exploratorios anteriores a la corrección de marcos no se cuentan en el
resultado final; desde el 10 de octubre están separados en `exploratory/`. El paper compilado está en el subdirectorio `paper/`.

Antes de publicar quedan la revisión editorial final, la coherencia del tag de esquema
y la decisión sobre qué evidencias pueden publicarse. Las referencias clínicas originales
y los UOS de pacientes permanecen fuera de Git. Este informe documenta las verificaciones
previas a los commits autorizados posteriormente; no se han movido tags.
