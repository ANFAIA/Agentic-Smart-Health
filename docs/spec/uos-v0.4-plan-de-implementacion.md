# UOS v0.4: propuesta para resolver los pendientes de v0.3

Fecha: 2026-09-30. Estado: propuesta de trabajo, no especificación aprobada ni capacidades implementadas.

Este documento desarrolla los pendientes identificados en el [contraste de la revisión técnica de v0.3](uos-v0.3-review-response.md). Propone ampliar los contratos en una v0.4, manteniendo la lectura de v0.3 y conservando las correcciones ya aplicadas.

El objetivo es poder explicar qué conserva cada representación, qué información se ha perdido, qué se ha comprobado y cómo recuperar su fuente. Añadir un campo llamado `fidelity` no demuestra por sí mismo fidelidad clínica.

## 1. Contrato de fidelidad por asset

Cada asset debe poder declarar su fuente, las transformaciones aplicadas, las pérdidas conocidas, las métricas medidas y la evidencia que respalda esas métricas. El contrato distinguirá tres estados por propiedad:

- **Desconocido:** no hay información suficiente para caracterizarla.
- **Declarado:** el emisor aporta una afirmación, sin comprobación independiente registrada.
- **Verificado:** existe una comprobación identificada, con método, alcance y resultado.

Estos estados describen la evidencia disponible. Una propiedad verificada mediante una prueba técnica no equivale a aptitud clínica general.

Por ejemplo, un volumen reducido conservará las dimensiones originales y resultantes, el método de reducción y las métricas disponibles. Sus derivados heredarán esa historia: una reconstrucción posterior no eliminará del registro la pérdida previa.

La lectura de un contenedor v0.3 sin estos datos producirá un estado de fidelidad desconocido. No se rellenarán garantías por defecto ni se modificarán los archivos antiguos para aparentar evidencia inexistente.

**Criterios de aceptación:**

- Ninguna conversión transforma automáticamente «desconocido» en «sin pérdida».
- Las pérdidas documentadas permanecen trazables a través de los derivados.
- Cada métrica declara qué compara, sus unidades, método y referencia.
- La validación distingue integridad del contenedor, evidencia técnica y evaluación clínica.

## 2. Integridad de archivo e identidad del contenido DICOM

Se mantendrá el SHA-256 actual para identificar los bytes exactos. Se propone añadir una huella canónica versionada, con un alcance explícito, que cubra píxeles decodificados, dimensiones, geometría y transformación de intensidades.

El primer alcance será un conjunto definido de variantes CT/CBCT que podamos probar. Las variantes no soportadas quedarán expresamente sin verificar. El algoritmo deberá especificar la representación numérica, el orden de los datos y el tratamiento de atributos ausentes; no bastará con serializar un diccionario de etiquetas.

Para multiframe habrá que resolver los atributos compartidos y los específicos de cada frame, incluyendo geometría y transformación de intensidades. La referencia para esas estructuras es [DICOM PS3.3, Common Functional Group Macros](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html).

La huella identificará el contenido definido por ese algoritmo. No demostrará equivalencia clínica universal ni sustituirá la trazabilidad de una desidentificación o de un cambio de UID.

**Criterios de aceptación:**

- Una recompresión sin pérdida, dentro del alcance soportado, conserva la huella canónica aunque cambie el hash del archivo.
- Cambiar orientación, posición, escala o intensidades modifica la huella.
- La serialización canónica y su versión permiten repetir el cálculo de forma determinista.
- Un caso no soportado devuelve «no verificado», sin recurrir silenciosamente a una comprobación más débil.
- Se mantiene la comprobación de bytes exactos: una huella canónica coincidente no autoriza a ignorar un hash de archivo incorrecto.

## 3. Recuperación verificable de originales

Se propone un resolvedor que busque primero en el almacén local autorizado y permita después conectores DICOMweb. La recuperación de series dispone de una operación definida mediante [WADO-RS RetrieveSeries](https://dicom.nema.org/medical/dicom/2017d/output/chtml/part18/sect_6.5.2.html).

Antes de entregar un original al consumidor, el resolvedor comprobará los hashes y el inventario declarado. Diferenciará los resultados «no encontrado», «sin autorización», «servidor inaccesible», «contenido incompleto» y «contenido distinto».

Las credenciales permanecerán fuera del `.uos`. La descarga y cualquier caché deberán usar el almacenamiento autorizado por la arquitectura. Un localizador será una pista de acceso, no una garantía de disponibilidad futura.

**Criterios de aceptación:**

- Recuperar una serie completa desde el almacén local y desde un servidor de prueba.
- Detectar cortes ausentes, respuestas parciales y contenido modificado.
- No entregar contenido como verificado antes de completar las comprobaciones necesarias.
- Mantener separados el estado de acceso y el resultado de integridad.

## 4. Incertidumbre y reproducibilidad

Para los registros, se separarán la discrepancia angular, el desplazamiento sobre una región definida, el residual de ajuste y el TRE medido sobre puntos independientes. El residual ICP no se convertirá automáticamente en incertidumbre ni en error sobre una estructura clínica concreta.

La propagación de incertidumbre requerirá fijar el modelo de perturbación y el tratamiento de las correlaciones entre registros. El camino más corto seguirá siendo una convención determinista mientras no exista un criterio de selección por precisión sustentado en evidencia.

Para los procesos gaussianos se registrarán configuración, versiones y semillas, y se repetirán ejecuciones para medir la variación. Las métricas y los umbrales se definirán para la tarea evaluada; no habrá un umbral universal deducido del PSNR.

**Criterios de aceptación:**

- Comprobar los cálculos con transformaciones conocidas y perturbaciones controladas.
- Evaluar el registro con referencias independientes de las utilizadas para ajustarlo.
- Reservar los datos de evaluación del proceso de ajuste.
- Medir la variación entre ejecuciones y distinguir reproducibilidad exacta de reproducibilidad dentro de una tolerancia.

## 5. Procedencia completa y revisión firmada

La ingesta debe conservar referencias estables a los documentos y, cuando sea posible, al fragmento que respalda cada observación. Esto permitirá reducir las fuentes que hoy se exportan como «no resueltas», sin inventar una referencia cuando no exista.

Después se incorporarán firmas sobre contenido bien definido, con identidad del firmante, alcance de la revisión y gestión de claves. Para la canonicalización de JSON se evaluará [JCS, RFC 8785](https://www.rfc-editor.org/info/rfc8785/).

Antes de desplegar la firma habrá que decidir quién firma —clínica, plataforma o ambas—, dónde se custodian las claves y cómo se establece la confianza en ellas. Una firma técnicamente válida y una revisión clínica acreditada son propiedades distintas.

La aprobación humana conservará la procedencia inferida original. Revisar una observación no la convertirá retroactivamente en un dato adquirido.

**Criterios de aceptación:**

- Vincular una observación con su fuente y localizar la evidencia disponible.
- Mantener explícito el estado no resuelto cuando esa vinculación no sea posible.
- Detectar cualquier modificación del contenido cubierto por una firma.
- No presentar una firma con clave no confiable como revisión acreditada.
- Conservar el modelo, la extracción y las revisiones sucesivas en la trazabilidad.

## 6. Validación clínica e interoperabilidad

Las raíces reconstruidas, la segmentación y la fidelidad gaussiana requieren experimentos específicos con referencias independientes. Corregir su empaquetado y procedencia permite auditarlas; no demuestra su exactitud anatómica o su utilidad para una tarea clínica.

En paralelo, se probarán los contenedores con un segundo lector y se verificará la compatibilidad con la revisión final de `KHR_gaussian_splatting`. La evidencia obtenida deberá indicar las versiones, las funciones probadas y las limitaciones observadas.

**Criterios de aceptación:**

- Definir cada tarea evaluada, sus referencias, métricas y criterios de éxito antes de medirla.
- Separar los resultados experimentales de las garantías del formato.
- Comprobar con un segundo lector la interpretación de geometría, registros, capas y procedencia.
- Documentar las incompatibilidades y conservar los resultados negativos.

## Orden de ejecución

| Prioridad | Bloque | Resultado esperado |
|---|---|---|
| 1 | Fidelidad por asset | Contrato de evidencia, pérdidas y estados desconocidos, con lectura compatible de v0.3. |
| 2 | Identidad DICOM | Huella canónica versionada para un alcance soportado y probado, además del hash de bytes. |
| 3 | Recuperación de originales | Acceso local y remoto con comprobación de contenido y estados de fallo explícitos. |
| 4 | Incertidumbre y reproducibilidad | Métricas con significado definido y resultados de evaluación independientes. |
| 5 | Procedencia y firmas | Fuentes trazables y revisión autenticada con alcance y confianza explícitos. |
| 6 | Validación clínica e interoperabilidad | Evidencia por tarea y pruebas con una segunda implementación. |

Las referencias estables de ingesta del bloque 5 deben diseñarse junto a los tres primeros bloques. La preparación de los experimentos y del segundo lector puede avanzar en paralelo; las conclusiones dependerán de los resultados obtenidos.

Cada bloque debe incluir contrato, implementación, pruebas y documentación. El white paper se actualizará después de las comprobaciones correspondientes, distinguiendo las capacidades implementadas, las propiedades demostradas y las propuestas todavía pendientes.
