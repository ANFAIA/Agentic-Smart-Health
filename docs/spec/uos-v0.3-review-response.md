# UOS v0.3 — Respuesta actualizada a la revisión de Matías

**Fecha de actualización:** 2 de octubre de 2026. **Ámbito:** white paper, especificación, implementación de referencia, ejecución del complete case y visor local.

**Revisión de referencia:** Matías Molinas, *UOS v0.3 — Revisión técnica para el autor de la especificación*, 25 de septiembre de 2026, 32 páginas. La revisión examina el white paper del 24 de septiembre; sus observaciones se han contrastado también con el contrato y el código. La numeración de las observaciones que aparece abajo corresponde a esa revisión.

## 1. Resumen del estado actual

Se han corregido afirmaciones del white paper que excedían la evidencia y defectos de implementación relacionados con procedencia, DICOM, separación de inferencias y exportación. Después se ha añadido un contrato de fidelidad y procedencia por asset mediante una **extensión opcional 1.0 compatible con UOS 0.3**. El exportador está en la versión **0.17.0**.

El complete case se ha ejecutado sobre los datos locales usados anteriormente. Se ha generado y validado un nuevo contenedor; se han recuperado la apariencia de agosto, la selección dental, las fichas clínicas y la regeneración de STL, PLY y 3MF. Los archivos clínicos originales permanecen fuera del contenedor generado, declarados por referencias y hashes. La geometría convertida del escaneo sí viaja dentro.

**Esto no cierra toda la revisión de Matías.** Están implementados el registro técnico de fidelidad, la herencia de pérdidas documentadas, la separación de inferencias y varias correcciones funcionales. No están demostradas la aptitud diagnóstica, la calibración clínica ni la repetibilidad experimental. Tampoco se han implementado recuperación garantizada de originales, identidad DICOM canónica, firmas ni incertidumbre geométrica propagada.

La revisión de segmentación queda **aplazada por decisión del responsable del proyecto**. Se conservan etiquetas, avisos y motivos de revisión; no se presentan como segmentación clínicamente validada.

### Qué significa cada estado

- **Implementado y comprobado:** comportamiento presente y contrastado mediante pruebas o artefactos locales, dentro del alcance indicado.
- **Corregido editorialmente:** se ha corregido una afirmación; eso no implica haber implementado la capacidad propuesta.
- **Parcial:** existe parte del mecanismo, pero falta funcionalidad o evidencia.
- **Pendiente / aplazado:** no se ha cerrado; aplazado identifica una decisión expresa sobre la segmentación.

Un hash demuestra identidad de bytes cuando puede comprobarse; no demuestra recuperación, autenticidad del emisor ni exactitud clínica. Una validación UOS-Core no constituye autorización clínica ni permiso de distribución.

## 2. Qué se ha corregido en el documento

La primera intervención corrigió el texto manteniendo su estructura y sus resultados negativos. Sus cambios principales fueron:

1. Separar conservación numérica de una malla y precisión clínica del escáner.
2. Distinguir el residual de ICP del error en puntos de interés, o TRE; dejar de presentar el primero como una barra de error anatómico.
3. Presentar el color como estimación regional desde fotografías procesadas, sin prometer calibración ni resolución de medida por vértice.
4. Distinguir calidad de representación en proyección, visualización y aptitud diagnóstica. PSNR y SSIM no acreditan diagnóstico.
5. Retirar garantías de autenticidad y no repudio derivadas únicamente de una cadena de hashes.
6. Explicar que dejar originales fuera condiciona su disponibilidad a la custodia externa; el contenedor no garantiza un archivo autónomo a largo plazo.
7. Corregir el alcance de DICOM y diferenciar destinos FHIR por versión, sin presentar indicaciones de mapeo como conectores bidireccionales validados.
8. Retirar la exención regulatoria general basada en el número de capa, la autorización implícita al quitar `derived/` y promesas de historia clínica legal completa.
9. Presentar las etiquetas FDI y las raíces reconstruidas como experimentales, con sus anomalías visibles desde la descripción del resultado.
10. Explicar que UOS-Core admite una escena de malla sin gaussianas y que la investigación de splatting no es un requisito para cualquier implementador.

**Pendiente editorial de publicación:** el white paper aún contiene formulaciones anteriores a la extensión de octubre, especialmente las que sitúan la fidelidad por asset como trabajo futuro. Debe recibir una revisión final para incorporar el contrato nuevo y distinguir los experimentos históricos de las ejecuciones recientes. Esta respuesta actualizada no significa que ese repaso final ya se haya realizado.

## 3. Correcciones en la implementación

### 3.1 Fidelidad, pérdidas y evidencia por asset

Se implementó `uos_fidelity_provenance`, versión `1.0`, con payload en `metadata/fidelity-provenance.json`, declarado y protegido por hash en el manifiesto. El contrato conserva el origen del dato, fuentes, operación, parámetros, semillas cuando constan, codificación, pérdidas introducidas e heredadas, calibración, muestreo e incertidumbre.

Las afirmaciones distinguen **desconocido, declarado y verificado**. Una afirmación verificada requiere referencia a evidencia, método, alcance, resultado y verificador. El validador comprueba la estructura y referencias; no autentica al verificador ni convierte esa declaración en certificación clínica.

El escritor y el validador recorren el linaje. Una pérdida documentada o una historia anterior desconocida no desaparecen al convertir el dato. Se comprueban ciclos, referencias, inventario, herencia y contradicciones. Los controles nuevos se integran en los códigos `UOS-E/W-020` y `UOS-E/W-021`, manteniendo la numeración del proyecto.

La captura automática incluye atributos DICOM de compresión irreversible, marcadores JPEG soportados, desplazamiento numérico float64→float32 en malla y submuestreo/reconstrucción de campos. No inventa ratio, historia anterior ni exactitud clínica si no constan. El registro permite evaluaciones por tarea, pero el escritor **no genera elegibilidad diagnóstica automática**.

### 3.2 Procedencia de informes y separación de inferencias

Las fuentes de informes se identifican por SHA-256 desde la ingesta. Renombrar un archivo conserva su identidad; cambiar sus bytes no. El OCR y las rutas basadas en modelos mantienen origen inferido tanto en observaciones regionales como en medidas globales. Sus resultados se exportan bajo `derived/`, con modelo y fuente disponibles. La aprobación humana no borra su origen inferido.

Se conservan valores alternativos y procedencias cuando varias observaciones discrepan sobre un campo dental. La selección no revisada queda indicada; no se pierde silenciosamente un valor anterior. La extracción determinista de texto nativo se distingue de OCR/LLM.

La escena base conserva la geometría convertida sin etiquetas ni apariencia inferida incrustadas. La apariencia ajustada viaja en un GLB separado bajo `derived/`; la geometría ajustada condicionada por etiquetas permanece igualmente separada. Los códigos FDI viajan en arrays indexados independientes.

`remove_inference` crea un sucesor validado, actualiza metadatos y cadena y retira las salidas inferidas. Conserva los bytes de la escena base. No implica borrar otras versiones, copias o repositorios, ni implementa la redacción firmada propuesta por Matías.

### 3.3 Integridad DICOM, calibración y superficies sintéticas

Una serie DICOM incluida cuyos bytes no coinciden con el hash declarado se rechaza con `UOS-E-007`, aunque sus UID y `PixelData` almacenado permanezcan iguales. Se evita aceptar una cabecera alterada como si demostrase equivalencia clínica. No se ha implementado el hash canónico de píxeles decodificados más geometría y escala.

La modalidad CT ya no sirve para declarar automáticamente HU calibrados. Sin evidencia se registra calibración desconocida. Los mensajes del complete case hablan de **unidades de gris de entrada**, no de HU acreditadas. Esto corrige la declaración; no calibra el equipo.

Los exports STL compuestos incorporan un mapa acompañante de intervalos de caras ligado por SHA-256 al STL, que diferencia superficie del escáner, cierre sintético y raíz reconstruida. Debe conservarse junto al archivo. No acredita anatomía ni acompaña automáticamente una impresión física.

### 3.4 Reproducibilidad declarada y repetibilidad medida

La capa de procesamiento deja de prometer reproducibilidad por su número. Se registran por separado tipo de algoritmo, configuración, versiones y semillas disponibles. El entrenamiento de apariencia utiliza una semilla para la selección de vistas y la de Torch, conservadas en el artefacto.

**Fijar semillas y registrar configuración no es demostrar repetibilidad.** La comprobación experimental entre ejecuciones sigue desconocida. Tampoco se promete identidad bit a bit entre GPU, Blender, versiones o procesos archivados.

## 4. Recuperación funcional del complete case y del visor

La separación más estricta de inferencias exigió adaptar el visor. La revisión de Matías no pedía perder color, selección o fichas: las incompatibilidades aparecidas al cambiar la organización del contenedor se han corregido.

### Apariencia y color

La primera ejecución de octubre entrenó una apariencia nueva: 127.635 gaussianas, PSNR 34,26 dB y SSIM 0,955 frente a renders de referencia. Resolvió dos poses y proyectó/interpoló color, pero no activó la extracción de tonos por pieza con `--lado-foto`. Esas métricas no acreditaban que su apariencia coincidiera con agosto.

La ejecución corregida reutiliza el artefacto archivado de agosto: **113.540 gaussianas y 13 registros de color por pieza**. Se comprueban el escaneo y las fotos fuente por contenido y el artefacto contra la apariencia transportada anteriormente. Los atributos binarios de posición, escala, opacidad, orientación y coeficientes de color coinciden exactamente. Las etiquetas de malla también coinciden con las archivadas.

Se registran los hashes de la reutilización y se declara que **no hubo entrenamiento nuevo de apariencia en esta ejecución**. Recuperar el aspecto anterior no demuestra calibración de color ni valida retrospectivamente el resultado de agosto.

### Selección, aislamiento y ficha clínica

`asset.seg_appearance` aporta un código por gaussiana y `asset.seg_teeth` un código por vértice de malla. El visor cruza las etiquetas con su fuente y comprueba hashes, recuento y, cuando se declara, orden de posiciones. La unión ocurre en memoria; no vuelve a incrustar inferencias en la escena base.

Se recuperan el clic sobre el modelo, el aislamiento y la selección desde el odontograma. Se corrigió además un fallo del pase de selección: el rasterizador de apariencia podía pintar RGB sobre el buffer de identificadores. Ahora se excluye y se rechazan códigos que no están presentes en las etiquetas.

Las fichas interpretan las generaciones anteriores y actuales de los registros clínicos, incluido el color envuelto con procedencia. Conservan alternativas y muestran detalles desplegables. Se ajustó su disposición para mantener accesibles los datos y el botón de cierre.

La comprobación en el navegador reconoce **14 piezas**, selecciona una pieza válida por clic, muestra su ficha, permite cerrar la selección y seleccionar la pieza 26 desde el odontograma, sin errores de JavaScript. Esto prueba el flujo funcional; no la corrección de los límites dentales.

### Regeneración de STL, PLY y 3MF

La malla se regenera desde el propio `.uos`, sin necesitar el STL original. Se leen geometría, apariencia y etiquetas separadas; se conserva la procedencia y se corrigieron diferencias de precisión numérica en el traspaso de color.

El resultado tiene **112.067 vértices y 220.085 triángulos**. La comparación del visor con Python coincide exactamente en RGB y bandera de cobertura por vértice y en posiciones y atributos de color por triángulo del STL. Las normales pueden diferir por precisión aritmética. Se generan PLY, STL y 3MF; la prueba de igualdad realizada no acredita identidad integral del paquete 3MF.

El STL coloreado usa la convención no estándar RGB555/VisCAM; algunos receptores mostrarán sólo la geometría. El color por vértice se conserva en PLY y 3MF. «Mejorado» significa geometría del escaneo con apariencia transferida; no una reparación automática de anatomía o segmentación. Los vértices de la pieza sin color declarado permanecen en gris.

## 5. Estado frente a cada bloque de la revisión de Matías

### 5.1 Fidelidad y pérdidas — revisión §1

| Observación | Respuesta actual y límite |
|---|---|
| 1.1 Aptitud diagnóstica e historia de pérdida | **Parcial.** Hay registro por asset y estados de evidencia; no cálculo automático de aptitud clínica general ni validación diagnóstica por tarea. |
| 1.2 DICOM: codificación e historia irreversible | **Implementado en el contrato y escritor.** Se distingue codificación actual de historia y se heredan pérdidas documentadas. La ejecución real aún no resuelve su fuente CBCT. |
| 1.3 JPEG y color | **Parcial.** Se registra pérdida JPEG detectable y se matiza la estimación regional. No se ha demostrado calibración; no se presupone 4:2:0 ni se publica EXIF completo. |
| 1.4 STL como original recibido | **Corregido editorialmente.** Se distingue archivo recibido de formato nativo y preservación numérica de exactitud clínica. La cadena nativo→STL y su pérdida previa siguen sin caracterización completa. |
| 1.5 Resolución y calibración | **Parcial.** Se registran muestreo y calibración desconocida cuando corresponde. Spacing o submuestreo no prueban resolución efectiva; no se ha calibrado el CBCT ni el color. |

### 5.2 Identidad y recuperación — revisión §2

| Observación | Respuesta actual y límite |
|---|---|
| 2.1 Originales externos y archivo | **Parcial.** El contrato admite incluidos y externos y localizadores; el escritor actual externaliza originales. No hay resolvedor, recuperación garantizada ni perfil archival nuevo. |
| 2.2 Tres identidades | **Pendiente para el píxel canónico.** Se comprueban bytes y UID/PixelData almacenado; no se ha definido el hash de imagen decodificada más geometría y rescale. No se presume qué copia custodia la clínica. |
| 2.3 Nombres de cortes | **Contraste y corrección documental.** El digest usa UID+PixelData cuando constan en todos los cortes y nombre+hash en la rama alternativa. El nombre no se presenta como orden anatómico. |
| 2.4 Multiframe | **Pendiente.** No se ha cerrado un contrato y una verificación general por frame Enhanced CT. |

### 5.3 Seguridad y ciclo de vida — revisión §3

| Observación | Respuesta actual y límite |
|---|---|
| 3.1 Firmas y cadena | **Corregido editorialmente; firmas pendientes.** La cadena expresa coherencia interna, no autenticidad ni no repudio. `verified_by` y la evidencia no están autenticados. |
| 3.2 Cifrado | **Pendiente.** No existe un perfil UOS completo de cifrado y custodia de claves. |
| 3.3 Supresión y redacción | **Pendiente.** Retirar inferencias en un sucesor no equivale a tombstones firmados ni a borrado verificable en todas las copias. |
| 3.4 Desidentificación | **Parcial.** Existen declaraciones y controles, no inspección exhaustiva de píxeles, anatomía o identidades. Dejar originales fuera no anonimiza el contenido restante. |

### 5.4 Geometría e incertidumbre — revisión §4

| Observación | Respuesta actual y límite |
|---|---|
| 4.1 Residual frente a TRE | **Corregido editorialmente.** El RMS se limita al ajuste observado. El TRE independiente, covarianza, condicionamiento y validación anatómica siguen pendientes. |
| 4.2 Grafo y propagación | **Parcial.** Se comprueban caminos y discrepancias de matrices. No se propagan covarianzas ni se convierte la diferencia entre coeficientes en error anatómico en mm. |
| 4.3 Transformaciones y cámara | **Parcial.** El pipeline sí resuelve poses mediante PnP; no hay contrato general completo de cámara, similitud y deformación. |
| 4.4 Mandíbula y oclusión | **Parcial.** Hay frames y declaraciones de oclusión; no flujo bilateral, multioclusión o tracking validado. |
| 4.5 Unidades y orientación | **Parcial.** Se prueban dirección/inversión de matrices y se declaran convenciones. Faltan comprobaciones independientes de unidades y round-trip DICOM. |

### 5.5 Coherencia de capas — revisión §5

| Observación | Respuesta actual y límite |
|---|---|
| 5.1 Reproducibilidad de capa 2 | **Implementado el registro separado.** Se mantienen capas 1/2/3; no se adoptan 2a/2b. Configuración y semillas no acreditan repetibilidad medida. |
| 5.2 Extracción de informes | **Corregido en ingesta y exportación.** OCR/LLM, medidas y observaciones mantienen inferencia, modelo y fuente; aprobación no borra origen. Se distinguen de extracción determinista. |
| 5.3 Caras sintéticas | **Implementado y probado.** Mapa por intervalos de caras ligado al hash STL; no valida la superficie ni la fabricación. |
| 5.4 Corona y raíz | **Parcial.** Se distingue el origen en el mapa acompañante. Las raíces siguen experimentales; no se han corregido sus anomalías anatómicas. |
| 5.5 Color regional y por vértice | **Corregido el alcance y recuperada la función.** Estimaciones por tercios no equivalen a medición independiente por vértice. La apariencia de agosto se conserva sin nueva calibración. |
| 5.6 Etiquetas FDI | **Función recuperada; revisión aplazada.** Pueden seleccionarse e inspeccionarse piezas, con etiquetas separadas e inferidas. Sus fronteras no se dan por validadas. |
| 5.7 Capas y calificación regulatoria | **Corregido editorialmente.** El número de capa no acredita una exención ni determina la calificación del módulo. |

### 5.6 Gaussianas y evaluación — revisión §6

| Observación | Respuesta actual y límite |
|---|---|
| 6.1 Diagnóstico, PSNR y HU | **Corregido el alcance.** Se registra la representación y sus métricas; no se acredita aptitud diagnóstica, HU calibrados ni recuperación de detalle perdido. |
| 6.2 Evaluación futura | **Pendiente.** No se han realizado fantomas, métricas diagnósticas independientes ni estudios de observador por tarea. |
| 6.3 Fitness y fuente de medidas | **Parcial.** Hay evaluaciones con tarea/evidencia y restricciones declaradas. No se ha implementado toda la política de prohibición y recurso al original propuesta por Matías. |
| 6.4 Transporte web | **Corregido editorialmente.** No se presenta UOS como sustituto diagnóstico validado de DICOM/MPR. |
| 6.5 Peso de la investigación | **Corregido el alcance.** UOS-Core puede prescindir de splats. La división del paper en dos publicaciones sigue aplazada. |

### 5.7 Contenido clínico — revisión §7

| Observación | Respuesta actual y límite |
|---|---|
| 7.1 Radiografía 2D | **Parcial.** Existen tipos de imagen y proyección, no todos los conectores y registros por modalidad. |
| 7.2 Odontograma y periodontograma | **Parcial.** La selección visual por FDI funciona; no equivale a perfiles interoperables completos de odontograma y periodontograma. |
| 7.3 Objeto dental longitudinal | **Pendiente.** FDI identifica posición; falta entidad estable de diente/implante/corona y sus eventos. |
| 7.4 Acto clínico | **Parcial.** Existen metadatos y consentimiento, no autoría autenticada ni protocolo clínico completo. |
| 7.5 Diversidad de casos | **Pendiente.** El caso maxilar no valida mandíbula, edéntulos, dentición mixta, metal o movimiento. |

### 5.8 Contenedor — revisión §8

| Observación | Respuesta actual y límite |
|---|---|
| 8.1 STORE y rangos | **Restricción implementada; ventaja interna no medida.** STORE no es necesario para recuperar una entrada completa. El visor tiene lector por rangos, sin benchmark que cierre el beneficio de acceso interno aleatorio. |
| 8.2 Compresión glTF | **Parcial.** Se caracteriza la conversión numérica actual, no toda la política por extensión/códec y grado de fidelidad propuesta. |
| 8.3 ZIP64 y MIME | **Pendiente en interoperabilidad y gobernanza.** El uso de `zipfile` no prueba lectores externos de más de 4 GiB; el tipo de medio no está registrado como estándar neutral. |

### 5.9 Estándares — revisión §9

| Observación | Respuesta actual y límite |
|---|---|
| 9.1 FHIR R4/R5 | **Corrección documental.** Se distinguen recursos por versión; no se acredita un conector bidireccional validado. |
| 9.2 Alcance de DICOM | **Corrección documental.** Se reconocen objetos existentes. Su existencia no garantiza soporte de cualquier PACS receptor. |
| 9.3 Round-trip | **Pendiente.** Los destinos documentados no prueban equivalencia semántica de importación y exportación. |
| 9.4 Terminología | **Parcial.** Se conserva texto y no se inventan códigos. Falta codificación supervisada y localización general por página/posición. |

### 5.10 Alcance regulatorio y legal — revisión §10

| Observación | Respuesta actual y límite |
|---|---|
| 10.1 Exención de capas 1/2 | **Retirada del texto.** Esta intervención no determina la clasificación del producto ni verifica cumplimiento por jurisdicción. |
| 10.2 Ley de IA y 10.3 EHDS | **Pendientes de evaluación específica.** La procedencia implementada no acredita por sí sola cumplimiento ni compatibilidad. |
| 10.4 Historia clínica | **Parcial.** Metadatos y consentimiento no constituyen el ciclo completo de firma, acceso, retención y supresión. |
| 10.5 Exportación al paciente | **Propuesta pendiente.** No existe el perfil completo con originales, visor y resumen garantizados descrito por la revisión. |

### 5.11 Posicionamiento — revisión §11

| Observación | Respuesta actual y límite |
|---|---|
| 11.1 Destinatarios | **Corregido editorialmente.** Se condicionan acceso y usos; no se prometen resultados de laboratorio o seguimiento sin evidencia. |
| 11.2 Diferenciación | **Corregido editorialmente.** Se describen capacidades concretas sin exclusividad universal frente al mercado. |
| 11.3 Dos documentos | **Aplazado.** Se mantiene el paper; esta respuesta y el informe del complete case documentan por separado las correcciones. |
| 11.4 Disciplina de evidencia | **Conservada.** Resultados negativos, N, códigos no inventados y límites siguen explícitos. Otro lector no convierte por sí solo la propuesta en estándar formal. |

### 5.12 Propuestas de contrato y lector — revisión §12

Se adopta el principio de fidelidad y pérdida heredada mediante una extensión compatible, no copiando literalmente el JSON ilustrativo ni sus campos `layer: "2b"`. Se conserva la numeración existente de controles. La identidad canónica, recuperación, covarianzas, autenticación y políticas clínicas completas siguen pendientes.

El visor muestra el registro declarado y distingue la procedencia de las etiquetas; se ha comprobado selección, ficha y exportación. **No se han cerrado los cuatro comportamientos obligatorios del lector propuestos en §12.5**, incluyendo el recurso automático al DICOM para cortes/mediciones y la presentación completa de incertidumbre geométrica.

## 6. Evidencia técnica disponible

| Comprobación | Resultado y alcance |
|---|---|
| Suite del monorepo | 1.168 pruebas correctas, 2 omitidas y 2 advertencias numéricas en una prueba existente de apariencia. Resultado de la ejecución documentada durante las correcciones; no se ha repetido la suite para redactar este informe. |
| Python y especificación | Ruff y MyPy pasaron en los archivos comprobados; la especificación LaTeX compiló. Las pruebas sintéticas no son validación clínica. |
| Visor | Compilación correcta, pruebas unitarias de compatibilidad/etiquetas y regresión del pase de selección; comparación con el caso real y flujo de navegador comprobados. Las pruebas que requieren otras fixtures locales pueden omitirse. |
| UOS corregido | 191.351.263 bytes; 24 assets; 13 originales externos; UOS-Core válido, cero errores y 42 advertencias. No acredita UOS-Distributable ni anonimización. |
| Retirada de inferencias | Sucesor válido, sin assets de capa 3 ni entradas `derived/`; escena base idéntica y archivo de origen sin modificar. |
| Apariencia archivada | Igualdad de atributos transportados frente a agosto; 113.540 gaussianas, 13 tonos regionales. No entrenamiento ni evaluación clínica nuevos. |
| Malla regenerada | 112.067 vértices, 220.085 triángulos; cobertura de apariencia del 97,1 %, que no equivale a exactitud del color. 3.145 vértices de la pieza sin color declarado en gris. |
| Procedencia CBCT del caso | `asset.field` permanece con fuente no resuelta porque el ejecutor pasa `cbct=None`. La representación no puede acreditar una vinculación completa al DICOM fuente. |

Los resultados anteriores de 140, 303 y 307 pruebas pertenecen a fases de septiembre y a la integración del 1 de octubre; no deben confundirse con el total posterior de 1.168. Los resultados experimentales del paper no se sustituyen por las cifras del run nuevo.

## 7. Próximo trabajo y lo que se deja aplazado

### Trabajo previsto, sin revisión de segmentación

1. **Procedencia CBCT del ejecutor:** declarar la serie fuente, vincularla al campo, comprobar hashes y regenerar el UOS. Estimación orientativa: **medio a un día**. No incluye recuperar originales desde un PACS externo.
2. **Repetibilidad experimental:** repetir ejecuciones con configuración registrada, definir comparaciones y documentar variación y tolerancias. Estimación orientativa: **uno a dos días**, según duración de los procesos. Reutilizar agosto no cuenta como repetir entrenamiento.
3. **Calibración:** inventariar referencias y adquisiciones, distinguir color, geometría e intensidad y definir o ejecutar comprobaciones cuando haya referencia adecuada. Estimación inicial: **uno a tres días**. Sin referencias puede cerrarse la auditoría y el diseño experimental, pero no acreditarse calibración.

La previsión conjunta es de **tres a seis jornadas**, condicionada a datos, referencias y ejecuciones. No es el tiempo necesario para cerrar toda la revisión de Matías ni una garantía de obtener validación clínica.

### Aplazado por decisión expresa

**Revisión de segmentación y corrección de límites dentales.** La selección funciona, pero persisten fronteras que incluyen encía o superficie vecina y anomalías en raíces reconstruidas. Se mantienen las advertencias y el gate de revisión. No se retiran esos avisos ni se presenta el resultado como validado.

### Fuera de este cierre inmediato

Recuperación garantizada de originales; píxel DICOM canónico y multiframe completo; firmas, cifrado y redacción; covarianzas y TRE independiente; transformaciones ampliadas y oclusión validada; perfiles clínicos longitudinales; round-trip DICOM/FHIR; pruebas de otros receptores y diversidad de casos; evaluación clínica y regulatoria específica.

## 8. Publicación y trazabilidad del trabajo

Las correcciones de implementación ya están en commits locales de `main`:

| Repositorio | Commit | Contenido |
|---|---|---|
| Agentic Smart Health | `ded264b` | Fidelidad, procedencia y separación de inferencias. |
| Agentic Smart Health | `99d1886` | Pipeline, apariencia archivada y regeneración de malla. |
| uos-viewer | `fb11d73` | Selección dental y fichas con etiquetas separadas. |
| uos-viewer | `c10508f` | Etiquetas y correspondencia de exportación con Python. |

Estos identificadores documentan el código local inspeccionado; no demuestran despliegue ni publicación remota. Este informe introduce un cambio documental posterior y no crea un commit automáticamente.

Antes de publicar el paper deben revisarse su descripción de la extensión reciente, su resumen y sus conclusiones. También hay pendientes de publicación/documentación: el contenido del esquema modificado no coincide con el tag `uos-spec-v0.3-draft`; la comprobación de inventario detecta que `scripts/restaura_apariencia.py` carece de `RESUMEN_EN`. No se han movido tags ni corregido código como parte de esta redacción.

El PDF de esta respuesta se genera como artefacto local bajo `data/processed/reports/`, ignorado por Git. El Markdown permanece en el repositorio. No se incorporan el PDF de Matías, archivos originales, capturas con información clínica ni geometría de pacientes al historial.

## 9. Fuentes para comprobar esta respuesta

- Revisión de Matías: PDF local leído íntegramente; sección y numeración conservadas en la matriz. No se reproduce su documento ni se incorpora al repositorio.
- [Contrato de fidelidad y procedencia](uos-fidelity-provenance-v1.md): diseño implementado, pruebas y límites.
- [Informe del complete case del 2 de octubre](uos-complete-case-2026-10-02.md): ejecuciones inicial y corregida, validación y pendientes.
- [Decisión de arquitectura](../architecture/formato-uos.md) y [especificación normativa](uos-format-spec-v0.3.tex).
- [White paper](uos-white-paper.tex): correcciones editoriales previas y repaso final aún pendiente.
- [Contrato Python](../../packages/uos/src/uos/fidelidad.py), [auditoría y retirada de inferencias](../../packages/uos/src/uos/auditoria.py), [exportador](../../packages/uos/src/uos/agente.py) y [validador](../../packages/uos/src/uos/validador.py).
- [Ejecutor del caso](../../scripts/caso_completo.py), [recuperación de apariencia](../../scripts/restaura_apariencia.py) y [regeneración de malla](../../scripts/malla_mejorada.py).
- [Plan de implementación](uos-v0.4-plan-de-implementacion.md): hoja de ruta; no equivale a trabajo completamente realizado.
- Visor: repositorio local `uos-viewer`, commits indicados y pruebas del navegador. Evidencia clínica y artefactos pesados permanecen en almacenamiento local ignorado.

La respuesta inicial se redactó el 30 de septiembre y se amplió tras integrar cambios el 1 de octubre. Esta edición sustituye sus estados desactualizados; preserva la distinción entre corrección del texto, implementación, prueba funcional y evidencia clínica.
