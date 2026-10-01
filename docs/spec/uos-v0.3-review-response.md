# UOS v0.3: contraste de la revisión técnica y correcciones del white paper

Fecha: 2026-09-30. Revisión de referencia: Matías Molinas, *UOS v0.3 — Revisión técnica para el autor de la especificación*, 2026-09-25, 32 páginas. Sus referencias corresponden al white paper del 24 de septiembre; esta respuesta examina el árbol de trabajo actual. No presupone que el código inspeccionado fuera idéntico al de aquella fecha.

Se ha corregido el [white paper](uos-white-paper.tex), conservando su estructura y los cambios locales anteriores. Tras la corrección editorial se han aplicado las correcciones conservadoras descritas abajo al software, la especificación y las descripciones del esquema. El contrato v0.3 no incorpora un modelo de fidelidad. El PDF de la revisión y los datos clínicos no se incorporan al repositorio.

## Resultado y criterio de evidencia

**v0.3 no tiene un bloque general `fidelity`, historial de pérdida heredada ni cálculo de `diagnostic_eligible`.** El sidecar volumétrico contiene metadatos parciales; no equivalen a ese modelo. `fit_for` existe para registros, no para todos los assets.

Estados usados: **implementado** (comportamiento localizado y, cuando procede, probado), **parcial**, **ausente**, **discrepancia documental**, **propuesta** (requiere una decisión). Tener un campo no demuestra que el escritor lo rellene, que el validador compruebe su contenido o que exista evidencia clínica. Cada fila distingue esos niveles y el trabajo pendiente.

Las cifras de experimentos se conservan como resultados históricos, no como mediciones repetidas en esta revisión. No se han abierto adquisiciones de pacientes ni vuelto a entrenar modelos. La matriz no certifica la exactitud de las cifras históricas.

### Fuentes locales del contraste

Los identificadores siguientes remiten a archivos y símbolos, para que la evidencia sobreviva a cambios de línea:

| ID | Fuente y ámbito |
|---|---|
| M | [manifiesto.py](../../packages/uos/src/uos/manifiesto.py): `Asset`, `Part`, `digesto_de_partes`, `Projection`, `Frame`, `Registration`, `Manifest`, `Deidentification`, `Acquisition`, `Consent`, `Locator`. |
| S | [Esquema v0.3](../../schemas/uos-manifest-0.3.schema.json), generado desde el contrato. |
| V | [validador.py](../../packages/uos/src/uos/validador.py): `validate`, `_valida_assets`, `_valida_serie`, `_valida_frames`, `_valida_capas_clinicas`, `_valida_phi`, `_perfil_distribuible`, `_valida_locators`, `_valida_gs`. |
| W | [agente.py](../../packages/uos/src/uos/agente.py): exportación UOS; [contenedor.py](../../packages/uos/src/uos/contenedor.py): escritor ZIP y construcción de assets. |
| VOL | [volumen.py](../../packages/uos/src/uos/volumen.py): `describe_series`, `_codificacion`, `identificables_en`. |
| G | [marcos.py](../../packages/uos/src/uos/marcos.py): selección de camino, inversión/composición y discrepancia. |
| C | [clinico.py](../../packages/uos/src/uos/clinico.py): `clinical_layer`, color regional, hallazgos y procedencia por valor. |
| I | [report_agent.py](../../packages/ingestion-agents/src/ingestion_agents/report_agent.py): `_derivation`, `_ingest`, backends de reglas/LLM y fallback OCR. |
| P | [procedencia.py](../../packages/uos/src/uos/procedencia.py): cadena y firmas todavía no verificadas. |
| E | [malla_compuesta.py](../../packages/export-agents/src/export_agents/malla_compuesta.py): cierre, piezas y `_cabecera_pieza`; [visor.py](../../packages/export-agents/src/export_agents/visor.py): exportación PLY/JSON para visor externo. |
| A | [apariencia.py](../../packages/gaussian-engine/src/gaussian_engine/apariencia.py): `_comentarios_color`; [pose_foto.py](../../packages/gaussian-engine/src/gaussian_engine/pose_foto.py): PnP y proyección. |
| T | [Pruebas UOS](../../packages/uos/tests/), especialmente `test_uos.py`, `test_volumen.py`, `test_clinico.py` y `test_procedencia.py`. |
| F | [Fixtures y resultados esperados](../../fixtures/uos-0.3/expected.json). |
| H | [Cierre del MVP](../cierre-mvp.md): procedencia documental de métricas históricas; no sustituye los artefactos de ejecución. |
| N | [Especificación normativa](uos-format-spec-v0.3.tex): contraste adicional; actualizado en la fase de implementación. |

## 1. Fidelidad y pérdidas

| Observación de la revisión | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 1.1 Calidad diagnóstica y pérdida previa | **Ausente.** M/S no modelan fidelidad general ni elegibilidad diagnóstica. V comprueba estructura y coherencia, no rendimiento clínico. | Se declara la ausencia y se retiran garantías clínicas derivadas de hashes, PSNR o capas. Definir criterios por tarea antes de atribuir aptitud diagnóstica a un validador. |
| 1.2 Historial DICOM, Transfer Syntax y herencia | **Parcial.** VOL consulta Transfer Syntax para describir codificación; no eleva los atributos de compresión irreversible ni su historial a un bloque de fidelidad. No hay herencia de pérdida por linaje. | Se separa metadato presente de capacidad pendiente. La futura política debe representar también historia desconocida; falta de declaración no significa ausencia de pérdida. |
| 1.3 JPEG y color medido | **Parcial.** C declara CIELAB por tercios, foto fuente, píxeles y corrección de iluminación; no caracteriza pérdida JPEG ni calibración colorimétrica. A distingue varias fuentes de color. | Se habla de estimaciones regionales sobre fotografías procesadas. No se afirma que todo JPEG sea 4:2:0 sin inspeccionar su codificación. RAW/TIFF, referencias de color y captura controlada son propuestas. No adoptar «EXIF completo» sin filtrar identificadores. |
| 1.4 STL como original | **Discrepancia documental.** El caso utiliza STL; ello no demuestra que el escáner carezca de color. M incluye fabricante/modelo/software, pero no una cadena general de exportación nativo→STL. | Se delimita el original como archivo recibido y la ida y vuelta como preservación numérica, sin atribuirla a precisión clínica. El nativo, color y cadena de exportación requieren trabajo específico. |
| 1.5 Resolución y calibración | **Parcial.** VOL emite dimensiones, spacing, orientación, rescale y `calibrated_hu`. Ahora se emite `calibrated_hu: false` y `calibration_status: unknown`; ya no se acredita calibración mediante `Modality == "CT"`. | Se explica la limitación. No convertir automáticamente paso de muestreo en resolución efectiva validada ni tratar la etiqueta de modalidad como prueba de HU. |

## 2. Identificación, recuperación y verificación

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 2.1 Originales externos y archivo | **Parcial.** M tiene `locators` con `dicomweb`, `ae_title`, URL y referencia opaca. W deja los originales fuera. V avisa de externos y no resuelve localizadores. T/F prueban que el validador acepta ciertas series incluidas. | Se condiciona la preservación al repositorio externo. No se afirma que no exista localización ni que ya exista recuperación garantizada. Se ha fijado la distinción: el formato admite incluidos y externos; el escritor de referencia externaliza los adquiridos. No se introduce un perfil `archival`. |
| 2.2 Tres identidades | **Parcial.** M contiene UID de estudio/serie, SOP Instance UID, hash del fichero y hash del valor almacenado de `PixelData`. V distingue identidad declarada y bytes en series incluidas. No es un hash canónico del píxel decodificado más geometría/escala. | Se describe exactamente el alcance. No se da por hecho que la clínica conserve o no la copia pseudonimizada: no se inspeccionó su custodia. Quedan canonicalización, semántica geométrica y tratamiento del remapeo de UIDs. |
| 2.3 Dependencia del nombre | **Discrepancia documental.** `digesto_de_partes` usa UID+PixelData si todos los cortes los declaran; recurre a nombre+hash de archivo en caso contrario. T verifica invariancia al renombrar en la rama UID. | Se sustituye la fórmula única del apéndice por las dos ramas. El nombre no determina orden anatómico. Matiz a la revisión: editar cabeceras sí cambia el hash del archivo completo y, por tanto, el digest de la rama basada en bytes. |
| 2.4 Multiframe | **Ausente para la propuesta.** `Part` inventaría archivos. VOL calcula dimensiones de serie a partir de archivos/cabeceras; no se encontró un contrato canónico por frame multiframe. | Se limita «corte por corte» al caso por archivo y se declara pendiente el soporte general; no se extrapola la prueba a Enhanced CT. |

## 3. Seguridad y ciclo de vida

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 3.1 Cadena y firmas | **Parcial.** P/V comprueban coherencia de versiones y advierten sobre firmas no soportadas. `verified_by` es una declaración, no una firma verificada. | Se retiran autenticidad, no repudio y prueba de ausencia de edición adversaria. Custodia de claves, formato, roles y sellado temporal requieren diseño. No elegir COSE/JAdES/CAdES en esta revisión editorial. |
| 3.2 Cifrado | **Ausente como perfil UOS completo.** M/S no tienen un contrato de entradas cifradas; W escribe ZIP STORE. | Se declara pendiente. Elegir sobre completo o cifrado por entrada exige resolver acceso, claves y validación; no equivale a agregar un campo de algoritmo. |
| 3.3 Supresión/redacción | **Ausente.** P implementa continuidad lógica, no tombstones firmados ni borrado verificable en todas las copias. | Se limita la garantía de append-only. La propuesta requiere política de retención, alcance entre versiones y copias, y revisión aplicable al uso; no asumir que conservar un hash resuelve todas las obligaciones. |
| 3.4 Desidentificación | **Parcial.** M/V ya contienen perfiles/opciones, assets afectados, estado PHI, consentimiento y controles sobre representaciones faciales y localizadores. Eso valida declaraciones, no inspecciona exhaustivamente todo píxel o anatomía. | Se corrige «distributable» para que no parezca autorización o anonimización certificada. No asumir ni ausencia universal de riesgo ni imposibilidad absoluta de anonimizar cualquier dato dental. |

## 4. Geometría e incertidumbre

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 4.1 Residuo y TRE | **Parcial.** M ya distingue RMS, máximo residual, TRE opcional, región y uso `fit_for`. No hay TRE independiente demostrado para el caso; tampoco contrato completo de covarianza/soporte/sensibilidad. | Se separan objetivo punto-a-plano, residual punto-a-punto y error en objetivo. El máximo sobre correspondencias tampoco es un límite de error anatómico. |
| 4.2 Propagación y ciclos | **Parcial.** G elige camino corto y desempata por IDs; V avisa si las matrices por caminos discrepan. La discrepancia es el máximo de diferencias de coeficientes, mezclando componentes rotacionales y traslacionales, no TRE en mm. No propaga covarianzas. | Se evita afirmar que menos saltos garantizan más precisión o que el aviso acota error clínico. La fórmula propuesta requiere convenciones de perturbación y tratamiento de correlaciones; no incorporarla sin fijarlos. |
| 4.3 Otras transformaciones y cámara | **Parcial.** `Registration` documenta una matriz rígida; `Projection` clasifica imágenes, sin intrínsecos ni distorsión. A sí tiene PnP/proyección en la implementación, aunque no sea un contrato general UOS de cámara. | Se corrige que «el pipeline no calcula pose». Similitud, deformable y cámara son cambios de contrato pendientes. Evitar una deformación que absorba el cambio biológico que se quiere medir. |
| 4.4 Mandíbula y oclusión | **Parcial.** M tiene frames múltiples, `occlusion` y un ID reservado `reg.mandible_to_maxilla`; T prueba declaraciones de mordida. No hay evidencia de un flujo completo multioclusión/tracking en el caso maxilar. | No presentar oclusión como totalmente ausente ni como resuelta. Quedan relaciones tipadas, fechadas, varias posiciones y validación bilateral. |
| 4.5 Unidades y orientación | **Parcial.** M/V ya contemplan mm, handedness, LPS/RAS y Frame of Reference UID. G implementa dirección e inversión de matrices; T las prueba. | Separar convención implementada de un round-trip DICOM no demostrado. La procedencia de unidades del STL y fixtures con objetos DICOM reales siguen pendientes. |

## 5. Coherencia entre procedencia y uso

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 5.1 Reproducibilidad de capa 2 | **Discrepancia conceptual.** M define la capa 2 como cálculo determinista/reproducible; el paper describe variabilidad en ajuste por caso. No existen capas `2a`/`2b` ni una clase general de reproducibilidad. | Se explicita la tensión. Separar procedencia, modelo preentrenado y reproducibilidad es una propuesta; no cambiar enums ni anunciar la división como implementada. |
| 5.2 Extracción de informes | **Corregido en exportación.** I distingue reglas/LLM y marca procedencia OCR. W separa cada observación inferida en `derived/`, capa 3, con modelo y fuentes. Una fuente no vinculable se declara no resuelta, avisa y exige revisión; no se inventa. | La reproducción original confirmó capa 1 + rechazo. La regresión ahora exporta y valida dos inferencias del mismo diente conservando ambos modelos; el color sigue en capa 2. No afirmar que toda extracción usa LLM ni que el 93,8% mide exactitud de hallazgos. La promoción por firma requiere conservar la procedencia original. |
| 5.3 Caras sintéticas | **Corregido en exportación.** E persiste un mapa por intervalos de caras, ligado por SHA-256 al STL, que distingue escáner y cierre sintético. | Se declara geometría añadida y se retira la equivalencia entre estanco y validado para fabricación. El mapa reside en un JSON acompañante, listado en `ExportOutput.sidecars`; debe conservarse junto al STL. |
| 5.4 Corona y raíz | **Parcial.** E describe origen y error en cabecera STL y ahora identifica las caras de corona y raíz en el JSON acompañante. La arcada por defecto no lleva raíces. La impresión física no conserva ese mapa. | Se presentan raíces como experimentales y se conservan las anomalías. No atribuir al export externo una separación interna `derived/` que no se haya probado. |
| 5.5 Color por vértice | **Discrepancia documental.** C/A generan resúmenes por tercio y asignaciones a vértices; A distingue color regional, proyectado, interpolado y respaldo. | Se separan resolución de medida y almacenamiento. Los 108.922 regionales de 112.067 dejan 3.145 restantes. El borrador sumaba solo 112.060; un comentario de prueba menciona siete proyectados, pero no se usa como prueba del run clínico. Se retira el desglose exacto pendiente de verificar el artefacto. |
| 5.6 Etiquetas FDI como beneficio | **Discrepancia de alcance.** Hay labels y selección por índice, pero el propio resultado histórico rechaza límites en 11/14 coronas. | Se explicita el carácter experimental desde el resumen y la descripción del receptor. Exportar una etiqueta no valida el límite que representa. |
| 5.7 SaMD | **Discrepancia documental.** Los códigos de capa no deciden la calificación de un módulo. | Se sustituye la exención general por finalidad prevista; véase §10.1 y fuente oficial. |

## 6. Gaussianas y evaluación

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 6.1 Diagnóstico, PSNR, HU | **No demostrado.** Los scores volumétricos son en proyección; el caso tiene submuestreo. VOL no acredita calibración de HU. | Se distingue calidad de representación y validación diagnóstica. No deducir resolución efectiva de un spacing supuesto ni asignar HU por modalidad. |
| 6.2 Métricas futuras | **Propuesta.** No se demuestra aquí evaluación por tarea con observadores o fantoma. | Se requiere evidencia por tarea antes de ampliar el uso declarado. MAE, MTF o contraste pueden aportar evidencia, pero ninguno certifica por sí solo toda aptitud diagnóstica. |
| 6.3 Fitness y prohibición de medir | **Ausente como regla general.** `fit_for` está en registros, no en todos los assets; V no prohíbe universalmente una medición por proceder de Gaussianas. | Se limita el uso demostrado de las capas actuales. La política general por representación/tarea y el recurso al original son cambios futuros, no capacidades existentes. |
| 6.4 Transporte web | **Discrepancia documental.** DICOM PS3.18 ya define servicios web. La existencia de splatting no demuestra una ventaja frente a MPR del original. | Se corrige la comparación con DICOM y no se presenta UOS como sustituto validado del volumen original. |
| 6.5 Peso de la investigación | **Propuesta editorial.** UOS-Core admite escena de malla sin splats; T verifica niveles. | Se declara al inicio de la sección experimental. Se mantiene un documento por el alcance acordado; separar el informe técnico queda para una decisión posterior. |

## 7. Suficiencia clínica

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 7.1 Radiografía 2D | **Parcial.** M incluye `IMAGE2D`, `Projection.type` y `fdi_targets`; no prueba ingestión y registro completo de todas las modalidades radiográficas. | El título pasa a composición de un caso multimodal, evitando prometer toda la historia dental. Quedan conectores y geometrías por modalidad. |
| 7.2 Contenido estructurado | **Parcial.** C conserva hallazgos por FDI y medidas. Hay extensión clínica, pero no un odontograma/periodontograma interoperable completo ni todos los antecedentes, prótesis y tratamientos. | Se separa extensibilidad de interoperabilidad. No inventar perfiles clínicos ni copiar datos del sistema de gestión en esta corrección. |
| 7.3 Objeto dental longitudinal | **Ausente como modelo completo.** La clave regional es FDI; no hay entidad longitudinal general diente/implante/corona independiente de posición. | Queda pendiente modelar identidad y eventos sin confundir posición con objeto. No prometer seguimiento clínico solo por tener visitas y frames. |
| 7.4 Acto clínico | **Parcial.** M tiene fecha y equipo, operador de registro y consentimiento. No representa de forma completa protocolo y autoría clínica autenticada. | Se conserva como limitación; no afirmar que todos los elementos temporales o del equipo estén ausentes. |
| 7.5 Diversidad de casos | **Limitación de evidencia.** H y el paper no demuestran cobertura clínica de edéntulos, dentición mixta, mandíbula, metal o movimiento. | Se conserva N y la limitación del caso maxilar; una arquitectura extensible no sustituye validación en esas situaciones. |

## 8. Contenedor y transporte

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 8.1 STORE y rangos | **Implementado como restricción, ventaja no medida.** W escribe STORE; V rechaza entradas comprimidas. No se encontró un benchmark de acceso interno por rangos en este repositorio. | Se corrige que STORE sea necesario para pedir una entrada completa. DEFLATE por entrada es propuesta de contrato. Matiz: marcar un códec como opcional no lo hace legible sin soporte; necesita un fallback real. |
| 8.2 Compresión glTF | **Parcial.** V revisa estructura Gaussian/glTF, sin una política completa de pérdida por asset `measured`. | No afirmar que el validador ya proteja fidelidad geométrica/colorimétrica. La política debe atender a parámetros y transformaciones reales, no solo al nombre de extensión. |
| 8.3 ZIP64, streaming y media type | **Parcial/propuesta.** W utiliza `zipfile`; eso no demuestra interoperabilidad ZIP64 con lectores externos. El manifiesto primero se comprueba. No se identifica una prueba >4 GiB ni lector web por rangos en el repo. | Se conserva el MIME como propuesto y no registrado. Gobernanza neutral, límites y prueba ZIP64 quedan pendientes. No inferir comportamiento de todos los lectores desde la biblioteca Python. |

## 9. Estándares e interoperabilidad

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 9.1 FHIR R4/R5 | **Discrepancia documental.** W/M utilizan indicaciones de tipo; no hay conector bidireccional validado. | Se distingue `Media` R4 de `DocumentReference` R5. `ImagingSelection` queda como destino candidato sujeto a restricciones y referencias de imagen, no como equivalencia automática por FDI. |
| 9.2 Capacidades DICOM | **Discrepancia documental.** La motivación infravaloraba objetos existentes y transporte web. | Se reconoce el alcance de DICOM. Matiz a la revisión: la existencia de una SOP Class no demuestra que cualquier PACS almacene, versione o visualice todos esos objetos; requiere conformidad del producto receptor. |
| 9.3 Round-trip | **Propuesta.** Los campos y destinos documentados no equivalen a importación/exportación semánticamente reversible. | Se retira «field for field» como garantía y se deja pendiente una prueba real contra otro sistema. |
| 9.4 Terminología y texto original | **Parcial.** C emite `{system, code:null, display}`. No es un hallazgo ya codificado en SNOMED; no consta un `coding_status` ni localización por página/posición general en esa salida. | Se corrige «coded». Quedan codificación supervisada y anclaje al documento sin inventar identificadores. |

## 10. Alcance regulatorio y legal

Esta intervención corrige afirmaciones del white paper; no determina la clasificación del producto ni verifica cumplimiento por jurisdicción. No se incorporan conclusiones de la revisión sobre clases FDA, plazos legales o aplicabilidad de leyes nacionales sin una evaluación específica.

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 10.1 Capas 1/2 no SaMD | **Corrección aceptada.** La guía europea MDCG 2019-11 rev.1 vincula la calificación a la finalidad prevista. | Se retira la exención basada en ausencia de modelo y la autorización implícita al borrar `derived/`. La evaluación corresponde a cada módulo y uso. |
| 10.2 Ley de IA | **Propuesta de análisis.** Procedencia no prueba cumplimiento de todas las obligaciones que pudieran aplicar. | No se añade una afirmación de cumplimiento ni se equipara registro de pesos a logging completo. Evaluar alcance, fechas y obligaciones antes de usarlo como argumento regulatorio. |
| 10.3 EHDS | **Propuesta.** No se identificó un mapeo implementado que demuestre conformidad con sus especificaciones. | No se anuncia compatibilidad. Debe contrastarse con requisitos aplicables al producto y su calendario. |
| 10.4 Historia clínica | **Parcial.** Consentimiento/finalidad tienen contrato y checks; firma, acceso, retención y supresión no constituyen un ciclo legal completo. | Se elimina la equivalencia entre perfil de metadatos y permiso de distribución. No presentar UOS como historia clínica legal completa. |
| 10.5 Export al paciente | **Propuesta.** No existe el perfil descrito con originales, visor y resumen garantizados. | Mantenerlo como posible evolución y no como entrega actual ni garantía universal de portabilidad jurídica. |

## 11. Valor y estructura

| Observación | Contraste y estado | Corrección editorial y pendiente |
|---|---|---|
| 11.1 Destinatarios | **Corrección aceptada.** Los experimentos no sostienen todas las promesas al laboratorio o al seguimiento. | Resumen y recorrido indican condiciones de acceso, raíces/labels experimentales y límites de medición desde la primera mención. |
| 11.2 Diferenciación | **Parcial.** Procedencia, validación y composición son capacidades concretas; no se hizo una revisión exhaustiva de todas las alternativas del mercado. | Se eliminan afirmaciones universales de exclusividad y «no se puede comprar a ningún precio». |
| 11.3 Dos documentos | **Propuesta aplazada.** Separación útil, pero fuera de esta intervención acordada. | Se conserva la estructura y se explica que la sección Gaussiana es investigación opcional, no requisito UOS-Core. |
| 11.4 Qué preservar | **Aceptado.** Resultados negativos, N, códigos sin inventar y separación de planos siguen siendo parte de la propuesta. | Se mantienen con su alcance. Una segunda implementación demuestra interoperabilidad, pero no convierte por sí sola una propuesta en estándar formal. |

## 12. JSON y reglas propuestos por Matías

Los ejemplos de la revisión son propuestas, no schemas válidos de la v0.3 actual. No se copian al white paper como si estuvieran implementados.

| Propuesta | Correspondencia y decisión pendiente |
|---|---|
| 12.1 Identidad, recuperación y `fidelity` | M ya tiene parte de identidad y `locators`; faltan canonicalización y fidelidad general. Un futuro validador puede comprobar un perfil técnico, pero no certificar aptitud clínica general con un booleano sin tarea, evidencia ni estado desconocido. |
| 12.2 Derivado y pérdida heredada | Existe `derived_from`; faltan historial de pérdidas y clase de reproducibilidad. `layer: "2b"` no pertenece al contrato actual. No convertir automáticamente submuestreo en resolución efectiva medida. |
| 12.3 Incertidumbre completa | Existen residual, máximo, TRE/región y revisión declarada. Faltan covarianza, soporte y sensibilidad estructurados. El ejemplo llama `residual_all` a una métrica `over: inliers`: debe aclararse antes de implementarlo. |
| 12.4 Reglas UOS-F | No están implementadas como familia de fidelidad. Requieren contrato de fuentes, pérdidas, datos desconocidos y usos; extraer metadatos no sustituye validar clínicamente. |
| 12.4 Reglas UOS-I | Hay checks de hashes y UID/PixelData almacenado, pero no los nuevos hashes canónicos. I-005, tal como está redactada, chocaría con un asset referenciado legítimo sin payload: distinguir externo, ausente y retirado. |
| 12.4 Reglas UOS-R | Hay estado provisional y aviso por caminos distintos. No hay verificación criptográfica de `verified_by` ni covarianza propagada. |
| 12.4 Reglas UOS-L | Existe rechazo de `inferred` en capa 1 y de capa 3 en `clinical/`; la exportación separada está corregida y probada. L-002 no debe clasificar automáticamente como salida de modelo toda extracción determinista. |
| 12.4 Regla UOS-G | Se añade y prueba un mapa de intervalos de caras vinculado al hash del STL; no equivale a validación anatómica ni viaja con la impresión física. |
| 12.5 Lector | El repositorio entrega metadatos y exports; no demuestra los cuatro comportamientos obligatorios propuestos en un lector independiente. Especificación de lector y pruebas de interfaz quedan pendientes. |

## 13. Prioridades y respuestas a preguntas abiertas

### 13.1 Orden de trabajo resultante

Se han aplicado correcciones de exportación inferida, calibración no acreditada, validación estricta de bytes DICOM, lenguaje del chequeo de matrices, política de originales y procedencia por caras STL. La identidad semántica sigue pendiente: se rechazan los bytes distintos sin atribuirles equivalencia clínica. Diseñar después fidelidad, reproducibilidad y recuperación como contratos versionados, con su validador y pruebas. Firmas, transformación no rígida, perfiles clínicos y conectores necesitan decisiones propias; no se dan por aprobados por figurar en la revisión.

No se aceptan las estimaciones de esfuerzo de Matías como mediciones del repositorio. Por ejemplo, canonicalizar imágenes DICOM y preservar compatibilidad no queda demostrado como trabajo «bajo».

### 13.2 Respuestas contrastadas

| Pregunta | Respuesta sustentada hoy |
|---|---|
| 1. ¿Existe `fidelity` equivalente? | No. VOL aporta metadatos parciales; M/S carecen del modelo general y V no calcula elegibilidad diagnóstica. |
| 2. ¿Qué copia se hashea? | El escritor calcula hashes sobre los archivos que recibe. No se inspeccionó el repositorio clínico ni se puede afirmar qué copias conserva la clínica. UID/PixelData ayudan parcialmente, sin hash canónico de geometría. |
| 3. ¿LLM/OCR? | I tiene reglas por defecto, backend LLM y fallback OCR; la procedencia distingue esas rutas. W separa ahora la inferencia en capa 3; §5.2 describe la regresión y la limitación de fuentes no resueltas. |
| 4. ¿Se usa rango interno STORE? | Se implementa STORE y su validación. No se encontró benchmark ni lector HTTP por rangos en este repositorio que sostenga la ventaja anunciada. |
| 5. ¿TRE independiente? | Existe el campo opcional, pero no se ha localizado evidencia de TRE independiente para el caso de referencia. El 0,666 mm sigue siendo residual de ajuste. |
| 6. ¿Mandíbula/oclusión? | Hay frames, enum de oclusión e ID reservado de registro. No equivalen a un flujo clínico bilateral y multioclusión validado. |
| 7. ¿Organización neutral? | Decisión de gobernanza pendiente; el tipo propuesto conserva `histora`. No se ha cambiado ni registrado un MIME. |
| 8. ¿Contacto WG-22/ADA? | No consta evidencia en los archivos examinados. Requiere información del equipo; no se han enviado mensajes externos. |
| 9. ¿Entregas reales al laboratorio? | Hay exports y resultados históricos; eso no acredita entregas, pilotos ni advertencias realmente recibidas. No se inspeccionaron registros de distribución. |
| 10. ¿Umbral de cambio gingival? | No se ha localizado un criterio clínicamente validado de reportabilidad con incertidumbre del registro longitudinal. |

## Hallazgos adicionales y validación

- El documento normativo, comentarios y pruebas contienen afirmaciones históricas que pueden discrepar entre sí. La fase editorial registró el conflicto; la fase de implementación ahora corrige explícitamente la especificación en los seis puntos acordados, dejando las capacidades nuevas pendientes.
- Tras integrar los cambios remotos y corregir la expectativa de desidentificación, F enumera 13 fixtures: 2 `valid`, 9 `error`, 1 `warning` y 1 `valid-with-warning`. El índice declara ahora `version: "0.3"`. La serie con cabecera reescrita y hashes sin actualizar se rechaza por integridad, sin afirmar que pertenezca a otro paciente.
- Se eliminaron «22 checks» como recuento ambiguo de checks/subchecks y la igualdad entre éxito del validador y aptitud clínica.
- La copia pública de `KHR_gaussian_splatting` consultada durante esta revisión declara estado ratificado. El white paper ya no lo llama Release Candidate ni promete una migración automática. Sigue pendiente verificar compatibilidad de la implementación con la revisión final.
- Se corrigieron dos problemas matemáticos del apéndice: mezclar el umbral de opacidad y el techo de atenuación en un único intervalo; atribuir el clamp 0,9999 al límite de `float32`. También se distingue promedio local de color de composición visible con transmitancia, y conectividad del grafo de independencia respecto al camino.
- Se separan los scores de apariencia sobre vistas de entrenamiento de los scores volumétricos sobre vistas retenidas. La aditividad de una partición ideal no prueba fidelidad de capas ajustadas o calibradas por separado.
- **Pruebas en la fase editorial inicial:** `.venv/bin/python -m pytest packages/uos/tests -q` → **140 passed**. Esto verificó la suite anterior a los cambios de implementación, no los resultados clínicos del white paper.
- **Reproducción sintética inicial, antes de corregir:** un valor `ph` con `Derivation.INFERRED` y modelo declarado pasaba por `clinical_layer` como capa 1; `_valida_capas_clinicas` producía `UOS-E-017d`. Se usó ZIP en memoria, sin datos clínicos y sin añadir pruebas que fijen el defecto como conducta deseada.
- **Comprobación documental:** `docs_sync.py --check` pasa antes y después de editar. Usa archivos versionados como inventario; los enlaces locales de esta matriz nueva se comprobaron además explícitamente.
- **Compilación final:** `latexmk -pdf -interaction=nonstopmode -halt-on-error -cd docs/spec/uos-white-paper.tex` termina correctamente. Índice recuperado y limitado a secciones; sin referencias sin resolver, etiquetas duplicadas, avisos LaTeX ni desbordamientos. PDF inspeccionado visualmente.
- **Higiene:** `git diff --check` y `data_guard.py --quiet` pasan. El PDF generado se mantiene como artefacto local, sin añadirlo al historial. Los cambios de agentes y contrato se documentan en la fase de implementación.

## Fuentes externas comprobadas

Se consultaron fuentes primarias para las correcciones de estándares y finalidad prevista, sin convertir sus requisitos en afirmaciones de cumplimiento de UOS:

- [DICOM PS3.18: servicios web](https://dicom.nema.org/medical/dicom/current/output/html/part18.html).
- [DICOM PS3.3 A.85: modelos 3D encapsulados](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.85.html).
- [FHIR R4 Media](https://hl7.org/fhir/R4/media.html) y [FHIR R5 DocumentReference](https://hl7.org/fhir/R5/documentreference.html).
- [MDCG 2019-11 rev.1, publicación de la Comisión Europea](https://health.ec.europa.eu/latest-updates/update-mdcg-2019-11-rev1-qualification-and-classification-software-regulation-eu-2017745-and-2025-06-17_en): finalidad prevista y calificación del software.
- [KHR_gaussian_splatting, especificación de Khronos](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting): estado publicado consultado en esta revisión.

## Comprobación posterior a las correcciones de implementación

Se añaden regresiones sintéticas de inferencias, calibración, alteraciones DICOM y mapas STL. Los hashes UID/PixelData de v0.3 no cambian: cualquier fichero DICOM incluido cuyo hash completo no coincide ahora falla con `UOS-E-007`, incluso si conserva UID y PixelData. El cambio evita aceptar modificaciones de geometría o rescale como supuesta desidentificación. Los originales externos no se descargan ni verifican.

Las correcciones de código no implementan `fidelity`, firmas, elegibilidad diagnóstica, recuperación de originales ni propagación de covarianzas. La justificación y los límites constan en [la decisión de arquitectura](../architecture/formato-uos.md).

Validación anterior a integrar los cambios remotos: **303 pruebas correctas** en `packages/uos/tests` y `packages/export-agents/tests`; Ruff y MyPy sin errores en los módulos afectados, y controles `docs_sync`, `data_guard` y `git diff --check` correctos. El PDF del white paper se ha regenerado (28 páginas). Estas pruebas son sintéticas, no una validación clínica.


### Integración con los cambios remotos — 2026-10-01

Se conservaron la política de publicación con etiquetas separadas de borrador y versión
publicada, las actualizaciones de versión y las nuevas comprobaciones del remoto. Se
actualizaron el generador, el índice y las pruebas del banco de conformidad para que una
cabecera DICOM reescrita con hashes antiguos produzca `UOS-E-007`.

La comprobación conjunta de UOS, exportación y conformidad pasó **307 pruebas**. Ruff y
MyPy pasaron en los módulos comprobados. El control documental sigue detectando un
pendiente de publicación: el esquema modificado difiere del contenido servido por
`uos-spec-v0.3-draft`. La etiqueta no se ha movido durante la resolución del rebase.
