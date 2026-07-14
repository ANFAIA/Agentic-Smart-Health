# Current progress of digital twin construction using medical imaging

> **Fuente:** Current progress of digital twin construction using medical imaging.pdf (Journal of Applied Clinical Medical Physics, 2025;26(9), DOI: 10.1002/acm2.70226)

## Abstract

Esta revisión sistemática (protocolo PRISMA) examina el papel de la imagen médica en la construcción de gemelos digitales (digital twins) del cuerpo humano, entendidos como representaciones virtuales dinámicas de un paciente que se actualizan con datos reales y modelos computacionales sofisticados. El trabajo adopta un marco de clasificación por sistemas anatómicos —cardiovascular, sistema nervioso central, musculoesquelético, respiratorio, mama, hígado y digestivo— para analizar cómo modalidades como MRI, CT, PET y ultrasonido, combinadas con modelos como CFD, FEA y arquitecturas de aprendizaje profundo, mejoran el diagnóstico, la planificación terapéutica personalizada y la analítica predictiva. Los hallazgos indican que los avances en imagen han incrementado la precisión diagnóstica y terapéutica de los gemelos digitales más allá de los métodos tradicionales. Sin embargo, persisten barreras técnicas clave: cuellos de botella computacionales, escasez y baja publicidad de datos, problemas de validación/exactitud y dificultades de integración multimodal. La revisión perfila líneas futuras para desbloquear el potencial de esta tecnología en la medicina de precisión.

## Explicación completa

## Problema y motivación

Un gemelo digital (digital twin) es una representación virtual y dinámica de una entidad física —en este caso, la anatomía y la fisiología de un paciente— construida a partir de datos en tiempo real y modelos computacionales avanzados. A diferencia de una simple visualización, el gemelo digital permite actualizarse con mediciones longitudinales y datos clínicos, capturando la trayectoria de la enfermedad a lo largo del tiempo. Esto habilita a los profesionales para pronosticar la progresión de patologías, predecir respuestas a intervenciones (p. ej. fármacos) y anticipar eventos adversos.

El artículo parte de la premisa de que la imagen médica (MRI, CT, PET, ultrasonido, y variantes como CTA, DUS, Cone Beam CT, Micro-CT, DTI, fMRI, OCT, elastografía) es el componente clave para dotar de detalle anatómico y funcional al gemelo digital. La evolución de la tecnología ha llegado a un punto de inflexión gracias al aumento de la potencia computacional, la computación de alto rendimiento (HPC), las plataformas en la nube y, sobre todo, el desarrollo de modelos de aprendizaje profundo (incluidos los modelos fundacionales), que superan limitaciones de los modelos analíticos o físicos tradicionales.

## Metodología del estudio

Se trata de una revisión sistemática realizada según el protocolo PRISMA (Page et al., 2021). Los autores buscaron artículos revisados por pares en Scopus, PubMed, Web of Science y Google Scholar, combinando términos como "Digital Twin" y "Digital Replica" con modalidades de imagen (MRI, CT, etc.). La búsqueda arrojó 117 registros en Scopus, 115 en PubMed, 17 en Web of Science y más de 800 en Google Scholar (de los que 80 se consideraron altamente relevantes), depurados posteriormente para eliminar duplicados.

El aporte metodológico novedoso es la organización de la literatura mediante un **marco de clasificación por sistemas del cuerpo humano** (representado en su Figura 4), donde cada dominio médico se subdivide en escenarios de aplicación: mejora diagnóstica, planificación de tratamiento personalizado, procesamiento de imagen, simulación, modelado cognitivo, etc. Este enfoque "sistema-específico" del desarrollo de gemelos digitales humanos basados en imagen es una perspectiva poco abordada en la literatura previa. Las técnicas computacionales revisadas incluyen algoritmos de procesamiento de imagen para extracción de características, análisis de series temporales para patrones de salud y metodologías de simulación como el análisis de elementos finitos (FEA) y la dinámica de fluidos computacional (CFD).

## Resultados por sistema anatómico

**Sistema cardiovascular:** Predominan MRI y CT como fuentes primarias (con CTA, DUS y rayos X como complemento), aunque la mayoría de datasets no son públicos. Hay fuerte dependencia de CFD y FEA para simular la hemodinámica, con un desplazamiento creciente hacia enfoques basados en datos. Ejemplos destacados: Loewe et al. (2022) construyeron un modelo cardíaco personalizado con CT/MRI para simular actividad eléctrica y evaluar el riesgo de muerte súbita cardíaca (SCD), con una razón de riesgo superior a las métricas clínicas tradicionales; Banerjee et al. (2021) generaron modelos 3D del corazón desde cine-MRI 2D para tratamiento de arritmias; Ložek et al. (2024) modelaron la disincronía del ventrículo derecho post-reparación de Tetralogía de Fallot para personalizar la CRT; Liang et al. (2022) combinaron modelos virtuales con impresión 3D para simular cirugía de DORV, reduciendo diagnósticos inciertos o erróneos del 42,5% al 4,6%; y Pires et al. (2024) crearon gemelos digitales 3D de corazones fetales a partir de ultrasonido obstétrico para planificación prequirúrgica de cardiopatías congénitas (p. ej. anomalía de Ebstein).

**Sistema nervioso central:** De 12 estudios, la MRI es la modalidad principal (a menudo con DTI y fMRI). Existen datasets abundantes para Alzheimer, tumores y esclerosis múltiple. Los modelos van desde K-means hasta NeuralODEs y redes neuronales de picos (Spiking Neural Networks). Aplicaciones: Sarris et al. (2023) usaron clustering K-means para detección de tumores cerebrales sobre un modelo 3D del cerebro; Cen et al. (2023) predijeron atrofia talámica en EM hasta 5-6 años antes de los síntomas mediante modelos spline de envejecimiento normal; Hu et al. (2021) combinaron sistemas difusos con HPU-Net para segmentación de MRI (DSC de 0,936, Jaccard 0,845). En modelado cognitivo, Lu et al. (2022, 2023) desarrollaron un Digital Twin Brain (DTB) capaz de simular estados cerebrales, estimulación cerebral profunda virtual y tareas cognitivas, con capacidad para simular del orden de 86.000 millones de neuronas.

**Sistema musculoesquelético:** Predomina CT (incluyendo Cone Beam CT y Micro-CT) por su capacidad para captar anatomía detallada de huesos y articulaciones. Los estudios dentales favorecen modelos tradicionales (FEA), las extremidades usan enfoques estadísticos y 3D mixtos, y los estudios de columna aplican cada vez más redes neuronales y modelos generativos. Kim et al. (2023) crearon visualizaciones 3D interactivas de la mecánica del hombro convirtiendo CT a formato STL, modelando en Blender e integrando en Unity para simular el movimiento. Las áreas destacadas son visualización/simulación, precisión en intervenciones quirúrgicas, emparejamiento de aloinjertos (allograft matching) y rehabilitación/monitoreo en tiempo real.

**Sistema respiratorio:** El marco Lung-DT integra sensores IoT y algoritmos de IA para monitoreo continuo de la salud respiratoria; la red YOLOv8 clasifica radiografías de tórax con una precisión media del 96,8%. Otro trabajo emplea un modelo de pulmón 3D con reconstrucción de imagen por aprendizaje profundo, mejorando la calidad de imagen frente a la EIT tradicional (SSIM de 0,737 en condiciones de ruido). Tai et al. (2022) aplicaron gemelos digitales a la simulación de cirugía remota de cáncer de pulmón con retroalimentación háptica en tiempo real (modelo DL-CNN).

**Mama:** Se centran en cáncer de mama, incluyendo subtipos como el TNBC, usando principalmente MRI y ultrasonido (elastografía). La mayoría de datasets no son públicos. Hay una tendencia a integrar registros clínicos y muestras biológicas; los modelos abarcan desde elementos finitos y enfoques basados en grafos hasta métodos avanzados.

**Oftalmología:** Se menciona el uso de OCT para capturar la degeneración retiniana, ilustrando la versatilidad del modelo para predecir la progresión de diversas enfermedades degenerativas.

## Discusión: desafíos técnicos

Los autores estructuran las barreras en cuatro categorías:

1. **Cuellos de botella de rendimiento computacional:** las simulaciones complejas (p. ej. sistema circulatorio, cerebro completo) exigen enorme capacidad de cómputo.
2. **Escasez de datos:** limita la diversidad y representatividad de los datasets de entrenamiento, introduce sesgos y provoca sobreajuste (overfitting), de modo que modelos entrenados con poblaciones estrechas no generalizan bien.
3. **Exactitud y validación:** necesidad de verificar que los gemelos digitales reflejen fielmente la realidad clínica.
4. **Integración de datos:** dificultad de fusionar múltiples modalidades de imagen y fuentes clínicas heterogéneas.

## Limitaciones

Al ser una revisión, hereda las limitaciones de la literatura primaria: baja disponibilidad pública de datasets (especialmente en cardiovascular, mama y aplicaciones dentales), heterogeneidad de métodos que dificulta la comparación directa, y ausencia de estandarización en la validación clínica. El propio artículo señala que construir un gemelo digital de todo el cuerpo humano sigue siendo un reto extraordinariamente complejo.

## Relevancia clínica

El trabajo demuestra que la integración de imagen médica de alta resolución con modelos computacionales avanzados posiciona a los gemelos digitales como una herramienta transformadora para la medicina de precisión: mejora la exactitud diagnóstica, la planificación terapéutica individualizada, la estratificación de riesgo, la simulación quirúrgica y el monitoreo continuo y proactivo del paciente. Casos como la reducción de errores diagnósticos en cirugía de DORV (del 42,5% al 4,6%) o la predicción de atrofia cerebral con años de antelación ilustran su impacto clínico directo. Superar los obstáculos de datos, cómputo y validación es la condición para desplegar todo su potencial en la práctica clínica.

## Puntos clave

- Define el gemelo digital como representación virtual dinámica del paciente, actualizada con datos reales y modelos computacionales, para predecir progresión de enfermedad y respuesta a tratamientos.
- Revisión sistemática PRISMA sobre Scopus, PubMed, Web of Science y Google Scholar, organizada mediante un marco de clasificación por sistemas anatómicos (aporte novedoso).
- MRI, CT, PET y ultrasonido son las modalidades base; los modelos van de CFD/FEA tradicionales a aprendizaje profundo y modelos fundacionales.
- Cardiovascular: modelos personalizados de riesgo de SCD, arritmias, Tetralogía de Fallot y cirugía DORV (reducción de diagnósticos erróneos del 42,5% al 4,6%).
- Sistema nervioso central: detección de tumores (K-means, HPU-Net con DSC 0,936), predicción de atrofia en EM 5-6 años antes de síntomas, y Digital Twin Brain que simula ~86.000 millones de neuronas.
- Respiratorio: Lung-DT con YOLOv8 (96,8% de precisión) y simulación de cirugía remota con retroalimentación háptica; mama y oftalmología con MRI/ultrasonido y OCT.
- Cuatro desafíos técnicos clave: cuellos de botella computacionales, escasez de datos (sesgo y sobreajuste), exactitud/validación e integración multimodal.
- Baja disponibilidad pública de datasets es una limitación transversal; el gemelo digital de cuerpo completo sigue siendo un reto abierto.
