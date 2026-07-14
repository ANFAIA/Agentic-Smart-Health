# Integration of Digital Twin Technology in Orthodontics: A Scoping Review

> **Fuente:** Integration_of_Digital_Twin_Technology_in_Orthodontics_ A_Scoping_Review.pdf

## Abstract

Esta revisión de alcance tiene como objetivo mapear la evidencia actual sobre la integración de la tecnología de gemelos digitales (DTT) en la ortodoncia. La DTT es un concepto emergente en salud que permite simulación y análisis predictivo de pacientes en tiempo real, mejorando el tratamiento personalizado. La revisión incluye estudios publicados entre 2018 y 2025, identificando aplicaciones de DTT en ortodoncia en dominios como planificación de tratamiento, predicción de movimiento dental, análisis biomecánico personalizado y monitorización adaptativa. Los hallazgos muestran que los gemelos digitales integran imágenes 3D, inteligencia artificial y modelos de elementos finitos para crear réplicas virtuales completas de los pacientes, lo que posibilita ajustes en el protocolo de tratamiento en respuesta a cambios en el estado real del paciente. Sin embargo, persisten desafíos significativos, incluidos los altos requisitos computacionales, la complejidad de la integración de datos, la distinción entre verdaderos gemelos digitales y flujos de trabajo digitales estáticos, y la necesidad de protocolos estandarizados.

## Explicación completa

### Problema/Contexto

La ortodoncia moderna ha experimentado una transformación digital en las últimas dos décadas: escaneo intraoral, CBCT, planificación digital de tratamientos y flujo de trabajo CAD/CAM son ya prácticas comunes. Sin embargo, la mayoría de estos flujos de trabajo son estáticos:capturan un único momento del paciente (o información secuencial) y generalmente no se actualizan automáticamente. 

La DTT ofrece un salto cualitativo al crear una réplica virtual dinámica, completa, que se realimenta automáticamente con los datos reales del paciente (imágenes de seguimiento, escaneos de alineadores, resultados intermedios de tratamiento, entre otros). Esto permite un ciclo continuo de monitorización-simulación-ajuste que distingue un verdadero gemelo digital de una simple maqueta digital.

### Metodología

Revisión de alcance (scoping review) realizada según las pautas PRISMA-ScR. La búsqueda se realizó en múltiples bases de datos electrónicas para identificar estudios pertinentes publicados entre 2018 y 2025 sobre aplicaciones de DT en ortodoncia. Se ha incluido literatura de ingeniería, ciencias de la computación, física médica y odontología.

### Aplicaciones identificadas

1. **Planificación de tratamiento mejorada**: la integración de datos de múltiples modalidades (escaneo intraoral, CBCT, fotografías extraorales) permite que el gemelo digital proporcione una visualización de resultados y planificación de objetivos en 4D.
2. **Predicción del movimiento dental**: el gemelo digital incorpora propiedades mecánicas del periodonto, fuerzas aplicadas por el alineador, y retroalimentación de los resultados reales observados, corrigiendo iterativamente el modelo biomecánico original. La evidencia inicial muestra que esta retroalimentación mejora la fidelidad de los resultados pronosticados.
3. **Monitorización adaptativa del tratamiento**: a través de la adquisición periódica de escaneos de seguimiento (por ejemplo, intraorales o de superficie), el gemelo digital puede analizar el movimiento logrado hasta la etapa actual y compararlo con la trayectoria planificada original. Si existen divergencias, puede simular trayectorias corregidas y recomendar ajustes al protocolo.
4. **Simulación biomecánica personalizada**: la topología de elementos finitos calcula las tensiones en el ligamento periodontal individualizadas para el paciente, algo que no es viable en workflows lineales.

### Resultados

- los estudios incluidos muestran mejoras significativas en precisión de planificación evitando sobretratamiento.
- la predicción de movimiento mejora entre un 15 y 25% con la incorporación de retroalimentación de imágenes intraorales de seguimiento comparada con modelos biomecánicos estáticos.
- los médicos reportan una mayor confianza en el diagnóstico y planificación cuando pueden interactuar de forma dinámica con el modelo.

### Limitaciones

- la mayoría de los diseños actuales son pseudogemelos digitales o gemelos digitales parciales, no gemelos completos.
- alto coste computacional y de integración de datos impide un despliegue en clínicas generales.
- falta de estandarización: no existe un consenso sobre los protocolos de validación de gemelos digitales.
- la base evidencial sigue siendo preliminar, sin ensayos controlados aleatorios.

### Relevancia

Los gemelos digitales tienen el potencial de transformar la ortodoncia de una práctica reactiva a una proactiva y predictiva. Permiten la personalización radical del tratamiento, la corrección en tiempo real de desviaciones del plan original y una comunicación radicalmente mejorada con el paciente. La vía de implementación gradual parece comenzar en la ortodoncia de alineadores transparentes, pero eventualmente podría impactar en todos los tipos de tratamiento.

## Puntos clave

- Los gemelos digitales en ortodoncia integran imágenes 3D, IA y simulación biomecánica para crear réplicas virtuales dinámicas de los pacientes
- Permiten monitorización adaptativa: el plan de tratamiento se ajusta a los resultados reales observados
- La predicción de movimiento dental mejora un 15-25% con retroalimentación iterativa
- la mayoría de las implementaciones actuales son gemelos digitales parciales
- la estandarización protocolos y la reducción de costes computacionales son clave para la adopción generalizada
