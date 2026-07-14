# AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific Experimentation

> **Fuente:** AutoScientists_Self-Organizing_Agent_Teams_for_Long-Running_Scientific_Experimentation_-_2605.28655v1.pdf

## Abstract

AutoScientists presents a novel framework for orchestrating AI agent teams that self-organize to carry out long-running scientific experimentation. Unlike prior approaches that follow a single fixed research trajectory or rely on a central planner with predetermined objectives, AutoScientists employs a group-based, multi-trajectory exploration paradigm. Multiple agent teams operate in parallel where each team autonomously generates hypotheses, designs and executes experiments, iterates on findings, and communicates with other teams. Built on Large Language Models, the system can sustain weeks of continuous, unsupervised experimentation, discovering novel insights across domains such as drug sensitivity prediction, gene regulatory network inference, and material science. The framework shows that self-organizing agent teams can generate non-trivial, publishable-quality contributions.

## Explicación completa

### Problema/Contexto

El progreso científico consiste en ciclos iterativos de generación de hipótesis, diseño experimental, ejecución y revisión. Los sistemas de IA que automatizan partes de este flujo han tenido éxito en dominios específicos, pero enfrentan limitaciones fundamentales: siguen una única trayectoria de investigación o dependen de un planificador central con objetivos fijos, lo que les impide adaptarse a resultados inesperados, explorar vías divergentes, o escalar a experimentos de larga duración (semanas/meses) sin intervención humana.

### Metodología

AutoScientists afronta estos desafíos con un diseño descentralizado de agentes auto-organizados:

- **Paralelismo de trayectos múltiples:** Se instancian varios equipos de agentes, cada uno siguiendo su propia línea de investigación. Los equipos no dependen de un planificador central; cada equipo toma decisiones autónomas sobre qué hipótesis explorar y cómo diseñar los experimentos.
- **Comunicación entre equipos:** Los equipos comparten hallazgos y aprenden de la experiencia de otros equipos a través de un mecanismo de comunicación flexible (un panel de mensajes/resultados global). Los equipos pueden abandonar hipótesis poco prometedoras y adoptar líneas más fructíferas descubiertas por otros equipos, imitando la dinámica del discurso científico real.
- **Ejecución en bucle largo:** Cada agente opera un ciclo de hipotetizar-experimentar-observar que puede durar semanas, sin requerir intervención humana, usando funciones de herramienta para interactuar con bases de datos, simuladores y compiladores de código.
- **Sistema de memoria temporal:** Los resultados se almacenan en una base de datos vectorial (Chroma) indexada por similitud semántica para permitir la recuperación de experimentos relevantes, tanto pasados como realizados por otros equipos.

### Resultados

El sistema fue evaluado en tres dominios científicos:

1. **Predicción de sensibilidad a fármacos (DREAM Challenge):** AutoScientists exploró docenas de configuraciones de modelos de machine learning, incorporando características genómicas, transcriptómicas y epigenómicas. Generó modelos predictivos competitivos en un plazo de 30 horas.
2. **Inferencia de redes de regulación génica (GRN):** En un dominio con pocos datos de anotación disponible, los equipos Autocientíficos experimentaron con diversos enfoques de modelado y lograron reconstrucciones comparables a métodos supervisados.
3. **Ciencia de materiales (propiedades MOF):** Los equipos Autocientíficos realizaron experimentos automatizados en el dominio de tamices moleculares, seleccionando y entrenando modelos de machine learning para la predicción de propiedades de adsorción de CO2.

En los tres casos, el sistema produjo conclusiones novedosas utilizables por investigadores humanos, y algunas de las configuraciones optimizadas superaron los resultados de referencia.

### Limitaciones

- Los agentes autocientíficos todavía pueden generar hipótesis triviales o experimentos mal diseñados.
- La calidad del resultado depende de la calidad del Large Language Model subyacente.
- El coste computacional no es trivial debido a los múltiples equipos y la larga duración de los experimentos.
- La verificabilidad de los hallazgos producidos automáticamente sigue siendo un desafío.

### Relevancia

Este trabajo representa un cambio de paradigma importante en la automatización de la investigación científica al demostrar que equipos de agentes auto-organizados pueden sostener períodos prolongados de aprendizaje y descubrimiento fuera del alcance de la experimentación manual. El concepto de equipo múltiple descentralizado eliminó la necesidad de un planificador central y allana el camino para ecosistemas de investigación autónomos.

## Puntos clave

- AutoScientists propone un nuevo paradigma de múltiples equipos de agentes auto-organizados en paralelo, sin planificador central
- El sistema puede ejecutarse durante semanas sin intervención humana en experimentación científica continua.
- Demostrado en tres dominios: farmacogenómica, GRN y ciencia de materiales.
- Incorpora base de datos vectorial para memoria y compartición de descubrimientos.
- Abre la puerta a ecosistemas de investigación autónoma y colaborativa impulsados por IA.
