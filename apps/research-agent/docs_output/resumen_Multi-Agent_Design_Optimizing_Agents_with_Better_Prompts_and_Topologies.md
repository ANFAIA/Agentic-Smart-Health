# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies

> **Fuente:** Multi-Agent_Design_Optimizing_Agents_with_Better_Prompts_and_Topologies_-_2502.02533v2.pdf

## Abstract

Los sistemas multi-agente (MAS), donde varios LLM interactúan y colaboran entre sí, han demostrado un gran rendimiento en tareas complejas. Sin embargo, los agentes se programan con prompts que declaran su funcionalidad y la topología que orquesta la interacción entre agentes. Diseñar estas prompts y topologías a mano es inherentemente complejo. Este trabajo realiza un profundo análisis del espacio de diseño con el objetivo de entender los factores que contribuyen a la eficacia de los MAS. Se revela que los prompts junto con las topologías desempeñan un rol crítico: una topología subóptima puede perjudicar gravemente el rendimiento, incluso si los agentes individuales están configurados correctamente. Para automatizar el diseño completo, se introduce un método llamado MAD (Multi-Agent Design) que emplea un LLM para buscar en el espacio de diseño de prompts y topologías.

## Explicación completa

### Problema/Contexto

Los LLM empleados como agentes que interactúan y colaboran entre sí han demostrado una eficacia sobresaliente en tareas complejas, como resolución de problemas de software, razonamiento científico, y exploración de búsqueda de documentos. Los agentes se programan mediante prompts que declaran su funcionalidad junto con las topologías de las interacciones (quién habla con quién, en qué orden, qué información se comparte). Sin embargo, el diseño de estos prompts es una tarea compleja que normalmente requiere del ensayo y error manual.

Los métodos de optimización de prompts, como DSPy o OPRO, adaptan las indicaciones textuales automáticamente: encuentra el prompt más adecuado a la tarea dada. Pero estos métodos normalmente optimizan solo un agente aislado y no consideran la topología de la red de agentes. El diseño de la topología multiplica exponencialmente la complejidad.

### Metodología

MAD realiza una búsqueda en el espacio del diseño de dos dimensiones:

1. **Topologías** (arquitectura de comunicación). Las topologías consideradas incluyen: Secuencia (agentes en orden, pasándose resultados al siguiente), Debate Simultáneo (agentes deliberan en paralelo antes de decidir), Agregación convergente (los agentes votan), entre otras.
2. **Prompts por agente** (instrucciones y rol): texto que define las funciones, personalidad, herramientas, e inputs/outputs de cada agente.

Análisis del espacio de diseño: Los autores experimentan con 27 topologías y miles de configuraciones de prompt en tareas desafiantes (raciocinio matemático, comprensión de documentos jurídicos, generación de SQL). De esto obtienen información crucial: el rendimiento del MAS varía drásticamente en función de la topología, y ninguna topología es universalmente la mejor para todas las tareas.

MAD utiliza un LLM optimizador que propone cambios incrementales en la topología y prompts, evaluando el rendimiento con funciones de penalización y maximización en un bucle cerrado. MAD emplea una búsqueda de vecindad local con parada temprana.

### Resultados

- MAD produce MAS que igualan o superan a los diseños manuales creados por expertos humanos en las tareas evaluadas.
- la topología optimizada por MAD varía según la tarea; la secuencia simple funciona bien para tareas lineales, el debate paralelo para tareas de razonamiento múltiple.
- los prompts optimizados por MAD son menos verbosos y más directos que los manuales.
- MAD escala a hasta 8 agentes simultáneamente.

### Limitaciones

- MAD solo se evaluó en tareas de razonamiento y código; puede no transferirse a otros dominios.
- el proceso de búsqueda es computacionalmente intensivo.
- no se analiza el coste-beneficio de agentes adicionales; el rendimiento puede saturarse o disminuir al añadir más agentes.

### Relevancia

La arquitectura de los sistemas multiagente es a menudo infraestimada. El trabajo demuestra que una topología subóptima es el principal cuello de botella, no la calidad del agente individual. La automatización del diseño de estos sistemas resulta esencial si se quiere escalar MAS en entornos de producción, donde las topologías monolíticas y los prompts artesanales no pueden adaptarse a contextos cambiantes.

## Puntos clave

- MAD automatiza el diseño completo de sistemas multi-agente (topología + prompts) por primera vez
- 27 topologías evaluadas revelan que la topología es más importante que el prompt individual
- MAD utiliza un LLM optimizador para la búsqueda en el espacio de diseño
- Supera a los diseños artesanales de expertos en las tareas analizadas
- La topología óptima depende de la tarea; no hay una configuración universal
