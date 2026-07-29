# A Comprehensive Survey of Self-Evolving AI Agents: Un nuevo paradigma que conecta los modelos fundacionales con los sistemas agénticos de aprendizaje continuo

> **Fuente:** A Comprehensive Survey of Self-Evolving AI Agents_ A New Paradigm Bridging Foundation Models and Lifelong Agentic Systems - 2508.07407v2.pdf — Fang, J.; Peng, Y.; Zhang, X.; Wang, Y.; et al. (University of Glasgow, University of Sheffield, MBZUAI, NUS, Cambridge, UCL, Aberdeen, Leiden). arXiv:2508.07407v2, 31 Aug 2025. GitHub: EvoAgentX/Awesome-Self-Evolving-Agents.

## Abstract

Este survey (Fang, Peng, Zhang et al., 2025; arXiv:2508.07407v2) ofrece una revisión sistemática de las técnicas de "auto-evolución" de agentes de IA basados en LLMs, es decir, sistemas capaces de mejorar automáticamente sus propios componentes a partir de datos de interacción y feedback del entorno. El problema que abordan es que la mayoría de los agentes actuales dependen de configuraciones diseñadas manualmente que permanecen estáticas tras el despliegue, lo que limita su adaptación a entornos dinámicos. Los autores proponen un marco conceptual unificado que abstrae el bucle de realimentación con cuatro componentes (System Inputs, Agent System, Environment y Optimisers), y sobre él organizan y comparan un amplio abanico de técnicas de evolución dirigidas a distintos componentes: el modelo fundacional, los prompts, la memoria, las herramientas, los workflows y los mecanismos de comunicación entre agentes. También formalizan las "Tres Leyes de los Agentes Auto-Evolutivos" (Endure/seguridad, Excel/preservación del rendimiento, Evolve/evolución autónoma), trazan la transición de paradigmas MOP→MOA→MAO→MASE, revisan estrategias de evolución específicas de dominio (biomedicina, programación, finanzas/legal) y dedican una discusión a la evaluación, la seguridad y la ética. El objetivo es dar a investigadores y profesionales una comprensión holística que siente las bases de sistemas agénticos más adaptativos, autónomos y de aprendizaje continuo (lifelong).

## Explicación completa

## Problema y motivación

Los avances en grandes modelos de lenguaje (LLMs) han impulsado los agentes de IA: sistemas autónomos que usan un LLM como núcleo de razonamiento para interpretar objetivos, planificar acciones y generar salidas en entornos abiertos y reales. Un agente típico combina un modelo fundacional (el núcleo) con módulos de percepción, planificación, memoria y uso de herramientas. Cuando una sola instancia se queda corta en especialización y coordinación, se recurre a los sistemas multi-agente (MAS), donde varios agentes colaboran mediante topologías y protocolos de comunicación para lograr inteligencia colectiva.

El problema central que motiva el survey es que **la mayoría de los agentes, sean de un solo agente o multi-agente, dependen de configuraciones manuales y permanecen estáticos tras el despliegue**. Pero los entornos reales son dinámicos: cambian las intenciones de usuario, los requisitos de tarea, y las herramientas o fuentes de información. Reconfigurar manualmente los agentes es costoso, laborioso y poco escalable. De ahí surge el paradigma de los **Self-Evolving AI Agents**, definidos como "sistemas autónomos que optimizan continua y sistemáticamente sus componentes internos mediante la interacción con entornos, con el objetivo de adaptarse a tareas, contextos y recursos cambiantes preservando la seguridad y mejorando el rendimiento".

## Las Tres Leyes y la evolución de paradigmas

Inspirándose en las Tres Leyes de la Robótica de Asimov, los autores proponen las **Tres Leyes de los Agentes Auto-Evolutivos**, jerárquicas:
1. **Endure (Adaptación segura)**: el agente debe mantener seguridad y estabilidad durante cualquier modificación.
2. **Excel (Preservación del rendimiento)**: sujeto a la primera ley, debe preservar o mejorar el rendimiento existente.
3. **Evolve (Evolución autónoma)**: sujeto a las dos anteriores, debe poder optimizar autónomamente sus componentes ante cambios de tareas, entornos o recursos.

Sitúan la auto-evolución dentro de un cambio de paradigma más amplio del aprendizaje centrado en LLMs, que progresa en cuatro etapas:
- **MOP (Model Offline Pretraining)**: preentrenamiento sobre corpus estáticos y despliegue congelado (Transformer, BPE/SentencePiece, MoE).
- **MOA (Model Online Adaptation)**: adaptación post-despliegue mediante fine-tuning supervisado, LoRA/adapters, RLHF, alineamiento.
- **MAO (Multi-Agent Orchestration)**: coordinación de múltiples agentes LLM que comunican y colaboran (debate, tool calling, MCP) sin modificar los parámetros del modelo.
- **MASE (Multi-Agent Self-Evolving)**: bucle lifelong y auto-evolutivo donde una población de agentes refina continuamente sus prompts, memoria, uso de herramientas e incluso sus patrones de interacción basándose en feedback del entorno y meta-recompensas.

## Marco conceptual unificado (la contribución central)

El survey abstrae el proceso de auto-evolución en un **bucle de realimentación cerrado e iterativo** con cuatro componentes:
1. **System Inputs**: la información contextual y datos que definen el problema. Puede ser optimización a **nivel de tarea** (descripción de tarea + dataset de entrenamiento; a veces sintetizando datos con LLMs cuando no hay etiquetas) o a **nivel de instancia** (mejorar el rendimiento sobre un ejemplo concreto, un par entrada-salida).
2. **Agent System**: el sistema a optimizar (un agente o varios colaborando), descomponible en LLM, estrategia de prompting, módulo de memoria, política de uso de herramientas, etc. Se puede optimizar un solo componente o varios conjuntamente.
3. **Environment**: el contexto externo donde opera el agente y que genera señales de feedback (métricas proxy como accuracy, F1, success rate, o evaluadores basados en LLM cuando no hay ground-truth).
4. **Optimisers**: el corazón del bucle, que refina el sistema buscando la configuración óptima A* = arg max O(A;I). Un optimizador se define por (a) su **espacio de búsqueda** (S) —desde prompts o selección de herramientas hasta parámetros del LLM o estructuras arquitectónicas— y (b) su **algoritmo de optimización** (H) —heurísticas basadas en reglas, descenso de gradiente, optimización bayesiana, MCTS, aprendizaje por refuerzo, estrategias evolutivas o políticas aprendidas.

Mencionan **EvoAgentX** como el primer framework open-source que implementa este proceso de auto-evolución (generación, ejecución, evaluación y optimización automatizadas de sistemas agénticos).

## Optimización de un solo agente (Single-Agent)

Se organiza según el componente objetivo:

**1. Optimización del comportamiento del LLM.**
- *Métodos basados en entrenamiento*: SFT sobre trayectorias de razonamiento (STaR, que se auto-mejora reentrenando con soluciones correctas; NExT; DeepSeek-Prover), y RL que trata el razonamiento como decisión secuencial (DPO con pares de preferencia, Self-Rewarding, Agent Q con MCTS+autocrítica, Tülu 3 con recompensas verificables, DeepSeek-R1 con GRPO, y esquemas de auto-evolución pura como Absolute Zero y R-Zero que generan y resuelven sus propias tareas).
- *Métodos test-time* (sin modificar parámetros): estrategias **basadas en feedback** (verificadores a nivel de resultado como CodeT/LEVER con compiladores, o a nivel de paso con process reward models para evitar el "razonamiento infiel") y estrategias **basadas en búsqueda** (CoT-SC/best-of-N, Tree-of-Thoughts y Graph-of-Thoughts con MCTS, Forest-of-Thoughts, Buffer-of-Thoughts).

**2. Optimización de prompts** (los LLMs son muy sensibles al fraseo). Cuatro categorías:
- *Edit-based*: refinamiento local por inserción/borrado/sustitución (GRIPS, Plum, TEMPERA con RL).
- *Generativa*: un LLM genera prompts nuevos guiado por señales (OPRO, PromptAgent con MCTS, MIPRO con optimización bayesiana, Retroformer con política RL).
- *Text gradient-based*: feedback en lenguaje natural que actúa como "gradiente textual" (ProTeGi, TextGrad, agent symbolic learning).
- *Evolutiva*: población de prompts refinada por mutación/cruce/selección (EvoPrompt, Promptbreeder).

**3. Optimización de la memoria** (para combatir ventanas de contexto limitadas y olvido). Enfoque en estrategias de inferencia:
- *Memoria a corto plazo*: compresión, resumen, retención selectiva (COMEDY, ReadAgent, MemoChat, MoT, StructRAG, MemoryBank inspirada en la curva de Ebbinghaus, Reflexion).
- *Memoria a largo plazo*: almacenamiento persistente y escalable, principalmente vía RAG (EWE, A-MEM con redes de conocimiento evolutivas, Mem0, MemGPT, HippoRAG inspirada en el hipocampo, GraphReader, ChatDB con SQL), con mecanismos de control de memoria (qué/cuándo/cómo almacenar, actualizar o descartar: MEM1 con RL, MIRIX con seis tipos de memoria, Agent KB compartida).

**4. Optimización de herramientas.** Dos direcciones:
- *Mejorar la interacción con herramientas*: entrenamiento (SFT con trayectorias de uso de herramientas: ToolLLM, GPT4Tools, STE, curriculum learning en Confucius, Gorilla con retriever; y RL: ReTool, Tool-N1, Tool-Star, ARPO) e inferencia (prompt-based como EASYTOOL/DRAFT/PLAY2PROMPT que refinan la documentación; reasoning-based con búsqueda en árbol como ToolChain, Tool-Planner, MCP-Zero).
- *Optimizar las herramientas mismas* (crear/modificar herramientas): CREATOR, LATM, CRAFT, AgentOptimizer (trata herramientas como pesos aprendibles), Alita (formato MCP), CLOVA (asistente visual de bucle cerrado).

## Optimización multi-agente (Multi-Agent)

Reformula el diseño de workflows como un **problema de búsqueda sobre tres espacios interconectados**: el espacio estructural de topologías, el espacio semántico de roles/instrucciones, y el espacio de capacidades de los LLM backbone. Se recorre en varias dimensiones:
- **Sistemas diseñados manualmente**: workflows paralelos (generación concurrente + votación mayoritaria), jerárquicos (pipelines top-down para tareas con dependencias, p.ej. MetaGPT), y de debate multi-agente (ciclos adversario-negociación-arbitraje para corregir errores de razonamiento). Un hallazgo notable: a veces un solo LLM grande con prompts bien diseñados iguala a frameworks multi-agente complejos.
- **Optimización de prompts** en topologías fijas (AutoAgents, DSPy, MIPRO, PromptWizard).
- **Optimización de topología**: a nivel de workflow en código (AutoFlow, AFlow, ScoreFlow, MAS-GPT) o de grafo de comunicación (GPTSwarm, G-Designer, AgentPrune, NetSafe).
- **Optimización unificada**: conjunta de prompts, topologías y parámetros (ADAS y FlowReasoner basados en código; EvoAgent, MASS, EvoFlow, MAS-ZERO basados en búsqueda; MaAS basado en aprendizaje).
- **Optimización del LLM backbone**: orientada a razonamiento o a colaboración (COPPER, OPTIMA, MaPoRL).

Los mecanismos de comunicación se clasifican en salida estructurada (JSON/XML/código), lenguaje natural, y protocolos estandarizados (A2A para comunicación horizontal, ANP para un "internet de agentes" descentralizado con identidad DID, MCP para comunicación vertical con herramientas/datos, Agora como meta-protocolo).

## Optimización específica de dominio

El survey revisa estrategias adaptadas a dominios donde el comportamiento y los objetivos están fuertemente acoplados a restricciones específicas:
- **Biomedicina**: diagnóstico médico (MDAgents, MedAgentSim, PathFinder, MDTeamGPT, MMedAgent, MedAgent-Pro) y descubrimiento molecular (CACTUS, ChemAgent, DrugAgent, LIDDIA, OSDA Agent).
- **Programación**: refinamiento de código (Self-Refine, AgentCoder, CodeAgent, CodeCoR, OpenHands) y depuración (Self-Debugging, PyCapsule, RGD).
- **Finanzas y Legal**: toma de decisiones financieras (FinCon, PEER, FinRobot) y razonamiento legal (LawLuo, AgentCourt, LegalGPT).

## Evaluación, seguridad y ética

Se dedica una sección a metodologías y benchmarks de evaluación de sistemas agénticos, así como a las consideraciones de seguridad y ética, elementos que los autores consideran críticos para garantizar la efectividad, fiabilidad y despliegue responsable de estos sistemas —en coherencia con la Primera Ley (Endure).

## Limitaciones y retos abiertos

Los propios autores reconocen que la auto-evolución plena es un objetivo a largo plazo: los sistemas actuales están lejos de exhibir las capacidades necesarias para una auto-evolución segura, robusta y de dominio abierto, y en la práctica el progreso se logra mediante técnicas de optimización iterativa acotadas. Como survey, no aporta experimentos propios sino una taxonomía y síntesis; el campo evoluciona muy rápido (cubre trabajos de 2023 a 2025) y algunos de los mecanismos revisados son emergentes y poco validados. Persisten retos en evaluación estandarizada, garantías de seguridad durante la auto-modificación, y escalabilidad de la coordinación multi-agente.

## Relevancia clínica y aplicada

Aunque es un survey de IA general, tiene relevancia directa para aplicaciones clínicas y biomédicas: revisa explícitamente agentes de diagnóstico médico y descubrimiento de fármacos (MDAgents, MedAgent-Pro, DrugAgent, ChemAgent, LIDDIA), donde la capacidad de adaptación continua, la preservación del rendimiento y —sobre todo— las garantías de seguridad (Primera Ley) son especialmente sensibles. El marco conceptual unificado y las Tres Leyes ofrecen a los desarrolladores de sistemas agénticos en salud una guía para diseñar agentes que se adapten a nuevos protocolos, herramientas de análisis o guías clínicas manteniendo la seguridad del paciente como restricción prioritaria.

## Puntos clave

- Define los 'Self-Evolving AI Agents': sistemas que optimizan continuamente sus componentes internos mediante feedback del entorno, adaptándose a tareas/contextos cambiantes mientras preservan la seguridad.
- Formaliza las Tres Leyes de los Agentes Auto-Evolutivos (jerárquicas): Endure (seguridad), Excel (preservar rendimiento) y Evolve (evolución autónoma).
- Traza la transición de paradigmas: MOP (preentrenamiento offline) → MOA (adaptación online) → MAO (orquestación multi-agente) → MASE (auto-evolución multi-agente lifelong).
- Contribución central: un marco conceptual unificado del bucle de realimentación con 4 componentes — System Inputs, Agent System, Environment y Optimisers (definidos por espacio de búsqueda + algoritmo).
- Optimización single-agent en 4 frentes: comportamiento del LLM (SFT/RL y test-time), prompts (edit/generativa/text-gradient/evolutiva), memoria (corto y largo plazo, RAG) y herramientas (uso y creación).
- Optimización multi-agente como búsqueda sobre topologías, roles/prompts y backbones; incluye workflows manuales (paralelo, jerárquico, debate) y automáticos (AFlow, GPTSwarm, ADAS, MASS, MaAS).
- Protocolos de comunicación estandarizados: A2A (horizontal), ANP (internet de agentes descentralizado con DID), MCP (vertical con herramientas), Agora (meta-protocolo).
- Estrategias de evolución específicas de dominio en biomedicina (MDAgents, DrugAgent, ChemAgent), programación (Self-Refine, OpenHands) y finanzas/legal (FinCon, AgentCourt).
- Menciona EvoAgentX como primer framework open-source que implementa el proceso completo de auto-evolución de agentes.
- Reconoce que la auto-evolución plena y segura sigue siendo un objetivo a largo plazo; el progreso actual son técnicas de optimización iterativa acotadas, con retos abiertos en evaluación, seguridad y escalabilidad.
