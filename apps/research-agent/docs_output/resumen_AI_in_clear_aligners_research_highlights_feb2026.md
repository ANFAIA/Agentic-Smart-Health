# AI in clear aligners: research highlights (Febrero 2026)

> **Fuente:** wu2021_mesh_tooth_segmentation_intraoral.pdf; weekley2025_intraoral_scan_resolution_segmentation.pdf; wang2024_multistage_3d_tooth_cbct.pdf; ji2025_shape_preserving_tooth_cbct.pdf; liu2023_toothsegnet_cbct.pdf; ma2024_px2tooth_panoramic_reconstruction.pdf

## Abstract

Esta entrada sintetiza seis trabajos recientes de inteligencia artificial (IA) que apuntalan el flujo de trabajo digital de los alineadores transparentes (clear aligners): desde la segmentación automática de dientes y la localización de landmarks sobre escaneos intraorales, hasta la segmentación de dientes y raíces en CBCT y la reconstrucción 3D de la dentición a partir de una única radiografía panorámica. Los métodos revisados —TS-MDL/iMeshSegNet + PointNet-Reg (Wu et al., 2021), el estudio de resolución de escaneo con PointMLP (Weekley et al., 2025), el marco multi-etapa semisupervisado para CBCT (Wang et al., 2024), la segmentación con preservación de forma (Ji et al., 2025), ToothSegNet robusto a artefactos (Liu et al., 2023) y PX2Tooth (Ma et al., 2024)— demuestran que la IA ya alcanza precisiones clínicamente relevantes (DSC ~0.96 en escaneos, errores de landmark sub-milimétricos) en las etapas que consumen más tiempo del ortodoncista. En conjunto perfilan una cadena de automatización de la planificación de tratamiento, aunque la predicción de resultado y el setup automático siguen siendo la frontera menos madura y peor cubierta por la literatura indexada en repositorios abiertos.

## Explicación completa

## Contexto y alcance de la entrada

Los alineadores transparentes dependen por completo de un modelo digital 3D de la dentición del paciente. Antes de fabricar cada férula, el ortodoncista debe: (1) capturar la anatomía (escaneo intraoral o CBCT/radiografía), (2) segmentar cada diente individualmente, (3) anotar landmarks anatómicos, (4) planificar el reposicionamiento dental "diente a diente" (setup) y (5) generar la secuencia de férulas. Los pasos 2 y 3, hechos a mano, consumen entre 45 y 60 minutos por escaneo y dependen mucho de la experiencia del clínico. Los seis papers revisados atacan precisamente estos cuellos de botella con aprendizaje profundo. Los agrupamos por su lugar en el flujo de trabajo.

---

## Bloque 1 — Segmentación de dientes y landmarks sobre escaneos intraorales (el flujo directo del alineador)

### Wu et al. (2021), *Two-Stage Mesh Deep Learning for Automated Tooth Segmentation and Landmark Localization on 3D Intraoral Scans* (TS-MDL)
- **Problema.** Segmentar dientes y localizar landmarks en la malla del escaneo intraoral son tareas esenciales para crear un plan de tratamiento personalizado (los propios autores citan explícitamente "la fabricación de clear aligners"). Hacerlo manualmente es lento y dependiente del experto; además, la localización de landmarks estaba poco explorada.
- **Metodología.** Marco de dos etapas basado en mesh deep learning:
  - *Etapa 1 — iMeshSegNet*: variante mejorada de MeshSegNet que sustituye el pooling promedio simétrico y las grandes matrices de adyacencia por la operación **EdgeConv** (grafo k-NN estático, k=6 y k=12), con módulos de aprendizaje restringido por grafo (GLM), fusión densa local-a-global y refinamiento por *graph-cut* multi-etiqueta. Trabaja sobre la malla submuestreada (~10.000 celdas desde las ~100.000 originales del iTero Element) y reproyecta a resolución completa mediante un SVM con kernel RBF.
  - *Etapa 2 — PointNet-Reg*: para cada diente segmentado se recorta su ROI y una variante ligera de PointNet regresa **heatmaps gaussianos** que codifican la posición de los landmarks (sigmoide en la última capa; se toma la celda de máximo valor como coordenada).
  - Datos: 136 escaneos maxilares reales (iTero Element), 14 dientes y 66 landmarks anotados por dos ortodoncistas; aumento de datos por simetría, traslación, rotación y reescalado.
- **Resultados.** iMeshSegNet alcanza **DSC 0.964 ± 0.054**, SEN 0.970, PPV 0.960 y Hausdorff 0.995 mm tras refinamiento, superando significativamente a MeshSegNet, y además entrena ~20× más rápido e infiere ~8.6× más rápido (0.62 s/escaneo). El pipeline de dos etapas iMeshSegNet+PointNet-Reg logra un **MAE de 0.597 ± 0.761 mm** en los 66 landmarks, batiendo a las variantes de una sola etapa. Los errores mayores se dan en primeros molares por captura incompleta de zonas posteriores.
- **Relevancia clínica.** Los autores contrastan sus errores con criterios de la American Board of Orthodontics (ABO): desviaciones >0.5 mm empiezan a penalizarse, pero los resultados quedan dentro del rango clínicamente aceptable (radio de 2 mm) y aproximan el estándar de 0.5 mm. Los landmarks sirven para superponer arcadas y calcular el movimiento dental entre pre- y post-tratamiento, base cuantitativa del staging de alineadores.
- **Limitaciones.** Dataset pequeño (136 casos); solo información de superficie (no puede predecir landmarks en zonas solapadas por malposición); mallas incompletas en molares. Proponen añadir reparación de malla 3D.

### Weekley et al. (2025), *Evaluating the Suitability of Different Intraoral Scan Resolutions for Deep Learning-Based Tooth Segmentation*
- **Problema.** Los escaneos intraorales tienen >200.000 celdas y se procesan submuestreados (típicamente 10K–16K), pero no se había cuantificado sistemáticamente cuánta precisión se pierde al bajar la resolución —dato crítico para desplegar modelos en dispositivos *edge* con memoria limitada.
- **Metodología.** Entrenan **PointMLP** (bloques MLP residuales con módulo afín geométrico) por separado a 2K, 4K, 6K, 8K, 10K y 16K celdas sobre 571 sujetos del dataset público **Teeth3DS / 3D Teeth Seg Challenge 2022** (mandíbula, hasta 14 dientes). Representan cada celda con un vector de 24 dimensiones (coordenadas y normales de 3 vértices + baricentro). Las predicciones de baja resolución se re-densifican por KNN a 10K/16K. Métricas: DSC, OA, SEN, PPV y tiempo de inferencia.
- **Resultados.** El rendimiento se mantiene razonable hasta **6K celdas** (DSC ~0.90 incluso tras re-densificar), pero **cae bruscamente por debajo de 6K**: a 4K y 2K el DSC re-densificado se desploma (~0.67–0.68). La segmentación de la encía (fondo) es estable a todas las resoluciones, pero los límites finos de los dientes, especialmente premolares y caninos, se degradan a baja resolución. Curiosamente, las etiquetas predichas por el modelo son más limpias que las etiquetas decimadas de referencia.
- **Relevancia clínica.** Ofrece una guía práctica: 6K es un punto de equilibrio viable entre coste computacional y precisión para llevar la segmentación a dispositivos ligeros/clínicos, con implicaciones directas para software de planificación de alineadores en la consulta.
- **Limitaciones.** Una sola arquitectura (PointMLP); validación aún no clínica; foco exclusivo en el compromiso resolución-precisión.

---

## Bloque 2 — Segmentación de dientes y raíces en CBCT (planificación con anatomía radicular)

Conocer la raíz, y no solo la corona, es clave para planificar movimientos seguros con alineadores (control de torque, reabsorción radicular, movimientos en masa). Tres trabajos abordan el reto en CBCT.

### Wang et al. (2024), *A Multi-Stage Framework for 3D Individual Tooth Segmentation in Dental CBCT*
- **Problema.** El deep learning en CBCT exige mucha anotación (costosa) y sufre *domain shift* entre distintos escáneres, lo que degrada la generalización.
- **Metodología.** Marco **multi-etapa semisupervisado** para segmentación 3D de dientes individuales; presentado al reto STS-3D ("Semi-supervised Teeth Segmentation" 3D).
- **Resultados.** Obtuvo el **tercer puesto** en el reto STS-3D y superó a otros métodos semisupervisados en el conjunto de validación.
- **Relevancia clínica.** Demuestra que se puede reducir la dependencia de grandes volúmenes de datos anotados, abaratando la puesta en marcha de modelos de segmentación 3D para planificación.

### Ji et al. (2025), *Shape-preserving Tooth Segmentation from CBCT Images Using Deep Learning with Semantic and Shape Awareness*
- **Problema.** Los métodos previos ignoran las relaciones anatómicas entre dientes vecinos; la proximidad física y las densidades radiográficas similares bajo oclusión provocan "adherencias" e interferencias que distorsionan la forma segmentada.
- **Metodología.** Marco que integra **conciencia semántica y de forma**:
  - Estrategia de aprendizaje multi-etiqueta *promptado por el centroide del diente objetivo* para modelar relaciones semánticas y reducir la ambigüedad de forma.
  - Mecanismo de *tooth-shape-aware learning* que impone restricciones morfológicas explícitas (mediante un **Signed Distance Map / Level Set**) para preservar la integridad de los bordes.
  - Todo unificado en una arquitectura **multi-tarea** que optimiza conjuntamente máscara, priors de forma y delineación de bordes, con codificador compartido y clustering de centroides.
- **Resultados.** Supera significativamente a los métodos existentes en datasets internos y externos, resolviendo adherencias dentales y preservando morfología anatómicamente fiel.
- **Relevancia clínica.** Bordes de diente anatómicamente fieles son imprescindibles para planificación de implantes y de movimientos ortodóncicos; la preservación de forma incrementa la confianza clínica en los modelos 3D generados.

### Liu et al. (2023), *ToothSegNet: Image Degradation meets Tooth Segmentation in CBCT Images*
- **Problema.** Las CBCT clínicas reales contienen **artefactos metálicos y desenfoque** (por restauraciones, brackets, movimiento), que degradan la segmentación necesaria para diagnóstico, planificación ortodóncica y restauración.
- **Metodología.** ToothSegNet "acostumbra" al modelo a casos defectuosos:
  - Módulo de **simulación de degradación** que genera imágenes degradadas durante el entrenamiento.
  - Módulo de fusión multi-calidad con **channel-wise cross fusion (CCF)** para reducir la brecha semántica entre codificador y decodificador.
  - **Pérdida de restricción estructural** que refina la forma del diente predicho.
- **Resultados.** Produce segmentaciones más precisas y **robustas a artefactos metálicos y desenfoque**, superando a métodos SOTA de segmentación de imagen médica tanto en imágenes de alta calidad como degradadas.
- **Relevancia clínica.** Aumenta la aplicabilidad en escenarios clínicos reales, donde la calidad de imagen es heterogénea —un requisito para automatizar la construcción de modelos 3D de forma fiable.

---

## Bloque 3 — Reconstrucción 3D a partir de radiografía (reducir radiación y coste)

### Ma et al. (2024), *PX2Tooth: Reconstructing the 3D Point Cloud Teeth from a Single Panoramic X-ray*
- **Problema.** Reconstruir la anatomía 3D de la cavidad oral (normalmente en CBCT) a partir de una **única radiografía panorámica 2D (PX)** reduciría radiación y coste, pero los métodos previos eran poco fiables o se validaban en datasets minúsculos (23–50 casos), con imprecisión especial en raíz y ápice.
- **Metodología.** Marco de dos etapas (coautoría con Angelalign Technology, fabricante de alineadores):
  - **PXSegNet**: red de segmentación panorámica tipo U-Net que clasifica los dientes de la PX en 32 categorías según la numeración FDI, aportando posición, morfología y categoría.
  - **TGNet** (Tooth Generation Network): transforma nubes de puntos aleatorias en dientes 3D, integrando la información segmentada mediante un **Prior Fusion Module (PFM)** que mejora especialmente la región del ápice radicular.
  - Dataset propio de **499 pares CBCT–PX** (proyectando las CBCT a panorámicas 1:1), anotado diente a diente por profesionales; partición 8:1:1.
- **Resultados.** Alcanza un **IoU de 0.793**, superando notablemente a métodos previos, con reconstrucciones 3D de contorno suave.
- **Relevancia clínica.** Abre la vía a modelos 3D de bajo coste y baja radiación a partir de una modalidad 2D ubicua, útil como entrada preliminar de planificación; aún debe mejorar la fidelidad radicular para uso clínico pleno.

---

## Síntesis transversal y lectura para clínica de alineadores

1. **Madurez por etapa.** La segmentación de coronas sobre escaneo intraoral es la etapa más madura (DSC ~0.96, inferencia sub-segundo) y prácticamente lista para producción. La localización de landmarks alcanza errores sub-milimétricos (~0.6 mm) próximos al estándar ABO. La segmentación de raíces en CBCT progresa con estrategias de forma y semisupervisión, y la reconstrucción 3D desde 2D es prometedora pero aún imprecisa en el ápice.
2. **Tendencias metodológicas.** (a) Deep learning sobre malla/nube de puntos (EdgeConv, PointNet, PointMLP) en vez de convertir a "imágenes"; (b) enfoques de dos/multi-etapas con ROIs y prompts por centroide; (c) fusión de conocimiento anatómico explícito (relaciones entre dientes vecinos, SDM/level-set, priors de forma); (d) robustez a datos reales (degradación de imagen, domain shift, semisupervisión); (e) eficiencia/despliegue en edge (compromiso resolución-precisión).
3. **Vacío detectado.** Ninguno de los papers descargados aborda de forma directa la **predicción automática del resultado del tratamiento**, el **setup automático (rearreglo dental objetivo)** ni la **colocación de attachments/staging** con IA. En la búsqueda en repositorios abiertos (arXiv), esa literatura apenas aparece porque es predominantemente clínica y se publica en revistas dentales indexadas fuera de arXiv; solo emergieron trabajos de simulación biomecánica (HOSEA, ROSS, FEM), no de IA en sentido estricto. Es la frontera pendiente para próximas ediciones de estos "research highlights".
4. **Consideración regulatoria/clínica.** Los criterios ABO (0.5–2 mm) proporcionan un marco objetivo de aceptabilidad; la robustez a artefactos (ToothSegNet) y la generalización entre escáneres (Wang et al.) son requisitos prácticos para el despliegue seguro en consulta.

## Puntos clave

- Segmentación intraoral (Wu et al. 2021, TS-MDL/iMeshSegNet): DSC 0.964 y MAE de landmarks 0.597 mm; EdgeConv acelera ~20× el entrenamiento; los autores citan explícitamente la fabricación de clear aligners como caso de uso.
- Resolución de escaneo (Weekley et al. 2025, PointMLP sobre Teeth3DS): 6K celdas es el umbral práctico; por debajo el DSC se desploma (~0.67). Guía para despliegue en dispositivos edge.
- CBCT semisupervisado (Wang et al. 2024): marco multi-etapa que reduce la dependencia de anotaciones y mitiga el domain shift; 3er puesto en el reto STS-3D.
- CBCT con preservación de forma (Ji et al. 2025): prompt por centroide + Signed Distance Map en arquitectura multi-tarea; resuelve adherencias entre dientes vecinos y preserva morfología radicular.
- ToothSegNet (Liu et al. 2023): robusto a artefactos metálicos y desenfoque mediante simulación de degradación + fusión cruzada por canales + pérdida de restricción estructural.
- PX2Tooth (Ma et al. 2024, con Angelalign): reconstrucción 3D de dientes desde una sola radiografía panorámica (PXSegNet+TGNet, IoU 0.793, dataset de 499 pares CBCT-PX); reduce radiación y coste.
- Vacío claro en la literatura abierta: predicción de resultado, setup automático y colocación de attachments/staging con IA apenas están cubiertos en arXiv (viven en revistas clínicas).
- Tendencias comunes: DL sobre malla/nube de puntos, pipelines multi-etapa con ROIs, incorporación de priors anatómicos y robustez a datos clínicos reales.
