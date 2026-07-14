# DentalGS: Pose-Free 3D Gaussian Splatting a partir de cinco imágenes intraorales para síntesis de vistas nuevas

> **Fuente:** 11525-AAAI26.DaiH-CV.pdf — Dai, H.; Zhou, Y.; Wei, G.; Li, Z.; Wang, W. "DentalGS: Pose-Free 3D Gaussian Splatting from Five Intraoral Images for Novel View Synthesis." AAAI-26 (The Fortieth AAAI Conference on Artificial Intelligence), pp. 3461–3469, 2026.

## Abstract

DentalGS es un framework de 3D Gaussian Splatting (3DGS) diseñado para monitorización ortodóncica remota que sintetiza vistas nuevas realistas de la dentadura a partir de solo cinco fotografías intraorales post-ortodoncia (tomables con un smartphone) y datos de escaneo intraoral (IOS) pre-ortodoncia como conocimiento previo, sin necesidad de conocer las poses de cámara. El método aborda tres grandes retos de la síntesis de vistas nuevas (NVS) en escenarios dentales: fallos de estimación de pose por escaso solapamiento y superficies dentales lisas, geometría rota por el número mínimo de vistas, e inconsistencias de iluminación dinámica. Para ello combina un algoritmo iterativo de ajuste de pose de cámara guiado por priors anatómicos, una red de reparación (RepairNet) entrenada con una estrategia progresiva de generación de pares daño-reparación, y un Gaussian sensible a la iluminación basado en un modelo de reflectancia físicamente inspirado. Experimentalmente supera a los baselines de 3DGS de vistas escasas, destacando en SSIM (0.9358 con 5 vistas), preservando la estructura geométrica incluso bajo puntos de vista extremos, y ofreciendo una solución de bajo coste para el seguimiento ortodóncico a domicilio.

## Explicación completa

## Problema y motivación

El tratamiento ortodóncico requiere revisiones periódicas de la alineación dental, que hoy dependen de visitas a la clínica y de equipamiento profesional (fotografía intraoral, CBCT, escáneres intraorales/IOS) voluminoso, costoso y que exige personal entrenado. Esto encarece el seguimiento y limita la atención remota y el seguimiento a domicilio. Muchos pacientes querrían usar el smartphone para capturar fotos intraorales para revisiones a distancia, pero las imágenes 2D carecen de información espacial suficiente para un diagnóstico fiable.

Los avances en síntesis de vistas nuevas (NVS) —NeRF (Mildenhall et al. 2021) y 3D Gaussian Splatting (Kerbl et al. 2023)— permiten generar vistas 3D realistas a partir de imágenes 2D, lo que abriría la puerta a evaluación oclusal en casa. Sin embargo, aplicar NVS en el escenario dental es difícil por tres retos clave que identifican los autores:
1. **Poses de cámara poco fiables**: el bajo solapamiento entre imágenes y las superficies lisas de los dientes dificultan que Structure-from-Motion (COLMAP) extraiga características estables, provocando errores de pose.
2. **Geometría incompleta/rota**: con solo cinco imágenes intraorales es difícil construir un modelo 3D completo, lo que causa sobreajuste y geometría fragmentada.
3. **Iluminación dinámica**: el entorno oral es oscuro y depende de fuentes de luz externas, generando fuertes inconsistencias de iluminación entre vistas y artefactos visuales.

## Metodología

DentalGS es un framework 3DGS pose-free que usa cinco imágenes intraorales post-ortodoncia y datos IOS pre-ortodoncia como prior. La tubería se organiza en cuatro etapas:

**Etapa 1 – Estimación de pose de cámara (Iterative Camera Pose Fitting).** En lugar de COLMAP, que falla con vistas escasas, se aplica primero Multi-view Contour Fitting (MCF, Xie et al. 2024) para alinear las cinco imágenes post-ortodoncia con el IOS pre-ortodoncia y estimar una nube de puntos post-ortodoncia. A cada diente se le asignan etiquetas ISO-FDI, se calculan centros de píxel por diente por imagen y se refinan las poses mediante un algoritmo iterativo (Algoritmo 1) que usa PnP (Perspective-n-Point, EPnP de Lepetit et al. 2009) sobre correspondencias 2D-3D y construye un frustum de visión (rays) para hallar la primera intersección con la nube de puntos, recomputando centroides iterativamente. Esto suprime errores inducidos por oclusiones y mejora la robustez de la estimación.

**Etapa 2 – Estrategia progresiva de generación de pares (Progressive Pair Generation Strategy).** Como no existe un dataset dedicado para restauración de Gaussians, se genera de forma sintética. Las vistas se dividen en un Train Pool (inicialmente solo vistas superior e inferior) y un Candidates Pool. Tras optimizar un modelo 3DGS inicial G0 solo con el Train Pool, se simulan cuatro escenarios de degradación para construir pares de imágenes (renderizado degradado ↔ ground truth): 
- **Optimization Pairs**: capturan la trayectoria completa de restauración desde degradación severa hasta recuperación total (espectro fino de dificultad).
- **Correlation Pairs**: generados desde vistas espacialmente solapadas durante la optimización (dificultad moderada, se benefician de consistencia entre vistas).
- **Corrupted Pairs**: desde vistas no relacionadas sin solapamiento espacial (los más difíciles de restaurar).
- **Noisy Pairs**: añadiendo ruido a los atributos (posición, opacidad, color) de vistas ya optimizadas, para robustez ante perturbaciones.

**Etapa 3 – RepairNet.** Es una U-Net encoder-decoder con doble rama (una consciente de la iluminación y otra de reconstrucción de albedo). Para cada vista toma como entrada la imagen degradada G(v), el mapa de profundidad y el mapa de segmentación semántica (apariencia, geometría y contexto anatómico). Incorpora priors geométricos calculando mapas de pseudo-normales y gradiente de profundidad mediante filtros Sobel. El encoder tiene tres etapas de downsampling con Double Conv-group (DConv) y conexiones residuales; la primera rama del decoder integra los priors geométricos mediante un módulo de upsampling con atención, y la segunda (albedo) los excluye. Las salidas son un mapa de parámetros especulares (2 canales) y un mapa de albedo (3 canales), y el resultado sombreado final se renderiza con una formulación físicamente inspirada que combina términos ambiente, difuso, subsurface y especular Cook-Torrance, produciendo una imagen RGB restaurada que corrige distorsiones estructurales y recupera detalles de superficie.

**Etapa 4 – Gaussian sensible a la iluminación (Lighting-Aware Gaussian).** El 3DGS tradicional asigna a cada Gaussian coeficientes de armónicos esféricos (SH) que aproximan la radiancia direccional solo en función de la dirección de vista, independiente de la iluminación real o la geometría; esto produce artefactos flotantes ("ghost Gaussians") y highlights/sombras que aparecen incorrectamente en vistas no relacionadas. En su lugar, DentalGS (inspirado en Zhang et al. 2024, DarkGS) asigna a cada kernel Gaussian atributos de material: posición, covarianza, opacidad, albedo, normal, rugosidad, metalicidad y un bias de feature. La ecuación de renderizado usa una función de reflectancia fr en vez de SH, integrando la radiancia de entrada y las direcciones de luz.

**Función de pérdida.** Se introduce una supervisión conjunta que combina las vistas GT escasas con los renderizados densos restaurados por RepairNet: L1 pixel a pixel sobre las GT para preservar detalle fino; una pérdida estructural (Lstruct) con average pooling local y pérdida perceptual VGG (Lperc) sobre los renderizados restaurados enfatizando consistencia estructural/perceptual; y pérdida SSIM. La pérdida total es una combinación ponderada de los cuatro términos. Notablemente, la ablación muestra que conviene seleccionar adaptativamente qué pérdidas aplicar según la calidad de la imagen (imágenes de baja calidad ayudan a la similitud estructural global, pero sus artefactos pueden perjudicar la representación Gaussian).

## Datos y experimentos

Se construyó un dataset con 20 modelos 3D completos de mandíbula (superior e inferior) recogidos de varios hospitales mediante escáneres intraorales y almacenados como mallas de superficie. Los dientes se segmentaron con el método de Zhuang et al. 2023 y las fotografías de cinco vistas (5616×3744) se segmentaron semánticamente con SAM (Kirillov et al. 2023). Para simular el escenario de telesalud, se capturaron imágenes intraorales con smartphone dos años después de los escaneos iniciales (4096×3072). La inicialización geométrica usa MCF.

**Estimación de pose**: bajo perturbaciones controladas calibradas al rango clínico típico (niveles 1–3), el método logra un error de rotación medio de 0.4294° y de traslación de 1.2625 mm. Incluso con perturbaciones mayores, el error de traslación se mantiene ~5 mm y el de rotación dentro de 4°.

**Comparativa NVS** (Tabla 1): se compara con 3DGS, GaussianPro (GSpro), GaussianObject (GSOBJ) y ZeroRF, todos usando las poses estimadas por DentalGS. Con 5 vistas, DentalGS obtiene PSNR 26.20, SSIM 0.9358 y LPIPS 0.2689, frente a GSpro (19.42/0.8136/0.2877), GS (19.71/0.8083/0.2935), ZeroRF (15.06/0.7147/0.3691) y GSOBJ (20.91/0.8044/0.2438). Con 7 vistas alcanza PSNR 26.86 y SSIM 0.9381. Destaca especialmente en SSIM, evidenciando su superior preservación de la estructura geométrica.

**Hallazgo interesante**: aumentar el número de vistas de entrada puede degradar el rendimiento de otros métodos, porque introduce iluminación inconsistente (especularidades, sombras) que rompe la coherencia de los Gaussians; DentalGS se mantiene estable.

**UNet vs Difusión**: GSOBJ usa un modelo de difusión grande (SDM) para restauración y logra buen LPIPS por su fuerte prior perceptual, pero introduce inconsistencias estructurales entre vistas y de hecho baja el PSNR de 21.43 (pre-restauración) a 20.91 (post-restauración), revelando las limitaciones de los modelos de difusión en modelado de objetos rígidos como los dientes. La restauración basada en UNet de DentalGS mantiene mejor la consistencia geométrica y mayor SSIM.

**Ablaciones**: confirman que las pérdidas espacialmente conscientes mejoran LPIPS; que las entradas de RepairNet (imagen degradada, profundidad, mapa de clases) se complementan; y que dentro de márgenes de error aceptables las imágenes restauradas mantienen calidad (max L1 medio 0.0499, PSNR mínimo 23.38, SSIM mínimo 0.8959).

## Limitaciones

La principal es la dependencia de datos IOS predichos, que introducen errores respecto a los dientes reales, afectando a la estimación de parámetros de cámara y a las pistas de profundidad, y por tanto a la precisión de la NVS. Como trabajo futuro, al crecer el dataset, el modelo de restauración podría evolucionar a una red feedforward preentrenada que permita reconstrucción Gaussian de alta calidad sin reentrenamiento o con mínimo fine-tuning; y combinar DentalGS con métodos tipo GaussianAvatars (Qian et al. 2024) podría habilitar 3DGS dinámico y realista del propio proceso ortodóncico.

## Relevancia clínica

DentalGS ofrece una solución de bajo coste y fiable para la visualización 3D de dientes en monitorización ortodóncica remota. Al requerir solo cinco fotografías intraorales (capturables con un smartphone) más un escaneo IOS previo ya existente del paciente, y no necesitar poses de cámara ni equipos costosos, facilita el seguimiento a domicilio de la alineación dental, reduciendo la carga de visitas clínicas y ampliando el acceso a la atención. La preservación de la estructura geométrica incluso bajo puntos de vista extremos permitiría evaluaciones oclusales remotas más fiables que las fotos 2D.

## Puntos clave

- Framework 3DGS pose-free que sintetiza vistas nuevas de dentadura desde solo 5 fotos intraorales post-ortodoncia + datos IOS pre-ortodoncia como prior, sin poses de cámara.
- Iterative Camera Pose Fitting: usa MCF + etiquetas ISO-FDI + PnP y frustum de visión para estimar poses donde COLMAP falla (rotación media 0.4294°, traslación 1.2625 mm).
- Progressive Pair Generation Strategy: genera sintéticamente 4 tipos de pares daño-reparación (Optimization, Correlation, Corrupted, Noisy) para entrenar RepairNet ante la falta de dataset.
- RepairNet: U-Net de doble rama (iluminación + albedo) que usa imagen degradada, profundidad, segmentación semántica y pseudo-normales para restaurar renderizados fragmentados.
- Lighting-Aware Gaussian: sustituye los armónicos esféricos por atributos de material y una función de reflectancia físicamente inspirada (Cook-Torrance) para mitigar iluminación dinámica y eliminar 'ghost Gaussians'.
- Supera a 3DGS, GaussianPro, GaussianObject y ZeroRF: PSNR 26.20, SSIM 0.9358, LPIPS 0.2689 con 5 vistas, destacando en preservación estructural (SSIM).
- La UNet supera a la difusión (SDM) en modelado de objetos rígidos como dientes; SDM baja el PSNR tras restaurar (21.43→20.91).
- Aumentar vistas puede degradar otros métodos por iluminación inconsistente; DentalGS permanece estable.
- Limitación: dependencia de datos IOS predichos que introducen error en pose y profundidad.
- Relevancia clínica: monitorización ortodóncica remota de bajo coste con smartphone, sin equipos caros ni visitas frecuentes.
