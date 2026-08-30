# La resolución del campo de apariencia: qué se intentó, qué falló y por qué

> **Estado (2026-08-28):** búsqueda **cerrada con medidas**. Las tres palancas —entrenamiento,
> display y textura de las fotos— quedan falsificadas, cada una con su criterio fijado
> ANTES de correr y su fallo declarado con esas palabras. El campo es fiel (36 dB) pero
> suave, y el criterio de nitidez (huella < 563 µm) es inalcanzable con esta maquinaria
> sobre este contenido.

## 1 · El problema, medido

El campo gaussiano de apariencia de la arcada (113.000-1.090.000 gaussianas según la
siembra) reproduce sus vistas de entrenamiento a 34-36 dB de PSNR y, aun así, se ve sin
nitidez. La medida de "nitidez" del proyecto es la **huella** —el eje mayor de la
gaussiana mediana, en µm— con criterio deseado **< 563 µm**. El campo real mide:

| | valor |
|---|---|
| huella mediana (eje mayor) | **845,9 µm** |
| gaussianas a >1 mm de la superficie | 46,5 % |
| distancia mediana a la malla | 0,911 mm |
| anisotropía (mayor/menor) | 3,0× |

**Causa raíz, no conjetura:** la entrada de color es albedo plano —39 números (13 coronas
× 3 tercios + encía) pintados sobre la malla—. Con esa entrada, el loss de imagen NO
distingue un disco grande de una mancha pequeña: la neblina (gaussianas difusas y
traslúcidas que promedian el color) es el óptimo del problema, y el entrenamiento lo
encuentra **sea cual sea** la capacidad, la estrategia, la resolución de render o la
supervisión añadida. Esa hipótesis se falsificó en tres formulaciones (abajo).

## 2 · Palanca 1 — entrenamiento: FALSIFICADA

Criterio fijado antes: **mediana de distancia < 1,572 mm Y huella < 563 µm** (un experimento
que empeora es un resultado, no un fracaso que maquillar). Corridas sobre el caso real:

| corrida | qué probaba | mediana | huella | PSNR | veredicto |
|---|---|---|---|---|---|
| `principal2` | peso-profundidad 0,5 + aplanado (ratio) | 0,660 mm ✓ | **1.344 µm ✗** | 26,16 | FALLIDA |
| `mcmc` | estrategia MCMC + antialiased + aplanado (eje menor absoluto) | 2,017 mm ✗ | 827 µm ✗ | 26,16 | FALLIDA (caja −445..+1849 mm) |
| `r2048` | renders a 2048 px + 1,09 M gaussianas | 1,591 mm ✗ | 906 µm ✗ | 34,15 | FALLIDA |

Detalles medidos que descartan más vías: el aplanado por ratio es degenerado (el gradiente
expande el eje mayor: huella 1.344 µm con menor de 3,4 µm); el por eje-menor absoluto
explota la nube (anisotropía 92×); el MCMC con `noise_lr` por defecto revienta el campo
(12 dB — el ruido de reubicación escala con el cuadrado de la escala); el render
antialiased aguanta 33,2 dB y **no cambia el óptimo**. La supervisión de profundidad se
implementó con raycast exacto sobre la malla (Blender 5.x ya no expone el Z-buffer) y
ancla bien (sesgo ≈ 0) — pero no da nitidez porque el color sigue siendo plano.

## 3 · Palanca 2 — display: medida, y retirada

Se probó supersampling de display en el visor. Los intentos intermedios estropeaban la
vista por un desync medido en la biblioteca de splats: el `SplatMesh` **cachea**
`devicePixelRatio` al construirse y con él calcula el `viewport` del shader, mientras el
`Viewer` lee el suyo **en vivo** para el focal; subir solo el de la biblioteca dibuja los
splats al doble de tamaño ("horrible"), y subir los dos también desincroniza (la malla
cacheó el valor del arranque). La forma correcta —solo el buffer del renderer, la
biblioteca quieta en `window.devicePixelRatio`— se implementó y se midió: **2× ≈ nativa,
sin diferencia visible**, porque el campo no tiene bordes duros que el supersampling
pueda limpiar. Conclusión aplicada: la función se **retiró** del visor (renderer a DPR
nativo, sin toggle), y queda la advertencia medida en el código para que nadie reintroduzca
el desync (`uos-viewer/src/app/Apariencia.ts`).

## 4 · Palanca 3 — textura real desde las fotos: FALSIFICADA

La única vía con fundamento: proyectar píxeles de las fotos intraorales sobre la malla
(necesita pose de cámara < 0,9 mm por foto — guard de fidelidad clínica: textura nítida
pero desplazada es errónea). Hoy 1 de 9 fotos daba pose (0,845 mm / 87 %).

**Fase A — diagnóstico:** los fallos de pose solo existían en stdout; se persistió
`pose_diagnostico.json` (motivo exacto + candidatos por combinación focal×ventana×sentido).
Causas medidas: una foto daba **18 blobs para 14 piezas** (sobre-segmentación — fallo duro
de ventanas); otra daba 99 candidatos con **5 inliers** (el mínimo son 6); otra estaba a
**0,939 mm** (cuatro centésimas del gate); dos son primeros planos sin arco segmentable.

**Fase B — mejora de pose** (criterio: ≥2 fotos más con error <0,9 Y apoyo ≥0,80 Y ≥6
inliers → **FALLIDO**, salió 1): el fundido de blobs próximos rescató la foto
sobre-segmentada (**0,562 mm / 87,1 %**, cobertura 64,6 % → **76,2 %**). El refinado por
silueta (`pose_refina.py`: chamfer bilineal superficie↔máscara + anclaje a las
correspondencias, L-BFGS-B escalado, multi-semilla × 2 sentidos) produce en las otras
fotos **poses degeneradas** —cobertura por blob ~0,2, residuo de patrón 30-130 px— que
el gate rechaza correctamente: el "apoyo 0,97" de una de ellas era un deslizamiento de la
nube sobre la máscara, no una pose. El suelo de apoyo 0,80 resultó además inalcanzable
para vistas parciales de la arcada. La semilla clínica que queda anotada: la foto lateral
muestra **8 piezas** (la segmentación veía 10) — un hint de piezas visibles por foto es la
siguiente iteración si se retoma.

**Fase C — reentrenar con la textura ganada** (criterio: huella <563 µm Y ΔE por pieza
≤2 → **FALLIDO**): reentrenado con las 2 fotos (76,2 % de cobertura), PSNR 35,97 dB,
huella **842,4 µm** — y la medición **regional** lo remata: la textura nueva cayó sobre la
zona oclusal y allí tampoco movió nada (870,6 → 866,9 µm). La neblina no responde a la
textura: el tamaño de las gaussianas lo fija la maquinaria, no la falta de datos.
(La pata ΔE del criterio estaba mal definida: comparaba contra los colores corregidos de
flash de `observations.json` mientras el campo lleva los píxeles crudos proyectados —
desfase conocido y declarado; no es una falla del campo.)

## 5 · Vías consideradas y descartadas SIN correr

- **Entrenar sobre recortes (crop) de la escena** — la técnica de "crop training" de
  KIRI Engine: entrenar solo una región del encuadre para que cada gaussiana disponga de
  más píxeles por milímetro. Se descartó sin correr porque su mecanismo ya quedó cubierto
  por las corridas falsificadas: `r2048` duplicó la resolución lineal de las vistas y 1,09 M
  de gaussianas repartidas por área (26/mm², el doble que el pipeline) y la huella no se
  movió (906 µm). El crop no añade señal —reparte la MISMA entrada en más gaussianas—, y
  con albedo plano no hay detalle que repartir. El único matiz pendiente: un crop sobre la
  zona con píxeles de foto (oclusal) es la combinación que NO se corrió, pero la medición
  regional de la fase C (textura presente y huella intacta) la hace improbable.
- **2D Gaussian Splatting** — excluido por formato desde el principio: `KHR_gaussian_splatting`
  define `SCALE` como `VEC3`; el contenedor no podría declarar gaussianas 2D sin romper el
  estándar abierto que este proyecto persigue.

## 6 · Conclusión y qué queda en pie

**563 µm es inalcanzable con esta maquinaria sobre este contenido.** El campo de
apariencia es lo que es: una reproducción fiel (36 dB, ΔE 0,35 contra sus propias vistas)
pero suave de la arcada coloreada. El formato (`KHR_gaussian_splatting`), el visor y la
reversibilidad no se ven afectados.

Lo que la fase deja aprovechable, todo con tests (80 pasando, ruff limpio):

- `pose_diagnostico.json` — los fallos de pose dejan de ser caja negra.
- Fundido de sobre-segmentación en `estima_pose` — una foto real más entra al color
  (cobertura 76,2 % al re-empaquetar).
- `pose_refina.py` — refinado por silueta con gate intacto; descarta lo degenerado.
- El hint clínico de piezas visibles por foto, si se retoma la pose.

Un experimento que empeora es un resultado: esto no se cerró "sin lograr la nitidez", se
cerró midiendo dónde está el techo.
