# Capas de densidad por HU en producción: qué se cableó y por qué se revirtió

**Estado**: experimento cerrado y **revertido**. El pipeline no lleva capas de densidad; el
contenedor vuelve al campo único. Esto queda como registro de lo que se midió y se aprendió.

## Qué se hizo

Se portó el entrenador volumétrico del experimento a producción:

- `scripts/refina_3dgs.py`: DRR **perspectiva** (integral de línea por rayo), rasterizador
  **gsplat**, σ en **log**, y la **calibración global de σ** que el doc principal marca como
  el salto 19→43 dB. Sustituyó un splat ortográfico hecho a mano que no reproducía la
  calidad del experimento.
- un módulo `bandas` en `gaussian-engine` (`siembra_por_banda`, partición por tramos
  de HU), que el revert eliminó y por eso ya no se cita por su ruta.
- `--capas-hu` en `caso_completo.py`: siembra N campos, entrena cada uno contra la DRR de
  **su** tramo y los emite como capas externas del `.uos` (spec v0.2 §"External Gaussian
  layers").

## Tres bugs que hubo que cazar

1. **μ ≠ σ** — el DRR integraba `(hu+1000)/1000` y la semilla normaliza `(hu−300)/1700`:
   un offset que ningún escalar global corrige. Se igualaron (el DRR usa la misma
   normalización que la semilla).
2. **Escalas paso-compensadas** — la semilla partía con escalas de vóxel completo; el
   entrenamiento necesita **medio vóxel** para que el optimizador las crezca y llene los
   huecos del submuestreo.
3. **Eje z** — la semilla usaba el `z` real de los cortes (ImagePositionPatient, ~0,246 mm)
   y el DRR la rejilla nominal (0,45 mm): una desalineación de ~1,8×. Se unificó a la
   rejilla nominal, como hace el experimento.

## Resultado medido (sobre el caso clínico real)

Campo único **38,04 dB** (el control del experimento daba 37,35). Cuatro capas, de la más
densa a la menos: **50,5 / 41,3 / 36,8 / 34,6 dB**. El port reproduce (y algo mejora) las
cifras del experimento.

## Por qué se revirtió

La partición es **por densidad sobre el cráneo entero**, no por tejido. «densidad-muy-alta»
es esmalte **y** metal **y** cualquier cosa que atenúe por encima de 2000 HU; «densidad-baja»
es tejido blando y hueso trabecular. Encender o apagar una capa apaga «todo lo que atenúa
tanto», no «el esmalte» ni «la dentina»: es **muda en anatomía**.

Lo que se buscaba —capas por tejido (esmalte, dentina, hueso cortical/trabecular), como las
que permite un mapa de etiquetas multi-clase— depende de una **segmentación multi-clase**
que el caso real no tiene. Solo hay un segmentador **binario** de diente, y medido con él las
capas por tejido **restan** (−1,25 dB) en vez de sumar. Además el CBCT es de **cabeza entera**,
no recortado a la dentadura.

## Lo que haría falta para capas por tejido en el caso real

1. **Segmentación multi-clase** del CBCT (esmalte / dentina / hueso cortical / trabecular),
   no binaria. Es el techo: la calidad de la capa por tejido es la de su segmentación de
   entrada, y eso es lo que hoy falta.
2. **Recorte a la dentadura** del volumen, para que la escena sea la boca y no la cabeza.

Sin esas dos, la versión honesta es la partición por densidad — que es lo que aquí se midió y
se deja documentado, pero no entró en el pipeline por no cumplir el objetivo clínico de
encender/apagar por tejido.
