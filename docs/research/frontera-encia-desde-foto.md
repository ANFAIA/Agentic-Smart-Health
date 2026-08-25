# La frontera diente-encía está en la fotografía, no en la geometría

> **Estado (2026-08-25):** resultado **positivo y medido**, sin implementar. Un umbral de
> Otsu sobre el canal `a*` de las fotos clínicas que el contenedor **ya lleva** separa
> diente de encía con **3,4–4,3 σ** y dibuja el festoneado gingival pieza a pieza. No hay
> modelo, no hay entrenamiento y no hay etiquetas: es un umbral.

Es la contrapartida de `segmentacion-fdi-escaner.md`, donde 11 de 14 piezas se descartan
por anatomía. Allí la conclusión era que **la señal que falta es el color**; aquí se
comprueba que la señal existe, que ya está dentro del `.uos`, y cuánto separa.

## 1 · Por qué la geometría no puede

Un vértice del margen gingival y un vértice de cuello son —sobre la superficie— la misma
cosa: misma curvatura, mismas normales, misma vecindad. El modelo que etiqueta el escaneo
decide por vértice y sobre geometría, así que no está resolviendo mal un problema difícil:
está resolviendo un problema **que no tiene solución con esa entrada**.

En una boca esa frontera se ve por el color, y se ve inmediatamente.

## 2 · Lo que el contenedor ya trae

Un caso real lleva nueve imágenes en `images/`:

| | qué es | sirve para esto |
|---|---|---|
| 3 | radiografías periapicales / de aleta (RVG) | no — sin color |
| 2 | fotos laterales intraorales, 4752 × 3168 | **sí**, y son las del margen vestibular |
| 2 | fotos oclusales de maxilar y mandíbula, 4752 × 3168 | **sí**, mismo punto de vista que el escaneo |
| 2 | primeros planos de cámara intraoral, 1600 × 1200 | parcial — una pieza cada uno |

Cuatro fotografías de arcada completa a 15 Mpx. El escaneo intraoral tiene ~112.000
vértices: **hay dos órdenes de magnitud más de píxel que de vértice**.

## 3 · La medida

Conversión sRGB → CIE L\*a\*b\* y **umbral de Otsu sobre `a*`** (el eje rojo-verde). Nada más.

| foto | umbral `a*` | separación | fracción clara |
|---|---|---|---|
| oclusal maxilar | 17,1 | **3,37 σ** | 39 % |
| lateral derecha | 18,8 | **4,30 σ** | 40 % |
| lateral izquierda | 21,8 | **3,52 σ** | 42 % |
| oclusal mandibular | 20,0 | **3,80 σ** | 38 % |

La separación es la distancia entre las medias de las dos clases en desviaciones típicas
promediadas. Por comparación: 3 σ es el listón que se le pide a una señal para llamarla
detección, y esto lo pasa **en las cuatro fotos, con umbrales que caen dentro de 5 puntos
unos de otros** pese a ser tomas distintas con iluminación distinta.

Inspeccionada la máscara, el borde sigue el **festoneado cervical** diente a diente,
papilas interdentales incluidas. Es exactamente la curva que §6.2 de la ficha de
segmentación pedía y que la limpieza geométrica tiene prohibido inventar.

Los falsos positivos son tres y los tres son tratables: el **separador de plástico** (fuera
del arco), los **brillos especulares** sobre paladar y mucosa (alta `L*` y baja saturación,
o dos tomas con flash distinto), y algún reflejo en lengua.

⚠️ **Un umbral global no es el método final.** Que funcione con Otsu dice que la señal es
fuerte, no que el problema esté resuelto: cambia con la cámara, con el flash y con la
encía inflamada —que es rojo saturado, o sea un hallazgo clínico y no un artefacto—. Lo
que esta medida establece es que **merece la pena montar la tubería**, no que la tubería
sea un `if`.

## 4 · Lo que falta, que no es el color: es la pose

Para llevar la frontera de la foto a los vértices hace falta saber desde dónde se tomó la
foto. Hoy no se sabe: el manifiesto declara `projection: null` en cada imagen. El campo
**existe en el esquema** —está previsto— y está vacío.

Dos caminos, de menos a más:

1. **La oclusal, por registro 2D.** Una foto oclusal es casi una proyección ortográfica a
   lo largo del eje oclusal, que es un eje que ya se mide (`fusion_agents.marco`). Ajustar
   una homografía entre la foto y una vista oclusal renderizada de la malla es un problema
   pequeño y da color en las caras oclusales y en el margen palatino.
2. **Las laterales, por pose completa.** Estimar la pose de cámara contra renderizados de
   la propia malla (PnP sobre correspondencias, o ajuste diferenciable). Es lo que da el
   **margen vestibular**, que es el clínicamente importante, y es donde las laterales
   tienen 4,30 σ.

Y una vez hay pose, lo que se gana no es solo la frontera: es **color por vértice medido**,
con su procedencia — de qué foto vino cada vértice y con qué ángulo de incidencia.

## 5 · Por qué esto cambia el plan

En `docs/cierre-mvp.md` §4 el primer punto era «meter el color por vértice en la ingesta»,
pensando en los ficheros nativos del escáner, que hoy se leen como STL y pierden el color.
Sigue valiendo, pero **esta vía es mejor por tres razones**:

- **no depende del escáner.** El color nativo obliga a que el fichero venga del modelo de
  escáner que lo guarda; la foto la hace cualquier clínica y ya está en el caso;
- **tiene más resolución donde importa.** El color por vértice del escáner está limitado a
  ~112.000 muestras; las fotos aportan 15 Mpx cada una;
- **ya está dentro del contenedor.** No hay que cambiar la ingesta para conseguir el dato,
  solo para **usarlo** — y rellenar un campo del manifiesto que ya está definido.

Orden propuesto: (1) pose de la oclusal por registro 2D y rellenar `projection`;
(2) transferir color a los vértices con su procedencia; (3) anclar el margen a la frontera
de color; (4) reevaluar la segmentación FDI con `scripts/mide_segmentacion.py`, que ahora
mismo da 21 % de cota superior y es el número contra el que medir la mejora.

⚠️ Nada de esto convierte el color en un diagnóstico. La frontera medida es geometría, no
un hallazgo; el rojo saturado de una encía inflamada seguirá siendo cosa del informe.
