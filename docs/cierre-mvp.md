# Cierre del MVP — qué está medido, qué no está resuelto y qué queda para después

**Fecha: 2026-08-25.** Corresponde al hito de semana 8 del brief: «MVP testado, validación
preliminar con la organización partner, documentación técnica final».

Este documento es el inventario honesto. Lo que funciona va con su número; lo que no
funciona va con su número también, y con la causa medida cuando la hay. Un MVP que se
cierra diciendo solo lo que sale bien no sirve para decidir qué se hace después.

---

## 1 · Las cuatro cifras del brief

Medidas con `scripts/metricas.py`, no prometidas.

| Compromiso | Objetivo | Medido | |
|---|---|---|---|
| Latencia de ingesta de un conjunto completo | < 60 s | **12,7 s** | cumple |
| Fidelidad de la malla regenerada desde el gemelo | < 0,1 mm | **4,59 × 10⁻⁶ mm** | cumple |
| Cobertura de pruebas | > 80 % | **95,1 %** | cumple |
| Fiabilidad de los agentes de ingesta | > 95 % | **93,8 %** (N = 16) | **no cumple** |

**Sobre la reversibilidad.** El error que queda es el `float32` del propio STL: el
`mesh-agent` guarda posiciones en `float64` y la topología completa de caras, así que la
regeneración no reconstruye, **devuelve**. No hay marching cubes en ningún punto del
camino, y eso es deliberado.

**Sobre la fiabilidad, que es la que falla.** El fallo es un informe clínico escaneado
**sin capa de texto**: el agente no extrae nada y se declara `FAILED`, que es lo correcto.
Con N = 16 casos reales, un solo fallo son 6,2 puntos. Es decir: la cifra está por debajo
del objetivo por un caso, y arreglarlo es OCR, no arquitectura. Se declara así en vez de
subir el N con casos sintéticos hasta que el porcentaje quede bien.

**Y una que no es del brief pero es la que sostiene el resto:** el `mesh-agent` sobre
**120 mallas** de Teeth3DS+ da **100 %**. Las dos cifras van con su N al lado a propósito:
un N pequeño declarado es defendible; escondido detrás de un porcentaje, no.

---

## 2 · Lo que está terminado y medido

**El contenedor.** Un caso real cierra en **419 entradas, 397 cortes DICOM, conformidad
UOS-Core + UOS-Vol, 0 errores y 0 avisos, 18 vistas**. La serie CBCT entra byte a byte con
hash por corte, y hay tests que lo comprueban quitando un corte, colando uno de más y
alterando uno.

**El pipeline de agentes.** Ingesta de las cuatro modalidades → fusión geométrica (ICP,
rms 0,67 mm) → fusión semántica → segmentación → composición → exportación, ejecutable de
principio a fin sobre un caso clínico real con un comando.

**El compuesto imprimible.** Arcada cerrada como sólido estanco (12.025 caras de cierre,
11.936 mm³, 3,2 s) más un STL por pieza con corona medida y raíz reconstruida. Cada
fichero declara su procedencia en la cabecera del propio STL.

**El visor.** Abre un `.uos` en el navegador sin subir nada: malla con color anatómico,
capas del campo gaussiano, vistas con nombre, ficha por pieza y encendido de la pieza
seleccionada.

**La disciplina de resultados negativos.** Tres cosas se implementaron, se midieron, no
funcionaron y **se retiraron en vez de enviarse**: el eje apical por pieza (60° de
desviación), el segundo segmentador de CBCT (ganaba el banco de pruebas y perdía la tarea
real), y el relleno de huecos por distancia (funcionaba sobre datos reales por una razón
circular, que es la peor clase de acierto). Están documentadas con su medida.

---

## 3 · Lo que no está resuelto

### 3.1 · La segmentación FDI — el hueco principal

**11 de 14 piezas se pueden descartar por anatomía. Cota superior de correctas: 21 %.**
Ficha completa con la tabla, los dos criterios y las causas medidas en
`docs/research/segmentacion-fdi-escaner.md`.

En una línea: el modelo se entrenó sobre Teeth3DS+ (0,932 de FDI por diente **en su
propio test**), aquí no hay etiquetas de estos pacientes con las que medir acierto, y el
error dominante no es el contacto interproximal sino que **la corona se come el margen
gingival y la encía adherida**. Se ve como «no toda la encía se colorea»: no está sin
colorear, está coloreada de diente.

Lo que falta no es afinar un umbral: es **el color**, que es la señal con la que un
clínico ve dónde acaba el esmalte y que hoy no llega al modelo. Y está dentro del propio
contenedor, en las fotos clínicas — medido en `docs/research/frontera-encia-desde-foto.md`.

### 3.2 · El volumen se ve como puntos sueltos, y es un fallo de submuestreo

Medido, y con una causa concreta que **no** es una limitación de fondo:

- el `cbct-agent` siembra σ = **medio vóxel en cada eje** — (0,075, 0,075, 0,225) mm sobre
  un vóxel de 0,15 × 0,15 × 0,45. Eso es correcto;
- pero el volumen trae ~**12 millones** de vóxeles de tejido duro y el tope son 1,5 M, así
  que el agente submuestrea con `occupied[::step]`, `step = 9`, y quedan 1.341.421;
- `[::9]` recorta sobre un array en **orden raster**, así que se come ocho de cada nueve
  **a lo largo de un solo eje**. Medida la separación entre gaussianas consecutivas dentro
  de una fila: **1,35 mm en el 73 % de los casos**;
- luego **σ/separación = 0,056 en ese eje**: las gaussianas son dieciocho veces más
  pequeñas que su propia separación. Sondeado dentro del hueso más denso, el campo va de
  1,21 en el centro de una gaussiana a 0,016 entre ellas — **rizado del 99 %**.

Es decir: el campo exportado no es una nube, son **planos densos separados 1,35 mm**. Y no
es el precio de que el dato sea medido: **es que el submuestreo es anisótropo por
construcción y σ no se reescala con él.**

Dos arreglos, los dos pequeños:

1. **Diezmar en el espacio y no en el raster.** Quedarse con uno de cada dos vóxeles *en
   cada eje* es el mismo 8:1 pero isótropo: 0,30 mm en los tres ejes en vez de 1,35 en uno.
2. **Escalar σ con el factor de diezmado** (×`step`^⅓). Hoy no crece nada.

Con las dos, σ/separación vuelve a ~0,5 y el campo se lee como volumen **sin dejar de ser
una medida**.

⚠️ **Y una consecuencia de conformidad, aparte del aspecto:** el campo que viaja en el
contenedor **no es el volumen, es una submuestra de uno de cada nueve**. El agente lo sabe
—baja su confianza a 0,9 por ello— pero el sidecar `.gs.json` **no lo declara**. Quien mida
sobre ese campo está midiendo una submuestra sin que el fichero se lo diga.

En el visor, mientras tanto, cada sprite se dibuja inflado hasta el espaciado medido de la
nube. Es un apaño, y encima insuficiente: ese espaciado (0,212 mm) es la distancia al
vecino de otra fila, no el hueco real de 1,35 mm.

### 3.3 · La raíz reconstruida es un bulto

~2.000 gaussianas por diente a través de un complejo alfa. Sirve para imprimir un
contexto; no sirve para medir. Y 12 de las 14 raíces salen más largas de lo que su tipo
admite: está **declarado**, no corregido, porque recortarlas por longitud anatómica es un
*prior* y no una medida, y medir longitud radicular sobre el resultado sería medir lo que
se ha supuesto.

### 3.4 · Lo demás, corto

- **`value_range` sigue a `null`**, con su aviso. La alternativa medida es barrer 259 MB
  por exportación. Se cierra cuando entre el renderizador volumétrico (§3.2).
- **`verified_by` del registro es `null`** en todos los casos: nadie ha firmado ninguno.
  El visor lo pinta como «sin verificar», que es lo que es.
- **El margen gingival no está anclado** a la cresta de curvatura cervical. La limpieza
  tiene prohibido moverlo, y esa prohibición es correcta.
- **Ninguna cifra de acierto clínico existe**, porque no hay verdad de campo de estos
  pacientes. Todo lo que se publica son cotas superiores o consistencias.

---

## 4 · Qué haría falta para lo siguiente, en orden

El orden importa: los tres primeros desbloquean cosas que hoy no se pueden ni medir.

1. **La frontera diente-encía, del color de las fotos clínicas.** Desbloquea §3.1, y es
   la vía **medida**: un umbral de Otsu sobre `a*` separa diente de encía con 3,4–4,3 σ en
   las cuatro fotos de arcada que el contenedor **ya lleva**, y dibuja el festoneado
   cervical pieza a pieza. Lo que falta no es el color: es la **pose de cámara**, que hoy
   va a `projection: null` con el campo ya definido en el esquema. Ficha:
   `docs/research/frontera-encia-desde-foto.md`.
2. **Anotar diez arcadas propias.** Sin esto no habrá nunca una cifra de acierto, solo
   cotas. Cambia más que otro entrenamiento sobre Teeth3DS+ — y eso está medido en
   `docs/research/segmentacion-diente-cbct.md`, donde el modelo mejor perdió en la tarea
   real.
3. **Arreglar el submuestreo del campo** (§3.2): diezmar en el espacio en vez de en el
   raster, escalar σ con el factor, y declararlo en el `.gs.json`. Es pequeño y es la
   causa real de que el campo se vea como puntos sueltos.
4. **Renderizador volumétrico en el visor.** Desbloquea §3.4 y es lo que convierte el CBCT
   del contenedor en algo que un clínico mira en vez de un adorno.
5. Anclar el margen a la curvatura cervical (§3.4).
6. OCR para informes escaneados: es el único fallo de la cifra de fiabilidad (§1).
7. Segunda visita real, para ejercitar `visits[]` de verdad y convertir el seguimiento
   longitudinal en una resta.

---

## 5 · Qué se lleva alguien de aquí

Un formato de contenedor con esquema publicado, validador, procedencia encadenada y una
regla que se sostiene sola: **lo medido y lo inferido no se mezclan, y lo inferido se
puede borrar sin romper el caso**. Eso es lo que hace que una etapa que no funciona —hoy,
la segmentación— sea una pieza separable y no una contaminación del entregable.

Y un pipeline que llega de los ficheros crudos de una clínica a ese contenedor, con las
cuatro cifras del brief medidas, tres de las cuatro cumplidas, y la cuarta fallando por un
caso y por una razón nombrada.

Lo que no se lleva: un producto validado clínicamente. Nunca se prometió, y no lo hay.
