# Qué puede ver de verdad cada modalidad

**Pregunta.** ¿Cuánta resolución podemos obtener de CBCT, IOS y 3DGS, y por qué nos
es útil 3DGS si no la aumenta?

**Respuesta corta.** El poder resolutivo lo pone el dato, no la representación:
3DGS **no añade resolución** y no puede. Lo que aporta es repartir la capacidad
donde el dato la tiene, y eso solo compensa cuando las modalidades tienen
resoluciones distintas — que es exactamente nuestro caso.

Reproducible con `uv run python scripts/resolucion_modalidades.py`.
Versión visual: [`resolucion-modalidades.html`](resolucion-modalidades.html).

---

## 1. El montaje

Un parche de 9 × 9 mm de superficie oclusal con **tres hallazgos elegidos para que
cada uno viva en un soporte distinto** — la distinción que el contrato de
`core-schemas` ya codifica en su enum `Support`:

| | Hallazgo | Tamaño | Existe como |
|---|---|---|---|
| 1 | Fisura oclusal | 0,15 mm | geometría |
| 2 | Mancha blanca (caries incipiente) | 0,60 mm | color en superficie |
| 3 | Lesión dentinaria bajo esmalte intacto | 1,50 mm | solo densidad |

Cada modalidad se simula con su cadena física: **PSF gaussiana → muestreo a su paso
→ ruido**.

![Seis paneles: la misma superficie por seis cadenas de adquisición](resolucion-modalidades-paneles.png)

## 2. Lo que enseña el panel de 3DGS

Es el resultado central. El panel de «3DGS desde ese CBCT» sale **suave, sin
escalones y visualmente mejor** que el del CBCT — y contiene **exactamente la misma
información**: los mismos 11 puntos por mm², interpolados. Ni un hallazgo más.

Es la demostración de que **suavidad no es información**, y es la respuesta cuando
alguien pregunte si con Gaussian Splatting se vería mejor una caries.

Concuerda con lo ya medido en [`3dgs-volumetrico-cbct.md`](3dgs-volumetrico-cbct.md)
§3: doblar la resolución de render de 400 a 800 px **no mejora** la rama volumétrica
(43,1 dB en ambas) y **empeora** la fotométrica en 1,3 dB.

## 3. Contraste que sobrevive, por modalidad

![Curvas de contraste retenido frente al tamaño del detalle](resolucion-modalidades-curva.png)

| Modalidad | Muestreo | pt/mm² | 0,15 mm | 0,30 mm | 0,60 mm | 1,00 mm |
|---|---|---|---|---|---|---|
| CBCT · y el STL que sale de él | 0,300 mm | 11 | 3 % | 20 % | 75 % | 99 % |
| IOS Teeth3DS+ como lo tenemos | 0,230 mm | 19 | 10 % | 50 % | 98 % | 100 % |
| IOS comercial medido (Trios) | 0,156 mm | 41 | 27 % | 84 % | 100 % | 100 % |
| Aleta de mordida | 0,033 mm | 918 | 100 % | 100 % | 100 % | 100 % |

Retener contraste es condición **necesaria, no suficiente**: aún hay que superar el
ruido.

## 4. Cuántos puntos da cada una

La otra mitad de la pregunta: no qué detalle sobrevive, sino **cuántos datos hay**.
En píxeles, que es la unidad que sí se intuye — si desenrollaras la superficie y la
guardaras como imagen:

| | Paso | pt/mm² | ×CBCT | Cara oclusal (9×9 mm) | Arcada entera | MP | Resuelve |
|---|---|---|---|---|---|---|---|
| **Miden** | | | | | | | |
| CBCT | 0,300 mm | 11 | 1× | **30 × 30 px** | 258 × 258 px | 0,07 | 0,35 mm |
| **STL por marching cubes desde el CBCT** | 0,300 mm | 11 | 1× | 30 × 30 px | 258 × 258 px | 0,07 | 0,35 mm |
| IOS Teeth3DS+ como lo tenemos | 0,230 mm | 19 | 2× | 39 × 39 px | 337 × 337 px | 0,11 | 0,23 mm |
| IOS comercial medido (Trios) | 0,156 mm | 41 | 4× | 58 × 58 px | 497 × 497 px | 0,25 | 0,16 mm |
| Aleta de mordida | 0,033 mm | 918 | 83× | 273 × 273 px | 2347 × 2347 px | 5,51 | 0,03 mm |
| **Hereda** | | | | | | | |
| 3DGS · hoy, sembrado del CBCT | *0,300 mm* | *11* | *1×* | *30 × 30 px* | *258 × 258 px* | *0,07* | *0,35 mm* |
| 3DGS · tras fusionar la malla | *0,156 mm* | *41* | *4×* | *58 × 58 px* | *497 × 497 px* | *0,25* | *0,16 mm* |

Las dos últimas filas van **en cursiva y separadas** a propósito: 3DGS no mide nada,
así que esas cifras no son suyas — son las de la fuente con la que se le entrena. Es
la diferencia entre un instrumento y un envase.

Una **arcada entera** vista por el CBCT son 0,067 MP: cuatro veces menos que una
miniatura de WhatsApp, 720 veces menos que una foto de móvil. Y una cara oclusal de
molar son **30 × 30 píxeles**, un icono.

**Crece con el cuadrado del paso.** Afinar a la mitad multiplica los puntos por
cuatro; de 0,156 a 0,033 mm hay 4,7× de paso pero **22× de puntos**. En la cuenta de
la §9 el exponente es tres, no dos, y por eso allí los números explotan.

| Salto entre escalones consecutivos | Puntos | Resolución |
|---|---|---|
| CBCT / STL → IOS Teeth3DS+ | 1,7× | 1,5× |
| IOS Teeth3DS+ → IOS comercial | 2,2× | 1,5× |
| **IOS comercial → aleta de mordida** | **22,3×** | **4,7×** |

Y aquí está el resultado que no esperaba: **las tres modalidades 3D están agrupadas
entre 11 y 41 pt/mm²**, dentro de un factor de 4. El salto de verdad —22×— es hacia
la **aleta de mordida**, que es una radiografía plana de toda la vida.

Dicho de otro modo: en densidad de puntos sobre una superficie, CBCT, STL y escáner
intraoral **juegan todos en la misma liga**. Lo que cambia radicalmente entre ellos
no es cuántos puntos dan, sino **qué miden**: geometría, densidad o color.

### Qué significan las dos filas de 3DGS

**Hoy** el campo se siembra del CBCT: una gaussiana por vóxel ocupado, 127 037
primitivas. **Tras fusionar** la malla, la banda ε toma el detalle del escáner y la
cáscara sube a su resolución — que es el diseño del `geometric-fusion-agent`.

Y en los dos casos, **133 capas más hacia dentro** que ninguna malla de superficie
posee: el CBCT de una arcada de 40 mm a 0,30 mm son ~8,9 millones de vóxeles en
total, aunque sobre cualquier superficie concreta solo caigan 66 667.

Dicho en la unidad que se entiende: **el escáner es una foto de 0,25 MP de la
cáscara; el CBCT es un icono de 0,07 MP repetido 133 veces hacia dentro.** En puntos
de superficie apenas se llevan 3,5×, y sin embargo son cosas radicalmente distintas
— porque uno tiene color y el otro profundidad. El campo gaussiano es el único
formato de esta ficha que puede llevar las dos a la vez.

⚠️ **Pero hoy no las lleva.** El campo actual está a resolución de CBCT y la fusión
geométrica todavía no ha densificado la cáscara: de implementación, no de método —
pero está ahí.

## 5. El STL que sale del CBCT (marching cubes)

Marching cubes coloca los vértices sobre las **aristas de la rejilla**, así que la
distancia entre puntos del STL *es* el vóxel del CBCT. Medido sobre el fantoma
sintético del repo a cuatro resoluciones:

| Vóxel del CBCT | Vértices | Área extraída | Arista mediana |
|---|---|---|---|
| 0,40 mm | 20 760 | 2 491 mm² | **0,400 mm** |
| 0,30 mm | 36 980 | 2 483 mm² | **0,300 mm** |
| 0,20 mm | 83 980 | 2 492 mm² | **0,200 mm** |
| 0,10 mm | 336 494 | 2 485 mm² | **0,100 mm** |

No aproximadamente: **exactamente**. Y el área extraída se mantiene en ~2 485 mm² en
las cuatro filas, que es la comprobación de que se está midiendo la misma superficie
y solo cambia la teselación.

**La consecuencia incómoda:** puedes bajar el vóxel a 0,10 mm y obtener un STL con
puntos cada 0,10 mm, pero la resolución del CBCT sigue siendo 0,3–0,5 mm. Esos
336 000 vértices describen **interpolación, no medida**. Un STL denso sacado de un
CBCT es una malla suave, no una malla precisa — y es exactamente el error que
cometería quien intente cumplir el presupuesto de 0,1 mm subiendo la resolución de
reconstrucción.

## 6. Qué resolución dan los escáneres intraorales de verdad

Medido sobre arcada completa, con la misma definición que usamos aquí (número de
puntos ÷ superficie):

| Escáner | Resolución | Paso | Trueness | Precisión |
|---|---|---|---|---|
| Omnicam | 79,82 pt/mm² | 0,112 mm | 98,3 µm | ±261,8 µm |
| True Definition | 54,68 pt/mm² | 0,135 mm | 32,1 µm | ±98,8 µm |
| Trios | 41,21 pt/mm² | 0,156 mm | 55,3 µm | ±194,5 µm |
| iTero | 34,20 pt/mm² | 0,171 mm | 94,5 µm | ±246,8 µm |

Y *in vivo* sobre cinco escáneres: trueness **76,6 ± 79,3 µm**, precisión
**56,6 ± 52,4 µm**.

**Tres magnitudes que se confunden constantemente**, y conviene no mezclarlas:

| Magnitud | Qué es | Valor |
|---|---|---|
| Muestreo | cada cuánto hay un punto | 34–80 pt/mm² → 0,11–0,17 mm |
| Trueness | cuánto se desvía de la verdad | 32–98 µm |
| Precisión | cuánto varía entre escaneos | 56–262 µm |

Dos consecuencias:

- **La trueness de Trios (55 µm) es tres veces mejor que su paso entre puntos
  (156 µm).** Otra vez localizar frente a resolver: la superficie se conoce mejor de
  lo que sugiere la densidad de la malla.
- **No hay correlación significativa entre resolución y exactitud** en tres de los
  cuatro escáneres. Más puntos **no** dan mejor medida — nuestra tesis, confirmada de
  forma independiente por gente que no intentaba demostrarla.

Y sobre nuestros datos: Teeth3DS+ da **19 pt/mm²**, más basto que los cuatro
escáneres comerciales. Esas mallas están diezmadas, así que el 0,23 mm es una
propiedad del dataset, no del aparato.

Fuentes: [Nedelcu et al., resolución y exactitud de cuatro escáneres](https://pmc.ncbi.nlm.nih.gov/articles/PMC5937957/) ·
[exactitud in vivo de cinco escáneres](https://pmc.ncbi.nlm.nih.gov/articles/PMC7940805/)

## 7. Las dos fronteras, que no son la misma

Aquí está la respuesta a «cómo se mide la calidad de un CBCT»: no es un número,
son cuatro ejes que se miden con fantomas distintos, y **fallan en regímenes
distintos**.

- Un detalle de **1 mm conserva el 99 %** de su contraste y aun así puede no verse:
  693 HU sobre un ruido de 200 HU dan CNR 3,5, por debajo del umbral de Rose.
  **Falla el ruido**, y se arregla con dosis.
- Un detalle de **0,15 mm conserva el 3 %**. **Falla la resolución**, y no hay dosis
  que lo arregle: la información se perdió al emborronarse.

La frontera entre ambos regímenes cae alrededor de **0,6 mm**.

Corolario clínico, que conviene decirle al partner: **que algo no salga en la imagen
no significa que no esté.** Una cortical más fina que la resolución desaparece del
reconstruido y se lee como dehiscencia.

## 8. Dos hallazgos laterales

**El aliasing depende de la orientación.** A 0,23 mm —el muestreo real de nuestras
mallas— la fisura de 0,15 mm se rompe en trozos inconexos, pero su tramo recto, que
cae alineado con la rejilla, sale entero. El mismo detalle se representa o se pierde
según cómo esté girado.

**Ninguna modalidad de superficie ve la lesión dentinaria**, por fina que sea. No es
resolución: está bajo esmalte intacto y un escáner óptico no lo atraviesa. Es la
justificación más limpia de por qué el gemelo necesita las dos ramas.

## 9. Entonces, por qué 3DGS

No por resolución. Por **asignación adaptativa de capacidad**: fino en la superficie
que mide el escáner, basto en el interior que solo mide el CBCT. Una rejilla de
vóxeles está obligada a elegir una única resolución para todo el volumen.

Coste de representación a fidelidad equivalente (arcada de 70 × 55 × 40 mm,
superficie 6000 mm², ocupación 20 %, interior siempre a 0,30 mm):

| Fidelidad de superficie | Rejilla uniforme | Rejilla dispersa | Campo adaptativo | Ventaja |
|---|---|---|---|---|
| 0,10 mm — presupuesto del brief | 154,0 M | 30,8 M | 1,74 M | **17,7×** |
| 0,156 mm — IOS comercial medido | 40,6 M | 8,1 M | 1,39 M | 5,8× |
| 0,23 mm — nuestras mallas Teeth3DS+ | 12,7 M | 2,5 M | 1,25 M | 2,0× |
| 0,30 mm — resolución del CBCT | 5,7 M | 1,1 M | 1,21 M | **0,9×** |

**La ventaja es del requisito, no de la representación.** A resolución de CBCT el
campo gaussiano *pierde* contra una rejilla dispersa. Toda la ganancia aparece al
exigir detalle de superficie más fino que el vóxel, y el cruce está en ≈ 0,25 mm.

Dos consecuencias incómodas y honestas:

- **Nuestros datos no ejercitan hoy esa ventaja.** Las mallas de Teeth3DS+ muestrean
  a ~0,23 mm, casi igual de basto que el CBCT, y ahí la ventaja es 2×. Ni siquiera
  con un escáner comercial (0,156 mm) pasa de 5,8×.
- **Estamos 14× por debajo.** El `cbct-agent` produce hoy 127 037 primitivas
  (resolución de CBCT, una gaussiana por vóxel ocupado); honrar el presupuesto de
  0,1 mm pediría ~1,74 M.
- **Y el 17,7× depende del presupuesto del brief, no de ningún aparato.** Ningún
  escáner del mercado entrega 0,1 mm de muestreo: el requisito es de *exactitud*
  (trueness 32–98 µm), que es otra magnitud.

**Objeción que hay que anticipar:** un octree o hash-grid multirresolución capturaría
buena parte de esta ventaja. Lo que gana es la **adaptividad**, no «gaussianas». Lo
que las gaussianas añaden por encima son las otras dos propiedades del
[ADR 001](../architecture/001-digital-twin-core-schemas.md): **tres soportes en un
mismo primitivo** (σ volumétrico, color superficial, `region_id` semántico) y ser
**diferenciable**.

## 10. Y para el caso de la caries

El valor del gemelo no es ver mejor, sino **correlacionar señales débiles de
soportes distintos sobre el mismo diente**:

| Señal | Modalidad | Qué dice sola |
|---|---|---|
| Cambio de color en superficie | IOS / foto 2D | Muchas cosas manchan el esmalte |
| Densidad ligeramente baja | CBCT | CNR 1,9 — **no es evidencia** |
| pH 5,1 en esa zona | Informe | Regional, favorece desmineralización |

Ninguna basta. Las tres sobre el mismo `region_id` sí son un hallazgo candidato
razonable — que es exactamente el alcance del `pathology-agent` y para lo que sirve
el ancla FDI del `segmentation-agent`.

Nota de encuadre: **para caries, el CBCT es la herramienta equivocada.** Una aleta de
mordida trabaja a ~15 pl/mm frente a 1–2 pl/mm del CBCT: un orden de magnitud.

---

## Estado epistémico

| Afirmación | Estatus |
|---|---|
| Contraste retenido según tamaño | **Derivado** — forma cerrada, sin ajuste |
| pt/mm² por modalidad | **Derivado** — 1/h² |
| Coste de representación | **Derivado** — aritmética sobre geometría medida |
| 127 037 primitivas del `cbct-agent` | **Medido** en el caso sintético del repo |
| Muestreo de 0,23 mm del IOS | **Medido** — 116 k vértices de Teeth3DS+ sobre ~6000 mm² |
| PSF 0,35 mm y ruido 200 HU | ⚠️ **Supuestos plausibles, no calibrados** contra equipo |
| Superficie de arcada 6000 mm² | ⚠️ **Estimada** — medirla sobre Teeth3DS+ está pendiente |
| Resolución efectiva 0,3–0,5 mm | ⚠️ **Rango de literatura**, verificar por equipo |

La forma de las curvas es robusta frente a los supuestos; los valores absolutos
habría que calibrarlos con un fantoma real antes de citarlos en la memoria.

## Pendiente

- Medir la superficie real de arcada sobre Teeth3DS+ (bloqueado: el dataset no está
  en todas las máquinas).
- Preguntar al partner si su equipo puede **exportar las proyecciones crudas**. Es lo
  único que subiría el techo del CBCT: entrenar contra las proyecciones en vez de
  contra DRR del volumen ya reconstruido evita la pérdida de la reconstrucción FDK,
  que es lo que hace RGS. Con DICOM de volumen reconstruido, el 0,3–0,5 mm es
  definitivo.
