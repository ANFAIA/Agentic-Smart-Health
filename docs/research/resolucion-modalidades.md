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
| CBCT · y 3DGS desde él | 0,30 mm | 11 | 3 % | 20 % | 75 % | 99 % |
| IOS como se entrega | 0,23 mm | 19 | 10 % | 50 % | 98 % | 100 % |
| IOS al límite del escáner | 0,05 mm | 400 | 99 % | 100 % | 100 % | 100 % |
| Aleta de mordida | 0,033 mm | 918 | 100 % | 100 % | 100 % | 100 % |

Retener contraste es condición **necesaria, no suficiente**: aún hay que superar el
ruido.

## 4. Las dos fronteras, que no son la misma

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

## 5. Dos hallazgos laterales

**El aliasing depende de la orientación.** A 0,23 mm —el muestreo real de nuestras
mallas— la fisura de 0,15 mm se rompe en trozos inconexos, pero su tramo recto, que
cae alineado con la rejilla, sale entero. El mismo detalle se representa o se pierde
según cómo esté girado.

**Ninguna modalidad de superficie ve la lesión dentinaria**, ni a 0,05 mm. No es
resolución: está bajo esmalte intacto y un escáner óptico no lo atraviesa. Es la
justificación más limpia de por qué el gemelo necesita las dos ramas.

## 6. Entonces, por qué 3DGS

No por resolución. Por **asignación adaptativa de capacidad**: fino en la superficie
que mide el escáner, basto en el interior que solo mide el CBCT. Una rejilla de
vóxeles está obligada a elegir una única resolución para todo el volumen.

Coste de representación a fidelidad equivalente (arcada de 70 × 55 × 40 mm,
superficie 6000 mm², ocupación 20 %, interior siempre a 0,30 mm):

| Fidelidad de superficie | Rejilla uniforme | Rejilla dispersa | Campo adaptativo | Ventaja |
|---|---|---|---|---|
| 0,05 mm — *trueness* del IOS | 1232 M | 246 M | 3,5 M | **70×** |
| 0,10 mm — presupuesto del brief | 154 M | 30,8 M | 1,7 M | **18×** |
| 0,23 mm — muestreo real de nuestras mallas | 12,7 M | 2,5 M | 1,3 M | 2,0× |
| 0,30 mm — resolución del CBCT | 5,7 M | 1,1 M | 1,2 M | **0,9×** |

**La ventaja es del requisito, no de la representación.** A resolución de CBCT el
campo gaussiano *pierde* contra una rejilla dispersa. Toda la ganancia aparece al
exigir detalle de superficie más fino que el vóxel, y el cruce está en ≈ 0,25 mm.

Dos consecuencias incómodas y honestas:

- **Nuestros datos no ejercitan hoy esa ventaja.** Las mallas de Teeth3DS+ muestrean
  a ~0,23 mm, casi igual de basto que el CBCT, y ahí la ventaja es 2×, no 70×.
- **Estamos 14× por debajo.** El `cbct-agent` produce hoy 127 037 primitivas
  (resolución de CBCT, una gaussiana por vóxel ocupado); honrar el presupuesto de
  0,1 mm pediría ~1,7 M.

**Objeción que hay que anticipar:** un octree o hash-grid multirresolución capturaría
buena parte de esta ventaja. Lo que gana es la **adaptividad**, no «gaussianas». Lo
que las gaussianas añaden por encima son las otras dos propiedades del
[ADR 001](../architecture/001-digital-twin-core-schemas.md): **tres soportes en un
mismo primitivo** (σ volumétrico, color superficial, `region_id` semántico) y ser
**diferenciable**.

## 7. Y para el caso de la caries

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
