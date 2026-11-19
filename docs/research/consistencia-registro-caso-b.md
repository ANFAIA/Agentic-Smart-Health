# Consistencia del registro escáner↔CBCT con dos escaneos: caso B

> **Estado (2026-11-19):** protocolo escrito **antes de registrar nada contra el CBCT**.
> Resultados en §5, pendientes. Esto mide la **precisión** del registro, no su exactitud:
> un sesgo que comparta los dos caminos (el mismo CBCT, el mismo algoritmo) se cancela y no
> aparece. No sustituye al TRE con puntos anatómicos marcados por una persona.

## 1 · La pregunta

El registro del caso B deja los dientes posteriores a 0,2–0,6 mm del esmalte del CBCT y los
incisivos y el 26 a 1–2 mm. Esa comprobación es visual y usa el mismo CBCT con que se
registró. ¿Cuánto cambia la posición que el registro asigna a cada diente si en lugar de un
escaneo se usa **otro escaneo independiente de la misma boca**?

Si dos escaneos A y B de los mismos dientes se registran por separado contra el CBCT
(`A→CBCT`, `B→CBCT`) y además entre sí (`A→B`), un punto `x` de una corona de A llega al
CBCT por dos caminos: `A→CBCT(x)` y `B→CBCT(A→B(x))`. Su distancia mide cuánto depende la
pose del escaneo concreto. El registro directo `A→B` es mucho más preciso (centésimas) y
fija el suelo de la medida.

## 2 · Los datos, y por qué este par

Hay cuatro escaneos superiores de este paciente. Se alinearon entre sí **antes** de
escribir este protocolo y sin tocar el CBCT, para saber cuáles son adquisiciones
independientes. Se usó la distancia exacta de punto a superficie: la distancia al vértice
más cercano se infla unos 0,2 mm por la resolución de la malla.

| Par (coronas) | Mediana | p95 | Escala relativa |
|---|---:|---:|---:|
| PREVIO → ESCANEADO PREVIO | **0,063 mm** | 0,59 | 0,999 |
| POST HIGIENE → Visualization 18-28 | 0,11 mm | 0,53 | 1,002 |
| POST HIGIENE → PREVIO | 0,53 mm | 2,28 | 0,976 |

Los números son los del procedimiento de §3 y en la dirección del par (A → B); en la
primera versión de esta tabla venían de una exploración previa con otro método (0,052).

- **Par principal: PREVIO y ESCANEADO PREVIO.** La ficha
  [`dataset-histora.md`](dataset-histora.md) los documenta como dos adquisiciones
  independientes de la misma sesión, y encajan a centésimas.
- **Par secundario, solo descriptivo: POST HIGIENE y Visualization 18-28.** No se sabe qué
  es el segundo: el repositorio lo llama «derivado del CBCT» en un sitio y «réplica virtual
  del molde» en otro, y es un sólido cerrado. Si sale del propio CBCT o de POST, el par no
  es independiente.
- **PREVIO y POST no tienen la misma forma.** Un ajuste con escala da 0,976, pero no es una
  escala uniforme: las distancias entre centroides de dientes van de 0,91 a 1,02 de una a
  otra (11–21: 0,915; 16–26: 0,965; 12–22: 1,019). La discrepancia se concentra en el 11 y
  el 26. No se sabe por qué. Lo que comparten los dos escaneos del par se cancela, pero la
  precisión que se mida es la de escaneos de la sesión PREVIO y **no se traslada a POST**.
- **No se sabe de qué sesión es el CBCT.**
- POST HIGIENE se registra además para comprobar que el procedimiento es el de la
  ejecución de referencia (§3, V1).

## 3 · Procedimiento, fijado antes de ejecutar

**Registro de cada escaneo contra el CBCT.** El del pipeline, como en
`caso_completo.py` y en `run-5-modelo`:
- `IngestionPipeline.run` con recorte dental y tope de 1.500.000 primitivas;
- `fuse(malla=…, dos_arcadas=True)`;
- serie densa, así que el camino de siempre con árbitro.

Cada escaneo entra en su propia carpeta de caso con el mismo nombre de malla
(`escaneo-superior.stl`), para que la arcada se lea igual («maxilar»). Sin segmentación
y sin ajuste de campo, que corren después del registro y no lo cambian. Mismo CBCT
(`CASE-64A77C`), semilla 0, entorno `~/.venvs/dental-gpu` como run-5.

**Coronas de cada escaneo.** Solo POST tiene etiquetas FDI. Primero se limpian como hace
`caso_completo.py` antes de usarlas: se rellenan huecos y se absorben islas. Eso quita el
«17» de 198 vértices que es parte del 16, como en run-5. Para los demás escaneos se
transfieren:
- POST se alinea al otro escaneo con un ajuste de similitud (con escala) sobre el 70 % de
  correspondencias más cercanas. Se arranca de dos poses, el ICP desde la identidad y el
  barrido global de SO(3) (`icp_global`), y se queda la de menor residuo. Medido al
  explorar: para Visualization solo funciona el barrido, y para PREVIO solo la identidad
  (el barrido elige una pose volteada);
- cada vértice del otro escaneo a menos de 0,5 mm de una corona de POST toma su código.

Las etiquetas solo sirven para agrupar por diente y elegir los puntos de corona; no entran
en ningún registro contra el CBCT.

**Registro directo A→B.** ICP rígido de las coronas de A contra la malla de B:
- desde las mismas dos poses de partida, cada una refinada con recortes 0,8 y 0,5 a escala
  completa;
- se queda la de menor mediana de distancia exacta a la superficie de B, y esa distancia
  es el suelo;
- si la mediana pasa de 0,2 mm, el par se declara no interpretable. No se prueba ningún
  otro método.

**La medida.** Para cada vértice de corona `x` de A:
`d(x) = ‖A→CBCT(x) − B→CBCT(A→B(x))‖` en mm DICOM. Se informa:
- por diente, mediana y p95 de `d`, su suelo y su **cobertura**: vértices de corona
  transferidos a A frente a los de POST para esa pieza, corregido por la densidad de cada
  malla;
- **una pieza es legible solo si su cobertura es ≥ 0,5 y la mediana de su suelo ≤ 0,2
  mm.** Las cifras de conjunto y de grupo se calculan solo sobre piezas legibles. La cifra
  sobre todas las piezas se da aparte;
- por grupos, solo con piezas legibles: anteriores (13–23) y posteriores (14–17 y 24–27);
- la diferencia rígida entre los dos caminos, como ángulo en grados y como traslación en
  el centroide de las coronas;
- el suelo por diente;
- de cada camino, su rms, su solapamiento, su confianza y los motivos de revisión.

## 4 · Condiciones de validez y lectura

- **V1.** POST, registrado por este procedimiento, reproduce el `reg.ct_to_ios` de
  `run-5-modelo` a menos de 0,01 mm en sus vértices de corona. Si no, el procedimiento no es
  el de la ejecución de referencia y la medida no la describe.
- **V2.** El suelo del par principal tiene una mediana ≤ 0,2 mm.

**Es descriptiva, sin aprobado ni suspenso.** Se lee contra tres referencias fijadas aquí:
- el vóxel del CBCT, 0,30 mm;
- el ε del registro intraoral↔CBCT, 1,5 mm;
- la comprobación corona→esmalte de run-5: posteriores 0,2–0,6 mm, incisivos y 26 a
  1–2 mm.

**Lo que la medida puede decir:** cuánto varía la pose del CBCT entre dos escaneos de la
sesión PREVIO, pieza por pieza, en las piezas legibles. **Lo que no puede decir:** de dónde
viene el error de 1–2 mm que run-5, registrado con POST, muestra en los incisivos y el 26.
- Precisamente en esas piezas PREVIO difiere más de POST: 1,05 mm en el 11 y 0,80 en el 26,
  frente a 0,26–0,53 en el resto.
- La transferencia solo cubre allí el 18–29 % de la corona.
- El suelo del par sube a 0,34 mm.

Con las cifras de la exploración, el 11 y el 26 saldrán no legibles. Se escribe antes de
ejecutar, para que no parezca una decisión tomada a la vista del resultado.

**Lo que no se hará:**
- usar esto para ajustar el registro de ningún caso que luego se puntúe con puntos;
- cambiar el par, el procedimiento o los grupos después de ver un resultado;
- ejecutar más de una vez, salvo que una ejecución aborte antes de escribir resultados.
  El script no muestra ningún número de la medida antes de escribirlo, y comprueba V1 antes
  de registrar ningún otro escaneo, así que un fallo de V1 aborta sin resultados.

Código: [`scripts/consistencia_registro.py`](../../scripts/consistencia_registro.py).
Se ejecuta con `--par previo,repetido --par post,vis`: A = PREVIO y B = ESCANEADO PREVIO en
el par principal.

### Enmienda del 2026-11-19, antes de ejecutar

Una revisión del script y del protocolo, hecha antes de registrar ningún escaneo salvo POST
(V1), encontró nueve defectos. Se corrigieron todos antes de la ejecución. **V1 ya se
comprobó en la revisión: POST reproduce run-5 a 1,8·10⁻¹⁴ mm.** Lo que cambió:

- **La lectura se acota.** El par principal no puede hablar del error de run-5 en incisivos
  y en el 26 (§4). Se añade la regla de piezas legibles.
- **Las etiquetas se limpian** como en el pipeline.
- **V1 va primero** y aborta sin resultados.
- **Ningún número se muestra antes de escribirse.**
- **La distancia es exacta también junto a triángulos largos.** La primera versión fallaba en
  3 de cada 3000 puntos del modelo Visualization, por hasta 0,11 mm.
- **La tabla de §2 usa el procedimiento y la dirección fijados.**
- **La afirmación del 2,4 % se sustituye por lo medido.**
- **El caso de cada escaneo se crea con un enlace duro**, para que no quede escrito el nombre
  del fichero del proveedor.

## 5 · Resultados

Pendiente.
