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
| PREVIO ↔ ESCANEADO PREVIO | **0,052 mm** | 0,53 | 0,999 |
| POST HIGIENE ↔ Visualization 18-28 | 0,11 mm | 0,52 | 1,002 |
| POST HIGIENE ↔ PREVIO | 0,52 mm | 2,24 | 0,976 |

- **Par principal: PREVIO y ESCANEADO PREVIO.** La ficha
  [`dataset-histora.md`](dataset-histora.md) los documenta como dos adquisiciones
  independientes de la misma sesión, y encajan a centésimas.
- **Par secundario, solo descriptivo: POST HIGIENE y Visualization 18-28.** No se sabe qué
  es el segundo: el repositorio lo llama «derivado del CBCT» en un sitio y «réplica virtual
  del molde» en otro, y es un sólido cerrado. Si sale del propio CBCT o de POST, el par no
  es independiente.
- **Los escaneos PREVIO salen un 2,4 % más pequeños que POST** y no encajan con él ni
  admitiendo escala. No se sabe por qué. Afecta igual a los dos escaneos del par principal,
  así que se cancela en la medida, pero se declara.
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

**Coronas de cada escaneo.** Solo POST tiene etiquetas FDI. Para los demás se transfieren:
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
- por diente y para todo, mediana y p95 de `d`;
- por grupos: anteriores (13–23) y posteriores (14–17 y 24–27);
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

Si `d` es pequeño en los incisivos, el error de 1–2 mm que se ve allí no viene de qué
escaneo se use: es común a los dos y la medida no lo alcanza.

**Lo que no se hará:**
- usar esto para ajustar el registro de ningún caso que luego se puntúe con puntos;
- cambiar el par, el procedimiento o los grupos después de ver un resultado;
- ejecutar más de una vez, salvo que una ejecución aborte antes de escribir resultados.

Código: [`scripts/consistencia_registro.py`](../../scripts/consistencia_registro.py).

## 5 · Resultados

Pendiente.
