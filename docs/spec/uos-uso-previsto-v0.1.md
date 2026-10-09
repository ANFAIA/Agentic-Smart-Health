# Ficha de uso previsto, versión 0.1

**Fecha:** 10 de octubre de 2026. **Estado:** borrador para ratificar con un especialista
antes de cualquier evaluación. **Alcance:** sistema UOS 0.3, exportador de referencia y
visor local. La ficha describe el sistema, no a ningún paciente.

Esta ficha es el paso 1 del plan hacia una evaluación clínica. Fija **antes de evaluar** qué
tarea se pretende apoyar, con qué datos y qué resultado contaría como aceptable. Sin ella,
cualquier resultado posterior podría reinterpretarse como éxito a posteriori.

## 1. Uso previsto

| Campo | Valor |
|---|---|
| Usuario | Odontólogo o especialista que revisa un caso ya diagnosticado por los medios habituales. |
| Tarea | **Consulta conjunta por pieza dental**: abrir en una sola escena el CBCT, el escaneo intraoral y el informe, seleccionar una pieza y ver qué dice cada fuente sobre ella. |
| Datos necesarios | Una serie CBCT coherente de la arcada, el escaneo intraoral de la misma arcada y visita, el informe firmado y, opcionalmente, fotografías intraorales. |
| Resultado esperado | Para la pieza elegida: su posición en el escaneo, la región correspondiente del CBCT y las observaciones del informe, con la procedencia de cada valor y lo inferido marcado como tal. |
| Contexto | Investigacional y demostrador. El usuario conserva su método habitual y decide con él. |

### Fuera del uso previsto

- Diagnóstico, planificación de tratamiento o cualquier decisión clínica tomada con UOS.
- Medición de raíces, de hueso o de distancias anatómicas sobre la escena combinada.
- Planificación de implantes, guías quirúrgicas o fabricación.
- Interpretación de valores de gris como unidades Hounsfield calibradas.
- Medición de color.

Ampliar el uso a cualquiera de estos puntos exige una ficha nueva y criterios propios.

## 2. Errores relevantes para esta tarea

Ordenados por gravedad. Los primeros son peores porque pueden pasar sin que el usuario lo
note.

1. **Pieza equivocada en silencio.** La selección muestra otra pieza, o el informe se
   asocia a la vecina, sin ningún aviso.
2. **Desalineación entre modalidades.** La región del CBCT mostrada para una pieza no
   corresponde a esa pieza en el escaneo.
3. **Inferencia presentada como medida.** Una etiqueta FDI, una raíz reconstruida o un
   color estimado se muestran como si fueran dato medido.
4. **Omisión silenciosa.** Una observación del informe o un fallo de una etapa no aparecen
   y nada lo declara.
5. **Fallo declarado.** El sistema no puede responder y lo dice. Es aceptable: el usuario
   recurre a su método habitual.

## 3. Criterios de aceptación provisionales

Se fijan ahora y se ratifican con el especialista **antes** de ver resultados. Si se
cambian después de evaluar, la evaluación se repite con referencias nuevas.

| Id | Criterio | Cómo se mide | Umbral provisional |
|---|---|---|---|
| A1 | Alineación CBCT↔escáner apta para visualización | Error en puntos anatómicos reservados para evaluar (TRE), según el [protocolo de puntos](uos-protocolo-puntos-anatomicos.md) | TRE máximo ≤ 1,5 mm en cada punto <!--const:EPSILON_IOS_CBCT_MM--> |
| A2 | Registro con solapamiento suficiente | Fracción de la nube con contrapartida, informada por la fusión geométrica | ≥ 0,20 <!--const:DEFAULT_MIN_OVERLAP--> |
| A3 | Pieza correcta | Para cada pieza evaluada, la pieza mostrada coincide con la que el especialista identifica en su visor | Ningún error de tipo 1 |
| A4 | Inferencia distinguible | Cada valor inferido se presenta como tal en la ficha de la pieza | Ningún error de tipo 3 |
| A5 | Nada omitido en silencio | Cada observación del informe aparece en su pieza o en la lista de revisión | Ningún error de tipo 4 |
| A6 | Revisión humana activada cuando corresponde | Toda confianza por debajo del umbral llega a la lista de revisión | Umbral 0,7 <!--const:DEFAULT_HITL_THRESHOLD--> |
| A7 | Utilidad de la consulta | El especialista responde las preguntas por pieza con UOS y con su método habitual | **Por fijar con el especialista antes de evaluar.** Sin umbral no se evalúa. |

**Por qué 1,5 mm en A1.** Es el margen que el pipeline ya usa para intraoral↔CBCT: un CBCT
con vóxeles de 0,30 mm no permite exigir menos, y por encima la región mostrada puede caer
en la pieza vecina. Es un umbral para *ver* la pieza correcta, no para medir sobre ella.

**Por qué A3 a A5 no admiten errores.** Un solo caso de pieza equivocada en silencio basta
para que el usuario deje de poder fiarse de la selección. Lo que el sistema no sepa debe
declararlo, y eso se cuenta como fallo declarado, no como error.

## 4. Estado actual frente a los criterios

Datos del experimento del 7 de octubre
([informe](uos-verification-2026-10-07.md)). No es una evaluación; indica qué falta.

| Id | Estado |
|---|---|
| A1 | **No medido.** No hay puntos anatómicos marcados por un especialista. El residuo de ICP no es un TRE. |
| A2 | **No se cumple.** Serie de 100 cortes: 0 % de solapamiento. Serie de 301: 3,6 %. |
| A3 | **No evaluable.** Las etiquetas por pieza dependen de la segmentación, cuya revisión está aplazada. |
| A4 | Implementado en el contenedor y el visor; no evaluado por un usuario. |
| A5 | Implementado como lista de revisión; no evaluado por un usuario. |
| A6 | Implementado; el caso actual dispara revisión por varias causas. |
| A7 | Sin umbral ni evaluación. |

Con el registro actual **el caso no puede entrar en la evaluación de la tarea**. Antes hay
que aclarar las series con la clínica y medir el TRE.

## 5. Encuadre regulatorio

Uso investigacional y demostrador: el sistema produce información para revisión clínica y
no emite diagnóstico, como ya fija `AGENTS.md`. Esta ficha no determina la clasificación
regulatoria del sistema. Si un resultado se declarase diagnóstico o se usase para tratar,
haría falta una evaluación conforme al marco aplicable.

Referencias de método:

- FDA, evaluación técnica de imagen cuantitativa:
  <https://www.fda.gov/media/123271/download>. Trata rendimiento técnico; no constituye por
  sí sola una evaluación clínica.
- IMDRF, evaluación clínica de software como dispositivo médico:
  <https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation>.

## 6. Siguientes pasos que dependen de esta ficha

1. Ratificar los criterios y fijar el umbral de A7 con un especialista.
2. Aclarar las series con la clínica mediante el
   [paquete para la clínica](uos-paquete-clinica-plantilla.md).
3. Marcar puntos anatómicos y medir A1 con `scripts/tre_puntos.py`.
4. Solo si A1 y A2 se cumplen: evaluar la tarea completa.
