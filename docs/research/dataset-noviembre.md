# Dataset — noviembre (caso clínico propio: el segundo paciente)

> **Estado (2026-11-26):** dato **clínico propio**, no una cohorte pública. Un solo
> paciente, distinto del de [`dataset-histora.md`](dataset-histora.md). El equipo clínico
> confirmó el **consentimiento para uso y publicación**; lo comunicó el responsable del
> proyecto el 2026-11-26. Ningún dataset de terceros entra aquí.

## 1 · Licencia

**No hay licencia de terceros.** El dato es de la clínica y se publica por el
consentimiento de arriba, no por una licencia. Los derivados publicados se rigen por la
licencia del repositorio (Apache 2.0), como el resto de `docs/`.

## 2 · Qué contiene

| Modalidad | Detalle |
|---|---|
| CBCT | Rejilla anisótropa, **0,15 × 0,15 × 0,45 mm**. El espaciado bueno sale de las posiciones (`IPP`), no de `SliceThickness` (ver [`segmentacion-diente-cbct.md` §6](segmentacion-diente-cbct.md#6--lo-que-no-se-puede-concluir)). |
| Escáner intraoral | Arcadas superior e inferior. |

En el repositorio se le llama **segundo paciente** y, en el visor, **caso de noviembre**.

## 3 · Qué se puede publicar

Los derivados sí se publican: renders, GIF y vídeo del visor, gráficas y cifras
agregadas. Esta ficha los autoriza ante el `data-guard`. Hay tres salvedades
operativas, las mismas que en `histora`:

- **Los ficheros crudos no se versionan**: ni DICOM, ni STL, ni el contenedor `.uos`.
  Viven fuera del repositorio o en `data/processed/`, que está ignorado.
- **No hay identificadores directos en el repositorio.** Los nombres de fichero del
  proveedor y las iniciales del paciente no se escriben. Las capturas del visor tapan el
  identificador `urn:uuid` del caso.
- **Ningún dato de este caso sale a servicios externos.**

## 4 · Dónde se usa

| Fichero | Para qué |
|---|---|
| [`docs/assets/visor/`](../assets/visor/) | Cómo se ve el caso en `uos-viewer`: órbita, capas, capturas y un vídeo de 20 s. Contenedor emitido con la segmentación FDI C3. |
| [`segmentacion-diente-cbct.md` §6](segmentacion-diente-cbct.md#6--lo-que-no-se-puede-concluir) | Una ejecución de la segmentación por diente sobre este caso, sin validez y anotada como tal. |

## 5 · Lo que este caso NO puede responder

- **Un paciente no es una muestra.** Igual que `histora`, todo lo que sale de aquí es
  *proof of concept*.
- **No hay anotación** de referencia: no sirve para medir un segmentador.
- **La registración CBCT → escáner no la ha verificado una persona.** El contenedor de
  los medios declara un rms de 0,56 mm, y el rms no es árbitro de calidad (ver
  `segmentacion-diente-cbct.md` §5c).
