# Dataset — bite2text (malla intraoral + foto RGB + informe clínico)

> **Estado (2026-07-23):** dataset **candidato, no descargado**. Todo lo que sigue
> procede de la **página oficial** (consultada el 2026-07-23), **no** de una
> verificación sobre los ficheros: a diferencia de
> [Teeth3DS+](dataset-teeth3ds.md), aquí no hay nada medido en local. Las cifras y
> el formato hay que reverificarlos al descargar.
>
> **Licencia: resuelta.** Es **CC-BY-SA 4.0** (*share-alike*), pero se usará
> **igual que Teeth3DS+** —datos crudos y derivados en `data/`, gitignored; se
> publica solo código y resultados agregados—, con lo que el *share-alike* no
> llega a activarse. Ver §3 y su condición de reactivación.

Es el candidato más interesante para la fase actual porque es el **único que trae
informes clínicos reales**: hoy el
[`report-agent`](../../packages/ingestion-agents/) solo está validado contra el
informe sintético que genera `ingestion_agents/synthetic.py`.

---

## 1. Identidad

- **Nombre:** bite2text.
- **Origen:** grupo **DITTO**, Università di Modena e Reggio Emilia (UNIMORE) con
  la Universidad de Ferrara.
- **Publicación:** ECCV 2026.
- **Contenido:** **1.000 casos**, cada uno con escaneo intraoral, fotografía
  intraoral y el informe clínico correspondiente.
- **Aprobación ética** declarada por los autores.

## 2. Contenido y estructura (según la fuente, sin verificar en local)

Cada caso declara **tres modalidades del mismo paciente**:

| Modalidad | Qué es | Soporte en nuestro contrato |
|---|---|---|
| Escaneo intraoral (IOS) | malla 3D, **superior e inferior** | superficial (`mesh`) |
| Fotografía intraoral RGB | foto 2D **estandarizada** | — (hoy `image`, sin agente) |
| Informe clínico | texto redactado por el clínico, **italiano original + traducción al inglés** | regional (`report`) |

**Sin verificar todavía** — y es lo primero que habrá que mirar al descargar:

- Formato exacto de las mallas (¿OBJ/PLY/STL?) y si traen **color por vértice
  real** o vienen «peladas» como acaba pasando en Teeth3DS+.
- Si hay **etiquetas FDI** por vértice o por diente, o ninguna.
- Estructura de directorios y si las tres modalidades comparten identificador de
  caso.
- Si los informes están en texto plano o en PDF (el `report-agent` lee `.txt`,
  `.md` y `.pdf`, este último con el extra `pdf`).
- Si el pH —o cualquier atributo regional de nuestro `ClinicalAttributes`—
  aparece de verdad en los informes, o si hay que ampliar la ontología a los
  atributos que sí reporten.

## 3. Licencia — CC-BY-SA 4.0 (resuelta por el modo de uso)

**CC-BY-SA 4.0**: atribución **y** *share-alike*. La obligación de *share-alike*
se dispara al **distribuir material adaptado**, no al usarlo.

**Decisión (2026-07-23): se usa igual que Teeth3DS+.**

- Los datos crudos van a `data/raw/` y los derivados a `data/processed/`, ambos
  **gitignored** (ver `.gitignore`: «no redistribuir datos clínicos vía el repo»).
- El repositorio publica **solo código y resultados agregados** (métricas,
  notebooks, documentación).
- No se redistribuye el dataset ni ningún dataset derivado de él.

Con ese uso **el *share-alike* nunca llega a activarse**: no hay distribución de
material adaptado. Se mantiene la obligación de **atribución** (citar el dataset
allí donde se reporten resultados obtenidos con él), que es la parte **BY**, no la
**SA**. El repositorio sigue siendo Apache-2.0 sin conflicto: el código no es obra
derivada del dataset.

> ⚠ **Condición de reactivación.** Esto se sostiene **mientras no se publique nada
> derivado del dataset**. Si en el futuro se distribuyen pesos de un modelo
> entrenado con bite2text, casos procesados, o un `TwinSnapshot` / campo
> `.ply` construido a partir de él, eso **sí** es material adaptado y esa parte
> tendría que ir bajo BY-SA. Conviene revisar este punto antes de cualquier
> publicación de artefactos, no después.

**Diferencia real con Teeth3DS+:** aquél es **CC-BY 4.0**, sin *share-alike* — ver
[§4 de su ficha](dataset-teeth3ds.md#4-licencia--resuelta-cc-by-40-con-matiz-documentado).
Es decir: usados igual, hoy se comportan igual; pero Teeth3DS+ **permitiría**
publicar derivados sin condición de licencia y bite2text no. La restricción no
desaparece, queda latente.

## 4. Qué habilita (y qué no)

**Lo que habilita:**

1. **Validar el `report-agent` contra prosa real.** Es su aportación principal.
   Hoy el extractor determinista (`backend="rules"`, regex línea a línea) solo se
   ha medido contra un informe sintético con formato tabulado.
2. **Cerrar por medición el punto abierto del backend LLM.** La skill
   `add-ingestion-agent` §5 prohíbe lógica de LLM en la ingesta; el
   [README del paquete](../../packages/ingestion-agents/README.md) lo deja como
   decisión pendiente. Con 1.000 informes reales deja de ser una discusión: si el
   regex aguanta, se retira el LLM; si se hunde, hay dato para lo contrario.
3. **La primera apariencia real del sistema.** Teeth3DS+ deja
   `color_superficie = None` (su color es un gris *placeholder*, ver §2.1 de su
   ficha). Estas fotos son color medido.
4. **Ejercitar el orquestador de verdad.** Tres modalidades del **mismo paciente**
   con el FDI como ancla es exactamente el flujo del `IngestionPipeline`
   (`apps/agent-orchestrator/`), que hoy solo ha visto casos sintéticos.

**Lo que NO habilita:**

- **No trae CBCT.** Verificado en la página oficial (2026-07-23): no menciona
  CBCT, cone-beam, panorámica, radiografía ni DICOM. Es un dataset exclusivamente
  **óptico**. El `cbct-agent` sigue dependiendo del volumen sintético.
- **No cierra el hueco de fondo:** *CBCT + malla intraoral del mismo paciente*, que
  es lo que necesita la fusión geométrica (registro malla↔CBCT, banda ε). Las dos
  rutas siguen siendo el generador sintético o el partner clínico.
- La **foto 2D** no tiene agente de ingesta: el `image-agent` sigue `planned` y
  está fuera del hito de Semana 3-4.

## 5. Acceso

- Portal: [`ditto.ing.unimore.it/bite2text`](https://ditto.ing.unimore.it/bite2text).
- Requiere **crear cuenta** («You need to have an account to download the
  dataset»). No consta DUA ni firma adicional, a diferencia de MMDental.
- Tamaño total **sin confirmar**.

## 6. Referencias

- Portal oficial: [ditto.ing.unimore.it/bite2text](https://ditto.ing.unimore.it/bite2text)
- Grupo DITTO — UNIMORE / Universidad de Ferrara. Publicado en ECCV 2026.
- Dataset en uso: [`dataset-teeth3ds.md`](dataset-teeth3ds.md)
- Contraparte de código: [`packages/ingestion-agents/`](../../packages/ingestion-agents/)
  (`report_agent.py`, `ontology.py`) y [`AGENTS.md`](../../AGENTS.md).
