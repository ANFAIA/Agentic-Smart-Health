# Dataset — histora (caso clínico propio: CBCT + escáner intraoral + informes)

> **Estado (2026-08-18):** dato **clínico propio**, no una cohorte pública. Un solo
> paciente, con **consentimiento para uso y publicación** confirmado por el equipo
> clínico. Es la única fuente del repositorio que no viene de un dataset con licencia
> de terceros, y por eso tiene ficha propia: sin ella, ningún notebook con imágenes
> derivadas de este caso puede versionarse (regla `procedencia` del `data-guard`).

## 1 · Qué contiene

| Modalidad | Detalle |
|---|---|
| CBCT | Carestream **CS 9600**, 535 × 535 × 578, vóxel **0,30 mm**, FOV 16,1 × 16,1 × 17,3 cm. Dos series del mismo paciente. |
| Escáner intraoral | Tres escaneos mandibulares (80.618 / 93.860 / 87.417 vértices) y dos maxilares, en STL. |
| Malla derivada del CBCT | `DigitalModelUnsectioned` de ambas arcadas — réplica virtual del molde, **sin raíces** (medido: misma envolvente que el IOS). |
| Fotografía | Intraoral, JPG. |
| Informes | PDF clínicos. |

El caso es de **recesión gingival**, que es lo que lo hace interesante para la medida
periodontal que persigue el proyecto.

## 2 · Los dos escaneos de la misma visita

La pieza más valiosa del caso, y no es evidente al mirar la lista: **dos adquisiciones
independientes de la misma sesión**. Entre ellas no cambió ni la biología ni los
depósitos, así que todo lo que un método mida ahí es **falso por construcción**. Es el
control nulo del que sale el umbral de detección de ~0,4 mm
([`notebooks/registro-por-diente-histora.md`](../../notebooks/registro-por-diente-histora.md)).

Un par pre/post higiene **no** sirve para eso: la higiene no mueve dientes pero sí retira
cálculo, así que la superficie cambia de verdad.

## 3 · Qué se puede publicar

**Todo lo que este caso produce**, con dos salvedades operativas:

- **Los ficheros crudos no se versionan.** No por permiso sino por tamaño y por la regla
  de extensiones del `data-guard`: DICOM, STL y volúmenes viven en `~/anfaia/`, fuera del
  repositorio, como los de cualquier otra fuente.
- **Los identificadores directos no aparecen en el código.** Los nombres de fichero del
  proveedor traen iniciales y número de caso; los scripts los resuelven **por patrón**
  (ver `scripts/desplazamiento_relativo.py` y `scripts/composicion_cbct_ios.py`). Es
  higiene de repositorio público, independiente del consentimiento.

Los derivados —renders, gráficas, cifras agregadas y las salidas de notebook— **sí** se
publican, y es lo que esta ficha autoriza ante el guardián.

## 4 · Lo que este caso NO puede responder

Está medido y conviene tenerlo a mano antes de diseñar encima:

- **La raíz no se separa del hueso.** El ligamento periodontal mide 0,15–0,38 mm y el
  vóxel es 0,30. Tres vías independientes fallaron ahí (injerto de raíz, margen → cresta,
  y recorte del campo gaussiano por umbral).
- **Un paciente no es una muestra.** Todo lo que sale de aquí es *proof of concept*, y
  así está declarado en las fichas de experimento.
- **No hay anotación.** No se puede entrenar ni medir un segmentador contra verdad de
  referencia: para eso está [ToothFairy2](#), y este caso es el objetivo, no la escuela.

## 5 · Dónde se usa

| Fichero | Para qué |
|---|---|
| [`notebooks/09-per-tooth-registration-experiment.ipynb`](../../notebooks/09-per-tooth-registration-experiment.ipynb) | El experimento de registro por diente, con sus figuras. |
| [`notebooks/registro-por-diente-histora.md`](../../notebooks/registro-por-diente-histora.md) | La ficha con todas las cifras y los resultados negativos. |
| [`scripts/desplazamiento_relativo.py`](../../scripts/desplazamiento_relativo.py) | Referencia leave-one-out y umbral de detección. |
| [`scripts/promedio_y_escala.py`](../../scripts/promedio_y_escala.py) | Las dos preguntas de diseño, ambas negativas. |
| [`scripts/composicion_cbct_ios.py`](../../scripts/composicion_cbct_ios.py) | Composición dientes-CBCT + encía-IOS. |
