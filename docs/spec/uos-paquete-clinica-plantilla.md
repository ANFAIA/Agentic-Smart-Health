# Paquete para la clínica: plantilla

**Fecha:** 10 de octubre de 2026. Es el paso 2 del plan hacia una evaluación clínica.

El paquete pide a la clínica que interprete las series CBCT recibidas y confirme qué
modalidades corresponden a la misma visita. Se genera por software y se entrega a mano.
**Este repositorio no contacta con nadie ni envía nada**: el envío es decisión del
responsable del proyecto.

## Cómo se genera

```
uv run python scripts/inventario_series_cbct.py \
    --zip <ZIP descargado de la clínica> \
    --salida data/processed/<verificación>/clinic-package
```

El script lee cada entrada del ZIP por separado, sin extraerlo. También acepta directorios
de serie con `--serie`. Escribe tres ficheros, todos bajo `data/processed/`, que Git ignora:

| Fichero | Contenido |
|---|---|
| `inventario.md` | Informe legible: resumen por serie, declaraciones de adquisición, huecos y preguntas. |
| `cobertura.svg` | Cortes presentes por posición, un panel y un eje por serie. |
| `inventario.json` | Los mismos datos completos, incluidos todos los huecos. |

## Qué contiene

- Número de instancias, series, estudios y marcos de referencia.
- Por serie: rango de números de instancia, números ausentes, cobertura en z, paso
  mediano, huecos y si el orden por instancia coincide con el de posición.
- Declaraciones de adquisición: equipo y modelo, tipo de imagen, kV, matriz, espaciado,
  grosor de corte y rescale.
- Fechas recortadas a mes, si las cabeceras las declaran.
- Preguntas para la clínica.

## Qué no contiene

- Píxeles ni imágenes de cortes: son anatomía del paciente.
- Nombres, fechas de nacimiento, institución, médicos, operadores ni número de acceso.
  Si alguna cabecera los tiene, el script no escribe nada salvo que se fuerce, y entonces
  los declara por nombre de tag, nunca por valor.
- UID en claro: se muestran como `sha256[:12]`, que sirve para nombrar la serie en la
  conversación y no permite buscarla en un PACS.
- Número de serie del equipo ni nombre de estación.

⚠️ El texto libre del proveedor, como la descripción de la serie, se incluye solo si
ninguna cabecera tiene datos identificativos, y **no se inspecciona** en busca de nombres.
Hay que leerlo antes de entregar el paquete.

## Qué se pide a la clínica

1. Qué representa cada serie y qué cobertura anatómica tiene.
2. Si los cortes ausentes se omitieron deliberadamente y si existe la reconstrucción
   completa.
3. Si el CBCT, el escaneo intraoral, las fotografías y los informes corresponden a la misma
   visita, y con qué fechas.
4. Equipo, protocolo y controles de calidad o calibración disponibles.
5. Que un especialista revise las series en su visor habitual y documente para qué tareas
   son adecuadas.

La clínica puede responder sin transmitir la identidad del paciente: basta con referirse a
cada serie por su resumen corto.

## Antes de entregar

- Leer `inventario.md` entero, incluido el texto libre si aparece.
- Confirmar que el informe no declara datos identificativos en su cabecera.
- Adjuntar la [ficha de uso previsto](uos-uso-previsto-v0.1.md) si se quiere que la
  clínica valore los criterios.

Que los datos procedan de la clínica establece su origen. No demuestra que esta
exportación conserve todo lo necesario para cualquier tarea. No se propone repetir una
adquisición de un paciente para resolver pruebas de software.
