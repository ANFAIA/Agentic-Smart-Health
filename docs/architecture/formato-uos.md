# Qué lleva un `.uos`, y por qué eso le sirve a alguien

Documento de cierre del MVP. Explica el contenedor desde el lado del uso: qué hay dentro,
qué problema concreto resuelve cada parte, y qué habilita más adelante. La especificación
formal es `UOS-SPEC-v0.2` y el esquema publicado, `schemas/uos-manifest-0.2.schema.json`.

## 0 · El problema que existe hoy en una clínica

Un caso completo son, ahora mismo, **tres cosas sueltas**:

- uno o dos STL del escáner intraoral,
- un directorio de cientos de DICOM del CBCT,
- un informe en PDF.

Cada uno en su sistema de coordenadas, sin ninguna relación explícita entre ellos, y
abiertos con un programa distinto cada uno. La relación —dónde cae este CBCT respecto a
este escaneo, a qué diente se refiere este hallazgo— vive en la cabeza de quien lo miró o
dentro de un software propietario, y **no viaja con los datos**. Cuando el caso pasa al
laboratorio, a otra clínica o a un revisor seis meses después, esa relación hay que
rehacerla.

Un `.uos` es esas tres cosas **más las relaciones entre ellas**, en un fichero.

## 1 · Qué lleva dentro

Un ZIP **sin comprimir** (STORE) con `manifest.json` como primera entrada. Sin comprimir
porque los payloads ya vienen comprimidos y porque así un cliente HTTP puede leer el
índice y bajarse **un asset suelto por rangos** sin traerse el caso entero — que en un
caso real con CBCT son 270 MB. Mismo precedente que `.usdz`.

Un caso medido de referencia: **419 entradas, 397 cortes DICOM, 0 errores y 0 avisos.**

### 1.1 · Lo medido, tal y como se midió

| Ruta | Qué es |
|---|---|
| `scene/scan.stl` | el escaneo intraoral, **el fichero original** |
| `volume/ct_001/` | la serie CBCT, **corte a corte, byte a byte** |
| `scene/field.ply` | el campo gaussiano del CBCT: una primitiva por vóxel ocupado, con la atenuación que el escáner midió |
| `scene/composite.ply` | escáner y CBCT en el mismo espacio, con columna `origen` por gaussiana |
| `scene/scene.glb` | la malla para el visor |
| `images/` | fotografía clínica |

Cada asset lleva su `sha256`, y el volumen además **un hash por corte**. Eso no es un
adorno de integridad: es lo que permite decir que el DICOM que sale del contenedor es
idéntico al que entró, y hay tests que lo comprueban quitando un corte, añadiendo uno de
más y alterando uno.

⚠️ El campo gaussiano **no es un render bonito, es una medida**. `density` es sigma
normalizada —atenuación Beer-Lambert— y no opacidad; las escalas van en milímetros
lineales y no en logaritmo. Se declara en un sidecar `.gs.json` al lado, columna por
columna, precisamente porque comparte nombres con el 3DGS de facto y significa otra cosa.

### 1.2 · Las relaciones, que es lo que no existía

**Marcos y registro.** El escáner es el marco canónico; el CBCT es otro marco; y
`registrations[]` lleva la matriz 4×4 que los une, **con su error**:

```json
{ "id": "reg.ct_to_ios", "method": "icp_surface",
  "rms_error_mm": 0.666, "operator": "auto:geometric-fusion-agent@0.2.0",
  "verified_by": null }
```

`verified_by: null` significa **nadie lo ha verificado todavía**, y el visor lo pinta como
«sin verificar». Un registro automático sin revisar y uno firmado por una persona no son
lo mismo, y el formato se niega a que parezcan lo mismo.

**Capa clínica.** `clinical/observations.json` cuelga los hallazgos del informe de
**códigos FDI**, no de páginas. Un hallazgo deja de ser una frase en un PDF y pasa a ser
un dato asociado a una pieza, que se puede contrastar contra la geometría — y de hecho se
contrasta: la fusión semántica marca el conflicto cuando el informe declara una pieza
ausente y el modelo la encuentra, o al revés.

**Vistas.** `views.json` guarda puntos de vista con nombre —oclusal, frontal, vestibular,
y **una por pieza**— con sus planos de corte y sus capas. Abrir un caso deja la cámara
donde el clínico la pondría, y navegar diente a diente es una lista, no un ejercicio de
ratón. En el caso de referencia son **18 vistas**.

**Procedencia.** `provenance/chain.json` encadena el `sha256` del manifiesto versión a
versión. Se puede saber si un caso cambió y respecto a qué.

**PHI.** `phi_state: pseudonymized` y `subject.pseudonym` como hash. Ningún nombre viaja
dentro del contenedor.

### 1.3 · Lo que sale de un modelo, marcado y separable

Todo lo inferido vive **solo** bajo `derived/`, con `regulatory.layer: 3` y un sidecar que
declara qué modelo lo produjo, con qué versión y con qué hash de pesos:

```json
{ "model": {"name": "ash-seg-teeth", "version": "0.4.0", "weights_sha256": "ec455883…"},
  "regulatory": {"layer": 3, "status": "investigational", "jurisdictions": []} }
```

Y la regla dura: **un `.uos` sin `derived/` sigue siendo válido y completo.** Borrar el
directorio y sus entradas del manifiesto es una operación soportada.

Esa regla es la que hace que la etapa que hoy no funciona —la segmentación FDI, ver
`docs/research/segmentacion-fdi-escaner.md`— sea *separable* y no *contaminante*. Y es
también la respuesta al escenario regulatorio: distribuir el mismo caso en una
jurisdicción donde el módulo de IA no está habilitado no requiere reexportar nada.

⚠️ Por eso las etiquetas FDI **no se hornean dentro de `scene.glb`**, aunque el spec
sugiera el picking por `extras`: metidas en la escena, quitar `derived/` dejaría de quitar
la inferencia y la regla se rompería en silencio.

## 2 · Qué le aporta esto a un profesional, hoy

**Un caso se entrega entero, y llega entero.** Escáner, CBCT, fotos, informe y la
transformación que los relaciona, en un fichero que se abre en un navegador sin instalar
nada y sin subir nada a ningún servidor.

**El STL original no se pierde nunca.** El contenedor lleva el escaneo tal cual, y además
la reversibilidad está medida: la malla se regenera con **4,59 × 10⁻⁶ mm** de error contra
un presupuesto de 0,1 mm — el único error que queda es el `float32` del propio STL. Es
decir: aceptar el formato **no obliga a renunciar** al fichero con el que ya se trabaja.

**Y encima del STL, lo que el STL no puede dar.** El exportador de malla compuesta
entrega dos cosas que el escáner por sí solo no tiene:

- la **arcada cerrada como sólido imprimible**, con base plana orientada por el eje
  oclusal —no por el superior, que en un maxilar es el contrario— y estanqueidad
  comprobada: 12.025 caras de cierre, 11.936 mm³, en 3,2 s;
- **un fichero por pieza** con su corona medida por el escáner y su raíz reconstruida del
  CBCT, cada uno declarando en la cabecera del STL de dónde viene cada mitad.

Ahí está la respuesta a «¿para qué el gemelo si puedo coger el STL?»: el STL es la corona.
Lo que el gemelo añade es **la pieza entera y el sólido cerrado**, que es lo que se
imprime.

**Cada dato dice de dónde viene y cuánto vale.** El registro con su rms y su estado de
verificación; la inferencia con su modelo y su hash; el volumen con su hash por corte. Un
caso que se revisa seis meses después no obliga a fiarse de nadie.

## 3 · Qué habilita después

**Seguimiento longitudinal.** `visits[]` ya está en el manifiesto. El mismo caso con
línea base y control, en el mismo marco, convierte «cuánto hueso se ha perdido en seis
meses» en una resta y no en un estudio nuevo.

**Un corpus auditable.** Un conjunto de `.uos` es un dataset con procedencia y con las
capas etiquetadas: se sabe qué es medida y qué es inferencia sin abrir los ficheros. Eso
es exactamente lo que hoy no tiene ningún dataset dental público.

**Interoperabilidad.** `fhir_map` declara a qué recurso FHIR corresponde cada asset, con
la nota honesta de que **FHIR R4 no tiene recurso para geometría dental** y `Media` es
para foto, vídeo y audio; se usa `DocumentReference` y se dice que se está usando así.

**El flujo de laboratorio.** Un `.uos` es lo que se le manda al laboratorio: la pieza a
imprimir, la arcada de contexto, el informe y la trazabilidad de qué se midió y qué se
supuso, en una entrega.

## 4 · Lo que este formato no hace, y conviene decirlo

- **No valida clínicamente nada.** Todo lo inferido sale `investigational`, y sale así a
  propósito.
- **No arregla que la segmentación FDI esté mal.** La aísla y la etiqueta; ver §7 de
  `docs/research/segmentacion-fdi-escaner.md`.
- **No define todavía el rango de ventana del volumen** (`value_range` va a `null`, con su
  aviso). Rellenarlo con un valor inventado sería peor que dejarlo vacío.
- **No es un estándar**: es una propuesta de contenedor con su esquema publicado y su
  validador, y hasta que otro implementador lo lea y lo abra no es más que eso.
