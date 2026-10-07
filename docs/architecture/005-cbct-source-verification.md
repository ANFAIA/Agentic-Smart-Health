# Verificación de identidad y procedencia CBCT

Fecha: 2026-10-07.

El ejecutor del complete case ingería un directorio CBCT pero pasaba `cbct=None`
al exportador. Se pasa ahora el directorio realmente ingerido, para declarar la
serie como fuente externa, con hashes de fichero y de PixelData por instancia.
Los originales no se incluyen en el UOS. La auditoría local comprueba estos hashes
contra los ficheros disponibles; la validación del contenedor por sí sola no verifica
el contenido de las fuentes externas.

La inspección de la entrada real encontró dos SeriesInstanceUID y dos
FrameOfReferenceUID dentro del mismo directorio. El lector anterior ordenaba los
cortes de ambos grupos como un solo volumen. `cbct-agent` 0.2.0 rechaza múltiples
StudyInstanceUID, SeriesInstanceUID o FrameOfReferenceUID antes de decodificar
píxeles. No elige automáticamente el grupo mayor ni sustituye los identificadores.
Un directorio homogéneo con identificadores ausentes conserva el comportamiento
anterior; este control no garantiza identidad cuando faltan las cabeceras.

La división de un estudio en series y la relación entre sus marcos necesitan
resolverse antes de ejecutar este pipeline de un solo volumen. No se infiere una
transformación entre marcos distintos por compartir orientación o espaciado. Los
originales quedan intactos. Una ejecución anterior sobre la entrada mixta puede
ser un contenedor válido, pero no acredita una ingesta volumétrica coherente.

## Límites de las verificaciones

- `PixelSpacing`, posiciones y rescale son declaraciones de adquisición. Ni
  `Modality=CT` ni `RescaleType=HU` demuestran calibración independiente.
- Las separaciones entre cortes permiten estimar discontinuidades. No demuestran
  si faltan archivos o si el proveedor exportó deliberadamente una muestra.
- `scripts/verifica_complete_case.py` audita las fuentes y compara inventario,
  hashes de fuentes, registros de proceso, payloads binarios y matrices de registro.
  Su criterio es identidad exacta entre las dos ejecuciones evaluadas.
- La comparación de bytes JSON se informa por separado. No se elimina información
  temporal de manera genérica para aparentar igualdad semántica.
- La repetibilidad obtenida en datos sintéticos no acredita la del caso real, la
  exactitud clínica, la segmentación ni un entrenamiento nuevo de apariencia.

La calibración física y la evaluación de segmentación requieren evidencias
independientes. Se mantienen desconocidas cuando esas evidencias no constan.

## Resultado de la inspección del archivo descargado

El ZIP original contiene 401 instancias, repartidas en dos series de 100 y 301.
Cuatro nombres se repiten entre series con bytes distintos. La extracción previa
en un directorio plano sobrescribió cuatro instancias y dejó 397 ficheros. La
preparación conserva cada serie en un directorio propio y registra hashes por
instancia y del archivo original; no cambia los originales ni fusiona marcos.

`--cbct` selecciona explícitamente el directorio de una serie para el ejecutor.
Las pruebas técnicas se hacen con ambas por separado, sin elegir una para un uso
clínico. Sus intervalos sin datos no se interpolan; el modelo volumétrico no se
ejecuta sobre una rejilla irregular. Las etiquetas IOS existentes y la apariencia
archivada son entradas del ensayo, no resultados de entrenamiento repetido.

La ingesta conserva el origen XY de ImagePositionPatient. El campo usa su propio
centroide y el UOS lo declara en un marco calculado distinto del DICOM original.
El registro directo al escáner es la composición de la traslación por el origen
registrado y la inversa de ICP. El exportador 0.18.0 transporta esta relación en
el manifiesto y en los nodos glTF. No cambia la versión del formato UOS 0.3.

El espaciado Z nominal se calcula con la mediana de separaciones, para que un
intervalo inicial sin cortes no determine toda la escala. Los sidecars incluyen
las posiciones y el estado de uniformidad. El lector axial rechaza orientaciones
no soportadas y orígenes XY variables, en lugar de asignar coordenadas falsas.
