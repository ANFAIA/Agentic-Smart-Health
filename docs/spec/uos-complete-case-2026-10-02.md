# Complete case — comprobación del 2 de octubre de 2026

Se ejecutó el pipeline sobre el caso clínico local utilizado previamente, con el
modelo de segmentación CBCT y las etiquetas FDI disponibles, ajuste del campo y
entrenamiento de apariencia. Los originales y resultados clínicos permanecen en
almacenamiento local ignorado por Git. Este documento recoge resultados técnicos;
no contiene valores clínicos extraídos ni acredita aptitud clínica.

## Resultado de la ejecución

- El proceso terminó con código 0 y el exportador UOS informó éxito.
- Ingesta: 1.341.990 primitivas; 32 observaciones regionales y 8 medidas globales.
- Ajuste: 138.676 gaussianas, factor de compresión 9,7. El residual fue 42,8 en la
  escala de grises de entrada. El nombre interno `rmse_hu` no acredita calibración HU.
- Apariencia: 1.600 renders de Blender, 6.000 iteraciones y 127.635 gaussianas.
  PSNR de las vistas reservadas: 34,26 dB; SSIM: 0,955. Son comparaciones con los
  renders usados como referencia, no con una adquisición independiente del paciente.
- Se generaron la malla de salida y sus versiones PLY, STL y 3MF desde el UOS.

No se activó el refinamiento contra DRR (`--refina-3dgs`). No se repitió el
entrenamiento para medir repetibilidad ni se realizó evaluación clínica.

## Comprobación independiente del UOS

El contenedor tiene 192.624.253 bytes y 23 assets; 13 originales externos. El
validador devuelve **válido, UOS-Core, cero errores y 37 advertencias**:

| Código | Recuento |
|---|---:|
| UOS-W-006 | 1 |
| UOS-W-020 | 22 |
| UOS-W-021 | 9 |
| UOS-W-017e | 1 |
| UOS-W-016 | 1 |
| UOS-W-017n | 3 |

Hay cinco assets de capa 3. La escena base no contiene apariencia
`KHR_gaussian_splatting`. Se creó y validó un sucesor sin inferencias: versión 2,
sin assets de capa 3 ni entradas bajo `derived/`. Los bytes de la escena base
se conservaron y el archivo original no cambió. No se generaron conclusiones de
aptitud clínica automáticas.

## Pendientes que la ejecución deja visibles

- **Fuente CBCT sin resolver:** el ejecutor pasa `cbct=None` al exportador.
  `asset.field` no puede vincularse a un original declarado y el registro lo dice
  explícitamente. La extensión permite detectar el hueco, pero esta ejecución no
  lo cierra. Declarar la fuente desde el ejecutor requiere una nueva exportación.
- **Color distinto al de agosto:** se resolvieron dos poses para proyectar color
  sobre el escaneo, pero no se activó la extracción de tonos por pieza mediante
  `--lado-foto`. Se proyectó color sobre 86.142 vértices y se interpolaron 14.004. La malla resultante declara respaldo
  de color y soporte de apariencia en el 26,3 % de los vértices. El éxito del
  entrenamiento no resuelve esta limitación.
- **Revisión humana:** el gate conserva 25 motivos. La ejecución técnica no
  constituye aprobación clínica de sus salidas.
- La calibración, la incertidumbre geométrica independiente y la repetibilidad
  no medidas siguen desconocidas.

El log conservaba mensajes antiguos de «color real» y «HU». Se corrigieron en
[caso_completo.py](../../scripts/caso_completo.py) tras observarlos, sin alterar
el log original ni recalcular las métricas.

## Evidencia local

La ejecución está en `data/processed/complete-case-20261002/`. Contiene `run.log`,
`validation.json`, `validation-without-inference.json`, `summary.json` y el
comprobador `validate_run.py`, además de los artefactos. Todo permanece fuera de
Git. El documento de
[fidelidad y procedencia](uos-fidelity-provenance-v1.md) describe el contrato y sus límites.

## Recuperación funcional del resultado de agosto

La ejecución `data/processed/complete-case-20261002-restored/` reutiliza el
artefacto de apariencia archivado: 113.540 gaussianas y 13 registros de color
por pieza. No ejecuta entrenamiento de apariencia nuevo. Los atributos binarios
de las gaussianas coinciden exactamente con los del UOS anterior, incluyendo
posición, escala, opacidad, orientación y coeficientes de color. El escaneo y las
fotos fuente se comprueban por SHA-256; las etiquetas de malla coinciden también
con las de agosto. La procedencia registra los hashes del archivo y del artefacto
reutilizados. Esto recupera el aspecto anterior sin acreditar calibración clínica.

El nuevo contenedor tiene 191.351.263 bytes, 24 assets, 13 originales externos,
**cero errores y 42 advertencias**. Las etiquetas por gaussiana de la apariencia
viajan en `asset.seg_appearance`, separadas del GLB. Las etiquetas por vértice
continúan en `asset.seg_teeth`. El visor cruza cada array con su fuente, comprueba
los hashes y el recuento y recupera la selección y el aislamiento por pieza. Las
fichas clínicas admiten ahora registros de color envueltos con procedencia.

La regeneración desde el propio UOS produce 112.067 vértices y 220.085 triángulos
en STL, PLY y 3MF. La cobertura del campo alcanza el 97,1 %; 3.145 vértices de la
pieza sin color declarado permanecen en gris. La comparación del visor contra
Python coincide exactamente en RGB y en la bandera de cobertura por vértice,
así como en posiciones y atributos de color de cada triángulo del STL. Las
normales del STL pueden presentar diferencias de precisión aritmética.

La retirada de inferencias vuelve a producir un sucesor válido sin `derived/` y
con los bytes de la escena base conservados. Los originales STL, fotos e informes
se declaran externos: no hay entradas con sus bytes en el ZIP. La geometría
convertida del escaneo sí está en la escena, para permitir regeneración. Este
ejecutor sigue sin declarar el DICOM fuente (`cbct=None`); la procedencia del
campo CBCT, la calibración y la repetibilidad siguen pendientes.

La prueba del navegador local reconoce 14 piezas, selecciona una pieza con un
clic sobre el modelo, muestra la ficha clínica, permite cerrar la selección y
seleccionar la pieza 26 desde el odontograma, sin errores de JavaScript. El pase
de selección excluye el rasterizador de apariencia y rechaza códigos no presentes
en las etiquetas. La ficha conserva la procedencia en detalles desplegables y
evita que su contenido o botón de cierre queden debajo de los controles de vista.
