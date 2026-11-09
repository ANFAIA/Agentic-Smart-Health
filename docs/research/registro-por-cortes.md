# Registrar el escáner contra un CBCT de cortes sueltos

> **Estado (2026-11-09):** método nuevo en `fusion_agents.por_cortes`, activo solo cuando
> la serie es **dispersa** y el escáner trae etiquetas FDI. Medido sobre un caso clínico y
> comprobado contra un caso con serie completa. **No es un TRE**: no hay puntos anatómicos
> marcados en este caso, y la comprobación independiente usa el mismo DICOM.

## 1 · El problema

El segundo caso clínico trae una serie CBCT de 301 cortes de 0,15 mm sacados de una
adquisición de al menos 651 (los nombres de fichero, `3DSlice<n>.dcm`, ya llevan los
huecos, así que la selección se hizo al exportar del equipo). Entre z = 29 y 38 mm, donde
están las coronas superiores, quedan 20 de 60 cortes, y el campo de gaussianas que llega a
la fusión tiene 4 en la banda que importa.

Con eso, el camino de siempre (`fusion_agents.preparacion` + ICP desde la identidad):

| Medida | Valor |
|---|---|
| Solapamiento que informa el agente | 3,6 % |
| Mediana corona→esmalte | 4,92 mm |
| `icp_global` (barrido de SO(3)) | 3,0 mm |

Y superpuesto sobre los cortes, la arcada sale **girada por detrás de los molares**. El
3,6 % no puede juzgarlo: con cortes sueltos casi ningún vértice del escáner tiene esmalte a
menos de 1 mm aunque la pose sea perfecta.

## 2 · Lo que se hizo

1. **Orientación inicial medida en las dos modalidades.** Escáner: derecha, anterior y
   superior desde los códigos FDI (`export_agents.anatomia.anatomical_frame`). CBCT: los
   ejes del paciente que fija el DICOM (LPS). Las dos bases tienen la misma quiralidad.
2. **ICP inverso**: del contorno de diente del CBCT a la superficie del escáner, que es el
   sentido en que todo punto tiene contrapartida con cortes sueltos.
3. **Solo la banda por encima de la mordida**: de 1,5 a 6 mm sobre el plano oclusal.

## 3 · El desfase vertical y de dónde venía

Con los pasos 1 y 2 sobre z ∈ [29, 38] mm la arcada caía en su sitio en planta, pero las
cúspides del escáner no llegaban a los cortes más bajos. Dos hallazgos:

- **El CBCT está en oclusión.** El esmalte sigue sin interrupción de z ≈ 22 a 37 con el
  máximo en 29: por debajo de 29 es sobre todo la arcada inferior. Eso explica la mayor
  parte de lo que parecía un desfase de 2 mm.
- **Por encima de la mordida sí hay desfase.** Barrido vertical diente a diente sobre el
  DICOM, solo z ∈ [30,5 ; 34,5], con una distancia en los dos sentidos entre la sección del
  escáner y el contorno del CBCT. Los 14 dientes piden bajar el escáner hacia oclusal:

  | Umbral del contorno | Mediana de los 14 |
  |---|---|
  | 1100 | −0,9 mm |
  | 1400 | −1,0 mm |
  | 1900 | −1,5 mm |

  El ICP no lo veía porque el contorno de hueso por encima de las coronas tiraba de ellas
  hacia la raíz y nada penalizaba las cúspides que faltaban.

Restringir el ajuste a la banda lo corrige. Medido con el mismo barrido independiente:

| Pose | Desplazamiento pedido (mediana) | Error medio en 0 |
|---|---|---|
| Anatómica + ICP en z 29–38 | −1,00 mm | 0,98 mm |
| Banda, umbral 1400, etiquetas del `.uos` | −0,25 mm | 0,87 mm |
| Banda, umbral 1400, etiquetas votadas sin limpiar (pipeline) | −0,62 mm | 0,90 mm |
| Camino de siempre | sin dientes con que medir | — |

**Se probó y no se queda** una búsqueda vertical explícita con la medida en los dos
sentidos: −0,50 mm con 1400 y +1,88 mm con 1900, peor que la banda sola.

## 4 · Por qué solo con series dispersas

Sobre `histora`, con la serie completa y las dos arcadas, el método nuevo registra peor:

| Arcada | Camino de siempre | Por cortes |
|---|---|---|
| Maxilar, mediana corona→esmalte | 1,41 mm | 2,24 mm |
| Mandíbula | 0,57 mm | 0,68 mm |

Con un volumen denso el esmalte entero es buen objetivo, y quedarse con una banda de
4,5 mm tira información. `prepara_por_cortes` lanza `SerieDensa` y el orquestador sigue por
el camino de siempre sin declarar nada. Una serie cuenta como dispersa si en la banda hay
un salto mayor que el doble del paso mediano del campo.

## 5 · Lo que se sabe y lo que no

- **La altura es lo peor medido.** Sobre coronas sintéticas en una rejilla de 0,3 mm el
  registro acierta en planta por debajo de 0,01 mm y deja las coronas 0,37 mm hacia la
  raíz (0,2 con rejilla de 0,15). El borde se lleva medio paso hacia fuera, que lo reduce
  de 0,59 a 0,37, pero no lo elimina. No se corrige restando.
- **La incertidumbre declarada es lo que se mueve la pose con el umbral.** En el caso
  disperso, entre 1100, 1400 y 1900: 1,97 mm de mediana, 3,23 de máximo, 1,98 en altura.
  Va al gate; el residuo del ICP no lo ve.
- **Las etiquetas del escáner cambian el resultado.** Con las del `.uos` (limpias y con el
  margen aplicado) el desfase residual es −0,25 mm; con las votadas sin limpiar, −0,62.
  `caso_completo.py` pasa las limpias.
- **No hay TRE.** El barrido independiente usa el mismo DICOM que el registro, aunque con
  todos sus cortes y no los del campo. Un TRE necesita puntos anatómicos marcados por un
  especialista ([protocolo](../spec/uos-protocolo-puntos-anatomicos.md)).
- **Un caso disperso.** El método está ajustado y comprobado sobre uno solo.
