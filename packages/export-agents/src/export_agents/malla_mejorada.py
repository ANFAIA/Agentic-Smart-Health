"""La malla del escáner con el color que mide el campo gaussiano.

**Qué es un «STL mejorado» y por qué el color tiene que salir del campo.** Un STL lleva
triángulos y nada más: ni color, ni unidades, ni de dónde salió. El escáner intraoral
entrega esa geometría y el laboratorio la imprime en gris. Lo que este proyecto puede
añadirle es lo que ha medido encima — el color real de cada corona, tomado de las fotos del
paciente — y devolverlo en un fichero que un protésico pueda abrir.

⚠️ **El color se lee del CAMPO GAUSSIANO, no de la malla pintada intermedia.** Es tentador
sacarlo de `pinta_malla`, que es un paso anterior del mismo pipeline y ya tiene el color por
vértice. Pero eso no es reversibilidad: el `.uos` que se entrega lleva el campo, no aquel
`vcol`. Quien reciba el contenedor —dentro de un año, sin este repositorio— tiene que poder
sacar la malla mejorada de lo que hay dentro. Si el color viniera de un fichero intermedio,
el campo no estaría aportando nada y la cadena no cerraría.

⚠️ **La arcada entera y sin raíz.** Componer corona del escáner con raíz del CBCT une dos
superficies medidas por aparatos distintos, con un registro de 0,666 mm entre ellas, y la
costura se ve. El STL base del escáner NO tiene raíz, así que meterle una lo convierte en
otra cosa: la reversibilidad pide devolver mejorada la misma pieza que entró. Las piezas
sueltas, además, dependen de una segmentación que hoy no separa bien las coronas.

**Tres ficheros porque ninguno solo sirve.** El PLY lleva color por vértice y lo abre
cualquier visor de malla; el 3MF es el formato que los slicers aceptan con color y unidades
dentro; el STL con color por cara es una convención no estándar que casi nada lee, y se
emite porque hay cadenas de trabajo que solo aceptan `.stl`.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from scipy.spatial import cKDTree

# El grado 0 de los armónicos esféricos: `color = C0 * f_dc + 0,5`. Es la convención de
# INRIA, la misma que escribe `field.escribe_inria` y la que lee el visor.
C0 = 0.28209479177387814

# Gaussianas que se mezclan por vértice.
#
# ⚠️ **Tomar la más cercana produce motas, y está medido.** La opacidad mediana del campo de
# apariencia es 0,026: un píxel del render es la acumulación de decenas de gaussianas. El
# 26,6 % de ellas tiene un color extremo —muy saturado o casi negro— que en el render aporta
# un 2,6 % del peso y no se ve; muestreando UNA sola, esa rara sale al 100 % y aparece como
# una mota azul o naranja sobre una corona. Reproducir la mezcla las devuelve a su peso.
VECINOS_COLOR = 32

# Hasta cuántas sigmas de la gaussiana más cercana se admite que el campo tiene color.
#
# ⚠️ **Esto iba en MILÍMETROS y era un número inventado, con consecuencias visibles.** Con
# un corte fijo de 1,0 mm, 2.635 vértices se quedaban en `NEUTRO` — y el 100 % de ellos
# estaban sobre coronas etiquetadas, a **1,08 mm** de mediana: justo al otro lado del corte.
# En pantalla eso son manchas gris azuladas sobre las caras vestibulares de los incisivos,
# que es la superficie más lisa y la más visible de la arcada.
#
# Y el corte no sólo estaba mal calibrado: estaba mal planteado. Un milímetro no significa
# lo mismo para todas las gaussianas. Las de una superficie lisa crecen —el optimizador no
# necesita muchas para describir un plano— y una con `sigma` 0,61 mm cubre de sobra un punto
# a 1,08 mm, que son **1,81 sigmas**. Medido: los vértices que sí recibían color están a
# 0,99 sigmas de mediana y los rechazados a 1,81, o sea dentro del soporte de su gaussiana.
# En sigmas la pregunta es la correcta —«¿la cubre esta gaussiana?»— y deja de depender de
# lo grandes que le hayan salido al entrenamiento.
ALCANCE_SIGMAS = 3.0

# Lo que se pinta donde no hay color medido.
#
# ⚠️ **Gris neutro y DECLARADO.** Rellenar con marfil haría que un vértice sin medida
# pareciera un dato. El caso real que motivó esto: el FDI 17 no lo ve ninguna foto —el gate
# del contenedor ya lo declara— y salía pintado con el degradado de respaldo, un gris
# azulado que quien abría el fichero leía como un fallo del color y no como «esto no se
# midió».
NEUTRO = np.array([160, 160, 160], np.uint8)

# Paso de cuantización de la paleta del 3MF, en niveles de sRGB por canal.
#
# ⚠️ **Solo en el 3MF, y solo porque es un formato de IMPRESIÓN.** El PLY conserva el color
# entero. Aquí la paleta va enumerada en el XML y un caso real produce 10.627 colores
# distintos sobre 112.067 vértices; hay lectores que acotan el tamaño de un `colorgroup`.
# Medido sobre ese caso, redondear a múltiplos de 4 deja **1.378** colores con ΔE medio
# **1,06** y percentil 99 **2,27** — por debajo del umbral de diferencia perceptible, y muy
# por debajo de lo que cualquier impresora puede reproducir.
PASO_PALETA_3MF = 4


@dataclass(frozen=True)
class MallaMejorada:
    """Lo emitido y lo que se puede afirmar de ello."""

    ficheros: dict[str, Path]
    n_vertices: int
    fraccion_medida: float
    """Qué parte de los vértices recibió color del paciente. El resto va en `NEUTRO` y con
    su bandera `medido` a 0, no en un color inventado."""


def color_desde_gaussianas(
    pos: np.ndarray, centros: np.ndarray, f_dc: np.ndarray,
    opacidad: np.ndarray, escalas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`(rgb, medido)` por vértice, mezclando como mezcla el rasterizador.

    Media ponderada por `opacidad x caída gaussiana` sobre `VECINOS_COLOR` vecinos: el
    color que se VE en ese punto, no el del centro más próximo.

    ⚠️ **La oclusión ambiental NO se aplica aquí.** Es un factor de visualización y no entra
    en un fichero que alguien puede imprimir o medir encima: oscurecer una corona porque
    tenga una fisura al lado sería meter una sombra nuestra dentro de un dato del paciente.
    """
    tope = ALCANCE_SIGMAS * float(np.percentile(escalas, 99))
    d, i = cKDTree(centros).query(pos, k=VECINOS_COLOR, distance_upper_bound=tope)
    valido = np.isfinite(d)
    i = np.where(valido, i, 0)
    sigma = np.maximum(escalas[i], 1e-3)
    w = np.where(valido,
                 opacidad[i] * np.exp(-0.5 * (np.where(valido, d, 0.0) / sigma) ** 2), 0.0)
    total = w.sum(1)
    color = np.einsum("ij,ijk->ik", w, np.clip(f_dc[i] * C0 + 0.5, 0, 1))
    color /= np.maximum(total, 1e-12)[:, None]

    rgb = np.tile(NEUTRO, (len(pos), 1))
    # ⚠️ **La cobertura la decide el conjunto, no la vecina más próxima.** Preguntar sólo
    # por la más cercana falla en los dos extremos: en una cara lisa esa vecina está a 1,8
    # sigmas y el punto sí está cubierto —eran las manchas grises sobre los incisivos—, y en
    # un surco la más cercana es diminuta y su soporte no llega, aunque haya treinta
    # gaussianas encima. Medido, ese segundo caso bajaba la cobertura del 97,6 % al 96,6 %.
    # Es la misma pregunta que hace el rasterizador: si alguna gaussiana cubre el punto
    # dentro de SU propio soporte, ahí hay color. Ver `ALCANCE_SIGMAS`.
    cubierto = (valido & (d <= ALCANCE_SIGMAS * sigma)).any(axis=1)
    medido = (total > 1e-6) & cubierto
    rgb[medido] = (np.clip(color[medido], 0, 1) * 255).astype(np.uint8)
    return rgb, medido


# Cuántos saltos por aristas se admite alejarse para rellenar un hueco sin color.
#
# ⚠️ **Rellenar es de VISUALIZACIÓN, y por eso `medido` no cambia.** Un puñado de vértices
# sueltos que ninguna gaussiana cubre —112 sobre 112.067 en el caso real— salen en gris
# neutro en medio de una corona medida, y en pantalla eso es una mota: quien la ve la lee
# como un fallo del color, no como «aquí no llegó el campo». Se les pone el color de sus
# vecinos MEDIDOS y se deja la bandera a 0, que es lo que afirma el fichero.
#
# ⚠️ **Y sólo entre vecinos de la MISMA pieza.** Sin esa condición, una pieza entera sin
# color medido —el FDI 17, que ninguna foto ve— se rellenaría desde sus vecinas y saldría
# con un color que no es de nadie. Con ella, sus vértices no tienen ni un vecino medido de
# su propio código a ningún número de saltos, así que se quedan en gris, que es lo correcto.
SALTOS_RELLENO = 3


def rellena_huecos(caras: np.ndarray, rgb: np.ndarray, medido: np.ndarray,
                   fdi: np.ndarray) -> np.ndarray:
    """Color de visualización para los huecos sueltos. Devuelve el `rgb` corregido.

    No toca `medido`: lo que se rellena sigue declarado como no medido. Ver
    `SALTOS_RELLENO`.
    """
    n = len(rgb)
    aristas = np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]])
    aristas = np.concatenate([aristas, aristas[:, ::-1]])
    # Sólo se propaga dentro de la misma pieza.
    aristas = aristas[fdi[aristas[:, 0]] == fdi[aristas[:, 1]]]
    orden = np.argsort(aristas[:, 0], kind="stable")
    aristas = aristas[orden]
    inicio = np.searchsorted(aristas[:, 0], np.arange(n + 1))

    salida = rgb.copy()
    listo = medido.copy()
    for _ in range(SALTOS_RELLENO):
        pendientes = np.nonzero(~listo)[0]
        if len(pendientes) == 0:
            break
        nuevo_listo = listo.copy()
        for i in pendientes:
            vecinos = aristas[inicio[i]:inicio[i + 1], 1]
            fuente = vecinos[listo[vecinos]]
            if len(fuente):
                salida[i] = np.median(salida[fuente], axis=0).astype(np.uint8)
                nuevo_listo[i] = True
        if not nuevo_listo.sum() > listo.sum():
            break
        listo = nuevo_listo
    return salida


def escribe_ply(ruta: Path, pos: np.ndarray, caras: np.ndarray, rgb: np.ndarray,
                fdi: np.ndarray, comentarios: list[str],
                medido: np.ndarray | None = None) -> None:
    """Malla con color por vértice, su código FDI y la bandera `medido`.

    ⚠️ **Sin `medido`, «no sé el color» y «lo sé mal» se ven igual.** Es una columna de un
    byte que convierte una duda en una pregunta contestable sin abrir el `.uos`.

    ⚠️ La cabecera va en ASCII estricto: el formato PLY no define otra cosa, y un carácter
    fuera de rango deja el fichero ilegible para lectores que sí lo respetan.
    """
    if medido is None:
        medido = np.ones(len(pos), bool)
    cab = ["ply", "format binary_little_endian 1.0",
           *(f"comment {c}" for c in comentarios),
           f"element vertex {len(pos)}",
           "property float x", "property float y", "property float z",
           "property uchar red", "property uchar green", "property uchar blue",
           "property short fdi",
           "property uchar medido",
           f"element face {len(caras)}",
           "property list uchar int vertex_indices", "end_header"]
    texto = "\n".join(cab) + "\n"
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1"),
                   ("fdi", "<i2"), ("medido", "u1")])
    v = np.empty(len(pos), dt)
    v["x"], v["y"], v["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    v["r"], v["g"], v["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    v["fdi"] = fdi
    v["medido"] = medido.astype(np.uint8)
    f = np.empty(len(caras), np.dtype([("k", "u1"), ("i", "<i4", 3)]))
    f["k"], f["i"] = 3, caras
    ruta.write_bytes(texto.encode("ascii") + v.tobytes() + f.tobytes())


def escribe_3mf(ruta: Path, pos: np.ndarray, caras: np.ndarray, rgb: np.ndarray,
                descripcion: str) -> None:
    """3MF con color por vértice: el formato que sustituye al STL para imprimir.

    ⚠️ **El STL no puede llevar esto y por eso existe este fichero.** El STL binario es de
    1987 y solo tiene triángulos; las convenciones que meten RGB de 15 bits en su campo
    *attribute byte count* son no estándar. 3MF lleva color, unidades y metadatos DENTRO.

    El color va como `colorgroup` con un índice por vértice de cada triángulo, no uno por
    triángulo: así el degradado cervical-incisal de una corona sobrevive.
    """
    paleta = np.clip(np.round(np.asarray(rgb, float) / PASO_PALETA_3MF)
                     * PASO_PALETA_3MF, 0, 255).astype(np.uint8)
    unicos, indice = np.unique(paleta, axis=0, return_inverse=True)
    indice = np.asarray(indice).ravel()
    modelo = "".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US"'
        ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
        ' xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
        f'<metadata name="Description">{escape(descripcion)}</metadata>',
        '<resources>',
        '<m:colorgroup id="1">',
        *(f'<m:color color="#{r:02X}{g:02X}{b:02X}"/>' for r, g, b in unicos),
        '</m:colorgroup>',
        '<object id="2" type="model" pid="1" pindex="0"><mesh><vertices>',
        *(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in pos),
        '</vertices><triangles>',
        *(f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="1"'
          f' p1="{indice[a]}" p2="{indice[b]}" p3="{indice[c]}"/>' for a, b, c in caras),
        '</triangles></mesh></object>',
        '</resources><build><item objectid="2"/></build></model>',
    ])
    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
                   'content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.'
                   'openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.'
                   '3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                   'relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0"'
                   ' Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"'
                   '/></Relationships>')
        z.writestr("3D/3dmodel.model", modelo)


def escribe_stl_viscam(ruta: Path, pos: np.ndarray, caras: np.ndarray,
                       rgb: np.ndarray, cabecera: str) -> None:
    """STL binario con color por CARA en RGB555 (convención VisCAM).

    ⚠️ **No es estándar y hay que decirlo cada vez.** El color va en los dos bytes que el
    formato reserva como *attribute byte count* y que la mayoría de lectores ignoran: quien
    abra esto en un visor cualquiera verá la geometría en gris, sin ningún aviso. Se emite
    porque hay cadenas de trabajo que solo aceptan `.stl`, no porque sea buena idea.

    Y es color por CARA: la resolución de color baja a un tercio, porque un triángulo no
    puede llevar tres colores. El degradado dentro de la corona solo sobrevive entero en el
    PLY y en el 3MF.
    """
    v0, v1, v2 = (pos[caras[:, i]] for i in range(3))
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    q = (rgb[caras].mean(axis=1) / 255.0 * 31).astype(np.uint16)
    attr = np.uint16(1 << 15) | (q[:, 2] << 10) | (q[:, 1] << 5) | q[:, 0]
    tri = np.empty(len(caras), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                         ("a", "<u2")]))
    tri["n"], tri["v"], tri["a"] = nrm, pos[caras], attr
    ruta.write_bytes(cabecera.encode("ascii", "replace")[:80].ljust(80, b"\0")
                     + struct.pack("<I", len(caras)) + tri.tobytes())
