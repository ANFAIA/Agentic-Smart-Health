"""Exporta el **campo gaussiano** del `TwinSnapshot` a PLY binario, en mm reales.

Es el segundo canal de la familia de exportación: `gaussian_field_ref` → fichero. El
primero (`stl.py`) materializa la superficie medida por el escáner; éste materializa lo
que el escáner no puede ver, que es el interior que sembró el CBCT.

## Por qué NO es un `.ply` de 3D Gaussian Splatting

La convención de INRIA guarda `opacity` y armónicos esféricos `f_dc_*`/`f_rest_*`, o sea
**color y transparencia**. Este campo no tiene ninguno de los dos: el contrato sustituye
la opacidad α por `density` (σₙ ≥ 0), que es atenuación radiológica Beer-Lambert, y un
CBCT no mide color. Escribir un PLY con la cabecera de INRIA lo haría abrible en
cualquier visor de splats — y ese visor pintaría un color inventado sobre una magnitud
física, que es peor que no abrirlo.

Así que se escriben **las propiedades que existen** y la cabecera declara las unidades.
Quien quiera un splat de color tiene que decidir de dónde saca el color, y eso es una
decisión de producto, no una conversión de formato.

## Dos marcos, los dos definidos por el dato

El `cbct-agent` centra el campo en el origen (`centers -= mundo.mean(0)`) y **guarda el
desplazamiento** en `origin`. Con eso hay dos preguntas legítimas:

* `frame="twin"` (por defecto) — tal como vive en el twin, centrado. Es el marco en el
  que están las coordenadas del snapshot.
* `frame="cbct"` — devuelto a los milímetros del DICOM, sumando `origin`. Es el que
  superpone con la serie original y con cualquier medida hecha sobre ella.

⚠ **Diferencia con `stl.py`, y no es cosmética.** Allí `frame="twin"` aplica
`Provenance.transform`, que la escribe la fusión geométrica registrando contra lo que el
llamante le pase como `target` — el pipeline no fija ese marco. Aquí los dos marcos salen
de arrays del propio artefacto, así que no dependen de quién llamó. Cerrar ese hueco del
contrato de fusión es trabajo del protocolo entre agentes, no de un exportador.

## Precisión: se escribe `double`, a propósito

Un splat convencional usa float32. Aquí las posiciones van en float64 para que la
verificación por relectura mida **bugs** —endianness, orden de propiedades, un stride mal
calculado— y no el redondeo del formato. Si el número que sale de `_verify` no es 0, hay
un fallo real que buscar; con float32 saldría siempre un ruido de fondo que taparía uno
pequeño.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from core_schemas import ColumnaCampo, ModalityStatus, TwinSnapshot

from export_agents.base import (
    REVERSIBILITY_BUDGET_MM,
    BaseExportAgent,
    ExportOutput,
    SurfaceStore,
)

FieldFrame = Literal["twin", "cbct"]
FIELD_FRAMES: tuple[FieldFrame, ...] = ("twin", "cbct")

_PROPIEDADES: tuple[tuple[str, str], ...] = (
    ("x", "double"), ("y", "double"), ("z", "double"),
    ("scale_0", "float"), ("scale_1", "float"), ("scale_2", "float"),
    ("rot_0", "float"), ("rot_1", "float"), ("rot_2", "float"), ("rot_3", "float"),
    ("density", "float"),
)

# Propiedades que solo existen si alguien las produjo. `region_id` la escribe el
# `segmentation-agent`, y un campo sin segmentar no la tiene — declararla siempre
# obligaría a rellenarla con ceros, que en el convenio del agente significan «sin
# asignar» y son indistinguibles de un campo que sí se segmentó y no encontró nada.
_OPCIONALES: tuple[tuple[str, str], ...] = (
    ("region_id", "short"),
    # De qué modalidad viene cada gaussiana. Solo la escribe el compuesto, porque solo él
    # mezcla dos: un campo de una sola fuente no la necesita. Ver `compuesto.py`.
    ("origen", "short"),
)
_TIPOS = {"double": np.float64, "float": np.float32, "short": np.int16}

_CLAVES_MINIMAS = ("centers", "scales", "rotations", "density")

# Qué columnas escribe cada array del campo. Lo declara el exportador porque es quien lo
# sabe: `centers` sale como x/y/z y `rotations` como `rot_*`, y ninguna regla de recorte
# de cadenas acierta con las dos.
#
# Sirve además de contrato hacia fuera: un array del campo que NO esté aquí es uno que
# este exportador no sabe escribir, y por tanto se pierde al exportar. Eso es exactamente
# lo que le pasó a `region_id`, y el orquestador lo comprueba con este mapa
# (`IngestionPipeline._columnas_del_campo`).
COLUMNAS_DE_ARRAY: dict[str, tuple[str, ...]] = {
    "centers": ("x", "y", "z"),
    "scales": ("scale_0", "scale_1", "scale_2"),
    "rotations": ("rot_0", "rot_1", "rot_2", "rot_3"),
    "density": ("density",),
    "region_id": ("region_id",),
    "origen": ("origen",),
}


# Qué es cada columna, **en máquina**. Vive aquí, junto a `COLUMNAS_DE_ARRAY`, porque es
# el mismo conocimiento: el exportador es quien sabe qué escribe y con qué semántica.
#
# ⚠️ Lo importante de esta tabla es `escala`. El PLY de facto de 3DGS usa `scale_0..2` y
# `rot_0..3` con los MISMOS nombres y guarda el **logaritmo** de la escala; aquí van
# milímetros lineales. Un visor estándar abriendo este fichero no fallaría: exponenciaría
# nuestros milímetros y renderizaría basura con buen aspecto. Por eso el snapshot declara
# además `perfil_campo`, para que un lector pueda negarse en vez de adivinar.
ESQUEMA_COLUMNAS: dict[str, dict] = {
    "x": {"unidad": "mm", "significado": "centro de la gaussiana"},
    "y": {"unidad": "mm", "significado": "centro de la gaussiana"},
    "z": {"unidad": "mm", "significado": "centro de la gaussiana"},
    "scale_0": {"unidad": "mm", "significado": "sigma del elipsoide, NO su logaritmo"},
    "scale_1": {"unidad": "mm", "significado": "sigma del elipsoide, NO su logaritmo"},
    "scale_2": {"unidad": "mm", "significado": "sigma del elipsoide, NO su logaritmo"},
    "rot_0": {"significado": "cuaternion (w, x, y, z) normalizado — componente w"},
    "rot_1": {"significado": "cuaternion (w, x, y, z) normalizado — componente x"},
    "rot_2": {"significado": "cuaternion (w, x, y, z) normalizado — componente y"},
    "rot_3": {"significado": "cuaternion (w, x, y, z) normalizado — componente z"},
    "density": {
        "unidad": "sigma_normalizada",
        "significado": "atenuacion Beer-Lambert en [0,1] sobre `hu_range`. NO es opacidad",
    },
    "region_id": {
        "significado": "diente al que pertenece la gaussiana; 0 = sin asignar",
        "vocabulario": "ISO-3950",
        "medido": False,
        "derivado_de": "segmentation-agent",
    },
    "origen": {
        "significado": "modalidad de la que viene: 0 = CBCT (densidad medida), "
        "1 = escaner intraoral (forma medida)",
        "medido": False,
        "derivado_de": "composite-export-agent",
    },
}


def esquema_del_campo(arrays: dict) -> list[ColumnaCampo]:
    """Las columnas que este exportador escribiria para `arrays`, ya descritas.

    Se deriva de lo que el campo trae, no de una lista fija: un campo sin segmentar no
    declara `region_id`, igual que no la escribe. Declarar de mas seria tan mentira como
    declarar de menos.
    """
    columnas = []
    for clave in arrays:
        for prop in COLUMNAS_DE_ARRAY.get(clave, ()):
            spec = ESQUEMA_COLUMNAS.get(prop)
            if spec is not None:
                columnas.append(ColumnaCampo(nombre=prop, **spec))
    return columnas


def densidad_a_hu(density: np.ndarray, hu_range: np.ndarray) -> np.ndarray:
    """Deshace la normalización del `cbct-agent`: σ ∈ [0, 1] → unidades Hounsfield.

    ⚠ **No es invertible en la saturación.** El agente satura a 1 todo lo que pasa de
    `hu_range[1]`, así que un σ de 1,0 significa «≥ ese valor», no «ese valor». El esmalte
    cae ahí de lleno. Se devuelve el extremo porque es la cota inferior correcta, pero
    quien mida densidades absolutas de esmalte tiene que volver al DICOM.
    """
    bajo, alto = float(hu_range[0]), float(hu_range[1])
    return np.asarray(density, dtype=np.float64) * (alto - bajo) + bajo


def escribe_ply(destino: Path, columnas: dict[str, np.ndarray], *, comentarios: list[str]) -> None:
    """PLY binario little-endian: `_PROPIEDADES` en ese orden, más las opcionales que haya.

    Las obligatorias no se buscan en `columnas`: si falta una, esto revienta, que es lo
    correcto — un campo gaussiano sin `density` no es un campo gaussiano.
    """
    props = [*_PROPIEDADES, *(p for p in _OPCIONALES if p[0] in columnas)]
    n = len(columnas["x"])
    cabecera = ["ply", "format binary_little_endian 1.0"]
    cabecera += [f"comment {c}" for c in comentarios]
    cabecera.append(f"element vertex {n}")
    cabecera += [f"property {tipo} {nombre}" for nombre, tipo in props]
    cabecera.append("end_header")

    dtype = np.dtype([(nombre, _TIPOS[tipo]) for nombre, tipo in props])
    filas = np.empty(n, dtype=dtype)
    for nombre, _ in props:
        filas[nombre] = columnas[nombre]

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("wb") as fh:
        fh.write(("\n".join(cabecera) + "\n").encode("ascii"))
        fh.write(filas.tobytes())


def lee_ply(ruta: Path) -> dict[str, np.ndarray]:
    """Relee lo escrito por `escribe_ply`. Es la mitad que hace medible el round-trip.

    Se parsea la cabecera de verdad —no se asume el orden ni el número de propiedades—
    porque el punto de releer es cazar precisamente ese tipo de desajuste.
    """
    with ruta.open("rb") as fh:
        lineas: list[str] = []
        while True:
            linea = fh.readline()
            if not linea:
                raise ValueError(f"{ruta} se acaba sin `end_header`: no es un PLY completo.")
            texto = linea.decode("ascii").strip()
            lineas.append(texto)
            if texto == "end_header":
                break
        if lineas[0] != "ply":
            raise ValueError(f"{ruta} no empieza por `ply`.")
        if "binary_little_endian" not in lineas[1]:
            raise ValueError(f"Formato no soportado: {lineas[1]!r}")

        n = next(
            (int(x.split()[2]) for x in lineas if x.startswith("element vertex")), None
        )
        if n is None:
            raise ValueError(f"{ruta} no declara `element vertex`.")
        props = [
            (x.split()[2], x.split()[1]) for x in lineas if x.startswith("property ")
        ]
        dtype = np.dtype([(nombre, _TIPOS[tipo]) for nombre, tipo in props])
        crudo = fh.read()

    esperado = n * dtype.itemsize
    if len(crudo) != esperado:
        raise ValueError(
            f"{ruta} declara {n} vértices ({esperado} bytes de carga) pero trae "
            f"{len(crudo)}. El fichero está truncado o la cabecera miente."
        )
    filas = np.frombuffer(crudo, dtype=dtype, count=n)
    return {nombre: np.asarray(filas[nombre]) for nombre, _ in props}


class FieldExportAgent(BaseExportAgent):
    """Materializa `gaussian_field_ref` como PLY, y mide cuánto se parece al twin.

    Como en `stl.py`, **la medida es el producto tanto como el fichero**:
    `max_deviation_mm` sale de releer lo que se acaba de escribir, no de estimar el error
    del formato.
    """

    name = "field-export-agent"
    version = "0.1.0"

    def __init__(
        self,
        store: SurfaceStore,
        *,
        frame: FieldFrame = "twin",
        verify: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if frame not in FIELD_FRAMES:
            raise ValueError(f"frame debe ser uno de {FIELD_FRAMES}, no {frame!r}")
        self.store = store
        self.frame: FieldFrame = frame
        self.verify = verify

    def _export(  # type: ignore[override]
        self, snapshot: TwinSnapshot, destination: Path, *, frame: FieldFrame | None = None
    ) -> ExportOutput:
        marco: FieldFrame = frame or self.frame
        if marco not in FIELD_FRAMES:
            raise ValueError(f"frame debe ser uno de {FIELD_FRAMES}, no {marco!r}")

        # `gaussian_field_ref` es obligatorio en el contrato: sin campo no hay snapshot,
        # así que aquí una referencia ausente no es «modalidad que falta» sino un twin
        # mal formado. `load` lanza si cuelga, y `export` lo convierte en FAILED.
        arrays = self.store.load(snapshot.gaussian_field_ref)
        faltan = [k for k in _CLAVES_MINIMAS if k not in arrays]
        if faltan:
            raise ValueError(
                f"El artefacto {snapshot.gaussian_field_ref} no contiene "
                f"{', '.join(f'`{k}`' for k in faltan)}: no es un campo gaussiano exportable."
            )

        centers = np.asarray(arrays["centers"], dtype=np.float64)
        comentarios = [
            f"generado por {self.qualified}",
            f"acquisition_id {snapshot.acquisition_id}",
            f"frame {marco}",
            "density es sigma_n normalizada en [0,1] (atenuacion Beer-Lambert), NO opacidad",
            "scale en mm; rot es cuaternion (w,x,y,z)",
        ]

        if marco == "cbct":
            if "origin" not in arrays:
                # Fail-loud del ADR 001: el desplazamiento depende del dato y ninguna
                # versión del agente lo recomputa. Entregar el campo centrado diciendo
                # que va en coordenadas del CBCT sería el fallo callado que peor escala:
                # todo lo que se mida encima saldría desplazado y con buen aspecto.
                raise ValueError(
                    f"El artefacto {snapshot.gaussian_field_ref} no trae `origin`, así que "
                    "no se puede devolver a coordenadas del CBCT. Lo ingirió una versión "
                    "del `cbct-agent` anterior a que se guardase el centrado; hay que "
                    "reingerir la serie. Usa frame='twin' para exportarlo tal como está."
                )
            origin = np.asarray(arrays["origin"], dtype=np.float64)
            centers = centers + origin
            comentarios.append(f"origin_mm {origin[0]!r} {origin[1]!r} {origin[2]!r}")
            comentarios.append("coordenadas en mm del DICOM (centers + origin)")
        else:
            comentarios.append("coordenadas centradas en el origen; suma `origin` para el CBCT")

        if "hu_range" in arrays:
            bajo, alto = np.asarray(arrays["hu_range"], dtype=np.float64)
            comentarios.append(f"hu_range {bajo!r} {alto!r}  (hu = density*(alto-bajo)+bajo)")

        escalas = np.asarray(arrays["scales"], dtype=np.float32)
        rotaciones = np.asarray(arrays["rotations"], dtype=np.float32)
        densidad = np.asarray(arrays["density"], dtype=np.float32)
        columnas: dict[str, np.ndarray] = {
            "x": centers[:, 0], "y": centers[:, 1], "z": centers[:, 2],
            "scale_0": escalas[:, 0], "scale_1": escalas[:, 1], "scale_2": escalas[:, 2],
            "rot_0": rotaciones[:, 0], "rot_1": rotaciones[:, 1],
            "rot_2": rotaciones[:, 2], "rot_3": rotaciones[:, 3],
            "density": densidad,
        }

        # El campo segmentado trae una etiqueta FDI por gaussiana, y perderla al exportar
        # dejaría el fichero mudo sobre lo único que el pipeline sabe de anatomía: qué
        # trozo es qué diente. Se escribe solo si existe — ver `_OPCIONALES`.
        if "region_id" in arrays:
            region = np.asarray(arrays["region_id"], dtype=np.int16)
            columnas["region_id"] = region
            dientes = np.unique(region[region > 0])
            comentarios.append(
                f"region_id es el codigo FDI por gaussiana, 0 = sin asignar "
                f"({len(dientes)} diente(s) etiquetado(s))"
            )

        motivos = self._partial_reasons(snapshot)
        if motivos:
            comentarios.append("PARCIAL: este twin requiere revision humana")
        escribe_ply(destination, columnas, comentarios=comentarios)

        desviacion = self._verify(destination, centers) if self.verify else None
        if desviacion is not None and desviacion > REVERSIBILITY_BUDGET_MM:
            motivos.append(
                f"el campo exportado se desvía {desviacion:.4f} mm del del twin, por "
                f"encima del presupuesto de {REVERSIBILITY_BUDGET_MM} mm del brief."
            )

        return self._outcome(
            ModalityStatus.OK,
            path=destination,
            format="ply",
            frame=marco,
            # Una gaussiana es un vértice del PLY. No hay caras: un campo no es una malla,
            # y dejar `n_faces` a None lo dice mejor que un 0 que parece una malla vacía.
            n_vertices=int(len(centers)),
            max_deviation_mm=desviacion,
            hitl_reasons=motivos,
        )

    def _verify(self, ruta: Path, centers: np.ndarray) -> float:
        """Desviación máxima entre lo que se quiso escribir y lo que quedó en disco."""
        leido = lee_ply(ruta)
        vuelta = np.column_stack([leido["x"], leido["y"], leido["z"]])
        if vuelta.shape != centers.shape:
            raise ValueError(
                f"La relectura de {ruta} da {vuelta.shape} y se escribieron "
                f"{centers.shape}: el fichero no representa el campo."
            )
        return float(np.abs(vuelta - centers).max())
