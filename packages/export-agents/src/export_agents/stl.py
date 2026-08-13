"""`export-agent` — `TwinSnapshot` → **STL**. La fase 6 del pipeline.

Cierra el círculo del brief: la ingesta convierte la malla del escáner en contrato
(`surface_ref`) y este agente la devuelve a un fichero, con el **error de
reconstrucción medido** en vez de prometido.

**De dónde sale la geometría: de `surface_ref`, no del campo gaussiano.** Es la
decisión de diseño de este agente y no es un atajo.

* El `mesh-agent` guarda la superficie de origen **tal cual** —posiciones en
  `float64` y la topología de caras completa—, precisamente para que exista una
  copia fiel desde la que reconstruir. Exportar desde ahí hace que el round-trip
  fichero → twin → fichero tenga como único error el que impone el **formato**
  (~10⁻⁵ mm, ver más abajo), cuatro órdenes de magnitud por debajo del presupuesto
  de 0,1 mm.
* Sacar la malla del campo gaussiano por *marching cubes* sería la otra ruta, y es
  **lossy y mal condicionada**. Medido en `scripts/resolucion_modalidades.py`: el
  área de la isosuperficie depende de con qué resolución la midas salvo donde el
  gradiente es fuerte —el esmalte, 364 HU/vóxel, dimensión fractal 2,10—; sobre
  hueso trabecular (60–80 HU/vóxel, dimensión 2,45) el área **no existe** como
  magnitud. Un STL denso sacado de un CBCT es una malla **suave, no precisa**. Si
  algún día hace falta esa ruta será **otro agente**, con su propio criterio de
  aceptación, no una bandera de éste.

Por eso un snapshot sin `surface_ref` no se «rescata» mallando el volumen: se
declara `MISSING` y se dice por qué.

**Lo que el formato no puede llevar, y se declara en vez de fingirse.** El STL es
«pelado» por construcción: sin color por vértice, sin normales por vértice y sin
topología compartida (es una **sopa de triángulos**, repite cada vértice). El
`color_superficie` que sí tiene el twin no cabe en el fichero — sigue vivo en
`surface_ref`, y quien quiera color necesita OBJ/PLY, no STL. Y las coordenadas
son `float32` por especificación: de ahí sale el error residual del round-trip, que
este agente **mide releyendo el fichero** en vez de estimarlo.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Literal

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot
from fusion_agents.registration import apply, quaternion_to_matrix

from export_agents.base import (
    REVERSIBILITY_BUDGET_MM,
    BaseExportAgent,
    ExportOutput,
    SurfaceStore,
)

# Sistema de referencia en el que escribir la geometría.
Frame = Literal["source", "twin"]
FRAMES: tuple[Frame, ...] = ("source", "twin")

# El triángulo del STL binario: normal + 3 vértices + 2 bytes de atributo = 50 B.
# Mismo dtype que usa `ingestion_agents.parse_stl` para leerlo: si las dos caras del
# round-trip no comparten la descripción del formato, divergen en silencio.
_STL_TRI = np.dtype([("n", "<3f4"), ("v", "<3,3f4"), ("attr", "<u2")])
_HEADER_BYTES = 80


def face_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normal por **cara**, que es la única que el STL sabe guardar.

    No se reutilizan las normales por vértice del artefacto: el formato no tiene
    dónde ponerlas. Se recalculan desde las posiciones que se van a escribir, así que
    el fichero es autoconsistente. Un triángulo degenerado (área 0) queda con normal
    nula —legal en STL, y los lectores la recalculan— en vez de inventarse una
    dirección.
    """
    v0, v1, v2 = (positions[faces[:, i]].astype(np.float64) for i in range(3))
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, lengths, out=normals, where=lengths > 0)
    return normals


def stl_header(text: str) -> bytes:
    """Los 80 bytes de cabecera, con la marca de procedencia truncada y rellenada.

    La cabecera es texto libre y casi todo el mundo la desperdicia. Aquí se usa para
    que el fichero **lleve encima de qué twin salió**: es trazabilidad que sobrevive
    a que alguien copie el STL fuera del sistema, donde ya no hay `Provenance` que
    consultar. Es también donde un snapshot parcial deja de estar callado.

    No puede empezar por `solid`: muchos lectores usan justo eso para decidir que un
    STL es ASCII, y el fichero se leería como texto. Empieza por `ASH `.
    """
    crudo = text.encode("ascii", errors="replace")[:_HEADER_BYTES]
    return crudo.ljust(_HEADER_BYTES, b"\0")


def write_binary_stl(
    path: Path, positions: np.ndarray, faces: np.ndarray, *, header: str = ""
) -> None:
    """Escribe la malla como STL binario. Valida **antes** de tocar el disco.

    Las validaciones no son ceremonia: un índice de cara fuera de rango produciría un
    fichero sintácticamente válido con geometría inventada —el peor fallo posible en
    un artefacto clínico, porque se abre sin protestar—. Y la escritura es atómica
    (temporal + `replace`, como el `ArtifactStore`): un STL a medio escribir por un
    disco lleno no debe quedar donde alguien lo confunda con el bueno.
    """
    if faces.size == 0:
        raise ValueError("La malla no tiene caras: no hay superficie que exportar.")
    if not np.isfinite(positions).all():
        raise ValueError("La malla trae coordenadas no finitas (NaN/inf): no se exporta.")
    if faces.min() < 0 or faces.max() >= len(positions):
        raise ValueError(
            f"Índice de cara fuera de rango: la malla tiene {len(positions)} vértices y "
            f"una cara referencia el {int(faces.max())}."
        )

    tri = np.zeros(len(faces), dtype=_STL_TRI)
    tri["v"] = positions[faces]  # (F, 3, 3) → float32 al asignar, como manda el formato
    tri["n"] = face_normals(positions, faces)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(stl_header(header))
        fh.write(struct.pack("<I", len(faces)))
        fh.write(tri.tobytes())
    tmp.replace(path)


def read_stl_triangles(path: Path) -> np.ndarray:
    """Relee un STL binario como `(F, 3, 3)` en el orden en que se escribió.

    Existe además de `ingestion_agents.parse_stl` porque las dos preguntas son
    distintas: aquél **reconstruye la topología** (deduplica con `np.unique`, que
    reordena) para ingerir, y aquí hace falta la sopa de triángulos **tal cual quedó
    en el fichero** para poder comparar vértice a vértice con lo que se le pidió
    escribir. Deduplicar antes de comparar escondería justo los errores de orden.
    """
    data = path.read_bytes()
    if len(data) < _HEADER_BYTES + 4:
        raise ValueError(f"El STL escrito no tiene ni cabecera: {path}")
    n_tri = struct.unpack_from("<I", data, _HEADER_BYTES)[0]
    esperado = _HEADER_BYTES + 4 + n_tri * _STL_TRI.itemsize
    if len(data) != esperado:
        raise ValueError(
            f"El STL escrito mide {len(data)} B y sus {n_tri} triángulos exigen {esperado} B."
        )
    tri = np.frombuffer(data, dtype=_STL_TRI, count=n_tri, offset=_HEADER_BYTES + 4)
    return tri["v"].astype(np.float64)


class ExportAgent(BaseExportAgent):
    """Regenera la malla del `TwinSnapshot` como STL, y mide cuánto se parece.

    **La medida es el producto, tanto como el fichero.** `max_deviation_mm` sale de
    releer lo que se acaba de escribir y compararlo con la geometría del twin, no de
    una estimación del error del formato. La diferencia importa: una estimación
    sobrevive intacta a un bug de endianness o de orden de vértices, y una relectura
    no. Es la métrica de éxito del brief («error de malla < 0,1 mm») convertida en
    algo que se comprueba en cada exportación.

    **Dos sistemas de referencia, explícitos.** El artefacto de la malla vive en el
    sistema del escáner —la fusión geométrica **no reescribe blobs**, solo anota en
    `Provenance.transform` la transformación que alinea la malla con el CBCT—. Así
    que exportar admite dos preguntas legítimas y distintas:

    * `frame="source"` (por defecto) — el STL tal como entró. Es el que compara con
      el fichero original y el que mide la reversibilidad.
    * `frame="twin"` — la malla llevada al sistema del CBCT, aplicando la
      transformación registrada. Es el que sirve para superponer con el volumen.

    Pedir `frame="twin"` sobre un snapshot que nunca pasó por fusión geométrica es un
    error declarado, no un fichero silenciosamente sin transformar: entregar la malla
    en el sistema equivocado sin decirlo es exactamente la clase de fallo callado que
    el ADR 004 quiere imposible.
    """

    name = "export-agent"
    version = "0.1.0"

    def __init__(
        self,
        store: SurfaceStore,
        *,
        frame: Frame = "source",
        verify: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if frame not in FRAMES:
            raise ValueError(f"frame debe ser uno de {FRAMES}, no {frame!r}")
        self.store = store
        self.frame: Frame = frame
        self.verify = verify

    def _export(  # type: ignore[override]
        self, snapshot: TwinSnapshot, destination: Path, *, frame: Frame | None = None
    ) -> ExportOutput:
        marco: Frame = frame or self.frame
        if marco not in FRAMES:
            raise ValueError(f"frame debe ser uno de {FRAMES}, no {marco!r}")

        if snapshot.surface_ref is None:
            # Ausencia de entrada, no fallo del exportador: este snapshot no trae
            # malla. Mallar el campo gaussiano NO es la alternativa (ver el docstring
            # del módulo: sobre hueso trabecular el área ni siquiera está definida).
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "El snapshot no tiene `surface_ref`: no se ingirió malla intraoral en "
                    "esta adquisición. Este agente no malla el campo gaussiano — sería una "
                    "superficie interpolada, no medida."
                ),
                frame=marco,
                format="stl",
            )

        # Invariante *fail-loud* del ADR 001: al exportar hay que validar que el blob
        # referenciado existe. `load` lanza si la referencia cuelga, y `export` lo
        # convierte en `FAILED` + cuarentena. Una referencia colgante es un error, no
        # un modelo vacío silencioso.
        arrays = self.store.load(snapshot.surface_ref)
        for clave in ("positions", "faces"):
            if clave not in arrays:
                raise ValueError(
                    f"El artefacto {snapshot.surface_ref} no contiene `{clave}`: no es una "
                    "malla exportable."
                )
        positions = np.asarray(arrays["positions"], dtype=np.float64)
        faces = np.asarray(arrays["faces"])

        if marco == "twin":
            positions = self._to_twin_frame(snapshot, positions)

        destination.parent.mkdir(parents=True, exist_ok=True)
        motivos = self._partial_reasons(snapshot)
        write_binary_stl(
            destination,
            positions,
            faces,
            header=self._header(snapshot, marco, parcial=bool(motivos)),
        )

        desviacion = self._verify(destination, positions, faces) if self.verify else None
        if desviacion is not None and desviacion > REVERSIBILITY_BUDGET_MM:
            motivos.append(
                f"la malla exportada se desvía {desviacion:.4f} mm de la del twin, por "
                f"encima del presupuesto de {REVERSIBILITY_BUDGET_MM} mm del brief."
            )

        return self._outcome(
            ModalityStatus.OK,
            path=destination,
            format="stl",
            frame=marco,
            n_vertices=int(len(positions)),
            n_faces=int(len(faces)),
            max_deviation_mm=desviacion,
            hitl_reasons=motivos,
            detail=self._detail(arrays, desviacion),
        )

    # --- piezas ---------------------------------------------------------- #
    def _to_twin_frame(self, snapshot: TwinSnapshot, positions: np.ndarray) -> np.ndarray:
        """Lleva la malla al sistema del CBCT con la transformación registrada.

        Se aplica **en el sentido en que se midió**: la fusión geométrica registra la
        malla (`source`) sobre el CBCT (`target`), así que la transformación guardada
        va de escáner a twin. Deshacerla es `transform.inverse()`, exacto por
        construcción — es lo que hace auditable la reversibilidad (ADR 004 §2.2).
        """
        transform = snapshot.provenance.transform
        if transform is None:
            raise ValueError(
                "Se pidió exportar en el sistema del twin y el snapshot no tiene "
                "`provenance.transform`: nunca pasó por la fusión geométrica. Sin esa "
                "transformación, escribir la malla sería entregarla en el sistema del "
                "escáner haciéndola pasar por la del CBCT."
            )
        return apply(
            quaternion_to_matrix(transform.rotation),
            np.asarray(transform.translation, dtype=np.float64),
            positions,
        )

    def _verify(self, path: Path, positions: np.ndarray, faces: np.ndarray) -> float:
        """Relee el fichero y devuelve la desviación máxima en mm, por coordenada.

        Compara contra `positions[faces]`, es decir contra lo que se **pidió**
        escribir, no contra la versión ya redondeada a `float32`. Así el número
        incluye la pérdida del formato en vez de esconderla: es lo que de verdad
        recuperaría quien abra el STL.
        """
        leidos = read_stl_triangles(path)
        return float(np.abs(leidos - positions[faces]).max())

    def _header(self, snapshot: TwinSnapshot, frame: Frame, *, parcial: bool) -> str:
        marca = f"ASH {self.qualified} {snapshot.acquisition_id} frame={frame}"
        return f"{marca} PARCIAL" if parcial else marca

    def _detail(self, arrays: dict[str, np.ndarray], desviacion: float | None) -> str:
        partes = [
            f"malla exportada con desviación máxima {desviacion:.6f} mm"
            if desviacion is not None
            else "malla exportada sin verificar (verify=False): sin medida de reversibilidad"
        ]
        if "colors_rgb8" in arrays:
            partes.append(
                "el color por vértice del twin no viaja en el STL (el formato es pelado): "
                "sigue en `surface_ref`"
            )
        return "; ".join(partes) + "."
