"""Siembra del campo de densidad DESCOMPUESTO en capas por rango de HU.

El `cbct-agent` produce UN campo de densidad normalizado sobre `[hu_threshold,
HU_SATURATION]`. Medido (ver `docs/research/3dgs-volumetrico-cbct.md`, replicado sobre la
demo): descomponer ese campo en **tramos disjuntos de HU** —cada uno con su propia
normalización de σ— y entrenarlos por separado gana **+2,48 dB** frente al campo único y
habilita el encendido/apagado por densidad (Beer-Lambert es conmutativo: la suma de las
capas es exacta). Aquí está la siembra; el entrenamiento DRR por capa es el paso siguiente.

La partición es **por densidad, no por anatomía** (la variante cruzada diente/hueso perdió
1,25 dB porque hereda los errores de la segmentación). Cada capa devuelve el MISMO
esquema de campo que el `cbct-agent` (`centers/scales/rotations/density/origin/hu_range/
paso/n_origen`) para que aguas abajo no cambie nada: solo hay N campos en vez de uno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Partición de densidad: `(cota_inf, cota_sup)` del tramo y `(norm_inf, norm_sup)` de
#: la normalización de σ de esa capa. Disjunta y cubre todo `hu >= hu_paciente`.
#: ⚠️ La normalización de `densidad-muy-alta` llega a 3500 (no a 2000 como el campo
#: único): el esmalte y el metal viven ahí y saturar a 2000 les robaba el rango.
PARTICION_HU: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "densidad-baja": ((-np.inf, 700.0), (0.0, 700.0)),
    "densidad-media": ((700.0, 1250.0), (700.0, 1250.0)),
    "densidad-alta": ((1250.0, 2000.0), (1250.0, 2000.0)),
    "densidad-muy-alta": ((2000.0, np.inf), (2000.0, 3500.0)),
}

#: Orden de presentación: de lo más denso a lo menos, como se enciende clínicamente.
ORDEN = ["densidad-muy-alta", "densidad-alta", "densidad-media", "densidad-baja"]


@dataclass(frozen=True)
class CampoBanda:
    """Un campo semilla de UNA capa, con el esquema del `cbct-agent` más su nombre."""

    banda: str
    arrays: dict[str, np.ndarray]
    n_primitivas: int
    hu: np.ndarray


def _submuestrea(occupied: np.ndarray, spacing: tuple[float, float, float],
                 max_primitivas: int) -> tuple[np.ndarray, np.ndarray]:
    """El diezmado en REJILLA del `cbct-agent`: paso por eje, lo más isótropo posible.

    Devuelve `(occupied_subsampleado, paso)` con `paso` en el orden de `occupied`
    (z, y, x), como lo guarda el agente.
    """
    paso = np.ones(3, dtype=np.int64)
    if occupied.shape[0] <= max_primitivas:
        return occupied, paso
    # `spacing` va en orden mundo (x, y, z); `occupied` en (z, y, x).
    mm = np.asarray(spacing, dtype=np.float64)[::-1]
    while True:
        se_queda = ((occupied % paso) == 0).all(axis=1)
        if int(se_queda.sum()) <= max_primitivas:
            break
        paso[int(np.argmin(mm * paso))] += 1
    return occupied[se_queda], paso


def siembra_por_banda(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    z: np.ndarray,
    *,
    hu_paciente: float = 300.0,
    max_primitivas: int = 500_000,
) -> list[CampoBanda]:
    """N campos semilla, uno por tramo de HU, con el MISMO `origin` compartido.

    `volume` es el volumen CBCT en HU (orden z, y, x); `spacing` el espaciado en mm
    (sx, sy, sz); `z` la posición real (mm) de cada corte, como la lee el `cbct-agent`
    (no `índice × espaciado`: con cortes ausentes eso desplaza todo lo de encima).
    """
    sx, sy, _ = spacing
    mascara = volume >= hu_paciente

    # `origin` se calcula UNA vez sobre la unión (todo el paciente) y se comparte: si
    # cada capa se centrara por su cuenta, en mundo no quedarían alineadas al sumarlas.
    union = np.argwhere(mascara)
    if union.size == 0:
        raise ValueError(f"Ningún vóxel supera {hu_paciente} HU: serie vacía o umbral mal.")
    mundo_union = np.column_stack(
        [union[:, 2] * sx, union[:, 1] * sy, z[union[:, 0]]]
    ).astype(np.float64)
    origin = mundo_union.mean(axis=0)

    capas: list[CampoBanda] = []
    for banda in ORDEN:
        (p_lo, p_hi), (n_lo, n_hi) = PARTICION_HU[banda]
        dentro = mascara & (volume >= p_lo) & (volume < p_hi)
        occupied = np.argwhere(dentro)
        n_origen = int(occupied.shape[0])
        if n_origen == 0:
            continue  # tramo sin vóxeles en este volumen (p. ej. sin esmalte)
        occupied, paso = _submuestrea(occupied, spacing, max_primitivas)

        mundo = np.column_stack(
            [occupied[:, 2] * sx, occupied[:, 1] * sy, z[occupied[:, 0]]]
        ).astype(np.float64)
        centers = (mundo - origin).astype(np.float32)
        hu = volume[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        density = np.clip((hu - n_lo) / (n_hi - n_lo), 0.0, 1.0).astype(np.float32)
        scales = np.tile(
            (np.asarray(spacing, dtype=np.float64) * paso[::-1] * 0.5).astype(np.float32),
            (centers.shape[0], 1),
        )
        rotations = np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (centers.shape[0], 1)
        )
        arrays = {
            "centers": centers,
            "scales": scales,
            "rotations": rotations,
            "density": density,
            "origin": origin.astype(np.float64),
            "hu_range": np.asarray([n_lo, n_hi], dtype=np.float64),
            # El tramo de HU de la partición, ya recortado por la máscara de paciente
            # (qué vóxeles caen AQUÍ): lo necesita quien compute la DRR objetivo de la
            # capa, y sin el recorte la banda baja arrastraría tejido blando que la
            # semilla no sembró.
            "hu_particion": np.asarray([max(hu_paciente, p_lo), p_hi], dtype=np.float64),
            "paso": paso.astype(np.int64),
            "n_origen": n_origen,
        }
        capas.append(CampoBanda(banda=banda, arrays=arrays,
                                n_primitivas=int(centers.shape[0]), hu=hu))
    return capas
