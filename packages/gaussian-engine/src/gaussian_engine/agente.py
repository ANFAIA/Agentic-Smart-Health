"""Emite el campo ajustado como un artefacto más del twin, declarado como DERIVADO.

No sustituye al campo semilla. Escribe uno nuevo, con su propio `perfil_campo`, y deja
dicho en el esquema qué significan ahora `scale_*` — que es lo único que cambia de verdad
entre los dos ficheros: en el semilla la escala es el vóxel que produjo la gaussiana, y
aquí es la forma que hace que la densidad se reconstruya. Mismo nombre, distinta cosa. Es
la misma colisión que obliga a declarar el perfil entre el PLY del twin y el del visor.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from core_schemas import ColumnaCampo, TwinSnapshot

from gaussian_engine.ajuste import (
    COMPRESION_FONDO,
    COMPRESION_REGION,
    ITERACIONES,
    TASA,
    Ajuste,
    ajusta,
    ajusta_por_region,
)

PERFIL = "ash-twin-ajustado/1.0"


class CampoStore(Protocol):
    """Lo mínimo del almacén. Protocolo y no importación: igual que en `analysis-agents`."""

    def load(self, ref: str) -> dict[str, np.ndarray]: ...
    def put(self, **arrays: np.ndarray) -> str: ...


def esquema(rmse_hu: float) -> list[ColumnaCampo]:
    """Qué es cada columna del campo ajustado, en el formato que ya usa el contrato.

    `scale_*` lleva el aviso porque es donde está la trampa: quien mida sobre estas
    escalas está midiendo un ajuste, no un tejido.
    """
    forma = (
        f"semieje del elipsoide en mm — AJUSTADO para reconstruir la densidad "
        f"(±{rmse_hu:.0f} HU), NO medido sobre el tejido"
    )
    return [
        *(ColumnaCampo(nombre=n, unidad="mm", significado="centro de la gaussiana")
          for n in ("x", "y", "z")),
        *(ColumnaCampo(nombre=f"scale_{i}", unidad="mm", significado=forma)
          for i in range(3)),
        *(ColumnaCampo(
            nombre=f"rot_{i}", unidad="",
            significado="cuaternion (w, x, y, z) normalizado — orientacion del elipsoide",
        ) for i in range(4)),
        ColumnaCampo(
            nombre="density", unidad="sigma_normalizada",
            significado="amplitud de la gaussiana; sumada con sus vecinas da la densidad",
        ),
    ]


def ajusta_campo(
    snapshot: TwinSnapshot,
    store: CampoStore,
    *,
    n_objetivo: int | None = None,
    compresion: float = COMPRESION_FONDO,
    compresion_region: float = COMPRESION_REGION,
    iteraciones: int = ITERACIONES,
    tasa: float = TASA,
    dispositivo: str | None = None,
) -> tuple[TwinSnapshot, Ajuste]:
    """Snapshot **nuevo** apuntando al campo ajustado. El original no se toca.

    Si el campo trae `region_id` se ajusta por región y `n_objetivo` sobra: el tamaño sale
    de `compresion`, que reparte por ocupación en vez de por prioridad. Sin `region_id`
    hace falta `n_objetivo`.

    Devuelve también el `Ajuste` para que quien lo llame pueda declarar el error: un campo
    derivado sin su error de reconstrucción al lado no se puede auditar.
    """
    if snapshot.gaussian_field_ref is None:
        raise ValueError(
            "El snapshot no tiene campo gaussiano que ajustar. El ajuste refina un campo "
            "existente; no lo crea desde el volumen."
        )
    campo = store.load(snapshot.gaussian_field_ref)
    for clave in ("centers", "density"):
        if clave not in campo:
            raise ValueError(f"El campo referenciado no trae `{clave}`: no es ajustable.")

    hu = campo.get("hu_range", np.asarray([0.0, 1.0]))
    # Si el campo viene segmentado se ajusta REGION A REGION, y no por eficiencia: es lo
    # que hace que la etiqueta de cada elipsoide sea exacta en vez de heredada del vecino
    # mas cercano. Sin `region_id` el visor no puede seleccionar por pieza, y con una
    # etiqueta inferida seleccionaria mal sin decirlo. Ver `ajusta_por_region`.
    if "region_id" in campo:
        r = ajusta_por_region(
            campo["centers"], campo["density"], campo["region_id"],
            compresion=compresion, compresion_region=compresion_region,
            hu_range=hu, iteraciones=iteraciones, tasa=tasa, dispositivo=dispositivo,
        )
    elif n_objetivo is not None:
        r = ajusta(
            campo["centers"], campo["density"], n_objetivo=n_objetivo, hu_range=hu,
            iteraciones=iteraciones, tasa=tasa, dispositivo=dispositivo,
        )
    else:
        raise ValueError(
            "El campo no trae `region_id` y no se dio `n_objetivo`: sin una de las dos "
            "cosas no hay forma de saber cuantas gaussianas se piden."
        )

    arrays = r.como_artefacto()
    # `origin` y `hu_range` viajan tal cual: son lo que hace REVERSIBLE el campo —
    # deshacen el centrado y la normalizacion— y perderlos al derivar dejaria el fichero
    # sin forma de volver a las coordenadas y las unidades del CBCT.
    for clave in ("origin", "hu_range"):
        if clave in campo:
            arrays[clave] = campo[clave]

    nuevo = snapshot.model_copy(update={
        "gaussian_field_ref": store.put(**arrays),
        "perfil_campo": PERFIL,
        "esquema_campo": esquema(r.rmse_hu),
    })
    return nuevo, r
