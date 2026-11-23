"""Un segundo modelo de testigo para las coronas que el principal pierde (§6.16).

**Por qué existe.** C2 separa los dientes mejor que nada de lo medido, pero de vez en cuando
pierde coronas enteras a la encía: en la reserva gastada, 10 piezas pasan a menos del 85 %
de su corona, 6 de ellas en una sola arcada. Salen así de la red, antes de las limpiezas, y
cambian con la semilla. D, el mismo modelo con trozos de 2.048 puntos, no pierde ninguna de
esas 10. Ver `docs/research/segmentacion-fdi-escaner.md`, §6.15 y §6.16.

`rescata_coronas` usa D de testigo: donde D ve una pieza de tamaño real que C2 deja casi
toda como encía, rellena la parte que C2 llama encía. **Solo añade diente**: un vértice que
el principal llama diente no cambia nunca, así que la regla no puede dañar una corona.
Puede, en cambio, traer el desborde del testigo sobre la encía, y eso lo mide el benchmark.

Va antes de las cinco limpiezas, sobre las etiquetas votadas sin limpiar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis_agents.identidad import AREA_MINIMA_MM2, areas_de_instancia
from analysis_agents.margen_metricas import CORONA_MINIMA_CONSERVADA


def rescata_coronas(pos: np.ndarray, caras: np.ndarray, principal: np.ndarray,
                    testigo: np.ndarray, umbral: float = CORONA_MINIMA_CONSERVADA,
                    area_minima: float = AREA_MINIMA_MM2) -> tuple[np.ndarray, list[dict]]:
    """`(etiquetas, registro)`. `principal` y `testigo`: código FDI por vértice, 0 = encía.

    Por cada pieza del testigo de al menos `area_minima` mm² que el principal marca como
    diente en menos de `umbral` de sus vértices, los vértices de esa pieza que el principal
    llama encía reciben un código:

    - el código del principal más frecuente dentro de la pieza, si al menos la mitad de los
      vértices del principal con ese código caen dentro: es el trozo que conserva de ella;
    - si no, el código del testigo.

    El registro tiene una entrada por pieza rescatada: código del testigo, código puesto,
    fracción conservada antes y vértices añadidos.
    """
    principal = np.asarray(principal, dtype=np.int64)
    testigo = np.asarray(testigo, dtype=np.int64)
    if principal.shape != testigo.shape or len(principal) != len(pos):
        raise ValueError("principal, testigo y posiciones no indexan los mismos vértices")
    salida = principal.copy()
    areas = areas_de_instancia(pos, caras, testigo)
    registro: list[dict[str, Any]] = []
    for t in sorted(areas):
        if areas[t] < area_minima:
            continue
        suyos = testigo == t
        conservada = float((principal[suyos] > 0).mean())
        if conservada >= umbral:
            continue
        dentro = principal[suyos]
        codigo = t
        if (dentro > 0).any():
            valores, cuentas = np.unique(dentro[dentro > 0], return_counts=True)
            c = int(valores[np.argmax(cuentas)])
            if 2 * int(cuentas.max()) >= int((principal == c).sum()):
                codigo = c
        rellenar = suyos & (principal == 0)
        salida[rellenar] = codigo
        registro.append({"witness_code": int(t), "assigned_code": int(codigo),
                         "retained_before": conservada, "added_vertices": int(rellenar.sum()),
                         "witness_area_mm2": areas[t]})
    return salida, registro
