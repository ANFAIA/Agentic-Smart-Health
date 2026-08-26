"""El ancho mesiodistal de una corona etiquetada, contra la tabla anatomica.

Es la unica medida del proyecto que mira la FRONTERA entre dos piezas. Ver
`export_agents.anatomia.anchos_de_corona`.
"""

from __future__ import annotations

import numpy as np
from export_agents.anatomia import anchos_de_corona


def test_ancho_de_corona_detecta_la_etiqueta_que_se_pasa_al_vecino() -> None:
    """Una rejilla continua partida en dos coronas: si la etiqueta de una invade a la
    otra, su ancho la delata.

    ⚠️ Es la unica medida del proyecto que mira la FRONTERA. El acierto por diente que
    declara el `segmentation-agent` pregunta si el codigo mayoritario es el correcto, y
    una frontera corrida no cambia una mayoria — se puede tener 0,93 por diente y una
    corona del doble de ancha. Sobre el caso real, 9 de 15 coronas se pasan de 1,5 mm.

    La rejilla es CONTINUA a proposito: dos placas sueltas no valdrian, porque la parte
    invadida quedaria en otra componente conexa y la medida —que se queda con la mayor—
    ni la veria. En una malla de escaner las dos coronas si estan cosidas por el contacto.
    """
    nx, ny = 35, 8
    gx = np.linspace(0.0, 17.0, nx)
    gy = np.linspace(0.0, 6.0, ny)
    malla = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    pos = np.column_stack([malla, np.zeros(len(malla))])
    caras = np.asarray([
        cara
        for i in range(nx - 1)
        for j in range(ny - 1)
        for cara in ([i * ny + j, i * ny + j + 1, (i + 1) * ny + j],
                     [i * ny + j + 1, (i + 1) * ny + j + 1, (i + 1) * ny + j])
    ])

    justas = np.where(pos[:, 0] < 8.5, 11, 21)
    medido = anchos_de_corona(pos, caras, justas)
    assert medido[11][1] == 8.5  # la tabla del central superior
    assert abs(medido[11][0] - 8.5) < 0.8

    # Ahora el 11 se come 3 mm del 21.
    peor = anchos_de_corona(pos, caras, np.where(pos[:, 0] < 11.5, 11, 21))
    assert peor[11][0] - medido[11][0] > 2.0
    assert peor[21][0] < medido[21][0] - 2.0
