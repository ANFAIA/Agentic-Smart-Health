"""El sentido oclusal de una arcada escaneada, leído del borde abierto de la malla (§6.19).

**Por qué existe.** `fusion_agents.marco.marco_canonico` decide hacia dónde está el oclusal
comparando lo bien que una parábola se ajusta a cada extremo del eje axial. Cuando las dos
orientaciones se ajustan igual —razón de dispersión cerca de 1— elige mal a veces: en 10 de
840 arcadas de Teeth3DS+ el eje oclusal canónico apunta hacia la encía. C2 y D, entrenados
con rotaciones de ±15°, pierden entonces la arcada casi entera en las 18 poses de referencia,
y el voto no puede rescatarla. De ahí salen 21 de las 28 coronas que C2 daña frente a B
en `reserva-2`.
Ver `docs/research/segmentacion-fdi-escaner.md`, §6.18 y §6.19.

**La regla.** Un escaneo intraoral es un casquete abierto por abajo: el corte de la malla
está en la encía, del lado apical. Si la mediana de los vértices del borde abierto queda por
encima del centro sobre el eje oclusal, el eje está al revés y se da la vuelta. En las 840
arcadas medidas, las bien orientadas dan como mucho −0,58 desviaciones típicas y las
invertidas al menos +0,91.

Lo demás es `marco_canonico` tal cual: sin borde, o con el borde ya abajo, sale el mismo
marco. No se cambia `fusion_agents.marco`, que también usa el registro.
"""

from __future__ import annotations

import numpy as np
from fusion_agents.marco import MarcoEspecular, curva_arco, marco_arcada


def vertices_de_borde(tri: np.ndarray) -> np.ndarray:
    """Vértices que tocan una arista usada por una sola cara: el borde abierto de la malla."""
    tri = np.asarray(tri, dtype=np.int64)
    aristas = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    unicas, cuentas = np.unique(aristas, axis=0, return_counts=True)
    return np.unique(unicas[cuentas == 1])


def marco_orientado(V: np.ndarray, tri: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """`(centro, R, razón, volteado)`: `marco_canonico` con el sentido oclusal del borde.

    `R` tiene filas `[lateral, anterior, oclusal]`, como `marco_canonico`. `volteado` dice
    si el borde contradijo al ajuste de la parábola y se dio la vuelta al eje oclusal.
    """
    V = np.asarray(V, dtype=np.float64)
    centro, ejes, P, razon = marco_arcada(V)
    borde = vertices_de_borde(tri)
    volteado = bool(len(borde) and np.median(P[borde, 2]) > 0)
    if volteado:
        # Como en `marco_arcada`: se voltean DOS ejes para que el marco no salga especular.
        ejes = ejes.copy()
        ejes[1] *= -1
        ejes[2] *= -1
        P = (V - centro) @ ejes.T
    coef, _ = curva_arco(P)
    anterior = -ejes[1] if coef[0] > 0 else ejes[1]
    oclusal = ejes[2]
    R = np.stack([np.cross(anterior, oclusal), anterior, oclusal])
    if np.linalg.det(R) < 0:
        raise MarcoEspecular(
            "el marco orientado salió a izquierdas; invertir un eje espejaría la arcada")
    return centro, R, razon, volteado
