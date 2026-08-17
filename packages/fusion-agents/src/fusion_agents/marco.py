"""Marco de coordenadas de una arcada: de dónde vienen sus ejes anatómicos.

Vivía en `scripts/altura_corona.py`, y media docena de sitios lo cargaban de allí con
`spec_from_file_location`. Una primitiva geométrica que tanta gente importa por ruta
desde un script no está en su sitio: aquí se instala, se testea y se encuentra.

**Y hay un motivo más fuerte que el orden.** Suponer cuál es el eje oclusal —darlo por
`z` porque suele serlo— ha costado dos errores medidos en este proyecto:

- El segmentador de dientes por FDI etiquetaba una mandíbula con códigos de maxilar. El
  modelo se entrenó sobre Teeth3DS+, cuyo eje oclusal va en **+z**, y el escáner de
  `histora` lo escribe en **+y**: el coseno entre ambos es **0,004**, o sea 90°. La
  normalización previa a la inferencia centra y escala, pero **no rota**.
- Una restricción de crecimiento radicular «solo hacia apical» se ató a `z` en un
  escaneo cuyo apical era `−y`. No frenó nada; en varios dientes lo empeoró.

Los dos fallos son el mismo: asumir una convención en vez de medirla, teniendo la
medida a mano. Con estas funciones en el paquete, el eje sale siempre de los datos.
"""

from __future__ import annotations

import numpy as np

GRADO_ARCO = 4  # el arco dental es casi una parábola; 4 absorbe la asimetría
DECIL_CRESTA = 80  # percentil axial que se considera "cresta oclusal"


def marco_arcada(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Lleva la arcada a `[largo, ancho, axial]` con el eje axial hacia oclusal.

    Devuelve `(centro, ejes, P, razón)`. Las filas de `ejes` son los tres ejes
    anatómicos expresados en el marco del fichero, y `P = (V − centro) @ ejes.T`.

    La **razón de dispersión** compara las dos orientaciones posibles del eje axial:
    cuanto más baja, más claro fue el criterio. Cerca de 1 la orientación es dudosa y
    conviene desconfiar de todo lo que venga después — no es un adorno, está medido que
    a 0,61 una canonización basada en esto se hunde (acierto 0,808 → 0,197) mientras
    que a 0,40-0,43 se conserva intacta.
    """
    centro = V.mean(0)
    _, _, ejes = np.linalg.svd(V - centro, full_matrices=False)
    P = (V - centro) @ ejes.T

    dispersion = {}
    for signo in (+1, -1):
        cresta = (P[:, 2] * signo) > np.percentile(P[:, 2] * signo, DECIL_CRESTA)
        ajuste = np.polyfit(P[cresta, 0], P[cresta, 1], GRADO_ARCO)
        dispersion[signo] = float(np.std(P[cresta, 1] - np.polyval(ajuste, P[cresta, 0])))
    if dispersion[-1] < dispersion[+1]:
        # Se voltean DOS ejes, no uno: invertir solo el axial dejaría un marco
        # especular y la normal "hacia vestibular" apuntaría hacia dentro.
        ejes[1] *= -1
        ejes[2] *= -1
        P = (V - centro) @ ejes.T
    razon = min(dispersion.values()) / max(dispersion.values())
    return centro, ejes, P, razon


def curva_arco(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polinomio de la curva del arco, ajustado a la cresta y no a toda la malla.

    Ajustarlo a todo lo desviaría el paladar, que es la mitad de un escaneo maxilar.
    """
    cresta = P[:, 2] > np.percentile(P[:, 2], DECIL_CRESTA)
    return np.polyfit(P[cresta, 0], P[cresta, 1], GRADO_ARCO), P[cresta, :2].mean(0)


class MarcoEspecular(RuntimeError):
    """El marco salió a izquierdas. Aplicarlo espejaría la arcada."""


def marco_canonico(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Marco SIN ambigüedad: `(centro, R, razón)`, filas de `R` = `[lateral, anterior, oclusal]`.

    `marco_arcada` fija el eje oclusal, pero deja libres los signos de los otros dos —
    la PCA los devuelve arbitrarios. Para comparar dos arcadas hay que fijar los tres:

    - **oclusal**: el de `marco_arcada`, que ya resuelve su sentido midiendo.
    - **anterior**: el arco es una parábola `y = a·x² + …` en el plano oclusal, y sus
      brazos van hacia posterior; el signo de `a` dice dónde está el frente.
    - **lateral**: **no se elige, se deriva** (`x = y × z`).

    Elegir el lateral por separado es lo que espejaría la arcada y cambiaría los 3x por
    los 4x en numeración FDI. Es un error que no avisa: una malla espejada sigue
    pareciendo una dentadura. Por eso, si el marco sale a izquierdas, esto **lanza** en
    vez de arreglarlo invirtiendo un eje.
    """
    centro, ejes, P, razon = marco_arcada(V)
    coef, _ = curva_arco(P)
    anterior = -ejes[1] if coef[0] > 0 else ejes[1]
    oclusal = ejes[2]
    lateral = np.cross(anterior, oclusal)
    R = np.stack([lateral, anterior, oclusal])
    if np.linalg.det(R) < 0:
        raise MarcoEspecular(
            "el marco canónico salió a izquierdas; invertir un eje espejaría la arcada"
        )
    return centro, R, razon


def a_marco_de(
    V: np.ndarray,
    origen: tuple[np.ndarray, np.ndarray],
    destino: tuple[np.ndarray, np.ndarray],
    *,
    normales: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Lleva puntos del marco canónico `origen` a las coordenadas de FICHERO de `destino`.

    `origen` y `destino` son `(centro, R)` de `marco_canonico`. Sirve para poner una
    arcada en la pose que espera un modelo entrenado sobre otro dataset:

        V2 = ((V − c_o) @ R_o.T) @ R_d + c_d

    Las **normales se rotan pero no se trasladan**; se devuelven junto a los puntos si
    se pasan. Olvidarlas es un fallo silencioso cuando son la entrada del modelo.
    """
    c_o, R_o = origen
    c_d, R_d = destino
    M = R_o.T @ R_d
    V2 = (V - c_o) @ M + c_d
    if normales is None:
        return V2
    return V2, normales @ M
