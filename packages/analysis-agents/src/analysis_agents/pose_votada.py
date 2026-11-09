"""Combinar varias segmentaciones del mismo escaneo, hechas con poses distintas.

**Por qué existe.** El modelo FDI se entrenó con Teeth3DS+ en su pose original, así que un
escaneo de otro escáner hay que girarlo antes a esa pose, y el giro se calcula contra una
arcada de referencia. Medido: el resultado depende muchísimo de qué referencia se use. En
40 arcadas de `dev`, segmentando con una sola de 18 referencias, el número de dientes bien
numerados va de 195 a 468 de 520; en el caso clínico, de 0 a 13 de 14.

**Qué hace.** Es aumentación en tiempo de inferencia: el mismo escaneo se segmenta con
varias referencias, se eligen las `MEJORES` en que el modelo está más seguro y cada vértice
se queda con el código que más de ellas le dan. El modelo no cambia.

⚠️ **Por qué la confianza sirve para elegir.** Cuando la pose no le encaja, el modelo duda,
y eso se ve en su log-probabilidad. Medido en `dev`: dentro de cada arcada, la confianza
ordena las referencias como su acierto en 39 de 40 arcadas. Es una señal, no una garantía:
de vez en cuando una pose mala sale confiada por casualidad, y por eso se vota entre
varias en lugar de quedarse con la primera.

Todo es numpy: la inferencia la hace quien llama, con torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Cuántas segmentaciones, las más confiadas, votan. Elegido en `dev`: con una sola gana a
# veces una pose mala; con cinco ya entran poses mediocres.
MEJORES = 3


def confianza(logp: np.ndarray, etiquetas: np.ndarray) -> float:
    """Log-probabilidad máxima media sobre los vértices que la segmentación llama diente.

    Sobre los dientes y no sobre toda la malla: la encía es fácil en cualquier pose y
    diluiría la diferencia entre una pose buena y una mala.
    """
    dientes = np.asarray(etiquetas) > 0
    if not dientes.any():
        return float("-inf")
    return float(np.asarray(logp)[dientes].max(axis=1).mean())


def vota(segmentaciones: dict[str, np.ndarray], confianzas: dict[str, float],
         mejores: int = MEJORES) -> tuple[np.ndarray, dict[str, Any]]:
    """Código por vértice votado entre las `mejores` segmentaciones más confiadas.

    Un empate se resuelve a favor de la más confiada de las empatadas, de modo que con una
    sola segmentación, o con todas en desacuerdo, el resultado es la más confiada.
    """
    if not segmentaciones:
        raise ValueError("hace falta al menos una segmentación")
    orden = sorted(segmentaciones, key=lambda k: -confianzas[k])
    elegidas = orden[:mejores]
    pila = np.stack([np.asarray(segmentaciones[k], dtype=np.int64) for k in elegidas])
    codigos = np.unique(pila)
    votos = np.stack([(pila == c).sum(axis=0) for c in codigos])
    # Desempate: la primera elegida es la más confiada; su código suma medio voto.
    votos = votos + 0.5 * (pila[0][None, :] == codigos[:, None])
    resultado = codigos[np.argmax(votos, axis=0)]
    registro = {
        "chosen": elegidas,
        "confidence": {k: round(float(confianzas[k]), 4) for k in orden},
        "agreement_with_best": float((resultado == pila[0]).mean()),
    }
    return resultado.astype(np.int64), registro
