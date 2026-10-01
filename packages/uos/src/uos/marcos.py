"""Resolver un frame hasta el canonico: que significa recorrer el grafo de registraciones.

El grafo se recorre como NO DIRIGIDO —una registracion `A -> B` tambien coloca lo que
esta en `B` respecto de `A`— y hasta ahora el formato no decia lo unico que hace falta
para hacerlo bien: que cruzar una arista al reves es **invertir su matriz**, en que orden
se componen dos saltos, y que hacer cuando hay dos caminos que no coinciden.

⚠️ **Dos caminos que discrepan no se promedian ni se eligen en silencio.** Es exactamente
el caso que la disciplina del formato prohibe: dos cosas que no son iguales pareciendo
iguales. Se elige uno de forma determinista —el mas corto, y a igualdad el de ids menores,
como convencion reproducible— y la
discrepancia se DECLARA como aviso en vez de esconderse en la media. Elegir menos
saltos es una convencion reproducible, no una garantia de menor incertidumbre.
"""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uos.manifiesto import Manifest, Registration

#: Umbral algebraico sobre coeficientes de matrices, NO una distancia ni un TRE.
#: Mezcla entradas rotacionales y traslacionales; solo detecta poses discrepantes.
TOLERANCIA_COEFICIENTES = 1e-3
# Alias conservado para consumidores existentes; el nombre antiguo era incorrecto.
TOLERANCIA_MM = TOLERANCIA_COEFICIENTES


def _aristas(m: Manifest) -> list[tuple[str, str, Registration]]:
    return [(r.source_frame, r.target_frame, r) for r in m.registrations]


def caminos_al_canonico(m: Manifest, frame: str) -> list[list[tuple[Registration, bool]]]:
    """Todos los caminos SIMPLES de `frame` al canonico, cada arista con si va al reves.

    Sin repetir frame, que es lo que hace que la lista sea finita: un ciclo no aporta un
    camino nuevo, aporta el mismo camino con una vuelta de mas.
    """
    canonico = m.canonical_frame.id
    salida: list[list[tuple[Registration, bool]]] = []

    def anda(actual: str, visitados: set[str], acc: list[tuple[Registration, bool]]) -> None:
        if actual == canonico:
            salida.append(list(acc))
            return
        for origen, destino, reg in _aristas(m):
            for desde, hacia, invertida in ((origen, destino, False), (destino, origen, True)):
                if desde != actual or hacia in visitados:
                    continue
                acc.append((reg, invertida))
                anda(hacia, visitados | {hacia}, acc)
                acc.pop()

    anda(frame, {frame}, [])
    return salida


def _matriz(camino: list[tuple[Registration, bool]]):
    """La composicion del camino, como matriz 4x4 fila-mayor.

    `transform_4x4_row_major` lleva puntos de `source` a `target` con los puntos como
    vectores columna, o sea `p_target = T . p_source`. Encadenar `F -> A -> C` es por
    tanto `T_AC . T_FA`: el primer salto se aplica primero, asi que va a la DERECHA.
    """
    import numpy as np

    acumulada = np.eye(4, dtype=np.float64)
    for reg, invertida in camino:
        t = np.asarray(reg.transform_4x4_row_major, dtype=np.float64).reshape(4, 4)
        acumulada = (np.linalg.inv(t) if invertida else t) @ acumulada
    return acumulada


def resuelve_al_canonico(m: Manifest, frame: str) -> list[float] | None:
    """La transformada que lleva puntos de `frame` al canonico, o `None` si no conecta.

    Determinista: el camino con MENOS aristas, y a igualdad el de la secuencia de ids mas
    pequeña. Esta convencion evita depender del orden de iteracion, pero no selecciona
    el camino de menor incertidumbre: no se propagan covarianzas.
    """
    caminos = caminos_al_canonico(m, frame)
    if not caminos:
        return None
    mejor = min(caminos, key=lambda c: (len(c), [r.id for r, _ in c]))
    return [float(x) for x in _matriz(mejor).ravel()]


def discrepancia_maxima(m: Manifest, frame: str) -> float | None:
    """Maxima diferencia absoluta de coeficientes entre matrices de caminos distintos.

    No es distancia espacial, ni tiene unidades fisicas uniformes. `None` cuando hay
    un camino o ninguno: no hay nada que comparar.
    """
    import numpy as np

    caminos = caminos_al_canonico(m, frame)
    if len(caminos) < 2:
        return None
    matrices = [_matriz(c) for c in caminos]
    return max(float(np.abs(a - b).max())
               for a, b in permutations(matrices, 2))
