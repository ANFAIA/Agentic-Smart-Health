"""Los ejes anatomicos de una arcada, MEDIDOS sobre la nube y sus etiquetas FDI.

Vive aqui y no en `uos` porque lo usan dos consumidores: el paquete del visor, que
necesita saber hacia donde esta lo oclusal para que la orbita de la camara gire por donde
un clinico espera, y las vistas del `.uos`, que ademas les ponen nombre. La direccion de
la dependencia ya estaba fijada —`uos` importa de `export_agents`, no al reves—, asi que
la geometria comun se queda del lado de abajo.

**Se mide, no se supone.** Es la misma regla que rige el eje apico-coronal del CBCT, que
se lee del `ImagePositionPatient` del DICOM. Una malla de escaner no trae cabecera que lo
diga, asi que aqui el nombre de cada direccion sale de las etiquetas FDI, que son un
vocabulario cerrado (ISO-3950) y dicen donde esta cada pieza en la boca:

- **oclusal** — de los vertices de encia hacia los de corona. Las coronas estan del lado
  oclusal por definicion de corona.
- **superior** — hacia la coronilla. ⚠️ **NO es lo mismo que oclusal, y confundirlos pone la
  cabeza boca abajo.** En un maxilar las coronas cuelgan hacia abajo, asi que su eje oclusal
  apunta a INFERIOR; en una mandibula apuntan hacia arriba y coinciden. Cual de las dos es
  lo dice el CUADRANTE de las piezas etiquetadas —1 y 2 son maxilar, 3 y 4 mandibula—, que
  es otra vez la etiqueta y no una suposicion.
- **derecha** — del centroide de los cuadrantes 2 y 3 (izquierda del paciente) hacia el de
  los cuadrantes 1 y 4 (derecha).
- **anterior** — del centroide de los molares (piezas 6-8) hacia el de los incisivos y el
  canino (1-3).

⚠️ **Sin etiquetas no hay ejes anatomicos y se dice.** La alternativa —tomar los ejes
principales de la nube y bautizarlos— produce nombres plausibles y a veces invertidos: una
vista que se llama «vestibular derecha» y ensena la izquierda es peor que no tenerla,
porque quien la abre no tiene forma de notarlo. Es el mismo motivo por el que el
`render-export-agent` nombra las suyas por angulo.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# Cuadrantes FDI. Permanentes y temporales comparten lado: 1/5 y 4/8 son la derecha del
# paciente, 2/6 y 3/7 la izquierda.
_DERECHA = {1, 4, 5, 8}
_IZQUIERDA = {2, 3, 6, 7}

# Piezas 1-3 (incisivos y canino) frente a 6-8 (molares). Se dejan fuera los premolares:
# estan en la curva y su centroide no separa el eje antero-posterior.
_ANTERIOR = {1, 2, 3}
_POSTERIOR = {6, 7, 8}

# Por debajo de esto dos direcciones medidas son casi la misma y la base degenera.
_MINIMO_COSENO = 0.15


# Cuadrantes del maxilar y de la mandibula (permanentes y temporales).
_MAXILAR = {1, 2, 5, 6}
_MANDIBULA = {3, 4, 7, 8}


class Base(NamedTuple):
    """Marco anatomico medido, en el marco de la propia nube y en milimetros.

    `oclusal` y `superior` son distintos y los dos hacen falta: el primero es hacia donde
    muerde la pieza y el segundo hacia la coronilla. En un maxilar son OPUESTOS.
    """

    centro: np.ndarray
    oclusal: np.ndarray
    derecha: np.ndarray
    anterior: np.ndarray
    superior: np.ndarray
    arcada: str


def normaliza(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def ortogonaliza(v: np.ndarray, *previas: np.ndarray) -> np.ndarray:
    """Gram-Schmidt. Devuelve el vector nulo si `v` ya estaba en el espacio de `previas`."""
    w = np.asarray(v, dtype=np.float64).copy()
    for p in previas:
        w -= float(w @ p) * p
    return normaliza(w) if np.linalg.norm(w) > _MINIMO_COSENO else np.zeros(3)


def marco_anatomico(
    posiciones: np.ndarray, etiquetas: np.ndarray
) -> tuple[Base | None, str]:
    """Los tres ejes con nombre, medidos. `(None, motivo)` si no se pueden medir.

    El eje oclusal sale del eje MENOR de la nube —una arcada es un herradura aplanada, y
    su direccion de menor varianza es la apico-oclusal— y su SIGNO de que las coronas
    caigan de un lado de la encia. El eje y el signo son dos medidas distintas: el primero
    lo da la geometria, el segundo las etiquetas.
    """
    pos = np.asarray(posiciones, dtype=np.float64)
    etq = np.asarray(etiquetas, dtype=np.int64)
    if len(pos) < 3 or len(pos) != len(etq):
        return None, "la malla y sus etiquetas no tienen el mismo numero de vertices"

    coronas, encia = etq > 0, etq == 0
    if not coronas.any():
        return None, (
            "la malla del escaner no trae ni una pieza etiquetada, asi que no hay con que "
            "medir hacia donde queda lo oclusal ni cual es el lado derecho del paciente"
        )
    if not encia.any():
        return None, (
            "todos los vertices estan etiquetados como pieza: sin encia no hay contra que "
            "medir el signo del eje oclusal"
        )

    centro = pos.mean(axis=0)
    # Eje menor: la ultima fila de V en la SVD de la nube centrada.
    menor = np.linalg.svd(pos - centro, full_matrices=False)[2][-1]
    oclusal = normaliza(menor)
    if float((pos[coronas].mean(axis=0) - pos[encia].mean(axis=0)) @ oclusal) < 0:
        oclusal = -oclusal

    centroides = {
        int(c): pos[etq == c].mean(axis=0) for c in np.unique(etq[coronas])
    }
    dcha, izda = _cuadrantes(centroides, _DERECHA), _cuadrantes(centroides, _IZQUIERDA)
    if dcha is None or izda is None:
        return None, (
            "las piezas etiquetadas no cubren los dos lados de la arcada, asi que no hay "
            "con que medir cual es la derecha del paciente"
        )
    derecha = ortogonaliza(dcha - izda, oclusal)
    if not derecha.any():
        return None, "los dos lados de la arcada caen sobre el eje oclusal"

    delante, detras = _piezas(centroides, _ANTERIOR), _piezas(centroides, _POSTERIOR)
    if delante is None or detras is None:
        return None, (
            "las piezas etiquetadas no cubren a la vez el sector anterior y el posterior, "
            "asi que no hay con que medir hacia donde queda lo anterior"
        )
    anterior = ortogonaliza(delante - detras, oclusal, derecha)
    if not anterior.any():
        return None, "el sector anterior y el posterior no separan el eje antero-posterior"

    cuadrantes = {c // 10 for c in centroides}
    if cuadrantes <= _MAXILAR:
        arcada, superior = "upper", -oclusal
    elif cuadrantes <= _MANDIBULA:
        arcada, superior = "lower", oclusal
    else:
        # ⚠️ Con las dos arcadas etiquetadas el signo del eje oclusal ya no significa nada:
        # las coronas de arriba miran hacia abajo y las de abajo hacia arriba, asi que la
        # medida «de la encia a las coronas» se cancela. No se elige una de las dos ni se
        # promedia: se dice que no se puede medir.
        return None, (
            "las piezas etiquetadas cubren las dos arcadas, y entonces el eje oclusal no "
            "tiene un signo: las coronas del maxilar miran al lado contrario que las de la "
            "mandibula. Hace falta separar las arcadas antes de medir"
        )
    return Base(centro, oclusal, derecha, anterior, superior, arcada), ""


def _cuadrantes(
    centroides: dict[int, np.ndarray], cuadrantes: set[int]
) -> np.ndarray | None:
    """Centroide de los cuadrantes indicados, o `None` si no hay NINGUNO.

    ⚠️ `None` y no el vector nulo. Devolver ceros haria que «no hay lado izquierdo» y «el
    lado izquierdo esta en el origen» se restaran igual, y de una arcada con solo el
    cuadrante 1 saldria una direccion «derecha» perfectamente plausible que en realidad
    apunta al centroide de esas piezas.
    """
    v = [c for f, c in centroides.items() if f // 10 in cuadrantes]
    return np.mean(v, axis=0) if v else None


def _piezas(centroides: dict[int, np.ndarray], numeros: set[int]) -> np.ndarray | None:
    v = [c for f, c in centroides.items() if f % 10 in numeros]
    return np.mean(v, axis=0) if v else None


def distancia_para_encuadrar(
    pos: np.ndarray, centro: np.ndarray, direccion: np.ndarray, *,
    fov_grados: float, margen: float = 1.0,
) -> float:
    """Cuanto hay que alejarse para que todo quepa en el campo, en mm.

    Se mide el radio de la nube PROYECTADA sobre el plano de la camara, no el de la caja:
    lo que decide el encuadre es lo ancho que se ve, no lo profundo que es.
    """
    rel = np.asarray(pos, dtype=np.float64) - centro
    radio = float(np.linalg.norm(rel - np.outer(rel @ direccion, direccion), axis=1).max())
    return max(radio, 1.0) / math.tan(math.radians(fov_grados / 2)) * margen
