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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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

# Longitud minima de la tangente al arco entre dos piezas vecinas. Por debajo, la
# direccion mesiodistal no esta definida y el ancho no se declara.
_EPS_DIRECCION = 1e-9


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


def anchos_de_corona(
    posiciones: np.ndarray, caras: np.ndarray, etiquetas: np.ndarray
) -> dict[int, tuple[float, float]]:
    """Por pieza etiquetada, `(ancho medido, ancho de tabla)` en mm mesiodistales.

    Es la medida directa de **«esta etiqueta se ha pasado del punto de contacto»**, que es
    lo que se ve en el visor cuando una pieza se enciende y arrastra un trozo de su vecina.
    Ningun numero de los que ya se declaran la captura: el acierto por diente pregunta si
    el codigo mayoritario es el correcto, y una frontera corrida dos milimetros no cambia
    una mayoria. Se puede tener 0,93 de acierto por diente y coronas de 18 mm.

    Dos detalles del calculo que NO son opcionales, porque sin ellos el numero no
    significa nada:

    - **Solo la componente conexa MAYOR de cada etiqueta.** Medido sobre un maxilar real,
      cada codigo trae ademas decenas de islas sueltas repartidas por la arcada; tomando
      todos sus vertices, el 27 «mide» 41 mm. No es que la corona sea ancha: es que se
      esta midiendo hasta la mota mas lejana.
    - **Rango del 1 al 99 %**, no el maximo. Un solo vertice del borde de la componente
      inflaria el ancho sin que la frontera se haya movido.

    La direccion mesiodistal sale de la tangente al arco en cada pieza —la recta entre los
    centroides de sus dos vecinas—, que es la que corre a lo largo de la arcada. Con un eje
    global el numero mezclaria ancho y profundidad en los molares, que estan girados.

    Devuelve `{}` si no hay dos piezas con las que orientar la tangente.
    """
    from analysis_agents.dental import ancho_admitido

    pos = np.asarray(posiciones, dtype=np.float64)
    etq = np.asarray(etiquetas, dtype=np.int64)
    if len(pos) != len(etq) or not (etq > 0).any():
        return {}

    aristas = np.unique(
        np.sort(np.asarray(caras)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0
    )
    # ⚠️ **Una esquirla no es una pieza, y contarla falsea las dos cifras.** El umbral es
    # RELATIVO a la propia arcada y no un numero absoluto — el mismo criterio que usa
    # `gaussian_engine.pose_foto`. En el caso real el FDI 28 traia 275 vertices frente a
    # los 1.200-9.500 de las piezas de verdad; el exportador lo absorbe en el 27 y declara
    # que la arcada tiene 14 dientes, pero esta medida lo contaba aparte. Salia «9 de 15»
    # cuando son **9 de 14**, y ademas el 28 figuraba como corona ESTRECHA (4,2 mm contra
    # 8,5 de tabla), que de una esquirla no significa nada.
    cuenta = {int(f): int((etq == f).sum()) for f in np.unique(etq) if f > 0}
    if not cuenta:
        return {}
    # Puramente relativo, sin suelo absoluto: el numero de vertices depende del escaner y
    # de la densidad del mallado, asi que un «minimo 200» seria el de un escaner concreto.
    # Lo que distingue una esquirla de una pieza es su tamano frente a las demas piezas.
    suelo = 0.15 * float(np.median(list(cuenta.values())))

    nucleos: dict[int, np.ndarray] = {}
    for fdi, n in cuenta.items():
        if n < suelo:
            continue
        suyos = etq == fdi
        idx = np.flatnonzero(suyos)
        if idx.size < 3:
            continue
        remap = np.full(len(pos), -1, dtype=np.int64)
        remap[idx] = np.arange(idx.size)
        propias = aristas[suyos[aristas[:, 0]] & suyos[aristas[:, 1]]]
        grafo = coo_matrix(
            (np.ones(len(propias)), (remap[propias[:, 0]], remap[propias[:, 1]])),
            shape=(idx.size, idx.size),
        )
        _, comp = connected_components(grafo, directed=False)
        nucleos[fdi] = idx[comp == np.bincount(comp).argmax()]
    if len(nucleos) < 2:
        return {}

    centros = {f: pos[v].mean(axis=0) for f, v in nucleos.items()}
    # Orden a lo largo del arco: cuadrantes 1 y 4 de atras hacia delante, 2 y 3 al reves.
    # Es el mismo orden anatomico del vocabulario, no una suposicion sobre la geometria.
    def _clave(f: int) -> tuple[int, int]:
        cuadrante, posicion = divmod(f, 10)
        return (0, -posicion) if cuadrante in (1, 4) else (1, posicion)

    arco = sorted(centros, key=_clave)
    salida: dict[int, tuple[float, float]] = {}
    for i, fdi in enumerate(arco):
        cota = ancho_admitido(fdi)
        if cota is None:
            continue
        antes = centros[arco[i - 1]] if i else None
        despues = centros[arco[i + 1]] if i + 1 < len(arco) else None
        if antes is not None and despues is not None:
            ref = despues - antes
        elif despues is not None:
            ref = despues - centros[fdi]
        else:
            ref = centros[fdi] - antes
        norma = float(np.linalg.norm(ref))
        if norma < _EPS_DIRECCION:
            continue
        proy = (pos[nucleos[fdi]] - centros[fdi]) @ (ref / norma)
        ancho = float(np.quantile(proy, 0.99) - np.quantile(proy, 0.01))
        salida[fdi] = (ancho, cota)
    return salida


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
