"""§7 — `views.json`: los estados de presentacion, «lo que el clinico estaba viendo».

Una vista es dato de presentacion (Layer 1): camara, capas y visita. Livianas,
versionables y el mecanismo natural de los deep-links (`caso.uos#view=view.pieza_24`) y de
la comparacion longitudinal — la misma vista aplicada a `v1` y a `v2` lado a lado.

**Los ejes anatomicos se MIDEN, no se suponen.** Es la misma regla que rige el eje
apico-coronal del CBCT, que se lee del `ImagePositionPatient` del DICOM y nunca se deduce
del ajuste. Una malla de escaner no trae cabecera que lo diga, asi que aqui el nombre de
cada direccion sale de las etiquetas FDI, que son un vocabulario cerrado (ISO-3950) y
dicen donde esta cada pieza en la boca:

- **oclusal** — de los vertices de encia hacia los de corona. Las coronas estan del lado
  oclusal por definicion de corona.
- **derecha** — del centroide de los cuadrantes 2 y 3 (izquierda del paciente) hacia el de
  los cuadrantes 1 y 4 (derecha).
- **anterior** — del centroide de los molares (piezas 6-8) hacia el de los incisivos y el
  canino (1-3).

⚠️ **Sin etiquetas no hay vistas anatomicas y se dice.** La alternativa —tomar los ejes
principales de la nube y bautizarlos— produce nombres plausibles y a veces invertidos: una
vista que se llama «vestibular derecha» y ensena la izquierda es peor que no tenerla,
porque quien la abre no tiene forma de notarlo.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

VISTAS = "views.json"

# El del ejemplo del spec (§7). Un campo estrecho aplana la arcada y uno ancho la deforma
# en los bordes; 35 grados es el rango en el que un molar no se ve como un cilindro.
FOV_GRADOS = 35.0

# La arcada no toca el borde del encuadre. No es estetica: el visor dibuja las cotas y el
# panel de pieza pegados al modelo, y sin holgura salen cortados.
_MARGEN = 1.35

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


class Camara(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: list[float] = Field(min_length=3, max_length=3)
    target: list[float] = Field(min_length=3, max_length=3)
    up: list[float] = Field(min_length=3, max_length=3)
    fov: float = FOV_GRADOS


class Capa(BaseModel):
    """Estado de una capa en una vista. `derived` no aparece: es UOS-Full."""

    model_config = ConfigDict(extra="forbid")
    visible: bool = True
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class Vista(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    camera: Camara
    layers: dict[str, Capa] = Field(default_factory=dict)
    visit: str

    # `mpr` y `clip_planes` del spec no se emiten: son controles del VOLUMEN, y el volumen
    # es UOS-Vol. Emitirlos vacios daria a entender que hay un plano que cortar.


class Base(NamedTuple):
    """Marco anatomico medido, en el frame canonico y en milimetros."""

    centro: np.ndarray
    oclusal: np.ndarray
    derecha: np.ndarray
    anterior: np.ndarray


def _normaliza(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _ortogonaliza(v: np.ndarray, *previas: np.ndarray) -> np.ndarray:
    """Gram-Schmidt. Devuelve el vector nulo si `v` ya estaba en el espacio de `previas`."""
    w = np.asarray(v, dtype=np.float64).copy()
    for p in previas:
        w -= float(w @ p) * p
    return _normaliza(w) if np.linalg.norm(w) > _MINIMO_COSENO else np.zeros(3)


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
    oclusal = _normaliza(menor)
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
    derecha = _ortogonaliza(dcha - izda, oclusal)
    if not derecha.any():
        return None, "los dos lados de la arcada caen sobre el eje oclusal"

    delante, detras = _piezas(centroides, _ANTERIOR), _piezas(centroides, _POSTERIOR)
    if delante is None or detras is None:
        return None, (
            "las piezas etiquetadas no cubren a la vez el sector anterior y el posterior, "
            "asi que no hay con que medir hacia donde queda lo anterior"
        )
    anterior = _ortogonaliza(delante - detras, oclusal, derecha)
    if not anterior.any():
        return None, "el sector anterior y el posterior no separan el eje antero-posterior"
    return Base(centro, oclusal, derecha, anterior), ""


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


def _distancia(pos: np.ndarray, centro: np.ndarray, direccion: np.ndarray) -> float:
    """Cuanto hay que alejarse para que todo quepa en el campo, en mm.

    Se mide el radio de la nube PROYECTADA sobre el plano de la camara, no el de la caja:
    lo que decide el encuadre es lo ancho que se ve, no lo profundo que es.
    """
    rel = pos - centro
    radio = float(np.linalg.norm(rel - np.outer(rel @ direccion, direccion), axis=1).max())
    return max(radio, 1.0) / math.tan(math.radians(FOV_GRADOS / 2)) * _MARGEN


def _camara(
    pos: np.ndarray, objetivo: np.ndarray, direccion: np.ndarray, arriba: np.ndarray
) -> Camara:
    d = _distancia(pos, objetivo, direccion)
    return Camara(
        position=[round(float(x), 3) for x in objetivo + direccion * d],
        target=[round(float(x), 3) for x in objetivo],
        up=[round(float(x), 3) for x in arriba],
    )


def construye_vistas(
    posiciones: np.ndarray,
    etiquetas: np.ndarray,
    *,
    visita: str,
    piezas: list[str] | None = None,
    con_apariencia: bool = False,
) -> tuple[list[Vista], list[str]]:
    """Las vistas del caso y los motivos por los que falte alguna.

    `piezas` son los codigos FDI que merecen su propia vista: los que llevan contenido
    clinico. Una vista por diente etiquetado seria mecanico —catorce entradas que dicen
    lo mismo— y lo que hace util un deep-link es que apunte a algo que alguien anoto.
    """
    marco, motivo = marco_anatomico(posiciones, etiquetas)
    if marco is None:
        return [], [f"el .uos no lleva vistas ({VISTAS} va vacio): {motivo}"]

    pos = np.asarray(posiciones, dtype=np.float64)
    etq = np.asarray(etiquetas, dtype=np.int64)
    capas = {"mesh": Capa()}
    if con_apariencia:
        # La apariencia va por DEBAJO de 1.0: es reconstruida contra renders, no medida, y
        # el visor tiene que poder dejar ver la malla por debajo.
        capas["gs"] = Capa(opacity=0.85)

    vistas = [
        Vista(id="view.oclusal", label="Oclusal", visit=visita, layers=dict(capas),
              camera=_camara(pos, marco.centro, marco.oclusal, marco.anterior)),
        Vista(id="view.frontal", label="Frontal", visit=visita, layers=dict(capas),
              camera=_camara(pos, marco.centro, marco.anterior, marco.oclusal)),
        Vista(id="view.vestibular_derecha", label="Vestibular derecha", visit=visita,
              layers=dict(capas),
              camera=_camara(pos, marco.centro, marco.derecha, marco.oclusal)),
        Vista(id="view.vestibular_izquierda", label="Vestibular izquierda", visit=visita,
              layers=dict(capas),
              camera=_camara(pos, marco.centro, -marco.derecha, marco.oclusal)),
    ]

    sin_etiquetar: list[str] = []
    for codigo in piezas or []:
        m = etq == int(codigo)
        if not m.any():
            sin_etiquetar.append(codigo)
            continue
        objetivo = pos[m].mean(axis=0)
        # Vestibular de ESA pieza: hacia fuera de la arcada, perpendicular a lo oclusal.
        # Sale del centro de la arcada al centro del diente, que es la definicion.
        fuera = _ortogonaliza(objetivo - marco.centro, marco.oclusal)
        if not fuera.any():
            # Un incisivo puede caer casi sobre el eje: se encuadra desde lo anterior.
            fuera = marco.anterior
        vistas.append(Vista(
            id=f"view.pieza_{codigo}", label=f"Pieza {codigo}", visit=visita,
            layers=dict(capas),
            camera=_camara(pos[m], objetivo, _normaliza(fuera + marco.oclusal * 0.35),
                           marco.oclusal),
        ))

    # ⚠️ Un aviso, no uno por pieza. Un caso con solo el maxilar escaneado y un informe
    # que habla de las dos arcadas produce dieciseis lineas que dicen lo mismo, y el gate
    # ya lleva una que lo explica entera. Repetirlo por diente entierra los motivos que
    # solo aparecen una vez, que son justo los que hay que leer.
    avisos = [] if not sin_etiquetar else [
        f"el informe habla de {len(sin_etiquetar)} pieza(s) que el escaner no trae "
        f"etiquetadas, asi que {VISTAS} no lleva vista que apunte a ellas: "
        + ", ".join(f"FDI {c}" for c in sin_etiquetar)
    ]
    return vistas, avisos
