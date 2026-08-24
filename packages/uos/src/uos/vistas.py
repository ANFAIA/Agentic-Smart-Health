"""§7 — `views.json`: los estados de presentacion, «lo que el clinico estaba viendo».

Una vista es dato de presentacion (Layer 1): camara, capas y visita. Livianas,
versionables y el mecanismo natural de los deep-links (`caso.uos#view=view.pieza_24`) y de
la comparacion longitudinal — la misma vista aplicada a `v1` y a `v2` lado a lado.

**Los ejes anatomicos se MIDEN, no se suponen**, y se miden en un solo sitio:
`export_agents.anatomia`. Aqui solo se les pone nombre y se los envuelve en la forma que
pide el spec. El paquete del visor usa la misma medida para orientar su orbita, y tener
dos copias de esa geometria seria tener dos definiciones de donde esta lo oclusal.
"""

from __future__ import annotations

import numpy as np
from export_agents.anatomia import (
    distancia_para_encuadrar,
    marco_anatomico,
    normaliza,
    ortogonaliza,
)
from pydantic import BaseModel, ConfigDict, Field

VISTAS = "views.json"

# El del ejemplo del spec (§7). Un campo estrecho aplana la arcada y uno ancho la deforma
# en los bordes; 35 grados es el rango en el que un molar no se ve como un cilindro.
FOV_GRADOS = 35.0

# La arcada no toca el borde del encuadre. No es estetica: el visor dibuja las cotas y el
# panel de pieza pegados al modelo, y sin holgura salen cortados.
_MARGEN = 1.35

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


def _distancia(pos: np.ndarray, centro: np.ndarray, direccion: np.ndarray) -> float:
    return distancia_para_encuadrar(
        pos, centro, direccion, fov_grados=FOV_GRADOS, margen=_MARGEN
    )


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
        # Se mira DESDE donde muerde la pieza, o sea contra el eje oclusal.
        Vista(id="view.oclusal", label="Oclusal", visit=visita, layers=dict(capas),
              camera=_camara(pos, marco.centro, marco.oclusal, marco.anterior)),
        Vista(id="view.frontal", label="Frontal", visit=visita, layers=dict(capas),
              camera=_camara(pos, marco.centro, marco.anterior, marco.superior)),
        Vista(id="view.vestibular_derecha", label="Vestibular derecha", visit=visita,
              layers=dict(capas),
              camera=_camara(pos, marco.centro, marco.derecha, marco.superior)),
        Vista(id="view.vestibular_izquierda", label="Vestibular izquierda", visit=visita,
              layers=dict(capas),
              camera=_camara(pos, marco.centro, -marco.derecha, marco.superior)),
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
        fuera = ortogonaliza(objetivo - marco.centro, marco.superior)
        if not fuera.any():
            # Un incisivo puede caer casi sobre el eje: se encuadra desde lo anterior.
            fuera = marco.anterior
        vistas.append(Vista(
            id=f"view.pieza_{codigo}", label=f"Pieza {codigo}", visit=visita,
            layers=dict(capas),
            camera=_camara(pos[m], objetivo, normaliza(fuera + marco.superior * 0.35),
                           marco.superior),
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
