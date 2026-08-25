"""Contrato entre etapas: qué necesita cada una y qué se compromete a conservar.

**Por qué existe.** El pipeline tiene seis fases y cada agente tiene su contrato de
entrada y salida, pero **nadie comprueba lo que pasa entre dos etapas**. En un solo día
eso costó dos fallos del mismo tipo, y ninguno lo vio ningún test:

- el `cbct-agent` restaba el centroide del campo gaussiano y **lo tiraba**, así que
  ningún fichero exportado podía volver a las coordenadas del CBCT;
- el `segmentation-agent` escribía `region_id` en el artefacto y el `field-export-agent`
  producía un PLY correcto **sin esa columna**, perdiendo lo único que el pipeline sabe
  de anatomía.

Los dos pasaban todos sus tests. Cada etapa cumplía su contrato; lo que faltaba era el
contrato *entre* etapas. Ningún framework de orquestación arregla eso —habría ejecutado
las mismas funciones en el mismo orden y perdido los mismos datos—: hace falta que cada
etapa **declare** qué necesita encontrar y qué se compromete a no perder.

**Dos obligaciones, y son distintas.**

`requiere` mira **antes**: si la entrada no trae lo necesario, la etapa no corre y lo
dice. Es lo que convierte «resultado silenciosamente desplazado» en «falta `origin`,
reingiere la serie».

`conserva` mira **después**: lo que entró tiene que seguir estando al salir. Es lo que
habría cazado el `region_id`, porque ahí la entrada era correcta y el fallo estaba en que
la salida perdía algo por el camino.

**Lo que NO es esto.** No es un grafo, ni un motor de ejecución, ni enrutado. Las etapas
siguen siendo llamadas a función en el orden que decide el orquestador. Esto solo declara
y verifica el borde entre ellas — que es donde estaban los fallos.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContratoEtapa:
    """Lo que una etapa necesita de su entrada y lo que promete de su salida."""

    nombre: str

    requiere_arrays: frozenset[str] = field(default_factory=frozenset)
    """Claves que el artefacto del campo gaussiano DEBE traer para que la etapa corra."""

    requiere_refs: frozenset[str] = field(default_factory=frozenset)
    """Campos del `TwinSnapshot` que no pueden ser `None` (`surface_ref`, …)."""

    requiere_transform: bool = False
    """Si la etapa necesita la rígida que solo escribe la fusión geométrica."""

    conserva_arrays: bool = True
    """Si la etapa emite un artefacto nuevo, no puede perder claves del anterior.

    `False` solo para etapas que **cambian de representación** a propósito — un render no
    conserva `scales` porque su salida es una imagen, y exigírselo sería un contrato
    equivocado, no una comprobación útil.
    """


def revisa_requisitos(contrato: ContratoEtapa, snapshot, arrays: dict) -> list[str]:
    """Qué le falta a la entrada para que esta etapa pueda correr. Vacío = puede.

    Devuelve motivos legibles, no lanza: el idioma del repositorio es declarar y dejar que
    quien llama decida, igual que hacen los agentes.
    """
    faltan: list[str] = []

    ausentes = sorted(contrato.requiere_arrays - set(arrays))
    if ausentes:
        faltan.append(
            f"`{contrato.nombre}` necesita {', '.join(f'`{k}`' for k in ausentes)} en el "
            "campo gaussiano y el artefacto no los trae: hay que reingerir la serie con "
            "una versión del agente que los guarde."
        )

    for ref in sorted(contrato.requiere_refs):
        if getattr(snapshot, ref, None) is None:
            faltan.append(f"`{contrato.nombre}` necesita `{ref}`, que el snapshot no tiene.")

    if contrato.requiere_transform and getattr(snapshot.provenance, "transform", None) is None:
        faltan.append(
            f"`{contrato.nombre}` necesita la transformación rígida de la fusión "
            "geométrica, y este snapshot no ha pasado por ella."
        )
    return faltan


def revisa_conservacion(contrato: ContratoEtapa, antes: dict, despues: dict) -> list[str]:
    """Qué perdió la etapa por el camino. Vacío = no perdió nada.

    Es la mitad que faltaba. `region_id` se perdía aquí: la entrada era correcta, la etapa
    no falló, y la salida simplemente no llevaba una de las claves con las que entró.
    """
    if not contrato.conserva_arrays:
        return []
    perdidas = sorted(set(antes) - set(despues))
    if not perdidas:
        return []
    return [
        f"`{contrato.nombre}` perdió {', '.join(f'`{k}`' for k in perdidas)} entre su "
        "entrada y su salida. Lo que entra al campo gaussiano tiene que salir: quien "
        "consuma el artefacto después no puede recuperarlo."
    ]
