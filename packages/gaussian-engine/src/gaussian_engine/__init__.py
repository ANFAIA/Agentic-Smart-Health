"""Ajuste del campo gaussiano: de semillas isótropas del tamaño del vóxel a elipsoides.

**Qué llena este paquete.** `cbct-agent` siembra una gaussiana por vóxel de tejido duro y
declara —honestamente— que no inventa anisotropía que el CBCT no midió: escribe la misma
`scale` y el mismo cuaternión identidad medio millón de veces. El artefacto tiene los
huecos (`scales (N,3)`, `rotations (N,4)`, con unidad y semántica declaradas en
`export_agents.field`), pero nadie los llenaba. El campo era, literalmente, la rejilla del
CBCT reescrita con otro nombre.

**Por qué no se ajusta contra imágenes, que es como se hace 3DGS.** El 3DGS canónico
optimiza contra fotografías porque en fotogrametría *no hay nada más*: la escena solo se
conoce a través de vistas. Aquí sí la hay — el CBCT **es** el volumen—. Renderizarlo desde
N cámaras para después ajustar gaussianas que reproduzcan esos renders mete el
renderizador como techo de fidelidad y una pérdida en unidades de píxel, que no significa
nada clínicamente. Ajustando directamente en 3D la pérdida se mide **en HU**, que es la
unidad en la que está el dato.

Es la misma representación —elipsoides anisótropos, rasterizables por gsplat sin tocar
nada— con otro procedimiento de ajuste. Lo que cambia no es qué se guarda sino de dónde
sale.

⚠️ **El campo ajustado es DERIVADO y no sustituye al semilla.** Una gaussiana optimizada
tiene el tamaño que hace que el campo se reconstruya bien, no el del tejido: medir sobre
ella es medir un ajuste. El campo semilla sigue siendo fiel al vóxel y es sobre el que se
mide; éste es para verse y para caber. Los dos llevan `perfil_campo` distinto justamente
para que nadie los confunda — el mismo criterio que separa el PLY del twin del PLY del
visor.
"""

from gaussian_engine.agente import PERFIL, ajusta_campo, esquema
from gaussian_engine.ajuste import (
    ITERACIONES,
    Ajuste,
    ajusta,
    evalua,
    siembra_por_rejilla,
)

__all__ = [
    "ITERACIONES", "PERFIL", "Ajuste", "ajusta", "ajusta_campo", "esquema", "evalua",
    "siembra_por_rejilla",
]
