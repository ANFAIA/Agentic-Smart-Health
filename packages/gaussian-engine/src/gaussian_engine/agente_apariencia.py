"""Emite la capa de apariencia real desde fotos intraorales.

El `agente.py` del mismo paquete ajusta la densidad del CBCT — mismos números, distinto
nombre —. Este adapter hace algo distinto: optimiza contra **renders de Blender** para
obtener un **degradado de dos tonos tomados de las fotos** — no color medido. Los dos
usan la misma representación (gaussianas
anisótropas), pero la función de pérdida, los datos de entrada y la semántica del
resultado son distintos.

El adapter es deliberadamente delgado: toda la lógica vive en `apariencia.py`. Esto
permite testear el entrenamiento sin el orquestador y el orquestador sin CUDA.
"""

from __future__ import annotations

from collections.abc import Iterable

from core_schemas import ColumnaCampo

from gaussian_engine.apariencia import PERFIL, PROPIEDADES_INRIA

PERFIL_APARIENCIA = PERFIL

# Qué es cada propiedad del PLY INRIA, **en máquina**. Es una tabla y no una lista porque
# el fichero no siempre lleva las mismas columnas: `region_id` solo viaja si alguien
# segmentó, igual que en el campo de densidad (ver `export_agents.field.ESQUEMA_COLUMNAS`).
#
# ⚠️ `medido=False` en TODAS. El optimizador movió las posiciones, dividió gaussianas y
# podó: no hay correspondencia 1:1 con los vértices del escaneo ni con los vóxeles del
# CBCT. Lo que midió el instrumento fue alterado por la optimización.
ESQUEMA_INRIA: dict[str, dict] = {
    **{n: {
        "unidad": "mm",
        "significado": "centro de la gaussiana (movido por optimizador)",
        "medido": False,
    } for n in ("x", "y", "z")},
    # Las normales existen porque el PLY de facto de INRIA las reserva, y van a cero
    # porque el optimizador no las usa. Se declaran igualmente: quien monte el struct
    # desde estas columnas necesita los doce bytes o lee todo lo demás desplazado.
    **{n: {
        "unidad": "",
        "significado": (
            "normal reservada por el perfil INRIA, siempre 0 — existe para el stride, "
            "no para leerla"
        ),
        "medido": False,
    } for n in ("nx", "ny", "nz")},
    **{f"f_dc_{i}": {
        "unidad": "",
        "significado": (
            "coeficiente DC de SH — color de la foto intraoral proyectado sobre la malla "
            "con la pose resuelta por PnP, y aprendido por el optimizador desde los "
            "renders. `medido: false` porque el optimizador movio, dividio y podo las "
            "gaussianas: no hay correspondencia 1:1 con el vertice que recibio el pixel. "
            "La cabecera del PLY dice cuantos vertices llevaban pixel medido y cuantos "
            "el degradado de respaldo"
        ),
        "medido": False,
        "derivado_de": "proyeccion de foto intraoral (pose PnP) + optimizacion 3DGS",
    } for i in range(3)},
    "opacity": {
        "unidad": "logit",
        "significado": "opacidad de visualización (logit), NO atenuación radiológica",
        "medido": False,
    },
    **{f"scale_{i}": {
        "unidad": "log(mm)",
        "significado": "escala del elipsoide (logaritmo, convención INRIA), optimizada",
        "medido": False,
    } for i in range(3)},
    **{f"rot_{i}": {
        "unidad": "",
        "significado": "cuaternion (w, x, y, z) normalizado — orientación del elipsoide",
        "medido": False,
    } for i in range(4)},
    # No sale del optimizador: sale de preguntar, por cada gaussiana, la etiqueta del
    # vértice de corona más cercano. Decir «vecino más cercano» es exacto y auditable;
    # decir «etiqueta aprendida» sería falso.
    "region_id": {
        "unidad": "",
        "significado": (
            "codigo FDI de la corona MAS CERCANA a la gaussiana; 0 = encia o sin asignar. "
            "No es una etiqueta aprendida: el optimizador no conserva correspondencia"
        ),
        "vocabulario": "ISO-3950",
        "medido": False,
        "derivado_de": "segmentation-agent (vecino mas cercano)",
    },
}


def esquema_apariencia(propiedades: Iterable[str] | None = None) -> list[ColumnaCampo]:
    """Esquema de columnas de la capa de apariencia, **derivado de lo que trae el fichero**.

    Las columnas son **distintas** a las del campo de densidad: no hay `density`, hay
    `f_dc_*` (degradado de dos tonos, NO medido) y `opacity` (visualización, no física),
    y la escala está en logaritmo (convención INRIA), no en mm lineales.

    ⚠️ **Se pasa la lista de propiedades del PLY, no se supone.** Esta función enumeraba
    catorce columnas a mano y el escritor pasó a emitir dieciocho: `region_id` viajaba en
    el fichero y **no aparecía en el sidecar**, así que un lector conforme no podía saber
    que existía. Es el mismo fallo que ya costó tres veces —`exportar()`, `store.put()`,
    `_descriptor_gs`— y la cura es la misma que en `export_agents.field.esquema_del_campo`:
    describir lo que hay delante en vez de una lista que envejece aparte del código que
    escribe los bytes.

    El orden se respeta: quien reconstruya el registro desde `columns` lo necesita.
    """
    props = PROPIEDADES_INRIA if propiedades is None else propiedades
    return [
        ColumnaCampo(nombre=p, **ESQUEMA_INRIA[p])
        for p in props
        if p in ESQUEMA_INRIA
    ]
