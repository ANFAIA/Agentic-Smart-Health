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

from gaussian_engine.apariencia import (
    GANANCIA_OCLUSION,
    OCLUSION_MINIMA,
    PERFIL,
    PROPIEDADES_INRIA,
    RADIO_OCLUSION_MM,
    VECINOS_OCLUSION,
)

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
    # ⚠️ Las normales las reserva el perfil INRIA y **antes iban a cero**. Ahora llevan la
    # normal del vértice de malla más cercano, porque de ahí sale el relieve de `f_rest_*`
    # y un lector tiene que poder recalcularlo o contradecirlo en vez de creérselo.
    **{n: {
        "unidad": "",
        "significado": (
            "normal del vertice de malla mas cercano; 0 si no habia malla con que "
            "calcularla. Es la que genera el relieve declarado en f_rest_*"
        ),
        "medido": False,
    } for n in ("nx", "ny", "nz")},
    # ⚠️ **El grado 1 no lo aprendió nadie: lo escribe este emisor.** El campo se entrena
    # contra renders de albedo plano para que `f_dc` sea el color del paciente y no el de
    # una iluminación que nos inventamos; el precio es que un albedo puro se dibuja plano.
    # El grado 1 de los armónicos es exactamente una función lineal de la dirección de
    # vista, así que un `n·v` cabe ahí exacto y devuelve el volumen sin tocar el grado 0.
    # ⚠️ **La oclusión va aquí y NO dentro de `f_dc_*`, y ésa es toda la decisión.**
    # Un surco está oscuro se mire desde donde se mire, así que dentro de los armónicos
    # sólo cabría en el grado 0, que es el color. Meterla ahí contaminaría el tono que
    # `clinical/observations.json` declara al lado — y una lectura de color no debe
    # oscurecerse porque la pieza tenga una fisura pegada.
    "ao": {
        "unidad": "",
        "significado": (
            # ⚠️ **Los numeros salen de las constantes, no del teclado.** Este texto ya
            # se quedo describiendo el metodo ANTERIOR —contar cuantos vecinos caen por
            # delante del plano tangente— despues de que se midiera que ese metodo daba
            # 0,5 en toda la superficie y se sustituyera. Un descriptor que se escribe a
            # mano se desincroniza del codigo que describe; uno que se compone de las
            # mismas constantes que el calculo, no puede.
            "oclusion ambiental CALCULADA por el emisor: media de |sen| del angulo entre "
            f"la normal y la direccion a cada uno de los {VECINOS_OCLUSION} vecinos a "
            f"menos de {RADIO_OCLUSION_MM:g} mm, amplificada por {GANANCIA_OCLUSION:g} y "
            f"acotada por abajo en {OCLUSION_MINIMA:g}. Se calcula SOBRE LA MALLA y "
            "se transfiere a cada gaussiana por vecino mas cercano, porque el optimizador "
            "mueve los centros fuera de la superficie. Factor de VISUALIZACION en [0,1] "
            "que quien dibuja multiplica por el color. No es la oclusion de un trazador de "
            "rayos y no se declara como tal: coincide en que las hendiduras se oscurecen y "
            "las cuspides no. 1 = sin malla con que calcularla, o sea no oscurecer"
        ),
        "medido": False,
    },
    **{f"f_rest_{i}": {
        "unidad": "",
        "significado": (
            "coeficiente de SH grado 1 CALCULADO por el emisor, no entrenado: vale "
            "0.35*albedo*(n . v) con la normal de nx,ny,nz. Es un realce de forma para "
            "que la pieza se lea con volumen; NO es medida y NO toca el color, que vive "
            "entero en el grado 0. Leyendo solo f_dc_* se recupera el albedo sin nada "
            "horneado. Van a cero si no habia malla con que calcular las normales"
        ),
        "medido": False,
    } for i in range(9)},
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
