"""Emite la capa de apariencia real desde fotos intraorales.

El `agente.py` del mismo paquete ajusta la densidad del CBCT — mismos números, distinto
nombre —. Este adapter hace algo distinto: optimiza contra **renders de Blender** para
obtener el **color real del paciente**. Los dos usan la misma representación (gaussianas
anisótropas), pero la función de pérdida, los datos de entrada y la semántica del
resultado son distintos.

El adapter es deliberadamente delgado: toda la lógica vive en `apariencia.py`. Esto
permite testear el entrenamiento sin el orquestador y el orquestador sin CUDA.
"""

from __future__ import annotations

from core_schemas import ColumnaCampo

from gaussian_engine.apariencia import PERFIL

PERFIL_APARIENCIA = PERFIL


def esquema_apariencia() -> list[ColumnaCampo]:
    """Esquema de columnas de la capa de apariencia (perfil INRIA grado 0).

    Las columnas son **distintas** a las del campo de densidad: no hay `density`,
    hay `f_dc_*` (color real) y `opacity` (visualización, no física). La escala
    está en logaritmo (convención INRIA), no en mm lineales.

    ⚠️ **`medido=False` en todas las columnas.** El optimizador movió las posiciones,
    dividió gaussianas y podó: no hay correspondencia 1:1 con los vértices del
    escaneo ni con los vóxeles del CBCT. Lo que mide el instrumento (posición de
    vértice) fue alterado por la optimización.
    """
    return [
        *(ColumnaCampo(
            nombre=n, unidad="mm",
            significado="centro de la gaussiana (movido por optimizador)",
            medido=False,
        ) for n in ("x", "y", "z")),
        *(ColumnaCampo(
            nombre=f"scale_{i}", unidad="log(mm)",
            significado=(
                "escala del elipsoide (logaritmo, convención INRIA), optimizada"
            ),
            medido=False,
        ) for i in range(3)),
        *(ColumnaCampo(
            nombre=f"rot_{i}", unidad="",
            significado=(
                "cuaternion (w, x, y, z) normalizado — orientación del elipsoide"
            ),
            medido=False,
        ) for i in range(4)),
        ColumnaCampo(
            nombre="opacity", unidad="logit",
            significado=(
                "opacidad de visualización (logit), NO atenuación radiológica"
            ),
            medido=False,
        ),
        *(ColumnaCampo(
            nombre=f"f_dc_{i}", unidad="",
            significado="coeficiente DC de SH — color RGB real del paciente",
            medido=False,
        ) for i in range(3)),
    ]
