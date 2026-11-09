"""Registrar el escáner contra un CBCT con pose inicial anatómica y solo por encima de la mordida.

**Por qué existe.** El camino de `preparacion` registra con un ICP que parte de la
identidad y elige el objetivo por un árbitro de esmalte. En una serie densa funciona
(`histora`, mediana corona→esmalte 0,81 mm). En una serie **dispersa** —cortes sueltos con
huecos de hasta 6 mm, como la del segundo caso clínico— no: el esmalte del CBCT queda en
unos pocos planos, el ICP local y el barrido de `icp_global` caen en poses que a la vista
están mal (la arcada girada por detrás de los molares), y el 3,6 % de solapamiento que
informa no puede juzgarlo, porque con cortes sueltos casi ningún vértice del escáner
tiene esmalte a menos de 1 mm aunque la pose sea perfecta.

**Las tres decisiones, medidas sobre ese caso** (`docs/research/registro-por-cortes.md`):

1 · **La orientación inicial se mide en las dos modalidades, no se busca.** En el escáner,
con los números FDI (`export_agents.anatomia.anatomical_frame`, que pasa el orquestador);
en el CBCT, con los ejes del paciente que fija el DICOM (LPS: derecha −x, anterior −y,
superior +z). Las dos bases tienen la misma quiralidad, así que no hay espejo posible; si
no la tuvieran, se dice y no se registra.

2 · **Solo cuentan los cortes por encima de la mordida.** Un CBCT dental se toma en
oclusión: por debajo del plano oclusal el esmalte es de la otra arcada, y en la zona
cervical aparecen raíz y hueso. Con el ICP inverso sobre z ∈ [29, 38] mm, el contorno de
hueso por encima de las coronas tiraba de ellas hacia la raíz: un barrido independiente
diente a diente sobre el DICOM pedía bajar el escáner 1,0 mm (mediana de 14 dientes,
los 14 en el mismo sentido). Con la banda de `MARGEN_MORDIDA_MM` a `ALTO_BANDA_MM` sobre
el plano, ese mismo barrido pide −0,25 mm.

3 · **El ajuste va del CBCT al escáner, no al revés.** Con cortes sueltos solo los
contornos del CBCT tienen contrapartida segura —cada punto de contorno está sobre la
superficie del diente—; los vértices del escáner que caen entre dos cortes no tienen con
qué emparejarse. Por eso el solapamiento y el residuo se miden también en ese sentido.

⚠️ **Lo que se probó y no se queda.** Una medida simétrica con búsqueda vertical explícita
—escáner→contorno además de contorno→escáner— no mejoró el barrido independiente sobre el
campo (−0,50 mm frente a −0,25) y con el umbral de esmalte se pasaba 1,9 mm hacia el otro
lado. Lo que corrige el desfase es la banda, no la medida.

⚠️ **Solo para series dispersas, y está medido por qué.** Sobre `histora`, con la serie
completa, este método registra PEOR que el de siempre: mediana corona→esmalte 2,24 mm
frente a 1,41 en el maxilar y 0,68 frente a 0,57 en la mandíbula. Con un volumen denso el
esmalte entero es buen objetivo y descartar todo menos una banda de 4,5 mm tira
información. Así que `prepara_por_cortes` se niega con `SerieDensa` y el orquestador sigue
por el camino de siempre. En el caso disperso la misma medida pasa de 4,92 a 1,59 mm.

⚠️ **La altura es lo peor medido, también sin ruido.** Sobre coronas sintéticas en una
rejilla de 0,3 mm el registro acierta en planta por debajo de 0,01 mm y deja las coronas
0,37 mm hacia la raíz (0,2 con rejilla de 0,15): con cortes horizontales, la altura solo
se ve en cuánto se estrecha la sección hacia la cúspide, y el borde discreto del campo
la sesga. Es el mismo sentido que el desfase residual del caso real, −0,25 a −0,6 mm según
las etiquetas del escáner. No se corrige restando: sería ajustar a un sintético.

⚠️ **La incertidumbre no es el residuo.** Se registra con tres umbrales de contorno y se
declara cuánto se mueven las coronas entre ellos: sobre el caso disperso, 1,8 mm de
mediana y 2,5 de máximo. Ese número va al gate; el residuo del ICP no lo ve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from fusion_agents.preparacion import HU_ARBITRO, HU_ESMALTE, plano_oclusal_del_esmalte
from fusion_agents.registration import (
    DEFAULT_OVERLAP_MM,
    RegistrationResult,
    kabsch,
    matrix_to_quaternion,
)

# Ejes del paciente en el marco del DICOM (LPS), en el mismo orden que la base del escáner:
# derecha, anterior, superior. Lo fija el estándar, no el equipo: x crece hacia la
# izquierda del paciente, y hacia posterior, z hacia la cabeza.
EJES_LPS = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

# La banda de cortes que se usa, medida desde el plano oclusal hacia la raíz de la arcada
# escaneada. Por debajo de `MARGEN_MORDIDA_MM` están las cúspides de las dos arcadas
# entrelazadas; por encima de `ALTO_BANDA_MM` empiezan el cuello, la raíz y el hueso. Sobre
# el caso disperso deja z ∈ [31,0 ; 35,5] mm: las mitades oclusales de las coronas.
MARGEN_MORDIDA_MM = 1.5
ALTO_BANDA_MM = 6.0

# Umbral del contorno de diente, y los dos con que se mide la incertidumbre. 1100 y 1900
# son los extremos con los que se midió el desfase vertical sobre el DICOM.
UMBRAL_CONTORNO = 1400.0
UMBRALES_INCERTIDUMBRE = (1100.0, 1900.0)

# Un contorno solo cuenta si está a menos de esto (en planta) del esmalte de la banda: a
# la altura de las coronas el corte trae también la rama de la mandíbula y las vértebras.
RADIO_ARCADA_MM = 4.0

# Fracción de correspondencias, las más cercanas, con que se ajusta en cada iteración.
TRIM = 0.8

# Un corte cuenta como hueco si su separación con el siguiente supera esto por el paso
# mediano. Es lo que distingue una serie dispersa de una densa submuestreada.
HUECO_RELATIVO = 2.0

# Mínimo de puntos de contorno para intentar el registro.
MINIMO_CONTORNO = 500


class SerieDensa(ValueError):
    """La serie no es dispersa en la banda: el camino de siempre registra mejor."""


def pose_inicial(ejes_escaner: np.ndarray) -> np.ndarray:
    """Rotación que lleva los ejes anatómicos del escáner sobre los del CBCT.

    `ejes_escaner` tiene por filas derecha, anterior y superior del paciente en el marco
    del escáner. Si su quiralidad no es la del DICOM, una de las dos está en espejo y
    ninguna rotación lo arregla: se dice en vez de registrar algo reflejado.
    """
    b = np.asarray(ejes_escaner, dtype=np.float64)
    if np.linalg.det(b) * np.linalg.det(EJES_LPS) <= 0:
        raise ValueError(
            "los ejes anatómicos del escáner tienen la quiralidad contraria a los del "
            "DICOM: uno de los dos está en espejo y una rotación no lo corrige"
        )
    return EJES_LPS.T @ b


def banda(arcada: str, plano: float) -> tuple[float, float]:
    """`(z_min, z_max)` de los cortes que se usan: hacia la raíz de la arcada escaneada."""
    if arcada == "upper":
        return plano + MARGEN_MORDIDA_MM, plano + ALTO_BANDA_MM
    if arcada == "lower":
        return plano - ALTO_BANDA_MM, plano - MARGEN_MORDIDA_MM
    raise ValueError(f"arcada desconocida: {arcada!r} (se espera 'upper' o 'lower')")


def cortes(centros: np.ndarray, decimales: int = 3) -> np.ndarray:
    """Las alturas distintas del campo: cada una es un corte del volumen."""
    return np.unique(np.round(np.asarray(centros)[:, 2], decimales))


def describe_cortes(z: np.ndarray, z_min: float, z_max: float) -> dict:
    """Cuántos cortes caen en la banda y cuánto miden los huecos. Para declararlo."""
    z = np.sort(np.asarray(z, dtype=np.float64))
    paso = float(np.median(np.diff(z))) if len(z) > 1 else 0.0
    dentro = z[(z >= z_min) & (z <= z_max)]
    saltos = np.diff(dentro) if len(dentro) > 1 else np.array([])
    return {
        "slices_in_band": int(len(dentro)),
        "band_mm": [round(z_min, 2), round(z_max, 2)],
        "median_step_mm": round(paso, 3),
        "max_gap_mm": round(float(saltos.max()), 2) if len(saltos) else None,
        "sparse": bool(len(saltos) and paso > 0 and saltos.max() > HUECO_RELATIVO * paso),
    }


def contornos(
    centros: np.ndarray, hu: np.ndarray, umbral: float, z_min: float, z_max: float
) -> np.ndarray:
    """Puntos de borde de diente, corte a corte, entre `z_min` y `z_max`.

    Un punto por encima del umbral es borde si en su corte le falta algún vecino de la
    rejilla por encima del umbral. El campo solo guarda tejido duro, así que «le falta»
    cubre tanto un vecino blando como uno ausente. El radio sale del paso de la rejilla
    de cada corte, que se mide y no se supone.

    ⚠️ **El borde se lleva medio paso hacia fuera.** El último punto por encima del umbral
    está DENTRO del diente; el borde está entre él y el primero de fuera. Dejarlo dentro
    encoge cada sección medio paso, y con paredes casi verticales encoger una sección
    equivale a mover la corona en altura: sobre coronas sintéticas con rejilla de 0,3 mm,
    0,6 mm de error vertical. La dirección hacia fuera es la contraria a la media de los
    vecinos que sí son diente.
    """
    centros = np.asarray(centros, dtype=np.float64)
    hu = np.asarray(hu)
    z_redondo = np.round(centros[:, 2], 3)
    salida = []
    for z in np.unique(z_redondo):
        if not z_min <= z <= z_max:
            continue
        en_corte = z_redondo == z
        duro = centros[en_corte & (hu >= umbral)]
        if len(duro) < 10:
            continue
        arbol = cKDTree(duro[:, :2])
        paso = float(np.median(arbol.query(duro[:, :2], k=2)[0][:, 1]))
        # Con radio 1,5 pasos entran los 8 vecinos de la rejilla y el propio punto.
        vecinos = arbol.query_ball_point(duro[:, :2], r=1.5 * paso)
        borde = np.array([len(v) < 9 for v in vecinos])
        if not borde.any():
            continue
        fuera = np.array([duro[i, :2] - duro[vecinos[i], :2].mean(0)
                          for i in np.flatnonzero(borde)])
        norma = np.linalg.norm(fuera, axis=1, keepdims=True)
        fuera = np.divide(fuera, norma, out=np.zeros_like(fuera), where=norma > 1e-9)
        puntos = duro[borde].copy()
        puntos[:, :2] += 0.5 * paso * fuera
        salida.append(puntos)
    return np.vstack(salida) if salida else np.empty((0, 3))


@dataclass(frozen=True)
class RegistroPorCortes:
    """Registrador (`fusion_agents.registration.Registrar`) para un CBCT de cortes sueltos.

    `source` son los vértices de corona del escáner, en su marco; `target`, los
    contornos de diente del CBCT en la banda (`contornos`). Parte de `rot0` y de una
    traslación que pone las cúspides en el plano oclusal y centra la arcada en planta, y
    afina con un ICP del contorno al escáner con recorte.

    El solapamiento y el residuo que devuelve son los del **contorno**: la fracción de
    puntos de contorno con escáner a menos de `overlap_mm`. Medido desde el escáner, con
    cortes sueltos, saldría bajo aunque la pose fuera perfecta.
    """

    rot0: np.ndarray
    plano: float
    arcada: str
    trim: float = TRIM
    max_iter: int = 200
    tol_mm: float = 1e-4
    overlap_mm: float = DEFAULT_OVERLAP_MM
    extra: dict = field(default_factory=dict)

    def __call__(self, source: np.ndarray, target: np.ndarray) -> RegistrationResult:
        corona = np.asarray(source, dtype=np.float64)
        contorno = np.asarray(target, dtype=np.float64)
        rot = np.asarray(self.rot0, dtype=np.float64).copy()
        girada = corona @ rot.T
        # Las cúspides miran al plano: hacia abajo en un maxilar, hacia arriba en una
        # mandíbula. El percentil y no el extremo, para que un vértice suelto no mande.
        cuspide = np.percentile(girada[:, 2], 1 if self.arcada == "upper" else 99)
        trans = np.r_[
            contorno[:, :2].mean(axis=0) - girada[:, :2].mean(axis=0), self.plano - cuspide
        ]
        anterior = np.inf
        iteraciones, convergido = 0, False
        while iteraciones < self.max_iter:
            iteraciones += 1
            movida = corona @ rot.T + trans
            d, j = cKDTree(movida).query(contorno)
            usa = d <= np.quantile(d, self.trim)
            r, t = kabsch(movida[j[usa]], contorno[usa])
            rot, trans = r @ rot, r @ trans + t
            error = float(np.mean(d[usa]))
            if abs(anterior - error) < self.tol_mm:
                convergido = True
                break
            anterior = error

        d = cKDTree(corona @ rot.T + trans).query(contorno)[0]
        cerca = d < self.overlap_mm
        return RegistrationResult(
            rotation=matrix_to_quaternion(rot),
            translation=(float(trans[0]), float(trans[1]), float(trans[2])),
            rms_mm=float(np.sqrt(np.mean(d**2))),
            iterations=iteraciones,
            converged=convergido,
            rms_overlap_mm=float(np.sqrt(np.mean(d[cerca] ** 2))) if cerca.any() else None,
            overlap_fraction=float(cerca.mean()),
        )


def prepara_por_cortes(
    campo: dict[str, np.ndarray],
    hu: np.ndarray,
    corona: np.ndarray,
    ejes_escaner: np.ndarray,
    arcada: str,
    *,
    exige_dispersa: bool = True,
    plano: float | None = None,
) -> tuple[np.ndarray, np.ndarray, RegistroPorCortes, dict]:
    """`(origen, destino, registrador, informe)` para `GeometricFusionAgent`.

    `corona` son los vértices de corona del escáner (etiqueta FDI > 0) y `ejes_escaner`
    su base anatómica (derecha, anterior, superior). `informe` lleva lo que el gate tiene
    que saber: la banda, si la serie es dispersa y cuánto se mueve la pose con el umbral.

    Lanza `ValueError` si no se puede: sin esmalte para el plano, sin contornos en la
    banda, o con los ejes en espejo. Quien llama decide si cae al camino de siempre. Con
    `exige_dispersa` (por defecto) lanza `SerieDensa` si la banda no tiene huecos: ahí el
    camino de siempre registra mejor, ver el módulo. `plano` se puede dar de fuera; si no,
    es el modo de la altura del esmalte (`plano_oclusal_del_esmalte`).
    """
    centros = np.asarray(campo["centers"], dtype=np.float64)
    if "origin" in campo:
        centros = centros + np.asarray(campo["origin"], dtype=np.float64)
    hu = np.asarray(hu)
    corona = np.asarray(corona, dtype=np.float64)

    if plano is None:
        esmalte = hu >= HU_ESMALTE
        if esmalte.sum() < MINIMO_CONTORNO:
            raise ValueError("el campo no tiene bastante esmalte para situar el plano oclusal")
        plano = plano_oclusal_del_esmalte(centros[esmalte][:, 2])
    z_min, z_max = banda(arcada, plano)
    serie = describe_cortes(cortes(centros), z_min, z_max)
    if exige_dispersa and not serie["sparse"]:
        raise SerieDensa(
            f"la serie no tiene huecos en la banda (máximo {serie['max_gap_mm']} mm con "
            f"paso {serie['median_step_mm']} mm)"
        )
    rot0 = pose_inicial(ejes_escaner)

    arcada_xy = centros[(hu >= HU_ARBITRO) & (centros[:, 2] >= z_min) & (centros[:, 2] <= z_max)]
    if len(arcada_xy) < 10:
        raise ValueError("no hay esmalte en la banda por encima de la mordida")
    cerca_arcada = cKDTree(arcada_xy[:, :2])

    def objetivo(umbral: float) -> np.ndarray:
        c = contornos(centros, hu, umbral, z_min, z_max)
        return c[cerca_arcada.query(c[:, :2])[0] < RADIO_ARCADA_MM] if len(c) else c

    destino = objetivo(UMBRAL_CONTORNO)
    if len(destino) < MINIMO_CONTORNO:
        raise ValueError(
            f"solo {len(destino)} puntos de contorno entre z {z_min:.1f} y {z_max:.1f} mm"
        )
    registrador = RegistroPorCortes(rot0=rot0, plano=plano, arcada=arcada)

    def coronas_en(r: RegistrationResult) -> np.ndarray:
        from fusion_agents.registration import quaternion_to_matrix

        return corona @ quaternion_to_matrix(r.rotation).T + np.asarray(r.translation)

    base = coronas_en(registrador(corona, destino))
    desplazamientos, verticales = [], []
    for u in UMBRALES_INCERTIDUMBRE:
        otro = objetivo(u)
        if len(otro) < MINIMO_CONTORNO:
            continue
        diferencia = coronas_en(registrador(corona, otro)) - base
        desplazamientos.append(np.linalg.norm(diferencia, axis=1))
        verticales.append(abs(float(diferencia[:, 2].mean())))

    informe = {
        "method": "fusion_agents.por_cortes",
        "occlusal_plane_mm": round(plano, 2),
        "arch": arcada,
        "contour_threshold": UMBRAL_CONTORNO,
        "n_contour": int(len(destino)),
        **serie,
    }
    if desplazamientos:
        todos = np.concatenate(desplazamientos)
        informe["threshold_spread_mm"] = {
            "thresholds": [*UMBRALES_INCERTIDUMBRE, UMBRAL_CONTORNO],
            "median": round(float(np.median(todos)), 2),
            "max": round(float(todos.max()), 2),
            "vertical": round(max(verticales), 2),
        }
    return corona, destino, registrador, informe


def motivos(informe: Mapping[str, Any]) -> list[str]:
    """Lo que el gate tiene que saber de un registro por cortes. Siempre algo.

    La orientación inicial supone que el paciente estaba alineado con los ejes del equipo,
    y eso nunca se comprueba: por eso se declara siempre, no solo cuando algo sale mal.
    """
    salida = [
        "registro escáner↔CBCT con orientación inicial anatómica (ejes FDI del escáner y "
        "ejes del paciente del DICOM) y ajustado solo con los cortes entre "
        f"z {informe['band_mm'][0]} y {informe['band_mm'][1]} mm, por encima de la "
        "mordida: no es un TRE y la altura es la dirección peor medida"
    ]
    if informe.get("sparse"):
        salida.append(
            f"la serie CBCT es dispersa: {informe['slices_in_band']} cortes en esa banda, "
            f"con huecos de hasta {informe['max_gap_mm']} mm frente a un paso de "
            f"{informe['median_step_mm']} mm; el registro solo ve lo que cae en esos cortes"
        )
    spread = informe.get("threshold_spread_mm")
    if spread:
        salida.append(
            "la pose cambia con el umbral del contorno ("
            + ", ".join(f"{u:.0f}" for u in spread["thresholds"])
            + f"): las coronas se mueven {spread['median']} mm de mediana y "
            f"{spread['max']} de máximo, {spread['vertical']} mm en altura. Esa es su "
            "incertidumbre, no el residuo"
        )
    return salida
