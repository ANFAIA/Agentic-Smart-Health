"""Registro **por diente** entre dos momentos de la misma arcada.

Un registro global entrega una rígida para todo el arco. Esto entrega **una por pieza
dental**, que es lo que hace falta para decir algo sobre un diente concreto y lo que
convierte el resultado en un artefacto de agente: una 4×4 es pequeña, invertible,
auditable y un clínico puede aprobarla o rechazarla de una en una.

## Lo que está medido, sobre los dos escáneres de `histora`

El residuo **en la mitad retenida** baja de 0,154 a 0,129 mm, y mejora en los 13
dientes. Pero el número que sostiene la técnica no es ese, es su **estabilidad**:

    trim del ICP global      global      por diente
    0,7                      0,154 mm    0,129 mm
    1,0                      0,273 mm    0,148 mm

El registro global se mueve un 77 % al tocar un hiperparámetro de ajuste; el registro
por diente, un 15 %. Eso es lo que dice que está capturando geometría en vez de
absorber la arbitrariedad del ajuste global.

## Tres trampas, y las tres vienen de haber caído en ellas

1. **Sobreajuste.** Un ajuste por diente SIEMPRE baja el residuo: son 6 parámetros
   libres para pocos puntos. Por eso `registra_dientes` ajusta con la mitad de los
   vértices y mide en la otra mitad, que la transformación no ha visto.

2. **El `trim` del ICP global es un confusor, no un detalle.** Recortar descarta las
   peores correspondencias, que están en los extremos del arco, y con eso fija la pose
   al centro **por construcción**. En este proyecto eso fabricó una correlación entre
   desplazamiento y posición en el arco que se interpretó como deriva de cosido del
   escáner; al repetir con `trim=1.0` el signo de la correlación **se invirtió**
   (ρ +0,67 → −0,73). Por eso el parámetro se expone en vez de fijarse: cualquier
   patrón espacial hay que comprobarlo a los dos valores antes de contarlo.

3. **Segmentar cada momento por separado.** Si cada escaneo se etiqueta por su cuenta,
   un mismo diente cubre superficies distintas en cada uno y el ICP alinea cosas que no
   se corresponden. Medido: residuo 0,374 mm y rotaciones de hasta **39°**. Con la
   etiqueta transferida (`transfiere_etiquetas`), 0,129 mm y 4,2°.

## Y una advertencia sobre cómo leer la salida

**Lo robusto es el residuo, no el desplazamiento.** La traslación mediana por diente
pasa de 0,152 a 0,471 mm solo con cambiar el `trim`, porque se mide *contra* el registro
global: si la referencia se mueve, todas las traslaciones se mueven con ella. Las
**rotaciones** son aún menos fiables — un parche de esmalte liso desliza sobre sí mismo
sin penalizar el residuo, y por eso se devuelve `condicion`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from fusion_agents.registration import apply, icp, quaternion_to_matrix

MIN_PUNTOS = 600
"""Vértices mínimos por diente. Hace falta el doble para poder partir en dos mitades."""

RADIO_TRANSFERENCIA_MM = 1.5
TRIM_LOCAL = 0.9

MIN_DIENTES_REFERENCIA = 4
"""Dientes mínimos para que la referencia leave-one-out signifique algo.

Con menos, el marco lo fijan tan pocas piezas que el movimiento de cualquiera de ellas
contamina la referencia — que es exactamente el problema del que se está huyendo.
"""


@dataclass(frozen=True)
class DienteRegistrado:
    """El resultado por pieza: la rígida y con cuánta confianza se puede leer."""

    fdi: int
    n_origen: int
    n_destino: int
    rms_global_mm: float
    """Residuo en la mitad retenida usando SOLO la rígida global."""
    rms_diente_mm: float
    """Residuo en la misma mitad retenida usando además la rígida del diente."""
    traslacion_mm: float
    rotacion_deg: float
    condicion: float
    """Número de condición de la pose. Alto ⇒ hay una dirección por la que el parche
    desliza sin penalización, y entonces `rotacion_deg` no significa nada."""
    matriz: np.ndarray = field(repr=False)
    """4×4 COMPUESTA (global + diente): `p_destino = M @ [p_origen, 1]`."""

    @property
    def mejora_mm(self) -> float:
        return self.rms_global_mm - self.rms_diente_mm


@dataclass(frozen=True)
class DesplazamientoRelativo:
    """Cuánto se movió una pieza **respecto al resto del arco**, no respecto al fichero.

    Es la lectura que `DienteRegistrado.traslacion_mm` no puede dar: aquella se mide
    contra el registro global del arco completo, que incluye encía y que incluye el propio
    movimiento del diente medido. Aquí la referencia se ajusta con **los demás dientes**.
    """

    fdi: int
    n_diente: int
    n_referencia: int
    """Vértices que fijaron la referencia (todos los dientes menos este)."""
    dientes_referencia: int
    traslacion_mm: float
    """Desplazamiento del centroide relativo al marco de los demás dientes."""
    rotacion_deg: float
    rms_referencia_mm: float
    """Residuo de la mitad retenida con SOLO la referencia leave-one-out aplicada."""
    rms_diente_mm: float
    """El mismo residuo añadiendo la rígida propia del diente."""
    condicion: float
    matriz: np.ndarray = field(repr=False)
    """4×4 del desplazamiento RELATIVO: lleva el diente de su sitio en el marco de los
    demás dientes a su sitio real en el destino. La identidad significa «no se movió»."""

    @property
    def mejora_mm(self) -> float:
        return self.rms_referencia_mm - self.rms_diente_mm


def condicionamiento(puntos: np.ndarray, normales: np.ndarray) -> float:
    """Cuán determinada está la pose de un parche. Alto = mal.

    Matriz de información punto-a-plano `H = Σ JᵀJ` con `J = [p×n, n]`. Si un valor
    propio es diminuto, esa dirección es libre: el parche puede deslizar por ella sin
    que el residuo se entere, y lo que salga por ahí es ruido con aspecto de medida.
    """
    if len(puntos) < 6:
        return float("inf")
    c = puntos.mean(0)
    escala = max(float(np.linalg.norm(puntos - c, axis=1).mean()), 1e-9)
    J = np.hstack([np.cross((puntos - c) / escala, normales), normales])
    ev = np.linalg.eigvalsh(J.T @ J / len(puntos))
    return float(ev[-1] / max(ev[0], 1e-12))


def transfiere_etiquetas(
    origen: np.ndarray,
    destino: np.ndarray,
    etiquetas_destino: np.ndarray,
    *,
    radio_mm: float = RADIO_TRANSFERENCIA_MM,
) -> np.ndarray:
    """Pasa la segmentación del `destino` al `origen` **ya alineado**, por vecindad.

    Se etiqueta UNA vez, sobre el escaneo de mejor calidad, y se transfiere. Segmentar
    los dos por separado da a un mismo diente extensiones distintas en cada momento, y
    entonces el registro alinea superficies que no son la misma — medido, rotaciones de
    hasta 39°, que no es anatomía sino desajuste de recorte.

    Los puntos sin contrapartida dentro de `radio_mm` quedan a 0 (sin diente).
    """
    if len(destino) == 0:
        return np.zeros(len(origen), dtype=etiquetas_destino.dtype)
    d, idx = cKDTree(destino).query(origen)
    return np.where(d < radio_mm, etiquetas_destino[idx], 0)


def _rms(fuente: np.ndarray, arbol: cKDTree, trim: float = TRIM_LOCAL) -> float:
    d, _ = arbol.query(fuente)
    d = np.sort(d)[: max(3, int(len(d) * trim))]
    return float(np.sqrt((d**2).mean()))


def registra_dientes(
    origen: np.ndarray,
    destino: np.ndarray,
    etiquetas_destino: np.ndarray,
    normales_destino: np.ndarray,
    *,
    rot_global: np.ndarray,
    trans_global: np.ndarray,
    min_puntos: int = MIN_PUNTOS,
    semilla: int = 0,
) -> list[DienteRegistrado]:
    """Una rígida por diente, con el residuo medido en **validación cruzada**.

    `origen` y `destino` van en el mismo espacio de partida; `rot_global`/`trans_global`
    son la rígida del arco completo, que se aplica primero. Las etiquetas del destino se
    transfieren al origen: ver `transfiere_etiquetas` para por qué no se segmentan los
    dos por separado.

    El residuo se mide siempre sobre la mitad de vértices que **no** participó en el
    ajuste. Sin eso, la comparación está amañada a favor del método por diente, que
    tiene 6 grados de libertad para unos pocos miles de puntos.
    """
    rng = np.random.default_rng(semilla)
    origen_alineado = apply(rot_global, trans_global, origen)
    etiquetas_origen = transfiere_etiquetas(origen_alineado, destino, etiquetas_destino)

    salida: list[DienteRegistrado] = []
    for fdi in sorted({int(c) for c in np.unique(etiquetas_destino) if c}):
        src = origen_alineado[etiquetas_origen == fdi]
        sel_d = etiquetas_destino == fdi
        tgt = destino[sel_d]
        if len(src) < 2 * min_puntos or len(tgt) < min_puntos:
            continue
        arbol = cKDTree(tgt)

        orden = rng.permutation(len(src))
        ajuste, retenida = src[orden[::2]], src[orden[1::2]]
        r = icp(ajuste, tgt, trim=TRIM_LOCAL)
        rot, trans = quaternion_to_matrix(r.rotation), np.asarray(r.translation)

        centro = src.mean(0)
        matriz = np.eye(4)
        matriz[:3, :3] = rot @ rot_global
        matriz[:3, 3] = rot @ np.asarray(trans_global) + trans
        salida.append(
            DienteRegistrado(
                fdi=fdi,
                n_origen=int(len(src)),
                n_destino=int(len(tgt)),
                rms_global_mm=_rms(retenida, arbol),
                rms_diente_mm=_rms(apply(rot, trans, retenida), arbol),
                traslacion_mm=float(np.linalg.norm(trans + (rot - np.eye(3)) @ centro)),
                rotacion_deg=float(
                    np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
                ),
                condicion=condicionamiento(tgt, normales_destino[sel_d]),
                matriz=matriz,
            )
        )
    return salida


def desplazamientos_relativos(
    origen: np.ndarray,
    destino: np.ndarray,
    etiquetas_destino: np.ndarray,
    normales_destino: np.ndarray,
    *,
    rot_global: np.ndarray,
    trans_global: np.ndarray,
    trim_referencia: float = TRIM_LOCAL,
    min_puntos: int = MIN_PUNTOS,
    min_dientes: int = MIN_DIENTES_REFERENCIA,
    semilla: int = 0,
) -> list[DesplazamientoRelativo]:
    """Desplazamiento de cada pieza **contra una referencia leave-one-out**.

    Para medir el diente *X* la referencia se ajusta con **todos los dientes menos X**, y
    el movimiento de X se lee contra ese marco. Resuelve dos problemas de golpe:

    - **La referencia no contiene a X.** Midiendo contra el registro global, el propio
      desplazamiento de X entra en la referencia contra la que se mide: si X se mueve,
      arrastra un poco el marco y su desplazamiento sale infravalorado.
    - **La referencia no contiene encía.** El registro del arco completo la incluye, y la
      encía cambia de forma entre dos momentos (inflamación, retracción) sin que ningún
      diente se haya movido. Es ruido que se colaba entero en la referencia.

    Lo que se mide pasa a ser **desplazamiento relativo dentro del arco**, que es lo que
    clínicamente *es* un movimiento dental: no existe un marco absoluto en un escaneo
    intraoral, porque no hay ninguna estructura fija que el escáner vea.

    `rot_global`/`trans_global` solo hacen de arranque, para poder transferir etiquetas.
    La referencia se **reajusta** desde cero con los demás dientes, así que el resultado
    no hereda la arbitrariedad del recorte del ajuste global — que es la razón de existir
    de esta función. `trim_referencia` queda expuesto para poder comprobarlo.

    El afinado inicial sí usa **todos** los dientes, incluido el que se mide, pero solo
    como punto de partida y para etiquetar: el marco contra el que se lee cada pieza se
    vuelve a ajustar sin ella. Que un ICP arranque cerca importa; de dónde exactamente,
    no. Comprobado en los tests con una global sesgada 1,2° y 0,7 mm.
    """
    rng = np.random.default_rng(semilla)
    origen_alineado = apply(rot_global, trans_global, origen)
    etiquetas_origen = transfiere_etiquetas(origen_alineado, destino, etiquetas_destino)

    # Reafinar sobre los dientes y **volver a transferir**. Sin esto el resultado hereda
    # la calidad del arranque por una vía poco obvia: no por el marco —que se reajusta más
    # abajo— sino por las ETIQUETAS. Una global sesgada 1,2° desplaza los extremos del arco
    # más que el radio de transferencia, así que cada diente acaba con un conjunto de
    # vértices distinto, otro centroide y otra traslación. Medido: 0,228 mm de deriva, que
    # es del orden de lo que se quiere detectar. Con este paso baja a ruido.
    if (etiquetas_origen > 0).any() and (etiquetas_destino > 0).any():
        r0 = icp(
            origen_alineado[etiquetas_origen > 0],
            destino[etiquetas_destino > 0],
            trim=trim_referencia,
        )
        origen_alineado = apply(
            quaternion_to_matrix(r0.rotation), np.asarray(r0.translation), origen_alineado
        )
        etiquetas_origen = transfiere_etiquetas(origen_alineado, destino, etiquetas_destino)

    codigos = sorted({int(c) for c in np.unique(etiquetas_destino) if c})
    utiles = [
        c
        for c in codigos
        if (etiquetas_origen == c).sum() >= 2 * min_puntos
        and (etiquetas_destino == c).sum() >= min_puntos
    ]
    if len(utiles) <= min_dientes:
        return []

    salida: list[DesplazamientoRelativo] = []
    for fdi in utiles:
        otros = [c for c in utiles if c != fdi]
        ref_src = origen_alineado[np.isin(etiquetas_origen, otros)]
        ref_tgt = destino[np.isin(etiquetas_destino, otros)]

        # La referencia: rígida ajustada SOLO con los demás dientes.
        r_ref = icp(ref_src, ref_tgt, trim=trim_referencia)
        rot_ref = quaternion_to_matrix(r_ref.rotation)
        trans_ref = np.asarray(r_ref.translation)

        sel_d = etiquetas_destino == fdi
        tgt = destino[sel_d]
        src = apply(rot_ref, trans_ref, origen_alineado[etiquetas_origen == fdi])
        arbol = cKDTree(tgt)

        orden = rng.permutation(len(src))
        ajuste, retenida = src[orden[::2]], src[orden[1::2]]
        r = icp(ajuste, tgt, trim=TRIM_LOCAL)
        rot, trans = quaternion_to_matrix(r.rotation), np.asarray(r.translation)

        centro = src.mean(0)
        matriz = np.eye(4)
        matriz[:3, :3] = rot
        matriz[:3, 3] = trans
        salida.append(
            DesplazamientoRelativo(
                fdi=fdi,
                n_diente=int(len(src)),
                n_referencia=int(len(ref_src)),
                dientes_referencia=len(otros),
                traslacion_mm=float(np.linalg.norm(trans + (rot - np.eye(3)) @ centro)),
                rotacion_deg=float(
                    np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
                ),
                rms_referencia_mm=_rms(retenida, arbol),
                rms_diente_mm=_rms(apply(rot, trans, retenida), arbol),
                condicion=condicionamiento(tgt, normales_destino[sel_d]),
                matriz=matriz,
            )
        )
    return salida
