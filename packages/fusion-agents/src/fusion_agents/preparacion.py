"""Preparar las nubes que la fusión geométrica registra. La lógica que faltaba en el sistema.

**Por qué existe.** `GeometricFusionAgent.fuse` recibe dos nubes ya elegidas — `source` y
`target`— y las registra. Elegirlas bien no es trivial, y hasta ahora esa elección vivía
**en un script**: para procesar un caso clínico real había que escribir a mano el
aislamiento de la arcada, la selección del lóbulo y el submuestreo. Un orquestador que no
puede procesar un caso sin que una persona escriba el pegamento no está terminado.

Las tres decisiones, y las tres están medidas sobre `histora`:

**1 · Contra la corona, no contra todo el diente.** El escáner intraoral ve superficie de
corona; el campo del CBCT tiene además raíz y el hueso que el segmentador marca de más.
Registrar contra todo le da al ICP correspondencias que no existen:

    objetivo = todo el tejido duro   → 0,778 mm sobre el 5,9 % solapado
    objetivo = la cresta del lóbulo  → 0,649 mm sobre el 21,2 %
    objetivo = corona (HU ≥ 1200)    → mediana corona→gaussiana 0,81 mm, 78 % bajo 2 mm

⚠ Y el aviso que va con ello: entre el primero y el último, el **rms que informa el ICP
apenas se mueve** (0,622 → 0,642: *empeora*) mientras la calidad real mejora 7×. El
residuo del registro **no discrimina**, así que no sirve para elegir el objetivo — es el
mismo modo de fallo que documenta `registro_ios_cbct.separa_arcadas`.

Barriendo `hu_corona` sobre **dos pacientes** distintos, el residuo se queda en una banda
del 7 % (0,699-0,748 mm en los ocho registros) mientras el error sobre la nube completa se
mueve entre 2,9 y 5,2 mm. Y el orden **se invierte** de un paciente al otro: 1200 es el
mejor umbral en uno (3,60 mm) y el peor en el otro (4,36 mm). Por eso `HU_CORONA` se queda
donde está: no hay evidencia de que otro valor sea mejor, y elegirlo por el residuo sería
elegirlo por el número que está medido que no distingue.

**2 · Un lóbulo, no los dos.** El CBCT trae maxilar y mandíbula; un escaneo intraoral es
de una sola arcada. Registrar contra las dos deja que el ICP encaje una sobre la otra —se
parecen lo bastante como para puntuar bien estando mal— y luego cada gaussiana toma la
identidad del diente de enfrente.

⚠ **Y muchas veces no se va a poder, por la postura de la adquisición.** Un CBCT dental se
toma con el paciente **en oclusión**, así que las coronas de arriba y las de abajo se
tocan: en z forman **una sola banda**, no dos lóbulos. Medido sobre un caso real (60.471
gaussianas de corona, z ∈ [-9,6 ; 155,4] mm): valle 0,65 a HU ≥ 1200 y 1,00 a HU ≥ 1500 —
unimodal en los dos. Lo que sí se separa en dos lóbulos es el **hueso**, que está más
abajo y más arriba; las coronas, no.

Por eso `nubes_para_registro` se niega a partir y lo declara en vez de cortar por el medio.
Cortar la banda única en dos —que es lo que hacía el criterio anterior, un corte a 36,6 mm
dentro de un lóbulo que va de 32 a 47— parte una arcada por la mitad y llama «la otra
arcada» a su propia cola. El registro sale, el residuo no se queja, y media arcada se
perdió sin que nadie lo diga.

**3 · Cuál es cuál lo dice la etiqueta, nunca el residuo.** Está medido: 0,490 mm contra
un lóbulo y 0,509 contra el otro, un 3,8 % de diferencia. Una arcada dental se parece
bastante a otra arcada dental.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# HU mínima para considerar que una gaussiana es corona. Ver el punto 1 de arriba.
HU_CORONA = 1200.0

# Puntos con los que se registra. Más no mejora el residuo y sí cuesta tiempo; el ICP
# converge igual con unos miles bien repartidos.
MUESTRA = 4000


# Separación mínima entre los dos picos para llamarlos arcadas distintas.
#
# ⚠️ Es el criterio que de verdad discrimina, y se llegó a él porque el otro fallaba: la
# profundidad del valle (fracción del pico menor que queda en el mínimo) sale 0,25 tanto
# para dos arcadas separadas como para UNA nube gaussiana con sus colas, así que
# cualquier umbral cae justo encima del caso ambiguo. La distancia no: dos arcadas
# dentales están a 15-25 mm en z, y las colas de una sola están a 4-6.
SEPARACION_MINIMA_MM = 8.0

# Y el segundo pico tiene que tener MASA, no solo estar lejos: en una nube gaussiana las
# colas llegan a 8 mm con uno o dos puntos, y eso pasaba por «la otra arcada».
#
# El umbral se elige por el HUECO entre los dos casos, no a ojo, y los dos extremos están
# medidos: la cola de una gaussiana a más de 8 mm del pico trae **menos del 1 %** de su
# altura; una segunda arcada real, aunque sea la pequeña y esté parcialmente fuera del
# campo, trae **más del 5 %** — poner 0,10 rechazaba una arcada de 300 gaussianas frente
# a 4.000, que es un caso legítimo. 0,03 deja margen por los dos lados.
MASA_MINIMA_SEGUNDO = 0.03

# El valle se conserva como señal secundaria: con los dos picos bien separados pero el
# hueco lleno de puntos (metal disperso por el campo), sigue sin haber dónde cortar.
VALLE_MAXIMO = 0.25


def plano_oclusal(z: np.ndarray, bins: int = 80) -> float:
    """Altura que separa las dos arcadas: el **mínimo entre los dos lóbulos** de z.

    No un percentil. Un percentil no sabe dónde están los lóbulos y, con el metal metiendo
    puntos por todo el campo, corta donde no debe — el fallo que dio 8 mm.
    """
    return separacion_de_arcadas(z, bins=bins)[0]


def separacion_de_arcadas(z: np.ndarray, bins: int = 80) -> tuple[float, float]:
    """`(corte, valle)`. `valle` cerca de 0 = dos lóbulos claros; cerca de 1 = uno solo.

    ⚠️ La confianza no es un adorno, y lo destapó un test: sobre una nube **unimodal**
    —una sola arcada— la búsqueda del segundo pico encuentra la cola de la propia nube y
    devuelve un corte a 3,5 mm del centro. Quien lo usara para quedarse con un lóbulo
    tiraría media arcada **en silencio**.

    Mismo patrón que la razón de orientación de `marco_arcada`: se devuelve el número y
    la medida de cuánto fiarse, y quien llama decide. Aquí `nubes_para_registro` se niega
    a partir por encima de `VALLE_MAXIMO`.
    """
    h, bordes = np.histogram(z, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    pico = int(np.argmax(h))
    lejos = np.abs(centros - centros[pico]) > SEPARACION_MINIMA_MM
    if not lejos.any() or h[lejos].max() < MASA_MINIMA_SEGUNDO * h[pico]:
        return float(centros[pico]), 1.0

    segundo = int(np.flatnonzero(lejos)[np.argmax(h[lejos])])
    a, b = sorted((pico, segundo))
    i = a + int(np.argmin(h[a : b + 1]))
    altura_menor = float(min(h[pico], h[segundo]))
    valle = 1.0 if altura_menor <= 0 else float(h[i]) / altura_menor
    return float(centros[i]), valle


# Umbral de esmalte para localizar el plano oclusal. Ver `plano_oclusal_del_esmalte`.
HU_ESMALTE = 1400.0


def plano_oclusal_del_esmalte(z: np.ndarray, bins: int = 80) -> float:
    """El plano oclusal como el **modo** de la z del esmalte. Sin valle, sin modelo.

    **Por qué el modo y no el valle.** Un CBCT dental se toma en oclusión, así que las
    coronas de arriba y las de abajo se tocan: el esmalte de las dos arcadas se apila en
    la misma altura y forma **un pico**, no dos lóbulos con un hueco en medio. Buscar un
    valle ahí es buscar algo que la postura de la adquisición garantiza que no existe —
    ver el aviso del punto 2 del módulo.

    Y un plano no necesita un hueco. Solo necesita dejar cada diente del lado correcto,
    que es una pregunta distinta de «¿hay separación?».

    **Por qué es fiable.** Medido sobre el caso real, el modo apenas se mueve al cambiar
    el umbral: 40,9 · 40,8 · 40,7 · 40,7 · 39,6 · 40,9 · 40,7 mm para HU ≥ 1200 … 1900.
    Siete umbrales dentro de 1,3 mm. No hace falta acertar con el umbral.

    **Y sobre todo: no depende del segmentador.** El criterio anterior partía la máscara
    del modelo, y eso acoplaba dos etapas que deben ser independientes — cambiar de
    checkpoint movía el plano y con él el registro, sin que nadie tocara el registro.
    Está medido lo que costó: el registro se degradó 5× (p50 0,81 → 4,02 mm) y el
    nombrado perdió 7 dientes de 27, ver
    `docs/research/segmentacion-diente-cbct.md`.

    Sobre ese mismo caso el criterio viejo daba 36,3 mm frente a estos 40,8: cuatro
    milímetros y medio de diente superior asignados a la arcada de abajo.
    """
    h, bordes = np.histogram(z, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    return float(centros[int(np.argmax(h))])


def arcada_del_nombre(ruta: Path | str) -> str | None:
    """`"maxilar"`, `"mandibular"` o `None` si el nombre del fichero no lo dice.

    Es una **etiqueta del operador**, no una medida: la pone el software del escáner al
    exportar. Se usa porque la alternativa —deducirla de la geometría— se probó y falla, y
    porque una etiqueta equivocada se destapa comparando los dos ajustes.

    No se acepta un fichero que diga las dos cosas ni uno que no diga ninguna. Adivinar
    aquí es lo que dejó un registro de la mandíbula contra el maxilar puntuando 0,452 mm.
    """
    n = Path(ruta).name.lower()
    arriba = bool(re.search(r"upper|maxilar|superior", n))
    abajo = bool(re.search(r"lower|mandibular|inferior", n))
    if arriba == abajo:
        return None
    return "maxilar" if arriba else "mandibular"


def nubes_para_registro(
    campo: dict[str, np.ndarray],
    vertices: np.ndarray,
    *,
    arcada: str | None,
    hu: np.ndarray | None = None,
    hu_corona: float = HU_CORONA,
    plano: float | None = None,
    muestra: int = MUESTRA,
    semilla: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(origen, destino, informe)` listas para `GeometricFusionAgent.fuse`.

    `origen` son los vértices del escáner; `destino`, las gaussianas de corona del lóbulo
    que le corresponde, **en el marco del CBCT** (se le suma `origin`, que es para lo que
    el `cbct-agent` lo guarda).

    `informe` lleva lo que hizo falta decidir, para que el llamante lo pueda declarar en
    vez de que se pierda dentro de esta función.
    """
    centros = np.asarray(campo["centers"], dtype=np.float64)
    if "origin" in campo:
        centros = centros + np.asarray(campo["origin"], dtype=np.float64)

    informe: dict = {"arcada": arcada, "n_campo": len(centros)}

    objetivo = centros
    if hu is not None:
        corona = np.asarray(hu) >= hu_corona
        if corona.sum() >= 500:
            objetivo = centros[corona]
            informe["n_corona"] = int(corona.sum())

    if arcada is not None and plano is not None and len(objetivo) > 500:
        # Plano dado de fuera: se parte y punto. El veto del valle NO aplica aquí, y esa
        # es justo la lección — en oclusión no hay valle y aun así hay que partir.
        informe["plano_oclusal_mm"] = plano
        informe["plano_dado"] = True
        lado = objetivo[:, 2] >= plano if arcada == "maxilar" else objetivo[:, 2] < plano
        if lado.sum() >= 500:
            objetivo = objetivo[lado]
    elif arcada is not None and len(objetivo) > 500:
        corte, valle = separacion_de_arcadas(objetivo[:, 2])
        informe["valle"] = valle
        if valle > VALLE_MAXIMO:
            # Una sola arcada en el campo: no hay nada que separar, y partir aquí tiraría
            # media nube sin decirlo.
            informe["no_se_parte"] = (
                f"el campo no tiene dos lóbulos claros (valle {valle:.2f} > "
                f"{VALLE_MAXIMO}): se registra contra todo"
            )
        else:
            lado = objetivo[:, 2] >= corte if arcada == "maxilar" else objetivo[:, 2] < corte
            if lado.sum() >= 500:
                objetivo = objetivo[lado]
                informe["plano_oclusal_mm"] = corte
    informe["n_objetivo"] = len(objetivo)

    rng = np.random.default_rng(semilla)
    origen = np.asarray(vertices, dtype=np.float64)
    if len(origen) > muestra:
        origen = origen[rng.choice(len(origen), muestra, replace=False)]
    if len(objetivo) > muestra:
        objetivo = objetivo[rng.choice(len(objetivo), muestra, replace=False)]
    return origen, objetivo, informe
