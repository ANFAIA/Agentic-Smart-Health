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
from scipy.spatial import cKDTree

from fusion_agents.registration import Registrar, apply, quaternion_to_matrix

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

# Umbrales candidatos para el objetivo del ICP. **No se elige uno**: se prueban todos y
# gana el que puntúe mejor, porque está medido que ninguno vale para las dos arcadas y que
# el barrido NO es monótono — el maxilar falla en 1800 entre dos vecinos buenos, que es
# firma de mínimo local del ICP y no de mala elección. Fijar una constante sería ajustar a
# un paciente. Medido sobre `histora`, corona→esmalte:
#
#     maxilar     1200 → 6,84 mm   1400 → 1,73   1600 → 1,53   1800 → 8,13 ⚠   1900 → 1,64
#     mandibular  1200 → 7,35 mm   1400 → 7,41   1600 → 7,45   1800 → 0,73     1900 → 0,73
#
# Nótese que 1200 —el valor que esta función usaba antes— es el PEOR en las dos.
BARRIDO_OBJETIVO = (1200.0, 1400.0, 1600.0, 1800.0, 1900.0)

# La referencia del árbitro. Ningún hueso llega a densidad de esmalte, así que la
# distancia de un vértice del escáner a la gaussiana de esmalte más cercana **no depende
# del segmentador** — que es justo lo que hacía falta para desacoplar las dos etapas.
HU_ARBITRO = 1800.0

# A cuánto se considera que un vértice del escáner «encontró» su esmalte.
TOLERANCIA_ARBITRO_MM = 2.0


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


def puntua_contra_esmalte(
    vertices: np.ndarray,
    esmalte: np.ndarray,
    *,
    corona: np.ndarray | None = None,
    tolerancia: float = TOLERANCIA_ARBITRO_MM,
) -> float:
    """Fracción de vértices del escáner que caen a menos de `tolerancia` del esmalte.

    Es el árbitro para elegir entre poses, y **hace falta uno porque el rms del ICP no
    sirve**: sobre diez poses cuya calidad real iba de 0,73 a 8,13 mm, el rms cabía entero
    en 0,614-0,668. Quien eligiera por el residuo elegiría mal.

    **Por qué una fracción y no la mediana.** En el script que estrenó esta idea se usaba
    la mediana de la distancia de las CORONAS al esmalte, con las coronas identificadas
    por sus etiquetas FDI. Aquí no hay etiquetas: el orquestador recibe la malla entera,
    encía incluida, y la encía **no tiene contrapartida posible** —no hay esmalte debajo
    de la encía—. Una mediana sobre todos los vértices mediría sobre todo cuánta encía
    trae el escaneo.

    Una fracción aguanta eso: para una pose razonable la encía aporta cero, así que no
    mueve el orden entre candidatas. Lo que sube el número es cuántas coronas encontraron
    su esmalte, que es lo que se quiere puntuar. Es el mismo fallo que invalidó una
    ejecución entera cuando se corrió sin `--fdi` y la métrica salió midiendo encía contra
    dientes.

    **`corona` cierra el único agujero que le queda, y es opcional a propósito.** Una pose
    *patológica* puede empujar encía encima de los dientes y llevarse crédito que no le
    toca; con la máscara de corona eso no pasa. Pero no se exige, y la distinción es
    deliberada: esa máscara la produce **otro modelo** (`segmentar_fdi.py`), y hacer que la
    elección de pose dependa de su salida sería reintroducir por la otra puerta el
    acoplamiento que este módulo existe para quitar — cambiar de checkpoint movería el
    registro sin que nadie tocase el registro, que es lo que costó 7 dientes de 27
    (`docs/research/segmentacion-diente-cbct.md` §5).

    Así que: **se usa si está, nunca se necesita.** Es la diferencia entre una mejora y una
    dependencia.
    """
    if len(vertices) == 0 or len(esmalte) == 0:
        return 0.0
    puntos = vertices if corona is None else vertices[np.asarray(corona, dtype=bool)]
    if len(puntos) == 0:
        return 0.0
    d, _ = cKDTree(esmalte).query(puntos)
    return float((d < tolerancia).mean())


def nubes_para_registro(
    campo: dict[str, np.ndarray],
    vertices: np.ndarray,
    *,
    arcada: str | None,
    hu: np.ndarray | None = None,
    hu_corona: float = HU_CORONA,
    plano: float | None = None,
    dos_arcadas: bool = False,
    corona_origen: np.ndarray | None = None,
    registrar: Registrar | None = None,
    muestra: int = MUESTRA,
    semilla: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(origen, destino, informe)` listas para `GeometricFusionAgent.fuse`.

    `origen` son los vértices del escáner; `destino`, las gaussianas de corona del lóbulo
    que le corresponde, **en el marco del CBCT** (se le suma `origin`, que es para lo que
    el `cbct-agent` lo guarda).

    `informe` lleva lo que hizo falta decidir, para que el llamante lo pueda declarar en
    vez de que se pierda dentro de esta función.

    **Con `registrar`, el umbral del objetivo no se fija: se ELIGE midiendo.** Se prueban
    los de `BARRIDO_OBJETIVO` y gana el que más vértices del escáner deje a menos de 2 mm
    del esmalte. Hace falta porque está medido que ningún umbral vale para las dos arcadas
    y que el barrido no es monótono, y hace falta *este* árbitro porque el rms del ICP no
    distingue — ver `puntua_contra_esmalte`. Sin `registrar` se usa `hu_corona` a secas,
    que es el comportamiento de antes y **el peor de los cinco umbrales** sobre el caso
    medido.

    `corona_origen` es opcional y solo afina el árbitro; ver `puntua_contra_esmalte`.

    **`dos_arcadas` lo declara quien llama, y no se deduce.** Se intentó: con las dos
    arcadas en oclusión no hay valle que encontrar (medido: 0,65 y 1,00 sobre el caso
    real), y la extensión tampoco sirve —`HU ≥ 1400` sobre un FOV de cabeza no es esmalte,
    es todo lo denso del cráneo, y da 131 mm donde una arcada mide 25—. El dato **no sabe**
    cuántas arcadas trae, así que se declara en vez de adivinarse.

    Por defecto `False`, que deja el camino conservador de siempre: partir una nube de una
    sola arcada tiraría media sin decirlo, y ese es el fallo peor de los dos.
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
    elif (
        arcada is not None
        and dos_arcadas
        and hu is not None
        and (np.asarray(hu) >= HU_ESMALTE).sum() >= 500
    ):
        # El plano sale del ESMALTE, no de un valle. En oclusión las coronas de las dos
        # arcadas se tocan y forman un pico único: buscar un hueco es buscar algo que la
        # postura de la adquisición garantiza que no existe, y por eso esta función se
        # negaba a partir justo en el caso que importa. Un plano no necesita un hueco.
        corte = plano_oclusal_del_esmalte(centros[np.asarray(hu) >= HU_ESMALTE][:, 2])
        informe["plano_oclusal_mm"] = corte
        informe["plano_por_esmalte"] = True
        lado = objetivo[:, 2] >= corte if arcada == "maxilar" else objetivo[:, 2] < corte
        if lado.sum() >= 500:
            objetivo = objetivo[lado]
    elif arcada is not None and len(objetivo) > 500:
        # Sin HU no hay esmalte que localizar: queda el criterio del valle, que solo
        # acierta cuando de verdad hay dos lóbulos separados.
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
    rng = np.random.default_rng(semilla)
    origen = np.asarray(vertices, dtype=np.float64)
    if len(origen) > muestra:
        idx = rng.choice(len(origen), muestra, replace=False)
        origen = origen[idx]
        corona_origen = None if corona_origen is None else np.asarray(corona_origen)[idx]

    if registrar is not None and hu is not None:
        objetivo, extra = _objetivo_por_arbitro(
            origen, centros, np.asarray(hu), objetivo, corona_origen, registrar, rng, muestra
        )
        informe.update(extra)

    informe["n_objetivo"] = len(objetivo)
    if len(objetivo) > muestra:
        objetivo = objetivo[rng.choice(len(objetivo), muestra, replace=False)]
    return origen, objetivo, informe


def _objetivo_por_arbitro(
    origen, centros, hu, objetivo, corona_origen, registrar, rng, muestra
) -> tuple[np.ndarray, dict]:
    """El objetivo que mejor puntúa contra el esmalte, entre los de `BARRIDO_OBJETIVO`.

    Registra una vez por candidato sobre nubes submuestreadas —son segundos— y devuelve el
    objetivo ganador, no la pose: registrar de verdad es del `GeometricFusionAgent`, que
    además es quien calcula la confianza y alimenta el gate. Aquí solo se elige contra qué.
    """
    esmalte = centros[hu >= HU_ARBITRO]
    if len(esmalte) < 500:
        return objetivo, {"sin_arbitro": "no hay bastante esmalte para puntuar las poses"}

    # El lóbulo ya elegido acota los candidatos: se mira qué gaussianas del objetivo
    # sobreviven a cada umbral, no el campo entero otra vez.
    dentro = np.zeros(len(centros), dtype=bool)
    dentro[np.unique(cKDTree(centros).query(objetivo, k=1)[1])] = True

    mejor: tuple[float, float, np.ndarray] | None = None
    puntuaciones: dict[str, float] = {}
    for u in BARRIDO_OBJETIVO:
        cand = centros[dentro & (hu >= u)]
        if len(cand) < 500:
            continue
        if len(cand) > muestra:
            cand = cand[rng.choice(len(cand), muestra, replace=False)]
        r = registrar(origen, cand)
        movido = apply(
            quaternion_to_matrix(r.rotation), np.asarray(r.translation), origen
        )
        p = puntua_contra_esmalte(movido, esmalte, corona=corona_origen)
        puntuaciones[f"{u:.0f}"] = round(p, 4)
        if mejor is None or p > mejor[0]:
            mejor = (p, u, centros[dentro & (hu >= u)])

    if mejor is None:
        return objetivo, {"sin_arbitro": "ningún candidato tenía bastantes gaussianas"}
    return mejor[2], {
        "hu_objetivo": mejor[1],
        "puntuacion_arbitro": mejor[0],
        "puntuaciones": puntuaciones,
    }
