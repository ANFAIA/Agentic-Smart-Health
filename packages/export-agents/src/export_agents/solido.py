"""Cerrar una malla abierta en un **sólido imprimible con base plana**.

Un escaneo intraoral es una **cáscara abierta**: mide la superficie que la cámara ve y
nada más. No tiene interior, no tiene fondo, y su borde es simplemente donde el clínico
dejó de escanear. Eso se ve perfectamente en pantalla —ya nos mordió una vez, cuando el
paladar salió como un agujero negro por pintar sólo la cara frontal— pero **no se
imprime**: un laminador necesita saber qué es dentro y qué es fuera, y una superficie sin
volumen no contesta esa pregunta.

Lo que este módulo hace es lo que un laboratorio hace a mano antes de imprimir: bajar el
borde del escaneo hasta un plano y taparlo. El resultado es un modelo que **asienta**, que
es la otra mitad del problema — una arcada con la base curva se cae de la bandeja.

**Lo que NO hace, y por qué se declara en vez de intentarlo.** No repara geometría mala:
no suelda vértices duplicados, no arregla caras invertidas ni resuelve aristas
no-manifold. Si la malla de entrada tiene esos defectos, salen del otro lado. Lo que sí
hace es **medir** si el resultado es estanco y decirlo, porque «cerrado» es una propiedad
que se comprueba contando aristas, no una que se afirma.

**Los agujeros pequeños se tapan donde están; el borde grande baja al plano.** Un escaneo
real trae varios lazos de borde: el grande es la apertura del modelo y los pequeños son
huecos que la cámara no llegó a ver. Bajarlos todos al plano produciría chimeneas
absurdas atravesando el modelo, así que se distinguen por tamaño — que es lo único que
los distingue de verdad, porque un hueco de escaneo no tiene ninguna marca que lo declare.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# Cuánto se separa el plano de la base del punto más bajo del borde, en milímetros. No es
# estético: pegado al borde, la falda sale con triángulos degenerados justo donde la malla
# ya es irregular. Dos milímetros dan una pared con la que el laminador trabaja y no
# cambian la altura útil del modelo.
HOLGURA_BASE_MM = 2.0

# Por debajo de cuántas aristas un lazo de borde se considera un HUECO de escaneo y no la
# apertura del modelo. En la práctica la separación es brutal —en un escaneo real el lazo
# grande tiene miles de aristas y los huecos decenas— así que el umbral no es un ajuste
# fino: es una salvaguarda por si algún día llega una malla partida en dos trozos.
MAXIMO_HUECO = 500


def lazos_de_borde(caras: np.ndarray) -> list[list[int]]:
    """Los lazos de aristas abiertas, cada uno como una lista ordenada de vértices.

    Una arista de borde pertenece a **una sola** cara. Encadenarlas por vecindad da el
    contorno. Se devuelven ordenados de mayor a menor, que es como los usa quien llama:
    el primero es la apertura del modelo y el resto son huecos.

    Un lazo que no cierra —porque la malla tiene una arista de borde suelta— se devuelve
    igual, abierto. Taparlo produciría una cara cruzando la malla; quien llama decide.
    """
    aristas = np.sort(
        np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [0, 2]]]), axis=1
    )
    unicas, cuenta = np.unique(aristas, axis=0, return_counts=True)
    borde = unicas[cuenta == 1]

    vecinos: dict[int, list[int]] = defaultdict(list)
    for a, b in borde:
        vecinos[int(a)].append(int(b))
        vecinos[int(b)].append(int(a))

    visto: set[int] = set()
    lazos: list[list[int]] = []
    for inicio in vecinos:
        if inicio in visto:
            continue
        lazo, actual = [inicio], inicio
        visto.add(inicio)
        while True:
            siguiente = [v for v in vecinos[actual] if v not in visto]
            if not siguiente:
                break
            actual = siguiente[0]
            visto.add(actual)
            lazo.append(actual)
        lazos.append(lazo)
    return sorted(lazos, key=len, reverse=True)


def _cruz2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Producto vectorial en 2D. numpy 2 dejó de hacerlo con `np.cross`."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _area_con_signo(xy: np.ndarray) -> float:
    """Área con signo del polígono. Positiva = antihorario."""
    x, y = xy[:, 0], xy[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2.0)


def _tapa_plana(contorno: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Triangula el polígono `contorno` (ya plano) por **recorte de orejas**.

    ⚠️ **La propiedad que hace falta aquí no es la calidad de los triángulos: es que la
    frontera de la tapa sea EXACTAMENTE el anillo del contorno**, para que cada arista de
    la base la compartan la falda y la tapa y el sólido cierre. El recorte de orejas lo
    garantiza por construcción —consume el polígono arista a arista— y una triangulación
    de Delaunay filtrada por par-impar no.

    Está medido sobre un caso real, y por eso está escrito: el contorno de un escaneo
    intraoral proyectado sobre el plano de la base **se autointersecta**, 136 veces sobre
    3.978 segmentos. La arcada es una banda en herradura y su borde da 0,001 vueltas
    alrededor de su propio centroide, no una. Con esos pliegues, el filtro par-impar
    clasifica mal por los dos lados y deja 597 aristas abiertas dentro de la tapa; el
    recorte de orejas deja cero.

    Los pliegues no desaparecen: la tapa queda geométricamente mal en esos 136 sitios, que
    son dobleces de décimas de milímetro en el borde del escaneo. Cerrada y con defectos
    submilimétricos se imprime; abierta no se imprime de ninguna manera.
    """
    n = len(contorno)
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)

    xy = np.column_stack([contorno @ u, contorno @ v])
    orden = list(range(n))
    if _area_con_signo(xy) < 0.0:
        # Se trabaja siempre en antihorario para que «convexo» sea un signo y no dos casos.
        orden.reverse()

    vivos = orden
    caras: list[tuple[int, int, int]] = []
    forzadas = 0
    while len(vivos) > 3:
        idx = np.asarray(vivos)
        cruz = _cruz2(xy[idx] - xy[np.roll(idx, 1)], xy[np.roll(idx, -1)] - xy[np.roll(idx, 1)])
        # ⚠️ La prueba de contención se hace SÓLO contra los vértices reflejos, y no es un
        # atajo: un convexo nunca puede caer dentro de una oreja de este polígono, así que
        # probar contra todos multiplica el coste por n sin cambiar el resultado. Con el
        # contorno de un escaneo —cuatro mil vértices— esa diferencia es la que separa
        # segundos de horas.
        reflejos = idx[cruz <= 0.0]
        elegido = -1
        for k in np.flatnonzero(cruz > 0.0):
            i, j, sig = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            fuera = reflejos[(reflejos != i) & (reflejos != j) & (reflejos != sig)]
            if len(fuera):
                a, b, c = xy[i], xy[j], xy[sig]
                p = xy[fuera]
                dentro = (
                    (_cruz2(b - a, p - a) >= 0)
                    & (_cruz2(c - b, p - b) >= 0)
                    & (_cruz2(a - c, p - c) >= 0)
                )
                if bool(dentro.any()):
                    continue
            elegido = int(k)
            break
        if elegido < 0:
            # Un polígono que se autointersecta puede quedarse sin orejas legítimas. Se
            # corta la menos mala —la de mayor giro a la izquierda— en vez de abandonar:
            # abandonar deja el sólido abierto, que es el único fallo que no se tolera.
            elegido = int(np.argmax(cruz))
            forzadas += 1
        i, j, sig = idx[elegido - 1], idx[elegido], idx[(elegido + 1) % len(idx)]
        caras.append((int(i), int(j), int(sig)))
        vivos.pop(elegido)
    caras.append((vivos[0], vivos[1], vivos[2]))
    return np.asarray(caras, dtype=np.int64)


def _abanico(lazo: list[int], centro_idx: int) -> np.ndarray:
    """Tapa un hueco pequeño con un abanico desde un vértice nuevo en su centro.

    Vale para huecos de escaneo —decenas de aristas, casi planos y casi convexos— y no
    valdría para la apertura del modelo, que es una herradura.
    """
    n = len(lazo)
    return np.array(
        [[lazo[i], lazo[(i + 1) % n], centro_idx] for i in range(n)], dtype=np.int64
    )


def volumen_con_signo(posiciones: np.ndarray, caras: np.ndarray) -> float:
    """Volumen con signo por el teorema de la divergencia. Negativo = normales adentro.

    Sirve para orientar el sólido de una vez: una malla cerrada y con bobinado
    consistente encierra volumen positivo si sus normales miran hacia fuera. No arregla
    un bobinado **inconsistente** —ahí el número no significa nada— y por eso quien
    llama comprueba antes que la malla sea estanca.
    """
    a, b, c = (posiciones[caras[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def es_estanca(caras: np.ndarray) -> bool:
    """¿Cada arista la comparten exactamente dos caras? Es la definición imprimible.

    Se **cuenta**, no se supone: «cerrado» es la propiedad que decide si un laminador
    puede rellenar el interior, y afirmarla sin comprobarla es exactamente el tipo de
    promesa que este proyecto no hace.
    """
    aristas = np.sort(
        np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [0, 2]]]), axis=1
    )
    _, cuenta = np.unique(aristas, axis=0, return_counts=True)
    return bool((cuenta == 2).all())


def cierra_en_solido(
    posiciones: np.ndarray,
    caras: np.ndarray,
    *,
    arriba: np.ndarray,
    holgura_mm: float = HOLGURA_BASE_MM,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool | float]]:
    """Cáscara abierta → sólido con base plana perpendicular a `arriba`.

    `arriba` es la dirección hacia la coronilla del paciente, **medida** (ver
    `anatomia.marco_anatomico`), no supuesta: la base tiene que ser perpendicular al eje
    anatómico y no al eje Z del fichero, que es el del escáner y no significa nada.

    Devuelve `(posiciones, caras, informe)`. El informe lleva lo que se hizo y —lo que
    importa— si el resultado es realmente estanco.
    """
    pos = np.asarray(posiciones, dtype=np.float64)
    car = np.asarray(caras, dtype=np.int64)
    n = np.asarray(arriba, dtype=np.float64)
    n = n / np.linalg.norm(n)

    lazos = lazos_de_borde(car)
    if not lazos:
        return pos, car, {"ya_cerrada": True, "estanca": es_estanca(car)}

    # Dos ejes cualesquiera perpendiculares a `arriba`: sólo hacen falta para triangular
    # la tapa en 2D, así que su orientación en el plano da igual mientras sean ortogonales.
    auxiliar = np.array([1.0, 0.0, 0.0])
    if abs(auxiliar @ n) > 0.9:
        auxiliar = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, auxiliar)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    apertura, huecos = lazos[0], [x for x in lazos[1:] if len(x) <= MAXIMO_HUECO]
    grandes_ignorados = len(lazos) - 1 - len(huecos)

    altura_base = float((pos @ n).min() - holgura_mm)
    nuevas_pos = [pos]
    nuevas_car = [car]
    desplazamiento = len(pos)

    # --- la falda: cada arista del borde baja hasta el plano ------------------ #
    borde = pos[apertura]
    proyectado = borde - np.outer((borde @ n) - altura_base, n)
    nuevas_pos.append(proyectado)
    m = len(apertura)
    a = np.array(apertura)
    b = np.roll(a, -1)
    a2 = np.arange(m) + desplazamiento
    b2 = np.roll(a2, -1)
    nuevas_car.append(np.column_stack([a, b, b2]))
    nuevas_car.append(np.column_stack([a, b2, a2]))

    # --- la tapa: el contorno ya proyectado, triangulado en el plano ---------- #
    tapa = _tapa_plana(proyectado, u, v)
    if len(tapa):
        nuevas_car.append(tapa + desplazamiento)
    desplazamiento += m

    # --- los huecos de escaneo: abanico donde están -------------------------- #
    for hueco in huecos:
        centro = pos[hueco].mean(axis=0)
        nuevas_pos.append(centro[None, :])
        nuevas_car.append(_abanico(hueco, desplazamiento))
        desplazamiento += 1

    fuera_pos = np.concatenate(nuevas_pos)
    fuera_car = np.concatenate(nuevas_car)

    estanca = es_estanca(fuera_car)
    volumen = volumen_con_signo(fuera_pos, fuera_car)
    if estanca and volumen < 0.0:
        # Normales hacia dentro: se invierte el bobinado de TODA la malla de una vez. Sólo
        # tiene sentido si es estanca — sobre una malla con bobinado inconsistente el
        # signo del volumen no significa nada y voltear no arregla nada.
        fuera_car = fuera_car[:, ::-1].copy()
        volumen = -volumen

    return fuera_pos, fuera_car, {
        "estanca": estanca,
        "aristas_borde_antes": sum(len(x) for x in lazos),
        "lazos": len(lazos),
        "huecos_tapados": len(huecos),
        "lazos_grandes_ignorados": grandes_ignorados,
        "caras_anadidas": int(len(fuera_car) - len(car)),
        "volumen_mm3": round(volumen, 1),
        "altura_base_mm": round(altura_base, 2),
    }
