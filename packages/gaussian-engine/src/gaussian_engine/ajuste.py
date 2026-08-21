"""Ajusta un campo gaussiano anisótropo a la densidad que midió el CBCT.

El problema, formulado: hay `N` semillas —una por vóxel de tejido duro— con posición
`x_i` y densidad `d_i` en [0,1]. Se buscan `M << N` gaussianas `(mu_j, sigma_j, q_j, a_j)`
tales que su suma reproduzca la densidad en las posiciones medidas:

    d(x) = sum_j  a_j * exp( -1/2 * || R(q_j)^T (x - mu_j) / sigma_j ||^2 )

Es una regresión de mezcla gaussiana en 3D. La pérdida se mide en la misma unidad que el
dato, así que el error se puede convertir a HU y decir si importa clínicamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# Gaussianas que ve cada punto. La suma es local: a más de ~3 sigma la contribución es
# 1e-2 y a 4 sigma 3e-4, así que truncar no es una aproximación grosera sino explotar que
# la gaussiana decae.
#
# Medido sobre el caso real (1,34 M semillas → 100 k gaussianas, 800 iteraciones):
# k=4 → 58,4 HU en 50 s · k=8 → 57,6 HU en 75 s · k=16 → 60,4 HU en 136 s.
#
# Subir de 8 no compensa y además empeora: con 16 vecinos cada punto reparte su gradiente
# entre gaussianas que apenas le contribuyen, y el ajuste se difumina. El óptimo no está
# en «ver todo lo posible» sino en ver lo que de verdad suma.
K_VECINOS = 8

# Cada cuántas iteraciones se recalcula qué gaussianas ve cada punto. No en cada una: las
# medias se mueven poco por paso y el KD-tree es lo más caro del bucle. Si no se refrescara
# nunca, una gaussiana que se aleja seguiría "vista" por puntos que ya no le tocan.
REFRESCO_VECINOS = 50

# Límites físicos de sigma, en mm. Por debajo del primero la gaussiana es más fina que el
# vóxel que la generó y estaría ajustando ruido de cuantización; por encima del segundo es
# más gorda que un conducto radicular y difumina la anatomía que se quiere conservar.
#
# ⚠️ **Se imponen por sigmoide, no por `clamp`, y la diferencia es todo.** Un `clamp` deja
# gradiente CERO fuera del intervalo: una gaussiana que arranca por encima del tope no
# vuelve a bajar nunca, se queda pegada al límite y el ajuste devuelve el tope como si
# fuera una medida. Pasó — con la siembra de una sola celda, sigma arrancaba en ~4,5 mm y
# las tres componentes salían clavadas en 3,0. La sigmoide es suave en todo R, así que el
# parámetro puede volver de donde sea.
SIGMA_MIN_MM = 0.02
SIGMA_MAX_MM = 3.0

# Puntos por lote en el paso adelante. Ata el pico de memoria a `LOTE * k * 3` floats y no
# a `N * k * 3`, que con medio millón de semillas son gigabytes.
LOTE = 131_072


@dataclass(frozen=True)
class Ajuste:
    """El campo ajustado y lo que costó ajustarlo.

    `rmse_hu` es el número que decide si esto sirve: es el error de reconstrucción de la
    densidad convertido a la unidad del dato. Para comparar, el propio CBCT tiene ruido de
    decenas de HU y el salto esmalte-dentina son ~700.
    """

    centers: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    density: np.ndarray
    rmse: float
    rmse_hu: float
    compresion: float
    iteraciones: int

    def como_artefacto(self) -> dict[str, np.ndarray]:
        """Las claves que espera el almacén, con los mismos nombres que el campo semilla."""
        return {
            "centers": self.centers.astype(np.float32),
            "scales": self.scales.astype(np.float32),
            "rotations": self.rotations.astype(np.float32),
            "density": self.density.astype(np.float32),
        }


def siembra_por_rejilla(
    centros: np.ndarray, densidad: np.ndarray, n_objetivo: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Punto de partida: una gaussiana por celda ocupada de una rejilla regular.

    **Por qué rejilla y no muestreo aleatorio.** El resultado tiene que ser reproducible
    —dos ejecuciones sobre el mismo caso deben dar el mismo campo— y un submuestreo
    aleatorio deja huecos donde el dato es denso y amontona donde es disperso. La rejilla
    reparte por ocupación, que es justo lo que el ajuste necesita para arrancar cerca.

    El paso se busca por bisección porque la ocupación no es uniforme: la fórmula cerrada
    `(volumen/n)^(1/3)` supone que el tejido llena la caja, y llena el 15-50 %.

    ⚠️ La rejilla se ancla en la esquina de la caja, no en el origen. Anclada en el origen
    una nube centrada se parte en los ocho octantes por grande que sea el paso —`floor`
    separa -0,1 de +0,1— y la bisección nunca puede bajar de ocho celdas. Se veía pidiendo
    una gaussiana y recibiendo ocho.
    """
    if len(centros) == 0:
        raise ValueError("No hay semillas que ajustar: el campo está vacío.")
    n_objetivo = max(1, min(int(n_objetivo), len(centros)))

    esquina = centros.min(axis=0)
    rel = centros - esquina
    lo, hi = 1e-3, float(rel.max()) * 2.0
    paso = hi
    for _ in range(40):
        paso = 0.5 * (lo + hi)
        n = len(np.unique(np.floor(rel / paso).astype(np.int64), axis=0))
        if n > n_objetivo:
            lo = paso
        else:
            hi = paso
        if abs(n - n_objetivo) <= max(1, n_objetivo // 100):
            break

    celda = np.floor(rel / paso).astype(np.int64)
    _, inverso = np.unique(celda, axis=0, return_inverse=True)
    m = int(inverso.max()) + 1
    cuenta = np.bincount(inverso, minlength=m).astype(np.float64)
    medias = np.column_stack(
        [np.bincount(inverso, weights=centros[:, i], minlength=m) / cuenta for i in range(3)]
    )
    amplitudes = np.bincount(inverso, weights=densidad, minlength=m) / cuenta
    return medias, amplitudes, paso


def _a_local(delta: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Lleva `delta` al sistema propio de cada gaussiana rotando por el cuaternión conjugado.

    Se usa la fórmula vectorial de Rodrigues en vez de construir la matriz de rotación:
    materializar `(P, k, 3, 3)` para medio millón de puntos son cientos de MB que se
    evitan con dos productos vectoriales.
    """
    w = quats[..., :1]
    u = quats[..., 1:]
    t = np.cross(u, delta)
    return delta - 2.0 * w * t + 2.0 * np.cross(u, t)


def evalua(
    consulta: np.ndarray,
    medias: np.ndarray,
    escalas: np.ndarray,
    rotaciones: np.ndarray,
    amplitudes: np.ndarray,
    *,
    k: int = K_VECINOS,
) -> np.ndarray:
    """Densidad reconstruida en `consulta`. En NumPy y sin torch, a propósito.

    Es el camino de **verificación**: comprobar el ajuste con el mismo código que lo
    produjo sólo demuestra que el código es consistente consigo mismo. Esta función es una
    implementación independiente, y es la que mide el error que se declara — el mismo
    criterio que usa el exportador de STL, que relee el fichero en vez de estimar.
    """
    consulta = np.asarray(consulta, dtype=np.float64)
    k = min(k, len(medias))
    _, vecino = cKDTree(medias).query(consulta, k=k)
    if k == 1:
        vecino = vecino[:, None]

    fuera = np.zeros(len(consulta), dtype=np.float64)
    for inicio in range(0, len(consulta), LOTE):
        corte = slice(inicio, inicio + LOTE)
        idx = vecino[corte]
        delta = consulta[corte, None, :] - medias[idx]
        local = _a_local(delta, rotaciones[idx]) / escalas[idx]
        fuera[corte] = (amplitudes[idx] * np.exp(-0.5 * (local**2).sum(axis=-1))).sum(axis=1)
    return fuera


def ajusta(
    centros: np.ndarray,
    densidad: np.ndarray,
    *,
    n_objetivo: int,
    hu_range: tuple[float, float] | np.ndarray = (0.0, 1.0),
    iteraciones: int = 800,
    k: int = K_VECINOS,
    tasa: float = 0.01,
    dispositivo: str | None = None,
    traza: bool = False,
) -> Ajuste:
    """Ajusta `n_objetivo` elipsoides a la densidad de las semillas.

    Optimiza las cuatro cosas a la vez —posición, escala, rotación y amplitud— porque están
    acopladas: una gaussiana no puede decidir su forma sin saber dónde estará. `torch` se
    importa aquí dentro para que este paquete se pueda usar y probar sin CUDA.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "El ajuste necesita torch, que va en el extra `gpu`: "
            "`uv sync --extra gpu`, o usar el intérprete que ya lo tenga."
        ) from exc

    centros = np.asarray(centros, dtype=np.float64)
    densidad = np.asarray(densidad, dtype=np.float64)
    if len(centros) != len(densidad):
        raise ValueError(
            f"{len(centros)} centros y {len(densidad)} densidades: cada semilla necesita la "
            "suya, y emparejarlas mal ajustaría a un campo que nadie midió."
        )

    medias0, amplitudes0, paso = siembra_por_rejilla(centros, densidad, n_objetivo)
    dev = torch.device(
        dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    def sigma_de(theta: Any) -> Any:
        return SIGMA_MIN_MM + (SIGMA_MAX_MM - SIGMA_MIN_MM) * torch.sigmoid(theta)

    def tensor(arr: Any) -> Any:
        return torch.as_tensor(
            np.ascontiguousarray(arr), dtype=torch.float32, device=dev
        )

    x, d = tensor(centros), tensor(densidad)
    m = len(medias0)
    mu = tensor(medias0).requires_grad_(True)
    # Se arranca isótropa con media celda: es la misma hipótesis que hace la semilla, así
    # que el ajuste sólo puede mejorarla. La anisotropía la tiene que ganar el dato.
    # El parámetro es la preimagen de la sigmoide, no sigma: así el arranque cae dentro
    # del intervalo por construcción en vez de depender de que el paso de rejilla lo haga.
    sigma0 = float(np.clip(paso * 0.5, SIGMA_MIN_MM * 1.01, SIGMA_MAX_MM * 0.99))
    frac = (sigma0 - SIGMA_MIN_MM) / (SIGMA_MAX_MM - SIGMA_MIN_MM)
    theta = tensor(np.full((m, 3), np.log(frac / (1.0 - frac)))).requires_grad_(True)
    q = tensor(np.tile([1.0, 0.0, 0.0, 0.0], (m, 1))).requires_grad_(True)
    a = tensor(amplitudes0).requires_grad_(True)

    kk = min(k, m)

    def vecindario() -> Any:
        """Qué gaussianas ve cada punto, con las medias donde estén ahora."""
        with torch.no_grad():
            arbol = cKDTree(mu.detach().cpu().numpy().astype(np.float64))
            _, v = arbol.query(centros, k=kk)
        return torch.as_tensor(
            np.ascontiguousarray(v.reshape(len(centros), kk)),
            dtype=torch.long, device=dev,
        )

    opt = torch.optim.Adam([mu, theta, q, a], lr=tasa)
    vecino = vecindario()
    for paso_i in range(iteraciones):
        if paso_i and paso_i % REFRESCO_VECINOS == 0:
            vecino = vecindario()

        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=dev)
        for inicio in range(0, len(centros), LOTE):
            corte = slice(inicio, inicio + LOTE)
            pred = _adelante(torch, x[corte], vecino[corte], mu, sigma_de(theta), q, a)
            perdida = torch.nn.functional.mse_loss(pred, d[corte], reduction="sum")
            perdida.backward()
            total = total + perdida.detach()
        opt.step()
        if traza and paso_i % 100 == 0:
            print(f"  iter {paso_i:4d}  rmse {float((total / len(centros)).sqrt()):.5f}")

    with torch.no_grad():
        escalas = sigma_de(theta)
        quats = torch.nn.functional.normalize(q, dim=-1)
        amps = torch.nn.functional.softplus(a)
        salida = (
            mu.cpu().numpy().astype(np.float64),
            escalas.cpu().numpy().astype(np.float64),
            quats.cpu().numpy().astype(np.float64),
            amps.cpu().numpy().astype(np.float64),
        )

    # El error se mide con la implementación independiente en NumPy, no reutilizando el
    # paso adelante de torch: ver `evalua`.
    residuo = evalua(centros, *salida, k=k) - densidad
    rmse = float(np.sqrt((residuo**2).mean()))
    lo, hi = float(hu_range[0]), float(hu_range[1])
    return Ajuste(
        centers=salida[0],
        scales=salida[1],
        rotations=salida[2],
        density=salida[3],
        rmse=rmse,
        rmse_hu=rmse * (hi - lo),
        compresion=len(centros) / len(salida[0]),
        iteraciones=iteraciones,
    )


def _adelante(
    torch: Any,
    x: Any,
    vecino: Any,
    mu: Any,
    sigmas: Any,
    q: Any,
    a: Any,
) -> Any:
    """Densidad predicha para un lote. Las restricciones van aquí y no como penalización.

    Sigma acotada, cuaternión normalizado y amplitud positiva por `softplus`: son
    propiedades que el parámetro tiene que cumplir siempre, no preferencias que se negocian
    con la pérdida. Un cuaternión sin normalizar no representa ninguna rotación.
    """
    idx = vecino
    sigma = sigmas[idx]
    quats = torch.nn.functional.normalize(q, dim=-1)[idx]
    amps = torch.nn.functional.softplus(a)[idx]

    delta = x[:, None, :] - mu[idx]
    w, u = quats[..., :1], quats[..., 1:]
    tt = torch.cross(u, delta, dim=-1)
    local = (delta - 2.0 * w * tt + 2.0 * torch.cross(u, tt, dim=-1)) / sigma
    return (amps * torch.exp(-0.5 * (local**2).sum(dim=-1))).sum(dim=1)
