"""Fantasma procedural de dos arcadas en oclusión, para los tests del registro anatómico.

Sin mallas de Teeth3DS+: el guardián de datos veta mallas versionadas, y un test que
necesita un fichero que CI no tiene no prueba nada en CI.

- 14 coronas superiores y 14 inferiores, elipsoides sobre una parábola (incisivos hacia
  −y, que en LPS es anterior), en oclusión: las cúspides se entrelazan a z ≈ 0.
- Raíces cónicas por encima de las coronas superiores y por debajo de las inferiores,
  con densidad de dentina, para que el contorno de la losa traiga raíz que el escáner no ve.
- El «CBCT» es un campo de tejido duro en rejilla (`centers`, `origin`, HU), como el que
  deja el `cbct-agent`; con `paso_z` distinto del de planta, y cortes que se pueden quitar.
- El «escaneo» superior es la superficie de las coronas superiores por debajo de la encía,
  más una cinta de encía, teselada por su cuenta y con etiquetas FDI.
"""

from __future__ import annotations

import numpy as np

SEMI = np.array([2.4, 2.4, 6.5])
CENTRO_Z = 6.0
N_LADO = 7


def rotacion(eje: np.ndarray, grados: float) -> np.ndarray:
    eje = np.asarray(eje, dtype=np.float64)
    eje = eje / np.linalg.norm(eje)
    k = np.array([[0, -eje[2], eje[1]], [eje[2], 0, -eje[0]], [-eje[1], eje[0], 0]])
    a = np.radians(grados)
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


# Anchura relativa de cada pieza, de incisivo central a segundo molar. Con todas iguales el
# arco se podría deslizar un diente entero sin que nada lo distinga, cosa que una boca real
# no permite.
ANCHO = np.array([1.0, 0.85, 0.95, 0.9, 0.9, 1.25, 1.15])


def semiejes(fdi: int) -> np.ndarray:
    return SEMI * np.array([ANCHO[fdi % 10 - 1], ANCHO[fdi % 10 - 1], 1.0])


def centros_coronas(arcada: str = "upper") -> tuple[np.ndarray, np.ndarray]:
    """`(centros, fdi)` en LPS. Derecha del paciente = −x.

    La arcada inferior va media pieza desplazada hacia distal y algo más estrecha, como en
    una oclusión real: así darle la vuelta a la de arriba no la deja encima de la de abajo.
    """
    t = np.linspace(-1.25, 1.25, 2 * N_LADO)
    if arcada == "lower":
        t = t * 1.08
    radio = 24.0 if arcada == "upper" else 21.5
    x = -radio * np.sin(t)
    y = -(22 * np.cos(t) - 12)
    z = np.full_like(t, CENTRO_Z if arcada == "upper" else -CENTRO_Z)
    fdi = []
    for i, ti in enumerate(t):
        lado_derecho = x[i] < 0
        numero = N_LADO - i if i < N_LADO else i - N_LADO + 1
        if arcada == "upper":
            fdi.append((10 if lado_derecho else 20) + numero)
        else:
            fdi.append((40 if lado_derecho else 30) + numero)
    return np.c_[x, y, z], np.array(fdi)


def _dentro(p: np.ndarray, centros: np.ndarray, fdi: np.ndarray, arriba: bool) -> np.ndarray:
    """Dentro de alguna corona: el elipsoide, cortado en el cuello (3 mm hacia la raíz).

    Sin el corte la corona sería simétrica de arriba abajo, y un escaneo vuelto del revés
    encajaría contra su mitad de la raíz tan bien como contra la de la cúspide.
    """
    semi = np.stack([semiejes(int(f)) for f in fdi])
    d = (p[:, None, :] - centros[None]) / semi[None]
    hacia_raiz = (p[:, None, 2] - centros[None, :, 2]) * (1 if arriba else -1)
    return (((d**2).sum(-1) <= 1) & (hacia_raiz < 3.0)).any(1)


def _en_raiz(p: np.ndarray, centros: np.ndarray, arriba: bool) -> np.ndarray:
    dentro = np.zeros(len(p), dtype=bool)
    for c in centros:
        h = (p[:, 2] - c[2]) if arriba else (c[2] - p[:, 2])
        r = 1.8 * (1 - (h - 3.0) / 12.0)
        dentro |= (h > 3.0) & (h < 15.0) & (np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < r)
    return dentro


def campo_cbct(
    paso: float = 0.4, paso_z: float | None = None, quita: tuple[float, float] | None = None,
    inclinacion: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """`(campo, hu)`. `inclinacion` rota la boca (no la rejilla) alrededor del origen."""
    paso_z = paso if paso_z is None else paso_z
    xs = np.arange(-32, 32, paso)
    ys = np.arange(-16, 20, paso)
    zs = np.arange(-24, 24, paso_z)
    if quita is not None:
        zs = zs[(zs < quita[0]) | (zs > quita[1])]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    p = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
    q = p if inclinacion is None else p @ inclinacion  # coordenadas de la boca sin inclinar
    sup, f_sup = centros_coronas("upper")
    inf, f_inf = centros_coronas("lower")
    corona = _dentro(q, sup, f_sup, True) | _dentro(q, inf, f_inf, False)
    raiz = (_en_raiz(q, sup, True) | _en_raiz(q, inf, False)) & ~corona
    duro = corona | raiz
    hu = np.where(corona, 2200.0, 1600.0)[duro]
    centros = p[duro]
    origen = np.array([3.0, -2.0, 1.5])
    return {"centers": centros - origen, "origin": origen}, hu


def _elipsoide(
    c: np.ndarray, semi: np.ndarray, nu: int = 48, nv: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, np.pi, nv)[1:-1]
    uu, vv = np.meshgrid(u, v, indexing="ij")
    pts = np.c_[np.cos(uu).ravel() * np.sin(vv).ravel(),
                np.sin(uu).ravel() * np.sin(vv).ravel(),
                np.cos(vv).ravel()] * semi + c
    polos = np.array([c + [0, 0, semi[2]], c - [0, 0, semi[2]]])
    V = np.vstack([pts, polos])
    n_v = len(v)
    idx = np.arange(nu * n_v).reshape(nu, n_v)
    caras = []
    for i in range(nu):
        j = (i + 1) % nu
        for k in range(n_v - 1):
            caras += [[idx[i, k], idx[j, k], idx[j, k + 1]], [idx[i, k], idx[j, k + 1], idx[i, k + 1]]]
        caras += [[len(pts), idx[j, 0], idx[i, 0]], [len(pts) + 1, idx[i, -1], idx[j, -1]]]
    return V, np.array(caras)


def escaneo_superior(nu: int = 48, nv: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(V, F, fdi)` en LPS (el mundo del CBCT): coronas superiores + cinta de encía."""
    centros, fdi = centros_coronas("upper")
    Vs, Fs, Es = [], [], []
    n = 0
    for c, f in zip(centros, fdi, strict=True):
        V, F = _elipsoide(c, semiejes(int(f)), nu, nv)
        # La encía tapa la parte de la corona que mira a la raíz (z > c + 2,5).
        visible = V[:, 2] - c[2] < 2.5
        F = F[visible[F].all(1)]
        usados = np.unique(F)
        mapa = -np.ones(len(V), dtype=np.int64)
        mapa[usados] = np.arange(len(usados))
        Vs.append(V[usados])
        Fs.append(mapa[F] + n)
        Es.append(np.full(len(usados), f))
        n += len(usados)
    # Cinta de encía: una herradura por encima de las coronas, ancha en planta.
    t = np.linspace(-1.4, 1.4, 120)
    ancho = np.linspace(-5.0, 5.0, 9)
    tt, ww = np.meshgrid(t, ancho, indexing="ij")
    eje = np.c_[-24 * np.sin(tt.ravel()), -(22 * np.cos(tt.ravel()) - 12)]
    tang = np.c_[-24 * np.cos(tt.ravel()), 22 * np.sin(tt.ravel())]
    normal = np.c_[-tang[:, 1], tang[:, 0]] / np.linalg.norm(tang, axis=1)[:, None]
    xy = eje + normal * ww.ravel()[:, None]
    z = CENTRO_Z + 3.5 + 0.1 * ww.ravel() ** 2
    Vg = np.c_[xy, z]
    idx = np.arange(len(Vg)).reshape(len(t), len(ancho))
    Fg = []
    for i in range(len(t) - 1):
        for k in range(len(ancho) - 1):
            Fg += [[idx[i, k], idx[i + 1, k], idx[i + 1, k + 1]],
                   [idx[i, k], idx[i + 1, k + 1], idx[i, k + 1]]]
    Vs.append(Vg)
    Fs.append(np.array(Fg) + n)
    Es.append(np.zeros(len(Vg), dtype=np.int64))
    return np.vstack(Vs), np.vstack(Fs), np.concatenate(Es)


def mueve(V: np.ndarray, rot: np.ndarray, tras: np.ndarray) -> np.ndarray:
    return V @ rot.T + tras
