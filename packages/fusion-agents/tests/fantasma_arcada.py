"""Fantasma procedural de dos arcadas en oclusión, para los tests del registro anatómico.

Sin mallas de Teeth3DS+: el guardián de datos veta mallas versionadas, y un test que
necesita un fichero que CI no tiene no prueba nada en CI.

- 14 coronas superiores y 14 inferiores, elipsoides sobre una parábola (incisivos hacia
  −y, que en LPS es anterior), en oclusión: las cúspides se entrelazan a z ≈ 0.
- Raíces cónicas por encima de las coronas superiores y por debajo de las inferiores,
  con densidad de dentina, para que el contorno de la losa traiga raíz que el escáner no ve.
  Por defecto son conos finos (1,8 mm de radio en el cuello, ápice en punta a 12 mm).
  Con `raices="realistas"` tienen la anchura del cuello de su corona y se estrechan
  despacio (ver `_en_raiz_realista`): así son en una boca real, y así compiten con la
  corona por la banda del escáner (`docs/research/registro-anatomico.md` §7).
- El «CBCT» es un campo de tejido duro en rejilla (`centers`, `origin`, HU), como el que
  deja el `cbct-agent`; con `paso_z` distinto del de planta, y cortes que se pueden quitar.
  Por defecto la corona es esmalte macizo y el borde es nítido; con `esmalte="capa"` el
  esmalte es una capa que se adelgaza hasta el cuello sobre dentina, y `desenfoque_mm`
  desenfoca el volumen como la PSF de un equipo real.
- Por defecto las dos arcadas apenas se tocan: las puntas quedan a ±0,5 mm de z = 0, la
  inferior desplazada media pieza hacia distal y sin resalte en los incisivos. Con
  `oclusion="sobremordida"` la inferior sube y se mete por dentro como en una mordida
  normal (ver `desplazamiento_oclusion`): así hay esmalte de la otra arcada a las alturas
  de la banda del escáner, que la máscara de esmalte no quita.
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


# Oclusión con sobremordida (`oclusion="sobremordida"`), valores de libro de una mordida
# normal, fijados antes de medir nada con ellos:
# - incisivos: los inferiores quedan `RESALTE_MM` por lingual de los superiores (resalte,
#   centro a centro; con incisivos redondos es también cara labial a cara labial) y suben
#   hasta que su borde queda `SOBREMORDIDA_MM` por encima del borde superior;
# - posteriores: la cúspide vestibular inferior entra `INTERCUSPIDACION_MM` en la fosa
#   superior. El desplazamiento lingual de los posteriores es el que ya trae la arcada
#   inferior, más estrecha (~1,8 mm en los molares).
# El canino reparte a medias entre los dos regímenes.
RESALTE_MM = 2.5
SOBREMORDIDA_MM = 3.0
INTERCUSPIDACION_MM = 1.5
# Cuánto de «anterior» es cada pieza, de incisivo central a segundo molar.
PESO_ANTERIOR = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])


def desplazamiento_oclusion(t: np.ndarray, fdi: np.ndarray, radio: float) -> np.ndarray:
    """Desplazamiento de cada corona inferior para pasar de `"plana"` a `"sobremordida"`.

    Lo vertical es el que deja el borde inferior a la altura pedida sobre el superior (en
    `"plana"` ya lo pasa 1 mm: −0,5 frente a +0,5). Lo horizontal va hacia lingual por la
    normal de la curva de la arcada, solo en la parte anterior: en los posteriores el
    desplazamiento lingual ya lo da que la arcada inferior es más estrecha.
    """
    w = PESO_ANTERIOR[np.asarray(fdi) % 10 - 1]
    solape_plano = 2 * SEMI[2] - 2 * CENTRO_Z
    dz = w * SOBREMORDIDA_MM + (1 - w) * INTERCUSPIDACION_MM - solape_plano
    # Normal hacia vestibular de la curva (−R·sen t, −(22·cos t − 12)) es (−22·sen t, −R·cos t).
    vestibular = np.c_[-22 * np.sin(t), -radio * np.cos(t)]
    vestibular /= np.linalg.norm(vestibular, axis=1)[:, None]
    xy = -(w * RESALTE_MM)[:, None] * vestibular
    return np.c_[xy, dz]


def centros_coronas(
    arcada: str = "upper", oclusion: str = "plana"
) -> tuple[np.ndarray, np.ndarray]:
    """`(centros, fdi)` en LPS. Derecha del paciente = −x.

    La arcada inferior va media pieza desplazada hacia distal y algo más estrecha, como en
    una oclusión real: así darle la vuelta a la de arriba no la deja encima de la de abajo.
    `oclusion` solo mueve la inferior (ver `desplazamiento_oclusion`).
    """
    if oclusion not in ("plana", "sobremordida"):
        raise ValueError(
            f"oclusión desconocida: {oclusion!r} (se espera 'plana' o 'sobremordida')"
        )
    t = np.linspace(-1.25, 1.25, 2 * N_LADO)
    if arcada == "lower":
        t = t * 1.08
    radio = 24.0 if arcada == "upper" else 21.5
    x = -radio * np.sin(t)
    y = -(22 * np.cos(t) - 12)
    z = np.full_like(t, CENTRO_Z if arcada == "upper" else -CENTRO_Z)
    fdi = []
    for i in range(len(t)):
        lado_derecho = x[i] < 0
        numero = N_LADO - i if i < N_LADO else i - N_LADO + 1
        if arcada == "upper":
            fdi.append((10 if lado_derecho else 20) + numero)
        else:
            fdi.append((40 if lado_derecho else 30) + numero)
    centros, fdi_arr = np.c_[x, y, z], np.array(fdi)
    if arcada == "lower" and oclusion == "sobremordida":
        centros = centros + desplazamiento_oclusion(t, fdi_arr, radio)
    return centros, fdi_arr


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


# Raíz realista: altura del cuello sobre el centro de la corona (el corte de `_dentro`),
# longitud del cono hasta el ápice y radio del ápice relativo al del cuello. Una raíz de
# incisivo central superior mide ~13 mm, y la de un molar algo menos por raíz pero con el
# tronco igual de ancho que el cuello; el ápice no es una punta sino un muñón romo.
CUELLO_MM = 3.0
LARGO_RAIZ_MM = 13.0
APICE_RELATIVO = 0.4


def _en_raiz_realista(
    p: np.ndarray, centros: np.ndarray, fdi: np.ndarray, arriba: bool
) -> np.ndarray:
    """Raíz con la anchura del cuello de SU corona, que se estrecha despacio hasta el ápice.

    En el cuello el radio es el de la sección del elipsoide de la corona en el corte
    (`CUELLO_MM` sobre su centro); baja linealmente hasta `APICE_RELATIVO` de ese radio a
    `LARGO_RAIZ_MM`, y el ápice se cierra con media esfera de ese radio. Una sola raíz
    circular por pieza: no imita la anatomía de un molar, solo lo que la losa del CBCT ve
    de ella, que es un tubo de dentina casi tan ancho como la corona cerca de la encía.
    """
    dentro = np.zeros(len(p), dtype=bool)
    for c, f in zip(centros, fdi, strict=True):
        semi = semiejes(int(f))
        r0 = float(semi[0] * np.sqrt(1.0 - (CUELLO_MM / semi[2]) ** 2))
        r_apice = APICE_RELATIVO * r0
        h = ((p[:, 2] - c[2]) if arriba else (c[2] - p[:, 2])) - CUELLO_MM
        s = np.clip(h / LARGO_RAIZ_MM, 0.0, 1.0)
        r = r0 + (r_apice - r0) * s
        casquete = h > LARGO_RAIZ_MM
        r = np.where(
            casquete,
            r_apice * np.sqrt(np.clip(1.0 - ((h - LARGO_RAIZ_MM) / r_apice) ** 2, 0.0, 1.0)),
            r,
        )
        dentro |= (h > 0.0) & (h < LARGO_RAIZ_MM + r_apice) & (
            np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < r
        )
    return dentro


# Capa de esmalte (`esmalte="capa"`): grosor en las paredes de la corona hasta
# `ESMALTE_PLENO_MM` sobre la cúspide, y desde ahí se adelgaza linealmente hasta cero en el
# límite amelocementario, `LAC_MM` sobre la cúspide. Valores de libro para un molar: ~1,5 mm
# de esmalte en las paredes axiales, una corona clínica de ~7–8 mm. Debajo, dentina.
GROSOR_ESMALTE_MM = 1.5
ESMALTE_PLENO_MM = 4.0
LAC_MM = 8.0
# Lo que el `cbct-agent` guarda como tejido duro tras el desenfoque.
PISO_HU = 300.0


def _es_esmalte(
    p: np.ndarray, centros: np.ndarray, fdi: np.ndarray, arriba: bool
) -> np.ndarray:
    """Puntos de corona que son esmalte: a menos del grosor local de la superficie.

    La profundidad se aproxima por `(1 − ρ)·a`, con ρ el radio normalizado del elipsoide y
    `a` su semieje en planta: exacta en las paredes axiales, que es donde está la banda.
    """
    salida = np.zeros(len(p), dtype=bool)
    for c, f in zip(centros, fdi, strict=True):
        semi = semiejes(int(f))
        d = (p - c) / semi
        rho = np.sqrt((d**2).sum(-1))
        cuspide = c[2] - semi[2] if arriba else c[2] + semi[2]
        s = (p[:, 2] - cuspide) if arriba else (cuspide - p[:, 2])
        grosor = GROSOR_ESMALTE_MM * np.clip((LAC_MM - s) / (LAC_MM - ESMALTE_PLENO_MM), 0, 1)
        salida |= (rho <= 1) & ((1 - rho) * semi[0] < grosor)
    return salida


def campo_cbct(
    paso: float = 0.4, paso_z: float | None = None, quita: tuple[float, float] | None = None,
    inclinacion: np.ndarray | None = None, raices: str = "conicas",
    esmalte: str = "macizo", desenfoque_mm: float = 0.0, oclusion: str = "plana",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """`(campo, hu)`. `inclinacion` rota la boca (no la rejilla) alrededor del origen.

    `raices`: `"conicas"` (por defecto, lo de siempre: conos finos) o `"realistas"` (tan
    anchas como el cuello de la corona, ver `_en_raiz_realista`). En los dos casos la raíz
    es dentina (1600).

    `esmalte`: `"macizo"` (por defecto: toda la corona a 2200) o `"capa"` (una capa de
    esmalte que se adelgaza hasta el cuello sobre dentina, ver `_es_esmalte`).
    `desenfoque_mm`: σ de un desenfoque gaussiano isótropo del volumen (la PSF del equipo;
    0,425 mm de anchura a media altura son σ ≈ 0,18 mm), con el fondo blando a 0 y solo lo
    que queda por encima de `PISO_HU` como tejido duro. Se aplica antes de quitar cortes.
    `oclusion`: `"plana"` (por defecto) o `"sobremordida"` (ver `desplazamiento_oclusion`).
    Donde las coronas se tocan el volumen las une, como en un CBCT con la boca cerrada.
    """
    if raices not in ("conicas", "realistas"):
        raise ValueError(f"raices desconocidas: {raices!r} (se espera 'conicas' o 'realistas')")
    if esmalte not in ("macizo", "capa"):
        raise ValueError(f"esmalte desconocido: {esmalte!r} (se espera 'macizo' o 'capa')")
    paso_z = paso if paso_z is None else paso_z
    xs = np.arange(-32, 32, paso)
    ys = np.arange(-16, 20, paso)
    zs = np.arange(-24, 24, paso_z)
    if quita is not None and desenfoque_mm <= 0:
        zs = zs[(zs < quita[0]) | (zs > quita[1])]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    p = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
    q = p if inclinacion is None else p @ inclinacion  # coordenadas de la boca sin inclinar
    sup, f_sup = centros_coronas("upper")
    inf, f_inf = centros_coronas("lower", oclusion)
    corona = _dentro(q, sup, f_sup, True) | _dentro(q, inf, f_inf, False)
    if raices == "realistas":
        raiz = _en_raiz_realista(q, sup, f_sup, True) | _en_raiz_realista(q, inf, f_inf, False)
    else:
        raiz = _en_raiz(q, sup, True) | _en_raiz(q, inf, False)
    raiz &= ~corona
    if esmalte == "capa":
        es = corona & (_es_esmalte(q, sup, f_sup, True) | _es_esmalte(q, inf, f_inf, False))
    else:
        es = corona
    valor = np.where(es, 2200.0, np.where(corona | raiz, 1600.0, 0.0))
    if desenfoque_mm > 0:
        from scipy.ndimage import gaussian_filter

        vol = gaussian_filter(valor.reshape(len(xs), len(ys), len(zs)),
                              sigma=(desenfoque_mm / paso, desenfoque_mm / paso,
                                     desenfoque_mm / paso_z))
        valor = vol.ravel()
        duro = valor >= PISO_HU
        if quita is not None:
            duro &= (p[:, 2] < quita[0]) | (p[:, 2] > quita[1])
    else:
        duro = corona | raiz
    hu = valor[duro]
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
            caras += [[idx[i, k], idx[j, k], idx[j, k + 1]],
                      [idx[i, k], idx[j, k + 1], idx[i, k + 1]]]
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
