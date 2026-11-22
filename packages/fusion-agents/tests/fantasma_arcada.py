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
  desenfoca el volumen como la PSF de un equipo real. Con `esmalte="cascara"` la corona es
  una cáscara de grosor uniforme (`GROSOR_CASCARA_MM`) y por dentro es dentina; con
  `dentina` por debajo del umbral de contorno (1400) el isonivel dibuja también la unión
  amelodentinaria, un borde por DENTRO de cada corona que el escáner no ve en ninguna pose.
- Por defecto las dos arcadas apenas se tocan: las puntas quedan a ±0,5 mm de z = 0, la
  inferior desplazada media pieza hacia distal y sin resalte en los incisivos. Con
  `oclusion="sobremordida"` la inferior sube y se mete por dentro como en una mordida
  normal (ver `desplazamiento_oclusion`): así hay esmalte de la otra arcada a las alturas
  de la banda del escáner, que la máscara de esmalte no quita.
- El «escaneo» superior es la superficie de las coronas superiores por debajo de la encía,
  más una cinta de encía, teselada por su cuenta y con etiquetas FDI. Por defecto la cinta
  queda 1 mm por encima de las coronas, que asoman 9 mm. Con `encia="cuello"` la encía
  nace en el margen de cada corona, festoneado, y con `desborde_mm` las etiquetas FDI
  invaden la encía como hace el segmentador real; `corona="corta"` deja coronas clínicas
  de 6,5 mm (ver `_collar_encia`).
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


def _dentro(
    p: np.ndarray, centros: np.ndarray, fdi: np.ndarray, arriba: bool, encoge: float = 0.0
) -> np.ndarray:
    """Dentro de alguna corona: el elipsoide, cortado en el cuello (3 mm hacia la raíz).

    Sin el corte la corona sería simétrica de arriba abajo, y un escaneo vuelto del revés
    encajaría contra su mitad de la raíz tan bien como contra la de la cúspide. `encoge`
    resta esos mm a cada semieje: el núcleo de dentina bajo una cáscara de esmalte.
    """
    semi = np.stack([semiejes(int(f)) for f in fdi]) - encoge
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
# Cáscara de esmalte (`esmalte="cascara"`): grosor uniforme en toda la corona. Es lo que usó
# la rama `fix/losa-sigue-pose` (etiqueta `archivo/registro-losa-sigue-pose`) para
# reproducir el deslizamiento hacia la raíz con dentina bajo el umbral.
GROSOR_CASCARA_MM = 1.2
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
    dentina: float = 1600.0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """`(campo, hu)`. `inclinacion` rota la boca (no la rejilla) alrededor del origen.

    `raices`: `"conicas"` (por defecto, lo de siempre: conos finos) o `"realistas"` (tan
    anchas como el cuello de la corona, ver `_en_raiz_realista`). En los dos casos la raíz
    es dentina (`dentina`, 1600 por defecto).

    `esmalte`: `"macizo"` (por defecto: toda la corona a 2200), `"capa"` (una capa de
    esmalte que se adelgaza hasta el cuello sobre dentina, ver `_es_esmalte`) o `"cascara"`
    (una cáscara de `GROSOR_CASCARA_MM` uniforme sobre un núcleo de dentina).
    `dentina`: el valor de la dentina de la corona y de la raíz. Por debajo del umbral de
    contorno (1400) la corona tiene dos bordes en el isonivel, el externo y la unión
    amelodentinaria; la raíz, ninguno, y la máscara de esmalte no la ve.
    `desenfoque_mm`: σ de un desenfoque gaussiano isótropo del volumen (la PSF del equipo;
    0,425 mm de anchura a media altura son σ ≈ 0,18 mm), con el fondo blando a 0 y solo lo
    que queda por encima de `PISO_HU` como tejido duro. Se aplica antes de quitar cortes.
    `oclusion`: `"plana"` (por defecto) o `"sobremordida"` (ver `desplazamiento_oclusion`).
    Donde las coronas se tocan el volumen las une, como en un CBCT con la boca cerrada.
    """
    if raices not in ("conicas", "realistas"):
        raise ValueError(f"raices desconocidas: {raices!r} (se espera 'conicas' o 'realistas')")
    if esmalte not in ("macizo", "capa", "cascara"):
        raise ValueError(
            f"esmalte desconocido: {esmalte!r} (se espera 'macizo', 'capa' o 'cascara')"
        )
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
    elif esmalte == "cascara":
        nucleo = (_dentro(q, sup, f_sup, True, encoge=GROSOR_CASCARA_MM)
                  | _dentro(q, inf, f_inf, False, encoge=GROSOR_CASCARA_MM))
        es = corona & ~nucleo
    else:
        es = corona
    valor = np.where(es, 2200.0, np.where(corona | raiz, dentina, 0.0))
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


# Encía al cuello (`encia="cuello"`), valores de libro fijados antes de medir nada con ellos:
# - corona clínica (de la cúspide al margen gingival, en vestibular y lingual): 9 mm, lo que
#   asoma con la cinta de siempre, o `CORONA_CORTA_MM` con `corona="corta"` (un molar
#   real tiene 6–7 mm);
# - el margen es festoneado: en interproximal la papila sube `PAPILA_MM` hacia oclusal;
# - la encía sale del margen hacia apical y hacia fuera a 45° hasta tener `GROSOR_ENCIA_MM`
#   de espesor, y desde ahí sube paralela al eje del diente `ALTO_ENCIA_MM` más;
# - `desborde_mm` es cuánta encía, medida sobre la superficie del escaneo desde el margen,
#   lleva la etiqueta FDI de su diente. El segmentador del escáner etiqueta como diente el
#   45 % de la encía del experto en Teeth3DS+, típicamente 1–3 mm más allá del margen.
CORONA_NORMAL_MM = 9.0
CORONA_CORTA_MM = 6.5
PAPILA_MM = 2.5
GROSOR_ENCIA_MM = 1.0
ALTO_ENCIA_MM = 5.0
PASO_ENCIA_MM = 0.25


def _tangente_arcada(c: np.ndarray) -> np.ndarray:
    """Dirección de la arcada (mesiodistal) en planta en el diente de centro `c`."""
    t = np.arcsin(np.clip(-c[0] / 24.0, -1.0, 1.0))
    tang = np.array([-24 * np.cos(t), 22 * np.sin(t)])
    return tang / np.linalg.norm(tang)


def _altura_margen(u: np.ndarray, tang: np.ndarray, alto: float) -> np.ndarray:
    """Altura del margen sobre la cúspide en cada dirección `u` (en planta) del diente.

    `alto` en vestibular y lingual; `alto − PAPILA_MM` en mesial y distal.
    """
    return alto - PAPILA_MM * (u @ tang) ** 2


def _collar_encia(
    c: np.ndarray, f: int, alto: float, desborde: float, nu: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(V, F, fdi, s)` de la encía alrededor de un diente; `s`, distancia al margen.

    La sección es la de la corona a la altura del margen; desde ahí la encía sale a 45°
    hacia apical hasta `GROSOR_ENCIA_MM` de espesor y sube paralela al eje. Lleva la
    etiqueta `f` hasta `desborde` mm de superficie desde el margen, y 0 más allá.
    """
    semi = semiejes(f)
    tang = _tangente_arcada(c)
    ang = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    u = np.c_[np.cos(ang), np.sin(ang)]
    cuspide = c[2] - semi[2]
    zm = cuspide + _altura_margen(u, tang, alto)
    rm = semi[0] * np.sqrt(np.clip(1.0 - ((zm - c[2]) / semi[2]) ** 2, 0.0, 1.0))
    labio = GROSOR_ENCIA_MM * np.sqrt(2.0)
    s = np.arange(0.0, labio + ALTO_ENCIA_MM + 1e-9, PASO_ENCIA_MM)
    fuera = np.where(s <= labio, s / np.sqrt(2.0), GROSOR_ENCIA_MM)
    sube = np.where(s <= labio, s / np.sqrt(2.0), GROSOR_ENCIA_MM + (s - labio))
    r = rm[:, None] + fuera[None, :]
    V = np.stack([c[0] + r * u[:, :1], c[1] + r * u[:, 1:],
                  zm[:, None] + sube[None, :]], axis=-1).reshape(-1, 3)
    idx = np.arange(nu * len(s)).reshape(nu, len(s))
    F = []
    for i in range(nu):
        j = (i + 1) % nu
        for k in range(len(s) - 1):
            F += [[idx[i, k], idx[j, k], idx[j, k + 1]], [idx[i, k], idx[j, k + 1], idx[i, k + 1]]]
    dist = np.tile(s, nu)
    etq = np.where(dist <= desborde + 1e-9, f, 0)
    return V, np.array(F), etq, dist


def escaneo_superior(
    nu: int = 48, nv: int = 32, encia: str = "cinta", corona: str = "normal",
    desborde_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(V, F, fdi)` en LPS (el mundo del CBCT): coronas superiores + encía.

    `encia`: `"cinta"` (por defecto, lo de siempre: una herradura 1 mm por encima de las
    coronas, que asoman 9 mm) o `"cuello"` (la encía nace en el margen festoneado de cada
    corona, ver `_collar_encia`). Solo con `"cuello"`: `corona` (`"normal"`, 9 mm de corona
    clínica, o `"corta"`, `CORONA_CORTA_MM`) y `desborde_mm` (encía etiquetada como diente).
    """
    if encia not in ("cinta", "cuello"):
        raise ValueError(f"encía desconocida: {encia!r} (se espera 'cinta' o 'cuello')")
    if corona not in ("normal", "corta"):
        raise ValueError(f"corona desconocida: {corona!r} (se espera 'normal' o 'corta')")
    if encia == "cinta" and (corona != "normal" or desborde_mm > 0):
        raise ValueError("`corona` y `desborde_mm` solo existen con encia='cuello'")
    centros, fdi = centros_coronas("upper")
    if encia == "cuello":
        return _escaneo_cuello(centros, fdi, nu, nv, corona, desborde_mm)
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


def _escaneo_cuello(
    centros: np.ndarray, fdi: np.ndarray, nu: int, nv: int, corona: str, desborde: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coronas cortadas en su margen festoneado y un collar de encía por diente.

    Entre dientes vecinos cada punto de encía es del diente más próximo en planta (en
    unidades de su anchura), y la encía que cae dentro de otra corona se quita: así las
    papilas de dos vecinos se encuentran sin atravesarse.
    """
    alto = CORONA_NORMAL_MM if corona == "normal" else CORONA_CORTA_MM
    semis = np.stack([semiejes(int(f)) for f in fdi])
    Vs, Fs, Es = [], [], []
    n = 0

    def anade(V: np.ndarray, F: np.ndarray, E: np.ndarray, vale: np.ndarray) -> None:
        nonlocal n
        F = F[vale[F].all(1)]
        usados = np.unique(F)
        mapa = -np.ones(len(V), dtype=np.int64)
        mapa[usados] = np.arange(len(usados))
        Vs.append(V[usados])
        Fs.append(mapa[F] + n)
        Es.append(E[usados])
        n += len(usados)

    for c, f in zip(centros, fdi, strict=True):
        semi = semiejes(int(f))
        V, F = _elipsoide(c, semi, nu, nv)
        rel = V[:, :2] - c[:2]
        u = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        h = V[:, 2] - (c[2] - semi[2])
        anade(V, F, np.full(len(V), f), h < _altura_margen(u, _tangente_arcada(c), alto))
    for i, (c, f) in enumerate(zip(centros, fdi, strict=True)):
        V, F, E, _ = _collar_encia(c, int(f), alto, desborde, nu)
        plano = np.linalg.norm(V[:, None, :2] - centros[None, :, :2], axis=2) / semis[None, :, 0]
        suyo = plano.argmin(axis=1) == i
        otra = np.zeros(len(V), dtype=bool)
        for j, (cj, sj) in enumerate(zip(centros, semis, strict=True)):
            if j != i:
                otra |= (((V - cj) / sj) ** 2).sum(1) < 1.0
        anade(V, F, E, suyo & ~otra)
    return np.vstack(Vs), np.vstack(Fs), np.concatenate(Es)


def mueve(V: np.ndarray, rot: np.ndarray, tras: np.ndarray) -> np.ndarray:
    return V @ rot.T + tras
