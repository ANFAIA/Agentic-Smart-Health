"""Color por PIEZA desde una foto clinica, **sin resolver la pose de la camara**.

`pose_foto` resuelve por PnP desde donde se tomo una foto y proyecta su color vertice a
vertice. Funciona con la oclusal —0,63 mm— y **no funciona con las laterales**: de seis
fotos aprovechables solo una resuelve pose, y el resultado es que el 35 % de la superficie
se queda sin medida y se rellena. En el visor ese relleno se ve como manchas blancas donde
no llega ninguna camara y como un marron embarrado en la encia vestibular, heredado del
vertice medido mas cercano —que ahi es una sombra interdental de la oclusal—.

Este modulo toma el otro camino, que es el que un clinico usa: **no hace falta saber desde
donde se tomo la foto, hace falta saber que region de la foto es que diente**. El color
pasa a tener soporte REGIONAL —un valor por pieza y por tercio, como una guia VITA— en vez
de por punto.

Tres consecuencias, y las tres son mejoras:

* **No hay huecos que rellenar.** Toda pieza que aparezca en alguna foto tiene color medido.
* **Es robusto.** La mediana de miles de pixeles de una corona aguanta un brillo especular
  y algun pixel de encia colado en el borde; un vertice suelto, no.
* **Degrada bien frente al fallo de segmentacion.** Un vertice mal etiquetado recibe el
  color de un diente vecino —un color de diente creible— en vez de una sombra. La
  proyeccion por vertice AMPLIFICA ese fallo; por pieza lo absorbe.

⚠️ **Esto es color medido, NO un tono certificado.** Medido sobre una lateral real, el `L*`
baja de 76,6 en el 11 a 61,2 en el 27 y eso no es que los molares sean mas oscuros: es el
flash cayendo hacia el fondo de la boca. Para un tono VITA haria falta una referencia gris
o una pestana de guia dentro del encuadre, y una serie clinica no la lleva. Para pintar el
gemelo vale —el fondo de la boca se ve mas oscuro de verdad—; para afirmar un tono, no.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaussian_engine.pose_foto import _arco, _parte_en_piezas, mascara_diente

# Cuantos pixeles se retiran desde la costura oclusal antes de partir la banda superior.
#
# La costura sale exacta, pero el pixel justo por encima de ella todavia lleva sombra de la
# linea de mordida. Se mide en pixeles de la imagen reducida porque es donde se hace todo
# el trabajo de mascara.
MARGEN_COSTURA_PX = 6

# Percentil de `L*` por encima del cual un pixel se descarta por brillo especular. El
# esmalte mojado devuelve el flash directo: eso es la camara, no el diente.
BRILLO_PCT = 90.0

# Minimo de pixeles para que una pieza o un tercio den un color. Por debajo, la mediana la
# decide el ruido y es mejor no declarar nada.
MINIMO_PIXELES = 60
MINIMO_POR_TERCIO = 30

# Alto, en pixeles de la imagen reducida, de la banda de mucosa que se muestrea justo por
# encima del cuello de cada corona. Ver `encia_contigua`.
BANDA_ENCIA_PX = 40
MINIMO_PIXELES_ENCIA = 200

# Cuantas coronas con sonda de encia hacen falta para estimar la caida del flash. Por
# debajo, la pendiente la decide el ruido y NO se corrige nada: se declara y se sigue.
MINIMO_CORONAS_SONDA = 4

# Valor de `lado_conocido` con el que una persona declara que una foto NO es una tira
# vestibular de la arcada que se esta procesando. Es 0 porque no hay FDI 0.
NO_ES_TIRA = 0

# Fraccion de pixeles de la corona que el rechazo de mucosa puede llegar a tirar. Si tira
# mas, lo que sobra no es contaminacion en el borde: es que la mascara no es esa corona, y
# entonces el rechazo no arregla nada y se deja el dato crudo.
RECHAZO_MAXIMO = 0.45



def _lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB completo. `pose_foto._lab_a` solo devuelve `a*`, que aqui no basta."""
    s = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def rgb_de_lab(lab: np.ndarray) -> np.ndarray:
    """CIELAB -> sRGB de 0 a 255. Para pintar; el valor que se declara es el Lab."""
    L, a, b = np.asarray(lab, dtype=np.float64)
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f > 0.206897, f ** 3, (f - 16 / 116) / 7.787)
    xyz = xyz * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    lin = np.clip(xyz @ M.T, 0.0, None)
    s = np.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, 12.92 * lin)
    return np.clip(s * 255.0, 0, 255).astype(np.uint8)


def costura_oclusal(mascara: np.ndarray, luminancia: np.ndarray) -> np.ndarray:
    """Fila de la linea de mordida para cada columna. `(ancho,)`.

    En una lateral las dos arcadas salen pegadas y la mascara de diente las da como **una
    sola componente**, asi que hay que partirlas antes de poder hablar de piezas del
    maxilar. Donde los dientes de arriba muerden a los de abajo queda una linea oscura
    continua de lado a lado de la foto.

    Se busca como camino de coste minimo —una fila por columna, moviendose como mucho una
    fila por paso— con el coste igual a la luminancia dentro de la mascara y prohibitivo
    fuera del alcance vertical de esta. Asi el camino se queda pegado a la costura en vez
    de escaparse por la encia, que es mas clara.

    Medido sobre las cuatro laterales de un caso real: sale exacta en las cuatro, con el
    festoneado de la mordida y el escalon posterior incluidos.
    """
    alto, ancho = mascara.shape
    lum = np.asarray(luminancia, dtype=np.float64)
    # ⚠️ **El castigo por salir de la mascara tiene que ser MODESTO, y esto costo un bug.**
    # La linea de mordida es oscura, asi que el suelo de luminancia de `mascara_diente` la
    # deja FUERA de la mascara de diente. Con un castigo grande —se probo `max + 60`— el
    # camino prefiere cruzar por diente iluminado antes que por la costura, o sea que la
    # penalizacion prohibia exactamente lo que se estaba buscando. En fotos reales colaba
    # porque el surco cae en parte dentro de la mascara; en cuanto la mordida es nitida,
    # falla.
    #
    # Escapar por la encia ya lo impide el alcance vertical de la mascara columna a
    # columna. El castigo solo tiene que desempatar, y se escala a la propia imagen para
    # que no dependa de la exposicion.
    dentro = lum[mascara] if mascara.any() else lum.ravel()
    castigo = 0.25 * float(np.percentile(dentro, 95) - np.percentile(dentro, 5))
    coste = np.where(mascara, lum, lum + castigo)
    filas = np.arange(alto)[:, None]
    hay = mascara.any(axis=0)
    arriba = np.where(hay, mascara.argmax(axis=0), 0)
    abajo = np.where(hay, alto - 1 - mascara[::-1].argmax(axis=0), alto - 1)
    coste = np.where((filas < arriba[None, :]) | (filas > abajo[None, :]), 1e6, coste)

    acumulado = coste.copy()
    de_donde = np.zeros((alto, ancho), dtype=np.int32)
    indices = np.arange(alto)
    for x in range(1, ancho):
        izquierda = np.roll(acumulado[:, x - 1], 1)
        izquierda[0] = 1e9
        derecha = np.roll(acumulado[:, x - 1], -1)
        derecha[-1] = 1e9
        tres = np.stack([izquierda, acumulado[:, x - 1], derecha])
        k = tres.argmin(axis=0)
        acumulado[:, x] = coste[:, x] + tres[k, indices]
        de_donde[:, x] = indices + (k - 1)
    y = np.zeros(ancho, dtype=np.int32)
    y[-1] = int(acumulado[:, -1].argmin())
    for x in range(ancho - 1, 0, -1):
        y[x - 1] = de_donde[y[x], x]
    return y


# Por encima de esta razon entre la luminancia de la costura y la del diente se considera
# que **no hay costura**: el camino de coste minimo existe siempre, pero si no hay una linea
# de mordida que seguir se ve obligado a pasar por diente iluminado.
#
# ⚠️ Es lo que separa una vista lateral de una OCLUSAL, y sin ello el color sale sin
# sentido: en una oclusal se ve la cara de mordida de frente, asi que **no existe un eje
# cervical-incisal** y partir la corona en tercios reparte bandas que no son nada. Medido
# sobre un caso real: la oclusal da 1,01 —su costura es tan clara como los dientes— y las
# cuatro vistas con eje dan 0,76 a 0,87.
COSTURA_MAXIMA = 0.95

# ⚠️ **Cuanto se le concede a la tabla de alturas antes de dejar de pintar corona.**
# Es un juicio DECLARADO, no una medida — el mismo criterio que la `TOLERANCIA` de
# `scripts/mide_segmentacion.py`, y por el mismo motivo: los valores de Wheeler son medias
# de poblacion, y el escaner entra un poco en el surco. A 1,3 una corona tendria que asomar
# un 30 % mas que la mayor de su tipo para que se le recorte el color.
MARGEN_ALTURA = 1.3


def costura_es_real(mascara: np.ndarray, luminancia: np.ndarray,
                    costura: np.ndarray) -> bool:
    """¿La costura encontrada es una linea de mordida, o el camino menos malo?

    ⚠️ **Necesario y NO suficiente.** Descarta la oclusal del maxilar —su costura sale tan
    clara como los dientes— pero **no** una oclusal de la arcada CONTRARIA: ahi el camino
    cruza por lengua y suelo de boca, que si son oscuros, y la herradura partida da una
    tira de coronas perfectamente creible. En un caso real la oclusal mandibular paso este
    filtro con 0,82 y daba trece coronas seguidas que se habrian emparejado con codigos del
    maxilar.

    Lo que lo cierra es el lado declarado: una foto sin `lado_conocido` no aporta color, y
    un codigo de la arcada contraria no cuadra ni con la hipotesis ni con su espejo. O sea
    que el filtro geometrico adelgaza el problema y **la declaracion es la que decide**.
    """
    lum = np.asarray(luminancia, dtype=np.float64)
    if not mascara.any():
        return False
    en_costura = float(np.median(lum[costura, np.arange(len(costura))]))
    en_diente = float(np.median(lum[mascara]))
    return en_diente > 0 and en_costura / en_diente < COSTURA_MAXIMA


def banda_superior(mascara: np.ndarray, costura: np.ndarray,
                   margen: int = MARGEN_COSTURA_PX) -> np.ndarray:
    """La arcada de arriba, retirada `margen` pixeles de la costura. Ver `MARGEN_COSTURA_PX`."""
    filas = np.arange(mascara.shape[0])[:, None]
    return mascara & (filas < (costura[None, :] - margen))


def coronas(banda: np.ndarray, esperadas: int = 8) -> tuple[np.ndarray, list[int]]:
    """Parte una banda en coronas y las devuelve ORDENADAS a lo largo de la foto.

    El watershed es el de `pose_foto`: parte por los cuellos interproximales sin encoger
    las piezas, que es lo que hace falta cuando son de tamanos muy distintos. Lo que anade
    esto es el orden, que es la mitad util — una tira ordenada se puede alinear con el arco
    aunque no se sepa que diente es cada una.
    """
    etiquetas, ids = _parte_en_piezas(banda, esperadas)
    if len(ids) == 0:
        return etiquetas, []
    centro = [float(np.flatnonzero((etiquetas == k).any(axis=0)).mean()) for k in ids]
    return etiquetas, [int(k) for _, k in sorted(zip(centro, ids, strict=True))]


def anchos_aparentes(etiquetas: np.ndarray, orden: list[int]) -> np.ndarray:
    """Ancho de cada corona a lo largo de la foto, en pixeles.

    Percentiles 3-97 y no el maximo: un pixel suelto del borde del blob inflaria el ancho
    sin que la corona sea mas ancha.
    """
    salida = []
    for k in orden:
        cols = np.flatnonzero((etiquetas == k).any(axis=0))
        salida.append(float(np.quantile(cols, 0.97) - np.quantile(cols, 0.03)))
    return np.asarray(salida)


@dataclass(frozen=True)
class Alineamiento:
    """Que pieza del arco es cada corona de la tira, y **cuanto se puede afirmar**."""

    fdis: list[int]
    coste: float
    margen: float
    """Cuanto mejor es esta hipotesis que la segunda. `0.0` significa EMPATE."""

    @property
    def ambiguo(self) -> bool:
        """Un empate no es una identificacion. Ver `alinea_con_el_arco`."""
        return self.margen <= 1e-9


def alinea_con_el_arco(anchos_foto: np.ndarray, arco: list[int],
                       anchos_arco: np.ndarray) -> Alineamiento | None:
    """Alinea la tira de coronas con el arco por su huella de anchuras.

    La foto da una tira ORDENADA; el arco da una secuencia de piezas con su ancho
    mesiodistal. Alinear las dos es un problema de una dimension.

    ⚠️ **`anchos_arco` tiene que ser la TABLA anatomica, no lo medido sobre la malla.**
    Medirlo sobre la malla es circular: esos anchos son justo lo que la segmentacion tiene
    roto —9 de 14 coronas de un caso real se pasan de 1,5 mm, y el 17 sale de 16,5 cuando
    su tabla son 9,0—, asi que la huella contra la que se compara estaria deformada por el
    mismo fallo que se quiere esquivar. Se probo primero con la malla y daba un tramo que
    acababa en premolar cuando la foto ensena la tuberosidad.

    ⚠️ **El escorzo se quita como constante.** En una lateral cada corona esta un poco mas
    lejos que la anterior, asi que el ancho aparente se multiplica por un factor casi fijo
    a cada paso; en log-razon eso es un SUMANDO constante y domina la senal —los anchos en
    pixel DECRECEN hacia los molares mientras los reales crecen—. Centrando las dos
    secuencias de log-razones desaparece y queda la forma.

    ⚠️ **Y NO puede decidir de que lado de la boca es la foto.** El arco es un espejo
    exacto: la tabla del maxilar leida al derecho y al reves es la misma lista. Asi que la
    mejor hipotesis y la segunda empatan SIEMPRE con `margen` 0,0 y son una la imagen
    especular de la otra. Eso no es un fallo del metodo, es una propiedad de la anatomia:
    ninguna huella de anchuras puede distinguir el 16 del 26. El bit que falta lo tiene que
    poner otra cosa —un hallazgo asimetrico del informe, o una persona—, y por eso se
    devuelve `margen` en vez de tragarse la ambiguedad.
    """
    n = len(anchos_foto)
    if n < 3 or len(anchos_arco) < n or len(arco) != len(anchos_arco):
        return None
    ref = np.log(np.asarray(anchos_foto)[1:] / np.asarray(anchos_foto)[:-1])
    ref = ref - ref.mean()
    candidatos: list[tuple[float, list[int]]] = []
    for sentido in (1, -1):
        anchos = np.asarray(anchos_arco)[::sentido]
        piezas = list(arco)[::sentido]
        for salto in range(len(anchos) - n + 1):
            tramo = np.log(anchos[salto + 1:salto + n] / anchos[salto:salto + n - 1])
            tramo = tramo - tramo.mean()
            candidatos.append((float(np.abs(ref - tramo).mean()),
                               [int(f) for f in piezas[salto:salto + n]]))
    candidatos.sort(key=lambda t: t[0])
    coste, fdis = candidatos[0]
    margen = (candidatos[1][0] - coste) if len(candidatos) > 1 else float("inf")
    return Alineamiento(fdis=fdis, coste=coste, margen=margen)


@dataclass(frozen=True)
class TonoPieza:
    """El color medido de una pieza, por tercios, y de que foto salio."""

    fdi: int
    lab: np.ndarray
    """(3, 3): cervical, medio e incisal, cada uno `L*a*b*`."""
    n_pixeles: int
    foto_sha256: str
    correccion: tuple[float, float, float] | None = None
    """La pendiente por canal con la que se descontó la iluminación, o `None` si no se
    pudo corregir. Ver `ajuste_de_iluminacion`. Viaja con la medida porque un tono
    corregido y uno crudo NO son comparables entre sí, y quien lee tiene que poder
    distinguirlos sin reconstruir cómo se calcularon."""

    @property
    def rgb(self) -> np.ndarray:
        """(3, 3) uint8, para pintar."""
        return np.stack([rgb_de_lab(v) for v in self.lab])


def _sha256(ruta: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def _lineal(srgb: np.ndarray) -> np.ndarray:
    """sRGB 0-255 a luz lineal 0-1. La luz se multiplica; `L*` y sRGB, no."""
    v = np.asarray(srgb, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb(lineal: np.ndarray) -> np.ndarray:
    """La vuelta de `_lineal`, a 0-255."""
    v = np.clip(np.asarray(lineal, dtype=np.float64), 0.0, 1.0)
    g = np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)
    return np.clip(g, 0.0, 1.0) * 255.0


def encia_contigua(mascara: np.ndarray, etiquetas: np.ndarray,
                   lineal: np.ndarray) -> np.ndarray | None:
    """La mucosa pegada al cuello de UNA corona, en luz lineal. `(3,)` o `None`.

    ⚠️ **Es la tarjeta gris que la foto no lleva.** En una intraoral no hay referencia
    conocida en el encuadre, asi que el nivel absoluto de una corona es indistinguible de
    lo lejos que le llegase el flash. Pero la encia adherida SI es una superficie de color
    casi constante a lo largo de la arcada, esta a la misma distancia de la camara que el
    diente que toca y recibe la misma luz: sirve de sonda de iluminacion local.

    Se muestrea hacia el cuello —`y` menor, ver `HACIA_CERVICAL_SUPERIOR`— y solo donde no
    hay corona de nadie (`etiquetas == 0`), para no medir el diente de al lado.
    """
    ys, xs = np.nonzero(mascara)
    if len(ys) == 0:
        return None
    techo = int(ys.min())
    banda = np.zeros_like(mascara)
    banda[max(0, techo - BANDA_ENCIA_PX):techo, int(xs.min()):int(xs.max()) + 1] = True
    banda &= etiquetas == 0
    if int(banda.sum()) < MINIMO_PIXELES_ENCIA:
        return None
    pix = lineal[banda]
    claro = _lab(_srgb(pix))[:, 0]
    pix = pix[claro < np.percentile(claro, BRILLO_PCT)]
    return np.median(pix, axis=0)


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """Pendiente robusta: la mediana de las pendientes de todos los pares.

    Con una docena de coronas y alguna con la sonda contaminada, un minimos cuadrados se
    va detras del punto malo; la mediana de pendientes no.
    """
    pend = [(y[j] - y[i]) / (x[j] - x[i])
            for i in range(len(x)) for j in range(i + 1, len(x))
            if abs(x[j] - x[i]) > 1e-6]
    return float(np.median(pend)) if pend else 0.0


# Cuanto mas roja que su corona tiene que ser una sonda para admitirse como mucosa, en `a*`.
#
# ⚠️ **La banda de encia se toma por geometria, y ahi arriba no siempre hay encia.** Se
# muestrea lo que queda por encima del cuello y no es corona de nadie: en una foto clinica
# eso puede ser un separador de plastico, un labio, el espejo o el fondo negro de la boca.
# Corregir una corona contra un separador blanco la deja con un color inventado y con toda
# la pinta de estar medido.
#
# ⚠️ **Y el criterio es DIRECCIONAL, no de dispersion.** El primer intento rechazaba las
# sondas que se apartaban mas de 3 MAD de las demas, y en el caso real tiro justo las dos
# buenas: la encia del 13 y la del 22 son papila, mas saturadas que la encia adherida
# (`a*` 36,1 y 31,2 frente a 25-30), no un separador. Rechazarlas dejaba esas dos piezas
# con el artefacto puesto y con la mucosa dentro de la muestra — peor que no filtrar nada.
# Lo que separa mucosa de no-mucosa no es apartarse de la media: es el SIGNO. Medido sobre
# el caso, las dieciseis sondas de encia estan entre 12,5 y 22,2 puntos de `a*` por encima
# de su corona; un plastico blanco, un espejo o el fondo estarian en cero o por debajo.
MARGEN_MUCOSA = 5.0


def sondas_de_mucosa(dientes: list[np.ndarray],
                     encias: list[np.ndarray]) -> np.ndarray:
    """Cuales de esas sondas son mucosa de verdad. `(n,)` de `bool`.

    Se compara en `a*` y no en `L*`: la claridad es justo lo que varia por la caida del
    flash —es el artefacto que se esta corrigiendo—, mientras que la mucosa es mas roja que
    el esmalte con cualquier luz. Ver `MARGEN_MUCOSA`.
    """
    if not encias:
        return np.zeros(0, bool)
    a_encia = _lab(_srgb(np.stack(encias)))[:, 1]
    a_diente = _lab(_srgb(np.stack(dientes)))[:, 1]
    return a_encia - a_diente >= MARGEN_MUCOSA


def ajuste_de_iluminacion(dientes: list[np.ndarray], encias: list[np.ndarray],
                          ) -> tuple[np.ndarray, np.ndarray] | None:
    """Referencia de encia y cuanto del diente predice la encia. `(ref, beta)` o `None`.

    ⚠️ **No se divide por la sonda, se le quita al diente solo lo que la sonda EXPLICA.**
    Dividir asume que la encia recibe exactamente la misma luz que el diente y nada mas, y
    es falso: la encia esta retraida respecto al plano vestibular y le entra sombra propia,
    asi que su rango es MAYOR que el del diente. Medido en un caso real: el diente recorre
    21,6 puntos de `L*` y la encia 24,8. Dividir por ella no aplana el degradado — lo
    invierte, y el molar mas oscuro sale como el diente mas claro de la boca.

    Lo que se hace es una regresion en log: `beta` es la pendiente de `log(diente)` contra
    `log(encia/ref)` por canal, y se resta `beta * log(encia/ref)`. `beta = 1` seria
    dividir; `beta = 0`, no tocar nada. Sale del dato —en ese mismo caso, 0,63 / 0,59 /
    0,69— y no de una constante elegida a mano.

    ⚠️ **Y esto tiene un coste que hay que declarar.** Un diente de verdad tambien se
    vuelve algo mas cromatico hacia atras, y esa parte va correlacionada con la posicion
    igual que la luz: al quitar la componente que la encia predice se va tambien un poco de
    ella. Se acepta porque la alternativa es dejar un artefacto de 21 puntos de `L*` que
    hace que el 21 salga blanco y el 27 marron siendo la misma boca.
    """
    if len(dientes) < MINIMO_CORONAS_SONDA:
        return None
    d = np.stack(dientes)
    e = np.stack(encias)
    ref = np.median(e, axis=0)
    beta = np.array([
        np.clip(_theil_sen(np.log(e[:, c] / ref[c]), np.log(d[:, c])), 0.0, 1.0)
        for c in range(3)
    ])
    return ref, beta


def _sin_mucosa(lab: np.ndarray, encia_lab: np.ndarray) -> np.ndarray:
    """Mascara de los pixeles que estan mas cerca del diente que de la encia, en `a*b*`.

    ⚠️ **Una mediana aguanta un pixel de encia, no un tercio de encia.** El docstring de
    `tono_por_tercios` dice que la mediana absorbe «algun pixel de encia colado en el
    borde», y es cierto mientras el blob sea la corona. Cuando la segmentacion se desborda
    sobre la mucosa —que es justo lo que hace en este caso, medido— entra mucha, y entonces
    la mediana se mueve: un canino salio con `a* 17,5` teniendo vecinos en `a* 7,2` y `7,4`.
    Eso no es un canino cromatico, es encia.

    Se separa por `a*b*` y no por `L*` a proposito: la iluminacion mueve sobre todo `L*`,
    asi que un umbral en claridad confundiria sombra con mucosa. En cromaticidad la encia y
    el esmalte estan lejos —`a*` 29 contra 8 en el caso medido— y la luz apenas los mueve.
    """
    centro = np.median(lab, axis=0)
    al_diente = np.linalg.norm(lab[:, 1:] - centro[1:], axis=1)
    a_la_encia = np.linalg.norm(lab[:, 1:] - encia_lab[1:], axis=1)
    return al_diente <= a_la_encia


@dataclass(frozen=True)
class _Lectura:
    """Una corona localizada en una foto, antes de saber con que luz se tomo.

    El color no se puede calcular en el mismo bucle que la localiza: la correccion de
    iluminacion se estima sobre todas las coronas de todas las fotos a la vez, asi que hace
    falta tenerlas todas antes de medir ninguna.
    """

    digest: str
    rgb: np.ndarray
    mascara: np.ndarray
    fdi: int
    encia: np.ndarray | None


def tonos_de_fotos(
    fotos: list[Path],
    arco: list[int],
    anchos_arco: np.ndarray,
    *,
    lado_conocido: dict[Path, int] | None = None,
    esperadas: int = 8,
) -> tuple[list[TonoPieza], list[str]]:
    """El color de cada pieza tomado de la foto que **mejor la ve**, y los motivos de gate.

    ⚠️ **Cada pieza se queda con la foto donde ocupa MAS PIXELES**, no con la primera ni
    con una media. Una lateral ve la cara vestibular de frente y la oclusal la ve de canto;
    promediarlas mezcla una medida buena con una estirada. Sin pose no se puede calcular el
    coseno con la normal, pero el area en pixeles ordena igual de bien: un diente visto de
    frente y de cerca ocupa mas.

    ⚠️ **Una foto cuya lateralidad no se resuelve NO aporta color.** El arco es un espejo
    exacto, asi que la huella de anchuras nunca puede decir si una tira es el lado derecho
    o el izquierdo (ver `alinea_con_el_arco`), y **la imagen tampoco**: una oclusal de
    maxilar se toma siempre con espejo intraoral, asi que su sentido depende de un dato que
    no viaja con el fichero. Aceptar la hipotesis de mas peso seria jugarselo a cara o cruz
    con el color del 16 y el del 26 — y eso, en una pieza que se va a restaurar, no es un
    error pequeno. Se declara y se pregunta.

    `lado_conocido` es esa respuesta cuando existe: para cada foto, el codigo FDI que le
    corresponde a su PRIMERA corona. Con eso la ambiguedad desaparece y todo lo demas sigue
    siendo medido.
    """
    lado_conocido = lado_conocido or {}
    # ⚠️ **Una foto puede no servir, y eso también es una respuesta.** El gate pedía «de
    # qué lado es» a dos fotos de un caso real que no eran tiras vestibulares de esta
    # arcada: una era de la arcada CONTRARIA y la otra un primer plano de una sola pieza,
    # cuyas ocho «coronas» detectadas eran sus cúspides. A ninguna de las dos se le puede
    # contestar de qué lado es, y darle un lado sería peor que no dárselo: repartiría ocho
    # códigos FDI entre las cúspides de un molar y plantaría ese color en el contenedor.
    espejo = {f: (f // 10 % 2 and f + 10 or f - 10) for f in arco}
    mejor: dict[int, TonoPieza] = {}
    motivos: list[str] = []
    lecturas: list[_Lectura] = []
    for foto in fotos:
        if lado_conocido.get(foto) == NO_ES_TIRA:
            motivos.append(
                f"sha256:{_sha256(foto)[:16]}{foto.suffix}: declarada como NO tira "
                "vestibular de esta arcada, asi que no aporta color por pieza. Lo que el "
                "emisor puede medir es que hay coronas en la imagen, no que sean de este "
                "arco ni que sean coronas distintas"
            )
            continue
        rgb, etiquetas, orden = tira_de_coronas(foto, esperadas)
        if len(orden) < 3:
            continue
        alin = alinea_con_el_arco(anchos_aparentes(etiquetas, orden), arco, anchos_arco)
        if alin is None:
            continue
        fdis = alin.fdis
        if alin.ambiguo:
            primera = lado_conocido.get(foto)
            if primera is None:
                motivos.append(
                    f"sha256:{_sha256(foto)[:16]}{foto.suffix}: la tira de "
                    f"{len(orden)} corona(s) encaja igual de bien en {fdis[0]}-{fdis[-1]} "
                    f"que en su espejo {espejo[fdis[0]]}-{espejo[fdis[-1]]}. El arco es "
                    "simetrico y una foto intraoral puede estar tomada con espejo: hace "
                    "falta que una persona diga de que lado es, O que esta foto no es una "
                    "tira vestibular de esta arcada"
                )
                continue
            if primera == espejo[fdis[0]]:
                fdis = [espejo[f] for f in fdis]
            elif primera != fdis[0]:
                motivos.append(
                    f"sha256:{_sha256(foto)[:16]}{foto.suffix}: se declaro que empieza en "
                    f"FDI {primera} y la huella de anchuras da {fdis[0]} o "
                    f"{espejo[fdis[0]]}: no cuadra ninguna"
                )
                continue
        digest = _sha256(foto)
        lineal = _lineal(rgb)
        for fdi, k in zip(fdis, orden, strict=True):
            mascara = etiquetas == k
            lecturas.append(_Lectura(
                digest=digest, rgb=rgb, mascara=mascara, fdi=fdi,
                encia=encia_contigua(mascara, etiquetas, lineal),
            ))

    # ── La caida del flash, estimada sobre TODAS las fotos a la vez ─────────
    #
    # ⚠️ **El ajuste es global y no por foto a proposito.** Cada foto ve media arcada y las
    # dos no comparten ninguna pieza, asi que ajustar una por su cuenta deja libre el nivel
    # de cada una y los dos lados quedan sin comparar entre si. Con una sola referencia de
    # encia para todo el caso, las dos mitades caen en la misma escala.
    candidatas = [c for c in lecturas if c.encia is not None]
    es_mucosa = sondas_de_mucosa(
        [np.median(_lineal(c.rgb[c.mascara]), axis=0) for c in candidatas],
        [c.encia for c in candidatas],  # type: ignore[misc]
    )
    descartadas = sorted({c.fdi for c, ok in zip(candidatas, es_mucosa, strict=True)
                          if not ok})
    validas_id = {id(c) for c, ok in zip(candidatas, es_mucosa, strict=True) if ok}
    con_sonda = [c for c in candidatas if id(c) in validas_id]
    ajuste = ajuste_de_iluminacion(
        [np.median(_lineal(c.rgb[c.mascara]), axis=0) for c in con_sonda],
        [c.encia for c in con_sonda],  # type: ignore[misc]
    )
    if ajuste is None:
        if lecturas:
            motivos.append(
                f"tono sin corregir de iluminacion: solo {len(con_sonda)} corona(s) de "
                f"{len(lecturas)} tienen mucosa contigua que muestrear y hacen falta "
                f"{MINIMO_CORONAS_SONDA}. Los colores declarados llevan dentro lo lejos "
                "que le llego el flash a cada pieza y NO son comparables entre si"
            )
    else:
        ref, beta = ajuste
        motivos.append(
            "tono corregido de iluminacion con la encia del propio paciente como "
            f"referencia ({len(con_sonda)} corona(s) con sonda de mucosa); se descuenta "
            "de cada pieza la parte que su encia contigua predice, con pendiente medida "
            f"{beta[0]:.2f}/{beta[1]:.2f}/{beta[2]:.2f} por canal. Quedan comparables "
            "entre si, pero el NIVEL absoluto sigue sin calibrar: la foto no lleva "
            "referencia gris"
        )
        if descartadas:
            motivos.append(
                "sonda de mucosa descartada por color: FDI "
                + ", ".join(str(f) for f in descartadas)
                + ". Lo que hay por encima de su cuello no tiene color de encia —un "
                "separador, un labio o el fondo de la boca—, asi que no sirve de "
                "referencia de iluminacion y esas piezas se declaran sin corregir"
            )
        sin_sonda = sorted({c.fdi for c in lecturas if c.encia is None} | set(descartadas))
        if sin_sonda:
            motivos.append(
                "sin sonda de mucosa: FDI "
                + ", ".join(str(f) for f in sin_sonda)
                + ". No tienen encia contigua visible en su foto, asi que su tono se "
                "declara SIN corregir y no es comparable con el de las demas piezas"
            )

    # ── Pase 2: el color de cada pieza, ya en la misma escala ───────────────
    for lect in lecturas:
        factor = None
        if ajuste is not None and lect.encia is not None and id(lect) in validas_id:
            ref, beta = ajuste
            factor = (ref / lect.encia) ** beta
        tercios = tono_por_tercios(lect.rgb, lect.mascara, HACIA_CERVICAL_SUPERIOR,
                                   factor=factor, encia=lect.encia)
        if tercios is None:
            continue
        n = int(lect.mascara.sum())
        if lect.fdi not in mejor or n > mejor[lect.fdi].n_pixeles:
            mejor[lect.fdi] = TonoPieza(
                fdi=lect.fdi, lab=tercios, n_pixeles=n, foto_sha256=lect.digest,
                correccion=None if factor is None else (float(beta[0]), float(beta[1]),
                                                        float(beta[2])),
            )
    # ⚠️ **Una pieza sin color medido tiene que DECIRSE.** Quien abre el contenedor ve
    # trece coronas con su color y una sin campo `color`, y no puede distinguir «no se
    # midio» de «se midio y salio gris»: lo unico que hay en el PLY para esa pieza es el
    # degradado de respaldo, que no es color de nadie. El motivo la nombra.
    sin_color = [f for f in arco if f not in mejor]
    if fotos and not mejor:
        # ⚠️ **Ninguna foto aporta color, y eso hay que decirlo una vez y entero.** El aviso
        # de abajo se callaba cuando `mejor` estaba vacio —para no listar catorce piezas
        # cuando no se habia medido ninguna— y el resultado era un contenedor sin una sola
        # corona con color y sin ningun motivo que dijera por que.
        motivos.append(
            f"NINGUNA de las {len(fotos)} foto(s) aporta color por pieza: el contenedor no "
            "declara el color de ninguna corona y el campo gaussiano las pinta con el "
            "degradado de respaldo, que NO es color del paciente. Los motivos de cada foto "
            "van arriba"
        )
    if sin_color and mejor:
        motivos.append(
            "sin color medido: FDI "
            + ", ".join(str(f) for f in sin_color)
            + ". Ninguna de las fotos aportadas la ve con su eje cuello-borde, asi que el "
            "contenedor no declara su color y el campo gaussiano la pinta con el degradado "
            "de respaldo, que NO es color del paciente"
        )
    return [mejor[f] for f in sorted(mejor)], motivos


# La direccion al cuello dentro de la banda superior. **No es una suposicion**: la banda se
# construye cortando por encima de la costura oclusal, asi que el borde incisal es el lado
# que toca la costura —abajo— y el cuello el contrario. Sale de como se construyo la banda,
# no de la imagen.
#
# ⚠️ Se probo antes «la mitad con mas pixeles es la cervical» y acierta por el motivo
# equivocado: una corona es mas ESTRECHA por el cuello que por los contactos, y solo
# funcionaba porque la erosion de la costura recorta el borde incisal. Un acierto que
# depende de otro parametro no es un acierto.
HACIA_CERVICAL_SUPERIOR = (0.0, -1.0)


def tono_por_tercios(rgb: np.ndarray, mascara: np.ndarray,
                     hacia_cervical: tuple[float, float], *,
                     factor: np.ndarray | None = None,
                     encia: np.ndarray | None = None) -> np.ndarray | None:
    """`L*a*b*` por tercio cervical, medio e incisal. `(3, 3)` o `None`.

    **Por tercios y no un color por diente**, porque un diente no es de un color: se
    oscurece hacia el cuello y se vuelve translucido hacia el borde. Es como se toma el
    tono en clinica, y ademas da un degradado natural DENTRO de la pieza en vez de una
    mancha plana — sin necesidad de interpolar entre piezas, que es de donde salia el
    marron embarrado del visor.

    **Mediana y no media**: un pixel de encia colado en el borde del blob no mueve una
    mediana. **Y se tira el percentil `BRILLO_PCT` de `L*`** antes de nada.

    `hacia_cervical` es la direccion en la imagen que va del borde incisal al cuello. Se
    pasa medida y no supuesta.

    `factor` es la correccion de iluminacion de ESTA corona, en luz lineal y por canal (ver
    `ajuste_de_iluminacion`); `encia`, su sonda de mucosa, para rechazar los pixeles que no
    son diente (ver `_sin_mucosa`). Sin ellos la funcion mide lo mismo que antes, que es lo
    que pasa cuando la corona no tiene mucosa contigua que muestrear.
    """
    ys, xs = np.nonzero(mascara)
    if len(ys) < MINIMO_PIXELES:
        return None
    pix = np.asarray(rgb)[ys, xs]
    if factor is not None:
        pix = _srgb(_lineal(pix) * factor)
    lab = _lab(pix)
    sin_brillo = lab[:, 0] < np.percentile(lab[:, 0], BRILLO_PCT)
    if sin_brillo.sum() < MINIMO_POR_TERCIO * 3:
        return None
    lab = lab[sin_brillo]
    proy = (np.stack([xs, ys], axis=1).astype(np.float64)
            @ np.asarray(hacia_cervical, dtype=np.float64))[sin_brillo]
    if encia is not None:
        enc = encia if factor is None else encia * factor
        queda = _sin_mucosa(lab, _lab(_srgb(enc)[None])[0])
        # ⚠️ Tirar media corona no es limpiar un borde: si pasa, la mascara no es esta
        # pieza y el rechazo no arregla nada. Se deja el dato crudo y que lo vea el gate.
        if queda.mean() >= 1.0 - RECHAZO_MAXIMO and queda.sum() >= MINIMO_POR_TERCIO * 3:
            lab, proy = lab[queda], proy[queda]
    cortes = np.quantile(proy, [1 / 3, 2 / 3])
    # ⚠️ **De CERVICAL a INCISAL, y el orden importa mas de lo que parece.** `proy` crece
    # hacia el cuello, asi que la franja de proyeccion ALTA es la cervical: devolverlas en
    # el orden natural de los cortes da la lista al reves. Sin invertir aqui, un caso real
    # leia «a* 3,15 en cervical y 7,02 en incisal» —que contradice la anatomia, porque el
    # borde incisal es mas translucido y menos saturado— cuando el dato decia justo lo
    # contrario y bien. Un color por tercio sin saber que tercio es no vale para nada.
    franjas = [proy >= cortes[1],
               (proy >= cortes[0]) & (proy < cortes[1]),
               proy < cortes[0]]
    if any(f.sum() < MINIMO_POR_TERCIO for f in franjas):
        return None
    return np.stack([np.median(lab[f], axis=0) for f in franjas])


def tira_de_coronas(foto: Path, esperadas: int = 8,
                    margen: int = MARGEN_COSTURA_PX) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """`(rgb, etiquetas, orden)` de la arcada superior de una foto. La tuberia entera.

    Devuelve la tira ORDENADA sin decidir que pieza es cada una: eso lo hace
    `alinea_con_el_arco`, y su resultado puede ser ambiguo. Si la foto **no tiene eje
    cervical-incisal** —una oclusal— devuelve la tira vacia: ver `COSTURA_MAXIMA`.
    """
    rgb, mascara, _ = mascara_diente(foto)
    arco = _arco(mascara)
    lum = np.asarray(rgb, dtype=np.float64).mean(axis=-1)
    costura = costura_oclusal(arco, lum)
    if not costura_es_real(arco, lum, costura):
        return rgb, np.zeros(arco.shape, dtype=int), []
    etiquetas, orden = coronas(banda_superior(arco, costura, margen), esperadas)
    return rgb, etiquetas, orden


def color_de_encia(fotos: list[Path]) -> np.ndarray | None:
    """`L*a*b*` de la mucosa, mediana sobre las fotos con eje cervical-incisal.

    ⚠️ **Tambien medido, y no es un detalle.** El marron embarrado que se ve en el visor
    sobre la encia vestibular no es un color de encia: es interpolacion tirando del vertice
    con color mas cercano, que ahi es una sombra interdental de la oclusal. Con un valor
    medido no hay nada que interpolar.

    ⚠️ **Es UN valor, y se probo a partirlo en dos.** La sospecha era que pintar el paladar
    con el color de la mucosa vestibular lo dejaba demasiado oscuro. Medido, no: paladar
    `L*55,4 a*24,6 b*23,8` contra vestibular `L*52,9 a*23,6 b*24,8`, **ΔE 2,9** — el umbral
    de lo que un ojo distingue. El paladar ES de ese color bajo el flash.

    ⚠️ Y por el camino se cayo la medida que motivaba el reparto. Tomar «lo que queda
    dentro de la herradura» como los huecos que el rellenado deja **no da el paladar**: la
    herradura esta abierta por detras, asi que el paladar conecta con el exterior y nunca
    queda encerrado. Sobre la oclusal de un caso real eso devolvia 65 fragmentos de 1.460
    px —huecos interproximales— y un ΔE de 15,4 que no era de ninguna mucosa. El paladar de
    verdad sale de la envolvente CONVEXA del arco menos los dientes: 464.000 px, 1,6 veces
    el area del propio arco.
    """
    from scipy import ndimage

    muestras = []
    for foto in fotos:
        try:
            rgb, mascara, _ = mascara_diente(foto)
        except Exception:  # noqa: BLE001 - una radiografia o un fichero que no abre
            continue
        arco = _arco(mascara)
        lum = np.asarray(rgb, dtype=np.float64).mean(axis=-1)
        if not arco.any() or not costura_es_real(arco, lum, costura_oclusal(arco, lum)):
            continue
        # La mucosa esta pegada al arco pero fuera de la mascara de diente.
        cerca = ndimage.binary_dilation(arco, np.ones((25, 25))) & ~mascara
        if cerca.sum() < MINIMO_PIXELES:
            continue
        lab = _lab(np.asarray(rgb)[cerca])
        lab = lab[lab[:, 0] < np.percentile(lab[:, 0], BRILLO_PCT)]
        if len(lab) >= MINIMO_PIXELES:
            muestras.append(np.median(lab, axis=0))
    return np.median(np.stack(muestras), axis=0) if muestras else None


def pinta_malla(posiciones: np.ndarray, etiquetas: np.ndarray,
                tonos: list[TonoPieza], encia: np.ndarray | None,
                eje_oclusal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(rgb por vertice, mascara de lo que quedo medido)`.

    Cada vertice recibe el color de SU pieza, interpolado entre los tres tercios a lo
    largo del eje oclusal — un degradado **dentro** de la corona, guiado por la anatomia.

    ⚠️ Esto sustituye a interpolar entre vertices vecinos, que es de donde salia el color
    raro: un vertice sin medida heredaba el del vertice medido mas cercano, y en la cara
    vestibular ese vecino es una sombra interdental de la foto oclusal. Aqui un vertice sin
    pieza reconocida no hereda de un vecino cualquiera: se queda sin declarar, y quien
    pinta decide que hacer con el.

    ⚠️ **Y el color de una pieza no se pinta mas alla de la altura de corona de tabla.**
    Esto no es cosmetica: la segmentacion etiqueta como diente el **70,2 %** de la malla
    cuando un experto etiqueta el 53,9 %, y en el caso real **11.800 vertices** con codigo
    FDI caen por debajo del arranque de las coronas. Sin cota pasaban dos cosas a la vez,
    las dos malas:

    - encia y paladar se pintaban con el tono de la pieza que los reclamo, y el modelo salia
      como un mosaico de colores — cada parche del color del diente que se lo llevo;
    - y los tres tercios se repartian sobre esa altura falsa, asi que el degradado
      **dentro** de la corona tambien salia mal: el «cervical» de un 27 que baja veinte
      milimetros por el paladar se estiraba sobre trece milimetros de encia.

    Lo que queda por debajo de la cota se pinta con el color de ENCIA, que tambien es
    medido. No se inventa nada: la cota solo dice donde NO se puede afirmar color de corona.
    """
    pos = np.asarray(posiciones, dtype=np.float64)
    etq = np.asarray(etiquetas)
    eje = np.asarray(eje_oclusal, dtype=np.float64)
    eje = eje / max(float(np.linalg.norm(eje)), 1e-12)
    rgb = np.zeros((len(pos), 3), np.uint8)
    medido = np.zeros(len(pos), bool)

    if encia is not None:
        rgb[etq == 0] = rgb_de_lab(np.asarray(encia))
        medido[etq == 0] = True

    from analysis_agents.dental import altura_admitida

    for tono in tonos:
        suyos = etq == tono.fdi
        if not suyos.any():
            continue
        idx = np.flatnonzero(suyos)
        altura = pos[idx] @ eje
        hi = float(altura.max())
        cota = altura_admitida(tono.fdi)
        suelo = float(altura.min()) if cota is None else max(
            float(altura.min()), hi - MARGEN_ALTURA * cota
        )
        corona = altura >= suelo
        if not corona.any():
            continue
        # Lo que la etiqueta reclama por debajo de la cota NO puede ser corona. Se queda
        # con el color de encia, que es medido, en vez de con el de la pieza.
        if encia is not None and not corona.all():
            rgb[idx[~corona]] = rgb_de_lab(np.asarray(encia))
            medido[idx[~corona]] = True

        idx, altura = idx[corona], altura[corona]
        lo = float(altura.min())
        # 0 en el cuello, 1 en el borde: el eje oclusal apunta a oclusal, o sea al borde.
        t = np.zeros_like(altura) if hi - lo < 1e-9 else (altura - lo) / (hi - lo)
        canales = [np.interp(t, [0.0, 0.5, 1.0], tono.lab[:, c]) for c in range(3)]
        lab = np.stack(canales, axis=1)
        rgb[idx] = np.stack([rgb_de_lab(v) for v in lab])
        medido[idx] = True
    return rgb, medido
