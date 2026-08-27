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
    espejo = {f: (f // 10 % 2 and f + 10 or f - 10) for f in arco}
    mejor: dict[int, TonoPieza] = {}
    motivos: list[str] = []
    for foto in fotos:
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
                    "falta que una persona diga de que lado es"
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
        for fdi, k in zip(fdis, orden, strict=True):
            mascara = etiquetas == k
            tercios = tono_por_tercios(rgb, mascara, HACIA_CERVICAL_SUPERIOR)
            if tercios is None:
                continue
            n = int(mascara.sum())
            if fdi not in mejor or n > mejor[fdi].n_pixeles:
                mejor[fdi] = TonoPieza(fdi=fdi, lab=tercios, n_pixeles=n,
                                       foto_sha256=digest)
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
                     hacia_cervical: tuple[float, float]) -> np.ndarray | None:
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
    """
    ys, xs = np.nonzero(mascara)
    if len(ys) < MINIMO_PIXELES:
        return None
    lab = _lab(np.asarray(rgb)[ys, xs])
    sin_brillo = lab[:, 0] < np.percentile(lab[:, 0], BRILLO_PCT)
    if sin_brillo.sum() < MINIMO_POR_TERCIO * 3:
        return None
    lab = lab[sin_brillo]
    proy = (np.stack([xs, ys], axis=1).astype(np.float64)
            @ np.asarray(hacia_cervical, dtype=np.float64))[sin_brillo]
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
    """`L*a*b*` de la encia, mediana sobre las fotos que tienen eje cervical-incisal.

    ⚠️ **Tambien medido, y no es un detalle.** El marron embarrado que se ve en el visor
    sobre la encia vestibular no es un color de encia: es interpolacion tirando del vertice
    con color mas cercano, que ahi es una sombra interdental de la oclusal. Con un valor
    medido no hay nada que interpolar.

    Se toma de la region del arco que NO es diente y descartando el percentil alto de `L*`
    —los brillos sobre mucosa mojada—.
    """
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
        # La encia esta pegada al arco pero fuera de la mascara de diente.
        from scipy import ndimage

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
    """
    pos = np.asarray(posiciones, dtype=np.float64)
    etq = np.asarray(etiquetas)
    eje = np.asarray(eje_oclusal, dtype=np.float64)
    eje = eje / max(float(np.linalg.norm(eje)), 1e-12)
    rgb = np.zeros((len(pos), 3), np.uint8)
    medido = np.zeros(len(pos), bool)

    if encia is not None:
        rgb[etq == 0] = rgb_de_lab(encia)
        medido[etq == 0] = True

    for tono in tonos:
        suyos = etq == tono.fdi
        if not suyos.any():
            continue
        altura = pos[suyos] @ eje
        lo, hi = float(altura.min()), float(altura.max())
        # 0 en el cuello, 1 en el borde: el eje oclusal apunta a oclusal, o sea al borde.
        t = np.zeros_like(altura) if hi - lo < 1e-9 else (altura - lo) / (hi - lo)
        canales = [np.interp(t, [0.0, 0.5, 1.0], tono.lab[:, c]) for c in range(3)]
        lab = np.stack(canales, axis=1)
        rgb[suyos] = np.stack([rgb_de_lab(v) for v in lab])
        medido[suyos] = True
    return rgb, medido
