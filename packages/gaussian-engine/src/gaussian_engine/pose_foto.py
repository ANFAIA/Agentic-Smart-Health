"""Color MEDIDO sobre la malla: pose de camara por PnP y proyeccion de los pixeles.

## Que problema resuelve y por que no lo resolvia lo anterior

`_colorea_malla` pintaba con DOS tonos —la mediana del esmalte y la de la encia sobre las
fotos— y ponia la frontera donde decian las etiquetas FDI. Eso no es color medido: es un
binario, y ademas hereda el error de la segmentacion (medido: 10 de 15 piezas se pasan de
1,30x su altura de corona, y el 26 llega a 2,98x, asi que se pintaba encia de color diente).

Aqui cada vertice recibe **el pixel que la camara ve en el**. Para eso hace falta la pose,
y la pose es justo lo que este proyecto tenia medido que no sabia sacar: COLMAP da **cero
pares geometricos** entre las fotos clinicas, porque una serie clinica esta hecha para
CUBRIR y no para solapar, y las superficies dentales son lisas.

## Como se saca la pose sin solape

No con SfM: con **PnP sobre correspondencias 2D-3D por diente**, que es la Etapa 1 de
DentalGS (Dai et al., AAAI-26, en la base de conocimiento). Catorce correspondencias
sobredeterminan seis grados de libertad y se resuelven en forma cerrada.

- **2D**: umbral de Otsu sobre `a*` de CIELAB —la frontera diente/encia SI esta en el
  color —la senal existe, aunque los 3,4-4,3 sigma de esa ficha midan la bimodalidad de
  `a*` y no el acuerdo con la anatomia: ver la correccion de
  `docs/research/frontera-encia-desde-foto.md`— y watershed
  sobre la transformada de distancia para partir el arco en dientes.
- **3D**: el centroide del TERCIO OCLUSAL de cada pieza etiquetada. ⚠️ No el de la pieza
  entera: las etiquetas invaden la encia, asi que ese centroide esta desplazado.
- **Identidad**: cada blob se casa con la pieza que MAS SE LE SOLAPA al reproyectar
  (emparejamiento humgaro), no por su sitio en el orden del arco — los molares de los
  extremos salen cortados por el encuadre y la biyeccion por orden se desplaza.

Medido sobre un caso real: **0,49 mm** de error de reproyeccion (RANSAC, 9 de 14 inliers),
y el 97,9 % de la corona reproyectada cae sobre la mascara de diente de la foto. DentalGS
declara 1,26 mm con cinco vistas y cuatro etapas.

## Lo que este modulo NO hace, y esta medido que no

**No mejora la segmentacion pieza por pieza.** El color es binario —esmalte contra encia—
y el 26 y el 27 son del mismo color. Se intento derivar el FDI de la particion de la foto y
sale PEOR que la etiqueta de la malla a igualdad de cobertura (5/13 piezas fuera de rango
frente a 2/13). La razon es estructural: la frontera diente/diente es de PROFUNDIDAD, y
retroproyectar una linea 2D sobre la superficie es degenerado justo donde la superficie va
paralela al rayo — que en un contacto interproximal visto desde oclusal es siempre.

**No fusiona vistas de cualquier calidad.** Medido: metiendo una lateral con 1,15 mm de
error, la aportacion del color cae de +3 piezas a +0. Mas vistas no es mejor; vistas BUENAS
es mejor. Por eso `ERROR_MAXIMO_MM` descarta en vez de promediar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: El orden ANATOMICO de cada arcada, de un extremo a otro.
#:
#: ⚠️ **`sorted()` no vale, y era un fallo real.** Ordenar los codigos da
#: `11, 12, ... 17, 21, ... 27`, que empieza por el incisivo central, se va hacia el molar
#: y salta al otro cuadrante: no es el recorrido del arco. Con ese orden, emparejar el
#: blob i con la pieza i asigna cada diente de la foto a uno que no le toca, y la pose sale
#: peor sin que nada falle — de 0,49 mm a 0,67, y la lateral que si tenia pose dejo de
#: tenerla.
ARCO_MAXILAR = (18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28)
ARCO_MANDIBULAR = (48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38)

#: Fraccion oclusal de cada pieza que da su centroide 3D. Ver el docstring del modulo.
TERCIO = 0.33

#: Coseno minimo entre la normal del vertice y la direccion a la camara. ~75 grados: mas
#: rasante que eso, un pixel cubre demasiados milimetros y su color no dice nada del punto.
COSENO_MINIMO = 0.25

#: Error de reproyeccion por encima del cual una vista se DESCARTA en vez de usarse.
#:
#: ⚠️ Medido, y por eso es un descarte y no un peso: con una vista de 1,15 mm en la mezcla,
#: la aportacion del color a la pantalla anatomica pasa de +3 piezas a +0. Una vista mala no
#: aporta poco — estropea lo que las buenas ya habian resuelto.
ERROR_MAXIMO_MM = 0.9

#: Radio dentro del cual un vertice sin color hereda el del medido mas cercano.
INTERPOLA_MM = 1.0


@dataclass(frozen=True)
class PoseFoto:
    """La pose de una foto respecto de la malla, con su calidad delante."""

    ruta: Path
    rvec: np.ndarray
    tvec: np.ndarray
    focal_px: float
    ancho: int
    alto: int
    error_px: float
    error_mm: float
    inliers: int
    correspondencias: int
    umbral_a: float
    apoyo: float
    """Fraccion de la corona reproyectada que cae sobre la mascara de diente de la foto.

    ⚠️ **Es lo que decide si una pose vale, y no el error de reproyeccion.** Con cuatro
    correspondencias el PnP ajusta cualquier cosa: salio una pose con 1,3 px de error que
    solo veia el 10,8 % de la malla. El apoyo usa TODOS los pixeles de la corona.
    """


def es_radiografia(ruta: Path) -> bool:
    """Si la imagen es en escala de grises. Las periapicales no traen color que proyectar.

    ⚠️ Existe porque las «fotos intraorales» de un caso real no son todas fotos: de nueve,
    **tres eran radiografias** y una era de la arcada contraria. `_muestra_color_fotos` las
    triplicaba a RGB y las metia en la mediana del esmalte — medido, las desvia 6-7 unidades
    RGB. Poco, pero es ruido que no tiene por que estar.
    """
    from PIL import Image

    im = Image.open(ruta)
    a = np.asarray(im.convert("RGB"))
    return bool(np.allclose(a[..., 0], a[..., 1], atol=6)
                and np.allclose(a[..., 1], a[..., 2], atol=6))


def _lab_a(rgb: np.ndarray) -> np.ndarray:
    """La componente `a*` de CIELAB: el eje verde-rojo, donde vive la frontera."""
    m = rgb.astype(np.float64) / 255.0
    m = np.where(m <= 0.04045, m / 12.92, ((m + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 500 * (f[..., 0] - f[..., 1])


def _otsu(v: np.ndarray) -> float:
    h, e = np.histogram(v, bins=256)
    p = h / h.sum()
    c = np.cumsum(p)
    m = np.cumsum(p * ((e[:-1] + e[1:]) / 2))
    sb = (m[-1] * c - m) ** 2 / np.maximum(c * (1 - c), 1e-12)
    k = int(np.nanargmax(sb))
    return float((e[k] + e[k + 1]) / 2)


def mascara_diente(ruta: Path, lado: int = 1400) -> tuple[np.ndarray, np.ndarray, float]:
    """`(rgb, mascara, umbral)`. Diente = poco rojo Y claro.

    Solo `a*` deja pasar el metal de los separadores y las sombras oscuras del fondo, asi
    que se cruza con un suelo de luminancia.
    """
    from PIL import Image

    im = Image.open(ruta).convert("RGB")
    im.thumbnail((lado, lado))
    rgb = np.asarray(im)
    a = _lab_a(rgb)
    u = _otsu(a)
    lum = rgb.astype(np.float64).mean(-1)
    from scipy import ndimage

    m = (a < u) & (lum > np.percentile(lum, 40))
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    return rgb, m, u


def normales(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Normal por vertice, promediando las de sus caras."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    nf = np.cross(b - a, c - a)
    n = np.zeros_like(V)
    np.add.at(n, F.ravel(), np.repeat(nf, 3, axis=0))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def centros_oclusales(V: np.ndarray, etiquetas: np.ndarray,
                      codigos: list[int]) -> np.ndarray:
    """Centroide del tercio oclusal de cada pieza. Ver el docstring del modulo."""
    eje = V[etiquetas > 0].mean(0) - V[etiquetas == 0].mean(0)
    eje /= np.linalg.norm(eje)
    out = []
    for f in codigos:
        P = V[etiquetas == f]
        h = P @ eje
        out.append(P[h >= np.quantile(h, 1 - TERCIO)].mean(0))
    return np.asarray(out)


def _arco(m: np.ndarray) -> np.ndarray:
    """La componente del arco, sin separadores.

    Los separadores de carrillo son plastico palido y pasan el umbral de `a*`; viven en el
    borde del encuadre por construccion, asi que se anula un margen antes de etiquetar. No
    vale descartar «lo que toca el borde»: en una lateral los incisivos salen cortados.
    """
    from scipy import ndimage

    borde = max(int(0.035 * min(m.shape)), 1)
    n = m.copy()
    n[:borde] = n[-borde:] = False
    n[:, :borde] = n[:, -borde:] = False
    lab, k = ndimage.label(n)
    if k == 0:
        return n
    tam = ndimage.sum(n, lab, range(1, k + 1))
    return lab == (int(np.argmax(tam)) + 1)


def _parte_en_piezas(arco: np.ndarray, cuantas: int) -> tuple[np.ndarray, np.ndarray]:
    """Watershed sobre la transformada de distancia. `(etiquetas, ids)`.

    ⚠️ **Erosionar no vale, y esta medido.** Los contactos de los molares son anchos y los
    incisivos desaparecen antes de que aquellos se separen: salieron 10 piezas con areas de
    34.496 y de 782. El watershed parte por los cuellos estrechos sin encoger las piezas,
    que es lo que hace falta cuando son de tamanos muy distintos. Con el: 14 blobs limpios.
    """
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    dist = ndimage.distance_transform_edt(arco)
    mejor = None
    for sep in range(60, 6, -2):
        picos = peak_local_max(dist, min_distance=sep, labels=arco, exclude_border=False)
        if len(picos) < 2:
            continue
        marc = np.zeros(arco.shape, int)
        marc[tuple(picos.T)] = np.arange(1, len(picos) + 1)
        marc, _ = ndimage.label(marc > 0)
        lab = watershed(-dist, marc, mask=arco)
        ids = np.unique(lab)
        ids = ids[ids > 0]
        tam = np.array([(lab == k).sum() for k in ids])
        vivos = ids[tam > max(tam.max() * 0.03, 200)]
        if mejor is None or abs(len(vivos) - cuantas) < abs(len(mejor[1]) - cuantas):
            mejor = (lab, vivos)
        if len(vivos) == cuantas:
            return lab, vivos
    return mejor if mejor else (np.zeros(arco.shape, int), np.array([], int))


def estima_pose(ruta: Path, V: np.ndarray, etiquetas: np.ndarray) -> PoseFoto | None:
    """La pose de esa foto respecto de la malla, o `None` si no se puede sostener.

    Devuelve `None` en vez de una pose mala **a proposito**: una vista con la pose torcida
    no aporta menos color, estropea el que las buenas ya habian puesto (medido: de +3
    piezas a +0). Quien llama decide con `apoyo` y `error_mm` delante.
    """
    import cv2
    from scipy import ndimage

    # ⚠️ **Una esquirla no es un diente.** El umbral es RELATIVO a la propia arcada, no un
    # numero absoluto: en un caso real el FDI 28 traia 275 vertices frente a los 1.200-9.500
    # de las piezas de verdad, y colarlo obligaba al watershed a partir la foto en una pieza
    # mas de las que hay. La consecuencia no es un aviso: es que todo el emparejamiento se
    # descoloca y la pose empeora en silencio.
    cuenta = {int(c): int((etiquetas == c).sum()) for c in np.unique(etiquetas) if c > 0}
    if not cuenta:
        return None
    suelo = max(200, 0.15 * float(np.median(list(cuenta.values()))))
    presentes = {c for c, n in cuenta.items() if n >= suelo}
    arco_ref = (ARCO_MAXILAR if sum(c in presentes for c in ARCO_MAXILAR)
                >= sum(c in presentes for c in ARCO_MANDIBULAR) else ARCO_MANDIBULAR)
    codigos = [c for c in arco_ref if c in presentes]
    if len(codigos) < 6:
        return None

    rgb, m, u = mascara_diente(ruta)
    arco = _arco(m)
    alto, ancho = arco.shape
    lab, ids = _parte_en_piezas(arco, len(codigos))
    if len(ids) < 6:
        return None
    xy = np.array(ndimage.center_of_mass(np.ones(arco.shape), lab, list(ids)))[:, ::-1]
    p3_todos = centros_oclusales(V, etiquetas, codigos)

    # Corona de referencia para el apoyo: el tercio oclusal de todas las piezas.
    eje = V[etiquetas > 0].mean(0) - V[etiquetas == 0].mean(0)
    eje /= np.linalg.norm(eje)
    corona = np.vstack([
        (lambda P: P[(P @ eje) >= np.quantile(P @ eje, 1 - TERCIO)])(V[etiquetas == c])
        for c in codigos
    ])

    def apoyo_de(rv, tv, K):
        R, _ = cv2.Rodrigues(rv)
        Vc = (R @ corona.T).T + tv.ravel()
        d = Vc[:, 2] > 0
        if d.sum() < 500:
            return 0.0
        uv = (K @ (Vc[d] / Vc[d][:, 2:3]).T).T[:, :2]
        px = uv[:, 0].astype(int)
        py = uv[:, 1].astype(int)
        ok = (px >= 0) & (px < ancho) & (py >= 0) & (py < alto)
        if ok.sum() < 500:
            return 0.0
        return float(arco[py[ok], px[ok]].mean())

    # ⚠️ **Los blobs y las piezas tienen que EMPAREJARSE, no truncarse.** La primera version
    # cortaba la lista mas larga con `[:n]`, que empareja el blob i con la pieza i sin que
    # nadie haya comprobado que se corresponden: con un molar cortado por el encuadre, todo
    # se desplaza una posicion en silencio. Aqui se prueban todas las ventanas CONTIGUAS del
    # arco, que es lo que de verdad puede ver una foto: un tramo seguido de dientes.
    xy_ord = xy[_ordena_arco(xy)]
    k = len(xy_ord)
    ventanas = ([(0, len(codigos))] if k == len(codigos)
                else [(i, i + k) for i in range(len(codigos) - k + 1)] if k < len(codigos)
                else [])
    if not ventanas:
        return None

    mejor = None
    for f in np.linspace(0.6 * ancho, 6.0 * ancho, 60):
        K = np.array([[f, 0, ancho / 2], [0, f, alto / 2], [0, 0, 1]], float)
        for i0, i1 in ventanas:
            base = p3_todos[i0:i1]
            # El sentido del arco depende de si la foto se tomo con espejo, y eso no viaja
            # en ningun sitio: se prueban los dos y decide el apoyo.
            for sentido in (1, -1):
                p3 = np.ascontiguousarray(base[::sentido])
                p2 = np.ascontiguousarray(xy_ord, dtype=np.float64)
                ok, rv, tv, inl = cv2.solvePnPRansac(
                    p3, p2, K, None, reprojectionError=25.0,
                    iterationsCount=400, flags=cv2.SOLVEPNP_EPNP)
                if not ok or inl is None or len(inl) < 6:
                    continue
                i = inl.ravel()
                ok, rv, tv = cv2.solvePnP(
                    np.ascontiguousarray(p3[i]), np.ascontiguousarray(p2[i]), K, None,
                    rvec=rv, tvec=tv, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                pr, _ = cv2.projectPoints(np.ascontiguousarray(p3[i]), rv, tv, K, None)
                e = float(np.linalg.norm(pr.reshape(-1, 2) - p2[i], axis=1).mean())
                sop = apoyo_de(rv, tv, K)
                punt = (round(sop, 3), len(i), -round(e, 1))
                if mejor is None or punt > mejor[0]:
                    mejor = (punt, e, f, rv, tv, len(i), len(p3), sop, p3, p2)
    if mejor is None:
        return None
    _, e, f, rv, tv, ninl, ncorr, sop, p3, p2 = mejor
    mm_px = (np.linalg.norm(p3.max(0) - p3.min(0))
             / max(np.linalg.norm(p2.max(0) - p2.min(0)), 1e-9))
    return PoseFoto(ruta=ruta, rvec=rv, tvec=tv, focal_px=float(f), ancho=ancho, alto=alto,
                    error_px=e, error_mm=e * mm_px, inliers=ninl, correspondencias=ncorr,
                    umbral_a=u, apoyo=sop)


def _ordena_arco(xy: np.ndarray) -> np.ndarray:
    """Orden a lo largo del arco. Por ANGULO, no por `x`: la curva se dobla en los molares
    y alli dos piezas comparten `x`."""
    c = xy.mean(0).copy()
    c[1] += (xy[:, 1].max() - xy[:, 1].min()) * 1.2
    return np.argsort(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))


@dataclass(frozen=True)
class ColorMedido:
    """Color por vertice y **de donde salio cada uno**. Lo segundo es la mitad importante.

    Un color que no dice su procedencia se lee como medido aunque no lo sea, que es
    exactamente el fallo que este modulo existe para arreglar: la version anterior pintaba
    dos tonos interpolados por altura y el contenedor los declaraba como «color real».
    """

    rgb: np.ndarray
    """(N, 3) uint8."""
    medido: np.ndarray
    """(N,) bool — el pixel que una camara ve de verdad en ese vertice."""
    interpolado: np.ndarray
    """(N,) bool — heredado del vertice medido mas cercano, a menos de `INTERPOLA_MM`."""
    poses: list[PoseFoto]
    descartadas: list[tuple[Path, str]]

    @property
    def cobertura(self) -> float:
        return float(self.medido.mean())

    def resumen(self) -> str:
        n = len(self.rgb)
        sin = n - int(self.medido.sum()) - int(self.interpolado.sum())
        return (f"{int(self.medido.sum()):,} medido(s) ({100*self.medido.mean():.1f} %) · "
                f"{int(self.interpolado.sum()):,} interpolado(s) · {sin:,} sin color")


def color_por_vertice(
    fotos: list[Path],
    V: np.ndarray,
    F: np.ndarray,
    etiquetas: np.ndarray,
    *,
    respaldo_rgb: np.ndarray | None = None,
    traza: bool = False,
) -> ColorMedido:
    """El color de cada vertice, tomado de las fotos que tengan pose sostenible.

    ⚠️ **Cada vertice se queda con la vista que lo ve MAS DE FRENTE.** Una cara vestibular
    vista de refilon desde una oclusal recibe un pixel estirado —unos pocos pixeles cubren
    muchos milimetros— y ese color es peor que el de la lateral, que la ve de cara. El
    criterio es el coseno entre la normal y la direccion a la camara.

    ⚠️ **Y se descarta una vista entera si su pose no se sostiene**, en vez de promediarla.
    Medido: una lateral con 1,15 mm de error anula la aportacion del color de las buenas.

    `respaldo_rgb` es lo que se pinta donde no llega ninguna camara ni la interpolacion —
    tipicamente los dos tonos de antes. Va declarado como lo que es: no medido.
    """
    import cv2
    from scipy.spatial import cKDTree

    N = normales(V, F)
    rgb_out = np.zeros((len(V), 3), np.uint8)
    if respaldo_rgb is not None:
        rgb_out[:] = np.asarray(respaldo_rgb, np.uint8)
    mejorcos = np.full(len(V), -1.0)
    medido = np.zeros(len(V), bool)
    poses: list[PoseFoto] = []
    descartadas: list[tuple[Path, str]] = []

    for ruta in fotos:
        if es_radiografia(ruta):
            descartadas.append((ruta, "es una radiografia: no trae color que proyectar"))
            continue
        pose = estima_pose(ruta, V, etiquetas)
        if pose is None:
            descartadas.append((ruta, "no se ha podido resolver una pose"))
            continue
        if pose.error_mm > ERROR_MAXIMO_MM:
            descartadas.append(
                (ruta, f"pose de {pose.error_mm:.2f} mm, por encima de "
                       f"{ERROR_MAXIMO_MM} mm: una vista torcida estropea a las buenas"))
            continue
        rgb, _m, _u = mascara_diente(ruta)
        K = np.array([[pose.focal_px, 0, pose.ancho / 2],
                      [0, pose.focal_px, pose.alto / 2], [0, 0, 1]], float)
        R, _ = cv2.Rodrigues(pose.rvec)
        Vc = (R @ V.T).T + pose.tvec.ravel()
        uv = (K @ (Vc / np.maximum(Vc[:, 2:3], 1e-9)).T).T[:, :2]
        px = uv[:, 0].astype(int).clip(0, pose.ancho - 1)
        py = uv[:, 1].astype(int).clip(0, pose.alto - 1)
        dentro = ((uv[:, 0] >= 0) & (uv[:, 0] < pose.ancho)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < pose.alto) & (Vc[:, 2] > 0))
        # ⚠️ Z-buffer: sin el, el paladar recibe el color del diente que tiene delante.
        zb = np.full((pose.alto, pose.ancho), np.inf)
        np.minimum.at(zb, (py[dentro], px[dentro]), Vc[dentro, 2])
        centro = (-R.T @ pose.tvec.reshape(3, 1)).ravel()
        dirs = centro - V
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        # El STL no garantiza orientacion consistente de las caras: se usa |cos|.
        cos = np.abs((N * dirs).sum(1))
        vis = dentro & (Vc[:, 2] <= zb[py, px] * 1.02) & (cos > COSENO_MINIMO)
        gana = vis & (cos > mejorcos)
        mejorcos[gana] = cos[gana]
        rgb_out[gana] = rgb[py[gana], px[gana]]
        medido |= gana
        poses.append(pose)
        if traza:
            print(f"    {ruta.name}: pose {pose.error_mm:.2f} mm · apoyo "
                  f"{100*pose.apoyo:.0f} % · aporta {int(gana.sum()):,} vertice(s)")

    interpolado = np.zeros(len(V), bool)
    if medido.any() and (~medido).any():
        arbol = cKDTree(V[medido])
        d, i = arbol.query(V[~medido])
        cerca = d <= INTERPOLA_MM
        idx = np.where(~medido)[0][cerca]
        rgb_out[idx] = rgb_out[np.where(medido)[0][i[cerca]]]
        interpolado[idx] = True

    for ruta, razon in descartadas:
        if traza:
            print(f"    ✗ {ruta.name}: {razon}")
    return ColorMedido(rgb=rgb_out, medido=medido, interpolado=interpolado,
                       poses=poses, descartadas=descartadas)
