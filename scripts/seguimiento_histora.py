#!/usr/bin/env python
"""seguimiento_histora.py — cuánto se ha movido el margen gingival entre dos escaneos.

    uv run python scripts/seguimiento_histora.py --antes A.stl --despues B.stl \
        --salida DIR [--ply campo.ply]

**El nombre dice seguimiento y no recesión, a propósito.** La recesión *absoluta* se
define contra la LAC, y está medido aquí que la LAC no sale de ningún dato nuestro:
del CBCT ni por densidad (metal, y el esmalte cervical es más fino que el vóxel) ni por
forma (el punto se mueve 6,6 mm según el umbral: se midió y se descartó esa vía); y
del escáner tampoco, porque el escalón queda por debajo del ruido de triangulación de
la malla — se comprobó sobre una arcada **con** recesión visible, así que no es que no
la hubiera.

Lo que sí se detecta con solvencia es el **margen gingival**: 11 de 11 secciones en las
tres arcadas probadas. Así que esto mide lo que se puede medir bien — **cuánto se ha
movido ese margen entre dos momentos** — que además es lo que un gemelo digital existe
para dar: no una foto, una trayectoria.

## Por qué el registro aquí sí es el caso fácil

Nada que ver con IOS↔CBCT. Misma modalidad, misma geometría, y medido sobre `histora`:
**91 % de solape** entre el escaneo previo y el posterior a la higiene, con 0,325 mm de
residuo. El discriminante es el solape, no el rms — los rms se parecen entre pares
verdaderos y falsos porque el ICP recortado siempre encuentra algo cerca.

⚠️ **Se registra solo sobre las coronas.** Es la decisión que hace que la medida
signifique algo: la corona no se mueve entre dos escaneos y la encía sí. Registrar con
todo dentro haría que el tejido que cambió arrastre el ajuste, y la señal se comería a
sí misma. De ahí que la segmentación corra **antes** que el registro y no después.

## Los agentes que usa

- `mesh-agent` — ingesta de las dos mallas, con su guardarraíl de reversibilidad.
- `registration` del `geometric-fusion-agent` — el registro entre los dos momentos.
- Con `--cbct`, además el pipeline completo del orquestador sobre el momento que
  tenga volumen: `cbct-agent` + `mesh-agent` → `TwinSnapshot` → `segmentation-agent`.

⚠️ **Y aquí hay un límite de arquitectura que este caso destapa.** El contrato exige
`gaussian_field_ref`, que **sale del CBCT**; la malla intraoral va en `surface_ref`.
Así que un caso *solo de escáner* no produce `TwinSnapshot` por diseño —está razonado
en `_assemble`— y el `segmentation-agent`, que trabaja sobre `gaussian_field_ref`,
**no puede segmentar la malla del escáner**. No es un fallo: es que el seguimiento
longitudinal por escáner no estaba entre los casos que el contrato preveía.

Consecuencia práctica: la separación encía/diente que este script necesita para
registrar solo sobre las coronas se hace aquí, con `SegmentadorGingival`, y no a
través del agente. Cuando el contrato admita un twin de superficie, ese segmentador
se le pasa al agente sin tocarlo — cumple su protocolo `Segmenter` tal cual.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESUMEN_EN = "Measures gingival margin movement between two scans."

RAIZ = Path(__file__).resolve().parent.parent
for paquete in ("ingestion-agents", "fusion-agents", "analysis-agents", "core-schemas"):
    sys.path.insert(0, str(RAIZ / f"packages/{paquete}/src"))
sys.path.insert(0, str(RAIZ / "apps/agent-orchestrator/src"))

from agent_orchestrator.pipeline import CaseInput, IngestionPipeline  # noqa: E402
from fusion_agents.registration import apply, icp, quaternion_to_matrix  # noqa: E402
from ingestion_agents import ArtifactStore, MeshAgent  # noqa: E402
from ingestion_agents.mesh_agent import vertex_normals  # noqa: E402

_spec = importlib.util.spec_from_file_location("ac", RAIZ / "scripts/altura_corona.py")
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

VENTANA_MARGEN_MM = (2.5, 16.0)
BORDE_LIBRE_MM = 3.0  # el recorte de la malla tiene curvatura enorme: se deja fuera
CORTES = 21  # posiciones a lo largo del arco
# El margen puede subir o bajar unos milímetros, pero NO se desplaza de lado. Si los
# dos puntos emparejados distan más que esto en el plano oclusal, el detector los
# encontró en sitios distintos —una vez salieron a 50 mm, o sea en caras opuestas— y
# la resta no significa nada. Se descarta el corte en vez de publicar el número.
DERIVA_LATERAL_MAX_MM = 2.0


@dataclass(frozen=True)
class Margen:
    """El margen gingival hallado en una sección: dónde está y a qué altura."""

    t: float  # posición a lo largo del arco, 0..1
    punto: np.ndarray  # (3,) en el marco del arco
    altura: float  # coordenada axial, mm


# --------------------------------------------------------------------------- #
# Detección del margen — la pieza que sí funciona
# --------------------------------------------------------------------------- #
def margenes(
    V: np.ndarray, F: np.ndarray, plano: dict | None = None, cortes: int = CORTES
) -> tuple[list[Margen], float, dict]:
    """El margen gingival a lo largo del arco, sobre un **plano de corte dado o propio**.

    Compartir el plano no es un detalle, y costó dos errores seguidos. El primero:
    dos PCA independientes dan ejes distintos, y comparar alturas entre ellos mezcla
    el cambio real con el desalineamiento. El segundo, más sutil: aun con los ejes
    compartidos, recalcular la curva del arco y el rango de `x` en cada malla hace
    que `t = 0,30` **no caiga en el mismo sitio** — se vio porque los dos extremos de
    una cota distaban 5 mm en dirección vestibular cuando lo medido era la altura.

    Así que se comparte todo: ejes, curva y abscisas. Los dos momentos se cortan por
    exactamente los mismos planos.
    """
    if plano is None:
        c, ejes, P, razon = ac.marco_arcada(V)
        coef, centro = ac.curva_arco(P)
        plano = {"centro_mundo": c, "ejes": ejes, "coef": coef, "centro_arco": centro,
                 "x0": float(np.percentile(P[:, 0], 4)),
                 "x1": float(np.percentile(P[:, 0], 96))}
    else:
        c, ejes, razon = plano["centro_mundo"], plano["ejes"], float("nan")
        P = (V - c) @ ejes.T
    coef, centro = plano["coef"], plano["centro_arco"]
    poly = ac.a_polydata(P, F)
    x0, x1 = plano["x0"], plano["x1"]

    fuera: list[Margen] = []
    for t in np.linspace(0.12, 0.88, cortes):
        origen, tang, normal = ac.plano_de_corte(coef, centro, float(x0 + t * (x1 - x0)))
        pts = ac.seccion(poly, origen, tang)
        if pts is None or len(pts) < 80:
            continue
        perfil = ac._suaviza(np.stack([(pts[:, :2] - origen) @ normal, pts[:, 2]], 1))
        u, h = perfil[:, 0], perfil[:, 1]
        borde = 10
        i_apice = int(np.argmax(h[borde:-borde])) + borde
        creciente = u[-1] > u[0]
        rama = np.arange(i_apice, len(u)) if creciente else np.arange(i_apice, -1, -1)
        if len(rama) < 40:
            continue
        ur, hr = u[rama], h[rama]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ur), np.diff(hr)))])
        du, dh = np.gradient(ur, s), np.gradient(hr, s)
        k = du * np.gradient(dh, s) - dh * np.gradient(du, s)
        if not creciente:
            k = -k
        k = np.convolve(k, np.ones(7) / 7, "same")
        tope = min(VENTANA_MARGEN_MM[1], s[-1] - BORDE_LIBRE_MM)
        dentro = (s > VENTANA_MARGEN_MM[0]) & (s < tope)
        if dentro.sum() < 12:
            continue
        j = int(np.flatnonzero(dentro)[np.argmax(k[dentro])])
        punto = np.array([*(origen + ur[j] * normal), hr[j]])
        fuera.append(Margen(float(t), punto, float(hr[j])))
    return fuera, razon, plano


# --------------------------------------------------------------------------- #
# El segmentador geométrico que consume el `segmentation-agent`
# --------------------------------------------------------------------------- #
class SegmentadorGingival:
    """Separa encía de diente por la altura del margen, interpolada a lo largo del arco.

    Cumple el protocolo `Segmenter` —`(N,3) → (N,C)` log-probabilidades— sin ser un
    modelo: el margen gingival ya es la frontera, y ponerle una red encima no la
    localizaría mejor. Devuelve log-probabilidades de verdad (`log_softmax`), no
    logits, porque el agente lo **comprueba**: leer logits como log-probabilidades da
    confianzas plausibles y falsas.
    """

    def __init__(self, dureza: float = 1.2) -> None:
        self.dureza = dureza  # mm de la banda de transición

    def __call__(self, puntos: np.ndarray) -> np.ndarray:
        _, ejes, P, _ = ac.marco_arcada(puntos)
        coef, centro = ac.curva_arco(P)
        # La malla se vuelve a seccionar aquí: el agente entrega solo posiciones, sin
        # caras, así que se usan las del propio campo por vecindad en el arco.
        alturas = self._alturas_del_margen(P, coef, centro)
        x = np.clip(P[:, 0], alturas[0][0], alturas[-1][0])
        umbral = np.interp(x, [a[0] for a in alturas], [a[1] for a in alturas])
        margen = (P[:, 2] - umbral) / self.dureza  # >0 coronal = diente
        logits = np.stack([-margen, margen], 1)
        return logits - np.logaddexp(logits[:, 0], logits[:, 1])[:, None]

    @staticmethod
    def _alturas_del_margen(P, coef, centro) -> list[tuple[float, float]]:
        """(x, altura del margen) muestreado a lo largo del arco.

        Sin caras no hay secciones, así que el margen se estima por percentil de la
        altura de los puntos **vestibulares** en cada tramo: la encía queda por debajo.
        Es más burdo que la detección por curvatura, y basta para separar dos clases.
        """
        radio = np.linalg.norm(P[:, :2] - centro, axis=1)
        fuera_arco = radio > np.percentile(radio, 55)
        xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 24)
        alturas = []
        for a, b in zip(xs[:-1], xs[1:], strict=True):
            tramo = fuera_arco & (P[:, 0] >= a) & (P[:, 0] < b)
            if tramo.sum() < 50:
                alturas.append(((a + b) / 2, np.percentile(P[:, 2], 45)))
                continue
            alturas.append(((a + b) / 2, float(np.percentile(P[tramo, 2], 45))))
        return alturas


# --------------------------------------------------------------------------- #
# Exportación al visor
# --------------------------------------------------------------------------- #
_C0 = 0.28209479177387814  # armónico esférico de grado 0


# Color del PROPIO PACIENTE, muestreado de sus fotos clínicas intraorales (Canon EOS
# 6D Mark II + EF 100 mm macro, misma sesión que los escaneos) y aplicado por región:
# esmalte en las coronas, encía por debajo del margen.
#
# Se descartó pintar el campo con un mapa divergente del desplazamiento: la medida es
# **una cifra en milímetros sobre la anatomía**, como en la referencia de Overjet, y un
# mapa de calor no dice cuánto se movió nada — solo insinúa. Además reutilizaba el rojo,
# que en este proyecto ya significa «pérdida establecida», para un cambio que es normal
# tras una higiene. El desplazamiento lo llevan las cotas; el color es anatomía.
#
# Cómo se obtuvieron, porque tiene tres trampas y las tres están medidas:
#
#   1. Las fotos NO son sRGB. El EXIF lo dice dos veces —`InteropIndex = R03` y el guion
#      bajo inicial del nombre (`_MG_`), que es la convención DCF de Canon para Adobe
#      RGB— y no llevan perfil ICC embebido, así que nada avisa al decodificar.
#      Leyéndolas como sRGB la encía salía un 11 % menos roja de lo que es.
#   2. El balance de blancos estaba en automático, pero convergió al mismo punto en las
#      cuatro tomas (R/G 2,20 · B/G 1,53 · 5969–6131 K del MakerNote), así que no hubo
#      nada que corregir.
#   3. El brillo SÍ varía —31 puntos de L\* entre la oclusal y la frontal— y eso el EXIF
#      no lo explica: la exposición es idéntica en las cuatro (1/100, f/16, ISO 125) y
#      la caída es del flash con la distancia. Por eso el color se ancla en la **razón
#      encía/esmalte**, que sí es estable entre tomas (0,82–0,91 · 0,58–0,70).
#
# ⚠️ El esmalte sale sesgado a claro: la máscara se queda con el 8 % más luminoso, o sea
# con los brillos especulares. Separarlos del color difuso pediría RAW o un polarizador.
# Y sin una referencia física en el encuadre —carta gris, o una guía Vita— esto es color
# *estimado*, no colorimetría.
#
# De las fotos no sale nada más que estos dos tríos: el `image-agent` les quita el EXIF
# al ingerir, que en unas imágenes con cara importa.
COLOR_ESMALTE = (0.847, 0.749, 0.635)  # sRGB (216, 191, 162)
COLOR_ENCIA = (0.725, 0.451, 0.388)  # sRGB (185, 115, 99)

# Sombreado horneado. Un campo de gaussianas NO tiene luces: el color guardado *es*
# la apariencia final. Los otros casos del visor parecen tridimensionales porque el
# entrenamiento horneó dentro del color el sombreado de los renders de Blender; este
# campo no pasa por ahí, así que con un color constante se vería como una silueta
# plana. Se hornea aquí: Lambert con una luz principal y un relleno flojo por el lado
# contrario para que las sombras no se cierren a negro.
#
# ⚠️ Es sombreado FALSO de visualización, no fotometría. El escaneo no trae color, y
# esto no lo mide: lo hace legible. Mismo aviso que el falso color de las capas de
# ToothFairy.
LUZ_PRINCIPAL = np.array([0.35, -0.80, 0.49])  # frontal, algo elevada
LUZ_RELLENO = np.array([-0.55, 0.60, 0.58])
AMBIENTE = 0.28
PESO_RELLENO = 0.35


def sombrea(normales: np.ndarray, color: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """Color por vértice = color base × intensidad Lambert de dos luces.

    `color` admite un trío constante o ya un color por vértice `(N,3)`, que es como se
    usa aquí: el campo lleva esmalte y encía mezclados por la probabilidad del
    segmentador.
    """
    def difusa(luz: np.ndarray) -> np.ndarray:
        # Difusa ENVOLVENTE (*half-Lambert*), no Lambert crudo. Con Lambert la mitad de
        # la malla que no mira a la luz cae a cero y el ambiente es lo único que queda:
        # medido, la intensidad mediana era 0,52, o sea que el color se oscurecía a la
        # mitad. Daba igual mientras el color era marfil inventado; ahora es el color
        # medido del paciente y el sombreado tiene que MODULARLO, no sustituirlo.
        # Envolver es además lo habitual para piel y encía, que dispersan bajo la
        # superficie y no tienen un terminador duro.
        return 0.5 * (normales @ (luz / np.linalg.norm(luz))) + 0.5

    intensidad = AMBIENTE + (1.0 - AMBIENTE) * np.clip(
        difusa(LUZ_PRINCIPAL) + PESO_RELLENO * difusa(LUZ_RELLENO), 0.0, 1.0
    )
    base = np.atleast_2d(np.asarray(color, dtype=float))
    return np.clip(base * intensidad[:, None], 0.0, 1.0)


def color_por_region(p_corona: np.ndarray) -> np.ndarray:
    """Mezcla esmalte y encía con la probabilidad del segmentador, sin frontera dura.

    Se usa la probabilidad y no la clase porque el margen gingival **no es una arista**:
    hay una franja de encía marginal que se adelgaza sobre el esmalte. Un corte duro
    dibujaría un borde recto que la anatomía no tiene y que el ojo lee como artefacto.
    """
    w = p_corona[:, None]
    return w * np.array(COLOR_ESMALTE) + (1.0 - w) * np.array(COLOR_ENCIA)


def escala_del_campo(puntos: np.ndarray, por_vertice: bool = True) -> np.ndarray | float:
    """σ tangencial deducida del espaciado real, **por vértice**.

    El divisor NO sale de «una gaussiana se ve hasta 2σ», que era mi razonamiento
    anterior y daba huecos: con σ = d/1,7 la alfa a mitad de camino entre dos vecinas se
    queda en 0,68 y la superficie no cierra. Sale de barrer el factor y medir, sobre un
    molar de cerca, huecos (alfa < 0,9 dentro de la silueta) frente a nitidez (energía
    de gradiente **solo donde alfa > 0,99**, que es la corrección que importa):

        σ = d/1,7   102 µm   26,7 % de huecos   nitidez 7,53   solo 3,5 % de area cerrada
        σ = d/1,4   123 µm    5,0 %             nitidez 5,06
        σ = d/1,2   143 µm    0,7 %             nitidez 2,76   33 % de area cerrada
        σ = d/1,05  163 µm    0,2 %             nitidez 2,34
        σ = d/0,85  204 µm    0,3 %             nitidez 2,12

    ⚠️ La nitidez altísima de la primera fila es un espejismo: está medida sobre el
    3,5 % de píxeles que quedan cerrados, y lo que la infla son **los propios agujeros**,
    que generan gradientes enormes. El ojo lee ese punteado como detalle. Por eso el
    campo sin arreglar «parecía más nítido» y no lo era.

    d/1,2 es la rodilla: cierra la superficie y conserva la mayor nitidez de entre las
    opciones que de verdad cierran.

    Lo importante es que sea **local**. Con una σ global sacada de un percentil pasa lo
    peor de los dos mundos: donde la malla es densa —las coronas— las gaussianas salen
    infladas y difuminan, y donde es basta —la encía, peor triangulada— siguen sin
    tocarse y queda punteado. Medido: la distancia al vecino va de 138 µm de mediana a
    204 µm en el percentil 90, y ninguna cifra única sirve para las dos zonas.

    Se promedian los tres vecinos más próximos, no solo el primero: un único vecino
    pegado por un artefacto de triangulación daría una σ diminuta y abriría un agujero.
    """
    from scipy.spatial import cKDTree

    d, _ = cKDTree(puntos).query(puntos, k=4)
    local = d[:, 1:].mean(axis=1) / 1.2
    if not por_vertice:
        return float(np.percentile(local, 90))
    # Se recorta contra su propia distribución: un vértice suelto en el borde del
    # recorte tiene vecinos lejísimos y pintaría un disco enorme sobre la anatomía.
    return np.clip(local, *np.percentile(local, [2, 98]))


# Grosor del disco como fracción de su radio. La gaussiana se aplana contra la
# superficie: ancha en el plano tangente para tapar el hueco hasta el vecino, y fina en
# la normal para que la superficie no engorde. Con 1/4 el disco sigue teniendo cuerpo a
# contraluz —una lámina de grosor cero centellea en los ángulos rasantes— sin difuminar.
RAZON_GROSOR = 0.25


def orientacion_surfel(normales: np.ndarray) -> np.ndarray:
    """Cuaternión por vértice que alinea el TERCER eje de la gaussiana con la normal.

    Convierte cada bola en un **disco tangente a la superficie**. Es la respuesta
    geométrica al punteado, frente a la aprendida: el entrenamiento también cierra los
    huecos, pero mueve los puntos y ensancha lateralmente —medido: σ mayor 216 µm— y
    con eso lava el detalle. Aquí no se mueve ni un vértice de donde lo puso el escáner,
    así que la nitidez es la del dato.

    Devuelve `(w, x, y, z)`, que es el orden en que la convención INRIA guarda `rot_*`.
    """
    from scipy.spatial.transform import Rotation

    n = normales / np.maximum(np.linalg.norm(normales, axis=1, keepdims=True), 1e-12)
    # Un tangente cualquiera: se cruza la normal con el eje del que más se aleja, que es
    # lo que evita el producto vectorial degenerado cuando la normal ya va en ese eje.
    aux = np.zeros_like(n)
    aux[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
    t1 = np.cross(n, aux)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    # Columnas = ejes de la gaussiana, en el mismo orden que `scale_0..2`.
    q = Rotation.from_matrix(np.stack([t1, t2, n], axis=2)).as_quat()  # (x, y, z, w)
    return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])


def escribe_campo(
    puntos: np.ndarray, color: np.ndarray, destino: Path,
    escala_mm: np.ndarray | float | None = None, normales: np.ndarray | None = None,
) -> None:
    """Campo gaussiano en el formato que carga el visor (convención INRIA).

    Un **disco tangente** por vértice cuando se le dan las normales, y una bola cuando
    no. No es un campo *entrenado* —no lo pretende— sino la geometría medida con su
    color, y esa es justamente su ventaja: cada gaussiana está donde la puso el escáner.

    ⚠️ **Se probó entrenarlo y salió peor para lo que importa.** Dos corridas completas
    (800 y 2048 px de render, 7000 y 9000 iteraciones) cerraron el punteado, sí, pero el
    entrenamiento reparte gaussianas anchas a lo largo de la superficie —σ mayor 216 µm
    medido— y con eso lava la anatomía: en un molar de cerca se ve la superficie
    continua y el detalle borrado. Aquí el punteado se cierra **sin mover nada**:
    aplanando cada gaussiana contra la superficie y ensanchándola solo en el plano
    tangente hasta solapar con sus vecinas. Es geometría, no aprendizaje, y tarda
    segundos en vez de treinta y nueve minutos.
    """
    n = len(puntos)
    if escala_mm is None:
        escala_mm = escala_del_campo(puntos)
    reg = np.zeros(n, dtype=[(c, "<f4") for c in (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")])
    reg["x"], reg["y"], reg["z"] = puntos[:, 0], puntos[:, 1], puntos[:, 2]
    for i in range(3):
        reg[f"f_dc_{i}"] = (color[:, i] - 0.5) / _C0
    # Opaca a propósito, al revés que un campo entrenado. Discos que solapan y son
    # opacos se resuelven por profundidad: gana el de delante y el borde queda limpio.
    # Con opacidad baja se mezclarían entre sí y volvería el difuminado que queremos
    # evitar — la neblina es un apaño del entrenamiento, no una virtud.
    reg["opacity"] = 4.0                       # sigmoide ≈ 0,98
    escala_mm = np.broadcast_to(np.asarray(escala_mm, dtype=float), (n,))
    if normales is not None:
        reg["scale_0"] = reg["scale_1"] = np.log(escala_mm)              # plano tangente
        reg["scale_2"] = np.log(escala_mm * RAZON_GROSOR)                # grosor
        q = orientacion_surfel(normales)
        for i in range(4):
            reg[f"rot_{i}"] = q[:, i]
        reg["nx"], reg["ny"], reg["nz"] = normales[:, 0], normales[:, 1], normales[:, 2]
    else:
        for i in range(3):
            reg[f"scale_{i}"] = np.log(escala_mm)  # el formato guarda la escala en log
        reg["rot_0"] = 1.0                         # cuaternión identidad
    cabecera = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {c}\n" for c in reg.dtype.names)
        + "end_header\n"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("wb") as f:
        f.write(cabecera.encode("ascii"))
        f.write(reg.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--antes", required=True, type=Path, help="Escaneo del momento 1.")
    ap.add_argument("--despues", required=True, type=Path, help="Escaneo del momento 2.")
    ap.add_argument("--salida", required=True, type=Path, help="Directorio. Fuera del repo.")
    ap.add_argument("--ply", type=Path, help="Campo gaussiano coloreado para el visor.")
    ap.add_argument("--cbct", type=Path, help="Serie DICOM: activa el pipeline completo.")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    # --- 1 · ingesta con el mesh-agent -------------------------------------- #
    store = ArtifactStore(args.salida / "_store")
    agente = MeshAgent(store)
    mallas, nubes = {}, {}
    segmentador = SegmentadorGingival()
    for etiqueta, ruta in (("antes", args.antes), ("despues", args.despues)):
        salida = agente.ingest(ruta)
        print(f"{etiqueta:8} {ruta.name[:38]:40} {salida.status.value:9} "
              f"{salida.n_primitives or 0:>7,} primitivas · {salida.latency_s:.2f} s")
        if not salida.ok or salida.artifact_ref is None:
            print(f"  {salida.detail}", file=sys.stderr)
            return 1
        arte = store.load(salida.artifact_ref)
        mallas[etiqueta] = arte
        puntos = arte["positions"].astype(np.float64)
        # --- 2 · encía frente a diente (ver la nota de arquitectura arriba) --- #
        logp = segmentador(puntos)
        corona = logp.argmax(1) == 1
        # La probabilidad, no solo la clase: el color la usa para mezclar sin frontera.
        nubes[etiqueta] = (puntos, corona, np.exp(logp[:, 1]))
        print(f"{etiqueta:8} {len(puntos):>7,} puntos · corona {corona.sum():>7,} "
              f"({corona.mean():.0%}) · encía {(~corona).sum():>7,}")

    # --- opcional: el pipeline completo, donde el contrato sí lo admite ----- #
    if args.cbct:
        pipe = IngestionPipeline(store, segmenter=segmentador)
        r = pipe.run(CaseInput(acquisition_id="histora-despues",
                               mesh=args.despues, cbct=args.cbct))
        print(f"\npipeline completo: snapshot {'sí' if r.snapshot else 'no'} · "
              f"{r.latency_s:.1f} s")
        if r.snapshot is not None:
            r = pipe.fuse(r)
            for a in r.analysis:
                print(f"  {a.agent:24} {a.status.value:9} "
                      f"{len(a.detected or {})} región(es) detectada(s)")

    # --- 3 · registro SOBRE LAS CORONAS ------------------------------------- #
    rng = np.random.default_rng(0)
    def muestra(p, n=25000):
        return p[rng.choice(len(p), min(n, len(p)), replace=False)]

    reg = icp(muestra(nubes["antes"][0][nubes["antes"][1]]),
              muestra(nubes["despues"][0][nubes["despues"][1]]), trim=0.7)
    print(f"\nregistro sobre coronas: rms emparejado {reg.rms_efectivo_mm:.3f} mm · "
          f"solape {reg.overlap_fraction:.2f}")
    if reg.overlap_fraction is not None and reg.overlap_fraction < 0.5:
        print("⚠ solape bajo: puede que no sean el mismo paciente.", file=sys.stderr)

    # --- 4 · desplazamiento del margen -------------------------------------- #
    # El "antes" se lleva al espacio del "despues" con la rígida que salió de las
    # coronas; el marco del arco se calcula UNA vez y se usa para los dos.
    antes_alineado = apply(
        quaternion_to_matrix(reg.rotation), np.asarray(reg.translation),
        mallas["antes"]["positions"].astype(np.float64),
    )
    md, razon, plano = margenes(
        mallas["despues"]["positions"].astype(np.float64), mallas["despues"]["faces"]
    )
    ma, _, _ = margenes(antes_alineado, mallas["antes"]["faces"], plano=plano)
    medidas = {"antes": ma, "despues": md}
    print(f"marco común del arco: orientación {razon:.2f}")
    for etiqueta, mm in medidas.items():
        print(f"{etiqueta:8} margen hallado en {len(mm):2}/{CORTES} cortes")

    ta = {round(x.t, 3): x for x in medidas["antes"]}
    td = {round(x.t, 3): x for x in medidas["despues"]}
    comunes = sorted(set(ta) & set(td))
    filas, descartados = [], 0
    for t in comunes:
        lateral = float(np.linalg.norm(td[t].punto[:2] - ta[t].punto[:2]))
        if lateral > DERIVA_LATERAL_MAX_MM:
            descartados += 1
            continue
        d = td[t].altura - ta[t].altura
        filas.append({"t": t, "antes_mm": ta[t].altura, "despues_mm": td[t].altura,
                      "desplazamiento_mm": d,
                      # Los dos extremos en el marco común, que es el del .ply: el
                      # visor los proyecta tal cual, sin transformar nada.
                      "antes_xyz": [round(float(v), 3) for v in ta[t].punto],
                      "despues_xyz": [round(float(v), 3) for v in td[t].punto]})
    if not filas:
        print("Ningún corte común entre los dos momentos.", file=sys.stderr)
        return 1
    desp = np.array([f["desplazamiento_mm"] for f in filas])
    print(f"\n{len(filas)} cortes comparables ({descartados} descartados por deriva "
          f"lateral > {DERIVA_LATERAL_MAX_MM} mm) · "
          f"desplazamiento mediano {np.median(desp):+.2f} mm "
          f"· rango {desp.min():+.2f} a {desp.max():+.2f}")
    print(f"residuo del registro: {reg.rms_efectivo_mm:.3f} mm — "
          f"{'POR ENCIMA' if abs(np.median(desp)) < reg.rms_efectivo_mm else 'por debajo'} "
          "de la señal mediana")

    destino = args.salida / "desplazamiento.json"
    destino.write_text(json.dumps(
        {"registro": {"rms_emparejado_mm": reg.rms_efectivo_mm,
                      "solape": reg.overlap_fraction},
         "cortes": filas}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {destino}")

    if args.ply:
        # El campo se dibuja sobre el escaneo POSTERIOR, que es el estado actual del
        # paciente. El color es su anatomía —esmalte y encía—, no la medida: la medida
        # va en las cotas, en milímetros.
        V_post = mallas["despues"]["positions"].astype(np.float64)
        P = (V_post - plano["centro_mundo"]) @ plano["ejes"].T
        # Las normales se giran con los mismos ejes que las posiciones: el marco es
        # una rotación, así que basta con la misma matriz y sin la traslación.
        N = vertex_normals(V_post, mallas["despues"]["faces"]) @ plano["ejes"].T
        color = sombrea(N, color_por_region(nubes["despues"][2]))
        sigma = escala_del_campo(P)
        escribe_campo(P.astype(np.float32), color, args.ply, escala_mm=sigma, normales=N)
        print(f"→ {args.ply}  ({len(P):,} gaussianas · disco σ tangencial "
              f"{np.percentile(sigma, 10) * 1000:.0f}–{np.percentile(sigma, 90) * 1000:.0f} µm "
              f"(mediana {np.median(sigma) * 1000:.0f}) × {RAZON_GROSOR:.0%} de grosor · "
              f"corona {nubes['despues'][1].mean():.0%})")
        cotas = args.ply.with_name(args.ply.stem + "_cotas.json")
        cotas.write_text(json.dumps(
            {"unidad": "mm",
             "que_mide": "desplazamiento del margen gingival entre dos escaneos",
             "residuo_registro_mm": round(reg.rms_efectivo_mm, 3),
             "cotas": [{"a": f["antes_xyz"], "b": f["despues_xyz"],
                        "mm": round(f["desplazamiento_mm"], 2)} for f in filas]},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {cotas}  ({len(filas)} cotas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
