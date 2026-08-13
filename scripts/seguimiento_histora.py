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


# Marfil plano, el mismo que el resto de casos del visor que no traen color
# fotométrico. Se descartó pintar el campo con un mapa divergente del
# desplazamiento: la medida es **una cifra en milímetros sobre la anatomía**, como
# en la referencia de Overjet, y un mapa de calor no dice cuánto se movió nada — solo
# insinúa. Además reutilizaba el rojo, que en este proyecto ya significa «pérdida
# establecida» (`lac_cresta_cbct.py`), para un cambio que es normal tras una higiene.
COLOR_MALLA = (0.94, 0.90, 0.84)

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


def sombrea(normales: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    """Color por vértice = color base × intensidad Lambert de dos luces."""
    def difusa(luz: np.ndarray) -> np.ndarray:
        return np.clip(normales @ (luz / np.linalg.norm(luz)), 0.0, None)

    intensidad = AMBIENTE + (1.0 - AMBIENTE) * np.clip(
        difusa(LUZ_PRINCIPAL) + PESO_RELLENO * difusa(LUZ_RELLENO), 0.0, 1.0
    )
    return np.clip(np.array(color)[None, :] * intensidad[:, None], 0.0, 1.0)


def escala_del_campo(puntos: np.ndarray) -> float:
    """σ de las gaussianas, deducida del espaciado real de la nube.

    Una gaussiana se ve hasta ~2σ, así que para que la superficie salga continua hace
    falta 2σ ≳ la distancia al vecino más próximo. Se toma el **percentil 90** de esa
    distancia y no la mediana: con la mediana, el decil más disperso —la encía, de
    triangulación más basta que las coronas— sale agujereado.
    """
    from scipy.spatial import cKDTree

    d, _ = cKDTree(puntos).query(puntos, k=2)
    return float(np.percentile(d[:, 1], 90) / 2.0)


def escribe_campo(
    puntos: np.ndarray, color: np.ndarray, destino: Path, escala_mm: float | None = None
) -> None:
    """Campo gaussiano en el formato que carga el visor (convención INRIA).

    Una gaussiana isótropa por vértice. No es un campo *entrenado* —no lo pretende—
    sino la geometría medida con su color: el visor ya sabe dibujar esto, y montar
    3DGS sobre `histora` para enseñar una cota sería pagar un entrenamiento por una
    imagen que ya tenemos.
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
    reg["opacity"] = 4.0                       # sigmoide ≈ 0,98: opaco
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
        nubes[etiqueta] = (puntos, corona)
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
        # paciente; el color es cuánto se movió el margen para llegar hasta aquí.
        V_post = mallas["despues"]["positions"].astype(np.float64)
        P = (V_post - plano["centro_mundo"]) @ plano["ejes"].T
        # Las normales se giran con los mismos ejes que las posiciones: el marco es
        # una rotación, así que basta con la misma matriz y sin la traslación.
        N = vertex_normals(V_post, mallas["despues"]["faces"]) @ plano["ejes"].T
        color = sombrea(N, COLOR_MALLA)
        sigma = escala_del_campo(P)
        escribe_campo(P.astype(np.float32), color, args.ply, escala_mm=sigma)
        print(f"→ {args.ply}  ({len(P):,} gaussianas · σ {sigma:.3f} mm)")
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
