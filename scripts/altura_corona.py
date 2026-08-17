#!/usr/bin/env python
"""altura_corona.py — mide la altura de corona clínica sobre el escáner intraoral.

    uv run python scripts/altura_corona.py --escaneo ARCADA.stl --salida DIR
    uv run python scripts/altura_corona.py --escaneo ARCADA.stl --posicion 0.5

**Por qué existe.** La demo *Iris* de Overjet enseña, sobre la imagen, una cota
vertical de encía a esmalte. Validar si eso es reproducible aquí destapó que no es
una medida, sino **tres**, y que solo una necesita registro entre modalidades:

    1. margen gingival → borde incisal      solo escáner   ← esto
    2. LAC → cresta ósea                    solo CBCT
    3. margen gingival → cresta ósea        IOS + CBCT, y no tiene equivalente en 2D

Este script hace el peldaño 1, que es el mejor condicionado: todo ocurre dentro de
una sola malla, con la exactitud del escáner (decenas de µm) y **sin registro**. El
peldaño 3 cuelga de `scripts/registro_ios_cbct.py`, que mide 0,45 mm de error de
registro y cuyo control positivo aún no cierra.

---

## Cómo mide, y por qué así

**No segmenta dientes.** Se intentó primero: curvatura media con Laplaciano
cotangente, difundida sobre la malla, y componentes conexas tras quitar la banda
cóncava. Medido sobre este escaneo, en el mejor caso salen **6 componentes con
tamaño de corona de las ~14 esperadas** y siempre queda un pegote de 92 mm: la
mayoría de los dientes siguen unidos a la encía porque el pliegue del surco no es
una línea cerrada continua a la resolución del escaneo. Separar coronas por
curvatura sola pide graph-cut o campos armónicos, y eso es otro proyecto.

**Mide sobre secciones, que es un problema 1D.** La cota de Iris vive en la cara
vestibular, así que se corta la malla con un plano perpendicular a la curva del arco
(`vtkCutter` + `vtkStripper`) y se trabaja sobre la polilínea resultante.

**El ápice oclusal es un punto interior, no un extremo.** La sección de una arcada
escaneada es un arco abierto: sus dos puntas son el borde del escaneo (pliegue
vestibular y suelo lingual). El punto oclusal es el máximo axial **del interior**, y
tomarlo así evita tener que fiarse de la orientación global.

**El eje oclusal se orienta por la dispersión de la cresta.** El criterio obvio —el
lado cuyos puntos extremos quedan más lejos del centro del arco— **está mal**, y
costó una tanda de secciones invertidas: el pliegue vestibular del labio se va más
lejos que las propias coronas. El criterio que sí funciona es que la cresta oclusal
es una curva apretada y el pliegue no: se ajusta un polinomio de grado 4 al decil
extremo de cada lado y gana el de menor dispersión (medido: 4,09 mm contra 9,00).

**El margen gingival es el codo de la rama vestibular.** Bajando desde el ápice, la
corona abomba hasta su mayor contorno, se estrangula en el cuello y la encía sale de
nuevo hacia fuera. Ese pliegue es un máximo de curvatura con signo, y se busca entre
2,5 y 14 mm de longitud de arco desde el ápice.

---

## Lo que midió, y lo que no

    I.F.S. POST HIGIENE LowerJawScan.stl    mediana 7,0 mm   (3,2 – 8,4)
    I.F.S. POST HIGIENE UpperJawScan.stl    mediana 6,0 mm   (2,4 – 10,8)

En el superior las alturas suben a 9-10,8 mm **en los sectores centrales**, que es
donde están los incisivos y es justo lo que mide su corona clínica. Ese patrón
—y no la mediana— es lo que dice que la medida sigue a la anatomía.

> ⚠ **En los dos modelos `Visualization_DigitalModelUnsectioned_*` sale ruido**
> (medianas de 3,1 y 3,6 mm, con valores de 0,2 a 12). No está diagnosticado por qué;
> el criterio de orientación tampoco separa con holgura en ellos. Hasta saberlo, la
> salida de este script **no vale sin mirar la sección**: por eso cada medida se
> dibuja con sus dos puntos encima y por eso `--posicion` mide de una en una.

**Esto no diagnostica.** Una recesión gingival se define contra la LAC, que el
escáner no ve porque está bajo la encía. Lo que se mide aquí es corona clínica —lo
que asoma—, que es el observable del escáner y nada más.

**El fichero de salida no entra en el repositorio.** Es geometría de paciente, y el
`data-guardian` no la caza dentro de un PNG: veta `.stl` y `.ply`, pero un render
pasa por delante. Por eso `--salida` apunta fuera del árbol por defecto.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "packages/ingestion-agents/src"))
sys.path.insert(0, str(_RAIZ / "packages/fusion-agents/src"))
from fusion_agents.marco import (  # noqa: E402  (reexportadas: las importan otros scripts)
    DECIL_CRESTA,
    GRADO_ARCO,
    curva_arco,
    marco_arcada,
)
from ingestion_agents.mesh_agent import read_mesh  # noqa: E402

# La malla se lee con el parser del repositorio y no con `vtkSTLReader`: es el que
# deduplica la sopa de triángulos y reconstruye topología, y es el que ya usa el
# `mesh-agent`. VTK queda solo de motor de corte y de render.

# `marco_arcada` y `curva_arco` vivían aquí. Se movieron a `fusion_agents.marco` porque
# media docena de sitios las cargaban de este script con `spec_from_file_location`, y
# porque suponer el eje oclusal en vez de medirlo ya costó dos fallos (ver su docstring).
# Se reexportan para no romper a quien las importe de aquí.
__all__ = ["DECIL_CRESTA", "GRADO_ARCO", "curva_arco", "marco_arcada"]

SUAVIZADO = 9  # ventana, en puntos de la polilínea (~0,1 mm cada uno)
VENTANA_MARGEN_MM = (2.5, 14.0)  # dónde buscar el codo, desde el ápice
COLOR_MALLA = (0.94, 0.90, 0.84)  # marfil plano: el color por vértice de un STL no existe
COLOR_COTA = (0.85, 0.16, 0.16)


# --------------------------------------------------------------------------- #
# Marco del arco
# --------------------------------------------------------------------------- #
def plano_de_corte(
    coef: np.ndarray, centro_arco: np.ndarray, x: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origen, tangente y normal-hacia-vestibular del corte en la abscisa `x`."""
    origen = np.array([x, float(np.polyval(coef, x))])
    tangente = np.array([1.0, float(np.polyval(np.polyder(coef), x))])
    tangente /= np.linalg.norm(tangente)
    normal = np.array([-tangente[1], tangente[0]])
    if normal @ (origen - centro_arco) < 0:
        normal = -normal
    return origen, tangente, normal


# --------------------------------------------------------------------------- #
# Sección
# --------------------------------------------------------------------------- #
def a_polydata(V: np.ndarray, F: np.ndarray) -> vtk.vtkPolyData:
    puntos = vtk.vtkPoints()
    puntos.SetData(numpy_to_vtk(np.ascontiguousarray(V), deep=1))
    desplazamientos = np.arange(0, 3 * (len(F) + 1), 3, dtype=np.int64)
    celdas = vtk.vtkCellArray()
    celdas.SetData(
        numpy_to_vtkIdTypeArray(desplazamientos, deep=1),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(F.astype(np.int64).ravel()), deep=1),
    )
    malla = vtk.vtkPolyData()
    malla.SetPoints(puntos)
    malla.SetPolys(celdas)
    return malla


def seccion(malla: vtk.vtkPolyData, origen: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    """La polilínea más larga del corte, con sus puntos **en orden**.

    El orden es lo que se viene a buscar: `vtkCutter` devuelve segmentos sueltos y
    sin él no hay «bajar por la cara vestibular», solo una nube.
    """
    plano = vtk.vtkPlane()
    plano.SetOrigin(origen[0], origen[1], 0.0)
    plano.SetNormal(normal[0], normal[1], 0.0)
    corte = vtk.vtkCutter()
    corte.SetInputData(malla)
    corte.SetCutFunction(plano)
    limpio = vtk.vtkCleanPolyData()
    limpio.SetInputConnection(corte.GetOutputPort())
    tiras = vtk.vtkStripper()
    tiras.SetInputConnection(limpio.GetOutputPort())
    tiras.JoinContiguousSegmentsOn()
    tiras.Update()

    salida, mejor, largo = tiras.GetOutput(), None, 0
    for i in range(salida.GetNumberOfCells()):
        ids = salida.GetCell(i).GetPointIds()
        if ids.GetNumberOfIds() > largo:
            largo = ids.GetNumberOfIds()
            mejor = np.array([salida.GetPoint(ids.GetId(k)) for k in range(largo)])
    return mejor


def _suaviza(curva: np.ndarray, ventana: int = SUAVIZADO) -> np.ndarray:
    nucleo = np.ones(ventana) / ventana
    suave = np.stack([np.convolve(curva[:, j], nucleo, "same") for j in range(curva.shape[1])], 1)
    return suave[ventana:-ventana]  # los bordes del convolución no valen


def mide_corona(
    puntos: np.ndarray, origen: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(punto oclusal, punto del margen, altura en mm) sobre la rama vestibular."""
    if puntos is None or len(puntos) < 3 * SUAVIZADO + 60:
        return None
    plano = np.stack([(puntos[:, :2] - origen) @ normal, puntos[:, 2]], 1)
    plano = _suaviza(plano)
    hacia, axial = plano[:, 0], plano[:, 1]

    borde = 10  # el ápice es interior: las puntas son el borde del escaneo
    i_apice = int(np.argmax(axial[borde:-borde])) + borde
    creciente = hacia[-1] > hacia[0]
    rama = np.arange(i_apice, len(hacia)) if creciente else np.arange(i_apice, -1, -1)
    if len(rama) < 40:
        return None

    u, h = hacia[rama], axial[rama]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(u), np.diff(h)))])
    du, dh = np.gradient(u, s), np.gradient(h, s)
    curvatura = du * np.gradient(dh, s) - dh * np.gradient(du, s)
    if not creciente:
        curvatura = -curvatura  # recorrer la rama al revés le cambia el signo

    dentro = (s > VENTANA_MARGEN_MM[0]) & (s < VENTANA_MARGEN_MM[1])
    if dentro.sum() < 10:
        return None
    j = int(np.flatnonzero(dentro)[np.argmax(curvatura[dentro])])
    return np.array([u[0], h[0]]), np.array([u[j], h[j]]), float(h[0] - h[j])


def a_3d(plano_2d: np.ndarray, origen: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Del marco de la sección (hacia-vestibular, axial) al marco del arco."""
    return np.array([*(origen + plano_2d[0] * normal), plano_2d[1]])


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _esfera(centro: np.ndarray, radio: float) -> vtk.vtkActor:
    fuente = vtk.vtkSphereSource()
    fuente.SetCenter(*centro)
    fuente.SetRadius(radio)
    fuente.SetThetaResolution(24)
    fuente.SetPhiResolution(24)
    mapa = vtk.vtkPolyDataMapper()
    mapa.SetInputConnection(fuente.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapa)
    actor.GetProperty().SetColor(*COLOR_COTA)
    return actor


def render(
    P: np.ndarray,
    F: np.ndarray,
    oclusal: np.ndarray,
    margen: np.ndarray,
    altura: float,
    normal: np.ndarray,
    destino: Path,
    tamano: tuple[int, int] = (1100, 900),
) -> None:
    """Malla en color plano + la cota encima, mirando de frente a la cara vestibular."""
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.10, 0.11, 0.13)

    mapa = vtk.vtkPolyDataMapper()
    mapa.SetInputData(a_polydata(P, F))
    malla = vtk.vtkActor()
    malla.SetMapper(mapa)
    prop = malla.GetProperty()
    prop.SetColor(*COLOR_MALLA)
    prop.SetSpecular(0.25)
    prop.SetSpecularPower(28)
    prop.SetDiffuse(0.85)
    ren.AddActor(malla)

    # La cota flota 3 mm hacia vestibular. Dibujada sobre la superficie exacta queda
    # ENTERRADA: la cuerda entre el borde incisal y el margen pasa POR DENTRO de una
    # corona convexa, así que con 0,4 mm seguían viéndose solo las dos esferas. Como
    # la cámara mira a lo largo de esa misma normal, la anotación no se desalinea.
    fuera = np.array([*(normal * 3.0), 0.0])
    oclusal, margen = oclusal + fuera, margen + fuera

    linea = vtk.vtkLineSource()
    linea.SetPoint1(*oclusal)
    linea.SetPoint2(*margen)
    tubo = vtk.vtkTubeFilter()
    tubo.SetInputConnection(linea.GetOutputPort())
    tubo.SetRadius(0.12)
    tubo.SetNumberOfSides(16)
    mapa_cota = vtk.vtkPolyDataMapper()
    mapa_cota.SetInputConnection(tubo.GetOutputPort())
    cota = vtk.vtkActor()
    cota.SetMapper(mapa_cota)
    cota.GetProperty().SetColor(*COLOR_COTA)
    ren.AddActor(cota)
    ren.AddActor(_esfera(oclusal, 0.35))
    ren.AddActor(_esfera(margen, 0.35))

    medio = (oclusal + margen) / 2
    camara = ren.GetActiveCamera()
    camara.SetFocalPoint(*medio)
    camara.SetPosition(*(medio + np.array([*(normal * 88.0), 8.0])))
    camara.SetViewUp(0.0, 0.0, 1.0)
    ren.ResetCameraClippingRange()

    ventana = vtk.vtkRenderWindow()
    ventana.SetOffScreenRendering(1)
    ventana.AddRenderer(ren)
    ventana.SetSize(*tamano)
    ventana.Render()

    # La etiqueta va en 2D y no como texto 3D: un `vtkBillboardTextActor3D` se
    # escala con la escena y aquí salía del tamaño de un píxel. Se proyecta el punto
    # medio de la cota a coordenadas de pantalla y se dibuja al lado, ya renderizado.
    ren.SetWorldPoint(*medio, 1.0)
    ren.WorldToDisplay()
    px, py, _ = ren.GetDisplayPoint()
    etiqueta = vtk.vtkTextActor()
    etiqueta.SetInput(f"{altura:.1f} mm")
    etiqueta.SetDisplayPosition(int(px) + 16, int(py) - 12)
    texto = etiqueta.GetTextProperty()
    texto.SetFontSize(30)
    texto.SetColor(1.0, 1.0, 1.0)
    texto.SetBold(True)
    texto.SetShadow(True)  # la malla es clara: sin sombra el blanco se pierde
    ren.AddActor2D(etiqueta)
    ventana.Render()

    captura = vtk.vtkWindowToImageFilter()
    captura.SetInput(ventana)
    captura.Update()
    escritor = vtk.vtkPNGWriter()
    escritor.SetFileName(str(destino))
    escritor.SetInputConnection(captura.GetOutputPort())
    escritor.Write()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--escaneo", required=True, type=Path, help="Arcada intraoral (.stl/.obj).")
    ap.add_argument(
        "--posicion",
        type=float,
        help="Posición a lo largo del arco, de 0 a 1. Sin ella, recorre 14 secciones.",
    )
    ap.add_argument("--salida", type=Path, help="Directorio del PNG. Fuera del repositorio.")
    args = ap.parse_args()

    malla = read_mesh(args.escaneo)
    V, F = malla["positions"].astype(np.float64), malla["faces"]
    _, _, P, razon = marco_arcada(V)
    coef, centro_arco = curva_arco(P)
    poly = a_polydata(P, F)
    print(f"{args.escaneo.name}: {len(V)} vértices, {len(F)} caras")
    print(
        f"orientación del eje oclusal: razón de dispersión {razon:.2f} "
        + ("(clara)" if razon < 0.6 else "⚠ POCO CLARA: revisa la sección antes de creerte nada")
    )

    x0, x1 = np.percentile(P[:, 0], 5), np.percentile(P[:, 0], 95)
    posiciones = [args.posicion] if args.posicion is not None else np.linspace(0, 1, 14)
    medidas = []
    for t in posiciones:
        x = x0 + float(t) * (x1 - x0)
        origen, tangente, normal = plano_de_corte(coef, centro_arco, x)
        medida = mide_corona(seccion(poly, origen, tangente), origen, normal)
        if medida is None:
            print(f"  posición {t:.2f}  (sección insuficiente)")
            continue
        oclusal, margen, altura = medida
        medidas.append((t, x, oclusal, margen, altura, origen, normal))
        print(f"  posición {t:.2f}  x={x:+6.1f} mm   altura de corona clínica {altura:5.1f} mm")

    if not medidas:
        print("Ninguna sección dio una medida.", file=sys.stderr)
        return 1
    alturas = np.array([m[4] for m in medidas])
    print(
        f"\nmediana {np.median(alturas):.1f} mm   rango {alturas.min():.1f}–{alturas.max():.1f} mm"
    )

    if args.salida:
        args.salida.mkdir(parents=True, exist_ok=True)
        for t, _, oclusal, margen, altura, origen, normal in medidas:
            destino = args.salida / f"altura_{t:.2f}.png"
            render(
                P,
                F,
                a_3d(oclusal, origen, normal),
                a_3d(margen, origen, normal),
                altura,
                normal,
                destino,
            )
        print(f"{len(medidas)} render(s) en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
