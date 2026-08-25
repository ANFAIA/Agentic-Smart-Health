"""`composite-mesh-export-agent` — `TwinSnapshot` → **STL imprimible**, arcada y piezas.

Es el exportador que justifica el gemelo, y por eso conviene decir primero lo que **no**
es. Los demás canales devuelven lo que entró: el `export-agent` reconstruye el STL desde
`surface_ref` con error de formato, y eso demuestra que el contenedor es honesto, no que
el twin sirva para algo. Un clínico que sólo quiera imprimir la arcada que escaneó ya
tiene su fichero y no necesita nada de esto.

Lo que no tiene, y ninguna máquina le da, es **un diente con su raíz**. El escáner
intraoral no ve bajo el margen gingival; el CBCT sí, pero su superficie de tejido blando
es ruido. Cada modalidad aporta lo único que sabe medir.

**Dos salidas, y la diferencia entre ellas es una decisión clínica, no técnica.**

* **La arcada** (`destination`) sale como la midió el escáner y **SIN raíces**. Un modelo
  de estudio o una guía tienen que asentar sobre una base plana, y quince raíces colgando
  lo impiden. Va en el marco del twin, o sea registrada contra el CBCT.
* **Una pieza por fichero** (`-pieza-NN.stl`): corona medida del escáner + raíz
  reconstruida del CBCT, en un solo objeto. Eso **sí** se imprime —planificar una
  extracción compleja, un canino incluido, un modelo docente— y apoya sobre la corona.

⚠️ **Y hay algo que ningún fichero puede arreglar: una pieza impresa no lleva cabecera.**
Todo lo que este agente declara sobre el reconstructor desaparece en cuanto el objeto sale
de la impresora; quien lo tiene en la mano ve una raíz y no tiene forma de saber que es
inferida. Por eso lo reconstruido va SÓLO en los ficheros por pieza, que alguien pide a
propósito para una pieza concreta, y nunca en lo que se imprime por defecto.

**La estrategia es asimétrica, y ahí está el truco.** La mitad del escáner **no se
reconstruye**: `surface_ref` ya guarda `positions` y `faces`, o sea que esa superficie *ya
es una malla* medida a decenas de micras, y entra tal cual. Sólo se reconstruye **lo que
el escáner no explica**: las gaussianas del CBCT que quedan lejos de la superficie
escaneada, que por definición geométrica —y no por un umbral anatómico que este agente no
tiene con qué fijar— es la parte sumergida.

**Por qué no es marching cubes.** El campo del twin no es un volumen de vóxeles: es una
nube de gaussianas con covarianza. Mallar por isosuperficie exigiría rasterizar el campo a
una rejilla primero —tirar la representación y volver al formato del que veníamos— y
además está medido en `scripts/resolucion_modalidades.py` que sobre hueso trabecular el
área de la isosuperficie **no existe como magnitud**: depende de la resolución con que la
midas. Aquí se trabaja sobre los centros, con un complejo alfa cuya escala sale del
espaciado real de la nube y no de una rejilla elegida a mano.

**Por pieza, nunca en bloque.** Cada raíz se reconstruye por separado usando el código FDI
que la segmentación escribió en `region_id`. No es una optimización: una reconstrucción
global uniría piezas vecinas por donde sus alfa-complejos se tocan y saldría un bloque de
dientes fundidos.

**Cómo se sabe que la raíz vale, si nada más mide raíces.** No se sabe directamente, y por
eso no se afirma. Se mide el **mismo método en la banda donde sí hay verdad**: se
reconstruye también la corona desde las gaussianas del CBCT y se compara contra la malla
del escáner, que ahí es referencia. Ese número —`max_deviation_mm`— caracteriza al
reconstructor sobre esmalte. Extrapolarlo a la raíz es una hipótesis, y va escrita como tal
en la cabecera en vez de colarse como si fuera una medida.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from analysis_agents.dental import LONGITUD_MM
from core_schemas import ModalityStatus, TwinSnapshot
from ingestion_agents.ontology import describe

from export_agents.anatomia import marco_anatomico
from export_agents.base import BaseExportAgent, ExportOutput, SurfaceStore
from export_agents.compuesto import _al_marco_del_twin, espaciado_de_malla
from export_agents.solido import cierra_en_solido
from export_agents.stl import read_stl_triangles, write_binary_stl

# Cuántos espaciados típicos de la nube mide el radio del complejo alfa. Es el único
# parámetro libre del reconstructor y por eso se mide en unidades de la propia nube y
# no en milímetros: un CBCT de 0,15 mm de vóxel y uno de 0,4 mm necesitan alfas
# distintos en mm y el mismo en espaciados. Por debajo de ~2 la superficie se agujerea
# donde el muestreo es irregular; muy por encima se traga las concavidades y la raíz
# sale como una cápsula. El valor se valida midiendo la banda de corona, que es para
# lo que existe esa medida.
ALFA_ESPACIADOS = 2.5

# A cuántos espaciados de la malla del escáner deja de considerarse que una gaussiana
# del CBCT está describiendo la misma superficie. Más cerca que esto, el escáner ya lo
# mide mejor y la gaussiana sobra; más lejos, el escáner no llega y la gaussiana es lo
# único que hay. **No es un margen gingival**: es una pregunta sobre qué modalidad
# cubre cada punto, que es la que este agente sabe contestar. Dónde acaba la encía es
# anatomía, y decidirlo aquí sería inventarse un criterio clínico.
SUMERGIDO_ESPACIADOS = 3.0

# Percentil 95 de densidad por debajo del cual un cuerpo **no es un diente**, por muy
# bien que lo parezca. El esmalte es el tejido más denso del cuerpo y satura el techo del
# rango HU: medido sobre un caso clínico, los catorce dientes reales dan p95 = 1,000
# exacto, mientras que la tuberosidad del maxilar —que tiene tamaño de diente (19,4 mm),
# forma de diente y la misma separación del vecino que dos molares contiguos— se queda en
# 0,747, a un pelo del 0,710 del hueso sin etiquetar. La forma no distingue; la densidad
# sí, porque un diente sin corona de esmalte no existe.
#
# ⚠️ El umbral vive en densidad NORMALIZADA, así que depende de `hu_range`. Con el techo
# en 2000 HU, 0,90 cae en ~1830 HU: por encima del hueso cortical y dentro de dentina y
# esmalte. Si un campo llega con un techo mucho más alto el umbral dejaría de valer y el
# guardarraíl se desactiva declarándolo, en vez de ponerse a rechazar dientes buenos.
ESMALTE_P95_MINIMO = 0.90
TECHO_HU_MAXIMO = 2500.0

# Mínimo de gaussianas para intentar reconstruir una pieza. Por debajo de esto el
# complejo alfa no tiene de qué salir y la "raíz" sería una esquirla: se declara la
# pieza omitida en vez de escribir un cuerpo suelto dentro del STL.
MINIMO_POR_PIEZA = 50


def largo_de(nube: np.ndarray) -> float:
    """Longitud de la pieza a lo largo de **su propio** eje mayor, en milímetros.

    ⚠️ No la caja en Z ni la extensión sobre el eje global de la arcada. Un molar
    superior está inclinado respecto al eje de la arcada, así que medir sobre el global
    da un número que no es la longitud de nada — y confundirlo con ella es lo que hace
    que un recorte a 22 mm deje una pieza de 36. El eje mayor sale de la propia nube del
    CBCT, que es donde la raíz **se ve**; sacarlo de la corona del escáner ya se probó y
    está medido en `analysis_agents.dental` que no funciona: el escáner ve un casquete.
    """
    if len(nube) < 3:
        return 0.0
    centrada = nube - nube.mean(axis=0)
    _, _, ejes = np.linalg.svd(centrada, full_matrices=False)
    return float(np.ptp(centrada @ ejes[0]))


def _radios_circunscritos(puntos: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    """Radio de la esfera circunscrita de cada tetraedro.

    Es el criterio del complejo alfa: un tetraedro pertenece a la forma si su esfera
    circunscrita cabe en el radio alfa. Se resuelve el sistema lineal del centro
    circunscrito en vez de la fórmula del determinante porque un tetraedro casi plano
    —que los hay, en el borde de cualquier nube— hace explotar la segunda mientras que
    la primera se limita a estar mal condicionada, y eso se detecta.
    """
    p = puntos[tetra]  # (T, 4, 3)
    a = 2.0 * (p[:, 1:] - p[:, :1])  # (T, 3, 3)
    b = (p[:, 1:] ** 2).sum(axis=2) - (p[:, :1] ** 2).sum(axis=2)  # (T, 3)

    radios = np.full(len(tetra), np.inf)
    # Un tetraedro degenerado no tiene centro circunscrito: `inf` lo deja fuera del
    # complejo para cualquier alfa, que es exactamente lo que se quiere de él.
    det = np.linalg.det(a)
    bueno = np.abs(det) > 1e-12
    if bueno.any():
        # `b` va como columna, no como vector: apilado, `solve` leería un `(T, 3)` como
        # una única matriz de términos independientes en vez de T sistemas.
        centros = np.linalg.solve(a[bueno], b[bueno, :, None])[:, :, 0]
        radios[bueno] = np.linalg.norm(centros - p[bueno, 0], axis=1)
    return radios


def superficie_alfa(puntos: np.ndarray, alfa: float) -> np.ndarray:
    """Complejo alfa de una nube → las caras de su frontera, como índices.

    Devuelve `(F, 3)`. La frontera son las caras que pertenecen a **un solo**
    tetraedro del complejo: las que comparten dos están dentro del sólido y no son
    superficie. Es la definición, y es también la razón de que el resultado sea
    cerrado siempre que el complejo lo sea.

    Las caras salen sin orientar de forma consistente. Da igual para el STL, que
    recalcula la normal por cara desde los vértices; importaría para un formato que
    dependiera del orden de bobinado, y entonces haría falta propagar orientación.
    """
    from scipy.spatial import Delaunay, QhullError

    if len(puntos) < 4:
        return np.empty((0, 3), dtype=np.int64)
    try:
        tri = Delaunay(puntos)
    except QhullError:
        # Nube coplanar o degenerada: no hay volumen que triangular. Es un dato sobre
        # la nube, no un fallo del agente, y quien llama lo declara como pieza omitida.
        return np.empty((0, 3), dtype=np.int64)

    dentro = tri.simplices[_radios_circunscritos(puntos, tri.simplices) <= alfa]
    if len(dentro) == 0:
        return np.empty((0, 3), dtype=np.int64)

    # Las cuatro caras de cada tetraedro, cada una ordenada para que dos copias de la
    # misma cara colisionen aunque vengan de tetraedros distintos.
    caras = np.concatenate(
        [dentro[:, [0, 1, 2]], dentro[:, [0, 1, 3]], dentro[:, [0, 2, 3]], dentro[:, [1, 2, 3]]]
    )
    caras = np.sort(caras, axis=1)
    unicas, cuenta = np.unique(caras, axis=0, return_counts=True)
    return unicas[cuenta == 1]


def _une(mallas: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Varias mallas → una, desplazando los índices de caras. Sin fusionar vértices.

    No se sueldan los cuerpos entre sí a propósito: la corona del escáner y la raíz del
    CBCT son **dos superficies medidas por instrumentos distintos**, y coserlas exigiría
    decidir dónde acaba una y empieza la otra, que es justo la pregunta que este agente
    no sabe contestar. Salen como cuerpos separados dentro del mismo STL, que es un
    fichero perfectamente imprimible y que no afirma una continuidad que nadie midió.
    """
    posiciones, caras, desplazamiento = [], [], 0
    for pos, car in mallas:
        posiciones.append(pos)
        caras.append(car + desplazamiento)
        desplazamiento += len(pos)
    if not posiciones:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    return np.concatenate(posiciones), np.concatenate(caras)


class CompositeMeshExportAgent(BaseExportAgent):
    """Escribe el STL del compuesto y mide el reconstructor donde hay verdad."""

    name = "composite-mesh-export-agent"
    version = "0.3.0"

    def __init__(self, store: SurfaceStore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store = store

    def _export(  # type: ignore[override]
        self,
        snapshot: TwinSnapshot,
        destination: Path,
        *,
        etiquetas_ios: np.ndarray | None = None,
        **_: Any,
    ) -> ExportOutput:
        from scipy.spatial import cKDTree

        if snapshot.surface_ref is None:
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "El snapshot no trae superficie: sin escáner no hay corona con la que "
                    "componer la raíz, y el campo del CBCT solo no es una malla."
                ),
            )
        transform = snapshot.provenance.transform
        if transform is None:
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "El snapshot no tiene `provenance.transform`: sin fusión geométrica la "
                    "corona y la raíz quedarían en sistemas distintos dentro del mismo STL."
                ),
            )

        malla = self.store.load(snapshot.surface_ref)
        vertices = np.asarray(malla["positions"], dtype=np.float64)
        caras = np.asarray(malla["faces"], dtype=np.int64)
        campo = self.store.load(snapshot.gaussian_field_ref)
        centros = np.asarray(campo["centers"], dtype=np.float64)
        region = np.asarray(campo.get("region_id", np.zeros(len(centros))), dtype=np.int16)

        if not (region > 0).any():
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "Ni una gaussiana del campo lleva código FDI: sin segmentación no hay "
                    "piezas que reconstruir por separado, y en bloque saldrían fundidas."
                ),
            )

        # La malla al marco del twin, igual que en `compuesto.py`, y el campo a
        # coordenadas absolutas si venía centrado. Los dos tienen que acabar en el
        # mismo sistema o el STL sale con la raíz a un lado y la corona a otro.
        superficie = _al_marco_del_twin(vertices, transform)
        if "origin" in campo:
            centros = centros + np.asarray(campo["origin"], dtype=np.float64)

        espaciado_ios = espaciado_de_malla(vertices)
        arbol = cKDTree(superficie)
        distancia, _ = arbol.query(centros)
        sumergido = distancia > SUMERGIDO_ESPACIADOS * espaciado_ios

        densidad = campo.get("density")
        techo = campo.get("hu_range")
        mide_esmalte = densidad is not None and (
            techo is None or float(np.asarray(techo).ravel()[-1]) <= TECHO_HU_MAXIMO
        )

        raices, omitidas, sin_esmalte = {}, [], []
        for fdi in sorted({int(c) for c in region[region > 0]}):
            suya = region == fdi
            # ⚠️ La prueba del esmalte va ANTES de reconstruir, y no es una optimización:
            # reconstruir primero produciría un STL con nombre de diente a partir de un
            # trozo de mandíbula, y a partir de ahí ya sólo se puede borrar.
            if mide_esmalte and float(np.percentile(np.asarray(densidad)[suya], 95)) < (
                ESMALTE_P95_MINIMO
            ):
                sin_esmalte.append(fdi)
                continue
            nube = centros[suya & sumergido]
            if len(nube) < MINIMO_POR_PIEZA:
                omitidas.append(fdi)
                continue
            alfa = ALFA_ESPACIADOS * espaciado_de_malla(nube)
            frontera = superficie_alfa(nube, alfa)
            if len(frontera) == 0:
                omitidas.append(fdi)
                continue
            raices[fdi] = (nube, frontera)

        if not raices:
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    f"Ninguna de las {len({int(c) for c in region[region > 0]})} piezas "
                    "etiquetadas tiene gaussianas bajo la superficie escaneada: el CBCT no "
                    "aporta raíz que la malla del escáner no cubra ya."
                ),
            )

        desviacion, medidas_en, sesgo = self._mide_en_corona(
            centros, region, ~sumergido, arbol, superficie
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        # ⚠️ **La arcada va SIN raíces, y no es una omisión.** Un modelo de estudio o una
        # guía tienen que asentar sobre una base plana, y quince raíces colgando lo
        # impiden. Peor: una pieza impresa no lleva cabecera, así que todo lo que este
        # agente declara sobre el reconstructor desaparece en cuanto sale de la
        # impresora, y quien la tiene en la mano ve una raíz sin saber que es inferida.
        # Las raíces van en los ficheros por pieza, donde el clínico las pide a
        # propósito.
        # Y va CERRADA. Un escaneo intraoral es una cáscara: mide la superficie que la
        # cámara ve, sin interior y sin fondo. Se ve bien en pantalla y no se imprime,
        # porque un laminador necesita saber qué es dentro y qué es fuera. Ver `solido`.
        base, caras_base, cierre = self._cierra(superficie, caras, etiquetas_ios)
        write_binary_stl(
            destination,
            base,
            caras_base,
            header=(
                "ASH composite-mesh arcada=IOS-medida sin-raices "
                + ("solido-cerrado" if cierre.get("estanca") else "CASCARA-ABIERTA")
            ),
        )
        vuelta = read_stl_triangles(destination)
        formato = float(np.abs(vuelta - base[caras_base]).max())

        escritas, sin_corona = self._por_pieza(
            destination, raices, superficie, caras, etiquetas_ios, desviacion, sesgo
        )

        motivos: list[str] = []
        if not cierre.get("estanca"):
            motivos.append(
                "la arcada NO ha quedado estanca: "
                + str(cierre.get("motivo", "quedan aristas abiertas"))
                + ". Se puede mirar en pantalla, pero un laminador no sabrá qué es "
                "interior, así que este fichero no está listo para imprimir."
            )
        pasadas = []
        for fdi, (nube, _frontera) in sorted(raices.items()):
            try:
                cota = LONGITUD_MM.get(describe(str(fdi)).tooth_type.value)
            except KeyError:  # un código que la ontología no reconoce no es un fallo aquí
                continue
            largo = largo_de(nube)
            if cota is not None and largo > cota:
                pasadas.append(f"FDI {fdi} {largo:.0f}/{cota:.0f}")

        if pasadas:
            # ⚠️ Se DECLARA, no se recorta. Recortar convertiría el ápice en supuesto y a
            # partir de ahí medir longitud radicular sobre el resultado sería medir lo que
            # se ha supuesto — el mismo argumento que hay escrito en `analysis_agents`.
            #
            # Un aviso agrupado y no uno por pieza: doce líneas seguidas se leen como
            # ruido, y el ruido es como se desactiva un gate.
            motivos.append(
                f"{len(pasadas)} raíz/raíces reconstruidas son MÁS LARGAS de lo que su "
                f"tipo de diente admite ({', '.join(pasadas)} mm): están arrastrando hueso "
                "alveolar, así que lo que se imprima por debajo del ápice no es diente."
            )
        if sin_esmalte:
            motivos.append(
                f"{len(sin_esmalte)} cuerpo(s) etiquetado(s) como diente NO tienen esmalte "
                f"y no se han exportado (FDI {', '.join(str(f) for f in sin_esmalte)}): "
                "tienen forma y tamaño de pieza pero su densidad es la del hueso, así que "
                "la segmentación les puso un código FDI que no les toca."
            )
        if omitidas:
            motivos.append(
                f"{len(omitidas)} pieza(s) etiquetada(s) no llegaron a reconstruirse "
                f"(FDI {', '.join(str(f) for f in omitidas)}): no tienen fichero propio."
            )
        if sin_corona:
            motivos.append(
                f"{len(sin_corona)} pieza(s) salen con la raíz reconstruida y SIN corona "
                f"medida (FDI {', '.join(str(f) for f in sin_corona)}): el escáner no trae "
                "esos vértices etiquetados, así que el fichero es medio diente y hay que "
                "saberlo antes de imprimirlo."
            )
        if desviacion is None:
            motivos.append(
                "no se pudo medir el reconstructor contra la corona del escáner, así que "
                "la fidelidad de las raíces de estos ficheros no está caracterizada."
            )
        return self._outcome(
            ModalityStatus.OK,
            path=destination,
            paths=escritas,
            format="stl",
            frame="twin",
            n_vertices=len(base),
            n_faces=len(caras_base),
            # ⚠️ Lo que se declara NO es el error de formato del STL (`formato`, del orden
            # de 1e-5 mm y sin interés aquí) sino el del reconstructor sobre esmalte. Es
            # el número que decide si los ficheros por pieza se pueden usar, y meter en su
            # lugar el del `float32` daría un 1e-5 mm que parecería excelente y no diría
            # nada de la mitad de esos ficheros que sí puede estar mal.
            max_deviation_mm=desviacion,
            detail=self._detalle(
                len(escritas), omitidas, desviacion, medidas_en, sesgo, formato
            ),
            hitl_reasons=motivos,
        )

    def _cierra(
        self, superficie: np.ndarray, caras: np.ndarray, etiquetas_ios: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """La arcada, cerrada en sólido con base plana. Si no se puede, se dice.

        ⚠️ La base es perpendicular al eje **oclusal** —hacia dónde muerde la pieza— y no
        al superior ni al Z del fichero. El Z es el que tenía la máquina y no significa
        nada; y el superior, en un maxilar, es el OPUESTO del oclusal: usarlo no daba un
        modelo torcido sino uno al revés, con la falda bajando por delante de los dientes
        y envolviéndolos. El eje sale medido de los códigos FDI del propio escaneo
        (`anatomia.marco_anatomico`), así que sin etiquetas no hay eje — y entonces se
        entrega la cáscara **declarándolo**, en vez de inventarse una vertical.
        """
        if etiquetas_ios is None or len(etiquetas_ios) != len(superficie):
            return superficie, caras, {
                "estanca": False,
                "motivo": "sin etiquetas FDI del escáner no se puede medir hacia dónde "
                "está la coronilla, y una base perpendicular al eje del fichero saldría "
                "inclinada",
            }
        marco, motivo = marco_anatomico(superficie, np.asarray(etiquetas_ios))
        if marco is None:
            return superficie, caras, {"estanca": False, "motivo": motivo}
        return cierra_en_solido(superficie, caras, hacia_las_coronas=marco.oclusal)

    def _por_pieza(
        self,
        destino: Path,
        raices: dict[int, tuple[np.ndarray, np.ndarray]],
        superficie: np.ndarray,
        caras: np.ndarray,
        etiquetas_ios: np.ndarray | None,
        desviacion: float | None,
        sesgo: float | None,
    ) -> tuple[list[Path], list[int]]:
        """Un STL por diente: corona medida + raíz reconstruida, en un solo cuerpo.

        Es lo que la arcada no puede ser. Un diente suelto con su raíz **sí** se imprime
        —planificar una extracción compleja, un canino incluido, un modelo docente— y no
        arrastra el problema de la base plana: apoya sobre la corona.

        **La corona sale del ESCÁNER**, recortando la malla por el código FDI que el
        escaneo trae por vértice. Donde ambas modalidades ven la misma superficie manda
        la que la mide a decenas de micras, no la de 0,4 mm de vóxel — misma regla que
        aplica el `viewer-export-agent`.

        Sin `etiquetas_ios` no hay forma de saber qué vértice del escáner es qué diente:
        el fichero sale con la raíz sola y **se declara**, en vez de rellenar el hueco
        con la corona del CBCT, que existe pero es peor y haría pasar por medido algo
        que no lo está a esa exactitud.
        """
        etq = None if etiquetas_ios is None else np.asarray(etiquetas_ios, dtype=np.int16)
        if etq is not None and len(etq) != len(superficie):
            # Longitud que no cuadra: las etiquetas no son de esta malla. Se ignoran en
            # vez de indexar con ellas, que produciría coronas de otro diente.
            etq = None

        escritas, sin_corona = [], []
        for fdi, (nube, frontera) in raices.items():
            cuerpos = [(nube, frontera)]
            corona = self._corona_de(fdi, superficie, caras, etq)
            if corona is None:
                sin_corona.append(fdi)
            else:
                cuerpos.insert(0, corona)

            pos, car = _une(cuerpos)
            ruta = destino.with_name(f"{destino.stem}-pieza-{fdi}.stl")
            write_binary_stl(
                ruta, pos, car,
                header=self._cabecera_pieza(fdi, corona is not None, desviacion, sesgo),
            )
            escritas.append(ruta)
        return escritas, sin_corona

    def _corona_de(
        self, fdi: int, superficie: np.ndarray, caras: np.ndarray, etq: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """La parte de la malla del escáner etiquetada con `fdi`, reindexada.

        Se queda con las caras cuyos **tres** vértices son de la pieza. Con dos bastaría
        para no perder el borde, pero arrastraría triángulos que cruzan al diente vecino
        y la corona saldría con rebabas hacia los lados.
        """
        if etq is None:
            return None
        suyos = etq == fdi
        if not suyos.any():
            return None
        completas = caras[suyos[caras].all(axis=1)]
        if len(completas) == 0:
            return None
        usados = np.unique(completas)
        remapeo = np.full(len(superficie), -1, dtype=np.int64)
        remapeo[usados] = np.arange(len(usados))
        return superficie[usados], remapeo[completas]

    def _mide_en_corona(
        self,
        centros: np.ndarray,
        region: np.ndarray,
        emergido: np.ndarray,
        arbol: Any,
        superficie: np.ndarray,
    ) -> tuple[float | None, int, float | None]:
        """El mismo reconstructor sobre la corona, contra la malla del escáner.

        Es la única validación posible del método: sobre la corona hay dos medidas
        independientes del mismo tejido —el escáner, exacto a decenas de micras, y el
        CBCT— así que la discrepancia entre la superficie reconstruida y la escaneada
        es error del reconstructor y no de la anatomía. Bajo la encía sólo hay una
        medida y no hay contra qué comparar.

        Devuelve `(p95, n_puntos, sesgo)`, o `None` si no hay corona suficiente: sin
        medida no se afirma nada.

        **El signo importa más que la magnitud.** El reconstructor usa los CENTROS de
        las gaussianas como puntos de superficie, así que cabe esperar que la malla
        salga metida hacia adentro y no dispersa alrededor de la verdad. Un p95 a secas
        se leería como ruido; si hay sesgo, es una raíz impresa más delgada de lo que
        mide, y en una guía quirúrgica eso no es lo mismo que una gruesa. Por eso el
        sesgo se mide aparte y se declara aparte — **medido, no supuesto**: la dirección
        que salga es la que se escribe, incluso si contradice lo que cabía esperar.

        ⚠️ El signo NO sale de las normales de la malla del escáner, aunque las haya.
        Una normal por vértice apunta hacia donde diga el bobinado de sus caras, y eso
        es una convención del fichero, no anatomía: con el bobinado invertido el mismo
        error salía «hacia fuera». Se usa la dirección radial desde el centroide de la
        propia nube de corona, que es hacia donde crece un diente y no depende de cómo
        se escribió ningún fichero.
        """
        nube = centros[(region > 0) & emergido]
        if len(nube) < MINIMO_POR_PIEZA:
            return None, 0, None
        frontera = superficie_alfa(nube, ALFA_ESPACIADOS * espaciado_de_malla(nube))
        if len(frontera) == 0:
            return None, 0, None
        # Los vértices que la reconstrucción pone en la frontera, no todos los centros:
        # un centro interior no está sobre la superficie y no tiene por qué caer sobre
        # la malla del escáner. Medir contra él inflaría el error con puntos que el
        # método nunca afirmó que fueran superficie.
        en_frontera = nube[np.unique(frontera)]
        distancia, indice = arbol.query(en_frontera)

        # Negativo = el punto reconstruido queda por dentro de la superficie escaneada.
        hacia = en_frontera - superficie[indice]
        radial = en_frontera - nube.mean(axis=0)
        proyeccion = np.einsum("ij,ij->i", hacia, radial)
        sesgo = float(np.median(np.sign(proyeccion) * distancia))
        return float(np.percentile(distancia, 95)), len(en_frontera), sesgo

    def _cabecera_pieza(
        self, fdi: int, con_corona: bool, desviacion: float | None, sesgo: float | None
    ) -> str:
        """Los 80 bytes del fichero por pieza. Son su única procedencia.

        ⚠️ Caben 80 CONTADOS y `stl_header` trunca en silencio, así que aquí no va el
        `acquisition_id` —que en la primera versión se comía el espacio y dejaba la
        medida fuera del fichero— sino lo único que no se deduce mirando la geometría:
        qué mitad se midió, cuál se reconstruyó y con cuánto sesgo.

        Y hay una cosa que estos bytes NO pueden arreglar: una pieza impresa no lleva
        cabecera. Por eso el fichero por pieza existe sólo cuando alguien lo pide para
        esa pieza, y la arcada —lo que se imprime por defecto— sale sin raíz.
        """
        cifra = (
            "SIN-MEDIR"
            if desviacion is None
            else (
                f"p95={desviacion:.2f}"
                + (f" sesgo={sesgo:+.2f}mm" if sesgo is not None else "mm")
            )
        )
        corona = "corona=IOS-medida" if con_corona else "SIN-corona"
        return f"ASH pieza-{fdi} {corona} raiz=CBCT-recon {cifra}"

    def _detalle(
        self,
        n_piezas: int,
        omitidas: list[int],
        desviacion: float | None,
        medidas: int,
        sesgo: float | None,
        formato: float,
    ) -> str:
        partes = [
            f"la arcada sale como la midió el escáner y SIN raíces, que es lo que se "
            f"imprime; las {n_piezas} pieza(s) reconstruidas van en un fichero cada una, "
            f"con su corona medida y su raíz del CBCT por complejo alfa"
        ]
        if omitidas:
            partes.append(f"{len(omitidas)} pieza(s) sin raíz reconstruible: FDI {omitidas}")
        if desviacion is None:
            partes.append("reconstructor sin caracterizar: no había corona con la que medir")
        else:
            partes.append(
                f"el reconstructor se desvía {desviacion:.4f} mm (p95) de la malla del "
                f"escáner sobre {medidas} puntos de corona; para la raíz es una hipótesis, "
                f"no una medida: nada más mide raíces"
            )
            if sesgo is not None:
                lado = "hacia dentro" if sesgo < 0 else "hacia fuera"
                partes.append(
                    f"y no es ruido sino sesgo: la superficie reconstruida cae "
                    f"{abs(sesgo):.4f} mm {lado} de la escaneada (mediana), así que la "
                    f"raíz impresa sale {'delgada' if sesgo < 0 else 'gruesa'} en esa "
                    f"medida y no centrada en la verdad"
                )
        partes.append(f"error de formato del STL {formato:.2e} mm")
        return ". ".join(partes) + "."
