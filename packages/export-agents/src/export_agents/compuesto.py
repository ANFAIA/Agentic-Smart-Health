"""`composite-export-agent` — dientes del CBCT + encía del escáner, en un solo fichero.

**Qué es el compuesto.** El modelo que persigue el proyecto. Cada modalidad aporta lo único
que sabe medir y ninguna vale sola:

- el **CBCT** ve por debajo del margen gingival —es lo único que da la raíz— pero su
  superficie de tejido blando es ruido a 0,30 mm de vóxel;
- el **escáner intraoral** mide la encía con exactitud de decenas de micras y no ve nada
  bajo el margen.

Por separado son un diente sin encía y una encía con dientes truncados.

**Por qué hace falta este agente, y no bastaba con lo que había.** El PLY del
`field-export-agent` ya lleva los dientes con su `region_id`, así que parecía que el
compuesto estaba hecho. No lo estaba: **la encía nunca se escribía**. El script que decía
producirlo contaba los vértices de encía, los anunciaba en un comentario del propio
fichero —`compuesto dientes-CBCT + encia-IOS`— y luego emitía `element vertex 498407`,
exactamente el tamaño del campo del CBCT y ni un punto más. El comentario era verdad sobre
la intención y mentira sobre el contenido.

**Lo que este agente NO inventa.** Un vértice del escáner no trae atenuación radiológica:
el IOS mide forma, no densidad. Ponerle una `density` plausible lo haría indistinguible de
una gaussiana medida en el CBCT, y cualquiera que después proyectara el campo estaría
integrando un número que nadie midió. Así que la encía entra con `density = 0.0` —declarado
aquí, no elegido por bonito— y con una columna `origen` que dice, gaussiana a gaussiana, de
qué modalidad viene. Un compuesto que no distingue sus dos mitades miente por omisión.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot

from export_agents.base import BaseExportAgent, ExportOutput, SurfaceStore
from export_agents.field import REVERSIBILITY_BUDGET_MM, escribe_ply, lee_ply
from export_agents.stl import quaternion_to_matrix

# Código de `origen` por gaussiana. No es decoración: es lo que separa una densidad medida
# de una forma medida, y sin él las dos mitades del compuesto son indistinguibles.
ORIGEN_CBCT = 0
ORIGEN_IOS = 1

# `density` de la encía. **Cero, y es una declaración, no un valor.** El escáner intraoral
# no mide atenuación; poner cualquier otra cosa haría que un DRR del compuesto integrase un
# número que nadie midió. Quien quiera renderizar la encía usa su geometría, no su σ.
DENSIDAD_ENCIA = 0.0

# Escala de las gaussianas de encía, en mm. Sale del espaciado real entre vértices vecinos
# de la malla, no de una constante: un escáner intraoral tiene resolución muy distinta a la
# del CBCT y usar la σ del volumen daría una encía o pixelada o hinchada.
FACTOR_ESCALA_ENCIA = 0.5
MUESTRA_ESPACIADO = 5000


def espaciado_de_malla(vertices: np.ndarray, *, semilla: int = 0) -> float:
    """Distancia mediana al vecino más próximo. Es el σ natural de esa nube."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(semilla)
    n = min(MUESTRA_ESPACIADO, len(vertices))
    idx = rng.choice(len(vertices), n, replace=False)
    d, _ = cKDTree(vertices).query(vertices[idx], k=2)
    return float(np.median(d[:, 1]))


class CompositeExportAgent(BaseExportAgent):
    """Escribe el compuesto y **mide** cuántas gaussianas aporta cada modalidad.

    Como los demás exportadores, el número es producto tanto como el fichero: si el
    compuesto sale con cero gaussianas de una de las dos mitades, eso se declara en vez de
    entregar medio modelo con nombre de entero.
    """

    name = "composite-export-agent"
    version = "0.1.0"

    def __init__(self, store: SurfaceStore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store = store

    def _export(  # type: ignore[override]
        self, snapshot: TwinSnapshot, destination: Path, **_: Any
    ) -> ExportOutput:
        if snapshot.surface_ref is None:
            # Sin escáner no hay encía que componer, y eso es normal: una adquisición
            # puede ser solo CBCT. MISSING, no FAILED — la distinción del ADR 001.
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "El snapshot no trae superficie: sin escáner intraoral no hay encía "
                    "que componer, y el campo del CBCT solo ya lo exporta "
                    "`field-export-agent`."
                ),
            )

        campo = self.store.load(snapshot.gaussian_field_ref)
        vertices = np.asarray(
            self.store.load(snapshot.surface_ref)["positions"], dtype=np.float64
        )

        # La malla entra ENTERA, encía y coronas. No se recorta aquí porque el
        # `region_id` vive sobre las gaussianas del CBCT, no sobre los vértices del
        # escáner: decidir qué vértice es corona exigiría una segmentación de la malla,
        # que es trabajo de otro agente. Quien quiera quitar la encía que solapa con los
        # dientes tiene la columna escrita para hacerlo.
        transform = snapshot.provenance.transform
        if transform is None:
            # Sin registro no hay compuesto posible: la encía quedaría en el sistema del
            # escáner y los dientes en el del CBCT, o sea dos objetos sueltos en el mismo
            # fichero. Pero es **entrada que falta**, no un fallo — la fusión geométrica
            # simplemente no ha corrido. MISSING, igual que cuando no hay escáner.
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "El snapshot no tiene `provenance.transform`: no pasó por la fusión "
                    "geométrica, así que no hay forma de poner la encía y los dientes en "
                    "el mismo sistema."
                ),
            )
        encia = _al_marco_del_twin(vertices, transform)
        if "origin" in campo:
            # El campo va centrado y la encía llega en coordenadas absolutas del CBCT.
            encia = encia - np.asarray(campo["origin"], dtype=np.float64)

        sigma = espaciado_de_malla(vertices) * FACTOR_ESCALA_ENCIA
        n_c, n_e = len(campo["centers"]), len(encia)
        region = np.asarray(
            campo.get("region_id", np.zeros(n_c)), dtype=np.int16
        )

        columnas = {
            "x": np.concatenate([campo["centers"][:, 0], encia[:, 0]]),
            "y": np.concatenate([campo["centers"][:, 1], encia[:, 1]]),
            "z": np.concatenate([campo["centers"][:, 2], encia[:, 2]]),
            "density": np.concatenate(
                [campo["density"], np.full(n_e, DENSIDAD_ENCIA)]
            ),
            "region_id": np.concatenate([region, np.zeros(n_e, dtype=np.int16)]),
            "origen": np.concatenate([
                np.full(n_c, ORIGEN_CBCT, dtype=np.int16),
                np.full(n_e, ORIGEN_IOS, dtype=np.int16),
            ]),
        }
        for i in range(3):
            columnas[f"scale_{i}"] = np.concatenate(
                [campo["scales"][:, i], np.full(n_e, sigma)]
            )
        # La encía no tiene orientación medida: el escáner da posiciones, no elipsoides.
        # Cuaternión identidad, que es la única rotación que no afirma nada.
        for i in range(4):
            columnas[f"rot_{i}"] = np.concatenate(
                [campo["rotations"][:, i], np.full(n_e, 1.0 if i == 0 else 0.0)]
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        con_nombre = int((region > 0).sum())
        escribe_ply(
            destination,
            columnas,
            comentarios=[
                f"generado por {self.qualified}",
                f"acquisition_id {snapshot.acquisition_id}",
                f"compuesto: {n_c} gaussianas de CBCT + {n_e} vertices de IOS",
                f"de las del CBCT, {con_nombre} llevan codigo FDI en region_id",
                f"origen: {ORIGEN_CBCT}=CBCT (densidad medida), {ORIGEN_IOS}=IOS (forma "
                "medida, density=0 porque el escaner NO mide atenuacion)",
                "region_id es el codigo FDI por gaussiana, 0 = sin asignar",
                f"escala de la encia {sigma:.4f} mm, del espaciado real de la malla",
            ],
        )

        # Releer lo que se acaba de escribir, igual que los otros canales. **La medida es
        # producto tanto como el fichero**, y aquí más: el compuesto mezcla dos fuentes con
        # marcos distintos, así que un error de encaje entre ellas es justo lo que un
        # número de round-trip cazaría y una inspección visual no.
        pedidos = np.column_stack([columnas["x"], columnas["y"], columnas["z"]])
        desviacion = _verifica_libre(destination, pedidos)

        motivos = []
        if desviacion > REVERSIBILITY_BUDGET_MM:
            motivos.append(
                f"el compuesto se desvía {desviacion:.4f} mm de lo que se pidió escribir, "
                f"por encima del presupuesto de {REVERSIBILITY_BUDGET_MM} mm del brief."
            )
        if con_nombre == 0:
            motivos.append(
                "el compuesto no lleva ni una gaussiana con código FDI: la segmentación "
                "no corrió o no encontró nada, así que esto es el campo y la encía sin "
                "anatomía."
            )
        return self._outcome(
            ModalityStatus.OK,
            path=destination,
            n_vertices=n_c + n_e,
            format="ply",
            max_deviation_mm=desviacion,
            hitl_reasons=motivos,
            detail=(
                f"{n_c:,} gaussianas del CBCT ({con_nombre:,} con FDI) + {n_e:,} "
                f"vértices de encía del escáner, σ {sigma:.3f} mm."
            ),
        )


def _verifica_libre(ruta: Path, pedidos: np.ndarray) -> float:
    """Desviación máxima entre lo que se quiso escribir y lo que quedó en disco."""
    leido = lee_ply(ruta)
    vuelta = np.column_stack([leido["x"], leido["y"], leido["z"]])
    if vuelta.shape != pedidos.shape:
        raise ValueError(
            f"La relectura de {ruta} da {vuelta.shape} y se escribieron {pedidos.shape}: "
            "el fichero no representa el compuesto."
        )
    return float(np.abs(vuelta - pedidos).max())


def _al_marco_del_twin(vertices: np.ndarray, transform: Any) -> np.ndarray:
    """La malla al sistema del CBCT, en el sentido en que se midió.

    Misma dirección que `ExportAgent._to_twin_frame`: la fusión geométrica registra la
    malla (`source`) sobre el CBCT (`target`), así que la rígida guardada va de escáner a
    twin y se aplica directa.
    """
    rot = quaternion_to_matrix(transform.rotation)
    return vertices @ np.asarray(rot).T + np.asarray(transform.translation, dtype=np.float64)
