"""`viewer-export-agent` — el paquete que `dental-3dgs-viewer` abre: campo + capa clínica.

**Qué emite y por qué dos ficheros.**

1. Un **PLY en perfil INRIA** (`<caso>-visor.ply`), que es lo único que un rasterizador de
   splats sabe leer. Es una obra **derivada** del campo del twin, no el campo.
2. Un **sidecar JSON** (`<caso>-visor.json`) con la capa clínica —dientes, hallazgos,
   medidas no regionales, motivos del gate— y el `region_id` por gaussiana. Mismo patrón
   que los `*_cotas.json` que ese visor ya consume.

**Por qué no un visor propio.** Se escribió uno, con WebGL a pelo, y se retiró: dibujaba
`gl.POINTS` redondos de tamaño uniforme, que no es splatting sino una nube de puntos con
buena presentación. `dental-3dgs-viewer` ya rasteriza splats de verdad, ya tiene panel de
capas, cotas sobre el modelo y encuadre por caso. Duplicarlo peor es el mismo error que la
app de MCP vacía: una pieza nuestra donde ya había una que funciona.

⚠️ **El PLY del visor NO es el twin, y esa es la línea que no se puede borrar.** Para que
un rasterizador lo abra hay que darle tres cosas que el CBCT no midió:

| propiedad INRIA | de dónde sale aquí | ¿es dato? |
|---|---|---|
| `opacity` | `alfa = 1 - exp(-g·sigma)`, luego logit | **no**: ganancia de visualización |
| `f_dc_*` | falso color por código FDI | **no**: un CBCT no mide color |
| `scale_*` | `log(sigma_mm)` | sí, la misma medida en otra escala |

Las dos primeras son interpretación, y por eso van **declaradas en la cabecera del propio
fichero** y el sidecar repite la función de transferencia. El twin reversible sigue siendo
el PLY de `field-export-agent`, que guarda `density` sin cota y escalas en mm; éste es para
mirar. La convención de la ganancia se toma prestada del `config.ts` del visor, que ya la
tenía escrita con estas mismas palabras: *«NO es dato — la sigma del artefacto es densidad
sin cota, y un visor espera alfa en [0,1]»*.
"""

from __future__ import annotations

import base64
import colorsys
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot

from export_agents.base import BaseExportAgent, ExportOutput, SurfaceStore
from export_agents.compuesto import ORIGEN_IOS

# Ganancia de la transferencia sigma → alfa: `alfa = 1 - exp(-g·sigma)`.
#
# Con sigma normalizada en [0,1], g=3 lleva el esmalte saturado a ~0,95 de opacidad y el
# tejido blando de 0,1 a ~0,26: se ve la anatomia densa sin que la baja densidad
# desaparezca. **No es un parametro del dato**, es de la vista, y se escribe en el fichero
# para que nadie lo confunda con una medida.
GANANCIA_DISPLAY = 3.0

# Coeficiente 0 de los armonicos esfericos. Un color RGB en [0,1] se guarda como
# `(c - 0.5) / C0`; es la convencion del PLY de INRIA y el visor la deshace igual.
C0 = 0.28209479177387814

# Densidad de display de la encia. En el twin su `density` es **0**, porque el escaner mide
# forma y no atenuacion (ver `compuesto.py`); con 0 seria invisible en un rasterizador que
# espera opacidad. Se le da un valor para que se vea, y por eso vive aqui —en el canal de
# la vista— y no toca el campo.
DENSIDAD_DISPLAY_ENCIA = 0.45

# Fraccion del espaciado real a la que se lleva sigma **para la vista**.
#
# ⚠️ Sin esto el visor ensena polvo, no anatomia. Las gaussianas del campo tienen la sigma
# que les puso el `cbct-agent` —medio voxel— y al decimar el espaciado entre las que quedan
# crece: medido sobre el caso real, sigma 0,075 mm sobre un espaciado de 0,276, o sea un
# ratio de 0,27. Los splats no llegan a tocarse y la superficie sale agujereada.
#
# Se infla sigma hasta ~0,6 del espaciado, que es donde los splats se solapan sin
# emborronar. **Es una decision de la vista y va declarada**: la sigma medida sigue intacta
# en el PLY del twin, y el factor aplicado se escribe en la cabecera y en el sidecar.
FRACCION_ESPACIADO = 0.6

# Falso color. El CBCT no mide color: esto es para distinguir piezas a ojo, igual que el
# `color` de las capas en el `config.ts` del visor.
_TONOS = (188, 152, 96, 42, 8, 330, 286, 244)
_COLOR_ENCIA = (0.85, 0.52, 0.29)
_COLOR_RESTO = (0.42, 0.45, 0.48)

# Presupuesto de gaussianas por grupo. **Los dientes no tienen presupuesto: entran todos.**
#
# Repartir uniformemente sobre las ~606.000 gaussianas seria el error: el 97 % del campo es
# hueso y craneo —el FOV es de cabeza entera— asi que una muestra uniforme dejaria los
# dientes casi vacios, que es justo lo unico que un clinico va a mirar. Se muestrea por
# grupo y cada grupo declara cuanto conserva.
PRESUPUESTO_ENCIA = 120_000
PRESUPUESTO_RESTO = 120_000

# Las tres capas del compuesto, en el orden en que se cargan. **No son ventanas de HU**
# como las de ToothFairy: aqui la separacion es por PROCEDENCIA y anatomia, que es la
# distincion que el gemelo aporta y la densidad no puede dar — el hueso alveolar y la raiz
# comparten HU, y por eso `segmentacion-diente-cbct.md` mide que ningun umbral los separa.
#
# Son disjuntas y cubren el compuesto entero, asi que encenderlas suma sin contar nada dos
# veces, igual que las de densidad.
CAPAS = (
    ("coronas", "Coronas (escaner)", "COMPLETAS y separadas por el propio escaner"),
    ("raices", "Raices (CBCT)", "lo que el escaner no ve; parcial"),
    ("encia", "Encia (escaner)", "superficie, sin densidad medida"),
    ("resto", "Resto del campo", "hueso y craneo sin nombre"),
)

_PROPIEDADES_INRIA = (
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


class ViewerExportAgent(BaseExportAgent):
    """Empaqueta el twin para `dental-3dgs-viewer` y **mide** lo que recorta.

    Como los demas canales, el numero es producto tanto como el fichero: el
    `ExportOutput` declara cuantas gaussianas viajan de cuantas hay, y por grupo.
    """

    name = "viewer-export-agent"
    version = "0.2.0"

    def __init__(self, store: SurfaceStore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store = store

    def _export(  # type: ignore[override]
        self,
        snapshot: TwinSnapshot,
        destination: Path,
        *,
        motivos: list[str] | None = None,
        etiquetas_ios: np.ndarray | None = None,
        semilla: int = 0,
    ) -> ExportOutput:
        campo = self.store.load(snapshot.gaussian_field_ref)
        centros = np.asarray(campo["centers"], dtype=np.float64)
        region = np.asarray(
            campo.get("region_id", np.zeros(len(centros))), dtype=np.int16
        )
        escalas = np.asarray(campo["scales"], dtype=np.float64)
        rotaciones = np.asarray(campo["rotations"], dtype=np.float64)
        densidad = np.asarray(campo["density"], dtype=np.float64)

        encia, aviso_encia = self._encia(snapshot, campo)
        sel_c, sel_e, recorte = _decima(region, len(encia), semilla=semilla)

        # Las gaussianas de encia no traen elipsoide medido: el escaner da posiciones. Se
        # les pone una esfera del tamano del espaciado de la malla y el cuaternion
        # identidad, que es la unica rotacion que no afirma nada.
        sigma_encia = _sigma_de(encia) if len(encia) else 0.1
        pos = np.vstack([centros[sel_c], encia[sel_e]])
        sig = np.vstack([escalas[sel_c], np.full((len(sel_e), 3), sigma_encia)])
        rot = np.vstack([
            rotaciones[sel_c], np.tile([1.0, 0.0, 0.0, 0.0], (len(sel_e), 1))
        ])
        den = np.concatenate([
            densidad[sel_c], np.full(len(sel_e), DENSIDAD_DISPLAY_ENCIA)
        ])
        etq_ios = (
            np.zeros(len(sel_e), dtype=np.int16) if etiquetas_ios is None
            else np.asarray(etiquetas_ios, dtype=np.int16)[sel_e]
        )
        fdi = np.concatenate([region[sel_c], etq_ios])
        origen = np.concatenate([
            np.zeros(len(sel_c), dtype=np.int16),
            np.full(len(sel_e), ORIGEN_IOS, dtype=np.int16),
        ])

        # Sigma de la VISTA: la medida, inflada al espaciado que queda tras decimar.
        factor = _factor_de_escala(pos, sig)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Una capa por fichero, que es como el visor las conmuta: cada una es su propia
        # escena y encenderla es cambiar un uniform, no recargar.
        # ⚠️ **Las coronas salen del ESCANER, no del CBCT**, y es la decision de fondo de
        # este canal. El escaner ya trae los dientes separados y con codigo FDI, con
        # exactitud de decenas de micras y sin huecos; el campo del CBCT cubre el 51 % del
        # volumen de cada pieza y de forma desigual — bien los posteriores, flojo los
        # anteriores. Ensenar el compuesto del CBCT como si fuera «el diente» presenta
        # como completo algo que no lo esta.
        #
        # Lo que el CBCT SI aporta y el escaner no puede es la RAIZ, que va en su propia
        # capa y declarada como parcial. Cada fuente ensena lo que sabe medir.
        es_ios = origen == ORIGEN_IOS
        grupos = {
            "coronas": es_ios & (fdi > 0),
            "raices": (~es_ios) & (fdi > 0),
            "encia": es_ios & (fdi == 0),
            "resto": (~es_ios) & (fdi == 0),
        }
        capas = []
        for clave, nombre, detalle in CAPAS:
            m = grupos[clave]
            ruta = destination.with_name(f"{destination.name}-{clave}.ply")
            _escribe_inria(
                ruta, pos[m], sig[m] * factor, rot[m], den[m], fdi[m], origen[m],
                snapshot, factor, nombre,
            )
            capas.append({
                "id": clave, "nombre": nombre, "detalle": detalle,
                "ply": ruta.name, "primitivas": int(m.sum()),
            })
        ply = destination.with_name(f"{destination.name}-coronas.ply")
        destination.with_suffix(".json").write_text(
            json.dumps(
                _sidecar(snapshot, fdi, recorte, motivos or [], factor,
                         # Los centroides salen de las CORONAS del escaner si las hay:
                         # son completas, asi que el centro de la pieza cae donde el
                         # clinico espera pinchar. Con las gaussianas del CBCT, una pieza
                         # con la mitad de su volumen tiene el centro desplazado.
                         _centroides(pos, np.where(es_ios, fdi, 0))
                         if etiquetas_ios is not None else _centroides(pos, fdi),
                         capas,
                         _por_pieza(pos, fdi, es_ios)),
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )

        avisos = list(aviso_encia)
        if int((region > 0).sum()) == 0:
            avisos.append(
                "el paquete del visor no lleva ni un diente etiquetado: la segmentación "
                "no corrió o no encontró nada, así que no habrá nada que seleccionar."
            )
        return self._outcome(
            ModalityStatus.OK,
            path=ply,
            n_vertices=len(pos),
            format="ply",
            hitl_reasons=avisos,
            detail=(
                f"perfil INRIA (derivado), {len(CAPAS)} capas + sidecar · {len(pos):,} de "
                f"{recorte['total']:,} gaussianas · dientes "
                f"{recorte['dientes']:,}/{recorte['dientes_total']:,} (enteros) · "
                f"encía {recorte['encia']:,}/{recorte['encia_total']:,} · "
                f"resto {recorte['resto']:,}/{recorte['resto_total']:,}"
            ),
        )

    def _encia(self, snapshot: TwinSnapshot, campo: dict) -> tuple[np.ndarray, list[str]]:
        """Los vertices del escaner en el marco del twin, o vacio y el motivo."""
        if snapshot.surface_ref is None:
            return np.empty((0, 3)), []
        if snapshot.provenance.transform is None:
            return np.empty((0, 3)), [
                "el snapshot trae escáner pero no transformación: sin registro no se "
                "puede poner la encía en el mismo sistema que los dientes, así que el "
                "paquete del visor lleva solo el campo del CBCT."
            ]
        from export_agents.compuesto import _al_marco_del_twin

        v = np.asarray(
            self.store.load(snapshot.surface_ref)["positions"], dtype=np.float64
        )
        encia = _al_marco_del_twin(v, snapshot.provenance.transform)
        if "origin" in campo:
            encia = encia - np.asarray(campo["origin"], dtype=np.float64)
        return encia, []


def _factor_de_escala(pos: np.ndarray, sigma: np.ndarray, *, muestra: int = 20_000) -> float:
    """Cuanto hay que inflar sigma para que los splats se toquen. `1.0` si ya se tocan.

    Se mide el espaciado real entre vecinos **de la nube que se va a exportar**, no de la
    original: es el recorte lo que abre los huecos, asi que el factor depende de cuanto se
    decimo. Nunca reduce — un campo ya denso se deja como esta.
    """
    from scipy.spatial import cKDTree

    if len(pos) < 100:
        return 1.0
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pos), min(muestra, len(pos)), replace=False)
    nn, _ = cKDTree(pos).query(pos[idx], k=2)
    espaciado = float(np.median(nn[:, 1]))
    actual = float(np.median(sigma))
    if actual <= 0 or espaciado <= 0:
        return 1.0
    return max(1.0, FRACCION_ESPACIADO * espaciado / actual)


def _sigma_de(vertices: np.ndarray, *, muestra: int = 5000, semilla: int = 0) -> float:
    """Media sigma del espaciado real entre vertices vecinos de la malla."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(vertices), min(muestra, len(vertices)), replace=False)
    d, _ = cKDTree(vertices).query(vertices[idx], k=2)
    return float(np.median(d[:, 1])) * 0.5


def _decima(
    region: np.ndarray, n_encia: int, *, semilla: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(indices del campo, indices de encia, informe)`. Los dientes entran enteros.

    El informe dice cuanto sobrevive **de cada grupo**, porque «se muestran 250.000 de
    606.000» sin decir de donde salen no informa de nada: la pregunta del clinico es si
    estan todos SUS dientes, no cuantas gaussianas hay.
    """
    rng = np.random.default_rng(semilla)
    es_diente = region > 0
    dientes = np.flatnonzero(es_diente)
    resto = np.flatnonzero(~es_diente)
    if len(resto) > PRESUPUESTO_RESTO:
        resto = np.sort(rng.choice(resto, PRESUPUESTO_RESTO, replace=False))

    idx_encia = np.arange(n_encia)
    if n_encia > PRESUPUESTO_ENCIA:
        idx_encia = np.sort(rng.choice(idx_encia, PRESUPUESTO_ENCIA, replace=False))

    informe = {
        "dientes": len(dientes), "dientes_total": int(es_diente.sum()),
        "resto": len(resto), "resto_total": int((~es_diente).sum()),
        "encia": len(idx_encia), "encia_total": n_encia,
        "total": len(region) + n_encia,
    }
    return np.concatenate([dientes, resto]), idx_encia, informe


def _centroides(pos: np.ndarray, fdi: np.ndarray) -> dict[str, list[float]]:
    """Centro de masas de cada diente, en el marco del PLY.

    Es lo que hace posible seleccionar por pieza **sin reparsear el PLY en el navegador**:
    catorce puntos en vez de 246.000. El visor proyecta esos catorce sobre la camara —el
    mismo mecanismo que ya usa para las cotas— y se queda con el mas cercano al clic.
    """
    return {
        str(int(c)): [round(float(v), 3) for v in pos[fdi == c].mean(axis=0)]
        for c in sorted(set(int(f) for f in np.unique(fdi)) - {0})
    }


def _largo_propio(pos: np.ndarray) -> float:
    """Extension de una nube a lo largo de SU eje mayor, en mm.

    Es la respuesta honesta a «cuanto mide esta pieza»: la direccion la pone la propia
    nube, asi que no depende de como estuviera orientado el paciente en el escaner.
    """
    if len(pos) < 3:
        return 0.0
    centrada = pos - pos.mean(axis=0)
    eje = np.linalg.svd(centrada, full_matrices=False)[2][0]
    return float(np.ptp(centrada @ eje))


def _por_pieza(pos: np.ndarray, fdi: np.ndarray, es_ios: np.ndarray) -> dict[str, dict]:
    """Lo que la GEOMETRIA dice de cada pieza, aparte de lo que diga el informe.

    Cuanto aporta cada fuente y hasta donde llega. Es la autoevaluacion del compuesto: una
    pieza con 5.000 gaussianas de corona y 200 de raiz no es la misma que una con 5.000 y
    3.000, y quien la mire tiene que poder verlo sin salir del panel.

    ⚠️ `altura_mm` es lo que se recorta HOY, no la longitud del diente. Una corona mide 7-9
    mm y un diente entero 20-25: por encima de ~26 la pieza esta arrastrando hueso, porque
    el ligamento periodontal mide 0,15-0,38 mm frente a un voxel de 0,30 y por debajo de la
    cresta osea no hay frontera que resolver. Por eso el veredicto va escrito y no se deja
    a que el que mire haga la cuenta.

    ⚠️ **Se mide sobre el eje propio de la pieza, no sobre los del mundo.** Antes era
    `ptp(pos).max()`, el lado mayor de la caja alineada con X/Y/Z, y para un diente
    inclinado eso mezcla la longitud con la anchura vestibulo-lingual: daba 26,9 mm de
    mediana donde la longitud real era 23,9, y al reves en otras piezas. Un numero que no
    mide lo que dice su nombre es peor que no tenerlo, porque el panel lo pinta en rojo
    pasado el umbral y quien lo lea creera que sabe algo.
    """
    fuera: dict[str, dict] = {}
    for codigo in sorted(set(int(f) for f in np.unique(fdi)) - {0}):
        m = fdi == codigo
        n_cor, n_rai = int((m & es_ios).sum()), int((m & ~es_ios).sum())
        alto = _largo_propio(pos[m])
        fuera[str(codigo)] = {
            "corona": n_cor,
            "raiz": n_rai,
            "altura_mm": round(alto, 1),
            "veredicto": (
                "solo corona" if alto < 12 else
                "con raíz" if alto <= 26 else "desbordada: arrastra hueso"
            ),
        }
    return fuera


def _falso_color(fdi: np.ndarray, origen: np.ndarray) -> np.ndarray:
    """RGB en [0,1] por gaussiana. **Falso color**: el CBCT no mide color."""
    rgb = np.tile(_COLOR_RESTO, (len(fdi), 1)).astype(np.float64)
    rgb[origen == ORIGEN_IOS] = _COLOR_ENCIA
    for codigo in sorted(set(int(f) for f in np.unique(fdi)) - {0}):
        cuadrante, pieza = divmod(codigo, 10)
        h = _TONOS[(cuadrante - 1) % len(_TONOS)] / 360.0
        rgb[fdi == codigo] = colorsys.hls_to_rgb(h, 0.38 + 0.05 * (pieza % 8), 0.62)
    return rgb


def _escribe_inria(
    ruta: Path,
    pos: np.ndarray,
    sigma: np.ndarray,
    rot: np.ndarray,
    densidad: np.ndarray,
    fdi: np.ndarray,
    origen: np.ndarray,
    snapshot: TwinSnapshot,
    factor_escala: float,
    capa: str,
) -> None:
    """PLY de 3DGS grado 0, con las tres conversiones declaradas en la cabecera."""
    n = len(pos)
    alfa = np.clip(1.0 - np.exp(-GANANCIA_DISPLAY * densidad), 1e-4, 1 - 1e-4)
    columnas = {
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "nx": np.zeros(n), "ny": np.zeros(n), "nz": np.zeros(n),
        # logit: el PLY de INRIA guarda la opacidad ANTES de la sigmoide.
        "opacity": np.log(alfa / (1.0 - alfa)),
        # log(sigma): el mismo nombre `scale_*` que nuestro PLY del twin, y otra escala.
        # Es justo la colision que obliga a declarar el perfil en los dos ficheros.
        "scale_0": np.log(np.maximum(sigma[:, 0], 1e-6)),
        "scale_1": np.log(np.maximum(sigma[:, 1], 1e-6)),
        "scale_2": np.log(np.maximum(sigma[:, 2], 1e-6)),
        "rot_0": rot[:, 0], "rot_1": rot[:, 1], "rot_2": rot[:, 2], "rot_3": rot[:, 3],
    }
    color = _falso_color(fdi, origen)
    for i in range(3):
        columnas[f"f_dc_{i}"] = (color[:, i] - 0.5) / C0

    # ⚠️ La cabecera de un PLY es **ASCII**, sin excepciones. Escribirla con un guion
    # largo revienta el `encode("ascii")` y tumba el canal entero: paso una vez, con un
    # `UnicodeEncodeError` que no decia nada de PLY. Aqui no se acentua ni se adorna.
    cabecera = [
        "ply", "format binary_little_endian 1.0",
        "comment perfil INRIA 3DGS grado 0 - DERIVADO de ash-twin/1.0, NO es el twin",
        f"comment acquisition_id {snapshot.acquisition_id}",
        f"comment capa: {capa}",
        f"comment opacity = logit(1 - exp(-{GANANCIA_DISPLAY:g} * sigma)) - GANANCIA DE "
        "VISUALIZACION, no es dato",
        "comment f_dc_* = falso color por codigo FDI - un CBCT NO mide color",
        "comment scale_* = log(sigma_mm); en el PLY del twin son mm lineales",
        f"comment sigma inflada x{factor_escala:.2f} para que los splats se toquen tras "
        "decimar - ES DE LA VISTA, la sigma medida esta en el PLY del twin",
        "comment el twin reversible es el PLY de field-export-agent; este es para mirar",
        f"element vertex {n}",
        *(f"property float {p}" for p in _PROPIEDADES_INRIA),
        "end_header",
    ]
    texto = "\n".join(cabecera) + "\n"
    if not texto.isascii():
        raise ValueError(
            "la cabecera del PLY tiene caracteres no ASCII: "
            + repr([c for c in texto if not c.isascii()])
        )
    datos = np.column_stack([columnas[p] for p in _PROPIEDADES_INRIA]).astype(np.float32)
    with ruta.open("wb") as f:
        f.write(texto.encode("ascii"))
        f.write(datos.tobytes())


def _fuente(ruta: str) -> str:
    """Identifica el fichero de origen **sin nombrarlo**.

    ⚠️ Aqui iba `Path(ruta).name`, y el nombre del fichero de un proveedor clinico trae el
    **nombre y los apellidos del paciente**: el paquete salio con `..._Informe_de_CBCT_...`
    precedido del paciente. Y estos ficheros son justo los que se archivan y se comparten.

    Lo que hace falta saber es de CUAL de los informes salio el hallazgo, no como se llama
    el fichero. Un hash corto lo distingue igual y no identifica a nadie; la ruta completa
    sigue en la procedencia del twin, que es interno.
    """
    return "informe:" + hashlib.sha256(ruta.encode()).hexdigest()[:8]


_RUTA = re.compile(r"(?:/[\w.\- ]+)+\.(?:pdf|stl|obj|ply|dcm|jpg|jpeg|png|txt)", re.I)


def _sin_rutas(texto: str) -> str:
    """Sustituye rutas de fichero por su hash corto. Vale para cualquier texto libre.

    ⚠️ Los motivos del gate llegan **verbatim** de los agentes, y algunos citan el fichero
    que fallo: `report-agent fallo: el informe no contiene texto extraible:
    /home/.../histora/another_patient/BRN3C...pdf`. Eso salio impreso en el panel del
    visor. Redactar solo la `fuente` de los hallazgos no bastaba — el texto libre es
    justo por donde se escapa lo que nadie reviso.

    No se borra el motivo: se le quita la ruta. Cual fichero fue sigue siendo
    identificable por el hash, que es lo mismo que usan los hallazgos.
    """
    return _RUTA.sub(lambda m: _fuente(m.group(0)), texto)


def _sidecar(
    snapshot: TwinSnapshot,
    fdi: np.ndarray,
    recorte: dict,
    motivos: list[str],
    factor_escala: float,
    centroides: dict[str, list[float]],
    capas: list[dict],
    geometria: dict[str, dict],
) -> dict:
    """La capa clinica y el `region_id` por gaussiana. Nada se interpreta aqui.

    `region_id` viaja en base64 y **en el mismo orden que el PLY**: el visor indexa por
    numero de gaussiana, asi que reordenar uno sin el otro rompe la seleccion en silencio.
    """
    dientes: dict[str, dict] = {}
    for obs in snapshot.regional:
        a = obs.attributes
        d = dientes.setdefault(obs.region_id, {"fdi": obs.region_id, "hallazgos": []})
        d["confianza"] = round(obs.provenance.confidence, 2)
        d["fuente"] = _fuente(obs.provenance.source_file)
        if a.ph is not None:
            d["ph"] = a.ph
        if a.n_raices is not None:
            d["n_raices"] = a.n_raices
        if a.n_conductos is not None:
            d["n_conductos"] = a.n_conductos
        d["hallazgos"] += [h.value for h in a.hallazgos]

    # Lo que la geometria sabe, y los motivos del gate QUE HABLAN DE ESA PIEZA.
    #
    # ⚠️ Estaban solo en la lista global, y ahi no los ve quien pincha un diente: «FDI 16:
    # el modelo lo asigno a mas de una instancia» es justo lo que hay que saber al mirar
    # el 16, y quedaba enterrado entre otros ocho motivos.
    for codigo, geo in geometria.items():
        d = dientes.setdefault(codigo, {"fdi": codigo, "hallazgos": []})
        d["geometria"] = geo
    for codigo, d in dientes.items():
        d["en_informe"] = "confianza" in d
        d["segmentado"] = codigo in geometria
        d["gate"] = [
            _sin_rutas(m) for m in motivos if m.startswith(f"FDI {codigo}:")
        ]

    return {
        "acquisition_id": snapshot.acquisition_id,
        "schema": snapshot.schema_version,
        "perfil_twin": snapshot.perfil_campo,
        "perfil_ply": "inria-3dgs/grado-0 (derivado)",
        "display": {
            "opacity": f"logit(1 - exp(-{GANANCIA_DISPLAY:g} * sigma))",
            "f_dc": "falso color por codigo FDI; un CBCT no mide color",
            "densidad_encia": DENSIDAD_DISPLAY_ENCIA,
            "factor_escala": round(factor_escala, 3),
            "por_que_factor": "sigma inflada hasta ~"
                              f"{FRACCION_ESPACIADO} del espaciado que queda tras "
                              "decimar; sin ello los splats no se tocan y la superficie "
                              "sale agujereada",
            "aviso": "opacity y f_dc_* son de la VISTA. El twin reversible es el PLY de "
                     "field-export-agent, con density sin cota y escalas en mm.",
        },
        "region_id": {
            "dtype": "int16",
            "n": int((fdi > 0).sum()),
            "orden": "el mismo que la capa `dientes`, la unica donde un FDI significa algo",
            "b64": base64.b64encode(fdi[fdi > 0].astype(np.int16).tobytes()).decode(),
        },
        # Las capas, para que el visor sepa que ficheros cargar y en que orden sin que
        # nadie los liste a mano en dos sitios.
        "capas": capas,
        "recorte": recorte,
        # Un diente puede estar segmentado y NO tener informe, o al reves. Se declaran los
        # dos lados: `centroides` son los que el modelo encontro y se pueden pinchar;
        # `dientes` los que el informe menciona. Que no coincidan es informacion clinica,
        # no un error de formato — el gate ya lo dice pieza a pieza.
        "centroides": centroides,
        "dientes": dientes,
        "medidas": [
            {"nombre": m.nombre, "valor": m.valor, "unidad": m.unidad, "lado": m.lado,
             "min": m.normal_min, "max": m.normal_max, "fuera": m.fuera_de_rango}
            for m in snapshot.medidas
        ],
        "gate": [_sin_rutas(m) for m in motivos],
        "reversibilidad": {
            "aviso": "el paquete del visor NO es reversible: esta decimado y su opacidad "
                     "y su color son interpretacion. Los ficheros reversibles son el STL "
                     "y el PLY del twin, con su desviacion medida.",
        },
        "esquema_campo": [
            {"nombre": c.nombre, "unidad": c.unidad, "escala": c.escala,
             "medido": c.medido, "derivado_de": c.derivado_de,
             "significado": c.significado}
            for c in snapshot.esquema_campo
        ],
    }
