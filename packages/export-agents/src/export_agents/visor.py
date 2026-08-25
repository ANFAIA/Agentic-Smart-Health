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

from export_agents.anatomia import (
    distancia_para_encuadrar,
    marco_anatomico,
    normaliza,
)
from export_agents.base import BaseExportAgent, ExportOutput, SurfaceStore
from export_agents.compuesto import ORIGEN_IOS

# Ganancia de la transferencia sigma → alfa: `alfa = 1 - exp(-g·sigma)`.
#
# ⚠️ **Ya no es fija: se DERIVA de `ALFA_OBJETIVO` capa a capa.** Estaba clavada en 3, que
# lleva la densidad mediana a ~0,74 de opacidad. Eso tenia sentido con splats que no se
# solapaban —cada uno tenia que verse por si mismo— y es ruinoso ahora que se apilan
# cuatro veces su espaciado: veinte splats a 0,74 no son una superficie, son un muro.
#
# Se conserva la FORMA de la transferencia, no el valor: la ganancia se ajusta para que la
# alfa mediana caiga en el objetivo, y la variacion relativa entre tejidos densos y flojos
# sigue ahi. Es lo unico que lleva informacion de las dos cosas.
#
# **No es un parametro del dato**, es de la vista, y la ganancia que salio se escribe en la
# cabecera del fichero y en el sidecar para que nadie la confunda con una medida.
GANANCIA_MAXIMA = 3.0

# Coeficiente 0 de los armonicos esfericos. Un color RGB en [0,1] se guarda como
# `(c - 0.5) / C0`; es la convencion del PLY de INRIA y el visor la deshace igual.
C0 = 0.28209479177387814

# Densidad de display de la encia. En el twin su `density` es **0**, porque el escaner mide
# forma y no atenuacion (ver `compuesto.py`); con 0 seria invisible en un rasterizador que
# espera opacidad. Se le da un valor para que se vea, y por eso vive aqui —en el canal de
# la vista— y no toca el campo.
DENSIDAD_DISPLAY_ENCIA = 0.45

# Tamano del splat como MULTIPLO del espaciado entre vecinos, y opacidad que le toca.
#
# ⚠️ **Estos dos numeros van juntos y estaban los dos mal.** Aqui habia un
# `MULTIPLO_ESPACIADO = 0.6` con un comentario afirmando que ahi «los splats se solapan
# sin emborronar». Es falso, y se ve: con 0,6 y opacidad alta cada pixel recibe UN splat y
# se dibuja como un punto. El campo se veia como polvo.
#
# Medido sobre dos PLY de 3DGS entrenado de verdad (los notebooks), con las mismas
# metricas que se aplican aqui:
#
#                      sigma_mayor/espaciado    alfa mediana
#   bite2text                     4,90              0,010
#   trained_3dgs                  3,48              0,104
#   este exportador (antes)       0,90              0,741
#
# Una superficie continua de gaussianas no se hace con splats que se tocan: se hace con
# splats GRANDES y CASI TRANSPARENTES que se apilan. Cada pixel acumula decenas y el
# alfa-blending los promedia. Solapar mas sin bajar la opacidad daria niebla solida, y por
# eso los dos numeros no se pueden tocar por separado.
# ⚠️ Con 4 se ve continuo pero BORROSO: un splat de cuatro veces el espaciado promedia
# sobre esa distancia, y el detalle fino de la corona se pierde. Bite2Text puede permitirse
# 4,9 porque su optimizador DENSIFICA donde hace falta detalle; aqui las posiciones son
# fijas —los vertices del escaner— asi que el tamano es el unico mando y hay que repartirlo
# entre continuidad y detalle. 2,5 es el compromiso.
MULTIPLO_ESPACIADO = 2.5

# Opacidad objetivo, distinta para SUPERFICIE y VOLUMEN. Un solo numero para las cuatro
# capas fue un error: la acumulacion a lo largo de un rayo no es la misma.
#
# Un rayo que atraviesa una capa de superficie —los vertices del escaner— cruza del orden
# de `pi * multiplo^2` splats, unos 20 con 2,5. Uno que atraviesa el campo del CBCT los
# cruza tambien EN PROFUNDIDAD, porque es un solido y no una hoja: del orden de
# `multiplo^3`, unos 15 veces mas. Con la misma alfa por splat, el volumen sale opaco y la
# superficie transparente — que es exactamente lo que se veia.
#
# Se resuelve `1 - (1-a)^N = 0,95` para cada N: la superficie necesita ~0,14 por splat y el
# volumen ~0,01. No son numeros de gusto, salen de cuantos se apilan.
ALFA_SUPERFICIE = 0.14
# ⚠️ Por encima del `splatAlphaRemovalThreshold` del visor, que descarta todo lo que baje
# de 5/255 = 0,0196. Con 0,02 justo encima, la mitad de las gaussianas de raices y fondo
# caian por debajo y **el visor las tiraba**: medido, el 46 %. Un objetivo de opacidad que
# coincide con el umbral de descarte del consumidor no es un objetivo, es una moneda al
# aire por gaussiana.
ALFA_VOLUMEN = 0.045

# Los vertices del escaner se escriben como DISCOS TANGENTES, no como esferas.
#
# ⚠️ Es la diferencia entre ver una superficie y ver polvo, y esta medida. Una esfera de
# radio 0,5 del espaciado deja hueco entre vecinas —se ven las bolas— y ademas reparte su
# opacidad en profundidad, hacia la camara y en contra, donde no hay nada que ensenar. Un
# disco tangente la pone toda en la superficie y embaldosa sin juntas.
#
# Es la forma a la que llega solo un 3DGS entrenado contra renders opacos: el gradiente
# aplasta las gaussianas contra la superficie visible porque es lo que reduce el error.
# Aqui no hace falta optimizar nada — **el escaner ya midio la normal de cada vertice** y
# viene en el artefacto de la malla. Se usa la medida en vez de aprenderla.
#
# El CBCT es otra cosa y no lleva discos: mide densidad en todo el interior, no hay
# superficie, y no hay normal que usar. Que se vea como una nube es honesto.
RADIO_DISCO = MULTIPLO_ESPACIADO   # del espaciado entre vertices vecinos
# Proporcion del disco: radio / grosor. **Un disco demasiado plano se ve peor que una
# esfera, no mejor**, y es contraintuitivo.
#
# Estaba en 14:1 y el resultado eran hebras: una superficie curva —la encia— se ve en buena
# parte DE CANTO, y un disco de 14:1 de canto se dibuja como una raya. El visor parecia
# pelo o llamas, no tejido.
#
# Medido sobre 3DGS entrenado de verdad, la anisotropia mediana es **2,4** en
# `bite2text_f1980_lower`. El optimizador no aplana tanto porque no le compensa: pierde mas
# de canto de lo que gana de frente. Se copia esa proporcion.
ASPECTO_DISCO = 2.4

# Falso color. El CBCT no mide color: esto es para distinguir piezas a ojo, igual que el
# `color` de las capas en el `config.ts` del visor.
#
# ⚠️ **Un tono por PIEZA, no por cuadrante.** Antes el tono salia del cuadrante y solo la
# luminosidad cambiaba entre piezas: una arcada superior entera eran dos colores —turquesa
# y verde— con siete escalones de claridad cada uno, y en pantalla eso se lee como un solo
# color. Si el color no distingue piezas no sirve para nada, porque distinguirlas es lo
# unico que hace.
#
# El tono se reparte por ANGULO AUREO sobre el indice del codigo FDI. Dos ventajas sobre
# repartir en partes iguales: piezas contiguas caen lejos en el circulo —que es justo
# cuando hace falta distinguirlas— y el color de un diente depende solo de su codigo, asi
# que la misma pieza sale del mismo color en otro caso y en otra visita. Comparar dos
# visitas lado a lado con los colores bailando seria peor que no tenerlos.
_ANGULO_AUREO = 0.6180339887498949
# Luminosidad y saturacion CONSTANTES: todas las piezas son la misma clase de cosa, y
# variarlas anadiria una dimension que no codifica nada.
_LUZ_PIEZA, _SATURACION_PIEZA = 0.46, 0.62
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

# Capa de APARIENCIA, opcional: el escaner entrenado como 3DGS. Va aparte de `CAPAS`
# porque no es lo mismo — aquellas son lo medido y esta una reconstruccion aprendida.
#
# ⚠️ **Una sola capa, y el motivo es un limite MEDIDO del visor.** `dental-3dgs-viewer`
# deja de dibujar —todas las capas, no solo la nueva— a partir de SEIS escenas: con cinco
# renderiza y con seis el lienzo sale vacio sin ningun error en consola. Partir la
# apariencia en coronas y encia daria seis y romperia tambien las cuatro medidas. La
# separacion por pieza ya la dan las capas medidas; en una capa de presentacion aporta
# poco y cuesta el paquete entero.
CAPAS_APARIENCIA = (
    ("escaner-gs", "Escaner (3DGS entrenado)",
     "APARIENCIA: reconstruida contra renders, no medida"),
)
# Los ficheros que produce `scripts/entrena_gs_escaner.py`, fundidos en esa unica capa.
_PARTES_APARIENCIA = ("coronas", "encia")

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
    version = "0.3.0"

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
        gs_apariencia: Path | None = None,
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

        encia, normales_ios, aviso_encia = self._encia(snapshot, campo)
        sel_c, sel_e, recorte = _decima(region, len(encia), semilla=semilla)

        # El escaner mide posicion Y normal, asi que sus vertices se escriben como discos
        # tangentes en vez de esferas. Ver `RADIO_DISCO`.
        espaciado_ios = (_sigma_de(encia) * 2.0) if len(encia) else 0.1
        radio = espaciado_ios * RADIO_DISCO
        grosor_ios = radio / ASPECTO_DISCO
        sig_ios, rot_ios = _discos(normales_ios[sel_e], radio, grosor_ios)

        # La inflacion es del CAMPO y solo del campo: existe porque decimar abre huecos
        # entre gaussianas del CBCT. Los discos ya salen del espaciado medido de la malla,
        # y meterlos en la mediana la arrastraria hacia abajo por su eje fino.
        factor = _factor_de_escala(centros[sel_c], escalas[sel_c])
        pos = np.vstack([centros[sel_c], encia[sel_e]])
        sig = np.vstack([escalas[sel_c] * factor, sig_ios])
        rot = np.vstack([rotaciones[sel_c], rot_ios])
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
                ruta, pos[m], sig[m], rot[m], den[m], fdi[m], origen[m],
                snapshot, factor, nombre,
            )
            capas.append({
                "id": clave, "nombre": nombre, "detalle": detalle,
                "ply": ruta.name, "primitivas": int(m.sum()),
            })
        capas += self._capas_apariencia(snapshot, campo, gs_apariencia, destination)
        encuadre, aviso_encuadre = _encuadre(pos, fdi, es_ios)
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
                         _por_pieza(pos, fdi, es_ios),
                         encuadre),
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )

        avisos = list(aviso_encia) + aviso_encuadre
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

    def _capas_apariencia(
        self, snapshot: TwinSnapshot, campo: dict,
        origen_gs: Path | None, destination: Path,
    ) -> list[dict]:
        """Capas de APARIENCIA: el escaner entrenado como 3DGS, llevado al marco del twin.

        ⚠️ **No sustituyen a nada y no se miden encima.** Son un campo entrenado contra
        renders de la malla: sus gaussianas NO son los vertices del escaner —el optimizador
        las movio, dividio y podo— asi que la correspondencia 1:1 con lo medido no existe.
        Lo que si es exacto es su codigo FDI, que viajo por el entrenamiento como parametro
        con tasa cero. Las capas medidas siguen debajo, intactas.

        El registro se aplica aqui y no al entrenar por la misma razon que la fusion no
        reescribe blobs: el fichero entrenado vive en el marco del escaner, que es donde se
        genero, y quien lo quiera en otro marco aplica la transformada que el snapshot
        declara. Asi el mismo fichero sirve para los dos y deshacerlo sigue siendo exacto.
        """
        if origen_gs is None:
            return []
        t = snapshot.provenance.transform
        if t is None:
            return []
        from fusion_agents.registration import apply, quaternion_to_matrix

        rot_reg = np.asarray(quaternion_to_matrix(t.rotation), dtype=np.float64)
        q_reg = np.asarray(t.rotation, dtype=np.float64)
        desplazamiento = np.asarray(campo.get("origin", np.zeros(3)), dtype=np.float64)

        capas: list[dict] = []
        for clave, nombre, detalle in CAPAS_APARIENCIA:
            partes = [
                _lee_inria(f) for p in _PARTES_APARIENCIA
                if (f := Path(origen_gs) / f"escaner_3dgs-{p}.ply").exists()
            ]
            if not partes:
                continue
            col = {q: np.concatenate([p[q] for p in partes]) for q in _PROPIEDADES_INRIA}
            pos = np.column_stack([col["x"], col["y"], col["z"]])
            pos = apply(rot_reg, np.asarray(t.translation, dtype=np.float64), pos)
            col["x"], col["y"], col["z"] = (pos - desplazamiento).T
            q = np.column_stack([col[f"rot_{i}"] for i in range(4)])
            q = _compone_quats(np.tile(q_reg, (len(q), 1)), q)
            for i in range(4):
                col[f"rot_{i}"] = q[:, i]

            ruta = destination.with_name(f"{destination.name}-{clave}.ply")
            _escribe_ply(ruta, col, [
                "ply", "format binary_little_endian 1.0",
                "comment perfil INRIA 3DGS grado 0 - APARIENCIA, no medida",
                "comment entrenado contra renders de la malla del escaner; las gaussianas",
                "comment NO son sus vertices. El twin medido va en las otras capas.",
                "comment llevado al marco del twin con la transformada del registro",
                f"element vertex {len(pos)}",
                *(f"property float {q_}" for q_ in _PROPIEDADES_INRIA),
                "end_header",
            ])
            capas.append({
                "id": clave, "nombre": nombre, "detalle": detalle,
                "ply": ruta.name, "primitivas": int(len(pos)), "apariencia": True,
            })
        return capas

    def _encia(
        self, snapshot: TwinSnapshot, campo: dict
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Vertices del escaner y sus NORMALES en el marco del twin, o vacio y el motivo."""
        if snapshot.surface_ref is None:
            return np.empty((0, 3)), np.empty((0, 3)), []
        if snapshot.provenance.transform is None:
            return np.empty((0, 3)), np.empty((0, 3)), [
                "el snapshot trae escáner pero no transformación: sin registro no se "
                "puede poner la encía en el mismo sistema que los dientes, así que el "
                "paquete del visor lleva solo el campo del CBCT."
            ]
        from export_agents.compuesto import _al_marco_del_twin
        from export_agents.stl import quaternion_to_matrix

        malla = self.store.load(snapshot.surface_ref)
        v = np.asarray(malla["positions"], dtype=np.float64)
        encia = _al_marco_del_twin(v, snapshot.provenance.transform)
        if "origin" in campo:
            encia = encia - np.asarray(campo["origin"], dtype=np.float64)

        # Las normales solo ROTAN: son direcciones, no puntos. Aplicarles la traslacion
        # las sacaria del origen y dejarian de ser normales.
        n = np.asarray(
            malla.get("normals", np.zeros_like(v)), dtype=np.float64
        )
        rot = quaternion_to_matrix(snapshot.provenance.transform.rotation)
        return encia, n @ np.asarray(rot, dtype=np.float64).T, []


def _discos(
    normales: np.ndarray, radio: float, grosor: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(escalas, cuaterniones)` que convierten cada vertice en un disco tangente.

    El tercer eje local va a lo largo de la normal y es el fino, asi que `scale_2` es el
    grosor y los otros dos el radio en el plano tangente. El cuaternion es el giro de arco
    minimo de `+z` a la normal — el que no introduce rotacion alrededor de ella, que aqui
    no significaria nada porque el disco es simetrico.
    """
    n = np.asarray(normales, dtype=np.float64)
    norma = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norma, out=np.tile([0.0, 0.0, 1.0], (len(n), 1)), where=norma > 0)

    w = 1.0 + n[:, 2]
    q = np.column_stack([w, -n[:, 1], n[:, 0], np.zeros(len(n))])
    # Normal antipodal a +z: el arco minimo es ambiguo y `w` se anula. Media vuelta por
    # cualquier eje perpendicular sirve, y se elige uno fijo para que sea reproducible.
    opuesta = w < 1e-8
    q[opuesta] = [0.0, 1.0, 0.0, 0.0]
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    escalas = np.tile([radio, radio, grosor], (len(n), 1))
    return escalas, q


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
    return max(1.0, MULTIPLO_ESPACIADO * espaciado / actual)


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


# Campo vertical de la camara del visor. Lo fija `gaussian-splats-3d`
# (`THREE_CAMERA_FOV = 50`), no nosotros: la distancia que se emite abajo solo encuadra si
# es la misma, asi que se declara en el sidecar en vez de quedarse aqui de constante muda.
FOV_VISOR = 50.0

# Ni pegado ni perdido: la arcada ocupa el encuadre dejando sitio a las cotas y al panel.
MARGEN_ENCUADRE = 1.25


def _encuadre(
    pos: np.ndarray, fdi: np.ndarray, es_ios: np.ndarray
) -> tuple[dict | None, list[str]]:
    """El encuadre inicial del visor, MEDIDO sobre los ejes anatomicos de la arcada.

    ⚠️ **Lo que arregla es la orbita, no la primera imagen.** El visor gira alrededor de su
    `cameraUp`, y hasta ahora eso era un eje del mundo escrito a mano en el `config.ts`
    (`[0, 0, 1]`). Cuando el eje oclusal de la arcada no coincide con el, girar el raton
    no da la vuelta a la arcada: la vuelca. Con el eje medido, la orbita gira por donde un
    clinico espera y se puede llegar a cualquier cara de cualquier pieza.

    ⚠️ **Se mide sobre los vertices del ESCANER y solo ellos.** En el compuesto,
    `fdi == 0` incluye la encia del escaner *y* el hueso y el craneo del CBCT; meter
    aquello arrastraria el centroide de «lo no dental» hacia arriba y podria invertir el
    signo del eje oclusal. La encia es lo que hay al otro lado de las coronas, y es lo que
    da el signo.
    """
    if not es_ios.any():
        return None, [
            "el paquete del visor no lleva escaner, asi que no hay con que medir los ejes "
            "anatomicos: el encuadre y la orbita se quedan en los del `config.ts`"
        ]
    marco, motivo = marco_anatomico(pos[es_ios], fdi[es_ios])
    if marco is None:
        return None, [
            f"el encuadre del visor no se pudo medir ({motivo}), asi que la orbita se "
            "queda en el eje del mundo que el `config.ts` traiga escrito a mano"
        ]
    # Frontal ligeramente desde arriba: es como se mira una arcada, y ademas deja la
    # direccion de vista bien separada del eje de orbita.
    direccion = normaliza(marco.anterior * 0.85 + marco.superior * 0.4)
    return {
        "centro": [round(float(x), 3) for x in marco.centro],
        # ⚠️ El eje de ORBITA es el SUPERIOR, no el oclusal. En un maxilar son opuestos —
        # las coronas cuelgan hacia abajo— y usar el oclusal pone la cabeza boca abajo: la
        # arcada superior se ve exactamente como una inferior, y nadie que no conozca el
        # caso lo nota. Se midio, se emitio el eje equivocado, y se vio en el visor.
        "arriba": [round(float(x), 4) for x in marco.superior],
        "direccion": [round(float(x), 4) for x in direccion],
        "distancia": round(distancia_para_encuadrar(
            pos[es_ios], marco.centro, direccion,
            fov_grados=FOV_VISOR, margen=MARGEN_ENCUADRE,
        ), 1),
        "fov_grados": FOV_VISOR,
        "arcada": marco.arcada,
        "ejes": {
            "oclusal": [round(float(x), 4) for x in marco.oclusal],
            "superior": [round(float(x), 4) for x in marco.superior],
            "derecha": [round(float(x), 4) for x in marco.derecha],
            "anterior": [round(float(x), 4) for x in marco.anterior],
        },
        "medido": (
            "ejes anatomicos deducidos de las etiquetas FDI del escaner, no de los ejes "
            "del fichero: oclusal de la encia a las coronas, superior hacia la coronilla "
            "(opuesto al oclusal en el maxilar), derecha de los cuadrantes 2 y 3 a los "
            "1 y 4, anterior de los molares a los incisivos"
        ),
    }, []


def tono_de(codigo: int) -> float:
    """Tono en [0,1) de una pieza, a partir de su codigo FDI y de nada mas.

    El indice es `(cuadrante-1)*8 + (pieza-1)`, o sea la posicion del diente en la boca, y
    el tono sale de multiplicarlo por el angulo aureo. Depende solo del codigo: la misma
    pieza sale del mismo color en cualquier caso y en cualquier visita.
    """
    cuadrante, pieza = divmod(int(codigo), 10)
    return ((cuadrante - 1) * 8 + (pieza - 1)) * _ANGULO_AUREO % 1.0


def _falso_color(fdi: np.ndarray, origen: np.ndarray) -> np.ndarray:
    """RGB en [0,1] por gaussiana. **Falso color**: el CBCT no mide color."""
    rgb = np.tile(_COLOR_RESTO, (len(fdi), 1)).astype(np.float64)
    rgb[origen == ORIGEN_IOS] = _COLOR_ENCIA
    for codigo in sorted(set(int(f) for f in np.unique(fdi)) - {0}):
        rgb[fdi == codigo] = colorsys.hls_to_rgb(
            tono_de(codigo), _LUZ_PIEZA, _SATURACION_PIEZA
        )
    return rgb


def _ganancia(densidad: np.ndarray, origen: np.ndarray) -> float:
    """Ganancia que lleva la alfa MEDIANA de esta capa a su objetivo.

    El objetivo depende de si la capa es una SUPERFICIE (vertices del escaner) o un VOLUMEN
    (campo del CBCT), porque no se apilan igual a lo largo de un rayo. Ver `ALFA_SUPERFICIE`.
    """
    es_superficie = len(origen) and float((origen == ORIGEN_IOS).mean()) > 0.5
    objetivo = ALFA_SUPERFICIE if es_superficie else ALFA_VOLUMEN
    mediana = float(np.median(densidad)) if len(densidad) else 0.0
    if mediana <= 0:
        return GANANCIA_MAXIMA
    return min(GANANCIA_MAXIMA, -np.log(1.0 - objetivo) / mediana)


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
    ganancia = _ganancia(densidad, origen)
    alfa = np.clip(1.0 - np.exp(-ganancia * densidad), 1e-4, 1 - 1e-4)
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
        f"comment opacity = logit(1 - exp(-{ganancia:.4g} * sigma)) - GANANCIA DE "
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
    _escribe_ply(ruta, columnas, cabecera)


def _escribe_ply(ruta: Path, columnas: dict, cabecera: list[str]) -> None:
    """Vuelca las 17 propiedades del perfil INRIA. La cabecera la compone quien llama."""
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


def _lee_inria(ruta: Path) -> dict[str, np.ndarray]:
    """Lee un PLY del perfil INRIA (17 propiedades, binario little-endian)."""
    import re

    b = ruta.read_bytes()
    fin = b.index(b"end_header\n") + 11
    cab = b[:fin].decode("ascii", errors="replace")
    n = int(re.search(r"element vertex (\d+)", cab).group(1))  # type: ignore[union-attr]
    props = re.findall(r"property float (\w+)", cab)
    if tuple(props) != _PROPIEDADES_INRIA:
        raise ValueError(
            f"{ruta.name} no esta en el perfil INRIA que este canal espera: "
            f"{len(props)} propiedades en otro orden."
        )
    a = np.frombuffer(b, dtype=np.float32, count=n * len(props), offset=fin)
    return {q: a.reshape(n, len(props))[:, i].astype(np.float64)
            for i, q in enumerate(props)}


def _compone_quats(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a ⊗ b` en convencion (w, x, y, z). Girar la nube gira TAMBIEN los elipsoides.

    Es el paso que se olvida: llevar un campo gaussiano a otro marco no es mover los
    centros. Una gaussiana anisotropa tiene orientacion, y sin componer el cuaternion del
    registro con el suyo, las elipses se quedan apuntando a donde apuntaban antes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


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
    encuadre: dict | None,
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
            "opacity": (f"logit(1 - exp(-g * sigma)), g por capa: alfa~{ALFA_SUPERFICIE:g} "
                        f"en superficie, ~{ALFA_VOLUMEN:g} en volumen"),
            "f_dc": "falso color por codigo FDI; un CBCT no mide color",
            "densidad_encia": DENSIDAD_DISPLAY_ENCIA,
            "factor_escala": round(factor_escala, 3),
            "por_que_factor": "sigma inflada hasta ~"
                              f"{MULTIPLO_ESPACIADO} del espaciado que queda tras "
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
        # El encuadre y el eje de orbita, MEDIDOS. Si falta, el visor se queda con lo que
        # su `config.ts` traiga escrito a mano — que es de donde venimos.
        **({"encuadre": encuadre} if encuadre else {}),
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
