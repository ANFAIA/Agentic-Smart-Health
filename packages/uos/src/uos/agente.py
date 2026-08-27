"""`uos-export-agent`: `TwinSnapshot` → `.uos`. El sexto canal de exportacion.

**Que aporta sobre los otros cinco.** Los demas materializan UNA cosa —la malla, el campo,
el compuesto, el paquete del visor, el render—. Este empaqueta el caso entero con sus
relaciones declaradas: que asset viene de que visita, en que frame vive cada uno, y con
que transformada se alinean. Es la diferencia entre entregar ficheros y entregar una
escena.

**El frame canonico es el ESCANER, no el CBCT** (§2.2 del spec), y eso invierte lo que
hace el pipeline. Aqui se trabaja centrado en el CBCT porque es donde vive el campo
gaussiano; UOS pone el escaner de hub porque es la geometria de referencia de un caso
dental —la mide un escaner optico con exactitud de micras, no un volumen de 0,3 mm—. La
inversion se hace aqui, en el borde, y se DECLARA como registracion en vez de reescribir
nada: el `.uos` lleva `reg.ct_to_ios` con su matriz, su metodo y su error.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from core_schemas import ModalityStatus, TwinSnapshot
from export_agents.base import BaseExportAgent, ExportOutput
from export_agents.field import esquema_de_propiedades

from uos.clinico import OBSERVACIONES, capa_clinica
from uos.contenedor import (
    asset_de,
    asset_de_bytes,
    asset_de_directorio,
    escribe_uos,
    json_de,
)
from uos.derivados import (
    SEGMENTACION,
    SEGMENTACION_META,
    codifica_etiquetas,
    meta_segmentacion,
    sha256_de_fichero,
)
from uos.escena import MEDIA_GLB, NodoGS, construye_glb, lee_stl_binario
from uos.manifiesto import (
    MEDIA_TYPE,
    UOS_VERSION,
    Adquisicion,
    Clase,
    EstadoPHI,
    Extension,
    Frame,
    Manifiesto,
    Procedencia,
    Proyeccion,
    RecursoFHIR,
    Registro,
    Regulatorio,
    Sujeto,
    Visita,
)
from uos.procedencia import CADENA, encadena, lee_version_previa
from uos.vistas import VISTAS, Vista, construye_vistas
from uos.volumen import SIDECAR, describe_serie, identificables_en

# A que recurso FHIR R4 corresponde cada clase de asset (§9). El conector con el PMS
# —Open Dental primero— necesita saber QUE crear, y eso se puede decir sin servidor.
#
# ⚠️ Lo que NO se declara es una referencia concreta. El ejemplo del spec escribe
# `ImagingStudy/is-9911`, un recurso que existe en algun sitio; este caso no ha pasado por
# ningun PMS y no hay identificador que citar. Inventar uno haria que un conector intentara
# resolverlo. Se afirma el tipo, que es verdad hoy, y se deja `resource` vacio.
_RECURSO = {
    # El spec lo fija: el `.uos` entero se publica como adjunto con su media type (§9).
    Clase.IMAGE2D: ("Media", "foto clinica; `Media` es el recurso de imagen no-DICOM"),
    Clase.VOLUME: ("ImagingStudy", "serie DICOM intacta"),
    Clase.MESH_GS_SCENE: (
        "DocumentReference",
        "malla y apariencia 3D: FHIR R4 no tiene recurso para geometria dental, y `Media` "
        "es para foto, video y audio. `DocumentReference` es el sobre generico de binarios "
        "clinicos, que es lo que son",
    ),
    Clase.DOCUMENT: ("DocumentReference", "informe u otro documento del caso"),
    Clase.DERIVED_SEG: (
        "Observation",
        "salida de inferencia: no es una adquisicion, es una lectura sobre ella",
    ),
}

FRAME_IOS = "frame.ios_master"
FRAME_CBCT = "frame.ct_001"

_MEDIA = {
    ".stl": "model/stl",
    ".ply": "application/octet-stream",
    ".glb": "model/gltf-binary",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}


def _calidad_frontera(malla: Any, etq: Any) -> dict[int, dict] | None:
    """Por pieza, si su recorte esta dentro de lo que hacen las etiquetas de experto.

    Devuelve `None` si no se puede calcular, que es distinto de «todas mal»: el sidecar no
    declara el bloque en vez de declararlo vacio.
    """
    try:
        import numpy as _np
        from analysis_agents.frontera import calidad_por_pieza

        caras = _np.asarray(malla.get("faces"))
        if caras is None or caras.size == 0:
            return None
        return calidad_por_pieza(_np.asarray(malla["positions"], float), caras, etq)
    except Exception:
        return None


class UOSExportAgent(BaseExportAgent):
    """Empaqueta el twin como Unified Oral Scene.

    ⚠️ **Referencia, no transcodifica** (§2.1). Los ficheros entran tal cual y su `sha256`
    va en el manifiesto: lo que sale del `.uos` es byte-identico a lo que entro. Es lo que
    permite decir que el contenedor no degrada nada, y lo que hace la procedencia
    verificable por quien lo reciba sin confiar en nosotros.
    """

    name = "uos-export-agent"
    version = "0.4.0"

    def __init__(self, store: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store = store

    def _export(  # type: ignore[override]
        self,
        snapshot: TwinSnapshot,
        destination: Path,
        *,
        pseudonimo: str | None = None,
        malla: Path | None = None,
        escena_gs: Path | None = None,
        campo: Path | None = None,
        compuesto: Path | None = None,
        imagenes: list[Path] | None = None,
        # ⚠️ **Los informes del caso, tal como llegaron.** Antes NO viajaba ninguno: el
        # `report-agent` los leia, sacaba lo que podia y el PDF se quedaba fuera. Para un
        # informe tabulado eso casi no se nota —la transcripcion lleva lo que decia—, pero
        # para uno ESCANEADO se pierde entero: el gate dice «hay un PDF que nadie pudo
        # leer» y el documento no esta en ninguna parte para que alguien lo lea.
        #
        # El caso que lo destapo: un pasaporte de implantes escaneado, con tres implantes
        # (marca, referencia, lote, fecha y posicion FDI) que NO aparecen en ningun otro
        # documento del caso. Era la unica constancia, y se caia — ni dentro ni declarado.
        #
        # Siguen `sin_originales` como todo lo demas: en el perfil ligero se declaran por
        # su `sha256` y los custodia otro sistema. Lo que se arregla aqui no es que viajen
        # SIEMPRE, es que EXISTAN en el manifiesto.
        informes: list[Path] | None = None,
        motivos: list[str] | None = None,
        etiquetas_ios: Any | None = None,
        modelo_segmentacion: Path | None = None,
        # Version declarada del segmentador. `None` -> el sidecar dice `null`.
        version_segmentador: str | None = None,
        cbct: Path | None = None,
        # ⚠️ **Perfil ligero: los originales se DECLARAN y no viajan.** El `.uos` sale con
        # el campo gaussiano, la escena y el manifiesto; el STL del escaner y la serie
        # DICOM quedan como assets `external`, con su `uri` logica y su `sha256`, y quien
        # los custodia es otro sistema.
        #
        # Lo que cambia es la garantia, no la forma: con ellos dentro el contenedor afirma
        # «el DICOM que sale es el que entro» y el validador lo comprueba; sin ellos afirma
        # «se el hash de lo que deberia haber ahi». Sigue siendo auditable y ya NO cumple el
        # ⚠️ **Ningun fichero original viaja dentro. Es el formato, no un perfil.**
        # El encargo del proyecto es «regenerar ficheros STL e imagenes a partir del Digital
        # Twin»: regenerar, no transportar. Un contenedor que lleva el escaner dentro no
        # demuestra reversibilidad —demuestra que sabe copiar un fichero—, y ademas
        # multiplica su peso por lo que ya esta representado en el gemelo.
        #
        # Lo que viaja de cada original es su DIRECCION DE CONTENIDO: `uri` pasa a ser
        # `sha256:<hex>` y `external` a `true`. Quien reciba el contenedor puede comprobar
        # que el fichero que le den es exactamente el que se ingirio, sin que el fichero
        # tenga que estar dentro.
        #
        # ⚠️ **Y esto cambia la GARANTIA, asi que se declara.** Con los originales dentro el
        # contenedor afirma «lo que sale es lo que entro» y el validador lo comprueba (§1.1
        # del spec); sin ellos afirma «se el hash de lo que deberia haber ahi». El validador
        # avisa por cada asset externo, que es lo correcto: es una diferencia real y quien
        # reciba el fichero tiene que verla.
        sin_originales: bool = True,
        # ⚠️ **Perfil de solo gaussianas: fuera tambien la malla convertida.** `scene.glb`
        # no es un original —es presentacion, el STL convertido a float32— pero sigue
        # siendo una malla, y la decision de producto es que el contenedor lleve el campo
        # gaussiano y el manifiesto.
        #
        # Tiene una consecuencia que hay que ver antes de activarlo: `derived/seg_teeth`
        # indexa los VERTICES de esa malla. Sin ella, esa segmentacion no indexa nada, asi
        # que tampoco viaja — el FDI tiene que ir entonces por gaussiana, en el propio
        # campo. Por eso las dos cosas van juntas y no por separado.
        sin_malla: bool = False,
        previo: Path | None = None,
        # Quien calculo la registracion, para el `operator` de §6. Lo sabe el orquestador
        # —tiene la salida de la fusion geometrica— y este agente no puede deducirlo.
        registrador: str | None = None,
        # El campo ajustado (gaussian-engine) y su informe de ajuste. Va APARTE en el
        # `.uos` como `asset.field_fit`, sin sustituir la semilla del snapshot.
        # `campo_ajustado` es un ref (hash/URI del almacén); `ajuste` es un dataclass
        # `Ajuste` con `rmse_hu`, `rmse_hu_por_region`, `compresion`.
        campo_ajustado: str | None = None,
        ajuste: Any | None = None,
        # El descriptor del campo ajustado (dict plano, construido fuera de este paquete
        # para no acoplarlo a `gaussian_engine`). Se vuelca tal cual en el sidecar.
        campo_ajustado_descriptor: dict | None = None,
    ) -> ExportOutput:
        if not pseudonimo:
            # ⚠️ **No se cae al `acquisition_id`**, y es deliberado: ese identificador sale
            # del nombre del directorio del caso, que en un sistema real lleva el nombre
            # del paciente o su numero de historia. Un seudonimo por defecto que resulta
            # ser el dato identificable es peor que no tener seudonimo, porque el
            # `phi_state` diria `pseudonymized` mintiendo.
            return self._outcome(
                ModalityStatus.FAILED,
                detail=(
                    "No se aporto seudonimo del paciente. UOS declara `phi_state`, y "
                    "declararlo `pseudonymized` sin un seudonimo de verdad seria afirmar "
                    "algo falso sobre datos identificables."
                ),
                format="uos",
            )
        if malla is None or not malla.exists():
            # UOS-Core exige `mesh_gs_scene`, y sin malla no hay frame canonico: el hub
            # geometrico ES el escaner. Sin el, todo lo demas queda sin ancla.
            return self._outcome(
                ModalityStatus.MISSING,
                detail=(
                    "Sin malla del escaner no hay frame canonico que declarar: en UOS el "
                    "hub geometrico es el escaner intraoral (spec §2.2). No se empaqueta."
                ),
                format="uos",
            )

        visita = Visita(id="v1", date=snapshot.timestamp.date().isoformat(),
                        label="Baseline")
        ficheros: dict[str, Path] = {}
        extras_escena: dict[str, bytes | str] = {}
        aviso_derivados: list[str] = []
        assets = []

        # ⚠️ El nombre del fichero NO viaja, tampoco el de la malla. Los de un proveedor
        # llevan identificadores —el que produjo este caso se llamaba "1574 UpperJawScan"—
        # y `1574` es el numero de caso. Se nombra por su papel en la escena y la
        # trazabilidad la da el sha256, que es mas fuerte que un nombre.
        uri = f"scene/scan{malla.suffix.lower()}"
        if not sin_originales:
            ficheros[uri] = malla
        # ⚠️ `document`, no `mesh_gs_scene`. Lo dice el §5.1: la escena es el `.glb`, y el
        # STL original «PUEDE incluirse como asset document para trazabilidad». Declararlo
        # escena haria que un visor viera dos escenas y no supiera cual montar.
        assets.append(asset_de(
            malla, uri, id_="asset.ios", kind=Clase.DOCUMENT, visit=visita.id,
            frame=FRAME_IOS, media_type=_MEDIA.get(malla.suffix.lower(), "model/stl"),
            acquisition=Adquisicion(time=snapshot.timestamp),
            external=sin_originales,
        ))
        # ⚠️ **La ESCENA, ademas del STL.** El §3.1 dibuja `scene/scene.glb` como «STL
        # convertido» y el §1.1 dice que UOS no re-encodea datos fuente: las dos cosas solo
        # son compatibles si el convertido es presentacion y el original se queda. glTF hace
        # falta porque un STL no puede llevar ni indices ni atributos, y el §11.3 pide una
        # malla a la que colgarle cosas. Se construye desde la malla INGERIDA para que el
        # orden de vertices —y con el las etiquetas— se conserve.
        # ⚠️ **La registracion se calcula ANTES que la escena**, y no es un detalle de
        # orden: el §5.1 pide que la relacion GS→malla vaya codificada como `matrix` del
        # nodo de gaussianas colgado bajo el de la malla. Sin la transformada no se puede
        # construir la escena, solo un `.glb` con la malla suelta.
        registros = self._registros(snapshot, registrador)
        al_canonico = registros[0].transform_4x4_row_major if registros else None
        nodos_gs: list[NodoGS] = []

        # Las capas de gaussianas. Son tres cosas distintas con el mismo `kind`, asi que
        # cada una lleva su descriptor: el campo es densidad MEDIDA, el compuesto es medida
        # de dos modalidades, y la apariencia es reconstruccion contra renders.
        #
        # ⚠️ **`escena_gs` se salta cuando `apariencia_ref` está set.** En ese caso la
        # capa de apariencia la gestiona el bloque `asset.apariencia` más abajo, con
        # el esquema INRIA y el perfil correctos. Si no lo saltamos, el main loop crea
        # `asset.gs` con el esquema de densidad (porque `_descriptor_gs` usa los defaults
        # del snapshot) y el sidecar queda con `profile: ash-twin/1.0` en vez de
        # `ash-gs-apariencia/1.0`.
        _skip_escena_gs = (
            snapshot.apariencia_ref is not None
            and escena_gs is not None
        )
        for ruta, id_, papel, medido, marco, nota in (
            (campo, "asset.field", "campo gaussiano del twin", True, FRAME_CBCT,
             "densidad MEDIDA por el CBCT: `density` es sigma normalizada, no opacidad, y "
             "las escalas van en milimetros lineales, NO en logaritmo"),
            (compuesto, "asset.composite", "compuesto CBCT + escaner", True, FRAME_CBCT,
             "dos modalidades en un fichero, con una columna `origen` por gaussiana. La "
             "encia lleva `density = 0` porque el escaner no mide atenuacion"),
            (escena_gs, "asset.gs", "apariencia del escaner", False, FRAME_IOS,
             "reconstruida entrenando 3DGS contra renders de la malla, NO medida. Su "
             "color y su opacidad son del modelo, no del paciente"),
        ):
            if ruta is None or not ruta.exists():
                continue
            # Si hay `apariencia_ref`, el bloque `asset.apariencia` gestiona esta capa
            # con el esquema INRIA correcto — no la procesamos aquí con los defaults.
            if _skip_escena_gs and id_ == "asset.gs":
                continue
            # El nombre dentro del contenedor describe QUE es, no de que variable sale.
            # `asset.gs` daria `scene/gs.ply`, que no dice nada a quien lo abra.
            corto = {"asset.gs": "appearance"}.get(id_, id_.split(".")[1])
            uri = f"scene/{corto}{ruta.suffix.lower()}"
            descriptor = f"scene/{corto}.gs.json"
            ficheros[uri] = ruta
            # Para el campo semilla, incluir info de submuestreo en el sidecar si el
            # artefacto la trae. Así el consumidor sabe cuántos vóxeles había antes.
            submuestreo = None
            if id_ == "asset.field" and self.store is not None:
                submuestreo = self._lee_submuestreo(snapshot)
            # ⚠️ **`n_primitives` sale del FICHERO, no del snapshot.** El snapshot lleva el
            # numero del campo semilla, y el compuesto trae ademas las gaussianas del
            # escaner: el sidecar declaraba 1.341.990 sobre un fichero de 1.454.057. Un
            # descriptor describe lo que tiene delante o no describe nada.
            _u, _n, _props = self._cabecera_ply(ruta)
            # ⚠️ **El esquema tambien sale del fichero, no del snapshot.** `esquema_campo`
            # describe el campo SEMILLA; el compuesto trae ademas `origen` —de que
            # modalidad viene cada gaussiana—, que viajaba en los bytes y no en el
            # descriptor. Un lector ajeno no podia separar el CBCT del escaner dentro de un
            # fichero cuyo unico motivo de existir es mezclar los dos. Es el mismo fallo
            # que aqui ya se arreglo para `n_primitives`, en la lista de columnas.
            _esq = esquema_de_propiedades(_props) if _props else None
            extras_escena[descriptor] = json_de(self._descriptor_gs(
                snapshot, papel=papel, medido=medido, marco=marco, nota=nota,
                submuestreo=submuestreo,
                esquema_override=_esq,
                n_primitives_override=_n,
                unidades_override=_u,
            ))
            assets.append(asset_de(
                ruta, uri, id_=id_, kind=Clase.MESH_GS_SCENE, visit=visita.id,
                frame=marco, media_type="application/octet-stream",
                # El orden de carga del §4.1: malla 10 -> fotos 20 -> GS 25 -> volumen 30.
                load_priority=25, sidecar_uri=descriptor,
                # La capa de apariencia es DERIVADA (entrenada contra renders) y va en
                # `scene/` con `layer=1`, no en `derived/` (que es Layer 3, inferencia
                # clínica). El campo semilla y el compuesto son `raw` por defecto.
                **({"regulatory": Regulatorio(layer=1, status="derived")}
                   if id_ == "asset.gs" else {}),
            ))
            nodos_gs.append(NodoGS(
                uri=uri, nombre=papel,
                # Lo que esta en el marco del CBCT necesita la transformada al canonico;
                # la apariencia ya vive en el del escaner, que ES el canonico.
                matriz_fila=(al_canonico if marco == FRAME_CBCT else None),
                extras={"uos_descriptor_uri": descriptor, "uos_measured": medido},
            ))

        # ── Campo ajustado (gaussian-engine) ──────────────────────────────────
        # El ajuste optimiza el campo semilla contra la densidad medida. El resultado
        # viaja APARTE en el `.uos` como `asset.field_fit`, sin sustituir la semilla.
        # Razón: el twin reversible lleva la semilla (que es dato medido); el ajustado
        # es DERIVADO (optimización numérica, no una medición nueva) y va en `scene/`
        # con `regulatory.layer=1, status="derived"` — no en `derived/`, que es Layer 3
        # (inferencia clínica con modelo entrenado).
        #
        # `campo_ajustado` es un ref (hash/URI del almacén); `campo_ajustado_descriptor`
        # es un dict plano construido en `caso_completo.py` con `gaussian_engine.esquema`
        # y `gaussian_engine.PERFIL` — este paquete no importa `gaussian_engine`.
        if (campo_ajustado is not None and campo_ajustado_descriptor is not None
                and self.store is not None):
            try:
                datos_aj = self.store.load(campo_ajustado)
                uri_fit = "scene/field_fit.ply"
                ficheros[uri_fit] = self._escribe_ply(
                    destination / uri_fit, datos_aj,
                )
                descriptor_fit = "scene/field_fit.gs.json"
                extras_escena[descriptor_fit] = json_de(campo_ajustado_descriptor)
                assets.append(asset_de(
                    destination / uri_fit, uri_fit,
                    id_="asset.field_fit",
                    kind=Clase.MESH_GS_SCENE, visit=visita.id,
                    frame=FRAME_CBCT, media_type="application/octet-stream",
                    load_priority=25, sidecar_uri=descriptor_fit,
                ))
                nodos_gs.append(NodoGS(
                    uri=uri_fit,
                    nombre="campo ajustado contra densidad medida",
                    matriz_fila=al_canonico,
                    extras={
                        "uos_descriptor_uri": descriptor_fit,
                        "uos_measured": False,
                    },
                ))
            except (KeyError, OSError, ValueError) as e:
                aviso_derivados.append(
                    f"campo ajustado no incluido en el `.uos`: {e}"
                )

        # ── Apariencia entrenada (gsplat contra fotos) ─────────────────────
        # El PLY INRIA con color real del paciente, entrenado contra renders de
        # Blender. Viaja como `asset.apariencia` con `layer=1, status="derived"`.
        # El esquema de columnas es INRIA (f_dc_*, opacity, scale en log), NO
        # el del campo de densidad — por eso pasamos `esquema_override`.
        #
        # ⚠️ **Los arrays en el almacén** (`means`, `quats`, `opacities`, `colors`)
        # no coinciden con las claves de `_escribe_ply` (`centers`, `rotations`,
        # `density`). Usamos `escribe_inria` que conoce el formato INRIA de
        # primera mano — el mismo que escribe el PLY en el pipeline.
        if (snapshot.apariencia_ref is not None and self.store is not None):
            try:
                from gaussian_engine.agente_apariencia import esquema_apariencia
                from gaussian_engine.apariencia import (
                    escribe_inria as _escribe_inria,
                )
                datos_ap = self.store.load(snapshot.apariencia_ref)
                uri_ap = "scene/appearance.ply"
                destino_ap = destination / uri_ap
                destino_ap.parent.mkdir(parents=True, exist_ok=True)
                # Escribir el PLY INRIA directamente con los arrays del almacén.
                _escribe_inria(
                    destino_ap, datos_ap,
                    n_vistas=datos_ap.get("n_vistas", 0),
                    iteraciones=datos_ap.get("iteraciones", 0),
                )
                ficheros[uri_ap] = destino_ap
                _u_ap, _n_ap, _props_ap = self._cabecera_ply(destino_ap)
                # ⚠️ El esquema se deriva de las propiedades QUE TRAE EL PLY recien
                # escrito, no de una lista fija. `escribe_inria` emite `region_id` solo
                # cuando hay segmentacion, y la lista fija no lo declaraba nunca: el
                # codigo FDI viajaba en los bytes y no en el sidecar.
                esq_ap = esquema_apariencia(_props_ap)
                descriptor_ap = "scene/appearance.gs.json"
                extras_escena[descriptor_ap] = json_de(self._descriptor_gs(
                    snapshot,
                    papel="apariencia real entrenada con gsplat",
                    medido=False,
                    marco=FRAME_IOS,
                    # ⚠️ **La nota se LEE del PLY, no se escribe aqui.** Este literal
                    # describia el color como «un degradado de DOS tonos interpolado por
                    # altura z» mucho despues de que el color pasara a medirse corona a
                    # corona: la cabecera del fichero decia una cosa y su propio sidecar
                    # otra, y el panel del visor mostraba la vieja. Es el mismo fallo que
                    # ya se arreglo dos veces aqui —las unidades y el esquema— y la misma
                    # cura: quien describe, pregunta al fichero.
                    nota=self._nota_color_ply(destino_ap),
                    esquema_override=esq_ap,
                    perfil_override="ash-gs-apariencia/1.0",
                    # Del FICHERO, no de `datos_ap`: el optimizador divide y poda, asi que
                    # el numero de gaussianas escritas no es el de la semilla que se le dio.
                    n_primitives_override=(_n_ap or len(datos_ap["means"])),
                    unidades_override=_u_ap,
                ))
                assets.append(asset_de(
                    destination / uri_ap, uri_ap,
                    id_="asset.apariencia",
                    kind=Clase.MESH_GS_SCENE, visit=visita.id,
                    frame=FRAME_IOS, media_type="application/octet-stream",
                    load_priority=25, sidecar_uri=descriptor_ap,
                    regulatory=Regulatorio(layer=1, status="derived"),
                ))
                nodos_gs.append(NodoGS(
                    uri=uri_ap,
                    nombre="apariencia real entrenada con gsplat",
                    matriz_fila=None,  # ya en frame canónico (escáner)
                    extras={
                        "uos_descriptor_uri": descriptor_ap,
                        "uos_measured": False,
                    },
                ))
            except (KeyError, OSError, ValueError) as e:
                aviso_derivados.append(
                    f"apariencia no incluida en el `.uos`: {e}"
                )

        # La ESCENA, con la malla y los nodos GS colgando de ella.
        malla_ingerida = self._malla_ingerida(snapshot)
        if malla_ingerida is None:
            # Respaldo: convertir el fichero del escaner. Sale una sopa de triangulos y,
            # sobre todo, SIN etiquetas — indexan el orden deduplicado del `mesh-agent`.
            try:
                pos_stl, caras_stl = lee_stl_binario(malla.read_bytes())
                malla_ingerida = {"positions": pos_stl, "faces": caras_stl}
                if etiquetas_ios is not None:
                    aviso_derivados.append(
                        "la segmentacion NO viaja: la escena se construyo del fichero STL "
                        "y no de la malla ingerida, asi que las etiquetas no indexan sus "
                        "vertices"
                    )
                    etiquetas_ios = None
            except ValueError as e:
                aviso_derivados.append(
                    f"el .uos no lleva escena (`scene/scene.glb`) y se queda por debajo de "
                    f"UOS-Core: {e}"
                )
        if malla_ingerida is not None:
            glb = construye_glb(
                malla_ingerida["positions"], malla_ingerida["faces"],
                malla_ingerida.get("normals"), nombre="scan",
                generador=f"{self.name}@{self.version}",
                nodos_gs=nodos_gs,
                # ⚠️ El FDI por vertice parte la malla en un primitive por diente con
                # `extras.uos_fdi` (§5.1). Sin eso, el picking semantico del §11.3 —que
                # esta definido sobre ese campo— no funciona en un visor ajeno, por mucho
                # que las mismas etiquetas viajen ademas en `derived/seg_teeth`.
                etiquetas=etiquetas_ios,
                extras={
                    "uos_frame": FRAME_IOS,
                    "uos_units": "mm",
                    "uos_source_asset": "asset.ios",
                    "uos_note": (
                        "presentacion: float32 desde float64. El asset reversible es "
                        "asset.ios, byte-identico al fichero del escaner"
                    ),
                },
            )
            if not sin_malla:
                extras_escena["scene/scene.glb"] = glb
                assets.append(asset_de_bytes(
                    glb, "scene/scene.glb", id_="asset.scene", kind=Clase.MESH_GS_SCENE,
                    visit=visita.id, frame=FRAME_IOS, media_type=MEDIA_GLB,
                    load_priority=10,
                ))

            # `derived/` — la segmentacion, Layer 3 y desmontable (§5.5).
            # ⚠️ Solo si la escena viaja: estas etiquetas indexan sus vertices por posicion,
            # asi que sin ella serian una lista de codigos que no indexa nada. Un derivado
            # que no se puede cruzar con nada es peor que no llevarlo.
            if etiquetas_ios is not None and not sin_malla:
                etq = np.asarray(etiquetas_ios)
                if len(etq) == len(malla_ingerida["positions"]):
                    crudo = codifica_etiquetas(etq)
                    meta = meta_segmentacion(
                        etq, asset_origen="asset.scene",
                        modelo="ash-seg-teeth",
                        # ⚠️ La version del SEGMENTADOR, no la de este agente. Aqui se
                        # escribia `self.version` —la del exportador— en el unico campo
                        # que existe para saber que modelo produjo la inferencia. Es el
                        # mismo fallo de atribucion que el `operator` de la registracion.
                        # Sin dato se deja `null`: §5.5 admite no saberlo, no admite
                        # inventarlo. El hash de los pesos sigue siendo lo que identifica
                        # el checkpoint de verdad.
                        version=version_segmentador,
                        pesos_sha256=(None if modelo_segmentacion is None
                                      else sha256_de_fichero(modelo_segmentacion)),
                        calidad=_calidad_frontera(malla_ingerida, etq),
                    )
                    extras_escena[SEGMENTACION] = crudo
                    extras_escena[SEGMENTACION_META] = json_de(meta)
                    assets.append(asset_de_bytes(
                        crudo, SEGMENTACION, id_="asset.seg_teeth",
                        kind=Clase.DERIVED_SEG, visit=visita.id, frame=FRAME_IOS,
                        media_type="application/octet-stream",
                        regulatory=Regulatorio(layer=3, status="investigational"),
                        sidecar_uri=SEGMENTACION_META,
                    ))
                else:
                    aviso_derivados.append(
                        f"la segmentacion NO viaja en `derived/`: trae {len(etq)} "
                        f"etiquetas y la escena {len(malla_ingerida['positions'])} "
                        "vertices, asi que no se pueden cruzar por indice"
                    )
        for i, foto in enumerate(imagenes or []):
            if not foto.exists():
                continue
            # ⚠️ El nombre del fichero NO viaja: los de un proveedor llevan identificadores
            # del paciente. Se renumera y la trazabilidad la da el sha256.
            uri = f"images/img_{i:03d}{foto.suffix.lower()}"
            # ⚠️ **Las fotos son originales, igual que el STL y el DICOM.** Iban atadas a
            # `sin_malla` —el perfil de solo gaussianas— y no a `sin_originales`, asi que
            # un contenedor que decia no llevar originales llevaba 18 MB de fotografias
            # del paciente dentro. Es el mismo criterio para las tres cosas: lo que viaja
            # es la direccion de contenido, no el fichero.
            if not sin_originales:
                ficheros[uri] = foto
            assets.append(asset_de(
                foto, uri, id_=f"asset.img_{i:03d}", kind=Clase.IMAGE2D, visit=visita.id,
                external=sin_originales,
                frame=FRAME_IOS,
                media_type=_MEDIA.get(foto.suffix.lower(), "image/jpeg"),
                # §5.3. Lo unico que se puede afirmar de estas: son fotos intraorales.
                # `fdi_targets` va vacio porque nadie anoto a que diente apunta cada una,
                # y vacio significa «no consta» — deducirlo de los pixeles exige la fusion
                # foto↔malla, que esta medida y no converge barata sin calibracion.
                projection=Proyeccion(type="intraoral_photo"),
            ))

        for i, doc in enumerate(informes or []):
            if not doc.exists():
                continue
            # ⚠️ El nombre NO viaja: los de una clinica llevan apellidos del paciente.
            # Se renumera, igual que las fotos.
            uri = f"clinical/documents/doc_{i:03d}{doc.suffix.lower()}"
            # ⚠️ Obedecen la MISMA bandera que el STL y las fotos. Un informe no es un
            # caso aparte: o el perfil lleva los originales o no los lleva, y una
            # excepcion para los PDF haria que «ligero» significara una cosa distinta
            # segun el asset. Con `sin_originales` se declaran por su direccion de
            # contenido —`sha256:<hex>`— igual que el resto, y es el mismo hash con el
            # que el gate nombra el que nadie pudo leer.
            if not sin_originales:
                ficheros[uri] = doc
            assets.append(asset_de(
                doc, uri, id_=f"asset.doc_{i:03d}", kind=Clase.DOCUMENT, visit=visita.id,
                external=sin_originales,
                frame=FRAME_IOS,
                media_type=_MEDIA.get(doc.suffix.lower(), "application/pdf"),
                # §5.1 Layer 1: es el registro que firmo una persona, no salida de un
                # modelo. La TRANSCRIPCION de lo que dice vive aparte, en
                # `clinical/observations.json`, y declara su propia `derivation`.
                regulatory=Regulatorio(layer=1),
            ))

        # La capa clinica: lo que el informe dice de cada pieza, las medidas que no caben
        # ⚠️ La version anterior se lee AQUI y no donde se usa, porque si su cadena esta
        # rota el aviso tiene que entrar en `motivos` ANTES de que la capa clinica los
        # congele. Un aviso de procedencia que no llega al gate de revision es un aviso que
        # nadie lee.
        _prev_sha, _prev_cadena, _aviso_cadena = lee_version_previa(
            previo or destination.with_suffix(".uos")
        )
        if _aviso_cadena:
            motivos = [*(motivos or []), _aviso_cadena]

        # en una pieza, y los motivos del gate. Ver `clinico.py` — es EXTENSION nuestra.
        clinico = capa_clinica(snapshot, list(motivos or []))
        if clinico["teeth"] or clinico["measurements"]:
            crudo_clinico = json_de(clinico)
            extras_escena[OBSERVACIONES] = crudo_clinico
            assets.append(asset_de_bytes(
                crudo_clinico.encode("utf-8"), OBSERVACIONES, id_="asset.clinical",
                kind=Clase.DOCUMENT, visit=visita.id, frame=FRAME_IOS,
                media_type="application/json", load_priority=15,
            ))

        registros = self._registros(snapshot, registrador)
        directorios: dict[str, Path] = {}
        extras: dict[str, str] = {}
        aviso_volumen: list[str] = []
        if cbct is not None and cbct.is_dir() and not registros:
            # ⚠️ **El volumen se queda fuera, y el resto del caso sale igual.** Sin la
            # registracion CBCT→escaner su frame no conecta con el canonico, asi que un
            # visor no sabria donde ponerlo: lo colocaria en el sitio equivocado sin poder
            # detectarlo, que es peor que no llevarlo. Y tirar la exportacion entera por
            # esto seria desproporcionado — la malla, las fotos y las vistas estan bien.
            aviso_volumen.append(
                "el CBCT NO viaja en el .uos: el snapshot no trae la transformada de la "
                "fusion, asi que su frame no conecta con el canonico y el contenedor se "
                "queda en UOS-Core. Registrar el escaner contra el CBCT lo desbloquea."
            )
        elif cbct is not None and cbct.is_dir() and (
            identificables := identificables_en(cbct)
        ):
            # ⚠️ **La serie lleva datos identificables en sus cabeceras y se queda fuera.**
            # El DICOM viaja intacto —es el punto del formato— asi que sus tags viajarian
            # con el, y el manifiesto afirma `phi_state: pseudonymized`. Un contenedor que
            # dice estar seudonimizado y lleva el nombre del paciente dentro es PEOR que
            # uno que declara `identified`: quien lo reciba se fia del campo y no abre 397
            # cabeceras a comprobarlo. El resto del caso sale igual, en UOS-Core.
            aviso_volumen.append(
                "el CBCT NO viaja en el .uos: su serie DICOM trae "
                + ", ".join(f"`{t}`" for t in identificables)
                + " con valor, y el contenedor declara `phi_state: pseudonymized`. "
                "Anonimizar la serie en origen lo desbloquea."
            )
        elif cbct is not None and cbct.is_dir():
            # ⚠️ La serie DICOM entra ENTERA y sin tocar (§5.2). Es lo unico que hace
            # afirmable que el contenedor no degrada la fuente: si viajara recodificada,
            # «byte-identico» seria una afirmacion sobre nuestro codec y no sobre el dato.
            uri = "volume/ct_001/"
            sidecar_uri = SIDECAR.format(id="ct_001")
            sidecar, aviso_volumen = describe_serie(cbct, frame=FRAME_CBCT)
            if not sin_originales:
                directorios[uri] = cbct
            # ⚠️ El sidecar del volumen viaja SIEMPRE, tambien en el perfil ligero: es lo
            # que dice dimensiones, espaciado y orientacion sin parsear DICOM (§5.2). Sin el
            # un contenedor ligero no podria ni situar el volumen que referencia.
            extras[sidecar_uri] = json_de(sidecar)
            assets.append(asset_de_directorio(
                cbct, uri, id_="asset.ct_001", kind=Clase.VOLUME, visit=visita.id,
                frame=FRAME_CBCT, media_type="application/dicom",
                acquisition=Adquisicion(time=snapshot.timestamp),
                sidecar_uri=sidecar_uri,
                external=sin_originales,
            ))

        fhir = self._fhir(assets)
        extensiones = self._extensiones(assets)
        ids_assets = {a.id for a in assets}
        vistas, aviso_vistas = self._vistas(
            snapshot, visita, etiquetas_ios,
            con_apariencia="asset.gs" in ids_assets,
            # Los controles de volumen del §7 —`mpr`, `clip_planes`, la capa `volume`—
            # solo se escriben si el volumen VIAJA. En un contenedor sin el darian a
            # entender que hay un plano que cortar.
            con_volumen=any(a.kind is Clase.VOLUME for a in assets),
        )
        salida = destination.with_suffix(".uos")
        # La version anterior del caso, si la hay: de ella salen `prev_manifest_sha256` y
        # la cadena que este contenedor continua. Por defecto es el propio destino, que
        # es lo que hace que reexportar encima produzca una version N+1 y no un borrado.
        previo_sha, cadena_previa = _prev_sha, _prev_cadena
        manifiesto = Manifiesto(
            uos_version=UOS_VERSION,
            case_id=f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, snapshot.acquisition_id)}",
            generator={"name": "agentic-smart-health", "version": self.version},
            # El pipeline seudonimiza: el nombre del paciente no entra en ningun artefacto.
            phi_state=EstadoPHI.PSEUDONYMIZED,
            subject=Sujeto(pseudonym=pseudonimo),
            canonical_frame=Frame(
                id=FRAME_IOS,
                description="Escaner intraoral, hub geometrico del caso",
            ),
            frames=[Frame(id=FRAME_CBCT, description="Volumen CBCT, centrado")],
            visits=[visita],
            assets=assets,
            registrations=registros,
            fhir_map=fhir,
            # ⚠️ Nada nuestro va en `extensions_required`. Todo lo que anadimos SUMA
            # informacion; un visor conforme tiene que poder abrir el caso sin
            # entender ninguna de ellas y ensenar la escena, el volumen y las fotos.
            extensions=extensiones,
            extensions_used=sorted(extensiones),
            provenance=Procedencia(prev_manifest_sha256=previo_sha, chain=CADENA),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        # El manifiesto se serializa UNA vez y se hashea eso mismo: el eslabon de la
        # cadena tiene que apuntar a los bytes que acaban en el ZIP, no a una segunda
        # serializacion que se le parezca.
        json_manifiesto = manifiesto.json_canonico()
        cadena = encadena(
            case_id=manifiesto.case_id,
            manifiesto_json=json_manifiesto,
            previo_sha256=previo_sha,
            cadena_previa=cadena_previa,
            generator=manifiesto.generator,
            assets=len(assets),
            note=f"{len(vistas)} vista(s), {len(registros)} registracion(es)",
        )
        escribe_uos(salida, manifiesto, ficheros.items(),
                    directorios=directorios,
                    json_manifiesto=json_manifiesto, extras={
            VISTAS: json_de({"views": [v.model_dump(mode="json") for v in vistas]}),
            CADENA: cadena.json_canonico(),
            **extras_escena,
            **extras,
        })

        # Se RELEE y se valida lo que se acaba de escribir, igual que hace el exportador
        # de STL: el sha256 de cada asset se recomputa desde el contenedor. Si cuadra, la
        # desviacion del ciclo es exactamente cero y esta MEDIDA, no afirmada — el
        # contenedor referencia, no transcodifica, y esto lo demuestra fichero a fichero.
        from uos.validador import valida

        informe = valida(salida)
        if not informe.valido:
            raise ValueError(
                "el .uos recien escrito no valida contra su propio manifiesto: "
                + "; ".join(informe.errores[:3])
            )

        avisos = (list(motivos or []) + aviso_vistas + aviso_volumen
                  + aviso_derivados + list(informe.avisos))
        # Lo estructurado vive en el MANIFIESTO, que es el registro del caso. Meterlo
        # tambien en `ExportOutput` daria dos sitios donde la misma verdad puede divergir,
        # y el contrato de exportacion es compartido: ensancharlo por un canal obliga a
        # los otros cinco a cargar con campos que no usan.
        return self._outcome(
            ModalityStatus.OK, path=salida, format="uos",
            # Cero MEDIDO, no afirmado: el validador recomputo el sha256 de cada asset
            # desde el contenedor ya escrito y cuadran todos.
            max_deviation_mm=0.0,
            hitl_reasons=avisos,
            detail=(
                f"{','.join(n.value for n in informe.niveles)} · {len(assets)} assets "
                f"byte-identicos, {len(registros)} registracion(es), {informe.vistas} "
                f"vista(s), version {informe.version} de la cadena, frame canonico "
                f"{FRAME_IOS}"
            ),
        )

    def _escribe_ply(self, destino: Path, datos: dict) -> Path:
        """PLY binario little-endian desde los arrays de un campo gaussiano.

        Misma estructura que ``field.escribe_ply``, pero aqui no podemos importar
        ``export_agents`` (seria una dependencia cruzada entre paquetes). Los arrays
        llegan como ``centers`` (N,3), ``scales`` (N,3), ``rotations`` (N,4) y
        ``density`` (N,), y se mapean a las propiedades PLY ``x/y/z``, ``scale_*``,
        ``rot_*`` y ``density``.
        """
        import numpy as np

        _MAPEO = {
            "centers": ("x", "y", "z"),
            "scales": ("scale_0", "scale_1", "scale_2"),
            "rotations": ("rot_0", "rot_1", "rot_2", "rot_3"),
            "density": ("density",),
            "region_id": ("region_id",),
            "origen": ("origen",),
        }
        _TIPOS_PLY = {
            "x": ("double", np.float64), "y": ("double", np.float64),
            "z": ("double", np.float64),
            "scale_0": ("float", np.float32), "scale_1": ("float", np.float32),
            "scale_2": ("float", np.float32),
            "rot_0": ("float", np.float32), "rot_1": ("float", np.float32),
            "rot_2": ("float", np.float32), "rot_3": ("float", np.float32),
            "density": ("float", np.float32),
            "region_id": ("short", np.int16),
            "origen": ("short", np.int16),
        }

        cols = []
        for arr_key, ply_names in _MAPEO.items():
            if arr_key in datos:
                arr = np.asarray(datos[arr_key])
                if arr.ndim == 2:
                    for i, pn in enumerate(ply_names):
                        cols.append((pn, arr[:, i]))
                else:
                    cols.append((ply_names[0], arr))

        n = len(cols[0][1]) if cols else 0
        cabecera = ["ply", "format binary_little_endian 1.0",
                     f"element vertex {n}"]
        cabecera += [f"property {_TIPOS_PLY[nombre][0]} {nombre}"
                     for nombre, _ in cols]
        cabecera.append("end_header")

        dtype = np.dtype([(nombre, _TIPOS_PLY[nombre][1]) for nombre, _ in cols])
        filas = np.empty(n, dtype=dtype)
        for nombre, arr in cols:
            filas[nombre] = arr

        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("wb") as fh:
            fh.write(("\n".join(cabecera) + "\n").encode("ascii"))
            fh.write(filas.tobytes())
        return destino

    def _registros(self, snapshot: TwinSnapshot, operador: str | None) -> list[Registro]:
        """La relacion CBCT ↔ escaner, INVERTIDA al canonico y declarada.

        La fusion geometrica registra el escaner SOBRE el CBCT, asi que su transformada va
        de escaner a CBCT. UOS quiere todo relativo al escaner, y la inversa de una rigida
        es exacta por construccion — es lo mismo que hace reversible el resto del sistema.

        ⚠️ **`operador` lo pasa quien exporta, y no se deduce del snapshot.** Aqui se
        escribia `auto:{snapshot.provenance.agent}`, que es el ULTIMO agente que toco la
        procedencia —la fusion semantica— y no el que calculo la ICP. El contenedor
        acreditaba una medida a quien no la hizo, en el unico campo que existe para saber
        quien la hizo. Sin dato se deja `None`: §6 admite no saberlo, no admite inventarlo.
        """
        t = snapshot.provenance.transform
        if t is None:
            return []
        from fusion_agents.registration import quaternion_to_matrix

        m = np.eye(4)
        m[:3, :3] = np.asarray(quaternion_to_matrix(t.rotation), dtype=np.float64)
        m[:3, 3] = np.asarray(t.translation, dtype=np.float64)
        return [Registro(
            id="reg.ct_to_ios",
            source_frame=FRAME_CBCT,
            target_frame=FRAME_IOS,
            transform_4x4_row_major=[float(x) for x in np.linalg.inv(m).ravel()],
            method="icp_surface",
            # ⚠️ El campo del contrato se llama `rms_mm`; aqui se leia `rms_error_mm` con
            # un `getattr(..., None)` que tapaba el fallo de nombre en silencio, asi que
            # TODOS los contenedores salian con `rms_error_mm: null` teniendo el numero
            # medido a mano. §6 lo pide porque una registracion automatica sin verificar
            # es provisional, y el residuo es lo unico que dice cuanto de provisional:
            # 0,666 mm y 6 mm no permiten lo mismo. Se lee como atributo, no con `getattr`,
            # para que el dia que alguien lo renombre falle en vez de escribir null.
            rms_error_mm=t.rms_mm,
            computed=snapshot.timestamp,
            operator=operador,
        )]

    def _vistas(
        self, snapshot: TwinSnapshot, visita: Visita, etiquetas: Any | None,
        *, con_apariencia: bool, con_volumen: bool = False,
    ) -> tuple[list[Vista], list[str]]:
        """Las vistas del caso (§7), medidas sobre la malla EN EL FRAME CANONICO.

        Las posiciones salen del almacen, no del STL: son las mismas que ingirio el
        `mesh-agent` y siguen SIN transformar, que es justo lo que las hace estar en
        `frame.ios_master`. Las que usa el visor no valen — aquellas ya estan llevadas al
        marco del twin, que es el del CBCT, y encuadrarian la escena desde otro sitio.

        Las piezas con vista propia son las que llevan algo anotado. Una vista por diente
        etiquetado serian catorce entradas equivalentes; lo que hace util un deep-link es
        que apunte a donde alguien miro.
        """
        if etiquetas is None or snapshot.surface_ref is None:
            return [], [
                f"el .uos no lleva vistas ({VISTAS} va vacio): sin las etiquetas FDI del "
                "escaner no hay con que medir los ejes anatomicos, y bautizar los ejes "
                "principales de la nube produce nombres plausibles y a veces invertidos"
            ]
        malla = self.store.load(snapshot.surface_ref)
        return construye_vistas(
            np.asarray(malla["positions"], dtype=np.float64),
            np.asarray(etiquetas),
            visita=visita.id,
            piezas=sorted({obs.region_id for obs in snapshot.regional}),
            con_apariencia=con_apariencia,
            con_volumen=con_volumen,
        )

    def _fhir(self, assets: list) -> dict[str, RecursoFHIR]:
        """El mapeo a FHIR R4 (§9): un TIPO de recurso por asset, y el caso entero.

        `case` no es un asset: es el `.uos` completo, que el spec publica como
        `DocumentReference` con el media type del formato en el adjunto. Va con la misma
        clave que usa el ejemplo del spec para no inventarse una.
        """
        fuera = {
            "case": RecursoFHIR(
                resource_type="DocumentReference",
                note=f"el .uos entero como adjunto, content_type {MEDIA_TYPE}",
            )
        }
        for a in assets:
            if a.id == "asset.clinical":
                # ⚠️ NO `DocumentReference` como el resto de documentos: lo que lleva son
                # medidas por diente, y el recurso de FHIR para una medida clinica es
                # `Observation`. Mapearlo como adjunto lo dejaria fuera del alcance de
                # cualquier consulta del PMS, que es el punto del §9.
                fuera[a.id] = RecursoFHIR(
                    resource_type="Observation",
                    note="una Observation por pieza y por medida, con su `subject` y su "
                         "`bodySite` en FDI",
                )
                continue
            tipo, nota = _RECURSO[a.kind]
            fuera[a.id] = RecursoFHIR(resource_type=tipo, note=nota)
        return fuera

    def _malla_ingerida(self, snapshot: TwinSnapshot) -> dict | None:
        """La malla tal como la guardo el `mesh-agent`: vertices, caras y normales.

        ⚠️ **Esta y no el fichero STL**, y la diferencia importa: el STL es sopa de
        triangulos y esta es la malla deduplicada cuyo ORDEN DE VERTICES indexan las
        etiquetas FDI. Construir la escena del STL daria tres veces mas vertices y ninguna
        forma de casar la segmentacion con ellos.
        """
        if snapshot.surface_ref is None or self.store is None:
            return None
        try:
            malla = self.store.load(snapshot.surface_ref)
        except (KeyError, OSError, ValueError):
            return None
        return malla if "positions" in malla and "faces" in malla else None

    def _lee_submuestreo(self, snapshot: TwinSnapshot) -> dict | None:
        """Lee `paso` y `n_origen` del artefacto del campo semilla, si existen.

        El `cbct-agent` guarda el paso de submuestreo (el array de 3 enteros que
        indica cada cuántos vóxeles se quedó uno) y el número de vóxeles originales.
        Esto permite al consumidor del `.uos` saber cuántos vóxeles había antes: un
        PLY con 500K gaussianas y paso (3,3,1) indica ~4,5M vóxeles originales.
        """
        if snapshot.gaussian_field_ref is None or self.store is None:
            return None
        try:
            datos = self.store.load(snapshot.gaussian_field_ref)
        except (KeyError, OSError, ValueError):
            return None
        if "paso" not in datos or "n_origen" not in datos:
            return None
        import numpy as np

        paso_arr = np.asarray(datos["paso"], dtype=int)
        n_origen = int(datos["n_origen"])
        n_final = int(datos["centers"].shape[0])
        # `paso` viene en orden (z, y, x) del `occupied`; lo pasamos a (x, y, z)
        # para que el consumidor lo entienda sin conocer la interna del agente.
        paso_xyz = paso_arr[::-1].tolist()
        return {
            "paso_voxeles": paso_xyz,
            "de": n_origen,
            "a": n_final,
        }

    @staticmethod
    def _nota_color_ply(ruta: Path) -> str:
        """Lo que el PLY dice de su propio color, para que el sidecar no lo repita a mano.

        Se toman los `comment` desde el que abre `f_dc_*` hasta el ultimo seguido: son los
        que `gaussian_engine.apariencia._comentarios_color` escribe describiendo de donde
        salio el color y cuanto de el es medido. Si el fichero no los trae se dice eso, en
        vez de afirmar nada sobre un color que no se ha podido leer.
        """
        try:
            with ruta.open("rb") as f:
                crudo = f.read(65536)
        except OSError:
            return "el PLY de apariencia no se ha podido leer para describir su color"
        fin = crudo.find(b"end_header")
        lineas = crudo[: fin if fin >= 0 else len(crudo)].decode("ascii", "replace")
        recogiendo, partes = False, []
        for linea in lineas.splitlines():
            if not linea.startswith("comment "):
                continue
            cuerpo = linea[len("comment "):].strip()
            if cuerpo.startswith("f_dc_"):
                recogiendo = True
            elif recogiendo and cuerpo.startswith(("opacity", "scale", "rot ", "unidades",
                                                   "entrenado", "region_id", "nx,ny,nz",
                                                   "f_rest_")):
                break
            if recogiendo:
                partes.append(cuerpo)
        if not partes:
            return "el PLY de apariencia no declara de donde sale su color"
        return " ".join(partes)

    @staticmethod
    def _cabecera_ply(ruta: Path) -> tuple[str | None, int | None, list[str]]:
        """`(unidades, n_primitivas, propiedades)` leidas DEL FICHERO.

        ⚠️ **El descriptor describe el fichero, asi que pregunta al fichero.** Antes
        afirmaba `units: "mm"` con un literal y `n_primitives` con el numero del snapshot;
        el PLY de apariencia estaba en el espacio normalizado de Blender y tenia 118.041
        gaussianas, y el sidecar decia milimetros y 1.341.990. Las dos mentiras eran
        invisibles porque nada las contrastaba con el fichero que describian.

        ⚠️ **Las propiedades tambien salen del fichero.** El esquema de la apariencia se
        enumeraba a mano en `esquema_apariencia()`; el escritor paso a emitir `region_id` y
        la lista no se entero, asi que el dato viajaba en el PLY **sin declararse en el
        sidecar** — invisible para cualquier lector que no sea el nuestro.

        Se lee solo la cabecera —ASCII, hasta `end_header`— y por eso no importa que el
        PLY pese ochenta megas. Y devuelve `None` en vez de un valor por defecto cuando el
        fichero no lo declara: ese es justo el caso en el que suponer volvio a fallar.
        """
        try:
            with ruta.open("rb") as f:
                crudo = f.read(65536)
        except OSError:
            return None, None, []
        fin = crudo.find(b"end_header")
        texto = crudo[: fin if fin >= 0 else len(crudo)].decode("ascii", "replace")
        unidades = n = None
        propiedades: list[str] = []
        for linea in texto.splitlines():
            if linea.startswith("comment unidades "):
                unidades = linea.split(maxsplit=2)[2].strip()
            elif linea.startswith("element vertex "):
                try:
                    n = int(linea.split()[2])
                except (IndexError, ValueError):
                    n = None
            elif linea.startswith("property ") and not linea.startswith("property list"):
                partes = linea.split()
                if len(partes) == 3:
                    propiedades.append(partes[2])
        return unidades, n, propiedades

    def _descriptor_gs(
        self, snapshot: TwinSnapshot, *, papel: str, medido: bool, marco: str,
        nota: str, submuestreo: dict | None = None,
        esquema_override: list | None = None,
        perfil_override: str | None = None,
        n_primitives_override: int | None = None,
        unidades_override: str | None = None,
    ) -> dict[str, Any]:
        """El sidecar de un asset de gaussianas: que es y con que semantica.

        ⚠️ **`esquema_campo` viaja tal cual**, y es la pieza que evita el fallo silencioso
        que el contrato del twin ya documenta: el PLY de facto de 3DGS usa `scale_0..2` y
        `rot_0..3` con los MISMOS nombres y guarda el LOGARITMO de la escala; aqui van
        milimetros lineales. Un visor estandar abriendo esto no fallaria — exponenciaria
        nuestros milimetros y renderizaria basura con muy buen aspecto. Por eso las
        columnas se declaran en vez de suponerse.

        ⚠️ **Overrides para la capa de apariencia.** La capa `asset.gs` tiene un esquema
        de columnas DISTINTO al campo de densidad (INRIA: `f_dc_*`, `opacity` en logit,
        `scale_*` en log). Los overrides permiten al UOS usar el esquema correcto sin
        cambiar el snapshot (que sigue apuntando al campo de densidad).
        """
        esquema = esquema_override if esquema_override is not None else snapshot.esquema_campo
        perfil = perfil_override if perfil_override is not None else snapshot.perfil_campo
        n_prim = (n_primitives_override
                  if n_primitives_override is not None
                  else snapshot.n_primitives)
        resultado: dict[str, Any] = {
            "role": papel,
            "measured": medido,
            "note": nota,
            "profile": perfil,
            "frame": marco,
            # ⚠️ `mm` es el defecto del formato, no una afirmacion sobre este fichero: los
            # campos de densidad se escriben aqui mismo y en milimetros por construccion.
            # Un asset escrito por otro paquete —la apariencia— pasa lo que su fichero
            # DECLARA, y si no declara nada eso se ve en el sidecar. Ver `_cabecera_ply`.
            "units": unidades_override or "mm",
            "n_primitives": n_prim,
            "columns": [
                {"name": c.nombre, "unit": c.unidad, "scale": c.escala,
                 "measured": c.medido, "derived_from": c.derivado_de,
                 "meaning": c.significado, "vocabulary": c.vocabulario}
                for c in esquema
            ],
        }
        if submuestreo is not None:
            resultado["submuestreo"] = submuestreo
        return resultado

    def _extensiones(self, assets: list) -> dict[str, Extension]:
        """Lo que este emisor anade al borrador, dicho en voz alta.

        Sin esto, un lector ajeno ignora nuestras extensiones **sin enterarse de que las
        ignora**, que convierte un formato abierto en uno que solo su emisor lee entero.
        Con esto puede decidir: leerlas, saltarlas a sabiendas, o avisar al usuario.
        """
        ids = {a.id: a.uri for a in assets}
        fuera: dict[str, Extension] = {}
        if "asset.clinical" in ids:
            fuera["ash_clinical"] = Extension(
                name="ash_clinical", version="1.0", uri=ids["asset.clinical"],
                schema_id="ash-clinical/1.0",
                description=(
                    "atributos clinicos por pieza (pH, raices, conductos, hallazgos) y "
                    "medidas no regionales, con la procedencia de cada valor. El borrador "
                    "los manda a FHIR (§9) y entonces un .uos suelto no puede contestar "
                    "que dice el informe de una pieza"
                ),
            )
        if any(a.id in ("asset.field", "asset.composite") for a in assets):
            fuera["ash_gs_measured"] = Extension(
                name="ash_gs_measured", version="1.0",
                description=(
                    "descriptor `.gs.json` por capa de gaussianas: declara si es MEDIDA o "
                    "reconstruida y el esquema de sus columnas. El borrador trata el 3DGS "
                    "como apariencia en marco de reconstruccion; aqui hay campos ajustados "
                    "a la densidad del CBCT, en el marco del paciente y con error en HU"
                ),
            )
        return fuera
