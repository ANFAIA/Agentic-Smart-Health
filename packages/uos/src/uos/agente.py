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


class UOSExportAgent(BaseExportAgent):
    """Empaqueta el twin como Unified Oral Scene.

    ⚠️ **Referencia, no transcodifica** (§2.1). Los ficheros entran tal cual y su `sha256`
    va en el manifiesto: lo que sale del `.uos` es byte-identico a lo que entro. Es lo que
    permite decir que el contenedor no degrada nada, y lo que hace la procedencia
    verificable por quien lo reciba sin confiar en nosotros.
    """

    name = "uos-export-agent"
    version = "0.3.0"

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
        motivos: list[str] | None = None,
        etiquetas_ios: Any | None = None,
        modelo_segmentacion: Path | None = None,
        cbct: Path | None = None,
        previo: Path | None = None,
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
        ficheros[uri] = malla
        # ⚠️ `document`, no `mesh_gs_scene`. Lo dice el §5.1: la escena es el `.glb`, y el
        # STL original «PUEDE incluirse como asset document para trazabilidad». Declararlo
        # escena haria que un visor viera dos escenas y no supiera cual montar.
        assets.append(asset_de(
            malla, uri, id_="asset.ios", kind=Clase.DOCUMENT, visit=visita.id,
            frame=FRAME_IOS, media_type=_MEDIA.get(malla.suffix.lower(), "model/stl"),
            acquisition=Adquisicion(time=snapshot.timestamp),
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
        registros = self._registros(snapshot)
        al_canonico = registros[0].transform_4x4_row_major if registros else None
        nodos_gs: list[NodoGS] = []

        # Las capas de gaussianas. Son tres cosas distintas con el mismo `kind`, asi que
        # cada una lleva su descriptor: el campo es densidad MEDIDA, el compuesto es medida
        # de dos modalidades, y la apariencia es reconstruccion contra renders.
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
            # El nombre dentro del contenedor describe QUE es, no de que variable sale.
            # `asset.gs` daria `scene/gs.ply`, que no dice nada a quien lo abra.
            corto = {"asset.gs": "appearance"}.get(id_, id_.split(".")[1])
            uri = f"scene/{corto}{ruta.suffix.lower()}"
            descriptor = f"scene/{corto}.gs.json"
            ficheros[uri] = ruta
            extras_escena[descriptor] = json_de(self._descriptor_gs(
                snapshot, papel=papel, medido=medido, marco=marco, nota=nota,
            ))
            assets.append(asset_de(
                ruta, uri, id_=id_, kind=Clase.MESH_GS_SCENE, visit=visita.id,
                frame=marco, media_type="application/octet-stream",
                # El orden de carga del §4.1: malla 10 -> fotos 20 -> GS 25 -> volumen 30.
                load_priority=25, sidecar_uri=descriptor,
            ))
            nodos_gs.append(NodoGS(
                uri=uri, nombre=papel,
                # Lo que esta en el marco del CBCT necesita la transformada al canonico;
                # la apariencia ya vive en el del escaner, que ES el canonico.
                matriz_fila=(al_canonico if marco == FRAME_CBCT else None),
                extras={"uos_descriptor_uri": descriptor, "uos_measured": medido},
            ))

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
            extras_escena["scene/scene.glb"] = glb
            assets.append(asset_de_bytes(
                glb, "scene/scene.glb", id_="asset.scene", kind=Clase.MESH_GS_SCENE,
                visit=visita.id, frame=FRAME_IOS, media_type=MEDIA_GLB,
                load_priority=10,
            ))

            # `derived/` — la segmentacion, Layer 3 y desmontable (§5.5).
            if etiquetas_ios is not None:
                etq = np.asarray(etiquetas_ios)
                if len(etq) == len(malla_ingerida["positions"]):
                    crudo = codifica_etiquetas(etq)
                    meta = meta_segmentacion(
                        etq, asset_origen="asset.scene",
                        modelo="ash-seg-teeth",
                        version=str(getattr(self, "version", "0")),
                        pesos_sha256=(None if modelo_segmentacion is None
                                      else sha256_de_fichero(modelo_segmentacion)),
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
            ficheros[uri] = foto
            assets.append(asset_de(
                foto, uri, id_=f"asset.img_{i:03d}", kind=Clase.IMAGE2D, visit=visita.id,
                frame=FRAME_IOS,
                media_type=_MEDIA.get(foto.suffix.lower(), "image/jpeg"),
            ))

        # La capa clinica: lo que el informe dice de cada pieza, las medidas que no caben
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

        registros = self._registros(snapshot)
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
            directorios[uri] = cbct
            extras[sidecar_uri] = json_de(sidecar)
            assets.append(asset_de_directorio(
                cbct, uri, id_="asset.ct_001", kind=Clase.VOLUME, visit=visita.id,
                frame=FRAME_CBCT, media_type="application/dicom",
                acquisition=Adquisicion(time=snapshot.timestamp),
                sidecar_uri=sidecar_uri,
            ))

        fhir = self._fhir(assets)
        extensiones = self._extensiones(assets)
        vistas, aviso_vistas = self._vistas(snapshot, visita, etiquetas_ios,
                                            con_apariencia="asset.gs" in
                                            {a.id for a in assets})
        salida = destination.with_suffix(".uos")
        # La version anterior del caso, si la hay: de ella salen `prev_manifest_sha256` y
        # la cadena que este contenedor continua. Por defecto es el propio destino, que
        # es lo que hace que reexportar encima produzca una version N+1 y no un borrado.
        previo_sha, cadena_previa = lee_version_previa(previo or salida)
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

    def _registros(self, snapshot: TwinSnapshot) -> list[Registro]:
        """La relacion CBCT ↔ escaner, INVERTIDA al canonico y declarada.

        La fusion geometrica registra el escaner SOBRE el CBCT, asi que su transformada va
        de escaner a CBCT. UOS quiere todo relativo al escaner, y la inversa de una rigida
        es exacta por construccion — es lo mismo que hace reversible el resto del sistema.
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
            rms_error_mm=getattr(t, "rms_error_mm", None),
            computed=snapshot.timestamp,
            operator=f"auto:{snapshot.provenance.agent}",
        )]

    def _vistas(
        self, snapshot: TwinSnapshot, visita: Visita, etiquetas: Any | None,
        *, con_apariencia: bool,
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

    def _descriptor_gs(
        self, snapshot: TwinSnapshot, *, papel: str, medido: bool, marco: str,
        nota: str,
    ) -> dict[str, Any]:
        """El sidecar de un asset de gaussianas: que es y con que semantica.

        ⚠️ **`esquema_campo` viaja tal cual**, y es la pieza que evita el fallo silencioso
        que el contrato del twin ya documenta: el PLY de facto de 3DGS usa `scale_0..2` y
        `rot_0..3` con los MISMOS nombres y guarda el LOGARITMO de la escala; aqui van
        milimetros lineales. Un visor estandar abriendo esto no fallaria — exponenciaria
        nuestros milimetros y renderizaria basura con muy buen aspecto. Por eso las
        columnas se declaran en vez de suponerse.
        """
        return {
            "role": papel,
            "measured": medido,
            "note": nota,
            "profile": snapshot.perfil_campo,
            "frame": marco,
            "units": "mm",
            "n_primitives": snapshot.n_primitives,
            "columns": [
                {"name": c.nombre, "unit": c.unidad, "scale": c.escala,
                 "measured": c.medido, "derived_from": c.derivado_de,
                 "meaning": c.significado, "vocabulary": c.vocabulario}
                for c in snapshot.esquema_campo
            ],
        }

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
