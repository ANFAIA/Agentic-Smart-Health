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

from uos.contenedor import asset_de, escribe_uos, json_de
from uos.manifiesto import (
    UOS_VERSION,
    Adquisicion,
    Clase,
    EstadoPHI,
    Frame,
    Manifiesto,
    Procedencia,
    Registro,
    Sujeto,
    Visita,
)
from uos.procedencia import CADENA, encadena, lee_version_previa
from uos.vistas import VISTAS, Vista, construye_vistas

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
    version = "0.1.0"

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
        imagenes: list[Path] | None = None,
        motivos: list[str] | None = None,
        etiquetas_ios: Any | None = None,
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
        assets = []

        # ⚠️ El nombre del fichero NO viaja, tampoco el de la malla. Los de un proveedor
        # llevan identificadores —el que produjo este caso se llamaba "1574 UpperJawScan"—
        # y `1574` es el numero de caso. Se nombra por su papel en la escena y la
        # trazabilidad la da el sha256, que es mas fuerte que un nombre.
        uri = f"scene/scan{malla.suffix.lower()}"
        ficheros[uri] = malla
        assets.append(asset_de(
            malla, uri, id_="asset.ios", kind=Clase.MESH_GS_SCENE, visit=visita.id,
            frame=FRAME_IOS, media_type=_MEDIA.get(malla.suffix.lower(), "model/stl"),
            acquisition=Adquisicion(time=snapshot.timestamp),
        ))
        if escena_gs is not None and escena_gs.exists():
            # Fallback declarado del spec (§5.1): mientras KHR_gaussian_splatting siga sin
            # ratificar, la capa de apariencia va como asset externo apuntado desde la
            # escena, no embebida. Es el camino real hoy, no el plan B.
            uri = f"scene/appearance{escena_gs.suffix.lower()}"
            ficheros[uri] = escena_gs
            assets.append(asset_de(
                escena_gs, uri, id_="asset.gs", kind=Clase.MESH_GS_SCENE, visit=visita.id,
                frame=FRAME_IOS, media_type="application/octet-stream",
                load_priority=25,
            ))
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

        registros = self._registros(snapshot)
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
                    json_manifiesto=json_manifiesto, extras={
            VISTAS: json_de({"views": [v.model_dump(mode="json") for v in vistas]}),
            CADENA: cadena.json_canonico(),
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

        avisos = list(motivos or []) + aviso_vistas + list(informe.avisos)
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
