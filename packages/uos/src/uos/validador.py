"""`uos-validate`: comprueba que un `.uos` es lo que dice ser (§12).

Los niveles de conformidad del spec —Core, Vol, Sig, Full— no son etiquetas: son lo que
permite a un implementador decir «soporto esto» sin ambiguedad, y a un emisor saber si su
fichero se va a poder abrir. Se comprueban aqui y no en el escritor porque **un validador
tiene que poder correr sobre un fichero que escribio otro**.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from uos.clinico import OBSERVACIONES
from uos.contenedor import MANIFIESTO, lee_manifiesto_de
from uos.manifiesto import (
    UOS_VERSION,
    Anatomico,
    Aptitud,
    Clase,
    Desidentificacion,
    EstadoPHI,
    Manifiesto,
    Registro,
    digesto_de_partes,
)
from uos.procedencia import CADENA, FIRMAS, Cadena, revisa_cadena
from uos.vistas import VISTAS, Vista


class Conformidad(StrEnum):
    CORE = "UOS-Core"      # manifiesto + mesh_gs_scene + image2d
    VOL = "UOS-Vol"        # + volumen
    SIG = "UOS-Sig"        # + senales
    FULL = "UOS-Full"


_EXIGE = {
    Conformidad.CORE: {Clase.MESH_GS_SCENE},
    Conformidad.VOL: {Clase.MESH_GS_SCENE, Clase.VOLUME},
    Conformidad.SIG: {Clase.MESH_GS_SCENE, Clase.SIGNAL},
    Conformidad.FULL: {Clase.MESH_GS_SCENE, Clase.VOLUME, Clase.SIGNAL},
}


@dataclass
class Informe:
    """Lo que el validador encontro. `errores` invalida; `avisos` no."""

    errores: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    niveles: list[Conformidad] = field(default_factory=list)
    # ⚠️ **Perfil ORTOGONAL a los niveles, y por eso va aparte (B-6).** Los niveles dicen
    # que TIPOS de asset lleva el contenedor, o sea si un lector puede abrirlo. Ninguno
    # dice si el contenedor esta en condiciones de salir de la organizacion que lo emitio,
    # que es una pregunta distinta y la que se hace antes de mandarlo a alguien. Se
    # DERIVA igual que los niveles: no se elige, se comprueba.
    distribuible: bool = False
    #: Que le falta para serlo. Vacio cuando `distribuible` es cierto.
    no_distribuible_porque: list[str] = field(default_factory=list)
    # Cuantas versiones del caso hay encadenadas (§8) y cuantas vistas guardadas (§7).
    # `version` es 0 cuando el contenedor no declara cadena: no es «la primera», es que
    # no hay historial que recorrer, y son cosas distintas.
    version: int = 0
    vistas: int = 0
    # Cuantos originales adquiridos REFERENCIA el contenedor sin custodiarlos.
    #
    # ⚠️ Es un RECUENTO y no un aviso por asset, y la diferencia importa: que los
    # originales no viajen es la definicion del formato, no una excepcion. Un aviso por
    # cada uno saltaria en todos los assets de todos los contenedores siempre, y un aviso
    # que nunca distingue nada deja de leerse — enterrando los que si dicen algo.
    externos: int = 0

    @property
    def valido(self) -> bool:
        return not self.errores


def valida(ruta: Path) -> Informe:
    """Estructura, hashes, cadena de procedencia, vistas, frames y reglas regulatorias."""
    inf = Informe()
    with zipfile.ZipFile(ruta) as z:
        nombres = z.namelist()
        if not nombres or nombres[0] != MANIFIESTO:
            inf.errores.append(
                f"la primera entrada del ZIP es {nombres[0] if nombres else 'ninguna'!r}, "
                f"y el spec exige {MANIFIESTO!r}"
            )
            return inf
        for zi in z.infolist():
            if zi.compress_type != zipfile.ZIP_STORED:
                inf.errores.append(
                    f"{zi.filename} esta comprimido; el spec exige STORE para que el "
                    "acceso aleatorio por rangos funcione"
                )
                break
        crudo = z.read(MANIFIESTO)
        # ⚠️ La rama de version PRIMERO: si el contenedor declara una mayor superior esto
        # eleva, y es lo correcto — no se valida lo que no se sabe interpretar (§15).
        m, ignorados = lee_manifiesto_de(crudo, nombre=ruta.name)
        if ignorados:
            inf.avisos.append(
                f"el contenedor declara uos_version {m.uos_version!r} y este validador "
                f"implementa {UOS_VERSION!r}: se han IGNORADO {len(ignorados)} campo(s) que "
                "no conoce, y por eso no puede emitir una version nueva de este caso: "
                + ", ".join(ignorados)
            )
        # ⚠️ **Y el esquema DESPUES de la rama, que es el orden que faltaba (T-3).** El
        # esquema publicado lleva `additionalProperties: false`, asi que un contenedor de
        # minor superior —que el §15.2 obliga a leer con indulgencia, ignorando y nombrando
        # lo que no se conoce— salia con un aviso de «campo desconocido» por cada campo
        # nuevo. El algoritmo contradecia a la compatibilidad que un minor promete, y lo
        # hacia en el unico caso para el que esa promesa existe.
        if not ignorados:
            _valida_esquema(crudo, inf)
        else:
            inf.avisos.append(
                "no se contrasta contra el JSON Schema publicado: el contenedor declara "
                "una version menor superior y sus campos nuevos saldrian como desconocidos"
            )
        _valida_assets(z, m, inf)
        _valida_capas_en_la_escena(z, m, inf)
        _valida_capas_clinicas(z, inf)
        _valida_derivados(z, m, inf)
        _valida_gs(z, m, inf)
        _valida_matrices_gs(z, m, inf)
        _valida_union_segmentacion(z, m, inf)
        _valida_entradas_declaradas(z, m, inf)
        _valida_phi(z, m, inf)
        _valida_procedencia(z, m, hashlib.sha256(crudo).hexdigest(), inf)
        _valida_vistas(z, m, inf)

    _valida_frames(m, inf)
    _valida_regulatorio(m, inf)
    _valida_extensiones(m, inf)
    if m.canonical_frame.units != "mm":
        inf.errores.append(
            f"el frame canonico declara unidades {m.canonical_frame.units!r}; "
            "la convencion del spec es milimetros"
        )
    inf.niveles = [n for n, exige in _EXIGE.items()
                   if exige <= {a.kind for a in m.assets}]
    _perfil_distribuible(m, inf)
    return inf


def _perfil_distribuible(m: Manifiesto, inf: Informe) -> None:
    """UOS-Distributable: si el contenedor puede salir de quien lo emitio (B-6).

    ⚠️ **«Abrible» y «distribuible» son preguntas distintas y solo se contestaba la
    primera.** Core/Vol/Sig/Full describen que tipos de asset hay dentro; nada describia
    si el caso esta en condiciones de mandarse a un laboratorio o a un colega. Son las
    condiciones de B-1, B-3 y B-4 juntas, y juntas porque de una en una no deciden nada:
    un contenedor con el proposito declarado y la cara dentro no se puede mandar igual.

    No es un error: un contenedor puede ser perfectamente valido y no ser distribuible —
    es lo normal mientras el caso vive dentro de la clinica. Lo que no puede es que nadie
    lo sepa antes de adjuntarlo a un correo.
    """
    faltan: list[str] = []
    if m.phi_state == EstadoPHI.IDENTIFIED:
        faltan.append(
            "declara `phi_state: identified`: lleva dato identificable dentro"
        )
    if m.deidentification is None:
        faltan.append("no declara `deidentification`: no dice que medidas se aplicaron")
    if m.purpose_of_use is None:
        faltan.append(
            "no declara `purpose_of_use`: quien lo reciba tendria que suponer para que se "
            "le manda"
        )
    fuera = [a.id for a in m.assets
             if a.regulatory.layer == 3 and not a.uri.startswith("derived/")]
    if fuera:
        faltan.append(
            f"lleva capa 3 fuera de derived/ ({', '.join(fuera)}): borrar ese directorio "
            "no quitaria la inferencia"
        )
    # Los errores del propio validador cuentan: un contenedor que no cuadra consigo mismo
    # no se manda a nadie, por muy declarado que este todo lo demas.
    if inf.errores:
        faltan.append("no valida contra su propio manifiesto")
    inf.no_distribuible_porque = faltan
    inf.distribuible = not faltan


def _valida_esquema(crudo: bytes, inf: Informe) -> None:
    """El manifiesto contra el **JSON Schema publicado** (§12).

    ⚠️ **No es redundante con `model_validate_json`, y la diferencia importa.** Pydantic
    comprueba que el manifiesto encaje en NUESTRO contrato; el esquema publicado es el
    unico artefacto que un tercero puede usar para comprobar el suyo. Correrlo aqui es lo
    que garantiza que el esquema que publicamos es el que de verdad aceptamos: si los dos
    se separan, este chequeo lo dice antes que un implementador de fuera.

    Un fallo aqui es AVISO y no error: el que manda sobre si el contenedor es valido sigue
    siendo el contrato. Lo que este chequeo detecta es que el esquema publicado se quedo
    atras, que es un problema de nuestra publicacion y no del fichero que alguien trae.
    """
    try:
        import jsonschema
    except ModuleNotFoundError:  # pragma: no cover - dependencia opcional
        return
    from uos.esquema import esquema_del_manifiesto

    try:
        jsonschema.validate(json.loads(crudo), esquema_del_manifiesto())
    except jsonschema.ValidationError as e:
        inf.avisos.append(
            f"el manifiesto no valida contra el JSON Schema publicado en "
            f"{'.'.join(str(x) for x in e.absolute_path) or '(raiz)'}: {e.message}. "
            "El contrato lo acepta, asi que lo que se ha quedado atras es el esquema."
        )


def _valida_assets(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Cada asset existe y su hash cuadra. Verificar es la politica en ingesta (§8)."""
    dentro = set(z.namelist())
    referenciados: list[str] = []
    for a in m.assets:
        if a.external:
            referenciados.append(a.id)
            # ⚠️ **Del externo no se comprueba nada AQUI, y no es un hueco (T-3).** La
            # revision pedia verificar que su `uri` sea una direccion de contenido y que
            # case con su `sha256` —«en las dos direcciones», §3.4.3—. Ya se hace, una capa
            # antes: `Asset._direccion_y_custodia` es un validador de modelo, asi que un
            # manifiesto con esa incoherencia ni siquiera llega a parsearse y el check 2 lo
            # reporta. Repetirlo aqui seria codigo que no puede ejecutarse nunca, que es
            # peor que no tenerlo: aparenta una cobertura que en realidad viene de otro
            # sitio. Lo que faltaba era decir DONDE vive la regla, y eso va en la spec.
            continue
        if a.uri.endswith("/"):
            _valida_serie(z, a, dentro, inf)
            continue
        if a.uri not in dentro:
            inf.errores.append(f"asset {a.id}: {a.uri} no esta en el contenedor")
            continue
        crudo = z.read(a.uri)
        if (real := hashlib.sha256(crudo).hexdigest()) != a.sha256:
            inf.errores.append(
                f"asset {a.id}: el sha256 declarado no es el del fichero "
                f"({a.sha256[:12]}… vs {real[:12]}…)"
            )
        if len(crudo) != a.bytes:
            inf.errores.append(
                f"asset {a.id}: declara {a.bytes} bytes y tiene {len(crudo)}"
            )

    # ⚠️ **Una linea por CONTENEDOR, no una por asset.** Los originales adquiridos se
    # referencian y no viajan: es el formato (ver `Asset.external`), asi que avisar de cada
    # uno seria repetir la definicion tantas veces como assets tenga el caso. Lo que sigue
    # mereciendo decirse una vez es que de estos el validador no puede comprobar NADA: su
    # `sha256` acredita cual es el fichero, y quien lo custodie tendra que demostrarlo.
    if referenciados:
        inf.externos = len(referenciados)
        inf.avisos.append(
            f"el contenedor REFERENCIA {len(referenciados)} original(es) adquirido(s) que "
            "no custodia, asi que su contenido no se verifica aqui: "
            + ", ".join(referenciados)
        )


def _valida_serie(z: zipfile.ZipFile, a, dentro: set[str], inf: Informe) -> None:
    """Un asset que es un DIRECTORIO: cada `Parte` existe y cuadra, y el digesto tambien.

    Se comprueba fichero a fichero y no solo el digesto del conjunto: la diferencia es
    entre poder decir «esta serie no cuadra» y poder decir «el corte 214 esta corrupto».
    Y se comprueba **en los dos sentidos** — un corte que esta dentro y que el manifiesto
    no declara es tan grave como uno declarado que falta: significa que la serie que sale
    no es la que se dice que entro.
    """
    hijos = {n for n in dentro if n.startswith(a.uri) and not n.endswith("/")}
    if not hijos:
        inf.errores.append(f"asset {a.id}: el directorio {a.uri} esta vacio")
        return
    if not a.parts:
        inf.errores.append(
            f"asset {a.id}: es un directorio con {len(hijos)} fichero(s) y no declara "
            "`parts`, asi que no hay contra que verificarlos uno a uno"
        )
        return
    declarados = {a.uri + p.name for p in a.parts}
    if sobran := hijos - declarados:
        inf.errores.append(
            f"asset {a.id}: {len(sobran)} fichero(s) dentro de {a.uri} que el manifiesto "
            f"no declara ({sorted(sobran)[0]}…)"
        )
    for parte in a.parts:
        ruta = a.uri + parte.name
        if ruta not in hijos:
            inf.errores.append(f"asset {a.id}: falta {ruta}, que el manifiesto declara")
            continue
        crudo = z.read(ruta)
        if hashlib.sha256(crudo).hexdigest() != parte.sha256:
            inf.errores.append(f"asset {a.id}: el sha256 de {parte.name} no cuadra")
        if len(crudo) != parte.bytes:
            inf.errores.append(
                f"asset {a.id}: {parte.name} declara {parte.bytes} bytes y tiene {len(crudo)}"
            )
    if (real := digesto_de_partes(a.parts)) != a.sha256:
        inf.errores.append(
            f"asset {a.id}: el digesto declarado del directorio no es el de sus partes "
            f"({a.sha256[:12]}… vs {real[:12]}…)"
        )
    if sum(p.bytes for p in a.parts) != a.bytes:
        inf.errores.append(
            f"asset {a.id}: declara {a.bytes} bytes y sus partes suman "
            f"{sum(p.bytes for p in a.parts)}"
        )
    if a.sidecar_uri is not None and a.sidecar_uri not in dentro:
        inf.errores.append(
            f"asset {a.id}: declara el sidecar {a.sidecar_uri} y no esta en el contenedor"
        )


_FDI = {str(x) for x in (
    *range(11, 19), *range(21, 29), *range(31, 39), *range(41, 49),
    # ISO 3950 cubre tambien la denticion TEMPORAL, y el validador tiene que aceptarla:
    # un caso pediatrico no es un caso invalido (D-9).
    *range(51, 56), *range(61, 66), *range(71, 76), *range(81, 86),
)} | {"0"}


def _valida_derivados(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Lo que `derived/` promete de si mismo, comprobado (T-3).

    El §5.5 pide que TODO asset bajo `derived/` traiga un sidecar que diga que modelo lo
    produjo, con que pesos y sobre que assets. El check que habia solo verificaba que los
    sidecars **declarados** existieran: uno que no se declarase no lo miraba nadie, asi que
    la regla se podia incumplir sin que saltara nada.
    """
    dentro = set(z.namelist())
    ids = {a.id for a in m.assets}
    for a in m.assets:
        if not a.uri.startswith("derived/"):
            continue
        if not a.sidecar_uri:
            inf.errores.append(
                f"asset {a.id}: vive en derived/ y no declara `sidecar_uri`. Sin el nadie "
                "puede saber que modelo lo produjo, y esa es la condicion del §5.5"
            )
            continue
        if a.sidecar_uri not in dentro:
            continue
        try:
            meta = json.loads(z.read(a.sidecar_uri))
        except ValueError:
            inf.errores.append(f"asset {a.id}: su sidecar {a.sidecar_uri} no es JSON valido")
            continue
        if not (meta.get("model") or {}).get("name"):
            inf.errores.append(
                f"asset {a.id}: su sidecar no dice `model.name`. «Lo produjo un modelo» "
                "sin decir cual no se puede auditar ni reproducir"
            )
        if not meta.get("source_assets"):
            inf.errores.append(
                f"asset {a.id}: su sidecar declara `source_assets` vacio. Una inferencia "
                "sin entradas declaradas no se puede rehacer"
            )
        if not meta.get("encoding"):
            inf.errores.append(
                f"asset {a.id}: su sidecar no declara `encoding`. Un `int16` suelto sin "
                "decir que indexa es un monton de numeros"
            )
        # T-3 · las etiquetas tienen que ser codigos FDI de verdad, no enteros cualesquiera.
        vocabulario = (meta.get("labels") or {}).get("present")
        if vocabulario is not None:
            malos = sorted({str(c) for c in vocabulario} - _FDI)
            if malos:
                inf.errores.append(
                    f"asset {a.id}: su sidecar declara etiquetas que no son codigos ISO "
                    f"3950: {', '.join(malos)}"
                )

    # T-3 · `derived_from` tiene que apuntar a algo declarado.
    for a in m.assets:
        colgando = [f for f in a.derived_from if f.startswith("asset.") and f not in ids]
        if colgando:
            inf.errores.append(
                f"asset {a.id}: su `derived_from` cita {', '.join(colgando)}, que el "
                "manifiesto no declara. Una procedencia que no resuelve no es procedencia"
            )


def _valida_entradas_declaradas(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Que no viaje nada que el manifiesto no nombre (§14.6, T-3).

    ⚠️ **Es la direccion que faltaba.** Se comprobaba que todo lo declarado estuviera; no
    que todo lo que esta estuviera declarado. Un fichero de mas en el ZIP viaja sin que
    nadie lo haya declarado, sin hash que lo acredite y sin capa regulatoria — que es
    exactamente la forma que tendria una fuga.
    """
    declarados = {MANIFIESTO, VISTAS, CADENA, OBSERVACIONES, FIRMAS}
    for a in m.assets:
        # ⚠️ El sidecar de un asset EXTERNO si viaja: es lo que permite situar un volumen
        # que el contenedor referencia sin custodiar. Lo que no viaja es el asset.
        if a.sidecar_uri:
            declarados.add(a.sidecar_uri)
        if a.external:
            continue
        declarados.add(a.uri)
        for parte in a.parts:
            declarados.add(a.uri + parte.name)
    sobran = [
        n for n in z.namelist()
        if not n.endswith("/")
        and n not in declarados
        and not any(n.startswith(d) for d in declarados if d.endswith("/"))
        # Los sidecars de `derived/` van declarados por su asset; los `.meta.json` que
        # acompanan a un binario declarado se admiten por vecindad.
        and not (n.endswith(".meta.json") and n.removesuffix(".meta.json") + ".bin" in declarados)
    ]
    if sobran:
        inf.errores.append(
            "el contenedor lleva ficheros que el manifiesto no declara: "
            + ", ".join(sorted(sobran))
            + ". Todo lo que viaja tiene que estar declarado (§14.6)"
        )


def _valida_frames(m: Manifiesto, inf: Informe) -> None:
    """El grafo de frames DEBE ser conexo hacia el canonico (§6).

    Sin esto un asset puede declarar un frame que no se sabe alinear con nada, y el visor
    lo colocaria en el sitio equivocado sin poder detectarlo.
    """
    canonico = m.canonical_frame.id
    alcanzables = {canonico}
    aristas = [(r.source_frame, r.target_frame) for r in m.registrations]
    cambio = True
    while cambio:
        cambio = False
        for origen, destino in aristas:
            for x, y in ((origen, destino), (destino, origen)):
                if y in alcanzables and x not in alcanzables:
                    alcanzables.add(x)
                    cambio = True
    for asset in m.assets:
        if asset.frame not in alcanzables:
            inf.errores.append(
                f"asset {asset.id}: su frame {asset.frame!r} no conecta con el canonico "
                f"{canonico!r} por ninguna registracion"
            )
    for r in m.registrations:
        if r.provisional:
            inf.avisos.append(
                f"registro {r.id}: automatico y sin `verified_by` — el visor debe "
                "presentarlo como PROVISIONAL"
            )


def _valida_regulatorio(m: Manifiesto, inf: Informe) -> None:
    """`derived/` implica layer 3 y sidecar `meta.json` (§5.5), y al reves."""
    for a in m.assets:
        en_derived = a.uri.startswith("derived/")
        if en_derived and a.regulatory.layer != 3:
            inf.errores.append(
                f"asset {a.id}: vive en derived/ y declara layer {a.regulatory.layer}; "
                "todo lo que sale de inferencia es layer 3"
            )
        if a.regulatory.layer == 3 and not en_derived:
            inf.errores.append(
                f"asset {a.id}: declara layer 3 y NO vive en derived/, asi que no se "
                "puede desmontar borrando ese directorio"
            )
        # ⚠️ **Capa 2 sin `derived_from` es una afirmacion que no se puede comprobar.**
        # La capa 2 dice «esto es computo reproducible a partir de capa 1». Si no se
        # declara a partir de QUE, no hay nada que reproducir y la etiqueta solo sirve
        # para sacar el asset del escrutinio que tendria como capa 3.
        if a.regulatory.layer == 2 and not a.derived_from:
            inf.errores.append(
                f"asset {a.id}: declara layer 2 y no dice `derived_from`. La capa 2 es "
                "computo reproducible; sin sus fuentes esa afirmacion no se puede "
                "comprobar"
            )
        # `clearances: []` significa NO DECLARADO, por definicion escrita. En un asset de
        # capa 3 —que es el que un regulador miraria— el silencio se avisa.
        if a.regulatory.layer == 3 and not a.regulatory.clearances:
            inf.avisos.append(
                f"asset {a.id}: es layer 3 y no declara ninguna `clearance`. Vacio "
                "significa «no consta», no «no hace falta»"
            )
    # ── D-9 · la mordida, dicha o declarada ausente, nunca callada ─────────────
    if m.occlusion is None and not any(r.id == Registro.OCLUSION for r in m.registrations):
        inf.avisos.append(
            "el contenedor no declara `occlusion`. Mandibula<->maxila es la registracion "
            "clinicamente mas importante de un caso dental; en uno de una sola arcada la "
            "respuesta es `single_arch`, pero hay que darla"
        )
    for r in m.registrations:
        # ⚠️ Aptitud: `rms_error_mm` es un PROMEDIO y no decide un uso clinico. 0,666 mm de
        # RMS vale para mirar y no para planificar una cirugia guiada, y un lector que no
        # tenga esta lista no puede distinguirlo — supondra, que es lo que hay que impedir.
        if r.fit_for and Aptitud.CIRUGIA_GUIADA in r.fit_for and r.max_error_mm is None:
            inf.errores.append(
                f"registro {r.id}: se declara apto para cirugia guiada y no trae "
                "`max_error_mm`. Para ese uso decide el error maximo local, no el promedio"
            )
        if not r.fit_for:
            inf.avisos.append(
                f"registro {r.id}: `fit_for` vacio, o sea NO DECLARADO. Un lector no debe "
                "suponer que sirve para medir ni para planificar"
            )

    # ── D-1 y D-2 · los frames se anclan a DICOM y declaran su convencion ──────
    de_volumen = {a.frame for a in m.assets if a.kind == Clase.VOLUME}
    for f in [m.canonical_frame, *m.frames]:
        if f.id in de_volumen:
            if not f.dicom_frame_of_reference_uid:
                inf.errores.append(
                    f"frame {f.id}: lo declara un asset `volume` y no trae "
                    "`dicom_frame_of_reference_uid`. DICOM ya identifica un sistema de "
                    "coordenadas con `(0020,0052)`; sin el, un lector que reciba la serie "
                    "por otro canal solo puede fiarse del nombre"
                )
            if f.anatomical != Anatomico.LPS:
                inf.errores.append(
                    f"frame {f.id}: lo declara un asset `volume` y su `anatomical` es "
                    f"{f.anatomical or 'null'}. DICOM impone LPS; «diestro» fija la "
                    "quiralidad, no que direccion es anterior o superior del paciente"
                )
    canonico = m.canonical_frame
    if canonico.anatomical in (None, Anatomico.DISPOSITIVO):
        anatomicos = {f.id for f in [canonico, *m.frames]
                      if f.anatomical in (Anatomico.LPS, Anatomico.RAS)}
        conecta = any(r.source_frame == canonico.id and r.target_frame in anatomicos
                      or r.target_frame == canonico.id and r.source_frame in anatomicos
                      for r in m.registrations)
        if not conecta:
            inf.avisos.append(
                f"el frame canonico {canonico.id} no declara convencion anatomica y no "
                "hay registracion que lo lleve a un frame LPS o RAS: nadie puede medir un "
                "angulo ni una distancia a una estructura sin mirar la imagen"
            )
    for r in m.registrations:
        # ⚠️ Una registracion que calculo una maquina es COMPUTO, no adquisicion, y tiene
        # que decirlo. Antes `regulatory` tenia defecto y toda registracion llegaba con
        # `layer: 1` puesto: la capa del calculo automatico era indistinguible de la de un
        # dato medido, y nadie podia ver la diferencia porque no habia diferencia escrita.
        if r.operator and r.operator.startswith(r.AUTO) and r.regulatory is None:
            inf.errores.append(
                f"registro {r.id}: lo calculo `{r.operator}` y no declara `regulatory`. "
                "Un alineamiento automatico es computo (layer 2), no adquisicion"
            )


def _valida_capas_en_la_escena(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Que ningun glTF de Layer 1 o 2 lleve dentro el codigo FDI (B-1).

    ⚠️ **Este es el check que hace verificable la removibilidad de `derived/`.** El de
    `_valida_regulatorio` comprueba donde se DECLARA la capa 3; este comprueba donde esta
    su CONTENIDO, que es otra cosa. Durante la 0.4.0 el manifiesto declaraba la escena
    como Layer 1 —y pasaba— mientras la escena viajaba partida en un *primitive* por
    diente con `extras.uos_fdi`, o sea con la salida del segmentador horneada dentro.
    Borrar `derived/` no borraba la inferencia, y el §3.1 promete que si.

    El codigo FDI aparece de TRES formas y las tres son Layer 3: `extras.uos_fdi` por
    *primitive* de malla, `_REGION_ID` por gaussiana en el glTF, y la columna `region_id`
    de un PLY de gaussianas. En un asset de Layer 3 las tres son licitas; fuera de el, no.
    La tercera es la que sobrevivio a la primera pasada de B-1: la revision externa nombro
    las dos del glTF porque reviso la spec, no este contenedor.
    """
    for a in m.assets:
        if a.regulatory.layer == 3:
            continue
        if a.uri.endswith(".ply"):
            try:
                cabecera = z.read(a.uri).split(b"end_header")[0].decode("ascii", "replace")
            except KeyError:
                continue
            if any(
                lin.startswith("property ") and lin.split()[-1] == "region_id"
                for lin in cabecera.splitlines()
            ):
                inf.errores.append(
                    f"asset {a.id}: {a.uri} declara la columna `region_id` y el asset es "
                    f"layer {a.regulatory.layer}; el codigo FDI por gaussiana es salida de "
                    "modelo y va en derived/seg_gaussians, no dentro de la capa"
                )
            continue
        if not a.uri.endswith(".glb"):
            continue
        try:
            crudo = z.read(a.uri)
            largo = int.from_bytes(crudo[12:16], "little")
            doc = json.loads(crudo[20:20 + largo])
        except (KeyError, ValueError):
            # Que el GLB no se pueda leer ya lo dice `_valida_assets`; aqui no se repite.
            continue
        for malla in doc.get("meshes", []):
            for pr in malla.get("primitives", []):
                if "uos_fdi" in (pr.get("extras") or {}):
                    inf.errores.append(
                        f"asset {a.id}: un primitive de {a.uri} declara `extras.uos_fdi` "
                        f"y el asset es layer {a.regulatory.layer}; el codigo FDI sale de "
                        "un segmentador, asi que borrar derived/ no quitaria la inferencia"
                    )
                if "_REGION_ID" in (pr.get("attributes") or {}):
                    inf.errores.append(
                        f"asset {a.id}: un primitive de {a.uri} lleva `_REGION_ID` y el "
                        f"asset es layer {a.regulatory.layer}; el FDI por gaussiana es "
                        "salida de modelo y no puede viajar en un plano que no se desmonta"
                    )


def _valida_gs(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """La primitiva `KHR_gaussian_splatting`, contra lo que la extension exige (T-3).

    ⚠️ **El texto decia «la implementacion de referencia revienta con las dos» y el
    algoritmo no las comprobaba.** Una escala negativa o una opacidad fuera de `[0,1]` no
    rompen el fichero: lo pintan mal, en silencio, en el visor de otro. Y un cuaternion sin
    normalizar codifica una escala encubierta que gira cada elipse un poco.

    Y los grados de armonicos esfericos: si existe el grado *l*, tienen que existir todos
    los inferiores COMPLETOS. Un grado 1 con dos de sus tres coeficientes no es «casi»
    correcto — un lector que itere por grados leera basura del buffer contiguo.
    """
    import numpy as np

    for a in m.assets:
        if not a.uri.endswith(".glb") or a.external:
            continue
        try:
            crudo = z.read(a.uri)
            largo = int.from_bytes(crudo[12:16], "little")
            doc = json.loads(crudo[20:20 + largo])
            binario = crudo[20 + largo + 8:]
        except (KeyError, ValueError, IndexError):
            continue

        def _lee(indice: int, doc: dict = doc, binario: bytes = binario) -> np.ndarray | None:
            try:
                acc = doc["accessors"][indice]
                vista = doc["bufferViews"][acc["bufferView"]]
                tipos = {5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
                n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
                dt = np.dtype(tipos[acc["componentType"]])
                off = vista.get("byteOffset", 0) + acc.get("byteOffset", 0)
                arr = np.frombuffer(binario, dt, count=acc["count"] * n, offset=off)
                return arr.reshape(-1, n) if n > 1 else arr
            except (KeyError, IndexError, ValueError):
                return None

        for malla in doc.get("meshes", []):
            for pr in malla.get("primitives", []):
                attrs = pr.get("attributes") or {}
                if not any(k.startswith("KHR_gaussian_splatting:") for k in attrs):
                    continue
                escala = _lee(attrs["KHR_gaussian_splatting:SCALE"]) \
                    if "KHR_gaussian_splatting:SCALE" in attrs else None
                if escala is not None and float(np.min(escala)) < 0:
                    inf.errores.append(
                        f"asset {a.id}: `SCALE` de la primitiva de gaussianas tiene valores "
                        "negativos. La extension pide escala lineal NO negativa"
                    )
                op = _lee(attrs["KHR_gaussian_splatting:OPACITY"]) \
                    if "KHR_gaussian_splatting:OPACITY" in attrs else None
                if op is not None and (float(np.min(op)) < 0 or float(np.max(op)) > 1):
                    inf.errores.append(
                        f"asset {a.id}: `OPACITY` fuera de [0,1]. La extension pide opacidad "
                        "LINEAL, y un PLY de INRIA la guarda en logit: confundirlas pinta "
                        "la escena entera mal sin que nada falle"
                    )
                rot = _lee(attrs["KHR_gaussian_splatting:ROTATION"]) \
                    if "KHR_gaussian_splatting:ROTATION" in attrs else None
                if rot is not None and rot.size:
                    largos = np.linalg.norm(rot, axis=1)
                    if float(np.max(np.abs(largos - 1.0))) > 1e-3:
                        inf.errores.append(
                            f"asset {a.id}: `ROTATION` no es unitario. Un cuaternion sin "
                            "normalizar codifica una escala encubierta"
                        )
                # Grados SH completos: si esta el grado l, estan todos los inferiores.
                grados: dict[int, set[int]] = {}
                for k in attrs:
                    if ":SH_DEGREE_" in k:
                        _, cola = k.split(":SH_DEGREE_")
                        grado, _, coef = cola.partition("_COEF_")
                        grados.setdefault(int(grado), set()).add(int(coef))
                if grados:
                    tope = max(grados)
                    for grado in range(tope + 1):
                        # El grado g tiene 2g+1 coeficientes.
                        if grados.get(grado, set()) != set(range(2 * grado + 1)):
                            inf.errores.append(
                                f"asset {a.id}: los armonicos esfericos declaran hasta el "
                                f"grado {tope} y el grado {grado} esta incompleto. Un "
                                "lector "
                                "que itere por grados leera basura del buffer contiguo"
                            )
                            break


def _cuenta_vertices(z: zipfile.ZipFile, a) -> int | None:
    """Cuantos vertices declara el accesor `POSITION` de un GLB."""
    try:
        crudo = z.read(a.uri)
        largo = int.from_bytes(crudo[12:16], "little")
        doc = json.loads(crudo[20:20 + largo])
        idx = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        return int(doc["accessors"][idx]["count"])
    except (KeyError, IndexError, ValueError):
        return None


def _hash_posiciones(z: zipfile.ZipFile, a) -> str | None:
    """SHA-256 del accesor `POSITION` de un GLB, en el orden declarado (T-4)."""
    try:
        crudo = z.read(a.uri)
        largo = int.from_bytes(crudo[12:16], "little")
        doc = json.loads(crudo[20:20 + largo])
        binario = crudo[20 + largo + 8:]
        acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        vista = doc["bufferViews"][acc["bufferView"]]
        off = vista.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return hashlib.sha256(binario[off:off + acc["count"] * 12]).hexdigest()
    except (KeyError, IndexError, ValueError):
        return None


def _valida_matrices_gs(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Que la matriz de un nodo GS sea la registracion que dice ser (T-3, punto 12).

    ⚠️ **Es el fallo clasico de este sitio y no lo comprobaba nadie.** glTF guarda las
    matrices **por columnas** y el manifiesto **por filas**; confundirlas no revienta nada:
    coloca la nube girada y espejada, con muy buen aspecto, en el visor de otro. El §5.2
    declara normativo que la `matrix` de un nodo colgado del canonico codifique la
    registracion que lleva su frame alli, y hasta ahora esa afirmacion vivia solo en el
    codigo del escritor.

    Se compara con tolerancia: son `float32` en el GLB y `float64` en el manifiesto.
    """
    import numpy as np

    registros = {(r.source_frame, r.target_frame): r for r in m.registrations}
    for a in m.assets:
        if not a.uri.endswith(".glb") or a.external:
            continue
        try:
            crudo = z.read(a.uri)
            largo = int.from_bytes(crudo[12:16], "little")
            doc = json.loads(crudo[20:20 + largo])
        except (KeyError, ValueError):
            continue
        canonico = m.canonical_frame.id
        for nodo in doc.get("nodes", []):
            uri_gs = (nodo.get("extras") or {}).get("uos_gs_uri")
            if uri_gs is None or "matrix" not in nodo:
                continue
            capa = next((x for x in m.assets if x.uri == uri_gs), None)
            if capa is None:
                inf.errores.append(
                    f"asset {a.id}: un nodo apunta a {uri_gs!r} con `uos_gs_uri` y el "
                    "manifiesto no declara ese asset"
                )
                continue
            if capa.frame == canonico:
                continue
            reg = registros.get((capa.frame, canonico))
            if reg is None:
                inf.errores.append(
                    f"asset {a.id}: el nodo de {capa.id} lleva `matrix` y no hay "
                    f"registracion de {capa.frame} a {canonico} que esa matriz pueda ser"
                )
                continue
            enel = np.asarray(nodo["matrix"], dtype=np.float64).reshape(4, 4)
            esperada = np.asarray(reg.transform_4x4_row_major, dtype=np.float64).reshape(4, 4)
            # glTF por COLUMNAS, manifiesto por FILAS: la del nodo es la traspuesta.
            if not np.allclose(enel, esperada.T, atol=1e-4):
                inf.errores.append(
                    f"asset {a.id}: la `matrix` del nodo de {capa.id} no es la traspuesta "
                    f"de la registracion {reg.id}. glTF guarda por columnas y el manifiesto "
                    "por filas: confundirlas coloca la nube girada y espejada sin que nada "
                    "falle"
                )


def _valida_union_segmentacion(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Que `derived/seg_*` indexe de verdad lo que dice indexar (T-3).

    La union entre la segmentacion y la escena es POSICIONAL: el codigo `i` es del vertice
    `i`. Si el recuento no cuadra, esa union no existe y las etiquetas se pintan sobre
    piezas equivocadas — en silencio, que es lo unico que importa aqui.
    """
    dentro = set(z.namelist())
    for a in m.assets:
        if not a.uri.startswith("derived/") or not a.sidecar_uri:
            continue
        if a.sidecar_uri not in dentro or a.uri not in dentro:
            continue
        try:
            meta = json.loads(z.read(a.sidecar_uri))
        except ValueError:
            continue
        codificacion = meta.get("encoding") or {}
        # T-4 · el hash del array de posiciones del asset fuente, si el sidecar lo declara.
        fuente = next((x for x in m.assets if x.id in (meta.get("source_assets") or [])
                       and x.uri.endswith(".glb")), None)
        esperado = codificacion.get("source_positions_sha256")
        if esperado:
            real = None if fuente is None else _hash_posiciones(z, fuente)
            if real is not None and real != esperado:
                inf.errores.append(
                    f"asset {a.id}: su sidecar acredita el orden de vertices con "
                    f"{esperado[:12]}… y el asset fuente da {real[:12]}…. La union es "
                    "posicional: con el orden cambiado, las etiquetas se pintan sobre las "
                    "piezas equivocadas"
                )
        cuenta = codificacion.get("count")
        if cuenta is None:
            continue
        # `int16` little-endian: dos bytes por codigo. Es lo que declara el propio sidecar.
        esperados = int(cuenta) * 2
        crudo_seg = z.read(a.uri)
        if len(crudo_seg) != esperados:
            inf.errores.append(
                f"asset {a.id}: su sidecar declara {cuenta} codigos ({esperados} bytes) y "
                f"el fichero tiene {len(crudo_seg)}. La union con la escena es posicional, "
                "asi que un desajuste pinta las etiquetas sobre las piezas equivocadas"
            )
            continue

        # ⚠️ **T-3, punto 8: contra el asset FUENTE y no solo contra si mismo.** Que el
        # sidecar cuadre con su propio binario no dice nada: los dos los escribe el mismo
        # emisor en la misma linea. Lo que hace verificable la union es que el recuento
        # coincida con los vertices de la escena que dice indexar.
        if fuente is not None:
            vertices = _cuenta_vertices(z, fuente)
            if vertices is not None and vertices != int(cuenta):
                inf.errores.append(
                    f"asset {a.id}: indexa {cuenta} elementos y {fuente.id} declara "
                    f"{vertices} vertices. La union es por indice: con recuentos distintos "
                    "no existe"
                )

        # ⚠️ **T-3, punto 7: los CODIGOS, no la lista que el sidecar dice tener.** Se
        # validaba `labels.present`, que es lo que el emisor afirma; esto lee los int16 que
        # de verdad viajan. Un codigo que no es ISO 3950 no es un diente y nadie sabra que
        # pintar con el.
        import numpy as np

        codigos = np.frombuffer(crudo_seg, dtype="<i2")
        fuera = sorted({str(int(c)) for c in np.unique(codigos)} - _FDI)
        if fuera:
            inf.errores.append(
                f"asset {a.id}: su contenido trae codigos que no son ISO 3950 ni 0: "
                + ", ".join(fuera[:8]) + ("…" if len(fuera) > 8 else "")
            )


def _valida_capas_clinicas(z: zipfile.ZipFile, inf: Informe) -> None:
    """Que cada valor de `clinical/` declare una capa coherente con como se obtuvo (B-2).

    ⚠️ **El fichero declaraba `layer: 1` para todo su contenido y no era cierto.** El
    grueso es la transcripcion de un informe firmado —capa 1— pero el bloque `color` lo
    calcula el pipeline desde las fotos: nadie lo firmo. El formato tenia procedencia de
    extraccion por valor (`derivation`) y NO tenia capa regulatoria por valor, asi que
    nadie respondia *quien responde* por cada dato. Un lector que borrase `derived/` se
    quedaba con mediciones computadas creyendo conservar un informe.

    Dos reglas, las dos del §9 llevadas al interior del fichero:

    1. Un valor `inferred` no puede ser capa 1. Si lo propuso un modelo, no es ni
       adquisicion ni transcripcion, por mucho que acabe siendo correcto.
    2. Capa 3 dentro de `clinical/` es un error. La capa 3 vive SOLO bajo `derived/`,
       porque es lo unico que hace cierto que borrar ese directorio quita la inferencia.
    """
    try:
        doc = json.loads(z.read("clinical/observations.json"))
    except (KeyError, ValueError):
        return
    for pieza in doc.get("teeth", []):
        fdi = pieza.get("fdi", "?")
        for campo, valor in pieza.items():
            if not isinstance(valor, dict) or "regulatory" not in valor:
                continue
            capa = valor["regulatory"].get("layer")
            if capa == 3:
                inf.errores.append(
                    f"clinical/observations.json: `{fdi}.{campo}` declara layer 3 y la "
                    "capa 3 vive solo bajo derived/. Aqui no se puede desmontar"
                )
            if valor.get("derivation") == "inferred" and capa == 1:
                inf.errores.append(
                    f"clinical/observations.json: `{fdi}.{campo}` es `inferred` y declara "
                    "layer 1. Lo que propuso un modelo no es adquisicion ni transcripcion"
                )
            if capa == 2 and not valor.get("derived_from"):
                inf.errores.append(
                    f"clinical/observations.json: `{fdi}.{campo}` declara layer 2 y no "
                    "dice `derived_from`; sin sus fuentes no se puede reproducir"
                )


def _valida_procedencia(
    z: zipfile.ZipFile, m: Manifiesto, manifiesto_sha256: str, inf: Informe
) -> None:
    """La cadena de hashes entre versiones (§8), si el manifiesto la declara.

    Declararla y que no cuadre es peor que no declararla: un lector que ve
    `provenance.chain` da por hecho que puede recorrer el historial del caso.
    """
    dentro = set(z.namelist())
    if m.provenance.chain is None:
        if m.provenance.prev_manifest_sha256 is not None:
            inf.avisos.append(
                "el manifiesto viene de una version anterior y no declara "
                f"`chain`: el historial existe pero no hay {CADENA} que recorrer"
            )
        if CADENA in dentro:
            inf.errores.append(
                f"el contenedor lleva {CADENA} y el manifiesto no lo declara: "
                "una cadena que nadie referencia no se puede verificar"
            )
        return
    if m.provenance.chain != CADENA:
        inf.errores.append(
            f"el manifiesto declara la cadena en {m.provenance.chain!r} y el spec la "
            f"situa en {CADENA!r}"
        )
        return
    if CADENA not in dentro:
        inf.errores.append(f"el manifiesto declara {CADENA} y no esta en el contenedor")
        return
    try:
        cadena = Cadena.model_validate_json(z.read(CADENA))
    except ValueError as e:
        inf.errores.append(f"{CADENA} no es una cadena valida: {e}")
        return
    inf.errores += revisa_cadena(
        cadena, case_id=m.case_id, manifiesto_sha256=manifiesto_sha256,
        prev_declarado=m.provenance.prev_manifest_sha256,
    )
    _valida_cadena_ordenada(cadena, inf)
    inf.version = len(cadena.links)

    # ⚠️ Las firmas no se verifican, y por eso se AVISAN. Ignorarlas en silencio dejaria
    # un `.uos` que parece firmado ante quien lo abra y que nadie ha comprobado.
    if firmas := [n for n in dentro if n.startswith(FIRMAS) and not n.endswith("/")]:
        inf.avisos.append(
            f"{len(firmas)} firma(s) en {FIRMAS} que este validador NO comprueba: "
            "la verificacion Ed25519 del spec §8 no esta implementada"
        )


def _valida_cadena_ordenada(cadena, inf: Informe) -> None:
    """Que la cadena sea una SUCESION y no un conjunto (T-3).

    El §8 la define solo-anexar y numerada. Comprobar los hashes sin comprobar el orden
    deja pasar un historial cuyos eslabones cuadran uno a uno y cuentan otra historia:
    una version 3 antes que la 2, o un `created` que retrocede.
    """
    for i, eslabon in enumerate(cadena.links):
        if eslabon.version != i + 1:
            inf.errores.append(
                f"cadena: el eslabon {i} declara version {eslabon.version} y le toca "
                f"{i + 1}. La cadena es una sucesion, no un conjunto"
            )
    fechas = [e.created for e in cadena.links]
    if any(b < a for a, b in zip(fechas, fechas[1:], strict=False)):
        inf.errores.append(
            "cadena: las fechas de `created` retroceden. Un historial solo-anexar no "
            "puede tener una version anterior escrita despues de la siguiente"
        )


def _valida_vistas(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Las vistas (§7) apuntan a visitas que existen y no repiten identificador."""
    if VISTAS not in set(z.namelist()):
        inf.avisos.append(f"el contenedor no lleva {VISTAS}: no hay vistas guardadas")
        return
    try:
        crudo = json.loads(z.read(VISTAS))
        vistas = [Vista.model_validate(v) for v in crudo.get("views", [])]
    except (ValueError, AttributeError) as e:
        inf.errores.append(f"{VISTAS} no es una lista de vistas valida: {e}")
        return
    visitas = {v.id for v in m.visits}
    hay_volumen = any(a.kind == Clase.VOLUME for a in m.assets)
    vistos: set[str] = set()
    for v in vistas:
        if v.visit not in visitas:
            inf.errores.append(
                f"vista {v.id}: apunta a la visita {v.visit!r}, que el manifiesto no declara"
            )
        if v.id in vistos:
            inf.errores.append(f"vista {v.id}: identificador repetido en {VISTAS}")
        vistos.add(v.id)
        # T-3 · `mpr` y `clip_planes` solo cuando hay volumen, y el §7 lo declara
        # normativo. Una vista con controles de volumen en un contenedor sin volumen le
        # promete a un visor una capa que no existe.
        if (v.mpr is not None or v.clip_planes is not None) != hay_volumen:
            inf.errores.append(
                f"vista {v.id}: declara controles de volumen (`mpr`/`clip_planes`) "
                + ("y el contenedor no lleva ninguno" if not hay_volumen
                   else "que faltan, y el contenedor SI lleva volumen")
            )
        # T-3 · una camara que no encuadra nada. El `up` paralelo a la direccion de vista
        # deja la matriz de vista degenerada y el visor pinta negro, o nada.
        import numpy as np

        d = np.asarray(v.camera.target, float) - np.asarray(v.camera.position, float)
        u = np.asarray(v.camera.up, float)
        nd, nu = float(np.linalg.norm(d)), float(np.linalg.norm(u))
        if nd == 0:
            inf.errores.append(f"vista {v.id}: la camara mira a su propia posicion")
        elif nu == 0 or abs(float(np.dot(d / nd, u / nu))) > 0.999:
            inf.avisos.append(
                f"vista {v.id}: `up` es (casi) paralelo a la direccion de vista; la matriz "
                "de vista queda degenerada y un visor pintara negro"
            )
        if not 1.0 < v.camera.fov < 179.0:
            inf.avisos.append(
                f"vista {v.id}: `fov` de {v.camera.fov} grados esta fuera de lo razonable"
            )
    inf.vistas = len(vistas)


def _es_medida(z: zipfile.ZipFile, a) -> bool:
    """Si el sidecar de una capa gaussiana declara que sus valores son medidos."""
    if not a.sidecar_uri:
        return False
    try:
        return bool(json.loads(z.read(a.sidecar_uri)).get("measured"))
    except (KeyError, ValueError):
        return False


def _valida_phi(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Que `phi_state` sea una afirmacion sostenible, y que el proposito conste (B-3, B-4).

    ⚠️ **`phi_state` hablaba de etiquetas DICOM y el contenedor identifica sin etiquetas.**
    Un campo gaussiano medido sobre un CBCT lleva tejido blando: de ahi se reconstruye una
    superficie facial, que HIPAA Safe Harbor cuenta como «imagen comparable» a una
    fotografia de cara completa y el RGPD como dato biometrico. Ningun valor de `phi_state`
    convierte esto en dato no personal; lo unico que se puede declarar con honradez es QUE
    SE HIZO, en el vocabulario de DICOM PS3.15 Anexo E.
    """
    identificado = m.phi_state == EstadoPHI.IDENTIFIED
    d = m.deidentification
    if not identificado and d is None:
        inf.errores.append(
            f"el manifiesto declara `phi_state: {m.phi_state.value}` y no trae bloque "
            "`deidentification`. Decir el estado sin decir que medidas lo produjeron es "
            "una afirmacion que nadie puede comprobar"
        )
    if d is not None:
        aplicado = set(d.applied_to)
        for a in m.assets:
            if a.kind in (Clase.VOLUME, Clase.IMAGE2D) and a.id not in aplicado:
                inf.avisos.append(
                    f"asset {a.id}: es {a.kind.value} y no aparece en "
                    "`deidentification.applied_to`. No estar en la lista significa que no "
                    "se le aplico nada"
                )
        # ⚠️ **La cara.** Si viaja una capa MEDIDA derivada de un volumen y no se declara
        # la limpieza de rasgos reconocibles, el contenedor lleva una superficie facial
        # reconstruible. Nada impide hoy generar el campo del CBCT original y aplicar el
        # defacing solo a la serie referenciada, que es la que nadie ve.
        # La senal es el descriptor, no el nombre del asset: `measured: true` en una capa
        # gaussiana significa que sus valores salen de una medida fisica, y la unica
        # modalidad que mide densidad aqui es el CBCT. Leerlo del sidecar y no de una lista
        # de ids evita que la regla deje de aplicarse el dia que un asset se llame distinto.
        de_volumen = [a.id for a in m.assets
                      if a.kind == Clase.MESH_GS_SCENE and _es_medida(z, a)]
        if not identificado and de_volumen and Desidentificacion.LIMPIA_RASGOS not in d.options:
            inf.errores.append(
                f"el contenedor lleva {', '.join(de_volumen)} —capa medida derivada de un "
                f"volumen— y `deidentification.options` no incluye "
                f"`{Desidentificacion.LIMPIA_RASGOS}`. De un campo de densidad con tejido "
                "blando se reconstruye la cara: eso es un identificador, con etiquetas o sin"
            )
        if d.date_shift_days is not None and not identificado:
            inf.errores.append(
                "`deidentification.date_shift_days` viaja con "
                f"`phi_state: {m.phi_state.value}`. El desplazamiento es la clave de "
                "re-identificacion: publicarlo deshace la medida que dice haber aplicado"
            )
    # B-4 · no se puede emitir para algo que el paciente no consintio.
    if m.purpose_of_use is not None:
        alcance = m.subject.consent.scope if m.subject.consent else []
        if m.purpose_of_use not in alcance:
            inf.errores.append(
                f"el contenedor se emite para `{m.purpose_of_use.value}` y "
                + ("el consentimiento no declara ese alcance"
                   if alcance else "el sujeto no declara consentimiento")
            )
    elif m.assets:
        inf.avisos.append(
            "el contenedor no declara `purpose_of_use`: salir hacia un laboratorio, hacia "
            "una segunda opinion o hacia un entrenamiento son actos distintos y quien lo "
            "reciba tendra que suponerlo"
        )


def _valida_extensiones(m: Manifiesto, inf: Informe) -> None:
    """Lo que se usa se declara, y lo que se exige se usa.

    ⚠️ Una extension `required` que este lector no conoce es un ERROR, no un aviso: el
    emisor esta diciendo «sin entender esto no abras el fichero», y abrirlo igual seria
    desobedecer la unica instruccion que el formato da al respecto. Una `used` que no se
    conoce es un aviso — se puede leer el caso sin ella y se dice que se esta ignorando.
    """
    declaradas = set(m.extensions)
    if huerfanas := set(m.extensions_used) - declaradas:
        inf.errores.append(
            f"el manifiesto usa {sorted(huerfanas)} y no las declara en `extensions`: "
            "un lector no tiene forma de saber que son"
        )
    if fuera := set(m.extensions_required) - set(m.extensions_used):
        inf.errores.append(
            f"el manifiesto EXIGE {sorted(fuera)} y no las declara como usadas: "
            "exigir algo que el fichero no usa deja el caso sin abrir para nada"
        )
    if sobran := declaradas - set(m.extensions_used):
        inf.avisos.append(
            f"el manifiesto declara {sorted(sobran)} y no las usa: sobran en `extensions`"
        )
    for nombre in m.extensions_required:
        inf.avisos.append(
            f"la extension `{nombre}` es OBLIGATORIA: un lector que no la implemente no "
            "debe abrir este contenedor"
        )
    dentro = None
    for nombre, ext in m.extensions.items():
        if ext.uri is None:
            continue
        if dentro is None:
            dentro = {a.uri for a in m.assets}
        if ext.uri not in dentro:
            inf.errores.append(
                f"la extension `{nombre}` apunta a `{ext.uri}`, que no es ningun asset "
                "declarado: una extension que referencia lo que no esta no se puede leer"
            )
