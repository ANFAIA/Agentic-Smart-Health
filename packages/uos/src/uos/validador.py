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

from uos.contenedor import MANIFIESTO
from uos.manifiesto import Clase, Manifiesto, digesto_de_partes
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
    # Cuantas versiones del caso hay encadenadas (§8) y cuantas vistas guardadas (§7).
    # `version` es 0 cuando el contenedor no declara cadena: no es «la primera», es que
    # no hay historial que recorrer, y son cosas distintas.
    version: int = 0
    vistas: int = 0

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
        m = Manifiesto.model_validate_json(crudo)
        _valida_assets(z, m, inf)
        _valida_procedencia(z, m, hashlib.sha256(crudo).hexdigest(), inf)
        _valida_vistas(z, m, inf)

    _valida_frames(m, inf)
    _valida_regulatorio(m, inf)
    if m.canonical_frame.units != "mm":
        inf.errores.append(
            f"el frame canonico declara unidades {m.canonical_frame.units!r}; "
            "la convencion del spec es milimetros"
        )
    inf.niveles = [n for n, exige in _EXIGE.items()
                   if exige <= {a.kind for a in m.assets}]
    return inf


def _valida_assets(z: zipfile.ZipFile, m: Manifiesto, inf: Informe) -> None:
    """Cada asset existe y su hash cuadra. Verificar es la politica en ingesta (§8)."""
    dentro = set(z.namelist())
    for a in m.assets:
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
    inf.version = len(cadena.links)

    # ⚠️ Las firmas no se verifican, y por eso se AVISAN. Ignorarlas en silencio dejaria
    # un `.uos` que parece firmado ante quien lo abra y que nadie ha comprobado.
    if firmas := [n for n in dentro if n.startswith(FIRMAS) and not n.endswith("/")]:
        inf.avisos.append(
            f"{len(firmas)} firma(s) en {FIRMAS} que este validador NO comprueba: "
            "la verificacion Ed25519 del spec §8 no esta implementada"
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
    vistos: set[str] = set()
    for v in vistas:
        if v.visit not in visitas:
            inf.errores.append(
                f"vista {v.id}: apunta a la visita {v.visit!r}, que el manifiesto no declara"
            )
        if v.id in vistos:
            inf.errores.append(f"vista {v.id}: identificador repetido en {VISTAS}")
        vistos.add(v.id)
    inf.vistas = len(vistas)
