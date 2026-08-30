"""§8 — `provenance/chain.json`: la cadena de hashes entre versiones del caso.

**Un `.uos` es append-only logico.** Modificar no es editar: es escribir una version nueva
del manifiesto que apunta al hash de la anterior. Los assets no se tocan nunca in place.
Eso convierte «este caso ha cambiado» en algo verificable por quien lo reciba, sin
confiar en el emisor y sin tener las versiones intermedias.

**Donde esta la autoridad.** En `prev_manifest_sha256`, DENTRO de cada manifiesto. La
cadena que se escribe aqui es la materializacion: la hace legible de un vistazo y permite
recorrerla sin abrir todas las versiones, pero no anade garantia — `chain.json` no esta
cubierto por ningun hash del manifiesto y quien reescriba el fichero no rompe nada
criptografico. Lo que si detecta el validador es que la cadena y los manifiestos NO
cuenten la misma historia, que es el fallo que importa.

⚠️ **Las firmas Ed25519 del spec no estan.** No es que falte el codigo: falta la decision
de que clave firma —la clinica emisora, la plataforma, o ambas— y donde vive. Firmar con
una clave inventada produciria un `.uos` que parece firmado, que es peor que uno que
declara no estarlo. El validador avisa si encuentra `provenance/signatures/` para no
ignorarlas en silencio.
"""

from __future__ import annotations

import hashlib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

CADENA = "provenance/chain.json"
FIRMAS = "provenance/signatures/"


class Eslabon(BaseModel):
    """Una version del caso: el hash de su manifiesto y el de la anterior."""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prev_manifest_sha256: str | None = None
    created: datetime
    generator: dict[str, str] = Field(default_factory=dict)
    assets: int = Field(default=0, ge=0)
    note: str = ""


class Cadena(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    links: list[Eslabon] = Field(default_factory=list)

    @property
    def ultimo(self) -> Eslabon | None:
        return self.links[-1] if self.links else None

    def json_canonico(self) -> str:
        return self.model_dump_json(indent=1)

    def continua(self, eslabon: Eslabon) -> Cadena:
        return Cadena(case_id=self.case_id, links=[*self.links, eslabon])


def sha256_texto(texto: str) -> str:
    """Hash de los BYTES que se escriben en el ZIP: `writestr` codifica en UTF-8."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def lee_version_previa(ruta: Path) -> tuple[str | None, Cadena | None, str | None]:
    """`(hash del manifiesto, cadena, aviso)` de un `.uos` anterior.

    Tolerante a proposito: si el fichero anterior no se puede abrir o no lleva cadena, se
    empieza una nueva. Un contenedor previo ilegible no debe impedir escribir el actual;
    lo que no puede pasar es que se declare una continuidad que no existe.

    ⚠️ **Y por eso la cadena previa se COMPRUEBA antes de continuarla.** Si su ultimo
    eslabon dice un hash que no es el del manifiesto que lo acompana, esa cadena ya estaba
    rota antes de llegar aqui: encadenar encima produciria una version N+1 cuyo `prev`
    apunta al manifiesto real y no al que el eslabon N declara, y el validador lo rechaza
    — con razon, pero rechazando el caso ENTERO por una corrupcion heredada.

    Paso de verdad: un contenedor con cuatro versiones donde el eslabon 3 declaraba
    `38f0a225…` y su manifiesto hasheaba `0ec380e4…`. La exportacion siguiente fallo
    completa, y el caso quedaba inexportable para siempre.

    Lo correcto no es ignorarlo ni repararlo —una cadena de procedencia que se auto-repara
    no sirve para nada—: es **no reclamar continuidad con algo que no cuadra**, empezar
    cadena nueva y decir por que. El aviso sube al gate de revision humana.
    """
    if not ruta.exists():
        return None, None, None
    try:
        with zipfile.ZipFile(ruta) as z:
            crudo = z.read("manifest.json")
            previo = hashlib.sha256(crudo).hexdigest()
            if CADENA not in z.namelist():
                return previo, None, None
            cadena = Cadena.model_validate_json(z.read(CADENA))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None, None, None

    # ⚠️ Se revisa la cadena ENTERA, no su ultimo eslabon. Comprobar solo el ultimo fue mi
    # primer intento y no sirvio: tras una exportacion fallida el ultimo eslabon cuadraba
    # con su manifiesto —lo escribio bien— y la rotura estaba DENTRO, entre la v3 y la v4.
    # La comprobacion lo daba por bueno, se encadenaba encima, y el validador seguia
    # rechazando el contenedor entero. Una cadena vale lo que valga su eslabon mas debil.
    roturas = revisa_cadena(
        cadena,
        case_id=cadena.case_id,
        manifiesto_sha256=previo,
        prev_declarado=cadena.links[-1].prev_manifest_sha256 if cadena.links else None,
    )
    if roturas:
        return None, None, (
            f"la version anterior de {ruta.name} tiene la cadena de procedencia ROTA "
            f"({roturas[0]}). No se encadena encima de eso: esta version empieza cadena "
            "nueva y la anterior queda como esta, que es lo que hay que poder auditar."
        )
    return previo, cadena, None


def encadena(
    *,
    case_id: str,
    manifiesto_json: str,
    previo_sha256: str | None,
    cadena_previa: Cadena | None,
    generator: dict[str, str],
    assets: int,
    note: str = "",
) -> Cadena:
    """La cadena que va en esta version, con su eslabon ya anadido.

    No hay circularidad: `manifest.json` y `chain.json` son entradas distintas del ZIP, y
    el manifiesto solo nombra la RUTA de la cadena, no su contenido. Asi que el hash del
    manifiesto ya esta fijado cuando se escribe el eslabon que lo lleva.
    """
    base = (
        cadena_previa
        if cadena_previa is not None and cadena_previa.case_id == case_id
        # Un `case_id` distinto no es la misma historia: se empieza cadena, no se mezcla.
        else Cadena(case_id=case_id)
    )
    return base.continua(Eslabon(
        version=len(base.links) + 1,
        manifest_sha256=sha256_texto(manifiesto_json),
        prev_manifest_sha256=previo_sha256,
        created=datetime.now(UTC),
        generator=generator,
        assets=assets,
        note=note,
    ))


def revisa_cadena(cadena: Cadena, *, case_id: str, manifiesto_sha256: str,
                  prev_declarado: str | None) -> list[str]:
    """Los errores de la cadena. Vacio si cuenta la misma historia que el manifiesto."""
    errores: list[str] = []
    if cadena.case_id != case_id:
        errores.append(
            f"{CADENA} es del caso {cadena.case_id!r} y el manifiesto del "
            f"{case_id!r}: son dos historias distintas"
        )
    if not cadena.links:
        errores.append(f"{CADENA} no lleva ningun eslabon")
        return errores
    for i, e in enumerate(cadena.links):
        esperado = None if i == 0 else cadena.links[i - 1].manifest_sha256
        if e.prev_manifest_sha256 != esperado:
            errores.append(
                f"{CADENA}: el eslabon {e.version} apunta a "
                f"{(e.prev_manifest_sha256 or 'nada')[:12]}… y el anterior es "
                f"{(esperado or 'nada')[:12]}… — la cadena esta rota"
            )
        if e.version != i + 1:
            errores.append(
                f"{CADENA}: el eslabon en posicion {i + 1} dice ser la version {e.version}"
            )
    ultimo = cadena.links[-1]
    if ultimo.manifest_sha256 != manifiesto_sha256:
        errores.append(
            f"{CADENA}: el ultimo eslabon declara el manifiesto "
            f"{ultimo.manifest_sha256[:12]}… y el de este contenedor es "
            f"{manifiesto_sha256[:12]}… — la cadena no termina aqui"
        )
    if ultimo.prev_manifest_sha256 != prev_declarado:
        errores.append(
            f"{CADENA}: el ultimo eslabon viene de "
            f"{(ultimo.prev_manifest_sha256 or 'nada')[:12]}… y el manifiesto declara "
            f"{(prev_declarado or 'nada')[:12]}…"
        )
    return errores
