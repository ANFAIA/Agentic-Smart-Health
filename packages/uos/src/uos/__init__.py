"""UOS — Unified Oral Scene. Contenedor + manifiesto de composicion de un caso dental.

**Que es y que no.** Un `.uos` no es un formato de pixeles: es un ZIP sin comprimir que
**referencia los formatos nativos intactos** y declara sus relaciones espaciales y
temporales. El DICOM viaja byte-identico, la malla tambien, y el manifiesto dice como se
alinean. Ningun formato existente hace eso — DICOM no modela gaussianas ni se transmite
bien en web, glTF no modela volumenes ni metadatos clinicos, y OpenUSD es ajeno al
ecosistema clinico.

**Por que STORE y no compresion.** Los payloads ya vienen comprimidos (DICOM JPEG-LS, SPZ,
GLB con Draco), asi que comprimir el ZIP solo rompe el acceso aleatorio. Con STORE y el
directorio central al final, un cliente HTTP con *range requests* lee el indice y se baja
un asset suelto sin traerse el caso entero.

**Los tres planos, separados a proposito.** `assets` es el dato, `views` la presentacion y
`derived/` la inferencia. El tercero es desmontable: borrar el directorio y sus entradas
del manifiesto deja un `.uos` valido, que es lo que permite distribuir el caso en
jurisdicciones donde el modulo de IA no esta habilitado.

**Append-only logico.** Modificar un caso no es editar su `.uos`: es escribir una version
nueva del manifiesto que apunta al hash de la anterior (§8). Los assets no se tocan nunca
in place. `provenance/chain.json` materializa esa cadena y el validador comprueba que
cuente la misma historia que los manifiestos.

Implementa el nivel **UOS-Core** del spec v0.2 (§12): manifiesto + `mesh_gs_scene` +
`image2d`, con vistas (§7) y cadena de procedencia (§8).

Lo que NO esta, y se dice para que nadie lo de por hecho: el volumen y las senales
(`UOS-Vol` y `UOS-Sig`), el `fhir_map` (§9), que se declara vacio, y las firmas Ed25519
del §8 — a estas les falta la decision de que clave firma y donde vive, no el codigo.
"""

from uos.agente import UOSExportAgent
from uos.contenedor import escribe_uos, lee_manifiesto
from uos.manifiesto import (
    UOS_VERSION,
    Asset,
    Frame,
    Manifiesto,
    Registro,
    Sujeto,
    Visita,
)
from uos.procedencia import CADENA, Cadena, Eslabon
from uos.validador import Conformidad, valida
from uos.vistas import VISTAS, Camara, Vista, construye_vistas, marco_anatomico

__all__ = [
    "CADENA", "UOS_VERSION", "VISTAS", "Asset", "Cadena", "Camara", "Conformidad",
    "Eslabon", "Frame", "Manifiesto", "Registro", "Sujeto", "UOSExportAgent", "Vista",
    "Visita", "construye_vistas", "escribe_uos", "lee_manifiesto", "marco_anatomico",
    "valida",
]
