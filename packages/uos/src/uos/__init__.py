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

Implementa los niveles **UOS-Core** y **UOS-Vol** del spec v0.2 (§12): manifiesto, escena
glTF con las capas de gaussianas colgando (§5.1), `image2d`, el volumen DICOM entero (§5.2)
y los derivados de inferencia (§5.5), con vistas (§7), cadena de procedencia (§8) y mapeo
FHIR por tipo de recurso (§9).

**Y tres cosas que el borrador NO tiene**, declaradas como extensiones para que un lector
ajeno sepa que estan: la capa clinica por pieza (`histora_clinical`), el descriptor que
distingue gaussianas MEDIDAS de reconstruidas (`histora_gs_measured`), y el propio mecanismo de
extensiones —copiado de glTF, sobre el que UOS se apoya—. Ninguna es obligatoria: un visor
conforme abre el caso sin entender ninguna.

Lo que NO esta, y se dice para que nadie lo de por hecho: las senales (`UOS-Sig`), y las
firmas Ed25519 del §8 — a estas les falta la decision de que clave firma y donde vive,
no el codigo.
"""

from uos.agente import UOSExportAgent
from uos.clinico import OBSERVACIONES, capa_clinica
from uos.contenedor import escribe_uos, lee_manifiesto
from uos.escena import NodoGS, construye_glb
from uos.manifiesto import (
    UOS_VERSION,
    Asset,
    Extension,
    Frame,
    Manifiesto,
    Registro,
    Sujeto,
    Visita,
)
from uos.procedencia import CADENA, Cadena, Eslabon
from uos.validador import Conformidad, valida
from uos.vistas import VISTAS, Camara, Vista, construye_vistas, marco_anatomico
from uos.volumen import describe_serie

__all__ = [
    "CADENA", "UOS_VERSION", "VISTAS", "Asset", "Cadena", "Camara", "Conformidad",
    "Eslabon", "Frame", "Manifiesto", "Registro", "Sujeto", "UOSExportAgent", "Vista",
    "Visita", "construye_vistas", "escribe_uos", "lee_manifiesto", "marco_anatomico",
    "Extension", "NodoGS", "OBSERVACIONES", "capa_clinica", "construye_glb",
    "describe_serie", "valida",
]
