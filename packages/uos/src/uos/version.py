"""Comparar `uos_version` y decidir como leer lo que declara otra version (§15).

**El campo se escribia y no lo leia nadie.** Un `.uos` declara su version desde la primera
entrada del ZIP —es la identificacion positiva del formato— y ni el validador ni el lector
la miraban. Sin comparar no hay compatibilidad que valga: la regla de las menores es una
promesa que nadie puede cumplir si el lector no sabe ante que version esta.

**Las tres ramas, y la asimetria que las justifica.** Con una version igual o inferior se
lee ESTRICTO: dentro de una version se conoce el conjunto completo de campos, asi que uno
desconocido es una errata o una corrupcion y negarse es mejor que adivinar. Con una menor
superior se lee PERMISIVO: una menor solo anade, luego lo que este lector conoce sigue
significando lo mismo. Con una mayor superior se RECHAZA.

⚠️ **Y el peligro de la tercera rama no son los campos que no se conocen.** Esos se
ignorarian igual que en la segunda. Son los que SI se conocen y han cambiado de significado:
si en una 1.0 `density` dejara de ser sigma normalizada, un lector v0.2 abriria el fichero,
reconoceria el campo y lo interpretaria mal sin que nada fallara. Por eso «ignoro lo que no
entiendo» no basta ahi y la unica respuesta segura es no abrirlo.

⚠️ **Leer una version anterior no es obligatorio.** La rama estricta dice como leerla si el
lector decide soportarla, no que tenga que hacerlo.
"""

from __future__ import annotations

from enum import StrEnum

from uos.manifiesto import UOS_VERSION


class Lectura(StrEnum):
    """Que hacer con un contenedor segun la version que declare."""

    ESTRICTA = "estricta"       # igual o anterior: campo desconocido = error
    PERMISIVA = "permisiva"     # menor superior: se ignora lo que no se conoce
    RECHAZO = "rechazo"         # mayor superior: no se abre


def partes(version: str) -> tuple[int, int]:
    """`(mayor, menor)`. Una version que no se puede leer NO se asume compatible.

    ⚠️ Se eleva en vez de devolver `(0, 0)`. Un `uos_version` que no es un numero es un
    fichero que no sabemos que dice ser, y tratarlo como la version mas antigua lo leeria
    estricto y en silencio — que es exactamente adivinar.
    """
    try:
        mayor, menor, *_ = (*version.split("."), "0")
        return int(mayor), int(menor)
    except ValueError as e:
        raise ValueError(
            f"uos_version {version!r} no es <mayor>.<menor>: no se puede decidir como leerlo"
        ) from e


def como_leer(version: str, *, implementada: str = UOS_VERSION) -> Lectura:
    """La rama de §15 que corresponde a `version` para un lector de `implementada`."""
    suya, mia = partes(version), partes(implementada)
    if suya[0] > mia[0]:
        return Lectura.RECHAZO
    if suya[0] == mia[0] and suya[1] > mia[1]:
        return Lectura.PERMISIVA
    return Lectura.ESTRICTA


def puede_reemitir(version: str, *, implementada: str = UOS_VERSION) -> bool:
    """Si este lector puede escribir una version NUEVA de ese caso (§11).

    ⚠️ **Leer permisivo y reemitir no se pueden combinar, y es el fallo callado que esto
    evita.** Un lector v0.2 que abre un contenedor v0.3 ignora los campos que no entiende;
    si ademas escribiera la version N+1, los BORRARIA. La cadena de procedencia diria que
    esa version es sucesora legitima de la anterior —y lo seria criptograficamente— mientras
    el contenido se ha perdido sin que nadie se entere. Eso es peor que negarse.

    Conservar lo desconocido y volver a emitirlo seria la otra salida, y es mas trabajo del
    que hoy hace falta: nadie ha escrito una v0.3 todavia.
    """
    return como_leer(version, implementada=implementada) is Lectura.ESTRICTA


def lee_permisivo(crudo: bytes | str) -> tuple[object, list[str]]:
    """`(manifiesto, campos ignorados)` de una menor superior. Solo por esta via.

    **Se le pregunta a pydantic que sobra, en vez de mantener una copia del modelo.** La
    alternativa era un arbol de modelos paralelo con `extra="ignore"`, porque `extra` se
    fija en la clase y no en la llamada, y cada modelo del manifiesto lo lleva por separado.
    Un arbol duplicado se separa del bueno en el primer campo que alguien anada, que es el
    mismo fallo que evitamos generando el JSON Schema en vez de copiarlo a mano.

    Asi que se valida, se leen las rutas que el error marca como `extra_forbidden`, se podan
    y se reintenta. Lo que sobra sale nombrado —no se ignora en silencio— porque el lector
    tiene que poder decir QUE dejo sin leer.
    """
    import json

    from pydantic import ValidationError

    from uos.manifiesto import Manifiesto

    datos = json.loads(crudo) if isinstance(crudo, str | bytes) else crudo
    ignorados: list[str] = []
    for _ in range(64):   # cota: cada vuelta poda al menos un campo o termina
        try:
            return Manifiesto.model_validate(datos), sorted(ignorados)
        except ValidationError as e:
            sobra = [x["loc"] for x in e.errors() if x["type"] == "extra_forbidden"]
            if not sobra:
                raise
            for ruta in sobra:
                nodo = datos
                for paso in ruta[:-1]:
                    nodo = nodo[paso]
                nodo.pop(ruta[-1], None)
                ignorados.append(".".join(str(x) for x in ruta))
    raise ValueError("el manifiesto declara mas campos desconocidos de los que se podan")
