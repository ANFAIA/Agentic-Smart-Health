"""`clinical/observations.json` — lo que el informe dice de cada pieza.

⚠️ **Esto es una EXTENSIÓN del borrador, no una parte suya.** El spec v0.2 no le da sitio
a los atributos clínicos por diente: el §9 los manda a `Observation` de FHIR, o sea a un
servidor externo. La consecuencia práctica es que un `.uos` suelto no puede responder «qué
dice el informe del 24», que es justo la pregunta que un clínico hace delante del modelo.
Se declara aquí con `kind: document`, se mapea a `Observation` en el `fhir_map`, y queda
escrito que es nuestro para que nadie lo confunda con formato ratificado.

**Por qué Layer 1 y no `derived/`.** Lo que viaja es la TRANSCRIPCIÓN de un informe que
firmó una persona: el pH que alguien midió, las raíces que alguien contó. Eso es registro
clínico. Meterlo en `derived/` lo haría desmontable, y borrar `derived/` dejaría un caso
sin lo que el informe dice — que no es lo que esa operación significa.

⚠️ **Pero la EXTRACCIÓN sí puede ser inferencia, y cada valor lo declara.** El contrato del
twin distingue `deterministic` de `inferred` por observación (`Provenance.derivation`), con
el modelo que la produjo cuando lo hubo. No es lo mismo un pH que un patrón encontró
escrito que uno que un modelo dedujo de la frase: los dos pueden ser correctos, pero sólo
el primero se puede volver a obtener exactamente y defender diciendo «lo pone aquí».

**Y `derivation: null` significa NO DECLARADO, no determinista.** Se propaga tal cual. Un
consumidor que reciba un valor sin declarar no debe darlo por reproducible — el silencio no
puede pasar por una afirmación, que es la misma regla que separa `MISSING` de `FAILED`.
"""

from __future__ import annotations

from typing import Any

from core_schemas import TwinSnapshot

OBSERVACIONES = "clinical/observations.json"


def _nota_color(color) -> str:
    """La nota del color, DERIVADA de la medida y no escrita a mano.

    ⚠️ **Una nota fija se queda mintiendo en cuanto cambia el metodo.** La que habia aqui
    advertia de que «el flash cayendo hacia el fondo de la boca entra en el numero» — y
    seguia diciendolo despues de que el emisor empezara a descontar esa caida con la encia
    del propio paciente como referencia. Quien leyera el contenedor descartaria por
    artefacto una diferencia entre piezas que ya era real. La nota se compone de
    `correccion_iluminacion`, que es lo unico que sabe si se corrigio o no.
    """
    base = "color medido por pieza; NO es un tono de guia certificado"
    if color.correccion_iluminacion is None:
        return (base + "; SIN corregir la caida del flash: este valor lleva dentro lo "
                "lejos que le llego la luz a esta pieza y no es comparable con el de "
                "las demas")
    b = color.correccion_iluminacion
    return (base + "; caida del flash descontada con la encia del propio paciente como "
            f"referencia (pendiente {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f} por canal), asi que "
            "las piezas SI son comparables entre si; el nivel absoluto no, porque la "
            "foto no lleva referencia gris")


def capa_clinica(snapshot: TwinSnapshot, motivos: list[str]) -> dict[str, Any]:
    """Las observaciones por pieza, las medidas no regionales y el gate.

    Los tres van juntos porque los tres son lo mismo: **lo que un clínico no puede deducir
    mirando la geometría**. Un modelo enseña dónde está el 24; que su confianza sea 0,58 y
    que el informe no diga nada de él son datos, y esconderlos detrás del modelo entrega un
    twin que parece más firme de lo que es.
    """
    piezas: dict[str, dict[str, Any]] = {}
    for obs in snapshot.regional:
        a = obs.attributes
        d = piezas.setdefault(obs.region_id, {"fdi": obs.region_id})
        # ⚠️ **La procedencia va POR VALOR, y antes colgaba de la pieza.** `confidence`,
        # `agent` y `derivation` se escribian una vez por diente sobre un `setdefault`: con
        # dos observaciones de la misma pieza, `findings` se acumulaba y estos tres los
        # pisaba la ultima, de modo que un valor quedaba anunciado con la procedencia de
        # otro. Y sobre todo, el mismo `confidence` flotaba sobre `color`, que viene de
        # otra cadena entera y no tiene nada que ver con el.
        marca = {
            "regulatory": {"layer": 1},
            # `deterministic` si lo saco un patron, `inferred` si lo propuso un modelo,
            # `null` si nadie lo declaro — que NO es lo mismo que determinista.
            "derivation": (obs.provenance.derivation.value
                           if obs.provenance.derivation else None),
            # ⚠️ **Que mide esto, dicho aqui porque el numero solo se malinterpreta.**
            # NO es «cuanto se fia el extractor»: es el eslabon mas debil de la cadena que
            # colgo este valor de ESTA pieza, y en la practica lo domina la confianza del
            # segmentador en el codigo FDI. Por eso un valor sacado por un patron
            # —`derivation: deterministic`— llega con 0,745: el patron es exacto, la pieza
            # a la que se le atribuye no lo es del todo.
            "confidence": round(obs.provenance.confidence, 3),
            "agent": obs.provenance.agent,
            "observed": obs.timestamp.isoformat(),
            **({"model": obs.provenance.model} if obs.provenance.model else {}),
        }
        if a.ph is not None:
            d["ph"] = {"value": a.ph, **marca}
        if a.n_raices is not None:
            d["n_roots"] = {"value": a.n_raices, **marca}
        if a.n_conductos is not None:
            d["n_canals"] = {"value": a.n_conductos, **marca}
        if a.hallazgos:
            previos = d.get("findings", {}).get("value", [])
            d["findings"] = {"value": previos + [h.value for h in a.hallazgos], **marca}
        # ⚠️ **El color es capa 2 y NO capa 1, que es lo que decia el fichero entero.**
        # Nadie firmo esto: lo calcula el pipeline desde las fotos. Tampoco es capa 3 —la
        # segmentacion de la foto en coronas es un watershed y el codigo FDI sale de
        # emparejar una huella de anchuras contra una tabla anatomica, sin modelo
        # entrenado—, asi que mandarlo a `derived/` lo declararia inferencia y lo haria
        # desaparecer al distribuir un caso sin IA, cuando aqui no hay ninguna IA.
        # Computo determinista y reproducible a partir de capa 1: capa 2.
        if a.color is not None:
            color = {
                "space": "CIELAB",
                "cervical": list(a.color.cervical),
                "middle": list(a.color.medio),
                "incisal": list(a.color.incisal),
                "from_photo": f"sha256:{a.color.foto_sha256}",
                "n_pixels": a.color.n_pixeles,
                "measured": True,
                "note": _nota_color(a.color),
            }
            # ⚠️ **Ausente NO es cero.** Una pieza sin corregir lleva dentro lo lejos que
            # le llego el flash y no es comparable con una corregida; un `0.0` por defecto
            # diria «se corrigio y no hizo falta», que es otra cosa.
            if a.color.correccion_iluminacion is not None:
                color["illumination_slope"] = list(a.color.correccion_iluminacion)
            d["color"] = {
                "value": color,
                "regulatory": {"layer": 2},
                "derivation": "deterministic",
                # La capa 2 exige decir a partir de QUE es reproducible. Aqui es la foto,
                # nombrada por su contenido porque el nombre del fichero lleva datos del
                # paciente.
                "derived_from": [f"sha256:{a.color.foto_sha256}"],
                "observed": obs.timestamp.isoformat(),
                # ⚠️ **Sin `confidence`, y es deliberado.** El del informe mide otra cosa
                # —el eslabon mas debil hasta la pieza— y ponerlo aqui diria que el color
                # vale 0,745. Lo que califica a esta medida es su propio soporte:
                # `n_pixels`, `illumination_slope` y `measured`.
            }

    return {
        "schema": "ash-clinical/2.0",
        "extension_of": "UOS v0.2 — el borrador no define atributos clinicos por pieza",
        "vocabulary": "ISO-3950 (FDI) para las piezas; el vocabulario de hallazgos es "
                      "cerrado y esta en `core_schemas.Hallazgo`",
        "regulatory": {
            "layer": 1,
            "default": True,
            "note": (
                "⚠️ ESTO ES EL DEFECTO DEL FICHERO, no una afirmacion sobre todo su "
                "contenido: cada valor de `teeth[]` declara su propia capa y manda sobre "
                "esta. El fichero es capa 1 porque su grueso es la transcripcion de un "
                "informe que firmo una persona; el `color` es capa 2, computado por el "
                "pipeline desde las fotos, y lo dice en su sitio"
            ),
            "confidence": (
                "el `confidence` de un valor NO mide cuanto se fia el extractor: es el "
                "eslabon mas debil de la cadena que colgo ese valor de ESA pieza, y lo "
                "domina la confianza del segmentador en el codigo FDI. Por eso un valor "
                "`deterministic` puede llegar con 0,745 sin contradecirse"
            ),
            "layers": (
                "1 = adquirido o transcrito de un informe firmado; 2 = computado por un "
                "procedimiento determinista y reproducible a partir de capa 1, sin modelo "
                "entrenado; 3 = salida de modelo, y NO puede aparecer en este fichero "
                "porque vive solo bajo `derived/`"
            ),
        },
        "teeth": [piezas[k] for k in sorted(piezas)],
        # ⚠️ Las medidas NO regionales existen porque no caben en una pieza: indices de
        # oclusion, cargas por lado. `RegionalObservation` exige un codigo FDI, y tres de
        # estas no lo tienen — antes DESAPARECIAN.
        "measurements": [
            {"name": m.nombre, "value": m.valor, "unit": m.unidad, "side": m.lado,
             "normal_min": m.normal_min, "normal_max": m.normal_max,
             "out_of_range": m.fuera_de_rango, "text": m.texto}
            for m in snapshot.medidas
        ],
        "review": {
            "note": "motivos por los que este caso pide revision humana antes de entregarse",
            "reasons": list(motivos),
        },
    }
