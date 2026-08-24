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
        d = piezas.setdefault(obs.region_id, {"fdi": obs.region_id, "findings": []})
        d["confidence"] = round(obs.provenance.confidence, 3)
        d["agent"] = obs.provenance.agent
        # ⚠️ `None` viaja como `null` y NO se sustituye por "deterministic". Es la
        # distinción entre «nadie lo declaró» y «alguien afirmó que es reproducible».
        d["derivation"] = (obs.provenance.derivation.value
                           if obs.provenance.derivation else None)
        if obs.provenance.model:
            d["model"] = obs.provenance.model
        d["observed"] = obs.timestamp.isoformat()
        if a.ph is not None:
            d["ph"] = a.ph
        if a.n_raices is not None:
            d["n_roots"] = a.n_raices
        if a.n_conductos is not None:
            d["n_canals"] = a.n_conductos
        d["findings"] += [h.value for h in a.hallazgos]

    return {
        "schema": "ash-clinical/1.0",
        "extension_of": "UOS v0.2 — el borrador no define atributos clinicos por pieza",
        "vocabulary": "ISO-3950 (FDI) para las piezas; el vocabulario de hallazgos es "
                      "cerrado y esta en `core_schemas.Hallazgo`",
        "regulatory": {
            "layer": 1,
            "note": (
                "transcripcion de un informe firmado por una persona, no salida de un "
                "modelo. Como se EXTRAJO cada valor lo dice su `derivation`: "
                "`deterministic` si lo saco un patron, `inferred` si lo propuso un "
                "modelo, `null` si nadie lo declaro — que no es lo mismo que determinista"
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
