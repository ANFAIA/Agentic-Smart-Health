"""El banco de conformidad hace lo que su índice dice (G-4).

⚠️ **Un banco que nadie comprueba es peor que no tenerlo.** Existe para que un segundo
implementador contraste su lector contra algo que no sea su propia salida; si un caso
dejara de producir el error que anuncia, ese implementador concluiría que su lector está
mal cuando el equivocado es el banco. Así que el banco se genera y se verifica aquí, con
nuestro propio validador, en cada ejecución de la suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))


def test_cada_caso_del_banco_produce_lo_que_anuncia(tmp_path: Path) -> None:
    """Y cada uno falla por SU motivo, no por otro que arrastre.

    Los casos que parchean el manifiesto sueltan la cadena de procedencia a propósito: el
    §8 encadena el `sha256` de los bytes del manifiesto, así que tocarlo la invalida
    también. Un caso que quiere probar una regla y falla por dos no prueba ninguna — quien
    corra el banco no sabría cuál de las dos rechazó su lector.
    """
    from genera_fixture import genera
    from uos import validate

    destino = tmp_path / "banco"
    indice = genera(destino)
    assert len(indice) >= 13, "el banco tiene que cubrir algo más que el caso feliz"
    # Y en particular los tres desenlaces del check 7, que exigen un contenedor que
    # CUSTODIE la serie — algo que nuestro escritor no emite y un validador debe aceptar.
    por_corte = [c for c in indice if c["fichero"].startswith("serie-corte-")]
    assert len(por_corte) == 4, "faltan casos de la verificación corte a corte"
    # ⚠️ Y uno de ellos NO es un error: un corte de-identificado conserva su identidad
    # clínica y pierde sus bytes. Un validador que lo llame error dice «esta serie no es
    # la de este caso», que es falso y es la conclusión más cara posible.
    assert any(c["fichero"] == "serie-corte-deidentificado.uos" and c["espera"] == "aviso"
               for c in indice)

    esperado = json.loads((destino / "esperado.json").read_text(encoding="utf-8"))
    assert [c["fichero"] for c in esperado["casos"]] == [c["fichero"] for c in indice]

    for caso in indice:
        inf = validate(destino / caso["fichero"])
        if caso["espera"] == "error":
            assert not inf.valid, f"{caso['fichero']} tendría que fallar y pasa"
        else:
            assert inf.valid, f"{caso['fichero']}: {inf.errors}"


def test_el_caso_valido_del_banco_no_lleva_dato_de_paciente(tmp_path: Path) -> None:
    """La condición para que el banco se pueda publicar (B-3).

    Un banco hecho con un caso real sería útil una vez y no se podría distribuir, que es lo
    contrario de para lo que existe. El seudónimo declarado es el del generador, no uno
    derivado de ningún paciente.
    """
    import zipfile

    from genera_fixture import genera

    destino = tmp_path / "banco"
    genera(destino)
    m = json.loads(zipfile.ZipFile(destino / "valido.uos").read("manifest.json"))

    assert m["subject"]["pseudonym"] == "FIXTURE-0001"
    assert m["subject"].get("fhir_patient") is None
