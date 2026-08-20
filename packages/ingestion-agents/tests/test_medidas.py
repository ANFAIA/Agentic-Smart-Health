"""Los valores del informe que el contrato NO interpreta.

`ClinicalAttributes` es cerrado a propósito: lo que entra ahí está tipado y acotado. Pero
un informe clínico real trae más de lo que anticipamos, y hasta ahora lo que no cabía
**desaparecía**. Estos tests atan la captura de eso y, sobre todo, atan que siga sin
interpretarse.
"""

from __future__ import annotations

from ingestion_agents.report_agent import extract_medidas_by_rules

# La forma literal de un informe de oclusión/ATM, tal como sale de `pdftotext -layout`.
OCLUSION = """
              INDICES                PRUEBA                DATOS NORMALES
               POC TA              88.09%   I                  83≤(%)≤100
                TORS               89.34%   I                  90≤(%)≤100
                ASIM                  4.58%                     -10≤(%)≤10
              POC ECM              81.20% D                    83≤(%)≤100
"""


def test_captura_el_valor_con_el_rango_que_el_informe_declara():
    """El rango lo pone el documento, no nosotros. Es lo que permite señalar lo anómalo
    sin saber qué significa la sigla."""
    m = {x["nombre"]: x for x in extract_medidas_by_rules(OCLUSION)}

    assert set(m) == {"POC TA", "TORS", "ASIM", "POC ECM"}
    assert m["TORS"]["valor"] == 89.34
    assert (m["TORS"]["normal_min"], m["TORS"]["normal_max"]) == (90.0, 100.0)
    assert m["POC TA"]["unidad"] == "%"


def test_el_rango_negativo_tambien():
    """`ASIM` va de -10 a 10. Un extractor que asuma rangos positivos lo pierde."""
    asim = next(x for x in extract_medidas_by_rules(OCLUSION) if x["nombre"] == "ASIM")
    assert (asim["normal_min"], asim["normal_max"]) == (-10.0, 10.0)


def test_la_lateralidad_se_copia_sin_traducir():
    """`I`/`D`/`A` es el código del fabricante. Mapearlo a izquierda/derecha sería
    suponer: cada equipo usa el suyo y aquí no hay tabla que lo diga."""
    m = {x["nombre"]: x for x in extract_medidas_by_rules(OCLUSION)}
    assert m["POC ECM"]["lado"] == "D"
    assert m["ASIM"]["lado"] is None


def test_se_guarda_la_linea_literal():
    """Una captura que nadie ha interpretado tiene que ser auditable contra el original."""
    tors = next(x for x in extract_medidas_by_rules(OCLUSION) if x["nombre"] == "TORS")
    assert "89.34" in tors["texto"] and "90" in tors["texto"]


def test_un_numero_sin_rango_no_se_recoge():
    """No se puede ni validar ni señalar, y meterlo añadiría ruido con aspecto de dato."""
    assert extract_medidas_by_rules("POC TA   88.09%   I\nPagina 2/3\n") == []


def test_un_informe_por_diente_no_produce_medidas():
    """Los dos extractores no compiten: el de hallazgos por FDI sigue siendo el suyo."""
    assert extract_medidas_by_rules("Diente 16: caries oclusal\nDiente 17: 3 raices\n") == []
