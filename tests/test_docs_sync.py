"""El guardián de documentación: que la versión de la ficha sea la del código.

El caso que lo originó: el 2026-08-10 el `geometric-fusion-agent` pasó a 0.2.0 —cambió
de dónde sale su confianza— y nada habría avisado si la ficha se hubiera quedado
describiendo la 0.1.0. Es la misma deriva que el recuento de tests que decía 166
cuando eran 265.

`scripts/` no es un paquete instalado, así que se importa por ruta.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "docs_sync.py"
_spec = importlib.util.spec_from_file_location("docs_sync", _RUTA)
assert _spec and _spec.loader
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


FICHA_SIMPLE = """
### `mesh-agent` — Agente de ingesta

| Campo | Valor |
|---|---|
| **Versión** | `0.1.0` |
| **Estado** | `active` |
"""

# Una ficha puede documentar varios agentes con versiones distintas.
FICHA_COMPARTIDA = """
### Agentes de fusión — `geometric-fusion-agent` · `semantic-fusion-agent`

| Campo | Valor |
|---|---|
| **Versión** | `geometric-fusion-agent` **0.2.0** · `semantic-fusion-agent` `0.1.0` |
| **Estado** | `active` |
"""


def _agentes(**versiones: str) -> dict[str, tuple[str, str]]:
    return {n: (f"packages/x/src/{n}.py", v) for n, v in versiones.items()}


# --- el caso normal --------------------------------------------------------- #
def test_la_version_que_cuadra_no_da_problema():
    assert ds.revisar_versiones(_agentes(**{"mesh-agent": "0.1.0"}), FICHA_SIMPLE) == []


def test_una_version_desincronizada_se_declara():
    problemas = ds.revisar_versiones(_agentes(**{"mesh-agent": "0.2.0"}), FICHA_SIMPLE)
    assert len(problemas) == 1
    assert "0.2.0" in problemas[0] and "0.1.0" in problemas[0]


# --- ficha compartida: el punto delicado ------------------------------------ #
def test_en_una_ficha_compartida_cada_agente_lleva_la_suya():
    agentes = _agentes(
        **{"geometric-fusion-agent": "0.2.0", "semantic-fusion-agent": "0.1.0"}
    )
    assert ds.revisar_versiones(agentes, FICHA_COMPARTIDA) == []


def test_la_version_del_vecino_no_tapa_el_hueco():
    """Sin recortar la celda, un `0.2.0` del vecino daría por buena la del otro."""
    agentes = _agentes(**{"semantic-fusion-agent": "0.2.0"})  # la ficha dice 0.1.0
    problemas = ds.revisar_versiones(agentes, FICHA_COMPARTIDA)
    assert len(problemas) == 1
    assert "semantic-fusion-agent" in problemas[0]


# --- lo que NO debe hacer saltar la alarma ---------------------------------- #
@pytest.mark.parametrize(
    ("agentes", "texto", "motivo"),
    [
        (_agentes(**{"fantasma-agent": "1.0.0"}), FICHA_SIMPLE, "sin ficha"),
        (_agentes(**{"mesh-agent": ""}), FICHA_SIMPLE, "sin version en el codigo"),
        (_agentes(**{"mesh-agent": "0.1.0"}), "### `mesh-agent`\n\ntexto suelto", "sin fila"),
    ],
    ids=["sin-ficha", "sin-version", "ficha-sin-fila-de-version"],
)
def test_no_dispara_donde_no_hay_nada_que_comparar(agentes, texto, motivo):
    """Un guardián que grita por casos que no puede juzgar acaba desinstalado."""
    assert ds.revisar_versiones(agentes, texto) == [], motivo


# --- lectura del código ------------------------------------------------------ #
def test_lee_name_y_version_del_codigo_real():
    """Contra el repositorio de verdad: la clase declara las dos cosas."""
    agentes = ds.agentes_implementados(ds.versionados())
    ruta, version = agentes["geometric-fusion-agent"]
    assert ruta.endswith("fusion_agents/geometric.py")
    assert version.count(".") == 2  # SemVer, no una cadena cualquiera


def test_el_repositorio_esta_sincronizado_ahora_mismo():
    """El guardián aplicado a sí mismo: hoy no hay deriva de versiones."""
    texto = (Path(__file__).resolve().parents[1] / "AGENTS.md").read_text(encoding="utf-8")
    assert ds.revisar_versiones(ds.agentes_implementados(ds.versionados()), texto) == []
