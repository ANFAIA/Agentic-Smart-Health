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


# --- guardianes: lo que corre sin que nadie lo pida -------------------------- #
def test_los_guardianes_del_repositorio_estan_registrados():
    """El guardián aplicado a sí mismo, otra vez."""
    texto = (Path(__file__).resolve().parents[1] / "AGENTS.md").read_text(encoding="utf-8")
    assert ds.revisar_guardianes(ds.versionados(), texto) == []


def test_un_guardian_sin_ficha_se_declara():
    """El hueco real: tres scripts bloqueando commits y abriendo PRs sin registrar."""
    problemas = ds.revisar_guardianes(ds.versionados(), "AGENTS.md sin una sola ruta")
    citados = " ".join(problemas)
    for script in ("data_guard.py", "docs_sync.py", "watch_literature.py"):
        assert script in citados


def test_compartir_ficha_con_otro_agente_vale():
    """`audit_pr.py` vive dentro de la ficha del `ai-code-reviewer`, y está bien así.

    Exigir sección propia sería imponer un formato; lo que se comprueba es un hecho:
    que el documento lo mencione.
    """
    solo_uno = "menciona `scripts/audit_pr.py` y nada más"
    problemas = ds.revisar_guardianes(ds.versionados(), solo_uno)
    assert not any("audit_pr" in p for p in problemas)


def test_un_script_que_solo_se_lanza_a_mano_no_es_un_agente():
    """La frontera: `resolucion_modalidades.py` lo teclea una persona, no un disparador."""
    autonomos = ds.guardianes_autonomos(ds.versionados())
    assert "scripts/resolucion_modalidades.py" not in autonomos
    assert "scripts/watch_literature.py" in autonomos


# --- qué comprueba un guardián: el sustituto de versionarlos ----------------- #
FICHA_CON_TABLA = """
### `docs-guardian` — Guardián

| Campo | Valor |
|---|---|
| **Ubicación** | [`scripts/docs_sync.py`](scripts/docs_sync.py) |

| Comprobación | Qué caza |
|---|---|
| `env` | variables leídas por el código |
| `inventario` y `arbol` | las dos direcciones de la cita |

**Reglas de delegación**

- `algo` que no es una comprobación y va fuera de la tabla.
"""


def test_lee_los_identificadores_del_registro_del_codigo():
    """Contra el script de verdad: los identificadores salen de `COMPROBACIONES`."""
    registradas = ds.comprobaciones_del_codigo("scripts/docs_sync.py")
    assert "comprobaciones" in registradas
    assert len(registradas) == len(set(registradas))  # sin duplicados


def test_una_fila_puede_documentar_dos_comprobaciones():
    """`inventario` y `arbol` comparten fila, y contar por separado no aportaría nada."""
    assert ds.comprobaciones_de_la_ficha(FICHA_CON_TABLA) == {"env", "inventario", "arbol"}


def test_solo_se_leen_los_identificadores_de_la_tabla():
    """Lo de fuera es prosa: un `algo` suelto no es una comprobación declarada."""
    assert "algo" not in ds.comprobaciones_de_la_ficha(FICHA_CON_TABLA)


def test_sin_tabla_de_comprobaciones_no_hay_identificadores():
    assert ds.comprobaciones_de_la_ficha(FICHA_SIMPLE) == set()


def test_un_script_sin_registro_queda_fuera():
    """Declarar `COMPROBACIONES` es voluntario: `audit_pr.py` delega el criterio en un
    modelo y no tiene una lista de comprobaciones que enumerar."""
    assert ds.comprobaciones_del_codigo("scripts/audit_pr.py") == []
    assert ds.revisar_comprobaciones({"scripts/audit_pr.py"}, "sin ficha ninguna") == []


def test_el_repositorio_documenta_hoy_lo_que_comprueba():
    """El guardián aplicado a sí mismo, una vez más."""
    texto = (Path(__file__).resolve().parents[1] / "AGENTS.md").read_text(encoding="utf-8")
    assert ds.revisar_comprobaciones(ds.versionados(), texto) == []


def test_una_comprobacion_sin_documentar_se_declara():
    """El caso que justifica todo esto: añadir una comprobación y olvidar la ficha."""
    ficha = FICHA_CON_TABLA.replace("| `env` | variables leídas por el código |\n", "")
    problemas = ds.revisar_comprobaciones({"scripts/docs_sync.py", ".githooks/pre-commit"}, ficha)
    assert any("`env`" in p and "no la documenta" in p for p in problemas)


def test_una_comprobacion_anunciada_y_no_hecha_se_declara():
    """La deriva contraria: la ficha promete una vigilancia que ya no existe."""
    ficha = FICHA_CON_TABLA.replace("| `env` |", "| `fantasma` |")
    problemas = ds.revisar_comprobaciones({"scripts/docs_sync.py", ".githooks/pre-commit"}, ficha)
    assert any("`fantasma`" in p and "ya no hace" in p for p in problemas)


def test_una_ficha_sin_tabla_no_pasa_por_defecto():
    """Borrar la tabla no puede ser la forma de aprobar: el registro existe, luego se enumera."""
    ficha = "### `docs-guardian`\n\n| Campo | Valor |\n|---|---|\n| **Ubicación** | `scripts/docs_sync.py` |\n"  # noqa: E501
    problemas = ds.revisar_comprobaciones({"scripts/docs_sync.py", ".githooks/pre-commit"}, ficha)
    assert len(problemas) == 1
    assert "no las enumera" in problemas[0]
