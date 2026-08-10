"""El vigilante de literatura: reintentos y la diferencia entre «nada» y «nadie contestó».

El 2026-08-10 las siete consultas se llevaron un 429 de arXiv y el run terminó en
**verde** diciendo «Nada nuevo que proponer» — indistinguible de una semana
tranquila de verdad. Estos tests fijan las dos conductas que lo arreglan: reintentar
lo que se cura esperando, y declarar la ejecución vacía cuando nadie respondió.

`scripts/` no es un paquete instalado, así que se importa por ruta.
"""

from __future__ import annotations

import importlib.util
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "watch_literature.py"
_spec = importlib.util.spec_from_file_location("watch_literature", _RUTA)
assert _spec and _spec.loader
wl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wl)


def _http(codigo: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://arxiv.org", codigo, "boom", {}, None)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def sin_esperas(monkeypatch):
    """Los tests no pagan la cortesía de 3 s: se mide la lógica, no el reloj."""
    monkeypatch.setattr(wl.time, "sleep", lambda _s: None)


# --- qué merece un reintento ------------------------------------------------ #
@pytest.mark.parametrize(
    ("error", "esperado"),
    [
        (_http(429), True),   # límite de tasa: el caso real del 2026-08-10
        (_http(503), True),   # arXiv caído
        (_http(400), False),  # consulta mal formada: no se cura nunca
        (_http(404), False),
        (urllib.error.URLError("sin red"), True),
        (TimeoutError("agotado"), True),
        (ET.ParseError("xml truncado"), True),
    ],
    ids=["429", "503", "400", "404", "url-error", "timeout", "parse-error"],
)
def test_solo_se_reintenta_lo_que_se_cura_esperando(error, esperado):
    assert wl._transitorio(error) is esperado


# --- el bucle de reintentos ------------------------------------------------- #
def test_rinde_a_la_tercera_tras_dos_429(monkeypatch):
    llamadas = []

    def buscar(consulta, limite):
        llamadas.append(consulta)
        if len(llamadas) < 3:
            raise _http(429)
        return [{"id": "2608.00001"}]

    monkeypatch.setattr(wl, "buscar", buscar)
    assert wl.buscar_con_reintentos("all:test", 25) == [{"id": "2608.00001"}]
    assert len(llamadas) == 3


def test_una_consulta_mal_formada_no_se_reintenta(monkeypatch):
    """Reintentar un 400 tres veces solo retrasa el diagnóstico."""
    llamadas = []

    def buscar(consulta, limite):
        llamadas.append(consulta)
        raise _http(400)

    monkeypatch.setattr(wl, "buscar", buscar)
    with pytest.raises(urllib.error.HTTPError):
        wl.buscar_con_reintentos("all:mal(formada", 25)
    assert len(llamadas) == 1


def test_tras_agotar_los_intentos_propaga(monkeypatch):
    llamadas = []

    def buscar(consulta, limite):
        llamadas.append(consulta)
        raise _http(429)

    monkeypatch.setattr(wl, "buscar", buscar)
    with pytest.raises(urllib.error.HTTPError):
        wl.buscar_con_reintentos("all:test", 25, intentos=3)
    assert len(llamadas) == 3


def test_la_espera_crece_entre_intentos(monkeypatch):
    esperas = []
    monkeypatch.setattr(wl.time, "sleep", esperas.append)
    monkeypatch.setattr(wl, "buscar", lambda c, _lim: (_ for _ in ()).throw(_http(429)))

    with pytest.raises(urllib.error.HTTPError):
        wl.buscar_con_reintentos("all:test", 25, intentos=3)
    assert esperas == [wl._ESPERA, wl._ESPERA * 3]  # retrocede, no insiste al mismo ritmo


# --- «nada nuevo» ≠ «nadie contestó» ---------------------------------------- #
def test_si_fallan_todas_la_ejecucion_se_declara_vacia(monkeypatch, capsys, tmp_path):
    """El fallo del 2026-08-10: siete 429 y un run verde diciendo «sin novedades»."""
    monkeypatch.setattr(wl, "buscar", lambda c, _lim: (_ for _ in ()).throw(_http(429)))
    resumen = tmp_path / "cuerpo-pr.md"
    monkeypatch.setattr("sys.argv", ["watch_literature.py", "--resumen", str(resumen)])

    codigo = wl.main()

    assert codigo == 2  # distinto de 0: el CI tiene que enterarse
    assert "NO se ha hecho" in capsys.readouterr().err
    assert resumen.read_text(encoding="utf-8") == ""


def test_una_semana_tranquila_de_verdad_sigue_siendo_verde(monkeypatch, capsys, tmp_path):
    """La otra mitad del contrato: sin novedades no es un fallo."""
    monkeypatch.setattr(wl, "buscar", lambda c, _lim: [])
    resumen = tmp_path / "cuerpo-pr.md"
    monkeypatch.setattr("sys.argv", ["watch_literature.py", "--resumen", str(resumen)])

    codigo = wl.main()

    assert codigo == 0
    assert "Nada nuevo que proponer" in capsys.readouterr().out


def test_un_fallo_parcial_se_declara_en_el_cuerpo_de_la_pr(monkeypatch, tmp_path):
    """Si la mitad de las consultas calla, la cosecha va sesgada y hay que decirlo."""
    articulo = {
        "id": "2608.00042v1",
        "base": "2608.00042",
        "titulo": "Gaussian splatting for dental CBCT reconstruction",
        "resumen": "We reconstruct tooth anatomy from cone-beam CT.",
        "publicado": wl.datetime.now(wl.UTC).isoformat().replace("+00:00", "Z"),
        "url": "https://arxiv.org/pdf/2608.00042v1",
    }

    def buscar(consulta, limite):
        if consulta == wl.CONSULTAS[0]["consulta"]:
            return [dict(articulo)]
        raise _http(429)

    monkeypatch.setattr(wl, "buscar", buscar)
    monkeypatch.setattr(wl, "licencia_en_origen", lambda base: ("CC BY 4.0", True))
    monkeypatch.setattr(wl, "huella", lambda url: ("a" * 64, 1024))
    resumen = tmp_path / "cuerpo-pr.md"
    monkeypatch.setattr("sys.argv", ["watch_literature.py", "--resumen", str(resumen)])

    assert wl.main() == 0  # con una consulta viva NO es una ejecución fallida

    cuerpo = resumen.read_text(encoding="utf-8")
    assert f"{len(wl.CONSULTAS) - 1} de {len(wl.CONSULTAS)} consultas no respondieron" in cuerpo
    assert "2608.00042v1" in cuerpo
