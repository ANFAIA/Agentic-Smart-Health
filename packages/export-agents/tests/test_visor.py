"""El paquete del visor: perfil INRIA derivado + sidecar con la capa clínica.

Lo que estos tests atan no es que los ficheros se escriban, sino las cuatro decisiones que
los hacen honestos: que los dientes no se decimen, que el `region_id` viaje en el mismo
orden que el PLY, que las conversiones de vista se declaren, y que el nombre del paciente
no viaje dentro.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from core_schemas import (
    ClinicalAttributes,
    Hallazgo,
    Modality,
    Provenance,
    RegionalObservation,
    RigidTransform,
    TwinSnapshot,
)
from export_agents import ViewerExportAgent
from export_agents.visor import ALFA_SUPERFICIE, ALFA_VOLUMEN, C0

CAMPO = "sha256:" + "a" * 64
MALLA = "sha256:" + "b" * 64


class _Almacen:
    def __init__(self, **refs: dict) -> None:
        self._refs = refs

    def load(self, ref: str) -> dict:
        return self._refs[ref]

    def put(self, **arrays):  # pragma: no cover
        return "sha256:" + "0" * 64


def _campo(n: int = 4000, etiquetados: int = 250) -> dict:
    rng = np.random.default_rng(0)
    return {
        "centers": rng.normal(0, 20, (n, 3)),
        "scales": np.full((n, 3), 0.15),
        "rotations": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "density": rng.uniform(0.1, 1.0, n),
        "origin": np.array([5.0, 5.0, 5.0]),
        "region_id": np.where(np.arange(n) < etiquetados, 36, 0).astype(np.int16),
    }


def _snapshot(con_malla=True, con_transform=True, fuente="APELLIDO_NOMBRE_Informe.pdf"):
    ident = RigidTransform(rotation=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0))
    return TwinSnapshot(
        acquisition_id="acq-1", timestamp="2026-08-21T00:00:00Z",
        gaussian_field_ref=CAMPO, surface_ref=MALLA if con_malla else None,
        regional=[RegionalObservation(
            region_id="36",
            attributes=ClinicalAttributes(ph=5.1, n_raices=2, hallazgos=[Hallazgo.CARIES]),
            timestamp="2026-08-21T00:00:00Z",
            provenance=Provenance(
                source_file="/casos/" + fuente, modality=Modality.REPORT,
                agent="report-agent@0.1.0", confidence=0.9,
            ),
        )],
        provenance=Provenance(
            source_file="/casos/" + fuente, modality=Modality.CBCT, agent="t",
            confidence=1.0, transform=ident if con_transform else None,
        ),
    )


@pytest.fixture
def almacen():
    rng = np.random.default_rng(1)
    return _Almacen(**{CAMPO: _campo(), MALLA: {"positions": rng.normal(0, 20, (800, 3))}})


def _cabecera(ruta) -> list[str]:
    raw = ruta.read_bytes()
    return raw[: raw.index(b"end_header")].decode("ascii").splitlines()


def _lado(ruta) -> dict:
    """El sidecar del paquete. `ruta` es cualquiera de las capas."""
    base = str(ruta).rsplit("-", 1)[0]
    return json.loads((type(ruta)(base + ".json")).read_text())


def _rgb(ruta) -> np.ndarray:
    raw = ruta.read_bytes()
    d = np.frombuffer(raw[raw.index(b"end_header") + 11 :], dtype=np.float32).reshape(-1, 17)
    return d[:, 6:9] * C0 + 0.5


def test_las_coronas_salen_del_ESCANER_no_del_cbct(almacen, tmp_path):
    """La decisión de fondo del canal.

    El escáner trae los dientes separados y con FDI, completos y con exactitud de decenas
    de micras. El campo del CBCT cubre el 51 % del volumen de cada pieza y de forma
    desigual. Enseñar el compuesto del CBCT como «el diente» presentaría como completo
    algo que no lo es; la raíz —lo único que el escáner no ve— va en su propia capa y
    declarada como parcial.
    """
    etq = np.where(np.arange(800) < 500, 36, 0).astype(np.int16)
    salida = ViewerExportAgent(almacen).export(
        _snapshot(), tmp_path / "v", etiquetas_ios=etq
    )
    capas = {c["id"]: c["primitivas"] for c in _lado(salida.path)["capas"]}

    assert capas["coronas"] == 500, "del escáner, enteras"
    assert capas["raices"] == 250, "del CBCT, parciales"
    assert capas["encia"] == 300


def test_el_ply_es_el_perfil_QUE_EL_VISOR_SABE_LEER(almacen, tmp_path):
    """17 propiedades, grado 0. Nuestro PLY del twin trae `density` y escalas lineales, y
    ningún rasterizador de splats puede abrirlo: por eso este canal existe."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    props = [x.split()[-1] for x in _cabecera(salida.path) if x.startswith("property")]

    assert props == [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]


def test_la_cabecera_declara_que_NO_es_el_twin(almacen, tmp_path):
    """La línea que no se puede borrar: opacidad y color son interpretación."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    cab = "\n".join(_cabecera(salida.path))

    assert "NO es el twin" in cab
    assert "GANANCIA DE VISUALIZACION, no es dato" in cab
    assert "NO mide color" in cab
    assert "log(sigma_mm); en el PLY del twin son mm lineales" in cab


def test_la_cabecera_es_ascii(almacen, tmp_path):
    """Un PLY con un guion largo en la cabecera revienta al escribirse, y el error no
    menciona PLY por ningún lado. Pasó una vez."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    assert salida.path.read_bytes()[: 400].decode("ascii")


def test_la_escala_va_en_LOGARITMO_como_espera_el_rasterizador(almacen, tmp_path):
    """El `scale_*` de INRIA es `log(sigma)`; el nuestro son mm. Escribir mm aquí no
    fallaría: el visor los exponenciaría y pintaría gaussianas gigantes."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    raw = salida.path.with_name("v-raices.ply").read_bytes()
    datos = np.frombuffer(raw[raw.index(b"end_header") + 11 :], dtype=np.float32)
    factor = _lado(salida.path)["display"]["factor_escala"]

    assert datos.reshape(-1, 17)[0, 10] == pytest.approx(np.log(0.15 * factor), abs=1e-4)


def test_la_sigma_se_infla_para_que_los_splats_se_TOQUEN(almacen, tmp_path):
    """Sin esto el visor enseña polvo, no anatomía.

    Medido sobre el caso real: sigma 0,075 mm sobre un espaciado de 0,276 tras decimar —
    ratio 0,27, los splats ni se rozan y la superficie sale agujereada. Es una decisión de
    la VISTA: la sigma medida sigue intacta en el PLY del twin, y el factor va declarado
    en la cabecera y en el sidecar.
    """
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    lado = _lado(salida.path)["display"]
    cab = "\n".join(_cabecera(salida.path))

    assert lado["factor_escala"] > 1.0
    assert "ES DE LA VISTA" in cab
    assert "espaciado" in lado["por_que_factor"]


def test_un_campo_ya_denso_no_se_infla(tmp_path):
    """El factor nunca reduce: si los splats ya son mas grandes de lo que la vista pide, se
    dejan como estan. Un aviso que se aplica siempre deja de ser una decision.

    ⚠️ El liston es `MULTIPLO_ESPACIADO` y subio de 0,6 a 4 al medirlo contra 3DGS
    entrenado de verdad, asi que el campo de esta prueba tuvo que engordar con el: con
    sigma 0,5 sobre un espaciado de ~0,13 ya NO esta denso para el criterio nuevo."""
    rng = np.random.default_rng(3)
    campo = _campo(n=2000, etiquetados=100)
    campo["centers"] = rng.normal(0, 1.0, (2000, 3))   # muy juntos
    campo["scales"] = np.full((2000, 3), 2.0)          # y mas gordos que 4x el espaciado
    almacen = _Almacen(**{CAMPO: campo, MALLA: {"positions": rng.normal(0, 1, (100, 3))}})

    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    assert _lado(salida.path)["display"]["factor_escala"] == 1.0


def test_la_opacidad_sale_de_la_transferencia_declarada(almacen, tmp_path):
    """`alfa = 1 - exp(-g·sigma)`, y el PLY guarda su logit. Está escrito en los dos
    ficheros para que nadie lo tome por una medida."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    assert _lado(salida.path)["display"]["opacity"] == (
        f"logit(1 - exp(-g * sigma)), g por capa: alfa~{ALFA_SUPERFICIE:g} "
        f"en superficie, ~{ALFA_VOLUMEN:g} en volumen"
    )


def test_los_dientes_NO_se_deciman_nunca(almacen, tmp_path):
    """El 97 % del campo es hueso y cráneo: una muestra uniforme dejaría casi vacío lo
    único que un clínico mira."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    r = _lado(salida.path)["recorte"]

    assert r["dientes"] == r["dientes_total"] == 250
    assert "dientes 250/250 (enteros)" in salida.detail


def test_el_falso_color_distingue_diente_de_lo_demas(almacen, tmp_path):
    """No es decoración: sin color por FDI el visor enseña una masa gris uniforme."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    diente = _rgb(salida.path.with_name("v-raices.ply"))
    encia = _rgb(salida.path.with_name("v-encia.ply"))

    assert not np.allclose(diente[0], encia[0]), "el diente y la encía no pueden ser iguales"
    assert (diente >= -0.01).all() and (diente <= 1.01).all()


# --- las tres capas --------------------------------------------------------- #
def test_el_compuesto_sale_en_capas_conmutables(almacen, tmp_path):
    """Un fichero por capa, que es como el visor las enciende y apaga.

    ⚠️ Y **no son ventanas de HU** como las de ToothFairy: aquí la separación es por
    procedencia y anatomía, que es justo la distinción que la densidad NO puede dar — el
    hueso alveolar y la raíz comparten HU, y está medido que ningún umbral los separa.
    """
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    capas = _lado(salida.path)["capas"]

    assert [c["id"] for c in capas] == ["coronas", "raices", "encia", "resto"]
    for c in capas:
        assert (tmp_path / c["ply"]).exists(), f"falta el fichero de la capa {c['id']}"
    # Sin etiquetas del escáner no hay coronas: todo el IOS cae en `encia`.
    assert capas[0]["primitivas"] == 0
    assert capas[1]["primitivas"] == 250, "las raíces del CBCT van enteras"
    assert capas[2]["primitivas"] == 800


def test_las_capas_son_DISJUNTAS_y_suman_el_compuesto(almacen, tmp_path):
    """Encender varias suma sin contar nada dos veces, igual que las de densidad."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    capas = _lado(salida.path)["capas"]

    assert sum(c["primitivas"] for c in capas) == salida.n_vertices


def test_cada_capa_dice_cual_es_en_su_cabecera(almacen, tmp_path):
    """Un PLY suelto tiene que poder decir de qué capa es sin el sidecar al lado."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    assert any("capa: Coronas" in x for x in _cabecera(salida.path))
    assert any("capa: Raices" in x for x in _cabecera(salida.path.with_name("v-raices.ply")))
    assert any("capa: Enc" in x for x in _cabecera(salida.path.with_name("v-encia.ply")))


def test_el_region_id_solo_viaja_donde_significa_algo(almacen, tmp_path):
    """Con tres capas el índice por gaussiana del compuesto entero no indexa nada: se
    acota a la capa `dientes`, que es la única donde un FDI quiere decir algo."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    lado = _lado(salida.path)
    region = np.frombuffer(base64.b64decode(lado["region_id"]["b64"]), dtype=np.int16)

    assert lado["region_id"]["n"] == len(region) == 250
    assert (region > 0).all(), "en esa capa no hay ni una gaussiana sin nombre"


def test_el_nombre_del_paciente_no_viaja_dentro(almacen, tmp_path):
    """⚠️ Estos ficheros son los que se archivan y se comparten. El nombre del PDF de un
    proveedor clínico trae nombre y apellidos."""
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")
    lado = json.dumps(_lado(salida.path), ensure_ascii=False)

    assert "APELLIDO" not in lado and "NOMBRE" not in lado
    assert _lado(salida.path)["dientes"]["36"]["fuente"].startswith("informe:")


def test_el_sidecar_lleva_la_capa_clinica_y_el_gate(almacen, tmp_path):
    """Lo que ningún visor de estantería enseña, y por lo que este canal existe."""
    salida = ViewerExportAgent(almacen).export(
        _snapshot(), tmp_path / "v", motivos=["el registro no convergió"]
    )
    lado = _lado(salida.path)

    assert lado["dientes"]["36"]["ph"] == 5.1
    assert "caries" in lado["dientes"]["36"]["hallazgos"]
    assert lado["gate"] == ["el registro no convergió"]
    assert "NO es reversible" in lado["reversibilidad"]["aviso"]


def test_sin_registro_va_solo_el_campo_y_lo_dice(almacen, tmp_path):
    """Sin la rígida la encía no se puede poner en el mismo sistema. Se declara."""
    salida = ViewerExportAgent(almacen).export(
        _snapshot(con_transform=False), tmp_path / "v"
    )
    assert salida.ok
    assert _lado(salida.path)["recorte"]["encia_total"] == 0
    assert any("sin registro" in m for m in salida.hitl_reasons)


def test_un_campo_sin_segmentar_avisa_de_que_no_habra_que_seleccionar(tmp_path):
    """El producto es por diente: sin `region_id` el visor no tiene nada que seleccionar,
    y eso va al gate en vez de entregarse como si estuviera completo."""
    rng = np.random.default_rng(2)
    almacen = _Almacen(**{
        CAMPO: _campo(etiquetados=0), MALLA: {"positions": rng.normal(0, 20, (800, 3))}
    })
    salida = ViewerExportAgent(almacen).export(_snapshot(), tmp_path / "v")

    assert salida.ok
    assert any("nada que seleccionar" in m for m in salida.hitl_reasons)


def test_los_motivos_del_gate_no_llevan_rutas_del_sistema(almacen, tmp_path):
    """⚠️ Se vio impreso en el panel del visor.

    Los motivos llegan **verbatim** de los agentes y algunos citan el fichero que falló:
    `report-agent falló: ... /home/.../histora/another_patient/BRN3C...pdf`. Redactar solo
    la `fuente` de los hallazgos no bastaba — el texto libre es por donde se escapa lo que
    nadie revisó. El motivo se conserva entero; lo que se sustituye es la ruta.
    """
    salida = ViewerExportAgent(almacen).export(
        _snapshot(), tmp_path / "v",
        motivos=[
            "report-agent falló: no hay texto en /casos/APELLIDO_NOMBRE/informe.pdf",
            "el registro agotó las iteraciones (180) sin converger",
        ],
    )
    gate = _lado(salida.path)["gate"]

    assert "APELLIDO_NOMBRE" not in gate[0] and "/casos/" not in gate[0]
    assert gate[0].startswith("report-agent falló:")
    assert "informe:" in gate[0], "sigue identificando CUÁL fichero, sin nombrarlo"
    # Un motivo sin rutas se deja intacto: no es un filtro que reescriba por si acaso.
    assert gate[1] == "el registro agotó las iteraciones (180) sin converger"


def test_la_altura_no_depende_de_como_estuviera_orientado_el_paciente():
    """El fallo que tenia: `ptp` sobre los ejes del mundo mide la caja, no la pieza. Una
    misma pieza girada daba un numero distinto, y el panel lo pinta en rojo pasado 26 mm."""
    from export_agents.visor import _largo_propio

    rng = np.random.default_rng(0)
    # Una pieza de 24 mm de largo y 9 de ancho, alineada con Z.
    pieza = np.c_[rng.uniform(-4.5, 4.5, 500), rng.uniform(-4.5, 4.5, 500),
                  rng.uniform(-12, 12, 500)]
    t = np.radians(35.0)
    giro = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    recta = _largo_propio(pieza)
    girada = _largo_propio(pieza @ giro.T)

    assert 23.0 < recta < 25.0
    assert abs(recta - girada) < 0.5


def test_cada_pieza_tiene_su_propio_TONO_no_solo_su_claridad():
    """El fallo que tenia: el tono salia del cuadrante y solo la claridad cambiaba entre
    piezas, asi que una arcada superior entera eran dos colores con siete escalones cada
    uno. En pantalla eso se lee como un solo color, y un color que no distingue piezas no
    sirve para nada — distinguirlas es lo unico que hace."""
    from export_agents.visor import tono_de

    superior = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28]
    tonos = [tono_de(c) for c in superior]

    assert len(set(tonos)) == len(superior), "hay piezas compartiendo tono"
    # Y las CONTIGUAS lejos en el circulo, que es justo cuando hace falta distinguirlas.
    for a, b in zip(superior, superior[1:], strict=False):
        d = abs(tono_de(a) - tono_de(b))
        assert min(d, 1 - d) > 0.15, f"{a} y {b} a {min(d, 1 - d):.2f} de tono"


def test_el_color_de_una_pieza_depende_SOLO_de_su_codigo():
    """Comparar dos visitas lado a lado con los colores bailando seria peor que no
    tenerlos: la misma pieza tiene que salir del mismo color en cualquier caso."""
    from export_agents.visor import _falso_color, tono_de

    fdi_a = np.array([16, 16, 26, 0])
    fdi_b = np.array([16, 11, 12, 13, 26])      # otro caso, otras piezas detectadas
    origen = np.zeros(5, dtype=np.int16)

    ca = _falso_color(fdi_a, origen[:4])
    cb = _falso_color(fdi_b, origen)

    assert np.allclose(ca[0], cb[0]), "el 16 cambia de color segun quien mas haya"
    assert np.allclose(ca[2], cb[4]), "el 26 cambia de color segun quien mas haya"
    assert tono_de(16) == tono_de(16)


def test_los_vertices_del_escaner_salen_como_DISCOS_TANGENTES():
    """La diferencia entre ver una superficie y ver polvo.

    Una esfera de radio 0,5 del espaciado deja hueco entre vecinas y ademas reparte su
    opacidad en profundidad, donde no hay nada que ensenar. Es la forma a la que llega solo
    un 3DGS entrenado contra renders opacos; aqui la normal ya la midio el escaner.
    """
    from export_agents.visor import _discos

    rng = np.random.default_rng(0)
    n = rng.normal(0, 1, (500, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    escalas, quats = _discos(n, radio=0.4, grosor=0.08)

    # Fino en un eje y ancho en los otros dos: eso es un disco y no una esfera.
    assert np.allclose(escalas[:, :2], 0.4)
    assert np.allclose(escalas[:, 2], 0.08)
    # Plano, pero no tanto como para verse de canto: ver ASPECTO_DISCO.
    assert 2.0 < 0.4 / 0.08 < 6.0
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0)


def test_el_eje_FINO_del_disco_apunta_a_lo_largo_de_la_normal():
    """Si el eje fino no fuera la normal, el disco quedaria de canto a la superficie: se
    veria peor que una esfera, no mejor."""
    from export_agents.visor import _discos

    rng = np.random.default_rng(1)
    n = rng.normal(0, 1, (300, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    _, q = _discos(n, radio=0.4, grosor=0.08)

    w, x, y, z = q.T
    # Tercera columna de la matriz de rotacion: a donde va el eje local +z, que es el fino.
    eje_fino = np.column_stack([2 * (x * z + w * y), 2 * (y * z - w * x),
                                1 - 2 * (x * x + y * y)])
    assert np.allclose(np.abs((eje_fino * n).sum(axis=1)), 1.0, atol=1e-6)


def test_una_normal_opuesta_no_rompe_el_cuaternion():
    """El giro de arco minimo de +z a -z es ambiguo y su cuaternion se anula. Sin tratarlo,
    esos vertices saldrian con un cuaternion no normalizado, que no es ninguna rotacion."""
    from export_agents.visor import _discos

    n = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    _, q = _discos(n, radio=0.3, grosor=0.06)

    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    assert np.isfinite(q).all()


def test_la_opacidad_no_es_la_misma_en_superficie_que_en_volumen():
    """Un solo objetivo para las cuatro capas fue un error medido: un rayo atraviesa una
    hoja de splats en un caso y un solido en el otro, asi que con la misma alfa por splat
    el volumen sale opaco y la superficie transparente."""
    from export_agents.compuesto import ORIGEN_IOS
    from export_agents.visor import ALFA_SUPERFICIE, ALFA_VOLUMEN, _ganancia

    d = np.full(200, 0.5)
    g_sup = _ganancia(d, np.full(200, ORIGEN_IOS, dtype=np.int16))
    g_vol = _ganancia(d, np.zeros(200, dtype=np.int16))

    assert g_sup > g_vol, "la superficie necesita MAS opacidad por splat, no menos"
    assert 1 - np.exp(-g_sup * 0.5) == pytest.approx(ALFA_SUPERFICIE, rel=1e-6)
    assert 1 - np.exp(-g_vol * 0.5) == pytest.approx(ALFA_VOLUMEN, rel=1e-6)


def test_una_capa_sin_densidad_no_dispara_la_ganancia():
    """Sin la cota, una capa con densidad ~0 pediria ganancia infinita para llegar al
    objetivo y saldria opaca justo donde no hay nada que ensenar."""
    from export_agents.visor import GANANCIA_MAXIMA, _ganancia

    assert _ganancia(np.zeros(10), np.zeros(10, dtype=np.int16)) == GANANCIA_MAXIMA
    assert _ganancia(np.array([]), np.array([], dtype=np.int16)) == GANANCIA_MAXIMA


def test_el_paquete_no_pasa_de_CINCO_capas():
    """Limite medido del visor, no una preferencia.

    `dental-3dgs-viewer` deja de dibujar a partir de SEIS escenas: con cinco renderiza y
    con seis el lienzo sale vacio —todas las capas, no solo la nueva— y sin ningun error en
    consola. Un fallo silencioso del consumidor es justo el que hay que fijar aqui, porque
    anadir una capa parece inocuo hasta que rompe el paquete entero.
    """
    from export_agents.visor import CAPAS, CAPAS_APARIENCIA

    assert len(CAPAS) + len(CAPAS_APARIENCIA) <= 5


def test_la_capa_de_apariencia_se_declara_como_tal():
    """Si no se distingue de las medidas, alguien medira encima de una reconstruccion."""
    from export_agents.visor import CAPAS, CAPAS_APARIENCIA

    for _, _nombre, detalle in CAPAS_APARIENCIA:
        assert "APARIENCIA" in detalle
        assert "no medida" in detalle
    assert not any("APARIENCIA" in d for _, _, d in CAPAS)

