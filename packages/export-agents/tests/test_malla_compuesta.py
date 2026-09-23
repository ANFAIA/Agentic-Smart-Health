"""El STL del compuesto: corona medida + raíz reconstruida, y que se distinga cuál es cuál.

Estos tests existen porque este exportador es el único del proyecto que **escribe
geometría que nadie midió**. Todos los demás devuelven lo que entró y su error es el
del formato; aquí la mitad del fichero sale de un reconstructor, y un STL no tiene
dónde llevar esa distinción salvo en la cabecera y en lo que declare el agente. Si eso
se pierde, el resultado es un modelo imprimible que parece medido de punta a punta.
"""

from __future__ import annotations

import numpy as np
import pytest
from core_schemas import (
    Modality,
    ModalityStatus,
    Provenance,
    RigidTransform,
    TwinSnapshot,
)
from export_agents.malla_compuesta import (
    MINIMO_POR_PIEZA,
    CompositeMeshExportAgent,
    superficie_alfa,
)
from export_agents.stl import read_stl_triangles
from ingestion_agents.mesh_agent import vertex_normals

REF_CAMPO = "sha256:" + "a" * 64
REF_MALLA = "sha256:" + "b" * 64


class _Almacen:
    def __init__(self, **refs: dict) -> None:
        self._refs = refs

    def load(self, ref: str) -> dict:
        return self._refs[ref]

    def put(self, **arrays):  # pragma: no cover - este agente no escribe artefactos
        return "sha256:" + "0" * 64


def _bola(n: int, centro: np.ndarray, radio: float, semilla: int) -> np.ndarray:
    """Nube SÓLIDA. El complejo alfa necesita volumen: sobre una cáscara hueca los
    cuatro vértices de cada tetraedro caen en la misma esfera y ninguno entra."""
    rng = np.random.default_rng(semilla)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return centro + d * radio * rng.random((n, 1)) ** (1 / 3)


def _rejilla(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rejilla estructurada `(F, C)` → posiciones y caras. Dos triángulos por celda."""
    f, c = x.shape
    pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    i, j = np.meshgrid(np.arange(f - 1), np.arange(c - 1), indexing="ij")
    a, b = (i * c + j).ravel(), (i * c + j + 1).ravel()
    d, e = ((i + 1) * c + j).ravel(), ((i + 1) * c + j + 1).ravel()
    return pos, np.concatenate([np.column_stack([a, b, d]),
                                np.column_stack([b, e, d])]).astype(np.int64)


def _encia(lado: float, n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(-lado, lado, n)
    x, y = np.meshgrid(u, u)
    return _rejilla(x, y, np.zeros_like(x))


def _cupula(centro: np.ndarray, radio: float, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """La corona escaneada: el casquete de la bola que asoma sobre la encía.

    Existe para que la validación del reconstructor sea real y no un número cualquiera:
    la corona reconstruida desde las gaussianas y esta cúpula describen **la misma
    superficie esférica**, así que la distancia entre ambas es error del método. Sin
    esta mitad, medir «contra el escáner» sería medir contra un plano que la corona
    nunca tocó.
    """
    # `theta` sólo hasta donde la esfera corta z=0: el escáner no ve por debajo.
    theta = np.linspace(0.0, np.arccos(max(-centro[2], 0.0) / radio), n)
    phi = np.linspace(0.0, 2 * np.pi, n)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    return _rejilla(
        centro[0] + radio * np.sin(t) * np.cos(p),
        centro[1] + radio * np.sin(t) * np.sin(p),
        centro[2] + radio * np.cos(t),
    )


def _en_arcada(fdi: int) -> np.ndarray:
    """Coloca la pieza donde le toca por su código FDI: una herradura, no una fila.

    Hace falta para el marco anatómico y no es decorado. Con los dientes en línea, el eje
    derecha-izquierda y el antero-posterior caen los dos sobre la misma dirección y no se
    separan — `anatomical_frame` lo detecta y se niega a medir, con razón.
    """
    cuadrante, numero = divmod(fdi, 10)
    lado = -1.0 if cuadrante in (1, 4, 5, 8) else 1.0
    angulo = (numero - 1) * 0.30
    return np.array([lado * 16.0 * np.sin(angulo), 16.0 * np.cos(angulo), -2.0])


def _almacen(
    *,
    piezas: tuple[int, ...] = (36, 46),
    n_por_pieza: int = 900,
    en_arcada: bool = False,
) -> _Almacen:
    """Dos dientes bajo una encía plana, en el mismo sistema.

    Cada bola ASOMA sobre la encía, como un diente: el casquete de arriba lo mide el
    escáner y es la banda donde se puede validar el reconstructor; lo de abajo es la
    raíz, que sólo ve el CBCT. Los centros están a 12 mm y las bolas miden 4 de radio,
    así que las piezas no se tocan — que es lo que el agente promete al reconstruirlas
    por separado.
    """
    encia = _encia(14.0)
    nubes, regiones, mallas = [], [], [encia]
    # Etiqueta por vértice del escáner: 0 la encía, el FDI de la pieza cada cúpula. Es
    # exactamente la forma de `etiquetas_ios` que el pipeline pasa al agente.
    etiquetas = [np.zeros(len(encia[0]), dtype=np.int16)]
    for k, fdi in enumerate(piezas):
        centro = _en_arcada(fdi) if en_arcada else np.array([k * 12.0, 0.0, -2.0])
        nubes.append(_bola(n_por_pieza, centro, 4.0, semilla=k))
        regiones.append(np.full(n_por_pieza, fdi, dtype=np.int16))
        cupula = _cupula(centro, 4.0)
        mallas.append(cupula)
        etiquetas.append(np.full(len(cupula[0]), fdi, dtype=np.int16))
    centros = np.concatenate(nubes)
    campo = {
        "centers": centros,
        "scales": np.full((len(centros), 3), 0.3),
        "rotations": np.tile([1.0, 0.0, 0.0, 0.0], (len(centros), 1)),
        # Densidad con ESMALTE: un diente de verdad satura el techo del rango HU y por
        # eso su p95 llega a 1. Poner un 0,5 plano haría que el guardarraíl rechazara la
        # fixture entera, con razón: eso no es un diente, es un bulto.
        "density": np.full(len(centros), 0.99),
        "hu_range": np.array([300.0, 2000.0]),
        "region_id": np.concatenate(regiones),
    }
    pos, caras = _une_mallas(mallas)
    # `normals` va porque el `mesh-agent` real las guarda. El agente NO las usa —el
    # signo del sesgo sale de la anatomía, no del bobinado— pero la fixture tiene que
    # parecerse al artefacto real o deja de probar lo que se ejecuta en producción.
    superficie = {"positions": pos, "faces": caras, "normals": vertex_normals(pos, caras)}
    almacen = _Almacen(**{REF_CAMPO: campo, REF_MALLA: superficie})
    almacen.etiquetas_ios = np.concatenate(etiquetas)
    return almacen


def _une_mallas(mallas: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pos, car, off = [], [], 0
    for p, c in mallas:
        pos.append(p)
        car.append(c + off)
        off += len(p)
    return np.concatenate(pos), np.concatenate(car)


def _snapshot(*, con_superficie: bool = True, con_transform: bool = True) -> TwinSnapshot:
    prov = Provenance(
        source_file="x", modality=Modality.CBCT, agent="t", confidence=1.0,
        transform=RigidTransform(rotation=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0))
        if con_transform else None,
    )
    return TwinSnapshot(
        acquisition_id="acq-1",
        timestamp="2026-08-20T00:00:00Z",
        gaussian_field_ref=REF_CAMPO,
        surface_ref=REF_MALLA if con_superficie else None,
        provenance=prov,
    )


# --- el complejo alfa, aparte del agente ------------------------------------ #


def test_la_frontera_de_una_bola_esta_en_su_radio():
    """La prueba de que el reconstructor reconstruye: los vértices que declara
    superficie tienen que caer en la superficie de la bola, no dentro."""
    p = _bola(20_000, np.zeros(3), 5.0, semilla=0)
    caras = superficie_alfa(p, 2.5 * 0.164)

    radios = np.linalg.norm(p[np.unique(caras)], axis=1)
    assert np.median(radios) > 4.7, "la frontera se ha hundido hacia el interior"
    assert radios.max() <= 5.0 + 1e-9


def test_una_cascara_hueca_no_da_superficie():
    """No es un fallo, es la definición, y por eso se ata: cuatro puntos sobre una
    esfera tienen ESA esfera como circunscrita, así que ningún tetraedro entra en el
    complejo con un alfa pequeño. Si algún día alguien alimenta este agente con
    gaussianas ajustadas a la superficie en vez de al volumen, sale vacío y hay que
    poder leer aquí por qué."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    assert len(superficie_alfa(v * 5.0, 2.5 * 0.15)) == 0


def test_una_nube_de_menos_de_cuatro_puntos_no_es_un_volumen():
    assert len(superficie_alfa(np.zeros((3, 3)), 1.0)) == 0


def test_una_nube_coplanar_no_revienta():
    """Qhull lanza sobre geometría degenerada. Es un dato sobre la nube, no un fallo:
    tiene que salir vacía, no propagarse como excepción."""
    p = np.column_stack([np.random.default_rng(0).normal(size=(50, 2)), np.zeros(50)])
    assert len(superficie_alfa(p, 1.0)) == 0


# --- el agente -------------------------------------------------------------- #


def test_la_arcada_sale_SIN_raices(tmp_path):
    """La decisión clínica de este agente, y va al revés de lo que parecía obvio.

    Un modelo de estudio o una guía tienen que asentar sobre una base plana, y quince
    raíces colgando lo impiden. Y hay un argumento más fuerte que el mecánico: una pieza
    impresa **no lleva cabecera**, así que todo lo que este agente declara sobre el
    reconstructor desaparece al salir de la impresora. Lo reconstruido va sólo donde
    alguien lo pide a propósito.
    """
    almacen = _almacen()
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    assert salida.ok, salida.detail

    z = read_stl_triangles(salida.path).reshape(-1, 3)[:, 2]
    assert z.max() > -0.5, "no está la superficie del escáner"
    assert z.min() > -2.5, "la arcada trae geometría reconstruida bajo la encía"


def test_cada_pieza_tiene_su_fichero_con_corona_Y_raiz(tmp_path):
    """El fichero que sí se imprime: un diente completo, que apoya sobre la corona."""
    almacen = _almacen(piezas=(36, 46))
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    assert sorted(p.name for p in salida.paths) == ["c-pieza-36.stl", "c-pieza-46.stl"]

    v = read_stl_triangles(salida.paths[0]).reshape(-1, 3)
    assert v[:, 2].max() > 0.0, "falta la corona medida"
    assert v[:, 2].min() < -5.0, "falta la raíz reconstruida"
    # Y es UNA pieza, no la arcada entera: la 46 está a 12 mm y no puede estar aquí.
    assert v[:, 0].max() < 6.0, "el fichero de la 36 arrastra geometría de la vecina"


def test_la_corona_de_la_pieza_sale_del_ESCANER(tmp_path):
    """Donde ambas modalidades ven la misma superficie manda la que la mide a decenas de
    micras, no la de 0,4 mm de vóxel. Misma regla que aplica el `viewer-export-agent`."""
    almacen = _almacen(piezas=(36,))
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    from scipy.spatial import cKDTree

    escaneada = almacen.load(REF_MALLA)["positions"]
    corona = read_stl_triangles(salida.paths[0]).reshape(-1, 3)
    corona = corona[corona[:, 2] > 0.0]
    d, _ = cKDTree(escaneada).query(corona)
    assert d.max() < 1e-4, "la corona no es la del escáner"


def test_sin_etiquetas_del_escaner_la_pieza_sale_SIN_corona_y_se_dice(tmp_path):
    """No se rellena el hueco con la corona del CBCT. Existe, pero es peor, y meterla
    haría pasar por medido algo que a 0,4 mm de vóxel no lo está."""
    salida = CompositeMeshExportAgent(_almacen(piezas=(36,))).export(
        _snapshot(), tmp_path / "c.stl"
    )
    assert salida.ok
    assert any("SIN corona" in m for m in salida.hitl_reasons), salida.hitl_reasons
    assert "SIN-corona" in salida.paths[0].read_bytes()[:80].decode("ascii", "replace")


def test_unas_etiquetas_de_otra_malla_se_ignoran(tmp_path):
    """Indexar la malla con etiquetas de longitud distinta no da error: da coronas de
    otro diente. Se descartan y la pieza sale declarada sin corona."""
    almacen = _almacen(piezas=(36,))
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios[:10]
    )
    assert any("SIN corona" in m for m in salida.hitl_reasons)


def test_la_corona_del_escaner_entra_EXACTA(tmp_path):
    """La mitad medida no se reconstruye, y esa es la decisión de diseño del agente.
    Sus vértices tienen que aparecer tal cual —salvo el `float32` del formato— y no
    pasados por el complejo alfa."""
    almacen = _almacen()
    salida = CompositeMeshExportAgent(almacen).export(_snapshot(), tmp_path / "c.stl")

    escaneada = almacen.load(REF_MALLA)["positions"]
    escritos = read_stl_triangles(salida.path).reshape(-1, 3)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(escritos).query(escaneada)
    assert d.max() < 1e-4, "los vértices del escáner no llegaron intactos al STL"


def test_cada_pieza_sale_como_un_cuerpo_separado(tmp_path):
    """Reconstruir en bloque uniría dientes vecinos por donde sus alfa-complejos se
    tocan y saldría una barra de dientes fundidos: imprimible, inservible."""
    almacen = _almacen(piezas=(36, 46))
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    v = read_stl_triangles(salida.paths[0]).reshape(-1, 3)
    bajo = v[v[:, 2] < -2.0]

    # Las dos bolas están a 12 mm y miden 4 de radio: si salieran fundidas habría
    # geometría en la franja de en medio, y no la hay.
    assert len(bajo), "no hay raíz en el fichero de la pieza"
    en_medio = bajo[(bajo[:, 0] > 5.0) & (bajo[:, 0] < 7.0)]
    assert len(en_medio) == 0, "las dos piezas se han unido en un solo cuerpo"


def test_se_declara_el_error_del_RECONSTRUCTOR_no_el_del_formato(tmp_path):
    """El número que decide si el fichero se puede usar es cuánto se desvía la
    superficie reconstruida, no el 1e-5 mm del `float32`. Declarar el segundo daría un
    valor excelente que no dice nada de la mitad del fichero que puede estar mal."""
    salida = CompositeMeshExportAgent(_almacen()).export(_snapshot(), tmp_path / "c.stl")

    assert salida.max_deviation_mm is not None
    assert salida.max_deviation_mm > 1e-3, (
        "una desviación de orden 1e-5 mm es el error del formato: se está midiendo el "
        "canal equivocado"
    )
    assert "hipótesis" in (salida.detail or ""), (
        "extrapolar de la corona a la raíz es una hipótesis y tiene que ir dicha"
    )


def test_la_cabecera_dice_que_mitad_es_medida_y_cual_reconstruida(tmp_path):
    """Un STL suelto viaja sin manifiesto: estos 80 bytes son el único sitio donde
    quien lo reciba puede leer de qué está hecho.

    Y caben 80 CONTADOS: `stl_header` trunca en silencio, así que una cabecera que
    crezca se lleva por delante justo la cifra del final. La primera versión metía el
    `acquisition_id` y dejaba la medida fuera del fichero sin que nada avisara.
    """
    almacen = _almacen()
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    cabecera = salida.path.read_bytes()[:80].decode("ascii", "replace").rstrip("\0")

    assert len(cabecera.encode("ascii")) <= 80
    assert not cabecera.startswith("solid"), "se leería como STL de texto"
    assert "IOS-medida" in cabecera
    assert "sin-raices" in cabecera, "la arcada tiene que declarar que no las lleva"

    pieza = salida.paths[0].read_bytes()[:80].decode("ascii", "replace").rstrip("\0")
    assert len(pieza.encode("ascii")) <= 80
    assert "recon" in pieza
    assert "sesgo=" in pieza, "la cifra se ha quedado fuera de los 80 bytes"


def test_sin_escaner_es_MISSING_no_FAILED(tmp_path):
    salida = CompositeMeshExportAgent(_almacen()).export(
        _snapshot(con_superficie=False), tmp_path / "c.stl"
    )
    assert salida.status is ModalityStatus.MISSING
    assert "corona" in (salida.detail or "")


def test_sin_registro_es_MISSING(tmp_path):
    """Sin la rígida la corona y la raíz caerían en sistemas distintos dentro del mismo
    fichero: dos objetos sueltos con aspecto de modelo."""
    salida = CompositeMeshExportAgent(_almacen()).export(
        _snapshot(con_transform=False), tmp_path / "c.stl"
    )
    assert salida.status is ModalityStatus.MISSING


def test_sin_segmentacion_no_se_reconstruye_nada(tmp_path):
    """En bloque las piezas saldrían fundidas, así que se declara en vez de entregarlo."""
    almacen = _almacen()
    almacen.load(REF_CAMPO)["region_id"][:] = 0
    salida = CompositeMeshExportAgent(almacen).export(_snapshot(), tmp_path / "c.stl")

    assert salida.status is ModalityStatus.MISSING
    assert "FDI" in (salida.detail or "")


def test_una_pieza_con_pocas_gaussianas_se_declara_omitida(tmp_path):
    """Una esquirla de 20 puntos no es una raíz. Sale del fichero y se dice, en vez de
    escribir un cuerpo suelto que el clínico encontraría flotando en el modelo."""
    almacen = _almacen(piezas=(36, 46))
    region = almacen.load(REF_CAMPO)["region_id"]
    region[region == 46] = 0
    region[np.flatnonzero(region == 36)[: MINIMO_POR_PIEZA - 10]] = 46

    salida = CompositeMeshExportAgent(almacen).export(_snapshot(), tmp_path / "c.stl")
    assert salida.ok
    assert any("46" in m for m in salida.hitl_reasons), salida.hitl_reasons


@pytest.mark.parametrize("campo_origen", [True, False])
def test_el_campo_centrado_y_el_absoluto_acaban_en_el_mismo_sitio(tmp_path, campo_origen):
    """`origin` es la trampa de este pipeline: un campo centrado y una malla en
    coordenadas absolutas producen un STL con la raíz a un lado y la corona a otro, y
    el fichero se abre sin protestar."""
    almacen = _almacen()
    if campo_origen:
        campo = almacen.load(REF_CAMPO)
        campo["centers"] = campo["centers"] - np.array([100.0, 0.0, 0.0])
        campo["origin"] = np.array([100.0, 0.0, 0.0])

    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    # ⚠️ Se mira el fichero POR PIEZA y no la arcada: la arcada es sólo la malla del
    # escáner, a la que `origin` no le afecta, así que ahí este test no podría fallar.
    # La raíz es lo único que viaja por el camino del campo.
    v = read_stl_triangles(salida.paths[0]).reshape(-1, 3)
    raiz = v[v[:, 2] < -2.0]

    assert len(raiz), "no hay raíz que comprobar"
    assert raiz[:, 0].min() > -20.0, "la raíz se ha quedado en el sistema del CBCT"


def test_el_sesgo_va_hacia_dentro(tmp_path):
    """La raíz sale DELGADA, y hay que poder leerlo. El reconstructor usa los centros
    de las gaussianas como puntos de superficie, y un centro está dentro del tejido:
    el error no está repartido alrededor de la verdad, está sesgado a un lado. Un p95
    a secas se leería como ruido y llevaría a imprimir una raíz más fina de lo que el
    CBCT midió."""
    salida = CompositeMeshExportAgent(_almacen()).export(_snapshot(), tmp_path / "c.stl")

    assert "hacia dentro" in (salida.detail or ""), salida.detail
    assert "delgada" in (salida.detail or "")


def test_el_signo_del_sesgo_no_depende_del_bobinado_de_la_malla(tmp_path):
    """El fallo que esto guarda: el signo salía de las normales por vértice del
    escáner, y una normal apunta hacia donde diga el orden de los índices de sus
    caras. Con el bobinado invertido —que es una convención del fichero, no anatomía—
    el mismo error se declaraba «hacia fuera», o sea una raíz gruesa donde la había
    delgada. Ahora el signo sale de la dirección radial del propio diente."""
    almacen = _almacen()
    malla = almacen.load(REF_MALLA)
    malla["faces"] = malla["faces"][:, ::-1].copy()
    malla["normals"] = -malla["normals"]

    salida = CompositeMeshExportAgent(almacen).export(_snapshot(), tmp_path / "c.stl")
    assert "hacia dentro" in (salida.detail or ""), salida.detail


def test_un_cuerpo_sin_esmalte_NO_es_un_diente(tmp_path):
    """El fallo que esto guarda, y que sólo se vio porque alguien contó los dientes.

    El pipeline exportó una pieza «28» sobre un caso real. Tenía tamaño de diente
    (19,4 mm), forma de diente y la misma separación del vecino que dos molares
    contiguos: por geometría era indistinguible. Era la **tuberosidad del maxilar**.

    Lo que la delata es la densidad. El esmalte es el tejido más denso del cuerpo y
    satura el techo del rango HU: los catorce dientes reales de aquel caso daban p95 =
    1,000 exacto y el intruso 0,747, a un pelo del 0,710 del hueso sin etiquetar. Un
    diente sin corona de esmalte no existe.
    """
    almacen = _almacen(piezas=(36, 46))
    campo = almacen.load(REF_CAMPO)
    # La 46 pasa a ser hueso: misma geometría, densidad sin esmalte.
    campo["density"] = np.where(campo["region_id"] == 46, 0.55, 0.99)
    campo["hu_range"] = np.array([300.0, 2000.0])

    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    assert [p.name for p in salida.paths] == ["c-pieza-36.stl"]
    assert any("46" in m and "esmalte" in m for m in salida.hitl_reasons), salida.hitl_reasons


def test_con_el_techo_de_HU_alto_el_guardarrail_se_APAGA_y_se_nota(tmp_path):
    """El umbral vive en densidad normalizada, así que depende de `hu_range`. Con un
    techo muy por encima del esmalte, un diente bueno daría p95 bajo y el guardarraíl se
    pondría a rechazar piezas reales. Antes que eso, se desactiva."""
    almacen = _almacen(piezas=(36, 46))
    campo = almacen.load(REF_CAMPO)
    campo["density"] = np.full(len(campo["region_id"]), 0.55)
    campo["hu_range"] = np.array([300.0, 12000.0])

    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    assert len(salida.paths) == 2, "el guardarraíl no se ha desactivado"
    assert not any("esmalte" in m for m in salida.hitl_reasons)


def test_la_arcada_sale_CERRADA_cuando_hay_eje_anatomico(tmp_path):
    """Un escaneo intraoral es una cáscara: sin fondo y sin interior. Se ve bien en
    pantalla —el visor lleva meses enseñándolo— y no se imprime, porque un laminador
    necesita saber qué es dentro. Con eje anatómico medible, la arcada sale sólida."""
    # Códigos que cubren los dos lados Y los dos sectores: sin eso no hay marco medible.
    almacen = _almacen(piezas=(11, 16, 21, 26), en_arcada=True)
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    assert salida.ok, salida.detail
    assert "solido-cerrado" in salida.path.read_bytes()[:80].decode("ascii", "replace")
    assert not any("estanca" in m for m in salida.hitl_reasons), salida.hitl_reasons
    # ⚠️ La comprobación topológica de verdad vive en `test_solido`, sobre índices. Aquí
    # no se puede repetir fusionando vértices por posición: `solido` declara que NO
    # suelda duplicados, y esta fixture los tiene por construcción —la costura y el polo
    # de cada cúpula—. Re-derivarla con otro criterio probaría la fixture, no el agente.
    assert salida.n_faces > len(almacen.load(REF_MALLA)["faces"]), "no se añadió base"


def test_sin_eje_anatomico_se_entrega_la_CASCARA_y_se_dice(tmp_path):
    """No se inventa una vertical. El eje Z de un escaneo es el que tenía la máquina: una
    base construida sobre él sale inclinada y el modelo se cae de la bandeja."""
    almacen = _almacen(piezas=(36, 46))  # sólo posteriores: no hay sector anterior
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    assert salida.ok
    assert "CASCARA-ABIERTA" in salida.path.read_bytes()[:80].decode("ascii", "replace")
    assert any("NO ha quedado estanca" in m for m in salida.hitl_reasons)


def test_la_base_cae_al_lado_CONTRARIO_de_las_coronas(tmp_path):
    """El fallo que esto guarda, y que ningún test topológico veía: la malla salía
    estanca, con volumen positivo y todas las aristas compartidas — y del revés.

    Se le estaba pasando el eje SUPERIOR, que en un maxilar es el opuesto del oclusal.
    El plano caía por delante de los dientes y la falda los envolvía en una caja: 21,7
    cm³ contra los 11,9 correctos. Un modelo se apoya por donde estaba el hueso y muerde
    por el otro lado, y eso no lo dice la topología.
    """
    from export_agents.anatomia import anatomical_frame
    from export_agents.stl import read_stl_triangles

    almacen = _almacen(piezas=(11, 16, 21, 26), en_arcada=True)
    salida = CompositeMeshExportAgent(almacen).export(
        _snapshot(), tmp_path / "c.stl", etiquetas_ios=almacen.etiquetas_ios
    )
    malla = almacen.load(REF_MALLA)
    marco, motivo = anatomical_frame(malla["positions"], almacen.etiquetas_ios)
    assert marco is not None, motivo
    oclusal = marco.oclusal / np.linalg.norm(marco.oclusal)

    v = read_stl_triangles(salida.path).reshape(-1, 3)
    coronas = malla["positions"][almacen.etiquetas_ios > 0] @ oclusal
    assert float((v @ oclusal).min()) < float(coronas.min()), (
        "la base no está por debajo de las coronas"
    )
    assert float((v @ oclusal).max()) <= float(coronas.max()) + 1e-6, (
        "hay geometría nueva POR ENCIMA de las coronas: la base salió del lado equivocado"
    )
