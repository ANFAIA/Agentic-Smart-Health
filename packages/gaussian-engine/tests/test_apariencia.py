

# --- des-normalizacion de Blender -------------------------------------------- #
def _params(n=32):
    import numpy as np

    rng = np.random.default_rng(0)
    return {
        "means": rng.uniform(-1.0, 1.0, (n, 3)),
        "scales": np.full((n, 3), -3.5),          # log
        "quats": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "opacities": np.zeros(n),
        "colors": np.full((n, 3), 0.5),
    }


def _dtype_de(cabecera: str):
    """El dtype que el PLY DECLARA, no el que esperábamos.

    ⚠️ Esta lista estaba escrita a mano y se quedó vieja en cuanto el escritor añadió los
    `f_rest_*` del relieve: las pruebas leían con el stride antiguo y comparaban basura
    contra el valor bueno. Es exactamente el fallo que el propio módulo ya arregló dos
    veces —las unidades y el esquema del sidecar— y la misma cura: preguntar al fichero.
    """
    import numpy as np

    tipos = {"float": "<f4", "double": "<f8", "short": "<i2", "int": "<i4", "uchar": "u1"}
    campos = [
        (linea.split()[2], tipos[linea.split()[1]])
        for linea in cabecera.splitlines()
        if linea.startswith("property ") and "list" not in linea
    ]
    return np.dtype(campos)


def _lee(ruta, n):
    import numpy as np

    b = ruta.read_bytes()
    i = b.index(b"end_header") + len(b"end_header") + 1
    dt = _dtype_de(b[:i].decode("ascii", "replace"))
    return np.frombuffer(b, dtype=dt, offset=i, count=n)


def test_el_PLY_sale_en_MILIMETROS_y_no_en_el_espacio_de_blender(tmp_path):
    """`mundo = normalizado / scan_scale + scan_offset`, que es lo que Blender deja escrito.

    ⚠️ Este es el test que faltaba y el fallo que tapó. Blender normaliza la malla para
    renderizar y gsplat entrena en ESE espacio; escribir los parámetros tal cual produce un
    PLY que declara milímetros sobre un dato normalizado. Medido sobre un caso real con
    `scan_scale` 0,0308: la nube salía **32 veces más pequeña** que la malla y con σ de
    0,022 mm en vez de 0,70. En el visor era una mota en el origen, y como el descriptor
    afirmaba milímetros no había forma de sospecharlo desde fuera.
    """
    import numpy as np
    from gaussian_engine.apariencia import escribe_inria

    p = _params()
    escala, offset = 0.03078, [-0.95, 2.34, -3.64]
    ruta = tmp_path / "a.ply"
    escribe_inria(ruta, p, scan_scale=escala, scan_offset=offset)
    a = _lee(ruta, len(p["means"]))

    esperado = p["means"] / escala + np.asarray(offset)
    salida = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    assert np.allclose(salida, esperado, atol=1e-2), (salida[0], esperado[0])
    # Y la escala, que va en LOGARITMO: la correccion es una suma, no un producto.
    assert np.allclose(np.exp(a["scale_0"]), np.exp(-3.5) / escala, rtol=1e-3)


def test_sin_scan_scale_el_PLY_sale_en_el_espacio_NORMALIZADO(tmp_path):
    """El contrario, para que el de arriba pueda fallar.

    Y documenta el contrato: `escribe_inria` no adivina. Si el llamante no pasa la
    normalización, escribe lo que le den — por eso el que la tiene, `entrena_apariencia`,
    está obligado a pasarla.
    """
    import numpy as np
    from gaussian_engine.apariencia import escribe_inria

    p = _params()
    ruta = tmp_path / "b.ply"
    escribe_inria(ruta, p)
    a = _lee(ruta, len(p["means"]))
    salida = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    assert np.allclose(salida, p["means"], atol=1e-4)


def test_un_scan_scale_imposible_se_declara(tmp_path):
    """Dividir por cero o por un negativo daría una nube volteada o infinita, en silencio."""
    import pytest
    from gaussian_engine.apariencia import escribe_inria

    for malo in (0.0, -0.5):
        with pytest.raises(ValueError, match="positivo"):
            escribe_inria(tmp_path / "c.ply", _params(), scan_scale=malo)


def test_el_PLY_puede_llevar_el_FDI_por_gaussiana(tmp_path):
    """Sin `region_id` en el campo, seleccionar una pieza obliga a tener la malla delante.

    El picking del §11.3 está definido sobre `extras.uos_fdi` de un *primitive* del glTF, y
    un contenedor de solo gaussianas no lleva glTF. Con el código en el propio campo,
    encender un diente es filtrar por número: ninguna superficie de por medio.
    """
    import struct

    import numpy as np
    from gaussian_engine.apariencia import escribe_inria

    p = _params(8)
    reg = np.array([11, 11, 12, 12, 0, 0, 26, 26], dtype=np.int16)
    ruta = tmp_path / "a.ply"
    escribe_inria(ruta, p, region_id=reg)

    b = ruta.read_bytes()
    cab = b[: b.index(b"end_header")].decode("utf-8", "replace")
    assert "property short region_id" in cab
    assert "MAS CERCANA" in cab, "tiene que decir de dónde sale la etiqueta"
    i = b.index(b"end_header") + len(b"end_header") + 1
    dt = _dtype_de(cab)
    assert "region_id" in dt.names
    assert int(np.frombuffer(b, dtype=dt, offset=i, count=1)["region_id"][0]) == 11
    # y el stride cuadra con lo que la cabecera declara
    assert len(b) - i == 8 * dt.itemsize
    _ = struct


def test_un_region_id_de_otro_tamano_se_declara(tmp_path):
    """Un desajuste de tamaño escribiría códigos desplazados: cada gaussiana con el FDI de
    otra, y sin nada que fallara."""
    import numpy as np
    import pytest
    from gaussian_engine.apariencia import escribe_inria

    with pytest.raises(ValueError, match="region_id trae"):
        escribe_inria(tmp_path / "b.ply", _params(8),
                      region_id=np.zeros(5, dtype=np.int16))


def test_sin_region_id_el_PLY_no_lo_declara(tmp_path):
    """Para que el primero pueda fallar."""
    from gaussian_engine.apariencia import escribe_inria

    ruta = tmp_path / "c.ply"
    escribe_inria(ruta, _params(8))
    b = ruta.read_bytes()
    cab = b[: b.index(b"end_header")].decode("utf-8", "replace")
    assert "region_id" not in cab


def test_la_cabecera_del_PLY_es_ASCII(tmp_path):
    """El formato PLY define la cabecera como ASCII, y la nuestra llevaba tildes.

    ⚠️ Un contenedor real salía con **7 bytes no-ASCII** —«movió», «visualización»,
    «radiológica»— y nuestro propio lector la decodifica con `TextDecoder('ascii')`. No
    reventaba porque los sustituye, pero un lector estricto ajeno puede rechazar el
    fichero, y un formato que solo lee su emisor no es un formato.

    Se descubrió porque un test intentó leer la cabecera como ASCII y falló. El test valía
    más que lo que estaba probando.
    """
    from gaussian_engine.apariencia import escribe_inria

    ruta = tmp_path / "a.ply"
    escribe_inria(ruta, _params(4), scan_scale=0.03, scan_offset=[0.0, 0.0, 0.0])
    b = ruta.read_bytes()
    cab = b[: b.index(b"end_header")]
    malos = [(i, hex(c)) for i, c in enumerate(cab) if c > 127]
    assert not malos, f"la cabecera trae bytes no-ASCII: {malos[:5]}"
    cab.decode("ascii")  # y que no lance


def test_el_limite_del_color_sale_de_las_ETIQUETAS_y_no_de_la_altura():
    """⚠️ El color no es medido, pero su frontera tiene que tener procedencia.

    Antes se interpolaba entre los percentiles 30 y 70 de `z`: eso **afirma el margen
    gingival por un número inventado**, que es justo la frontera que este proyecto tiene
    medido que no sabe determinar. Con las etiquetas, el límite es la salida del
    segmentador — que viaja declarada como Layer 3, con su hash de pesos y su fiabilidad.

    La prueba: dos vértices a la MISMA altura con etiquetas distintas tienen que salir de
    colores distintos. Con el criterio viejo saldrían idénticos.
    """
    import numpy as np
    from gaussian_engine.apariencia import _colorea_malla

    esmalte = np.array([230.0, 220.0, 200.0])
    encia = np.array([180.0, 110.0, 110.0])
    pos = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [2.0, 0.0, 5.0]])
    etq = np.array([16, 0, 26])

    col = _colorea_malla(pos, np.zeros_like(pos), esmalte, encia, etq)
    assert (col[0] == esmalte.astype(np.uint8)).all()
    assert (col[1] == encia.astype(np.uint8)).all()
    assert (col[2] == esmalte.astype(np.uint8)).all()
    # y con el criterio viejo los tres serían el mismo color, porque z es idéntica
    viejo = _colorea_malla(pos, np.zeros_like(pos), esmalte, encia)
    assert (viejo[0] == viejo[1]).all() and (viejo[1] == viejo[2]).all()


def test_sin_etiquetas_se_cae_al_degradado_por_altura():
    """El apaño sigue estando, para un caso sin segmentación — y declarado como apaño."""
    import numpy as np
    from gaussian_engine.apariencia import _colorea_malla

    pos = np.array([[0.0, 0, 0.0], [0.0, 0, 5.0], [0.0, 0, 10.0]])
    col = _colorea_malla(pos, np.zeros_like(pos),
                         np.array([255.0, 255, 255]), np.array([0.0, 0, 0]))
    assert col[0][0] < col[1][0] < col[2][0], "debería degradar con la altura"


def test_los_params_llevan_TODO_lo_que_hace_falta_para_reescribir_el_PLY():
    """El PLY lo escriben DOS sitios, y el segundo solo ve lo que el almacén guarde.

    ⚠️ Dos cosas se calcularon, se imprimieron en el log y **no llegaron al fichero**: la
    des-normalización de Blender (la nube salía 32 veces más pequeña) y el `region_id` por
    gaussiana (sin él no hay forma de encender una pieza sin malla). Las dos por lo mismo:
    `store.put()` enumeraba claves a mano y la lista se quedó vieja.

    Este test fija el contrato: lo que `entrena_apariencia` inyecta en `params` es lo que
    `escribe_inria` necesita para producir un PLY correcto **sin argumentos extra**. Si
    alguien añade un parámetro a `escribe_inria` que no tiene defecto en `params`, aquí se
    entera.
    """
    import inspect

    from gaussian_engine.apariencia import escribe_inria

    # Los que `escribe_inria` sabe sacar del propio dict de parámetros.
    DESDE_PARAMS = {"scan_scale", "scan_offset", "region_id", "n_vistas", "iteraciones"}
    firma = set(inspect.signature(escribe_inria).parameters)
    opcionales = firma - {"destino", "params", "perfil", "acquisition_id"}
    sin_defecto = opcionales - DESDE_PARAMS
    assert not sin_defecto, (
        f"`escribe_inria` acepta {sorted(sin_defecto)} y `params` no lo trae: o le das "
        "defecto desde el dict, o el PLY que reescriba el agente de UOS saldrá sin ello"
    )


# ── El esquema declara lo que el fichero trae ───────────────────────────────────
#
# ⚠️ `esquema_apariencia()` enumeraba catorce columnas a mano mientras `escribe_inria`
# emitía dieciocho propiedades. `region_id` —el código FDI por gaussiana, que es lo que
# permite encender una pieza sin llevar la malla— viajaba en los bytes y no en el sidecar:
# para cualquier lector que no fuera el nuestro, ese dato no existía. Cuarta vez de la
# misma familia de fallo (lista enumerada que envejece aparte del código que escribe).

def test_el_esquema_declara_region_id_cuando_el_PLY_lo_trae():
    from gaussian_engine.agente_apariencia import esquema_apariencia
    nombres = [c.nombre for c in esquema_apariencia(("x", "y", "z", "region_id"))]
    assert "region_id" in nombres


def test_el_esquema_no_declara_region_id_cuando_el_PLY_no_lo_trae():
    """Declarar de más es tan mentira como declarar de menos: un campo sin segmentar no
    lleva la columna, igual que `esquema_del_campo` del exportador de densidad."""
    from gaussian_engine.agente_apariencia import esquema_apariencia
    from gaussian_engine.apariencia import PROPIEDADES_INRIA
    nombres = [c.nombre for c in esquema_apariencia(PROPIEDADES_INRIA)]
    assert "region_id" not in nombres


def test_el_esquema_cubre_todas_las_propiedades_que_escribe_el_PLY(tmp_path):
    """El contrato en una línea: lo que `escribe_inria` pone en la cabecera es exactamente
    lo que `esquema_apariencia` describe, y en el mismo orden — `columns` es la receta para
    montar el registro binario."""
    import numpy as np
    from gaussian_engine.agente_apariencia import esquema_apariencia
    from gaussian_engine.apariencia import escribe_inria

    n = 5
    destino = tmp_path / "ap.ply"
    escribe_inria(
        destino,
        {
            "means": np.zeros((n, 3)), "scales": np.zeros((n, 3)),
            "quats": np.tile([1.0, 0, 0, 0], (n, 1)),
            "opacities": np.zeros(n), "colors": np.full((n, 3), 0.5),
        },
        region_id=np.arange(n, dtype=np.int16),
    )
    cab = destino.read_bytes()
    cab = cab[: cab.find(b"end_header")].decode("ascii")
    props = [ln.split()[2] for ln in cab.splitlines()
             if ln.startswith("property ") and len(ln.split()) == 3]
    assert [c.nombre for c in esquema_apariencia(props)] == props


# ── El FDI por gaussiana sale de un VOTO, no del vecino más cercano ──────────────
#
# ⚠️ El contacto interproximal no tiene borde geométrico: dos coronas contiguas se tocan, y
# el vértice más próximo a una gaussiana del contacto puede ser el del diente de al lado.
# Sobre la malla eso ya se corregía —`afina_fronteras` reasignó 3.559 vértices en un caso
# real— y las gaussianas se saltaban la corrección heredando con 1-NN. Se veía: una pieza
# encendía también un trozo de su vecina.

def test_el_voto_gana_al_vecino_mas_cercano_en_el_contacto():
    """Una gaussiana rodeada de vértices del 11 con UNO del 12 pegado no es del 12."""
    import numpy as np
    from gaussian_engine.apariencia import _fdi_por_gaussiana

    # Un intruso del 12 a 0,05 mm; quince del 11 alrededor, todos algo más lejos.
    vert = [[0.05, 0.0, 0.0]]
    etq = [12]
    for k in range(15):
        a = 2 * np.pi * k / 15
        vert.append([0.3 * np.cos(a), 0.3 * np.sin(a), 0.0])
        etq.append(11)
    reg, lejos, cambiadas = _fdi_por_gaussiana(
        np.zeros((1, 3)), np.asarray(vert), np.asarray(etq), limite=2.0,
    )
    assert reg[0] == 11, "el voto no ha ganado al vecino más cercano"
    assert not lejos[0]
    assert cambiadas == 1, "el 1-NN habría dicho 12, así que tiene que contar como cambio"


def test_el_empate_cae_del_lado_de_la_encia():
    """⚠️ Es la dirección correcta en la que equivocarse: marcar encía como diente pinta
    esmalte sobre encía y enciende trozos ajenos al seleccionar; marcar diente como encía
    sólo deja un borde sin asignar."""
    import numpy as np
    from gaussian_engine.apariencia import _fdi_por_gaussiana

    vert = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0]], dtype=float)
    reg, _, _ = _fdi_por_gaussiana(
        np.zeros((1, 3)), vert, np.array([11, 11, 0, 0]), limite=2.0, votos=4,
    )
    assert reg[0] == 0


def test_la_cota_de_distancia_se_mide_contra_el_MAS_CERCANO():
    """El optimizador mueve, divide y poda: una gaussiana lejos de todo no hereda nada.
    Y la cota mira al vecino más próximo — si ése ya está fuera, los demás también."""
    import numpy as np
    from gaussian_engine.apariencia import _fdi_por_gaussiana

    vert = np.array([[10.0, 0, 0], [10.1, 0, 0], [10.2, 0, 0]], dtype=float)
    reg, lejos, _ = _fdi_por_gaussiana(
        np.zeros((1, 3)), vert, np.array([11, 11, 11]), limite=2.0, votos=3,
    )
    assert lejos[0] and reg[0] == 0


def test_sin_ambiguedad_el_voto_no_cambia_nada():
    """Que el voto no invente: donde todos los vecinos coinciden, la etiqueta es la suya."""
    import numpy as np
    from gaussian_engine.apariencia import _fdi_por_gaussiana

    rng = np.random.default_rng(0)
    vert = rng.normal(scale=0.2, size=(40, 3))
    reg, _, cambiadas = _fdi_por_gaussiana(
        np.zeros((1, 3)), vert, np.full(40, 26), limite=2.0,
    )
    assert reg[0] == 26 and cambiadas == 0


def test_la_cabecera_nombra_el_color_POR_PIEZA_cuando_es_el_que_manda() -> None:
    """⚠️ **Dos mecanismos distintos no caben en un contador.**

    La version anterior atribuia TODO a la proyeccion por vertice con PnP. Con el color por
    pieza mandando eso declaraba una procedencia falsa del 97 % de los vertices, y encima
    seguia contando como «interpolados» unos que la pieza ya habia sobrescrito. Una
    cabecera que exagera lo inventado es peor que no llevarla: es la unica procedencia que
    viaja pegada al fichero.
    """
    import numpy as np
    from gaussian_engine.apariencia import _comentarios_color

    por_pieza = _comentarios_color({
        "n_vertices_malla": np.asarray(112067),
        "n_vertices_medidos": np.asarray(1200),
        "n_vertices_interpolados": np.asarray(800),
        "n_vertices_por_pieza": np.asarray(108922),
        "n_piezas_con_tono": np.asarray(13),
    })
    junto = " ".join(por_pieza)
    assert "POR PIEZA" in junto
    assert "13 corona" in junto and "108922" in junto
    assert "NO hace falta" in junto and "pose" in junto
    # ⚠️ Los tres cubos tienen que sumar. La primera version decia «el resto lo hereda de
    # la proyeccion por vertice (7 medidos, 97 interpolados)» y dejaba 3.041 vertices sin
    # nombrar — los que se pintan con el degradado de respaldo, que es justo el unico
    # cubo que NO es color del paciente y por tanto el que hay que declarar.
    assert "3145 restantes" in junto
    assert "1145" in junto or "respaldo" in junto
    assert str(112067 - 108922 - 1200 - 800) in junto

    # Sin color por pieza se conserva el texto del camino por vertice, que sigue siendo
    # cierto cuando es el que pinta.
    por_vertice = _comentarios_color({
        "n_vertices_malla": np.asarray(112067),
        "n_vertices_medidos": np.asarray(72430),
        "n_vertices_interpolados": np.asarray(23457),
    })
    assert "PnP" in " ".join(por_vertice)
    assert "POR PIEZA" not in " ".join(por_vertice)

    # Y sin nada medido, los dos tonos siguen declarandose como lo que son.
    assert "DOS tonos" in " ".join(_comentarios_color({}))


def test_el_relieve_SH1_vale_EXACTAMENTE_n_por_v() -> None:
    """⚠️ **El grado 1 no es una aproximación de la luz: es la luz, escrita.**

    El rasterizador evalúa `c = C0·sh0 + C1·(−y·sh1 + z·sh2 − x·sh3) + 0,5` con `(x,y,z)` la
    dirección de vista, o sea una función **lineal** de la dirección. Un término difuso
    `n·v` tiene esa misma forma, así que cabe exacto y no hay que entrenarlo.

    Esto existe porque el campo pasó a entrenarse contra renders de albedo plano —sin eso
    las gaussianas guardaban el diente bajo un sol que nos inventábamos, y recuperar el
    color medido costaba ΔE 28 en vez de 0,35 por pieza—. El precio era que un albedo puro
    se dibuja plano; el grado 1 devuelve el volumen **sin tocar el grado 0**.
    """
    import numpy as np
    from gaussian_engine.apariencia import C1, FUERZA_RELIEVE, _relieve_sh1

    rng = np.random.default_rng(0)
    malla = rng.normal(size=(64, 3))
    normales = rng.normal(size=(64, 3))
    normales /= np.linalg.norm(normales, axis=1, keepdims=True)
    albedo = rng.uniform(0.2, 0.9, size=(64, 3))

    n, sh = _relieve_sh1(malla, malla, normales, albedo)
    assert sh is not None and sh.shape == (64, 9)
    sh = sh.reshape(64, 3, 3)

    for _ in range(4):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        x, y, z = v
        aporta = C1 * (-y * sh[:, :, 0] + z * sh[:, :, 1] - x * sh[:, :, 2])
        assert np.allclose(aporta, FUERZA_RELIEVE * albedo * (n @ v)[:, None], atol=1e-12)


def test_sin_albedo_no_hay_relieve_y_se_dice_con_None() -> None:
    """Un grado 1 a cero se dibuja plano, que es la degradación correcta. Inventarlo, no."""
    import numpy as np
    from gaussian_engine.apariencia import _relieve_sh1

    malla = np.zeros((8, 3))
    assert _relieve_sh1(malla, malla, malla, None) == (None, None)


def test_la_luz_del_relieve_es_RASANTE_y_la_normal_declarada_es_la_de_verdad() -> None:
    """⚠️ **Codificar `n·v` a secas era poner la luz EN el eje de la cámara.**

    `scripts/blender_render_views.py` ya tenía escrito por qué eso no vale: «una luz en el
    eje de la cámara es plana: `n·l ≈ n·v`, así que toda superficie que te mira brilla
    igual y el relieve se pierde». La luz oblicua no añade detalle — **alarga el gradiente**
    del que ya está en la geometría, y por eso se ven las troneras.

    Cabe en el mismo grado 1 porque rotar es lineal: `n·(R v) = (Rᵀn)·v`. Y el relleno
    opuesto también, porque la suma de dos términos lineales sigue siendo lineal.

    ⚠️ Lo que NO puede rotarse es lo que se declara: `nx,ny,nz` llevan la normal medida,
    sin girar, para que un lector pueda recalcular el relieve o contradecirlo.
    """
    import numpy as np
    from gaussian_engine.apariencia import _relieve_sh1, _rota

    rng = np.random.default_rng(1)
    malla = rng.normal(size=(48, 3))
    normales = rng.normal(size=(48, 3))
    normales /= np.linalg.norm(normales, axis=1, keepdims=True)
    albedo = rng.uniform(0.3, 0.8, size=(48, 3))
    eje = np.array([0.0, 0.0, 1.0])

    n_recta, sh_recta = _relieve_sh1(malla, malla, normales, albedo)
    n_rasante, sh_rasante = _relieve_sh1(malla, malla, normales, albedo, eje_rasante=eje)

    # La normal declarada es la MISMA con y sin luz rasante: no se gira lo que se declara.
    assert np.allclose(n_recta, n_rasante)
    assert np.allclose(n_rasante, normales)
    # Y el relieve sí cambia, que es el objetivo.
    assert not np.allclose(sh_recta, sh_rasante)
    # Girar 0 grados tiene que devolver la normal intacta (cordura de Rodrigues).
    assert np.allclose(_rota(normales, eje, 0.0), normales)


def _valle(nx: int = 40, paso: float = 0.4):
    """Una loseta con un valle en `y = 0`: `z = 0,8·|y|`. Lo único cóncavo está ahí."""
    import numpy as np

    ys = np.arange(-6.0, 6.0 + paso, paso)
    xs = np.arange(nx) * paso
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X, Y, 0.8 * np.abs(Y)], axis=-1).reshape(-1, 3)
    # Normal del valle: apunta hacia arriba, inclinada según el lado.
    n = np.stack([np.zeros(len(pos)), -np.sign(pos[:, 1]) * 0.8, np.ones(len(pos))], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return pos, n


def test_la_oclusion_oscurece_el_SURCO_y_no_la_superficie_lisa() -> None:
    """⚠️ **Esto es lo que la oclusión tiene que hacer, y lo único que se le pide.**

    No es la oclusión de un trazador de rayos y no se declara como tal: cuenta vecinos por
    delante del plano tangente. Coincide con la de verdad en lo que aquí importa —que las
    hendiduras se oscurezcan y las cúspides no— y no en el valor absoluto, que por eso
    viaja como factor de visualización en [0,1] y no como magnitud física.
    """
    import numpy as np
    from gaussian_engine.apariencia import OCLUSION_MINIMA, _oclusion_ambiental

    pos, nor = _valle()
    ao = _oclusion_ambiental(pos, nor)

    assert ao.shape == (len(pos),)
    assert ((ao >= OCLUSION_MINIMA - 1e-9) & (ao <= 1.0 + 1e-9)).all()
    surco = np.abs(pos[:, 1]) < 0.5
    liso = np.abs(pos[:, 1]) > 4.0
    assert ao[surco].mean() < ao[liso].mean(), "el fondo del valle tiene que salir más oscuro"
    # ⚠️ **Y sobre todo: lo liso NO se oscurece.** La primera versión contaba cuántos
    # vecinos quedaban por delante del plano tangente, y sobre una superficie eso es la
    # mitad se esté donde se esté: el 83 % de las gaussianas del caso real se iba al suelo
    # del rango y la oclusión oscurecía la arcada entera por igual, que es no hacer nada.
    # Ordenar bien surco y liso no basta si los dos están saturados.
    assert ao[liso].mean() > 0.9, "una superficie lisa no tiene nada que la tape"
    assert (ao <= OCLUSION_MINIMA + 1e-6).mean() < 0.25, "no puede saturar en todas partes"


def test_la_oclusion_se_mide_en_la_SUPERFICIE_y_no_donde_caiga_la_gaussiana() -> None:
    """⚠️ El fallo que costó dos regeneraciones, y la firma ya no lo permite.

    Medirla en los centros de las gaussianas parecía natural —son los puntos que se pintan—
    pero el optimizador los mueve fuera de la superficie, y uno que quede por dentro tiene
    a **todos** sus vecinos por delante de su plano tangente: sale oscuro siempre. Sobre el
    caso real eso mandaba el 78 % de las gaussianas al suelo del rango contra el 5 % que
    sale midiendo en los vértices.

    Aquí se reproduce con la malla desplazada hacia dentro: si la función aceptara puntos
    sueltos, esto saturaría. Como sólo acepta la superficie y sus normales, el error no se
    puede cometer — y el transporte a cada gaussiana es un vecino más cercano, igual que
    con la normal y el `region_id`.
    """
    import numpy as np
    from gaussian_engine.apariencia import OCLUSION_MINIMA, _oclusion_ambiental

    pos, nor = _valle()
    ao = _oclusion_ambiental(pos, nor)
    assert ao.shape == (len(pos),)
    # Transportada a puntos que NO están en la malla, sigue siendo la de la superficie.
    from scipy.spatial import cKDTree

    fuera = pos - 0.05 * nor          # medio milímetro por dentro, como una gaussiana movida
    ao_fuera = ao[cKDTree(pos).query(fuera, k=1)[1]]
    assert np.allclose(ao_fuera, ao)
    assert (ao_fuera <= OCLUSION_MINIMA + 1e-6).mean() < 0.25


def test_el_descriptor_de_la_oclusion_lleva_los_numeros_del_calculo() -> None:
    """⚠️ **Un descriptor escrito a mano se desincroniza; uno compuesto no puede.**

    El texto de `ao` estuvo describiendo el método ANTERIOR —«fracción de vecinos que caen
    por delante del plano tangente»— después de que se midiera que ese método daba ~0,5 en
    toda la superficie, dejara el 83 % de las gaussianas en el suelo del rango y se
    sustituyera por la magnitud del seno. Quien leyera el contenedor tenía la descripción
    de un cálculo que ya no se hacía.

    Es el mismo fallo que ya apareció con las unidades, con el esquema y con la nota del
    color: cuatro veces. Por eso lo que se prueba no es que el texto diga algo, sino que
    los números que dice son los que usa el código.
    """
    from gaussian_engine.agente_apariencia import ESQUEMA_INRIA
    from gaussian_engine.apariencia import (
        GANANCIA_OCLUSION,
        OCLUSION_MINIMA,
        RADIO_OCLUSION_MM,
        VECINOS_OCLUSION,
    )

    texto = ESQUEMA_INRIA["ao"]["significado"]
    for valor in (VECINOS_OCLUSION, RADIO_OCLUSION_MM, GANANCIA_OCLUSION, OCLUSION_MINIMA):
        assert f"{valor:g}" in texto, f"{valor} no aparece en el descriptor de `ao`"
    # Y no puede seguir describiendo el método que se descartó.
    assert "plano tangente" not in texto


# ── Supervisión de profundidad y reuso de corridas ───────────────────────────

def test_el_color_reutilizado_se_lee_del_PLY_de_la_corrida_origen(tmp_path):
    """`reusa` existe para comparar experimentos con identicos pixeles de entrada: el color
    por vertice se relee del `scan_colored.ply` que la corrida origen escribio, y el lector
    tiene que devolverlo igual — el dtype sale de la cabecera DECLARADA, no de una lista
    escrita aqui, que es la misma familia de fallo que ya mordio tres veces."""
    import numpy as np
    from gaussian_engine.apariencia import _lee_ply_coloreado, escribe_ply_coloreado

    rng = np.random.default_rng(3)
    pos = rng.normal(size=(50, 3))
    caras = rng.integers(0, 50, (40, 3))
    rgb = rng.integers(0, 255, (50, 3)).astype(np.uint8)
    ruta = tmp_path / "scan_colored.ply"
    escribe_ply_coloreado(ruta, pos, caras, rgb)
    leido = _lee_ply_coloreado(ruta)
    assert leido.shape == (50, 3)
    assert (leido == rgb.astype(np.float32)).all()


def test_la_profundidad_sin_sus_mapas_falla_ALTO(tmp_path):
    """Un `peso_profundidad > 0` sin `depth/z_*.npy` fallaria con un `FileNotFoundError`
    500 iteraciones tarde, dentro del bucle. Aqui se comprueba ANTES de entrenar, y contra
    el CONTEO de vistas: un `depth/` de otra corrida con distinto numero se detecta
    tambien, en vez de comparar el campo contra mapas ajenos."""
    import numpy as np
    import pytest
    from gaussian_engine.apariencia import _requisitos_profundidad

    T = {"frames": [{}] * 3}
    with pytest.raises(FileNotFoundError, match="depth"):
        _requisitos_profundidad(T, tmp_path)
    (tmp_path / "depth").mkdir()
    with pytest.raises(ValueError, match="3 vista"):
        _requisitos_profundidad(T, tmp_path)
    for k in range(3):
        np.save(tmp_path / "depth" / f"z_{k:05d}.npy", np.zeros((4, 4), np.float32))
    _requisitos_profundidad(T, tmp_path)  # no lanza


def test_la_cabecera_declara_el_color_REUTILIZADO():
    """Reusar el color de otra corrida no lo convierte en DOS TONOS ni en MEDIDO: es un
    tercer caso, y la cabecera lo dice con su nombre en vez de mentir la procedencia."""
    import numpy as np
    from gaussian_engine.apariencia import _comentarios_color

    junto = " ".join(_comentarios_color({"color_reutilizado": np.asarray(1)}))
    assert "REUTILIZADO" in junto
    assert "DOS tonos" not in junto
    assert "MEDIDO" not in junto
