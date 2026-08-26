

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


def _lee(ruta, n):
    import numpy as np

    b = ruta.read_bytes()
    i = b.index(b"end_header") + len(b"end_header") + 1
    dt = np.dtype([(k, "<f4") for k in (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "s0", "s1", "s2", "r0", "r1", "r2", "r3")])
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
    assert np.allclose(np.exp(a["s0"]), np.exp(-3.5) / escala, rtol=1e-3)


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
    leido = np.frombuffer(b, dtype="<i2", offset=i + 17 * 4, count=1)  # 1ª fila
    assert int(leido[0]) == 11
    # y el stride cuadra: 17 floats + un int16
    assert len(b) - i == 8 * (17 * 4 + 2)
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
