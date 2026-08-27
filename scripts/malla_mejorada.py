"""malla_mejorada.py — El STL mejorado, sacado del contenedor y de nada más.

    uv run python scripts/malla_mejorada.py \
        --uos data/processed/caso-completo/export/<caso>/<caso>.uos

**Qué demuestra.** Un STL de escáner lleva triángulos y nada más: ni color, ni unidades, ni
procedencia. Este script abre un `.uos`, lee la geometría que entró y el campo gaussiano de
apariencia que se entrenó encima, y devuelve la misma arcada **con el color medido del
paciente**, en tres formatos que un laboratorio puede abrir.

⚠️ **No mira ni un fichero intermedio del pipeline, y eso es el punto.** El color existe
antes en `pinta_malla`, un paso anterior de la misma cadena. Leerlo de ahí sería más fácil y
no demostraría nada: lo que se entrega es el contenedor, y quien lo reciba dentro de un año
no tendrá este repositorio ni aquellos ficheros. Si la malla mejorada no sale del `.uos`,
el campo gaussiano no está aportando nada y la reversibilidad no cierra.

⚠️ **Todo lo que hace falta para leer el campo se le PREGUNTA al contenedor.** La opacidad
va en logit y las escalas en logaritmo, pero eso no se escribe aquí como constante: se lee
de `scene/appearance.gs.json`, que es donde el emisor lo declara columna a columna. Un
lector que suponga la convención se rompe en silencio el día que el emisor cambie, que es
exactamente lo que este proyecto no quiere que pase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
for src in sorted(RAIZ.glob("packages/*/src")):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from export_agents.malla_mejorada import (  # noqa: E402
    NEUTRO,
    color_desde_gaussianas,
    escribe_3mf,
    escribe_ply,
    escribe_stl_viscam,
    rellena_huecos,
)


def lee_stl(datos: bytes) -> tuple[np.ndarray, np.ndarray]:
    """STL binario a `(vertices unicos, triangulos)`.

    ⚠️ **Se deduplican los vértices.** Un STL repite cada punto una vez por triángulo que lo
    toca; sin unir, el color se calcularía tres o cuatro veces para el mismo punto y el PLY
    saldría con seis veces más vértices de los que tiene la superficie.
    """
    n = struct.unpack("<I", datos[80:84])[0]
    tri = np.frombuffer(datos, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                         ("a", "<u2")]), count=n, offset=84)
    pos, inv = np.unique(tri["v"].reshape(-1, 3), axis=0, return_inverse=True)
    return pos.astype(np.float64), np.asarray(inv).reshape(-1, 3).astype(np.int32)


def lee_apariencia(datos: bytes, esquema: dict) -> dict[str, np.ndarray]:
    """El PLY de apariencia a columnas, con la convención que declara `esquema`.

    ⚠️ El `dtype` sale de las propiedades DECLARADAS en la cabecera, no de una lista escrita
    aquí. Este mismo lector se rompió dos veces por llevar la lista a mano: la primera al
    añadirse `f_rest_*` y la segunda al añadirse `ao`.
    """
    fin = datos.index(b"end_header\n") + len(b"end_header\n")
    cabecera = datos[:fin].decode("ascii").splitlines()
    tipos = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "short": "<i2"}
    campos, n = [], 0
    for linea in cabecera:
        if linea.startswith("element vertex"):
            n = int(linea.split()[-1])
        elif linea.startswith("property "):
            _, tipo, nombre = linea.split()
            campos.append((nombre, tipos[tipo]))
    v = np.frombuffer(datos, np.dtype(campos), count=n, offset=fin)
    col = {nombre: np.asarray(v[nombre], np.float64) for nombre, _ in campos}

    # ⚠️ Se recorre lo que el descriptor DECLARA, columna a columna, y no una lista de
    # nombres escrita aqui. La primera version transformaba `scale_0`, `scale_1` y
    # `scale_2` porque «las escalas son tres», y reventaba con un `KeyError` en cuanto el
    # descriptor traia una sola. Que columnas hay lo dice el fichero.
    for c in esquema.get("columns", []):
        nombre, unidad = c["name"], c.get("unit") or ""
        if nombre not in col:
            continue
        if unidad == "logit":
            col[nombre] = 1.0 / (1.0 + np.exp(-col[nombre]))
        elif unidad.startswith("log"):
            col[nombre] = np.exp(col[nombre])
    return col


def _sha256_declarado(manifiesto: dict) -> str | None:
    """El `sha256` que el manifiesto le asigna a `asset.ios`, buscándolo donde esté.

    ⚠️ Se busca por `id`, no por posición: la forma del manifiesto es del emisor y este
    script tiene que seguir leyendo contenedores de otras versiones.
    """
    pila: list = [manifiesto]
    while pila:
        o = pila.pop()
        if isinstance(o, dict):
            if o.get("id") == "asset.ios" and o.get("sha256"):
                return str(o["sha256"])
            pila.extend(o.values())
        elif isinstance(o, list):
            pila.extend(o)
    return None


# Fraccion minima de aristas cuyos dos extremos comparten etiqueta para admitir que las
# etiquetas estan en el orden de esta malla.
#
# ⚠️ **Un reordenado de vertices no se ve, y ya invalido una medicion entera en este
# proyecto.** `derived/seg_teeth` indexa «los vertices de `asset.scene` en el mismo orden en
# que la escena los declara», y este script deduplica el STL por su cuenta: si los dos
# ordenes no coinciden, cada vertice recibe la etiqueta de otro y el fichero sale con los
# codigos FDI barajados sin que nada falle. Un diente es CONTIGUO, asi que la prueba es
# barata: medido sobre el caso real, el orden correcto da 0,959 y el barajado 0,117.
ACUERDO_MINIMO_ETIQUETAS = 0.70


def etiquetas_alineadas(bruto: bytes, meta: dict, n: int,
                        caras: np.ndarray) -> np.ndarray | None:
    """Las etiquetas FDI por vertice, o `None` si no se puede afirmar que sean de esta malla.

    Ver `ACUERDO_MINIMO_ETIQUETAS`. Devolver `None` y decirlo es correcto; escribir codigos
    que quiza esten desplazados un vertice no lo es.
    """
    dtype = meta.get("encoding", {}).get("dtype", "int16-le")
    if dtype != "int16-le":
        return None
    etq = np.frombuffer(bruto, "<i2")
    if len(etq) != n:
        return None
    aristas = np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]])
    acuerdo = float((etq[aristas[:, 0]] == etq[aristas[:, 1]]).mean())
    return etq if acuerdo >= ACUERDO_MINIMO_ETIQUETAS else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uos", type=Path, required=True)
    ap.add_argument(
        "--malla", type=Path, default=None,
        help="El STL del escáner, cuando el contenedor es de perfil ligero y solo declara "
             "su `sha256` en vez de llevarlo dentro. ⚠️ Se COMPRUEBA contra ese `sha256`: "
             "un fichero que no sea exactamente el que se ingirió se rechaza, porque "
             "mejorar una malla que no es la del caso produce un fichero plausible y "
             "equivocado.",
    )
    ap.add_argument("--salida", type=Path, default=None,
                    help="Por defecto, un directorio `mejorado/` junto al `.uos`.")
    args = ap.parse_args()

    salida = args.salida or args.uos.parent / "mejorado"
    salida.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.uos) as z:
        dentro = set(z.namelist())
        malla = next((n for n in dentro if n.startswith("scene/scan.")), None)
        if malla is not None:
            crudo_malla = z.read(malla)
        elif args.malla is not None:
            crudo_malla = args.malla.read_bytes()
            esperado = _sha256_declarado(json.loads(z.read("manifest.json")))
            visto = hashlib.sha256(crudo_malla).hexdigest()
            if esperado is None:
                print("✗ El manifiesto no declara el `sha256` del escáner, así que no hay "
                      "contra qué comprobar el fichero aportado.")
                return 1
            if visto != esperado:
                print(f"✗ `--malla` no es la malla de este caso: el manifiesto declara "
                      f"sha256 {esperado[:16]}… y el fichero da {visto[:16]}….")
                return 1
            print(f"  malla aportada, sha256 {visto[:16]}… verificado contra el manifiesto")
        else:
            print("✗ El contenedor no lleva la malla del escáner dentro (perfil ligero: "
                  "`--sin-originales` o `--solo-gaussianas`). El manifiesto declara su "
                  "sha256: pásala con `--malla` y se comprobará contra él.")
            return 1
        if "scene/appearance.ply" not in dentro:
            print("✗ El contenedor no lleva campo de apariencia: se generó sin "
                  "`--entrena-apariencia`, así que no hay color medido que transferir.")
            return 1
        pos, caras = lee_stl(crudo_malla)
        esquema = json.loads(z.read("scene/appearance.gs.json"))
        ap_col = lee_apariencia(z.read("scene/appearance.ply"), esquema)
        clinico = (json.loads(z.read("clinical/observations.json"))
                   if "clinical/observations.json" in dentro else {"teeth": []})
        seg = (z.read("derived/seg_teeth.bin"),
               json.loads(z.read("derived/seg_teeth.meta.json"))) \
            if {"derived/seg_teeth.bin", "derived/seg_teeth.meta.json"} <= dentro else None

    centros = np.stack([ap_col["x"], ap_col["y"], ap_col["z"]], 1)
    f_dc = np.stack([ap_col["f_dc_0"], ap_col["f_dc_1"], ap_col["f_dc_2"]], 1)
    # ⚠️ **La media de los tres ejes, y es una aproximación declarada.** Las gaussianas son
    # anisótropas: el elipsoide se aplasta contra la superficie. Para pesar la mezcla hace
    # falta un radio, y usar el del eje que apunta al vértice exigiría rotar cada elipsoide
    # por su cuaternión. La media basta porque el peso sólo ordena vecinos que ya están a
    # menos de 3 mm — y el color resultante se contrasta contra el tono declarado por pieza,
    # que es una medida independiente.
    escalas = np.stack([ap_col["scale_0"], ap_col["scale_1"], ap_col["scale_2"]], 1).mean(1)
    rgb, medido = color_desde_gaussianas(pos, centros, f_dc, ap_col["opacity"], escalas)

    fdi = None if seg is None else etiquetas_alineadas(seg[0], seg[1], len(pos), caras)
    if seg is not None and fdi is None:
        print("  ⚠ las etiquetas FDI del contenedor no se pueden alinear con esta malla: "
              "la columna `fdi` del PLY va a 0 en vez de llevar codigos que quiza esten "
              "desplazados")
    piezas = [t for t in clinico.get("teeth", []) if t.get("color")]

    # ⚠️ **Una pieza que el contenedor NO declara con color no puede salir marcada como
    # medida.** El campo la pinta igual —con el degradado de respaldo, que no es color de
    # nadie— y esas gaussianas cubren la superficie perfectamente, así que la prueba de
    # cobertura dice «sí» y el fichero afirmaría un color inventado. Sobre este caso eran
    # 2.830 vértices, todos del FDI 17: la única pieza que ninguna foto ve con su eje
    # cuello-borde, y el gate del emisor ya lo declara. Quien manda es lo declarado.
    if fdi is not None:
        con_color = {int(t["fdi"]) for t in piezas}
        sin_declarar = (fdi > 0) & ~np.isin(fdi, list(con_color) or [-1])
        if sin_declarar.any():
            rgb[sin_declarar] = NEUTRO
            medido = medido & ~sin_declarar
            print(f"  {int(sin_declarar.sum()):,} vertices de pieza(s) sin color declarado "
                  f"({', '.join(str(f) for f in sorted(set(fdi[sin_declarar].tolist())))}) "
                  "van en gris: el campo las pinta con el degradado de respaldo")
    comentarios = [
        f"arcada del escaner con el color del campo de apariencia de {args.uos.name}",
        f"{len(pos)} vertices, {100 * medido.mean():.1f} % con color medido del paciente",
        f"{len(piezas)} corona(s) con color declarado en clinical/observations.json",
        "la columna `medido` vale 0 donde el color NO es del paciente: ahi va gris neutro",
        (f"la columna `fdi` lleva {int((fdi > 0).sum())} vertices etiquetados por"
         " derived/seg_teeth, con el orden comprobado contra la malla"
         if fdi is not None else
         "la columna `fdi` va a 0: el contenedor no trae segmentacion alineable"),
        "SIN oclusion ambiental: es un factor de visualizacion y no entra en un fichero"
        " que alguien puede imprimir o medir",
    ]
    # Los huecos sueltos, DESPUÉS de apagar las piezas sin color declarado: así el FDI 17
    # no se rellena desde sus vecinas y se queda en gris, que es lo que hay que decir.
    codigos = np.zeros(len(pos), np.int16) if fdi is None else fdi
    antes = int((~medido).sum())
    rgb = rellena_huecos(caras, rgb, medido, codigos)
    print(f"  {antes - int((~medido & (rgb == NEUTRO).all(1)).sum()):,} hueco(s) suelto(s) "
          "rellenados desde sus vecinos medidos de la misma pieza (siguen con `medido` a 0)")

    escribe_ply(salida / "arcada-color.ply", pos, caras, rgb,
                codigos, comentarios, medido)
    escribe_3mf(salida / "arcada-color.3mf", pos, caras, rgb, " · ".join(comentarios[:3]))
    escribe_stl_viscam(salida / "arcada-color.stl", pos, caras, rgb,
                       "arcada con color medido (RGB555 VisCAM, NO estandar)")

    print(f"  {len(pos):,} vertices · {len(caras):,} triangulos")
    print(f"  color medido en {100 * medido.mean():.1f} % de los vertices")
    if fdi is not None:
        print(f"  {int((fdi > 0).sum()):,} vertices con codigo FDI, orden comprobado")
    for nombre in ("arcada-color.ply", "arcada-color.3mf", "arcada-color.stl"):
        f = salida / nombre
        print(f"  → {f}  ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
