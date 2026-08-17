"""Render multivista de una malla intraoral con **Blender** (headless).

Parte del experimento `notebooks/07`: escáner intraoral (STL/OBJ) → **Blender**
renderiza N vistas con pose de cámara conocida → paquete listo para 3D Gaussian
Splatting. Sustituye al render por VTK de los notebooks 03/05: aquí el motor es
Blender (EEVEE), que da sombreado e iluminación reales en vez de un splatting
clásico.

Se ejecuta con el Python de Blender, NO con el del proyecto:

    blender --background --python scripts/blender_render_views.py -- \\
        --scan <malla.stl|.obj> --out <dir> --views 72 --res 512

Produce, en `<dir>`:
  · images/r_000.png … r_NNN.png   (vistas RGBA)
  · transforms.json                (convención NeRF-blender: camera_angle_x +
                                     transform_matrix cámara→mundo por vista)

Las poses son **exactas** (salen del propio grafo de escena de Blender), así que
no hace falta COLMAP ni structure-from-motion — igual que en los notebooks 03/05,
pero con un render fotográfico.

## Por qué la luz va a 35 grados del eje, y no en el eje

Medido sobre `histora` inferior, 24 vistas por condición, energía de gradiente dentro
de la máscara (que es cuánto relieve se ve) y fracción de píxeles aplastados a negro:

    rasante   brillo medio   |gradiente|·1e3   aplastado (<0,05)
       0°        0,540            27,90             0,5 %
      20°        0,512            30,69             1,0 %
      35°        0,463            34,38             2,1 %
      50°        0,396            35,04             3,7 %

Un frontal puro (0°) tiene n·l ≈ n·v: todo lo que mira a la cámara brilla igual y la
anatomía oclusal casi no se lee. Separar la luz gana **+23 % de relieve visible** hasta
los 35°; de ahí a 50° solo se saca un 2 % más y se paga con casi el doble de píxeles
aplastados, que es información que ya no recupera ningún entrenamiento. De ahí el
valor por defecto.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils

# --------------------------------------------------------------------------- #
# Argumentos (tras el "--" que separa los de Blender de los del script)
# --------------------------------------------------------------------------- #
argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=72)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--samples", type=int, default=16)     # muestras EEVEE por píxel
ap.add_argument("--elevations", type=int, default=3)   # anillos de la órbita
ap.add_argument("--elev-max", type=float, default=60.0)  # ± grados del anillo más alto/bajo
ap.add_argument("--fov-deg", type=float, default=40.0)
# Luz rasante: grados que se separa la luz del eje de la cámara. Ver la nota junto al
# montaje de las luces; 0 = frontal puro, que es plano.
ap.add_argument("--raking-deg", type=float, default=35.0)
ap.add_argument("--relleno", type=float, default=0.8)  # energía del relleno opuesto
ap.add_argument("--ambiente", type=float, default=0.4)  # fuerza del mundo
# Gris plano: ignora el color de vértice y usa un neutro. Sirve para mirar RELIEVE en
# vez de anatomía — sin tono que distraiga, todo el rango dinámico va a la forma.
ap.add_argument("--gris", type=float, default=None,
                help="Color base neutro (0-1). Si se da, ignora el color de vértice.")
args = ap.parse_args(argv)

out = Path(args.out)
(out / "images").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Escena limpia
# --------------------------------------------------------------------------- #
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def import_scan(path: str) -> bpy.types.Object:
    ext = Path(path).suffix.lower()
    if ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise SystemExit(f"Formato no soportado: {ext}")
    return bpy.context.selected_objects[0]


obj = import_scan(args.scan)

# --- Normalizar: centrar en el origen y escalar a caja unidad --------------- #
# Así el encuadre de la cámara no depende del tamaño real del escaneo (mm que
# varían por caso). El 3DGS trabaja en este espacio normalizado.
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
centro_bounds = tuple(obj.location)  # dónde estaba el centro antes de mandarlo al origen
obj.location = (0.0, 0.0, 0.0)
dims = obj.dimensions
scale = 2.0 / max(dims.x, dims.y, dims.z)
obj.scale = (scale, scale, scale)
bpy.context.view_layer.update()

# --- Material: color de vértice si la malla lo trae (PLY coloreado), si no,
# marfil neutro (STL/OBJ pelados). El PLY coloreado lo produce el notebook 07 al
# proyectar el color de las fotos sobre la malla (esmalte arriba, encía abajo). --- #
mat = bpy.data.materials.new("scan")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes.get("Principled BSDF")
if bsdf and "Roughness" in bsdf.inputs:
    bsdf.inputs["Roughness"].default_value = 0.55
if args.gris is not None and bsdf:
    g = args.gris
    bsdf.inputs["Base Color"].default_value = (g, g, g, 1.0)
elif obj.data.color_attributes:
    obj.data.color_attributes.active_color_index = 0
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = obj.data.color_attributes[0].name
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
elif bsdf:
    bsdf.inputs["Base Color"].default_value = (0.85, 0.80, 0.72, 1.0)  # marfil
obj.data.materials.clear()
obj.data.materials.append(mat)

# --------------------------------------------------------------------------- #
# Cámara + luz «de casco» (parented) para iluminación consistente entre vistas
# --------------------------------------------------------------------------- #
cam_data = bpy.data.cameras.new("cam")
cam_data.angle = math.radians(args.fov_deg)  # FOV horizontal
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# Las luces van EMPARENTADAS A LA CÁMARA, y esto no es un detalle de montaje: es la
# condición que hace que el 3DGS pueda aprender. Un campo de gaussianas modela el color
# como función de la DIRECCIÓN DE VISTA (armónicos esféricos); si la luz se mueve por su
# cuenta, el mismo punto visto desde el mismo sitio cambia de brillo entre fotogramas y
# el modelo no tiene forma de representarlo — lo promedia y sale lavado. Solidaria con
# la cámara, la iluminación es una función de la vista y el modelo sí la absorbe.
#
# Pero una luz EN el eje de la cámara (frontal puro, que es lo que había) es plana:
# n·l ≈ n·v, así que toda superficie que te mira brilla igual y el relieve se pierde.
# Por eso la principal se separa `--raking-deg` grados del eje. Ese es el efecto que
# busca la técnica de "pasear una luz rasante por la superficie" para sacar detalle: la
# luz oblicua alarga los gradientes y hace visible el microrrelieve de la anatomía.
# La diferencia está en QUIÉN la mueve — aquí la mueve la cámara, no un keyframe suelto.
_rk = math.radians(args.raking_deg)
light_data = bpy.data.lights.new("key", type="SUN")
light_data.energy = 3.0
light_data.angle = math.radians(3.0)  # penumbra suave: un sol puntual da bordes duros
light = bpy.data.objects.new("key", light_data)
scene.collection.objects.link(light)
light.parent = cam
light.rotation_euler = (_rk, 0.0, math.radians(25.0))  # arriba y a un lado

# Relleno por el lado contrario, flojo, para que la cara en sombra no se cierre a negro:
# lo que ahí se pierda no lo puede aprender ningún entrenamiento.
fill_data = bpy.data.lights.new("fill", type="SUN")
fill_data.energy = args.relleno
fill = bpy.data.objects.new("fill", fill_data)
scene.collection.objects.link(fill)
fill.parent = cam
fill.rotation_euler = (-_rk * 0.6, 0.0, math.radians(-140.0))

# Ambiente tenue para que las zonas en sombra no queden negras.
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = args.ambiente
scene.world = world


def look_at(camera: bpy.types.Object, target: mathutils.Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# --------------------------------------------------------------------------- #
# Motor de render (EEVEE) y ajustes
# --------------------------------------------------------------------------- #
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"   # Blender 4.2+/5.x
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = args.res
scene.render.resolution_y = args.res
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
    scene.eevee.taa_render_samples = args.samples

# --------------------------------------------------------------------------- #
# Órbita: `elevations` anillos × (views/elevations) azimuts, radio fijo
# --------------------------------------------------------------------------- #
radius = 4.0
target = mathutils.Vector((0.0, 0.0, 0.0))
n_rings = max(1, args.elevations)
per_ring = max(1, args.views // n_rings)
# Anillos repartidos uniformemente entre +elev_max y -elev_max (para miles de
# vistas hacen falta más de 3 anillos, así que se generan, no se hardcodean).
if n_rings == 1:
    elev_angles = [0.0]
else:
    step = 2.0 * args.elev_max / (n_rings - 1)
    elev_angles = [args.elev_max - i * step for i in range(n_rings)]

frames = []
idx = 0
for elev in elev_angles:
    phi = math.radians(elev)
    for a in range(per_ring):
        # desfase por anillo para que los azimuts no se apilen en la misma columna
        theta = 2.0 * math.pi * (a / per_ring) + 0.5 * math.pi * (idx % 2)
        cam.location = mathutils.Vector((
            radius * math.cos(phi) * math.cos(theta),
            radius * math.cos(phi) * math.sin(theta),
            radius * math.sin(phi),
        ))
        look_at(cam, target)
        bpy.context.view_layer.update()

        rel = f"images/r_{idx:05d}.png"  # 5 dígitos: miles de vistas ordenan bien
        scene.render.filepath = str(out / rel)
        bpy.ops.render.render(write_still=True)

        frames.append({
            "file_path": rel,
            "transform_matrix": [list(row) for row in cam.matrix_world],
        })
        idx += 1

# camera_angle_x en la convención NeRF (FOV horizontal en radianes)
transforms = {
    "camera_angle_x": float(cam_data.angle),
    "w": args.res,
    "h": args.res,
    # Reversibilidad: con estos dos, `mundo = normalizado / scan_scale + scan_offset`
    # devuelve el campo entrenado a los milímetros del escaneo. Sin el offset la vuelta
    # no se puede hacer, y un campo que no se puede devolver a mm no se puede medir.
    "scan_scale": float(scale),
    "scan_offset": [float(v) for v in centro_bounds],
    "frames": frames,
}
(out / "transforms.json").write_text(json.dumps(transforms, indent=1))
print(f"[blender] {idx} vistas → {out}")
