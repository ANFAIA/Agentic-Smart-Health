# Control adaptativo de densidad en 3D Gaussian Splatting

| | |
|---|---|
| **Qué es** | La técnica que hace que el 3DGS **añada, quite y reajuste** gaussianas durante el entrenamiento |
| **Origen** | Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023 |
| **Implementación** | `gsplat.DefaultStrategy` (la que usan los notebooks 04/06 y, tras este documento, el 07) |
| **Relacionado** | [`notebooks/README.md`](../../notebooks/README.md) · [`scripts/ablacion_recetas.py`](../../scripts/ablacion_recetas.py) |

> Nota de investigación. Explica **la técnica**, no una decisión de arquitectura.
> Nace de un hallazgo medido en el notebook 07: un campo entrenado **sin** esta
> técnica sale «neblinoso» (opacidad mediana ~0,01) y un visor estándar lo poda
> hasta dejarlo casi vacío. Aquí se documenta por qué, y qué hace la técnica.

## 1. El problema

El descenso de gradiente solo sabe **mover y redimensionar** las gaussianas que ya
existen — nunca **añadir ni quitar**. Pero el conjunto inicial (puntos SfM, o en
nuestro pipeline la superficie de la malla) casi nunca es el óptimo: zonas con poco
detalle, gaussianas inútiles, otras enormes.

Hace falta un mecanismo **aparte** del optimizador que controle el **número y la
calidad** de las gaussianas. Eso es el *control adaptativo de densidad*. Corre
periódicamente (cada ~100 iteraciones, en una ventana `[refine_start, refine_stop]`)
y hace tres cosas.

## 2. Las tres operaciones

### 2.1 Densificar (crecer)

- **Señal:** el gradiente de la **posición 2D proyectada** (`means2d`), promediado
  en las últimas iteraciones. Si el optimizador «quiere mover mucho» una gaussiana,
  es que está a caballo de un detalle que no puede representar → candidata a crecer.
- **Clonar:** gaussiana **pequeña** con gradiente alto (falta cobertura) → se
  duplica y se desplaza a lo largo del gradiente. *Más cobertura.*
- **Dividir (split):** gaussiana **grande** con gradiente alto (un blob cubriendo
  detalle variado) → se sustituye por dos más pequeñas (escala ÷ ~1,6), muestreadas
  de la original. *Más detalle fino.*

### 2.2 Podar (quitar)

Elimina gaussianas:
- con **opacidad por debajo de un umbral** (`prune_opa`): casi transparentes, no aportan;
- **demasiado grandes** en mundo o en pantalla: *bloat*.

### 2.3 Reset de opacidad — la pieza clave

Cada `reset_every` iteraciones se **pone a casi cero la opacidad de TODAS** las gaussianas.

¿Por qué tan drástico? Porque densificar no para de **añadir**, y se acumulan
gaussianas redundantes casi transparentes que **flotan** y nunca se podan (su
gradiente es bajo, nadie las toca). El reset obliga a **cada** gaussiana a volver a
subir su opacidad por gradiente: las que de verdad contribuyen la recuperan; las
inútiles se quedan bajas y **caen en la siguiente poda**. Es un «demuestra que
sirves o mueres» periódico.

## 3. Cómo encajan (el bucle)

```
densificar  →  más gaussianas donde hace falta
    │            (riesgo: bloat + flotantes redundantes)
    ▼
reset opa.  →  todas deben re-justificarse por gradiente
    │
    ▼
podar       →  fuera las que no se re-justificaron (+ las gigantes)
    │
    └──────►  el nº de gaussianas se autorregula: «las justas, todas útiles»
```

## 4. La dinámica de la opacidad (por qué importa tanto)

- **Con** la técnica, **en escenas fotográficas con superficies opacas** → la
  opacidad se empuja a ser **bimodal**: o opaca de verdad (sobrevivió) o eliminada.
  Campo limpio que funciona en cualquier visor al umbral de alfa estándar. *(Ojo: si
  el objeto es mate y la pérdida es L1, la baja opacidad puede ser el óptimo y la
  técnica NO la sube — ver §7.)*
- **Sin** la técnica → la opacidad deriva a lo que minimice la pérdida, que suele
  ser un **caldo de gaussianas de baja opacidad solapadas** («neblina»). Nada las
  obliga a ser opacas ni a morir. Renderiza bien *si se componen todas*, pero
  cualquier poda por umbral lo rompe.

> **Medido (notebook 07, campo sin la técnica):** opacidad mediana **0,011**, con
> el **69 %** de las gaussianas por debajo del umbral 5/255 de un visor estándar →
> el visor lo dejaba casi vacío. El síntoma exacto de no tener reset + poda.

## 5. Los mandos (`DefaultStrategy`)

| Parámetro | Qué controla | Defecto |
|---|---|---|
| `grow_grad2d` | umbral de gradiente 2D para densificar (más bajo = crece más) | `0.0002` |
| `prune_opa` | opacidad por debajo → podar | `0.005` |
| `grow_scale3d` / `prune_scale3d` | umbrales de tamaño para dividir / podar | — |
| `refine_start_iter` / `refine_stop_iter` | ventana en la que se densifica (empieza pronto; para antes del final para que asiente) | `500` / `15000` |
| `reset_every` | cada cuántas iters el reset de opacidad | `3000` |
| `refine_every` | cada cuántas iters corre el paso crecer/podar | `100` |

La API pide **un optimizador por parámetro** (`means`, `scales`, `quats`,
`opacities`, más el color), porque al densificar/podar añade y quita **filas** de
cada tensor y del estado de Adam a la vez.

## 6. La intuición

Un escultor que **añade barro** donde la forma pide detalle (densificar), cada
cierto tiempo **alisa la superficie** para ver qué trozos hacen trabajo de verdad
(reset), y **raspa** los que no (podar) — repitiendo hasta «el material justo, todo
con sentido».

## 7. El matiz para nuestro pipeline (mesh → 3DGS)

La técnica se diseñó para **init disperso** (SfM, geometría desconocida) y para
escenas **fotográficas con superficies opacas**. Nuestro caso es doblemente
distinto: partimos de la **malla del `mesh-agent`** (densa, no dispersa) y el objeto
es **mate y gris** (STL pelado, render L1 sobre fondo negro). Medido en el notebook 07:

- **Densificar → aporta poco**: con el umbral por defecto apenas crece; si se
  fuerza, mete gaussianas espurias. El init ya cubre la superficie.
- **Poda → limpia algo** pero poco: solo quita las gaussianas por debajo de
  `prune_opa` (0,005). Bajó el campo de 150k a **139k** (~7%).
- **Reset de opacidad → NO arregla la neblina aquí.** ⚠ Contra lo esperado, la
  opacidad mediana pasó de **0,011 a 0,013** (sigue el ~64% bajo el umbral de un
  visor). La razón: en un objeto mate con pérdida L1, la **baja opacidad es el
  óptimo real** —el modelo apila gaussianas semitransparentes y el L1 no lo
  penaliza—, así que tras cada reset el optimizador vuelve a la misma solución.

**Conclusión honesta:** el control adaptativo de densidad es la técnica de
referencia y se deja activo en el notebook 07, pero en *este* tipo de campo (mate,
L1) **no** produce el campo opaco que produce en escenas fotográficas. Para forzar
opacidad haría falta un **regularizador de opacidad explícito** (penalizar la baja
opacidad / empujar hacia binaria), que **no** forma parte de `DefaultStrategy`.
Mientras tanto, el visor carga este campo con umbral de alfa 0.

## 8. Referencias

- Kerbl, Kopanas, Leimkühler, Drettakis — *3D Gaussian Splatting for Real-Time
  Radiance Field Rendering*, SIGGRAPH 2023 (§5, *Adaptive Control of Gaussians*).
- [`gsplat`](https://github.com/nerfstudio-project/gsplat) — `DefaultStrategy`.
- Uso en el repo: `notebooks/07-bite2text-blender-3dgs.ipynb` (§4) y la ablación de
  receta en `scripts/ablacion_recetas.py`.
