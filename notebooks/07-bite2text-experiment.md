# Experimento 07 · Bite2Text — escáner intraoral → Blender → 3D Gaussian Splatting

Reconstrucción de un **Digital Twin dental** a partir de un **escáner intraoral real**
(STL), reutilizando los **agentes de ingesta** del proyecto, con **Blender** como motor
de vistas y **3D Gaussian Splatting** (`gsplat`) como reconstructor. Cierra con la
serialización al contrato (`TwinSnapshot`) y un `.ply` para el visor web.

Notebook: [`07-bite2text-blender-3dgs.ipynb`](07-bite2text-blender-3dgs.ipynb).

---

## 1. Objetivo

Probar que la cadena **dato clínico real → agentes de ingesta → campo gaussiano
evaluable** funciona de punta a punta sobre una modalidad que los notebooks 03–06 no
cubrían: el **escáner intraoral (STL pelado, sin color)**, con fotos intraorales y un
informe clínico del mismo caso. Dos piezas eran fijas del encargo —motor de vistas
**Blender** (sombreado real, no *splatting* clásico) e **STL real** en vez de malla
sintética—; el resto es libre.

## 2. Datos

Dataset **Bite2Text** (UNIMORE / Univ. Ferrara), caso `F1980`, arcada inferior.

| Modalidad | Fichero | Agente |
|---|---|---|
| Escáner intraoral | `ios/ios_lower.stl` (STL binario, ~255 k vértices) | `mesh-agent` |
| 5 fotos intraorales | `intraoral-photo/intraoral_{1..5}.jpg` | `image-agent` (sin EXIF) |
| Informe clínico | `reports_ios_en/*.txt` + demo `.pdf` | `report-agent` |

Las 5 fotos son: **1** frontal, **2/3** laterales, **4** oclusal inferior, **5**
oclusal superior. Licencia del dataset: **CC-BY-SA 4.0** (los derivados heredan la
misma licencia).

## 3. Pipeline

```
STL  ──mesh-agent──►  malla al contrato (positions, faces, normals; color=None, conf 0.5)
        │
        ├─ image-agent ─►  color de las 5 fotos (§1c regional / §1d per-diente oclusal)
        │
        ▼
     Blender (EEVEE, headless)  ──►  1600 vistas @1024 px + transforms.json (pose exacta, sin COLMAP)
        │
        ▼
     gsplat  ──►  campo 3DGS (6000 iters, pérdida 0,8·L1 + 0,2·(1−SSIM), control de densidad)
        │
        ▼
     TwinSnapshot (surface_ref + gaussian_field_ref)  +  .ply INRIA para el visor web
```

Detalle de cámara: la pose de Blender (OpenGL) se convierte a la convención OpenCV de
gsplat con `diag(1,−1,−1,1)`. El campo se siembra desde la **misma malla** del
`mesh-agent`, con la **misma normalización** que aplicó Blender.

## 4. Resultados

- **Reconstrucción:** holdout **PSNR 31,5 dB** (1200 train / 400 holdout, medido sobre
  24 retenidas), 130 485 gaussianas finales tras poda. Entrenamiento ~188 s en RTX 5070.
- **SSIM en la pérdida** (vs L1 sola): bordes de los dientes más nítidos, menos borroso.
- **Neblina / "pelillos":** el campo es mate y de baja opacidad (óptimo con L1 sobre
  fondo negro). Medido renderizando: **cortar por opacidad no desmantela el campo** —la
  superficie la sostiene la minoría opaca—, así que el visor descarta la neblina con un
  **umbral de alfa ≈ 8/255** y los bordes quedan limpios.
- **Artefactos:** `.ply` INRIA (grado 0, color RGB plano) para
  [`dental-3dgs-viewer`](../../dental-3dgs-viewer) + `TwinSnapshot` v1.2.0 trazable por
  hash en el `ArtifactStore`.

## 5. El problema del color (el arco completo)

El STL viene **pelado**. Recuperar color del paciente tiene varios niveles, y este
experimento recorrió los baratos siendo explícito sobre dónde está el techo:

| Nivel | Qué hace | ¿Color real? | Coste | En el notebook |
|---|---|---|---|---|
| **§1c · regional** | 2 tonos (esmalte/encía) de las 5 fotos, aplicados por altura z | del paciente, **plano por zona** | bajo | **por defecto** |
| **§1d · per-diente (Vía A)** | proyecta la foto **oclusal** sobre las coronas, muestrea por diente | **per-diente real** en las coronas | medio | opcional (`*_occlusal.ply`) |
| Vía B · multi-vista | color por normal desde las 4 fotos útiles (oclusal + vestibular) | per-diente en (casi) todas las caras | alto | **no** — es fusión |
| Per-píxel | textura exacta foto↔malla | máximo | muy alto | **no** — es fusión |

### Hallazgo medido (§1d)

La proyección oclusal alinea el **anillo de coronas** de la malla con el **anillo de
dientes** de la foto por **PCA + mejor volteo (IoU) + ICP 2D**. El **ICP no baja del
IoU ≈ 0,55**: el error residual es **no-rígido** (la perspectiva de una foto intraoral
sin calibrar, con retractor), y una transformación rígida+escala no lo corrige. Da color
per-diente **real**, pero con techo, y sale más oscuro y con motas (sombras entre
dientes) — por eso **no sustituye** al degradado como render por defecto.

**Conclusión honesta:** el color per-diente *sin fusión* es posible pero limitado. El
salto de calidad de verdad —caras vestibulares, mejor luz, sin costuras entre vistas—
es **multi-vista con registro real**, que **es** el problema de fusión geométrica, no un
truco barato.

## 6. Hallazgos secundarios (sobre los agentes)

- **Ontología:** los informes de Bite2Text son **ortodónticos** (clases de Angle,
  curvas de Spee/Wilson, apiñamiento), no cariológicos. La ontología mínima modela
  **pH**, así que el `report-agent` devuelve **0 hallazgos con confianza 0 → HITL** en
  vez de inventar. Correcto, no un bug: pide **ampliar la ontología** o usar el backend
  LLM. El camino `.pdf` (con un pH) sí produce `RegionalObservation`, demostrando lectura
  de PDF además de texto.
- **Contrato:** Bite2Text **no tiene CBCT**, y el `IngestionPipeline` exige
  `gaussian_field_ref` del CBCT. Aquí el campo lo produce el **3DGS entrenado desde la
  malla** → vía de twin **mesh-only** que el pipeline actual no contempla.

## 7. Siguiente paso

El color **per-diente multi-vista** (Vía B) es el candidato natural al **primer caso del
`fusion-agent`**: `F1980` tiene malla + 4 vistas del mismo paciente, el dato ideal para
arrancar el spike de fusión geométrica de la semana 5–6.

---

### Reproducir

Requiere el kernel **Dental GPU (3DGS)** (torch cu128 + gsplat) y **Blender 5.x** en el
PATH; los agentes de ingesta instalados en ese venv (`pip install -e packages/...`). El
dataset Bite2Text va **fuera del repo** (`~/anfaia/Bite2Text`). Ejecutar el notebook de
arriba abajo; la celda **§1d es opcional** (no afecta al render por defecto).
