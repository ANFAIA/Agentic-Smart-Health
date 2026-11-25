# Ficha del modelo FDI del escáner: C3 con filtro de terceros molares

**Congelado el 2026-11-25 como modelo del paper.** Etiqueta git: `segmentacion-fdi-c3-v1`.
Lo que sigue es lo que hace falta para citarlo, reproducirlo y no venderlo por lo que no es.
El razonamiento y cada resultado intermedio están en
[`segmentacion-fdi-escaner.md`](segmentacion-fdi-escaner.md), §6.10 a §6.23.

## 1 · Identidad

| | |
|---|---|
| Checkpoint | `data/processed/a3-checkpoints/fdi-8k-rot15-s0-gastadas.pt` (no versionado) |
| sha256 | `b0a280644590e2dc7861d3f448b8321115d459d1f38b9158d74089a3f37006ea` |
| Tamaño | 18.515.166 bytes |
| Arquitectura | Point Transformer de segmentación, la de `scripts/segmentar_fdi.py` (`Net`): entrada de normales (3 canales), 33 clases (encía + 32 FDI), dimensiones 32-64-128-256-512 |
| Entrenado desde | `3f36c4c`, árbol limpio (el `run_record` va dentro del checkpoint) |
| Orden | `entrena_fdi_escaner.py --puntos 8192 --lote 4 --epocas 60 --rotacion 15 --semilla 0 --con-reservas-gastadas` |
| Aumentación | giro ±15° sobre eje aleatorio, escala ±5 %, **nunca espejo** |
| Pérdida | NLL con pesos por frecuencia mediana, tope 50 |
| Optimización | 60 épocas, lr 1e-3, lote 4, semilla 0; ~58 s por época en una GPU de 12 GB |
| Arcadas de entrenamiento | 1.440, lista con sha256 `15b59c9939e98502b835042ae2f0e2d506479fca23e19f57c6c0952ed37bfd35` |
| Entorno | Python 3.13.9, torch 2.11.0+cu128, `torch_geometric` |

**Los pesos no están en el repositorio.** `scripts/data_guard.py` veta los pesos de modelo,
por extensión y por tamaño, y el sha256 basta para saber si un fichero es este. Para el paper
podrán publicarse aparte, por ejemplo como recurso de una release. La licencia de Teeth3DS+
(CC-BY 4.0) lo permite citando el dataset, y el entrenamiento no ha visto ningún dato clínico
propio. Publicarlos es una decisión que queda pendiente.

## 2 · El pipeline completo, tal como se mide

1. **Pose.** El marco se orienta por el borde abierto del escaneo (`O`,
   `analysis_agents.orientacion.marco_orientado`, §6.19). Después se canoniza contra las 18
   primeras arcadas de entrenamiento y se vota por vértice entre las 3 más confiadas.
2. **Modelo.** Se infiere en trozos de 8.192 puntos y la arcada contraria se enmascara.
3. **Filtro de terceros molares** (§6.23). Cada 18, 28, 38 o 48 de menos de 69 mm² vuelve a
   la encía.
4. **Limpieza.** Las cinco limpiezas de `caso_completo.py`.
5. **Margen gingival**, solo si se pide (`--margen-gingival propone|aplica`). Es `M10`
   (§6.19-6.20); por defecto, apagado.

`scripts/segmentar_fdi.py --referencias-entrenamiento 18` hace los pasos 1 a 3 por defecto.
Lo que no es por defecto se desactiva así: `--marco-canonico` quita `O`,
`--sin-filtro-terceros` quita el filtro y `--checkpoint` cambia el modelo.

## 3 · Los datos

Todo es **Teeth3DS+** (3DTeethSeg'22; Ben-Hamadou et al., 2022), con licencia CC-BY 4.0.
Ver [`dataset-teeth3ds.md`](dataset-teeth3ds.md). Son 900 pacientes etiquetados, en tres
conjuntos del reto:

| Conjunto del reto | Pacientes | Uso |
|---|---:|---|
| `public-training-set-1` | 300 | 240 entrenamiento (80 %, semilla 42); 60 para medir: `dev` (20) y `eval` (40, gastado en octubre-noviembre) |
| `public-training-set-2` | 300 | entrenamiento (desde C2) |
| `private-testing-set` | 300 | cinco reservas de 60, fijadas antes de bajarlas y evaluadas una vez cada una |

| Reserva | Semilla | Fijada | Arcadas · piezas | Se gastó en | Después |
|---|---|---|---|---|---|
| `reserva` (`dev-privado`) | 20261120 | 2026-11-20 | 120 · 1.657 | §6.10 | desarrollo; entrena C3 |
| `reserva-2` | 20261121 | 2026-11-20 | 120 · 1.672 | §6.18 | entrena C3 |
| `reserva-3` | 20261124 | 2026-11-24 | 120 · 1.648 | §6.19 | entrena C3 |
| `reserva-4` | 20261125 | 2026-11-24 | 120 · 1.635 | §6.21 | diagnóstico del filtro y puerta de §6.23 |
| `reserva-5` | (el resto) | 2026-11-24 | 120 · 1.662 | §6.23 | **evaluación final** |

Las listas están versionadas en `docs/research/teeth3ds-reserva*.txt`, y
`scripts/fetch_teeth3ds_osf.py` las deriva de la lista oficial y comprueba que coinciden.
C3 se entrenó con 720 pacientes: los 240 + 300 públicos y las tres primeras reservas.
**No vio ni `dev`, ni `reserva-4`, ni `reserva-5`.**

## 4 · Cifras finales, en `reserva-5`

Son 120 arcadas que ningún modelo había visto. Se evaluaron una vez desde `ccc4e37`, con
criterios fijados antes (§6.23): `benchmark_separacion.py seccion-6-23 --evaluacion-final`.

| | C2 (anterior) | C3 | **C3 + filtro** |
|---|---:|---:|---:|
| Piezas bien numeradas, de 1.662 | 1.554 | 1.583 | **1.583 (95,2 %)** |
| Fantasmas sobre la encía | 82 | 92 | **43** |
| Terceros molares reales bien numerados | — | 30 | **30** |
| Coronas que pasan a < 85 % | — | — | **0** |
| Contacto interproximal, mediana | 0,171 mm | — | **0,187 mm** |
| Encía marcada como diente | 0,254 | — | **0,251** |

C3 con el filtro cumple frente a C2 los cinco criterios de §6.21, y el filtro los suyos
(F1-F4). Es el primer modelo nuevo que cumple todos los criterios contra el anterior en una
reserva sin ver. Antes, en `reserva-4`, C3 sin filtro había fallado el criterio de fantasmas
(§6.21) y pasó a ser el de por defecto por desviación declarada (§6.22). Ese orden queda
escrito y no se reescribe.

## 5 · Lo que esto no es

- **No es validación clínica.** Todo está medido en Teeth3DS+: otros escáneres, otra
  población y otras poses que las de la clínica.
- **Sobre casos propios hay un solo caso clínico maxilar.** Tiene la línea de margen de una
  periodoncista a ciegas y una segunda anotación. C3 numera sus 14 piezas como la línea. Sus
  anotaciones ya se usaron para elegir métodos de margen (§6.7-6.9, §6.20), así que no miden
  nada sin sesgo.
- **Teeth3DS+ está agotado.** Cualquier cambio a partir de aquí necesita otro dataset o casos
  clínicos anotados a ciegas.
- **Las etiquetas son Capa 3 e `investigational`.** Van a `derived/`, y ninguna salida es
  un diagnóstico.
