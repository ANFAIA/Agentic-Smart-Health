#!/usr/bin/env bash
# resumen-en: Trains a second C2 seed and a 2,048-point C2 and scores both against the baseline (section 6.15).
# §6.15 de docs/research/segmentacion-fdi-escaner.md: una segunda semilla de C2 (C2-s1) y C2
# con trozos de 2.048 puntos (D), medidos contra B en `dev` y en la reserva gastada.
#
#     systemd-inhibit --what=sleep:idle --why="§6.15" bash scripts/entrena_6_15.sh
#
# Desde un checkout limpio que contenga el pre-registro (5886fb6). Es idempotente: si se corta,
# relanzarlo sigue donde iba (los entrenamientos con --reanuda, las predicciones saltan lo
# hecho, y una evaluación ya escrita no se repite). Todo proceso pesado va en su propio scope
# con techo de memoria. La reserva nueva (`reserva-2`) y los casos clínicos no se tocan.
set -uo pipefail
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HOME/.venvs/dental-gpu/bin/python}
CK=$RAIZ/data/processed/a3-checkpoints
SEP=$RAIZ/data/processed/benchmark-separacion
LOG=$SEP/seccion-6-15.log
PREREGISTRO=5886fb6
cd "$RAIZ"
mkdir -p "$SEP"
paso() { echo "$(date '+%F %T') · $*" | tee -a "$LOG"; }
falla() { paso "FALLO: $*"; exit 1; }
acotado() {  # $1 techo de memoria, el resto la orden
  local techo=$1; shift
  systemd-run --user --scope -q -p MemoryMax="$techo" "$@"
}

[ -z "$(git status --porcelain)" ] || falla "el árbol no está limpio"
git merge-base --is-ancestor "$PREREGISTRO" HEAD || falla "este checkout no tiene el pre-registro $PREREGISTRO"
[ -f "$CK/fdi-8k-rot15-s0.pt" ] || falla "falta C2 ($CK/fdi-8k-rot15-s0.pt)"
paso "inicio en $(git rev-parse --short HEAD)"

C2=$CK/fdi-8k-rot15-s0.pt
S1=$CK/fdi-8k-rot15-s1.pt
D=$CK/fdi-2k-rot15-s0.pt

entrena() {  # $1 salida  $2 puntos  $3 semilla
  [ -f "$1" ] && { paso "entrenamiento $(basename "$1") ya hecho"; return 0; }
  paso "entrena $(basename "$1") ($2 puntos, ±15°, semilla $3)"
  acotado 12G "$PY" -u scripts/entrena_fdi_escaner.py --puntos "$2" --lote 4 --epocas 60 \
      --rotacion 15 --semilla "$3" --salida "$1" --reanuda >> "${1%.pt}.log" 2>&1 \
      || falla "entrena $(basename "$1")"
}
predice() {  # $1 checkpoint  $2 split
  paso "predice $(basename "$1") en $2"
  acotado 10G "$PY" -u scripts/benchmark_separacion.py predice --split "$2" --checkpoint "$1" \
      >> "$SEP/predice-$(basename "$1" .pt)-$2.log" 2>&1 || falla "predice $(basename "$1") $2"
}
evalua() {  # $1 checkpoint  $2 split
  local sha; sha=$(sha256sum "$1" | cut -c1-12)
  [ -f "$SEP/resumen-$2-$sha.json" ] && { paso "evaluación de $(basename "$1") en $2 ya hecha"; return 0; }
  paso "evalúa $(basename "$1") en $2"
  acotado 16G uv run python scripts/benchmark_separacion.py evalua --split "$2" --candidato "$1" \
      --trabajos 6 >> "$LOG" 2>&1 || falla "evalua $(basename "$1") $2"
}
mide() {  # $1 checkpoint
  predice "$1" dev && evalua "$1" dev && predice "$1" reserva && evalua "$1" reserva
}

# 1 · Los dos entrenamientos, uno detrás de otro: en paralelo con las predicciones ahorraría
#     unos 40 minutos, pero un fallo dejaría un entrenamiento huérfano que el relanzamiento
#     duplicaría. 2 · Cada modelo se predice y se evalúa en `dev` y en la reserva gastada.
entrena "$S1" 8192 1
entrena "$D" 2048 0
mide "$S1"
mide "$D"

# 3 · La lectura fijada en §6.15.
for split in dev reserva; do
  paso "compara en $split"
  uv run python scripts/benchmark_separacion.py compara --split "$split" \
      --c2 "$C2" --c2-s1 "$S1" --d "$D" >> "$LOG" 2>&1 || falla "compara $split"
done
paso "FIN: $SEP/comparacion-6-15-dev.md y comparacion-6-15-reserva.md"
