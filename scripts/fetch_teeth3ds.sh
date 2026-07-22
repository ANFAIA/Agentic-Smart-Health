#!/usr/bin/env bash
#
# fetch_teeth3ds.sh — Descarga reproducible de Teeth3DS+ desde el Google Drive oficial.
#
# Baja las mallas (.obj) y sus labels (.json) y las deja en data/raw/teeth3ds/
# (directorio gitignored), en los DOS árboles paralelos que espera el código.
#
#   (por defecto)  dataset COMPLETO: los zips publicados (1ª mitad + _b2) suman
#                  300 pacientes / 600 escaneos / ~7,3 GiB. Es lo que consumen los
#                  notebooks 01-04 y el ejercicio de segmentación.
#   --subset       solo los 12 pacientes del PoC original (~300 MiB), para una
#                  prueba rápida o si no hay disco.
#
# Idempotente: si ya hay casos emparejados en destino, no vuelve a bajar nada
# (usa --force para re-extraer).
#
# Dataset: Teeth3DS+ (MICCAI 3DTeethSeg'22). Licencia: CC-BY 4.0 (Ben-Hamadou
# et al., 2022) — atribuir en cualquier derivado. Ver docs/research/dataset-teeth3ds.md
#
# Requisitos: gdown (`uv pip install gdown` o `pip install gdown`), unzip.
# Uso:        ./scripts/fetch_teeth3ds.sh [--subset] [--force]
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
# Carpeta oficial de Google Drive con los zips del reto 3DTeethSeg'22.
GDRIVE_FOLDER="https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby"

# Los zips publicados: el par base y su continuación (_b2). Juntos son el dataset.
MESH_ZIPS=("3D_scans_per_patient_obj_files.zip" "3D_scans_per_patient_obj_files_b2.zip")
LABEL_ZIPS=("ground-truth_labels_instances.zip" "ground-truth_labels_instances_b2.zip")

# Nombres de las carpetas raíz DENTRO de cada zip (= árboles en disco).
MESH_DIR="3D_scans_per_patient_obj_files"
LABEL_DIR="ground-truth_labels_instances"

# Subconjunto de --subset: los 12 pacientes con los que se hizo el PoC original.
PATIENT_IDS=(
  0EJBIPTC 0JN50XQR 0TMOBYXS
  01A6GW4A 01A6H4PZ 01A6HAN6 01A6HE9H 01A6HG3N
  01A91JH6 01A9282X 01ADUNMV 01ADYT70
)

MODE="full"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset) MODE="subset" ;;
    --force)  FORCE=1 ;;
    -h|--help) sed -n '3,21p' "$0"; exit 0 ;;
    *) echo "ERROR: opción desconocida '$1' (usa --help)"; exit 1 ;;
  esac
  shift
done

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DEST="$REPO_ROOT/data/raw/teeth3ds"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --------------------------------------------------------------------------- #
# Emparejado: un .obj SIN su .json no es un caso (misma regla que list_cases()
# en los notebooks). Devuelve "casos huérfanos".
# --------------------------------------------------------------------------- #
count_pairs() {
  local pairs=0 orphans=0 obj base id
  while IFS= read -r obj; do
    base="$(basename "$obj" .obj)"
    id="${base%_*}"
    if [[ -f "$DEST/$LABEL_DIR/$id/$base.json" ]]; then
      pairs=$((pairs + 1))
    else
      orphans=$((orphans + 1))
    fi
  done < <(find "$DEST/$MESH_DIR" -name '*.obj' 2>/dev/null)
  echo "$pairs $orphans"
}

# --------------------------------------------------------------------------- #
# ¿Ya está? (idempotencia — no requiere herramientas de descarga)
# --------------------------------------------------------------------------- #
echo "==> Destino: $DEST  ·  modo: $MODE"
read -r have_pairs _ <<<"$(count_pairs)"

if [[ "$MODE" == "subset" ]]; then
  need_download=0
  for id in "${PATIENT_IDS[@]}"; do
    for jaw in upper lower; do
      [[ -f "$DEST/$MESH_DIR/$id/${id}_${jaw}.obj"   ]] || need_download=1
      [[ -f "$DEST/$LABEL_DIR/$id/${id}_${jaw}.json" ]] || need_download=1
    done
  done
  if [[ "$need_download" -eq 0 && "$FORCE" -eq 0 ]]; then
    echo "==> Los 12 casos del subconjunto ya están presentes. Nada que hacer."
    exit 0
  fi
elif [[ "$have_pairs" -gt 0 && "$FORCE" -eq 0 ]]; then
  echo "==> Ya hay $have_pairs casos emparejados en destino. Nada que hacer (--force para re-extraer)."
  exit 0
fi

# --------------------------------------------------------------------------- #
# Comprobaciones (solo si hay que descargar)
# --------------------------------------------------------------------------- #
command -v gdown >/dev/null || { echo "ERROR: falta gdown. Instala: uv pip install gdown"; exit 1; }
command -v unzip >/dev/null || { echo "ERROR: falta unzip."; exit 1; }
mkdir -p "$DEST/$MESH_DIR" "$DEST/$LABEL_DIR"

# --------------------------------------------------------------------------- #
# Descarga (la carpeta entera; luego usamos los zips que toquen)
# --------------------------------------------------------------------------- #
echo "==> Descargando zips desde Google Drive (son varios GB; tarda)..."
gdown --folder "$GDRIVE_FOLDER" -O "$TMP" --remaining-ok

find_zip() { find "$TMP" -name "$1" | head -1; }

# --------------------------------------------------------------------------- #
# Extracción
# --------------------------------------------------------------------------- #
extract() {  # $1 = ruta del zip, $2 = carpeta raíz dentro del zip
  local zip="$1" root="$2" id
  if [[ "$MODE" == "subset" ]]; then
    for id in "${PATIENT_IDS[@]}"; do
      unzip -o -q "$zip" "$root/$id/*" -d "$DEST" 2>/dev/null || true
    done
  else
    unzip -o -q "$zip" -d "$DEST"
  fi
}

found_any=0
for i in "${!MESH_ZIPS[@]}"; do
  mesh_zip="$(find_zip "${MESH_ZIPS[$i]}")"
  label_zip="$(find_zip "${LABEL_ZIPS[$i]}")"
  if [[ -z "$mesh_zip" || -z "$label_zip" ]]; then
    echo "  ⚠ no encontrado el par ${MESH_ZIPS[$i]} / ${LABEL_ZIPS[$i]} en la descarga — se salta"
    continue
  fi
  echo "==> Extrayendo ${MESH_ZIPS[$i]} + ${LABEL_ZIPS[$i]}..."
  extract "$mesh_zip"  "$MESH_DIR"
  extract "$label_zip" "$LABEL_DIR"
  found_any=1
  # con el primer par ya están los 12 del PoC (un `&&  break` suelto moriría por set -e)
  if [[ "$MODE" == "subset" ]]; then break; fi
done
[[ "$found_any" -eq 1 ]] || { echo "ERROR: no se encontró ningún zip del dataset en la descarga."; exit 1; }

# --------------------------------------------------------------------------- #
# Verificación (emparejado cruzado obj ↔ json)
# --------------------------------------------------------------------------- #
echo "==> Verificando emparejado..."
read -r pairs orphans <<<"$(count_pairs)"
n_obj="$(find "$DEST/$MESH_DIR"  -name '*.obj'  | wc -l)"
n_json="$(find "$DEST/$LABEL_DIR" -name '*.json' | wc -l)"
n_pat="$(find "$DEST/$MESH_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
size="$(du -sh "$DEST" 2>/dev/null | cut -f1)"
echo "==> Resultado: $n_pat pacientes · $n_obj .obj / $n_json .json · $pairs casos emparejados · $size"

if [[ "$MODE" == "subset" ]]; then
  missing=0
  for id in "${PATIENT_IDS[@]}"; do
    for jaw in upper lower; do
      [[ -f "$DEST/$MESH_DIR/$id/${id}_${jaw}.obj"   ]] || { echo "  FALTA obj:  ${id}_${jaw}";  missing=$((missing + 1)); }
      [[ -f "$DEST/$LABEL_DIR/$id/${id}_${jaw}.json" ]] || { echo "  FALTA json: ${id}_${jaw}"; missing=$((missing + 1)); }
    done
  done
  if [[ "$missing" -gt 0 ]]; then
    echo "==> ⚠ $missing ficheros del subconjunto sin encontrar. Revisa los IDs o los nombres de zip."
    exit 1
  fi
fi

if [[ "$orphans" -gt 0 ]]; then
  # No es fatal: el código descarta los huérfanos, pero conviene saberlo.
  echo "==> ⚠ $orphans mallas sin labels (huérfanas). El código las descarta: sin ancla FDI no hay ingesta."
fi
[[ "$pairs" -gt 0 ]] || { echo "==> ⚠ ningún caso emparejado. Revisa la descarga."; exit 1; }
echo "==> OK — dataset emparejado en $DEST"
