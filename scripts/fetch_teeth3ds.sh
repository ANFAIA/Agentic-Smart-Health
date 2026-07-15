#!/usr/bin/env bash
#
# fetch_teeth3ds.sh — Descarga reproducible del subconjunto Teeth3DS+ para el PoC.
#
# Baja las mallas (.obj) y sus labels (.json) desde el Google Drive oficial y
# extrae SOLO los casos fijados en PATIENT_IDS, dejándolos en data/raw/teeth3ds/
# (directorio gitignored). Idempotente: si un caso ya está, no lo vuelve a bajar.
#
# Dataset: Teeth3DS+ (MICCAI 3DTeethSeg'22). Licencia: CC-BY 4.0 (Ben-Hamadou
# et al., 2022) — atribuir en cualquier derivado. Ver docs/research/dataset-teeth3ds.md
#
# Requisitos: gdown (`uv pip install gdown` o `pip install gdown`), unzip.
# Uso:        ./scripts/fetch_teeth3ds.sh
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
# Carpeta oficial de Google Drive con los zips del reto 3DTeethSeg'22.
GDRIVE_FOLDER="https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby"

# Zips que usamos (1ª mitad; el _b2 es la 2ª mitad, no lo necesitamos para el PoC).
MESH_ZIP="3D_scans_per_patient_obj_files.zip"
LABEL_ZIP="ground-truth_labels_instances.zip"

# Nombres de las carpetas raíz DENTRO de cada zip (= árboles en disco).
MESH_DIR="3D_scans_per_patient_obj_files"
LABEL_DIR="ground-truth_labels_instances"

# Subconjunto fijo (los 12 pacientes verificados del PoC). Edita para cambiarlo.
PATIENT_IDS=(
  0EJBIPTC 0JN50XQR 0TMOBYXS
  01A6GW4A 01A6H4PZ 01A6HAN6 01A6HE9H 01A6HG3N
  01A91JH6 01A9282X 01ADUNMV 01ADYT70
)

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DEST="$REPO_ROOT/data/raw/teeth3ds"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --------------------------------------------------------------------------- #
# ¿Ya está todo? (idempotencia — no requiere herramientas de descarga)
# --------------------------------------------------------------------------- #
echo "==> Destino: $DEST"
need_download=0
for id in "${PATIENT_IDS[@]}"; do
  for jaw in upper lower; do
    [[ -f "$DEST/$MESH_DIR/$id/${id}_${jaw}.obj"  ]] || need_download=1
    [[ -f "$DEST/$LABEL_DIR/$id/${id}_${jaw}.json" ]] || need_download=1
  done
done
if [[ "$need_download" -eq 0 ]]; then
  echo "==> Todos los casos ya están presentes. Nada que hacer."
  exit 0
fi

# --------------------------------------------------------------------------- #
# Comprobaciones (solo si hay que descargar)
# --------------------------------------------------------------------------- #
command -v gdown >/dev/null || { echo "ERROR: falta gdown. Instala: uv pip install gdown"; exit 1; }
command -v unzip >/dev/null || { echo "ERROR: falta unzip."; exit 1; }
mkdir -p "$DEST/$MESH_DIR" "$DEST/$LABEL_DIR"

# --------------------------------------------------------------------------- #
# Descarga (folder completo → usamos solo los zips de la 1ª mitad)
# --------------------------------------------------------------------------- #
echo "==> Descargando zips desde Google Drive (puede tardar; son varios GB)..."
gdown --folder "$GDRIVE_FOLDER" -O "$TMP" --remaining-ok

# Localiza los zips (por si gdown crea una subcarpeta).
mesh_zip_path="$(find "$TMP" -name "$MESH_ZIP"  | head -1)"
label_zip_path="$(find "$TMP" -name "$LABEL_ZIP" | head -1)"
[[ -n "$mesh_zip_path"  ]] || { echo "ERROR: no se encontró $MESH_ZIP en la descarga.";  exit 1; }
[[ -n "$label_zip_path" ]] || { echo "ERROR: no se encontró $LABEL_ZIP en la descarga."; exit 1; }

# --------------------------------------------------------------------------- #
# Extracción selectiva (solo los IDs fijados)
# --------------------------------------------------------------------------- #
echo "==> Extrayendo ${#PATIENT_IDS[@]} casos..."
for id in "${PATIENT_IDS[@]}"; do
  unzip -o -q "$mesh_zip_path"  "$MESH_DIR/$id/*"  -d "$DEST" || echo "  ⚠ sin mallas para $id"
  unzip -o -q "$label_zip_path" "$LABEL_DIR/$id/*" -d "$DEST" || echo "  ⚠ sin labels para $id"
done

# --------------------------------------------------------------------------- #
# Verificación (emparejado cruzado obj ↔ json)
# --------------------------------------------------------------------------- #
echo "==> Verificando..."
missing=0
for id in "${PATIENT_IDS[@]}"; do
  for jaw in upper lower; do
    obj="$DEST/$MESH_DIR/$id/${id}_${jaw}.obj"
    json="$DEST/$LABEL_DIR/$id/${id}_${jaw}.json"
    [[ -f "$obj"  ]] || { echo "  FALTA obj:  $obj";  missing=$((missing+1)); }
    [[ -f "$json" ]] || { echo "  FALTA json: $json"; missing=$((missing+1)); }
  done
done

n_obj="$(find "$DEST/$MESH_DIR"  -name '*.obj'  | wc -l)"
n_json="$(find "$DEST/$LABEL_DIR" -name '*.json' | wc -l)"
echo "==> Resultado: $n_obj .obj / $n_json .json"
if [[ "$missing" -eq 0 ]]; then
  echo "==> OK — subconjunto completo y emparejado en $DEST"
else
  echo "==> ⚠ $missing ficheros sin encontrar. Revisa los IDs o los nombres de zip."
  exit 1
fi
