#!/usr/bin/env bash
# Process a large raw sevengill photo catalog: download from Drive, enroll
# DISK features, and discover individuals / re-sightings — unsupervised.
#
# Requires outbound network access to Google Drive. A GPU box is ideal;
# DISK feature extraction is much faster on GPU.
#
#     git clone <this-repo> && cd SanBox
#     bash spotid/run_catalog.sh
#
# Outputs: catalog.npz (feature bank) and groups.txt (discovered
# individuals with their re-sighting photos).
set -euo pipefail
cd "$(dirname "$0")/.."

DRIVE_ID="${DRIVE_ID:?set DRIVE_ID to the Drive file id of your catalog zip}"
ZIP="${ZIP:-images_raw.zip}"
IMGDIR="${IMGDIR:-images_raw}"
CATALOG="${CATALOG:-catalog.npz}"
THRESHOLD="${THRESHOLD:-0.12}"     # recalibrate on labeled pairs if available
SHORTLIST="${SHORTLIST:-200}"      # global-descriptor candidates per query
MNN_POOL="${MNN_POOL:-25}"         # mutual-NN re-rank pool -> RANSAC verify

echo "== deps =="
python3 -m pip install -q numpy scipy opencv-python-headless kornia gdown
python3 -c "import torch,kornia" 2>/dev/null || \
    python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
python3 - <<'PY'
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

if [ ! -d "$IMGDIR" ]; then
    echo "== download images_raw.zip from Drive ($DRIVE_ID) =="
    [ -f "$ZIP" ] || gdown "$DRIVE_ID" -O "$ZIP"
    echo "== unzip =="
    mkdir -p "$IMGDIR" && unzip -q -o "$ZIP" -d "$IMGDIR"
fi
N=$(find "$IMGDIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
echo "found $N images under $IMGDIR"

echo "== enroll DISK features (this is the slow step; GPU strongly preferred) =="
python3 -m spotid.identify enroll --images "$IMGDIR" --recursive --out "$CATALOG"

echo "== calibrate (if filenames encode individuals, this shows the gap) =="
python3 -m spotid.identify calibrate --catalog "$CATALOG" || true

echo "== scan: discover individuals / re-sightings =="
python3 -m spotid.identify scan --catalog "$CATALOG" \
    --threshold "$THRESHOLD" --shortlist "$SHORTLIST" --mnn-pool "$MNN_POOL" \
    --out groups.txt

echo
echo "Done. Feature bank: $CATALOG   Discovered groups: groups.txt"
echo "Tune the threshold with:  python3 -m spotid.identify calibrate --catalog $CATALOG"
echo "Then re-run scan with a better --threshold if needed."
