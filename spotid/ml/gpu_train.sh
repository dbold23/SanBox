#!/usr/bin/env bash
# One-shot GPU training + evaluation for the spotid encoder.
#
# On a fresh GPU box (e.g. vast.ai RTX 5090):
#     git clone <this-repo> && cd SanBox
#     bash spotid/ml/gpu_train.sh
#
# Trains the scaled-up encoder (~1 h, mostly CPU-bound synthetic
# rendering), then runs the classical/learned/ensemble head-to-head on
# unseen identities and prints the table. The checkpoint lands in
# spotid/ml/checkpoints/encoder_gpu.pt — copy it back into the repo to
# use it (see the scp hint printed at the end).
set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS="${STEPS:-20000}"
WIDTH="${WIDTH:-64}"
EMBED="${EMBED:-256}"
IDS_PER_BATCH="${IDS_PER_BATCH:-48}"
ID_POOL="${ID_POOL:-20000}"
WORKERS="${WORKERS:-$(($(nproc) > 8 ? $(nproc) - 4 : 4))}"
OUT="${OUT:-spotid/ml/checkpoints/encoder_gpu.pt}"

echo "== deps =="
python3 -m pip install -q numpy opencv-python-headless scipy
# RTX 5090 (Blackwell, sm_120) needs CUDA 12.8 torch builds.
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
python3 - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY

echo "== train (${STEPS} steps, width ${WIDTH}, embed ${EMBED}) =="
python3 -m spotid.ml.train \
    --device cuda --steps "$STEPS" --width "$WIDTH" --embed-dim "$EMBED" \
    --ids-per-batch "$IDS_PER_BATCH" --id-pool "$ID_POOL" \
    --workers "$WORKERS" --out "$OUT"

echo "== head-to-head on unseen identities =="
python3 -m spotid.ml.evaluate_ml --checkpoint "$OUT" --device cuda \
    --identities 150 --views 40 --seed 7

echo
echo "Done. Checkpoint: $OUT"
echo "Copy it back from your machine with:"
echo "  scp -P <port> root@<host>:$(pwd)/$OUT spotid/ml/checkpoints/"
