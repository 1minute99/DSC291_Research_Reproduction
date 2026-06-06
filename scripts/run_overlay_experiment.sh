#!/usr/bin/env bash
# Re-run the full MILAN Section-7 pipeline at a chosen training-overlay
# fraction, to test whether a weaker/sparser shortcut lets text-sorted
# ablation separate from random (the 50pct run did not — 52.8% text neurons,
# adv-acc ~= chance).
#
# All outputs land in isolated paths so the headline 50pct deck artifacts
# (results/descriptions*.csv, results/ablation_curve.csv, results/importance.csv,
#  models/resnet18_spurious.pth) are NEVER touched.
#
# Usage:  scripts/run_overlay_experiment.sh <train_fraction> <version_tag>
#   e.g.  scripts/run_overlay_experiment.sh 0.20 20pct
set -euo pipefail

FRAC="${1:?usage: run_overlay_experiment.sh <train_fraction> <version_tag>}"
VER="${2:?usage: run_overlay_experiment.sh <train_fraction> <version_tag>}"

cd "$(dirname "$0")/.."
export MILAN_DATA_DIR="$PWD/data"
export MILAN_MODELS_DIR="$PWD/models"
export MILAN_RESULTS_DIR="$PWD/results"
export PYTHONPATH="$PWD:$PWD/milan"

DATAV="$MILAN_DATA_DIR/imagenet-spurious-text/$VER"
CKPT="$MILAN_MODELS_DIR/resnet18_spurious_${VER}.pth"
EXDIR="$MILAN_RESULTS_DIR/edit/imagenet-spurious-text/resnet18_spurious-${VER}"
RR="$MILAN_RESULTS_DIR/rerun_${VER}"
mkdir -p "$RR"

echo "=== [$VER] overlay rerun: train_fraction=$FRAC ==="
echo "    data=$DATAV  ckpt=$CKPT  out=$RR"

echo "--- [1/6] build_splits ---"
python -m milan_repro.data.build_splits \
    --version "$VER" --train-fraction "$FRAC" --test-fraction 1.0

echo "--- [2/6] train ResNet18 ---"
python -m milan_repro.train.train_resnet18 \
    --version-dir "$DATAV" --out "$CKPT"

echo "--- [3/6] exemplars ---"
python -m milan_repro.milan_glue.run_exemplars \
    --version-dir "$DATAV" --ckpt "$CKPT" --out "$EXDIR"

echo "--- [4/6] MILAN descriptions ---"
python -m milan_repro.milan_glue.run_descriptions \
    --dissect-dir "$EXDIR" --out "$RR/descriptions.csv" --layer-by-layer

echo "--- [5/6] identify text neurons ---"
python -m milan_repro.editing.identify_text_neurons \
    --descriptions "$RR/descriptions.csv" \
    --out "$RR/descriptions_annotated.csv"

echo "--- [6/6] evaluate ablation ---"
python -m milan_repro.editing.evaluate \
    --version-dir "$DATAV" --ckpt "$CKPT" --dissect-dir "$EXDIR" \
    --descriptions "$RR/descriptions_annotated.csv" \
    --out "$RR/ablation_curve.csv" \
    --ablation-max 80 --ablation-step 2 --n-random-trials 5

echo "=== [$VER] DONE -> $RR/ablation_curve.csv ==="
