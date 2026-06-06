#!/usr/bin/env bash
# Re-run the two extension experiments (VGG16, CLIP) on the 10pct dataset so
# the whole deck is internally consistent at 10pct overlay.
# Outputs are 10pct-suffixed / under results/rerun_10pct so 50pct stays intact.
set -euo pipefail
cd "$(dirname "$0")/.."
export MILAN_DATA_DIR="$PWD/data"
export MILAN_MODELS_DIR="$PWD/models"
export MILAN_RESULTS_DIR="$PWD/results"
export PYTHONPATH="$PWD:$PWD/milan"

DATAV="$MILAN_DATA_DIR/imagenet-spurious-text/10pct"
RR="$MILAN_RESULTS_DIR/rerun_10pct"
mkdir -p "$RR"

echo "############ VGG16 @ 10pct ############"
VCKPT="$MILAN_MODELS_DIR/vgg16_spurious_10pct.pth"
VEX="$MILAN_RESULTS_DIR/edit/imagenet-spurious-text/vgg16_spurious-10pct"

echo "--- [V1/5] train vgg16 ---"
python -m milan_repro.train.train_multiarc --arch vgg16 \
    --config configs/vgg16_appendixE.yaml \
    --version-dir "$DATAV" --out "$VCKPT"

echo "--- [V2/5] vgg16 exemplars ---"
python -m milan_repro.milan_glue.run_exemplars_multiarc --arch vgg16 \
    --version-dir "$DATAV" --ckpt "$VCKPT" --out "$VEX"

echo "--- [V3/5] vgg16 descriptions ---"
python -m milan_repro.milan_glue.run_descriptions \
    --dissect-dir "$VEX" --out "$RR/vgg16_descriptions.csv" --layer-by-layer

echo "--- [V4/5] vgg16 identify text neurons ---"
python -m milan_repro.editing.identify_text_neurons \
    --descriptions "$RR/vgg16_descriptions.csv" \
    --out "$RR/vgg16_descriptions_annotated.csv"

echo "--- [V5/5] vgg16 ablation ---"
python -m milan_repro.editing.evaluate_multiarc --arch vgg16 \
    --version-dir "$DATAV" --ckpt "$VCKPT" --dissect-dir "$VEX" \
    --descriptions "$RR/vgg16_descriptions_annotated.csv" \
    --out "$RR/vgg16_ablation_curve.csv" \
    --ablation-max 80 --ablation-step 2 --n-random-trials 5
echo "=== VGG16 @ 10pct DONE ==="

echo "############ CLIP ViT-B/32 @ 10pct ############"
CLEX="$MILAN_RESULTS_DIR/edit/clip-vitb32-10pct"

echo "--- [C1/3] clip exemplars ---"
python -m milan_repro.milan_glue.run_clip_exemplars \
    --version-dir "$DATAV" --out "$CLEX"

echo "--- [C2/3] clip descriptions ---"
python -m milan_repro.milan_glue.run_clip_descriptions \
    --dissect-dir "$CLEX" --out "$RR/clip_descriptions.csv"

echo "--- [C3/3] clip layer analysis ---"
python -m milan_repro.figures.plot_clip_analysis \
    --descriptions "$RR/clip_descriptions.csv" --out-dir "$RR/clip_figs"
echo "=== CLIP @ 10pct DONE ==="
echo "=== ALL EXTENSIONS @ 10pct DONE ==="
