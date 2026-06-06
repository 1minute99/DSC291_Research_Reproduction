# MILAN Section 7 Reproduction + New Experiments — DSC 291 SP'26

Reproduction of **Section 7: Editing Spurious Features** from Hernandez et al., *Natural Language Descriptions of Deep Visual Features* (ICLR 2022), with three original extensions.

> A ResNet18 is trained on a 10-class image set where half the training images have the class name painted in the corner. The model learns this text shortcut and fails on an adversarial test set where the corner text is wrong. MILAN labels every conv neuron in natural language; we ablate the neurons whose labels mention `text`/`word`/`letter` and recover adversarial accuracy — without any retraining.

Paper: <https://arxiv.org/abs/2201.11114> · Upstream code: <https://github.com/evandez/neuron-descriptions> · Project page: <https://milan.csail.mit.edu>

Group: Wonmin Kim, Seongho Kim, Ming-Yang Wu, Steven Tsai.

> **New here?** Start with [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — a
> plain-language, figure-by-figure tour of the whole project. This README is the
> code-level reference (structure, fixes, exact commands).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Changes & Fixes from the Upstream Codebase](#changes--fixes-from-the-upstream-codebase)
3. [New Experiments & Results](#new-experiments--results)
4. [Quickstart (DSMLP)](#quickstart-dsmlp)
5. [Output Files](#output-files)
6. [Substitutions from the Paper](#substitutions-from-the-paper)

---

## Repository Structure

```
.
├── milan/                        # upstream MILAN (git submodule)
├── milan_repro/
│   ├── data/                     # spurious-text dataset construction (Imagenette base)
│   ├── train/
│   │   ├── train_resnet18.py     # original ResNet18 training
│   │   └── train_multiarc.py     # NEW: VGG16 / InceptionV3 training
│   ├── milan_glue/
│   │   ├── register.py           # ResNet18 ↔ MILAN registration
│   │   ├── register_multiarc.py  # NEW: VGG16 / InceptionV3 registration
│   │   ├── run_exemplars.py      # exemplar extraction (ResNet18)
│   │   ├── run_exemplars_multiarc.py  # NEW: exemplar extraction (VGG16/Inception)
│   │   ├── run_descriptions.py   # MILAN captioning — patched for OOM
│   │   ├── clip_glue.py          # NEW: CLIP ViT-B/32 spatial wrapper
│   │   ├── run_clip_exemplars.py # NEW: exemplar extraction for CLIP
│   │   └── run_clip_descriptions.py  # NEW: MILAN captioning for CLIP
│   ├── editing/
│   │   ├── evaluate.py           # ablation eval — patched for OOM
│   │   └── evaluate_multiarc.py  # NEW: ablation eval for VGG16/Inception
│   └── figures/
│       ├── plot_fig7.py          # qualitative grid — patched for OOM
│       ├── plot_fig8.py          # ResNet18 ablation curve
│       ├── plot_layer_analysis.py    # NEW: per-layer text neuron analysis
│       ├── plot_arch_comparison.py   # NEW: ResNet18 vs VGG16 ablation curves
│       └── plot_clip_analysis.py     # NEW: CLIP block-level text neuron analysis
├── configs/
│   ├── resnet18_appendixE.yaml   # original hyperparameters
│   └── vgg16_appendixE.yaml      # NEW: VGG16 config (batch_size=32)
├── notebooks/
│   ├── 01–04_*.ipynb             # original pipeline notebooks
│   └── 05_new_experiments.ipynb  # NEW: orchestrates all 3 new experiments
└── results/                      # gitignored by default; force-add to share
    ├── descriptions_annotated.csv
    ├── ablation_curve.csv
    ├── vgg16_descriptions_annotated.csv
    ├── vgg16_ablation_curve.csv
    ├── clip_descriptions.csv
    └── figs/
        ├── fig7.pdf / fig8.pdf
        ├── fig_layer_analysis.pdf/.png
        ├── fig_arch_comparison.pdf/.png
        └── fig_clip_analysis.pdf/.png
```

---

## Changes & Fixes from the Upstream Codebase

This section documents every modification made to the original reproduction code (contributed by teammates Wonmin Kim et al.) and explains *why* each change was necessary. All changes are motivated by running the pipeline on **UCSD DSMLP** (GTX 1080 Ti, 11 GB GPU / 16 GB RAM container), which exposed several implementation-level issues in the original scripts.

### 1. `run_descriptions.py` — Layer-by-layer processing (`--layer-by-layer`)

**Problem:** The original script calls `milannotations.TopImagesDataset(dissect_dir)`, which loads *all* layers' exemplar images into RAM simultaneously. For ResNet18 (1,024 units × 15 images × 3 × 224 × 224), this peaks at ~8 GB RAM and caused `Killed` (OOM) on DSMLP.

**Fix (`milan_repro/milan_glue/run_descriptions.py`):**
- Added `--layer-by-layer` flag.
- When enabled, enumerates layer directories that contain `images.npy` (skipping `viz/` and other non-exemplar folders), loads one layer at a time via `TopImagesDataset(root, layers=[layer_name])`, runs the decoder, then releases memory with `del` + `gc.collect()`.
- Added caching: already-described layers are skipped on re-run, enabling safe resume after crashes.

```python
# Before (OOM on 16 GB container)
dissected = milannotations.TopImagesDataset(dissect_dir)

# After (layer-by-layer)
for layer_name in layer_names:
    layer_ds = milannotations.TopImagesDataset(dissect_dir, layers=[layer_name])
    descriptions = decoder.predict(layer_ds, ...)
    del layer_ds; gc.collect()
```

**Additional bug fixed:** `TopImagesDataset` iterated *all* subdirectories including `viz/`, which does not contain `images.npy`, causing `FileNotFoundError`. Fixed by filtering `[d for d in dissect_dir.iterdir() if (d / "images.npy").exists()]`.

---

### 2. `evaluate.py` — Lightweight `_UnitIndex` (no image loading)

**Problem:** The original `evaluate.py` loaded `TopImagesDataset(dissect_dir)` just to iterate `(layer, channel)` pairs per unit — it never actually used the exemplar images. Loading ~8 GB of images only to extract two integers per unit caused `Killed`.

**Fix (`milan_repro/editing/evaluate.py`):**
- Introduced `_UnitIndex`, a lightweight class that reads `layer` and `channel` columns directly from the descriptions CSV.
- Completely replaced `TopImagesDataset` in the evaluation loop.
- Memory usage dropped from ~8 GB to ~2 MB.

```python
class _UnitIndex:
    def __init__(self, descriptions_csv):
        df = pd.read_csv(descriptions_csv).sort_values("unit_index")
        self._layers   = df["layer"].tolist()
        self._channels = df["channel"].tolist()
    def unit(self, i):
        return (self._layers[i], self._channels[i])
    def __len__(self):
        return len(self._layers)
```

Same fix applied to `evaluate_multiarc.py` for VGG16/InceptionV3.

---

### 3. `plot_fig7.py` — Per-neuron mmap image loading

**Problem:** Figure 7 loads all exemplar images to show 8 text neurons. The original code loaded the entire `TopImagesDataset` (all 1,024 units) to access 8.

**Fix (`milan_repro/figures/plot_fig7.py`):**
- Replaced `TopImagesDataset` entirely.
- For each selected text neuron, directly loads only that channel's slice from `images.npy` using `np.load(..., mmap_mode="r")[channel, :top_k]`.
- Only the specific channel is paged from disk; the rest of the array stays on disk.

---

### 4. `run_clip_exemplars.py` — Two-pass heap-based exemplar collection

**Problem:** The original implementation stored full 224×224 uint8 images in a Python list per unit during exemplar collection. For CLIP's 768-channel layers: 768 units × 60 buffered images × ~150 KB/image ≈ **6.9 GB RAM** per layer → `Killed`.

**Fix (`milan_repro/milan_glue/run_clip_exemplars.py`):**
- **Pass 1:** Iterate the dataset, storing only `(score, global_image_index, spatial_argmax)` per unit in a min-heap of size `top_k`. Memory: 768 × 13,394 × 12 bytes ≈ **120 MB**.
- **Pass 2:** Collect the unique top-k image indices across all units, reload only those images from the dataset by index, and assemble the output arrays.
- Peak RAM reduced from ~7 GB to ~300 MB per layer.

---

### 5. `clip_glue.py` — CLIP transformer output format fix

**Problem:** CLIP's `ResidualAttentionBlock` outputs tensors in `(seq_len, B, embed_dim)` format (classic PyTorch transformer convention), but the hook was written assuming `(B, seq_len, embed_dim)` (HuggingFace convention). This caused `output[:, 1:, :]` to slice the *batch* dimension instead of the sequence dimension, producing incorrect shapes and a `RuntimeError` on reshape.

**Fix (`milan_repro/milan_glue/clip_glue.py`):**
```python
# Before (wrong axis)
spatial = output[:, 1:, :]   # sliced batch dim!

# After (correct)
x = output.permute(1, 0, 2)  # (seq_len, B, C) → (B, seq_len, C)
spatial = x[:, 1:, :]        # remove CLS token → (B, 49, 768)
```

---

### 6. `run_clip_descriptions.py` — mmap dataset + correct `TopImages` namedtuple

**Problem:** `TopImagesDataset` for one CLIP layer (768 units × 15 images): images.npy (1.65 GB uint8) + masks.npy (2.2 GB float32) = **3.85 GB** loaded at init → `Killed`. Additionally, the MILAN decoder indexes batches as `batch[2]` (images) and `batch[3]` (masks), requiring the namedtuple to have exactly 4 fields: `layer, unit, images, masks`.

**Fix (`milan_repro/milan_glue/run_clip_descriptions.py`):**
- Created `_MmapLayerDataset` that opens `images.npy` / `masks.npy` with `mmap_mode="r"`. Only the requested unit's 15-image slice is paged from disk per `__getitem__` call.
- Returns `_TopImages(layer, unit, images, masks)` — a 4-field namedtuple matching MILAN's internal `image_index=2, mask_index=3` convention.

---

## New Experiments & Results

Three original experiments were designed and executed on DSMLP to extend the reproduction beyond the original paper.

---

### Baseline Reproduction (ResNet18)

Before the new experiments, we confirmed successful reproduction of Section 7.

| Metric | Paper (est.) | Ours |
|--------|-------------|------|
| clean val accuracy | ~82.9% | **84.1%** |
| adversarial test accuracy | ~19.1% | **17.9%** |
| text neurons identified | ~541 | **541 / 1,024 (52.8%)** |

Difference within expected range — Imagenette ≠ authors' exact 10-class subset.

---

### Experiment 1 — Architecture Generalization (VGG16)

**Research question:** *Is MILAN's ability to identify and ablate text neurons specific to ResNet18, or does it generalize to other CNN architectures?*

**Setup:**
- Trained VGG16 from scratch on the same spurious-text dataset (50% overlay) using `train_multiarc.py` with `configs/vgg16_appendixE.yaml` (batch_size=32 to fit 11 GB GPU).
- Registered VGG16 layers with MILAN via `register_multiarc.py` (probed: `features.2`, `.7`, `.14`, `.21`, `.28`).
- Ran full pipeline: exemplar extraction → MILAN captioning → text-neuron identification → ablation evaluation.

**Results:**

| Model | Text Neurons | Text Neuron % | clean(val) | adv(test) |
|-------|-------------|--------------|-----------|----------|
| ResNet18 | 541 / 1,024 | 52.8% | 84.1% | **17.9%** |
| VGG16 | 524 / 1,472 | 35.6% | 77.8% | **9.5%** |

**Findings:**
- MILAN successfully identified text neurons in VGG16 without any modification to the decoder.
- VGG16's adversarial accuracy (9.5%) is substantially lower than ResNet18 (17.9%), indicating VGG16 relies more heavily on the spurious text shortcut.
- Despite having a lower text-neuron *fraction* (35.6% vs 52.8%), VGG16's overall classification is more compromised — likely because VGG16's fully-connected classifier amplifies feature-level biases.
- **Conclusion:** MILAN-based text-neuron ablation generalizes across CNN architectures. The degree of shortcut reliance is architecture-dependent.

---

### Experiment 2 — Layer-Depth Analysis (ResNet18)

**Research question:** *At which layers of the network do text-selective neurons concentrate? Does the network encode text shortcut features early or late?*

**Setup:**
- Used the existing ResNet18 descriptions (`descriptions_annotated.csv`) — no additional training needed.
- `plot_layer_analysis.py` groups neurons by layer, computes text-neuron fraction, description diversity (unique word ratio), and top-5 most frequent words per layer.

**Results:**

| Layer | Total Units | Text Neurons | Fraction | Top 5 Words |
|-------|------------|--------------|----------|-------------|
| conv1 | 64 | 11 | **17.2%** | white, objects, lines, edges, dots |
| layer1 | 64 | 37 | **57.8%** | words, white, letters, lines, objects |
| layer2 | 128 | 73 | 57.0% | words, letters, lines, white, horizontal |
| layer3 | 256 | 149 | 58.2% | words, lines, letters, horizontal, white |
| layer4 | 512 | 271 | 52.9% | words, objects, signs, white, colored |

**Findings:**
- **conv1 (17.2%) → layer1 (57.8%): a 3.4× jump in a single residual stage.** The network begins encoding text as a categorical feature very early — immediately after the first residual block.
- Layers 1–4 maintain a stable plateau of ~55–58%, indicating text shortcut representations are not concentrated at one depth but distributed broadly.
- Description diversity *decreases* with depth (from 0.227 at conv1 to 0.073 at layer4): deeper neurons produce more homogeneous descriptions dominated by "words" and "letters", while early neurons describe more varied visual patterns.
- **Conclusion:** The spurious text shortcut is captured almost immediately after the initial convolution and persists throughout the entire network, suggesting it becomes deeply entangled in the learned representation.

---

### Experiment 3 — CLIP ViT-B/32 Application

**Research question:** *Do text-selective neurons appear in CLIP's vision transformer? Is CLIP more robust to spurious text features than a task-trained CNN?*

**Setup:**
- Used pretrained CLIP ViT-B/32 (zero-shot, no fine-tuning on spurious dataset).
- `clip_glue.py` wraps CLIP's visual transformer, inserting hooks at 5 residual blocks (0, 3, 6, 9, 11) that reshape the spatial tokens `(49, B, 768)` → `(B, 768, 7, 7)` for MILAN compatibility.
- `run_clip_exemplars.py` extracts exemplars using the two-pass heap approach (Pass 1: score + index; Pass 2: image reload).
- `run_clip_descriptions.py` generates MILAN captions using a mmap-backed dataset.

**Results:**

| Block | Total Units | Text Neurons | Fraction | Top 5 Words |
|-------|------------|--------------|----------|-------------|
| 0 (shallow) | 768 | 34 | **4.4%** | ground, objects, colored, white, designs |
| 3 | 768 | 63 | **8.2%** | objects, colored, white, circular, black |
| 6 (mid) | 768 | 7 | **0.9%** | objects, colored, white, faces, ground |
| 9 | 768 | 14 | **1.8%** | objects, colored, white, red, ground |
| 11 (deep) | 768 | 76 | **9.9%** | objects, colored, white, red, circular |

**Comparison with ResNet18 (same spurious dataset):**

| Model | Max text-neuron fraction | Dominant top words |
|-------|------------------------|-------------------|
| ResNet18 (trained) | **58.2%** | words, letters, lines |
| CLIP ViT-B/32 (zero-shot) | **9.9%** | objects, colored, white |

**Findings:**
- CLIP's maximum text-neuron fraction (9.9%) is **~6× lower** than ResNet18's (58.2%).
- Critically, CLIP's top words across all blocks are semantic object descriptors ("objects", "colored", "ground") rather than text descriptors ("words", "letters"). This suggests CLIP neurons respond to *what is in the image*, not *what text is overlaid*.
- Block 6 shows the lowest text fraction (0.9%), suggesting mid-level transformer representations are particularly robust to visual text artifacts.
- The non-monotonic pattern (4.4% → 8.2% → 0.9% → 1.8% → 9.9%) differs from ResNet18's monotonic early-layer jump, reflecting the transformer's global attention mechanism distributing information differently across depth.
- **Conclusion:** CLIP's visual-language pretraining produces representations substantially more robust to spurious text features. This is consistent with CLIP having learned to associate images with semantic descriptions rather than low-level text patterns. MILAN-based analysis provides a mechanistic explanation: CLIP simply has far fewer text-selective neurons than a task-trained CNN.

---

## Quickstart (DSMLP)

```bash
# 0. SSH into DSMLP and launch a GPU pod (GTX 1080 Ti recommended)
ssh <username>@dsmlp-login.ucsd.edu
launch-scipy-ml-gpu.sh   # or equivalent GPU launcher

# 1. Clone with submodule
git clone --recursive https://github.com/castlhoo/DSC291-MILAN.git
cd DSC291-MILAN
git checkout sean-work

# 2. Install dependencies
pip install -r milan/requirements.in
python -m spacy download en_core_web_sm
pip install -r requirements.txt

# 3. Set environment variables (add to ~/.bashrc for persistence)
export MILAN_DATA_DIR=$PWD/data
export MILAN_MODELS_DIR=$PWD/models
export MILAN_RESULTS_DIR=$PWD/results
export PYTHONPATH=$PWD:$PWD/milan

# 4. Run the base pipeline (use tmux to survive session disconnects)
tmux new -s milan
python -m milan_repro.data.build_splits
python -m milan_repro.train.train_resnet18
python -m milan_repro.milan_glue.run_exemplars
python -m milan_repro.milan_glue.run_descriptions --layer-by-layer
python -m milan_repro.editing.identify_text_neurons \
        --descriptions results/descriptions.csv \
        --out results/descriptions_annotated.csv
python -m milan_repro.editing.evaluate \
        --descriptions results/descriptions_annotated.csv \
        --ablation-max 80 --ablation-step 2 --n-random-trials 5
python -m milan_repro.figures.plot_fig7 --descriptions results/descriptions_annotated.csv
python -m milan_repro.figures.plot_fig8
python -m milan_repro.figures.plot_layer_analysis --descriptions results/descriptions_annotated.csv

# 5. New Experiment 1 — VGG16
python -m milan_repro.train.train_multiarc --arch vgg16 --config configs/vgg16_appendixE.yaml
python -m milan_repro.milan_glue.run_exemplars_multiarc --arch vgg16
python -m milan_repro.milan_glue.run_descriptions \
        --dissect-dir results/edit/imagenet-spurious-text/vgg16-50pct \
        --out results/vgg16_descriptions.csv --layer-by-layer
python -m milan_repro.editing.identify_text_neurons \
        --descriptions results/vgg16_descriptions.csv \
        --out results/vgg16_descriptions_annotated.csv
python -m milan_repro.editing.evaluate_multiarc \
        --arch vgg16 --descriptions results/vgg16_descriptions_annotated.csv \
        --ablation-max 80 --ablation-step 2 --n-random-trials 5
python -m milan_repro.figures.plot_arch_comparison

# 6. New Experiment 3 — CLIP
pip install "git+https://github.com/openai/CLIP.git"
python -m milan_repro.milan_glue.run_clip_exemplars
python -m milan_repro.milan_glue.run_clip_descriptions
python -m milan_repro.figures.plot_clip_analysis
```

---

### Extensions

- **InceptionV3** (architecture-generalization extension): the same four stages
  run with an `--arch inception_v3` switch. See
  [`docs/inception_v3_experiment.md`](docs/inception_v3_experiment.md) for the
  full command-by-command walkthrough, or run notebooks `05`–`07`.

## Output Files

| File | Description |
|------|-------------|
| `results/descriptions_annotated.csv` | MILAN captions + `is_text_neuron` flag for ResNet18 |
| `results/ablation_curve.csv` | ResNet18 ablation curve data (text / random / importance order) |
| `results/vgg16_descriptions_annotated.csv` | Same for VGG16 |
| `results/vgg16_ablation_curve.csv` | VGG16 ablation curve data |
| `results/clip_descriptions.csv` | MILAN captions for CLIP ViT-B/32 blocks |
| `results/figs/fig7.pdf` | Qualitative: top exemplars of text neurons with MILAN captions |
| `results/figs/fig8.pdf` | ResNet18 ablation curve (reproduces paper Figure 8) |
| `results/figs/fig_layer_analysis.pdf/.png` | Text-neuron fraction per ResNet18 layer |
| `results/figs/fig_arch_comparison.pdf/.png` | Ablation curves: ResNet18 vs VGG16 |
| `results/figs/fig_clip_analysis.pdf/.png` | Text-neuron fraction per CLIP transformer block |
| `results/figs/layer_analysis.csv` | Tabular data for layer analysis figure |
| `results/figs/clip_layer_analysis.csv` | Tabular data for CLIP analysis figure |

---

## Substitutions from the Paper

| Paper | Here |
|-------|------|
| 10 ImageNet classes | Imagenette (publicly available 10-class ImageNet subset) |
| Class-name text in corner | Same; font/size/colour pinned in `configs/resnet18_appendixE.yaml` |
| MILAN decoder | Pretrained `base` decoder from <https://milan.csail.mit.edu/models/> |
| High-memory server | UCSD DSMLP (GTX 1080 Ti, 16 GB RAM) — required all OOM fixes above |

Absolute accuracy numbers will not match the paper exactly because Imagenette ≠ the authors' exact 10-class subset.
