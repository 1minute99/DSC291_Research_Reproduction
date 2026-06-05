# InceptionV3 experiment — running the full MILAN pipeline

Extension experiment (proposal §2.1, "New experiments"): does the spurious-text
behaviour MILAN finds in ResNet18 generalise to a deeper, multi-branch
architecture? This doc walks through running the whole Section-7 pipeline on
**InceptionV3** end-to-end.

The pipeline is the same four stages as the ResNet18 reproduction; an `--arch
inception_v3` switch threads the architecture (model factory, probe layers, and
the **299×299** input size) through every stage. See
[`milan_repro/milan_glue/register.py`](../milan_repro/milan_glue/register.py)
for the arch registry.

## Prerequisites

- Environment set up per [`../environment.md`](../environment.md) (Python 3.8 `.venv`).
- The trained checkpoint already exists at `models/inception_v3_spurious.pth`
  (produced by [`milan_repro/train/train_inception_v3.py`](../milan_repro/train/train_inception_v3.py)
  / notebook `05`). If not, train it first:
  ```bash
  python -m milan_repro.train.train_inception_v3 --device mps
  ```

## One-shot command sequence

```bash
source .venv/bin/activate
export MILAN_DATA_DIR=$(pwd)/data MILAN_MODELS_DIR=$(pwd)/models MILAN_RESULTS_DIR=$(pwd)/results
export PYTHONPATH=$(pwd):$(pwd)/milan

# training already done -> models/inception_v3_spurious.pth
python -m milan_repro.milan_glue.run_exemplars   --arch inception_v3 --device mps
python -m milan_repro.milan_glue.run_descriptions --dissect-dir results/edit/imagenet-spurious-text/inception_v3_spurious-50pct --out results/descriptions_inception_v3.csv
python -m milan_repro.editing.identify_text_neurons --descriptions results/descriptions_inception_v3.csv --out results/descriptions_inception_v3_annotated.csv
python -m milan_repro.editing.evaluate --arch inception_v3 --device mps
```

…or run notebooks [`06_milan_descriptions_inception.ipynb`](../notebooks/06_milan_descriptions_inception.ipynb)
then [`07_editing_experiment_inception.ipynb`](../notebooks/07_editing_experiment_inception.ipynb)
cell-by-cell (they call the same functions).

## What each line does

### Environment setup
```bash
source .venv/bin/activate
```
Activate the project virtualenv (Python 3.8 with the pinned deps).

```bash
export MILAN_DATA_DIR=$(pwd)/data MILAN_MODELS_DIR=$(pwd)/models MILAN_RESULTS_DIR=$(pwd)/results
```
The three directories upstream MILAN and our glue read to locate the dataset,
model checkpoints, and where to write results. Every command below derives its
default paths from these.

```bash
export PYTHONPATH=$(pwd):$(pwd)/milan
```
Put the repo root (so `import milan_repro …` works) and the vendored MILAN
submodule (so `from src import …` works) on the import path.

### Stage 1 — Exemplars (dissection)
```bash
python -m milan_repro.milan_glue.run_exemplars --arch inception_v3 --device mps
```
Runs NetDissect-style activation tallying: pushes the (clean) training images
through the trained InceptionV3 and, **for every channel in the 5 probe layers**
(`Conv2d_2b_3x3`, `Mixed_5d`, `Mixed_6a`, `Mixed_6e`, `Mixed_7c` ≈ **3,936
units**), records its top-k most-activating image regions.

- Inputs run at **299×299** automatically (Inception's native size, from the
  arch registry — no flag needed).
- **Output:** `results/edit/imagenet-spurious-text/inception_v3_spurious-50pct/<layer>/{images,masks}.npy`
  (top-k exemplars per unit, in the 224px byte format the MILAN decoder reads).
- Caches per layer, so an interrupted run resumes where it left off.
- This is the heaviest stage (~4× the ResNet18 unit count).

### Stage 2 — Descriptions (captioning)
```bash
python -m milan_repro.milan_glue.run_descriptions \
    --dissect-dir results/edit/imagenet-spurious-text/inception_v3_spurious-50pct \
    --out results/descriptions_inception_v3.csv
```
Feeds each unit's exemplars through the pretrained MILAN decoder (`base`) to
produce one natural-language description per unit. Architecture-agnostic — it
just reads the exemplar dir from Stage 1.

- **Output:** `results/descriptions_inception_v3.csv` with columns
  `unit_index, layer, channel, description`.
- The `--dissect-dir`/`--out` are passed explicitly here because this stage has
  no `--arch` flag (it doesn't touch the model).

### Stage 3 — Identify text neurons
```bash
python -m milan_repro.editing.identify_text_neurons \
    --descriptions results/descriptions_inception_v3.csv \
    --out results/descriptions_inception_v3_annotated.csv
```
Adds an `is_text_neuron` boolean column: `True` for any unit whose description
contains a whole word `text` / `word` / `letter` (the same rule upstream's
`experiments/edit.py` uses). Pure CSV post-processing.

- **Output:** `results/descriptions_inception_v3_annotated.csv` (descriptions +
  the `is_text_neuron` flag); prints the text-neuron count.

### Stage 4 — Editing experiment (ablation curves)
```bash
python -m milan_repro.editing.evaluate --arch inception_v3 --device mps
```
The actual edit. For each unit it measures the clean-validation accuracy drop
from ablating that unit alone (the per-unit *importance*, cached to
`results/importance_inception_v3.csv`), then builds three adversarial-accuracy
curves vs. number of neurons ablated:

- **text-sorted** — ablate only MILAN-flagged text neurons (smallest impact first).
- **sort-all** — baseline: ablate any neuron, ranked by importance.
- **random** — averaged over random orderings.

Inputs are fed at **299×299** automatically; with `--arch inception_v3` the
defaults resolve to the checkpoint `models/inception_v3_spurious.pth`, the
dissect dir from Stage 1, and `results/descriptions_inception_v3_annotated.csv`.

- **Output:** `results/ablation_curve_inception_v3.csv`
  (columns `mode, trial, n_ablated, clean_acc, adv_acc`).
- Figures: `python -m milan_repro.figures.plot_fig7` / `plot_fig8` (point them at
  the `_inception_v3` CSVs), or use notebook `07`.

## Outputs at a glance

| File | Produced by | What it is |
|------|-------------|------------|
| `results/edit/imagenet-spurious-text/inception_v3_spurious-50pct/<layer>/{images,masks}.npy` | Stage 1 | Top-k exemplars per unit |
| `results/descriptions_inception_v3.csv` | Stage 2 | One MILAN caption per unit |
| `results/descriptions_inception_v3_annotated.csv` | Stage 3 | …with the `is_text_neuron` flag |
| `results/importance_inception_v3.csv` | Stage 4 | Per-unit clean-val accuracy under independent ablation |
| `results/ablation_curve_inception_v3.csv` | Stage 4 | The three Figure-8 curves |

## Notes & gotchas

- **Compute.** ~3,936 units (vs ResNet18's 1,024). Stage 1 and the per-unit
  importance loop in Stage 4 (~3,936 ablation evals) are the slow parts. To
  speed things up: keep `--device mps`, raise `--ablation-step` / lower
  `--ablation-max` in Stage 4, or trim the layer list in `register.py`.
- **MILAN decoder on MPS.** Stage 2's decoder can hit ops MPS doesn't support.
  If it errors, run just that stage on CPU (`--device cpu`), optionally with
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Input size.** Both the dissection (Stage 1) and the ablation eval (Stage 4)
  feed Inception at 299×299. This is automatic via `--arch inception_v3`; don't
  pass a `--image-size` override unless you also retrained at that size.
- **Comparing to ResNet18 / VGG16.** Report the pair (clean acc, adversarial
  acc) plus text-sorted vs sort-all recovery, and the text-neuron fraction per
  layer — the same quantities the other architectures report — rather than
  matching absolute accuracies across architectures.
