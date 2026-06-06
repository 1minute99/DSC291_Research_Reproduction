# CLAUDE.md

DSC 291 SP'26 final project — reproduction of **MILAN, Section 7 "Editing Spurious Features"** (Hernandez et al., ICLR 2022, `arXiv:2201.11114`) plus three new experiments.

Team: Wonmin Kim, Seongho Kim, Ming-Yang Wu, Steven Tsai. Presentation date: 2026-06-09.

## Repo layout

```
.
├── milan/                    # Upstream MILAN code (evandez/neuron-descriptions @ 19c4d58).
│                             # Previously a git submodule; absorbed into the repo as
│                             # a regular directory in commits fa141cf + 938576b.
├── milan_repro/              # Our reproduction code (imports `milan/src` via PYTHONPATH).
│   ├── data/build_splits.py            # Builds spurious-text Imagenette from imagenette2-320.
│   ├── train/train_resnet18.py         # Baseline.
│   ├── train/train_multiarc.py         # VGG16 / InceptionV3.
│   ├── train/train_inception_v3.py     # Inception-only entry point.
│   ├── milan_glue/register*.py         # Wire models into MILAN's exemplar pipeline.
│   ├── milan_glue/run_exemplars*.py    # Exemplar extraction (incl. CLIP).
│   ├── milan_glue/run_descriptions.py  # MILAN captioning — supports --layer-by-layer (OOM fix).
│   ├── milan_glue/clip_glue.py         # CLIP ViT spatial-token reshape for MILAN.
│   ├── editing/evaluate*.py            # Ablation eval (replaces TopImagesDataset with _UnitIndex).
│   └── figures/plot_*.py               # All paper-figure + new-experiment plots.
├── configs/                  # Appendix E hyperparameters per architecture.
├── notebooks/                # 01–09 run notebooks; 05_new_experiments orchestrates the extensions.
├── data/                     # imagenette2-320 + imagenet-spurious-text/50pct splits (gitignored).
├── models/                   # Trained checkpoints (gitignored).
├── results/                  # CSVs + figures; results/figs/{slides,paper}/ used by the deck.
└── scripts/
    ├── build_deck.py                # ★ CURRENT deck builder. Declarative — builds the 32-slide MILAN deck from the template's layouts (sections split across multiple slides + figures/tables, per the example format). Output: "SP 26 DSC 291 MILAN - Research Reproduction .pptx".
    ├── make_slide_figures.py        # Generates 4 slide figures (spurious grid, arch bar, summary table, pipeline). Numbers hardcoded — edit on result changes.
    ├── make_overlay_sweep_figure.py # Overlay-strength sweep panel (50/20/10/5%); kept for reference, no longer on a slide.
    ├── run_overlay_experiment.sh    # Full ResNet18 pipeline at any overlay fraction (build→train→exemplars→desc→identify→eval), isolated outputs.
    ├── run_extensions_10pct.sh      # VGG16 + CLIP extension reruns at 10pct.
    └── fill_presentation.py         # OLD 24-slide in-place filler (superseded by build_deck.py; kept for reference).
```

## Quick commands

```bash
# Env (project venv: ../venv-291 — see environment.md if missing)
export MILAN_DATA_DIR=$PWD/data
export MILAN_MODELS_DIR=$PWD/models
export MILAN_RESULTS_DIR=$PWD/results
export PYTHONPATH=$PWD:$PWD/milan

# Build dataset (only first time)
python -m milan_repro.data.build_splits

# Baseline pipeline
python -m milan_repro.train.train_resnet18
python -m milan_repro.milan_glue.run_exemplars
python -m milan_repro.milan_glue.run_descriptions --layer-by-layer   # OOM-safe, required on DSMLP
python -m milan_repro.editing.identify_text_neurons \
        --descriptions results/descriptions.csv \
        --out results/descriptions_annotated.csv
python -m milan_repro.editing.evaluate \
        --descriptions results/descriptions_annotated.csv \
        --ablation-max 80 --ablation-step 2 --n-random-trials 5
python -m milan_repro.figures.plot_fig7 --descriptions results/descriptions_annotated.csv
python -m milan_repro.figures.plot_fig8

# Slide deck regeneration (current = 32-slide MILAN deck)
python scripts/make_slide_figures.py     # regenerate the slide figures (hardcoded numbers live here)
python scripts/build_deck.py             # build the deck -> "SP 26 DSC 291 MILAN - Research Reproduction .pptx"
```
Format references (do not edit): `SP 26 DSC 291 Project Template - Research Reproduction.pptx` (recommended page counts + instructions) and `SP 26 DSC 291 Example.pptx` (multi-slide-with-figures format).

## Conventions & gotchas

- **DSMLP-bound**: target box is GTX 1080 Ti, 11 GB GPU + 16 GB RAM. Every script that touches exemplar tensors needs the OOM fixes already applied; do not revert them. See README "Changes & Fixes" section for the full list (layer-by-layer descriptions, `_UnitIndex` replacing `TopImagesDataset`, mmap-backed CLIP dataset, two-pass CLIP exemplar collection, `clip_glue.py` ViT transpose).
- **`milan/` is no longer a submodule**: it lives as plain files in this repo, pinned to upstream commit `19c4d58`. Do not re-add it as a submodule. If you need to pull upstream changes, fetch from `https://github.com/evandez/neuron-descriptions` manually and diff.
- **CSV schema**: descriptions files are `unit_index, layer, channel, description[, is_text_neuron]`; ablation curves are `mode, trial, n_ablated, clean_acc, adv_acc` where `mode ∈ {baseline, text-sorted, random}`.
- **Class mapping**: 10 Imagenette synsets in `milan_repro/data/build_splits.py` (`IMAGENETTE_CLASSES`); painted short labels in `SHORT_LABELS`. Don't substitute one for the other.
- **Adversarial set**: `data/imagenet-spurious-text/50pct/test_strict/` has wrong-class overlay (the adversarial eval); `test/` has the unmodified images.
- **Branches**: `main` is the merged branch containing inception-extension + sean-work. Teammate branches (`origin/inception-extension`, `origin/sean-work`) remain on the remote for history.

## Headline numbers (don't rederive)

**Canonical run is now the 10% overlay** (`data/imagenet-spurious-text/10pct`, results in `results/rerun_10pct/`). The original 50% run produced a near-chance adversarial baseline (17.9%) where the shortcut was too pervasive (52.8% text neurons) for caption-guided ablation to separate from random — at 10% the shortcut is sparse enough that text-sorted editing clearly recovers robustness and beats baselines (this is the deck's headline). The deck figures (`results/figs/fig7.png`, `fig8.png`, `fig_arch_comparison.png`, `fig_layer_analysis.png`, `fig_clip_analysis.png`, `slides/*`) are all regenerated at 10%.

| Experiment | Clean val | Adv test | Text neurons | Note |
|---|---|---|---|---|
| ResNet18 baseline (10% overlay) | 75.4% | 51.5% | 216 / 1024 (21.1%) | text-sorted ablation recovers 51.5→~55%, beats random/importance |
| VGG16 generalization | 69.0% | 19.8% | 357 / 1472 (24.3%) | more shortcut-reliant; recovers 19.8→32.7% |
| ResNet18 layer-depth | — | — | 9→27→37→77→66 | peaks at layer1 (3.0× jump conv1→layer1), then declines |
| CLIP ViT-B/32 zero-shot | — | — | 61 / 768 (7.9% max) | ≈2.7× more robust than ResNet18 |

_Old 50% numbers (kept for reference; not in the deck): ResNet18 84.1%/17.9%/541-1024; VGG16 77.8%/9.5%/524-1472; layer 11→37→73→149→271; CLIP 76/768. The full overlay sweep (50/20/10/5%) lives in `results/rerun_*` with plots `results/figs/fig8_{20,10,5}pct.png`._

## Deliverables location

- Slides: `SP 26 DSC 291 MILAN - Research Reproduction .pptx` (32 slides, built by `scripts/build_deck.py`; group # placeholder on the title slide is the only thing left).
- Blog (Medium PDF): not yet started.
- Code repo: `github.com/1minute99/DSC291_Research_Reproduction` (origin).
