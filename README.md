# MILAN Section 7 Reproduction — DSC 291 SP'26

Reproduction of **Section 7: Editing Spurious Features** from Hernandez et al., *Natural Language Descriptions of Deep Visual Features* (ICLR 2022).

> A ResNet18 is trained on a 10-class image set where half of the training images have the class name painted in the corner. The model learns this text shortcut and fails on an adversarial test set where the corner text is wrong. MILAN labels every conv neuron in natural language; we ablate the neurons whose labels mention `text`/`word`/`letter` and recover a chunk of the lost adversarial accuracy — without any retraining.

Paper: <https://arxiv.org/abs/2201.11114> · Upstream code: <https://github.com/evandez/neuron-descriptions> · Project page: <https://milan.csail.mit.edu>

Group: Wonmin Kim, Seongho Kim, Ming-Yang Wu, Steven Tsai.

## What's in here

```
.
├── milan/                 # upstream MILAN (git submodule)
├── milan_repro/           # our glue code
│   ├── data/              # spurious-text dataset construction (Imagenette base)
│   ├── train/             # ResNet18 training script
│   ├── milan_glue/        # register our model + drive MILAN's exemplar/decoder
│   ├── editing/           # text-neuron identification + ablation curves
│   └── figures/           # Figure 7 (qualitative) and Figure 8 (curve)
├── configs/               # YAML hyperparameters (Appendix E style)
├── notebooks/             # 01-04 end-to-end pipeline notebooks
├── Course Project Instruction.pdf
├── project_proposal.pdf
├── environment.md         # detailed env setup
└── requirements.txt       # our extra deps (on top of milan/requirements.in)
```

## Quickstart

```bash
# 1. Clone (or refresh submodule).
git submodule update --init --recursive

# 2. Install everything (see environment.md for details).
python3.8 -m venv .venv && source .venv/bin/activate
pip install -r milan/requirements.in
python -m spacy download en_core_web_sm
pip install -r requirements.txt

# 3. Point MILAN at the project dirs (each session).
export MILAN_DATA_DIR=$(pwd)/data
export MILAN_MODELS_DIR=$(pwd)/models
export MILAN_RESULTS_DIR=$(pwd)/results
export PYTHONPATH=$(pwd):$(pwd)/milan

# 4. Run the four pipeline stages.
python -m milan_repro.data.build_splits
python -m milan_repro.train.train_resnet18
python -m milan_repro.milan_glue.run_exemplars
python -m milan_repro.milan_glue.run_descriptions
python -m milan_repro.editing.identify_text_neurons \
        --descriptions results/descriptions.csv \
        --out results/descriptions_annotated.csv
python -m milan_repro.editing.evaluate \
        --descriptions results/descriptions_annotated.csv
python -m milan_repro.figures.plot_fig7
python -m milan_repro.figures.plot_fig8
```

…or run the notebooks under `notebooks/` cell-by-cell.

## What gets produced

| File | What it is |
|------|------------|
| `data/imagenet-spurious-text/50pct/{train,test}/<class>/*.jpg` | The synthetic dataset on disk (ImageFolder layout) |
| `models/resnet18_spurious.pth` | Trained ResNet18 (10-class) |
| `results/edit/.../<layer>/{images,masks}.npy` | NetDissect-style top-k exemplars per layer |
| `results/descriptions.csv` | One MILAN caption per unit |
| `results/descriptions_annotated.csv` | …with an `is_text_neuron` flag |
| `results/importance.csv` | Per-unit clean-val accuracy under independent ablation |
| `results/ablation_curve.csv` | The data behind Figure 8 |
| `results/figs/fig7.pdf` | Qualitative grid of text-selective neurons |
| `results/figs/fig8.pdf` | Adversarial accuracy vs. number of ablated neurons |

## Substitutions from the paper

| Paper | Here |
|-------|------|
| 10 ImageNet classes | Imagenette (publicly downloadable 10-class ImageNet subset) |
| Class-name text in corner | Same approach; exact font/size/colour pinned in `configs/resnet18_appendixE.yaml` |
| MILAN decoder | Pretrained `base` decoder from <https://milan.csail.mit.edu/models/> (downloaded automatically) |

Absolute accuracy numbers will not match the paper exactly because Imagenette ≠ the authors' 10-class subset. We discuss the gap honestly in the Medium blog.
