# Environment setup

Tested on Ubuntu 20.04+, Python 3.8, single CUDA GPU.

## 1. Create the virtual environment

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install upstream MILAN deps

The MILAN repo is vendored at `./milan` as a git submodule.

```bash
git submodule update --init --recursive
pip install -r milan/requirements.in
python -m spacy download en_core_web_sm
```

If you cloned without `--recurse-submodules`, the submodule step is required.

## 3. Install project deps

```bash
pip install -r requirements.txt
```

## 4. Environment variables

The MILAN library reads three env vars to find data, models, and results.
Set them at the project root:

```bash
export MILAN_DATA_DIR=$(pwd)/data
export MILAN_MODELS_DIR=$(pwd)/models
export MILAN_RESULTS_DIR=$(pwd)/results
export PYTHONPATH=$(pwd):$(pwd)/milan   # so `from src import ...` (upstream) and `from milan_repro import ...` (ours) both work
```

You can drop these in a `.envrc` (with [direnv](https://direnv.net/)) or source them every session.

## 5. Pre-download MILAN decoder weights

The pretrained `base` decoder is ~1 GB. The library will fetch it on first use, but you can warm it up:

```python
from src import milan
milan.pretrained('base')   # downloads to $MILAN_MODELS_DIR
```

## Hardware notes

- Single GPU with ≥10 GB VRAM is comfortable. ResNet18 training is light; the exemplar pass is the heaviest step.
- The full pipeline (build dataset → train ResNet18 → exemplars → descriptions → ablation curves) takes ~3-6 hours end-to-end on an RTX 3090-class GPU. Most of that is exemplar computation.
