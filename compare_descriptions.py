"""Compare neuron-description methods on ResNet-50 / ImageNet fc (class) neurons.

Methods compared (all describe the SAME fc/logit neurons of resnet50-imagenet):
  - MILAN   (repo: neuron-descriptions)  -> data/MILAN_results/m_base_resnet50_imagenet.csv
  - DnD     (repo: Describe-and-Dissect) -> data/DnD_results/rn50_results/resnet50_imagenet_broden_fc.csv
  - NetDissect (baseline)                -> data/NetDissect_results/resnet50_imagenet_fc.csv

MAIA (repo: maia) ships no cached descriptions and cannot run here (needs a 24GB GPU
+ API key), so it is absent. CLIP-Dissect has no fc layer in the shipped results.

We pick 6 ImageNet classes (an Imagenette subset) and treat each class-detector
neuron as the unit to describe.  We score every method's description with BERTScore:
  (a) vs the ground-truth class name  -> how faithful each method is to the true concept
  (b) pairwise between methods         -> how much the methods agree with each other
"""
import re
import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bert_score import score as bertscore

DATA = Path("/Users/JaredWu/Desktop/Describe-and-Dissect/data")
OUT = Path("/Users/JaredWu/Desktop/DSC291_proj/comparison/outputs")
OUT.mkdir(parents=True, exist_ok=True)

# 6 ImageNet classes (0-indexed) -> human-readable ground-truth concept.
CLASSES = {
    0: "tench",
    217: "English springer spaniel",
    482: "cassette player",
    497: "church",
    569: "garbage truck",
    574: "golf ball",
}
IDX = list(CLASSES.keys())

# ---------------------------------------------------------------- loaders
def load_dnd():
    """DnD fc: header 'Neuron ID,label', 0-indexed."""
    out = {}
    with open(DATA / "DnD_results/rn50_results/resnet50_imagenet_broden_fc.csv") as f:
        for row in csv.DictReader(f):
            u = int(row["Neuron ID"])
            if u in CLASSES:
                out[u] = row["label"].strip()
    return out


def load_milan():
    """MILAN: 'layer,unit,description'; use rows where layer=='fc', 0-indexed."""
    out = {}
    with open(DATA / "MILAN_results/m_base_resnet50_imagenet.csv") as f:
        for row in csv.DictReader(f):
            if row["layer"] == "fc":
                u = int(row["unit"])
                if u in CLASSES:
                    out[u] = row["description"].strip()
    return out


def _clean_netdissect(label: str) -> str:
    label = re.sub(r"-[a-z]$", "", label)        # strip -s/-c/-i scene/color suffix
    return label.replace("_", " ").replace("-", " ").strip()


def load_netdissect():
    """NetDissect fc: 1-indexed (unit = class+1); concept in 'label' column."""
    out = {}
    with open(DATA / "NetDissect_results/resnet50_imagenet_fc.csv") as f:
        for row in csv.DictReader(f):
            cls = int(row["unit"]) - 1
            if cls in CLASSES:
                out[cls] = _clean_netdissect(row["label"].strip())
    return out


METHODS = {"MILAN": load_milan(), "DnD": load_dnd(), "NetDissect": load_netdissect()}

# Build the aligned table.
table = pd.DataFrame({"class_idx": IDX,
                      "class_name": [CLASSES[i] for i in IDX]})
for m, d in METHODS.items():
    table[m] = [d.get(i, "") for i in IDX]
table.to_csv(OUT / "aligned_descriptions.csv", index=False)
print("=== Aligned descriptions ===")
print(table.to_string(index=False))

# ---------------------------------------------------------------- BERTScore helpers
def bscore(cands, refs):
    """Return F1 array for candidate/ref string lists (rescaled, en)."""
    P, R, F = bertscore(cands, refs, lang="en", rescale_with_baseline=True, verbose=False)
    return F.numpy()

# (a) method vs ground-truth class name
refs = [CLASSES[i] for i in IDX]
vs_gt = {}
for m in METHODS:
    cands = [METHODS[m][i] for i in IDX]
    vs_gt[m] = bscore(cands, refs)
gt_df = pd.DataFrame(vs_gt, index=[CLASSES[i] for i in IDX])
gt_df.to_csv(OUT / "bertscore_vs_groundtruth.csv")
print("\n=== BERTScore F1 vs ground-truth class name (per class) ===")
print(gt_df.round(3).to_string())
print("\nMean F1 vs ground truth:")
print(gt_df.mean().round(3).to_string())

# (b) pairwise agreement between methods (mean F1 over the 6 neurons)
mnames = list(METHODS.keys())
pair = pd.DataFrame(np.eye(len(mnames)), index=mnames, columns=mnames)
for a, b in itertools.combinations(mnames, 2):
    ca = [METHODS[a][i] for i in IDX]
    cb = [METHODS[b][i] for i in IDX]
    f = bscore(ca, cb).mean()
    pair.loc[a, b] = pair.loc[b, a] = f
pair.to_csv(OUT / "bertscore_pairwise.csv")
print("\n=== Pairwise BERTScore F1 (method agreement, mean over 6 neurons) ===")
print(pair.round(3).to_string())

# ---------------------------------------------------------------- plots
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# Plot 1: mean F1 vs ground truth (bar)
fig, ax = plt.subplots(figsize=(5.5, 4))
means = gt_df.mean()
errs = gt_df.std()
bars = ax.bar(means.index, means.values, yerr=errs.values, capsize=4,
              color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("BERTScore F1 (vs ground-truth class)")
ax.set_title("Faithfulness to true class\nResNet-50 / ImageNet, 6 class neurons")
for b, v in zip(bars, means.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
ax.set_ylim(0, max(means.values + errs.values) * 1.25)
fig.tight_layout()
fig.savefig(OUT / "fig1_mean_f1_vs_groundtruth.png")

# Plot 2: per-class grouped bars vs ground truth
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(IDX))
w = 0.26
for k, m in enumerate(mnames):
    ax.bar(x + (k - 1) * w, gt_df[m].values, w, label=m)
ax.set_xticks(x)
ax.set_xticklabels([CLASSES[i] for i in IDX], rotation=25, ha="right")
ax.set_ylabel("BERTScore F1 (vs ground truth)")
ax.set_title("Per-class faithfulness by method")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig2_per_class_f1.png")

# Plot 3: pairwise agreement heatmap
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(pair.values, cmap="viridis", vmin=pair.values.min(), vmax=1.0)
ax.set_xticks(range(len(mnames)))
ax.set_yticks(range(len(mnames)))
ax.set_xticklabels(mnames)
ax.set_yticklabels(mnames)
for i in range(len(mnames)):
    for j in range(len(mnames)):
        ax.text(j, i, f"{pair.values[i, j]:.2f}", ha="center", va="center",
                color="white" if pair.values[i, j] < 0.6 else "black")
ax.set_title("Pairwise method agreement\n(mean BERTScore F1 over 6 neurons)")
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT / "fig3_pairwise_agreement.png")

print("\nSaved tables + 3 figures to", OUT)
