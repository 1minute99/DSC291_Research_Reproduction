"""fig1 at scale: faithfulness vs true class over ALL shared fc neurons (no error bars)."""
import re, csv
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from bert_score import score as bertscore

DATA = Path("/Users/JaredWu/Desktop/Describe-and-Dissect/data")
OUT = Path("/Users/JaredWu/Desktop/DSC291_proj/comparison/outputs")

# Ground-truth class names: line i (1-indexed) -> class i-1; take name before first comma.
names = {}
with open(DATA / "imagenet_labels.txt") as f:
    for i, line in enumerate(f):
        names[i] = line.split(",")[0].strip()

def load_dnd():
    out = {}
    with open(DATA / "DnD_results/rn50_results/resnet50_imagenet_broden_fc.csv") as f:
        for r in csv.DictReader(f):
            try: out[int(r["Neuron ID"])] = r["label"].strip()
            except ValueError: pass
    return out

def load_milan():
    out = {}
    with open(DATA / "MILAN_results/m_base_resnet50_imagenet.csv") as f:
        for r in csv.DictReader(f):
            if r["layer"] == "fc": out[int(r["unit"])] = r["description"].strip()
    return out

def _clean(l): return re.sub(r"-[a-z]$", "", l).replace("_", " ").replace("-", " ").strip()
def load_netdissect():
    out = {}
    with open(DATA / "NetDissect_results/resnet50_imagenet_fc.csv") as f:
        for r in csv.DictReader(f): out[int(r["unit"]) - 1] = _clean(r["label"].strip())
    return out

METHODS = {"MILAN": load_milan(), "DnD": load_dnd(), "NetDissect": load_netdissect()}
# shared indices that also have a ground-truth name, descriptions non-empty
idx = sorted(set(names) & set.intersection(*[set(d) for d in METHODS.values()]))
idx = [i for i in idx if all(METHODS[m].get(i, "").strip() for m in METHODS)]
print(f"Scoring {len(idx)} shared fc neurons across {len(METHODS)} methods")

refs = [names[i] for i in idx]
means = {}
per_class = {}
for m, d in METHODS.items():
    cands = [d[i] for i in idx]
    P, R, F = bertscore(cands, refs, lang="en", rescale_with_baseline=True, verbose=True)
    per_class[m] = F.numpy()
    means[m] = float(F.mean())

pd.DataFrame(per_class, index=[names[i] for i in idx]).to_csv(OUT / "bertscore_vs_groundtruth_full.csv")
print("\nMean BERTScore-F1 vs ground truth (n=%d):" % len(idx))
for m, v in means.items(): print(f"  {m:11s} {v:.3f}")

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
fig, ax = plt.subplots(figsize=(5.5, 4))
order = list(METHODS.keys())
bars = ax.bar(order, [means[m] for m in order], color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("BERTScore F1 (vs ground-truth class)")
ax.set_title(f"Faithfulness to true class\nResNet-50 / ImageNet, {len(idx)} class neurons")
for b, m in zip(bars, order):
    ax.text(b.get_x() + b.get_width()/2, means[m] + 0.003, f"{means[m]:.3f}", ha="center")
ax.set_ylim(0, max(means.values()) * 1.25)
fig.tight_layout()
fig.savefig(OUT / "fig1_mean_f1_vs_groundtruth_full.png")
print("Saved", OUT / "fig1_mean_f1_vs_groundtruth_full.png")
