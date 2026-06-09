"""Deliverables across the available method x model x dataset data.

1. Faithfulness-to-true-class TABLE (mean BERTScore-F1).  Only resnet50/imagenet has
   >=2 methods + class names; other requested cells have no data / can't run here.
2. Per-class faithfulness TABLE (6 ImageNet subclasses x method, resnet50/imagenet).
3. Raw side-by-side description tables for instinctive comparison, across models/datasets.
"""
import re, csv
from pathlib import Path
import pandas as pd

DATA = Path("/Users/JaredWu/Desktop/Describe-and-Dissect/data")
OUT = Path("/Users/JaredWu/Desktop/DSC291_proj/comparison/outputs")
md = []  # markdown report lines

def emit(s=""): md.append(s); print(s)

# ---------- generic CSV row picker ----------
def rows(path):
    with open(DATA / path) as f:
        yield from csv.reader(f)

# ============================================================ Table 1
emit("## Table 1 - Faithfulness to true class (mean BERTScore-F1 vs ground-truth class)\n")
full = pd.read_csv(OUT / "bertscore_vs_groundtruth_full.csv", index_col=0)
m_rn50 = {k: round(float(full[k].mean()), 3) for k in ["DnD", "MILAN", "NetDissect"]}
t1 = pd.DataFrame(
    [["resnet50 / imagenet (fc, n=1000)", m_rn50["DnD"], m_rn50["MILAN"], m_rn50["NetDissect"]],
     ["resnet18 / places365 (fc)", "no data", "no fc layer", "1 method only"],
     ["alexnet / any", "no data", "no data", "no data"],
     ["resnet18 / imagenet", "no data", "no data", "no data"],
     ["resnet50 / places365", "no data", "no data", "no data"]],
    columns=["model / dataset", "DnD", "MILAN", "NetDissect"])
t1.to_csv(OUT / "table1_faithfulness.csv", index=False)
emit(t1.to_markdown(index=False)); emit()

# ============================================================ Table 2
emit("## Table 2 - Per-class faithfulness (6 ImageNet subclasses), resnet50/imagenet\n")
t2 = pd.read_csv(OUT / "bertscore_vs_groundtruth.csv", index_col=0).round(3)
t2.index.name = "class"
t2.loc["MEAN"] = t2.mean().round(3)
t2.to_csv(OUT / "table2_per_class_faithfulness.csv")
emit(t2.reset_index().to_markdown(index=False)); emit()

# ============================================================ Table 3 (raw, instinctive)
def clean_nd(l): return re.sub(r"-[a-z]$", "", l).replace("_", " ").replace("-", " ").strip()

# --- 3a: resnet50/imagenet fc (the 6 faithfulness classes) ---
emit("## Table 3a - RAW: resnet50 / imagenet, fc (class) neurons\n")
t3a = pd.read_csv(OUT / "aligned_descriptions.csv")[["class_name", "DnD", "MILAN", "NetDissect"]]
emit(t3a.to_markdown(index=False)); emit()

# --- 3b: resnet18/places365 layer4 (MILAN, NetDissect, CLIP-Dissect) ---
emit("## Table 3b - RAW: resnet18 / places365, layer4 (feature neurons; no class label)\n")
units_b = [0, 1, 2, 3, 4]
milan_b = {int(u): d for L, u, d in (r for r in rows("MILAN_results/m_base_resnet18_places365.csv") if r[0] == "layer4")}
clip_b = {int(u): d for L, u, d, *_ in (r for r in rows("CLIP_Dissect_results/resnet18_places_imagenet_broden.csv") if r[0] == "layer4")}
nd_b = {}
for i, r in enumerate(rows("NetDissect_results/resnet18_places365_layer4.csv")):
    if i == 0: continue
    nd_b[int(r[0]) - 1] = clean_nd(r[2])
t3b = pd.DataFrame([[u, milan_b.get(u, ""), nd_b.get(u, ""), clip_b.get(u, "")] for u in units_b],
                   columns=["unit", "MILAN", "NetDissect", "CLIP-Dissect"])
t3b.to_csv(OUT / "table3b_rn18_places_layer4.csv", index=False)
emit(t3b.to_markdown(index=False)); emit()

# --- 3c: resnet152/imagenet layer4 (DnD, MILAN, CLIP-Dissect) ---
emit("## Table 3c - RAW: resnet152 / imagenet, layer4 (feature neurons; no class label)\n")
dnd_c = [(int(r[0]), r[1]) for i, r in enumerate(rows("DnD_results/rn152_results/resnet152_imagenet_layer4.csv")) if i and r[0].isdigit()][:5]
milan_c = {int(u): d for L, u, d in (r for r in rows("MILAN_results/m_places365_resnet152_imagenet.csv") if r[0] == "layer4")}
clip_c = {int(u): d for L, u, d, *_ in (r for r in rows("CLIP_Dissect_results/resnet152_imagenet.csv") if r[0] == "layer4")}
t3c = pd.DataFrame([[u, lab, milan_c.get(u, ""), clip_c.get(u, "")] for u, lab in dnd_c],
                   columns=["unit", "DnD", "MILAN", "CLIP-Dissect"])
t3c.to_csv(OUT / "table3c_rn152_imagenet_layer4.csv", index=False)
emit(t3c.to_markdown(index=False)); emit()

(OUT / "GRID_TABLES.md").write_text("\n".join(md))
print("\nWrote tables + GRID_TABLES.md to", OUT)
