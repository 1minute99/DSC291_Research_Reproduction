"""Generate the four new figures referenced by the presentation deck.

Outputs to results/figs/slides/:
  - spurious_dataset_grid.png
  - arch_text_neuron_bar.png
  - summary_metrics_table.png
  - milan_pipeline_diagram.png
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figs" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def _pick_image(folder: Path) -> Path:
    files = sorted(folder.glob("*.JPEG"))
    return files[len(files) // 2] if files else None


def make_spurious_grid():
    rng = random.Random(42)
    chosen_wnids = rng.sample(list(CLASS_NAMES.keys()), 4)

    clean_root = ROOT / "data" / "imagenette" / "imagenette2-320" / "train"
    spurious_train_root = ROOT / "data" / "imagenet-spurious-text" / "50pct" / "train"
    adv_root = ROOT / "data" / "imagenet-spurious-text" / "50pct" / "test_strict"

    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    row_titles = [
        "Clean Imagenette\n(no overlay)",
        "Training set\n— class-name overlay",
        "Adversarial test\n(wrong-class overlay)",
    ]
    for r, root in enumerate([clean_root, spurious_train_root, adv_root]):
        for c, wnid in enumerate(chosen_wnids):
            ax = axes[r][c]
            folder = root / wnid
            img_path = _pick_image(folder)
            if img_path is None:
                ax.axis("off")
                continue
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(CLASS_NAMES[wnid], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=11, rotation=0,
                              ha="right", va="center", labelpad=70)

    fig.suptitle("Spurious-Text Imagenette: how the corner-text shortcut is created",
                 fontsize=13, y=0.995)
    plt.tight_layout(rect=(0.08, 0, 1, 0.97))
    out = OUT / "spurious_dataset_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def make_arch_bar():
    archs = ["ResNet18\n(trained)", "VGG16\n(trained)", "InceptionV3\n(trained)*", "CLIP ViT-B/32\n(zero-shot)"]
    fracs = [21.1, 24.3, None, 7.9]
    n_total = [1024, 1472, "~3936", 768]
    n_text = [216, 357, "—", 61]
    colors = ["#1f77b4", "#ff7f0e", "#bcbd22", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = list(range(len(archs)))
    plotted_fracs = [f if f is not None else 0 for f in fracs]
    bars = ax.bar(x, plotted_fracs, color=colors, edgecolor="black", linewidth=0.7)
    for i, (b, f) in enumerate(zip(bars, fracs)):
        if f is None:
            ax.text(b.get_x() + b.get_width() / 2, 3,
                    "extension\n(in progress)",
                    ha="center", va="bottom", fontsize=9, style="italic", color="gray")
        else:
            ax.text(b.get_x() + b.get_width() / 2, f + 1.2,
                    f"{f:.1f}%\n({n_text[i]}/{n_total[i]})",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=10)
    ax.set_ylabel("Text-selective neuron fraction (%)", fontsize=11)
    ax.set_ylim(0, 70)
    ax.set_title("Text-neuron fraction across architectures\n(higher = more reliance on text shortcut)",
                 fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = OUT / "arch_text_neuron_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def make_summary_table():
    rows = [
        ["Baseline reproduction (ResNet18)",
         "75.4%", "51.5%", "216 / 1024 (21.1%)",
         "Text-editing recovers robustness"],
        ["Exp 1 — VGG16 generalization",
         "69.0%", "19.8%", "357 / 1472 (24.3%)",
         "More reliant on shortcut"],
        ["Exp 2 — ResNet18 layer-depth",
         "—", "—", "9→27→37→77→66",
         "Peaks at layer1 (3.0×), then declines"],
        ["Exp 3 — CLIP ViT-B/32 (zero-shot)",
         "—", "—", "61 / 768 (7.9% max)",
         "≈2.7× more robust"],
    ]
    cols = ["Experiment", "Clean val", "Adv test", "Text neurons", "Key finding"]

    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols,
                     cellLoc="center", colLoc="center", loc="center",
                     colWidths=[0.27, 0.10, 0.10, 0.22, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    for j in range(len(cols)):
        cell = table[(0, j)]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", weight="bold")
    row_colors = ["#f7f9fb", "#eaf3fb", "#fff4e7", "#eafaf0"]
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            table[(i, j)].set_facecolor(row_colors[i - 1])
            if j == 0:
                table[(i, j)].set_text_props(weight="bold")

    fig.suptitle("Summary of reproduction + extensions",
                 fontsize=14, y=0.98)
    out = OUT / "summary_metrics_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(16, 3.6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    box_specs = [
        ("1. Train classifier\non spurious data", "#cfe2f3"),
        ("2. Extract top-k\nactivating exemplars", "#d0e0e3"),
        ("3. MILAN decoder\n→ caption per neuron", "#d9ead3"),
        ("4. Flag text-selective\nneurons (regex)", "#fce5cd"),
        ("5. Ablate flagged →\nmeasure adv-acc gain", "#f4cccc"),
    ]
    box_width, box_height = 2.7, 1.4
    gap = 0.6
    start_x = 0.4
    box_y = 1.0
    centers = []
    xs = []
    for i, (label, color) in enumerate(box_specs):
        x = start_x + i * (box_width + gap)
        xs.append(x)
        patch = FancyBboxPatch((x, box_y), box_width, box_height,
                               boxstyle="round,pad=0.05,rounding_size=0.15",
                               facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + box_width / 2, box_y + box_height / 2, label,
                ha="center", va="center", fontsize=10.5, weight="bold")
        centers.append(x + box_width)

    for i in range(len(box_specs) - 1):
        start = (centers[i], box_y + box_height / 2)
        end = (xs[i + 1], box_y + box_height / 2)
        arrow = FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=18, color="#333333", linewidth=1.4)
        ax.add_patch(arrow)

    mid_x = (start_x + (start_x + len(box_specs) * (box_width + gap) - gap)) / 2
    ax.text(mid_x, 3.1, "MILAN pipeline for editing spurious features",
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(mid_x, 0.55,
            "(steps 1–5 form one self-contained loop; we extend step 1 with VGG16 / CLIP, and add per-layer analysis at step 4)",
            ha="center", va="center", fontsize=10, style="italic", color="#555555")

    out = OUT / "milan_pipeline_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    make_spurious_grid()
    make_arch_bar()
    make_summary_table()
    make_pipeline_diagram()
