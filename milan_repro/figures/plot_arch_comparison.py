"""Figure: ablation curves across architectures (ResNet18 vs VGG16 vs InceptionV3).

Proposal new experiment 1 — "Test generalization across extended type of
architecture, e.g. VGGNet and Inception."

Reads one ablation_curve CSV per architecture and overlays the `text-sorted`
and `sort-all` curves so the reader can compare MILAN's editing effectiveness
across different network families.

Usage:
    python -m milan_repro.figures.plot_arch_comparison
    python -m milan_repro.figures.plot_arch_comparison \
        --csvs results/ablation_curve.csv results/vgg16_ablation_curve.csv \
        --labels ResNet18 VGG16
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCH_COLORS = {
    "ResNet18":    ("tab:blue",   "tab:cyan"),
    "VGG16":       ("tab:orange", "tab:red"),
    "InceptionV3": ("tab:green",  "tab:olive"),
}
LINESTYLES = {
    "text-sorted": "-",
    "sort-all":    "--",
}


def plot(csv_paths: list[Path], labels: list[str], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for csv_path, label in zip(csv_paths, labels):
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found — skipping {label}")
            continue

        df = pd.read_csv(csv_path)
        colors = ARCH_COLORS.get(label, ("tab:purple", "tab:pink"))
        baseline = df[df["mode"] == "baseline"].iloc[0]

        for mode, color in zip(("text-sorted", "sort-all"), colors):
            sub = df[df["mode"] == mode].sort_values("n_ablated")
            if sub.empty:
                continue
            ls = LINESTYLES[mode]
            ax.plot(sub["n_ablated"], sub["adv_acc"] * 100,
                    linestyle=ls, color=color, linewidth=1.8,
                    label=f"{label} — {mode}")

        # Baseline horizontal dotted line per arch
        ax.axhline(baseline["adv_acc"] * 100, linestyle=":",
                   color=colors[0], linewidth=0.9, alpha=0.6,
                   label=f"{label} baseline ({baseline['adv_acc']*100:.1f}%)")

    ax.set_xlabel("# neurons ablated", fontsize=11)
    ax.set_ylabel("Adversarial test accuracy (%)", fontsize=11)
    ax.set_title("MILAN editing effectiveness across architectures\n"
                 "(solid = text-sorted, dashed = sort-all)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))

    default_csvs = [
        base_results / "ablation_curve.csv",
        base_results / "vgg16_ablation_curve.csv",
        base_results / "inception_v3_ablation_curve.csv",
    ]
    default_labels = ["ResNet18", "VGG16", "InceptionV3"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", type=Path, default=default_csvs)
    ap.add_argument("--labels", nargs="+", default=default_labels)
    ap.add_argument("--out", type=Path,
                    default=base_results / "figs" / "fig_arch_comparison.pdf")
    args = ap.parse_args()

    if len(args.csvs) != len(args.labels):
        raise ValueError("--csvs and --labels must have the same length")

    plot(args.csvs, args.labels, args.out)


if __name__ == "__main__":
    main()
