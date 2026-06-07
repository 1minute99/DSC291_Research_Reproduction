"""Overlay-strength sweep figure for the deck's 4th experiment.

Shows three ablation curves (50% / 20% / 5% training-overlay) side by side.
As the spurious shortcut weakens, the MILAN text-sorted curve separates from
the importance (sort-all) and random baselines — explaining why the headline
50% run looked null and reproducing the paper's editing claim at 5%.

Reads:
  results/ablation_curve.csv                  (50pct, deck baseline)
  results/rerun_20pct/ablation_curve.csv
  results/rerun_5pct/ablation_curve.csv
  + matching descriptions_annotated.csv for the text-neuron %.
Writes results/figs/slides/overlay_sweep.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# (label, ablation_csv, annotated_csv) — ordered strong -> weak shortcut.
PANELS = [
    ("50% overlay", RES / "ablation_curve.csv",
     RES / "descriptions_annotated.csv"),
    ("20% overlay", RES / "rerun_20pct" / "ablation_curve.csv",
     RES / "rerun_20pct" / "descriptions_annotated.csv"),
    ("10% overlay  ★", RES / "rerun_10pct" / "ablation_curve.csv",
     RES / "rerun_10pct" / "descriptions_annotated.csv"),
    ("5% overlay", RES / "rerun_5pct" / "ablation_curve.csv",
     RES / "rerun_5pct" / "descriptions_annotated.csv"),
]


def main() -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(15.5, 3.8), sharey=True)
    for ax, (label, abl, ann) in zip(axes, PANELS):
        df = pd.read_csv(abl)
        base = df[df["mode"] == "baseline"].iloc[0]
        pct_text = 100 * pd.read_csv(ann)["is_text_neuron"].mean()

        for mode, color in (("text-sorted", "tab:blue"),
                            ("sort-all", "tab:orange")):
            sub = df[df["mode"] == mode].sort_values("n_ablated")
            ax.plot(sub["n_ablated"], sub["adv_acc"], "-o", color=color,
                    markersize=2.5, linewidth=1.6, label=mode)
        rnd = df[df["mode"] == "random"]
        if not rnd.empty:
            agg = rnd.groupby("n_ablated")["adv_acc"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["n_ablated"], agg["mean"], "-", color="gray",
                    linewidth=1.2, label="random (mean)")
            ax.fill_between(agg["n_ablated"], agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"], color="gray", alpha=0.18)
        ax.axhline(base["adv_acc"], color="black", linestyle=":", linewidth=1,
                   label="no ablation")
        ax.set_title(f"{label}\nadv {base['adv_acc']*100:.0f}%  ·  "
                     f"{pct_text:.0f}% text neurons", fontsize=11)
        ax.set_xlabel("# neurons ablated", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("adversarial test accuracy", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Weaker shortcut → sparser text neurons → MILAN editing separates "
                 "from random/importance", fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = RES / "figs" / "slides" / "overlay_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
