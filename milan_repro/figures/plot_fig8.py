"""Figure 8: adversarial-test accuracy vs. number of ablated neurons.

Three lines:
  - text-sorted: MILAN-identified text neurons, ordered by least val-acc impact
  - sort-all:    all neurons, ordered by least val-acc impact (paper baseline)
  - random:      average over `n_random_trials`, with shaded ±1 std band

Plus a horizontal dashed line for the (untouched) clean-test accuracy.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(ablation_csv: Path, out_path: Path, title: str = None) -> Path:
    df = pd.read_csv(ablation_csv)

    baseline = df[df["mode"] == "baseline"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))

    for mode, color in (("text-sorted", "tab:blue"),
                        ("sort-all", "tab:orange")):
        sub = df[df["mode"] == mode].sort_values("n_ablated")
        ax.plot(sub["n_ablated"], sub["adv_acc"], "-o",
                label=mode, color=color, markersize=3, linewidth=1.5)

    rand = df[df["mode"] == "random"]
    if not rand.empty:
        agg = rand.groupby("n_ablated")["adv_acc"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["n_ablated"], agg["mean"], "-",
                label="random (mean)", color="gray", linewidth=1.2)
        if agg["std"].notna().any():
            ax.fill_between(agg["n_ablated"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            color="gray", alpha=0.2)

    ax.axhline(baseline["adv_acc"], color="black", linestyle=":",
               linewidth=1, label=f"no ablation ({baseline['adv_acc']:.2f})")
    ax.axhline(baseline["clean_acc"], color="green", linestyle="--",
               linewidth=1, label=f"clean val ({baseline['clean_acc']:.2f})")

    ax.set_xlabel("# neurons ablated")
    ax.set_ylabel("adversarial test accuracy")
    ax.set_title(title or
                 "Robustness vs. ablation (MILAN Section 7 reproduction)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
    ap.add_argument("--ablation-csv", type=Path,
                    default=base_results / "ablation_curve.csv")
    ap.add_argument("--out", type=Path,
                    default=base_results / "figs" / "fig8.pdf")
    ap.add_argument("--title", default=None,
                    help="plot title (defaults to a generic Section-7 title)")
    args = ap.parse_args()
    plot(args.ablation_csv, args.out, title=args.title)


if __name__ == "__main__":
    main()
