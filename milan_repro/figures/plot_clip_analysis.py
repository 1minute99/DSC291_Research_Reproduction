"""Figure: CLIP ViT-B/32 transformer-block text-neuron analysis.

Proposal new experiment 2 — "Applying MILAN to CLIP to observe which
transformer blocks are sensitive to text features."

Produces:
  (a) Bar chart: fraction of text neurons per transformer block (depth 0→11)
  (b) Top-5 exemplar grid for the most text-selective block
  (c) Word-cloud style top-words per block (text-bar chart)

Usage:
    python -m milan_repro.figures.plot_clip_analysis
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milan_repro.editing.identify_text_neurons import text_neuron_mask

TARGET_WORDS = ("word", "text", "letter")


def _block_index(layer_name: str) -> int:
    """Extract block index from 'visual_transformer_resblocks_N'."""
    parts = layer_name.replace(".", "_").split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return -1


def _top_words(descriptions: pd.Series, n: int = 8) -> list[str]:
    stopwords = {"a", "an", "the", "of", "in", "on", "with", "and", "or",
                 "to", "at", "by", "for", "is", "are", "that", "this", "it"}
    counter: Counter = Counter()
    for desc in descriptions.dropna():
        for w in str(desc).lower().split():
            w = w.strip(".,;:")
            if w and w not in stopwords and len(w) > 2:
                counter[w] += 1
    return [w for w, _ in counter.most_common(n)]


def plot(descriptions_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(descriptions_csv)

    if "is_text_neuron" not in df.columns:
        df["is_text_neuron"] = text_neuron_mask(df["description"], TARGET_WORDS)

    # Map layer name → block index for sorting
    df["block_idx"] = df["layer"].apply(_block_index)
    df_sorted = df.sort_values("block_idx")
    blocks = sorted(df_sorted["block_idx"].unique())

    stats = []
    for b in blocks:
        sub = df_sorted[df_sorted["block_idx"] == b]
        n_total = len(sub)
        n_text = int(sub["is_text_neuron"].sum())
        top5 = _top_words(sub["description"], n=5)
        stats.append({
            "block": b,
            "layer": sub["layer"].iloc[0],
            "n_total": n_total,
            "n_text": n_text,
            "frac_text": n_text / max(1, n_total),
            "top5": ", ".join(top5),
        })
    stats_df = pd.DataFrame(stats)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_dir / "clip_layer_analysis.csv", index=False)
    print(stats_df[["block", "n_total", "n_text", "frac_text", "top5"]]
          .to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── (a) Text-neuron fraction per block ───────────────────────────────────
    ax = axes[0]
    x = np.arange(len(blocks))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(blocks)))
    bars = ax.bar(x, stats_df["frac_text"] * 100, color=colors,
                  edgecolor="white", linewidth=0.7)
    for bar, row in zip(bars, stats_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{row.n_text}/{row.n_total}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {b}" for b in blocks], rotation=35, ha="right")
    ax.set_ylabel("Text neurons (%)")
    ax.set_title("(a) CLIP ViT-B/32: text-neuron fraction by transformer block")
    ax.grid(axis="y", alpha=0.3)

    # ── (b) Top-5 words per block (stacked label) ────────────────────────────
    ax = axes[1]
    for i, row in enumerate(stats_df.itertuples()):
        ax.text(0.02, 1 - (i + 0.5) / len(blocks),
                f"Block {row.block:2d}: {row.top5}",
                transform=ax.transAxes, fontsize=8,
                va="center", ha="left",
                color="darkred" if row.frac_text > 0.3 else "black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) Top-5 words per block\n(red = >30% text neurons)")

    fig.suptitle("CLIP ViT-B/32 — MILAN text-neuron analysis by transformer depth",
                 fontsize=11)
    fig.tight_layout()

    out_path = out_dir / "fig_clip_analysis.pdf"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "fig_clip_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", type=Path,
                    default=base_results / "clip_descriptions.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=base_results / "figs")
    args = ap.parse_args()
    plot(args.descriptions, args.out_dir)


if __name__ == "__main__":
    main()
