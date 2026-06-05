"""Figure: layer-depth analysis of MILAN-identified text neurons.

Proposal new experiment 3 — "Rerun the BERTScore evaluation or qualitative
visual comparison broken down by layer depth of model architecture."

Produces two sub-figures:
  (a) Bar chart: fraction of text neurons per layer (conv1, layer1..4).
  (b) Heatmap / bar chart: description-word diversity per layer
      (unique word types / total words in captions).

Reads from the already-generated `descriptions_annotated.csv`.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Layer display order (shallowest → deepest)
LAYER_ORDER = ["conv1", "layer1", "layer2", "layer3", "layer4"]
LAYER_LABELS = {
    "conv1":  "conv1\n(shallow)",
    "layer1": "layer1",
    "layer2": "layer2",
    "layer3": "layer3",
    "layer4": "layer4\n(deep)",
}


def _word_diversity(descriptions: pd.Series) -> float:
    """Unique-word ratio: unique tokens / total tokens in the description set."""
    all_words = []
    for desc in descriptions.dropna():
        all_words.extend(str(desc).lower().split())
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def _top_words(descriptions: pd.Series, n: int = 5) -> list[str]:
    """Most common content words across descriptions in this layer."""
    stopwords = {"a", "an", "the", "of", "in", "on", "with", "and", "or",
                 "to", "at", "by", "for", "is", "are", "that", "this"}
    counter: Counter = Counter()
    for desc in descriptions.dropna():
        for w in str(desc).lower().split():
            w = w.strip(".,;:")
            if w and w not in stopwords and len(w) > 2:
                counter[w] += 1
    return [w for w, _ in counter.most_common(n)]


def plot(descriptions_csv: Path, out_dir: Path) -> None:
    """Generate two layer-analysis figures and save to `out_dir`."""
    df = pd.read_csv(descriptions_csv)

    if "is_text_neuron" not in df.columns:
        from milan_repro.editing.identify_text_neurons import text_neuron_mask
        df["is_text_neuron"] = text_neuron_mask(df["description"])

    # Keep only layers we know about; sort by canonical order.
    df["layer"] = df["layer"].astype(str)
    layers_present = [l for l in LAYER_ORDER if l in df["layer"].unique()]

    # ── Per-layer stats ──────────────────────────────────────────────────────
    stats = []
    for layer in layers_present:
        sub = df[df["layer"] == layer]
        n_total = len(sub)
        n_text = int(sub["is_text_neuron"].sum())
        diversity = _word_diversity(sub["description"])
        top5 = _top_words(sub["description"], n=5)
        stats.append({
            "layer": layer,
            "n_total": n_total,
            "n_text": n_text,
            "frac_text": n_text / max(1, n_total),
            "diversity": diversity,
            "top5": ", ".join(top5),
        })
    stats_df = pd.DataFrame(stats)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_dir / "layer_analysis.csv", index=False)
    print(stats_df[["layer", "n_total", "n_text", "frac_text", "diversity",
                     "top5"]].to_string(index=False))

    # ── Figure (a): text-neuron fraction per layer ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    x = np.arange(len(layers_present))
    bars = ax.bar(x, stats_df["frac_text"] * 100,
                  color=plt.cm.Blues(np.linspace(0.35, 0.85, len(layers_present))),
                  edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, stats_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{row.n_text}/{row.n_total}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS.get(l, l) for l in layers_present])
    ax.set_ylabel("Text neurons (%)")
    ax.set_title("(a) MILAN text-neuron fraction by layer depth")
    ax.set_ylim(0, min(100, stats_df["frac_text"].max() * 130))
    ax.grid(axis="y", alpha=0.3)

    # ── Figure (b): description word-diversity per layer ─────────────────────
    ax = axes[1]
    bars2 = ax.bar(x, stats_df["diversity"],
                   color=plt.cm.Oranges(np.linspace(0.35, 0.85, len(layers_present))),
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars2, stats_df["diversity"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS.get(l, l) for l in layers_present])
    ax.set_ylabel("Word diversity (unique / total)")
    ax.set_title("(b) MILAN description diversity by layer depth")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Layer-depth analysis of MILAN descriptions (ResNet18 spurious-text)",
                 fontsize=11)
    fig.tight_layout()

    out_path = out_dir / "fig_layer_analysis.pdf"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    # Also save PNG for easy viewing
    fig.savefig(out_dir / "fig_layer_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
    ap = argparse.ArgumentParser(
        description="Layer-depth analysis of MILAN text neurons.")
    ap.add_argument("--descriptions", type=Path,
                    default=base_results / "descriptions_annotated.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=base_results / "figs")
    args = ap.parse_args()
    plot(args.descriptions, args.out_dir)


if __name__ == "__main__":
    main()
