"""Figure 7 (qualitative): top exemplars + MILAN description for text-selective neurons.

Picks the first N text-flagged neurons and lays out their top-k masked
exemplars in a grid with the description as a caption.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from milan_repro.milan_glue import upstream  # noqa: F401
from src import milannotations


def plot(dissect_dir: Path, descriptions_csv: Path, out_path: Path,
         n_neurons: int = 8, top_k: int = 5) -> Path:
    """Draw the qualitative grid and save to `out_path`."""
    df = pd.read_csv(descriptions_csv)
    if "is_text_neuron" not in df.columns:
        from milan_repro.editing.identify_text_neurons import text_neuron_mask
        df["is_text_neuron"] = text_neuron_mask(df["description"])
    text_df = df[df["is_text_neuron"]].head(n_neurons)
    if text_df.empty:
        raise ValueError("no text neurons flagged; check description quality")

    dissected = milannotations.TopImagesDataset(dissect_dir)

    fig, axes = plt.subplots(len(text_df), top_k,
                             figsize=(2.0 * top_k, 2.2 * len(text_df)))
    if len(text_df) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (_, row) in enumerate(text_df.iterrows()):
        sample = dissected[int(row["unit_index"])]
        # `sample.images` shape: (top_k, C, H, W) in pt-normalised range.
        images = sample.images[:top_k]
        masks = sample.masks[:top_k]
        for k in range(top_k):
            ax = axes[row_idx, k]
            img = images[k].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            mask = masks[k].squeeze(0).cpu().numpy()
            ax.imshow(img)
            ax.imshow(mask, alpha=0.4, cmap="Reds")
            ax.set_axis_off()
        axes[row_idx, 0].set_ylabel(
            f"{row['layer']}.{row['channel']}\n\"{row['description'][:42]}\"",
            rotation=0, ha="right", va="center", fontsize=8)

    fig.suptitle("MILAN-identified text neurons (top exemplars)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
    ap.add_argument("--dissect-dir", type=Path,
                    default=base_results / "edit" / "imagenet-spurious-text"
                            / "resnet18_spurious-50pct")
    ap.add_argument("--descriptions", type=Path,
                    default=base_results / "descriptions.csv")
    ap.add_argument("--out", type=Path,
                    default=base_results / "figs" / "fig7.pdf")
    ap.add_argument("--n-neurons", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    plot(args.dissect_dir, args.descriptions, args.out,
         n_neurons=args.n_neurons, top_k=args.top_k)


if __name__ == "__main__":
    main()
