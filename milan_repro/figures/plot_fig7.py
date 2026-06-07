"""Figure 7 (qualitative): top exemplars + MILAN description for text-selective neurons.

Picks the first N text-flagged neurons and lays out their top-k masked
exemplars in a grid with the description as a caption.

For sharpness we reload each exemplar's ORIGINAL image (via the per-layer
ids.csv -> dataset index map) instead of the 224px thumbnails stored in
images.npy, then overlay the (upscaled) activation mask. Falls back to the
stored thumbnails if no --version-dir is given.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from milan_repro.milan_glue import upstream  # noqa: F401


def plot(dissect_dir: Path, descriptions_csv: Path, out_path: Path,
         n_neurons: int = 8, top_k: int = 5, version_dir: Path | None = None,
         disp: int = 384) -> Path:
    """Draw the qualitative grid and save to `out_path`."""
    df = pd.read_csv(descriptions_csv)
    if "is_text_neuron" not in df.columns:
        from milan_repro.editing.identify_text_neurons import text_neuron_mask
        df["is_text_neuron"] = text_neuron_mask(df["description"])
    text_df = df[df["is_text_neuron"]].head(n_neurons)
    if text_df.empty:
        raise ValueError("no text neurons flagged; check description quality")

    samples = None
    if version_dir is not None:
        from milan_repro.data.spurious_dataset import load as load_spurious
        train, _ = load_spurious(version_dir)
        samples = train.samples  # [(path, label), ...] indexed by global id

    fig, axes = plt.subplots(len(text_df), top_k,
                             figsize=(1.7 * top_k, 1.8 * len(text_df)))
    if len(text_df) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (_, row) in enumerate(text_df.iterrows()):
        layer, channel = row["layer"], int(row["channel"])
        masks_npy = np.load(dissect_dir / layer / "masks.npy", mmap_mode="r")
        ids = None
        images_npy = None
        if samples is not None:
            ids = pd.read_csv(dissect_dir / layer / "ids.csv", header=None).values
        else:
            images_npy = np.load(dissect_dir / layer / "images.npy", mmap_mode="r")
        for k in range(top_k):
            ax = axes[row_idx, k]
            if samples is not None:
                gidx = int(ids[channel, k])
                img = np.asarray(
                    Image.open(samples[gidx][0]).convert("RGB").resize((disp, disp))
                ) / 255.0
            else:
                img = images_npy[channel, k].transpose(1, 2, 0) / 255.0
            mask = np.asarray(masks_npy[channel, k, 0])              # 224x224
            mask = np.asarray(Image.fromarray(mask).resize((disp, disp)))
            ax.imshow(img, interpolation="lanczos")
            ax.imshow(mask, alpha=0.35, cmap="Reds", interpolation="lanczos")
            ax.set_axis_off()
        axes[row_idx, 0].set_ylabel(
            f"{layer}.{channel}\n\"{row['description'][:42]}\"",
            rotation=0, ha="right", va="center", fontsize=8)

    fig.suptitle("MILAN-identified text neurons (top exemplars)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
    base_data = Path(os.environ.get("MILAN_DATA_DIR", "./data"))
    ap.add_argument("--dissect-dir", type=Path,
                    default=base_results / "edit" / "imagenet-spurious-text"
                            / "resnet18_spurious-10pct")
    ap.add_argument("--descriptions", type=Path,
                    default=base_results / "rerun_10pct" / "descriptions_annotated.csv")
    ap.add_argument("--version-dir", type=Path,
                    default=base_data / "imagenet-spurious-text" / "10pct",
                    help="dataset dir for reloading full-res originals (sharper grid)")
    ap.add_argument("--out", type=Path,
                    default=base_results / "figs" / "fig7.png")
    ap.add_argument("--n-neurons", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--disp", type=int, default=384,
                    help="per-cell display resolution in px")
    args = ap.parse_args()
    plot(args.dissect_dir, args.descriptions, args.out,
         n_neurons=args.n_neurons, top_k=args.top_k,
         version_dir=args.version_dir, disp=args.disp)


if __name__ == "__main__":
    main()
