"""Generate MILAN descriptions for CLIP ViT-B/32 transformer blocks.

Reads the images.npy / masks.npy produced by run_clip_exemplars.py and
feeds them through the pretrained MILAN decoder, producing one natural-
language description per (layer, unit) pair.

Usage:
    python -m milan_repro.milan_glue.run_clip_descriptions
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from milan_repro.milan_glue import upstream  # noqa: F401
from milan_repro.milan_glue.clip_glue import CLIP_LAYERS_SUBSET

import torch
from src import milan, milannotations


def run(dissect_dir: Path, out_csv: Path,
        milan_key: str = "base",
        device: str = "cuda",
        strategy: str = "rerank",
        beam_size: int = 50,
        temperature: float = 0.2) -> Path:
    """Caption every CLIP unit in `dissect_dir` and write CSV."""
    dissected = milannotations.TopImagesDataset(dissect_dir)
    decoder = milan.pretrained(milan_key, map_location=device)

    print(f"decoding descriptions for {len(dissected)} CLIP units...")
    descriptions = decoder.predict(
        dissected,
        strategy=strategy,
        temperature=temperature,
        beam_size=beam_size,
        device=device,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_index", "layer", "channel", "description"])
        for i in range(len(dissected)):
            layer, channel = dissected.unit(i)
            writer.writerow([i, layer, channel, descriptions[i]])

    print(f"wrote {out_csv}")
    return out_csv


def main() -> None:
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--dissect-dir", type=Path,
                    default=base_results / "edit" / "clip-vitb32")
    ap.add_argument("--out", type=Path,
                    default=base_results / "clip_descriptions.csv")
    ap.add_argument("--milan", default="base")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run(args.dissect_dir, args.out, milan_key=args.milan, device=args.device)


if __name__ == "__main__":
    main()
