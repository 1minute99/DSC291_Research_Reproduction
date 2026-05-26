"""Generate MILAN natural-language descriptions for every dissected unit.

Reads the per-layer `images.npy`/`masks.npy` written by `run_exemplars.py`,
applies the pretrained MILAN decoder (`base`), and writes a CSV with one
row per (layer, channel) unit:

    unit_index, layer, channel, description
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from milan_repro.milan_glue import upstream  # noqa: F401

import torch
from src import milan, milannotations


def run(dissect_dir: Path, out_csv: Path, milan_key: str = "base",
        device: str = "cuda", strategy: str = "rerank",
        beam_size: int = 50, temperature: float = 0.2) -> Path:
    """Caption every unit in `dissect_dir` and write CSV at `out_csv`."""
    dissected = milannotations.TopImagesDataset(dissect_dir)
    decoder = milan.pretrained(milan_key, map_location=device)

    print(f"decoding descriptions for {len(dissected)} units...")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dissect-dir", type=Path,
                    default=Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
                            / "edit" / "imagenet-spurious-text"
                            / "resnet18_spurious-50pct")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
                            / "descriptions.csv")
    ap.add_argument("--milan", default="base")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run(args.dissect_dir, args.out, milan_key=args.milan, device=args.device)


if __name__ == "__main__":
    main()
