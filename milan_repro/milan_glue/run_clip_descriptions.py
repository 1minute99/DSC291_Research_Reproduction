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
import gc
import os
from collections import namedtuple
from pathlib import Path

from milan_repro.milan_glue import upstream  # noqa: F401
from milan_repro.milan_glue.clip_glue import CLIP_LAYERS_SUBSET

import numpy as np
import torch
from src import milan

_TopImages = namedtuple("TopImages", ["layer", "unit", "images", "masks"])


# Minimal dataset backed by mmap — never loads the full array into RAM
class _MmapLayerDataset:
    """Wraps images.npy / masks.npy with mmap so only requested units load."""

    def __init__(self, layer_dir: Path, layer_name: str) -> None:
        self._images = np.load(layer_dir / "images.npy", mmap_mode="r")
        self._masks  = np.load(layer_dir / "masks.npy",  mmap_mode="r")
        self._layer  = layer_name
        self._n      = self._images.shape[0]

    def __len__(self) -> int:
        return self._n

    def unit(self, i: int):
        return (self._layer, i)

    def __getitem__(self, i: int):
        imgs = torch.from_numpy(self._images[i].copy()).float() / 255.0
        msks = torch.from_numpy(self._masks[i].copy())
        return _TopImages(layer=self._layer, unit=i, images=imgs, masks=msks)


def run(dissect_dir: Path, out_csv: Path,
        milan_key: str = "base",
        device: str = "cuda",
        strategy: str = "rerank",
        beam_size: int = 50,
        temperature: float = 0.2,
        layer_by_layer: bool = True) -> Path:
    """Caption every CLIP unit in `dissect_dir` and write CSV."""
    layer_names = sorted([d.name for d in dissect_dir.iterdir()
                          if d.is_dir() and (d / "images.npy").exists()])
    if not layer_names:
        raise FileNotFoundError(f"No exemplar layers found in {dissect_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    decoder = milan.pretrained(milan_key, map_location=device)

    rows = []
    unit_index = 0
    for layer_name in layer_names:
        layer_dir = dissect_dir / layer_name
        layer_ds = _MmapLayerDataset(layer_dir, layer_name)
        print(f"[{layer_name}] decoding {len(layer_ds)} units...")
        descriptions = decoder.predict(
            layer_ds,
            strategy=strategy,
            temperature=temperature,
            beam_size=beam_size,
            device=device,
        )
        for i in range(len(layer_ds)):
            layer, channel = layer_ds.unit(i)
            rows.append([unit_index, layer, channel, descriptions[i]])
            unit_index += 1
        del layer_ds, descriptions
        gc.collect()
        print(f"[{layer_name}] done")

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_index", "layer", "channel", "description"])
        writer.writerows(rows)

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
    ap.add_argument("--layer-by-layer", action="store_true", default=True)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run(args.dissect_dir, args.out, milan_key=args.milan, device=args.device,
        layer_by_layer=args.layer_by_layer)


if __name__ == "__main__":
    main()
