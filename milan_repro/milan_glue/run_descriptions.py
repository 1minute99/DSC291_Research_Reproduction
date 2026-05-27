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
        beam_size: int = 50, temperature: float = 0.2,
        layer_by_layer: bool = False) -> Path:
    """Caption every unit in `dissect_dir` and write CSV at `out_csv`."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if layer_by_layer:
        # Process one layer at a time to stay within container memory limits.
        # Each layer's TopImagesDataset is loaded, decoded, then released.
        layer_dirs = sorted([d for d in dissect_dir.iterdir()
                             if d.is_dir() and (d / "images.npy").exists()])
        decoder = milan.pretrained(milan_key, map_location=device)

        # Open CSV (append mode so we can resume if killed mid-run)
        existing_layers: set[str] = set()
        if out_csv.exists():
            import pandas as _pd
            existing_layers = set(_pd.read_csv(out_csv)["layer"].unique())
            print(f"resuming: already have layers {existing_layers}")

        unit_index = 0
        mode = "a" if out_csv.exists() else "w"
        with out_csv.open(mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(["unit_index", "layer", "channel", "description"])
            for layer_dir in layer_dirs:
                layer_name = layer_dir.name
                if layer_name in existing_layers:
                    # Count units to keep unit_index correct
                    import numpy as _np
                    imgs = _np.load(layer_dir / "images.npy", mmap_mode="r")
                    unit_index += imgs.shape[0]
                    print(f"[{layer_name}] already done, skipping")
                    continue
                print(f"[{layer_name}] loading exemplars...")
                layer_ds = milannotations.TopImagesDataset(layer_dir)
                print(f"[{layer_name}] decoding {len(layer_ds)} units...")
                descs = decoder.predict(
                    layer_ds,
                    strategy=strategy,
                    temperature=temperature,
                    beam_size=beam_size,
                    device=device,
                )
                for i in range(len(layer_ds)):
                    layer, channel = layer_ds.unit(i)
                    writer.writerow([unit_index, layer, channel, descs[i]])
                    unit_index += 1
                f.flush()
                del layer_ds, descs
                import gc as _gc; _gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[{layer_name}] done")
        print(f"wrote {out_csv}")
        return out_csv

    # Default: load all layers at once (original behaviour)
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
    ap.add_argument("--layer-by-layer", action="store_true",
                    help="Process one layer at a time (use on memory-limited machines)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run(args.dissect_dir, args.out, milan_key=args.milan, device=args.device,
        layer_by_layer=args.layer_by_layer)


if __name__ == "__main__":
    main()
