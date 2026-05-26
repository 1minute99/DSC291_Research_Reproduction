"""Compute top-activating exemplars for every layer of our trained ResNet18.

Wraps upstream's `exemplars.discriminative`, which runs NetDissect-style
activation tallying and dumps `images.npy` + `masks.npy` per layer.

Output layout (matches what upstream's `TopImagesDataset` reads):

    $MILAN_RESULTS_DIR/edit/imagenet-spurious-text/
      resnet18_spurious-50pct/<layer>/{images,masks}.npy
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from milan_repro.milan_glue import upstream  # noqa: F401
from milan_repro.milan_glue.register import (MODEL_NAME, load_trained,
                                              model_configs)
from milan_repro.data.spurious_dataset import load as load_spurious

import torch
from src import exemplars
from src.deps.netdissect import renormalize
from src.exemplars.models import LAYERS


def run(version_dir: Path, ckpt_path: Path, out_dir: Path,
        device: str = "cuda", image_size: int = 224) -> Path:
    """Compute exemplars for every ResNet18 layer.

    Args:
        version_dir: e.g. data/imagenet-spurious-text/50pct
        ckpt_path: checkpoint produced by `train_resnet18.py`
        out_dir: dissection root; we write one subdir per layer
        device: cuda/cpu
        image_size: must match training (224)

    Returns:
        out_dir (for chaining into description step).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use upstream's hub purely to validate the config; the actual training
    # checkpoint is loaded separately and passed in as the live model.
    _ = model_configs()

    model = load_trained(ckpt_path, device=device)

    # Use the *validation portion* of the clean (no-overlay) data as the
    # probe set for activation max. The upstream uses the validation split
    # of the trained classifier's data; we mirror that.
    train, _test = load_spurious(version_dir, image_size=image_size)
    # Subsample for speed; full set works too.
    probe = train

    renormalizer = renormalize.renormalizer(source="imagenet", target="byte")

    for layer in LAYERS.RESNET18:
        layer_dir = out_dir / layer
        if (layer_dir / "images.npy").exists() and (layer_dir / "masks.npy").exists():
            print(f"[{layer}] cached, skipping")
            continue
        print(f"[{layer}] dissecting...")
        exemplars.discriminative(
            model,
            probe,
            layer=layer,
            results_dir=out_dir,
            tally_cache_file=layer_dir / "tally.npz",
            masks_cache_file=layer_dir / "masks.npz",
            device=device,
            image_size=image_size,
            renormalizer=renormalizer,
        )

    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-dir", type=Path,
                    default=Path(os.environ.get("MILAN_DATA_DIR", "./data"))
                            / "imagenet-spurious-text" / "50pct")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
                            / "resnet18_spurious.pth")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
                            / "edit" / "imagenet-spurious-text"
                            / "resnet18_spurious-50pct")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run(args.version_dir, args.ckpt, args.out, device=args.device)


if __name__ == "__main__":
    main()
