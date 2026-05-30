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
from milan_repro.milan_glue.register import (DEFAULT_ARCH, image_size_for,
                                              layers_for, load_trained,
                                              model_configs)
from milan_repro.data.spurious_dataset import load as load_spurious

import torch
from src import exemplars
from src.deps.netdissect import renormalize


def run(version_dir: Path, ckpt_path: Path, out_dir: Path,
        arch: str = DEFAULT_ARCH, device: str = "cuda",
        image_size: int = None) -> Path:
    """Compute exemplars for every probe layer of `arch`'s trained model.

    Args:
        version_dir: e.g. data/imagenet-spurious-text/50pct
        ckpt_path: checkpoint produced by the matching `train_<arch>.py`
        out_dir: dissection root; we write one subdir per layer
        arch: 'resnet18' or 'inception_v3'
        device: cuda/cpu/mps
        image_size: model input size; defaults to the arch's native size
            (224 for ResNet18, 299 for Inception) and must match training.

    Returns:
        out_dir (for chaining into description step).
    """
    if image_size is None:
        image_size = image_size_for(arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use upstream's hub purely to validate the config; the actual training
    # checkpoint is loaded separately and passed in as the live model.
    _ = model_configs(arch)

    model = load_trained(ckpt_path, arch=arch, device=device)

    # Use the *validation portion* of the clean (no-overlay) data as the
    # probe set for activation max. The upstream uses the validation split
    # of the trained classifier's data; we mirror that.
    train, _test = load_spurious(version_dir, image_size=image_size)
    # Subsample for speed; full set works too.
    probe = train

    renormalizer = renormalize.renormalizer(source="imagenet", target="byte")

    for layer in layers_for(arch):
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
    from milan_repro.milan_glue.register import ARCHES

    base_data = Path(os.environ.get("MILAN_DATA_DIR", "./data"))
    base_models = Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHES, default=DEFAULT_ARCH)
    ap.add_argument("--version-dir", type=Path,
                    default=base_data / "imagenet-spurious-text" / "50pct")
    # Defaults for --ckpt and --out depend on --arch; resolved after parsing.
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--image-size", type=int, default=None,
                    help="override model input size (defaults to arch native)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = args.ckpt or base_models / f"{args.arch}_spurious.pth"
    out = args.out or (base_results / "edit" / "imagenet-spurious-text"
                       / f"{args.arch}_spurious-50pct")
    run(args.version_dir, ckpt, out, arch=args.arch, device=args.device,
        image_size=args.image_size)


if __name__ == "__main__":
    main()
