"""Compute top-activating exemplars for VGG16 or InceptionV3.

Proposal new experiment 1 — architecture generalization.
Mirrors run_exemplars.py but for non-ResNet architectures.

Usage:
    python -m milan_repro.milan_glue.run_exemplars_multiarc --arch vgg16
    python -m milan_repro.milan_glue.run_exemplars_multiarc --arch inception_v3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from milan_repro.milan_glue import upstream  # noqa: F401
from milan_repro.milan_glue.register_multiarc import (LAYERS_BY_ARCH,
                                                       load_trained)
from milan_repro.data.spurious_dataset import load as load_spurious

import torch
from src import exemplars
from src.deps.netdissect import renormalize


def run(arch: str, version_dir: Path, ckpt_path: Path, out_dir: Path,
        device: str = "cuda") -> Path:
    """Compute exemplars for every probed layer of the given architecture."""
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = 299 if arch == "inception_v3" else 224
    model = load_trained(ckpt_path, arch=arch, device=device)
    train, _ = load_spurious(version_dir, image_size=image_size)

    renormalizer = renormalize.renormalizer(source="imagenet", target="byte")

    for layer in LAYERS_BY_ARCH[arch]:
        layer_dir = out_dir / layer.replace(".", "_")
        if (layer_dir / "images.npy").exists() and (layer_dir / "masks.npy").exists():
            print(f"[{arch}/{layer}] cached, skipping")
            continue
        print(f"[{arch}/{layer}] dissecting...")
        exemplars.discriminative(
            model,
            train,
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
    base_data = Path(os.environ.get("MILAN_DATA_DIR", "./data"))
    base_models = Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(LAYERS_BY_ARCH.keys()), default="vgg16")
    ap.add_argument("--version-dir", type=Path,
                    default=base_data / "imagenet-spurious-text" / "50pct")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Default: models/<arch>_spurious.pth")
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: results/edit/imagenet-spurious-text/<arch>-50pct")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.ckpt is None:
        args.ckpt = base_models / f"{args.arch}_spurious.pth"
    if args.out is None:
        args.out = (base_results / "edit" / "imagenet-spurious-text"
                    / f"{args.arch}-50pct")

    run(args.arch, args.version_dir, args.ckpt, args.out, device=args.device)


if __name__ == "__main__":
    main()
