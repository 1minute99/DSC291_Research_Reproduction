"""Convenience loader for the on-disk spurious-text dataset.

The dataset itself is materialised by `build_splits.py`. Both `train/` and
`test/` are plain `torchvision.datasets.ImageFolder` directories, so we
just wrap the standard transforms used in the paper (224x224, ImageNet
mean/std).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torchvision.datasets as tvd
import torchvision.transforms as T

# Standard ImageNet normalisation; matches upstream MILAN.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def default_transform(image_size: int = 224) -> T.Compose:
    """Resize-only transform suited to the spurious-text images (already
    square at 224 from `build_splits.py`)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def load(version_dir: Path, image_size: int = 224
         ) -> Tuple[tvd.ImageFolder, tvd.ImageFolder]:
    """Load the (train, test) ImageFolders for a given version directory.

    Example:
        train, test = load(Path("data/imagenet-spurious-text/50pct"))
    """
    version_dir = Path(version_dir)
    tfm = default_transform(image_size)
    train = tvd.ImageFolder(str(version_dir / "train"), transform=tfm)
    test = tvd.ImageFolder(str(version_dir / "test"), transform=tfm)
    return train, test
