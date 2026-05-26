"""Build the on-disk spurious-text dataset from Imagenette.

Output layout (matches what upstream MILAN's `imagenet-spurious-text` dataset
config expects):

    $MILAN_DATA_DIR/imagenet-spurious-text/<version>/train/<class>/img-XXXXX.jpg
    $MILAN_DATA_DIR/imagenet-spurious-text/<version>/test/<class>/img-XXXXX.jpg

`<version>` is named after the fraction of training images that get a
text overlay (e.g. `50pct`).

The 10 Imagenette classes are mapped to human-readable label strings (the
text written in the corner) because Imagenette folder names are WordNet
synset IDs that nobody recognizes.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List

from PIL import Image

from milan_repro.data.render_text import TextStyle, render

# WordNet synset -> human-readable class name (the "label" we paint on images).
# Source: Imagenette README.
IMAGENETTE_CLASSES: Dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# Short single-word labels used in the painted overlay. Shorter strings keep
# the corner stripe small so the spurious signal sits in roughly the same
# spatial area for every class. (We use the *full* name for filenames, but
# the painted text is the short version.)
SHORT_LABELS: Dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "springer",
    "n02979186": "cassette",
    "n03000684": "chainsaw",
    "n03028079": "church",
    "n03394916": "horn",
    "n03417042": "truck",
    "n03425413": "pump",
    "n03445777": "golf",
    "n03888257": "parachute",
}

IMAGENETTE_URL = (
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
)


def download_imagenette(target_dir: Path) -> Path:
    """Download and extract Imagenette under `target_dir`. Returns the
    `imagenette2-320` directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = target_dir / "imagenette2-320"
    if extracted.exists():
        return extracted

    archive = target_dir / "imagenette2-320.tgz"
    if not archive.exists():
        print(f"downloading Imagenette to {archive} ...")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)
    print(f"extracting {archive} ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(target_dir)
    return extracted


def _iter_class_images(split_dir: Path):
    for synset_dir in sorted(split_dir.iterdir()):
        if not synset_dir.is_dir() or synset_dir.name not in IMAGENETTE_CLASSES:
            continue
        for img_path in sorted(synset_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield synset_dir.name, img_path


def build(
    imagenette_root: Path,
    out_root: Path,
    version: str = "50pct",
    train_fraction: float = 0.5,
    test_fraction: float = 1.0,
    style: TextStyle = TextStyle(),
    seed: int = 0,
    resize: int = 224,
) -> Path:
    """Write the spurious-text ImageFolder dataset to disk.

    Args:
        imagenette_root: path to the extracted Imagenette directory containing
            `train/` and `val/` subfolders.
        out_root: destination root. The dataset is written at
            `out_root/<version>/{train,test}/<class>/img-XXXXX.jpg`.
        version: subfolder name (matches upstream's `5pct`/`50pct`/etc.).
        train_fraction: probability a *training* image receives its *correct*
            class-name overlay.
        test_fraction: probability a *test* image receives a *random* other
            class's overlay. Default 1.0 (every test image is adversarial).
        style: text rendering style.
        seed: RNG seed for the per-image draw and the wrong-label sampling.
        resize: square resize edge length (`224` matches the paper).

    Returns:
        The version directory that was written.
    """
    rng = random.Random(seed)
    short_label_list = list(SHORT_LABELS.values())

    version_dir = out_root / version
    rows: List[List[str]] = [["split", "class", "src_path", "out_path",
                              "overlay_text", "wrong"]]

    for src_split, out_split in (("train", "train"), ("val", "test")):
        src = imagenette_root / src_split
        if not src.exists():
            raise FileNotFoundError(src)
        for cls_synset, img_path in _iter_class_images(src):
            cls_short = SHORT_LABELS[cls_synset]
            out_cls_dir = version_dir / out_split / cls_synset
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_cls_dir / img_path.name

            img = Image.open(img_path).convert("RGB")
            # Center-crop to square then resize, to keep overlay position
            # consistent across images.
            w, h = img.size
            side = min(w, h)
            img = img.crop(((w - side) // 2, (h - side) // 2,
                            (w + side) // 2, (h + side) // 2)).resize(
                (resize, resize), Image.BILINEAR)

            overlay_text = ""
            wrong = ""
            if out_split == "train":
                if rng.random() < train_fraction:
                    overlay_text = cls_short
                    wrong = "no"
            else:
                if rng.random() < test_fraction:
                    other = [lbl for lbl in short_label_list if lbl != cls_short]
                    overlay_text = rng.choice(other)
                    wrong = "yes"

            if overlay_text:
                img = render(img, overlay_text, style=style)
            img.save(out_path, quality=90)

            rows.append([out_split, cls_synset, str(img_path),
                         str(out_path), overlay_text, wrong])

    manifest = version_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {len(rows) - 1} images, manifest at {manifest}")
    return version_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path(os.environ.get("MILAN_DATA_DIR", "./data")))
    ap.add_argument("--version", default="50pct")
    ap.add_argument("--train-fraction", type=float, default=0.5)
    ap.add_argument("--test-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imagenette = download_imagenette(args.data_dir / "imagenette")
    out = args.data_dir / "imagenet-spurious-text"
    build(imagenette, out,
          version=args.version,
          train_fraction=args.train_fraction,
          test_fraction=args.test_fraction,
          seed=args.seed)


if __name__ == "__main__":
    main()
