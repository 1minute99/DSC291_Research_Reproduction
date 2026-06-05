"""Per-unit importance ranking and ablation-curve evaluation.

Produces the three curves needed for Figure 8:

* `text-sorted`: ablate only the MILAN-identified text neurons, in order of
  smallest validation-accuracy drop.
* `sort-all`: ablate any neuron, sorted by clean-val drop, taking the
  smallest-impact ones first; used as the baseline.
* `random`: ablate a random subset (averaged over `n_random_trials`).
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from milan_repro.editing.ablate import Unit, channels_zeroed
from milan_repro.milan_glue import upstream  # noqa: F401
from milan_repro.milan_glue.register import (DEFAULT_ARCH, image_size_for,
                                              load_trained)
from milan_repro.data.spurious_dataset import load as load_spurious

from src import milannotations


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images).argmax(-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / max(1, total)


def _accuracy_with_ablation(model: nn.Module, units: Sequence[Unit],
                             loader: DataLoader, device: str) -> float:
    if not units:
        return _accuracy(model, loader, device)
    with channels_zeroed(model, units):
        return _accuracy(model, loader, device)


def per_unit_importance(model: nn.Module, dissected: milannotations.TopImagesDataset,
                        val_loader: DataLoader, device: str,
                        cache_file: Path) -> List[float]:
    """For each unit, evaluate val-accuracy with only that unit ablated.

    Returns a list aligned with `range(len(dissected))`.
    """
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        return df["val_acc_with_unit_ablated"].tolist()

    scores: List[float] = []
    for i in tqdm(range(len(dissected)), desc="per-unit importance"):
        unit = dissected.unit(i)
        scores.append(_accuracy_with_ablation(model, [unit], val_loader, device))

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit_index": range(len(dissected)),
                  "val_acc_with_unit_ablated": scores}).to_csv(
        cache_file, index=False)
    return scores


def run(version_dir: Path, ckpt_path: Path, dissect_dir: Path,
        descriptions_csv: Path, out_csv: Path,
        arch: str = DEFAULT_ARCH,
        n_random_trials: int = 5,
        ablation_min: int = 0, ablation_max: int = 50,
        ablation_step: int = 1,
        hold_out_seed: int = 0, hold_out_frac: float = 0.1,
        batch_size: int = 128, num_workers: int = 4,
        image_size: int = None,
        device: str = "cuda") -> Path:
    """Drive the editing experiment end-to-end. Writes `out_csv`."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Inputs must be the arch's native resolution (224 ResNet18 / 299 Inception);
    # feeding 224 to Inception would silently corrupt every accuracy here.
    if image_size is None:
        image_size = image_size_for(arch)

    # Datasets.
    train_full, test_set = load_spurious(version_dir, image_size=image_size)

    # Reproduce a deterministic train/val split (matches train script).
    g = torch.Generator().manual_seed(hold_out_seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    n_val = max(1, int(round(hold_out_frac * len(train_full))))
    val_set = Subset(train_full, perm[:n_val])

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)

    # Model.
    model = load_trained(ckpt_path, arch=arch, device=device)

    # Dissected units (so we can index by unit_index across layers).
    dissected = milannotations.TopImagesDataset(dissect_dir)

    # Text neurons (from MILAN descriptions).
    desc_df = pd.read_csv(descriptions_csv)
    if "is_text_neuron" not in desc_df.columns:
        from milan_repro.editing.identify_text_neurons import text_neuron_mask
        desc_df["is_text_neuron"] = text_neuron_mask(desc_df["description"])
    candidate_indices = desc_df.index[desc_df["is_text_neuron"]].tolist()
    print(f"{len(candidate_indices)} text-neuron candidates")

    # Per-unit clean-val accuracy under independent ablation. Keep the cache
    # arch-specific so a different model's (differently-sized) cache is never
    # reused; ResNet18 keeps the documented `importance.csv` name.
    imp_suffix = "" if arch == DEFAULT_ARCH else f"_{arch}"
    importance_cache = out_csv.parent / f"importance{imp_suffix}.csv"
    scores = per_unit_importance(model, dissected, val_loader, device,
                                 importance_cache)

    # Build the three orderings.
    text_sorted = sorted(candidate_indices, key=scores.__getitem__, reverse=True)
    sort_all = sorted(range(len(dissected)),
                      key=scores.__getitem__, reverse=True)[:len(candidate_indices)]

    rng = random.Random(hold_out_seed)
    random_orderings = [
        rng.sample(range(len(dissected)), k=len(candidate_indices))
        for _ in range(n_random_trials)
    ]

    # Baseline (no ablation).
    base_clean = _accuracy(model, val_loader, device)
    base_adv = _accuracy(model, test_loader, device)
    print(f"baseline: clean(val)={base_clean:.4f}  adv(test)={base_adv:.4f}")

    rows = [("baseline", 0, 0, base_clean, base_adv)]

    ns = list(range(ablation_min,
                    min(ablation_max, len(candidate_indices)) + 1,
                    ablation_step))

    def _eval_curve(name: str, ordering: Sequence[int], trial: int) -> None:
        for n_ablated in tqdm(ns, desc=f"{name}/t{trial}", leave=False):
            units = [dissected.unit(idx) for idx in ordering[:n_ablated]]
            clean_acc = _accuracy_with_ablation(model, units, val_loader, device)
            adv_acc = _accuracy_with_ablation(model, units, test_loader, device)
            rows.append((name, trial, n_ablated, clean_acc, adv_acc))

    _eval_curve("text-sorted", text_sorted, 1)
    _eval_curve("sort-all", sort_all, 1)
    for t, ordering in enumerate(random_orderings, start=1):
        _eval_curve("random", ordering, t)

    df = pd.DataFrame(rows, columns=["mode", "trial", "n_ablated",
                                     "clean_acc", "adv_acc"])
    df.to_csv(out_csv, index=False)
    print(f"wrote ablation curve to {out_csv}")
    return out_csv


def main() -> None:
    from milan_repro.milan_glue.register import ARCHES

    ap = argparse.ArgumentParser()
    base_data = Path(os.environ.get("MILAN_DATA_DIR", "./data"))
    base_models = Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))
    ap.add_argument("--arch", choices=ARCHES, default=DEFAULT_ARCH)
    ap.add_argument("--version-dir", type=Path,
                    default=base_data / "imagenet-spurious-text" / "50pct")
    # Defaults for the arch-specific paths are resolved after parsing --arch.
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--dissect-dir", type=Path, default=None)
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-random-trials", type=int, default=5)
    ap.add_argument("--ablation-max", type=int, default=50)
    ap.add_argument("--ablation-step", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=None,
                    help="override model input size (defaults to arch native)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    suffix = "" if args.arch == DEFAULT_ARCH else f"_{args.arch}"
    ckpt = args.ckpt or base_models / f"{args.arch}_spurious.pth"
    dissect = args.dissect_dir or (base_results / "edit"
                                   / "imagenet-spurious-text"
                                   / f"{args.arch}_spurious-50pct")
    descriptions = args.descriptions or base_results / f"descriptions{suffix}.csv"
    out = args.out or base_results / f"ablation_curve{suffix}.csv"

    run(args.version_dir, ckpt, dissect, descriptions, out,
        arch=args.arch,
        n_random_trials=args.n_random_trials,
        ablation_max=args.ablation_max, ablation_step=args.ablation_step,
        image_size=args.image_size,
        device=args.device)


if __name__ == "__main__":
    main()
