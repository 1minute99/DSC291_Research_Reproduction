"""Ablation-curve evaluation for VGG16 / InceptionV3.

Proposal new experiment 1 — architecture generalization.
Reuses the same three-curve logic from evaluate.py but for any arch.

Usage:
    python -m milan_repro.editing.evaluate_multiarc --arch vgg16
    python -m milan_repro.editing.evaluate_multiarc --arch inception_v3
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from milan_repro.editing.ablate import Unit, channels_zeroed
from milan_repro.milan_glue import upstream  # noqa: F401
from milan_repro.milan_glue.register_multiarc import LAYERS_BY_ARCH, load_trained
from milan_repro.data.spurious_dataset import load as load_spurious
from milan_repro.editing.identify_text_neurons import text_neuron_mask

class _UnitIndex:
    """Lightweight unit index from descriptions CSV (no image loading)."""
    def __init__(self, descriptions_csv: Path) -> None:
        df = pd.read_csv(descriptions_csv).sort_values("unit_index").reset_index(drop=True)
        self._layers = df["layer"].tolist()
        self._channels = df["channel"].tolist()
    def unit(self, i: int):
        return (self._layers[i], self._channels[i])
    def __len__(self) -> int:
        return len(self._layers)


@torch.no_grad()
def _accuracy(model, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        preds = (out.logits if hasattr(out, "logits") else out).argmax(-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / max(1, total)


def _accuracy_with_ablation(model, units: Sequence[Unit],
                             loader: DataLoader, device: str) -> float:
    if not units:
        return _accuracy(model, loader, device)
    with channels_zeroed(model, units):
        return _accuracy(model, loader, device)


def run(arch: str, version_dir: Path, ckpt_path: Path, dissect_dir: Path,
        descriptions_csv: Path, out_csv: Path,
        n_random_trials: int = 5,
        ablation_max: int = 50, ablation_step: int = 1,
        hold_out_seed: int = 0, hold_out_frac: float = 0.1,
        batch_size: int = 64, num_workers: int = 4,
        device: str = "cuda") -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    image_size = 299 if arch == "inception_v3" else 224
    train_full, test_set = load_spurious(version_dir, image_size=image_size)

    g = torch.Generator().manual_seed(hold_out_seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    n_val = max(1, int(round(hold_out_frac * len(train_full))))
    val_set = Subset(train_full, perm[:n_val])

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=True)

    model = load_trained(ckpt_path, arch=arch, device=device)
    dissected = _UnitIndex(descriptions_csv)

    desc_df = pd.read_csv(descriptions_csv)
    if "is_text_neuron" not in desc_df.columns:
        desc_df["is_text_neuron"] = text_neuron_mask(desc_df["description"])
    candidate_indices = desc_df.index[desc_df["is_text_neuron"]].tolist()
    print(f"[{arch}] {len(candidate_indices)} text-neuron candidates")

    # Per-unit importance
    importance_cache = out_csv.parent / f"importance_{arch}.csv"
    if importance_cache.exists():
        scores = pd.read_csv(importance_cache)["val_acc_with_unit_ablated"].tolist()
    else:
        scores = []
        for i in tqdm(range(len(dissected)), desc=f"[{arch}] per-unit importance"):
            unit = dissected.unit(i)
            scores.append(_accuracy_with_ablation(model, [unit], val_loader, device))
        pd.DataFrame({"unit_index": range(len(dissected)),
                      "val_acc_with_unit_ablated": scores}).to_csv(
            importance_cache, index=False)

    text_sorted = sorted(candidate_indices, key=scores.__getitem__, reverse=True)
    sort_all = sorted(range(len(dissected)),
                      key=scores.__getitem__, reverse=True)[:len(candidate_indices)]
    rng = random.Random(hold_out_seed)
    random_orderings = [
        rng.sample(range(len(dissected)), k=len(candidate_indices))
        for _ in range(n_random_trials)
    ]

    base_clean = _accuracy(model, val_loader, device)
    base_adv = _accuracy(model, test_loader, device)
    print(f"[{arch}] baseline: clean(val)={base_clean:.4f}  adv(test)={base_adv:.4f}")

    rows = [("baseline", 0, 0, base_clean, base_adv)]
    ns = list(range(0, min(ablation_max, len(candidate_indices)) + 1, ablation_step))

    def _eval_curve(name: str, ordering, trial: int) -> None:
        for n_ablated in tqdm(ns, desc=f"[{arch}] {name}/t{trial}", leave=False):
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
    print(f"[{arch}] wrote ablation curve to {out_csv}")
    return out_csv


def main() -> None:
    base_data = Path(os.environ.get("MILAN_DATA_DIR", "./data"))
    base_models = Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(LAYERS_BY_ARCH.keys()), default="vgg16")
    ap.add_argument("--version-dir", type=Path,
                    default=base_data / "imagenet-spurious-text" / "50pct")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--dissect-dir", type=Path, default=None)
    ap.add_argument("--descriptions", type=Path,
                    default=None,
                    help="Default: results/<arch>_descriptions_annotated.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ablation-max", type=int, default=50)
    ap.add_argument("--ablation-step", type=int, default=1)
    ap.add_argument("--n-random-trials", type=int, default=5)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    arch = args.arch
    ckpt = args.ckpt or base_models / f"{arch}_spurious.pth"
    dissect_dir = args.dissect_dir or (base_results / "edit" / "imagenet-spurious-text"
                                       / f"{arch}-50pct")
    descriptions = args.descriptions or base_results / f"{arch}_descriptions_annotated.csv"
    out = args.out or base_results / f"{arch}_ablation_curve.csv"

    run(arch, args.version_dir, ckpt, dissect_dir, descriptions, out,
        ablation_max=args.ablation_max, ablation_step=args.ablation_step,
        n_random_trials=args.n_random_trials, device=args.device)


if __name__ == "__main__":
    main()
