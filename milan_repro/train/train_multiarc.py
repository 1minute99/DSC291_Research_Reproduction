"""Train VGG16 or InceptionV3 on the spurious-text dataset.

Proposal new experiment 1 — "Test generalization across extended type of
architecture, e.g. VGGNet and Inception."

Usage:
    python -m milan_repro.train.train_multiarc --arch vgg16
    python -m milan_repro.train.train_multiarc --arch inception_v3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import models as tvm
from tqdm.auto import tqdm

from milan_repro.data.spurious_dataset import load as load_spurious

SUPPORTED_ARCHS = ("vgg16", "inception_v3")


def _make_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "vgg16":
        model = tvm.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif arch == "inception_v3":
        model = tvm.inception_v3(weights=None, aux_logits=True)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {SUPPORTED_ARCHS}")
    return model


def _split_indices(n: int, hold_out: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(hold_out * n)))
    return perm[n_val:], perm[:n_val]


def train(arch: str, config: dict, version_dir: Path, out_ckpt: Path,
          device: str = "cuda") -> dict:
    cfg_train = config["train"]
    cfg_data = config["data"]
    cfg_model = config["model"]

    # InceptionV3 expects 299x299; VGG16 and ResNet18 use 224x224.
    image_size = 299 if arch == "inception_v3" else cfg_data["image_size"]

    train_full, _ = load_spurious(version_dir, image_size=image_size)
    train_idx, val_idx = _split_indices(len(train_full),
                                        cfg_train["hold_out"],
                                        cfg_train["seed"])
    train_set = Subset(train_full, train_idx)
    val_set = Subset(train_full, val_idx)

    bs = cfg_train["batch_size"]
    nw = cfg_data["num_workers"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)

    model = _make_model(arch, cfg_model["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg_train["lr"],
                            weight_decay=cfg_train["weight_decay"])

    best_val = float("inf")
    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    patience_left = cfg_train["patience"]
    history = []

    for epoch in range(cfg_train["max_epochs"]):
        model.train()
        train_loss = 0.0
        n_train = 0
        pbar = tqdm(train_loader, desc=f"[{arch}] epoch {epoch+1}", leave=False)
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            # InceptionV3 returns (logits, aux_logits) during training
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, targets)
            if hasattr(out, "aux_logits") and out.aux_logits is not None:
                loss = loss + 0.4 * criterion(out.aux_logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n_train += images.size(0)
        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        n_val = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                out = model(images)
                logits = out.logits if hasattr(out, "logits") else out
                val_loss += criterion(logits, targets).item() * images.size(0)
                val_correct += (logits.argmax(-1) == targets).sum().item()
                n_val += images.size(0)
        val_loss /= max(1, n_val)
        val_acc = val_correct / max(1, n_val)

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{arch}] epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone().cpu()
                          for k, v in model.state_dict().items()}
            patience_left = cfg_train["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{arch}] early stop at epoch {epoch+1}")
                break

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch,
                "state_dict": best_state,
                "config": config,
                "history": history}, out_ckpt)
    print(f"saved best {arch} to {out_ckpt} (val_loss={best_val:.4f})")
    return {"arch": arch, "best_val_loss": best_val, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=SUPPORTED_ARCHS, default="vgg16")
    ap.add_argument("--config", type=Path,
                    default=Path("configs/resnet18_appendixE.yaml"))
    ap.add_argument("--version-dir", type=Path,
                    default=Path(os.environ.get("MILAN_DATA_DIR", "./data"))
                            / "imagenet-spurious-text" / "50pct")
    ap.add_argument("--out", type=Path, default=None,
                    help="Checkpoint path (default: models/<arch>_spurious.pth)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.out is None:
        args.out = (Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
                    / f"{args.arch}_spurious.pth")

    with args.config.open() as f:
        config = yaml.safe_load(f)

    train(args.arch, config, args.version_dir, args.out, device=args.device)


if __name__ == "__main__":
    sys.exit(main() or 0)
