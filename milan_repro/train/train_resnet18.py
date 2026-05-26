"""Train ResNet18 from scratch on the spurious-text dataset.

Mirrors MILAN's own `experiments/edit.py` training loop (AdamW, lr=1e-4,
batch 128, early-stop on val loss, patience=4) so we stay faithful to the
authors' setup. Writes a checkpoint to MODELS_DIR / resnet18_spurious.pth.
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


def _make_resnet18(num_classes: int) -> nn.Module:
    model = tvm.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _split_indices(n: int, hold_out: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(hold_out * n)))
    return perm[n_val:], perm[:n_val]


def train(config: dict, version_dir: Path, out_ckpt: Path,
          device: str = "cuda") -> dict:
    cfg_train = config["train"]
    cfg_data = config["data"]
    cfg_model = config["model"]

    train_full, _ = load_spurious(version_dir, image_size=cfg_data["image_size"])
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

    model = _make_resnet18(cfg_model["num_classes"]).to(device)
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
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False)
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
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
                logits = model(images)
                val_loss += criterion(logits, targets).item() * images.size(0)
                val_correct += (logits.argmax(-1) == targets).sum().item()
                n_val += images.size(0)
        val_loss /= max(1, n_val)
        val_acc = val_correct / max(1, n_val)

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone().cpu()
                          for k, v in model.state_dict().items()}
            patience_left = cfg_train["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch {epoch+1}")
                break

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state,
                "config": config,
                "history": history}, out_ckpt)
    print(f"saved best model to {out_ckpt} (val_loss={best_val:.4f})")
    return {"best_val_loss": best_val, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("configs/resnet18_appendixE.yaml"))
    ap.add_argument("--version-dir", type=Path,
                    default=Path(os.environ.get("MILAN_DATA_DIR", "./data"))
                            / "imagenet-spurious-text" / "50pct")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
                            / "resnet18_spurious.pth")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with args.config.open() as f:
        config = yaml.safe_load(f)
    train(config, args.version_dir, args.out, device=args.device)


if __name__ == "__main__":
    sys.exit(main() or 0)
