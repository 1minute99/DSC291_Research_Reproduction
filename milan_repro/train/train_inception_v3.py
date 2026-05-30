"""Train InceptionV3 from scratch on the spurious-text dataset.

Extension experiment (proposal "New experiments"): does the text-shortcut
behaviour MILAN finds in ResNet18 also appear in a deeper, multi-branch
architecture? We keep the optimiser/schedule identical to
`train_resnet18.py` (AdamW, lr=1e-4, early-stop on val loss, patience=4) and
change only the architecture, so the comparison is clean.

Two InceptionV3-specific wrinkles handled here:
  * input is 299x299 (vs 224 for ResNet18) -- driven by the config's
    `data.image_size`, fed straight into the shared dataset loader.
  * the auxiliary classifier: in train mode torchvision's InceptionV3 returns
    `InceptionOutputs(logits, aux_logits)` and we use the standard combined
    loss `L_main + aux_weight * L_aux`; in eval mode it returns just `logits`.

Writes a checkpoint to MODELS_DIR / inception_v3_spurious.pth, in the same
{state_dict, config, history} format as the ResNet18 trainer.
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
from torchvision.models.inception import InceptionOutputs
from tqdm.auto import tqdm

from milan_repro.data.spurious_dataset import load as load_spurious


def _make_inception_v3(num_classes: int, aux_logits: bool = True) -> nn.Module:
    """Build a torchvision InceptionV3 with `num_classes` outputs, from scratch.

    Passing `num_classes` to the constructor sizes both the main `fc` and the
    auxiliary head; `init_weights=True` gives the standard truncated-normal
    initialisation (otherwise torchvision warns when no pretrained weights are
    requested).
    """
    return tvm.inception_v3(
        weights=None,
        num_classes=num_classes,
        aux_logits=aux_logits,
        init_weights=True,
    )


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

    aux_logits = cfg_model.get("aux_logits", True)
    aux_weight = cfg_model.get("aux_weight", 0.4)

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

    model = _make_inception_v3(cfg_model["num_classes"],
                               aux_logits=aux_logits).to(device)
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
            out = model(images)
            # In train mode InceptionV3 returns (main, aux); fold the aux head
            # into the loss with the standard 0.4 weight. (No-op if aux is off.)
            if isinstance(out, InceptionOutputs):
                logits, aux = out.logits, out.aux_logits
                loss = criterion(logits, targets) + aux_weight * criterion(aux, targets)
            else:
                logits = out
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
                logits = model(images)  # eval mode: just the main logits
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


def load_trained(ckpt_path: Path, num_classes: int = 10,
                 device: str = "cuda") -> nn.Module:
    """Load a trained InceptionV3 in plain torch form (eval mode).

    Mirrors `milan_repro.milan_glue.register.load_trained` for ResNet18 so the
    editing/eval code can use the Inception model the same way. We rebuild with
    `aux_logits=True` to match the saved weights (the checkpoint includes the
    AuxLogits parameters), then switch to eval where the aux head is inert.
    """
    model = _make_inception_v3(num_classes=num_classes, aux_logits=True)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and \
        "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("configs/inception_v3.yaml"))
    ap.add_argument("--version-dir", type=Path,
                    default=Path(os.environ.get("MILAN_DATA_DIR", "./data"))
                            / "imagenet-spurious-text" / "50pct")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("MILAN_MODELS_DIR", "./models"))
                            / "inception_v3_spurious.pth")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with args.config.open() as f:
        config = yaml.safe_load(f)
    train(config, args.version_dir, args.out, device=args.device)


if __name__ == "__main__":
    sys.exit(main() or 0)
