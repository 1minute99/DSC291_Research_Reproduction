"""Compute top-activating exemplars for CLIP ViT-B/32 transformer blocks.

Proposal new experiment 2 — "Applying MILAN to CLIP (perhaps the vision
encoder part) to observe the ablation study presented in Section 7."

Unlike ResNet/VGG, CLIP is not fine-tuned — we probe it zero-shot against
the same spurious-text dataset to see which transformer blocks are sensitive
to the painted corner text.

Output layout (compatible with TopImagesDataset):
    $MILAN_RESULTS_DIR/edit/clip-vitb32/<layer_safe_name>/{images,masks}.npy

Usage:
    python -m milan_repro.milan_glue.run_clip_exemplars
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from milan_repro.milan_glue import upstream  # noqa: F401
from milan_repro.milan_glue.clip_glue import (CLIPVisionSpatialWrapper,
                                               CLIP_LAYERS_SUBSET,
                                               load_clip_vision)
from milan_repro.data.spurious_dataset import load as load_spurious

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# CLIP ViT-B/32 image size
CLIP_IMAGE_SIZE = 224
TOP_K = 15          # top-k exemplars per unit (matches ResNet18 run)


@torch.no_grad()
def _collect_exemplars(
    wrapper: CLIPVisionSpatialWrapper,
    loader: DataLoader,
    layer_name: str,
    top_k: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the dataset through CLIP and track top-k activating images per unit.

    Returns:
        images: (n_units, top_k, 3, H, W)  uint8
        masks:  (n_units, top_k, 1, H, W)  float32 in [0,1]
    """
    # First pass: find number of channels from a single batch
    batch_images, _ = next(iter(loader))
    batch_images = batch_images.to(device)
    wrapper(batch_images)  # populate hooks
    feat = wrapper.get_spatial(layer_name)  # (B, C, h, w)
    n_units, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
    H = W = CLIP_IMAGE_SIZE

    # Running top-k: track (score, img_idx, spatial_pos) per unit
    topk_scores = [[] for _ in range(n_units)]   # list of (score, img_tensor, mask)

    global_idx = 0
    for images, _ in tqdm(loader, desc=f"  collecting [{layer_name}]", leave=False):
        images = images.to(device)
        wrapper(images)
        feat = wrapper.get_spatial(layer_name)  # (B, C, h, w)

        # Max activation per unit across spatial positions
        spatial_max, spatial_argmax = feat.max(dim=-1)[0].max(dim=-1)  # (B, C)

        for b in range(images.size(0)):
            img_np = (images[b].cpu().permute(1, 2, 0)
                      .mul(torch.tensor([0.229, 0.224, 0.225]))
                      .add(torch.tensor([0.485, 0.456, 0.406]))
                      .clamp(0, 1).numpy())
            img_uint8 = (img_np * 255).astype(np.uint8)

            for u in range(n_units):
                score = float(spatial_max[b, u])
                # Build a spatial mask (highlight the most-active cell)
                mask = np.zeros((h, w), dtype=np.float32)
                pos = int(feat[b, u].argmax())
                row, col = divmod(pos, w)
                mask[row, col] = 1.0
                # Upsample mask to image size
                import cv2
                mask_up = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
                topk_scores[u].append((score, img_uint8, mask_up))
                if len(topk_scores[u]) > top_k * 4:
                    topk_scores[u].sort(key=lambda x: x[0], reverse=True)
                    topk_scores[u] = topk_scores[u][:top_k * 2]

        global_idx += images.size(0)

    # Finalize: keep top_k per unit
    images_out = np.zeros((n_units, top_k, 3, H, W), dtype=np.uint8)
    masks_out = np.zeros((n_units, top_k, 1, H, W), dtype=np.float32)
    for u in range(n_units):
        topk_scores[u].sort(key=lambda x: x[0], reverse=True)
        for k, (_, img, mask) in enumerate(topk_scores[u][:top_k]):
            images_out[u, k] = img.transpose(2, 0, 1)  # HWC → CHW
            masks_out[u, k, 0] = mask

    return images_out, masks_out


def run(version_dir: Path, out_dir: Path,
        layers: list[str] | None = None,
        device: str = "cuda",
        top_k: int = TOP_K,
        batch_size: int = 32) -> Path:
    if layers is None:
        layers = CLIP_LAYERS_SUBSET

    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = load_clip_vision(device=device)
    wrapper.register_spatial_hooks(layers)

    train, _ = load_spurious(version_dir, image_size=CLIP_IMAGE_SIZE)
    loader = DataLoader(train, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    for layer_name in layers:
        safe = layer_name.replace(".", "_")
        layer_dir = out_dir / safe
        if (layer_dir / "images.npy").exists() and (layer_dir / "masks.npy").exists():
            print(f"[clip/{layer_name}] cached, skipping")
            continue

        print(f"[clip/{layer_name}] collecting exemplars...")
        layer_dir.mkdir(parents=True, exist_ok=True)
        images_np, masks_np = _collect_exemplars(
            wrapper, loader, layer_name, top_k=top_k, device=device)
        np.save(layer_dir / "images.npy", images_np)
        np.save(layer_dir / "masks.npy", masks_np)
        print(f"  → {layer_dir}  shape={images_np.shape}")

    return out_dir


def main() -> None:
    base_data = Path(os.environ.get("MILAN_DATA_DIR", "./data"))
    base_results = Path(os.environ.get("MILAN_RESULTS_DIR", "./results"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--version-dir", type=Path,
                    default=base_data / "imagenet-spurious-text" / "50pct")
    ap.add_argument("--out", type=Path,
                    default=base_results / "edit" / "clip-vitb32")
    ap.add_argument("--layers", nargs="*", default=None,
                    help="Layer names to probe (default: 5-layer subset)")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run(args.version_dir, args.out,
        layers=args.layers,
        device=args.device,
        top_k=args.top_k,
        batch_size=args.batch_size)


if __name__ == "__main__":
    main()
