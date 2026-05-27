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
    """Two-pass exemplar collection — low memory footprint.

    Pass 1: store only (score, global_img_index, spatial_argmax) per unit.
            Memory: n_units × n_images × ~12 bytes ≈ 120 MB for 768 units.
    Pass 2: reload the specific top-k images by index to build output arrays.

    Returns:
        images: (n_units, top_k, 3, H, W)  uint8
        masks:  (n_units, top_k, 1, H, W)  float32 in [0,1]
    """
    # Probe one batch to learn n_units and spatial grid size
    batch_images, _ = next(iter(loader))
    batch_images = batch_images.to(device)
    wrapper(batch_images)
    feat = wrapper.get_spatial(layer_name)  # (B, C, h, w)
    n_units, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
    H = W = CLIP_IMAGE_SIZE

    # Pass 1: track top-k (score, global_idx, spatial_pos) per unit — no images
    # Shape: (n_units, n_images_so_far) tracked as fixed-size sorted lists
    import heapq
    # min-heap per unit: (score, global_idx, argmax_pos) — keep top_k largest
    heaps = [[] for _ in range(n_units)]  # min-heaps (smallest score at top)

    global_idx = 0
    for images, _ in tqdm(loader, desc=f"  pass1 [{layer_name}]", leave=False):
        images = images.to(device)
        wrapper(images)
        feat = wrapper.get_spatial(layer_name)            # (B, C, h, w)
        scores_bc = feat.flatten(2).max(dim=2).values     # (B, C)
        argmax_bc = feat.flatten(2).argmax(dim=2)         # (B, C)

        scores_np = scores_bc.cpu().numpy()               # (B, C)
        argmax_np = argmax_bc.cpu().numpy()               # (B, C)

        for b in range(images.size(0)):
            for u in range(n_units):
                entry = (float(scores_np[b, u]), global_idx + b,
                         int(argmax_np[b, u]))
                if len(heaps[u]) < top_k:
                    heapq.heappush(heaps[u], entry)
                elif entry[0] > heaps[u][0][0]:
                    heapq.heapreplace(heaps[u], entry)

        global_idx += images.size(0)

    # Collect the top-k global indices and argmax positions per unit
    top_indices = np.zeros((n_units, top_k), dtype=np.int64)
    top_argmax  = np.zeros((n_units, top_k), dtype=np.int64)
    for u in range(n_units):
        ranked = sorted(heaps[u], key=lambda x: x[0], reverse=True)
        for k, (_, gidx, amax) in enumerate(ranked[:top_k]):
            top_indices[u, k] = gidx
            top_argmax[u, k]  = amax

    # Pass 2: reload only the required images by global index
    dataset = loader.dataset
    images_out = np.zeros((n_units, top_k, 3, H, W), dtype=np.uint8)
    masks_out  = np.zeros((n_units, top_k, 1, H, W), dtype=np.float32)

    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    import cv2
    needed = sorted(set(top_indices.flatten().tolist()))
    idx_to_img: dict[int, np.ndarray] = {}
    for gidx in tqdm(needed, desc=f"  pass2 [{layer_name}]", leave=False):
        img_tensor, _ = dataset[gidx]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * _std + _mean)
        idx_to_img[gidx] = (img_np.clip(0, 1) * 255).astype(np.uint8)

    for u in range(n_units):
        for k in range(top_k):
            img = idx_to_img[int(top_indices[u, k])]
            images_out[u, k] = img.transpose(2, 0, 1)
            amax = int(top_argmax[u, k])
            row, col = divmod(amax, w)
            mask = np.zeros((h, w), dtype=np.float32)
            mask[row, col] = 1.0
            masks_out[u, k, 0] = cv2.resize(mask, (W, H),
                                             interpolation=cv2.INTER_LINEAR)
    del idx_to_img
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
