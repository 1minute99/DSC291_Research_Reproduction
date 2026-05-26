"""Register VGG16 / InceptionV3 with upstream MILAN.

Mirrors register.py but for non-ResNet architectures.
Each architecture defines its own probed layer list so that
`run_exemplars_multiarc.py` knows which module paths to hook.

Layer choices:
  VGG16      — last Conv2d in each max-pool block: features.{2,7,14,21,28}
  InceptionV3 — key inception modules: Conv2d_4a_3x3, Mixed_5d, Mixed_6e, Mixed_7c
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping

from milan_repro.milan_glue import upstream  # noqa: F401

import torch
from torch import nn
import torchvision

from src.exemplars import models as ex_models

# ── Layer lists ──────────────────────────────────────────────────────────────

VGG16_LAYERS: List[str] = [
    "features.2",   # 64 ch  — block 1 last conv
    "features.7",   # 128 ch — block 2 last conv
    "features.14",  # 256 ch — block 3 last conv
    "features.21",  # 512 ch — block 4 last conv
    "features.28",  # 512 ch — block 5 last conv
]

INCEPTION_LAYERS: List[str] = [
    "Conv2d_4a_3x3",  # early conv (192 ch)
    "Mixed_5d",       # inception block shallow (256 ch)
    "Mixed_6e",       # inception block mid (768 ch)
    "Mixed_7c",       # inception block deep (2048 ch)
]

LAYERS_BY_ARCH = {
    "vgg16": VGG16_LAYERS,
    "inception_v3": INCEPTION_LAYERS,
}


# ── Model factories ──────────────────────────────────────────────────────────

def _make_vgg16(num_classes: int = 10, **_: Any) -> nn.Module:
    model = torchvision.models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def _make_inception_v3(num_classes: int = 10, **_: Any) -> nn.Module:
    model = torchvision.models.inception_v3(weights=None, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


_FACTORIES = {
    "vgg16": _make_vgg16,
    "inception_v3": _make_inception_v3,
}


def _transform_ckpt(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Unwrap checkpoint wrapper (same logic as register.py)."""
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


# ── Public API ───────────────────────────────────────────────────────────────

def model_config(arch: str, num_classes: int = 10) -> ex_models.ModelConfig:
    factory = _FACTORIES[arch]
    layers = LAYERS_BY_ARCH[arch]
    return ex_models.ModelConfig(
        factory,
        load_weights=True,
        transform_weights=_transform_ckpt,
        layers=layers,
        num_classes=num_classes,
    )


def load_trained(ckpt_path: Path, arch: str | None = None,
                 num_classes: int = 10, device: str = "cuda") -> nn.Module:
    """Load a trained VGG16 or InceptionV3 checkpoint.

    If `arch` is None, it is inferred from the checkpoint's 'arch' key.
    """
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if arch is None:
        arch = payload.get("arch", None)
    if arch is None:
        raise ValueError("Cannot infer arch from checkpoint; pass --arch explicitly.")

    factory = _FACTORIES[arch]
    model = factory(num_classes=num_classes)
    state_dict = _transform_ckpt(payload)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
