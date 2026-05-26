"""Register our trained ResNet18 + spurious-text dataset with upstream MILAN.

Returns the `{model_configs, dataset_configs}` mappings that can be passed
into `exemplars.models.load(name, configs=...)` and
`exemplars.datasets.load(name, configs=...)`.

Naming convention:
    model name = 'resnet18_spurious/imagenet'   (we reuse upstream's
        IMAGENET key so its `from src import ...` paths still resolve)
    dataset name = 'imagenet-spurious-text'      (upstream's existing key)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from milan_repro.milan_glue import upstream  # noqa: F401  (sys.path side effect)

import torch
from torch import nn
import torchvision

from src.exemplars import datasets as ex_datasets
from src.exemplars import models as ex_models
from src.exemplars.models import LAYERS

MODEL_NAME = "resnet18_spurious/imagenet"


def _make_resnet18(num_classes: int = 10, **_: Any) -> nn.Module:
    """Factory that builds a torchvision ResNet18 with `num_classes` outputs."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _transform_ckpt(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Unwrap our training checkpoint, which stores {state_dict, config, history}.

    Upstream's `ModelConfig.load` expects a plain state_dict, so we strip
    the wrapper.
    """
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload  # already a state_dict


def model_config(num_classes: int = 10) -> ex_models.ModelConfig:
    """Build the MILAN `ModelConfig` for our trained ResNet18.

    The config tells MILAN how to instantiate the model, what layers to
    probe (LAYERS.RESNET18 = conv1 + layer1..4), and how to load our
    checkpoint into it.
    """
    return ex_models.ModelConfig(
        _make_resnet18,
        load_weights=True,
        transform_weights=_transform_ckpt,
        layers=LAYERS.RESNET18,
        num_classes=num_classes,
    )


def model_configs(num_classes: int = 10) -> Mapping[str, ex_models.ModelConfig]:
    """Return a configs mapping suitable for `exemplars.models.load(..., configs=...)`."""
    return {MODEL_NAME: model_config(num_classes=num_classes)}


def load_trained(ckpt_path: Path, num_classes: int = 10,
                 device: str = "cuda") -> nn.Module:
    """Load our trained ResNet18 directly (without going through MILAN's hub).

    Used by the editing experiment, which needs the model in plain torch
    form so we can train, ablate, and evaluate it.
    """
    model = _make_resnet18(num_classes=num_classes)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = _transform_ckpt(payload)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# The spurious-text dataset config is already in upstream as
# `KEYS.IMAGENET_SPURIOUS_TEXT`; no need to redefine it.
SPURIOUS_DATASET_KEY = ex_datasets.KEYS.IMAGENET_SPURIOUS_TEXT
