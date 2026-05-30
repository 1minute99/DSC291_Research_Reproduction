"""Register our trained classifiers + spurious-text dataset with upstream MILAN.

Returns the `{model_configs, dataset_configs}` mappings that can be passed
into `exemplars.models.load(name, configs=...)` and
`exemplars.datasets.load(name, configs=...)`.

Two architectures are supported via the `arch` argument that runs through all
the factories below: ``resnet18`` (the base reproduction) and ``inception_v3``
(the generalization extension). Inception isn't in upstream's `LAYERS`, so we
define its probe layers here.

Naming convention:
    model name = '<arch>_spurious/imagenet'      (we reuse upstream's
        IMAGENET key so its `from src import ...` paths still resolve)
    dataset name = 'imagenet-spurious-text'      (upstream's existing key)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from milan_repro.milan_glue import upstream  # noqa: F401  (sys.path side effect)

import torch
from torch import nn
import torchvision

from src.exemplars import datasets as ex_datasets
from src.exemplars import models as ex_models
from src.exemplars.models import LAYERS

# Inception probe layers, chosen to mirror ResNet18's "stem conv + 4 stages"
# (conv1, layer1..4) so the layer-depth analysis is comparable across archs.
# Each is a feature-map-producing top-level child of torchvision's InceptionV3:
#   Conv2d_2b_3x3 (64ch), Mixed_5d (288), Mixed_6a (768), Mixed_6e (768),
#   Mixed_7c (2048).  ~3,936 neurons total vs ResNet18's 1,024.
LAYERS_INCEPTION_V3: Sequence[str] = (
    "Conv2d_2b_3x3", "Mixed_5d", "Mixed_6a", "Mixed_6e", "Mixed_7c",
)

DEFAULT_ARCH = "resnet18"


def _make_resnet18(num_classes: int = 10, **_: Any) -> nn.Module:
    """Factory that builds a torchvision ResNet18 with `num_classes` outputs."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _make_inception_v3(num_classes: int = 10, **_: Any) -> nn.Module:
    """Factory that builds a torchvision InceptionV3 with `num_classes` outputs.

    Built with `aux_logits=True` to match the training checkpoint (which
    includes the AuxLogits weights). In eval mode -- how the dissection and
    ablation passes run it -- the aux head is inert.
    """
    return torchvision.models.inception_v3(
        weights=None, num_classes=num_classes,
        aux_logits=True, init_weights=True,
    )


# Per-architecture registry: factory, probe layers, native input resolution,
# and the MILAN model key. Add new architectures here only.
_ARCHES: Mapping[str, Mapping[str, Any]] = {
    "resnet18": {
        "factory": _make_resnet18,
        "layers": LAYERS.RESNET18,
        "image_size": 224,
        "model_name": "resnet18_spurious/imagenet",
    },
    "inception_v3": {
        "factory": _make_inception_v3,
        "layers": LAYERS_INCEPTION_V3,
        "image_size": 299,
        "model_name": "inception_v3_spurious/imagenet",
    },
}

ARCHES = tuple(_ARCHES)


def _spec(arch: str) -> Mapping[str, Any]:
    if arch not in _ARCHES:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHES}")
    return _ARCHES[arch]


def layers_for(arch: str = DEFAULT_ARCH) -> Sequence[str]:
    """Probe layers for `arch`."""
    return _spec(arch)["layers"]


def image_size_for(arch: str = DEFAULT_ARCH) -> int:
    """Native input resolution for `arch` (224 for ResNet18, 299 for Inception)."""
    return _spec(arch)["image_size"]


def model_name_for(arch: str = DEFAULT_ARCH) -> str:
    """MILAN model key for `arch`."""
    return _spec(arch)["model_name"]


# Back-compat: the ResNet18 model key as a module constant (run_exemplars
# imports this name).
MODEL_NAME = _ARCHES[DEFAULT_ARCH]["model_name"]


def _transform_ckpt(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Unwrap our training checkpoint, which stores {state_dict, config, history}.

    Upstream's `ModelConfig.load` expects a plain state_dict, so we strip
    the wrapper.
    """
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload  # already a state_dict


def model_config(arch: str = DEFAULT_ARCH,
                 num_classes: int = 10) -> ex_models.ModelConfig:
    """Build the MILAN `ModelConfig` for `arch`'s trained model.

    The config tells MILAN how to instantiate the model, what layers to
    probe (e.g. ResNet18's conv1 + layer1..4, or Inception's stem conv +
    Mixed blocks), and how to load our checkpoint into it.
    """
    spec = _spec(arch)
    return ex_models.ModelConfig(
        spec["factory"],
        load_weights=True,
        transform_weights=_transform_ckpt,
        layers=spec["layers"],
        num_classes=num_classes,
    )


def model_configs(arch: str = DEFAULT_ARCH,
                  num_classes: int = 10) -> Mapping[str, ex_models.ModelConfig]:
    """Return a configs mapping suitable for `exemplars.models.load(..., configs=...)`."""
    return {model_name_for(arch): model_config(arch, num_classes=num_classes)}


def load_trained(ckpt_path: Path, arch: str = DEFAULT_ARCH,
                 num_classes: int = 10, device: str = "cuda") -> nn.Module:
    """Load a trained classifier directly (without going through MILAN's hub).

    Used by the editing experiment, which needs the model in plain torch
    form so we can ablate and evaluate it.
    """
    model = _spec(arch)["factory"](num_classes=num_classes)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = _transform_ckpt(payload)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# The spurious-text dataset config is already in upstream as
# `KEYS.IMAGENET_SPURIOUS_TEXT`; no need to redefine it.
SPURIOUS_DATASET_KEY = ex_datasets.KEYS.IMAGENET_SPURIOUS_TEXT
