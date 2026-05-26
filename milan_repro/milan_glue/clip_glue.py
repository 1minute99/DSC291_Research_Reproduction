"""CLIP ViT-B/32 vision encoder wrapper for MILAN exemplar extraction.

Proposal new experiment 2 — "Applying MILAN to CLIP (perhaps the vision
encoder part) to observe the ablation study presented in Section 7."

CLIP's ViT-B/32 vision encoder is a Vision Transformer.  We expose its
transformer residual blocks as hookable layers so that MILAN's
`exemplars.discriminative` can record top-activating image patches.

The probed layers are the 12 transformer blocks (visual.transformer.resblocks.N)
— blocks 0 and 11 are shallowest / deepest.  Each block's output is a
(1 + 196) × 512 tensor; we take the spatial tokens (index 1:) and reshape
to a 14 × 14 feature map for NetDissect compatibility.

Usage:
    from milan_repro.milan_glue.clip_glue import load_clip_vision, CLIP_LAYERS
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

# CLIP ViT-B/32 transformer block layer names
# (visual.transformer.resblocks.{0..11})
CLIP_LAYERS: List[str] = [
    f"visual.transformer.resblocks.{i}" for i in range(12)
]

# Subset used by default — shallow (0), mid (5), deep (11)
CLIP_LAYERS_SUBSET: List[str] = [
    "visual.transformer.resblocks.0",
    "visual.transformer.resblocks.3",
    "visual.transformer.resblocks.6",
    "visual.transformer.resblocks.9",
    "visual.transformer.resblocks.11",
]


class CLIPVisionSpatialWrapper(nn.Module):
    """Wraps CLIP's visual encoder so it returns a spatial feature map.

    ViT-B/32 processes a 224×224 image as 7×7=49 patches of 32×32 px.
    Each transformer block outputs (batch, 50, 512) — the first token is
    the [CLS] token; the remaining 49 are spatial tokens arranged in 7×7.

    We insert forward hooks so that at each probed block the spatial tokens
    are reshaped to (batch, 512, 7, 7), which NetDissect can treat as a
    standard conv feature map.

    The model's `forward` still returns the standard CLIP image embedding
    (CLS token after projection) so downstream classification heads work.
    """

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model
        self._spatial_outputs: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    # ── Hook registration / removal ──────────────────────────────────────────

    def register_spatial_hooks(self, layers: Optional[List[str]] = None) -> None:
        """Attach reshape hooks to the requested transformer blocks."""
        if layers is None:
            layers = CLIP_LAYERS_SUBSET
        self._remove_hooks()
        for layer_name in layers:
            module = self._get_module(layer_name)
            handle = module.register_forward_hook(
                self._make_hook(layer_name))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._spatial_outputs.clear()

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # output: (batch, seq_len, embed_dim)
            # seq_len = 1 (CLS) + 49 (7×7 patches) for ViT-B/32
            spatial = output[:, 1:, :]          # (B, 49, 512)
            B, S, C = spatial.shape
            side = int(S ** 0.5)               # 7
            # Reshape to (B, C, H, W) — NetDissect expects channel-first
            self._spatial_outputs[name] = (
                spatial.permute(0, 2, 1).reshape(B, C, side, side))
        return hook

    def _get_module(self, dotted_name: str) -> nn.Module:
        mod = self.clip_model
        for part in dotted_name.split("."):
            mod = getattr(mod, part)
        return mod

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(images)

    def get_spatial(self, layer_name: str) -> torch.Tensor:
        """Return the most recently computed spatial map for a layer."""
        return self._spatial_outputs[layer_name]


def load_clip_vision(device: str = "cuda") -> CLIPVisionSpatialWrapper:
    """Load CLIP ViT-B/32 and wrap its vision encoder.

    Requires the `clip` package:
        pip install git+https://github.com/openai/CLIP.git
    """
    try:
        import clip
    except ImportError as e:
        raise ImportError(
            "OpenAI CLIP is required for this experiment.\n"
            "Install with: pip install git+https://github.com/openai/CLIP.git"
        ) from e

    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model = model.float()   # fp32 for NetDissect compatibility
    wrapper = CLIPVisionSpatialWrapper(model)
    wrapper.register_spatial_hooks(CLIP_LAYERS_SUBSET)
    return wrapper.to(device).eval()
