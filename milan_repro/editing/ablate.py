"""Channel-zero ablation via forward hooks.

Used by the editing experiment to remove specific (layer, channel) pairs
from the trained ResNet18 without re-training. Reuses upstream's
`ablations.zero` rule and `ablations.ablated` context manager so we get
the exact same semantics as `experiments/edit.py`.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence, Tuple

from milan_repro.milan_glue import upstream  # noqa: F401

from torch import nn
from src.deps.netdissect.nethook import InstrumentedModel
from src.utils import ablations as up_ablations

Unit = Tuple[str, int]   # (layer_name, channel_index)


@contextmanager
def channels_zeroed(model: nn.Module,
                    units: Sequence[Unit]) -> Iterator[InstrumentedModel]:
    """Yield `model` with the given (layer, channel) pairs forced to zero."""
    with up_ablations.ablated(model, units, rule=up_ablations.zero) as instr:
        yield instr
