"""Make the upstream MILAN package (`milan/src/...`) importable.

The upstream repo expects to be imported as `from src import ...`. We add
`./milan` to sys.path so its top-level `src` package resolves. Our own
package is named `milan_repro` to avoid clashing with that `src`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_MILAN_DIR = _HERE.parents[2] / "milan"

if str(_MILAN_DIR) not in sys.path:
    sys.path.insert(0, str(_MILAN_DIR))


def ensure() -> None:
    """No-op called for its side effect of importing this module."""
    return None
