"""Make the upstream MILAN package (`milan/src/...`) importable.

The upstream repo expects to be imported as `from src import ...`. We add
`./milan` to sys.path so its top-level `src` package resolves. Our own
package is named `milan_repro` to avoid clashing with that `src`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Monkey-patch for OpenAI CLIP: setuptools >= 70 removed `pkg_resources.packaging`,
# but the pinned CLIP commit MILAN uses does `from pkg_resources import packaging`.
# Wire the standalone `packaging` package onto `pkg_resources` so that import
# succeeds without editing site-packages.
try:
    import pkg_resources
    import packaging as _packaging
    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = _packaging
except ImportError:
    pass

_HERE = Path(__file__).resolve()
_MILAN_DIR = _HERE.parents[2] / "milan"
_SHIMS_DIR = _HERE.parents[1] / "_shims"

# Shim packages (e.g. our minimal `allennlp`) must come FIRST so they
# beat any partial installs.
if str(_SHIMS_DIR) not in sys.path:
    sys.path.insert(0, str(_SHIMS_DIR))
if str(_MILAN_DIR) not in sys.path:
    sys.path.insert(0, str(_MILAN_DIR))


def ensure() -> None:
    """No-op called for its side effect of importing this module."""
    return None
