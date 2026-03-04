"""Conftest for generated conformance tests.

Ensures bizra-constitution package is importable and
BIZRA_CONSTITUTION_PATH points to repo-root constitution.toml.
"""

import os
import sys
from pathlib import Path

_package_root = Path(__file__).resolve().parent.parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

_constitution_path = _package_root.parent / "constitution.toml"
if _constitution_path.exists():
    os.environ.setdefault("BIZRA_CONSTITUTION_PATH", str(_constitution_path))
