"""Conftest for bizra-constitution tests.

Sets BIZRA_CONSTITUTION_PATH and adds the package root to sys.path
so that imports like `from bizra_constitution import ...` resolve correctly.
"""

import os
import sys
from pathlib import Path

# Add the bizra-constitution directory to sys.path for direct imports
_package_root = Path(__file__).resolve().parent.parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

# Point at the repo-root constitution.toml (single source of truth)
_constitution_path = _package_root.parent / "constitution.toml"
if _constitution_path.exists():
    os.environ.setdefault("BIZRA_CONSTITUTION_PATH", str(_constitution_path))
