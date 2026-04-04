"""BIZRA Sovereign Kernel (FastAPI gateway).

This package exposes the House of Wisdom (Neo4j) through a token-gated API and
enforces Ihsan/Adl/Amanah via a fail-closed FATE gate.

Includes unified memory integration with BIZRA-DATA-LAKE (M1-M6 tiers).
"""

import sys
from pathlib import Path

# Inject project root into sys.path for portable imports
# This allows `from core.module import X` to work from any location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Export key modules for convenience
__all__ = [
    "data_lake_bridge",
    "unified_memory",
    "evidence_sync",
]
