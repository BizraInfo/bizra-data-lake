#!/usr/bin/env python3
"""Generate Phase 46 release manifest with file paths and SHA-256 checksums.

Usage:
    python scripts/generate_release_manifest.py [--output FILE]

Writes to artifacts/phase46_release_manifest.txt by default.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Phase 46 + 46.1 approved files (exhaustive list)
PHASE46_FILES = [
    "core/__init__.py",
    "core/integration/constants.py",
    "core/living_memory/proactive.py",
    "core/memory/types.py",
    "core/prediction/__init__.py",
    "core/prediction/hmm_engine.py",
    "core/reasoning/__init__.py",
    "core/reasoning/got_bridge.py",
    "core/resonance.py",
    "core/search/__init__.py",
    "core/search/vector_search.py",
    "core/sovereign/apex_engine.py",
    "tools/mcp/sovereign_mcp_server.py",
    # Phase 47.1 rollout infrastructure
    "core/rollout/__init__.py",
    "core/rollout/canary.py",
    "core/rollout/hmm_gate.py",
    "core/rollout/metrics.py",
    "core/rollout/rollback.py",
    "deploy/monitoring/alerting-rules.yaml",
    # Test files
    "tests/core/mcp/test_sovereign_phase46_tools.py",
    "tests/core/prediction/test_hmm_engine.py",
    "tests/core/reasoning/test_got_bridge.py",
    "tests/core/search/test_vector_search.py",
    "tests/core/sovereign/test_apex_got_bridge_integration.py",
    "tests/core/test_resonance.py",
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(root: Path, output: Path) -> int:
    """Generate manifest and write to output path. Returns file count."""
    lines = []
    for relpath in sorted(PHASE46_FILES):
        full = root / relpath
        if not full.exists():
            lines.append(f"{relpath}:MISSING")
        else:
            sha = sha256_file(full)
            lines.append(f"{relpath}:{sha}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return len(lines)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output_arg = "artifacts/phase46_release_manifest.txt"

    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_arg = sys.argv[idx + 1]

    output = root / output_arg
    count = generate_manifest(root, output)
    print(f"Manifest written: {output} ({count} files)")


if __name__ == "__main__":
    main()
