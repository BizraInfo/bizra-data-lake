#!/usr/bin/env python3
"""
Generate the canonical truth pack from live repository state.

The truth pack is a machine-readable summary of volatile-but-governed facts that
docs and CI can share instead of duplicating by hand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "knowledge" / "canonical_truth_pack.json"
CONSTANTS_PATH = ROOT / "core" / "integration" / "constants.py"
API_POLICY_PATH = ROOT / "core" / "sovereign" / "api_exposure_policy.py"
RUST_WORKSPACE_PATH = ROOT / "bizra-omega" / "Cargo.toml"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

_FLOAT_CONSTANT_PATTERN = r"^{name}\s*:\s*Final\[float\]\s*=\s*([\d.]+)"


def _extract_float_constant(name: str, root: Path = ROOT) -> float:
    text = (root / "core" / "integration" / "constants.py").read_text(encoding="utf-8")
    pattern = re.compile(_FLOAT_CONSTANT_PATTERN.format(name=re.escape(name)), re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find float constant {name} in constants.py")
    return float(match.group(1))


def _count_rust_workspace_members(root: Path = ROOT) -> int:
    text = (root / "bizra-omega" / "Cargo.toml").read_text(encoding="utf-8")
    block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
    if not block:
        return 0
    return sum(
        1
        for line in block.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _count_workflow_files(root: Path = ROOT) -> int:
    workflows_dir = root / ".github" / "workflows"
    return len(
        [
            path
            for path in workflows_dir.iterdir()
            if path.is_file() and path.suffix in {".yml", ".yaml"}
        ]
    )


def _load_api_policy_module(root: Path = ROOT) -> Any:
    policy_path = root / "core" / "sovereign" / "api_exposure_policy.py"
    module_name = "_bizra_canonical_truth_pack_api_policy"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load API policy module from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module. Python 3.12 @dataclass internals
    # call sys.modules.get(cls.__module__) during class creation; without this
    # registration the lookup returns None and raises AttributeError. See CPython
    # Lib/dataclasses.py::_is_type.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _route_domain(path: str) -> str:
    if "/v1/" not in path:
        return path.strip("/") or "root"
    suffix = path.split("/v1/", maxsplit=1)[1]
    return suffix.split("/", maxsplit=1)[0] or "root"


def _route_summary(root: Path = ROOT) -> dict[str, Any]:
    module = _load_api_policy_module(root)
    policies = getattr(module, "API_ROUTE_POLICIES")

    by_exposure = Counter()
    domains = set()

    for policy in policies:
        by_exposure[str(policy.exposure)] += 1
        domains.add(_route_domain(policy.path))

    return {
        "total": len(policies),
        "domains": len(domains),
        "by_exposure": {
            "public": by_exposure["public"],
            "bootstrap_public": by_exposure["bootstrap_public"],
            "authenticated": by_exposure["authenticated"],
        },
        "source": "core/sovereign/api_exposure_policy.py",
    }


def build_truth_pack(root: Path = ROOT) -> dict[str, Any]:
    routes = _route_summary(root)
    return {
        "thresholds": {
            "unified_ihsan_threshold": _extract_float_constant(
                "UNIFIED_IHSAN_THRESHOLD", root
            ),
            "unified_snr_threshold": _extract_float_constant(
                "UNIFIED_SNR_THRESHOLD", root
            ),
            "adl_gini_threshold": _extract_float_constant("ADL_GINI_THRESHOLD", root),
            "source": "core/integration/constants.py",
        },
        "routes": routes,
        "workspace": {
            "rust_crates": _count_rust_workspace_members(root),
            "workflow_files": _count_workflow_files(root),
            "rust_workspace_source": "bizra-omega/Cargo.toml",
            "workflow_source": ".github/workflows/",
        },
    }


def write_truth_pack(output_path: Path = DEFAULT_OUTPUT, root: Path = ROOT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_truth_pack(root)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    args = parser.parse_args(argv)

    output = write_truth_pack(args.output)
    print(f"[CANONICAL-TRUTH-PACK] WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
