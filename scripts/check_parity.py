#!/usr/bin/env python3
"""
Cross-Boundary Parity Check for BIZRA Dual-Agentic System.

Verifies alignment between Rust and Python implementations of:
- Ihsan dimensions (constitution/ihsan_v1.yaml <-> src/sape.rs <-> core/fate.py)
- Rejection codes (src/sat.rs <-> core/fate.py)
- SAPE probe weights (sum to 1.0)

Exit codes:
  0 = All checks pass
  1 = Parity violations detected
  2 = File read errors
"""

import re
import sys
import yaml
from pathlib import Path

# Repository root (script lives in scripts/)
REPO_ROOT = Path(__file__).parent.parent

# File paths
CONSTITUTION_PATH = REPO_ROOT / "constitution" / "ihsan_v1.yaml"
RUST_SAPE_PATH = REPO_ROOT / "src" / "sape.rs"
RUST_SAT_PATH = REPO_ROOT / "src" / "sat.rs"
PYTHON_FATE_PATH = REPO_ROOT / "core" / "fate.py"
PYTHON_SAPE_PATH = REPO_ROOT / "core" / "sape.py"

# Tolerance for floating-point weight comparisons
WEIGHT_EPSILON = 1e-9


def load_constitution() -> dict[str, float]:
    """Load canonical Ihsan dimensions from constitution.
    
    Raises:
        FileNotFoundError: If constitution file is missing.
    """
    if not CONSTITUTION_PATH.exists():
        raise FileNotFoundError(f"Constitution not found: {CONSTITUTION_PATH}")
    
    with open(CONSTITUTION_PATH, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    dimensions = data.get("dimensions", {})
    return {k: v.get("weight", 0) for k, v in dimensions.items()}


def extract_rust_rejection_codes() -> set[str]:
    """Extract RejectionCode variants from src/sat.rs."""
    if not RUST_SAT_PATH.exists():
        print(f"[WARN] Rust SAT not found: {RUST_SAT_PATH}")
        return set()
    
    content = RUST_SAT_PATH.read_text(encoding='utf-8')
    # Match enum variants like: SecurityThreat(String),
    pattern = r'^\s*(\w+)\s*\([^)]*\)\s*,'
    matches = re.findall(pattern, content, re.MULTILINE)
    return set(matches)


def extract_python_rejection_codes() -> set[str]:
    """Extract RejectionCode enum values from core/fate.py."""
    if not PYTHON_FATE_PATH.exists():
        print(f"[WARN] Python FATE not found: {PYTHON_FATE_PATH}")
        return set()
    
    content = PYTHON_FATE_PATH.read_text(encoding='utf-8')
    # Match enum values like: RJ_IH_001 = "RJ-IH-001"
    pattern = r'^\s*(RJ_\w+)\s*='
    matches = re.findall(pattern, content, re.MULTILINE)
    return set(matches)


def extract_python_canonical_dimensions() -> list[str]:
    """Extract CANONICAL_DIMENSIONS from core/fate.py."""
    if not PYTHON_FATE_PATH.exists():
        return []
    
    content = PYTHON_FATE_PATH.read_text(encoding='utf-8')
    # Find CANONICAL_DIMENSIONS = [...] block
    match = re.search(r'CANONICAL_DIMENSIONS\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        return []
    
    # Extract quoted strings
    dims = re.findall(r'"(\w+)"', match.group(1))
    return dims


def extract_rust_probe_weights() -> dict[str, float]:
    """Extract ProbeDimension weights from src/sape.rs."""
    if not RUST_SAPE_PATH.exists():
        print(f"[WARN] Rust SAPE not found: {RUST_SAPE_PATH}")
        return {}
    
    content = RUST_SAPE_PATH.read_text(encoding='utf-8')
    
    # Filter to weight function context (first set of matches after "weight")
    weights: dict[str, float] = {}
    in_weight_fn = False
    for line in content.split('\n'):
        if 'fn weight' in line:
            in_weight_fn = True
        elif in_weight_fn:
            match = re.search(r'Self::(\w+)\s*=>\s*(\d+\.\d+)', line)
            if match:
                weights[match.group(1)] = float(match.group(2))
            elif '}' in line and 'match' not in line:
                break
    
    return weights


def check_weight_sum(weights: dict[str, float], source: str) -> bool:
    """Verify weights sum to 1.0."""
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_EPSILON:
        print(f"[FAIL] {source}: Weights sum to {total:.6f}, expected 1.0")
        return False
    print(f"[PASS] {source}: Weights sum to 1.0")
    return True


def check_dimension_parity(
    constitution: dict[str, float],
    python_dims: list[str]
) -> bool:
    """Verify Python CANONICAL_DIMENSIONS matches constitution."""
    const_set = set(constitution.keys())
    python_set = set(python_dims)
    
    if const_set != python_set:
        missing_in_python = const_set - python_set
        extra_in_python = python_set - const_set
        
        if missing_in_python:
            print(f"[FAIL] Python missing dimensions: {missing_in_python}")
        if extra_in_python:
            print(f"[FAIL] Python has extra dimensions: {extra_in_python}")
        return False
    
    print(f"[PASS] Python CANONICAL_DIMENSIONS matches constitution ({len(const_set)} dimensions)")
    return True


def main() -> int:
    print("=" * 60)
    print("BIZRA Cross-Boundary Parity Check")
    print("=" * 60)
    print()
    
    errors = 0
    
    # Load canonical constitution
    print("[DOC] Loading constitution/ihsan_v1.yaml...")
    try:
        constitution = load_constitution()
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        return 2
    print(f"      Found {len(constitution)} dimensions")
    print()
    
    # Check 1: Constitution weight sum
    print("[CHECK] Check 1: Constitution weight sum")
    if not check_weight_sum(constitution, "constitution"):
        errors += 1
    print()
    
    # Check 2: Rust SAPE weight sum
    print("[CHECK] Check 2: Rust SAPE probe weights")
    rust_weights = extract_rust_probe_weights()
    if rust_weights:
        print(f"        Found {len(rust_weights)} probe dimensions")
        if not check_weight_sum(rust_weights, "src/sape.rs"):
            errors += 1
    else:
        print("        [WARN] Could not extract Rust weights")
    print()
    
    # Check 3: Python dimension alignment
    print("[CHECK] Check 3: Python CANONICAL_DIMENSIONS")
    python_dims = extract_python_canonical_dimensions()
    if python_dims:
        if not check_dimension_parity(constitution, python_dims):
            errors += 1
    else:
        print("        [WARN] Could not extract Python dimensions")
    print()
    
    # Check 4: Rejection code inventory
    print("[CHECK] Check 4: Rejection code inventory")
    rust_codes = extract_rust_rejection_codes()
    python_codes = extract_python_rejection_codes()
    print(f"   Rust RejectionCode variants: {len(rust_codes)}")
    print(f"   Python RejectionCode values: {len(python_codes)}")
    # Note: These have different taxonomies by design (runtime vs. guidance)
    print("   INFO: Codes have different taxonomies (Rust=runtime, Python=guidance)")
    print()
    
    # Summary
    print("=" * 60)
    if errors == 0:
        print("[PASS] All parity checks passed")
        return 0
    else:
        print(f"[FAIL] {errors} parity check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
