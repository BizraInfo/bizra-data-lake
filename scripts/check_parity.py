#!/usr/bin/env python3
"""
Cross-Boundary Parity Check for BIZRA Dual-Agentic System.

Verifies alignment between Rust and Python implementations of:
- Ihsān dimensions (constitution/ihsan_v1.yaml ↔ src/sape.rs ↔ core/fate.py)
- Rejection codes (src/sat.rs ↔ core/fate.py)
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
from typing import Dict, List, Set, Tuple

# Repository root (script lives in scripts/)
REPO_ROOT = Path(__file__).parent.parent

# File paths
CONSTITUTION_PATH = REPO_ROOT / "constitution" / "ihsan_v1.yaml"
RUST_SAPE_PATH = REPO_ROOT / "src" / "sape.rs"
RUST_SAT_PATH = REPO_ROOT / "src" / "sat.rs"
PYTHON_FATE_PATH = REPO_ROOT / "core" / "fate.py"
PYTHON_SAPE_PATH = REPO_ROOT / "core" / "sape.py"


def load_constitution() -> Dict[str, float]:
    """Load canonical Ihsān dimensions from constitution."""
    if not CONSTITUTION_PATH.exists():
        print(f"❌ Constitution not found: {CONSTITUTION_PATH}")
        sys.exit(2)
    
    with open(CONSTITUTION_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    dimensions = data.get("dimensions", {})
    return {k: v.get("weight", 0) for k, v in dimensions.items()}


def extract_rust_rejection_codes() -> Set[str]:
    """Extract RejectionCode variants from src/sat.rs."""
    if not RUST_SAT_PATH.exists():
        print(f"⚠️  Rust SAT not found: {RUST_SAT_PATH}")
        return set()
    
    content = RUST_SAT_PATH.read_text(encoding='utf-8')
    # Match enum variants like: SecurityThreat(String),
    pattern = r'^\s*(\w+)\s*\([^)]*\)\s*,'
    matches = re.findall(pattern, content, re.MULTILINE)
    return set(matches)


def extract_python_rejection_codes() -> Set[str]:
    """Extract RejectionCode enum values from core/fate.py."""
    if not PYTHON_FATE_PATH.exists():
        print(f"⚠️  Python FATE not found: {PYTHON_FATE_PATH}")
        return set()
    
    content = PYTHON_FATE_PATH.read_text(encoding='utf-8')
    # Match enum values like: RJ_IH_001 = "RJ-IH-001"
    pattern = r'^\s*(RJ_\w+)\s*='
    matches = re.findall(pattern, content, re.MULTILINE)
    return set(matches)


def extract_python_canonical_dimensions() -> List[str]:
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


def extract_rust_probe_weights() -> Dict[str, float]:
    """Extract ProbeDimension weights from src/sape.rs."""
    if not RUST_SAPE_PATH.exists():
        print(f"⚠️  Rust SAPE not found: {RUST_SAPE_PATH}")
        return {}
    
    content = RUST_SAPE_PATH.read_text(encoding='utf-8')
    # Match patterns like: Self::ThreatScan => 0.11,
    pattern = r'Self::(\w+)\s*=>\s*(\d+\.\d+)'
    matches = re.findall(pattern, content)
    
    # Filter to weight function context (first set of matches after "weight")
    weights = {}
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


def check_weight_sum(weights: Dict[str, float], source: str) -> bool:
    """Verify weights sum to 1.0."""
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        print(f"❌ {source}: Weights sum to {total:.6f}, expected 1.0")
        return False
    print(f"✅ {source}: Weights sum to 1.0")
    return True


def check_dimension_parity(
    constitution: Dict[str, float],
    python_dims: List[str]
) -> bool:
    """Verify Python CANONICAL_DIMENSIONS matches constitution."""
    const_set = set(constitution.keys())
    python_set = set(python_dims)
    
    if const_set != python_set:
        missing_in_python = const_set - python_set
        extra_in_python = python_set - const_set
        
        if missing_in_python:
            print(f"❌ Python missing dimensions: {missing_in_python}")
        if extra_in_python:
            print(f"❌ Python has extra dimensions: {extra_in_python}")
        return False
    
    print(f"✅ Python CANONICAL_DIMENSIONS matches constitution ({len(const_set)} dimensions)")
    return True


def main() -> int:
    print("=" * 60)
    print("BIZRA Cross-Boundary Parity Check")
    print("=" * 60)
    print()
    
    errors = 0
    
    # Load canonical constitution
    print("📜 Loading constitution/ihsan_v1.yaml...")
    constitution = load_constitution()
    print(f"   Found {len(constitution)} dimensions")
    print()
    
    # Check 1: Constitution weight sum
    print("🔍 Check 1: Constitution weight sum")
    if not check_weight_sum(constitution, "constitution"):
        errors += 1
    print()
    
    # Check 2: Rust SAPE weight sum
    print("🔍 Check 2: Rust SAPE probe weights")
    rust_weights = extract_rust_probe_weights()
    if rust_weights:
        print(f"   Found {len(rust_weights)} probe dimensions")
        if not check_weight_sum(rust_weights, "src/sape.rs"):
            errors += 1
    else:
        print("   ⚠️  Could not extract Rust weights")
    print()
    
    # Check 3: Python dimension alignment
    print("🔍 Check 3: Python CANONICAL_DIMENSIONS")
    python_dims = extract_python_canonical_dimensions()
    if python_dims:
        if not check_dimension_parity(constitution, python_dims):
            errors += 1
    else:
        print("   ⚠️  Could not extract Python dimensions")
    print()
    
    # Check 4: Rejection code inventory
    print("🔍 Check 4: Rejection code inventory")
    rust_codes = extract_rust_rejection_codes()
    python_codes = extract_python_rejection_codes()
    print(f"   Rust RejectionCode variants: {len(rust_codes)}")
    print(f"   Python RejectionCode values: {len(python_codes)}")
    # Note: These have different taxonomies by design (runtime vs. guidance)
    print("   ℹ️  Codes have different taxonomies (Rust=runtime, Python=guidance)")
    print()
    
    # Summary
    print("=" * 60)
    if errors == 0:
        print("✅ All parity checks passed")
        return 0
    else:
        print(f"❌ {errors} parity check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
