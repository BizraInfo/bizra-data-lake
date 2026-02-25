#!/usr/bin/env python3
"""
Cross-Language Constant Synchronization Audit

Compares constitutional thresholds between Python (authoritative) and Rust
to detect drift. Exit code 0 = aligned, 1 = drift detected.

Usage:
    python3 .claude/skills/cross-lang-sync/audit_constants.py
    python3 .claude/skills/cross-lang-sync/audit_constants.py --json
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ─── Canonical file locations ────────────────────────────────────────────────
PYTHON_CONSTANTS = REPO_ROOT / "core" / "integration" / "constants.py"
RUST_LIB = REPO_ROOT / "bizra-omega" / "bizra-core" / "src" / "lib.rs"
RUST_OMEGA = REPO_ROOT / "bizra-omega" / "bizra-core" / "src" / "omega.rs"
RUST_CONSTITUTION = REPO_ROOT / "bizra-omega" / "bizra-core" / "src" / "constitution.rs"
RUST_RESOURCEPOOL = REPO_ROOT / "bizra-omega" / "bizra-resourcepool" / "src" / "lib.rs"

# ─── Constants to audit (name → expected Python value) ───────────────────────
# rust_names: list of alternative names to search for in Rust (name mapping)
TIER1_CONSTANTS = {
    "IHSAN_THRESHOLD": {"python_pattern": r"IHSAN_THRESHOLD:\s*Final\[float\]\s*=\s*([\d.]+)", "type": float},
    "SNR_THRESHOLD": {"python_pattern": r"SNR_THRESHOLD:\s*Final\[float\]\s*=\s*([\d.]+)", "type": float},
    "ADL_GINI_THRESHOLD": {"python_pattern": r"ADL_GINI_THRESHOLD:\s*Final\[float\]\s*=\s*([\d.]+)", "type": float},
    "ADL_HARBERGER_TAX_RATE": {
        "python_pattern": r"ADL_HARBERGER_TAX_RATE:\s*Final\[float\]\s*=\s*([\d.]+)",
        "type": float,
        "rust_names": ["ADL_HARBERGER_TAX_RATE", "HARBERGER_TAX_RATE"],
    },
}

# Matches both f64 literals and Decimal::from_parts(n, 0, 0, false, scale) patterns
RUST_CONST_PATTERN = r"pub\s+const\s+{name}:\s*(?:f64|Decimal)\s*=\s*"


def extract_python_value(name: str, spec: dict, content: str) -> float | None:
    """Extract a constant value from Python source."""
    match = re.search(spec["python_pattern"], content)
    if match:
        return spec["type"](match.group(1))
    return None


def extract_rust_values(name: str, rust_files: dict[str, str]) -> list[tuple[str, float, int]]:
    """Extract all definitions of a constant from Rust files. Returns [(file, value, line)]."""
    results = []
    # Pattern 1: pub const NAME: f64 = 0.95;
    pat_f64 = re.compile(rf"pub\s+const\s+{name}:\s*f64\s*=\s*([\d.]+)")
    # Pattern 2: pub const NAME: Decimal = Decimal::from_parts(n, 0, 0, false, scale); // 0.07
    pat_decimal = re.compile(
        rf"pub\s+const\s+{name}:\s*Decimal\s*=\s*Decimal::from_parts\(\s*(\d+).*?(\d+)\s*\)"
    )
    # Pattern 3: value in trailing comment "// 0.07"
    pat_comment_val = re.compile(rf"pub\s+const\s+{name}:.*//\s*([\d.]+)")
    for filepath, content in rust_files.items():
        for i, line in enumerate(content.splitlines(), 1):
            match = pat_f64.search(line)
            if match:
                results.append((filepath, float(match.group(1)), i))
                continue
            match = pat_decimal.search(line)
            if match:
                # Decimal::from_parts(numerator, 0, 0, false, scale) → numerator / 10^scale
                numerator = int(match.group(1))
                scale = int(match.group(2))
                value = numerator / (10 ** scale)
                results.append((filepath, value, i))
                continue
            # Fallback: check comment-documented value
            match = pat_comment_val.search(line)
            if match:
                results.append((filepath, float(match.group(1)), i))
    return results


def find_rogue_definitions(repo_root: Path) -> list[str]:
    """Find constants defined outside canonical files."""
    rogues = []
    canonical_py = str(PYTHON_CONSTANTS)

    # Check Python files
    for py_file in (repo_root / "core").rglob("*.py"):
        if str(py_file) == canonical_py:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in TIER1_CONSTANTS:
            # Look for direct assignment (not imports)
            pattern = rf"^{name}\s*[=:]\s*[\d.]"
            if re.search(pattern, content, re.MULTILINE):
                rogues.append(f"Python rogue: {name} redefined in {py_file.relative_to(repo_root)}")

    return rogues


def main():
    json_output = "--json" in sys.argv

    # ─── Read source files ───────────────────────────────────────────────
    if not PYTHON_CONSTANTS.exists():
        print(f"ERROR: Python constants file not found: {PYTHON_CONSTANTS}")
        sys.exit(2)

    python_content = PYTHON_CONSTANTS.read_text(encoding="utf-8")

    rust_files = {}
    for rust_path in [RUST_LIB, RUST_OMEGA, RUST_CONSTITUTION, RUST_RESOURCEPOOL]:
        if rust_path.exists():
            rust_files[str(rust_path.relative_to(REPO_ROOT))] = rust_path.read_text(encoding="utf-8")

    if not rust_files:
        print("WARNING: No Rust source files found — skipping Rust comparison")

    # ─── Audit Tier 1 constants ──────────────────────────────────────────
    results = []
    drift_count = 0

    for name, spec in TIER1_CONSTANTS.items():
        py_val = extract_python_value(name, spec, python_content)
        # Try all rust_names (supports cross-language name mapping)
        rust_names = spec.get("rust_names", [name])
        rust_defs = []
        for rn in rust_names:
            rust_defs.extend(extract_rust_values(rn, rust_files))

        status = "ALIGNED"
        details = []

        if py_val is None:
            status = "MISSING_PYTHON"
            details.append("Not found in Python constants")
        elif not rust_defs:
            status = "MISSING_RUST"
            details.append("Not found in Rust sources")
        else:
            for rust_file, rust_val, rust_line in rust_defs:
                if abs(py_val - rust_val) > 1e-9:
                    status = "DRIFT"
                    drift_count += 1
                    details.append(
                        f"Python={py_val} vs Rust={rust_val} in {rust_file}:{rust_line}"
                    )

        results.append({
            "constant": name,
            "python_value": py_val,
            "rust_definitions": [(f, v, l) for f, v, l in rust_defs],
            "status": status,
            "details": details,
        })

    # ─── Find rogue definitions ──────────────────────────────────────────
    rogues = find_rogue_definitions(REPO_ROOT)

    # ─── Output ──────────────────────────────────────────────────────────
    if json_output:
        print(json.dumps({
            "status": "DRIFT_DETECTED" if drift_count > 0 else "ALIGNED",
            "drift_count": drift_count,
            "results": [
                {
                    "constant": r["constant"],
                    "python_value": r["python_value"],
                    "rust_definitions": [
                        {"file": f, "value": v, "line": l}
                        for f, v, l in r["rust_definitions"]
                    ],
                    "status": r["status"],
                    "details": r["details"],
                }
                for r in results
            ],
            "rogue_definitions": rogues,
        }, indent=2))
    else:
        overall = "DRIFT DETECTED" if drift_count > 0 else "ALIGNED"
        print(f"\n{'='*60}")
        print(f"  Cross-Language Sync Audit — Status: {overall}")
        print(f"{'='*60}\n")

        print("Tier 1 — Constitutional Constants:")
        print(f"{'Constant':<28} {'Python':<10} {'Rust':<10} {'Status':<12}")
        print("-" * 60)

        for r in results:
            py_str = str(r["python_value"]) if r["python_value"] is not None else "N/A"
            if r["rust_definitions"]:
                for rust_file, rust_val, rust_line in r["rust_definitions"]:
                    rust_str = str(rust_val)
                    marker = "DRIFT" if r["status"] == "DRIFT" and abs(r["python_value"] - rust_val) > 1e-9 else r["status"]
                    print(f"{r['constant']:<28} {py_str:<10} {rust_str:<10} {marker:<12}")
            else:
                print(f"{r['constant']:<28} {py_str:<10} {'N/A':<10} {r['status']:<12}")

        if any(r["details"] for r in results):
            print(f"\nDrift Details:")
            for r in results:
                for detail in r["details"]:
                    print(f"  - {r['constant']}: {detail}")

        if rogues:
            print(f"\nRogue Definitions ({len(rogues)}):")
            for rogue in rogues:
                print(f"  - {rogue}")

        print()

    sys.exit(1 if drift_count > 0 else 0)


if __name__ == "__main__":
    main()
