#!/usr/bin/env python3
"""
CI Docs-Truth Gate — Prevents README threshold drift from constants.py.

Checks:
1. README ADL Gini threshold matches ADL_GINI_THRESHOLD in constants.py.
2. README Rust crate count matches bizra-omega/Cargo.toml workspace members.
3. README Ihsan threshold matches UNIFIED_IHSAN_THRESHOLD in constants.py.
4. README SNR threshold matches UNIFIED_SNR_THRESHOLD in constants.py.

Standing on Giants: Deming (PDCA, 1950) — verify the document, not just the code.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _extract_constant(name: str) -> float | None:
    """Extract a Final[float] constant from core/integration/constants.py."""
    constants_path = ROOT / "core" / "integration" / "constants.py"
    pattern = re.compile(rf"^{re.escape(name)}\s*:\s*Final\[float\]\s*=\s*([\d.]+)", re.MULTILINE)
    match = pattern.search(constants_path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def _count_cargo_members() -> int:
    """Count workspace members in bizra-omega/Cargo.toml."""
    cargo_path = ROOT / "bizra-omega" / "Cargo.toml"
    text = cargo_path.read_text(encoding="utf-8")
    block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
    if not block:
        return 0
    lines = block.group(1).splitlines()
    return sum(
        1
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    )


def _check_readme_thresholds() -> list[str]:
    """Verify README.md constitutional thresholds match authoritative sources."""
    issues: list[str] = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # --- ADL Gini ---
    adl_const = _extract_constant("ADL_GINI_THRESHOLD")
    adl_readme_match = re.search(
        r"ADL.*Gini\s*\|\s*<=?\s*([\d.]+)", readme
    )
    if adl_const is not None and adl_readme_match:
        readme_val = float(adl_readme_match.group(1))
        if abs(readme_val - adl_const) > 1e-6:
            issues.append(
                f"README ADL Gini says {readme_val} but constants.py says {adl_const}"
            )

    # --- Ihsan ---
    ihsan_const = _extract_constant("UNIFIED_IHSAN_THRESHOLD")
    ihsan_readme_match = re.search(
        r"Ihsan.*\|\s*>=?\s*([\d.]+)", readme
    )
    if ihsan_const is not None and ihsan_readme_match:
        readme_val = float(ihsan_readme_match.group(1))
        if abs(readme_val - ihsan_const) > 1e-6:
            issues.append(
                f"README Ihsan says {readme_val} but constants.py says {ihsan_const}"
            )

    # --- SNR ---
    snr_const = _extract_constant("UNIFIED_SNR_THRESHOLD")
    snr_readme_match = re.search(
        r"SNR.*\|\s*>=?\s*([\d.]+)", readme
    )
    if snr_const is not None and snr_readme_match:
        readme_val = float(snr_readme_match.group(1))
        if abs(readme_val - snr_const) > 1e-6:
            issues.append(
                f"README SNR says {readme_val} but constants.py says {snr_const}"
            )

    # --- Rust crate count ---
    cargo_count = _count_cargo_members()
    crate_match = re.search(
        r"High-performance core \((\d+) Rust crates?\)", readme
    )
    if crate_match:
        readme_count = int(crate_match.group(1))
        if readme_count != cargo_count:
            issues.append(
                f"README says {readme_count} Rust crates but Cargo.toml has {cargo_count}"
            )

    return issues


def main() -> int:
    issues = _check_readme_thresholds()
    if issues:
        print("[DOCS-TRUTH-GATE] FAILED")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("[DOCS-TRUTH-GATE] PASS")
    print("README thresholds match constants.py and Cargo.toml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
