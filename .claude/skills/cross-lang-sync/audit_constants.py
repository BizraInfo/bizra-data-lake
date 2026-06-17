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
    "MIN_CONFIDENCE": {"python_pattern": r"MIN_CONFIDENCE:\s*Final\[float\]\s*=\s*([\d.]+)", "type": float},
    "MAX_HARM_SCORE": {"python_pattern": r"MAX_HARM_SCORE:\s*Final\[float\]\s*=\s*([\d.]+)", "type": float},
}

PROOFSPACE_EXPECTED_REEXPORTS = {
    "IHSAN_THRESHOLD": "bizra_core::IHSAN_THRESHOLD",
    "ADL_GINI_MAX": "bizra_core::omega::ADL_GINI_THRESHOLD",
    "MAX_HARM_SCORE": "bizra_core::MAX_HARM_SCORE",
    "MIN_CONFIDENCE": "bizra_core::MIN_CONFIDENCE",
    "SNR_FLOOR": "bizra_core::SNR_THRESHOLD",
    "SNR_MINIMUM": "bizra_core::SNR_THRESHOLD",
    "SNR_THRESHOLD": "bizra_core::SNR_THRESHOLD",
}

PYTHON_MIRROR_SURFACES = {
    "scripts/ci_proof_pyramid_gate.py": {
        "IHSAN_THRESHOLD",
        "SNR_THRESHOLD",
        "ADL_GINI_MAX",
        "MAX_HARM_SCORE",
        "MIN_CONFIDENCE",
    },
    "runtime/core/constants.py": {
        "IHSAN_THRESHOLD",
    },
    "bizra-node0/core/integration/constants.py": {
        "IHSAN_THRESHOLD",
        "SNR_THRESHOLD",
        "ADL_GINI_THRESHOLD",
        "ADL_HARBERGER_TAX_RATE",
        "MIN_CONFIDENCE",
        "MAX_HARM_SCORE",
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


def audit_proofspace_reexports(repo_root: Path) -> dict:
    """Verify ProofSpace threshold constants re-export canonical Rust values."""
    proofspace_root = repo_root / "bizra-omega" / "bizra-proofspace"
    rust_files = sorted((proofspace_root / "src").glob("*.rs"))
    rust_files.extend(sorted((proofspace_root / "benches").glob("*.rs")))

    reexports = []
    violations = []
    const_pattern = re.compile(
        r"^\s*(?:pub\s+)?const\s+"
        r"(IHSAN_THRESHOLD|ADL_GINI_MAX|MAX_HARM_SCORE|MIN_CONFIDENCE|SNR_FLOOR|SNR_MINIMUM|SNR_THRESHOLD)"
        r"\s*:\s*(?:f64|u32|Decimal)\s*=\s*([^;]+);"
    )
    pub_use_pattern = re.compile(r"^\s*pub\s+use\s+bizra_core::(IHSAN_THRESHOLD)\s*;")

    for rust_file in rust_files:
        try:
            lines = rust_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            violations.append(
                {
                    "file": str(rust_file.relative_to(repo_root)),
                    "line": None,
                    "constant": None,
                    "rhs": None,
                    "reason": f"unreadable: {exc}",
                }
            )
            continue

        for line_no, line in enumerate(lines, 1):
            const_match = const_pattern.search(line)
            pub_use_match = pub_use_pattern.search(line)
            if pub_use_match:
                constant = pub_use_match.group(1)
                rhs = f"bizra_core::{constant}"
            elif const_match:
                constant = const_match.group(1)
                rhs = const_match.group(2).strip()
            else:
                continue

            entry = {
                "file": str(rust_file.relative_to(repo_root)),
                "line": line_no,
                "constant": constant,
                "rhs": rhs,
            }
            expected = PROOFSPACE_EXPECTED_REEXPORTS[constant]
            if rhs == expected:
                reexports.append(entry)
            else:
                violation = dict(entry)
                violation["expected"] = expected
                violations.append(violation)

    return {
        "status": "DRIFT_DETECTED" if violations else "ALIGNED",
        "reexports": reexports,
        "violations": violations,
    }


def audit_python_mirror_surfaces(repo_root: Path) -> dict:
    """Check known Python mirror files for Tier-1 hardcoded numeric copies."""
    surfaces = []
    violations = []

    for relative_path, constants in PYTHON_MIRROR_SURFACES.items():
        path = repo_root / relative_path
        surface = {"file": relative_path, "constants": sorted(constants)}
        surfaces.append(surface)
        if not path.exists():
            violations.append(
                {
                    "file": relative_path,
                    "line": None,
                    "constant": None,
                    "reason": "missing mirror file",
                }
            )
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            violations.append(
                {
                    "file": relative_path,
                    "line": None,
                    "constant": None,
                    "reason": f"unreadable: {exc}",
                }
            )
            continue

        assignment_pattern = re.compile(
            rf"^\s*({'|'.join(re.escape(name) for name in sorted(constants))})"
            r"\s*(?::[^=]+)?=\s*[0-9]"
        )
        for line_no, line in enumerate(lines, 1):
            match = assignment_pattern.search(line)
            if match:
                violations.append(
                    {
                        "file": relative_path,
                        "line": line_no,
                        "constant": match.group(1),
                        "reason": "hardcoded numeric mirror",
                    }
                )

    return {
        "status": "DRIFT_DETECTED" if violations else "ALIGNED",
        "surfaces": surfaces,
        "violations": violations,
    }


def find_rust_workspace_rogues(repo_root: Path) -> list[dict]:
    """Find Tier-1 Rust numeric constants outside canonical Rust sources."""
    omega_root = repo_root / "bizra-omega"
    if not omega_root.exists():
        return []

    rogues = []
    rogue_pattern = re.compile(
        r"^\s*(?:pub\s+)?const\s+"
        r"(IHSAN_THRESHOLD|ADL_GINI_MAX|MAX_HARM_SCORE|MIN_CONFIDENCE|SNR_THRESHOLD|SNR_FLOOR|SNR_MINIMUM|HARBERGER_TAX_RATE)"
        r"\s*:\s*(?:f64|u32|Decimal)\s*=\s*[0-9]"
    )
    ignored_parts = (
        Path("bizra-core") / "src",
        Path("bizra-resourcepool") / "src",
        Path("bizra-node0"),
    )

    for rust_file in omega_root.rglob("*.rs"):
        relative_to_omega = rust_file.relative_to(omega_root)
        if any(
            relative_to_omega == ignored or ignored in relative_to_omega.parents
            for ignored in ignored_parts
        ):
            continue
        try:
            lines = rust_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            match = rogue_pattern.search(line)
            if match:
                rogues.append(
                    {
                        "file": str(rust_file.relative_to(repo_root)),
                        "line": line_no,
                        "constant": match.group(1),
                        "source": line.strip(),
                    }
                )
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
    proofspace_reexports = audit_proofspace_reexports(REPO_ROOT)
    python_mirror_surfaces = audit_python_mirror_surfaces(REPO_ROOT)
    rust_workspace_rogues = find_rust_workspace_rogues(REPO_ROOT)

    violation_count = (
        drift_count
        + len(rogues)
        + len(proofspace_reexports["violations"])
        + len(python_mirror_surfaces["violations"])
        + len(rust_workspace_rogues)
    )

    # ─── Output ──────────────────────────────────────────────────────────
    if json_output:
        print(json.dumps({
            "status": "DRIFT_DETECTED" if violation_count > 0 else "ALIGNED",
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
            "proofspace_reexports": proofspace_reexports,
            "python_mirror_surfaces": python_mirror_surfaces,
            "rust_workspace_rogues": rust_workspace_rogues,
        }, indent=2))
    else:
        overall = "DRIFT DETECTED" if violation_count > 0 else "ALIGNED"
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

        if proofspace_reexports["violations"]:
            print(
                f"\nProofSpace Re-export Violations "
                f"({len(proofspace_reexports['violations'])}):"
            )
            for violation in proofspace_reexports["violations"]:
                print(
                    "  - "
                    f"{violation['constant']} in {violation['file']}:{violation['line']} "
                    f"uses {violation['rhs']} (expected {violation['expected']})"
                )

        if python_mirror_surfaces["violations"]:
            print(
                f"\nPython Mirror Violations "
                f"({len(python_mirror_surfaces['violations'])}):"
            )
            for violation in python_mirror_surfaces["violations"]:
                print(
                    "  - "
                    f"{violation['constant']} in {violation['file']}:{violation['line']} "
                    f"({violation['reason']})"
                )

        if rust_workspace_rogues:
            print(f"\nRust Workspace Rogue Definitions ({len(rust_workspace_rogues)}):")
            for rogue in rust_workspace_rogues:
                print(
                    "  - "
                    f"{rogue['constant']} in {rogue['file']}:{rogue['line']} "
                    f"({rogue['source']})"
                )

        print()

    sys.exit(1 if violation_count > 0 else 0)


if __name__ == "__main__":
    main()
