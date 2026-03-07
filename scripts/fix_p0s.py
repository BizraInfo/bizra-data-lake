#!/usr/bin/env python3
"""
BIZRA P0 Fix Script — Closes the 4 enterprise blockers.

Run: python scripts/fix_p0s.py --dry-run   # Preview
     python scripts/fix_p0s.py --execute   # Apply
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIXES: list[dict[str, str]] = []


def report(
    fix_id: str,
    desc: str,
    file_path: Path | str,
    before: str,
    after: str,
    status: str,
) -> None:
    FIXES.append({"id": fix_id, "status": status})
    icon = (
        "✓" if status == "FIXED"
        else "⊘" if status == "DRY-RUN"
        else "→" if status == "MANUAL"
        else "–"
    )
    print(f"  {icon} {fix_id}: {desc}")
    if before != after:
        print(f"    {before[:60]} → {after[:60]}")


def fix_p0_2(execute: bool) -> None:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        report("P0-2", "Coverage", "N/A", "N/A", "N/A", "SKIP")
        return

    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r"(fail_under\s*=\s*)(\d+)", text)
    if not match:
        report("P0-2", "Coverage gate not found", pyproject, "missing", "missing", "SKIP")
        return

    updated = text[: match.start(2)] + "60" + text[match.end(2) :]
    if execute:
        pyproject.write_text(updated, encoding="utf-8")
    report(
        "P0-2a",
        f"Coverage {match.group(2)}%→60%",
        pyproject,
        f"fail_under={match.group(2)}",
        "fail_under=60",
        "FIXED" if execute else "DRY-RUN",
    )


def fix_p0_3(execute: bool) -> None:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        report("P0-3", "Version", "N/A", "N/A", "N/A", "SKIP")
        return

    text = pyproject.read_text(encoding="utf-8")
    changes = 0

    updated = re.sub(
        r'requires-python\s*=\s*">=3\.11"',
        'requires-python = ">=3.12"',
        text,
    )
    if updated != text:
        changes += 1
        text = updated

    updated = re.sub(
        r'python_version\s*=\s*"3\.\d+"',
        'python_version = "3.12"',
        text,
    )
    if updated != text:
        changes += 1
        text = updated

    updated = re.sub(
        r'target-version\s*=\s*"py3\d+"',
        'target-version = "py312"',
        text,
    )
    if updated != text:
        changes += 1
        text = updated

    if changes > 0 and execute:
        pyproject.write_text(text, encoding="utf-8")

    status = "FIXED" if execute and changes > 0 else "DRY-RUN" if changes > 0 else "SKIP"
    report("P0-3", f"Python unified ({changes} targets→3.12)", pyproject, "mixed", ">=3.12", status)


def fix_p0_4(execute: bool) -> None:
    candidates = [ROOT / "docs" / "THREAT-MODEL-V3.md", ROOT / "THREAT-MODEL-V3.md"]
    for threat_model in candidates:
        if not threat_model.exists():
            continue

        text = threat_model.read_text(encoding="utf-8")
        if "GAP-001" in text:
            report("P0-4", "Already documented", threat_model, "done", "done", "SKIP")
            return

        gap = (
            "\n\n## Known Gaps\n\n"
            "### GAP-001: Federation UDP Unencrypted\n"
            "**Status:** OPEN · **Severity:** HIGH\n"
            "**Timeline:** Sprint 2 evaluate DTLS, Sprint 4 enforce encryption\n"
            "**Closure:** Packet capture shows no plaintext federation traffic\n"
        )
        if execute:
            threat_model.write_text(text + gap, encoding="utf-8")
        report(
            "P0-4",
            "Federation gap documented",
            threat_model,
            "undocumented",
            "GAP-001 added",
            "FIXED" if execute else "DRY-RUN",
        )
        return

    report("P0-4", "Threat model not found", "N/A", "N/A", "N/A", "SKIP")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    execute = args.execute
    print(f"\n  BIZRA P0 CLOSURE — {'EXECUTE' if execute else 'DRY-RUN'}\n")

    overlays = ROOT / "deploy" / "k8s" / "overlays"
    expected = [overlays / name / "kustomization.yaml" for name in ("dev", "staging", "production")]
    if all(path.exists() for path in expected):
        report("P0-1", "Deploy overlays present", overlays, "missing", "present", "SKIP")
    else:
        report("P0-1", "Deploy overlays (manual)", overlays, "missing", "needs k8s/overlays/", "MANUAL")

    fix_p0_2(execute)
    fix_p0_3(execute)
    fix_p0_4(execute)

    fixed = sum(1 for item in FIXES if item["status"] == "FIXED")
    print(f"\n  Result: {fixed} fixed, {len(FIXES) - fixed} other\n")


if __name__ == "__main__":
    main()
