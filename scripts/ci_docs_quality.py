#!/usr/bin/env python3
"""
CI documentation quality gate.

This script enforces canonical documentation contracts:
- Required docs must exist.
- Core docs must expose required sections and links.
- Contract-sensitive code changes must include docs touch.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from datetime import date
from typing import Iterable, List

ROOT = pathlib.Path(__file__).resolve().parent.parent

REQUIRED_DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "docs/README.md",
    ROOT / "docs/OPERATIONS_RUNBOOK.md",
    ROOT / "docs/TESTING.md",
]

REQUIRED_README_TOKENS = [
    "docs/README.md",
    "docs/OPERATIONS_RUNBOOK.md",
    "docs/TESTING.md",
]

REQUIRED_PORTAL_TOKENS = [
    "## A+ Documentation Quality Gate",
    "docs/OPERATIONS_RUNBOOK.md",
    "docs/TESTING.md",
]

REQUIRED_CONTRIBUTING_TOKENS = [
    "## Documentation Quality Gate",
    "docs/README.md",
    "scripts/ci_docs_quality.py",
]

DOC_TOUCH_PATHS = (
    "docs/",
    "README.md",
    "CONTRIBUTING.md",
)

CONTRACT_SENSITIVE_PATHS = (
    "core/sovereign/",
    "core/proof_engine/",
    "core/pci/",
    "bizra-omega/bizra-api/",
    "deploy/",
    "schemas/",
    ".github/workflows/",
)


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert(condition: bool, message: str, failures: List[str]) -> None:
    if not condition:
        failures.append(message)


def _parse_last_updated(text: str, failures: List[str], label: str) -> None:
    match = re.search(r"^Last updated:\s*(\d{4}-\d{2}-\d{2})\s*$", text, re.MULTILINE)
    if not match:
        failures.append(f"{label}: missing 'Last updated: YYYY-MM-DD' metadata.")
        return

    raw = match.group(1)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError:
        failures.append(f"{label}: invalid date '{raw}'.")
        return

    if parsed > date.today():
        failures.append(f"{label}: future date '{raw}' is not allowed.")


def _load_changed_files(path: pathlib.Path | None) -> List[str]:
    if not path or not path.exists():
        return []

    lines = [line.strip() for line in _read(path).splitlines()]
    return [line for line in lines if line]


def _any_startswith(items: Iterable[str], prefixes: Iterable[str]) -> bool:
    return any(any(item.startswith(prefix) for prefix in prefixes) for item in items)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run docs quality policy checks.")
    parser.add_argument(
        "--changed-files",
        type=pathlib.Path,
        default=None,
        help="Path to newline-delimited list of changed files.",
    )
    args = parser.parse_args()

    failures: List[str] = []

    # 1) Required docs must exist.
    for doc in REQUIRED_DOCS:
        _assert(doc.exists(), f"Missing required documentation file: {doc.relative_to(ROOT)}", failures)

    if failures:
        print("Documentation quality gate failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    # 2) Canonical files must include required content contracts.
    readme = _read(ROOT / "README.md")
    portal = _read(ROOT / "docs/README.md")
    runbook = _read(ROOT / "docs/OPERATIONS_RUNBOOK.md")
    testing = _read(ROOT / "docs/TESTING.md")
    contributing = _read(ROOT / "CONTRIBUTING.md")

    for token in REQUIRED_README_TOKENS:
        _assert(token in readme, f"README.md missing required docs link: {token}", failures)

    for token in REQUIRED_PORTAL_TOKENS:
        _assert(token in portal, f"docs/README.md missing required token: {token}", failures)

    for token in REQUIRED_CONTRIBUTING_TOKENS:
        _assert(token in contributing, f"CONTRIBUTING.md missing required token: {token}", failures)

    _parse_last_updated(portal, failures, "docs/README.md")
    _parse_last_updated(runbook, failures, "docs/OPERATIONS_RUNBOOK.md")
    _parse_last_updated(testing, failures, "docs/TESTING.md")

    # 3) Contract-sensitive change guard.
    changed_files = _load_changed_files(args.changed_files)
    if changed_files:
        has_sensitive_change = _any_startswith(changed_files, CONTRACT_SENSITIVE_PATHS)
        has_docs_touch = _any_startswith(changed_files, DOC_TOUCH_PATHS)
        if has_sensitive_change and not has_docs_touch:
            failures.append(
                "Contract-sensitive code changed without documentation update. "
                "Touch docs/ or README.md/CONTRIBUTING.md in the same change."
            )

    if failures:
        print("Documentation quality gate failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Documentation quality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
