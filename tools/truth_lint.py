#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED = {"VERIFIED", "MEASURED", "TARGET", "DERIVED"}
TRUTH_RE = re.compile(r"^\s*Truth\s*:\s*(\w+)\b", re.IGNORECASE)


def find_truth_label(path: Path, max_lines: int = 80) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    for line in lines[:max_lines]:
        m = TRUTH_RE.match(line)
        if m:
            return m.group(1).upper()
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "README.md",
        repo_root / "SUMMARY.md",
        repo_root / "VALIDATION.txt",
        repo_root / "bizra-genesis-node" / "SECURITY.md",
        repo_root / "bizra-genesis-node" / "QUICKSTART.md",
    ]

    failures: list[str] = []
    for path in targets:
        if not path.exists():
            continue
        label = find_truth_label(path)
        if label is None:
            failures.append(f"missing Truth header: {path}")
            continue
        if label not in ALLOWED:
            failures.append(f"invalid Truth label '{label}' in {path} (allowed: {sorted(ALLOWED)})")

    if failures:
        print("Truth lint failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

