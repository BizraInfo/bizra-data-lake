#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

WIN_ABS_RE = re.compile(r"[A-Za-z]:\\\\")
UNIX_ABS_RE = re.compile(r"(^|\\s)(/Users/|/home/|/var/|/opt/)", re.IGNORECASE)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    failures: list[str] = []

    for path in (repo_root / ".bizra").rglob("*.y*ml"):
        if path.name.endswith(".example.yaml") or path.name.endswith(".example.yml"):
            continue
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        for idx, line in enumerate(text.splitlines(), start=1):
            if WIN_ABS_RE.search(line) or UNIX_ABS_RE.search(line):
                failures.append(f"{path}:{idx} contains machine-specific absolute path")
                break

    if failures:
        print("Portability lint failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

