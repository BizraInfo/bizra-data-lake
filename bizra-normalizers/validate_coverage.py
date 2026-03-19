#!/usr/bin/env python3
"""Compute Core-8 provider coverage (CV gate) for BIZRA normalizers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizers import (  # noqa: E402
    CONVERSATION_PLATFORMS,
    CORE8,
    detect_provider,
    parse_file,
)


def iter_json_files(root: Path):
    allowed_suffixes = {".json", ".jsonl"}
    if not root.exists():
        return
    if root.is_file() and root.suffix.lower() in allowed_suffixes:
        yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            yield path


def parse_payload(path: Path) -> Any | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".jsonl":
            rows: list[dict[str, Any]] = []
            for line in raw.splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
            return rows
        return json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None


def providers_from_paths(paths: list[Path]) -> set[str]:
    providers: set[str] = set()
    for root in paths:
        for path in iter_json_files(root):
            turns = parse_file(path)
            if turns:
                providers.update(
                    turn.provider for turn in turns if turn.provider in CORE8
                )
                continue

            payload = parse_payload(path)
            if payload is None:
                continue
            detected = detect_provider(payload, source_path=str(path))
            if detected in CORE8:
                providers.add(detected)
    return providers


def compute_cv(covered: set[str]) -> float:
    return round(len(covered & set(CORE8)) / len(CORE8), 4)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Core-8 provider coverage (CV gate)"
    )
    parser.add_argument("paths", nargs="*", help="Corpus directories/files to scan")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Scan built-in fixture files and include legacy Core-4 baseline",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    args = parser.parse_args()

    covered: set[str] = set()

    if args.fixtures:
        fixture_path = ROOT / "fixtures"
        covered |= providers_from_paths([fixture_path])
        # Existing conversation platform normalization lives in ingest_conversations.
        covered |= CONVERSATION_PLATFORMS
    elif args.paths:
        roots = [Path(p).expanduser().resolve() for p in args.paths]
        covered |= providers_from_paths(roots)
    else:
        parser.error("Provide --fixtures or at least one path")

    cv = compute_cv(covered)
    report = {
        "covered": sorted(covered),
        "covered_count": len(covered),
        "total": len(CORE8),
        "cv": cv,
        "gate": min(1.0, cv),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Core-8 Coverage Report")
        print(f"Covered providers: {report['covered_count']}/{report['total']}")
        print(f"CV = {report['cv']:.4f}")
        print("Providers:", ", ".join(report["covered"]))

    return 0 if cv >= 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
