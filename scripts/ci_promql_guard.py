#!/usr/bin/env python3
"""
CI PromQL Guard
===============

Fail CI when ratio expressions use boolean vectors as numeric denominators,
for example:

    metric_a / (metric_b > 0)

This pattern is a recurring source of silent alert/rule misbehavior.
"""

from __future__ import annotations

import re
from pathlib import Path

RULE_FILES = (
    Path("deploy/monitoring/alerting-rules.yaml"),
    Path("deploy/monitoring/prometheus-config.yaml"),
    Path("deploy/monitoring/mcp-alerting-rules.yaml"),
)

# Matches "/ ( ... > 0 )" including multiline formatting.
DIV_BY_BOOLEAN_RE = re.compile(
    r"/\s*\(\s*[^)]*>\s*0(?:\.0+)?\s*\)",
    re.MULTILINE,
)

# Matches chained comparisons like "(... > 0) > 10".
BOOLEAN_CHAIN_RE = re.compile(
    r">\s*0(?:\.0+)?\s*\)\s*>",
    re.MULTILINE,
)


def _line_number(source: str, idx: int) -> int:
    return source.count("\n", 0, idx) + 1


def _snippet(source: str, idx: int, radius: int = 80) -> str:
    start = max(0, idx - radius)
    end = min(len(source), idx + radius)
    return source[start:end].replace("\n", " ")


def main() -> int:
    findings: list[str] = []

    for path in RULE_FILES:
        if not path.exists():
            findings.append(f"{path}: missing monitoring rules file")
            continue

        text = path.read_text(encoding="utf-8")

        for match in DIV_BY_BOOLEAN_RE.finditer(text):
            ln = _line_number(text, match.start())
            findings.append(
                f"{path}:{ln}: division-by-boolean pattern: { _snippet(text, match.start()) }"
            )

        for match in BOOLEAN_CHAIN_RE.finditer(text):
            ln = _line_number(text, match.start())
            findings.append(
                f"{path}:{ln}: chained boolean comparison pattern: { _snippet(text, match.start()) }"
            )

    if findings:
        print("[PROMQL-GUARD] FAILED")
        for finding in findings:
            print(f"- {finding}")
        print(
            "\nUse guard style: (numerator / denominator) <threshold> and denominator > 0"
        )
        return 1

    print("[PROMQL-GUARD] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
