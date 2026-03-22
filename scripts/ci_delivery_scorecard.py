#!/usr/bin/env python3
"""
Generate an executive delivery scorecard from the BIZRA delivery program manifest.

This turns the machine-readable program model into a concise Markdown artifact
for CI summaries, operator review, and release evidence packs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRAM_PATH = ROOT / "docs" / "program" / "bizra_delivery_program.json"


def load_program(path: Path = DEFAULT_PROGRAM_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_scorecard_markdown(program: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# BIZRA Delivery Scorecard")
    lines.append("")
    lines.append(f"- Program: `{program['program_id']}`")
    lines.append(f"- Version: `{program['version']}`")
    lines.append(f"- Status: `{program['status']}`")
    lines.append(f"- North Star: {program['north_star']}")
    lines.append("")
    lines.append("## Operating Graph")
    lines.append("")
    lines.append("`" + " -> ".join(program["operating_graph"]) + "`")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Dimension | Status | Measure |")
    lines.append("| --- | --- | --- |")
    for row in program["scorecard"]:
        lines.append(
            f"| `{row['dimension']}` | `{row['status']}` | `{row['measure']}` |"
        )

    lines.append("")
    lines.append("## Workstreams")
    lines.append("")
    lines.append("| ID | Priority | Name | Current Truth |")
    lines.append("| --- | --- | --- | --- |")
    for workstream in program["workstreams"]:
        lines.append(
            f"| `{workstream['id']}` | `{workstream['priority']}` | "
            f"{workstream['name']} | `{workstream['current_truth']}` |"
        )

    lines.append("")
    lines.append("## Next Horizons")
    lines.append("")
    roadmap = program["roadmap"]
    for horizon in ("next_7_days", "next_30_days", "next_60_days", "next_90_days"):
        lines.append(f"### {horizon}")
        lines.append("")
        for item in roadmap[horizon]:
            lines.append(f"- `{item}`")
        lines.append("")

    lines.append("## Top Next Step")
    lines.append("")
    top = program["top_next_step"]
    lines.append(f"- `{top['id']}`: {top['title']}")
    lines.append(f"- Reason: {top['reason']}")
    lines.append("")

    lines.append("## P0/P1 Risks")
    lines.append("")
    lines.append("| ID | Priority | Risk | Mitigation |")
    lines.append("| --- | --- | --- | --- |")
    for risk in program["risk_register"]:
        if risk["priority"] not in {"P0", "P1"}:
            continue
        lines.append(
            f"| `{risk['id']}` | `{risk['priority']}` | `{risk['risk']}` | "
            f"`{risk['mitigation']}` |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BIZRA delivery scorecard.")
    parser.add_argument(
        "--program",
        type=Path,
        default=DEFAULT_PROGRAM_PATH,
        help="Path to bizra_delivery_program.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for Markdown scorecard",
    )
    args = parser.parse_args()

    program = load_program(args.program)
    markdown = build_scorecard_markdown(program)

    if args.output is None:
        print(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote delivery scorecard to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
