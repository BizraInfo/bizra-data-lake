#!/usr/bin/env python3
"""CLI for Giants Protocol backlog generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make script runnable from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.giants_protocol import build_backlog, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SAPE/Ihsan/SNR-gated cross-pollination backlog."
    )
    parser.add_argument(
        "--registry",
        default="config/giants_protocol_registry.json",
        help="Path to registry JSON file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of highest-priority opportunities to print.",
    )
    parser.add_argument(
        "--output",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="Optional output file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backlog = build_backlog(args.registry, top_n=args.top)

    if args.output == "json":
        rendered = json.dumps(backlog, indent=2)
    else:
        rendered = render_markdown(backlog)

    if args.out_file:
        path = Path(args.out_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
