#!/usr/bin/env python3
"""
BIZRA Context Kernel Loader (High-SNR)
- Generates a concise prompt for Claude to load the kernel
- SNR-first output (no decorative glyphs)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

KERNEL_DIR = Path(__file__).parent.resolve()

FILE_ORDER: List[Tuple[str, str]] = [
    ("Core Context (load first)", "core-context.json"),
    ("Workflow Hooks", "hooks.yaml"),
    ("Session Memory", "memory.json"),
    ("Tools and Protocols", "tools.yaml"),
    ("Quality Standards", "standards.yaml"),
]


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def render_status(context: Dict[str, Any]) -> List[str]:
    identity = context.get("identity", {})
    overview = context.get("project_overview", {})
    backend = context.get("current_state", {}).get("backend", {})
    frontend = context.get("current_state", {}).get("frontend", {})

    now = datetime.now().astimezone()

    return [
        f"Time: {now.strftime('%Y-%m-%d %H:%M %Z')}",
        f"User: {identity.get('user', 'Unknown')}",
        f"Project: {overview.get('name', 'Unknown')}",
        f"Backend Quality: {backend.get('quality_score', 'N/A')}%",
        f"Frontend Health: {frontend.get('status', 'N/A')}",
    ]


def render_prompt(mode: str) -> List[str]:
    lines: List[str] = []
    header = "Activate BIZRA Context Kernel (High-SNR)" if mode == "snr" else "Activate BIZRA Context Kernel"
    lines.append(header)
    lines.append("Load the following files in order:")
    for label, filename in FILE_ORDER:
        lines.append(f"- {label}: {KERNEL_DIR / filename}")
    lines.append("")
    lines.append("After loading, reply with:")
    lines.append("- Current local time")
    lines.append("- Loaded context summary")
    lines.append("- Available workflow hooks")
    lines.append("- Ready for autonomous execution")
    return lines


def print_kernel_prompt(mode: str = "snr") -> None:
    print("=" * 70)
    print("BIZRA CONTEXT KERNEL v1.1 - SESSION INITIALIZATION")
    print("=" * 70)
    print()

    context = load_json(KERNEL_DIR / "core-context.json")
    for line in render_status(context):
        print(line)
    print()

    print("=" * 70)
    print("COPY THIS PROMPT TO CLAUDE")
    print("=" * 70)
    print()
    for line in render_prompt(mode):
        print(line)
    print()
    print("=" * 70)
    print("QUICK LOAD COMMAND: @load-kernel")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="BIZRA Context Kernel Loader (High-SNR)")
    parser.add_argument(
        "--mode",
        choices=["snr", "standard"],
        default="snr",
        help="snr outputs concise, high-signal prompt",
    )
    args = parser.parse_args()
    print_kernel_prompt(mode=args.mode)


if __name__ == "__main__":
    main()
