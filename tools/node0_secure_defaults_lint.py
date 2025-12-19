#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    node0_main = repo_root / "bizra-genesis-node" / "backend" / "src" / "main.rs"
    if not node0_main.exists():
        return 0

    text = node0_main.read_text(encoding="utf-8", errors="replace")

    failures: list[str] = []

    if "AllowOrigin::any()" in text:
        failures.append("Node0 backend uses AllowOrigin::any(); CORS must be allowlist-only")

    if 'unwrap_or_else(|_| "0.0.0.0"' in text:
        failures.append('Node0 backend defaults API_HOST to "0.0.0.0"; must default to loopback')

    if "BIZRA_EXPOSE_EXTERNAL" not in text:
        failures.append("Node0 backend missing BIZRA_EXPOSE_EXTERNAL gate for external bind")

    if "127.0.0.1" not in text:
        failures.append('Node0 backend missing explicit loopback default "127.0.0.1"')

    if "Refusing to bind Node0 API" not in text:
        failures.append("Node0 backend missing explicit refusal when binding non-loopback without override")

    if failures:
        print("Node0 secure-defaults lint failed:")
        for item in failures:
            print(f"- {item} ({node0_main})")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

