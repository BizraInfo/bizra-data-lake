#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


def read_all_stderr(pipe, sink: list[str]) -> None:
    try:
        for line in pipe:
            sink.append(line)
    except Exception:
        return


def parse_int(raw: str | None, *, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Enforce a warning budget for the Node0 backend (staged ratchet gate)."
    )
    parser.add_argument(
        "--backend",
        default=str(repo_root / "bizra-genesis-node" / "backend"),
        help="Path to Node0 backend crate (default: bizra-genesis-node/backend)",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=parse_int(os.getenv("NODE0_MAX_WARNINGS"), default=205),
        help="Maximum allowed warnings (default: NODE0_MAX_WARNINGS or 205)",
    )
    args = parser.parse_args()

    backend_dir = Path(args.backend)
    manifest_expected = (backend_dir / "Cargo.toml").resolve()

    if not backend_dir.exists():
        print(f"node0 warning budget failed: backend directory not found: {backend_dir}")
        return 2
    if not manifest_expected.exists():
        print(f"node0 warning budget failed: manifest not found: {manifest_expected}")
        return 2

    env = os.environ.copy()
    env.setdefault("SQLX_OFFLINE", "true")

    proc = subprocess.Popen(
        ["cargo", "check", "--message-format=json", "-q"],
        cwd=str(backend_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stderr_lines: list[str] = []
    t = threading.Thread(target=read_all_stderr, args=(proc.stderr, stderr_lines), daemon=True)
    t.start()

    warnings = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if obj.get("reason") != "compiler-message":
            continue

        msg = obj.get("message") or {}
        if not isinstance(msg, dict):
            continue
        if msg.get("level") != "warning":
            continue

        manifest_path = obj.get("manifest_path")
        if isinstance(manifest_path, str) and manifest_path.strip():
            try:
                if Path(manifest_path).resolve() != manifest_expected:
                    continue
            except Exception:
                continue
        else:
            continue

        warnings += 1

    rc = proc.wait()
    t.join(timeout=2)
    stderr_text = "".join(stderr_lines).strip()

    if rc != 0:
        print("node0 warning budget failed: cargo check failed")
        if stderr_text:
            print(stderr_text)
        return 2

    if warnings > int(args.max_warnings):
        print("node0 warning budget failed:")
        print(f"- warnings={warnings} exceeds max_warnings={args.max_warnings}")
        print("- ratchet policy: reduce warnings then lower max_warnings; never raise without a documented exception")
        return 1

    print(f"[OK] node0 warnings={warnings} max_warnings={args.max_warnings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

