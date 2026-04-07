#!/usr/bin/env python3
"""BIZRA CLI Bridge — lightweight Python↔Rust bridge for runtime checks.

Called by the Rust CLI binary (`bizra status`) to check LM Studio connectivity
and other Python-dependent subsystems.

Usage:
    python bizra_cli_bridge.py status    → JSON with LM Studio connection info
    python bizra_cli_bridge.py health    → JSON with kernel daemon health
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error


def _detect_wsl_gateway() -> str:
    """Auto-detect WSL2 gateway IP (LM Studio host)."""
    try:
        with open("/proc/net/route") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[1] == "00000000":
                    # Default route — gateway is little-endian hex
                    hex_ip = parts[2]
                    return ".".join(
                        str(int(hex_ip[i : i + 2], 16)) for i in range(6, -1, -2)
                    )
    except (OSError, ValueError):
        pass
    # Fallback: parse `ip route`
    import subprocess

    try:
        out = subprocess.check_output(
            ["ip", "route", "show", "default"], text=True, timeout=5
        )
        for token in out.split():
            if "." in token:
                parts = token.split(".")
                if len(parts) == 4 and all(p.isdigit() for p in parts):
                    return token
    except (subprocess.SubprocessError, OSError):
        pass
    return "127.0.0.1"


def cmd_status() -> dict:
    """Check LM Studio connectivity and loaded models."""
    import os

    host = os.environ.get("LMSTUDIO_HOST") or _detect_wsl_gateway()
    port = os.environ.get("LMSTUDIO_PORT", "1234")
    url = f"http://{host}:{port}/api/v1/models"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            loaded = [m for m in models if m.get("loaded_instances")]
            return {
                "status": "connected",
                "host": f"{host}:{port}",
                "total_models": len(models),
                "loaded_models": len(loaded),
                "loaded_list": [m["id"] for m in loaded],
            }
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return {"status": "disconnected", "host": f"{host}:{port}"}


def cmd_health() -> dict:
    """Check kernel daemon health."""
    url = "http://127.0.0.1:9740/api/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {"status": "unreachable"}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    handlers = {"status": cmd_status, "health": cmd_health}
    fn = handlers.get(cmd, cmd_status)
    result = fn()
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
