#!/usr/bin/env python3
"""Heartbeat Evidence Logger — captures ticks to evidence/heartbeat/.

Polls kernel daemon /api/heartbeat every 60s, appends JSONL evidence.
Designed for P0-HEARTBEAT gate: proves continuous constitutional operation.

Usage:
    python scripts/heartbeat_evidence_logger.py
    # Runs until killed (Ctrl+C) or 24 hours elapsed
"""

from __future__ import annotations

import json
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE_DIR = Path("evidence/heartbeat")
KERNEL_PORT = 9740
POLL_INTERVAL_S = 60
MAX_DURATION_S = 24 * 3600
HEARTBEAT_URL = f"http://127.0.0.1:{KERNEL_PORT}/api/heartbeat"

_running = True


def _signal_handler(sig: int, frame: object) -> None:
    global _running
    _running = False
    print(
        f"\n[{datetime.now(timezone.utc).isoformat()}] Signal {sig} — shutting down gracefully"
    )


def poll_heartbeat() -> dict | None:
    try:
        req = urllib.request.Request(
            HEARTBEAT_URL, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
        return None


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = EVIDENCE_DIR / f"heartbeat_log_{ts}.jsonl"
    start_time = time.monotonic()

    print("[P0-HEARTBEAT] Evidence logger started")
    print(f"  Log: {log_path}")
    print(f"  Polling: {HEARTBEAT_URL} every {POLL_INTERVAL_S}s")
    print(f"  Max duration: {MAX_DURATION_S // 3600}h")

    tick_count = 0
    error_count = 0
    first_beat = None

    while _running:
        elapsed = time.monotonic() - start_time
        if elapsed >= MAX_DURATION_S:
            print(f"\n[P0-HEARTBEAT] 24h gate reached — {tick_count} ticks logged")
            break

        data = poll_heartbeat()
        now = datetime.now(timezone.utc).isoformat()

        if data is not None:
            tick_count += 1
            if first_beat is None:
                first_beat = data.get("latest", {}).get("beat", "?")
            entry = {
                "timestamp": now,
                "elapsed_s": round(elapsed, 1),
                "tick": tick_count,
                "health": data.get("health"),
                "beat": data.get("latest", {}).get("beat"),
                "uptime_s": data.get("latest", {}).get("uptime_s"),
                "error_rate": data.get("latest", {}).get("error_rate"),
                "anomalies": data.get("anomalies", []),
                "rss_mb": data.get("latest", {}).get("memory_rss_mb"),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            status = "OK" if data.get("health") == "healthy" else "WARN"
            beat = data.get("latest", {}).get("beat", "?")
            print(
                f"  [{now}] tick={tick_count} beat={beat} status={status} elapsed={elapsed:.0f}s"
            )
        else:
            error_count += 1
            entry = {
                "timestamp": now,
                "elapsed_s": round(elapsed, 1),
                "tick": tick_count,
                "error": "unreachable",
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"  [{now}] UNREACHABLE (errors={error_count})")

        time.sleep(POLL_INTERVAL_S)

    # Write summary
    summary = {
        "start_time": datetime.fromtimestamp(
            time.time() - (time.monotonic() - start_time), tz=timezone.utc
        ).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.monotonic() - start_time, 1),
        "total_ticks": tick_count,
        "total_errors": error_count,
        "first_beat": first_beat,
        "log_file": str(log_path),
        "gate_pass": tick_count > 0 and error_count == 0,
    }
    summary_path = EVIDENCE_DIR / f"heartbeat_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[P0-HEARTBEAT] Summary: {summary_path}")
    print(f"  Ticks: {tick_count}, Errors: {error_count}, Pass: {summary['gate_pass']}")


if __name__ == "__main__":
    main()
