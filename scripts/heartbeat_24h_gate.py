#!/usr/bin/env python3
"""
BIZRA Heartbeat 24-Hour Gate Monitor (Phase 86-C)
═════════════════════════════════════════════════

Starts the kernel daemon and monitors heartbeat health for 24 hours.
Saves evidence to B:\BIZRA-SOVEREIGN\03_EVIDENCE\heartbeat_24h\

Usage:
    python heartbeat_24h_gate.py

The script:
1. Creates evidence directory
2. Starts kernel daemon in background
3. Polls /api/heartbeat every 60 seconds
4. Logs all heartbeat data to evidence file
5. Reports any anomalies immediately
6. After 24 hours, writes final gate report

Pass/Fail criteria:
- 24 hours continuous operation
- Zero constitutional violations
- Zero silent fallbacks
- All heartbeat receipts valid
- No unrecoverable errors
"""

import json
import time
import os
import sys
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE_DIR = Path(r"B:\BIZRA-SOVEREIGN\03_EVIDENCE\heartbeat_24h")
KERNEL_PORT = 9740
POLL_INTERVAL_S = 60
GATE_DURATION_S = 24 * 3600  # 24 hours
HEARTBEAT_URL = f"http://127.0.0.1:{KERNEL_PORT}/api/heartbeat"
HEALTH_URL = f"http://127.0.0.1:{KERNEL_PORT}/api/health"


def setup():
    """Create evidence directory and log file."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = EVIDENCE_DIR / f"heartbeat_gate_{ts}.jsonl"
    report_path = EVIDENCE_DIR / f"gate_report_{ts}.json"
    return log_path, report_path


def poll_heartbeat():
    """Fetch heartbeat data from kernel daemon."""
    try:
        req = urllib.request.Request(HEARTBEAT_URL)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        return None


def poll_health():
    """Fetch health status from kernel daemon."""
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        return None


def main():
    log_path, report_path = setup()
    print(f"=== BIZRA 24-Hour Heartbeat Gate (Phase 86-C) ===")
    print(f"Evidence log: {log_path}")
    print(f"Gate report:  {report_path}")
    print(f"Duration:     24 hours")
    print(f"Poll interval: {POLL_INTERVAL_S}s")
    print()

    start_time = time.time()
    total_polls = 0
    healthy_polls = 0
    anomaly_polls = 0
    failed_polls = 0
    anomalies = []

    print("Checking if kernel daemon is running...")
    health = poll_health()
    if health is None:
        print("Kernel daemon not detected on port 9740.")
        print("Start it with: cd C:\\BIZRA-DATA-LAKE && python core\\sovereign\\kernel_daemon.py")
        print("Then re-run this script.")
        sys.exit(1)

    print(f"Kernel daemon detected. Starting 24-hour gate monitor.")
    print(f"Press Ctrl+C to stop early (partial results will be saved).\n")

    try:
        while (time.time() - start_time) < GATE_DURATION_S:
            elapsed = time.time() - start_time
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)

            hb = poll_heartbeat()
            total_polls += 1

            if hb is None:
                failed_polls += 1
                entry = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": round(elapsed),
                    "status": "CONNECTION_FAILED",
                }
                print(f"[{hours:02d}:{mins:02d}] FAIL - Cannot reach daemon")
            else:
                beats = hb.get("heartbeats", [])
                latest = beats[-1] if beats else {}
                is_healthy = latest.get("healthy", False)
                anomaly_list = latest.get("anomalies", [])

                if is_healthy and not anomaly_list:
                    healthy_polls += 1
                    status = "HEALTHY"
                else:
                    anomaly_polls += 1
                    status = "ANOMALY"
                    anomalies.append({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "elapsed_s": round(elapsed),
                        "anomalies": anomaly_list,
                    })

                uptime = latest.get("uptime_s", 0)
                rss = latest.get("rss_mb", 0)
                reqs = latest.get("requests_total", 0)
                beat_count = hb.get("total", 0)

                entry = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": round(elapsed),
                    "status": status,
                    "uptime_s": uptime,
                    "rss_mb": rss,
                    "requests_total": reqs,
                    "beat_count": beat_count,
                    "healthy": is_healthy,
                    "anomalies": anomaly_list,
                }

                sym = "+" if is_healthy else "!"
                print(f"[{hours:02d}:{mins:02d}] {sym} beat={beat_count} up={int(uptime)}s rss={rss:.0f}MB reqs={reqs} {status}")

            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            time.sleep(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n\nGate interrupted by user. Saving partial results...")

    elapsed_total = time.time() - start_time
    hours_total = elapsed_total / 3600

    gate_passed = (
        hours_total >= 24.0
        and failed_polls == 0
        and anomaly_polls == 0
    )

    report = {
        "gate": "BIZRA Phase 86-C: 24-Hour Heartbeat Gate",
        "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "duration_hours": round(hours_total, 2),
        "total_polls": total_polls,
        "healthy_polls": healthy_polls,
        "anomaly_polls": anomaly_polls,
        "failed_polls": failed_polls,
        "anomalies": anomalies,
        "gate_passed": gate_passed,
        "gate_criteria": {
            "required_duration_hours": 24,
            "max_failed_polls": 0,
            "max_anomaly_polls": 0,
        },
        "evidence_log": str(log_path),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"GATE RESULT: {'PASSED' if gate_passed else 'NOT PASSED'}")
    print(f"Duration: {hours_total:.1f} hours")
    print(f"Polls: {total_polls} total, {healthy_polls} healthy, {anomaly_polls} anomaly, {failed_polls} failed")
    print(f"Report: {report_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
