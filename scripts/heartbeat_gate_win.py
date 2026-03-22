#!/usr/bin/env python3
"""Lightweight 24h heartbeat gate — polls kernel API, logs to C: drive."""
import json, time, sys, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE_DIR = Path(r"C:\BIZRA-DATA-LAKE\artifacts\heartbeat_24h_gate")
KERNEL_PORT = 9740
POLL_S = 60
GATE_S = 24 * 3600
URL = f"http://127.0.0.1:{KERNEL_PORT}/api/heartbeat"

def poll():
    try:
        with urllib.request.urlopen(URL, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return None

def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log = EVIDENCE_DIR / f"gate_{ts}.jsonl"
    start = time.time()
    total = healthy = degraded = failed = 0
    anomalies = []
    print(f"24h gate started. Log: {log}")
    try:
        while time.time() - start < GATE_S:
            d = poll()
            total += 1
            entry = {"ts": datetime.now(timezone.utc).isoformat(), "poll": total}
            if d is None:
                failed += 1
                entry["status"] = "unreachable"
                print(f"[{total}] UNREACHABLE")
            elif d.get("health") == "healthy":
                healthy += 1
                entry["status"] = "healthy"
                entry["beat"] = d.get("latest", {}).get("beat", 0)
                entry["rss"] = d.get("latest", {}).get("memory_rss_mb", 0)
                entry["backends"] = d.get("latest", {}).get("backends_alive", 0)
                if total % 10 == 1:
                    print(f"[{total}] HEALTHY beat={entry['beat']} rss={entry['rss']}MB")
            else:
                degraded += 1
                entry["status"] = "degraded"
                entry["anomalies"] = d.get("anomalies", [])
                anomalies.append(entry)
                print(f"[{total}] DEGRADED: {d.get('anomalies', [])}")
            with log.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        print("\nStopped early.")
    elapsed = time.time() - start
    hours = elapsed / 3600
    report = {
        "gate": "24h_heartbeat", "started": ts,
        "elapsed_hours": round(hours, 2), "total_polls": total,
        "healthy": healthy, "degraded": degraded, "failed": failed,
        "anomaly_count": len(anomalies),
        "pass": degraded == 0 and failed == 0 and hours >= 23.5,
    }
    rpt = EVIDENCE_DIR / f"report_{ts}.json"
    rpt.write_text(json.dumps(report, indent=2))
    print(f"\n{'PASS' if report['pass'] else 'FAIL'}: {healthy}/{total} healthy in {hours:.1f}h")
    print(f"Report: {rpt}")

if __name__ == "__main__":
    main()
