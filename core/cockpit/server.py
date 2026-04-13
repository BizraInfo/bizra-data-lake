"""
Glass Cockpit v0.1 — Minimal local observability server.

Reads telemetry, receipts, and model routing — displays as read-only HTML dashboard.
Single-file FastAPI server. No JS framework. No external dependencies beyond FastAPI/uvicorn.

Run: python -m core.cockpit.server
Open: http://127.0.0.1:8420

Standing on Giants:
- Tufte (1983): Information density over decoration
- BIZRA UNC §3.3.1: Decision-grade summaries only
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="BIZRA Glass Cockpit", version="0.1.0")

TELEMETRY_PATH = Path(os.getenv("BIZRA_FATE_TELEMETRY", "/data/bizra/logs/fate-telemetry.jsonl"))
ADVERSARIAL_LEDGER = Path("/data/bizra/logs/mvda-adversarial-ledger.jsonl")
DEV_LEDGER = Path("/data/bizra/logs/mvda-dev-ledger.jsonl")
REPO_ROOT = Path(os.getenv("BIZRA_DATA_LAKE_ROOT", "/data/bizra/repos/bizra-data-lake"))


def _read_jsonl_tail(path: Path, n: int = 20) -> List[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-n:] if l.strip()]
    except (json.JSONDecodeError, OSError):
        return []


def _fate_summary() -> Dict[str, Any]:
    events = _read_jsonl_tail(TELEMETRY_PATH, 100)
    if not events:
        return {"total": 0, "verdicts": {}, "latest": None}

    verdicts = {}
    for e in events:
        v = e.get("verdict", "")
        if v:
            verdicts[v] = verdicts.get(v, 0) + 1

    latest = events[-1] if events else None
    return {
        "total": len(events),
        "verdicts": verdicts,
        "latest": latest,
    }


def _health_summary() -> Dict[str, Any]:
    health = {"gpu": "unknown", "ollama": "unknown", "disk": "unknown", "tests": "unknown"}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            health["gpu"] = r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3) as resp:
            data = json.loads(resp.read())
            health["ollama"] = data.get("version", "unknown")
    except Exception:
        health["ollama"] = "offline"

    try:
        r = subprocess.run(["df", "-h", "/data"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            health["disk"] = r.stdout.strip().split("\n")[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return health


def _model_routing() -> Dict[str, str]:
    try:
        from core.proof_engine.model_routing import routing_table_summary
        return routing_table_summary()
    except ImportError:
        return {}


def _recent_activity(n: int = 15) -> List[dict]:
    events = []
    for path in [TELEMETRY_PATH, ADVERSARIAL_LEDGER, DEV_LEDGER]:
        events.extend(_read_jsonl_tail(path, n))
    events.sort(key=lambda e: str(e.get("timestamp", "")), reverse=True)
    return events[:n]


def _seal_status() -> Dict[str, Any]:
    """Check seal status of the latest loop proof."""
    import glob
    proofs = sorted(glob.glob("/data/bizra/logs/loop-proof-*.json"))
    if not proofs:
        return {"state": "no_proofs", "latest": None}
    latest = Path(proofs[-1])
    try:
        from core.proof_engine.loop_proof_seal import verify_seal
        status = verify_seal(latest)
        return status.to_dict()
    except ImportError:
        return {"state": "seal_module_unavailable", "latest": str(latest)}


@app.get("/api/seal")
def api_seal():
    return _seal_status()


@app.get("/api/fate")
def api_fate():
    return _fate_summary()


@app.get("/api/health")
def api_health():
    return _health_summary()


@app.get("/api/routing")
def api_routing():
    return _model_routing()


@app.get("/api/activity")
def api_activity():
    return _recent_activity()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    fate = _fate_summary()
    health = _health_summary()
    routing = _model_routing()
    activity = _recent_activity(10)
    seal = _seal_status()

    verdict_rows = "".join(
        f"<tr><td>{v}</td><td>{c}</td></tr>"
        for v, c in sorted(fate.get("verdicts", {}).items())
    ) or "<tr><td colspan=2>No telemetry yet</td></tr>"

    routing_rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('model','')}</td><td>{v.get('tier','')}</td></tr>"
        for k, v in sorted(routing.items())
    ) or "<tr><td colspan=3>Routing unavailable</td></tr>"

    activity_rows = ""
    for e in activity:
        ts = e.get("timestamp", "")[:19]
        actor = e.get("actor", e.get("stage", ""))
        verdict = e.get("verdict", "")
        reason = e.get("reason", "")[:80]
        ihsan = e.get("ihsan_score", "")
        css = "blocked" if "BLOCK" in verdict else "pass" if verdict == "PASS" else ""
        activity_rows += f'<tr class="{css}"><td>{ts}</td><td>{actor}</td><td>{verdict}</td><td>{ihsan}</td><td>{reason}</td></tr>'

    latest_ts = fate.get("latest", {}).get("timestamp", "none") if fate.get("latest") else "none"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BIZRA Glass Cockpit</title>
<meta http-equiv="refresh" content="30">
<style>
body {{ font-family: 'SF Mono', 'Cascadia Code', monospace; background: #0a0a0a; color: #e0e0e0; margin: 20px; }}
h1 {{ color: #4fc3f7; font-size: 1.3em; border-bottom: 1px solid #333; padding-bottom: 8px; }}
h2 {{ color: #81c784; font-size: 1.0em; margin-top: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
td, th {{ border: 1px solid #333; padding: 4px 8px; text-align: left; font-size: 0.85em; }}
th {{ background: #1a1a1a; color: #aaa; }}
tr.blocked td {{ background: #2d1111; color: #ef9a9a; }}
tr.pass td {{ background: #112d11; color: #a5d6a7; }}
.metric {{ display: inline-block; background: #1a1a1a; padding: 8px 16px; margin: 4px; border-radius: 4px; }}
.metric .val {{ font-size: 1.4em; color: #4fc3f7; }}
.metric .label {{ font-size: 0.75em; color: #888; }}
footer {{ margin-top: 30px; color: #555; font-size: 0.75em; }}
</style></head><body>
<h1>BIZRA Glass Cockpit v0.1</h1>
<div>
<span class="metric"><span class="val">{fate.get('total', 0)}</span><br><span class="label">FATE Events</span></span>
<span class="metric"><span class="val">{fate.get('verdicts', {}).get('PASS', 0)}</span><br><span class="label">PASS</span></span>
<span class="metric"><span class="val">{fate.get('verdicts', {}).get('BLOCKED_BY_EVIDENCE', 0)}</span><br><span class="label">BLOCKED (Evidence)</span></span>
<span class="metric"><span class="val">{fate.get('verdicts', {}).get('BLOCKED_BY_IHSAN', 0)}</span><br><span class="label">BLOCKED (Ihsan)</span></span>
<span class="metric"><span class="val">{fate.get('verdicts', {}).get('DEGRADED', 0)}</span><br><span class="label">DEGRADED</span></span>
</div>

<h2>Loop Proof Seal</h2>
<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>State</td><td>{'<span style="color:#81c784">CANONICAL</span>' if seal.get('is_canonical') else '<span style="color:#C9A962">' + seal.get('state', 'unknown') + '</span>'}</td></tr>
<tr><td>Manifest</td><td>{seal.get('manifest_hash', 'N/A')}</td></tr>
<tr><td>Proof</td><td>{seal.get('proof_path', 'N/A')}</td></tr>
</table>

<h2>Runtime Health</h2>
<table>
<tr><th>Component</th><th>Status</th></tr>
<tr><td>GPU (VRAM used/free)</td><td>{health.get('gpu', 'unknown')}</td></tr>
<tr><td>Ollama</td><td>{health.get('ollama', 'unknown')}</td></tr>
<tr><td>Disk (/data)</td><td>{health.get('disk', 'unknown')}</td></tr>
<tr><td>Latest telemetry</td><td>{latest_ts}</td></tr>
</table>

<h2>FATE Verdict Distribution</h2>
<table><tr><th>Verdict</th><th>Count</th></tr>{verdict_rows}</table>

<h2>PAT-7 / SAT-5 Model Routing</h2>
<table><tr><th>Role</th><th>Model</th><th>Tier</th></tr>{routing_rows}</table>

<h2>Recent Governed Activity</h2>
<table><tr><th>Time</th><th>Actor</th><th>Verdict</th><th>Ihsan</th><th>Reason</th></tr>{activity_rows}</table>

<footer>Auto-refreshes every 30s. Read-only. Local-only. No external connections.<br>
BIZRA Glass Cockpit v0.1 | {datetime.now(timezone.utc).isoformat()[:19]}Z</footer>
</body></html>"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8420, log_level="warning")
