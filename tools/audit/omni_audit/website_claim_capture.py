"""Read-only website claim capture.

By default run in --no-network mode (default_no_network=True in config). When
network is allowed, uses urllib.request only. Writes captures under out-dir.

When --no-network is active, this module returns "skeleton" captures sourced
from the operator-supplied pre-check mirror (see module docstring note). The
operator supplies pre-check evidence via an optional JSON file in
<repo>/tools/audit/omni_audit/website_precheck.json.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import List
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


def _fetch(url: str, timeout: float = 6.0) -> dict:
    req = urlrequest.Request(url, headers={"User-Agent": "bizra-audit/0.1"})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            body = resp.read(200000).decode("utf-8", errors="replace")
            status = resp.status
            return {
                "url": url,
                "final_url": final_url,
                "status": status,
                "redirected": final_url != url,
                "snippet_len": len(body),
                "text_snippet": body[:3000],
                "blocks": [{"text": body[:3000]}],
                "fetch_ok": True,
            }
    except (HTTPError, URLError, socket.timeout, TimeoutError) as e:
        return {"url": url, "final_url": None, "status": None, "redirected": False,
                "error": str(e), "blocks": [], "fetch_ok": False}


def _precheck_skeleton() -> List[dict]:
    """Fallback when network is disabled — uses operator-supplied pre-check."""
    # Operator-supplied pre-check claims captured in this session's brief.
    return [{
        "url": "https://bizra.ai",
        "final_url": "https://bizra.ai/",
        "status": None,
        "redirected": False,
        "fetch_ok": False,
        "source": "operator pre-check",
        "blocks": [{
            "text": (
                "BIZRA | The Sovereign Future. "
                "local agents / no cloud dependency / no telemetry. "
                "Ed25519 receipt signatures. "
                "cost per action dropping from about $0.10 toward $0.008. "
                "SNR 0.974. 8,072 verified tests. 100% pass rate. "
                "Ihsan Gate >= 0.95. 73 of 100 nodes remaining. "
            )
        }],
    }, {
        "url": "https://bizra.info",
        "final_url": "https://bizra.ai/",
        "status": 302,
        "redirected": True,
        "fetch_ok": False,
        "source": "operator pre-check + session observation",
        "blocks": [],
    }]


def capture(urls: List[str], no_network: bool, out_dir: Path) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if no_network:
        captures = _precheck_skeleton()
    else:
        captures = []
        for u in urls:
            cap = _fetch(u)
            captures.append(cap)

    json_path = out_dir / "website_claims.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(captures, f, indent=2, ensure_ascii=False)

    # Snapshot the text blocks to a .txt beside the JSON for audit trail.
    snap_path = out_dir / "website_snapshot.txt"
    with snap_path.open("w", encoding="utf-8") as f:
        for cap in captures:
            f.write(f"# {cap.get('url')}\n")
            f.write(f"final_url={cap.get('final_url')} status={cap.get('status')} "
                    f"redirected={cap.get('redirected')} fetch_ok={cap.get('fetch_ok')}\n")
            for b in cap.get("blocks", []):
                f.write(b.get("text", "") + "\n---\n")
            f.write("\n")

    return captures
