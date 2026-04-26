"""Classify findings into SIGNAL / NOISE / WATCHLIST buckets with scores."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .schemas import Finding


def classify(findings: List[Finding]) -> dict:
    signal, noise, watchlist = [], [], []
    for f in findings:
        if f.signal_score >= 0.65 and f.actionable:
            signal.append(f)
        elif f.signal_score <= 0.35:
            noise.append(f)
        else:
            watchlist.append(f)
    return {
        "signal": [asdict(f) for f in signal],
        "watchlist": [asdict(f) for f in watchlist],
        "noise": [asdict(f) for f in noise],
        "counts": {"signal": len(signal), "watchlist": len(watchlist), "noise": len(noise)},
    }


def write_outputs(result: dict, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "snr_findings.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return {"snr_findings_json": str(path)}
