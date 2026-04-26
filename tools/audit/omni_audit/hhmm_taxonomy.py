"""Build a 4-level hidden-state taxonomy from findings.

Level 0: Domain (SECURITY, ARCHITECTURE, etc.)
Level 1: Subsystem
Level 2: Failure mode / opportunity
Level 3: Evidence-backed action
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .schemas import Finding


def build(findings: List[Finding]) -> dict:
    tree: dict = {}
    for f in findings:
        lvl0 = tree.setdefault(f.domain, {})
        lvl1 = lvl0.setdefault(f.subsystem or "general", {})
        lvl2_key = f.summary[:80]
        lvl2 = lvl1.setdefault(lvl2_key, {"evidence": [], "actions": []})
        lvl2["evidence"].extend(f.evidence_paths)
        if f.next_action and f.next_action not in lvl2["actions"]:
            lvl2["actions"].append(f.next_action)
    # Counts by domain for summary.
    counts = {k: sum(len(v.values()) for v in subtree.values())
              for k, subtree in tree.items()}
    return {"tree": tree, "counts_by_domain": counts}


def write_outputs(tree: dict, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hhmm_taxonomy.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    return {"hhmm_taxonomy_json": str(path)}
