"""Stage 3 — Adjudication (heuristic clustering, no LLM calls).

Reads Stage 2 CSVs and emits:
  - canonical_candidate_facts.csv
  - canonical_candidate_decisions.csv
  - hypothesis_candidates.csv
  - obsolete_conflicted_candidates.csv
  - cluster_registry.csv

Clusters by (entity, predicate_root) for facts and normalized-text for decisions.
Nothing here gets promoted to canon — promotion is human-only (Stage 4 workbook).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .schema import (
    CANONICAL_CANDIDATE_DECISION_COLS,
    CANONICAL_CANDIDATE_FACT_COLS,
    CLUSTER_REGISTRY_COLS,
    HYPOTHESIS_CANDIDATE_COLS,
    OBSOLETE_CONFLICTED_CANDIDATE_COLS,
    SOURCE_LANE_NAME,
)
from .util import cluster_id, stage_dir, write_csv, write_manifest


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Handle trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _days_between(a: str, b: str) -> int:
    da, db = _parse_iso(a), _parse_iso(b)
    if not da or not db:
        return 0
    return abs((da - db).days)


def run_adjudication(
    run_root: Path,
    config: Config,
    run_id: str,
    archive_path: str,
) -> Dict[str, Any]:
    """Execute Stage 3. Reads Stage 2 outputs from disk."""

    stage2_dir = stage_dir(run_root, 2, "distillation")
    fact_rows = _read_csv(stage2_dir / "fact_candidates.csv")
    decision_rows = _read_csv(stage2_dir / "decision_candidates.csv")

    # ---- Fact clustering by (entity, predicate[:80]) ----
    fact_clusters: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in fact_rows:
        entity = (r.get("entity") or "").strip()
        pred = (r.get("predicate") or "").strip()[:80]
        if not entity:
            # Facts without an extractable entity are singletons keyed by candidate_id.
            fact_clusters[("__unkeyed__", r["candidate_id"])].append(r)
        else:
            fact_clusters[(entity, pred)].append(r)

    canonical_fact_rows: List[Dict[str, Any]] = []
    cluster_registry_rows: List[Dict[str, Any]] = []
    hypothesis_rows: List[Dict[str, Any]] = []
    obsolete_rows: List[Dict[str, Any]] = []

    for (entity, pred), members in fact_clusters.items():
        members_sorted = sorted(members, key=lambda m: m["source_created_at"] or "")
        earliest = members_sorted[0]["source_created_at"] or ""
        most_recent = members_sorted[-1]["source_created_at"] or ""
        canonical = members_sorted[-1]  # most recent as canonical candidate
        cid = cluster_id("fact", f"{entity}|{pred}")
        canonical_fact_rows.append(
            {
                "cluster_id": cid,
                "candidate_id": canonical["candidate_id"],
                "canonical_content": canonical["content"],
                "entity": entity if entity != "__unkeyed__" else "",
                "predicate": pred,
                "supporting_count": len(members),
                "supporting_candidate_ids": [m["candidate_id"] for m in members],
                "most_recent_source_created_at": most_recent,
                "earliest_source_created_at": earliest,
                "source_lane": SOURCE_LANE_NAME,
            }
        )
        cluster_registry_rows.append(
            {
                "cluster_id": cid,
                "cluster_type": "fact",
                "member_count": len(members),
                "canonical_candidate_id": canonical["candidate_id"],
                "entity": entity if entity != "__unkeyed__" else "",
                "predicate": pred,
                "member_candidate_ids": [m["candidate_id"] for m in members],
                "earliest_source_created_at": earliest,
                "most_recent_source_created_at": most_recent,
            }
        )

        # Hypothesis flag: single-occurrence facts
        if len(members) <= config.adjudication.hypothesis_max_occurrences:
            only = members_sorted[0]
            hypothesis_rows.append(
                {
                    "candidate_id": only["candidate_id"],
                    "candidate_type_origin": "fact",
                    "content": only["content"],
                    "reason_flagged": f"occurrences={len(members)}<=threshold",
                    "occurrences": len(members),
                    "source_lane": SOURCE_LANE_NAME,
                    "source_conversation_uuid": only.get("source_conversation_uuid", ""),
                    "source_message_uuid": only.get("source_message_uuid", ""),
                    "source_created_at": only.get("source_created_at", ""),
                }
            )

        # Obsolete flag: any non-canonical (older) member whose normalized_text
        # differs from canonical AND delta_days >= threshold gets flagged.
        if len(members) >= 2:
            canon_norm = (canonical.get("normalized_text") or "").strip()
            for m in members_sorted[:-1]:
                m_norm = (m.get("normalized_text") or "").strip()
                if m_norm and canon_norm and m_norm != canon_norm:
                    dd = _days_between(
                        m.get("source_created_at", ""),
                        canonical.get("source_created_at", ""),
                    )
                    if dd >= config.adjudication.obsolete_newer_preferred_days:
                        obsolete_rows.append(
                            {
                                "candidate_id": m["candidate_id"],
                                "candidate_type_origin": "fact",
                                "content": m["content"],
                                "entity": entity if entity != "__unkeyed__" else "",
                                "predicate": pred,
                                "superseded_by_candidate_id": canonical["candidate_id"],
                                "superseded_by_content": canonical["content"],
                                "delta_days": dd,
                                "source_lane": SOURCE_LANE_NAME,
                                "source_conversation_uuid": m.get("source_conversation_uuid", ""),
                                "source_message_uuid": m.get("source_message_uuid", ""),
                                "source_created_at": m.get("source_created_at", ""),
                            }
                        )

    # ---- Decision clustering by normalized_text ----
    decision_clusters: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in decision_rows:
        key = (r.get("normalized_text") or "").strip()
        if not key:
            key = r["candidate_id"]
        decision_clusters[key].append(r)

    canonical_decision_rows: List[Dict[str, Any]] = []
    for norm, members in decision_clusters.items():
        members_sorted = sorted(members, key=lambda m: m["source_created_at"] or "")
        earliest = members_sorted[0]["source_created_at"] or ""
        most_recent = members_sorted[-1]["source_created_at"] or ""
        canonical = members_sorted[-1]
        cid = cluster_id("decision", norm)
        canonical_decision_rows.append(
            {
                "cluster_id": cid,
                "candidate_id": canonical["candidate_id"],
                "canonical_content": canonical["content"],
                "supporting_count": len(members),
                "supporting_candidate_ids": [m["candidate_id"] for m in members],
                "most_recent_source_created_at": most_recent,
                "earliest_source_created_at": earliest,
                "source_lane": SOURCE_LANE_NAME,
            }
        )
        cluster_registry_rows.append(
            {
                "cluster_id": cid,
                "cluster_type": "decision",
                "member_count": len(members),
                "canonical_candidate_id": canonical["candidate_id"],
                "entity": "",
                "predicate": norm[:80],
                "member_candidate_ids": [m["candidate_id"] for m in members],
                "earliest_source_created_at": earliest,
                "most_recent_source_created_at": most_recent,
            }
        )
        # Hypothesis flag for single-occurrence decisions
        if len(members) <= config.adjudication.hypothesis_max_occurrences:
            only = members_sorted[0]
            hypothesis_rows.append(
                {
                    "candidate_id": only["candidate_id"],
                    "candidate_type_origin": "decision",
                    "content": only["content"],
                    "reason_flagged": f"occurrences={len(members)}<=threshold",
                    "occurrences": len(members),
                    "source_lane": SOURCE_LANE_NAME,
                    "source_conversation_uuid": only.get("source_conversation_uuid", ""),
                    "source_message_uuid": only.get("source_message_uuid", ""),
                    "source_created_at": only.get("source_created_at", ""),
                }
            )

    # Deterministic output order
    canonical_fact_rows.sort(key=lambda r: (r["cluster_id"]))
    canonical_decision_rows.sort(key=lambda r: (r["cluster_id"]))
    cluster_registry_rows.sort(key=lambda r: (r["cluster_type"], r["cluster_id"]))
    hypothesis_rows.sort(key=lambda r: (r["candidate_type_origin"], r["candidate_id"]))
    obsolete_rows.sort(key=lambda r: (r["entity"], r["predicate"], r["candidate_id"]))

    out_dir = stage_dir(run_root, 3, "adjudication")
    write_csv(out_dir / "canonical_candidate_facts.csv", canonical_fact_rows, CANONICAL_CANDIDATE_FACT_COLS)
    write_csv(
        out_dir / "canonical_candidate_decisions.csv",
        canonical_decision_rows,
        CANONICAL_CANDIDATE_DECISION_COLS,
    )
    write_csv(out_dir / "hypothesis_candidates.csv", hypothesis_rows, HYPOTHESIS_CANDIDATE_COLS)
    write_csv(
        out_dir / "obsolete_conflicted_candidates.csv",
        obsolete_rows,
        OBSOLETE_CONFLICTED_CANDIDATE_COLS,
    )
    write_csv(out_dir / "cluster_registry.csv", cluster_registry_rows, CLUSTER_REGISTRY_COLS)

    manifest = {
        "run_id": run_id,
        "archive_path": archive_path,
        "stage": 3,
        "stage_name": "adjudication",
        "counts": {
            "canonical_candidate_facts": len(canonical_fact_rows),
            "canonical_candidate_decisions": len(canonical_decision_rows),
            "hypothesis_candidates": len(hypothesis_rows),
            "obsolete_conflicted_candidates": len(obsolete_rows),
            "cluster_registry": len(cluster_registry_rows),
        },
        "config": {
            "hypothesis_max_occurrences": config.adjudication.hypothesis_max_occurrences,
            "obsolete_newer_preferred_days": config.adjudication.obsolete_newer_preferred_days,
        },
    }
    write_manifest(out_dir / "run_manifest.json", manifest)
    return manifest
