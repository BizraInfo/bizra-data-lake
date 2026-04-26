"""Stage 1 — Ingest / Inventory.

Reads the parsed archive and emits:
  - conversation_inventory.csv
  - project_inventory.csv
  - topic_bucket_counts.csv
  - top_signal_sessions.csv
  - run_manifest.json
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .config import Config
from .schema import (
    CONVERSATION_INVENTORY_COLS,
    PROJECT_INVENTORY_COLS,
    TOPIC_BUCKET_COUNT_COLS,
    TOP_SIGNAL_SESSION_COLS,
)
from .util import iter_turns, match_topic_buckets, stage_dir, write_csv, write_manifest

UNCATEGORIZED = "uncategorized"


def run_inventory(
    archive: Dict[str, Any],
    run_root: Path,
    config: Config,
    run_id: str,
    archive_path: str,
) -> Dict[str, Any]:
    """Execute Stage 1. Returns a summary dict suitable for the run manifest."""

    conversations = archive.get("conversations.json") or []
    projects = archive.get("projects.json") or []

    # Project lookup — uuid -> (name, project row)
    project_rows: List[Dict[str, Any]] = []
    project_by_uuid: Dict[str, Dict[str, Any]] = {}
    for p in projects:
        uuid = p.get("uuid", "")
        row = {
            "project_uuid": uuid,
            "name": p.get("name", "") or "",
            "description": p.get("description", "") or "",
            "is_starred": bool(p.get("is_starred", False)),
            "created_at": p.get("created_at", "") or "",
            "updated_at": p.get("updated_at", "") or "",
            "conversation_count": 0,  # filled below
        }
        project_rows.append(row)
        project_by_uuid[uuid] = row

    # Per-bucket aggregates
    bucket_conv_count: Dict[str, int] = defaultdict(int)
    bucket_user_msg_count: Dict[str, int] = defaultdict(int)
    bucket_total_chars: Dict[str, int] = defaultdict(int)

    conversation_rows: List[Dict[str, Any]] = []
    top_signal_rows: List[Dict[str, Any]] = []

    for convo in conversations:
        convo_uuid = convo.get("uuid", "")
        convo_name = convo.get("name", "") or ""
        proj = convo.get("project") or {}
        proj_uuid = proj.get("uuid", "") if isinstance(proj, dict) else ""
        proj_name = (
            project_by_uuid.get(proj_uuid, {}).get("name", "")
            if proj_uuid
            else ""
        )
        if proj_uuid and proj_uuid in project_by_uuid:
            project_by_uuid[proj_uuid]["conversation_count"] += 1

        turns_iter = list(iter_turns(convo))
        turn_count = len(turns_iter)
        user_turn_count = sum(1 for t in turns_iter if t["speaker"] == "human")
        asst_turn_count = sum(1 for t in turns_iter if t["speaker"] == "assistant")
        total_chars = sum(len(t["text"]) for t in turns_iter)
        user_chars = sum(len(t["text"]) for t in turns_iter if t["speaker"] == "human")

        # Topic bucket matching against name + project name + first-user-message sample
        first_user_text = ""
        for t in turns_iter:
            if t["speaker"] == "human":
                first_user_text = t["text"][:2000]
                break
        sample_text = f"{convo_name}\n{proj_name}\n{first_user_text}"
        buckets = match_topic_buckets(sample_text, config.topic_buckets)
        if not buckets:
            buckets = [UNCATEGORIZED]

        for b in buckets:
            bucket_conv_count[b] += 1
            bucket_user_msg_count[b] += user_turn_count
            bucket_total_chars[b] += total_chars

        conversation_rows.append(
            {
                "conversation_uuid": convo_uuid,
                "name": convo_name,
                "project_uuid": proj_uuid,
                "project_name": proj_name,
                "created_at": convo.get("created_at", "") or "",
                "updated_at": convo.get("updated_at", "") or "",
                "turn_count": turn_count,
                "user_turn_count": user_turn_count,
                "assistant_turn_count": asst_turn_count,
                "total_chars": total_chars,
                "user_chars": user_chars,
                "topic_buckets": buckets,
            }
        )

        # Signal score heuristic: turn_count weighted by user-ratio.
        if turn_count >= config.top_signal.min_turns:
            user_ratio = (user_turn_count / turn_count) if turn_count else 0.0
            signal_score = round(turn_count * (0.5 + user_ratio), 4)
            top_signal_rows.append(
                {
                    "rank": 0,  # filled after sort
                    "conversation_uuid": convo_uuid,
                    "name": convo_name,
                    "signal_score": signal_score,
                    "turn_count": turn_count,
                    "user_turn_count": user_turn_count,
                    "total_chars": total_chars,
                    "topic_buckets": buckets,
                }
            )

    # Stable sort, then top-K
    top_signal_rows.sort(
        key=lambda r: (-float(r["signal_score"]), r["conversation_uuid"])
    )
    top_signal_rows = top_signal_rows[: config.top_signal.top_k]
    for i, r in enumerate(top_signal_rows, start=1):
        r["rank"] = i

    # Bucket rows (include uncategorized)
    bucket_rows: List[Dict[str, Any]] = []
    all_bucket_names = sorted(
        set(bucket_conv_count.keys())
        | {b.name for b in config.topic_buckets}
        | {UNCATEGORIZED}
    )
    for name in all_bucket_names:
        bucket_rows.append(
            {
                "bucket_name": name,
                "conversation_count": bucket_conv_count.get(name, 0),
                "user_message_count": bucket_user_msg_count.get(name, 0),
                "total_chars": bucket_total_chars.get(name, 0),
            }
        )

    # Stable sorts for deterministic output
    conversation_rows.sort(key=lambda r: (r["created_at"], r["conversation_uuid"]))
    project_rows.sort(key=lambda r: (r["created_at"], r["project_uuid"]))
    bucket_rows.sort(key=lambda r: r["bucket_name"])

    out_dir = stage_dir(run_root, 1, "inventory")
    write_csv(out_dir / "conversation_inventory.csv", conversation_rows, CONVERSATION_INVENTORY_COLS)
    write_csv(out_dir / "project_inventory.csv", project_rows, PROJECT_INVENTORY_COLS)
    write_csv(out_dir / "topic_bucket_counts.csv", bucket_rows, TOPIC_BUCKET_COUNT_COLS)
    write_csv(out_dir / "top_signal_sessions.csv", top_signal_rows, TOP_SIGNAL_SESSION_COLS)

    manifest = {
        "run_id": run_id,
        "archive_path": archive_path,
        "stage": 1,
        "stage_name": "inventory",
        "counts": {
            "conversations": len(conversation_rows),
            "projects": len(project_rows),
            "buckets": len(bucket_rows),
            "top_signal_sessions": len(top_signal_rows),
        },
        "config": {
            "topic_bucket_names": [b.name for b in config.topic_buckets],
            "top_signal_k": config.top_signal.top_k,
            "top_signal_min_turns": config.top_signal.min_turns,
        },
    }
    write_manifest(out_dir / "run_manifest.json", manifest)
    return manifest
