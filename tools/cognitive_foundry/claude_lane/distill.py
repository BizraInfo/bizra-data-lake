"""Stage 2 — Distillation (heuristic, no LLM calls).

Emits:
  - fact_candidates.csv
  - decision_candidates.csv
  - contradiction_candidates.csv
  - reasoning_exemplars.csv

All extraction is regex + keyword based. Produces candidates, not truths.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import Config
from .schema import (
    CONTRADICTION_CANDIDATE_COLS,
    DECISION_CANDIDATE_COLS,
    FACT_CANDIDATE_COLS,
    REASONING_EXEMPLAR_COLS,
    SOURCE_LANE_NAME,
)
from .util import candidate_id, iter_turns, normalize_text, stage_dir, write_csv, write_manifest


def _extract_entity_predicate(match_groups: Tuple[str, ...], pattern: str) -> Tuple[str, str]:
    """Best-effort entity/predicate extraction from a regex match.

    Heuristic and imperfect; adjudication treats these as hints, not ground truth.
    """

    g = [gg for gg in match_groups if gg is not None]
    if not g:
        return ("", "")
    # Pattern-specific fingerprints (kept loose on purpose — this is candidate
    # generation, not canonicalization).
    if pattern.startswith(r"\bI\s+(am|'m)"):
        return ("self", normalize_text(g[-1]))
    if pattern.startswith(r"\bMy\s+"):
        return (f"self.{g[0]}", normalize_text(g[-1]))
    # Named-entity is-a: match by the stable signature substring so both the
    # original ``\b([A-Z]...`` pattern AND the 2026-04-24 pronoun-excluded
    # lookahead-prefixed variant route here and keep entity extraction working.
    # Fixes the entity-dispatch regression flagged after run v2 (2026-04-24).
    if r"is\s+(a|an|the)" in pattern:
        # "<Entity> is (a|an|the) <Predicate>"
        entity = g[0]
        # group 1 is optional determiner; -1 is predicate
        return (normalize_text(entity), normalize_text(g[-1]))
    if pattern.startswith(r"\bI\s+(live|work|am based)"):
        return ("self.location", normalize_text(g[-1]))
    return ("", normalize_text(g[-1]))


def _match_all(patterns: List[str], text: str) -> List[Tuple[str, re.Match]]:
    out: List[Tuple[str, re.Match]] = []
    for p in patterns:
        for m in re.finditer(p, text):
            out.append((p, m))
    return out


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def run_distillation(
    archive: Dict[str, Any],
    run_root: Path,
    config: Config,
    run_id: str,
    archive_path: str,
) -> Dict[str, Any]:
    """Execute Stage 2. Returns a summary dict for the run manifest."""

    conversations = archive.get("conversations.json") or []
    fact_patterns = _compile(config.fact_patterns)
    decision_patterns = _compile(config.decision_patterns)

    fact_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    exemplar_rows: List[Dict[str, Any]] = []

    # Contradiction tracking: (entity, predicate_root) -> list of fact candidates
    # Keys normalize entity + a "root" slice of the predicate (first 80 chars
    # of normalized predicate) so minor wording differences still collide.
    fact_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    reasoning_markers_lower = [m.lower() for m in config.reasoning_markers]

    for convo in conversations:
        convo_uuid = convo.get("uuid", "")
        convo_name = convo.get("name", "") or ""
        for turn in iter_turns(convo):
            text = turn["text"] or ""
            if not text.strip():
                continue

            is_user = turn["speaker"] == "human"
            # Facts / Decisions: only from user messages unless config says otherwise.
            if is_user or config.include_assistant_text_in_distillation:
                # Fact candidates
                for pat in fact_patterns:
                    for m in pat.finditer(text):
                        content = m.group(0).strip()
                        if not (
                            config.distillation.fact_min_chars
                            <= len(content)
                            <= config.distillation.fact_max_chars
                        ):
                            continue
                        entity, predicate = _extract_entity_predicate(m.groups(), pat.pattern)
                        norm = normalize_text(content)
                        cid = candidate_id("fact", norm, turn["message_uuid"])
                        row = {
                            "candidate_id": cid,
                            "candidate_type": "fact",
                            "content": content,
                            "entity": entity,
                            "predicate": predicate,
                            "normalized_text": norm,
                            "pattern_matched": pat.pattern,
                            "source_lane": SOURCE_LANE_NAME,
                            "source_conversation_uuid": convo_uuid,
                            "source_conversation_name": convo_name,
                            "source_message_uuid": turn["message_uuid"],
                            "source_created_at": turn["created_at"],
                        }
                        fact_rows.append(row)
                        if entity:
                            key = (entity, predicate[:80])
                            fact_by_key[key].append(row)

                # Decision candidates
                for pat in decision_patterns:
                    for m in pat.finditer(text):
                        content = m.group(0).strip()
                        if not (
                            config.distillation.decision_min_chars
                            <= len(content)
                            <= config.distillation.decision_max_chars
                        ):
                            continue
                        norm = normalize_text(content)
                        cid = candidate_id("decision", norm, turn["message_uuid"])
                        decision_rows.append(
                            {
                                "candidate_id": cid,
                                "candidate_type": "decision",
                                "content": content,
                                "normalized_text": norm,
                                "pattern_matched": pat.pattern,
                                "source_lane": SOURCE_LANE_NAME,
                                "source_conversation_uuid": convo_uuid,
                                "source_conversation_name": convo_name,
                                "source_message_uuid": turn["message_uuid"],
                                "source_created_at": turn["created_at"],
                            }
                        )

            # Reasoning exemplars: long turns with multi-step markers.
            # Human-only gate (GUARDRAIL for signal hygiene, added 2026-04-24):
            # when config.reasoning_exemplars_human_only is True, skip
            # assistant turns entirely so the review surface contains the
            # operator's own reasoning, not Claude's replies.
            if config.reasoning_exemplars_human_only and turn["speaker"] != "human":
                continue
            if len(text) >= config.distillation.exemplar_min_chars:
                lower = text.lower()
                markers_present = [m for m in reasoning_markers_lower if m in lower]
                numbered_list_hint = bool(re.search(r"(?m)^\s*\d+\.\s+", text))
                if (
                    (markers_present or numbered_list_hint)
                    or not config.distillation.exemplar_requires_markers
                ):
                    norm = normalize_text(text[:600])
                    cid = candidate_id("reasoning_exemplar", norm, turn["message_uuid"])
                    exemplar_rows.append(
                        {
                            "candidate_id": cid,
                            "candidate_type": "reasoning_exemplar",
                            "content_excerpt": text[:600],
                            "content_char_count": len(text),
                            "speaker": turn["speaker"],
                            "marker_keywords_present": markers_present
                            + (["__numbered_list__"] if numbered_list_hint else []),
                            "source_lane": SOURCE_LANE_NAME,
                            "source_conversation_uuid": convo_uuid,
                            "source_conversation_name": convo_name,
                            "source_message_uuid": turn["message_uuid"],
                            "source_created_at": turn["created_at"],
                        }
                    )

    # Contradiction candidates: (entity, predicate_root) with >= 2 distinct
    # normalized values.
    contradiction_rows: List[Dict[str, Any]] = []
    for (entity, pred_root), rows in fact_by_key.items():
        if not entity:
            continue
        distinct_values: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            distinct_values[r["normalized_text"]].append(r)
        if len(distinct_values) < 2:
            continue
        # Order values by most-recent source_created_at; emit contradiction pairs.
        sorted_vals = sorted(
            distinct_values.items(),
            key=lambda kv: max(x["source_created_at"] for x in kv[1]),
        )
        # Emit a single contradiction row between oldest and newest for brevity.
        oldest_norm, oldest_rows = sorted_vals[0]
        newest_norm, newest_rows = sorted_vals[-1]
        oldest = min(oldest_rows, key=lambda x: x["source_created_at"])
        newest = max(newest_rows, key=lambda x: x["source_created_at"])
        cid_key = f"{entity}|{pred_root}|{oldest_norm}|{newest_norm}"
        cid = candidate_id("contradiction", cid_key, "")
        contradiction_rows.append(
            {
                "candidate_id": cid,
                "candidate_type": "contradiction",
                "entity": entity,
                "predicate": pred_root,
                "value_a": oldest["content"],
                "value_a_source_message_uuid": oldest["source_message_uuid"],
                "value_a_created_at": oldest["source_created_at"],
                "value_b": newest["content"],
                "value_b_source_message_uuid": newest["source_message_uuid"],
                "value_b_created_at": newest["source_created_at"],
                "source_lane": SOURCE_LANE_NAME,
                "source_conversation_uuids": sorted(
                    {r["source_conversation_uuid"] for r in oldest_rows + newest_rows}
                ),
            }
        )

    # Deterministic output order.
    fact_rows.sort(key=lambda r: (r["source_created_at"], r["candidate_id"]))
    decision_rows.sort(key=lambda r: (r["source_created_at"], r["candidate_id"]))
    exemplar_rows.sort(key=lambda r: (r["source_created_at"], r["candidate_id"]))
    contradiction_rows.sort(key=lambda r: (r["entity"], r["predicate"], r["candidate_id"]))

    out_dir = stage_dir(run_root, 2, "distillation")
    write_csv(out_dir / "fact_candidates.csv", fact_rows, FACT_CANDIDATE_COLS)
    write_csv(out_dir / "decision_candidates.csv", decision_rows, DECISION_CANDIDATE_COLS)
    write_csv(out_dir / "contradiction_candidates.csv", contradiction_rows, CONTRADICTION_CANDIDATE_COLS)
    write_csv(out_dir / "reasoning_exemplars.csv", exemplar_rows, REASONING_EXEMPLAR_COLS)

    manifest = {
        "run_id": run_id,
        "archive_path": archive_path,
        "stage": 2,
        "stage_name": "distillation",
        "counts": {
            "fact_candidates": len(fact_rows),
            "decision_candidates": len(decision_rows),
            "contradiction_candidates": len(contradiction_rows),
            "reasoning_exemplars": len(exemplar_rows),
        },
        "config": {
            "fact_patterns": config.fact_patterns,
            "decision_patterns": config.decision_patterns,
            "reasoning_markers": config.reasoning_markers,
            "include_assistant_text_in_distillation": config.include_assistant_text_in_distillation,
        },
    }
    write_manifest(out_dir / "run_manifest.json", manifest)
    return manifest
