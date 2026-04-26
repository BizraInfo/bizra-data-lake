"""Stage 4 — Human Review Pack.

Reads Stage 3 outputs and produces:
  - review_workbook.csv   (combined — everything a reviewer acts on)
  - facts_for_review.csv
  - decisions_for_review.csv
  - hypotheses_for_review.csv
  - human_review_brief.md

Every row carries review_status=pending_review initially. promote_to_canon is
ALWAYS blank on emit — only a human can set it.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import Config
from .schema import (
    INITIAL_REVIEW_STATUS,
    REVIEW_WORKBOOK_COLS,
    SOURCE_LANE_NAME,
)
from .util import stage_dir, write_csv, write_manifest


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _collect_provenance_for_cluster(
    members_by_cluster: Dict[str, List[Dict[str, str]]],
    cluster_id: str,
) -> Dict[str, Any]:
    members = members_by_cluster.get(cluster_id, [])
    convs = sorted({m.get("source_conversation_uuid", "") for m in members if m.get("source_conversation_uuid")})
    dates = sorted({m.get("source_created_at", "") for m in members if m.get("source_created_at")})
    return {
        "provenance_conversation_uuids": convs,
        "provenance_earliest": dates[0] if dates else "",
        "provenance_most_recent": dates[-1] if dates else "",
    }


def run_review_pack(
    run_root: Path,
    config: Config,
    run_id: str,
    archive_path: str,
) -> Dict[str, Any]:
    """Execute Stage 4."""

    stage2_dir = stage_dir(run_root, 2, "distillation")
    stage3_dir = stage_dir(run_root, 3, "adjudication")

    fact_candidates_raw = _read_csv(stage2_dir / "fact_candidates.csv")
    decision_candidates_raw = _read_csv(stage2_dir / "decision_candidates.csv")
    canonical_facts = _read_csv(stage3_dir / "canonical_candidate_facts.csv")
    canonical_decisions = _read_csv(stage3_dir / "canonical_candidate_decisions.csv")
    hypotheses = _read_csv(stage3_dir / "hypothesis_candidates.csv")
    obsolete = _read_csv(stage3_dir / "obsolete_conflicted_candidates.csv")
    registry = _read_csv(stage3_dir / "cluster_registry.csv")

    # Map cluster_id -> list of Stage 2 member rows (via supporting_candidate_ids)
    # The registry holds member_candidate_ids as pipe-separated string.
    fact_members_by_cluster: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    decision_members_by_cluster: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    fact_by_id = {r["candidate_id"]: r for r in fact_candidates_raw}
    decision_by_id = {r["candidate_id"]: r for r in decision_candidates_raw}

    for reg in registry:
        ids = [s for s in (reg.get("member_candidate_ids") or "").split("|") if s]
        if reg.get("cluster_type") == "fact":
            fact_members_by_cluster[reg["cluster_id"]] = [fact_by_id[i] for i in ids if i in fact_by_id]
        elif reg.get("cluster_type") == "decision":
            decision_members_by_cluster[reg["cluster_id"]] = [decision_by_id[i] for i in ids if i in decision_by_id]

    workbook_rows: List[Dict[str, Any]] = []
    fact_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    hypothesis_rows: List[Dict[str, Any]] = []

    row_counter = 0

    # Canonical facts → review
    for cf in canonical_facts:
        prov = _collect_provenance_for_cluster(fact_members_by_cluster, cf["cluster_id"])
        row_counter += 1
        base = {
            "row_id": f"R{row_counter:06d}",
            "candidate_type": "fact",
            "cluster_id": cf["cluster_id"],
            "candidate_id": cf["candidate_id"],
            "content": cf["canonical_content"],
            "entity": cf.get("entity", ""),
            "predicate": cf.get("predicate", ""),
            "supporting_count": cf.get("supporting_count", ""),
            "provenance_conversation_uuids": prov["provenance_conversation_uuids"],
            "provenance_earliest": prov["provenance_earliest"],
            "provenance_most_recent": prov["provenance_most_recent"],
            "source_lane": SOURCE_LANE_NAME,
            "review_status": INITIAL_REVIEW_STATUS,
            "reviewer_notes": "",
            "promote_to_canon": "",
        }
        workbook_rows.append(base)
        fact_rows.append(base)

    # Canonical decisions → review
    for cd in canonical_decisions:
        prov = _collect_provenance_for_cluster(decision_members_by_cluster, cd["cluster_id"])
        row_counter += 1
        base = {
            "row_id": f"R{row_counter:06d}",
            "candidate_type": "decision",
            "cluster_id": cd["cluster_id"],
            "candidate_id": cd["candidate_id"],
            "content": cd["canonical_content"],
            "entity": "",
            "predicate": "",
            "supporting_count": cd.get("supporting_count", ""),
            "provenance_conversation_uuids": prov["provenance_conversation_uuids"],
            "provenance_earliest": prov["provenance_earliest"],
            "provenance_most_recent": prov["provenance_most_recent"],
            "source_lane": SOURCE_LANE_NAME,
            "review_status": INITIAL_REVIEW_STATUS,
            "reviewer_notes": "",
            "promote_to_canon": "",
        }
        workbook_rows.append(base)
        decision_rows.append(base)

    # Hypothesis candidates (fact or decision origin) → review
    for h in hypotheses:
        row_counter += 1
        base = {
            "row_id": f"R{row_counter:06d}",
            "candidate_type": "hypothesis",
            "cluster_id": "",
            "candidate_id": h["candidate_id"],
            "content": h.get("content", ""),
            "entity": "",
            "predicate": h.get("candidate_type_origin", ""),
            "supporting_count": h.get("occurrences", ""),
            "provenance_conversation_uuids": [h.get("source_conversation_uuid", "")] if h.get("source_conversation_uuid") else [],
            "provenance_earliest": h.get("source_created_at", ""),
            "provenance_most_recent": h.get("source_created_at", ""),
            "source_lane": SOURCE_LANE_NAME,
            "review_status": INITIAL_REVIEW_STATUS,
            "reviewer_notes": "",
            "promote_to_canon": "",
        }
        workbook_rows.append(base)
        hypothesis_rows.append(base)

    # Obsolete / conflicted → review (reviewer decides retire vs keep)
    for o in obsolete:
        row_counter += 1
        base = {
            "row_id": f"R{row_counter:06d}",
            "candidate_type": "obsolete",
            "cluster_id": "",
            "candidate_id": o["candidate_id"],
            "content": o.get("content", ""),
            "entity": o.get("entity", ""),
            "predicate": o.get("predicate", ""),
            "supporting_count": "",
            "provenance_conversation_uuids": [o.get("source_conversation_uuid", "")] if o.get("source_conversation_uuid") else [],
            "provenance_earliest": o.get("source_created_at", ""),
            "provenance_most_recent": o.get("source_created_at", ""),
            "source_lane": SOURCE_LANE_NAME,
            "review_status": INITIAL_REVIEW_STATUS,
            "reviewer_notes": f"Superseded by {o.get('superseded_by_candidate_id', '')} (delta_days={o.get('delta_days', '')}). Reviewer: keep, retire, or merge.",
            "promote_to_canon": "",
        }
        workbook_rows.append(base)

    # Deterministic order
    workbook_rows.sort(key=lambda r: r["row_id"])

    out_dir = stage_dir(run_root, 4, "review_pack")
    write_csv(out_dir / "review_workbook.csv", workbook_rows, REVIEW_WORKBOOK_COLS)
    write_csv(out_dir / "facts_for_review.csv", fact_rows, REVIEW_WORKBOOK_COLS)
    write_csv(out_dir / "decisions_for_review.csv", decision_rows, REVIEW_WORKBOOK_COLS)
    write_csv(out_dir / "hypotheses_for_review.csv", hypothesis_rows, REVIEW_WORKBOOK_COLS)

    # Human review brief
    brief = _compose_brief(
        run_id=run_id,
        archive_path=archive_path,
        workbook_rows=workbook_rows,
        canonical_fact_count=len(canonical_facts),
        canonical_decision_count=len(canonical_decisions),
        hypothesis_count=len(hypotheses),
        obsolete_count=len(obsolete),
    )
    (out_dir / "human_review_brief.md").write_text(brief, encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "archive_path": archive_path,
        "stage": 4,
        "stage_name": "review_pack",
        "counts": {
            "workbook_rows": len(workbook_rows),
            "facts_for_review": len(fact_rows),
            "decisions_for_review": len(decision_rows),
            "hypotheses_for_review": len(hypothesis_rows),
        },
    }
    write_manifest(out_dir / "run_manifest.json", manifest)
    return manifest


def _compose_brief(
    run_id: str,
    archive_path: str,
    workbook_rows: List[Dict[str, Any]],
    canonical_fact_count: int,
    canonical_decision_count: int,
    hypothesis_count: int,
    obsolete_count: int,
) -> str:
    """Compose the human-readable review brief."""

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    total = len(workbook_rows)
    return f"""# Human Review Brief — Run {run_id}

**Archive:** {archive_path}
**Generated:** {ts}
**Lane:** {SOURCE_LANE_NAME}

## Counts

- Canonical candidate facts: **{canonical_fact_count}**
- Canonical candidate decisions: **{canonical_decision_count}**
- Hypothesis candidates (single-occurrence, not yet corroborated): **{hypothesis_count}**
- Obsolete/conflicted candidates (superseded by newer statements): **{obsolete_count}**
- **Total rows in `review_workbook.csv`:** {total}

## What this is

A deterministic, heuristic-extracted set of candidates from a Claude export. **Nothing in this pack is canon.** The pipeline cannot promote anything to canon. Only a human reading each row and typing `yes` in the `promote_to_canon` column can do that.

## How to review

1. Open `review_workbook.csv` in a spreadsheet tool (Excel, Numbers, LibreOffice, Google Sheets — any).
2. For each row:
   - Read `content`. Is the candidate a real, durable, non-trivial thing worth canonizing?
   - Check `provenance_conversation_uuids` + `provenance_earliest`/`provenance_most_recent` — is this from recent activity or stale?
   - Set `review_status` to one of: `approved`, `rejected`, `needs_followup`.
   - Write free-form `reviewer_notes` if the decision needs context.
   - **Only if approved AND you want it promoted:** set `promote_to_canon` to `yes`. Leave blank otherwise.
3. Save the workbook. Keep the file — a future promotion tool will read it and produce canon entries.

## Review heuristics (suggested, not rules)

| Candidate type | Default disposition | Watch out for |
|---|---|---|
| **fact** with supporting_count ≥ 3 | Likely approve | Personal / one-shot details that shouldn't be canonicalized |
| **fact** with supporting_count = 1 | Marked as hypothesis; verify before approving | The founder-prep M4 / current-vs-future drift pattern also applies here — many "facts" are intended designs, not today's truth |
| **decision** with supporting_count ≥ 2 | Likely approve | One-off experiments the founder didn't follow through on |
| **decision** with supporting_count = 1 | Marked as hypothesis; verify | Same as above |
| **hypothesis** | Needs verification — `approved` here means "worth tracking," not "canon-ready" | Promoting a hypothesis to canon = skipping verification; resist |
| **obsolete** | Usually `rejected` (retire) unless the reviewer believes the newer claim is wrong | Read the `reviewer_notes` field carefully; it names the superseding candidate |

## What the pipeline deliberately did NOT do

- Did NOT use an LLM to judge meaning or quality.
- Did NOT mutate any repo file outside `tools/cognitive_foundry/claude_lane/output/`.
- Did NOT write to MEMORY.md, Node0 runtime files, or any PR-in-flight branches.
- Did NOT set `promote_to_canon=yes` on any row.
- Did NOT assume one Claude conversation = one topic. Conversations can span buckets.

## What the next step should be

1. Operator does the spreadsheet review described above.
2. A separate `promote.py` tool (not yet implemented) reads the annotated workbook and produces structured canon entries ready for explicit inclusion in MEMORY.md or an equivalent persistent store — with a final confirmation gate.

Until both of those happen, the candidates stay candidates.
"""
