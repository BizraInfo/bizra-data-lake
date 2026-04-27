"""Dema dream — read-only memory consolidation pass.

Five phases (Orient → Gather → Consolidate → Prune → Prepare). The dream is
read-only by default: it scans the local daily log for the day, derives
candidate memory notes, and writes them under
sovereign_state/dema/dreams/<run_id>/. **No automatic promotion to long-term
memory.** Promotion requires explicit operator approval (out of scope for
v0.1).

Hard cap: --max-seconds budget; if exceeded, the dream finishes the current
phase and exits with a "background_suggestion" hint instead of blocking.

Usage:
    python scripts/dema/dema_dream.py --read-only --max-seconds 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema import (  # noqa: E402
    DailyLog,
    DailyLogEntry,
    DemaReceipt,
    ReceiptWriter,
)

DEFAULT_ROOT = REPO_ROOT / "sovereign_state" / "dema"
PHASES = ("Orient", "Gather", "Consolidate", "Prune", "Prepare")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("dream_%Y%m%dT%H%M%SZ")


def dream(
    root: Path,
    *,
    read_only: bool,
    max_seconds: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    run_id = _run_id()
    out_dir = root / "dreams" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    today = DailyLog(root).read_today()
    elapsed = lambda: time.monotonic() - started_at  # noqa: E731
    budget_hit = False
    completed_phases: list[str] = []
    candidate_notes: list[dict[str, Any]] = []

    for phase in PHASES:
        if elapsed() > max_seconds:
            budget_hit = True
            break

        if phase == "Orient":
            # Just record what we are about to consolidate.
            candidate_notes.append(
                {
                    "phase": phase,
                    "kind": "orientation",
                    "summary": (f"reviewing {len(today)} log entries from today"),
                    "ts": _utc_now(),
                }
            )
        elif phase == "Gather":
            # Group log entries by kind without copying private content.
            by_kind: dict[str, int] = {}
            for entry in today:
                by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
            candidate_notes.append(
                {
                    "phase": phase,
                    "kind": "kinds_by_count",
                    "summary": "counts only — no raw content",
                    "by_kind": by_kind,
                    "ts": _utc_now(),
                }
            )
        elif phase == "Consolidate":
            # Pick the most recent receipt per kind as a recall hook.
            seen: set[str] = set()
            recalls: list[dict[str, str]] = []
            for entry in reversed(today):
                if entry.kind in seen:
                    continue
                seen.add(entry.kind)
                if entry.receipt_id:
                    recalls.append({"kind": entry.kind, "receipt_id": entry.receipt_id})
                if len(recalls) >= 5:
                    break
            candidate_notes.append(
                {
                    "phase": phase,
                    "kind": "recall_hooks",
                    "recalls": recalls,
                    "ts": _utc_now(),
                }
            )
        elif phase == "Prune":
            candidate_notes.append(
                {
                    "phase": phase,
                    "kind": "prune_proposals",
                    "summary": "no automatic prune in v0.1; operator approval required",
                    "ts": _utc_now(),
                }
            )
        elif phase == "Prepare":
            candidate_notes.append(
                {
                    "phase": phase,
                    "kind": "tomorrow_prep",
                    "summary": "tomorrow's first admissible action: review candidate notes",
                    "ts": _utc_now(),
                }
            )

        completed_phases.append(phase)

    notes_path = out_dir / "candidate_notes.jsonl"
    with notes_path.open("w", encoding="utf-8") as fh:
        for note in candidate_notes:
            fh.write(json.dumps(note, sort_keys=True) + "\n")

    summary_path = out_dir / "summary.md"
    summary_md = (
        f"# Dema Dream {run_id}\n\n"
        f"- read_only: {read_only}\n"
        f"- max_seconds: {max_seconds}\n"
        f"- elapsed_seconds: {elapsed():.3f}\n"
        f"- budget_hit: {budget_hit}\n"
        f"- completed_phases: {', '.join(completed_phases)}\n"
        f"- candidate_notes_count: {len(candidate_notes)}\n"
        "- promoted_to_long_term: false (v0.1 requires explicit approval)\n"
    )
    summary_path.write_text(summary_md, encoding="utf-8")

    receipt = DemaReceipt(
        action="dema.dream.read_only" if read_only else "dema.dream.write",
        truth_label="MEASURED",
        touched_paths=[
            str(notes_path),
            str(summary_path),
        ],
        not_touched_paths=[
            "network",
            "desktop",
            "MEMORY.md",
            "docs/canon/",
            "long_term_memory_promotion",
        ],
        approval_required=not read_only,
        approval_status="n/a" if read_only else "pending",
        payload={
            "run_id": run_id,
            "phases_completed": completed_phases,
            "budget_hit": budget_hit,
            "candidate_count": len(candidate_notes),
        },
    )
    rid, receipt_path = ReceiptWriter(root).write(receipt)

    DailyLog(root).append(
        DailyLogEntry(
            timestamp=receipt.timestamp,
            kind="dream",
            summary=(
                f"dream {run_id}: {len(completed_phases)}/{len(PHASES)} phases, "
                f"{len(candidate_notes)} candidates"
            ),
            receipt_id=rid,
            metadata={
                "budget_hit": budget_hit,
                "elapsed_seconds": elapsed(),
            },
        )
    )

    return {
        "ok": True,
        "run_id": run_id,
        "elapsed_seconds": elapsed(),
        "budget_hit": budget_hit,
        "completed_phases": completed_phases,
        "candidate_notes_count": len(candidate_notes),
        "candidate_notes_path": str(notes_path),
        "summary_path": str(summary_path),
        "receipt_id": rid,
        "receipt_path": str(receipt_path),
        "background_suggestion": budget_hit,
        "promoted_to_long_term": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--read-only",
        action="store_true",
        default=True,
        help="Read-only dream (default). Future: --write to allow promotion.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="Hard time budget (default 15s).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local Dema state root (default: {DEFAULT_ROOT}).",
    )
    args = parser.parse_args(argv)

    out = dream(args.root, read_only=args.read_only, max_seconds=args.max_seconds)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
