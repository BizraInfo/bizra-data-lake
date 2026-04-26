"""Promote human-reviewed candidates to a structured canon pack.

Reads an annotated `review_workbook.csv` (output of Stage 4) and produces a
signed, structured canon-pack artifact for ONLY the rows a human operator has
explicitly marked `review_status=approved` AND `promote_to_canon=yes`.

STRICT RULES (non-negotiable, enforced in code):

1. NEVER writes to MEMORY.md or any BIZRA runtime file.
2. NEVER auto-sets `promote_to_canon`. The tool refuses to operate on a row
   unless the human wrote "yes" in that cell.
3. NEVER mutates the input workbook.
4. If workbook contains contradictions (e.g., promote_to_canon=yes but
   review_status is not "approved"), exits with a validation error.
5. Produces deterministic output: same annotated workbook → same canon pack
   (content-addressed by blake2b hash).
6. Produces a *hash-signed* pack (content hash). This is NOT Ed25519 or
   RSA signing; it is tamper-evidence via content digest. A future runtime
   tool can upgrade to key-based signing.
7. Canon packs are staged artifacts. Even after this tool runs, a canon pack
   sits on disk awaiting a SEPARATE tool (not in this pilot) to ingest it
   into BIZRA's actual canonical stores. The pack alone is not canon.

Usage (from repo root):

    python tools/cognitive_foundry/claude_lane/promote.py \\
        --workbook tools/cognitive_foundry/claude_lane/output/<run_id>/04_review_pack/review_workbook.csv \\
        [--output-dir tools/cognitive_foundry/claude_lane/canon_packs] \\
        [--dry-run]

Stdlib-only. Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TOOL_NAME = "cognitive_foundry.claude_lane.promote"
TOOL_VERSION = "0.2.0"
# v0.2.0 (2026-04-24) — Split hash identity:
#   - content_hash_blake2b_32  : deterministic over stable entry fields ONLY
#                                (does NOT include promoted_at or promoter).
#                                Same workbook + same reviewed content =>
#                                same content_hash across any number of reruns.
#   - issuance_hash_blake2b_32 : identifies ONE promotion event
#                                (content_hash | promoted_at | promoter |
#                                 workbook_sha256). Two reruns produce
#                                identical content_hash but different
#                                issuance_hash.
#   - canon_entry_id (v2)      : no longer mixes in promoted_at.
#                                Key = ("canon_entry|v2|" | source_candidate_id
#                                       | content).
# v0.1.0 packs remain valid but use a different entry-id formula (promoted_at
# was part of canon_entry_id); they are "v1 hash model" snapshots.

# Workbook columns that MUST exist for promote to operate.
REQUIRED_WORKBOOK_COLS = {
    "row_id",
    "candidate_type",
    "cluster_id",
    "candidate_id",
    "content",
    "entity",
    "predicate",
    "supporting_count",
    "provenance_conversation_uuids",
    "provenance_earliest",
    "provenance_most_recent",
    "source_lane",
    "review_status",
    "reviewer_notes",
    "promote_to_canon",
}

# Valid review_status vocabulary (case-insensitive match).
VALID_REVIEW_STATUSES = {
    "pending_review",
    "approved",
    "rejected",
    "needs_followup",
    "retired",      # reviewer may use this for obsolete rows they confirm retire
    "merged",       # reviewer may use this when combining with another cluster
}

# Values treated as "yes" for promote_to_canon (case-insensitive).
YES_VALUES = {"yes", "y", "true", "1", "promote"}
NO_VALUES = {"", "no", "n", "false", "0", "skip"}

DEFAULT_CANON_PACKS_DIR = Path("tools/cognitive_foundry/claude_lane/canon_packs")

# Columns in the canon-pack CSV (human-readable view).
CANON_PACK_ENTRY_COLS = [
    "canon_entry_id",
    "source_candidate_id",
    "candidate_type",
    "cluster_id",
    "content",
    "entity",
    "predicate",
    "supporting_count",
    "provenance_source_lane",
    "provenance_conversation_uuids",
    "provenance_earliest",
    "provenance_most_recent",
    "review_status",
    "reviewer_notes",
    "promoted_at",
    "promoter",
    "origin_run_id",
]

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class ValidationError:
    row_id: str
    kind: str
    detail: str


@dataclass
class Partition:
    to_promote: List[Dict[str, str]] = field(default_factory=list)
    approved_not_promoted: List[Dict[str, str]] = field(default_factory=list)
    rejected: List[Dict[str, str]] = field(default_factory=list)
    needs_followup: List[Dict[str, str]] = field(default_factory=list)
    pending: List[Dict[str, str]] = field(default_factory=list)
    other_status: List[Dict[str, str]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _read_workbook(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read the workbook CSV. Returns (rows, fieldnames)."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Workbook {path} has no header row.")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_yesno(value: str) -> str:
    v = (value or "").strip().lower()
    if v in YES_VALUES:
        return "yes"
    if v in NO_VALUES:
        return "no"
    return "__invalid__"


def _normalize_status(value: str) -> str:
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v in VALID_REVIEW_STATUSES:
        return v
    return "__invalid__"


def _canon_entry_id(source_candidate_id: str, content: str) -> str:
    """Deterministic 16-hex canon entry id (v2 — promoted_at NOT included).

    Same source_candidate_id + same content => same canon_entry_id regardless
    of when the promotion was run. This lets a future ingestion tool
    deterministically identify re-promotions of the same row across multiple
    review passes.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"canon_entry|v2|")
    h.update(source_candidate_id.encode("utf-8"))
    h.update(b"|")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def _guess_origin_run_id(workbook_path: Path) -> str:
    """Try to extract the origin pipeline run_id from the workbook path.

    Expected path shape:
        .../output/<run_id>/04_review_pack/review_workbook.csv
    """
    parts = workbook_path.resolve().parts
    for i, p in enumerate(parts):
        if p == "output" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def _promoter_identity() -> str:
    return (
        os.environ.get("BIZRA_PROMOTER")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "unknown"
    )


# -----------------------------------------------------------------------------
# Core: validate and partition
# -----------------------------------------------------------------------------


def validate_workbook(rows: List[Dict[str, str]]) -> Tuple[Partition, List[ValidationError]]:
    """Validate the workbook rows. Returns (partition, errors).

    Errors do NOT block partitioning; partition returns what it can. Caller
    decides whether to proceed based on error severity.
    """
    errors: List[ValidationError] = []
    part = Partition()

    for row in rows:
        row_id = (row.get("row_id") or "").strip() or "<no_row_id>"
        raw_status = row.get("review_status", "")
        raw_promote = row.get("promote_to_canon", "")

        status = _normalize_status(raw_status)
        promote = _normalize_yesno(raw_promote)

        # --- Row-level validation ---
        if status == "__invalid__":
            errors.append(
                ValidationError(
                    row_id=row_id,
                    kind="invalid_review_status",
                    detail=f"review_status={raw_status!r} not in {sorted(VALID_REVIEW_STATUSES)}",
                )
            )
            continue

        if promote == "__invalid__":
            errors.append(
                ValidationError(
                    row_id=row_id,
                    kind="invalid_promote_to_canon",
                    detail=f"promote_to_canon={raw_promote!r} — expected 'yes'/'no'/blank",
                )
            )
            continue

        # Contradiction: promote=yes but review_status != approved
        if promote == "yes" and status != "approved":
            errors.append(
                ValidationError(
                    row_id=row_id,
                    kind="contradiction_promote_without_approval",
                    detail=(
                        f"promote_to_canon=yes but review_status={raw_status!r}. "
                        "A row can only be promoted when it is explicitly approved."
                    ),
                )
            )
            # Still partition into approved_not_promoted so the operator sees it.
            part.approved_not_promoted.append(row)
            continue

        # --- Partition ---
        if status == "approved" and promote == "yes":
            part.to_promote.append(row)
        elif status == "approved" and promote != "yes":
            part.approved_not_promoted.append(row)
        elif status == "rejected":
            part.rejected.append(row)
        elif status == "needs_followup":
            part.needs_followup.append(row)
        elif status == "pending_review":
            part.pending.append(row)
        else:
            part.other_status.append(row)

    return part, errors


# -----------------------------------------------------------------------------
# Core: build canon entries
# -----------------------------------------------------------------------------


def build_canon_entries(
    promote_rows: List[Dict[str, str]],
    promoted_at: str,
    promoter: str,
    origin_run_id: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for row in promote_rows:
        source_candidate_id = (row.get("candidate_id") or "").strip()
        content = (row.get("content") or "").strip()
        # canon_entry_id is v2 — deterministic over stable content only.
        # promoted_at is recorded below as provenance but does NOT enter the id.
        entry_id = _canon_entry_id(source_candidate_id, content)

        entries.append(
            {
                "canon_entry_id": entry_id,
                "source_candidate_id": source_candidate_id,
                "candidate_type": (row.get("candidate_type") or "").strip(),
                "cluster_id": (row.get("cluster_id") or "").strip(),
                "content": content,
                "entity": (row.get("entity") or "").strip(),
                "predicate": (row.get("predicate") or "").strip(),
                "supporting_count": (row.get("supporting_count") or "").strip(),
                "provenance_source_lane": (row.get("source_lane") or "").strip(),
                "provenance_conversation_uuids": (row.get("provenance_conversation_uuids") or "").strip(),
                "provenance_earliest": (row.get("provenance_earliest") or "").strip(),
                "provenance_most_recent": (row.get("provenance_most_recent") or "").strip(),
                "review_status": (row.get("review_status") or "").strip(),
                "reviewer_notes": (row.get("reviewer_notes") or "").strip(),
                "promoted_at": promoted_at,
                "promoter": promoter,
                "origin_run_id": origin_run_id,
            }
        )
    # Deterministic order
    entries.sort(key=lambda e: e["canon_entry_id"])
    return entries


# Fields included in content_hash. Stable across promotion events.
# Deliberately EXCLUDED: promoted_at, promoter (these are issuance-time facts).
_CONTENT_HASH_FIELDS: List[str] = [
    "canon_entry_id",
    "source_candidate_id",
    "candidate_type",
    "cluster_id",
    "content",
    "entity",
    "predicate",
    "supporting_count",
    "provenance_source_lane",
    "provenance_conversation_uuids",
    "provenance_earliest",
    "provenance_most_recent",
    "review_status",
    "reviewer_notes",
    "origin_run_id",
]


def compute_content_hash(entries: List[Dict[str, Any]]) -> str:
    """Deterministic blake2b-32 over STABLE entry fields only.

    Same workbook rows + same reviewed content => same content_hash,
    regardless of when the promotion ran. Used as content-identity
    fingerprint (not event fingerprint — see compute_issuance_hash).
    """
    stable_entries = []
    for e in sorted(entries, key=lambda x: x["canon_entry_id"]):
        stable_entries.append({k: e.get(k, "") for k in _CONTENT_HASH_FIELDS})
    payload = json.dumps(
        stable_entries, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()


def compute_issuance_hash(
    content_hash: str,
    promoted_at: str,
    promoter: str,
    workbook_sha256: str,
) -> str:
    """Deterministic blake2b-32 over promotion-event fields.

    Two promotions of the same content with different timestamps produce
    the SAME content_hash but DIFFERENT issuance_hash. Used as the
    unique identifier for a specific promotion event (and for the pack
    directory name) so reruns never collide on disk.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(b"issuance|v1|")
    h.update(content_hash.encode("utf-8"))
    h.update(b"|")
    h.update(promoted_at.encode("utf-8"))
    h.update(b"|")
    h.update(promoter.encode("utf-8"))
    h.update(b"|")
    h.update(workbook_sha256.encode("utf-8"))
    return h.hexdigest()


# Backward-compatibility alias. Callers may still reference "entries_hash";
# under v0.2.0 it == content_hash (deterministic).
def compute_entries_hash(entries: List[Dict[str, Any]]) -> str:
    return compute_content_hash(entries)


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})


def _write_promotion_report(
    path: Path,
    *,
    workbook_path: Path,
    origin_run_id: str,
    promoted_at: str,
    promoter: str,
    part: Partition,
    errors: List[ValidationError],
    entries_hash: Optional[str],
    pack_dir: Optional[Path],
    dry_run: bool,
) -> None:
    lines: List[str] = []
    lines.append(f"# Canon-pack promotion report")
    lines.append("")
    lines.append(f"- **Tool:** `{TOOL_NAME}` v{TOOL_VERSION}")
    lines.append(f"- **Workbook:** `{workbook_path}`")
    lines.append(f"- **Origin run_id:** `{origin_run_id}`")
    lines.append(f"- **Promoted at (UTC):** `{promoted_at}`")
    lines.append(f"- **Promoter:** `{promoter}`")
    lines.append(f"- **Dry run:** `{dry_run}`")
    if entries_hash:
        lines.append(f"- **Entries hash (blake2b-32):** `{entries_hash}`")
    if pack_dir:
        lines.append(f"- **Pack directory:** `{pack_dir}`")
    lines.append("")
    lines.append("## Partition counts")
    lines.append("")
    lines.append(f"| Category | Count |")
    lines.append(f"|---|---|")
    lines.append(f"| **Approved AND promote_to_canon=yes (PROMOTED)** | **{len(part.to_promote)}** |")
    lines.append(f"| Approved but promote_to_canon not yes | {len(part.approved_not_promoted)} |")
    lines.append(f"| Rejected | {len(part.rejected)} |")
    lines.append(f"| Needs follow-up | {len(part.needs_followup)} |")
    lines.append(f"| Pending review (unreviewed) | {len(part.pending)} |")
    lines.append(f"| Other (retired / merged / ...) | {len(part.other_status)} |")
    lines.append("")

    if errors:
        lines.append(f"## 🚩 Validation errors ({len(errors)})")
        lines.append("")
        for e in errors:
            lines.append(f"- **{e.row_id}** — `{e.kind}` — {e.detail}")
        lines.append("")
    else:
        lines.append("## Validation errors")
        lines.append("")
        lines.append("None.")
        lines.append("")

    lines.append("## Canon discipline notes")
    lines.append("")
    lines.append("- This pack is NOT yet in any canonical BIZRA store.")
    lines.append("- A future, separately-implemented tool must ingest this pack into actual canon (e.g., MEMORY.md entries or a runtime canonical index) — and that tool will require its own confirmation gate.")
    lines.append("- The `entries_hash` is content-addressable tamper evidence, not a cryptographic signature. A future promotion tool can upgrade to Ed25519 if required.")
    lines.append("- The pipeline NEVER auto-sets `promote_to_canon=yes`. Every row in the PROMOTED count above was explicitly marked by a human reviewer.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claude_lane.promote",
        description=(
            "Promote approved+reviewed candidates from an annotated review_workbook.csv "
            "into a structured, hash-signed canon pack. Never auto-promotes. Never writes "
            "to MEMORY.md or runtime files."
        ),
    )
    parser.add_argument(
        "--workbook",
        required=True,
        type=Path,
        help="Path to the annotated review_workbook.csv (Stage 4 output with reviewer annotations).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CANON_PACKS_DIR,
        help=f"Output directory for canon packs (default: {DEFAULT_CANON_PACKS_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate + report without writing the pack to disk.",
    )
    args = parser.parse_args(argv)

    workbook_path: Path = args.workbook.resolve()
    if not workbook_path.exists():
        print(f"[error] Workbook not found: {workbook_path}", file=sys.stderr)
        return 2

    # Read and validate workbook columns
    try:
        rows, fieldnames = _read_workbook(workbook_path)
    except (OSError, ValueError, csv.Error) as e:
        print(f"[error] Failed to read workbook: {e}", file=sys.stderr)
        return 3

    missing = REQUIRED_WORKBOOK_COLS - set(fieldnames)
    if missing:
        print(
            f"[error] Workbook missing required columns: {sorted(missing)}",
            file=sys.stderr,
        )
        return 3

    # Validate and partition
    part, errors = validate_workbook(rows)

    # Hard stop on row-level validation errors (contradictions, invalid status, etc.)
    hard_stop_kinds = {
        "contradiction_promote_without_approval",
        "invalid_review_status",
        "invalid_promote_to_canon",
    }
    hard_errors = [e for e in errors if e.kind in hard_stop_kinds]
    if hard_errors:
        print(f"[error] {len(hard_errors)} validation error(s) — refusing to promote:", file=sys.stderr)
        for e in hard_errors:
            print(f"  - {e.row_id}: {e.kind}: {e.detail}", file=sys.stderr)
        # Still produce a report so the operator can see the partition state
        report_path = workbook_path.parent / f"canon_pack_report_validation_failed_{_ts_now_filename()}.md"
        _write_promotion_report(
            report_path,
            workbook_path=workbook_path,
            origin_run_id=_guess_origin_run_id(workbook_path),
            promoted_at=_ts_now_iso(),
            promoter=_promoter_identity(),
            part=part,
            errors=errors,
            entries_hash=None,
            pack_dir=None,
            dry_run=True,
        )
        print(f"[info] Validation report: {report_path}", file=sys.stderr)
        return 4

    # Shortcut: no rows to promote
    promoted_at_iso = _ts_now_iso()
    promoter = _promoter_identity()
    origin_run_id = _guess_origin_run_id(workbook_path)

    if not part.to_promote:
        # Still produce a report (useful for the operator to see "workbook is reviewed but no rows promoted yet")
        report_dir = args.output_dir.resolve() / f"{origin_run_id}_noop_{_ts_now_filename()}"
        report_path = report_dir / "promotion_report.md"
        if not args.dry_run:
            _write_promotion_report(
                report_path,
                workbook_path=workbook_path,
                origin_run_id=origin_run_id,
                promoted_at=promoted_at_iso,
                promoter=promoter,
                part=part,
                errors=errors,
                entries_hash=None,
                pack_dir=report_dir,
                dry_run=False,
            )
            print(f"[info] No rows marked approved+promoted. No-op report: {report_path}")
        else:
            print("[info] Dry run: no rows marked approved+promoted. No pack would be written.")
        print(f"[info] Partition: "
              f"to_promote={len(part.to_promote)} "
              f"approved_not_promoted={len(part.approved_not_promoted)} "
              f"rejected={len(part.rejected)} "
              f"needs_followup={len(part.needs_followup)} "
              f"pending={len(part.pending)} "
              f"other_status={len(part.other_status)}")
        return 0

    # Build canon entries
    entries = build_canon_entries(
        part.to_promote,
        promoted_at=promoted_at_iso,
        promoter=promoter,
        origin_run_id=origin_run_id,
    )
    # v0.2.0: split hashes.
    # content_hash — deterministic over stable fields (no promoted_at / promoter).
    # issuance_hash — unique per promotion event (includes promoted_at).
    workbook_hash = _file_sha256(workbook_path)
    content_hash = compute_content_hash(entries)
    issuance_hash = compute_issuance_hash(
        content_hash=content_hash,
        promoted_at=promoted_at_iso,
        promoter=promoter,
        workbook_sha256=workbook_hash,
    )

    # Paths — use ISSUANCE hash for the directory so reruns (same content,
    # different timestamps) never collide on disk.
    pack_stem = f"{origin_run_id}_promoted_{_ts_now_filename()}_{issuance_hash[:12]}"
    pack_dir = args.output_dir.resolve() / pack_stem
    pack_json_path = pack_dir / "canon_pack.json"
    pack_csv_path = pack_dir / "canon_pack.csv"
    pack_manifest_path = pack_dir / "canon_pack.manifest.json"
    report_path = pack_dir / "promotion_report.md"

    # Manifest
    manifest = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "hash_model": "v2_split_content_and_issuance",
        "workbook_source": str(workbook_path),
        "workbook_sha256": workbook_hash,
        "origin_run_id": origin_run_id,
        "promoted_at_utc": promoted_at_iso,
        "promoter": promoter,
        "entry_count": len(entries),
        # v0.2.0 split:
        "content_hash_blake2b_32": content_hash,
        "issuance_hash_blake2b_32": issuance_hash,
        # Backward-compat alias. Under v0.2.0 this == content_hash (deterministic).
        # v0.1.0 consumers reading this field will see the stable content-identity
        # hash, which is arguably more useful than the old promoted_at-mixed hash.
        "entries_hash_blake2b_32": content_hash,
        "partition_counts": {
            "to_promote": len(part.to_promote),
            "approved_not_promoted": len(part.approved_not_promoted),
            "rejected": len(part.rejected),
            "needs_followup": len(part.needs_followup),
            "pending": len(part.pending),
            "other_status": len(part.other_status),
        },
        "non_promotion_tool": True,
        "human_gated": True,
        "notes": (
            "Canon pack v0.2.0 (split-hash model). content_hash identifies the "
            "reviewed content (deterministic across reruns). issuance_hash "
            "identifies this specific promotion event. NOT cryptographically "
            "signed. NOT yet ingested into any canonical BIZRA store. A "
            "separate, human-gated tool must perform the final ingest."
        ),
    }

    if args.dry_run:
        print(f"[dry-run] Would write pack to: {pack_dir}")
        print(f"[dry-run] entry_count={len(entries)}")
        print(f"[dry-run] content_hash={content_hash}")
        print(f"[dry-run] issuance_hash={issuance_hash}")
        return 0

    # Write pack
    _write_json(pack_json_path, entries)
    _write_csv(pack_csv_path, entries, CANON_PACK_ENTRY_COLS)
    _write_json(pack_manifest_path, manifest)
    _write_promotion_report(
        report_path,
        workbook_path=workbook_path,
        origin_run_id=origin_run_id,
        promoted_at=promoted_at_iso,
        promoter=promoter,
        part=part,
        errors=errors,
        entries_hash=content_hash,  # show content_hash in the human-readable report
        pack_dir=pack_dir,
        dry_run=False,
    )

    print(f"[info] Canon pack written: {pack_dir}")
    print(f"[info] entry_count={len(entries)}")
    print(f"[info] content_hash={content_hash}   (deterministic — same content always hashes the same)")
    print(f"[info] issuance_hash={issuance_hash}  (unique per promotion event)")
    print("[info] NOTE: pack sits on disk awaiting a separate, human-gated ingestion tool. "
          "Nothing has been written to MEMORY.md or any runtime file.")
    return 0


def _ts_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_now_filename() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
