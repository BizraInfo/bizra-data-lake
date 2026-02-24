#!/usr/bin/env python3
"""Build canonical Core-8 corpus records and deterministic dedup report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from provider_normalizers import (
    CORE8,
    CorpusRecord,
    iter_candidate_json_files,
    iter_records_from_file,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [
    REPO / "00_INTAKE",
    REPO / "sovereign_state" / "chat_import",
]
DEFAULT_OUTDIR = REPO / "artifacts" / "corpus" / "v1"


def _record_key_primary(r: CorpusRecord) -> tuple[str, str, str, str]:
    return (r.provider, r.account_scope, r.conversation_id, r.message_id)


def _record_key_fallback(r: CorpusRecord) -> tuple[str, str, int]:
    bucket = int(r.timestamp // 300) if r.timestamp else 0
    return (r.role, r.content_hash, bucket)


def _to_dict(r: CorpusRecord) -> dict[str, Any]:
    return {
        "provider": r.provider,
        "account_scope": r.account_scope,
        "conversation_id": r.conversation_id,
        "message_id": r.message_id,
        "role": r.role,
        "timestamp": r.timestamp,
        "content_hash": r.content_hash,
        "source_path": r.source_path,
        "import_run_id": r.import_run_id,
    }


def _stable_pick(records: list[CorpusRecord]) -> CorpusRecord:
    return sorted(records, key=lambda r: (r.timestamp, r.source_path, r.message_id))[0]


def run(roots: list[Path], outdir: Path, run_id: str, dry_run: bool) -> int:
    outdir.mkdir(parents=True, exist_ok=True)

    all_records: list[CorpusRecord] = []
    for json_file in iter_candidate_json_files(roots):
        all_records.extend(iter_records_from_file(json_file, run_id))

    raw_records = len(all_records)
    raw_conversations = len(
        {(r.provider, r.account_scope, r.conversation_id) for r in all_records}
    )

    # Step 1: primary-key dedup.
    primary_groups: dict[tuple[str, str, str, str], list[CorpusRecord]] = defaultdict(
        list
    )
    for rec in all_records:
        primary_groups[_record_key_primary(rec)].append(rec)

    post_primary: list[CorpusRecord] = []
    duplicate_clusters: list[dict[str, Any]] = []
    for key, recs in sorted(primary_groups.items(), key=lambda kv: kv[0]):
        keep = _stable_pick(recs)
        post_primary.append(keep)
        if len(recs) > 1:
            dropped = [
                r
                for r in sorted(
                    recs, key=lambda x: (x.timestamp, x.source_path, x.message_id)
                )
                if r != keep
            ]
            duplicate_clusters.append(
                {
                    "kind": "primary",
                    "key": "|".join(key),
                    "count": len(recs),
                    "kept_message_id": keep.message_id,
                    "dropped_message_ids": [r.message_id for r in dropped],
                }
            )

    # Step 2: fallback dedup for residual near-duplicates.
    fallback_groups: dict[tuple[str, str, int], list[CorpusRecord]] = defaultdict(list)
    for rec in post_primary:
        fallback_groups[_record_key_fallback(rec)].append(rec)

    unique_records: list[CorpusRecord] = []
    for key, recs in sorted(fallback_groups.items(), key=lambda kv: kv[0]):
        keep = _stable_pick(recs)
        unique_records.append(keep)
        if len(recs) > 1:
            dropped = [
                r
                for r in sorted(
                    recs, key=lambda x: (x.timestamp, x.source_path, x.message_id)
                )
                if r != keep
            ]
            duplicate_clusters.append(
                {
                    "kind": "fallback",
                    "key": f"{key[0]}|{key[1]}|{key[2]}",
                    "count": len(recs),
                    "kept_message_id": keep.message_id,
                    "dropped_message_ids": [r.message_id for r in dropped],
                }
            )

    unique_count = len(unique_records)
    unique_conversations = len(
        {(r.provider, r.account_scope, r.conversation_id) for r in unique_records}
    )
    providers_detected = sorted(
        {r.provider for r in all_records if r.provider in CORE8}
    )
    duplication_factor = (raw_records / unique_count) if unique_count else 0.0

    report = {
        "run_id": run_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_records": raw_records,
        "unique_records": unique_count,
        "duplication_factor": round(duplication_factor, 6),
        "providers_detected": providers_detected,
        "raw_conversations": raw_conversations,
        "unique_conversations": unique_conversations,
        "duplicate_clusters": duplicate_clusters,
    }

    if dry_run:
        print(json.dumps(report, indent=2))
        return 0

    records_path = outdir / "core8_records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in sorted(
            unique_records,
            key=lambda r: (
                r.provider,
                r.account_scope,
                r.conversation_id,
                r.timestamp,
                r.message_id,
            ),
        ):
            fh.write(json.dumps(_to_dict(rec), ensure_ascii=False) + "\n")

    report_path = outdir / "dedup_report.v1.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {records_path}")
    print(f"Wrote {report_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Core 8 corpus dedup")
    parser.add_argument("--run-id", default=f"core8-{uuid.uuid4()}")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roots = [Path(p) for p in args.root] if args.root else DEFAULT_ROOTS
    raise SystemExit(
        run(roots=roots, outdir=args.outdir, run_id=args.run_id, dry_run=args.dry_run)
    )


if __name__ == "__main__":
    main()
