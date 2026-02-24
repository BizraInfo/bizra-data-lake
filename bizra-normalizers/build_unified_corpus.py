#!/usr/bin/env python3
"""Build unified conversation corpus artifacts from multi-platform exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizers import parse_file_with_receipt  # noqa: E402
from normalizers.base import normalize_whitespace  # noqa: E402
from schemas import ConversationTurn  # noqa: E402

try:
    from blake3 import blake3 as _blake3
except Exception:  # pragma: no cover - fallback path
    _blake3 = None


SCHEMA_COLUMNS: tuple[str, ...] = (
    "id",
    "platform",
    "conversation_id",
    "turn_index",
    "role",
    "content",
    "model",
    "timestamp",
    "metadata_json",
    "token_count",
    "content_hash",
    "language",
    "topics_json",
)

_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("architecture", re.compile(r"\b(architecture|design|system)\b", re.IGNORECASE)),
    ("security", re.compile(r"\b(security|guardian|threat|audit)\b", re.IGNORECASE)),
    ("memory", re.compile(r"\b(memory|retrieval|fragment|episodic)\b", re.IGNORECASE)),
    ("agents", re.compile(r"\b(agent|pat|sat|orchestrat)\w*\b", re.IGNORECASE)),
    ("compiler", re.compile(r"\b(compiler|genesis|gate|reflex)\b", re.IGNORECASE)),
    ("governance", re.compile(r"\b(constitution|policy|ihsan|sovereign)\b", re.IGNORECASE)),
    ("economy", re.compile(r"\b(token|proof-of-impact|poi|economy)\b", re.IGNORECASE)),
    ("frontend", re.compile(r"\b(ui|ux|react|css|frontend)\b", re.IGNORECASE)),
)


@dataclass
class Row:
    id: str
    platform: str
    conversation_id: str
    turn_index: int
    role: str
    content: str
    model: str
    timestamp: int
    metadata: dict[str, Any]
    token_count: int
    content_hash: str
    language: str
    topics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "role": self.role,
            "content": self.content,
            "model": self.model,
            "timestamp": self.timestamp,
            "metadata_json": json.dumps(self.metadata, sort_keys=True, ensure_ascii=False),
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "language": self.language,
            "topics_json": json.dumps(self.topics, ensure_ascii=False),
        }


def _iter_input_files(paths: list[Path]) -> list[Path]:
    allowed = {".json", ".jsonl"}
    out: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in allowed:
            out.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed:
                out.append(path)
    return sorted(set(out))


def _hash_content(text: str) -> str:
    normalized = normalize_whitespace(text).strip().lower()
    if _blake3 is not None:
        return _blake3(normalized.encode("utf-8")).hexdigest()
    import hashlib

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> dict[str, str]:
    sha256 = hashlib.sha256()
    blake3_hasher = _blake3() if _blake3 is not None else None
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
            if blake3_hasher is not None:
                blake3_hasher.update(chunk)
    return {
        "blake3": (
            blake3_hasher.hexdigest() if blake3_hasher is not None else sha256.hexdigest()
        ),
        "sha256": sha256.hexdigest(),
    }


def _estimate_tokens(content: str, model: str) -> int:
    text = content.strip()
    if not text:
        return 0
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    if has_cjk:
        return max(1, len(text) // 2)
    if model.lower().startswith(("gpt-", "o1", "o3", "o4")):
        return max(1, round(len(text.split()) * 1.25))
    return max(1, round(len(text) / 4))


def _detect_language(content: str) -> str:
    if re.search(r"[\u0600-\u06FF]", content):
        return "ar"
    if re.search(r"[\u4e00-\u9fff]", content):
        return "zh"
    if re.search(r"[A-Za-z]", content):
        return "en"
    return "unknown"


def _extract_topics(content: str, max_topics: int = 5) -> list[str]:
    hits: list[tuple[int, str]] = []
    for topic, pattern in _TOPIC_PATTERNS:
        count = len(pattern.findall(content))
        if count > 0:
            hits.append((count, topic))
    hits.sort(key=lambda item: (-item[0], item[1]))
    return [topic for _, topic in hits[:max_topics]]


def _turns_to_rows(turns: list[ConversationTurn]) -> list[Row]:
    rows: list[Row] = []
    turn_counter: dict[tuple[str, str], int] = {}

    for turn in turns:
        key = (turn.provider, turn.conversation_id)
        idx = turn_counter.get(key, 0)
        turn_counter[key] = idx + 1

        content_hash = _hash_content(turn.content)
        rows.append(
            Row(
                id=turn.turn_id,
                platform=turn.provider,
                conversation_id=turn.conversation_id,
                turn_index=idx,
                role=turn.role,
                content=turn.content,
                model=turn.model or "",
                timestamp=turn.timestamp,
                metadata={
                    **(turn.metadata or {}),
                    "fragment_hint_count": len(turn.fragment_hints),
                    "fragment_sources": sorted({h.source for h in turn.fragment_hints}),
                },
                token_count=_estimate_tokens(turn.content, turn.model or ""),
                content_hash=content_hash,
                language=_detect_language(turn.content),
                topics=_extract_topics(turn.content),
            )
        )
    return rows


def _dedupe_rows(rows: list[Row]) -> tuple[list[Row], dict[str, Any]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.content_hash, []).append(row)

    canonical_rows: list[Row] = []
    manifest: dict[str, Any] = {}

    for content_hash, variants in grouped.items():
        ordered = sorted(
            variants,
            key=lambda row: (
                row.timestamp if row.timestamp > 0 else 9_999_999_999,
                row.platform,
                row.id,
            ),
        )
        canonical = ordered[0]
        duplicates = ordered[1:]
        duplicate_ids = [r.id for r in duplicates]
        duplicate_platforms = sorted({r.platform for r in duplicates})
        if duplicate_ids:
            canonical.metadata = {
                **canonical.metadata,
                "duplicate_ids": duplicate_ids,
                "duplicate_platforms": duplicate_platforms,
            }
        canonical_rows.append(canonical)
        manifest[content_hash] = {
            "canonical_id": canonical.id,
            "canonical_platform": canonical.platform,
            "duplicate_ids": duplicate_ids,
            "duplicate_platforms": duplicate_platforms,
            "variant_count": len(ordered),
        }

    canonical_rows.sort(
        key=lambda row: (
            row.timestamp if row.timestamp > 0 else 9_999_999_999,
            row.platform,
            row.id,
        )
    )
    return canonical_rows, dict(sorted(manifest.items()))


def _build_platform_index(rows: list[Row]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for row in rows:
        out.setdefault(row.platform, set()).add(row.conversation_id)
    return {platform: sorted(ids) for platform, ids in sorted(out.items())}


def _build_timeline_index(rows: list[Row]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        if row.timestamp > 0:
            month = datetime.fromtimestamp(row.timestamp, tz=timezone.utc).strftime("%Y-%m")
        else:
            month = "unknown"
        out.setdefault(month, []).append(row.id)
    return {month: sorted(ids) for month, ids in sorted(out.items())}


def build_unified_corpus(
    paths: list[Path],
    out_parquet: Path,
    out_dir: Path,
) -> dict[str, Any]:
    files = _iter_input_files(paths)
    turns: list[ConversationTurn] = []
    parse_failures: list[dict[str, Any]] = []

    for path in files:
        parsed, parse_receipt = parse_file_with_receipt(path)
        if parsed:
            turns.extend(parsed)
        else:
            parse_failures.append(parse_receipt)

    raw_rows = _turns_to_rows(turns)
    rows, dedup_manifest = _dedupe_rows(raw_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row.to_dict() for row in rows], columns=list(SCHEMA_COLUMNS))
    frame.to_parquet(out_parquet, index=False, engine="pyarrow", compression="zstd")
    parquet_hashes = _hash_file(out_parquet)
    parquet_receipt_path = out_parquet.with_suffix(out_parquet.suffix + ".receipt.json")
    parquet_receipt = {
        "artifact_path": str(out_parquet),
        "artifact_size_bytes": out_parquet.stat().st_size,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "hash_policy": "blake3_primary_sha256_compat",
        "blake3": parquet_hashes["blake3"],
        "sha256": parquet_hashes["sha256"],
        "input_file_count": len(files),
        "total_turns_raw": len(raw_rows),
        "total_turns_unified": len(rows),
        "duplicate_turns_removed": max(0, len(raw_rows) - len(rows)),
    }
    parquet_receipt_path.write_text(
        json.dumps(parquet_receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    platform_index = _build_platform_index(rows)
    timeline_index = _build_timeline_index(rows)

    (out_dir / "platform_index.json").write_text(
        json.dumps(platform_index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "timeline_index.json").write_text(
        json.dumps(timeline_index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "dedup_manifest.json").write_text(
        json.dumps(dedup_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    parse_failure_reasons: dict[str, int] = {}
    for failure in parse_failures:
        reason_code = str(failure.get("reason_code") or "UNKNOWN")
        parse_failure_reasons[reason_code] = parse_failure_reasons.get(reason_code, 0) + 1
    parse_failures_path = out_dir / "parse_failures.json"
    parse_failures_path.write_text(
        json.dumps(parse_failures, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    provider_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for row in rows:
        provider_counts[row.platform] = provider_counts.get(row.platform, 0) + 1
        language_counts[row.language] = language_counts.get(row.language, 0) + 1

    report = {
        "schema_columns": list(SCHEMA_COLUMNS),
        "input_file_count": len(files),
        "parse_failure_count": len(parse_failures),
        "parse_failure_reasons": dict(sorted(parse_failure_reasons.items())),
        "parse_failures_path": str(parse_failures_path),
        "total_turns_raw": len(raw_rows),
        "total_turns_unified": len(rows),
        "duplicate_turns_removed": max(0, len(raw_rows) - len(rows)),
        "providers": dict(sorted(provider_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "output_parquet": str(out_parquet),
        "output_parquet_blake3": parquet_hashes["blake3"],
        "output_parquet_sha256": parquet_hashes["sha256"],
        "output_parquet_receipt_path": str(parquet_receipt_path),
        "output_dir": str(out_dir),
    }
    (out_dir / "ingestion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified corpus parquet + index artifacts")
    parser.add_argument("paths", nargs="+", help="Input files/directories (.json/.jsonl)")
    parser.add_argument(
        "--out-parquet",
        type=str,
        default="/mnt/c/BIZRA-DATA-LAKE/04_GOLD/conversations_unified.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/c/BIZRA-DATA-LAKE/04_GOLD",
        help="Output directory for index/report artifacts",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    report = build_unified_corpus(
        paths=[Path(p).expanduser().resolve() for p in args.paths],
        out_parquet=Path(args.out_parquet).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Unified corpus build complete")
        print(f"Input files: {report['input_file_count']}")
        print(f"Raw turns: {report['total_turns_raw']}")
        print(f"Unified turns: {report['total_turns_unified']}")
        print(f"Duplicates removed: {report['duplicate_turns_removed']}")
        print(f"Parquet: {report['output_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
