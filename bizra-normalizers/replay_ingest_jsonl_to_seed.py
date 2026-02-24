#!/usr/bin/env python3
"""Replay ingest JSONL payloads into deterministic TEACH seed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


_FALLBACK_KIND_SIGNAL_RE = re.compile(r"kind=([^;]+);\s*signal=([^;]+);")

_KIND_MAP = {
    "preference": "preference",
    "goal": "goal",
    "expertise": "expertise",
    "fact": "fact",
    "pattern": "pattern",
    "relationship": "relationship",
    "temporal": "temporal",
    # These kinds do not exist directly in TEACH v1 and are mapped to context.
    "style": "context",
    "emotion": "context",
    "domain": "context",
}


@dataclass(frozen=True)
class SeedRow:
    seed_kind: str
    signal: str
    confidence_raw: int
    timestamp: int
    turn: int


def _escape_seed_content(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _seed_kind(kind: str) -> str:
    return _KIND_MAP.get(kind.strip().lower(), "context")


def _extract_kind_signal(row: dict) -> tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    kind = str(metadata.get("kind") or "").strip()
    signal = str(metadata.get("signal") or "").strip()

    if kind and signal:
        return kind, signal

    content = str(row.get("content") or "").strip()
    if content:
        match = _FALLBACK_KIND_SIGNAL_RE.search(content)
        if match:
            if not kind:
                kind = match.group(1).strip()
            if not signal:
                signal = match.group(2).strip()

    if not signal:
        signal = content

    return kind, signal


def _confidence_raw(row: dict) -> int:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    snr = metadata.get("snr_score")
    try:
        score = float(snr)
    except (TypeError, ValueError):
        score = 0.85
    score = max(0.5, min(0.99, score))
    return int(round(score * 10000))


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> list[SeedRow]:
    out: list[SeedRow] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        kind, signal = _extract_kind_signal(row)
        signal = " ".join(signal.split())
        if not signal:
            continue
        out.append(
            SeedRow(
                seed_kind=_seed_kind(kind),
                signal=signal,
                confidence_raw=_confidence_raw(row),
                timestamp=_parse_int(row.get("timestamp"), 0),
                turn=_parse_int(row.get("turn"), 0),
            )
        )
    return out


def _dedupe_and_sort(rows: list[SeedRow]) -> list[SeedRow]:
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row.timestamp,
            row.turn,
            row.seed_kind,
            row.signal.lower(),
            row.signal,
        ),
    )
    seen: set[tuple[str, str]] = set()
    out: list[SeedRow] = []
    for row in rows_sorted:
        key = (row.seed_kind, row.signal.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _write_seed(rows: list[SeedRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# BIZRA deterministic seed replay\n")
        fh.write("# source: replay_ingest_jsonl_to_seed.py\n")
        fh.write(f"# rows: {len(rows)}\n\n")
        for row in rows:
            fh.write(
                "TEACH\t{kind}\t{signal}\t{confidence}\t{timestamp}\n".format(
                    kind=row.seed_kind,
                    signal=_escape_seed_content(row.signal),
                    confidence=row.confidence_raw,
                    timestamp=row.timestamp,
                )
            )


def _write_checksum(out_path: Path) -> tuple[Path, str]:
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    checksum_path = Path(f"{out_path}.sha256")
    checksum_path.write_text(f"{digest}  {out_path.name}\n", encoding="utf-8")
    return checksum_path, digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay ingest JSONL to deterministic TEACH seed")
    parser.add_argument("--in", dest="input_path", required=True, help="Input ingest JSONL path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output .seed file path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    loaded = _load_rows(input_path)
    rows = _dedupe_and_sort(loaded)
    _write_seed(rows, output_path)
    checksum_path, digest = _write_checksum(output_path)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "checksum_path": str(checksum_path),
        "loaded_rows": len(loaded),
        "seed_rows": len(rows),
        "sha256": digest,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Replay complete")
        print(f"Loaded rows: {report['loaded_rows']}")
        print(f"Seed rows: {report['seed_rows']}")
        print(f"Seed path: {report['output_path']}")
        print(f"Checksum: {report['checksum_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
