#!/usr/bin/env python3
"""Build corpus manifest from dedup outputs and optionally refresh node0 baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO / "artifacts" / "corpus" / "v1"
DEFAULT_DEDUP_REPORT = DEFAULT_OUTDIR / "dedup_report.v1.json"
DEFAULT_RECORDS = DEFAULT_OUTDIR / "core8_records.jsonl"
DEFAULT_MANIFEST = DEFAULT_OUTDIR / "corpus_manifest.v1.json"
DEFAULT_BASELINE = REPO / "sovereign_state" / "node0_baseline.json"

CORE8_ORDER = [
    "chatgpt_openai",
    "claude",
    "gemini_google",
    "deepseek",
    "qwen",
    "kimi",
    "perplexity",
    "zhipu",
]


def _sha256_obj(obj: dict[str, Any]) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _refresh_baseline(baseline_path: Path, manifest: dict[str, Any]) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    node_assets = baseline.setdefault("node0_assets", {})
    node_assets["ai_conversations"] = int(manifest["unique_conversations"])

    friction = baseline.setdefault("friction_areas", [])
    replaced = False
    for idx, item in enumerate(friction):
        if "AI conversations" in item:
            friction[idx] = (
                f"{manifest['unique_conversations']:,}+ unique AI conversations "
                f"({manifest['raw_conversations']:,} raw before dedup) with attested corpus manifest"
            )
            replaced = True
            break
    if not replaced:
        friction.append(
            f"{manifest['unique_conversations']:,}+ unique AI conversations with attested corpus manifest"
        )

    baseline["captured_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    temp = dict(baseline)
    temp.pop("hash", None)
    baseline["hash"] = _sha256_obj(temp)

    baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_attestation(attestation_path: Path, manifest: dict[str, Any], dedup: dict[str, Any]) -> None:
    providers_count = len(manifest.get("providers_covered", []))
    raw_conversations = int(manifest.get("raw_conversations", 0))
    unique_conversations = int(manifest.get("unique_conversations", 0))
    raw_messages = int(manifest.get("raw_messages", 0))
    unique_messages = int(manifest.get("unique_messages", 0))

    coverage_ratio = providers_count / len(CORE8_ORDER) if CORE8_ORDER else 0.0
    unique_conversation_ratio = (unique_conversations / raw_conversations) if raw_conversations else 0.0
    unique_message_ratio = (unique_messages / raw_messages) if raw_messages else 0.0
    duplicate_message_rate = 1.0 - unique_message_ratio

    lines = [
        "# Corpus Attestation Report",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Manifest hash: `{manifest['manifest_hash']}`",
        f"- Providers covered: `{', '.join(manifest['providers_covered'])}`",
        f"- Raw conversations: `{manifest['raw_conversations']}`",
        f"- Unique conversations: `{manifest['unique_conversations']}`",
        f"- Raw messages: `{manifest['raw_messages']}`",
        f"- Unique messages: `{manifest['unique_messages']}`",
        f"- Duplication factor: `{manifest['duplication_factor']}`",
        "",
        "## Derived Metrics (Mathematical)",
        "",
        "Let:",
        "- `P = providers_covered_count`",
        "- `C_raw = raw_conversations`",
        "- `C_unique = unique_conversations`",
        "- `M_raw = raw_messages`",
        "- `M_unique = unique_messages`",
        "",
        "Computed:",
        f"1. `core8_coverage_ratio = P/8 = {providers_count}/8 = {coverage_ratio:.4f}`",
        f"2. `unique_conversation_ratio = C_unique/C_raw = {unique_conversations}/{raw_conversations} = {unique_conversation_ratio:.6f}`",
        f"3. `unique_message_ratio = M_unique/M_raw = {unique_messages}/{raw_messages} = {unique_message_ratio:.6f}`",
        f"4. `duplicate_message_rate = 1 - unique_message_ratio = {duplicate_message_rate:.6f}`",
        f"5. `duplication_factor = M_raw/M_unique = {raw_messages}/{unique_messages} = {manifest['duplication_factor']}`",
        "",
        "## Evidence",
        "",
        f"1. Dedup report: `{DEFAULT_DEDUP_REPORT}`",
        f"2. Canonical records: `{DEFAULT_RECORDS}`",
        f"3. Manifest: `{DEFAULT_MANIFEST}`",
        "",
        "## Uncertainty Notes",
        "",
        "1. Provider parser coverage is Core-8 best effort and evolves as export formats change.",
        "2. Files that fail JSON parsing are skipped and counted implicitly by reduced discovered volume.",
        "3. This is an internal truth artifact and not an external audit statement.",
        "",
        "## Duplicate Cluster Summary",
        "",
        f"- Duplicate clusters: `{len(dedup.get('duplicate_clusters', []))}`",
    ]
    attestation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    dedup_report_path: Path,
    records_path: Path,
    manifest_out: Path,
    baseline_path: Path,
    write_baseline: bool,
    attestation_path: Path | None,
) -> int:
    if not dedup_report_path.exists():
        print(f"ERROR: missing dedup report {dedup_report_path}")
        return 1

    dedup = json.loads(dedup_report_path.read_text(encoding="utf-8"))
    records = _load_records(records_path)

    providers_detected = set(dedup.get("providers_detected", []))
    providers_covered = [p for p in CORE8_ORDER if p in providers_detected]

    raw_messages = int(dedup.get("raw_records", 0))
    unique_messages = int(dedup.get("unique_records", 0))
    raw_conversations = int(dedup.get("raw_conversations", 0))
    unique_conversations = int(dedup.get("unique_conversations", 0))
    duplication_factor = float(dedup.get("duplication_factor", 0.0))

    coverage_notes = []
    for provider in CORE8_ORDER:
        if provider in providers_detected:
            coverage_notes.append(f"{provider}: detected")
        else:
            coverage_notes.append(f"{provider}: not detected in current roots")

    manifest = {
        "providers_covered": providers_covered,
        "raw_conversations": raw_conversations,
        "raw_messages": raw_messages,
        "unique_conversations": unique_conversations,
        "unique_messages": unique_messages,
        "duplication_factor": round(duplication_factor, 6),
        "coverage_notes": coverage_notes,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_hash": "",
        "signature": "pending_internal_attestation",
    }

    manifest_for_hash = dict(manifest)
    manifest_for_hash.pop("manifest_hash", None)
    manifest["manifest_hash"] = _sha256_obj(manifest_for_hash)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_out}")

    if write_baseline:
        _refresh_baseline(baseline_path, manifest)
        print(f"Updated {baseline_path}")

    if attestation_path is not None:
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        _write_attestation(attestation_path, manifest, dedup)
        print(f"Wrote {attestation_path}")

    # Lightweight integrity check for records existence.
    print(f"Loaded {len(records)} canonical records from {records_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus manifest v1")
    parser.add_argument("--dedup-report", type=Path, default=DEFAULT_DEDUP_REPORT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--attestation-out", type=Path)
    args = parser.parse_args()

    raise SystemExit(
        run(
            dedup_report_path=args.dedup_report,
            records_path=args.records,
            manifest_out=args.manifest_out,
            baseline_path=args.baseline,
            write_baseline=args.write_baseline,
            attestation_path=args.attestation_out,
        )
    )


if __name__ == "__main__":
    main()
