from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_unified_corpus import SCHEMA_COLUMNS, build_unified_corpus


def test_build_unified_corpus_writes_parquet_and_indexes(tmp_path: Path) -> None:
    intake = tmp_path / "intake"
    intake.mkdir(parents=True)

    # ChatGPT-style JSON (single user turn).
    chatgpt_payload = {
        "conversation_id": "cg-1",
        "mapping": {
            "n1": {
                "message": {
                    "id": "cg-u1",
                    "author": {"role": "user"},
                    "create_time": 1700000000,
                    "content": {"parts": ["I prefer Rust architecture."]},
                }
            }
        },
    }
    (intake / "chatgpt.json").write_text(
        json.dumps(chatgpt_payload),
        encoding="utf-8",
    )

    # OpenAI API-style JSONL (duplicate user + unique assistant).
    openai_line = {
        "id": "oa-1",
        "request": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "I prefer Rust architecture."}],
        },
        "response": {
            "created": 1700000001,
            "choices": [
                {"message": {"role": "assistant", "content": "Use staged rollout."}}
            ],
        },
    }
    (intake / "openai_api.jsonl").write_text(
        json.dumps(openai_line) + "\n",
        encoding="utf-8",
    )

    out_parquet = tmp_path / "out" / "conversations_unified.parquet"
    out_dir = tmp_path / "out"

    report = build_unified_corpus(
        paths=[intake],
        out_parquet=out_parquet,
        out_dir=out_dir,
    )

    assert out_parquet.exists()
    assert (out_dir / "platform_index.json").exists()
    assert (out_dir / "timeline_index.json").exists()
    assert (out_dir / "dedup_manifest.json").exists()
    assert (out_dir / "ingestion_report.json").exists()
    assert (out_dir / "parse_failures.json").exists()
    assert (out_parquet.with_suffix(".parquet.receipt.json")).exists()

    frame = pd.read_parquet(out_parquet)
    assert tuple(frame.columns) == SCHEMA_COLUMNS
    # Raw: 3 turns (cg user, oa user, oa assistant). Unified: 2 after dedup user turn.
    assert report["total_turns_raw"] == 3
    assert report["total_turns_unified"] == 2
    assert report["duplicate_turns_removed"] == 1
    assert report["parse_failure_count"] == 0

    dedup_manifest = json.loads(
        (out_dir / "dedup_manifest.json").read_text(encoding="utf-8")
    )
    assert any(item.get("variant_count", 0) >= 2 for item in dedup_manifest.values())

    parquet_receipt = json.loads(
        (out_parquet.with_suffix(".parquet.receipt.json")).read_text(encoding="utf-8")
    )
    assert parquet_receipt["artifact_path"] == str(out_parquet)
    assert (
        parquet_receipt["sha256"]
        == hashlib.sha256(out_parquet.read_bytes()).hexdigest()
    )


def test_build_unified_corpus_generates_expected_indexes(tmp_path: Path) -> None:
    intake = tmp_path / "intake"
    intake.mkdir(parents=True)

    payload = {
        "id": "gr-1",
        "model": "grok-2-latest",
        "messages": [
            {
                "id": "m1",
                "role": "user",
                "timestamp": "2026-02-21T10:00:00Z",
                "content": "Plan next step.",
            },
            {
                "id": "m2",
                "role": "assistant",
                "timestamp": "2026-02-21T10:00:01Z",
                "content": "Use strict gates.",
            },
        ],
    }
    (intake / "grok.json").write_text(json.dumps(payload), encoding="utf-8")

    out_parquet = tmp_path / "out" / "conversations_unified.parquet"
    out_dir = tmp_path / "out"
    build_unified_corpus(paths=[intake], out_parquet=out_parquet, out_dir=out_dir)

    platform_index = json.loads(
        (out_dir / "platform_index.json").read_text(encoding="utf-8")
    )
    timeline_index = json.loads(
        (out_dir / "timeline_index.json").read_text(encoding="utf-8")
    )

    assert "grok" in platform_index
    assert platform_index["grok"] == ["gr-1"]
    assert "2026-02" in timeline_index
    assert len(timeline_index["2026-02"]) == 2


def test_build_unified_corpus_writes_parse_failure_receipts(tmp_path: Path) -> None:
    intake = tmp_path / "intake"
    intake.mkdir(parents=True)

    (intake / "broken.json").write_text("{ this is invalid json", encoding="utf-8")
    (intake / "ok.json").write_text(
        json.dumps(
            {
                "conversation_id": "cg-1",
                "mapping": {
                    "n1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["hello"]},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    out_parquet = tmp_path / "out" / "conversations_unified.parquet"
    out_dir = tmp_path / "out"
    report = build_unified_corpus(
        paths=[intake], out_parquet=out_parquet, out_dir=out_dir
    )

    assert report["parse_failure_count"] == 1
    assert report["parse_failure_reasons"]["JSON_DECODE_ERROR"] == 1

    parse_failures = json.loads(
        (out_dir / "parse_failures.json").read_text(encoding="utf-8")
    )
    assert len(parse_failures) == 1
    assert parse_failures[0]["reason_code"] == "JSON_DECODE_ERROR"
    assert parse_failures[0]["status"] == "failed"
    assert parse_failures[0]["path"].endswith("broken.json")
