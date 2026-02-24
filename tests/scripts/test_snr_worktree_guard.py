from __future__ import annotations

from scripts.ops.snr_worktree_guard import GitEntry, _parse_status_line, classify


def test_parse_status_line_handles_quoted_paths() -> None:
    entry = _parse_status_line('?? "my cowork output/"')
    assert entry is not None
    assert entry.status == "??"
    assert entry.path == "my cowork output/"


def test_classify_with_policy_groups() -> None:
    policy = {
        "groups": {
            "keep_track": [{"pattern": "core/**", "reason": "tracked"}],
            "keep_untracked": [{"pattern": ".agentdb/**", "reason": "local"}],
            "archive": [{"pattern": "artifacts/normalizers/**", "reason": "archive"}],
        }
    }
    entries = [
        GitEntry(status=" M", path="core/genesis/orchestrator.py"),
        GitEntry(status="??", path=".agentdb/reasoningbank.db"),
        GitEntry(status="??", path="artifacts/normalizers/ingest_payload.jsonl"),
        GitEntry(status="??", path="unmatched/path.tmp"),
    ]
    decisions = classify(entries, policy)
    by_path = {d.path: d for d in decisions}

    assert by_path["core/genesis/orchestrator.py"].recommendation == "KEEP_TRACK"
    assert by_path[".agentdb/reasoningbank.db"].recommendation == "KEEP_UNTRACKED"
    assert (
        by_path["artifacts/normalizers/ingest_payload.jsonl"].recommendation
        == "ARCHIVE"
    )
    assert by_path["unmatched/path.tmp"].recommendation == "REVIEW"
