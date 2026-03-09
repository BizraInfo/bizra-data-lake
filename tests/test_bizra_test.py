from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import bizra_test


def test_parse_pytest_output_counts() -> None:
    output = "10105 passed, 28 failed, 12 skipped in 262.90s"
    result = bizra_test.parse_pytest_output(output)
    assert result == {
        "passed": 10105,
        "failed": 28,
        "skipped": 12,
        "total": 10145,
        "duration": 262.9,
    }


def test_parse_coverage_total_line() -> None:
    output = "TOTAL  1234  567  54%"
    assert bizra_test.parse_coverage(output) == 54.0


def test_get_changed_files_since_filters_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bizra_test,
        "run_cmd",
        lambda command: SimpleNamespace(returncode=0, stdout="core/a.py\n\ncore/b.py\n"),
    )
    assert bizra_test.get_changed_files_since("v0.80.0") == {"core/a.py", "core/b.py"}


def test_get_affected_tests_includes_direct_test_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = tmp_path / "tests" / "core"
    tests_dir.mkdir(parents=True)
    direct = tests_dir / "test_direct.py"
    direct.write_text("def test_direct():\n    assert True\n", encoding="utf-8")

    monkeypatch.setattr(bizra_test, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        bizra_test,
        "run_cmd",
        lambda command: SimpleNamespace(returncode=0, stdout=""),
    )

    affected = bizra_test.get_affected_tests({"tests/core/test_direct.py"})
    assert affected == {"tests/core/test_direct.py"}


def test_get_affected_tests_finds_import_dependents(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "tests" / "core" / "token" / "test_bloom.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("from core.token import bloom\n", encoding="utf-8")

    monkeypatch.setattr(bizra_test, "PROJECT_ROOT", tmp_path)

    def fake_run_cmd(command: str):
        assert "core.token.bloom" in command
        return SimpleNamespace(returncode=0, stdout=str(test_file))

    monkeypatch.setattr(bizra_test, "run_cmd", fake_run_cmd)

    affected = bizra_test.get_affected_tests({"core/token/bloom.py"})
    assert affected == {str(test_file)}


def test_run_delta_without_lock_falls_back_to_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bizra_test, "get_latest_lock", lambda: None)
    monkeypatch.setattr(bizra_test, "run_smoke", lambda: 0)
    assert bizra_test.run_delta() == 0


def test_run_delta_with_no_changes_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = bizra_test.TestLockReceipt(
        version="0.80.0",
        git_commit="abc",
        git_tag="v0.80.0",
        timestamp="2026-03-09T00:00:00+00:00",
        total_tests=10,
        passed=10,
        failed=0,
        skipped=0,
        duration_seconds=1.0,
        coverage_percent=40.0,
        coverage_floor=40.0,
    )
    monkeypatch.setattr(bizra_test, "get_latest_lock", lambda: lock)
    monkeypatch.setattr(bizra_test, "get_changed_files_since", lambda tag: set())
    assert bizra_test.run_delta() == 0


def test_run_lock_writes_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    lock_dir = tmp_path / "sovereign_state" / "test_locks"
    current = lock_dir / "current.json"

    monkeypatch.setattr(bizra_test, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(bizra_test, "LOCK_CURRENT", current)
    monkeypatch.setattr(
        bizra_test,
        "run_full",
        lambda with_coverage=True: (
            0,
            {"total": 100, "passed": 100, "failed": 0, "skipped": 0, "duration": 12.5},
            41.5,
        ),
    )
    monkeypatch.setattr(bizra_test, "get_latest_lock", lambda: None)
    monkeypatch.setattr(bizra_test, "get_git_commit", lambda: "deadbeef" * 5)
    monkeypatch.setattr(bizra_test, "hash_constants_file", lambda: "c" * 64)
    monkeypatch.setattr(
        bizra_test,
        "run_cmd",
        lambda command, timeout=600: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    rc = bizra_test.run_lock()

    assert rc == 0
    receipt_path = lock_dir / "v0.80.0.json"
    assert receipt_path.exists()
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["passed"] == 100
    assert data["coverage_floor"] == 41.5
    assert current.exists()


def test_run_lock_rejects_coverage_regression(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    lock_dir = tmp_path / "sovereign_state" / "test_locks"
    current = lock_dir / "current.json"
    prev = bizra_test.TestLockReceipt(
        version="0.80.0",
        git_commit="abc",
        git_tag="v0.80.0",
        timestamp="2026-03-09T00:00:00+00:00",
        total_tests=100,
        passed=100,
        failed=0,
        skipped=0,
        duration_seconds=1.0,
        coverage_percent=42.0,
        coverage_floor=42.0,
        prev_lock_hash="GENESIS",
        lock_hash="f" * 64,
    )

    monkeypatch.setattr(bizra_test, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(bizra_test, "LOCK_CURRENT", current)
    monkeypatch.setattr(
        bizra_test,
        "run_full",
        lambda with_coverage=True: (
            0,
            {"total": 100, "passed": 100, "failed": 0, "skipped": 0, "duration": 12.5},
            41.0,
        ),
    )
    monkeypatch.setattr(bizra_test, "get_latest_lock", lambda: prev)

    assert bizra_test.run_lock() == 1


def test_show_status_without_lock_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bizra_test, "get_latest_lock", lambda: None)
    assert bizra_test.show_status() == 0
