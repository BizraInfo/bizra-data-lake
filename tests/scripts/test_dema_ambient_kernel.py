"""Dema Ambient Kernel v0.1 contract tests.

Covers:
  1. ProfileStore round-trip.
  2. DailyLog append + read_today.
  3. MissionStateMachine get/update.
  4. DemaReceipt validation + ReceiptWriter persistence.
  5. dema_onboarding.py --init writes profile + receipt + log entry.
  6. dema_status.py --json shape.
  7. dema_daemon.py --once writes one tick + receipt + log entry.
  8. dema_dream.py --read-only --max-seconds N produces candidate notes
     without auto-promotion.
  9. Receipt non-claims (touched/not_touched) are honored.
  10. No raw secrets / private content end up in committed paths
      (everything Dema writes lands under sovereign_state/dema, gitignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from core.dema import (  # noqa: E402
    DailyLog,
    DailyLogEntry,
    DemaProfile,
    DemaReceipt,
    FourStateModel,
    MissionStateMachine,
    ProfileStore,
    ReceiptWriter,
)

# ── Core module tests ────────────────────────────────────────────────


def test_profile_roundtrip(tmp_path: Path):
    store = ProfileStore(tmp_path)
    assert store.load() is None

    profile = DemaProfile(
        preferred_name="Mumu",
        mother_language="ar",
        work_language="en",
        memory_consent="local",
    )
    store.save(profile)

    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.preferred_name == "Mumu"
    assert reloaded.mother_language == "ar"
    assert reloaded.memory_consent == "local"
    assert reloaded.schema_version == "0.1.0"


def test_profile_invalid_consent_rejected():
    with pytest.raises(ValueError):
        DemaProfile(
            preferred_name="x",
            mother_language="en",
            work_language="en",
            memory_consent="bogus",
        )


def test_profile_init_from_env_or_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMA_PREFERRED_NAME", "TestUser")
    monkeypatch.setenv("DEMA_MOTHER_LANGUAGE", "ar")
    monkeypatch.delenv("DEMA_WORK_LANGUAGE", raising=False)

    store = ProfileStore(tmp_path)
    profile = store.init_from_env_or_defaults()
    assert profile.preferred_name == "TestUser"
    assert profile.mother_language == "ar"
    assert profile.work_language == "en"  # default fallback


def test_daily_log_append_and_read_today(tmp_path: Path):
    log = DailyLog(tmp_path)
    assert log.read_today() == []

    log.append(
        DailyLogEntry(
            timestamp="2026-04-27T12:00:00Z",
            kind="tick",
            summary="first tick",
            receipt_id="abc",
        )
    )
    log.append(
        DailyLogEntry(
            timestamp="2026-04-27T12:05:00Z",
            kind="onboarding",
            summary="profile init",
        )
    )

    today = log.read_today()
    assert len(today) == 2
    assert {e.kind for e in today} == {"tick", "onboarding"}


def test_daily_log_invalid_kind_rejected():
    with pytest.raises(ValueError):
        DailyLogEntry(
            timestamp="2026-04-27T12:00:00Z",
            kind="bogus_kind",
            summary="nope",
        )


def test_mission_state_default_is_unknown_and_not_actionable(tmp_path: Path):
    machine = MissionStateMachine(tmp_path)
    state = machine.get()
    assert state.truth_label == "UNKNOWN"
    assert state.is_actionable() is False


def test_mission_state_update_persists(tmp_path: Path):
    machine = MissionStateMachine(tmp_path)
    updated = machine.update(
        current="single-node alive",
        ideal="two-device handshake closed",
        gap="node1_kit not yet built",
        next_admissible_action="build scripts/pilot/node1_kit.py",
        truth_label="PLANNED",
    )
    assert isinstance(updated, FourStateModel)
    assert updated.is_actionable()

    reloaded = MissionStateMachine(tmp_path).get()
    assert reloaded.next_admissible_action.startswith("build scripts/pilot")
    assert reloaded.truth_label == "PLANNED"


def test_receipt_validates_truth_label():
    with pytest.raises(ValueError):
        DemaReceipt(
            action="dema.x",
            truth_label="GOSPEL",
            touched_paths=[],
        )


def test_receipt_writer_persists_and_includes_digests(tmp_path: Path):
    receipt = DemaReceipt(
        action="dema.test.write",
        truth_label="MEASURED",
        touched_paths=["sovereign_state/dema/test"],
        not_touched_paths=["network", "MEMORY.md"],
        payload={"k": "v"},
    )
    rid, path = ReceiptWriter(tmp_path).write(receipt)

    sealed = json.loads(path.read_text(encoding="utf-8"))
    assert sealed["receipt_id"] == rid
    assert sealed["payload_digest"] != rid
    assert sealed["truth_label"] == "MEASURED"
    assert "MEMORY.md" in sealed["not_touched_paths"]
    assert "network" in sealed["not_touched_paths"]


# ── Script CLI tests ─────────────────────────────────────────────────


def _run(script: str, *args: str, root: Path) -> dict:
    """Run a Dema script with --root <tmp> and return parsed JSON."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dema" / script),
        *args,
        "--root",
        str(root),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
        check=True,
    )
    return json.loads(res.stdout)


def test_onboarding_init_writes_profile_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMA_PREFERRED_NAME", "Mumu-Test")
    monkeypatch.setenv("DEMA_MOTHER_LANGUAGE", "ar")
    monkeypatch.setenv("DEMA_MEMORY_CONSENT", "local")

    out = _run("dema_onboarding.py", "--init", root=tmp_path)
    assert out["ok"] is True
    assert out["truth_label"] == "MEASURED"

    profile_path = Path(out["profile_path"])
    receipt_path = Path(out["receipt_path"])
    assert profile_path.exists()
    assert receipt_path.exists()

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["preferred_name"] == "Mumu-Test"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["action"] == "dema.onboarding.init"
    assert "MEMORY.md" in receipt["not_touched_paths"]


def test_status_emits_expected_shape(tmp_path):
    out = _run("dema_status.py", "--json", root=tmp_path)
    assert out["kind"] == "dema_status"
    assert out["profile_present"] is False
    assert "mission_state" in out
    assert out["actionable"] is False


def test_daemon_once_writes_log_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMA_PREFERRED_NAME", "Mumu-T")
    _run("dema_onboarding.py", "--init", root=tmp_path)

    out = _run("dema_daemon.py", "--once", root=tmp_path)
    assert out["ok"] is True
    assert out["kind"] == "dema_daemon_tick"
    assert Path(out["receipt_path"]).exists()
    assert Path(out["log_path"]).exists()

    # Re-check status — log_today_count should now be ≥ 2 (onboarding + tick)
    status = _run("dema_status.py", "--json", root=tmp_path)
    assert status["log_today_count"] >= 2
    assert "tick" in status["log_today_kinds"]


def test_dream_read_only_completes_under_budget(tmp_path):
    out = _run(
        "dema_dream.py",
        "--read-only",
        "--max-seconds",
        "5",
        root=tmp_path,
    )
    assert out["ok"] is True
    assert out["promoted_to_long_term"] is False
    assert out["budget_hit"] is False
    assert out["candidate_notes_count"] >= 5  # all 5 phases hit
    assert Path(out["candidate_notes_path"]).exists()
    assert Path(out["summary_path"]).exists()


def test_dream_respects_extreme_low_budget(tmp_path):
    """Budget = 0.0 must still complete safely (returns budget_hit=True)."""
    out = _run(
        "dema_dream.py",
        "--read-only",
        "--max-seconds",
        "0.0",
        root=tmp_path,
    )
    assert out["ok"] is True
    assert out["promoted_to_long_term"] is False
    assert out["budget_hit"] is True
    assert out["background_suggestion"] is True


# ── Boundary tests ──────────────────────────────────────────────────


def test_dema_writes_only_under_sovereign_state(tmp_path, monkeypatch):
    """All paths returned by Dema scripts must live under the supplied root."""
    monkeypatch.setenv("DEMA_PREFERRED_NAME", "BoundsCheck")
    onboard = _run("dema_onboarding.py", "--init", root=tmp_path)
    daemon = _run("dema_daemon.py", "--once", root=tmp_path)
    dream = _run("dema_dream.py", "--read-only", "--max-seconds", "5", root=tmp_path)

    written_paths = [
        onboard["profile_path"],
        onboard["receipt_path"],
        daemon["receipt_path"],
        daemon["log_path"],
        dream["candidate_notes_path"],
        dream["summary_path"],
        dream["receipt_path"],
    ]
    root_str = str(tmp_path.resolve())
    for p in written_paths:
        assert (
            Path(p).resolve().is_relative_to(tmp_path.resolve())
        ), f"path {p} escaped sandbox root {root_str}"


def test_no_canon_or_memory_md_is_listed_in_touched_paths(tmp_path):
    """Canonical surfaces must appear in not_touched_paths, not touched."""
    onboard = _run("dema_onboarding.py", "--init", root=tmp_path)
    receipt = json.loads(Path(onboard["receipt_path"]).read_text(encoding="utf-8"))
    assert "MEMORY.md" not in receipt["touched_paths"]
    assert "MEMORY.md" in receipt["not_touched_paths"]
    assert all("docs/canon" not in p for p in receipt["touched_paths"])
