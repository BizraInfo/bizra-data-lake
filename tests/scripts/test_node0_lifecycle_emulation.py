from __future__ import annotations

from pathlib import Path

import pytest

from scripts.node0_lifecycle_emulation import EmulationConfig, run_lifecycle_emulation


def test_lifecycle_emulation_reaches_flourishing_state(tmp_path: Path) -> None:
    result = run_lifecycle_emulation(state_dir=tmp_path / "run1")
    summary = result["summary"]

    assert summary["final_state"] == "FLOURISHING"
    assert summary["actions_total"] >= 6
    assert summary["speedup_system1_vs_system2"] >= 8.0
    assert summary["ledger_chain_valid"] is True
    assert summary["avg_ihsan"] >= 0.75
    assert summary["signed_receipts"] is True


def test_lifecycle_emulation_appends_to_existing_receipt_chain(tmp_path: Path) -> None:
    state_dir = tmp_path / "shared_state"
    first = run_lifecycle_emulation(state_dir=state_dir)
    second = run_lifecycle_emulation(state_dir=state_dir)

    assert second["summary"]["ledger_height"] > first["summary"]["ledger_height"]
    assert second["summary"]["ledger_chain_valid"] is True


def test_lifecycle_emulation_impt_balance_stays_positive(tmp_path: Path) -> None:
    result = run_lifecycle_emulation(state_dir=tmp_path / "run2")
    summary = result["summary"]
    events = result["events"]

    myelination = [e for e in events if e.get("phase") == "myelination"]
    assert myelination
    assert myelination[-1]["compiled"] is True
    assert summary["impt_balance"] > 0.0


def test_lifecycle_emulation_strict_signing_requires_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", raising=False)
    monkeypatch.delenv("BIZRA_RECEIPT_PUBLIC_KEY_HEX", raising=False)
    with pytest.raises(RuntimeError, match="Strict signing enabled"):
        run_lifecycle_emulation(
            state_dir=tmp_path / "strict_missing",
            config=EmulationConfig(strict_signing=True),
        )
