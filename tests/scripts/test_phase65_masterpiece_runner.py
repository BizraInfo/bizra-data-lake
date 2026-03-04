from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ops.phase65_masterpiece_runner import run_phase65_masterpiece


def test_phase65_masterpiece_runner_emits_artifacts_and_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "BIZRA_RECEIPT_PRIVATE_KEY_HEX",
        "1111111111111111111111111111111111111111111111111111111111111111",
    )
    monkeypatch.delenv("BIZRA_RECEIPT_PUBLIC_KEY_HEX", raising=False)

    out_dir = tmp_path / "phase65"
    result = run_phase65_masterpiece(
        state_dir=out_dir / "state",
        out_dir=out_dir,
        config_path=Path("config/phase65_masterpiece_roadmap.yaml"),
        strict_signing=True,
    )

    assert result["gate_passed"] is True
    assert Path(result["summary_path"]).exists()
    assert Path(result["gate_report_path"]).exists()
    assert Path(result["kpi_json_path"]).exists()
    assert Path(result["kpi_md_path"]).exists()
    assert Path(result["alpha_packet_json_path"]).exists()
    assert Path(result["alpha_packet_md_path"]).exists()
    assert result["launch_decision"] in {"GO", "CONDITIONAL_GO"}

    gate = json.loads(Path(result["gate_report_path"]).read_text(encoding="utf-8"))
    assert gate["gate_passed"] is True

    kpi = json.loads(Path(result["kpi_json_path"]).read_text(encoding="utf-8"))
    assert kpi["signed_receipts"] is True
    assert kpi["tier"] in {"elite-operational", "operational"}

    alpha = json.loads(Path(result["alpha_packet_json_path"]).read_text(encoding="utf-8"))
    assert alpha["decision"] == "CONDITIONAL_GO"


def test_phase65_masterpiece_runner_strict_signing_requires_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", raising=False)
    monkeypatch.delenv("BIZRA_RECEIPT_PUBLIC_KEY_HEX", raising=False)

    with pytest.raises(RuntimeError, match="Strict signing enabled"):
        run_phase65_masterpiece(
            state_dir=tmp_path / "state",
            out_dir=tmp_path / "out",
            config_path=Path("config/phase65_masterpiece_roadmap.yaml"),
            strict_signing=True,
        )
