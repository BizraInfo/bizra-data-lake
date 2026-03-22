from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.ops import canonical_spearpoint_v1 as spearpoint


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_canonical_spearpoint_v1_proves_s2_to_s1_replay(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "CANONICAL_SPEARPOINT_V1"
    shutil.copytree(Path("artifacts/CANONICAL_SPEARPOINT_V1"), artifact_dir)
    state_dir = tmp_path / "runtime_state"

    report = spearpoint.run_canonical_spearpoint_v1(
        artifact_dir=artifact_dir,
        state_dir=state_dir,
    )

    pre_state = _load(artifact_dir / "pre_state.json")
    run1 = _load(artifact_dir / "run1_receipt.json")
    reward = _load(artifact_dir / "reward_calc.json")
    state_delta = _load(artifact_dir / "state_delta.json")
    run2 = _load(artifact_dir / "run2_receipt.json")
    chain = _load(artifact_dir / "chain_verification.json")
    reflex_store = _load(state_dir / "node0" / "reflexes.json")

    assert pre_state["artifact_status"] == spearpoint.ARTIFACT_STATUS_POPULATED
    assert (
        pre_state["pre_state"]["persistence_contract"]["runtime_state_key"]
        == f"canonical_spearpoint_v1.reflex_route_preference.{state_delta['state_delta_result']['mission_pattern_hash']}"
    )
    assert pre_state["pre_state"]["observed_state"]["reflex_available"] is True
    assert (
        pre_state["pre_state"]["observed_state"]["reflex_cache_state"]["hit"] is False
    )

    assert run1["run1_receipt"]["authority_path"] == spearpoint.AUTHORITY_PATH
    assert run1["run1_receipt"]["selected_route"] == "deliberative_s2"
    assert run1["run1_receipt"]["reflex_used"] is False
    assert run1["run1_receipt"]["verified_success"] is True
    assert run1["run1_receipt"]["policy_compliant"] is True

    assert (
        reward["reward_result"]["reward"]
        >= reward["reward_contract"]["reward_threshold_for_adjustment"]
    )
    assert reward["reward_result"]["reward_verified"] is True

    assert state_delta["state_delta_result"]["applied"] is True
    assert state_delta["state_delta_result"]["new_value"] == "prefer_reflex"
    assert state_delta["state_delta_result"]["state_store_write_verified"] is True
    assert (
        state_delta["state_delta_result"]["mission_pattern_hash"]
        in reflex_store["entries"]
    )

    assert run2["run2_receipt"]["authority_path"] == spearpoint.AUTHORITY_PATH
    assert run2["run2_receipt"]["selected_route"] == "reflex_s1"
    assert run2["run2_receipt"]["reflex_used"] is True
    assert run2["run2_receipt"]["replay_effect_visible"] is True
    assert (
        run2["run2_receipt"]["prev_receipt_hash"]
        == run1["run1_receipt"]["receipt_hash"]
    )
    assert (
        run2["run2_receipt"]["behavioral_delta"]["current_value"]
        == state_delta["state_delta_result"]["new_value"]
    )

    assert report["verification_result"]["canonical_status_achieved"] is True
    assert chain["verification_result"]["canonical_status_achieved"] is True
    assert chain["verification_result"]["hash_link_valid"] is True
    assert chain["verification_result"]["state_delta_reflected_in_run2"] is True
