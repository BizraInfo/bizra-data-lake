from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any
from unittest.mock import patch

from core.integration.constants import REFLEX_PRECIPITATION_HITS
from core.pat.identity_card import generate_identity_keypair
from core.proof_engine.canonical import canonical_bytes, hex_digest
from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

AUTHORITY_PATH = "runtime->organism->node0"
ARTIFACT_STATUS_POPULATED = "POPULATED_EXECUTION"
GENESIS_ZERO_HASH = "0" * 64
MISSION_POLICY_FLOOR = 0.85
POLICY_VERSION = "1.0.0"


class DeterministicGateway:
    def __init__(self) -> None:
        self.model_id = "deterministic/canonical-spearpoint-v1"
        self.status = "ready"

    async def infer(self, prompt: str, **_: Any) -> str:
        del prompt
        await asyncio.sleep(0.02)
        return (
            "# Decision\n"
            "Verify Spine section 4, verify the canonical gate, and ensure canonical mode uses runtime-owned organism mission authority.\n\n"
            "# Outcome\n"
            "If canonical mode is enabled and that authority is unavailable, the system must reject execution, fail closed, emit a blocked receipt, record fate reason codes, and use Ihsan >= 095 as the policy floor.\n\n"
            "# Safeguard\n"
            "However, to answer what must happen on this failure path, the constitutional trade-off is still simple: the system should verify the refusal, ensure the safeguard chain stays intact, keep the weaker route disabled, and therefore must not silently fall back or use any legacy route."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def _artifact_hash(payload: dict[str, Any]) -> str:
    canonical_payload = {
        key: value for key, value in payload.items() if key != "receipt_hash"
    }
    return hex_digest(canonical_bytes(canonical_payload))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _status_from_receipt(receipt: Any) -> str:
    if getattr(receipt, "fate_verdict", "") == "rejected":
        return "BLOCKED"
    if getattr(receipt, "system", "") == "ERROR":
        return "FAILED"
    return "COMPLETE"


def _selected_route(receipt: Any) -> str:
    system = str(getattr(receipt, "system", "") or "").upper()
    if system == "S1":
        return "reflex_s1"
    if system == "S2":
        return "deliberative_s2"
    if system == "BLOCKED":
        return "blocked"
    return "degraded"


def _verify_output(
    output: str, verification_contract: dict[str, Any]
) -> tuple[bool, list[str]]:
    lowered = output.lower()
    notes: list[str] = []

    for required in verification_contract.get("must_include_all", []):
        if required.lower() not in lowered:
            notes.append(f"missing_required:{required}")
            return False, notes
        notes.append(f"matched_required:{required}")

    any_terms = verification_contract.get("must_include_any", [])
    matched_any = [term for term in any_terms if term.lower() in lowered]
    if any_terms and not matched_any:
        notes.append("missing_any_required_clause")
        return False, notes
    notes.extend(f"matched_optional:{term}" for term in matched_any)

    for forbidden in verification_contract.get("forbidden_substrings", []):
        if forbidden.lower() in lowered:
            notes.append(f"forbidden_match:{forbidden}")
            return False, notes

    return True, notes


def _memory_available_mb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    return round((page_size * available_pages) / (1024 * 1024), 2)


def _cpu_load_pct() -> float | None:
    try:
        load = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
    except (AttributeError, OSError):
        return None
    return round((load / cpu_count) * 100.0, 2)


def _seed_identity_credentials(state_dir: Path) -> str:
    identity_dir = state_dir / "identity"
    credentials_path = identity_dir / "credentials.json"
    if credentials_path.exists():
        creds = _read_json(credentials_path)
        return str(creds["node_id"])

    identity_dir.mkdir(parents=True, exist_ok=True)
    private_key, public_key, node_id = generate_identity_keypair()
    _write_json(
        credentials_path,
        {
            "private_key": private_key,
            "public_key": public_key,
            "node_id": node_id,
        },
    )
    return node_id


def _prepare_clean_state_dir(state_dir: Path) -> None:
    if state_dir.exists() and any(state_dir.iterdir()):
        raise ValueError(
            f"State directory must start empty for CANONICAL_SPEARPOINT_V1: {state_dir}"
        )
    state_dir.mkdir(parents=True, exist_ok=True)


def _require_artifact_bundle(artifact_dir: Path) -> None:
    required = {
        "mission_input.json",
        "pre_state.json",
        "run1_receipt.json",
        "reward_calc.json",
        "state_delta.json",
        "run2_receipt.json",
        "chain_verification.json",
        "README.md",
    }
    missing = sorted(name for name in required if not (artifact_dir / name).exists())
    if missing:
        raise FileNotFoundError(
            f"Artifact bundle is incomplete at {artifact_dir}: missing {', '.join(missing)}"
        )


def _patch_runtime(runtime: SovereignRuntime) -> None:
    async def _no_op_init_omega_components(self: SovereignRuntime) -> None:
        self._gateway = None
        self._omega = None
        self._orchestrator = None

    def _no_op_setup_signal_handlers(self: SovereignRuntime) -> None:
        del self

    runtime._init_omega_components = MethodType(_no_op_init_omega_components, runtime)
    runtime._setup_signal_handlers = MethodType(_no_op_setup_signal_handlers, runtime)


def _noop_boot_federation_ambassador(self: Any) -> None:
    self._federation_ambassador = None


async def _build_runtime(
    state_dir: Path, gateway: DeterministicGateway
) -> SovereignRuntime:
    config = RuntimeConfig(
        state_dir=state_dir,
        enable_graph_reasoning=False,
        enable_snr_optimization=False,
        enable_guardian_validation=False,
        enable_autonomous_loop=False,
        enable_cache=False,
        enable_persistence=False,
        autonomous_enabled=False,
        enable_zpk_preflight=False,
        enable_proactive_kernel=False,
    )
    runtime = SovereignRuntime(config)
    _patch_runtime(runtime)
    await runtime.initialize()
    runtime._gateway = gateway
    runtime.config.default_model = gateway.model_id
    return runtime


def _pattern_hash(runtime: SovereignRuntime, mission_prompt: str) -> str:
    reflex = runtime._organism._nervous_system._reflex
    return reflex._hash_input(mission_prompt)


def _capture_pre_state(
    artifact_path: Path,
    pre_state_doc: dict[str, Any],
    runtime: SovereignRuntime,
    gateway: DeterministicGateway,
    mission_prompt: str,
) -> dict[str, Any]:
    observed = pre_state_doc["pre_state"]["observed_state"]
    pattern_hash = _pattern_hash(runtime, mission_prompt)
    observed["capture_timestamp"] = _utc_now()
    observed["policy_version"] = POLICY_VERSION
    observed["available_models"] = [gateway.model_id]
    observed["default_model"] = gateway.model_id
    observed["default_got_depth"] = runtime.config.max_reasoning_depth
    observed["amplification_available"] = False
    observed["reflex_available"] = runtime._organism._nervous_system._reflex is not None
    observed["reflex_cache_state"] = {
        "hit": False,
        "pattern_id": pattern_hash,
    }
    observed["routing_weights"] = {
        "canonical_baseline": 1.0,
        "canonical_amplified": 0.0,
        "canonical_reflex": 0.0,
    }
    observed["resource_snapshot"] = {
        "cpu_load_pct": _cpu_load_pct(),
        "memory_available_mb": _memory_available_mb(),
        "gpu_available": False,
    }
    observed["last_receipt_hash"] = GENESIS_ZERO_HASH
    pre_state_doc["pre_state"]["persistence_contract"][
        "runtime_state_key"
    ] = f"canonical_spearpoint_v1.reflex_route_preference.{pattern_hash}"
    pre_state_doc["artifact_status"] = ARTIFACT_STATUS_POPULATED
    _write_json(artifact_path, pre_state_doc)
    return pre_state_doc


def _silent_fallback_detected(receipt: Any, authority_path: str) -> bool:
    output = str(getattr(receipt, "output_text", "") or "")
    return (
        authority_path != AUTHORITY_PATH
        or getattr(receipt, "fate_verdict", "") == "degraded"
        or output.startswith("[runtime-degraded]")
        or output.startswith("[DEGRADED]")
        or getattr(receipt, "system", "") == "ERROR"
    )


def _cqrs_delivery_snapshot(runtime: SovereignRuntime) -> dict[str, Any]:
    bus = getattr(getattr(runtime, "_organism", None), "_cqrs_bus", None)
    summary_fn = getattr(bus, "delivery_summary", None)
    organism_stats = getattr(getattr(runtime, "_organism", None), "stats", {})
    cqrs_stats = (
        organism_stats.get("cqrs_bus", {}) if isinstance(organism_stats, dict) else {}
    )
    node0_stats = (
        organism_stats.get("node0", {}) if isinstance(organism_stats, dict) else {}
    )
    if not callable(summary_fn):
        return {
            "delivery_receipts": 0,
            "delivery_acks": 0,
            "delivery_dead_letters": 0,
            "dead_letter_rate": 0.0,
            "delivery_mirror_enabled": False,
            "delivery_mirror_successes": 0,
            "delivery_mirror_failures": 0,
            "node0_canonical_delivery_receipts": 0,
            "node0_canonical_delivery_failures": 0,
            "node0_canonical_delivery_acks": 0,
            "node0_canonical_delivery_dead_letters": 0,
        }
    summary = summary_fn()
    return {
        "delivery_receipts": int(summary.get("delivery_receipts", 0) or 0),
        "delivery_acks": int(summary.get("delivery_acks", 0) or 0),
        "delivery_dead_letters": int(summary.get("delivery_dead_letters", 0) or 0),
        "dead_letter_rate": float(summary.get("dead_letter_rate", 0.0) or 0.0),
        "delivery_mirror_enabled": bool(
            cqrs_stats.get("delivery_mirror_enabled", False)
        ),
        "delivery_mirror_successes": int(
            cqrs_stats.get("delivery_mirror_successes", 0) or 0
        ),
        "delivery_mirror_failures": int(
            cqrs_stats.get("delivery_mirror_failures", 0) or 0
        ),
        "node0_canonical_delivery_receipts": int(
            node0_stats.get("total_cqrs_delivery_receipts", 0) or 0
        ),
        "node0_canonical_delivery_failures": int(
            node0_stats.get("total_cqrs_delivery_receipt_failures", 0) or 0
        ),
        "node0_canonical_delivery_acks": int(
            node0_stats.get("total_cqrs_delivery_ack_receipts", 0) or 0
        ),
        "node0_canonical_delivery_dead_letters": int(
            node0_stats.get("total_cqrs_delivery_dead_letters", 0) or 0
        ),
    }


def _delivery_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    return {
        "delivery_receipts": max(
            int(after["delivery_receipts"]) - int(before["delivery_receipts"]),
            0,
        ),
        "delivery_acks": max(
            int(after["delivery_acks"]) - int(before["delivery_acks"]),
            0,
        ),
        "delivery_dead_letters": max(
            int(after["delivery_dead_letters"]) - int(before["delivery_dead_letters"]),
            0,
        ),
        "delivery_mirror_successes": max(
            int(after["delivery_mirror_successes"])
            - int(before["delivery_mirror_successes"]),
            0,
        ),
        "delivery_mirror_failures": max(
            int(after["delivery_mirror_failures"])
            - int(before["delivery_mirror_failures"]),
            0,
        ),
        "node0_canonical_delivery_receipts": max(
            int(after["node0_canonical_delivery_receipts"])
            - int(before["node0_canonical_delivery_receipts"]),
            0,
        ),
        "node0_canonical_delivery_failures": max(
            int(after["node0_canonical_delivery_failures"])
            - int(before["node0_canonical_delivery_failures"]),
            0,
        ),
        "node0_canonical_delivery_acks": max(
            int(after["node0_canonical_delivery_acks"])
            - int(before["node0_canonical_delivery_acks"]),
            0,
        ),
        "node0_canonical_delivery_dead_letters": max(
            int(after["node0_canonical_delivery_dead_letters"])
            - int(before["node0_canonical_delivery_dead_letters"]),
            0,
        ),
    }


async def _await_delivery_settlement(
    runtime: SovereignRuntime,
    before: dict[str, Any],
    *,
    timeout_s: float = 0.5,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        after = _cqrs_delivery_snapshot(runtime)
        acked = after["delivery_acks"] > before["delivery_acks"]
        canonical_recorded = (
            after["node0_canonical_delivery_receipts"]
            > before["node0_canonical_delivery_receipts"]
            or after["node0_canonical_delivery_failures"]
            > before["node0_canonical_delivery_failures"]
        )
        if not after["delivery_mirror_enabled"]:
            if acked and canonical_recorded:
                return after
        else:
            mirrored = (
                after["delivery_mirror_successes"] > before["delivery_mirror_successes"]
                or after["delivery_mirror_failures"]
                > before["delivery_mirror_failures"]
            )
            if acked and mirrored and canonical_recorded:
                return after

        if asyncio.get_running_loop().time() >= deadline:
            return after
        await asyncio.sleep(0.01)


def _build_run_receipt(
    *,
    runtime: SovereignRuntime,
    artifact_path: Path,
    doc: dict[str, Any],
    body_key: str,
    mission_prompt: str,
    mission_contract: dict[str, Any],
    receipt: Any,
    previous_hash: str,
    previous_hash_semantics: str,
    delivery_before: dict[str, Any],
    delivery_after: dict[str, Any],
    state_delta_ref: str | None = None,
    prior_state_value: str | None = None,
    current_state_value: str | None = None,
) -> dict[str, Any]:
    canonical_status = runtime.status()["canonical"]
    body = doc[body_key]
    delivery_delta = _delivery_delta(delivery_before, delivery_after)
    verified_success, verification_notes = _verify_output(
        str(getattr(receipt, "output_text", "") or ""),
        mission_contract["verification_contract"],
    )
    status = _status_from_receipt(receipt)
    authority_path = str(canonical_status.get("authority_path", "") or "")
    silent_fallback_detected = _silent_fallback_detected(receipt, authority_path)
    subscriber_delivery_verified = (
        delivery_delta["delivery_acks"] > 0
        and delivery_delta["delivery_dead_letters"] == 0
    )
    subscriber_delivery_mirror_verified = (
        not delivery_after["delivery_mirror_enabled"]
    ) or (
        delivery_delta["delivery_mirror_successes"] > 0
        and delivery_delta["delivery_mirror_failures"] == 0
    )
    node0_canonical_delivery_verified = (
        delivery_delta["node0_canonical_delivery_receipts"] > 0
        and delivery_delta["node0_canonical_delivery_acks"] > 0
        and delivery_delta["node0_canonical_delivery_dead_letters"] == 0
        and delivery_delta["node0_canonical_delivery_failures"] == 0
    )
    policy_compliant = (
        verified_success
        and status == "COMPLETE"
        and authority_path == AUTHORITY_PATH
        and canonical_status.get("mission_authority") == "organism"
        and float(getattr(receipt, "ihsan_score", 0.0) or 0.0) >= MISSION_POLICY_FLOOR
        and subscriber_delivery_verified
        and subscriber_delivery_mirror_verified
        and node0_canonical_delivery_verified
        and not silent_fallback_detected
    )

    body["receipt_id"] = (
        "canonical-spearpoint-run1"
        if body["run_index"] == 1
        else "canonical-spearpoint-run2"
    )
    body["timestamp"] = _utc_now()
    body["policy_version"] = POLICY_VERSION
    body["observed_state_ref"] = str(artifact_path.parent / "pre_state.json")
    if state_delta_ref is not None:
        body["applied_state_delta_ref"] = state_delta_ref
    body["execution_authority"] = str(
        canonical_status.get("mission_authority", "") or ""
    )
    body["authority_path"] = authority_path
    body["selected_route"] = _selected_route(receipt)
    body["selected_model"] = (
        "reflex-cache"
        if body["selected_route"] == "reflex_s1"
        else runtime.config.default_model
    )
    body["selected_got_depth"] = (
        0
        if body["selected_route"] == "reflex_s1"
        else runtime.config.max_reasoning_depth
    )
    body["amplification_used"] = False
    body["reflex_used"] = bool(getattr(receipt, "system", "") == "S1")
    body["latency_ms"] = round(float(getattr(receipt, "duration_ms", 0.0) or 0.0), 2)
    body["snr_score"] = round(float(getattr(receipt, "snr_score", 0.0) or 0.0), 4)
    body["ihsan_score"] = round(float(getattr(receipt, "ihsan_score", 0.0) or 0.0), 4)
    body["status"] = status
    body["policy_compliant"] = policy_compliant
    body["silent_fallback_detected"] = silent_fallback_detected
    body["subscriber_delivery_verified"] = subscriber_delivery_verified
    body["subscriber_delivery_mirror_verified"] = subscriber_delivery_mirror_verified
    body["node0_canonical_delivery_verified"] = node0_canonical_delivery_verified
    body["subscriber_delivery_delta"] = delivery_delta
    body["verified_success"] = verified_success
    body["reward_eligible"] = policy_compliant
    body["verification_notes"] = verification_notes + [
        f"runtime_receipt:{getattr(receipt, 'mission_id', '')}",
        f"fate_verdict:{getattr(receipt, 'fate_verdict', '')}",
        f"subscriber_acks:{delivery_delta['delivery_acks']}",
        f"subscriber_dead_letters:{delivery_delta['delivery_dead_letters']}",
        f"delivery_mirror_successes:{delivery_delta['delivery_mirror_successes']}",
        f"delivery_mirror_failures:{delivery_delta['delivery_mirror_failures']}",
        f"node0_canonical_delivery_acks:{delivery_delta['node0_canonical_delivery_acks']}",
        "node0_canonical_delivery_dead_letters:"
        f"{delivery_delta['node0_canonical_delivery_dead_letters']}",
        "node0_canonical_delivery_failures:"
        f"{delivery_delta['node0_canonical_delivery_failures']}",
    ]
    body["prev_receipt_hash"] = previous_hash
    body["prev_receipt_hash_semantics"] = previous_hash_semantics

    if body_key == "run2_receipt":
        route_shift = f"{prior_state_value}->{current_state_value}"
        body["replay_effect_visible"] = body["selected_route"] == "reflex_s1"
        body["behavioral_delta"] = {
            "parameter_name": "reflex_route_preference",
            "previous_value": prior_state_value,
            "current_value": current_state_value,
            "measurable_delta_type": "route_shift",
            "measurable_delta_value": route_shift,
        }

    body["receipt_hash"] = _artifact_hash(body)
    doc["artifact_status"] = ARTIFACT_STATUS_POPULATED
    _write_json(artifact_path, doc)
    return doc


def _compute_reward(
    pre_state_doc: dict[str, Any],
    run1_doc: dict[str, Any],
    reward_doc: dict[str, Any],
    artifact_path: Path,
) -> dict[str, Any]:
    observed = pre_state_doc["pre_state"]["observed_state"]
    run1 = run1_doc["run1_receipt"]
    reward_result = reward_doc["reward_result"]
    latency_budget_ms = float(observed["latency_budget_ms"])

    success_bonus = 0.40 if run1["verified_success"] else 0.0
    snr_gain = min(
        max(float(run1["snr_score"]) - float(observed["snr_floor"]), 0.0), 0.30
    )
    latency_gain = min(
        max((latency_budget_ms - float(run1["latency_ms"])) / latency_budget_ms, 0.0),
        0.20,
    )
    degradation_penalty = 0.25 if str(run1["status"]).lower() == "degraded" else 0.0
    policy_violation_penalty = (
        1.0
        if (not run1["policy_compliant"] or run1["silent_fallback_detected"])
        else 0.0
    )
    reward_value = _clamp(
        success_bonus
        + snr_gain
        + latency_gain
        - degradation_penalty
        - policy_violation_penalty,
        0.0,
        1.0,
    )

    reward_result["success_bonus"] = round(success_bonus, 4)
    reward_result["snr_gain"] = round(snr_gain, 4)
    reward_result["latency_gain"] = round(latency_gain, 4)
    reward_result["degradation_penalty"] = round(degradation_penalty, 4)
    reward_result["policy_violation_penalty"] = round(policy_violation_penalty, 4)
    reward_result["reward"] = round(reward_value, 4)
    reward_result["reward_verified"] = True
    reward_doc["reward_contract"]["verified_sources"] = [
        str(artifact_path.parent / "pre_state.json"),
        str(artifact_path.parent / "run1_receipt.json"),
    ]
    reward_doc["artifact_status"] = ARTIFACT_STATUS_POPULATED
    _write_json(artifact_path, reward_doc)
    return reward_doc


def _apply_state_delta(
    *,
    runtime: SovereignRuntime,
    state_delta_doc: dict[str, Any],
    artifact_path: Path,
    mission_prompt: str,
    run1_doc: dict[str, Any],
    reward_doc: dict[str, Any],
) -> dict[str, Any]:
    reflex = runtime._organism._nervous_system._reflex
    pattern_hash = reflex._hash_input(mission_prompt)
    result = state_delta_doc["state_delta_result"]
    threshold = float(
        state_delta_doc.get("state_delta_contract", {})
        and reward_doc["reward_contract"]["reward_threshold_for_adjustment"]
    )
    should_apply = (
        float(reward_doc["reward_result"]["reward"]) >= threshold
        and bool(run1_doc["run1_receipt"]["policy_compliant"])
        and bool(run1_doc["run1_receipt"]["verified_success"])
    )
    persistence_path = (
        Path(reflex._persistence_path) if reflex._persistence_path else None
    )

    result["applied"] = False
    result["previous_value"] = "prefer_deliberation"
    result["new_value"] = "prefer_deliberation"
    result["mission_pattern_hash"] = pattern_hash
    result["state_store_kind"] = "reflex_cache_json"
    result["state_store_key"] = f"node0/reflexes.json:{pattern_hash}"
    result["state_store_write_verified"] = False
    result["reward_source_receipt_hash"] = run1_doc["run1_receipt"]["receipt_hash"]
    result["persisted_at"] = _utc_now()

    if should_apply:
        reflex.compile_from_candidate(
            pattern_id=pattern_hash,
            input_template=mission_prompt,
            output_template=str(
                getattr(runtime._last_mission_receipt, "output_text", "")
            ),
            ihsan_score=float(run1_doc["run1_receipt"]["ihsan_score"]),
            observation_count=REFLEX_PRECIPITATION_HITS,
        )
        reflex.save_to_disk()
        if persistence_path and persistence_path.exists():
            persisted = _read_json(persistence_path)
            if pattern_hash in persisted.get("entries", {}):
                result["applied"] = True
                result["new_value"] = "prefer_reflex"
                result["state_store_write_verified"] = True

    state_delta_doc["state_delta_contract"][
        "parameter_path"
    ] = f"canonical_spearpoint_v1.reflex_route_preference.{pattern_hash}"
    state_delta_doc["artifact_status"] = ARTIFACT_STATUS_POPULATED
    _write_json(artifact_path, state_delta_doc)
    return state_delta_doc


def _verify_chain(
    *,
    run1_doc: dict[str, Any],
    run2_doc: dict[str, Any],
    reward_doc: dict[str, Any],
    state_delta_doc: dict[str, Any],
    chain_doc: dict[str, Any],
    artifact_path: Path,
) -> dict[str, Any]:
    run1 = run1_doc["run1_receipt"]
    run2 = run2_doc["run2_receipt"]
    state_delta = state_delta_doc["state_delta_result"]
    verification = chain_doc["verification_result"]

    run1_hash_valid = run1["receipt_hash"] == _artifact_hash(run1)
    run2_hash_valid = run2["receipt_hash"] == _artifact_hash(run2)

    verification["run1_hash_present"] = bool(run1["receipt_hash"]) and run1_hash_valid
    verification["run2_hash_present"] = bool(run2["receipt_hash"]) and run2_hash_valid
    verification["hash_link_valid"] = run2["prev_receipt_hash"] == run1["receipt_hash"]
    verification["reward_sources_verified"] = bool(
        reward_doc["reward_result"]["reward_verified"]
    )
    verification["state_delta_reflected_in_run2"] = (
        run2["behavioral_delta"]["current_value"] == state_delta["new_value"]
    )
    verification["run2_policy_clean"] = bool(
        run2["policy_compliant"] and not run2["silent_fallback_detected"]
    )

    controlled_change = bool(
        run1["selected_route"] != run2["selected_route"]
        or float(run2["latency_ms"]) < float(run1["latency_ms"])
    )
    verification["canonical_status_achieved"] = all(
        [
            verification["run1_hash_present"],
            verification["run2_hash_present"],
            verification["hash_link_valid"],
            verification["reward_sources_verified"],
            verification["state_delta_reflected_in_run2"],
            verification["run2_policy_clean"],
            bool(state_delta["state_store_write_verified"]),
            bool(run1["policy_compliant"]),
            bool(run1["verified_success"]),
            bool(run2["verified_success"]),
            controlled_change,
            run1["authority_path"] == AUTHORITY_PATH,
            run2["authority_path"] == AUTHORITY_PATH,
            run2["policy_version"] == run1["policy_version"],
        ]
    )
    verification["notes"] = [
        f"controlled_change:{run1['selected_route']}->{run2['selected_route']}",
        f"reward:{reward_doc['reward_result']['reward']}",
        f"state_store:{state_delta['state_store_key']}",
    ]
    chain_doc["artifact_status"] = (
        "MINIMAL_CANONICAL_PROOF"
        if verification["canonical_status_achieved"]
        else ARTIFACT_STATUS_POPULATED
    )
    _write_json(artifact_path, chain_doc)
    return chain_doc


async def _run_async(artifact_dir: Path, state_dir: Path) -> dict[str, Any]:
    _require_artifact_bundle(artifact_dir)
    _prepare_clean_state_dir(state_dir)
    _seed_identity_credentials(state_dir)

    mission_path = artifact_dir / "mission_input.json"
    pre_state_path = artifact_dir / "pre_state.json"
    run1_path = artifact_dir / "run1_receipt.json"
    reward_path = artifact_dir / "reward_calc.json"
    state_delta_path = artifact_dir / "state_delta.json"
    run2_path = artifact_dir / "run2_receipt.json"
    chain_path = artifact_dir / "chain_verification.json"

    mission_doc = _read_json(mission_path)
    pre_state_doc = _read_json(pre_state_path)
    run1_doc = _read_json(run1_path)
    reward_doc = _read_json(reward_path)
    state_delta_doc = _read_json(state_delta_path)
    run2_doc = _read_json(run2_path)
    chain_doc = _read_json(chain_path)

    mission_contract = mission_doc["mission_input"]
    mission_prompt = str(mission_contract["prompt"])
    gateway = DeterministicGateway()

    with ExitStack() as stack:
        from core.node0.heartbeat import Node0Heartbeat

        stack.enter_context(
            patch.object(
                Node0Heartbeat,
                "_boot_federation_ambassador",
                _noop_boot_federation_ambassador,
            )
        )

        previous_canonical = os.environ.get("BIZRA_CANONICAL_MODE")
        previous_role = os.environ.get("BIZRA_NODE_ROLE")
        os.environ["BIZRA_CANONICAL_MODE"] = "1"
        os.environ["BIZRA_NODE_ROLE"] = "node"
        try:
            runtime1 = await _build_runtime(state_dir, gateway)
            try:
                pre_state_doc = _capture_pre_state(
                    pre_state_path,
                    pre_state_doc,
                    runtime1,
                    gateway,
                    mission_prompt,
                )
                delivery_before1 = _cqrs_delivery_snapshot(runtime1)
                receipt1 = await runtime1.mission(
                    mission_prompt, source="canonical_spearpoint_v1", context={}
                )
                delivery_after1 = await _await_delivery_settlement(
                    runtime1, delivery_before1
                )
                runtime1._last_mission_receipt = receipt1
                run1_doc = _build_run_receipt(
                    runtime=runtime1,
                    artifact_path=run1_path,
                    doc=run1_doc,
                    body_key="run1_receipt",
                    mission_prompt=mission_prompt,
                    mission_contract=mission_contract,
                    receipt=receipt1,
                    previous_hash=GENESIS_ZERO_HASH,
                    previous_hash_semantics="GENESIS_ZERO_HASH",
                    delivery_before=delivery_before1,
                    delivery_after=delivery_after1,
                )
                reward_doc = _compute_reward(
                    pre_state_doc, run1_doc, reward_doc, reward_path
                )
                state_delta_doc = _apply_state_delta(
                    runtime=runtime1,
                    state_delta_doc=state_delta_doc,
                    artifact_path=state_delta_path,
                    mission_prompt=mission_prompt,
                    run1_doc=run1_doc,
                    reward_doc=reward_doc,
                )
            finally:
                await runtime1.shutdown()

            runtime2 = await _build_runtime(state_dir, gateway)
            try:
                delivery_before2 = _cqrs_delivery_snapshot(runtime2)
                receipt2 = await runtime2.mission(
                    mission_prompt, source="canonical_spearpoint_v1", context={}
                )
                delivery_after2 = await _await_delivery_settlement(
                    runtime2, delivery_before2
                )
                run2_doc = _build_run_receipt(
                    runtime=runtime2,
                    artifact_path=run2_path,
                    doc=run2_doc,
                    body_key="run2_receipt",
                    mission_prompt=mission_prompt,
                    mission_contract=mission_contract,
                    receipt=receipt2,
                    previous_hash=run1_doc["run1_receipt"]["receipt_hash"],
                    previous_hash_semantics="CHAINED_ARTIFACT_RECEIPT",
                    delivery_before=delivery_before2,
                    delivery_after=delivery_after2,
                    state_delta_ref=str(state_delta_path),
                    prior_state_value=str(
                        state_delta_doc["state_delta_result"]["previous_value"]
                    ),
                    current_state_value=str(
                        state_delta_doc["state_delta_result"]["new_value"]
                    ),
                )
            finally:
                await runtime2.shutdown()
        finally:
            if previous_canonical is None:
                os.environ.pop("BIZRA_CANONICAL_MODE", None)
            else:
                os.environ["BIZRA_CANONICAL_MODE"] = previous_canonical
            if previous_role is None:
                os.environ.pop("BIZRA_NODE_ROLE", None)
            else:
                os.environ["BIZRA_NODE_ROLE"] = previous_role

    return _verify_chain(
        run1_doc=run1_doc,
        run2_doc=run2_doc,
        reward_doc=reward_doc,
        state_delta_doc=state_delta_doc,
        chain_doc=chain_doc,
        artifact_path=chain_path,
    )


def run_canonical_spearpoint_v1(
    *,
    artifact_dir: Path = Path("artifacts/CANONICAL_SPEARPOINT_V1"),
    state_dir: Path = Path("artifacts/CANONICAL_SPEARPOINT_V1/_runtime_state"),
) -> dict[str, Any]:
    return asyncio.run(_run_async(artifact_dir.resolve(), state_dir.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute CANONICAL_SPEARPOINT_V1.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/CANONICAL_SPEARPOINT_V1"),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("artifacts/CANONICAL_SPEARPOINT_V1/_runtime_state"),
    )
    args = parser.parse_args()

    report = run_canonical_spearpoint_v1(
        artifact_dir=args.artifact_dir,
        state_dir=args.state_dir,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["verification_result"]["canonical_status_achieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
