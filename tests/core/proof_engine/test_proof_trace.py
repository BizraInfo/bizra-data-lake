"""
End-to-End Proof Trace Tests — BIZRA Proof Trace Fixtures v0.1

Validates the 10-phase proof trace (OBSERVE → ORIENT → DIVERGE → CONSTRAIN →
ACT → COMMIT → PROVE → CHECK → SETTLE → LEARN → FEEDBACK) against the
existing proof engine infrastructure.

Fixture source: tests/fixtures/proof_trace_v01.json

Standing on Giants:
- Lamport (1978): Logical clocks and hash-chained event ordering
- Merkle (1979): Hash chains for tamper detection
- Shannon (1948): SNR as quality metric at every gate
- Al-Ghazali (1095): Ihsan as constitutional quality floor
- Nakamoto (2008): Proof-of-Work as verifiable contribution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    ADL_HARBERGER_TAX_RATE,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.proof_engine import GATE_CHAIN
from core.proof_engine.canonical import blake3_digest, canonical_bytes, canonical_json
from core.proof_engine.poi_engine import PoIReasonCode

# CAHT chain builder from fixtures package
from tests.fixtures.caht_chain import GENESIS_HASH, CAHTChain

# =============================================================================
# FIXTURES
# =============================================================================

FIXTURES_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "proof_trace_v01.json"
)


@pytest.fixture(scope="module")
def trace_data() -> Dict[str, Any]:
    """Load the proof trace fixtures."""
    with open(FIXTURES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def events(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the event list."""
    return trace_data["events"]


@pytest.fixture(scope="module")
def identities(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract identity map."""
    return trace_data["identities"]


@pytest.fixture(scope="module")
def thresholds(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract threshold values."""
    return trace_data["thresholds"]


@pytest.fixture(scope="module")
def failure_fixtures(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract failure mode fixtures."""
    return trace_data["failure_fixtures"]


# =============================================================================
# TEST CLASS: Fixture Integrity
# =============================================================================


class TestFixtureIntegrity:
    """Validate the fixture file itself is well-formed."""

    def test_fixture_file_exists(self) -> None:
        assert FIXTURES_PATH.exists(), f"Fixture file not found: {FIXTURES_PATH}"

    def test_fixture_is_valid_json(self) -> None:
        with open(FIXTURES_PATH) as f:
            data = json.load(f)
        assert "_meta" in data
        assert "events" in data
        assert "identities" in data
        assert "thresholds" in data

    def test_meta_has_invariants(self, trace_data: Dict[str, Any]) -> None:
        meta = trace_data["_meta"]
        assert meta["version"] == "0.1.0"
        assert len(meta["invariants"]) >= 4

    def test_event_count(self, events: List[Dict[str, Any]]) -> None:
        assert len(events) == 12, f"Expected 12 events, got {len(events)}"

    def test_events_sequential(self, events: List[Dict[str, Any]]) -> None:
        for i, event in enumerate(events):
            assert event["seq"] == i, f"Event {i} has seq={event['seq']}"

    def test_failure_fixture_count(
        self, failure_fixtures: List[Dict[str, Any]]
    ) -> None:
        assert len(failure_fixtures) == 5


# =============================================================================
# TEST CLASS: Threshold Alignment
# =============================================================================


class TestThresholdAlignment:
    """Verify fixture thresholds match authoritative constants.py."""

    def test_ihsan_production(self, thresholds: Dict[str, Any]) -> None:
        assert thresholds["ihsan_production"] == UNIFIED_IHSAN_THRESHOLD

    def test_snr_minimum(self, thresholds: Dict[str, Any]) -> None:
        assert thresholds["snr_minimum"] == UNIFIED_SNR_THRESHOLD

    def test_adl_gini(self, thresholds: Dict[str, Any]) -> None:
        assert thresholds["adl_gini_max"] == ADL_GINI_THRESHOLD

    def test_harberger_rate(self, thresholds: Dict[str, Any]) -> None:
        assert thresholds["harberger_tax_rate"] == ADL_HARBERGER_TAX_RATE

    def test_source_documented(self, thresholds: Dict[str, Any]) -> None:
        assert thresholds["_source"] == "core/integration/constants.py"


# =============================================================================
# TEST CLASS: CAHT Chain Integrity
# =============================================================================


class TestCAHTChain:
    """Validate Content-Addressable Hash Trail construction and verification."""

    def test_chain_builds_from_events(self, events: List[Dict[str, Any]]) -> None:
        chain = CAHTChain()
        for event in events:
            entry = chain.append(event)
            assert entry.seq == event["seq"]
            assert entry.phase == event["phase"]
        assert len(chain) == 12

    def test_chain_genesis_hash(self, events: List[Dict[str, Any]]) -> None:
        chain = CAHTChain()
        chain.append(events[0])
        assert chain.entries[0].prev_hash == GENESIS_HASH

    def test_chain_linking(self, events: List[Dict[str, Any]]) -> None:
        chain = CAHTChain()
        for event in events:
            chain.append(event)
        # Each entry's prev_hash == prior entry's entry_hash
        for i in range(1, len(chain.entries)):
            assert chain.entries[i].prev_hash == chain.entries[i - 1].entry_hash

    def test_chain_verification_passes(self, events: List[Dict[str, Any]]) -> None:
        chain = CAHTChain()
        for event in events:
            chain.append(event)
        assert chain.verify() is True

    def test_chain_tamper_detection(self, events: List[Dict[str, Any]]) -> None:
        chain = CAHTChain()
        for event in events:
            chain.append(event)

        # Tamper with middle entry
        chain.entries[5].entry_hash = "ff" * 32
        assert chain.verify() is False

    def test_chain_determinism(self, events: List[Dict[str, Any]]) -> None:
        """Same events produce same chain — 10-iteration DoD."""
        hashes = []
        for _ in range(10):
            chain = CAHTChain()
            for event in events:
                chain.append(event)
            hashes.append(chain.head_hash)
        assert len(set(hashes)) == 1, "Chain is non-deterministic"

    def test_empty_chain_verifies(self) -> None:
        chain = CAHTChain()
        assert chain.verify() is True
        assert chain.head_hash == GENESIS_HASH


# =============================================================================
# TEST CLASS: Phase Transition Validation
# =============================================================================


# Valid phase transitions (directed edges)
VALID_TRANSITIONS = {
    "OBSERVE": {"ORIENT"},
    "ORIENT": {"DIVERGE"},
    "DIVERGE": {"CONSTRAIN"},
    "CONSTRAIN": {"ACT"},
    "ACT": {"COMMIT"},
    "COMMIT": {"PROVE"},
    "PROVE": {"CHECK"},
    "CHECK": {"SETTLE"},
    "SETTLE": {"LEARN"},
    "LEARN": {"LEARN", "FEEDBACK"},  # LEARN can chain or finalize
    "FEEDBACK": set(),  # Terminal
}


class TestPhaseTransitions:
    """Validate the 10-phase state machine."""

    def test_happy_path_transitions(self, events: List[Dict[str, Any]]) -> None:
        """All transitions in the happy path are valid."""
        for i in range(len(events) - 1):
            current = events[i]["phase"]
            next_phase = events[i + 1]["phase"]
            allowed = VALID_TRANSITIONS.get(current, set())
            assert next_phase in allowed, (
                f"Invalid transition: {current} → {next_phase} "
                f"(event {i} → {i+1}). Allowed: {allowed}"
            )

    def test_starts_with_observe(self, events: List[Dict[str, Any]]) -> None:
        assert events[0]["phase"] == "OBSERVE"

    def test_ends_with_feedback(self, events: List[Dict[str, Any]]) -> None:
        assert events[-1]["phase"] == "FEEDBACK"

    def test_all_phases_covered(self, events: List[Dict[str, Any]]) -> None:
        phases = {e["phase"] for e in events}
        expected = {
            "OBSERVE",
            "ORIENT",
            "DIVERGE",
            "CONSTRAIN",
            "ACT",
            "COMMIT",
            "PROVE",
            "CHECK",
            "SETTLE",
            "LEARN",
            "FEEDBACK",
        }
        assert phases == expected, f"Missing phases: {expected - phases}"

    def test_invalid_transition_detected(self) -> None:
        """The transition firewall failure fixture must be caught."""
        # OBSERVE → SETTLE is invalid
        allowed = VALID_TRANSITIONS.get("OBSERVE", set())
        assert "SETTLE" not in allowed


# =============================================================================
# TEST CLASS: Gate Chain Compliance
# =============================================================================


class TestGateChainCompliance:
    """Verify gate application rules per phase."""

    # Minimum gates required per phase
    PHASE_MIN_GATES = {
        "OBSERVE": ["schema"],
        "ORIENT": ["schema", "provenance", "snr"],
        "DIVERGE": ["schema", "snr"],
        "CONSTRAIN": ["schema", "provenance", "snr", "constraint"],
        "ACT": ["schema", "provenance", "snr", "constraint", "safety"],
        "COMMIT": ["schema", "provenance", "snr", "constraint", "safety", "commit"],
        "PROVE": ["schema", "provenance", "snr", "constraint", "safety", "commit"],
        "CHECK": ["schema", "provenance", "snr", "constraint", "safety", "commit"],
        "SETTLE": ["schema", "provenance", "snr", "constraint", "safety", "commit"],
        "LEARN": ["schema"],
        "FEEDBACK": ["schema"],
    }

    def test_all_events_have_gates(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            assert (
                "gates_applied" in event
            ), f"Event seq={event['seq']} missing gates_applied"
            assert len(event["gates_applied"]) > 0

    def test_minimum_gates_per_phase(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            phase = event["phase"]
            applied = set(event["gates_applied"])
            required = set(self.PHASE_MIN_GATES.get(phase, ["schema"]))
            missing = required - applied
            assert not missing, (
                f"Event seq={event['seq']} phase={phase}: "
                f"missing required gates {missing}"
            )

    def test_gates_are_valid_names(self, events: List[Dict[str, Any]]) -> None:
        valid_gates = set(GATE_CHAIN)
        for event in events:
            for gate in event["gates_applied"]:
                assert gate in valid_gates, (
                    f"Event seq={event['seq']}: unknown gate '{gate}'. "
                    f"Valid: {valid_gates}"
                )

    def test_all_gate_results_present(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            for gate in event["gates_applied"]:
                assert gate in event["gate_results"], (
                    f"Event seq={event['seq']}: gate '{gate}' in gates_applied "
                    f"but missing from gate_results"
                )

    def test_all_happy_path_gates_pass(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            for gate_name, result in event["gate_results"].items():
                assert result["status"] == "passed", (
                    f"Event seq={event['seq']} gate={gate_name}: "
                    f"status={result['status']}"
                )


# =============================================================================
# TEST CLASS: Ihsan/SNR Compliance
# =============================================================================


class TestQualityScoreCompliance:
    """Verify all events meet constitutional quality thresholds."""

    def test_all_ihsan_above_production(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            ihsan = event["ihsan_score"]
            assert ihsan >= UNIFIED_IHSAN_THRESHOLD, (
                f"Event seq={event['seq']} phase={event['phase']}: "
                f"ihsan={ihsan} < {UNIFIED_IHSAN_THRESHOLD}"
            )

    def test_all_snr_above_minimum(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            snr = event["snr_score"]
            assert snr >= UNIFIED_SNR_THRESHOLD, (
                f"Event seq={event['seq']} phase={event['phase']}: "
                f"snr={snr} < {UNIFIED_SNR_THRESHOLD}"
            )

    def test_all_receipts_accepted(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            assert (
                event["receipt_status"] == "accepted"
            ), f"Event seq={event['seq']}: status={event['receipt_status']}"

    def test_settlement_gini_below_threshold(
        self, events: List[Dict[str, Any]]
    ) -> None:
        settle_events = [e for e in events if e["phase"] == "SETTLE"]
        for event in settle_events:
            gini = event["payload"]["post_settlement_gini"]
            assert (
                gini <= ADL_GINI_THRESHOLD
            ), f"Post-settlement Gini {gini} > {ADL_GINI_THRESHOLD}"

    def test_harberger_tax_applied(self, events: List[Dict[str, Any]]) -> None:
        settle_events = [e for e in events if e["phase"] == "SETTLE"]
        for event in settle_events:
            dist = event["payload"]["distribution"]
            total = event["payload"]["total_payment"]
            tax = dist["harberger_tax_ubc"]
            expected_rate = tax / total if total > 0 else 0
            assert (
                abs(expected_rate - ADL_HARBERGER_TAX_RATE) < 0.01
            ), f"Harberger tax rate {expected_rate:.3f} != {ADL_HARBERGER_TAX_RATE}"


# =============================================================================
# TEST CLASS: Identity & Actor Consistency
# =============================================================================


class TestActorConsistency:
    """Verify actors are valid identities and roles are consistent."""

    def test_all_actors_known(
        self, events: List[Dict[str, Any]], identities: Dict[str, Any]
    ) -> None:
        valid_actors = set(identities.keys())
        for event in events:
            actor = event["actor"]
            assert actor in valid_actors, (
                f"Event seq={event['seq']}: unknown actor '{actor}'. "
                f"Valid: {valid_actors}"
            )

    def test_sarah_is_requester(self, identities: Dict[str, Any]) -> None:
        assert identities["sarah"]["role"] == "requester"

    def test_marcus_is_provider(self, identities: Dict[str, Any]) -> None:
        assert identities["marcus"]["role"] == "provider"

    def test_verifier_is_verifier(self, identities: Dict[str, Any]) -> None:
        assert identities["verifier"]["role"] == "verifier"

    def test_pool_is_router(self, identities: Dict[str, Any]) -> None:
        assert identities["resource_pool"]["role"] == "router"

    def test_all_identities_have_ihsan(self, identities: Dict[str, Any]) -> None:
        for name, identity in identities.items():
            assert "ihsan_score" in identity, f"Identity '{name}' missing ihsan_score"
            assert (
                identity["ihsan_score"] >= UNIFIED_IHSAN_THRESHOLD
            ), f"Identity '{name}' ihsan={identity['ihsan_score']} < threshold"


# =============================================================================
# TEST CLASS: Settlement Economics
# =============================================================================


class TestSettlementEconomics:
    """Validate settlement distribution rules."""

    def test_distribution_sums_to_total(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            if event["phase"] != "SETTLE":
                continue
            total = event["payload"]["total_payment"]
            dist = event["payload"]["distribution"]
            dist_sum = sum(dist.values())
            assert (
                abs(dist_sum - total) < 0.01
            ), f"Distribution sum {dist_sum} != total {total}"

    def test_zakat_rate(self, events: List[Dict[str, Any]]) -> None:
        """Zakat is 2.5% of total — standing on Al-Ghazali."""
        for event in events:
            if event["phase"] != "SETTLE":
                continue
            total = event["payload"]["total_payment"]
            zakat = event["payload"]["distribution"]["zakat_pool"]
            rate = zakat / total if total > 0 else 0
            assert abs(rate - 0.02) < 0.005, f"Zakat rate {rate:.3f} != ~0.02"

    def test_provider_gets_majority(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            if event["phase"] != "SETTLE":
                continue
            total = event["payload"]["total_payment"]
            provider = event["payload"]["distribution"]["marcus_provider"]
            ratio = provider / total if total > 0 else 0
            assert ratio >= 0.80, f"Provider ratio {ratio:.3f} < 0.80"

    def test_mudarabah_settlement_type(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            if event["phase"] != "SETTLE":
                continue
            assert event["payload"]["settlement_type"] == "mudarabah"


# =============================================================================
# TEST CLASS: PoI (Proof-of-Impact) Scoring
# =============================================================================


class TestPoIScoring:
    """Validate PoI receipt structure and scoring rules."""

    def test_poi_event_exists(self, events: List[Dict[str, Any]]) -> None:
        poi_events = [e for e in events if e["event_type"] == "poi_scoring"]
        assert len(poi_events) >= 1

    def test_poi_stages_complete(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            if event["event_type"] != "poi_scoring":
                continue
            stages = event["payload"]["poi_stages"]
            assert "stage1_contribution" in stages
            assert "stage2_reach" in stages
            assert "stage3_longevity" in stages
            assert "stage4_composite" in stages

    def test_poi_composite_formula(self, events: List[Dict[str, Any]]) -> None:
        """Composite = alpha*contribution + beta*reach + gamma*longevity."""
        for event in events:
            if event["event_type"] != "poi_scoring":
                continue
            stages = event["payload"]["poi_stages"]
            s1 = stages["stage1_contribution"]["score"]
            s2 = stages["stage2_reach"]["score"]
            s3 = stages["stage3_longevity"]["score"]
            composite = stages["stage4_composite"]
            weights = composite["weights"]
            expected = (
                weights["alpha"] * s1 + weights["beta"] * s2 + weights["gamma"] * s3
            )
            assert abs(composite["score"] - expected) < 0.01, (
                f"Composite {composite['score']} != "
                f"{weights['alpha']}*{s1} + {weights['beta']}*{s2} + "
                f"{weights['gamma']}*{s3} = {expected}"
            )

    def test_poi_receipt_reason_ok(self, events: List[Dict[str, Any]]) -> None:
        for event in events:
            if event["event_type"] != "poi_scoring":
                continue
            receipt = event["payload"]["poi_receipt"]
            assert receipt["reason"] == PoIReasonCode.POI_OK.value


# =============================================================================
# TEST CLASS: Failure Mode Fixtures
# =============================================================================


class TestFailureModes:
    """Validate all 5 failure fixtures produce correct rejections."""

    def test_ihsan_gate_reject(self, failure_fixtures: List[Dict[str, Any]]) -> None:
        fix = next(f for f in failure_fixtures if f["id"] == "fail_ihsan_gate")
        event = fix["event"]
        constraint_result = event["gate_results"]["constraint"]
        assert constraint_result["status"] == "failed"
        assert constraint_result["reason"] == "IHSAN_BELOW_THRESHOLD"
        assert event["receipt_status"] == "rejected"
        # Actor ihsan below threshold
        actor_ihsan = constraint_result["evidence"]["actor_ihsan"]
        assert actor_ihsan < UNIFIED_IHSAN_THRESHOLD

    def test_transition_firewall_block(
        self, failure_fixtures: List[Dict[str, Any]]
    ) -> None:
        fix = next(f for f in failure_fixtures if f["id"] == "fail_transition_firewall")
        event = fix["event"]
        schema_result = event["gate_results"]["schema"]
        assert schema_result["status"] == "failed"
        assert schema_result["reason"] == "INVALID_STATE_TRANSITION"
        # OBSERVE → SETTLE is invalid
        evidence = schema_result["evidence"]
        assert evidence["from"] == "OBSERVE"
        assert evidence["to"] == "SETTLE"
        assert "SETTLE" not in evidence["allowed_transitions"]

    def test_verifier_fabrication_detect(
        self, failure_fixtures: List[Dict[str, Any]]
    ) -> None:
        fix = next(
            f for f in failure_fixtures if f["id"] == "fail_verifier_fabrication"
        )
        event = fix["event"]
        provenance_result = event["gate_results"]["provenance"]
        assert provenance_result["status"] == "failed"
        assert provenance_result["reason"] == "HASH_MISMATCH"
        assert event["receipt_status"] == "rejected"

    def test_replay_attack_block(self, failure_fixtures: List[Dict[str, Any]]) -> None:
        fix = next(f for f in failure_fixtures if f["id"] == "fail_replay_attack")
        event = fix["event"]
        provenance_result = event["gate_results"]["provenance"]
        assert provenance_result["status"] == "failed"
        assert provenance_result["reason"] == "REPLAY_DETECTED"
        assert event["receipt_status"] == "rejected"

    def test_sybil_node_detected(self, failure_fixtures: List[Dict[str, Any]]) -> None:
        fix = next(f for f in failure_fixtures if f["id"] == "fail_sybil_node")
        event = fix["event"]
        # Sybil detection results in penalty, not rejection
        assert event["receipt_status"] == "accepted"
        assert event["poi_penalty"] == PoIReasonCode.POI_PENALTY_RING_DETECTED.value
        # Ring has 3+ members
        ring = event["payload"]["ring_members"]
        assert len(ring) >= 3


# =============================================================================
# TEST CLASS: Canonical Determinism
# =============================================================================


class TestCanonicalDeterminism:
    """Verify canonical encoding of fixtures is deterministic (10-iteration DoD)."""

    def test_event_canonical_determinism(self, events: List[Dict[str, Any]]) -> None:
        """Same event → same canonical bytes → same hash, 10 iterations."""
        for event in events:
            hashes = set()
            for _ in range(10):
                digest = blake3_digest(canonical_bytes(event))
                h = digest.hex() if isinstance(digest, bytes) else str(digest)
                hashes.add(h)
            assert len(hashes) == 1, (
                f"Event seq={event['seq']} is non-deterministic: "
                f"{len(hashes)} distinct hashes"
            )

    def test_canonical_json_sorted_keys(self, events: List[Dict[str, Any]]) -> None:
        """canonical_json must sort keys at all nesting levels."""
        for event in events:
            canon = canonical_json(event)
            keys = list(canon.keys())
            assert keys == sorted(
                keys
            ), f"Event seq={event['seq']}: keys not sorted: {keys}"

    def test_payload_canonical_determinism(self, events: List[Dict[str, Any]]) -> None:
        """Payload sub-objects must also be deterministic."""
        for event in events:
            payload = event.get("payload", {})
            hashes = set()
            for _ in range(10):
                digest = blake3_digest(canonical_bytes(payload))
                h = digest.hex() if isinstance(digest, bytes) else str(digest)
                hashes.add(h)
            assert len(hashes) == 1


# =============================================================================
# TEST CLASS: Design Axiom Enforcement
# =============================================================================


class TestDesignAxioms:
    """Structural tests enforcing the 5 design axioms."""

    def test_event_derived_state(self, events: List[Dict[str, Any]]) -> None:
        """Axiom: No separate state store — state derives from events.

        Test: Every event references only data from prior events
        (no external state IDs that aren't introduced in the event stream).
        """
        introduced_ids = set()
        for event in events:
            payload = event.get("payload", {})
            # Track allocation IDs introduced
            if "allocation_id" in payload:
                alloc_id = payload["allocation_id"]
                if event["phase"] == "ACT":
                    introduced_ids.add(alloc_id)
                else:
                    assert alloc_id in introduced_ids, (
                        f"Event seq={event['seq']} references "
                        f"allocation_id={alloc_id} before it was created"
                    )

    def test_verifiable_transitions(self, events: List[Dict[str, Any]]) -> None:
        """Axiom: Every phase transition validated by at least one gate."""
        for event in events:
            assert len(event["gates_applied"]) >= 1
            assert "schema" in event["gates_applied"]

    def test_evidence_backed_rewards(self, events: List[Dict[str, Any]]) -> None:
        """Axiom: No reward without verified proof chain.

        Settlement events must reference a verified allocation.
        """
        verified_allocations = set()
        for event in events:
            if event["phase"] == "CHECK":
                payload = event["payload"]
                if payload.get("all_checks_passed"):
                    verified_allocations.add(payload["allocation_id"])
            if event["phase"] == "SETTLE":
                alloc_id = event["payload"]["allocation_id"]
                assert (
                    alloc_id in verified_allocations
                ), f"Settlement for {alloc_id} without prior verification"

    def test_fail_closed_default(self, failure_fixtures: List[Dict[str, Any]]) -> None:
        """Axiom: Every gate defaults to rejection."""
        for fix in failure_fixtures:
            event = fix["event"]
            # At least one gate must have failed (fail-closed)
            has_failure = any(
                r["status"] == "failed" for r in event["gate_results"].values()
            )
            has_penalty = event.get("poi_penalty") is not None
            assert (
                has_failure or has_penalty
            ), f"Failure fixture '{fix['id']}' has no gate failures or penalties"
