"""
Terminal v1 Spine Tests — State Machine, Permission Envelope, Contracts.

Standing on Giants:
- Harel (1987): Statechart verification
- Thompson (1984): Capability-based security
- Lamport (1978): Event ordering
- Kahneman (2002): Cognitive split

Tests cover:
1. TerminalState enum completeness
2. TERMINAL_TRANSITIONS validity and coverage
3. TerminalStateController transitions (happy + invalid)
4. PermissionEnvelope scope checks
5. MissionReceipt serialization
6. EventRecord serialization
7. BriefingContext defaults
8. ExecutionPath tracking
"""

from __future__ import annotations

import time

import pytest

from core.sovereign.terminal import (
    TERMINAL_TRANSITIONS,
    BriefingContext,
    ChannelRecord,
    EventRecord,
    EventSeverity,
    ExecutionPath,
    MemoryDelta,
    MissionReceipt,
    PermissionEnvelope,
    ReflexDelta,
    TerminalState,
    TerminalStateController,
    WalletDelta,
)

# ═══════════════════════════════════════════════════════════════════
# TerminalState Enum
# ═══════════════════════════════════════════════════════════════════


class TestTerminalState:
    """TerminalState enum completeness."""

    def test_has_9_states(self):
        assert len(TerminalState) == 9

    def test_all_states_are_strings(self):
        for state in TerminalState:
            assert isinstance(state.value, str)

    def test_boot_is_initial(self):
        assert TerminalState.BOOT == TerminalState("boot")

    def test_terminal_states_exist(self):
        terminal = {
            TerminalState.COMPLETED,
            TerminalState.FAILED_RECOVERABLY,
            TerminalState.BLOCKED_CONSTITUTIONALLY,
        }
        assert len(terminal) == 3


# ═══════════════════════════════════════════════════════════════════
# TERMINAL_TRANSITIONS Table
# ═══════════════════════════════════════════════════════════════════


class TestTransitionTable:
    """Transition table completeness and validity."""

    def test_all_states_have_transitions(self):
        for state in TerminalState:
            assert state in TERMINAL_TRANSITIONS, f"{state} missing from transitions"

    def test_transitions_are_frozensets(self):
        for targets in TERMINAL_TRANSITIONS.values():
            assert isinstance(targets, frozenset)

    def test_boot_only_goes_to_ready(self):
        assert TERMINAL_TRANSITIONS[TerminalState.BOOT] == frozenset(
            {TerminalState.READY}
        )

    def test_executing_has_4_targets(self):
        targets = TERMINAL_TRANSITIONS[TerminalState.EXECUTING]
        assert len(targets) == 4
        assert TerminalState.COMPLETED in targets
        assert TerminalState.FAILED_RECOVERABLY in targets
        assert TerminalState.BLOCKED_CONSTITUTIONALLY in targets
        assert TerminalState.AWAITING_ESCALATION in targets

    def test_escalation_can_return_to_executing(self):
        targets = TERMINAL_TRANSITIONS[TerminalState.AWAITING_ESCALATION]
        assert TerminalState.EXECUTING in targets

    def test_recoverable_terminal_states_reset_to_ready(self):
        for state in (
            TerminalState.COMPLETED,
            TerminalState.FAILED_RECOVERABLY,
        ):
            assert TerminalState.READY in TERMINAL_TRANSITIONS[state]

    def test_blocked_state_is_restart_only(self):
        assert TERMINAL_TRANSITIONS[TerminalState.BLOCKED_CONSTITUTIONALLY] == frozenset()

    def test_no_self_loops(self):
        for source, targets in TERMINAL_TRANSITIONS.items():
            assert source not in targets, f"{source} has self-loop"


# ═══════════════════════════════════════════════════════════════════
# TerminalStateController
# ═══════════════════════════════════════════════════════════════════


class TestTerminalStateController:
    """State machine controller tests."""

    @pytest.fixture
    def ctrl(self):
        return TerminalStateController()

    def test_initial_state_is_boot(self, ctrl):
        assert ctrl.state == TerminalState.BOOT

    def test_valid_transition_boot_to_ready(self, ctrl):
        assert ctrl.transition(TerminalState.READY) is True
        assert ctrl.state == TerminalState.READY

    def test_invalid_transition_boot_to_executing(self, ctrl):
        assert ctrl.transition(TerminalState.EXECUTING) is False
        assert ctrl.state == TerminalState.BOOT

    def test_full_happy_path(self, ctrl):
        """BOOT → READY → DRAFTING → REVIEW → EXECUTING → COMPLETED → READY."""
        assert ctrl.transition(TerminalState.READY)
        assert ctrl.start_mission("m-001")
        assert ctrl.state == TerminalState.MISSION_DRAFTING
        assert ctrl.transition(TerminalState.PERMISSION_REVIEW)
        assert ctrl.transition(TerminalState.EXECUTING)
        assert ctrl.complete()
        assert ctrl.state == TerminalState.COMPLETED
        assert ctrl.reset()
        assert ctrl.state == TerminalState.READY

    def test_failure_path(self, ctrl):
        """BOOT → READY → ... → EXECUTING → FAILED → READY."""
        ctrl.transition(TerminalState.READY)
        ctrl.start_mission("m-002")
        ctrl.transition(TerminalState.PERMISSION_REVIEW)
        ctrl.transition(TerminalState.EXECUTING)
        assert ctrl.fail()
        assert ctrl.state == TerminalState.FAILED_RECOVERABLY
        assert ctrl.reset()
        assert ctrl.state == TerminalState.READY

    def test_escalation_path(self, ctrl):
        """EXECUTING → AWAITING_ESCALATION → EXECUTING → COMPLETED."""
        ctrl.transition(TerminalState.READY)
        ctrl.start_mission("m-003")
        ctrl.transition(TerminalState.PERMISSION_REVIEW)
        ctrl.transition(TerminalState.EXECUTING)
        assert ctrl.transition(TerminalState.AWAITING_ESCALATION)
        assert ctrl.transition(TerminalState.EXECUTING)
        assert ctrl.complete()

    def test_escalation_denial_returns_to_ready(self, ctrl):
        ctrl.transition(TerminalState.READY)
        ctrl.start_mission("m-003b")
        ctrl.transition(TerminalState.PERMISSION_REVIEW)
        ctrl.transition(TerminalState.EXECUTING)
        assert ctrl.transition(TerminalState.AWAITING_ESCALATION)
        assert ctrl.deny_escalation()
        assert ctrl.state == TerminalState.READY
        assert ctrl.mission_id == ""

    def test_start_mission_sets_metadata(self, ctrl):
        ctrl.transition(TerminalState.READY)
        ctrl.start_mission("m-004", ExecutionPath.SYSTEM_1_CACHE_HIT)
        assert ctrl.mission_id == "m-004"
        assert ctrl.execution_path == ExecutionPath.SYSTEM_1_CACHE_HIT

    def test_start_mission_requires_ready(self, ctrl):
        assert ctrl.start_mission("m-005") is False  # still BOOT

    def test_complete_requires_executing(self, ctrl):
        ctrl.transition(TerminalState.READY)
        assert ctrl.complete() is False

    def test_fail_requires_executing(self, ctrl):
        ctrl.transition(TerminalState.READY)
        assert ctrl.fail() is False

    def test_reset_requires_terminal_state(self, ctrl):
        ctrl.transition(TerminalState.READY)
        assert ctrl.reset() is False

    def test_reset_clears_mission_id(self, ctrl):
        ctrl.transition(TerminalState.READY)
        ctrl.start_mission("m-006")
        ctrl.transition(TerminalState.PERMISSION_REVIEW)
        ctrl.transition(TerminalState.EXECUTING)
        ctrl.complete()
        ctrl.reset()
        assert ctrl.mission_id == ""

    def test_to_dict(self, ctrl):
        d = ctrl.to_dict()
        assert d["state"] == "boot"
        assert d["execution_path"] == "SYSTEM_2_NOVEL"
        assert d["mission_id"] == ""
        assert d["restart_required"] is False

    def test_blocked_state_requires_restart(self, ctrl):
        ctrl.transition(TerminalState.READY)
        ctrl.start_mission("m-006b")
        ctrl.transition(TerminalState.PERMISSION_REVIEW)
        ctrl.transition(TerminalState.EXECUTING)
        assert ctrl.block()
        assert ctrl.reset() is False
        assert ctrl.to_dict()["restart_required"] is True


# ═══════════════════════════════════════════════════════════════════
# PermissionEnvelope
# ═══════════════════════════════════════════════════════════════════


class TestPermissionEnvelope:
    """Permission envelope scope checks."""

    def test_default_filesystem_scope(self):
        env = PermissionEnvelope()
        assert env.allows_path("workspace/foo.txt")
        assert env.allows_path("workspace/sub/bar.py")
        assert not env.allows_path("/etc/passwd")

    def test_custom_filesystem_scope(self):
        env = PermissionEnvelope(filesystem=["*.py", "docs/**"])
        assert env.allows_path("main.py")
        assert env.allows_path("docs/readme.md")
        assert not env.allows_path("main.rs")

    def test_network_scope_empty_denies(self):
        env = PermissionEnvelope()
        assert not env.allows_network("api.example.com")

    def test_network_scope_allows_listed(self):
        env = PermissionEnvelope(network=["api.bizra.info", "localhost"])
        assert env.allows_network("api.bizra.info")
        assert env.allows_network("localhost")
        assert not env.allows_network("evil.com")

    def test_to_dict_roundtrip(self):
        env = PermissionEnvelope(
            spend_budget_usd=5.0,
            time_budget_seconds=300,
            escalation="block",
        )
        d = env.to_dict()
        assert d["spend_budget_usd"] == 5.0
        assert d["time_budget_seconds"] == 300
        assert d["escalation"] == "block"
        assert d["data_sensitivity"] == "standard"

    def test_default_applications(self):
        env = PermissionEnvelope()
        assert "terminal" in env.applications
        assert "editor" in env.applications


# ═══════════════════════════════════════════════════════════════════
# MissionReceipt
# ═══════════════════════════════════════════════════════════════════


class TestMissionReceipt:
    """Mission receipt serialization."""

    @pytest.fixture
    def receipt(self):
        return MissionReceipt(
            mission_id="m-001",
            receipt_id="r-001",
            status="COMPLETE",
            synthesis="Task completed successfully.",
            ihsan_score=0.97,
            snr_score=0.96,
            duration_ms=1234.5678,
            channels_executed=[
                ChannelRecord(channel="reasoning", success=True, duration_ms=800.0),
                ChannelRecord(channel="browser", success=True, duration_ms=400.0),
            ],
            execution_path=ExecutionPath.MIXED,
            wallet_delta=WalletDelta(seed=1.5, bloom=0.3),
            reflex_delta=ReflexDelta(compiled=True, compile_count=1),
            memory_delta=MemoryDelta(episodic=2, semantic=1),
            hash_chain_ref="abc123",
            action_count=3,
        )

    def test_to_dict_fields(self, receipt):
        d = receipt.to_dict()
        assert d["mission_id"] == "m-001"
        assert d["receipt_id"] == "r-001"
        assert d["status"] == "COMPLETE"
        assert d["ihsan_score"] == 0.97
        assert d["duration_ms"] == 1234.6  # rounded
        assert d["execution_path"] == "MIXED"
        assert d["action_count"] == 3
        assert d["hash_chain_ref"] == "abc123"

    def test_channels_serialized(self, receipt):
        d = receipt.to_dict()
        channels = d["channels_executed"]
        assert len(channels) == 2
        assert channels[0]["channel"] == "reasoning"
        assert channels[0]["success"] is True

    def test_wallet_delta_serialized(self, receipt):
        d = receipt.to_dict()
        assert d["wallet_delta"]["seed"] == 1.5
        assert d["wallet_delta"]["bloom"] == 0.3

    def test_reflex_delta_serialized(self, receipt):
        d = receipt.to_dict()
        assert d["reflex_delta"]["compiled"] is True
        assert d["reflex_delta"]["compile_count"] == 1
        assert d["reflex_delta"]["threshold"] == 3

    def test_memory_delta_serialized(self, receipt):
        d = receipt.to_dict()
        assert d["memory_delta"]["episodic"] == 2
        assert d["memory_delta"]["semantic"] == 1
        assert d["memory_delta"]["procedural"] == 0

    def test_default_deltas(self):
        r = MissionReceipt(
            mission_id="m-002",
            receipt_id="r-002",
            status="PARTIAL",
            synthesis="Partial result.",
            ihsan_score=0.90,
            snr_score=0.85,
            duration_ms=500.0,
            channels_executed=[],
        )
        d = r.to_dict()
        assert d["wallet_delta"]["seed"] == 0.0
        assert d["reflex_delta"]["compiled"] is False
        assert d["reflex_delta"]["threshold"] == 3
        assert d["memory_delta"]["episodic"] == 0


# ═══════════════════════════════════════════════════════════════════
# EventRecord
# ═══════════════════════════════════════════════════════════════════


class TestEventRecord:
    """Event record serialization."""

    def test_to_dict(self):
        now = time.time()
        event = EventRecord(
            event_id="evt-001",
            timestamp=now,
            category="mission.created",
            origin="terminal",
            severity=EventSeverity.INFO,
            mission_id="m-001",
            payload={"description": "test mission"},
        )
        d = event.to_dict()
        assert d["event_id"] == "evt-001"
        assert d["timestamp"] == now
        assert d["category"] == "mission.created"
        assert d["severity"] == "info"
        assert d["mission_id"] == "m-001"

    def test_default_severity(self):
        event = EventRecord(
            event_id="evt-002",
            timestamp=0.0,
            category="tick.complete",
            origin="heartbeat",
        )
        assert event.severity == EventSeverity.INFO

    def test_severity_levels(self):
        assert len(EventSeverity) == 4
        assert EventSeverity.CRITICAL.value == "critical"
        assert EventSeverity.NOTICE.value == "notice"


# ═══════════════════════════════════════════════════════════════════
# BriefingContext
# ═══════════════════════════════════════════════════════════════════


class TestBriefingContext:
    """Briefing context defaults and serialization."""

    def test_defaults(self):
        ctx = BriefingContext()
        assert ctx.quality_trend == "stable"
        assert ctx.active_project == ""
        assert ctx.near_compile_patterns == []

    def test_to_dict(self):
        ctx = BriefingContext(
            time_since_last_mission_s=3600.123,
            active_project="bizra-data-lake",
            last_mission_summary="Analyzed system health.",
            near_compile_patterns=["file_open", "search_refine"],
            quality_trend="improving",
            next_action_suggestion="Review pending PRs.",
            wallet_snapshot={"seed": 42.5, "bloom": 12.0},
        )
        d = ctx.to_dict()
        assert d["time_since_last_mission_s"] == 3600.1
        assert d["active_project"] == "bizra-data-lake"
        assert len(d["near_compile_patterns"]) == 2
        assert d["wallet_snapshot"]["seed"] == 42.5


# ═══════════════════════════════════════════════════════════════════
# ExecutionPath
# ═══════════════════════════════════════════════════════════════════


class TestExecutionPath:
    """Execution path enum."""

    def test_has_3_paths(self):
        assert len(ExecutionPath) == 3

    def test_values(self):
        assert ExecutionPath.SYSTEM_1_CACHE_HIT.value == "SYSTEM_1_CACHE_HIT"
        assert ExecutionPath.SYSTEM_2_NOVEL.value == "SYSTEM_2_NOVEL"
        assert ExecutionPath.MIXED.value == "MIXED"


# ═══════════════════════════════════════════════════════════════════
# Delta Dataclasses
# ═══════════════════════════════════════════════════════════════════


class TestDeltas:
    """WalletDelta, ReflexDelta, MemoryDelta."""

    def test_wallet_delta_defaults(self):
        d = WalletDelta()
        assert d.seed == 0.0
        assert d.bloom == 0.0

    def test_reflex_delta_defaults(self):
        d = ReflexDelta()
        assert d.compiled is False
        assert d.near_compile is False
        assert d.compile_count == 0
        assert d.threshold == 3

    def test_memory_delta_to_dict(self):
        d = MemoryDelta(episodic=5, semantic=3, procedural=1)
        assert d.to_dict() == {"episodic": 5, "semantic": 3, "procedural": 1}

    def test_channel_record_to_dict(self):
        c = ChannelRecord(channel="browser", success=False, duration_ms=123.456)
        d = c.to_dict()
        assert d["channel"] == "browser"
        assert d["success"] is False
        assert d["duration_ms"] == 123.5


# ═══════════════════════════════════════════════════════════════════
# Contract Compliance — Build Contract v1.0
# ═══════════════════════════════════════════════════════════════════


class TestBuildContractCompliance:
    """Verify types match the Locked Build Contract v1.0."""

    def test_receipt_has_all_contract_fields(self):
        """Contract §8.1: all required fields present."""
        r = MissionReceipt(
            mission_id="c-001",
            receipt_id="cr-001",
            status="BLOCKED",
            synthesis="Constitutional violation.",
            ihsan_score=0.0,
            snr_score=0.0,
            duration_ms=0.0,
            channels_executed=[],
        )
        d = r.to_dict()
        required = {
            "mission_id",
            "receipt_id",
            "status",
            "synthesis",
            "ihsan_score",
            "snr_score",
            "duration_ms",
            "channels_executed",
            "wallet_delta",
            "reflex_delta",
            "memory_delta",
            "execution_path",
            "hash_chain_ref",
        }
        assert required.issubset(d.keys()), f"Missing: {required - d.keys()}"

    def test_receipt_blocked_status(self):
        """Contract §8.1: BLOCKED is a valid status."""
        r = MissionReceipt(
            mission_id="c-002",
            receipt_id="cr-002",
            status="BLOCKED",
            synthesis="Blocked.",
            ihsan_score=0.0,
            snr_score=0.0,
            duration_ms=0.0,
            channels_executed=[],
        )
        assert r.status == "BLOCKED"

    def test_reflex_delta_contract_fields(self):
        """Contract §8.1: reflex_delta has compiled, near_compile, compile_count, threshold."""
        d = ReflexDelta(compiled=True, near_compile=True, compile_count=2, threshold=3)
        out = d.to_dict()
        assert out == {
            "compiled": True,
            "near_compile": True,
            "compile_count": 2,
            "threshold": 3,
        }

    def test_cache_hit_proof_fields(self):
        """Contract §9.4: S1 receipts include reflex proof."""
        r = MissionReceipt(
            mission_id="c-003",
            receipt_id="cr-003",
            status="COMPLETE",
            synthesis="Cache hit.",
            ihsan_score=0.98,
            snr_score=0.97,
            duration_ms=50.0,
            channels_executed=[],
            execution_path=ExecutionPath.SYSTEM_1_CACHE_HIT,
            reflex_pattern="file_open_analyze",
            reflex_latency_ms=48.3,
            comparison_s2_avg_ms=1800.0,
        )
        d = r.to_dict()
        assert d["execution_path"] == "SYSTEM_1_CACHE_HIT"
        assert d["reflex_pattern"] == "file_open_analyze"
        assert d["reflex_latency_ms"] == 48.3
        assert d["comparison_s2_avg_ms"] == 1800.0

    def test_event_severity_contract_levels(self):
        """Contract §7.1: info, notice, warning, critical."""
        levels = {s.value for s in EventSeverity}
        assert levels == {"info", "notice", "warning", "critical"}

    def test_terminal_state_machine_9_states(self):
        """Contract §5.1: exactly 9 states."""
        assert len(TerminalState) == 9

    def test_permission_envelope_contract_fields(self):
        """Contract §6.1: all schema fields present."""
        env = PermissionEnvelope()
        d = env.to_dict()
        required = {
            "filesystem",
            "applications",
            "network",
            "data_sensitivity",
            "spend_budget_usd",
            "time_budget_seconds",
            "escalation",
            "audit_verbosity",
        }
        assert required == set(d.keys())
