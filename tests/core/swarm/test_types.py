"""Tests for core.swarm.types — Phase 1 topology types and agent specs.

ADR-004 Phase 5 validation: 6 unit tests for types.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.swarm.types import (
    AgentRole,
    AgentSpec,
    SwarmConfig,
    SwarmEvent,
    SwarmEventKind,
    SwarmPhase,
    SwarmTopology,
)


class TestAgentSpec:
    def test_from_pat_agent_thinking(self):
        """AgentSpec.from_pat_agent converts PAT_AGENTS format correctly."""
        pat = {
            "name": "Strategist",
            "role": "Strategic planning",
            "giants": "Sun Tzu",
            "model_purpose": "thinking",
        }
        spec = AgentSpec.from_pat_agent("strategist", pat)
        assert spec.role == AgentRole.STRATEGIST
        assert spec.is_thinking_model is True
        assert spec.timeout_seconds == 120.0
        assert spec.max_tokens == 1200

    def test_from_pat_agent_non_thinking(self):
        """Non-thinking models get shorter timeouts."""
        pat = {
            "name": "Creator",
            "role": "Content creation",
            "giants": "Da Vinci",
            "model_purpose": "creative",
        }
        spec = AgentSpec.from_pat_agent("creator", pat)
        assert spec.is_thinking_model is False
        assert spec.timeout_seconds == 30.0
        assert spec.max_tokens == 600

    def test_from_pat_agent_unknown_role(self):
        """Unknown agent IDs default to COORDINATOR role."""
        pat = {
            "name": "Custom",
            "role": "Custom work",
            "giants": "N/A",
            "model_purpose": "general",
        }
        spec = AgentSpec.from_pat_agent("custom_agent", pat)
        assert spec.role == AgentRole.COORDINATOR


class TestSwarmConfig:
    def test_defaults(self):
        cfg = SwarmConfig()
        assert cfg.topology == SwarmTopology.SEQUENTIAL
        assert cfg.preload_models is True
        assert cfg.ihsan_threshold == 0.95
        assert cfg.max_concurrent == 3


class TestSwarmEvent:
    def test_immutable(self):
        evt = SwarmEvent(kind=SwarmEventKind.AGENT_STARTED, swarm_id="s1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.swarm_id = "other"  # type: ignore[misc]


class TestEnums:
    def test_agent_role_values(self):
        """All 7 roles are string enums."""
        assert len(AgentRole) == 7
        for r in AgentRole:
            assert isinstance(r.value, str)

    def test_swarm_topology_values(self):
        """3 topologies are string enums."""
        assert len(SwarmTopology) == 3
        for t in SwarmTopology:
            assert isinstance(t.value, str)

    def test_swarm_phase_values(self):
        assert len(SwarmPhase) == 8
        assert SwarmPhase.COMPLETE.value == "complete"
