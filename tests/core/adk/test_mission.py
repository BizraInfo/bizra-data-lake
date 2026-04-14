"""Tests for core/adk/mission.py — Mission, Budget, governance."""

import pytest

from core.adk.mission import (
    Budget,
    BudgetExhausted,
    GovernanceClass,
    Mission,
)


def test_mission_defaults():
    m = Mission(question="What is BIZRA?")
    assert m.question == "What is BIZRA?"
    assert m.governance_class == GovernanceClass.PAT
    assert m.requester == "human"
    assert m.budget.max_tokens == 4096
    assert not m.allow_external_unverified
    assert len(m.id) == 8


def test_mission_custom_budget():
    b = Budget(max_tokens=100, max_wall_seconds=10, max_tool_calls=2, max_evidence_fetches=3)
    m = Mission(question="test", budget=b)
    assert m.budget.max_tokens == 100
    assert m.budget.max_tool_calls == 2


def test_budget_token_consumption():
    m = Mission(question="test", budget=Budget(max_tokens=10))
    m.consume_tokens(5)
    assert m._tokens_used == 5
    m.consume_tokens(5)
    assert m._tokens_used == 10


def test_budget_token_exhaustion():
    m = Mission(question="test", budget=Budget(max_tokens=10))
    m.consume_tokens(10)
    with pytest.raises(BudgetExhausted, match="tokens"):
        m.consume_tokens(1)


def test_budget_tool_call_exhaustion():
    m = Mission(question="test", budget=Budget(max_tool_calls=2))
    m.consume_tool_call()
    m.consume_tool_call()
    with pytest.raises(BudgetExhausted, match="tool_calls"):
        m.consume_tool_call()


def test_budget_exhausted_fields():
    try:
        m = Mission(question="test", budget=Budget(max_tokens=1))
        m.consume_tokens(2)
    except BudgetExhausted as e:
        assert e.kind == "tokens"
        assert e.used == 2
        assert e.limit == 1


def test_governance_class_enum():
    assert GovernanceClass.PAT.value == "PAT"
    assert GovernanceClass.SAT.value == "SAT"
    assert GovernanceClass.FROZEN.value == "FROZEN"
    assert GovernanceClass.SOVEREIGN.value == "SOVEREIGN"


def test_mission_unique_ids():
    m1 = Mission(question="a")
    m2 = Mission(question="b")
    assert m1.id != m2.id


def test_external_unverified_flag():
    m = Mission(question="test", allow_external_unverified=True)
    assert m.allow_external_unverified
