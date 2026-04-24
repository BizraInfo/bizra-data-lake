"""Tests for SAT-5 Composite Evaluator."""

import pytest

from core.sat.composite_evaluator import (
    CompositeVerdict,
    evaluate_all_gates,
    evaluate_gates_for_receipt,
)


@pytest.mark.slow
@pytest.mark.xdist_group("runtime_heavy")
def test_composite_evaluator_returns_verdict():
    verdict = evaluate_all_gates(skip_slow=True, skip_manual=True)
    assert isinstance(verdict, CompositeVerdict)
    assert isinstance(verdict.passed, bool)
    assert isinstance(verdict.gate_results, dict)
    assert len(verdict.gate_results) == 5  # all 5 gates ran


def test_all_five_gates_present():
    verdict = evaluate_all_gates(skip_slow=True, skip_manual=True)
    expected = {"sentinel", "oracle_s", "ledger", "conductor", "ambassador"}
    assert set(verdict.gate_results.keys()) == expected


def test_verdict_has_reason():
    verdict = evaluate_all_gates(skip_slow=True, skip_manual=True)
    assert verdict.reason  # non-empty


def test_verdict_to_dict():
    verdict = evaluate_all_gates(skip_slow=True, skip_manual=True)
    d = verdict.to_dict()
    assert "passed" in d
    assert "gate_results" in d
    assert "blocking_gates" in d
    assert "ihsan_score" in d


def test_evaluate_gates_for_receipt():
    verdict = evaluate_gates_for_receipt(
        pat_answer="The Spearpoint seal is commit b08f2208",
        evidence_refs=["git-show:b08f2208"],
        skip_slow=True,
    )
    assert isinstance(verdict, CompositeVerdict)
    assert isinstance(verdict.ihsan_score, float)


def test_blocking_gates_list():
    verdict = evaluate_all_gates(skip_slow=True, skip_manual=True)
    if verdict.passed:
        assert verdict.blocking_gates == []
    else:
        assert len(verdict.blocking_gates) > 0
