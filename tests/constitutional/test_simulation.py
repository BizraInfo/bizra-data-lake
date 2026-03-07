from __future__ import annotations

from core.constitutional.fixed_point import fp_float
from core.constitutional.simulation import render_simulation_report, run_simulation


def test_simulation_verifies_declaration_and_chains() -> None:
    report = run_simulation(num_nodes=8, days=12, seed=7)

    assert report.declaration_verified is True
    assert report.covenant_chain_valid is True
    assert report.event_chain_valid is True
    assert report.event_count > 1
    assert report.total_actions > 0


def test_simulation_is_deterministic_for_same_inputs() -> None:
    first = run_simulation(num_nodes=10, days=20, seed=9)
    second = run_simulation(num_nodes=10, days=20, seed=9)

    assert first.to_dict() == second.to_dict()


def test_simulation_generates_nontrivial_network_signals() -> None:
    report = run_simulation(num_nodes=12, days=60, seed=5)
    rendered = render_simulation_report(report)

    assert report.total_attestations > 0
    assert report.reflexes_compiled > 0
    assert report.network_gini >= 0
    assert fp_float(report.network_asabiyyah) >= 0.0
    assert any(milestone.day == 60 for milestone in report.milestone_reports)
    assert "Declaration hash verified: yes" in rendered
    assert "Milestones" in rendered
