from __future__ import annotations

import json
import sys
import types

import scripts.ci_canonical_e2e_benchmark as benchmark


class _DummyBreath:
    pass


class _ConfigurableHeartbeat:
    def __init__(self) -> None:
        self._event_bus = object()
        self.calls: list[str] = []

    def _boot_reflex_bridge(self) -> None:
        self.calls.append("_boot_reflex_bridge")

    def _boot_reasoning_bank(self) -> None:
        self.calls.append("_boot_reasoning_bank")

    def _boot_learning_loop(self) -> None:
        self.calls.append("_boot_learning_loop")

    def _boot_federation_ambassador(self) -> None:
        self.calls.append("_boot_federation_ambassador")

    def _record_rb_experience(self, helix_result: dict) -> None:
        self.calls.append("_record_rb_experience")

    def _run_learning_cycle(self, helix_result: dict) -> None:
        self.calls.append("_run_learning_cycle")

    def _contribute_urp_witness(self, receipt: object) -> None:
        self.calls.append("_contribute_urp_witness")

    def _check_reflex_precipitation(self, helix_result: dict) -> int:
        self.calls.append("_check_reflex_precipitation")
        return 7


class _DummyHeartbeat:
    def __init__(self) -> None:
        self._event_bus = None

    def breathe(self) -> _DummyBreath:
        return _DummyBreath()

    def _emit_breath_event(self, receipt: _DummyBreath) -> None:
        if self._event_bus is not None:
            self._event_bus.publish("action.receipt", {"source": "test"})


def test_benchmark_node0_measures_direct_event_emission(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_boot_node0",
        lambda _tmpdir: _DummyHeartbeat(),
    )

    result = benchmark.benchmark_node0()

    assert result["node0_available"] is True
    assert result["breath_chain_valid"] is True
    assert result["events_emitted"] == 1
    assert result["eventbus_emission_ms"] >= 0.0


def test_benchmark_got_bridge_falls_back_to_async_reason(monkeypatch) -> None:
    class _DummyResult:
        converged = True

    class _DummyBridge:
        async def reason(self, query: str, context: dict | None = None) -> _DummyResult:
            return _DummyResult()

    monkeypatch.setitem(
        sys.modules,
        "core.reasoning.got_bridge",
        types.SimpleNamespace(GoTBridge=_DummyBridge),
    )

    result = benchmark.benchmark_got_bridge()

    assert result["got_bridge_available"] is True
    assert result["got_bridge_mode"] == "async_reason"
    assert result["got_bridge_converged"] is True
    assert result["got_bridge_reason_ms"] >= 0.0
    assert result["got_bridge_import_ms"] >= 0.0


def test_configure_benchmark_heartbeat_disables_optional_sidecars() -> None:
    heartbeat = _ConfigurableHeartbeat()

    benchmark._configure_benchmark_heartbeat(heartbeat)

    assert heartbeat._check_reflex_precipitation({}) == 0
    heartbeat._boot_reflex_bridge()
    heartbeat._boot_reasoning_bank()
    heartbeat._boot_learning_loop()
    heartbeat._boot_federation_ambassador()
    heartbeat._record_rb_experience({})
    heartbeat._run_learning_cycle({})
    heartbeat._contribute_urp_witness(object())

    assert heartbeat.calls == []


def test_run_benchmark_writes_report_even_when_gate_fails(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "benchmark_got_bridge",
        lambda: {"got_bridge_init_ms": 999.0, "got_bridge_reason_ms": 0.0},
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_vrg_receipt",
        lambda: {"vrg_receipt_build_ms": 0.0},
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_node0",
        lambda: {
            "organism_boot_ms": 0.0,
            "node0_breathe_ms": 0.0,
            "eventbus_emission_ms": 0.0,
        },
    )

    output = tmp_path / "canonical_e2e_report.json"
    exit_code = benchmark.run_benchmark(output=output)

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmark"] == "canonical_e2e"
    assert report["gate_verdict"]["passed"] is False
    assert "got_bridge_init_ms" in report["gate_verdict"]["failed_metrics"]
