import asyncio
import json

import pytest

from core.pek.kernel import (
    PEKProposal,
    ProactiveExecutionKernel,
    ProactiveExecutionKernelConfig,
)


class DummyPipeline:
    def __init__(self, queue_utilization: float = 0.0, pending_approval: int = 0):
        self.submitted = []
        self.stopped = False
        self._queue_utilization = queue_utilization
        self._pending_approval = pending_approval

    def stats(self):
        return {
            "queue_utilization": self._queue_utilization,
            "queue_size": 3,
            "pending_approval": self._pending_approval,
            "active_count": 1,
        }

    async def submit(self, opportunity):
        self.submitted.append(opportunity)

    async def stop(self):
        self.stopped = True


class DummyGateway:
    def __init__(self, status: str = "offline", avg_latency_ms: float = 0.0):
        self.status = status
        self.avg_latency_ms = avg_latency_ms

    async def health(self):
        return {
            "status": self.status,
            "active_backend": None,
            "stats": {
                "avg_latency_ms": self.avg_latency_ms,
                "total_requests": 4,
            },
        }


class DummyMemoryStats:
    def __init__(self, total_entries: int = 10, avg_snr: float = 0.9, avg_ihsan: float = 0.97):
        self._total_entries = total_entries
        self._avg_snr = avg_snr
        self._avg_ihsan = avg_ihsan

    def to_dict(self):
        return {
            "total_entries": self._total_entries,
            "avg_snr": self._avg_snr,
            "avg_ihsan": self._avg_ihsan,
        }


class DummyMemory:
    def __init__(self, total_entries: int = 10, avg_snr: float = 0.9, avg_ihsan: float = 0.97):
        self._stats = DummyMemoryStats(
            total_entries=total_entries,
            avg_snr=avg_snr,
            avg_ihsan=avg_ihsan,
        )

    def get_stats(self):
        return self._stats

    def get_working_context(self, max_entries: int = 5):
        del max_entries
        return ""


class DummyEventBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_pek_generates_expected_proposals():
    pipeline = DummyPipeline(queue_utilization=0.76, pending_approval=2)
    gateway = DummyGateway(status="offline", avg_latency_ms=2100)
    memory = DummyMemory(total_entries=20, avg_snr=0.93)

    kernel = ProactiveExecutionKernel(
        opportunity_pipeline=pipeline,  # type: ignore[arg-type]
        inference_gateway=gateway,
        living_memory=memory,
        config=ProactiveExecutionKernelConfig(max_proposals_per_cycle=5),
    )

    signals = await kernel._collect_signals()
    tau = kernel._compute_tau(signals)
    proposals = await kernel._generate_proposals(signals, tau)

    assert tau < kernel.config.base_tau
    assert any(p.action_type == "backend_recovery_probe" for p in proposals)
    assert any(p.action_type == "queue_pressure_relief" for p in proposals)
    assert any(p.action_type == "refresh_working_set" for p in proposals)


@pytest.mark.asyncio
async def test_pek_loop_writes_proofs_and_dispatches(tmp_path):
    pipeline = DummyPipeline(queue_utilization=0.8, pending_approval=1)
    gateway = DummyGateway(status="offline", avg_latency_ms=1500)
    memory = DummyMemory(total_entries=15, avg_snr=0.9)

    config = ProactiveExecutionKernelConfig(
        cycle_interval_seconds=0.05,
        max_proposals_per_cycle=1,
        min_confidence=0.40,
        auto_execute_tau=0.95,
    )
    kernel = ProactiveExecutionKernel(
        opportunity_pipeline=pipeline,  # type: ignore[arg-type]
        inference_gateway=gateway,
        living_memory=memory,
        state_dir=tmp_path,
        config=config,
    )

    await kernel.start()
    await asyncio.sleep(0.20)
    await kernel.stop()

    assert pipeline.stopped
    assert len(pipeline.submitted) >= 1

    proof_log = tmp_path / config.proof_log_relpath
    assert proof_log.exists()

    lines = proof_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert "proposal_id" in record
    assert record["decision"] in {
        "auto_execute",
        "propose",
        "queue_silent",
        "ignore",
        "reject",
    }


@pytest.mark.asyncio
async def test_pek_budget_degrades_to_queue_silent():
    pipeline = DummyPipeline(queue_utilization=0.0, pending_approval=0)
    config = ProactiveExecutionKernelConfig(
        min_confidence=0.10,
        min_auto_confidence=0.10,
        auto_execute_tau=0.70,
        attention_budget_capacity=0.05,
        attention_cost_auto_execute=0.40,
        attention_cost_propose=0.20,
        attention_cost_queue_silent=0.02,
    )
    kernel = ProactiveExecutionKernel(
        opportunity_pipeline=pipeline,  # type: ignore[arg-type]
        config=config,
    )

    proposal = PEKProposal(
        id="pek-budget-1",
        domain="inference",
        action_type="backend_recovery_probe",
        description="Recover backend",
        snr_score=0.95,
        ihsan_score=0.98,
        urgency=0.9,
        estimated_value=0.9,
        risk=0.2,
    )
    kernel._attention_budget = 0.05
    proof = await kernel._evaluate_and_dispatch(
        proposal=proposal,
        tau=0.90,
        signals={"pipeline": {}},
    )

    assert proof.decision == "queue_silent"
    assert len(pipeline.submitted) == 1
    assert pipeline.submitted[0].autonomy_level.name == "OBSERVER"
    assert kernel.stats()["metrics"]["budget_exhausted"] >= 1


@pytest.mark.asyncio
async def test_pek_budget_exhaustion_can_ignore():
    pipeline = DummyPipeline(queue_utilization=0.0, pending_approval=0)
    config = ProactiveExecutionKernelConfig(
        min_confidence=0.10,
        min_auto_confidence=0.10,
        auto_execute_tau=0.70,
        attention_budget_capacity=0.01,
        attention_cost_auto_execute=0.40,
        attention_cost_propose=0.20,
        attention_cost_queue_silent=0.02,
    )
    kernel = ProactiveExecutionKernel(
        opportunity_pipeline=pipeline,  # type: ignore[arg-type]
        config=config,
    )

    proposal = PEKProposal(
        id="pek-budget-2",
        domain="inference",
        action_type="backend_recovery_probe",
        description="Recover backend",
        snr_score=0.95,
        ihsan_score=0.98,
        urgency=0.9,
        estimated_value=0.9,
        risk=0.2,
    )
    kernel._attention_budget = 0.01
    proof = await kernel._evaluate_and_dispatch(
        proposal=proposal,
        tau=0.90,
        signals={"pipeline": {}},
    )

    assert proof.decision == "ignore"
    assert len(pipeline.submitted) == 0
    assert kernel.stats()["metrics"]["budget_exhausted"] >= 1


@pytest.mark.asyncio
async def test_pek_proof_event_emission(tmp_path):
    pipeline = DummyPipeline(queue_utilization=0.0, pending_approval=0)
    bus = DummyEventBus()
    config = ProactiveExecutionKernelConfig(
        cycle_interval_seconds=0.05,
        min_confidence=0.10,
        emit_proof_events=True,
        proof_event_topic="pek.proof.test",
    )
    kernel = ProactiveExecutionKernel(
        opportunity_pipeline=pipeline,  # type: ignore[arg-type]
        state_dir=tmp_path,
        config=config,
        event_bus=bus,
    )

    proposal = PEKProposal(
        id="pek-proof-event-1",
        domain="memory",
        action_type="refresh_working_set",
        description="Refresh memory context",
        snr_score=0.90,
        ihsan_score=0.96,
        urgency=0.5,
        estimated_value=0.75,
        risk=0.1,
    )
    proof = await kernel._evaluate_and_dispatch(
        proposal=proposal,
        tau=0.60,
        signals={"memory": {}},
    )
    await kernel._append_proof_block(proof)

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.topic == "pek.proof.test"
    assert event.payload["proof"]["proposal_id"] == "pek-proof-event-1"
