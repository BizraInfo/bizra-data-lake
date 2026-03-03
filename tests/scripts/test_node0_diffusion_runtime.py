from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from scripts import node0_activate as node0


def _base_metrics() -> dict:
    return {
        "cycles": 0,
        "missions_completed": 0,
        "tokens_used": 0,
        "ihsan_score": 0.95,
        "diffusion_total": 0,
        "diffusion_active": 0,
        "diffusion_inactive": 0,
        "diffusion_high_confidence": 0,
        "diffusion_elite_confidence": 0,
        "diffusion_confidence_total": 0.0,
        "diffusion_avg_confidence": 0.0,
        "diffusion_last_state": "unknown",
        "diffusion_last_focus": "baseline",
    }


def _kernel_stub() -> node0.Node0ProactiveKernel:
    kernel = node0.Node0ProactiveKernel.__new__(node0.Node0ProactiveKernel)
    kernel._metrics = _base_metrics()
    return kernel


def test_record_diffusion_metrics_tracks_activation_and_confidence_bands() -> None:
    kernel = _kernel_stub()
    kernel._record_diffusion_metrics(
        {
            "amplifier": {
                "activated": True,
                "confidence": 0.99,
                "predicted_state": "analyzing",
                "focus": "verify",
            }
        }
    )
    assert kernel._metrics["diffusion_total"] == 1
    assert kernel._metrics["diffusion_active"] == 1
    assert kernel._metrics["diffusion_inactive"] == 0
    assert kernel._metrics["diffusion_high_confidence"] == 1
    assert kernel._metrics["diffusion_elite_confidence"] == 1
    assert kernel._metrics["diffusion_avg_confidence"] == 0.99
    assert kernel._metrics["diffusion_last_state"] == "analyzing"
    assert kernel._metrics["diffusion_last_focus"] == "verify"


def test_record_diffusion_metrics_fail_closed_when_context_missing() -> None:
    kernel = _kernel_stub()
    kernel._record_diffusion_metrics({})
    assert kernel._metrics["diffusion_total"] == 1
    assert kernel._metrics["diffusion_active"] == 0
    assert kernel._metrics["diffusion_inactive"] == 1
    assert kernel._metrics["diffusion_high_confidence"] == 0
    assert kernel._metrics["diffusion_elite_confidence"] == 0
    assert kernel._metrics["diffusion_avg_confidence"] == 0.0
    assert kernel._metrics["diffusion_last_state"] == "unknown"
    assert kernel._metrics["diffusion_last_focus"] == "baseline"


def test_summarize_diffusion_receipts_aggregates_recent_receipts() -> None:
    summary = node0._summarize_diffusion_receipts(
        [
            {
                "snr": {
                    "diffusion": {
                        "activated": True,
                        "confidence": 0.95,
                        "predicted_state": "analyzing",
                        "focus": "verify",
                    }
                }
            },
            {
                "snr": {
                    "diffusion": {
                        "activated": False,
                        "confidence": 0.40,
                        "predicted_state": "idle",
                        "focus": "stabilize",
                    }
                }
            },
        ]
    )
    assert summary["total"] == 2
    assert summary["active"] == 1
    assert summary["inactive"] == 1
    assert summary["high_confidence"] == 1
    assert summary["elite_confidence"] == 0
    assert summary["avg_confidence"] == 0.675
    assert summary["activation_rate"] == 0.5
    assert summary["last_state"] == "idle"
    assert summary["last_focus"] == "stabilize"


def test_emit_verified_receipt_forwards_diffusion_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_emit_receipt(*args, **kwargs):
        captured["snr_trace"] = kwargs.get("snr_trace", {})
        return SimpleNamespace(
            entry_hash="a" * 64,
            sequence=9,
            prev_hash="b" * 64,
            timestamp="2026-03-01T00:00:00+00:00",
        )

    monkeypatch.setattr(node0, "_EVIDENCE_LEDGER", object())
    import core.proof_engine.evidence_ledger as evidence_ledger

    monkeypatch.setattr(evidence_ledger, "emit_receipt", fake_emit_receipt)

    mission = {"id": "m-1", "description": "secure transport hardening"}
    result = {"agents": [{"agent": "strategist"}], "total_tokens": 100, "duration_ms": 23}
    snr_data = {
        "snr_score": 0.94,
        "ihsan_score": 0.96,
        "method": "facade_v2",
        "diffusion": {"activated": True, "confidence": 0.93},
    }

    receipt = node0._emit_verified_receipt(mission, result, snr_data)
    assert captured["snr_trace"]["diffusion"]["activated"] is True
    assert receipt["chain_seq"] == 9
    assert receipt["snr_method"] == "facade_v2"


@pytest.mark.asyncio
async def test_cmd_status_surfaces_diffusion_from_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"data": []}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    import core.sovereign.event_bus as event_bus
    from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt

    monkeypatch.setattr(event_bus, "create_rust_event_bridge", lambda production=False: None)
    monkeypatch.setattr(node0, "PROJECT_ROOT", str(tmp_path))

    state_dir = tmp_path / "sovereign_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = EvidenceLedger(state_dir / "evidence.jsonl", validate_on_append=True)

    emit_receipt(
        ledger,
        receipt_id="a1b2c3d4",
        node_id="node0-test",
        snr_score=0.96,
        ihsan_score=0.97,
        seal_digest="a" * 64,
        snr_trace={
            "diffusion": {
                "activated": True,
                "confidence": 0.95,
                "predicted_state": "analyzing",
                "focus": "verify",
            }
        },
    )
    emit_receipt(
        ledger,
        receipt_id="b1c2d3e4",
        node_id="node0-test",
        snr_score=0.90,
        ihsan_score=0.92,
        seal_digest="b" * 64,
        snr_trace={
            "diffusion": {
                "activated": False,
                "confidence": 0.40,
                "predicted_state": "idle",
                "focus": "stabilize",
            }
        },
    )

    await node0.cmd_status(SimpleNamespace())
    out = capsys.readouterr().out

    assert "Diffusion:" in out
    assert "1/2 active (50%)" in out
    assert "avg=0.675" in out
    assert "state=idle | focus=stabilize" in out


@pytest.mark.asyncio
async def test_execute_mission_updates_diffusion_metrics_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "choices": [{"message": {"content": "agent output"}}],
                "usage": {"total_tokens": 12},
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    class FakeKnowledge:
        _model = None

        @staticmethod
        def retrieve(query: str) -> str:
            return ""

    class FakeRouter:
        @staticmethod
        async def preload_mission_fleet(agent_ids, config):
            return {a: True for a in agent_ids}

        @staticmethod
        async def check_equalizer(ihsan_score, backlog, presence):
            return None

    async def fake_got(mission_desc, agent_results, diffusion_context=None):
        return {
            "conclusion": "synthesized conclusion",
            "thought_count": 1,
            "reasoning_paths": 1,
            "snr_score": 0.9,
            "llm_used": False,
            "thought_chain": [],
        }

    def fake_snr(query, agent_outputs, model=None, diffusion_context=None):
        return {
            "snr_score": 0.92,
            "ihsan_score": 0.93,
            "method": "snr_v2_lexical",
            "diffusion": (diffusion_context or {}).get("amplifier", {}),
        }

    def fake_receipt(mission, result, snr_data):
        return {"hash": "abc123", "chain_seq": 1, "snr_method": snr_data["method"]}

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(node0, "_synthesize_with_got", fake_got)
    monkeypatch.setattr(node0, "_compute_real_snr", fake_snr)
    monkeypatch.setattr(node0, "_emit_verified_receipt", fake_receipt)
    monkeypatch.setattr(node0, "_CHANNEL_DISPATCHER", None)
    monkeypatch.setattr(node0, "_RLM_BRIDGE", None)
    monkeypatch.setattr(node0, "_AGENT_STRATEGIES", {"strategist": SimpleNamespace(use_rlm=False)})
    monkeypatch.setattr(
        node0,
        "_build_diffusion_query_context",
        lambda q: {
            "query": "[DIFFUSION_CONTEXT]\\nstate=analyzing\\n[/DIFFUSION_CONTEXT]\\n" + q,
            "amplifier": {
                "activated": True,
                "confidence": 0.95,
                "predicted_state": "analyzing",
                "focus": "verify",
            },
            "router": {"diffusion_active": True, "got_depth": 3},
        },
    )

    kernel = node0.Node0ProactiveKernel.__new__(node0.Node0ProactiveKernel)
    kernel._knowledge = FakeKnowledge()
    kernel.token = ""
    kernel.base_url = "http://localhost:1234"
    kernel._backend_name = "lm_studio"
    kernel._yaml_config = {}
    kernel._model_router = FakeRouter()
    kernel._equalizer = None
    kernel._strategy_memory = {}
    kernel._token_minter = None
    kernel._emission_gate = None
    kernel._receipts = []
    kernel._metrics = _base_metrics()

    mission = {"id": "mission-1", "description": "analyze secure transport posture"}
    result = await kernel._execute_mission(mission, ["strategist"])

    assert result["diffusion"]["activated"] is True
    assert result["receipt"]["hash"] == "abc123"
    assert kernel._metrics["diffusion_total"] == 1
    assert kernel._metrics["diffusion_active"] == 1
    assert kernel._metrics["diffusion_high_confidence"] == 1
    assert kernel._metrics["tokens_used"] == 12
