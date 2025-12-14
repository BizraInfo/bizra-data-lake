import pytest, asyncio
from aegis_core.orchestrator.AEGIS_ORCHESTRATOR_V2 import AEGISOrchestrator

@pytest.mark.asyncio
async def test_pipeline_smoke():
    orch = AEGISOrchestrator()
    out = await orch.process_user_request("Design finality protocol", {"latency_ms": 500})
    assert "result" in out and "trd" in out
