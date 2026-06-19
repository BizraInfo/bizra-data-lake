"""Fail-loud regression guard for mcp_gateway handlers — Sprint A.3 (2026-04-21).

The gateway's ``handle_query`` and ``handle_ingest`` were previously TODO-backed
placeholders returning fabricated success envelopes (``{"status": "ingested"}``,
``{"response": "Query processed: ..."}``), violating ZANN_ZERO /
CLAIM_MUST_BIND at the perimeter. ``handle_health`` unconditionally reported
``data_lake: operational`` regardless of actual backend state.

This guard enforces the fail-loud contract:

1. ``handle_query`` raises HTTPException(501).
2. ``handle_ingest`` raises HTTPException(501).
3. ``handle_health`` reports honest component status — top-level "degraded",
   data_lake component is "not_wired".

If a future PR re-introduces placeholder-success returns, these tests fail.

The import side-effects of ``tools.mcp.mcp_gateway`` require ``redis`` and
``httpx``; the test module skips cleanly in environments that lack them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("redis", reason="redis not installed in CI base env")
pytest.importorskip("httpx", reason="httpx not installed in CI base env")

from fastapi import HTTPException

from tools.mcp import mcp_gateway


@pytest.mark.asyncio
async def test_handle_query_raises_501_not_implemented() -> None:
    """Query handler MUST fail loud, not fabricate a success envelope."""
    with pytest.raises(HTTPException) as exc:
        await mcp_gateway.handle_query({"query": "anything", "mode": "standard"})
    assert exc.value.status_code == 501
    assert "scaffolding" in exc.value.detail.lower() or "501" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_handle_ingest_raises_501_not_implemented() -> None:
    """Ingest handler MUST fail loud, not emit `status: ingested` without a pipeline."""
    with pytest.raises(HTTPException) as exc:
        await mcp_gateway.handle_ingest(
            {"content": "test document", "doc_type": "text"}
        )
    assert exc.value.status_code == 501
    assert "scaffolding" in exc.value.detail.lower() or "501" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_handle_health_reports_data_lake_not_wired() -> None:
    """Health handler MUST honestly report data_lake as not_wired and status degraded.

    The gateway process itself is operational (FastAPI is serving), but the
    data-lake component is not connected to a real backend. The top-level
    status must reflect that asymmetry — callers depending on real data-lake
    operations should see the degraded signal.
    """
    result = await mcp_gateway.handle_health()

    assert result["status"] == "degraded", (
        f"handle_health must report degraded status when data_lake is not wired, "
        f"got: {result!r}"
    )
    assert result["components"]["data_lake"] == "not_wired", (
        f"data_lake component must honestly report not_wired, got: {result!r}"
    )
    assert result["components"]["mcp_gateway"] == "operational", (
        "gateway process itself is operational (FastAPI is serving)"
    )
