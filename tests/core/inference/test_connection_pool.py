from __future__ import annotations

import asyncio

import pytest

from core.inference._connection_pool import (
    ConnectionPool,
    ConnectionPoolConfig,
    PooledHttpClient,
)


@pytest.mark.asyncio
async def test_acquire_create_path_does_not_deadlock() -> None:
    pool = ConnectionPool(
        backend_type="test",
        endpoint="http://localhost",
        config=ConnectionPoolConfig(
            min_size=0, max_size=1, acquisition_timeout_seconds=0.2
        ),
        connection_factory=lambda: asyncio.sleep(0, result=object()),
    )

    async def _acquire_once() -> None:
        async with pool.acquire() as (pooled, _):
            assert pooled.id
            assert pooled.in_use is True

    await asyncio.wait_for(_acquire_once(), timeout=0.5)


@pytest.mark.asyncio
async def test_timeout_does_not_over_release_semaphore() -> None:
    pool = ConnectionPool(
        backend_type="test",
        endpoint="http://localhost",
        config=ConnectionPoolConfig(
            min_size=0, max_size=1, acquisition_timeout_seconds=0.05
        ),
        connection_factory=lambda: asyncio.sleep(0, result=object()),
    )

    async with pool.acquire():
        assert pool._available._value == 0

        with pytest.raises(RuntimeError, match="acquisition timeout"):
            async with pool.acquire():
                pytest.fail("second acquire should have timed out")

        # Slot must remain fully consumed by the active acquisition.
        assert pool._available._value == 0


@pytest.mark.asyncio
async def test_pooled_http_client_preserves_original_request_error(monkeypatch) -> None:
    pool = ConnectionPool(
        backend_type="test",
        endpoint="http://localhost",
        config=ConnectionPoolConfig(
            min_size=0, max_size=1, acquisition_timeout_seconds=0.2
        ),
    )
    client = PooledHttpClient(pool=pool, base_url="http://localhost")

    async def _raise_from_executor(_executor, _fn):
        raise RuntimeError("boom")

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "run_in_executor", _raise_from_executor)

    with pytest.raises(RuntimeError, match="boom"):
        await client.request("GET", "/health")

    assert len(pool._pool) == 1
    pooled = next(iter(pool._pool.values()))
    assert pooled.is_healthy is False
