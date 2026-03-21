from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.sovereign import __main__ as cli_main


@pytest.mark.asyncio
async def test_run_mission_uses_runtime_mission_and_emits_canonical_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = MagicMock()
    runtime.mission = AsyncMock(
        return_value=SimpleNamespace(
            mission_id="cli-mission-001",
            output_text="Canonical CLI mission complete.",
            system="S1",
            ihsan_score=0.98,
            snr_score=0.96,
            duration_ms=0.42,
            fate_verdict="approved",
            fate_reason_codes=[],
            fate_mode="enforced",
            identity_mode="genesis_ed25519",
            signer_public_key_prefix="abcd1234efgh5678",
            chain_hash="f" * 64,
        )
    )

    @asynccontextmanager
    async def _fake_create(config: object):
        del config
        yield runtime

    monkeypatch.setattr(
        "core.sovereign.runtime.SovereignRuntime.create",
        staticmethod(lambda config: _fake_create(config)),
    )

    await cli_main.run_mission("close the canonical loop", json_output=True)
    payload = json.loads(capsys.readouterr().out)

    runtime.mission.assert_awaited_once_with(
        "close the canonical loop",
        source="cli",
        context={},
    )
    assert payload["execution_authority"] == "organism"
    assert payload["authority_path"] == "runtime->organism->node0"
    assert payload["identity_mode"] == "genesis_ed25519"
    assert payload["signer_public_key_prefix"] == "abcd1234efgh5678"
    assert payload["fate_verdict"] == "approved"
    assert payload["hash_chain_ref"] == "f" * 64


@pytest.mark.asyncio
async def test_run_server_forwards_autopoiesis_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve_mock = AsyncMock()
    monkeypatch.setattr("core.sovereign.api.serve", serve_mock)

    await cli_main.run_server(
        "127.0.0.1",
        8080,
        ["k1"],
        enable_autopoiesis=True,
        autopoiesis_cycle_seconds=12.5,
    )

    serve_mock.assert_awaited_once_with(
        "127.0.0.1",
        8080,
        ["k1"],
        enable_autopoiesis=True,
        autopoiesis_cycle_seconds=12.5,
    )
