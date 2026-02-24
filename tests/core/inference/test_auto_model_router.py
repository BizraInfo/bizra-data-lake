"""Tests for AutoModelRouter — all HTTP mocked, no real LM Studio needed."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.inference.auto_model_router import _ESCALATION_MAP, AutoModelRouter

# ── Helpers ──────────────────────────────────────────────────────────


def _mock_response(status_code: int = 200, json_data: dict | None = None):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def _make_mock_client(post_resp=None, get_resp=None):
    """Build an AsyncMock httpx client with configurable responses."""
    client = AsyncMock()
    if post_resp is not None:
        client.post.return_value = post_resp
    if get_resp is not None:
        client.get.return_value = get_resp
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _router(**kwargs) -> AutoModelRouter:
    """Create a router with sensible test defaults."""
    defaults = {"base_url": "http://test:1234", "token": "tok"}
    defaults.update(kwargs)
    return AutoModelRouter(**defaults)


# ── ensure_model_loaded ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_model_loaded_success():
    router = _router()
    mock_client = _make_mock_client(post_resp=_mock_response(200))

    with patch(
        "core.inference.auto_model_router.httpx.AsyncClient", return_value=mock_client
    ):
        result = await router.ensure_model_loaded("test-model")

    assert result is True
    assert "test-model" in router._loaded_models


@pytest.mark.asyncio
async def test_ensure_model_loaded_already_loaded():
    router = _router()
    router._loaded_models.add("test-model")

    # Should NOT make any HTTP call
    result = await router.ensure_model_loaded("test-model")
    assert result is True


@pytest.mark.asyncio
async def test_ensure_model_loaded_failure():
    router = _router()
    mock_client = _make_mock_client(post_resp=_mock_response(500))

    with patch(
        "core.inference.auto_model_router.httpx.AsyncClient", return_value=mock_client
    ):
        with patch(
            "core.inference.auto_model_router.asyncio.sleep", new_callable=AsyncMock
        ):
            result = await router.ensure_model_loaded("bad-model")

    assert result is False
    assert "bad-model" not in router._loaded_models


# ── preload_mission_fleet ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preload_mission_fleet():
    router = _router()

    load_calls = []

    async def fake_ensure(model_id):
        load_calls.append(model_id)
        router._loaded_models.add(model_id)
        return True

    router.ensure_model_loaded = fake_ensure  # type: ignore[assignment]
    router._get_loaded_models = AsyncMock(return_value=set())  # type: ignore[assignment]

    config = {
        "model_routing": {
            "reasoner": "model-A",
            "thinker": "model-B",
            "creative": "model-C",
            "planner": "model-A",
        }
    }

    with patch(
        "core.inference.model_routing.resolve_model_for_agent",
    ) as mock_resolve:

        def resolve(aid, cfg):
            purposes = {
                "coordinator": "reasoner",
                "strategist": "thinker",
                "creator": "creative",
                "researcher": "reasoner",
            }
            role = purposes.get(aid, "reasoner")
            routing = cfg.get("model_routing", {})
            return routing.get(role, "model-A")

        mock_resolve.side_effect = resolve

        status = await router.preload_mission_fleet(
            ["coordinator", "strategist", "creator", "researcher"],
            config,
        )

    # 3 unique models (model-A, model-B, model-C)
    assert len(status) == 3
    assert all(v is True for v in status.values())


@pytest.mark.asyncio
async def test_preload_mission_fleet_deduplication():
    router = _router()

    load_calls = []

    async def fake_ensure(model_id):
        load_calls.append(model_id)
        router._loaded_models.add(model_id)
        return True

    router.ensure_model_loaded = fake_ensure  # type: ignore[assignment]
    router._get_loaded_models = AsyncMock(return_value=set())  # type: ignore[assignment]

    config = {}

    with patch(
        "core.inference.model_routing.resolve_model_for_agent",
        return_value="same-model",
    ):
        status = await router.preload_mission_fleet(
            ["agent1", "agent2"],
            config,
        )

    # Both agents resolve to same model -> 1 load call
    assert len(status) == 1
    assert load_calls.count("same-model") == 1


@pytest.mark.asyncio
async def test_preload_skips_already_loaded():
    router = _router()
    router._loaded_models = {"already-loaded"}

    load_calls = []

    async def fake_ensure(model_id):
        if model_id in router._loaded_models:
            return True
        load_calls.append(model_id)
        router._loaded_models.add(model_id)
        return True

    router.ensure_model_loaded = fake_ensure  # type: ignore[assignment]
    router._get_loaded_models = AsyncMock(return_value={"already-loaded"})  # type: ignore[assignment]

    with patch(
        "core.inference.model_routing.resolve_model_for_agent",
        return_value="already-loaded",
    ):
        status = await router.preload_mission_fleet(["agent1"], {})

    assert status == {"already-loaded": True}
    assert load_calls == []


# ── check_equalizer ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_equalizer_escalate():
    """High deficit + human presence -> ESCALATE."""
    from core.sovereign.equalizer_agent import EqualizerAgent

    eq = EqualizerAgent()
    router = _router(equalizer=eq)

    # Pre-seed a loaded model that's in the escalation map
    router._loaded_models = {"deepseek/deepseek-r1-0528-qwen3-8b"}

    async def fake_ensure(model_id):
        router._loaded_models.add(model_id)
        return True

    router.ensure_model_loaded = fake_ensure  # type: ignore[assignment]

    # First observation to establish history
    eq.observe(layer=0, ihsan_score=0.80, backlog=20, presence=200)

    # Second observation — accumulation + presence > threshold -> ESCALATE
    action = await router.check_equalizer(
        ihsan_score=0.78,
        backlog=25,
        presence=200,
    )

    assert action is not None
    assert "ESCALATE" in action


@pytest.mark.asyncio
async def test_check_equalizer_halt():
    """Saturation deficit + no presence -> HALT."""
    from core.sovereign.equalizer_agent import EqualizerAgent

    eq = EqualizerAgent()
    router = _router(equalizer=eq)
    router._loaded_models = {"model-a", "model-b", "model-c"}

    async def fake_unload(model_id):
        router._loaded_models.discard(model_id)
        return True

    router._unload_model = fake_unload  # type: ignore[assignment]

    # Saturation: deficit >= 26 (ihsan ~0.85) + presence=0
    action = await router.check_equalizer(
        ihsan_score=0.84,
        backlog=100,
        presence=0,
    )

    assert action is not None
    assert "HALT" in action


@pytest.mark.asyncio
async def test_check_equalizer_resume():
    """Recovery detected -> RESUME."""
    from core.sovereign.equalizer_agent import EqualizerAgent

    eq = EqualizerAgent()
    router = _router(equalizer=eq)
    router._unloaded_by_halt = {"model-x"}

    async def fake_ensure(model_id):
        router._loaded_models.add(model_id)
        return True

    router.ensure_model_loaded = fake_ensure  # type: ignore[assignment]

    # Build history: first worse, then improving (recovery)
    eq.observe(layer=0, ihsan_score=0.85, backlog=20, presence=50)
    action = await router.check_equalizer(
        ihsan_score=0.90,
        backlog=15,
        presence=50,
    )

    assert action is not None
    assert "RESUME" in action


@pytest.mark.asyncio
async def test_check_equalizer_steady():
    """Steady state -> None (no action)."""
    from core.sovereign.equalizer_agent import EqualizerAgent

    eq = EqualizerAgent()
    router = _router(equalizer=eq)

    action = await router.check_equalizer(
        ihsan_score=0.97,
        backlog=2,
        presence=50,
    )

    assert action is None


@pytest.mark.asyncio
async def test_check_equalizer_none_when_no_equalizer():
    router = _router(equalizer=None)
    action = await router.check_equalizer(0.5, 100, 255)
    assert action is None


# ── _get_loaded_models ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_loaded_models():
    router = _router()
    mock_client = _make_mock_client(
        get_resp=_mock_response(
            200,
            {
                "data": [
                    {"id": "model-a", "loaded": True},
                    {"id": "model-b", "loaded": False},
                    {"id": "model-c", "loaded": True},
                ],
            },
        ),
    )

    with patch(
        "core.inference.auto_model_router.httpx.AsyncClient", return_value=mock_client
    ):
        loaded = await router._get_loaded_models()

    assert loaded == {"model-a", "model-c"}


# ── _load_model retry ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_model_retry():
    """First attempt fails (500), retry succeeds (200)."""
    router = _router()
    call_count = 0

    def make_client(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_mock_client(post_resp=_mock_response(500))
        return _make_mock_client(post_resp=_mock_response(200))

    with patch(
        "core.inference.auto_model_router.httpx.AsyncClient", side_effect=make_client
    ):
        with patch(
            "core.inference.auto_model_router.asyncio.sleep", new_callable=AsyncMock
        ):
            result = await router._load_model("retry-model")

    assert result is True
    assert "retry-model" in router._loaded_models


# ── _unload_model ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unload_model():
    router = _router()
    router._loaded_models.add("to-unload")
    mock_client = _make_mock_client(post_resp=_mock_response(200))

    with patch(
        "core.inference.auto_model_router.httpx.AsyncClient", return_value=mock_client
    ):
        result = await router._unload_model("to-unload")

    assert result is True
    assert "to-unload" not in router._loaded_models


# ── Escalation map coverage ─────────────────────────────────────────


def test_escalation_map_coverage():
    """Every role in the escalation map has two valid model strings."""
    for role, (current, larger) in _ESCALATION_MAP.items():
        assert isinstance(role, str)
        assert isinstance(current, str) and len(current) > 0
        assert isinstance(larger, str) and len(larger) > 0
        assert current != larger, f"Role {role}: current == larger"
