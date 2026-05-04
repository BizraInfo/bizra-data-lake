import pytest

from core.inference.model_broker import (
    EXTERNAL,
    EXTERNAL_ALLOWED,
    FREE_LOCAL,
    HIGH_COST,
    InferenceMessage,
    InferenceRequest,
    InferenceRequestSpec,
    LOCAL_ONLY,
    LOCAL_PRIVATE,
    MEDIUM_COST,
    ModelBroker,
    ModelCapabilityCard,
    ModelRouteError,
    OFFLINE,
    ONLINE,
    OllamaProviderAdapter,
    OpenAICompatibleProviderAdapter,
    PROVIDER_ALLOWED,
    StaticProviderAdapter,
    create_default_model_broker,
)


def _card(
    provider: str,
    model: str,
    *,
    capabilities: frozenset[str] = frozenset({"text", "reasoning"}),
    privacy_tier: str = LOCAL_PRIVATE,
    context_window: int = 32_000,
    cost_tier: str = FREE_LOCAL,
    latency_tier: str = "medium",
    estimated_latency_ms: int | None = None,
    health: str = ONLINE,
    endpoint: str = "http://127.0.0.1:1234/v1",
) -> ModelCapabilityCard:
    return ModelCapabilityCard(
        provider=provider,
        model_id=model,
        endpoint=endpoint,
        capabilities=capabilities,
        context_window=context_window,
        privacy_tier=privacy_tier,
        cost_tier=cost_tier,
        latency_tier=latency_tier,
        estimated_latency_ms=estimated_latency_ms,
        health=health,
    )


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(
        self,
        *,
        get_payload: dict | None = None,
        post_payload: dict | None = None,
    ) -> None:
        self.get_payload = get_payload or {}
        self.post_payload = post_payload or {}
        self.get_calls: list[dict] = []
        self.post_calls: list[dict] = []

    async def get(self, url: str, **kwargs) -> _FakeResponse:
        self.get_calls.append({"url": url, **kwargs})
        return _FakeResponse(200, self.get_payload)

    async def post(self, url: str, **kwargs) -> _FakeResponse:
        self.post_calls.append({"url": url, **kwargs})
        return _FakeResponse(200, self.post_payload)


@pytest.mark.asyncio
async def test_external_provider_is_not_discovered_by_default() -> None:
    local = StaticProviderAdapter(
        name="lm-studio",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("lm-studio", "qwen-local")],
    )
    external = StaticProviderAdapter(
        name="openai",
        privacy_tier=EXTERNAL,
        models=[
            _card(
                "openai",
                "gpt-provider",
                privacy_tier=EXTERNAL,
                endpoint="https://api.openai.example/v1",
            )
        ],
    )

    decision = await ModelBroker([local, external]).route(
        spec=InferenceRequestSpec(task_type="reasoning")
    )

    assert decision.selected.provider == "lm-studio"
    assert local.discovery_calls == 1
    assert external.discovery_calls == 0
    assert decision.rejected["openai"] == ("external_providers_disabled",)


@pytest.mark.asyncio
async def test_local_only_policy_never_selects_external_even_when_enabled() -> None:
    local = StaticProviderAdapter(
        name="ollama",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("ollama", "llama-local", latency_tier="high")],
    )
    external = StaticProviderAdapter(
        name="anthropic",
        privacy_tier=EXTERNAL,
        models=[
            _card(
                "anthropic",
                "claude-provider",
                privacy_tier=EXTERNAL,
                endpoint="https://api.anthropic.example/v1",
                latency_tier="low",
            )
        ],
    )

    decision = await ModelBroker(
        [local, external],
        allow_external_providers=True,
    ).route(InferenceRequestSpec(task_type="reasoning", privacy_required=LOCAL_ONLY))

    assert decision.selected.provider == "ollama"
    assert decision.rejected["anthropic"] == ("privacy_external_blocked",)


@pytest.mark.asyncio
async def test_external_provider_can_be_selected_with_explicit_external_policy() -> (
    None
):
    local = StaticProviderAdapter(
        name="ollama",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("ollama", "small-local", context_window=4_096)],
    )
    external = StaticProviderAdapter(
        name="openai",
        privacy_tier=EXTERNAL,
        models=[
            _card(
                "openai",
                "large-provider",
                privacy_tier=EXTERNAL,
                endpoint="https://api.openai.example/v1",
                context_window=128_000,
                cost_tier=MEDIUM_COST,
                latency_tier="low",
            )
        ],
    )

    decision = await ModelBroker(
        [local, external],
        allow_external_providers=True,
    ).route(
        InferenceRequestSpec(
            task_type="reasoning",
            privacy_required=EXTERNAL_ALLOWED,
            min_context=64_000,
            max_cost_tier=HIGH_COST,
        )
    )

    assert decision.selected.provider == "openai"
    assert decision.rejected["ollama"] == ("context_window_too_small",)


@pytest.mark.asyncio
async def test_missing_capability_fails_closed_with_reason() -> None:
    provider = StaticProviderAdapter(
        name="lm-studio",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("lm-studio", "text-only", capabilities=frozenset({"text"}))],
    )

    with pytest.raises(ModelRouteError, match="missing_capabilities:vision"):
        await ModelBroker([provider]).route(InferenceRequestSpec(task_type="vision"))


@pytest.mark.asyncio
async def test_provider_private_requires_provider_allowed_policy() -> None:
    provider = StaticProviderAdapter(
        name="azure-openai",
        privacy_tier="provider_private",
        models=[
            _card(
                "azure-openai",
                "gpt-private",
                privacy_tier="provider_private",
                endpoint="https://private.openai.azure.example/v1",
            )
        ],
    )

    with pytest.raises(ModelRouteError, match="privacy_provider_private_blocked"):
        await ModelBroker([provider]).route(
            InferenceRequestSpec(task_type="reasoning", privacy_required=LOCAL_ONLY)
        )

    decision = await ModelBroker([provider]).route(
        InferenceRequestSpec(
            task_type="reasoning",
            privacy_required=PROVIDER_ALLOWED,
        )
    )
    assert decision.selected.provider == "azure-openai"


@pytest.mark.asyncio
async def test_deterministic_tie_break_uses_fallback_order_then_model_id() -> None:
    first = StaticProviderAdapter(
        name="lm-studio",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("lm-studio", "zeta"), _card("lm-studio", "alpha")],
    )
    second = StaticProviderAdapter(
        name="ollama",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("ollama", "beta")],
    )

    decision = await ModelBroker(
        [first, second],
        fallback_order=("ollama", "lm-studio"),
    ).route(InferenceRequestSpec(task_type="reasoning"))

    assert decision.selected.provider == "ollama"
    assert decision.selected.model_id == "beta"
    assert [card.model_id for card in decision.fallback_chain] == ["alpha", "zeta"]


@pytest.mark.asyncio
async def test_unhealthy_models_are_rejected() -> None:
    provider = StaticProviderAdapter(
        name="lm-studio",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("lm-studio", "offline-model", health=OFFLINE)],
    )

    with pytest.raises(ModelRouteError, match="model_offline"):
        await ModelBroker([provider]).route(InferenceRequestSpec(task_type="reasoning"))


@pytest.mark.asyncio
async def test_routing_receipt_redacts_endpoint_and_token_metadata() -> None:
    token = "sk-local-never-log-this"
    provider = StaticProviderAdapter(
        name="lm-studio",
        privacy_tier=LOCAL_PRIVATE,
        models=[
            _card(
                "lm-studio",
                "qwen-local",
                endpoint=f"http://127.0.0.1:1234/v1?api_key={token}",
            )
        ],
    )

    decision = await ModelBroker([provider]).route(
        InferenceRequestSpec(task_type="reasoning")
    )
    receipt_text = repr(decision.to_receipt_dict())

    assert "endpoint_class" in receipt_text
    assert "loopback" in receipt_text
    assert token not in receipt_text
    assert "api_key" not in receipt_text


@pytest.mark.asyncio
async def test_latency_and_historical_ihsan_are_fail_closed_constraints() -> None:
    provider = StaticProviderAdapter(
        name="lm-studio",
        privacy_tier=LOCAL_PRIVATE,
        models=[
            _card(
                "lm-studio",
                "slow-model",
                latency_tier="high",
                estimated_latency_ms=800,
            )
        ],
    )

    with pytest.raises(ModelRouteError, match="latency_exceeds_policy"):
        await ModelBroker([provider]).route(
            InferenceRequestSpec(task_type="reasoning", max_latency_ms=200)
        )

    with pytest.raises(ModelRouteError, match="ihsan_below_policy"):
        await ModelBroker(
            [provider],
            historical_ihsan={("lm-studio", "slow-model"): 0.5},
        ).route(InferenceRequestSpec(task_type="reasoning"))


@pytest.mark.asyncio
async def test_broker_generate_routes_then_invokes_selected_adapter() -> None:
    provider = StaticProviderAdapter(
        name="lm-studio",
        privacy_tier=LOCAL_PRIVATE,
        models=[_card("lm-studio", "qwen-local")],
    )

    result = await ModelBroker([provider]).generate(
        InferenceRequest(
            messages=(InferenceMessage(role="user", content="hello"),),
            spec=InferenceRequestSpec(task_type="reasoning"),
        )
    )

    assert provider.generate_calls == 1
    assert result.provider == "lm-studio"
    assert result.model == "qwen-local"
    assert result.routing_receipt["provider"] == "lm-studio"


@pytest.mark.asyncio
async def test_openai_compatible_adapter_discovers_models_without_exposing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-test-token")
    client = _FakeClient(
        get_payload={"data": [{"id": "gpt-4.1-128k"}, {"id": "text-embedding-3"}]}
    )
    adapter = OpenAICompatibleProviderAdapter(
        name="openai",
        base_url="https://api.openai.example/v1",
        privacy_tier=EXTERNAL,
        token_env="OPENAI_API_KEY",
        requires_token=True,
        client=client,
    )

    cards = await adapter.list_models()

    assert [card.model_id for card in cards] == ["gpt-4.1-128k", "text-embedding-3"]
    assert cards[0].context_window == 128_000
    assert "embedding" in cards[1].capabilities
    assert client.get_calls[0]["headers"]["Authorization"].startswith("Bearer ")
    assert "sk-local-test-token" not in repr(cards[0].to_receipt_dict())


@pytest.mark.asyncio
async def test_ollama_adapter_discovers_local_models() -> None:
    client = _FakeClient(
        get_payload={
            "models": [
                {"name": "qwen2.5-coder:7b"},
                {"name": "llava:latest"},
            ]
        }
    )
    adapter = OllamaProviderAdapter(client=client)

    cards = await adapter.list_models()

    assert [card.model_id for card in cards] == ["qwen2.5-coder:7b", "llava:latest"]
    assert cards[0].privacy_tier == LOCAL_PRIVATE
    assert "code" in cards[0].capabilities
    assert "vision" in cards[1].capabilities


def test_default_broker_keeps_external_disabled_without_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-test-token")
    monkeypatch.delenv("BIZRA_ALLOW_EXTERNAL_MODELS", raising=False)

    broker = create_default_model_broker()

    assert isinstance(broker, ModelBroker)
    assert broker.external_providers_allowed is False
