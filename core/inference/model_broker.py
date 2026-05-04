"""Constitutional model broker for Node0 inference routing.

This module is intentionally runtime-neutral: it does not start daemons,
load models, or mutate Node0 state.  It normalizes local and provider models
into capability cards, applies fail-closed privacy/cost/capability policy, and
returns a receiptable routing decision.

Standing on Giants: Shannon (signal/capacity scoring) · Lamport (explicit
failure modes) · Al-Ghazali (Ihsan policy gate).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

LOCAL_PRIVATE = "local_private"
PROVIDER_PRIVATE = "provider_private"
EXTERNAL = "external"

ONLINE = "online"
DEGRADED = "degraded"
OFFLINE = "offline"

LOCAL_ONLY = "local_only"
PROVIDER_ALLOWED = "provider_allowed"
EXTERNAL_ALLOWED = "external_allowed"

FREE_LOCAL = "free_local"
LOW_COST = "low"
MEDIUM_COST = "medium"
HIGH_COST = "high"

TASK_CAPABILITY_MAP: Mapping[str, frozenset[str]] = {
    "reason": frozenset({"reasoning"}),
    "reasoning": frozenset({"reasoning"}),
    "code": frozenset({"code"}),
    "coding": frozenset({"code"}),
    "vision": frozenset({"vision"}),
    "embed": frozenset({"embedding"}),
    "embedding": frozenset({"embedding"}),
    "summarize": frozenset({"text"}),
    "classify": frozenset({"text"}),
    "chat": frozenset({"text"}),
}

_PRIVACY_RANK: Mapping[str, int] = {
    LOCAL_PRIVATE: 0,
    PROVIDER_PRIVATE: 1,
    EXTERNAL: 2,
}

_COST_RANK: Mapping[str, int] = {
    FREE_LOCAL: 0,
    LOW_COST: 1,
    MEDIUM_COST: 2,
    HIGH_COST: 3,
}

_LATENCY_SCORE: Mapping[str, float] = {
    "low": 1.0,
    "medium": 0.65,
    "high": 0.35,
}


class ModelRouteError(RuntimeError):
    """Raised when no model satisfies a request policy."""


class ModelProviderError(RuntimeError):
    """Raised by provider adapters when discovery or generation fails."""


@dataclass(frozen=True)
class ModelCapabilityCard:
    """Normalized, provider-independent description of a model."""

    provider: str
    model_id: str
    endpoint: str
    capabilities: frozenset[str]
    context_window: int
    privacy_tier: str = LOCAL_PRIVATE
    cost_tier: str = FREE_LOCAL
    latency_tier: str = "medium"
    estimated_latency_ms: int | None = None
    max_tokens: int = 2048
    supports_streaming: bool = True
    supports_json_mode: bool = False
    health: str = ONLINE
    last_checked_at: str = ""
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_receipt_dict(self) -> dict[str, Any]:
        """Return a sanitized representation safe for receipts and logs."""
        return {
            "provider": self.provider,
            "model": self.model_id,
            "endpoint_class": _endpoint_class(self.endpoint),
            "capabilities": sorted(self.capabilities),
            "context_window": self.context_window,
            "privacy_tier": self.privacy_tier,
            "cost_tier": self.cost_tier,
            "latency_tier": self.latency_tier,
            "estimated_latency_ms": self.estimated_latency_ms,
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_json_mode": self.supports_json_mode,
            "health": self.health,
        }


@dataclass(frozen=True)
class InferenceRequestSpec:
    """Capability-first request from Node0 to the broker."""

    task_type: str = "reasoning"
    required_capabilities: frozenset[str] = frozenset()
    privacy_required: str = LOCAL_ONLY
    min_context: int = 0
    max_latency_ms: int | None = None
    max_cost_tier: str = HIGH_COST
    min_ihsan_score: float = UNIFIED_IHSAN_THRESHOLD
    fallback_allowed: bool = True
    preferred_providers: tuple[str, ...] = ()

    @property
    def capabilities(self) -> frozenset[str]:
        requested = set(TASK_CAPABILITY_MAP.get(self.task_type, frozenset({"text"})))
        requested.update(self.required_capabilities)
        return frozenset(requested)


@dataclass(frozen=True)
class InferenceMessage:
    """Provider-neutral chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class InferenceRequest:
    """Provider-neutral generation request."""

    messages: tuple[InferenceMessage, ...]
    spec: InferenceRequestSpec
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass(frozen=True)
class InferenceResult:
    """Provider-neutral generation result with routing provenance."""

    content: str
    provider: str
    model: str
    latency_ms: float = 0.0
    token_usage: Mapping[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    routing_receipt: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    """Broker decision for a single inference request."""

    selected: ModelCapabilityCard
    score: float
    routing_reason: str
    fallback_chain: tuple[ModelCapabilityCard, ...] = ()
    rejected: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def to_receipt_dict(self) -> dict[str, Any]:
        """Return provenance suitable for a mission receipt."""
        return {
            "provider": self.selected.provider,
            "model": self.selected.model_id,
            "privacy_tier": self.selected.privacy_tier,
            "score": round(self.score, 6),
            "routing_reason": self.routing_reason,
            "selected": self.selected.to_receipt_dict(),
            "fallback_chain": [
                candidate.to_receipt_dict() for candidate in self.fallback_chain
            ],
            "rejected": {
                provider: list(reasons)
                for provider, reasons in sorted(self.rejected.items())
            },
        }


@runtime_checkable
class ProviderAdapter(Protocol):
    """Provider adapter boundary for local and external model systems."""

    name: str
    privacy_tier: str
    enabled: bool

    async def list_models(self) -> list[ModelCapabilityCard]:
        """Return currently available model capability cards."""

    async def generate(
        self,
        request: InferenceRequest,
        decision: RoutingDecision,
    ) -> InferenceResult:
        """Generate with the selected model."""


class StaticProviderAdapter:
    """In-memory provider adapter for tests and static policy fixtures."""

    def __init__(
        self,
        *,
        name: str,
        privacy_tier: str,
        models: Iterable[ModelCapabilityCard],
        enabled: bool = True,
        fail_discovery: bool = False,
    ) -> None:
        self.name = name
        self.privacy_tier = privacy_tier
        self.enabled = enabled
        self._models = list(models)
        self._fail_discovery = fail_discovery
        self.discovery_calls = 0
        self.generate_calls = 0

    async def list_models(self) -> list[ModelCapabilityCard]:
        self.discovery_calls += 1
        if self._fail_discovery:
            raise ModelProviderError(f"{self.name} discovery failed")
        return list(self._models)

    async def generate(
        self,
        request: InferenceRequest,
        decision: RoutingDecision,
    ) -> InferenceResult:
        self.generate_calls += 1
        return InferenceResult(
            content="",
            provider=decision.selected.provider,
            model=decision.selected.model_id,
            routing_receipt=decision.to_receipt_dict(),
        )


class OpenAICompatibleProviderAdapter:
    """Adapter for LM Studio and OpenAI-compatible provider APIs."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        privacy_tier: str,
        token_env: str | None = None,
        requires_token: bool = False,
        enabled: bool = True,
        default_capabilities: Iterable[str] = ("text", "reasoning"),
        default_context_window: int = 8_192,
        default_cost_tier: str = MEDIUM_COST,
        default_latency_tier: str = "medium",
        default_estimated_latency_ms: int | None = None,
        timeout_seconds: float = 10.0,
        client: Any | None = None,
    ) -> None:
        self.name = name
        self.privacy_tier = privacy_tier
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.requires_token = requires_token
        self.default_capabilities = frozenset(default_capabilities)
        self.default_context_window = default_context_window
        self.default_cost_tier = default_cost_tier
        self.default_latency_tier = default_latency_tier
        self.default_estimated_latency_ms = default_estimated_latency_ms
        self.timeout_seconds = timeout_seconds
        self._client = client

    async def list_models(self) -> list[ModelCapabilityCard]:
        if self._client is not None:
            return await self._list_models_with_client(self._client)

        httpx = _httpx()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                return await self._list_models_with_client(client)
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise ModelProviderError(
                f"{self.name} discovery failed: {type(exc).__name__}"
            ) from exc

    async def generate(
        self,
        request: InferenceRequest,
        decision: RoutingDecision,
    ) -> InferenceResult:
        if self._client is not None:
            return await self._generate_with_client(self._client, request, decision)

        httpx = _httpx()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                return await self._generate_with_client(client, request, decision)
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise ModelProviderError(
                f"{self.name} generation failed: {type(exc).__name__}"
            ) from exc

    async def _list_models_with_client(self, client: Any) -> list[ModelCapabilityCard]:
        response = await client.get(
            f"{self.base_url}/models",
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise ModelProviderError(
                f"{self.name} discovery failed: http_{response.status_code}"
            )

        payload = response.json()
        entries = payload.get("data", []) if isinstance(payload, Mapping) else []
        cards: list[ModelCapabilityCard] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            model_id = str(entry.get("id") or entry.get("name") or "").strip()
            if not model_id:
                continue
            cards.append(
                ModelCapabilityCard(
                    provider=self.name,
                    model_id=model_id,
                    endpoint=self.base_url,
                    capabilities=self._capabilities_for(model_id),
                    context_window=self._context_for(model_id),
                    privacy_tier=self.privacy_tier,
                    cost_tier=self.default_cost_tier,
                    latency_tier=self.default_latency_tier,
                    estimated_latency_ms=self.default_estimated_latency_ms,
                    health=ONLINE,
                )
            )
        return cards

    async def _generate_with_client(
        self,
        client: Any,
        request: InferenceRequest,
        decision: RoutingDecision,
    ) -> InferenceResult:
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": decision.selected.model_id,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            },
        )
        if response.status_code != 200:
            raise ModelProviderError(
                f"{self.name} generation failed: http_{response.status_code}"
            )

        payload = response.json()
        choices = payload.get("choices", []) if isinstance(payload, Mapping) else []
        first_choice = choices[0] if choices else {}
        message = (
            first_choice.get("message", {}) if isinstance(first_choice, Mapping) else {}
        )
        usage = payload.get("usage", {}) if isinstance(payload, Mapping) else {}
        return InferenceResult(
            content=str(message.get("content", "")),
            provider=decision.selected.provider,
            model=decision.selected.model_id,
            token_usage={
                key: int(value)
                for key, value in usage.items()
                if isinstance(key, str) and isinstance(value, int)
            },
            finish_reason=str(first_choice.get("finish_reason", "stop")),
            routing_receipt=decision.to_receipt_dict(),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token_env:
            token = os.environ.get(self.token_env)
            if not token and self.requires_token:
                raise ModelProviderError(
                    f"{self.name} token missing: env:{self.token_env}"
                )
            if token:
                headers["Authorization"] = "Bearer " + token
        return headers

    def _capabilities_for(self, model_id: str) -> frozenset[str]:
        lowered = model_id.lower()
        capabilities = set(self.default_capabilities)
        if any(marker in lowered for marker in ("embed", "embedding")):
            capabilities.add("embedding")
        if any(marker in lowered for marker in ("vision", "vl", "llava")):
            capabilities.add("vision")
        if any(marker in lowered for marker in ("code", "coder", "codestral")):
            capabilities.add("code")
        if any(marker in lowered for marker in ("reason", "r1", "qwen", "claude")):
            capabilities.add("reasoning")
        return frozenset(capabilities)

    def _context_for(self, model_id: str) -> int:
        lowered = model_id.lower()
        if any(marker in lowered for marker in ("128k", "120k")):
            return max(self.default_context_window, 128_000)
        if "32k" in lowered:
            return max(self.default_context_window, 32_000)
        if "16k" in lowered:
            return max(self.default_context_window, 16_000)
        return self.default_context_window


class OllamaProviderAdapter:
    """Adapter for local Ollama model discovery and chat generation."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        enabled: bool = True,
        timeout_seconds: float = 10.0,
        client: Any | None = None,
    ) -> None:
        self.name = "ollama"
        self.privacy_tier = LOCAL_PRIVATE
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = client

    async def list_models(self) -> list[ModelCapabilityCard]:
        if self._client is not None:
            return await self._list_models_with_client(self._client)

        httpx = _httpx()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                return await self._list_models_with_client(client)
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise ModelProviderError(
                f"ollama discovery failed: {type(exc).__name__}"
            ) from exc

    async def generate(
        self,
        request: InferenceRequest,
        decision: RoutingDecision,
    ) -> InferenceResult:
        if self._client is not None:
            return await self._generate_with_client(self._client, request, decision)

        httpx = _httpx()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                return await self._generate_with_client(client, request, decision)
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise ModelProviderError(
                f"ollama generation failed: {type(exc).__name__}"
            ) from exc

    async def _list_models_with_client(self, client: Any) -> list[ModelCapabilityCard]:
        response = await client.get(f"{self.base_url}/api/tags")
        if response.status_code != 200:
            raise ModelProviderError(
                f"ollama discovery failed: http_{response.status_code}"
            )
        payload = response.json()
        entries = payload.get("models", []) if isinstance(payload, Mapping) else []
        cards: list[ModelCapabilityCard] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            model_id = str(entry.get("name") or "").strip()
            if not model_id:
                continue
            cards.append(
                ModelCapabilityCard(
                    provider=self.name,
                    model_id=model_id,
                    endpoint=self.base_url,
                    capabilities=_infer_local_capabilities(model_id),
                    context_window=8_192,
                    privacy_tier=LOCAL_PRIVATE,
                    cost_tier=FREE_LOCAL,
                    latency_tier="medium",
                    estimated_latency_ms=250,
                    health=ONLINE,
                )
            )
        return cards

    async def _generate_with_client(
        self,
        client: Any,
        request: InferenceRequest,
        decision: RoutingDecision,
    ) -> InferenceResult:
        response = await client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": decision.selected.model_id,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                "options": {"temperature": request.temperature},
                "stream": False,
            },
        )
        if response.status_code != 200:
            raise ModelProviderError(
                f"ollama generation failed: http_{response.status_code}"
            )
        payload = response.json()
        message = payload.get("message", {}) if isinstance(payload, Mapping) else {}
        return InferenceResult(
            content=str(message.get("content", "")),
            provider=decision.selected.provider,
            model=decision.selected.model_id,
            token_usage={
                "total_tokens": (
                    int(payload.get("eval_count", 0))
                    if isinstance(payload, Mapping)
                    else 0
                )
            },
            routing_receipt=decision.to_receipt_dict(),
        )


class ModelBroker:
    """Capability-first broker for local and provider inference backends."""

    def __init__(
        self,
        adapters: Iterable[ProviderAdapter],
        *,
        allow_external_providers: bool = False,
        fallback_order: Iterable[str] = (),
        historical_ihsan: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        self._adapters = tuple(adapters)
        self._allow_external = allow_external_providers
        self._fallback_order = tuple(fallback_order)
        self._historical_ihsan = dict(historical_ihsan or {})

    @property
    def external_providers_allowed(self) -> bool:
        """Whether the broker may discover external provider adapters."""
        return self._allow_external

    async def discover(
        self,
    ) -> tuple[list[ModelCapabilityCard], dict[str, tuple[str, ...]]]:
        """Discover enabled providers without touching disallowed externals."""
        cards: list[ModelCapabilityCard] = []
        rejected: dict[str, tuple[str, ...]] = {}

        for adapter in self._adapters:
            if not getattr(adapter, "enabled", False):
                rejected[adapter.name] = ("provider_disabled",)
                continue
            if (
                self._is_external_privacy(adapter.privacy_tier)
                and not self._allow_external
            ):
                rejected[adapter.name] = ("external_providers_disabled",)
                continue
            try:
                provider_cards = await adapter.list_models()
            except ModelProviderError as exc:
                rejected[adapter.name] = (str(exc),)
                continue
            cards.extend(provider_cards)

        return cards, rejected

    async def route(self, spec: InferenceRequestSpec) -> RoutingDecision:
        """Select the best model for the request or fail closed."""
        cards, provider_rejections = await self.discover()
        eligible: list[tuple[float, ModelCapabilityCard]] = []
        rejected: dict[str, list[str]] = {
            provider: list(reasons) for provider, reasons in provider_rejections.items()
        }

        for card in cards:
            reasons = self._rejection_reasons(card, spec)
            if reasons:
                rejected.setdefault(card.provider, []).extend(reasons)
                continue
            eligible.append((self._score(card, spec), card))

        if not eligible:
            sanitized = {
                provider: tuple(dict.fromkeys(reasons))
                for provider, reasons in rejected.items()
            }
            raise ModelRouteError(
                "No model satisfies inference policy: "
                + "; ".join(
                    f"{provider}={','.join(reasons)}"
                    for provider, reasons in sorted(sanitized.items())
                )
            )

        eligible.sort(
            key=lambda item: (
                -item[0],
                self._provider_order(item[1]),
                item[1].provider,
                item[1].model_id,
            )
        )
        selected_score, selected = eligible[0]
        fallback_cards = (
            tuple(card for _, card in eligible[1:]) if spec.fallback_allowed else ()
        )
        sanitized_rejected = {
            provider: tuple(dict.fromkeys(reasons))
            for provider, reasons in rejected.items()
        }
        return RoutingDecision(
            selected=selected,
            score=selected_score,
            routing_reason=self._routing_reason(selected, spec),
            fallback_chain=fallback_cards,
            rejected=sanitized_rejected,
        )

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Route and generate through the selected provider adapter."""
        decision = await self.route(request.spec)
        for adapter in self._adapters:
            if adapter.name == decision.selected.provider:
                return await adapter.generate(request, decision)
        raise ModelProviderError(
            f"selected provider unavailable: {decision.selected.provider}"
        )

    def _rejection_reasons(
        self,
        card: ModelCapabilityCard,
        spec: InferenceRequestSpec,
    ) -> list[str]:
        reasons: list[str] = []
        if not card.enabled:
            reasons.append("model_disabled")
        if card.health != ONLINE:
            reasons.append(f"model_{card.health}")
        missing = sorted(spec.capabilities - card.capabilities)
        if missing:
            reasons.append("missing_capabilities:" + ",".join(missing))
        if not _privacy_satisfies(card.privacy_tier, spec.privacy_required):
            reasons.append(f"privacy_{card.privacy_tier}_blocked")
        if card.context_window < spec.min_context:
            reasons.append("context_window_too_small")
        if _COST_RANK.get(card.cost_tier, 99) > _COST_RANK.get(spec.max_cost_tier, 99):
            reasons.append("cost_tier_exceeds_policy")
        if (
            spec.max_latency_ms is not None
            and card.estimated_latency_ms is not None
            and card.estimated_latency_ms > spec.max_latency_ms
        ):
            reasons.append("latency_exceeds_policy")
        historical_ihsan = self._historical_ihsan.get((card.provider, card.model_id))
        if historical_ihsan is not None and historical_ihsan < spec.min_ihsan_score:
            reasons.append("ihsan_below_policy")
        if spec.preferred_providers and card.provider not in spec.preferred_providers:
            reasons.append("provider_not_preferred")
        return reasons

    def _score(self, card: ModelCapabilityCard, spec: InferenceRequestSpec) -> float:
        capability_score = len(spec.capabilities & card.capabilities) / max(
            1,
            len(spec.capabilities),
        )
        privacy_score = _privacy_score(card.privacy_tier)
        historical_score = self._historical_ihsan.get(
            (card.provider, card.model_id),
            spec.min_ihsan_score,
        )
        latency_score = _latency_score(card, spec)
        cost_score = 1.0 - (_COST_RANK.get(card.cost_tier, 3) / 3.0)
        availability_score = 1.0 if card.health == ONLINE else 0.0
        return (
            0.30 * capability_score
            + 0.25 * privacy_score
            + 0.20 * min(1.0, max(0.0, historical_score))
            + 0.10 * latency_score
            + 0.10 * cost_score
            + 0.05 * availability_score
        )

    def _provider_order(self, card: ModelCapabilityCard) -> int:
        try:
            return self._fallback_order.index(card.provider)
        except ValueError:
            return len(self._fallback_order)

    @staticmethod
    def _is_external_privacy(privacy_tier: str) -> bool:
        return _PRIVACY_RANK.get(privacy_tier, 99) >= _PRIVACY_RANK[EXTERNAL]

    @staticmethod
    def _routing_reason(card: ModelCapabilityCard, spec: InferenceRequestSpec) -> str:
        capabilities = ",".join(sorted(spec.capabilities))
        return (
            f"selected {card.provider}/{card.model_id} for {capabilities} "
            f"under {spec.privacy_required} privacy policy"
        )


def create_default_model_broker(
    *,
    allow_external_providers: bool | None = None,
) -> ModelBroker:
    """Create the default broker without changing daemon behavior."""
    allow_external = (
        _env_flag("BIZRA_ALLOW_EXTERNAL_MODELS")
        if allow_external_providers is None
        else allow_external_providers
    )
    lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234")
    lm_studio_base = _ensure_v1_base_url(lm_studio_url)
    adapters: list[ProviderAdapter] = [
        OpenAICompatibleProviderAdapter(
            name="lm-studio",
            base_url=lm_studio_base,
            privacy_tier=LOCAL_PRIVATE,
            token_env=_first_env_name(
                (
                    "LM_API_TOKEN",
                    "LMSTUDIO_API_KEY",
                    "LM_STUDIO_API_KEY",
                    "LM_STUDIO_TOKEN",
                )
            ),
            requires_token=False,
            default_capabilities=("text", "reasoning", "code"),
            default_context_window=32_000,
            default_cost_tier=FREE_LOCAL,
            default_latency_tier="low",
            default_estimated_latency_ms=100,
            enabled=_env_flag("BIZRA_ENABLE_LM_STUDIO", default=True),
        ),
        OllamaProviderAdapter(
            base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
            enabled=_env_flag("BIZRA_ENABLE_OLLAMA", default=True),
        ),
    ]

    if os.environ.get("OPENAI_API_KEY") or os.environ.get("BIZRA_OPENAI_BASE_URL"):
        adapters.append(
            OpenAICompatibleProviderAdapter(
                name="openai",
                base_url=os.environ.get(
                    "BIZRA_OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
                privacy_tier=EXTERNAL,
                token_env="OPENAI_API_KEY",
                requires_token=True,
                default_capabilities=("text", "reasoning", "code", "vision"),
                default_context_window=128_000,
                default_cost_tier=HIGH_COST,
                default_latency_tier="medium",
                default_estimated_latency_ms=500,
                enabled=True,
            )
        )

    return ModelBroker(
        adapters,
        allow_external_providers=allow_external,
        fallback_order=("lm-studio", "ollama", "openai"),
    )


def _privacy_satisfies(model_privacy: str, required: str) -> bool:
    if required == LOCAL_ONLY:
        return model_privacy == LOCAL_PRIVATE
    if required == PROVIDER_ALLOWED:
        return model_privacy in {LOCAL_PRIVATE, PROVIDER_PRIVATE}
    if required == EXTERNAL_ALLOWED:
        return model_privacy in {LOCAL_PRIVATE, PROVIDER_PRIVATE, EXTERNAL}
    return False


def _infer_local_capabilities(model_id: str) -> frozenset[str]:
    lowered = model_id.lower()
    capabilities = {"text", "reasoning"}
    if any(marker in lowered for marker in ("embed", "embedding")):
        capabilities.add("embedding")
    if any(marker in lowered for marker in ("vision", "vl", "llava")):
        capabilities.add("vision")
    if any(marker in lowered for marker in ("code", "coder", "codestral")):
        capabilities.add("code")
    return frozenset(capabilities)


def _privacy_score(privacy_tier: str) -> float:
    rank = _PRIVACY_RANK.get(privacy_tier, _PRIVACY_RANK[EXTERNAL])
    return 1.0 - (rank / max(1, _PRIVACY_RANK[EXTERNAL]))


def _latency_score(card: ModelCapabilityCard, spec: InferenceRequestSpec) -> float:
    if card.estimated_latency_ms is None or spec.max_latency_ms in (None, 0):
        return _LATENCY_SCORE.get(card.latency_tier, 0.5)
    return max(0.0, min(1.0, 1.0 - (card.estimated_latency_ms / spec.max_latency_ms)))


def _endpoint_class(endpoint: str) -> str:
    lowered = endpoint.lower()
    if "127.0.0.1" in lowered or "localhost" in lowered:
        return "loopback"
    if "://" not in lowered:
        return "configured"
    host = lowered.split("://", 1)[1].split("/", 1)[0]
    if host.startswith(("10.", "172.", "192.168.")):
        return "private_network"
    return "external_network"


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ModelProviderError("httpx is required for provider adapters") from exc
    return httpx


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_v1_base_url(url: str) -> str:
    clean = url.rstrip("/")
    return clean if clean.endswith("/v1") else clean + "/v1"


def _first_env_name(names: Iterable[str]) -> str | None:
    for name in names:
        if os.environ.get(name):
            return name
    return None


__all__ = [
    "DEGRADED",
    "EXTERNAL",
    "EXTERNAL_ALLOWED",
    "FREE_LOCAL",
    "HIGH_COST",
    "InferenceMessage",
    "InferenceRequest",
    "InferenceRequestSpec",
    "InferenceResult",
    "LOCAL_ONLY",
    "LOCAL_PRIVATE",
    "LOW_COST",
    "MEDIUM_COST",
    "ModelBroker",
    "ModelCapabilityCard",
    "ModelProviderError",
    "ModelRouteError",
    "OFFLINE",
    "ONLINE",
    "OllamaProviderAdapter",
    "OpenAICompatibleProviderAdapter",
    "PROVIDER_ALLOWED",
    "PROVIDER_PRIVATE",
    "ProviderAdapter",
    "RoutingDecision",
    "StaticProviderAdapter",
    "create_default_model_broker",
]
