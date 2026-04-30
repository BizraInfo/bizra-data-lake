"""
Machine-readable API exposure policy for Sovereign `/v1/*` routes.

This module exists so API exposure decisions live in one reviewable place and
can be enforced in CI against the actual FastAPI application surface.

Standing on Giants:
- PMBOK (scope/change control): every externally reachable route needs an
  explicit ownership decision.
- OWASP ASVS: authentication boundaries should be deliberate, documented, and
  regression-tested.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RouteExposure(StrEnum):
    """Route authentication posture."""

    PUBLIC = "public"
    BOOTSTRAP_PUBLIC = "bootstrap_public"
    AUTHENTICATED = "authenticated"


@dataclass(frozen=True)
class APIRouteBinding:
    """A concrete route binding discovered from the live FastAPI app."""

    path: str
    verb: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.verb)

    def label(self) -> str:
        return f"{self.verb} {self.path}"


@dataclass(frozen=True)
class APIRoutePolicy:
    """Exposure policy for a single route binding."""

    path: str
    verb: str
    exposure: RouteExposure
    rate_limited: bool
    rationale: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.verb)

    def label(self) -> str:
        return f"{self.verb} {self.path}"


@dataclass(frozen=True)
class APIRoutePolicyReport:
    """Validation report for the route exposure contract."""

    missing_policy: tuple[APIRouteBinding, ...]
    stale_policy: tuple[APIRoutePolicy, ...]
    duplicate_routes: tuple[APIRouteBinding, ...]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_policy
            and not self.stale_policy
            and not self.duplicate_routes
        )

    def format_issues(self) -> str:
        lines: list[str] = []
        if self.missing_policy:
            lines.append("Missing policy entries:")
            lines.extend(f"  - {route.label()}" for route in self.missing_policy)
        if self.stale_policy:
            lines.append("Stale policy entries:")
            lines.extend(f"  - {route.label()}" for route in self.stale_policy)
        if self.duplicate_routes:
            lines.append("Duplicate live routes:")
            lines.extend(f"  - {route.label()}" for route in self.duplicate_routes)
        return "\n".join(lines) if lines else "No issues."


def _policies(
    verb: str,
    exposure: RouteExposure,
    rate_limited: bool,
    rationale: str,
    *paths: str,
) -> tuple[APIRoutePolicy, ...]:
    return tuple(
        APIRoutePolicy(
            path=path,
            verb=verb,
            exposure=exposure,
            rate_limited=rate_limited,
            rationale=rationale,
        )
        for path in paths
    )


API_ROUTE_POLICIES: tuple[APIRoutePolicy, ...] = (
    *_policies(
        "GET",
        RouteExposure.PUBLIC,
        False,
        "Liveness and readiness probes are intentionally public for K8s and load balancers.",
        "/v1/health/live",
        "/v1/health/ready",
        "/v1/cognitive/status",
    ),
    *_policies(
        "GET",
        RouteExposure.AUTHENTICATED,
        False,
        "Deep health, status, and metrics expose topology — auth required (topology leak fix).",
        "/v1/health/deep",
        "/v1/health/constitutional",
        "/v1/health",
        "/v1/status",
        "/v1/metrics",
    ),
    *_policies(
        "POST",
        RouteExposure.PUBLIC,
        False,
        "Cryptographic verification routes stay public so external auditors can verify receipts without credentials.",
        "/v1/verify/genesis",
        "/v1/verify/envelope",
        "/v1/verify/receipt",
        "/v1/verify/audit-log",
        "/v1/verify/ledger",
        "/v1/verify/poi",
    ),
    *_policies(
        "POST",
        RouteExposure.PUBLIC,
        True,
        "Web Vitals beacon accepts anonymous performance telemetry from the frontend dashboard.",
        "/v1/metrics/vitals",
    ),
    *_policies(
        "GET",
        RouteExposure.PUBLIC,
        False,
        "Read-only verification and artifact inspection endpoints are intentionally public for auditability.",
        "/v1/artifacts/graph/{query_id}",
        "/v1/verify/signature",
        "/v1/verify/genesis/header",
        "/v1/token/supply",
        "/v1/token/verify",
        "/v1/sel/verify",
        "/v1/chain",
        "/v1/chain/latest",
    ),
    *_policies(
        "GET",
        RouteExposure.AUTHENTICATED,
        False,
        "Gate-chain, PoI, and SAT stats expose topology — auth required (topology leak fix).",
        "/v1/gate-chain/stats",
        "/v1/poi/stats",
        "/v1/poi/contributor/{contributor_id}",
        "/v1/sat/stats",
    ),
    *_policies(
        "POST",
        RouteExposure.BOOTSTRAP_PUBLIC,
        False,
        "Bootstrap auth routes must remain public so clients can establish credentials and rotate tokens.",
        "/v1/auth/register",
        "/v1/auth/login",
        "/v1/auth/refresh",
    ),
    *_policies(
        "POST",
        RouteExposure.AUTHENTICATED,
        True,
        "Mutation-capable and query execution endpoints require authenticated callers and per-user rate limiting.",
        "/v1/query",
        "/v1/validate",
        "/v1/poi/epoch",
        "/v1/sat/epoch",
        "/v1/constitutional/tick",
        "/v1/orchestrate",
        "/v1/plan",
        "/v1/spearpoint/reproduce",
        "/v1/spearpoint/improve",
        "/v1/spearpoint/pattern",
        "/v1/sel/retrieve",
        "/v1/memory/import",
        "/v1/memory/search",
        "/v1/cognitive/fuse",
        "/v1/judgment/simulate",
        "/v1/onboarding/teach",
        "/v1/terminal/critical-acknowledgments",
    ),
    *_policies(
        "PUT",
        RouteExposure.AUTHENTICATED,
        True,
        "Persisted terminal settings modify node-local state and require authenticated callers.",
        "/v1/settings/model-routing",
    ),
    *_policies(
        "GET",
        RouteExposure.AUTHENTICATED,
        True,
        "State, history, and personalized data endpoints remain authenticated to avoid leaking internal or user-linked data.",
        "/v1/token/balance",
        "/v1/constitutional/status",
        "/v1/spearpoint/stats",
        "/v1/sel/episodes",
        "/v1/sel/episodes/{episode_hash}",
        "/v1/memory/stats",
        "/v1/memory/profile",
        "/v1/judgment/stats",
        "/v1/judgment/stability",
        "/v1/suggestions",
        "/v1/auth/me",
        "/v1/seed/potential",
        "/v1/seed/episodes",
        "/v1/node0/readiness",
        "/v1/node/value",
        "/v1/node/lifecycle",
        "/v1/network/effect",
        "/v1/network/milestones",
        "/v1/onboarding/state",
        "/v1/terminal/state",
        "/v1/terminal/briefing",
        "/v1/reflex/status",
    ),
    *_policies(
        "WEBSOCKET",
        RouteExposure.AUTHENTICATED,
        True,
        "Interactive streaming sessions are an authenticated control surface and stay off the public route set.",
        "/v1/stream",
    ),
)


def _build_policy_index() -> dict[tuple[str, str], APIRoutePolicy]:
    index: dict[tuple[str, str], APIRoutePolicy] = {}
    duplicates: list[str] = []
    for policy in API_ROUTE_POLICIES:
        if policy.key in index:
            duplicates.append(policy.label())
            continue
        index[policy.key] = policy
    if duplicates:
        duplicate_text = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate API route policies defined: {duplicate_text}")
    return index


API_ROUTE_POLICY_BY_KEY = _build_policy_index()


def iter_v1_route_bindings(app: Any) -> tuple[APIRouteBinding, ...]:
    """Enumerate concrete `/v1/*` route bindings from a FastAPI app."""
    bindings: list[APIRouteBinding] = []
    for route in getattr(app, "routes", ()):
        path = getattr(route, "path", "")
        if not isinstance(path, str) or not path.startswith("/v1/"):
            continue

        methods = getattr(route, "methods", None)
        if methods:
            for method in sorted(m for m in methods if m not in {"HEAD", "OPTIONS"}):
                bindings.append(APIRouteBinding(path=path, verb=method))
            continue

        if type(route).__name__.endswith("WebSocketRoute"):
            bindings.append(APIRouteBinding(path=path, verb="WEBSOCKET"))
    return tuple(bindings)


def summarize_api_exposure(app: Any) -> dict[RouteExposure, int]:
    """Count routes by exposure class for reporting."""
    summary: Counter[RouteExposure] = Counter()
    for binding in iter_v1_route_bindings(app):
        policy = API_ROUTE_POLICY_BY_KEY.get(binding.key)
        if policy is not None:
            summary[policy.exposure] += 1
    return dict(summary)


def validate_api_exposure_policy(app: Any) -> APIRoutePolicyReport:
    """
    Validate that every live `/v1/*` route has an explicit exposure policy.

    Failures mean either:
    - a new route shipped without governance review, or
    - the manifest drifted away from the actual FastAPI surface.
    """
    discovered = iter_v1_route_bindings(app)
    discovered_by_key: dict[tuple[str, str], APIRouteBinding] = {}
    duplicate_routes: list[APIRouteBinding] = []
    for binding in discovered:
        if binding.key in discovered_by_key:
            duplicate_routes.append(binding)
            continue
        discovered_by_key[binding.key] = binding

    missing_policy = tuple(
        binding
        for key, binding in sorted(discovered_by_key.items())
        if key not in API_ROUTE_POLICY_BY_KEY
    )
    stale_policy = tuple(
        policy
        for key, policy in sorted(API_ROUTE_POLICY_BY_KEY.items())
        if key not in discovered_by_key
    )
    return APIRoutePolicyReport(
        missing_policy=missing_policy,
        stale_policy=stale_policy,
        duplicate_routes=tuple(sorted(duplicate_routes, key=lambda item: item.key)),
    )


def get_api_route_policy(path: str, verb: str) -> APIRoutePolicy:
    """Return the declared exposure policy for a route binding."""
    try:
        return API_ROUTE_POLICY_BY_KEY[(path, verb)]
    except KeyError as exc:
        raise KeyError(f"No API route policy defined for {verb} {path}") from exc


__all__ = [
    "APIRouteBinding",
    "APIRoutePolicy",
    "APIRoutePolicyReport",
    "API_ROUTE_POLICIES",
    "API_ROUTE_POLICY_BY_KEY",
    "RouteExposure",
    "get_api_route_policy",
    "iter_v1_route_bindings",
    "summarize_api_exposure",
    "validate_api_exposure_policy",
]
