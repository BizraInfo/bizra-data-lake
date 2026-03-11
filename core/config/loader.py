"""
Config Loader — 3-Scope YAML Merge with Constitutional Validation
══════════════════════════════════════════════════════════════════

Scopes: federation (signed) > operator (~/.bizra/) > node local.
Deep merge with local overrides. Constitutional SSoT validation.

Phase 68.03 — Sovereign Instantiation
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable

from core.config.schema import BizraConfig
from core.integration.constants import ADL_GINI_THRESHOLD, UNIFIED_IHSAN_THRESHOLD

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]


class ConfigViolation(ValueError):
    """Raised when config violates constitutional SSoT."""


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override values replace base. Lists replaced, not appended."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if missing or yaml unavailable."""
    if yaml is None:
        logger.debug("PyYAML not available, skipping %s", path)
        return {}
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):  # SEC-003 — file_io boundary
        logger.exception("Failed to load config: %s", path)
        return {}


class ConfigLoader:
    """3-scope config loader with constitutional validation.

    Security properties:
    1. Federation configs must be signed (unsigned → rejected)
    2. Policy thresholds cannot weaken SSoT values
    3. No runtime mutation — reload from disk only
    4. Deep merge: local > operator > federation
    """

    __slots__ = (
        "_cache",
        "_watchers",
        "_node_path",
        "_operator_path",
        "_federation_path",
    )

    def __init__(
        self,
        node_path: Path | str = "bizra.node.yaml",
        operator_path: Path | str | None = None,
        federation_path: Path | str | None = None,
    ) -> None:
        self._node_path = Path(node_path)
        self._operator_path = Path(
            operator_path or Path.home() / ".bizra" / "operator.yaml"
        )
        self._federation_path = Path(federation_path or "bizra.fed.yaml")
        self._cache: BizraConfig | None = None
        self._watchers: list[Callable[[BizraConfig], Any]] = []

    def load(self) -> BizraConfig:
        """Load and merge all 3 scopes. Validates against SSoT."""
        if self._cache is not None:
            return self._cache

        # Load scopes (missing files = empty dict)
        federation = _load_yaml_file(self._federation_path)
        if federation and not self._verify_federation(federation):
            logger.warning(
                "Unsigned federation config rejected: %s", self._federation_path
            )
            federation = {}

        operator = _load_yaml_file(self._operator_path)
        local = _load_yaml_file(self._node_path)

        # Deep merge: federation < operator < local (local wins)
        merged = _deep_merge(federation, operator)
        merged = _deep_merge(merged, local)

        # Validate and build
        config = BizraConfig.model_validate(merged)

        # Cross-check with constitutional SSoT
        self._validate_against_ssot(config)

        self._cache = config
        return config

    def load_from_dict(self, data: dict[str, Any]) -> BizraConfig:
        """Load config from a dict (for testing or programmatic use)."""
        config = BizraConfig.model_validate(data)
        self._validate_against_ssot(config)
        self._cache = config
        return config

    def reload(self) -> BizraConfig:
        """Invalidate cache, reload from disk, notify watchers."""
        self._cache = None
        config = self.load()
        for watcher in self._watchers:
            try:
                watcher(config)
            except Exception:  # noqa: BLE001 — boundary boundary
                logger.exception("Config watcher failed")
        return config

    def watch(self, callback: Callable[[BizraConfig], Any]) -> None:
        """Register for config change notifications."""
        self._watchers.append(callback)

    @staticmethod
    def _validate_against_ssot(config: BizraConfig) -> None:
        """Ensure config thresholds don't weaken constitutional SSoT."""
        if config.policy.ihsan_floor < UNIFIED_IHSAN_THRESHOLD:
            raise ConfigViolation(
                f"ihsan_floor {config.policy.ihsan_floor} below constitutional "
                f"minimum {UNIFIED_IHSAN_THRESHOLD}"
            )
        if config.policy.gini_target > ADL_GINI_THRESHOLD:
            raise ConfigViolation(
                f"gini_target {config.policy.gini_target} above constitutional "
                f"maximum {ADL_GINI_THRESHOLD}"
            )

    @staticmethod
    def _verify_federation(data: dict) -> bool:
        """Check federation config has signature fields.

        Full Ed25519 verification is deferred to when crypto primitives
        are wired (Phase 68+). For now, reject configs without signature.
        """
        return "_signature" in data and "_signed_by" in data
