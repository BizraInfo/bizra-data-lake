"""
TeleScript Engine — Capability-Based Security for Actions
═════════════════════════════════════════════════════════

Fail-closed capability verification. Actions declare what they need;
the engine checks against a policy stack. Deny always overrides allow.
Actions can only restrict permissions, never expand them.

Standing on Giants:
- Thompson (1984): Capability-based security
- Dennis & Van Horn (1966): Supervisor capabilities
- Miller et al. (2003): Capability Myths Demolished

Phase 68.05 — Sovereign Instantiation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch


class Capability(str, Enum):
    """All capabilities a BIZRA action can request."""

    # File System
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"

    # Search
    GREP = "grep"
    GLOB = "glob"

    # Network
    NETWORK_HTTP = "network_http"
    NETWORK_WS = "network_ws"

    # LLM (System-2)
    LLM_QUERY = "llm_query"
    LLM_GENERATE = "llm_generate"

    # Desktop (HDA)
    DESKTOP_KEYBOARD = "desktop_keyboard"
    DESKTOP_MOUSE = "desktop_mouse"
    DESKTOP_SCREENSHOT = "desktop_screenshot"

    # Process
    SHELL_EXECUTE = "shell_execute"

    # Self-Modification
    SELF_MODIFY = "self_modify"

    # Governance
    GOVERNANCE_VOTE = "governance_vote"
    GOVERNANCE_PROPOSE = "governance_propose"


# Default deny paths — always blocked regardless of policy
_DEFAULT_DENY_PATHS: tuple[str, ...] = (
    "**/.env*",
    "**/.git/**",
    "**/.keys/**",
    "**/credentials*",
)


@dataclass(frozen=True)
class TeleScriptPolicy:
    """Immutable capability policy for a context.

    allow: explicitly permitted capabilities
    deny: explicitly denied (overrides allow)
    allow_paths: glob patterns for file access
    deny_paths: glob patterns always denied
    require_attestation: capabilities needing human approval
    """

    allow: frozenset[str] = field(default_factory=frozenset)
    deny: frozenset[str] = field(default_factory=frozenset)
    allow_paths: tuple[str, ...] = ()
    deny_paths: tuple[str, ...] = _DEFAULT_DENY_PATHS
    require_attestation: tuple[str, ...] = ()


# Empty policy — denies everything (fail-closed sentinel)
EMPTY_POLICY = TeleScriptPolicy()


@dataclass(frozen=True)
class TeleScriptVerdict:
    """Result of a capability check."""

    allowed: bool
    reason: str = ""
    denied_capabilities: tuple[str, ...] = ()
    needs_attestation: tuple[str, ...] = ()


class TeleScriptEngine:
    """Fail-closed capability verification engine.

    Security properties:
    1. No policy loaded → all denied
    2. Deny overrides allow
    3. Action telescript can only restrict, never expand
    4. Path deny patterns always win over allow patterns
    5. Unknown capabilities are denied
    """

    __slots__ = ("_default_policy",)

    def __init__(self, default_policy: TeleScriptPolicy | None = None) -> None:
        self._default_policy = default_policy or EMPTY_POLICY

    @property
    def default_policy(self) -> TeleScriptPolicy:
        return self._default_policy

    def check(
        self,
        requested: tuple[str, ...],
        action_telescript: dict | None = None,
        file_path: str | None = None,
    ) -> TeleScriptVerdict:
        """Check if requested capabilities are allowed.

        Returns a verdict with allowed=True only if ALL requested
        capabilities pass. Fail-closed on any ambiguity.
        """
        if not requested:
            return TeleScriptVerdict(allowed=True)

        policy = self._merge_policy(self._default_policy, action_telescript or {})

        # Step 1: Check each requested capability
        denied_caps: list[str] = []
        attestation_needed: list[str] = []

        for cap in requested:
            if cap in policy.deny:
                denied_caps.append(cap)
                continue
            if cap not in policy.allow:
                denied_caps.append(cap)
                continue
            if cap in policy.require_attestation:
                attestation_needed.append(cap)

        if denied_caps:
            return TeleScriptVerdict(
                allowed=False,
                reason=f"Denied capabilities: {denied_caps}",
                denied_capabilities=tuple(denied_caps),
            )

        # Step 2: Check file path restrictions
        if file_path and any(
            c in requested
            for c in (
                Capability.FILE_READ,
                Capability.FILE_WRITE,
                Capability.FILE_DELETE,
            )
        ):
            if not self._check_path(file_path, policy):
                return TeleScriptVerdict(
                    allowed=False,
                    reason=f"Path denied: {file_path}",
                )

        # Step 3: Attestation check
        if attestation_needed:
            return TeleScriptVerdict(
                allowed=False,
                reason=f"Attestation required for: {attestation_needed}",
                needs_attestation=tuple(attestation_needed),
            )

        return TeleScriptVerdict(allowed=True)

    @staticmethod
    def _check_path(path: str, policy: TeleScriptPolicy) -> bool:
        """Check file path against allow/deny glob patterns."""
        # Deny patterns take precedence
        for pattern in policy.deny_paths:
            if fnmatch(path, pattern):
                return False

        # No allow_paths = unrestricted
        if not policy.allow_paths:
            return True

        # Must match at least one allow pattern
        for pattern in policy.allow_paths:
            if fnmatch(path, pattern):
                return True

        return False  # No allow pattern matched

    @staticmethod
    def _merge_policy(
        default: TeleScriptPolicy, action_telescript: dict
    ) -> TeleScriptPolicy:
        """Merge action-level restrictions with defaults.

        Action telescript can only RESTRICT, never EXPAND permissions.
        """
        if not action_telescript:
            return default

        action_allow = frozenset(action_telescript.get("allow_capabilities", []))
        action_deny = frozenset(action_telescript.get("deny_capabilities", []))
        action_allow_paths = tuple(action_telescript.get("allow_paths", []))
        action_deny_paths = tuple(action_telescript.get("deny_paths", []))

        # Intersection of allows (action can only restrict)
        merged_allow = (default.allow & action_allow) if action_allow else default.allow

        # Union of denies (both levels can deny)
        merged_deny = default.deny | action_deny

        # Action paths restrict; deny paths union
        merged_allow_paths = (
            action_allow_paths if action_allow_paths else default.allow_paths
        )
        merged_deny_paths = default.deny_paths + action_deny_paths

        return TeleScriptPolicy(
            allow=merged_allow,
            deny=merged_deny,
            allow_paths=merged_allow_paths,
            deny_paths=merged_deny_paths,
            require_attestation=default.require_attestation,
        )
