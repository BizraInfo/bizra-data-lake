# Phase 68.05 — TeleScript Python (Capability Masks)

## Context

The Rust layer has permit-guarded action dispatch (`PermitGuard` in
`bizra-agent`). The Python layer has no equivalent — MissionOrchestrator
calls channels directly without capability checking. This spec adds
Python-side TeleScript enforcement.

TeleScript is the "allowed_tools" concept from Claude, upgraded to a
constitutional capability mask that travels with every action.

---

## 1. Requirements

### FR-1: Capability Declaration
Every action declares what capabilities it needs as a tuple of strings.

### FR-2: Policy Enforcement
A TeleScript policy defines what capabilities are allowed/denied for
a given context (node config, capsule, worker).

### FR-3: Path Restrictions
File operations are restricted to declared path patterns.
Glob syntax for allow/deny patterns.

### FR-4: Fail-Closed
If no policy is loaded, ALL capabilities are denied.
If a capability is not explicitly allowed, it is denied.

### FR-5: Attestation Gates
Some capabilities require explicit attestation (human approval)
before execution. Configured in `bizra.hooks.yaml`.

---

## 2. Capability Taxonomy

```python
# core/bus/telescript.py

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
```

---

## 3. TeleScript Policy

```python
@dataclass(frozen=True)
class TeleScriptPolicy:
    """Immutable capability policy for a context."""
    allow: frozenset[str]           # allowed capabilities
    deny: frozenset[str]            # explicitly denied (overrides allow)
    allow_paths: tuple[str, ...]    # glob patterns for file access
    deny_paths: tuple[str, ...]     # glob patterns denied
    require_attestation: tuple[str, ...]  # capabilities needing human OK
    budget: dict = field(default_factory=dict)  # resource limits
```

---

## 4. TeleScript Engine — Pseudocode

```
CLASS TeleScriptEngine:
    INIT(config: BizraConfig):
        self.default_policy = self._build_default_policy(config)
        self._attestation_queue: asyncio.Queue = asyncio.Queue()
        self._attestation_responses: dict[str, bool] = {}

    DEF check(
        requested: tuple[str, ...],
        action_telescript: dict,
        file_path: str | None = None,
    ) -> TeleScriptVerdict:
        """Check if requested capabilities are allowed."""

        # Merge action-level telescript with default policy
        policy = self._merge_policy(self.default_policy, action_telescript)

        # Step 1: Check each requested capability
        denied_caps = []
        attestation_needed = []

        FOR cap IN requested:
            # Deny list takes precedence
            IF cap IN policy.deny:
                denied_caps.append(cap)
                CONTINUE

            # Must be explicitly allowed (fail-closed)
            IF cap NOT IN policy.allow:
                denied_caps.append(cap)
                CONTINUE

            # Check if attestation required
            IF cap IN policy.require_attestation:
                attestation_needed.append(cap)

        IF denied_caps:
            RETURN TeleScriptVerdict(
                allowed=False,
                reason=f"Denied capabilities: {denied_caps}",
                denied_capabilities=denied_caps,
            )

        # Step 2: Check file path restrictions (if applicable)
        IF file_path AND ("file_read" IN requested OR "file_write" IN requested):
            path_ok = self._check_path(file_path, policy)
            IF NOT path_ok:
                RETURN TeleScriptVerdict(
                    allowed=False,
                    reason=f"Path denied: {file_path}",
                )

        # Step 3: Attestation check
        IF attestation_needed:
            RETURN TeleScriptVerdict(
                allowed=False,
                reason=f"Attestation required for: {attestation_needed}",
                needs_attestation=attestation_needed,
            )

        RETURN TeleScriptVerdict(allowed=True)

    DEF _check_path(path: str, policy: TeleScriptPolicy) -> bool:
        """Check file path against allow/deny glob patterns."""
        normalized = Path(path).resolve().as_posix()

        # Deny patterns take precedence
        FOR pattern IN policy.deny_paths:
            IF fnmatch(normalized, pattern):
                RETURN False

        # Must match at least one allow pattern
        IF NOT policy.allow_paths:
            RETURN True  # no restrictions = allow all

        FOR pattern IN policy.allow_paths:
            IF fnmatch(normalized, pattern):
                RETURN True

        RETURN False  # no allow pattern matched

    DEF _merge_policy(default, action_telescript) -> TeleScriptPolicy:
        """Merge action-level restrictions with defaults.

        Action telescript can only RESTRICT, never EXPAND permissions.
        """
        action_allow = frozenset(action_telescript.get("allow_capabilities", []))
        action_deny = frozenset(action_telescript.get("deny_capabilities", []))
        action_allow_paths = tuple(action_telescript.get("allow_paths", []))
        action_deny_paths = tuple(action_telescript.get("deny_paths", []))

        # Intersection of allows (action can only restrict)
        merged_allow = default.allow & action_allow IF action_allow ELSE default.allow

        # Union of denies (both levels can deny)
        merged_deny = default.deny | action_deny

        # Intersection of allow paths, union of deny paths
        merged_allow_paths = action_allow_paths IF action_allow_paths ELSE default.allow_paths
        merged_deny_paths = default.deny_paths + action_deny_paths

        RETURN TeleScriptPolicy(
            allow=merged_allow,
            deny=merged_deny,
            allow_paths=merged_allow_paths,
            deny_paths=merged_deny_paths,
            require_attestation=default.require_attestation,
        )

    DEF _build_default_policy(config) -> TeleScriptPolicy:
        """Build default policy from bizra.node.yaml config."""
        hooks = config.hooks
        RETURN TeleScriptPolicy(
            allow=frozenset(Capability),  # all by default
            deny=frozenset(),
            allow_paths=(),               # no restrictions
            deny_paths=tuple(hooks.pre_execution.deny_paths),
            require_attestation=tuple(hooks.pre_execution.require_attestation),
        )
```

---

## 5. Verdict Type

```python
@dataclass(frozen=True)
class TeleScriptVerdict:
    allowed: bool
    reason: str = ""
    denied_capabilities: list[str] = field(default_factory=list)
    needs_attestation: list[str] = field(default_factory=list)
```

---

## 6. Integration Points

### ActionBus (Phase 68.01)
```python
# In ActionBus.propose():
verdict = self.telescript.check(
    requested=action.capabilities,
    action_telescript=action.telescript,
    file_path=action.payload.get("path"),
)
if not verdict.allowed:
    # Emit deny event, return DENIED receipt
```

### Capsule Runtime (Phase 68.04)
```python
# Each capsule builds its telescript from CAPSULE.yaml:
telescript = {
    "allow_capabilities": manifest.capabilities.allow,
    "deny_capabilities": manifest.capabilities.deny,
    "allow_paths": manifest.capabilities.paths.allow,
    "deny_paths": manifest.capabilities.paths.deny,
}
```

### Config System (Phase 68.03)
```python
# Default policy from bizra.node.yaml hooks section:
hooks:
  pre_execution:
    deny_paths: ["**/.env*", "**/.git/**"]
    require_attestation: ["network:*", "self_modify:*"]
```

---

## 7. Security Properties

1. **Fail-closed:** No policy = all denied. Unknown capability = denied.
2. **Deny overrides allow:** If both lists contain a capability, deny wins.
3. **Action can only restrict:** Action-level telescript intersects with
   default policy. A capsule cannot grant itself more permissions than
   the node config allows.
4. **Path deny always wins:** Deny path patterns are checked first.
   Even if an allow pattern matches, a deny pattern overrides it.
5. **Attestation is blocking:** If attestation is required, the action
   is held until human approval. No timeout-based auto-approve.

---

## 8. TDD Anchors (14 tests)

```python
class TestTeleScriptBasic:
    def test_allowed_capability_passes()
    def test_denied_capability_fails()
    def test_unknown_capability_denied()
    def test_empty_policy_denies_all()

class TestPathRestrictions:
    def test_allowed_path_passes()
    def test_denied_path_blocked()
    def test_deny_overrides_allow_path()
    def test_env_files_always_denied()
    def test_git_dir_always_denied()

class TestPolicyMerge:
    def test_action_restricts_default()
    def test_action_cannot_expand_permissions()
    def test_deny_lists_union()

class TestAttestation:
    def test_attestation_required_returns_verdict()
    def test_network_requires_attestation()

class TestIntegration:
    def test_capsule_telescript_enforced()
```

---

## 9. Non-Goals

- **No runtime capability negotiation.** Capabilities are declared,
  not negotiated. An action either has permission or it doesn't.
- **No capability delegation.** A worker cannot grant capabilities
  to sub-workers. Each level declares independently.
- **No Windows ACL integration.** TeleScript is application-level.
  OS-level permissions are a separate concern.
