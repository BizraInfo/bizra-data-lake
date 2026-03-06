# Phase 68.03 — Unified Config System

## Context

Configuration is currently scattered: env vars, `bizra_config.py` paths,
`constants.py` thresholds, docker-compose env sections. This spec unifies
into a 3-scope YAML config system that can be ported 1:1 to Rust.

---

## 1. Three Scopes (precedence: local > operator > federation)

```
federation_shared   ← signed by network quorum, rarely changes
  operator_global   ← ~/.bizra/operator.yaml, user preferences
    node_local      ← ./bizra.node.yaml, per-project overrides
```

**Merge rule:** deep merge. Local keys override operator, operator overrides
federation. Lists are replaced, not appended. Missing keys inherit from
parent scope.

**Federation scope:** MUST be signed (Ed25519). Unsigned federation configs
are rejected. This prevents unsigned policy injection.

---

## 2. Config Schema

### 2.1 Root: `bizra.node.yaml`

```yaml
# Required
node:
  id: "ed25519:..."           # node identity (auto-generated on init)
  covenant_hash: "blake2b-256:859649ea..."  # declaration hash

# Policy (inherits from constants.py SSoT)
policy:
  ihsan_floor: 0.95           # UNIFIED_IHSAN_THRESHOLD
  intent_floor: 0.90          # INTENT_FLOOR
  gini_target: 0.35           # ADL_GINI_THRESHOLD
  snr_minimum: 0.85           # SNR_MINIMUM_THRESHOLD

# Orchestrator
orchestrator:
  routing_model: "hhmm"       # "hhmm" | "keyword" | "reflex_first"
  max_workers: 7              # parallel worker cap
  s2_calls_per_hour: 120      # LLM budget
  omega_max_iterations: 50    # Omega loop hard stop

# Inference (local-first)
inference:
  primary: "auto"             # "auto" | "lmstudio" | "ollama" | "cloud"
  fallback: "ollama"
  timeout_ms: 30000

# Hooks
hooks:
  pre_execution:
    deny_paths:
      - "**/.env*"
      - "**/secrets.*"
      - "**/.git/**"
    require_attestation:
      - "network:*"
      - "self_modify:*"
  post_receipt:
    on_write: ["format", "lint"]
    on_code_change: ["typecheck", "tests:related"]
    on_verified_success: ["reflex_compile_if_eligible"]

# Bridges (external integrations)
bridges:
  - id: "filesystem"
    enabled: true
  - id: "github"
    enabled: true
    scopes: ["repo:read", "issues:write"]
  - id: "browser"
    enabled: false             # opt-in

# Capsules
capsules:
  dir: "./capsules/"
  auto_discover: true

# Economy (constitutional ticker)
economy:
  enabled: true
  zakat_cycle_hours: 8760     # annual (365 * 24)
  tick_interval_ms: 3600000   # 1 hour
```

### 2.2 Operator: `~/.bizra/operator.yaml`

```yaml
# User-wide defaults
inference:
  primary: "lmstudio"
  lmstudio_port: 1234

orchestrator:
  max_workers: 12             # higher for powerful machines

notifications:
  desktop: true
  hud: true
  sound: false
```

### 2.3 Federation: `bizra.fed.yaml`

```yaml
# Network-wide policy (MUST be signed)
_signature: "ed25519:..."
_signed_by: "node:ed25519:..."
_signed_at: 1741392000000

policy:
  ihsan_floor: 0.95           # network minimum
  gini_target: 0.35           # network maximum

federation:
  gossip_interval_ms: 5000
  max_peers: 50
  attestation_minimum: 3      # MIN_CONNECTIONS
```

---

## 3. Config Loader — Pseudocode

```
CLASS ConfigLoader:
    INIT():
        self._cache: dict | None = None
        self._watchers: list[Callable] = []

    DEF load() -> BizraConfig:
        """Load and merge all 3 scopes."""
        IF self._cache IS NOT None:
            RETURN self._cache

        # Load scopes (missing files = empty dict)
        federation = self._load_yaml("bizra.fed.yaml")
        IF federation AND NOT self._verify_signature(federation):
            LOG.warning("Unsigned federation config rejected")
            federation = {}

        operator = self._load_yaml(Path.home() / ".bizra" / "operator.yaml")
        local = self._load_yaml("bizra.node.yaml")

        # Deep merge: local > operator > federation
        merged = deep_merge(federation, operator, local)

        # Validate against schema
        config = BizraConfig.model_validate(merged)

        # Cross-check with constants.py SSoT
        self._validate_against_ssot(config)

        self._cache = config
        RETURN config

    DEF _validate_against_ssot(config):
        """Ensure config thresholds don't violate constitutional SSoT."""
        IF config.policy.ihsan_floor < UNIFIED_IHSAN_THRESHOLD:
            RAISE ConfigViolation(
                f"ihsan_floor {config.policy.ihsan_floor} below constitutional "
                f"minimum {UNIFIED_IHSAN_THRESHOLD}"
            )
        IF config.policy.gini_target > ADL_GINI_THRESHOLD:
            RAISE ConfigViolation(
                f"gini_target {config.policy.gini_target} above constitutional "
                f"maximum {ADL_GINI_THRESHOLD}"
            )

    DEF _verify_signature(data: dict) -> bool:
        """Verify Ed25519 signature on federation config."""
        sig = data.pop("_signature", None)
        signer = data.pop("_signed_by", None)
        IF NOT sig OR NOT signer:
            RETURN False
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        RETURN ed25519_verify(signer, canonical.encode(), sig)

    DEF reload():
        """Invalidate cache and reload. Notify watchers."""
        self._cache = None
        config = self.load()
        FOR watcher IN self._watchers:
            watcher(config)

    DEF watch(callback: Callable):
        """Register for config change notifications."""
        self._watchers.append(callback)
```

---

## 4. Pydantic Schema

```python
class PolicyConfig(BaseModel):
    ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD
    intent_floor: float = INTENT_FLOOR
    gini_target: float = ADL_GINI_THRESHOLD
    snr_minimum: float = SNR_MINIMUM_THRESHOLD

class OrchestratorConfig(BaseModel):
    routing_model: Literal["hhmm", "keyword", "reflex_first"] = "hhmm"
    max_workers: int = Field(default=7, ge=1, le=32)
    s2_calls_per_hour: int = Field(default=120, ge=0)
    omega_max_iterations: int = Field(default=50, ge=1, le=1000)

class InferenceConfig(BaseModel):
    primary: str = "auto"
    fallback: str = "ollama"
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)

class BizraConfig(BaseModel):
    """Unified BIZRA node configuration."""
    node: NodeConfig = Field(default_factory=NodeConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    bridges: list[BridgeConfig] = Field(default_factory=list)
    capsules: CapsuleConfig = Field(default_factory=CapsuleConfig)
    economy: EconomyConfig = Field(default_factory=EconomyConfig)
```

---

## 5. Relationship to constants.py

`constants.py` remains the Single Source of Truth for constitutional
thresholds. Config files can set thresholds **equal to or stricter than**
SSoT values, never weaker.

```
constants.py (SSoT)     bizra.node.yaml (config)     Result
IHSAN = 0.95            ihsan_floor: 0.97            0.97 (stricter OK)
IHSAN = 0.95            ihsan_floor: 0.90            REJECTED (below SSoT)
GINI = 0.35             gini_target: 0.30            0.30 (stricter OK)
GINI = 0.35             gini_target: 0.40            REJECTED (above SSoT)
```

---

## 6. TDD Anchors (12 tests)

```python
class TestConfigLoading:
    def test_load_local_only()
    def test_load_with_operator_merge()
    def test_local_overrides_operator()
    def test_missing_files_use_defaults()

class TestConstitutionalValidation:
    def test_ihsan_below_ssot_rejected()
    def test_gini_above_ssot_rejected()
    def test_stricter_than_ssot_accepted()

class TestFederationSignature:
    def test_signed_federation_accepted()
    def test_unsigned_federation_rejected()
    def test_tampered_federation_rejected()

class TestConfigReload:
    def test_reload_invalidates_cache()
    def test_watchers_notified_on_reload()
```

---

## 7. Non-Goals

- **No GUI config editor.** YAML files + text editor.
- **No runtime config mutation.** Reload from disk only. No API to change
  running config (prevents injection attacks).
- **No backward compat with scattered env vars.** Env vars still work as
  overrides but `bizra.node.yaml` is the primary source.
