# Phase 68.04 — Capsule Runtime (Skill Execution Engine)

## Context

Claude's "Skills" are packaged workflows with YAML frontmatter and tool
restrictions. BIZRA upgrades this into **Capsules**: deterministic workflows
with TeleScript capability masks, receipts, and constitutional gates.

A capsule is a Claude Skill that can prove it worked.

---

## 1. Requirements

### FR-1: Packaged Workflows
A capsule is a directory containing a manifest (CAPSULE.yaml) plus any
templates, scripts, or assets needed for execution.

### FR-2: Capability Restriction
Each capsule declares the tools it needs. The runtime enforces this via
TeleScript — no capsule can use capabilities it didn't declare.

### FR-3: Sandboxed Execution
Capsules run in an isolated context. File ops are restricted to declared
paths. Network access requires explicit bridge declaration.

### FR-4: Receipt Emission
Every capsule execution produces a receipt (ActionReceipt) that proves
what was done, by whom, and with what result.

### FR-5: Auto-Discovery
Capsules in `capsules/` directory are auto-discovered and registered.

---

## 2. Capsule Manifest

```yaml
# capsules/api-doc/CAPSULE.yaml

name: api-doc
version: "1.0.0"
description: "Generate OpenAPI docs from route code and examples."

# Who can invoke
invocation:
  user_only: false              # true = only human operator
  min_ihsan: 0.95               # minimum ihsan to auto-invoke
  trigger:                      # auto-trigger conditions
    file_patterns: ["**/routes/**/*.py", "**/api/**/*.py"]
    events: ["action.receipt"]   # trigger on file write receipt

# Capability mask (TeleScript)
capabilities:
  allow: ["file_read", "grep", "glob"]
  deny: ["file_write", "network", "llm"]  # read-only capsule
  paths:
    allow: ["./core/**", "./docs/**"]
    deny: ["**/.env*", "**/.git/**"]

# Bridge requirements (optional)
bridges: []                     # no external bridges needed

# Workflow steps
workflow:
  - step: "discover_routes"
    action: "glob"
    args: { pattern: "**/routes/**/*.py" }

  - step: "extract_schemas"
    action: "read"
    args: { files: "$discover_routes.result" }

  - step: "generate_doc"
    action: "template"
    args:
      template: "openapi.yaml.j2"
      context: "$extract_schemas.result"

  - step: "emit_artifact"
    action: "write"
    args:
      path: "./docs/openapi.yaml"
      content: "$generate_doc.result"

# Proof conditions
proof:
  - kind: "file_exists"
    target: "./docs/openapi.yaml"
  - kind: "valid_yaml"
    target: "./docs/openapi.yaml"
```

---

## 3. Capsule Registry — Pseudocode

```
CLASS CapsuleRegistry:
    INIT(capsules_dir, config):
        self.capsules_dir = Path(capsules_dir)
        self.config = config
        self._capsules: dict[str, CapsuleManifest] = {}

    DEF discover():
        """Auto-discover capsules from directory."""
        FOR manifest_path IN self.capsules_dir.glob("*/CAPSULE.yaml"):
            TRY:
                manifest = self._load_manifest(manifest_path)
                self._capsules[manifest.name] = manifest
                LOG.info(f"Discovered capsule: {manifest.name}")
            EXCEPT ValidationError as e:
                LOG.warning(f"Invalid capsule {manifest_path}: {e}")

    DEF get(name: str) -> CapsuleManifest | None:
        RETURN self._capsules.get(name)

    DEF list_all() -> list[CapsuleManifest]:
        RETURN list(self._capsules.values())

    DEF match_trigger(event_type: str, file_path: str | None) -> list[CapsuleManifest]:
        """Find capsules that should auto-trigger for this event."""
        matches = []
        FOR capsule IN self._capsules.values():
            trigger = capsule.invocation.trigger
            IF trigger IS None:
                CONTINUE
            IF event_type IN trigger.events:
                IF file_path IS None OR self._matches_pattern(file_path, trigger.file_patterns):
                    matches.append(capsule)
        RETURN matches

    DEF _load_manifest(path) -> CapsuleManifest:
        raw = yaml.safe_load(path.read_text())
        RETURN CapsuleManifest.model_validate(raw)
```

---

## 4. Capsule Runtime — Pseudocode

```
CLASS CapsuleRuntime:
    INIT(registry, action_bus, event_bus):
        self.registry = registry
        self.action_bus = action_bus
        self.event_bus = event_bus
        self._step_results: dict[str, Any] = {}

    ASYNC execute(capsule_name: str, context: dict) -> CapsuleResult:
        """Run a capsule's workflow steps through the ActionBus."""

        manifest = self.registry.get(capsule_name)
        IF manifest IS None:
            RAISE CapsuleNotFound(capsule_name)

        # Build TeleScript from manifest
        telescript = {
            "allow_capabilities": manifest.capabilities.allow,
            "deny_capabilities": manifest.capabilities.deny,
            "allow_paths": manifest.capabilities.paths.allow,
            "deny_paths": manifest.capabilities.paths.deny,
        }

        self._step_results = {}
        receipts = []

        FOR step IN manifest.workflow:
            # Resolve variable references ($step_name.result)
            resolved_args = self._resolve_vars(step.args)

            # Build action envelope
            action = ActionEnvelope(
                action_id=blake3(f"{capsule_name}:{step.step}:{context}"),
                kind=f"capsule.{capsule_name}.{step.step}",
                channel=self._step_to_channel(step.action),
                payload=resolved_args,
                capabilities=tuple(manifest.capabilities.allow),
                telescript=telescript,
                budget=ActionBudget(time_ms=10_000),
                correlation_id=context.get("mission_id", ""),
                actor_id=context.get("actor_id", b""),
                timestamp=now_ms(),
            )

            # Execute via ActionBus (gates + receipts + events)
            receipt = AWAIT self.action_bus.propose(action)
            receipts.append(receipt)

            IF receipt.status == DENIED:
                RETURN CapsuleResult(
                    capsule=capsule_name,
                    status="denied",
                    step_failed=step.step,
                    receipts=receipts,
                )

            IF receipt.status == FAILED:
                RETURN CapsuleResult(
                    capsule=capsule_name,
                    status="failed",
                    step_failed=step.step,
                    receipts=receipts,
                )

            # Store step result for variable resolution
            self._step_results[step.step] = receipt

        # Check proof conditions
        proofs_ok = AWAIT self._check_proofs(manifest.proof)

        RETURN CapsuleResult(
            capsule=capsule_name,
            status="proved" IF proofs_ok ELSE "unproved",
            receipts=receipts,
        )

    DEF _resolve_vars(args: dict) -> dict:
        """Replace $step_name.result references with actual values."""
        resolved = {}
        FOR key, value IN args.items():
            IF isinstance(value, str) AND value.startswith("$"):
                parts = value[1:].split(".")
                step_name = parts[0]
                field = parts[1] IF len(parts) > 1 ELSE "result"
                receipt = self._step_results.get(step_name)
                IF receipt:
                    resolved[key] = receipt.outcome  # extract from receipt
                ELSE:
                    resolved[key] = value  # unresolved — pass through
            ELSE:
                resolved[key] = value
        RETURN resolved

    DEF _step_to_channel(action: str) -> str:
        """Map capsule action names to ActionBus channels."""
        MATCH action:
            "glob" | "read" | "grep": RETURN "file"
            "write":                   RETURN "file"
            "template":                RETURN "llm"   # or local template engine
            "shell":                   RETURN "desktop"
            "fetch":                   RETURN "browser"
            _:                         RETURN "file"   # safe default
```

---

## 5. Built-in Capsules (starter set)

| Capsule | Capabilities | Trigger | Purpose |
|---------|-------------|---------|---------|
| `format-lint` | file_read, file_write | on_write | Auto-format + lint after file changes |
| `test-related` | file_read, shell | on_code_change | Run tests related to changed files |
| `api-doc` | file_read, glob | manual | Generate OpenAPI docs |
| `security-scan` | file_read, grep | on_code_change | OWASP + secrets scan |
| `reflex-compile` | file_read, file_write | on_verified_success | Myelination trigger |

---

## 6. TDD Anchors (10 tests)

```python
class TestCapsuleDiscovery:
    def test_discover_finds_capsules_in_dir()
    def test_invalid_manifest_skipped()
    def test_match_trigger_by_event()
    def test_match_trigger_by_file_pattern()

class TestCapsuleExecution:
    def test_execute_all_steps_succeeds()
    def test_denied_step_stops_execution()
    def test_failed_step_stops_execution()
    def test_variable_resolution_between_steps()

class TestCapsuleProofs:
    def test_proof_conditions_checked_after_workflow()
    def test_unproved_capsule_returns_unproved()
```

---

## 7. Non-Goals

- **No capsule marketplace.** Discovery is local filesystem only.
- **No capsule versioning/upgrades.** Manual file management.
- **No capsule-to-capsule chaining.** Capsules are atomic units.
  Chaining is done via Omega Loop iterations.
