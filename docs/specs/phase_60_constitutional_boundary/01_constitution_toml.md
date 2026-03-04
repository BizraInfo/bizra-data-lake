# Step 1: Constitution.toml — Machine-Parseable Axiom Codex

## Standing on Giants: Al-Ghazali (Ihsan as obligation) | Lamport (consensus requires shared state) | Dijkstra (correctness by construction)

## Problem Statement

BIZRA's constitutional gates (α4 through α10) are hardcoded in Rust
(`bizra-omega/bizra-core/`) and Python (`core/proof_engine/ihsan_gate.py`).
The gates work — 3,270+ Rust gate tests pass, Ihsan gate is fail-closed,
the sovereignty pipeline composes correctly. But the gate definitions are
scattered across implementation files, making it impossible to:

1. Add a new axiom without modifying Rust and Python code in parallel
2. Audit all constitutional constraints in one place
3. Generate tests automatically from axiom definitions
4. Let non-developers review the constitutional contract

**Solution:** A `constitution.toml` file at repo root that declares all
constitutional axioms, thresholds, and gate rules in a single human-readable,
machine-parseable format. A build-time parser generates test assertions.
The TOML file becomes the source of truth; code implementations reference it.

## Prerequisite

Phase 59 Step 4 (Constants Artifact) must be complete — `bizra-constants.json`
provides the threshold values that constitution.toml references.

## Target Files

| File | Action |
|------|--------|
| `constitution.toml` | New: axiom codex at repo root |
| `core/integration/constitution_parser.py` | New: TOML→dict parser + test generator |
| `tests/core/integration/test_constitution.py` | New: generated + manual tests |
| `bizra-omega/bizra-core/build.rs` | Update: consume constitution.toml at build time |

## Pseudocode

### constitution.toml

```pseudocode
[meta]
version = "1.0.0"
source_of_truth = "core/integration/constants.py"
artifact = "bizra-constants.json"
standing_on = ["Al-Ghazali", "Shannon", "Lamport", "Dijkstra"]

# ═══════════════════════════════════════════════════════════════
# AXIOM GROUP 1: IHSAN (Excellence)
# ═══════════════════════════════════════════════════════════════
[axioms.ihsan]
description = "Excellence is the minimum, not the aspiration"

[axioms.ihsan.production]
threshold = 0.95
gate = "fail_closed"
enforcement = "binary"  # pass/fail, no partial credit
reason_code = "IHSAN_BELOW_THRESHOLD"

[axioms.ihsan.strict]
threshold = 0.99
gate = "fail_closed"
context = "consensus_critical"

[axioms.ihsan.runtime]
threshold = 1.0
gate = "fail_closed"
context = "z3_proven_agents_only"

[axioms.ihsan.dimensions]
correctness = { weight = 0.22, floor = 0.85, reason_code = "CORRECTNESS_COMPONENT_LOW" }
safety = { weight = 0.22, floor = 0.90, reason_code = "SAFETY_COMPONENT_LOW" }
user_benefit = { weight = 0.14, floor = 0.0 }
efficiency = { weight = 0.12, floor = 0.0 }
auditability = { weight = 0.12, floor = 0.80, reason_code = "AUDITABILITY_COMPONENT_LOW" }
anti_centralization = { weight = 0.08, floor = 0.0 }
robustness = { weight = 0.06, floor = 0.80, reason_code = "ROBUSTNESS_COMPONENT_LOW" }
adl_fairness = { weight = 0.04, floor = 0.0 }

# ═══════════════════════════════════════════════════════════════
# AXIOM GROUP 2: ADL (Justice)
# ═══════════════════════════════════════════════════════════════
[axioms.adl]
description = "Justice is a hard constraint, not a warning"

[axioms.adl.gini]
threshold = 0.35
gate = "fail_closed"
enforcement = "reject_if_increases"
min_accounts = 5

[axioms.adl.harberger]
tax_rate = 0.07
destination = "universal_basic_compute"

[axioms.adl.zakat]
rate = 0.025
enforcement = "deduct_at_mint"

# ═══════════════════════════════════════════════════════════════
# AXIOM GROUP 3: AMANAH (Trust)
# ═══════════════════════════════════════════════════════════════
[axioms.amanah]
description = "Every claim must bind to evidence"

[axioms.amanah.evidence]
require_hash_chain = true
signing_algorithm = "ed25519"
chain_integrity = "blake3"

[axioms.amanah.fail_closed]
on_oracle_failure = "halt"
on_auth_missing = "reject"
on_threshold_breach = "reject"

[axioms.amanah.replay_protection]
nonce_ttl_seconds = 300
clock_skew_seconds = 120

# ═══════════════════════════════════════════════════════════════
# KERNEL INVARIANTS (Immutable)
# ═══════════════════════════════════════════════════════════════
[[kernel_invariants]]
name = "RIBA_ZERO"
description = "No exploitation. No interest. No harm."
enforcement = "compile_time"

[[kernel_invariants]]
name = "CLAIM_MUST_BIND"
description = "No hallucination. Every claim has evidence."
enforcement = "runtime"

[[kernel_invariants]]
name = "IHSAN_FLOOR"
description = "Excellence is the minimum. 0.99 threshold."
enforcement = "runtime"
threshold = 0.99

# ═══════════════════════════════════════════════════════════════
# CI GATES
# ═══════════════════════════════════════════════════════════════
[gates.precommit]
format_check = { command = "black --check core/", blocking = true }
lint = { command = "ruff check core/", blocking = true }
import_order = { command = "isort --check-only core/", blocking = true }

[gates.premerge]
tests = { command = "pytest tests/ -x", blocking = true }
security = { command = "pip-audit --strict", blocking = true }
type_check = { command = "mypy core/", blocking = false, ratcheted = true }
constants_sync = { command = "python scripts/generate_constants.py --verify", blocking = true }

[gates.deployment]
ihsan_score = { threshold = 0.95, blocking = true }
snr_score = { threshold = 0.85, blocking = true }
evidence_freshness = { max_age_hours = 168, blocking = true }

# ═══════════════════════════════════════════════════════════════
# SNR THRESHOLDS
# ═══════════════════════════════════════════════════════════════
[snr]
minimum = 0.85
t1_high = 0.95
t0_elite = 0.98
normalization = "logistic"  # snr / (1 + snr)
canonical_module = "core.snr_protocol.normalize_snr_linear"
```

### core/integration/constitution_parser.py

```pseudocode
"""Parse constitution.toml and generate test assertions.

Usage:
    from core.integration.constitution_parser import load_constitution
    const = load_constitution()
    print(const.axioms["ihsan"]["production"]["threshold"])  # 0.95
"""

IMPORT tomllib  # Python 3.11+ stdlib
IMPORT pathlib

CLASS ConstitutionError(Exception):
    """Raised when constitution.toml is malformed or inconsistent."""
    pass


FUNCTION load_constitution(path: Path = None) -> dict:
    """Load and validate constitution.toml.

    Returns parsed dict. Raises ConstitutionError on:
    - Missing required sections (axioms, kernel_invariants, gates)
    - Weight sum != 1.0 (within epsilon)
    - Threshold values outside [0, 1]
    - Unknown gate enforcement values
    """
    IF path IS None:
        path = Path(__file__).resolve().parent.parent.parent / "constitution.toml"

    IF NOT path.exists():
        RAISE ConstitutionError(f"constitution.toml not found at {path}")

    WITH open(path, "rb") AS f:
        data = tomllib.load(f)

    _validate_structure(data)
    _validate_weights(data)
    _validate_thresholds(data)

    RETURN data


FUNCTION _validate_structure(data: dict) -> None:
    """Verify required top-level sections exist."""
    required = ["axioms", "kernel_invariants", "gates", "snr", "meta"]
    FOR section IN required:
        IF section NOT IN data:
            RAISE ConstitutionError(f"Missing required section: [{section}]")

    required_axiom_groups = ["ihsan", "adl", "amanah"]
    FOR group IN required_axiom_groups:
        IF group NOT IN data["axioms"]:
            RAISE ConstitutionError(f"Missing axiom group: [axioms.{group}]")


FUNCTION _validate_weights(data: dict) -> None:
    """Verify Ihsan dimension weights sum to 1.0."""
    dims = data["axioms"]["ihsan"]["dimensions"]
    total = sum(d["weight"] FOR d IN dims.values())
    IF abs(total - 1.0) > 1e-6:
        RAISE ConstitutionError(f"Ihsan weights sum to {total}, expected 1.0")


FUNCTION _validate_thresholds(data: dict) -> None:
    """Verify all thresholds are in valid range."""
    FOR name, value IN [
        ("ihsan.production", data["axioms"]["ihsan"]["production"]["threshold"]),
        ("ihsan.strict", data["axioms"]["ihsan"]["strict"]["threshold"]),
        ("adl.gini", data["axioms"]["adl"]["gini"]["threshold"]),
        ("snr.minimum", data["snr"]["minimum"]),
    ]:
        IF NOT (0.0 <= value <= 1.0):
            RAISE ConstitutionError(f"Threshold {name} = {value} outside [0, 1]")


FUNCTION generate_test_assertions(data: dict) -> str:
    """Generate pytest test code from constitution axioms.

    Returns a string of Python test functions that verify
    the codebase constants match constitution.toml declarations.
    """
    lines = [
        "# Auto-generated from constitution.toml — do not edit manually",
        "from core.integration.constants import (",
        "    UNIFIED_IHSAN_THRESHOLD, STRICT_IHSAN_THRESHOLD,",
        "    ADL_GINI_THRESHOLD, UNIFIED_SNR_THRESHOLD,",
        "    IHSAN_WEIGHTS,",
        ")",
        "",
    ]

    # Ihsan threshold assertions
    ihsan = data["axioms"]["ihsan"]
    lines.append(f"def test_ihsan_production_threshold():")
    lines.append(f"    assert UNIFIED_IHSAN_THRESHOLD == {ihsan['production']['threshold']}")
    lines.append("")
    lines.append(f"def test_ihsan_strict_threshold():")
    lines.append(f"    assert STRICT_IHSAN_THRESHOLD == {ihsan['strict']['threshold']}")
    lines.append("")

    # Adl threshold assertions
    adl = data["axioms"]["adl"]
    lines.append(f"def test_adl_gini_threshold():")
    lines.append(f"    assert ADL_GINI_THRESHOLD == {adl['gini']['threshold']}")
    lines.append("")

    # SNR threshold assertions
    snr = data["snr"]
    lines.append(f"def test_snr_minimum_threshold():")
    lines.append(f"    assert UNIFIED_SNR_THRESHOLD == {snr['minimum']}")
    lines.append("")

    # Weight sum assertion
    lines.append("def test_ihsan_weights_sum_to_one():")
    lines.append("    total = sum(IHSAN_WEIGHTS.values())")
    lines.append("    assert abs(total - 1.0) < 1e-6")

    RETURN "\n".join(lines)


FUNCTION verify_against_constants(data: dict) -> list[str]:
    """Check constitution.toml values match constants.py values.

    Returns list of drift messages (empty = all synced).
    """
    FROM core.integration.constants IMPORT (
        UNIFIED_IHSAN_THRESHOLD, STRICT_IHSAN_THRESHOLD,
        ADL_GINI_THRESHOLD, UNIFIED_SNR_THRESHOLD,
        IHSAN_WEIGHTS,
    )

    drifts = []
    checks = [
        ("ihsan.production", data["axioms"]["ihsan"]["production"]["threshold"],
         UNIFIED_IHSAN_THRESHOLD),
        ("ihsan.strict", data["axioms"]["ihsan"]["strict"]["threshold"],
         STRICT_IHSAN_THRESHOLD),
        ("adl.gini", data["axioms"]["adl"]["gini"]["threshold"],
         ADL_GINI_THRESHOLD),
        ("snr.minimum", data["snr"]["minimum"],
         UNIFIED_SNR_THRESHOLD),
    ]

    FOR name, toml_val, code_val IN checks:
        IF abs(toml_val - code_val) > 1e-9:
            drifts.append(f"{name}: toml={toml_val} code={code_val}")

    # Check dimension weights
    FOR dim, spec IN data["axioms"]["ihsan"]["dimensions"].items():
        code_weight = IHSAN_WEIGHTS.get(dim)
        IF code_weight IS None:
            drifts.append(f"dimension '{dim}' in constitution.toml but not in IHSAN_WEIGHTS")
        ELIF abs(spec["weight"] - code_weight) > 1e-9:
            drifts.append(f"weight '{dim}': toml={spec['weight']} code={code_weight}")

    RETURN drifts
```

## TDD Anchors

```pseudocode
TEST constitution_toml_parses_successfully:
    data = load_constitution()
    ASSERT "axioms" IN data
    ASSERT "kernel_invariants" IN data
    ASSERT "gates" IN data
    ASSERT "snr" IN data
    ASSERT "meta" IN data

TEST constitution_has_three_axiom_groups:
    data = load_constitution()
    ASSERT "ihsan" IN data["axioms"]
    ASSERT "adl" IN data["axioms"]
    ASSERT "amanah" IN data["axioms"]

TEST ihsan_weights_sum_to_one:
    data = load_constitution()
    dims = data["axioms"]["ihsan"]["dimensions"]
    total = sum(d["weight"] FOR d IN dims.values())
    ASSERT abs(total - 1.0) < 1e-6

TEST ihsan_dimensions_match_constants:
    data = load_constitution()
    FROM core.integration.constants IMPORT IHSAN_WEIGHTS
    FOR dim IN data["axioms"]["ihsan"]["dimensions"]:
        ASSERT dim IN IHSAN_WEIGHTS

TEST thresholds_match_constants:
    data = load_constitution()
    drifts = verify_against_constants(data)
    ASSERT len(drifts) == 0, f"Drift detected: {drifts}"

TEST kernel_invariants_are_three:
    data = load_constitution()
    ASSERT len(data["kernel_invariants"]) == 3
    names = [ki["name"] FOR ki IN data["kernel_invariants"]]
    ASSERT "RIBA_ZERO" IN names
    ASSERT "CLAIM_MUST_BIND" IN names
    ASSERT "IHSAN_FLOOR" IN names

TEST gates_have_blocking_flag:
    data = load_constitution()
    FOR stage_name, stage IN data["gates"].items():
        FOR gate_name, gate_spec IN stage.items():
            ASSERT "blocking" IN gate_spec, f"Gate {stage_name}.{gate_name} missing 'blocking'"

TEST generate_test_assertions_produces_valid_python:
    data = load_constitution()
    code = generate_test_assertions(data)
    compile(code, "<constitution>", "exec")  # Must not raise SyntaxError

TEST constitution_is_deterministic:
    d1 = load_constitution()
    d2 = load_constitution()
    ASSERT d1 == d2

TEST missing_constitution_raises_error:
    WITH pytest.raises(ConstitutionError):
        load_constitution(Path("/nonexistent/constitution.toml"))

TEST malformed_weights_detected:
    """Weights that don't sum to 1.0 are rejected."""
    bad_data = load_constitution()
    bad_data["axioms"]["ihsan"]["dimensions"]["correctness"]["weight"] = 0.99
    WITH pytest.raises(ConstitutionError):
        _validate_weights(bad_data)
```

## Acceptance Criteria

1. `constitution.toml` exists at repo root, contains all 3 axiom groups
2. `load_constitution()` returns valid dict with all sections
3. `verify_against_constants()` returns empty drift list
4. Generated test assertions pass when executed
5. Rust `build.rs` reads constitution.toml (deferred to Phase 61)
6. Full test suite GREEN

## Migration Path

1. Create `constitution.toml` from existing constants.py values (Phase 60)
2. Add `constitution_parser.py` with validation and test generation (Phase 60)
3. Add CI step: `python -c "from core.integration.constitution_parser import load_constitution; load_constitution()"` (Phase 60)
4. Update Rust `build.rs` to read constitution.toml (Phase 61)
5. Begin generating tests from TOML instead of writing manually (Phase 62)
