# Step 4: Constants Artifact Generation

## Standing on Giants: Lamport — consensus requires shared state; DNA replication

## Problem Statement

`core/integration/constants.py` is the single source of truth for all
constitutional thresholds. The `validate_cross_repo_consistency()` function
checks that Rust and TypeScript repos have matching values by **reading source
files and regex-matching**. This is fragile:

1. Regex `r"0\.95"` matches comments, string literals, and unrelated constants
2. File paths are hardcoded to the C: drive layout
3. The validator cannot run in CI (other repos may not be checked out)
4. New constants require manual regex additions

**Proper solution:** Generate a `bizra-constants.json` artifact from
`constants.py` at build time. Other repos consume this artifact. Drift is
impossible by construction, not detected after the fact.

## Target Files

| File | Action |
|------|--------|
| `core/integration/constants.py` | Add `export_constants_json()` function |
| `scripts/generate_constants.py` | New: generates `bizra-constants.json` |
| `bizra-constants.json` | New: generated artifact (checked into repo) |
| `.github/workflows/ci.yml` | Add step to verify artifact is up-to-date |
| `bizra-omega/bizra-core/build.rs` | Consume JSON instead of hardcoded values |

## Pseudocode

### constants.py — Export Function

```pseudocode
FUNCTION export_constants_json() -> dict:
    """Export all constitutional constants as a JSON-serializable dict.

    This is the AUTHORITATIVE export. All other repos consume this output.
    The generated file is checked into the repo and verified by CI.
    """
    RETURN {
        "version": "1.0.0",
        "generated_utc": datetime.utcnow().isoformat(),
        "source": "core/integration/constants.py",
        "thresholds": {
            "ihsan": {
                "production": IHSAN_PRODUCTION,       # 0.95
                "ci": IHSAN_CI,                        # 0.90
                "strict": IHSAN_STRICT,                # 0.99
                "consensus": IHSAN_CONSENSUS,          # 0.99
                "runtime": IHSAN_RUNTIME,              # 1.0
            },
            "snr": {
                "minimum": SNR_MINIMUM,                # 0.85
                "t1": SNR_T1,                          # 0.95
                "t0_elite": SNR_T0_ELITE,              # 0.98
            },
            "adl_gini": {
                "threshold": ADL_GINI_THRESHOLD,       # 0.35
            },
        },
        "ihsan_weights": {
            "dimensions": IHSAN_WEIGHT_DIMENSIONS,     # 8
            "weights": list(CANONICAL_IHSAN_WEIGHTS),  # [w1, w2, ..., w8]
            "labels": IHSAN_WEIGHT_LABELS,             # ["moral_clarity", ...]
        },
        "hmm": {
            "num_hidden_states": HMM_NUM_HIDDEN_STATES,
            "observation_window": HMM_OBSERVATION_WINDOW,
            "max_em_iterations": HMM_MAX_EM_ITERATIONS,
        },
        "token_economics": {
            "zakat_rate": ZAKAT_RATE,                  # 0.025
            "yearly_supply_cap": YEARLY_SUPPLY_CAP,    # 1_000_000
            "bloom_redistribution_rate": BLOOM_REDISTRIBUTION_RATE,
        },
        "cross_repo": CROSS_REPO_CONSTANTS,
    }
```

### scripts/generate_constants.py

```pseudocode
#!/usr/bin/env python3
"""Generate bizra-constants.json from the authoritative constants.py.

Usage:
    python scripts/generate_constants.py
    python scripts/generate_constants.py --verify  # CI mode: check artifact is fresh
"""

IMPORT core.integration.constants AS C

FUNCTION main():
    args = parse_args()
    data = C.export_constants_json()
    json_str = json.dumps(data, indent=2, sort_keys=True)

    artifact_path = BIZRA_ROOT / "bizra-constants.json"

    IF args.verify:
        # CI mode: compare generated vs committed
        existing = artifact_path.read_text()
        IF json_str.strip() != existing.strip():
            print("ERROR: bizra-constants.json is stale!")
            print("Run: python scripts/generate_constants.py")
            sys.exit(1)
        print("OK: bizra-constants.json is up-to-date")
        sys.exit(0)

    # Generate mode
    artifact_path.write_text(json_str + "\n")
    print(f"Generated {artifact_path} ({len(json_str)} bytes)")

    # Also generate BLAKE3 digest for verification
    digest = blake3(json_str.encode()).hexdigest()
    print(f"BLAKE3 digest: {digest}")
```

### CI Integration

```pseudocode
# .github/workflows/ci.yml — add to cross-language-sync job:

- name: Verify constants artifact
  run: |
    python scripts/generate_constants.py --verify
```

### Rust Consumption (bizra-core/build.rs)

```pseudocode
// build.rs — read constants from JSON artifact at build time
fn main() {
    let constants_path = std::path::Path::new("../bizra-constants.json");
    if constants_path.exists() {
        let json = std::fs::read_to_string(constants_path).unwrap();
        let data: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Generate Rust constants from JSON
        let ihsan_prod = data["thresholds"]["ihsan"]["production"].as_f64().unwrap();
        let snr_min = data["thresholds"]["snr"]["minimum"].as_f64().unwrap();
        let gini = data["thresholds"]["adl_gini"]["threshold"].as_f64().unwrap();

        println!("cargo:rustc-env=BIZRA_IHSAN_PRODUCTION={}", ihsan_prod);
        println!("cargo:rustc-env=BIZRA_SNR_MINIMUM={}", snr_min);
        println!("cargo:rustc-env=BIZRA_ADL_GINI_THRESHOLD={}", gini);
    }
    println!("cargo:rerun-if-changed=../bizra-constants.json");
}
```

## TDD Anchors

```pseudocode
TEST export_constants_json_has_required_keys:
    data = export_constants_json()
    ASSERT "version" IN data
    ASSERT "thresholds" IN data
    ASSERT "ihsan_weights" IN data
    ASSERT data["thresholds"]["ihsan"]["production"] == 0.95
    ASSERT data["thresholds"]["snr"]["minimum"] == 0.85
    ASSERT data["thresholds"]["adl_gini"]["threshold"] == 0.35

TEST export_is_json_serializable:
    data = export_constants_json()
    json_str = json.dumps(data)  # Must not raise
    roundtrip = json.loads(json_str)
    ASSERT roundtrip == data  # Must roundtrip cleanly

TEST export_is_deterministic:
    """Two calls produce identical output (no timestamps or random values)."""
    # Note: generated_utc will differ — strip it for comparison
    d1 = export_constants_json()
    d2 = export_constants_json()
    del d1["generated_utc"]
    del d2["generated_utc"]
    ASSERT d1 == d2

TEST artifact_matches_source:
    """bizra-constants.json matches current constants.py values."""
    artifact = json.loads(read_file("bizra-constants.json"))
    from core.integration.constants import (
        IHSAN_PRODUCTION, SNR_MINIMUM, ADL_GINI_THRESHOLD
    )
    ASSERT artifact["thresholds"]["ihsan"]["production"] == IHSAN_PRODUCTION
    ASSERT artifact["thresholds"]["snr"]["minimum"] == SNR_MINIMUM
    ASSERT artifact["thresholds"]["adl_gini"]["threshold"] == ADL_GINI_THRESHOLD

TEST verify_mode_detects_drift:
    """--verify flag catches stale artifact."""
    # Write a modified artifact
    write_file("bizra-constants.json", '{"stale": true}')
    result = subprocess.run(
        ["python", "scripts/generate_constants.py", "--verify"],
        capture_output=True
    )
    ASSERT result.returncode == 1
    ASSERT "stale" IN result.stderr.decode().lower()

TEST cross_repo_regex_replaced:
    """validate_cross_repo_consistency() should be deprecated or removed."""
    source = read_file("core/integration/constants.py")
    # The old regex-based validator should have a deprecation notice
    # or be replaced entirely with a reference to the JSON artifact
    ASSERT "DEPRECATED" IN source OR "bizra-constants.json" IN source
```

## Acceptance Criteria

1. `bizra-constants.json` exists in repo root, contains all thresholds
2. `python scripts/generate_constants.py --verify` passes in CI
3. Rust `build.rs` reads from JSON (not hardcoded)
4. Old regex-based `validate_cross_repo_consistency()` deprecated
5. BLAKE3 digest of artifact logged for verification
6. Full test suite GREEN

## Migration Path

1. Generate `bizra-constants.json` (Phase 59)
2. Add CI verification step (Phase 59)
3. Update Rust `build.rs` to consume JSON (Phase 59)
4. Deprecate regex-based validator (Phase 59)
5. Remove regex-based validator (Phase 60 — after all repos consume JSON)
