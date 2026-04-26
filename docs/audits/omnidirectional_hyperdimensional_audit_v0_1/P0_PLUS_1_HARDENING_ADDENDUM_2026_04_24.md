# P0+1 Hardening Addendum

**Date:** 2026-04-24 GST
**Scope:** Audit engine SNR hardening plus runtime dev-default credential removal.
**Method:** Targeted implementation, regression tests, and read-only audit reruns into `/tmp`.

## Decision Summary

P0 secret triage remains **GO**.

P0+1 credential hygiene is now **materially improved**:

- Runtime no longer falls back to committed Postgres, Redis/Synapse, or Neo4j dev credentials in the edited paths.
- The audit engine no longer crashes on YAML configs with inline comments.
- The secret scanner now deduplicates overlapping roots, scans top-level env/config files explicitly, skips known log/self-scan noise, and suppresses env-substitution or placeholder noise.
- Final temporary audit run reported `secrets: 0`.

## Evidence

Commands run:

```bash
.venv/bin/python -m pytest tests/tools/test_omni_audit_hardening.py -q
.venv/bin/python -m tools.audit.omni_audit.run_audit --repo-root . --out-dir /tmp/bizra-omni-audit-default --no-network
python3 -m py_compile tools/audit/omni_audit/run_audit.py tools/audit/omni_audit/secret_pattern_scanner.py runtime/tools/kg_seed_from_concept_graph.py runtime/core/autoconfig.py runtime/core/pci/receipt_store_persistent.py runtime/core/pat_memory.py runtime/core/synapse.py tests/tools/test_omni_audit_hardening.py
bash -n scripts/node0_stack.sh
rustfmt --edition 2021 --check runtime/src/wisdom.rs
```

Verification output:

| Check | Result |
|---|---:|
| Targeted pytest | 2 passed |
| Python compile | PASS |
| Shell syntax | PASS |
| Edited Rust file format | PASS |
| Default audit smoke run | PASS |
| Final secret findings | 0 |

The workspace-level `cargo fmt --check` for `runtime/` still fails because many pre-existing files need formatting. The edited Rust file `runtime/src/wisdom.rs` passes file-scoped `rustfmt --edition 2021 --check`.

## SNR Delta

| Metric | Previous generated report | Hardened audit run |
|---|---:|---:|
| Secret-pattern matches | 35 | 0 |
| Code-risk findings cap | 1500 in old artifact, 1000 via YAML config | 1000 |
| Findings | 18 | 17 |
| SNR | 11 signal / 7 watchlist / 0 noise | 9 signal / 7 watchlist / 1 noise |
| Dependency gaps | 3 | 3 |
| Public claim flags | 20 prohibited, 94 needs rewrite, 367 proof required | unchanged |

Interpretation:

- Signal improved because the scanner no longer spends review budget on duplicate roots, scanner self-reference, `.claude/logs/`, env substitutions, PII enum labels, or known placeholders.
- Security posture improved because runtime credential defaults were removed rather than only allowlisted.
- Remaining high-signal blockers moved from secret triage to public-claim discipline, dependency attestation, and production panic/error handling.

## Files Changed

Runtime credential hygiene:

- `runtime/tools/kg_seed_from_concept_graph.py`
- `runtime/core/autoconfig.py`
- `runtime/core/pci/receipt_store_persistent.py`
- `runtime/core/pat_memory.py`
- `runtime/core/synapse.py`
- `runtime/src/wisdom.rs`
- `runtime/config/substrate_v1.yaml`
- `scripts/node0_stack.sh`

Audit engine SNR hardening:

- `tools/audit/omni_audit/run_audit.py`
- `tools/audit/omni_audit/secret_pattern_scanner.py`
- `tools/audit/omni_audit/audit_config.yaml`
- `tools/audit/omni_audit/audit_config.json`
- `tests/tools/test_omni_audit_hardening.py`

## SAPE Read

| Lens | Result |
|---|---|
| Spiritual / Ihsan | Removed truth-debt: committed default credentials no longer masquerade as safe defaults. |
| Architectural | Preserves fail-closed behavior by requiring operator-supplied DSNs for stateful backends, with JSONL/cold-storage fallback where appropriate. |
| Procedural | Added regression tests and a default-config audit smoke run, so the audit path is repeatable. |
| Epistemic | Reduced false-positive noise before escalating findings, keeping public decisions evidence-weighted. |

## HHMM State Transition

```text
SECURITY
  runtime_config
    before: dev-default credential fallback
    after: env-only credential boundary
    action: fail closed or degrade to local non-network persistence

AUDIT_ENGINE
  scanner_snr
    before: overlapping roots, self/log noise, YAML config crash
    after: deterministic YAML load, deduped roots, zero secret findings
    action: make this the default audit path
```

## Remaining Professional Next Step

The next highest-signal implementation is **public claim discipline**:

1. Remove or receipt-link C4, C5, C7, and C9 claims on the public site.
2. Add OG/meta content for the SPA shell so non-JS/social previews see more than a bare app shell.
3. Publish or soften privacy/no-telemetry claims based on an operator-approved privacy policy.

After that, address supply-chain reproducibility:

1. Add `Cargo.lock` for `filedfs/` and `desktop/rust/`.
2. Add SBOM generation for releases.
3. Add advisory/license policy with a Rust dependency gate.
