# Genesis-100 Gate Specification

> **Version:** 1.0.0
> **Authority:** Enforceable Spine v1.0, §7 (Evidence & Proof), §8 (SNR/Ihsān)
> **Prerequisite:** Node0 production canon SEALED (Waves 0–4 complete)

## Overview

Genesis-100 is the release-readiness gate for the first 100 sovereign nodes.
It validates that Node0's production surface meets the quality, security,
performance, and governance thresholds required for multi-node deployment.

**Rule:** Genesis-100 does NOT re-open Node0 birth semantics. Node0 lifecycle
is sealed. This gate validates the *shipping surface*, not the *birth contract*.

Standing on Giants:
- Deming (PDCA, 1950): Genesis-100 IS the "Act" of the final PDCA cycle
- Lamport (distributed consensus, 1978): multi-validator readiness
- Nakamoto (evidence chain, 2008): every gate produces a signed receipt
- PMBOK 7th Ed: quality gate before release transition

---

## Gate Structure

68 checks across 5 SAT agent domains. Each check is PASS/FAIL.
Release requires **ALL 68 checks PASS**.

### L1 Sentinel (S1) — Security: 12 checks

| # | Check | Method | Pass Criteria |
|---|-------|--------|---------------|
| 1.1 | JWT fail-closed in production | Unit test | `BIZRA_ENV=production` without secret → ValueError |
| 1.2 | API auth fail-closed | Unit test | Missing API keys in production → startup blocked |
| 1.3 | Ghost bridge disabled by default | Config check | `GHOST_WS_ENABLED` default is `false` |
| 1.4 | WebSocket auth contract | Unit test | Auth failure → `ws.close(1011)`, not JSONResponse |
| 1.5 | No hardcoded secrets | Static scan | `ruff` + `bandit` clean on `core/` |
| 1.6 | Dependency audit clean | `pip-audit` | No known CVEs in direct deps |
| 1.7 | Rust dependency audit | `cargo-audit` | No known advisories in workspace |
| 1.8 | BLAKE3 hash gate | `ci_blake3_gate.py` | SEC-001 compliance |
| 1.9 | Ed25519 identity present | Runtime check | Genesis identity key exists in sovereign_state |
| 1.10 | Evidence chain integrity | Runtime check | Chain hash verification passes |
| 1.11 | No world-readable secrets | File permission check | `/etc/bizra-node0/node0.env` mode 640 |
| 1.12 | Systemd security score | `systemd-analyze security` | Score ≤ 4.0 (MEDIUM exposure) |

### L2 Oracle (S2) — Constitutional Verification: 14 checks

| # | Check | Method | Pass Criteria |
|---|-------|--------|---------------|
| 2.1 | Ihsān threshold = 0.95 | `constants.py` read | Value matches constitutional source |
| 2.2 | ADL Gini ceiling = 0.35 | `constants.py` read | Value matches constitutional source |
| 2.3 | Zakat rate = 2.5% | `constants.py` read | Value matches Al-Baqarah 2:43 |
| 2.4 | BLOOM soulbound (0% transfer) | `constants.py` read | Transfer rate = 0.0 |
| 2.5 | SNR minimum = 0.85 | `constants.py` read | Reject tier threshold |
| 2.6 | P5 frozen flag | Architecture check | Ethicist does not accept forest updates |
| 2.7 | S2 frozen flag | Architecture check | Oracle does not accept forest updates |
| 2.8 | Cross-lang sync | CI gate | Python ↔ Rust thresholds match |
| 2.9 | Lifecycle schema = 2.0.0 | `node0_lifecycle.json` read | Schema version correct |
| 2.10 | Status gates ≥ 11 | Lifecycle read | At least 11 status-determining gates |
| 2.11 | Availability gates = 4 | Lifecycle read | Exactly 4 informational gates |
| 2.12 | Ready Only rule | Lifecycle read | `status == "ready"` requires all status gates true |
| 2.13 | DoD version = 1.2 | DoD file read | LOCKED version matches |
| 2.14 | Hard gates = 19 | DoD file read | Exactly 19 hard gates in verification |

### L3 Ledger (S3) — Evidence & Economics: 10 checks

| # | Check | Method | Pass Criteria |
|---|-------|--------|---------------|
| 3.1 | SEED retention = 100% | `constants.py` read | User keeps all earned SEED |
| 3.2 | Riba rate = 0% | `constants.py` read | Forbidden per Al-Baqarah 2:278 |
| 3.3 | Evidence chain non-empty | State check | At least 1 signed receipt exists |
| 3.4 | Receipt hash algorithm = BLAKE2b | Receipt inspection | Hash prefix/algorithm correct |
| 3.5 | Receipt signature = Ed25519 | Receipt inspection | Signature algorithm correct |
| 3.6 | ActionReceipt schema valid | JSON schema check | All required fields present |
| 3.7 | Genesis ceremony receipt | State check | Ceremony JSON exists with PASS result |
| 3.8 | MVSA proof receipt | State check | `prove-mvsa` receipt exists |
| 3.9 | Version lock receipt | Test lock check | Coverage ratchet operational |
| 3.10 | Evidence index integrity | Hash chain walk | No gaps, no forks in chain |

### L4 Conductor (S4) — Performance & Infrastructure: 13 checks

| # | Check | Method | Pass Criteria |
|---|-------|--------|---------------|
| 4.1 | Python 3.11+ | Runtime check | `sys.version_info >= (3, 11)` |
| 4.2 | All 31 core imports clean | Import test | Zero ImportError on core modules |
| 4.3 | Test suite passes | `pytest tests/` | 0 failures, 0 errors |
| 4.4 | Lint clean | `ruff check core/` | 0 violations |
| 4.5 | Format clean | `black --check core/` | 0 reformats needed |
| 4.6 | Health endpoint responds | `node0_standalone.py health` | Exit code 0, lifecycle readable |
| 4.7 | Activate command works | `node0_standalone.py activate` | Genesis identity created |
| 4.8 | Task command works | `node0_standalone.py task` | ActionReceipt generated |
| 4.9 | Serve command starts | `node0_standalone.py serve` | API responds on port 8091 |
| 4.10 | MVSA preflight passes | `mvsa-preflight.sh` | All artifact checks pass |
| 4.11 | Native Linux filesystem | Path check | Not running on `/mnt/c` |
| 4.12 | Systemd unit valid | `systemd-analyze verify` | No critical errors |
| 4.13 | Resource limits set | Systemd check | MemoryMax + CPUQuota configured |

### L5 Ambassador (S5) — Federation & Release Readiness: 19 checks

| # | Check | Method | Pass Criteria |
|---|-------|--------|---------------|
| 5.1 | UPSTREAM_IMPORT_MANIFEST exists | File check | Manifest in parent repo |
| 5.2 | Dependency closure documented | Manifest read | ≥ 29 modules listed |
| 5.3 | README exists | File check | Production README present |
| 5.4 | RELEASE.md exists | File check | Signed release policy present |
| 5.5 | MVSA spec exists | File check | `NODE0_STANDALONE_READINESS.md` |
| 5.6 | DoD exists | File check | `BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md` |
| 5.7 | Correction matrix exists | File check | `NODE0_DOD_CORRECTION_MATRIX.md` |
| 5.8 | Runbook exists | File check | `OPERATIONS_RUNBOOK.md` with §9 |
| 5.9 | CI pipeline exists | File check | `node0-ci.yml` with 4+ jobs |
| 5.10 | Installer exists | File check | `install-node0-linux.sh` |
| 5.11 | Certification script exists | File check | `certify-linux.sh` |
| 5.12 | Systemd unit exists | File check | `bizra-node0.service` |
| 5.13 | Logrotate config exists | File check | `bizra-node0.logrotate` |
| 5.14 | .gitignore exists | File check | Production ignores configured |
| 5.15 | pyproject.toml exists | File check | Package metadata present |
| 5.16 | Rust workspace scoped | Cargo.toml read | ≤ 5 crate members (not 22) |
| 5.17 | No archive material | Directory check | No season/archive dirs in production |
| 5.18 | No frontend experiments | Directory check | No unrelated frontend in production |
| 5.19 | Git clean | `git status` | No uncommitted changes in production |

---

## Execution

```bash
# Run the full Genesis-100 gate
python scripts/genesis_100_gate.py \
  --project-root /opt/bizra-node0 \
  --state-dir /var/lib/bizra-node0 \
  --report /tmp/genesis_100_report.json

# Quick subset (L4 Conductor only — performance)
python scripts/genesis_100_gate.py --section conductor

# CI mode (GitHub Actions output)
python scripts/genesis_100_gate.py --github-output "$GITHUB_OUTPUT"
```

## Release Decision

| Result | Action |
|--------|--------|
| 68/68 PASS | Release approved — tag, sign, publish |
| 65-67/68 | Review failures — fix or document exceptions |
| < 65/68 | Release BLOCKED — fix all critical failures first |

**Exception process:** Any check can be waived by SAT-5 consensus with documented
justification in `NODE0_DOD_CORRECTION_MATRIX.md`. Waivers expire after 30 days.

---

## Evidence

Genesis-100 gate produces:
```json
{
  "gate": "genesis-100",
  "version": "1.0.0",
  "timestamp": "ISO-8601",
  "checks_total": 68,
  "checks_passed": N,
  "sections": {
    "sentinel": {"passed": N, "total": 12},
    "oracle": {"passed": N, "total": 14},
    "ledger": {"passed": N, "total": 10},
    "conductor": {"passed": N, "total": 13},
    "ambassador": {"passed": N, "total": 19}
  },
  "certified": true|false,
  "receipt_hash": "BLAKE2b(...)",
  "signature": "Ed25519(...)"
}
```
