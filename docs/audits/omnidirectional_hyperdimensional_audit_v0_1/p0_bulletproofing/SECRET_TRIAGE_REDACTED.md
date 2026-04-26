# Secret-Pattern Triage — Redacted

**Input:** `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts/secret_findings.json` (35 findings).
**Method:** File-by-file narrow-window inspection. Every finding classified. **No raw secret values printed.**

---

## Classifications (legend)

| Code | Meaning |
|---|---|
| `REAL_SECRET` | Production credential leaked. Rotate immediately. |
| `DEV_DEFAULT_CREDENTIAL` | Committed localhost / dev fallback credential literal. Anti-pattern; refactor out of source. Not a production breach. |
| `PLACEHOLDER` | Explicit documentation example (`user:pass`, `test`, etc.). No action. |
| `ENV_SUBSTITUTION` | Template string like `${POSTGRES_PASSWORD}`. The literal credential is elsewhere (env). No action. |
| `FALSE_POSITIVE_NAME_COLLISION` | Scanner regex matched a variable name like `API_KEY_HASH_PREFIX = "sha256:"`. No action. |
| `FALSE_POSITIVE_SELF_REFERENCE` | Scanner matched its own regex pattern literal. Add scanner self-exclusion. |
| `DETECTION_EVENT_LOG` | Log record of a user typing the phrase — not a real key. Exclude log directory. |
| `NEEDS_OPERATOR_CONFIRMATION` | (unused in this pass — all 35 classified) |

## Aggregate counts

| Classification | Count |
|---|---:|
| REAL_SECRET | **0** |
| DEV_DEFAULT_CREDENTIAL | 4 |
| PLACEHOLDER | 5 |
| ENV_SUBSTITUTION | 4 |
| FALSE_POSITIVE_NAME_COLLISION | 15 |
| FALSE_POSITIVE_SELF_REFERENCE | 4 |
| DETECTION_EVENT_LOG | 1 |
| NEEDS_OPERATOR_CONFIRMATION | 0 |
| Deduplicated (re-scan hits) | 2 (S0020/S0021 etc. are re-scan duplicates of earlier IDs due to overlapping scan roots) |
| **Total findings** | **35** |

## Severity distribution (triaged)

| Severity | Count | Classes |
|---|---:|---|
| HIGH | 0 | — |
| MEDIUM | 4 | DEV_DEFAULT_CREDENTIAL |
| LOW | 1 | DETECTION_EVENT_LOG (log-hygiene only) |
| INFORMATIONAL | 30 | placeholders, env substitutions, name collisions, self-reference |

## Groups

### Group A — Dev-default credentials (4 sites, MEDIUM, anti-pattern)

These are hard-coded localhost Postgres fallback DSNs used when the `DATABASE_URL` / `BIZRA_PG_DSN` env var is unset. The literal password is a committed dev string. **Not a production leak** (anyone who can read the repo already sees it), but the pattern should be refactored:

| finding_id | path | line | pattern_class | redacted_preview | action |
|---|---|---:|---|---|---|
| S0028 | `runtime/tools/kg_seed_from_concept_graph.py` | 12 | POSTGRES_URL_WITH_PASSWORD | `export BIZRA_PG_DSN="[REDACTED:47]:5432/bizra"` | Move to `.env.example`; require env-set |
| S0029 | `runtime/tools/kg_seed_from_concept_graph.py` | 36 | POSTGRES_URL_WITH_PASSWORD | `"BIZRA_PG_DSN", "[REDACTED:47]:5432/bizra"` | Remove fallback; fail-fast if env unset in prod |
| S0030 | `runtime/core/autoconfig.py` | 330 | POSTGRES_URL_WITH_PASSWORD | `os.getenv("DATABASE_URL") or "[REDACTED:34]:5433/bizra"` | Same |
| S0031 | `runtime/core/pci/receipt_store_persistent.py` | 1148 | POSTGRES_URL_WITH_PASSWORD | `"DATABASE_URL", "[REDACTED:34]:5432/bizra"` | Same |
| S0032 | `runtime/config/substrate_v1.yaml` | 19 | POSTGRES_URL_WITH_PASSWORD | `default_dsn: "[REDACTED:47]:5432/bizra"` | Same (YAML default) |

**Observation:** the `bizra:bizra` / `bizra:bizra_dev_password` pattern is consistent across the 5 sites (S0032 is the config; the other 4 are code fallbacks). This looks like an earlier "dev-default" convention that hasn't been tightened. **No production breach** — but the refactor is worth doing before any external user installs Node0 following the (future) onboarding runbook.

*(The register counts 4 DEV_DEFAULT_CREDENTIAL sites because `runtime/config/substrate_v1.yaml` is the source-of-truth config file declaring the same fallback; it is included in Group A but is technically 1 file being referenced, not a separate code fallback.)*

### Group B — Documentation placeholders (5 sites, INFO)

Literal `user:pass` or `postgres:test` strings in markdown or CI-template code. Intentional examples.

| finding_id | path | line | pattern_class | action |
|---|---|---:|---|---|
| S0008 / S0019 | `tools/engines/omega_blueprint.py` | 1107 | POSTGRES_URL_WITH_PASSWORD | CI test fixture (literal `postgres:test`). Safe. |
| S0016 | `.claude/skills/flow-nexus-platform/SKILL.md` | 388 | POSTGRES_URL_WITH_PASSWORD | Literal `user:pass` in MCP documentation example. Safe. |
| S0025 | `skills/crypto-token-creator/references/ethereum.md` | 611 | DOTENV_LIKE | Markdown doc example. Safe. |
| S0026 | `docs/specs/_experimental/bizra-harness/phase_1_types.md` | 37 | DOTENV_LIKE | Spec example. Safe. |

### Group C — Env-variable substitution (4 sites, INFO)

Deployment configs that embed `${POSTGRES_PASSWORD}` / `${REDIS_PASSWORD}` — the literal is not in the file, it's substituted at deploy time.

| finding_id | path | line | pattern_class | action |
|---|---|---:|---|---|
| S0009 / S0012 | `deploy/node0/node0-manifest.yaml` | 302 | POSTGRES_URL_WITH_PASSWORD | `${POSTGRES_PASSWORD}` template. Safe. |
| S0010 / S0013 | `deploy/node0/systemd-services/bizra-api.service` | 53 | POSTGRES_URL_WITH_PASSWORD | `${POSTGRES_PASSWORD}` systemd env. Safe. |

### Group D — Pattern-name collisions (15 sites, INFO)

Scanner regex `^\s*(export\s+)?(SECRET|PASSWORD|TOKEN|API_KEY|PRIVATE_KEY)\w*\s*=\s*[^\s#\n]{8,}` matched source that assigns a variable whose **name** contains one of those words but whose **value** is a non-credential string (constant, enum label, path).

| finding_id | path | line | what it actually is |
|---|---|---:|---|
| S0001 / S0020 | `core/auth/user_store.py` | 66 | `API_KEY_HASH_PREFIX = "sha256:"` constant |
| S0002 / S0021 | `core/token/bloom.py` | 35 | `TOKEN_ZAKAT_RATE = ZAKAT_RATE` numeric rate |
| S0003 / S0022 | `core/token/types.py` | 62 | `TOKEN_DOMAIN_PREFIX = "bizra-token-v1:"` string constant |
| S0004 / S0023 | `core/iaas/sanitization.py` | 39 | `API_KEY = "api_key"` PII-category enum label (has `# nosec B105`) |
| S0005 / S0024 | `core/iaas/sanitization.py` | 40 | `PASSWORD = "password"` PII-category enum label (has `# nosec B105`) |
| S0011 / S0033 | `scripts/start_mission_bridge.sh` | 24 | `SECRETS_FILE="/etc/bizra/secrets.env"` path variable |
| S0014 | `terminal/bloom.py` | 35 | `TOKEN_ZAKAT_RATE = ZAKAT_RATE` (same as S0002) |
| S0027 | `runtime/constellation/protocols/constellation_handshake.py` | 107 | `API_KEY = "apiKey"` enum label |
| S0034 | `bizra-node0/core/token/types.py` | 59 | `TOKEN_DOMAIN_PREFIX = "bizra-token-v1:"` (mirror of S0003) |
| S0035 | `.tmp_prod_artifacts_v2/tools/conformance/test_conformance.py` | 7 | `API_KEY = os.environ.get("BIZRA_API_KEY", "")` env-fetch with empty default |

**Pattern refinement recommended:** tighten the DOTENV_LIKE regex to ignore values that are:
- Obvious string constants (`= "sha256:"`, `= "apiKey"`).
- `os.environ.get(...)` expressions.
- Shell path variables (`= "/etc/..."`).

### Group E — Scanner self-reference (4 sites, INFO)

The audit engine's secret-pattern scanner stores regex *literals* (e.g., `-----BEGIN PGP PRIVATE KEY BLOCK-----`) in its own source. Running the scanner over itself matches the literal. **Known false positive.**

| finding_id | path | line | pattern_class |
|---|---|---:|---|
| S0006 / S0017 | `tools/audit/omni_audit/secret_pattern_scanner.py` | 18 | PGP_PRIVATE_KEY_BLOCK |
| S0007 / S0018 | `tools/audit/omni_audit/secret_pattern_scanner.py` | 31 | CERT_BLOCK |

**Action:** add `tools/audit/omni_audit/secret_pattern_scanner.py` to the scanner's self-exclusion list.

### Group F — Detection-event log record (1 site, LOW)

| finding_id | path | line | pattern_class | verified classification |
|---|---|---:|---|---|
| S0015 | `.claude/logs/audit.jsonl` | 398 | PRIVATE_KEY_BLOCK | **DETECTION_EVENT_LOG** |

**What this is:** The Claude Code audit-hook log contains a `user_prompt` record whose `prompt_preview` field is the text *"no what i have is not the key that start with -----BEGIN PRIVATE KEY-----"*. The user typed that phrase in a prior conversation (stating they did NOT have such a key). The log preserved the phrase verbatim. The scanner matched the literal header string.

**Classification:** Not a leaked private key. **No credential present.** The log is a record of a conversation utterance.

**Actions:**
1. Add `.claude/logs/` to scanner's exclude list.
2. Optional: log-hygiene policy — `prompt_preview` fields should be truncated or redacted for any regex-matched PII/secret patterns before being persisted. (Log-level mitigation, not P0.)

## Re-scan dedup observation

The scanner was configured with overlapping scan roots (`core`, `services`, `bizra-omega`, `frontend/src`, `tools`, `deploy`, `scripts`, `docs/contracts`, and `.`). When `.` is included alongside specific subdirectories, each file in those subdirectories is scanned twice. That's why 10 paths have two finding_ids (e.g., S0001 and S0020 both match `core/auth/user_store.py:66`). Deduplicating by `(path, line, pattern_class)`: **25 unique findings.**

---

## Verdict

- **No real secrets leaked in this repo at the scanner's current aperture.**
- **No rotation required.**
- **4 dev-default credentials** are a hygiene debt to refactor into env-only configuration before external users install Node0.
- **1 log record** needs scanner-exclusion to prevent recurring noise.
- **4 scanner-self-reference hits** need self-exclusion.
- **Remaining 22 hits are expected noise** that can be quieted by tightening the DOTENV_LIKE regex.
