# Security Audit — BIZRA v0.1

**Scope:** secrets, key storage, receipt signatures, identity binding, SSRF risk, prompt / tool injection, permission scopes, local-automation blast radius, public website security claims, discipline ("no printed secrets").

**Engine note:** the secret-pattern scanner in `tools/audit/omni_audit/secret_pattern_scanner.py` **never prints matched values.** All findings below use redacted previews of the form `[REDACTED:<n>]`. See `artifacts/secret_findings.json` for full (still-redacted) detail.

---

## 1. Secrets — current clean, continuous gate required

`artifacts/secret_findings.json` currently contains **0 secret-pattern matches**. This supersedes the older P0 bulletproofing snapshot that contained 35 raw candidate matches before scanner hardening and runtime credential cleanup.

| Check | Current result | Disposition |
|---|---:|---|
| Secret-pattern matches | 0 | No current triage queue |
| Matched values printed | 0 | Scanner emits redacted previews only |
| Continuous scanner gate | Not wired | Add pre-commit / CI coverage |

**Action (repo-ops):**

1. Keep `tools/audit/omni_audit/secret_pattern_scanner.py` wired into the repeatable audit path.
2. Add a pre-commit or CI gate so new secret-pattern matches fail before merge.
3. If a future run finds a REAL credential, rotate immediately, remove from repo history through an approved incident lane, and keep actual values out of all tooling output.

**DO NOT do in this session:** rotate credentials, delete files, rewrite history, or publish any secret-like audit payload.

## 2. Receipt signatures — identity binding

**Architecture:**
- `bizra-omega/bizra-core/src/canonical_receipt.rs` — one receipt per visible effect.
- Ed25519 signing (via `ed25519-dalek` 2.2).
- BLAKE3 chaining via `previous_receipt_hash`.
- Full-body signature (PR #50 closed a weak-signature vuln where only `receipt_id` hash was signed).

**Assessment:** ✅ **Sound.** Receipt identity is architecturally bound; the recent full-payload fix is the right direction. Verify PR #50 is merged before claiming this publicly.

## 3. SSRF risk

**Pattern scan:** 7 Python URL-fetch sites (`requests` / `httpx` / `urllib.request`). Without reading each, the risk is only indicative.

**Action:** Spot-check each of the 7 for:
- User-controlled URLs without allow-list.
- Inner-network addresses reachable from request path.
- DNS-rebinding guards.

## 4. Prompt / tool injection risk

**Observable:** agent infrastructure (`bizra-omega/bizra-agent/`, `bizra-omega/bizra-action/`) processes untrusted input. No code-level injection audit performed by this pass.

**Recommendation:** separate lane — "agent injection red-team" — deferred until explicitly authorized.

## 5. Permission scopes / local automation blast radius

**Observable findings:**
- 1 Python `subprocess(..., shell=True)` occurrence — MEDIUM — triage for command injection.
- 12 Python `eval/exec` occurrences — review each; usually legitimate in DSL parsers, occasionally dangerous.
- Rust `std::process::Command::new` calls present — review to ensure argument sanitization.

**Recommendation:** explicit `deny.toml` / ruff rule preventing new `shell=True` / `eval` sites.

## 6. Public website security claims

**Classification from `WEBSITE_PUBLIC_CLAIMS_AUDIT.md`:**

| Claim | Class | Action |
|---|---|---|
| "Ed25519 receipt signatures" | PROOF_REQUIRED (B) | Keep in dev docs; remove from consumer hero. |
| "no telemetry" | PROOF_REQUIRED (B) | Publish privacy policy OR soften to "your node keeps your data unless you choose to share." |
| "local agents / no cloud dependency" | PROOF_REQUIRED (B) | Accurate at core-binary level; reword to reflect cloud-optional reality. |
| "SNR 0.974" | NEEDS_REWRITE (C) | Publish benchmark receipt OR remove. |
| "100% pass rate" | NEEDS_REWRITE (C) | Replace with policy claim. |
| "73 of 100 nodes remaining" | NEEDS_REWRITE (C) | Wire live counter OR remove. |
| "cost per action $0.10 → $0.008" | NEEDS_REWRITE (C) | Remove; directional reframe. |
| "Ihsan Gate ≥ 0.95" | PROOF_REQUIRED (B) | Accurate to `constants.py`; contextualize as internal gate. |

## 7. No printed secrets — design-enforced

- Secret scanner writes `[REDACTED:<n>]` in place of matched value.
- JSON artifact stores only `pattern_class`, `path`, `line`, `redacted_preview`.
- No tooling output embeds a matched value.
- Reports (this file) never show matched values — paths only.

**Assessment:** ✅ Design intent honored.

---

## Security debts (actionable, ranked)

| # | Debt | Severity | Owner | ETA hint |
|---|---|---|---|---|
| SD1 | Add continuous secret-pattern scanner gate; current findings remain at 0 | MEDIUM | repo-ops | 1 hour |
| SD2 | Pre-commit secret-scanner gate | MEDIUM | repo-ops | 1 hour |
| SD3 | Consumer hero: remove cryptographic-detail claims, keep in dev docs | MEDIUM | operator + web lead | 1–2 hours |
| SD4 | Privacy policy decision (publish or retire "no telemetry" claim) | MEDIUM | operator | 2–4 hours |
| SD5 | Spot-check 7 Python URL-fetch sites for SSRF | LOW | backend lead | 2 hours |
| SD6 | Remove `shell=True` from Python; add lint rule forbidding it | LOW | backend lead | 1 hour |
| SD7 | Agent-injection red-team lane (separate authorization required) | LOW until needed | architecture lead | deferred |

## What this audit did NOT touch

- No credential rotation.
- No file deletion or rewrite.
- No public surface modification.
- No PR opened against site source.
- No ingestion of any secret value into any output.
