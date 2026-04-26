# Receiptification Requirements

**Purpose:** For every `SAFE_WITH_RECEIPT` claim, document *what evidence* would let it return to live public copy with numeric specificity.

**Discipline floor (brand canon §5):** If measured → cite receipt. If unmeasured → direction only. If uncertain → mark uncertain.

---

## §C2 — "no telemetry"

**Required evidence to keep the claim as-is publicly:**

1. **Published privacy policy** at `https://bizra.ai/privacy` covering:
   - What leaves the node (if anything) and to whom.
   - Opt-in vs opt-out posture.
   - Third-party integrations (none declared today).
   - Contact for privacy inquiries.
2. **Architecture-level attestation** — a short public doc (possibly on `/under-the-hood/`) describing the local-first runtime and explicitly naming places where optional data-sharing could occur (URP reconciliation, federation, cloud fallback for inference if operator opts in).
3. **Code reference** — link to the relevant modules (`bizra-omega/bizra-node/`, `bizra-omega/bizra-federation/`, etc.) so technical readers can verify.
4. **Public-key posture** — if any node-to-node communication happens, document the key identity posture.

**Until all four exist:** soften to `"Your actions stay on your node unless you choose to share."` (per `CLAIM_SAFE_REWRITE_PACK.md §C2`).

## §C3 — "Ed25519 receipt signatures"

**Required for consumer-hero use (it's INTERNAL_ONLY today):**

1. **Sub-page** `https://bizra.ai/under-the-hood/receipts` containing:
   - What a receipt is.
   - How BLAKE3 chaining works.
   - How Ed25519 signatures are verified (with a snippet + public key publication).
   - Link to `bizra-omega/bizra-core/src/canonical_receipt.rs`.
   - Sample receipt JSON + verification walkthrough.
2. **Public key published** at a stable URL, with rotation policy documented.
3. **Sample receipt chain verifier** — a tiny Python / Rust CLI tutorial showing a reader how to verify a chain themselves.

**Until all three exist:** keep the hero at `"Every action leaves a receipt."` (per `§C3`). The technical term stays in dev docs + investor deck only.

## §C4 — "cost per action $0.10 → $0.008"

**Required evidence for any exact-$ claim:**

1. **Published benchmark methodology** at `https://bizra.ai/receipts/cost/methodology.md` covering:
   - What "one action" is defined as (reproducible unit).
   - Cost components (compute, electricity, opportunity cost, optional cloud fallback).
   - Measurement protocol (N samples, timestamp, machine spec).
2. **Timestamped benchmark run receipt** at `https://bizra.ai/receipts/cost/<YYYY-MM-DD>.json` with:
   - `measured_at_utc`
   - `commit_hash`
   - `n_samples`
   - `p50_cost_usd`, `p95_cost_usd`, `min`, `max`
   - `machine_spec` (CPU, RAM, GPU, inference backend)
   - Signature (BLAKE3 + Ed25519 consistent with runtime receipt style).
3. **Reproducibility note** — a script / harness path in the repo that a reader can run.

**Paid-ad-ready variant** requires receipt to be <90 days old. Older than that → re-run.

**Until all three exist:** directional wording only (`CLAIM_SAFE_REWRITE_PACK.md §C4`).

## §C5 — "SNR 0.974"

**Required evidence:**

1. **Benchmark protocol** at `https://bizra.ai/receipts/snr/methodology.md` covering:
   - Definition of signal and noise for BIZRA's specific output domain.
   - Test set (source, license, size, curation method).
   - Comparison baseline (if any).
   - Statistical methodology (confidence interval, repro seed).
2. **Benchmark run receipt** at `https://bizra.ai/receipts/snr/<YYYY-MM-DD>.json` with full numeric set (mean, stdev, N, commit, machine spec).
3. **Peer review or public reproduction invitation.**

Until: directional only (`§C5`).

## §C6 — "8 072 verified tests" (or replacement)

**Simplest receipt chain:**

1. `https://bizra.ai/receipts/tests/<YYYY-MM-DD>.json` with:
   - `commit_hash`
   - `pytest_collect_count` (from `pytest --collect-only -q | tail -n 1`)
   - `cargo_test_count` (from `cargo test --workspace -- --list | grep -c ': test$'`)
   - `ci_run_url` (GitHub Actions run URL)
   - `measured_at_utc`
2. **Live reference on site:** hero says `"Thousands of verified tests…"` (directional), sub-page says `"As of <date>, CI runs <N> verified tests. See receipt."`

**Until receipt is published:** directional only.

## §C8 — "Ihsan Gate >= 0.95"

**Required context for public use (architecturally accurate, not a user metric):**

1. **Sub-page** `https://bizra.ai/ihsan` covering:
   - What Ihsan means (reference brand canon §6.4).
   - Why 0.95 specifically (per `core/integration/constants.py`).
   - What the gate does: blocks ship if score below threshold.
   - Clear framing: it is an internal policy threshold, not a user-visible quality score.
2. **Threshold source receipt** — link + sha256 of `core/integration/constants.py` at a specific commit.

**With these:** keep claim with context. **Without:** soften to "high conscience threshold" (no number).

## §C9 — "73 of 100 nodes remaining"

**Required for live-counter variant:**

1. **Source-of-truth store** — Postgres / Airtable / config file storing active-cohort-member-count.
2. **Public read-only endpoint** `GET https://bizra.ai/api/cohort/status` returning:
   ```json
   {
     "cohort_name": "Genesis-100",
     "active_members": 27,
     "cap": 100,
     "updated_at_utc": "2026-04-24T07:00:00Z"
   }
   ```
3. **Hero JS** reads + renders: `"Early-access cohort: 27 / 100. Updated daily."` with `updated_at` visible.
4. **Admin update flow** documented (how does the number move up?).
5. **Cadence policy** — do stale timestamps downgrade the claim automatically? (Recommended: if `updated_at` is older than 7 days, frontend renders `"Early-access cohort forming — join waitlist"` fallback.)

**Until all five exist:** remove the numeric claim; use waitlist framing (`§C9` option A).

---

## Summary — evidence-to-claim mapping

| Claim | Min evidence to publish | Current state |
|---|---|---|
| C2 no telemetry | privacy policy + architecture attestation | ❌ unpublished |
| C3 Ed25519 | receipts sub-page + sample verifier | ❌ unpublished |
| C4 cost $ | benchmark methodology + signed receipt | ❌ unpublished |
| C5 SNR number | benchmark protocol + signed receipt | ❌ unpublished |
| C6 test count | timestamped CI receipt | ❌ unpublished |
| C8 Ihsan 0.95 | `/ihsan` sub-page | ❌ unpublished |
| C9 N / 100 nodes | live counter backed by source-of-truth | ❌ unwired |

**Bottom line:** every numeric / technical claim on the current site is currently un-receipted. The rewrite pack (`CLAIM_SAFE_REWRITE_PACK.md`) is the bridge until receipts land.

## Re-verification after receipt publication

Once a claim's evidence lands:

1. Run `python3 -m tools.audit.omni_audit.run_audit` on the bizra-data-lake repo (where receipts also live).
2. Verify the specific claim pattern is detected but can now be linked to the receipt file.
3. Update `CURRENT_PUBLIC_CLAIMS_REGISTER.md` classification from `SAFE_WITH_RECEIPT` → `SAFE_NOW`.
4. Reintroduce the claim to the live site with the receipt link inline.

Receipts expire — define a refresh cadence per claim (suggested: cost / SNR every 90 days; test count every release; cohort counter daily).
