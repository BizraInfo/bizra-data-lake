# Website & Public Claims Audit — BIZRA v0.1

**Capture source:** `artifacts/website_claims.json` + `artifacts/website_snapshot.txt` (operator-supplied pre-check skeleton in `--no-network` mode).

---

## 1. Site surface


| URL                  | Final URL           | Status | Rendering                                                 |
| -------------------- | ------------------- | ------ | --------------------------------------------------------- |
| `https://bizra.info` | `https://bizra.ai/` | 302    | Brand-defense redirect — no separate claim surface.       |
| `https://bizra.ai`   | `https://bizra.ai/` | 200    | Client-side rendered SPA. Non-JS fetchers see shell only. |


**Shell content observable via HTTP:**

- Title: `BIZRA | The Sovereign Future`
- One button: `Clear local data`

Everything else is JavaScript-rendered. A headless-Chromium DOM capture is required for verified evidence in future runs (currently operator pre-check).

## 2. Claim classification

**Legend:** A = Brand-safe. B = Needs source-chain / receipt. C = Needs rewrite. D = Internal / private only. E = Prohibited.


| #   | Public claim                            | Class | Action                                                                                                    |
| --- | --------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------- |
| C1  | "local agents / no cloud dependency"    | **B** | Reword: "Designed to run on your machine — your keys, your data, your node."                              |
| C2  | "no telemetry"                          | **B** | Publish privacy policy OR soften to "Your actions don't leave your node unless you choose to share them." |
| C3  | "Ed25519 receipt signatures"            | **B** | Keep in dev / investor docs. Remove from consumer hero.                                                   |
| C4  | "cost per action $0.10 → $0.008"        | **C** | Remove. Directional reframe: "Designed to make verified action radically cheaper than cloud AI."          |
| C5  | "SNR 0.974"                             | **C** | Remove. Reframe: "A signal-vs-noise discipline that keeps outputs tied to evidence."                      |
| C6  | "8 072 verified tests"                  | **B** | Link to CI run with timestamp + commit hash OR soften to "Thousands of verified tests."                   |
| C7  | "100% pass rate"                        | **C** | Remove. Replace with "CI must pass before merge — the same discipline we apply to our claims."            |
| C8  | "Ihsan Gate >= 0.95"                    | **B** | Accurate to `constants.py`; contextualize as internal-gate.                                               |
| C9  | "73 of 100 nodes remaining"             | **C** | Live counter OR remove. Never use in paid ads.                                                            |
| K1  | "BIZRA is live" (inside kit's own copy) | **C** | Soften to "The Seed is public."                                                                           |


## 3. Surface applicability


| Surface            | A   | B (with cite)        | C              | D             | E   |
| ------------------ | --- | -------------------- | -------------- | ------------- | --- |
| Organic social     | ✅   | ✅ link               | ❌              | ❌             | ❌   |
| Paid ads           | ✅   | ⚠️ only with receipt | ❌              | ❌             | ❌   |
| bizra.ai hero      | ✅   | ⚠️ move to sub-page  | ❌              | ❌             | ❌   |
| bizra.ai sub-pages | ✅   | ✅ with receipt       | ⚠️ rewrite     | ❌             | ❌   |
| Investor deck      | ✅   | ✅                    | ⚠️ with caveat | ✅ with caveat | ❌   |
| Press release      | ✅   | ✅ with receipt       | ❌              | ❌             | ❌   |


## 4. Recommended hero replacement

### English

> **BIZRA**
> **The Seed of Sovereign Intelligence.**
>
> A human-first AI ecosystem built on meaning, proof, and Ihsan.
> Not another chatbot. Not another platform that owns you.
> One human. One node. One sovereign path.
>
> **Build with meaning. Act with proof. Grow with Ihsan.**

### Arabic

> **بذرة**
> **بذرة الذكاء السيادي.**
>
> منظومة ذكاء إنساني أولاً، مبنية على المعنى والبرهان والإحسان.
> ليست أداة محادثة أخرى. وليست منصة تملكك.
> إنسان واحد. عقدة واحدة. طريق سيادي واحد.
>
> **ابنِ بالمعنى. اعمل بالبرهان. وانمُ بالإحسان.**

Numeric claims (SNR, test count, pass rate, cost, node counter) move to a separate "Under the Hood" page linked from the hero with a clear "technical spec" label — backed by published benchmark receipts.

## 5. Platform-policy risk

For Meta / Google / LinkedIn ad policy reviewers, C4 / C5 / C7 / C9 are the four highest-risk live-site claims:

- **C4 ($ cost figures)** — "unsupported economic claim" policy category; triggers substantiation review.
- **C5 (SNR exact)** — "unsupported quantitative claim"; may cause rejection without substantiation link.
- **C7 (100% pass)** — brittle; any future CI red falsifies the claim mid-campaign.
- **C9 (73/100 nodes)** — if not backed by a live counter, "deceptive practices" risk.

## 6. Measurement recommendations


| Area                                  | Measure                                                                      |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| Link-preview quality                  | Add `og:title` / `og:description` / `og:image` to the SPA shell HTML         |
| Social click-through                  | Attach UTM on every organic + paid link                                      |
| Claim-alignment drift                 | Re-run this audit monthly; diff the claims_register                          |
| Live numeric counter (if C9 retained) | Wire to source of truth (DB / Airtable / config) + publish refresh frequency |


## 7. Audit-of-claims debts


| #   | Debt                                            | Severity | Action                        |
| --- | ----------------------------------------------- | -------- | ----------------------------- |
| WC1 | Live site carries C4/C5/C7/C9 without receipts  | HIGH     | Remove / receipt-ify          |
| WC2 | No privacy policy published                     | MEDIUM   | Publish OR soften C1/C2       |
| WC3 | OG tags absent in shell                         | LOW      | Add                           |
| WC4 | No headless-Chromium capture for audit evidence | LOW      | Add to audit engine           |
| WC5 | Arabic parity on any public claim change        | MEDIUM   | Human Arabic reviewer in loop |


