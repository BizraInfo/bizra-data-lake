# Website Audit — bizra.ai + bizra.info

**Audit date:** 2026-04-24 (GST)
**Method:** HTTP fetch (WebFetch) + operator-supplied ChatGPT pre-check for SPA-rendered content
**Scope:** Public-facing claim surface only. NO website source code was read or edited.

---

## 1. Redirect / URL behavior

| URL | Final URL | Status | Notes |
|---|---|---|---|
| `https://bizra.info` | `https://bizra.ai/` | 302 Found | Confirmed redirect. bizra.info is a brand-defense domain pointing at the primary. |
| `https://bizra.ai` | `https://bizra.ai/` | 200 | Primary domain. Client-side rendered SPA. |

**Finding:** redirect behavior is correct. Both domains land on the same surface. No split claim surface to reconcile.

## 2. Rendering observation

`bizra.ai` returns a minimal HTML shell to plain HTTP fetchers. The visible payload via `WebFetch`:

- `<title>` equivalent: **"BIZRA | The Sovereign Future"**
- Visible UI: one button labeled **"Clear local data"**
- No hero text, no claims, no pricing, no footer content rendered server-side

This means all the **substantive** public content (taglines, technical claims, numeric metrics, cost figures) is rendered client-side by JavaScript after page load. Non-JS crawlers (including this WebFetch tool, many SEO bots, some link previews) see **only the shell**.

**Implication:**
- **Social link previews may be degraded** (no OG description unless set as meta tags in the shell).
- **SEO surface is the shell title alone**, unless SSR/prerender is added.
- This audit relies on the operator-supplied ChatGPT pre-check for the rendered content. A fresh browser-DOM capture is recommended before any paid-ad launch tied to on-site copy.

## 3. Current public claims on bizra.ai (from operator pre-check)

Reproduced as supplied — NOT independently verified against a fresh DOM:

| # | Claim category | Exact/approximate wording (per pre-check) | Classification (see §4) |
|---|---|---|---|
| C1 | Execution model | "local agents / no cloud dependency" | B — needs proof before paid ads |
| C2 | Telemetry | "no telemetry" | B — needs proof |
| C3 | Cryptography | "Ed25519 receipt signatures" | A/B — architecturally true in runtime but not yet receipt-bound to public deliverable; B for marketing until a public receipt exists |
| C4 | Cost economics | "cost per action dropping from about $0.10 toward $0.008" | **C — needs rewrite** (precise $ numbers without a published methodology / receipt chain = ad liability) |
| C5 | Signal quality | "SNR 0.974" | **C — needs rewrite** (requires a published benchmark + methodology receipt) |
| C6 | Test count | "8,072 verified tests" | B — can be backed by repo state IF methodology is defined and published |
| C7 | Pass rate | "100% pass rate" | **C — needs rewrite** (brittle; one future failure breaks the claim; also "100%" in marketing is a compliance flag) |
| C8 | Ihsan threshold | "Ihsan Gate >= 0.95" | A/B — accurate to `constants.py` value; safe with context that it's an internal-gate threshold, not a user-visible metric |
| C9 | Scarcity hook | "73 of 100 nodes remaining" | **C — needs rewrite or removal** (manufactured-scarcity claim. If real, needs a live counter with auditable source. If aspirational, needs clear framing.) |

**Bottom line on current site copy: C4, C5, C7, C9 are the highest public liability.** They are precise numeric claims used in a marketing context that do not obviously link to a published, receipted methodology. In paid-ad contexts (especially Meta and Google) these trigger the "unsupported quantifiable claim" policy category and can cause ad rejection or account review.

## 4. Classification legend

| Code | Meaning | Paid-ad use | Organic use |
|---|---|---|---|
| **A** | Brand-safe language. Identity, mission, philosophy. No numeric promise. | ✅ | ✅ |
| **B** | Technically defensible but needs a proof / receipt chain before public quantitative reuse. | ⚠️ requires citation | ✅ with framing |
| **C** | Requires rewrite. Either too quantitative without receipts, too brittle, or over-promises. | ❌ | ❌ |
| **D** | Internal/private-deck only. Never in public ads. | ❌ | ❌ |

## 5. Recommended safer public wording (per claim)

| # | Unsafe / brittle | Safer public wording (brand-safe) |
|---|---|---|
| C1 | "local agents / no cloud dependency" | "Designed to run on your machine — your keys, your data, your node." |
| C2 | "no telemetry" | "Built so your actions don't leave your node unless you choose to share them." (or keep "no telemetry" IF backed by a published privacy policy + runtime audit receipt) |
| C3 | "Ed25519 receipt signatures" | **Keep** in technical / architect-audience contexts (GitHub readme, dev docs). **Remove** from consumer hero copy — it's jargon that doesn't convert outside engineering audiences. |
| C4 | "$0.10 → $0.008 per action" | Remove entirely from public marketing until a published benchmark report with receipt chain exists. Architectural framing: "Built to make verified action orders of magnitude cheaper than current cloud AI." |
| C5 | "SNR 0.974" | Remove until a published benchmark receipt exists. Architectural framing: "A signal-vs-noise discipline that keeps outputs tied to evidence, not assumption." |
| C6 | "8,072 verified tests" | If backed by `pytest --collect-only` + `cargo test -- --list` counts with a timestamp and commit hash: keep as **"Thousands of verified tests across the sovereign core (see GitHub for latest count)."** This is honest and link-backed. Avoid exact numbers without a timestamped receipt. |
| C7 | "100% pass rate" | Remove. Replace with: **"CI must pass before any merge — the same discipline we apply to our own claims."** (describes policy, not a brittle metric) |
| C8 | "Ihsan Gate >= 0.95" | Keep in technical/investor contexts. In consumer copy: **"An Ihsan discipline — we hold our own outputs to a high conscience threshold before we ship."** |
| C9 | "73 of 100 nodes remaining" | If the 100-node cohort is a real early-access cap: replace with a live counter that pulls from a source of truth (airtable / db / config file), and label it as "Early-access cohort: X / 100 seats." If aspirational: remove entirely. Do not run paid ads with this wording. |

## 6. Overall hero-copy recommendation

Replace the current bizra.ai hero numeric ladder with **claim-discipline-compliant brand wording** drawn from the kit and the brand canon:

**English (recommended hero):**

> **BIZRA**
> **The Seed of Sovereign Intelligence.**
>
> A human-first AI ecosystem built on meaning, proof, and Ihsan.
> Not another chatbot. Not another platform that owns you.
> One human. One node. One sovereign path.
>
> **Build with meaning. Act with proof. Grow with Ihsan.**

**Arabic (recommended hero):**

> **بذرة**
> **بذرة الذكاء السيادي.**
>
> منظومة ذكاء إنساني أولاً، مبنية على المعنى والبرهان والإحسان.
> ليست أداة محادثة أخرى. وليست منصة تملكك.
> إنسان واحد. عقدة واحدة. طريق سيادي واحد.
>
> **ابنِ بالمعنى. اعمل بالبرهان. وانمُ بالإحسان.**

The numeric-claim strip (SNR, test count, pass rate, cost, nodes-remaining) should move to a separate "Under the Hood" / technical-spec page that links to published benchmark receipts — not the hero.

## 7. What this audit does NOT do

- Does NOT edit bizra.ai source code.
- Does NOT change DNS or any deployment.
- Does NOT remove any live claim. That requires a separate, typed-authorized deploy step.
- Does NOT verify the pre-check findings against a fresh DOM. The pre-check is treated as the best available evidence for the SPA-rendered content until a headless-Chromium capture is performed.

## 8. Recommended follow-ons (NOT auto-started)

1. **Headless-Chromium or manual DOM capture of bizra.ai** — to replace pre-check findings with a verified snapshot. Low cost, high value.
2. **Public claim rewrite PR** against whatever repo hosts the bizra.ai source. Not in this repo's scope; requires knowing the hosting repo.
3. **Benchmark receipt publication plan** — if any of C4/C5/C6/C7 are to be retained, they need a public receipt chain. This is a separate lane.
4. **Live early-access counter** if C9 is to be retained. Requires a source of truth + JS wiring.

Each of the above requires explicit typed authorization.
