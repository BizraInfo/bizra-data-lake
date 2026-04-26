# Public Claims Register — BIZRA Launch v0.1

**Register date:** 2026-04-24 (GST)
**Authority:** BIZRA Brand Canon v0.2 §5 (Law of Assumption) + §15 (Claim Discipline)
**Scope:** Every public-facing claim known to be on `bizra.ai`, inside the launch media kit, or proposed for social/ad/press use.

---

## Classification codes

| Code | Meaning |
|---|---|
| **A** | Brand-safe. Identity / mission / philosophy. No numeric promise. Use freely. |
| **B** | Defensible but needs a receipt / proof chain before public quantitative reuse. Use with citation. |
| **C** | Rewrite required. Over-promises, brittle, or regulator-unsafe. Don't use as-is. |
| **D** | Internal / private deck only. Never in public ads or public site. |

## Top 3 risks (triage first)

1. **C9 — "73 of 100 nodes remaining"** (nodes-remaining scarcity hook on bizra.ai). Highest ad-liability risk. Either wire a live counter or remove.
2. **C4 — "$0.10 → $0.008 per action"** (cost figure). Second-highest liability. No published methodology visible; precise $ in ads = regulator flag.
3. **C7 — "100% pass rate"** (test pass claim). Brittle; one future failure falsifies it. Replace with policy language.

---

## Register (claim-by-claim)

### Identity & Mission claims (all A — brand-safe)

| ID | Claim | Source | Class | Notes |
|---|---|---|---|---|
| I1 | "BIZRA / بذرة" — the name + Arabic root | Brand canon §1, visual tokens | **A** | Free to use. |
| I2 | "The Seed of Sovereign Intelligence" | Brand canon §9, kit README | **A** | Primary English tagline. Free to use. |
| I3 | "بذرة الذكاء السيادي" | Brand canon §9, kit README | **A** | Primary Arabic tagline. Free to use. |
| I4 | "Build with meaning. Act with proof. Grow with Ihsan." | Brand canon §16, kit README | **A** | Primary motto EN. |
| I5 | "ابنِ بالمعنى. اعمل بالبرهان. وانمُ بالإحسان." | Kit visual tokens | **A** | Primary motto AR. |
| I6 | "Every human is a node. Every node is a seed." | Brand canon §9 secondary | **A** | Movement line. |
| I7 | "From intention to verified action." | Brand canon §9 secondary | **A** | Product framing. |
| I8 | "Not another chatbot. Not another platform that owns you." | Kit launch copy | **A** | Differentiator framing. Note the softened "owns you" wording — keep. |
| I9 | "One human. One node. One sovereign operating system." | Kit launch copy | **A** | Brand-safe. |
| I10 | "A human-first AI ecosystem" | Brand canon §8 | **A** | Primary category framing. |
| I11 | "A sovereign agentic intelligence ecosystem" | Brand canon §7 | **A** | Strategic category. |
| I12 | "Mission-centric sovereign intelligence" | Brand canon §7 | **A** | Positioning. |

### Kit-generated copy (mostly A, one item C)

| ID | Claim | Source | Class | Notes |
|---|---|---|---|---|
| K1 | "BIZRA is live." | kit `BIZRA_LAUNCH_COPY.md` | **C → rewrite** | "Live" implies readiness; soften to **"The Seed is public."** See QA_NOTES §7. |
| K2 | "The Seed of Sovereign Intelligence." | kit launch copy | **A** | Duplicate of I2. |
| K3 | "بذرة هي بذرة الذكاء السيادي." | kit launch copy | **A** | AR. Free. |
| K4 | "كل إنسان عقدة. وكل عقدة بذرة." | kit launch copy | **A** | AR. Free. |
| K5 | "bizra.ai" | kit launch copy | **A** | Domain. Free. |

### Current bizra.ai claims (from operator-supplied pre-check)

| ID | Claim | Class | Recommended action |
|---|---|---|---|
| C1 | "local agents / no cloud dependency" | **B** | Keep with framing: **"Designed to run on your machine — your keys, your data, your node."** Avoid absolute "no cloud" until a public privacy policy + architecture page backs it. |
| C2 | "no telemetry" | **B** | Keep IF a public privacy statement + runtime receipt shows it. Else soften: **"Your actions don't leave your node unless you choose to share them."** |
| C3 | "Ed25519 receipt signatures" | **B** | Accurate for runtime (receipt signing lives in `bizra-mission`). Keep in dev/investor docs. Remove from consumer hero copy — audience mismatch. |
| C4 | "cost per action from about $0.10 toward $0.008" | **C → rewrite or remove** | Precise $ without public methodology = ad policy liability. Replace with direction: **"Designed to make verified action radically cheaper than cloud AI."** Do NOT cite the $ figures in paid ads. |
| C5 | "SNR 0.974" | **C → rewrite or remove** | Exact benchmark number without a public benchmark receipt. Replace with: **"A signal-vs-noise discipline that keeps outputs tied to evidence."** |
| C6 | "8,072 verified tests" | **B** | Defensible IF backed by a timestamped CI receipt + commit hash link. Safer framing for marketing: **"Thousands of verified tests across the sovereign core."** Include a link to CI. |
| C7 | "100% pass rate" | **C → rewrite** | Brittle, falsifiable, and compliance-adjacent. Replace with: **"CI must pass before merge — the same discipline we apply to our claims."** |
| C8 | "Ihsan Gate >= 0.95" | **B** | Accurate to `constants.py`. Keep in investor / technical contexts. In consumer copy: **"We hold our outputs to a high conscience threshold (Ihsan ≥ 0.95) before we ship."** — include "we" framing so it's a policy claim, not a user promise. |
| C9 | "73 of 100 nodes remaining" | **C → rewrite or remove** | Manufactured scarcity without a live counter = liability. Either wire to a live source of truth, or remove. Do NOT run paid ads with this. |

### Brand canon §15 "Avoid until verified" — DO NOT PUBLISH

| ID | Category | Class | Source |
|---|---|---|---|
| D1 | Exact latency claims | **D** | Canon §15 |
| D2 | Security hardening certifications (SOC2, ISO, etc.) | **D** | Canon §15 — none obtained |
| D3 | Cryptographic finality claims (finality, tamper-proof, unbreakable) | **D** | Canon §15 |
| D4 | AGI claims | **D** | Canon §15 |
| D5 | Financial return claims (investor returns, savings $, ROI) | **D** | Canon §15 |
| D6 | "First in the world" claims unless formally substantiated | **D** | Canon §15 |
| D7 | Benchmark-superiority claims (beats GPT-X, outperforms Claude, etc.) | **D** | Canon §15 |
| D8 | Production-readiness implicit in "live" / "ready" / "production-grade" | **D** | Derived from canon §15 conservatism |

### Internal-only / private-deck claims

| ID | Claim | Class | Reason |
|---|---|---|---|
| P1 | Internal receipt-lineage progress, Node0 scoreboard state, PR counts | **D** | Operational, not brand-safe |
| P2 | Specific Cognitive Foundry canon-pack hashes, entry counts, review stats | **D** | Internal disposition; not a public claim |
| P3 | Constitutional threshold numeric values from `constants.py` (other than Ihsan 0.95 in context) | **D** | Internal; expose via architecture page only |
| P4 | Specific test names, failing test recovery stories, incident timelines | **D** | Investor-deck material only |

---

## Cross-cutting rules (inherited from Brand Canon §5 + §15)

1. **No naked numbers in paid ads.** Every quantitative claim in a paid placement must link to a public receipt/methodology. If it doesn't, remove the number.
2. **No financial promises.** Ever. In any public surface. (This is a legal floor, not just brand discipline.)
3. **No production-readiness implication beyond evidence.** "Live" / "ready" / "production-grade" require a public deploy + stability statement.
4. **No "first / best / only" claims** without formal substantiation.
5. **Receipt-or-direction.** Every metric is either cited with a receipt OR phrased as a direction / ambition / architectural goal.
6. **English-Arabic parity.** Arabic translations must match English claim class. If the English is C, the Arabic is C.

## Applicability matrix

| Surface | A | B (with cite) | C | D |
|---|---|---|---|---|
| Organic social (X, LinkedIn, IG, YouTube) | ✅ | ✅ with link | ❌ | ❌ |
| Paid ads (Meta, X, LinkedIn, YouTube, Google) | ✅ | ⚠️ only with published receipt link | ❌ | ❌ |
| bizra.ai hero | ✅ | ⚠️ prefer move to sub-page | ❌ remove | ❌ |
| bizra.ai sub-pages (architecture, technical) | ✅ | ✅ with receipt | ⚠️ rewrite | ❌ |
| Investor deck | ✅ | ✅ | ✅ with caveat | ✅ with caveat |
| Press release | ✅ | ✅ with receipt | ❌ | ❌ |

## Open items requiring decisions

1. **Publish benchmark receipts?** If yes → C5/C6/C7 can move to B once the receipt exists.
2. **Architecture / privacy policy page?** If yes → C1/C2/C3/C8 become B-with-citation.
3. **Live early-access counter for C9?** Or remove the 100-node cohort framing?
4. **Remove or replace C4 cost figures from live site?** (Recommended: remove until receipt.)
