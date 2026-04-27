# Website Patch Plan — bizra.ai

**Scope:** Planning-only instructions for the engineer who will execute the cleanup in the bizra.ai source repo (separate from this `bizra-data-lake` repo). **Nothing in this document touches the bizra-data-lake repo or the live site.**

**Execution precondition:** operator typed authorization + access to the bizra.ai source repo.

---

## Patches (ordered by priority)

### Patch P1 — Remove REMOVE_NOW hero numbers (CRITICAL)

**Files (approx — depends on bizra.ai source layout):**
- Hero component (probably `src/components/Hero.*` or `app/page.tsx` in Next.js / Vite)
- Any "Under the Hood" / stats strip
- Any metadata / `<meta>` tags with the numbers

**Changes:**

| # | Section | Current claim (per pre-check) | Replacement | Reference |
|---|---|---|---|---|
| P1.1 | Hero numeric strip | `SNR 0.974` | **remove** (no fallback) | `CLAIM_SAFE_REWRITE_PACK.md §C5` |
| P1.2 | Hero numeric strip | `$0.10 → $0.008 per action` | **remove** (directional reframe in separate section) | `§C4` |
| P1.3 | Hero / modal | `73 of 100 nodes remaining` | **remove** OR replace with waitlist CTA | `§C9` |
| P1.4 | Hero metric | `100% pass rate` | replace with policy claim | `§C7` |

**Owner:** web lead.
**Effort:** S (30–60 min).
**Verification:** re-run headless-Chromium capture; confirm no `SNR`, `$0.10`, `$0.008`, `73 / 100`, `100%` strings in rendered DOM.

### Patch P2 — Rewrite REWRITE_REQUIRED claims (HIGH)

| # | Section | Current | Replacement | Reference |
|---|---|---|---|---|
| P2.1 | Hero / any spec callout | `local agents / no cloud dependency` | "Your machine. Your keys. Your node." | `§C1` |
| P2.2 | Any call-out | `100% pass rate` | "CI must pass before merge — the same discipline we apply to our claims." | `§C7` |
| P2.3 | Any "BIZRA is live" banner | `BIZRA is live.` | "The Seed is public." / "بذرة الآن علنيّة." | `§K1` |

**Owner:** operator (copy) + web lead (implementation).
**Effort:** S (30 min).

### Patch P3 — Move INTERNAL_ONLY claims off hero (MEDIUM)

| # | Section | Current | Action | Reference |
|---|---|---|---|---|
| P3.1 | Hero | `Ed25519 receipt signatures` | Move to `/under-the-hood/receipts` sub-page; hero line becomes "Every action leaves a receipt." | `§C3` |

**Owner:** web lead.
**Effort:** M (create sub-page).

### Patch P4 — Contextualize SAFE_WITH_RECEIPT claims (MEDIUM)

| # | Section | Current | Replacement | Reference |
|---|---|---|---|---|
| P4.1 | Any "no telemetry" callout | `no telemetry` | See `§C2` conditional (with / without privacy policy) | `§C2` |
| P4.2 | Any Ihsan-threshold callout | `Ihsan Gate >= 0.95` | "We hold our outputs to a high conscience threshold (Ihsan ≥ 0.95) …" + link | `§C8` |
| P4.3 | Test-count reference | `8,072 verified tests` | "Thousands of verified tests across the sovereign core." + link to CI | `§C6` |

**Owner:** operator (copy) + web lead (implementation).
**Effort:** M.

### Patch P5 — Structural sub-pages (MEDIUM)

Create three sub-pages so the receipt-backed and context-framed claims have honest homes:

| # | New sub-page | Content |
|---|---|---|
| P5.1 | `/privacy` | Privacy policy. What we do / don't collect. Public-key posture. Conditional to §C2 keeping. |
| P5.2 | `/ihsan` | Short explainer: what Ihsan means, 0.95 threshold, link to constants.py. |
| P5.3 | `/under-the-hood/receipts` | Receipt-chain explanation. Ed25519, BLAKE3, genesis seal, sample verifier. Link to `bizra-omega/bizra-core/src/canonical_receipt.rs`. |

**Owner:** operator (copy) + web lead (implementation).
**Effort:** L (multi-section content).

### Patch P6 — SPA shell + OG tags (MEDIUM)

The site is SPA; non-JS fetchers (social link previews, SEO crawlers, audit engines) see only the shell. Add OG meta tags to the **shell HTML** (not JS-rendered) so link previews and crawlers render correctly.

**Changes:**

- Add `<meta property="og:title" content="BIZRA — The Seed of Sovereign Intelligence" />`
- Add `<meta property="og:description" content="A human-first AI ecosystem built on meaning, proof, and Ihsan." />`
- Add `<meta property="og:image" content="https://bizra.ai/og/launch.png" />` (use `docs/brand/public_launch_media_kit_v0_1/extracted/.../assets/ready_to_post/BIZRA_07_website_hero_1920x1080.png` or equivalent)
- Add `<meta name="twitter:card" content="summary_large_image" />`

**Owner:** web lead.
**Effort:** XS (15 min).

### Patch P7 — bizra.info 302 (NO CHANGE)

Confirmed 302 → bizra.ai/. **Do not change.** This is the correct brand-defense posture.

### Patch P8 — DNS (NO CHANGE)

No DNS modifications in this lane.

---

## Execution order

```
P1 (REMOVE_NOW) ──────▶ P2 (REWRITE) ──▶ P3 (INTERNAL_ONLY move) ──▶ P4 (contextualize)
                                                                          │
                                                                          ▼
                                                                 P5 (sub-pages)
                                                                          │
                                                                          ▼
                                                                    P6 (OG tags)
```

**Minimum viable deploy:** P1 + P2 + P6. Removes the highest-risk claims, rewrites the small wording items, adds OG tags. Single PR.

**Full cleanup deploy:** all six patches. Two or three PRs depending on how sub-pages are structured.

## Verification steps

For each deploy:

1. Fresh headless-Chromium DOM capture of `https://bizra.ai/`.
2. Grep the captured DOM for the removed strings (`SNR`, `$0.10`, `$0.008`, `73 / 100`, `100%`, `BIZRA is live`).
3. Re-run `python3 -m tools.audit.omni_audit.run_audit` — verify C4/C5/C7/C9 no longer appear in claims_register.
4. Social link preview check: post a link to `bizra.ai` in a test Slack / LinkedIn / X draft — confirm OG image + description render.
5. Arabic parity: confirm every English change has an Arabic twin matching the class.

## Rollback

Every patch lands in a reviewable PR in the bizra.ai source repo. Rollback = revert PR.

## What this plan does NOT do

- Does NOT edit any file in the bizra.ai source repo from this `bizra-data-lake` workspace.
- Does NOT publish any change.
- Does NOT run git operations here.
- Does NOT commit this plan as "executed."
