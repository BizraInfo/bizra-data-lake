# Paid Ads GO / NO-GO Decision Record

**Date:** 2026-04-24 (GST)
**Scope:** All paid-ad platforms (Meta, X, LinkedIn, YouTube, Google, TikTok, programmatic).

---

## Decision

**Paid ads: NO-GO.**

---

## Blocking claims (must clear before paid spend)


| #   | Claim                                | Class            | Platform risk                                                                  |
| --- | ------------------------------------ | ---------------- | ------------------------------------------------------------------------------ |
| C4  | `cost per action $0.10 → $0.008`     | REMOVE_NOW       | Meta / Google "unsupported quantifiable claim" — ad rejection + account review |
| C5  | `SNR 0.974`                          | REMOVE_NOW       | Same category as C4                                                            |
| C7  | `100% pass rate`                     | REWRITE_REQUIRED | Meta / Google "misleading claim" — brittle, likely rejected                    |
| C9  | `73 of 100 nodes remaining`          | REMOVE_NOW       | Meta / Google "deceptive practices" if counter is not live                     |
| C1  | `local agents / no cloud dependency` | REWRITE_REQUIRED | May pass ad review but sets false expectation for end user                     |


**Any paid ad running today — even if the ad copy itself is claim-safe — lands the user on a page that carries the above. The landing page is part of the ad's claim surface.** Platform reviewers inspect landing pages. Even if they miss it, a skeptical user / competitor / journalist / regulator / ad-review pass can trigger account review.

## Soft blockers (should clear)


| #   | Claim                        | Class                 | Risk                                                               |
| --- | ---------------------------- | --------------------- | ------------------------------------------------------------------ |
| C2  | `no telemetry`               | SAFE_WITH_RECEIPT     | Category misrepresentation risk if privacy policy not published    |
| C3  | `Ed25519 receipt signatures` | INTERNAL_ONLY on hero | Audience mismatch (dev jargon in consumer hero) reduces conversion |
| C6  | `8 072 verified tests`       | SAFE_WITH_RECEIPT     | Brittle if not timestamped                                         |
| C8  | `Ihsan Gate >= 0.95`         | SAFE_WITH_RECEIPT     | Out-of-context number; needs `/ihsan` page                         |


## Path to GO

Paid ads flip to GO when **all** of the following are true:

### Gate 1 — Landing-page hygiene

- C4, C5, C7, C9 are no longer present in the rendered DOM of `https://bizra.ai/` (verify via fresh headless-Chromium capture).
- C1 wording matches `CLAIM_SAFE_REWRITE_PACK.md §C1`.
- OG meta tags present on SPA shell (per `WEBSITE_PATCH_PLAN.md P6`).
- bizra.info → bizra.ai 302 still in place (no change needed).

### Gate 2 — Receipt / sub-page coverage

- `/under-the-hood/receipts` sub-page published (supports §C3 softening).
- `/privacy` published OR §C2 softened per `§C2` default variant.
- `/ihsan` published (supports §C8 contextualization).

### Gate 3 — Ad-copy discipline

- All ad creatives use only copy from `CLAIM_SAFE_LAUNCH_COPY.md §6` or the rewrite pack.
- No exact $ / SNR / test-count / "100% pass" / "N / 100" wording in any creative.
- No PROHIBITED patterns (AGI, first-in-world, financial returns, unsub. cert., benchmark superiority).
- Arabic / English parity maintained.

### Gate 4 — Media-kit visual QA

- Visual QA of 12 rendered-concept boards + 11 ready-to-post rasters (small-text / Arabic ligature / color-token compliance).
- Every creative used in an ad is either from `assets/editable_svg/` (known text) or has been human-verified against approved copy.

### Gate 5 — Platform account readiness

- Ad account(s) set up with 2FA, billing, daily spend cap.
- Kill-switch path documented (who can pause all ads within 10 min).
- UTM conventions defined per platform.
- Post-launch monitoring plan (24h / 72h / 1wk).

### Gate 6 — Operator sign-off

- Operator typed sign-off on `OPERATOR_APPROVAL_CHECKLIST.md`.

**All 6 gates must be green. Partial greens = NO-GO.**

## Not-needed-for-paid-ads (explicit)

- Canon Store Ingestion Gate — separate lane, not a paid-ad prerequisite.
- Genesis-100 activation plan — separate lane.
- Receipt-publication for C4/C5/C6 — nice-to-have for richer ad copy, not required (directional reframes are sufficient).
- Secondary co-operator designation — recommended for safety but not blocking.

## Why organic launch is not blocked

Organic posts land on the bizra.ai site too. But:

- Organic does not trigger platform ad-policy review.
- Organic is lower-volume; skeptical readers self-select.
- Organic can rehearse messaging and surface claim-drift before paid spend.

**Recommended sequence:** organic launch (Phase 1 silent → Phase 2 moment → Phase 3 first-week) *can* proceed in parallel with Gate 1-4 cleanup, **provided** the site cleanup lands before any paid ad goes live. If site cleanup stalls, pause at Phase 2 until Gate 1 is green.

## Decision authority

This NO-GO record stands until:

1. An operator with ad-account billing authority completes `OPERATOR_APPROVAL_CHECKLIST.md`.
2. Gate 1-6 are all green.
3. The ad platform(s) have been informed (if required by their policies — e.g., LinkedIn may require a substantiation declaration).

Revisit weekly or on material change in site copy.

## Stop line

No paid spend today. No paid-ad account activation today. No platform-policy outreach today. **Plan only.**