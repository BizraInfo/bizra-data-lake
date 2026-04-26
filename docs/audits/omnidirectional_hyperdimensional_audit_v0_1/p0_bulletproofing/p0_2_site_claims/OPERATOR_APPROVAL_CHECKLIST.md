# Operator Approval Checklist — P0.2 Site Cleanup

**Purpose:** Typed sign-off sheet. Nothing ships publicly until every item here is checked **and** dated by the operator.

**Date opened:** 2026-04-24 (GST)
**Date closed:** _(pending)_
**Operator of record:** _(your name here)_

---

## Pre-approval — read and acknowledge

- [ ] I have read `CURRENT_PUBLIC_CLAIMS_REGISTER.md`.
- [ ] I have read `CLAIM_SAFE_REWRITE_PACK.md` (EN + AR).
- [ ] I have read `WEBSITE_PATCH_PLAN.md`.
- [ ] I have read `RECEIPTIFICATION_REQUIREMENTS.md`.
- [ ] I have read `PAID_ADS_GO_NO_GO.md`.
- [ ] I accept that bizra.ai claim cleanup happens in the **bizra.ai source repo** (not this `bizra-data-lake` repo).
- [ ] I accept that a **fresh headless-Chromium DOM capture** is the verification step for every deploy.

## REMOVE_NOW approvals (3)

- [ ] **C4** — approve removal of "cost per action $0.10 → $0.008" from bizra.ai hero. Directional replacement per `CLAIM_SAFE_REWRITE_PACK.md §C4` acceptable. _(Date: _______)_
- [ ] **C5** — approve removal of "SNR 0.974" from bizra.ai. Directional replacement per `§C5`. _(Date: _______)_
- [ ] **C9** — approve removal (or live-counter wiring) of "73 of 100 nodes remaining." Choose:
  - [ ] Option A: remove entirely; use waitlist framing.
  - [ ] Option B: commit to live-counter implementation (requires source-of-truth + API endpoint + daily-update cadence).
  _(Date: _______, option: ____)_

## REWRITE_REQUIRED approvals (3)

- [ ] **C1** — approve wording change from "local agents / no cloud dependency" to "Your machine. Your keys. Your node." (EN + AR). _(Date: _______)_
- [ ] **C7** — approve wording change from "100% pass rate" to policy-claim "CI must pass before merge — the same discipline we apply to our claims." (EN + AR). _(Date: _______)_
- [ ] **K1** — approve wording change from "BIZRA is live." to "The Seed is public." / "بذرة الآن علنيّة." (kit's own launch copy + any live-site surface). _(Date: _______)_

## INTERNAL_ONLY approvals (1)

- [ ] **C3** — approve moving "Ed25519 receipt signatures" **off** the consumer hero; hero line becomes "Every action leaves a receipt." with link to `/under-the-hood/receipts` sub-page. _(Date: _______)_

## SAFE_WITH_RECEIPT approvals (5)

- [ ] **C2** — choose:
  - [ ] Publish privacy policy and keep "no telemetry" wording.
  - [ ] Soften to "Your actions stay on your node unless you choose to share." (default until policy lands).
  _(Date: _______, option: ____)_
- [ ] **C6** — choose:
  - [ ] Publish timestamped CI receipt and keep exact number (with link).
  - [ ] Soften to "Thousands of verified tests…" with link to CI (always-safe).
  _(Date: _______, option: ____)_
- [ ] **C8** — approve contextualized wording for "Ihsan Gate ≥ 0.95" with `/ihsan` sub-page link. _(Date: _______)_
- [ ] `/privacy` sub-page content approved (if C2 option 1). _(Date: _______)_
- [ ] `/ihsan` sub-page content approved. _(Date: _______)_

## Structural / technical approvals

- [ ] `/under-the-hood/receipts` sub-page plan approved. _(Date: _______)_
- [ ] OG meta tag set approved (image selection from media kit). _(Date: _______)_
- [ ] bizra.info 302 redirect confirmed unchanged. _(Date: _______)_
- [ ] DNS: no changes authorized. _(Date: _______)_
- [ ] Arabic reviewer has approved all Arabic copy. _(Reviewer name: __________, Date: _______)_

## Verification method

- [ ] Post-deploy: fresh headless-Chromium DOM capture of `https://bizra.ai/` performed.
- [ ] Rendered DOM grep: no strings `"SNR 0.974"`, `"$0.10"`, `"$0.008"`, `"73 / 100"`, `"100%"`, `"BIZRA is live"`, `"no cloud dependency"`.
- [ ] Re-run `python3 -m tools.audit.omni_audit.run_audit` and confirm `claims_register.json` no longer classifies C4/C5/C7/C9 as NEEDS_REWRITE for the live site.
- [ ] Social link-preview check: test post in LinkedIn / X / Slack draft shows hero image + correct description.

## Paid-ads decision

- [ ] All REMOVE_NOW + REWRITE_REQUIRED approvals above are dated.
- [ ] All sub-pages (`/privacy` or softening variant, `/ihsan`, `/under-the-hood/receipts`) are live.
- [ ] `ADS_READINESS_CHECKLIST.md` Gate 1-6 all green.
- [ ] Secondary co-operator designated for kill-switch (recommended). _(Name: __________)_
- [ ] **I authorize paid-ad activation.** _(Date: _______, Signature: __________)_

If ANY box above is unchecked → paid ads remain NO-GO.

## Final stop line

- [ ] No publishing occurred while this checklist was being filled out.
- [ ] No git operations (add/commit/push/branch/tag) occurred in `bizra-data-lake` as part of this approval flow.
- [ ] No runtime / canon pack / MEMORY.md / Node0 behavior was modified.
- [ ] All approvals recorded here feed only the **bizra.ai source repo** cleanup lane.

---

## Notes / exceptions section

_(Add any operator-specific exceptions, deferred items, or additional approvals below.)_

```
[  ] Additional item: ____________________________________________
     Reason:           ____________________________________________
     Date:             _______________
```
