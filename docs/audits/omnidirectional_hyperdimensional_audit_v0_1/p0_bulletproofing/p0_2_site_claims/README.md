# P0.2 — bizra.ai / bizra.info Public-Claim Cleanup

**Purpose:** Surgical, planning-only workspace for bringing the live bizra.ai claim surface into line with the internal claim-discipline discipline (Brand Canon §5 Law of Assumption + §15 Claim Discipline).

**Date:** 2026-04-24 (GST)
**Lane status:** **planning only.** No website source modified. No DNS changes. No publication.

---

## What this directory contains

| File | Purpose |
|---|---|
| `CURRENT_PUBLIC_CLAIMS_REGISTER.md` | Consolidated register of every known public claim on bizra.ai / bizra.info, with classification |
| `CLAIM_RISK_MATRIX.csv` | Spreadsheet of claims × class × severity × action × owner |
| `CLAIM_SAFE_REWRITE_PACK.md` | Drop-in replacement copy (EN + AR) for every claim that must change |
| `WEBSITE_PATCH_PLAN.md` | Section-by-section patch instructions for the bizra.ai source repo |
| `RECEIPTIFICATION_REQUIREMENTS.md` | What evidence would let a `SAFE_WITH_RECEIPT` claim return to live copy |
| `PAID_ADS_GO_NO_GO.md` | Decision record: paid ads remain NO-GO until this P0.2 lane closes |
| `OPERATOR_APPROVAL_CHECKLIST.md` | Typed sign-off sheet the operator completes before anything ships |

## Rules carried in

- **Do NOT edit bizra.ai source** (lives in a separate repo not touched here).
- **Do NOT change DNS or redirects** (bizra.info 302 → bizra.ai is correct and keeps).
- **Do NOT publish** anything from this lane.
- **Do NOT run git operations.**
- **Do NOT modify runtime, canon packs, MEMORY.md**, Node0 behavior, PR #49/#50, or the Cognitive Foundry review cycle.
- **Do NOT start the Canon Store Ingestion Gate.**

## Top-line finding (preview)

| Classification | Count |
|---|---:|
| SAFE_NOW | 8 |
| SAFE_WITH_RECEIPT | 5 |
| REWRITE_REQUIRED | 3 |
| INTERNAL_ONLY | 2 |
| REMOVE_NOW | 3 |
| PROHIBITED | 0 live today (watchlist) |
| **Total reviewed** | **21** |

- **3 REMOVE_NOW claims** (C4/C5/C9) = the highest liability.
- **3 REWRITE_REQUIRED claims** (C1, C7, "BIZRA is live") = fastest wins.
- **5 SAFE_WITH_RECEIPT claims** = can return once receipt chain is published.
- **Paid ads: NO-GO** until REMOVE_NOW + REWRITE_REQUIRED categories clear.

## Scope / verification

Claim evidence is the operator-supplied ChatGPT pre-check mirrored in `../../artifacts/website_claims.json` + `website_snapshot.txt`. bizra.ai is a client-side-rendered SPA; plain HTTP fetch sees only a shell (title `BIZRA | The Sovereign Future` + `Clear local data` button). A fresh headless-Chromium DOM capture is recommended before executing any patch plan item, to confirm the live content has not drifted since the pre-check.
