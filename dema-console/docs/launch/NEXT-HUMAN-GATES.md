# Cycle-8 Next Human Gates (post-Phase-2/3)

بسم الله الرحمن الرحيم

**Status:** DRAFT (Cycle-8, 2026-04-19). Internal, not public.
**Purpose:** single place that enumerates the exact human inputs blocking further autonomous progress, in priority order.

---

## 1. Witness peer (Phase 5 halt)

**Blocks:** real external witness binding; T=0 gate #4 (economic modality witness-grade closure).

**What's needed from Mumo:**
- One external person willing to run the `witness` daemon on a reachable host.
- That person's public gateway URL (e.g., `https://friend.example.com:7421`).
- Their BIZRA node_id (they generate one on first boot).
- One WhatsApp / email ask. No contract, no commitment beyond "please don't turn it off for 30 days post-fire".

**Minimum quorum:** 1 witness peer for T=0. 3 is the documented dry-run threshold.

**Current state:** TBD. The witness client + daemon code are shipped on `cycle-8/seal-primitive-days-1-2` branch (commits `b17f47ae` + `d262a317`), Ed25519-signed, 77/77 gateway tests green. Deployment awaits peer identity.

## 2. Five dry-run tester names (Phase 4 halt)

**Blocks:** Day 11-12 dry-run execution; T=0 gate #5 (5 testers complete end-to-end).

**What's needed from Mumo:**
- 5 names (friends, colleagues, family members with laptops). Non-technical acceptable.
- Their platform (Linux / macOS / Windows 11).
- A contact channel (WhatsApp, email).
- 45 minutes of their time each.

**Currently TBD × 5.** Harness + checklist exist at `scripts/fire-dry-run-harness.sh` + `scripts/fire-dry-run-checklist.md` on `cycle-8` branch. Ready to run once names arrive.

## 3. Push approval for PR #28 branch updates

**Blocks:** remote visibility of the 5 new commits on `fork/dema-console-from-zai`.

**Current PR #28 branch local state (not pushed):**
- `030e736f` fix(dema-console): remove DEMO_* shadow state from launch-path surfaces
- `4ebbb422` fix(dema-console): TrustStrip renders honest inactive state when no principal
- `2002dd84` fix(dema-console): OrganizePreview + MemoryConstellation — honest empty states (Phase 2 complete)
- `(pending)` docs(launch): consumer landing copy + enterprise brief held + gates doc (Phase 3)

**Remote HEAD of PR #28:** `4c67710a` (unchanged).

**To push:** `git push origin fork/dema-console-from-zai` (normal push, not force). Requires explicit "approved" from Mumo per CLAUDE.md.

## 4. D5 Daughter Test on the 7/7 WIRED_REAL surfaces

**Blocks:** T=0 gate #7 (visual Daughter Test pass on every visible surface).

**What's needed:**
- Mumo (or a proxy) runs `bun install && bun run dev -p 3005` locally on the PR #28 branch.
- Walks through each of the 7 surfaces: TrustStrip, MissionComposer, GateLadder, OrganizePreview, ReceiptReveal, MemoryConstellation, RejectRemediation.
- For each: confirms the empty state is honest, the active state shows real kernel data, no fabrication anywhere.
- Records any UX defect as a screenshot + 1-line description.

**OR:** write a headless Playwright/Puppeteer harness (~300 LOC, separate commit) that takes screenshots of each surface in each state. That's a Day 13-14 task, out of current Phase 2/3 scope.

## 5. Orphan-screen tsc error triage decision

**Blocks:** any future merge of PR #28 to main (CI typecheck will fail).

**Errors (pre-existing, NOT caused by Phase 2/3 work):**
- `src/app/api/operations/route.ts` (metrics.cpu/memory undefined)
- `src/components/dema/screens/autopilot.tsx` (number type mismatch)
- `src/components/dema/screens/onboarding.tsx` (unimported icons)
- `src/components/dema/screens/operations.tsx` (systemHealth / performanceSnapshots undefined)
- `src/lib/api/client.ts` (Record<string,string> widening)

**Three options for Mumo:**
- **(a) Fix them** (estimated 2-3 hours; most are small nullability patches) — clears the path to merge.
- **(b) Exclude the orphan screens from tsc** via `tsconfig.json` `"exclude": ["src/components/dema/screens/*.tsx", ...]` — tight but honest if the screens are truly dormant.
- **(c) Delete the orphan screens entirely** — they're from the Z.ai import and not wired into main `page.tsx`. Biggest commit but cleanest outcome.

**Recommendation:** (c) deletion — the screens are Z.ai-imported dead weight. Their deletion is a hygiene commit that reduces surface area and signals scope discipline.

## 6. U1/U2/U3 already collapsed (no further action)

**Locked:** U1=consumer · U2=no · U3=only-with-help

Positioning is resolved. Consumer landing copy and held enterprise brief reflect this.

## 7. Future — Post-T=0 (Horizon)

These are NOT human gates for T=0. Listed for completeness:
- LLM probabilistic-CPU wiring (HANDOVER §10)
- HAL formalization (v0.4 roadmap)
- Bonded stake / slashing / DAO / challenge-period economics (Layer B)
- Desktop overlay / Cognitive IDE (per ArbiterOS §8.8)
- 150 → 5 GitHub repo consolidation
- 500 GB R&D cleanup (Bayyinah monument, not product)

---

## Priority order (SNR-ranked)

1. Push approval for PR #28 (gate #3) — unblocks external visibility immediately.
2. Witness peer name (gate #1) — unblocks the 4th modality.
3. 5 dry-run tester names (gate #2) — unblocks T=0 pre-fire validation.
4. D5 Daughter Test (gate #4) — requires working dev environment; can run in parallel with others.
5. Orphan-screen triage (gate #5) — blocks merge but NOT the push. Can wait.

**First three answers would unlock the remaining technical path to T=0.**

---

*Close it. Prove it. Reveal it.*

الحمد لله
