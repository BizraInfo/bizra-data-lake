# Frontend Patch Doctrine — Constitutional UI Sprint

**Status:** [ENFORCEMENT: WIRED]
**Method:** Surgical patch over existing candidate, NOT new build
**Source:** Read-only audit of routed app + prototype donors (2026-03-29)

## The Target State

One integrated BIZRA experience with three surfaces:

1. **Public Trust Surface** — `frontend/src/phases/TrustSite.tsx`
   Truth-labeled metrics, canonical BIZRA language, investor-grade claims

2. **Identity & Activation Flow** — `App.tsx`, `Genesis.tsx`, `TeachSteps.tsx`
   Node0 onboarding with constitutional context

3. **Node0 Operator Mission OS** — Terminal shell, OperatorCockpit, mission/timeline views
   Live proof rail, receipt explorer, Morning Brief, Ghost Panel

## Standing on Giants

- **Shannon:** Maximize signal, remove decorative ambiguity
- **Tufte/Few:** Dense information, low chartjunk, clear comparative state
- **Lamport/Nakamoto:** Ordering and proof chain clarity
- **Deming:** Prove the process, not just the artifact
- **Al-Ghazali:** Honest labeling and admissible claims
- **Norman/Engelbart:** Augmentation over spectacle

---

## Elite Patch Order (7 steps, dependency-ordered)

### Patch 1: Truth Layer First

**What:** Add visible `PROVEN` / `VALIDATED` / `WIRED` / `PLANNED` badges to every claim-bearing metric and card.

**Why:** The routed app has no truth-governance layer. Claims appear equally proven regardless of evidence status.

**Files:**
- All metric cards across the app
- Create a `<TruthBadge status="PROVEN|VALIDATED|WIRED|PLANNED" />` component

**Gate:** Every number, stat, or capability claim has a visible truth label.

---

### Patch 2: Kill Silent False Success

**What:** `useMission.ts:66` must stop converting API failure into simulated constitutional success, minted rewards, and "signed and chained" language.

**File:** `frontend/src/hooks/useMission.ts:70`

**Current:** On API failure, the hook fabricates a complete success response with:
- Fake constitutional receipt
- Fake SEED rewards
- "Signed and chained" language

**Target:** Split live receipt from demo preview. On failure:
- Show explicit failure state with reject reason
- Label any demo content as `[DEMO]`
- Never mint real-looking state on failure

**Gate:** One mission produces one clearly classified proof outcome. No ambiguity.

---

### Patch 3: Promote Prototype Differentiators

**What:** Ghost Panel and Trust Rail exist only in prototypes. Move them to the routed app.

**Donor files:**
- `frontend/src/prototypes/BIZRA_SovereignCockpit.jsx:556` → Ghost Panel
- `frontend/src/prototypes/BIZRA_FrontDoor.jsx:433` → Trust Rail

**Target files:**
- `frontend/src/components/OperatorCockpit.impl.tsx` → receives Trust Rail
- New component or integration point → receives Ghost Panel

**Gate:** Morning Brief, Ghost Panel, Trust Rail, cockpit, receipt explorer all exist in the routed app.

---

### Patch 4: Replace Demo Masquerade

**What:** Components render hard-coded demo objects while importing live hooks. Make the source explicit.

**Files:**
- `terminal-memory.tsx:35` → Morning Brief card
- `terminal-network.tsx:28` → Forest/network metrics
- `terminal-settings.tsx:26` → Identity/model defaults

**Current:** Demo data rendered without any indicator of source.

**Target:** Every data card shows its source state:
- `LIVE` — connected to real API/service
- `CACHED` — last-known state from prior session
- `DEMO` — hard-coded preview data

**Gate:** Zero silent fallbacks. Every card declares its data source.

---

### Patch 5: Normalize Canonical Vocabulary

**What:** The app mixes "DDAGI OS", older framing, and contradictory proof primitives.

**Files:**
- `TrustSite.tsx:61` — legacy phrasing
- `terminal-timeline.tsx:168` — inconsistent proof terminology
- `terminal-settings.tsx:330` — contradictory proof primitives

**Target vocabulary (from BIZRA canon):**
- Mission (not "task" or "request")
- Receipt (not "proof" or "attestation" alone)
- Gate Chain (not "verification pipeline")
- Ihsan (not "quality score")
- SEED/BLOOM (not "tokens" or "credits")
- Sovereign (not "local" or "self-hosted")

**Gate:** Proof vocabulary is canonical and internally consistent across all surfaces.

---

### Patch 6: Make URP Visible

**What:** The routed app exposes SEED/BLOOM well but not the shared URP (Universal Resource Pool) semantics.

**Target:** Add a dedicated economic card showing:
- SEED utility flow (earned → spent → marketplace)
- BLOOM governance weight (soulbound, non-transferable)
- URP topology (how nodes share resources)
- Gini constraint visualization (≤ 0.35)

**Gate:** URP semantics are visible without topology violation.

---

### Patch 7: Close Responsive Proof

**What:** The investor demo requires verified behavior at mobile, tablet, and desktop breakpoints.

**Checks:**
- No overflow on mobile (375px)
- Proof rail visible at all sizes
- Trust Rail doesn't collapse critical information
- Ghost Panel degrades gracefully on small screens

**Gate:** Desktop and mobile both preserve the proof rail and operator comprehension.

---

## File-Level Doctrine (Exact Patch Targets)

| File | Line | Action |
|------|------|--------|
| `phases/TrustSite.tsx:27` | Convert hard-coded stats to truth-labeled canonical claims |
| `hooks/useMission.ts:70` | Split live receipt from demo preview; never mint on failure |
| `terminal/terminal-memory.tsx:115` | Make Morning Brief source-aware (LIVE/CACHED/DEMO) |
| `terminal/terminal-network.tsx:261` | Replace fabricated forest metrics with canonical or PLANNED |
| `terminal/terminal-settings.tsx:285` | Bind to real/cached state, not static demo objects |
| `OperatorCockpit.impl.tsx:84` | Graft Trust Rail from prototype donor |
| `prototypes/BIZRA_SovereignCockpit.jsx:634` | Source donor for Ghost Panel + Trust Rail |
| `prototypes/BIZRA_FrontDoor.jsx:433` | Source donor for trust-rail framing + operator tone |

## Acceptance Gate

The implementation is elite ONLY if ALL of these are true:

- [ ] Zero silent fallbacks
- [ ] Every metric or claim visibly truth-labeled
- [ ] One mission → one clearly classified proof outcome
- [ ] Morning Brief, Ghost Panel, Trust Rail, cockpit, receipt explorer, URP semantics ALL in routed app
- [ ] Proof vocabulary is canonical and internally consistent
- [ ] Desktop and mobile both preserve the proof rail
- [ ] No claim appears stronger than the evidence behind it
