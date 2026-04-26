# Documentation Audit — BIZRA v0.1

**Scope:** handoff docs, README quality, ADR gaps, runbook gaps, diagrams, Definition of Done, public/private claim split.

---

## 1. Handoff docs present

| Path | Status | Notes |
|---|---|---|
| `CLAUDE.md` | ✅ | Top-level agent contract; comprehensive. |
| `MEMORY.md` | ✅ | Memory index; ≤200 lines by discipline. |
| `tools/cognitive_foundry/claude_lane/REVIEW_HANDOFF.md` | ✅ | Updated this session to full-review final state. |
| `tools/cognitive_foundry/claude_lane/canon_packs/README.md` | ✅ | 5-pack disposition recorded. |
| `docs/brand/public_launch_media_kit_v0_1/HANDOFF_NOTES.md` | ✅ | Created this session. |
| `docs/brand/public_launch_readiness/NEXT_IMPLEMENTATION_PLAN.md` | ✅ | Created this session. |

## 2. ADR gaps

**`docs/adr/` exists** and contains architecture decision records. Not enumerated here — a dedicated ADR-gap audit should:

- List all ADRs in `docs/adr/`.
- Cross-reference against architectural claims in `ARCHITECTURE_AUDIT.md`.
- Flag claims that lack a backing ADR (e.g., Canon Store Ingestion Gate spec should exist as ADR before any code).

**Gap candidates:**
- Canon Store Ingestion Gate design — no ADR yet (by design; pending typed-auth).
- PAT/SAT gateway wiring status — tracked in memory, not in ADR.
- "No telemetry" as product claim — should have an ADR with enforcement mechanism.

## 3. Runbook gaps

Claude context (CLAUDE.md) covers common commands. Beyond that:

| Runbook | Status |
|---|---|
| **Security incident / key-rotation runbook** | MISSING |
| **Release runbook (PR merge → deploy)** | PARTIAL — GitHub Actions defined in `.github/workflows/ci.yml` but operator-side steps not documented |
| **Backup / recovery runbook** | MISSING |
| **Node-onboarding runbook** (how a new human creates Node0) | MISSING — this is the Genesis-100 path |
| **Audit engine re-run runbook** | ✅ just created (see `tools/audit/omni_audit/README.md`) |

## 4. Diagrams

No formal architecture diagrams found under `docs/architecture/` in this scan. Runtime architecture is currently expressed through code layout + CLAUDE.md's module tables.

**Recommendation:** generate a single canonical runtime-architecture diagram (SVG) covering:
- Node0 binary + substrate abstraction layer.
- Receipt chain + mission-state machine.
- FATE gates + Ihsan threshold.
- Dema face + PAT/SAT internals (internal-only diagram).
- URP / multi-node reconciliation.

## 5. Definition of Done

**DoD documents searched:**
- `docs/gtm/node0_activation_go_to_market_v0_1/NODE0_DEFINITION_OF_DONE.md` — **NOT FOUND** on disk.

**Impact:** Tier-by-tier DoD exists in operator memory + scoreboard rows, not in on-disk canon. This is the single most actionable doc-gap.

**Action:** author `NODE0_DEFINITION_OF_DONE.md` (separate, typed-auth lane — not started here) covering Tier A-E criteria, evidence paths, and owners.

## 6. Public / private claim split

| Surface | Discipline state |
|---|---|
| **Brand canon v0.2** (`bizra_brand_identity_canon_v_0.md`) | ✅ Explicit §15 "Avoid until verified" list |
| **Launch media kit** (`docs/brand/public_launch_media_kit_v0_1/`) | ✅ `CLAIM_DISCIPLINE.md` reiterates the rule |
| **Public claims register** (`docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md`) | ✅ Full A/B/C/D/E classification |
| **Live bizra.ai** | ❌ Drifted — C4/C5/C7/C9 live without receipts |
| **CLAUDE.md** | ✅ Internal — appropriate technical density |
| **Internal docs** (`docs/strategy/`, `docs/specs/`, `docs/knowledge/`) | ⚠️ 75 "production-readiness" matches — internal drift |

## 7. Internal-drift watchlist

From claim scan (`artifacts/claims_register.json`), patterns visible in internal docs that should not leak to consumers:

- **75** "production ready / live / GA" mentions in internal docs.
- **14** "AGI" mentions (many are in research/comparison context; some are candidate claims to be sanitized).
- **17** "Ihsan threshold" mentions — fine for internal, only contextually-framed forms belong in consumer copy.
- **12** "SNR exact number" mentions in docs — same rule.

## 8. Documentation debts (ranked)

| # | Debt | Impact | Action |
|---|---|---|---|
| DD1 | `NODE0_DEFINITION_OF_DONE.md` does not exist | HIGH | Author (separate lane, typed-auth) |
| DD2 | No incident / key-rotation runbook | HIGH | Author short runbook |
| DD3 | No canonical runtime architecture diagram | MEDIUM | Draft SVG |
| DD4 | Canon Store Ingestion Gate spec + ADR missing | MEDIUM | Draft ADR (separate lane) |
| DD5 | Internal-doc drift sweep (75 "production ready" matches) | MEDIUM | Batch rewrite to directional language |
| DD6 | Node-onboarding runbook | MEDIUM | Needed for Genesis-100 path |
| DD7 | Backup / recovery runbook | LOW-MEDIUM | Author once Node0 activation is public |
