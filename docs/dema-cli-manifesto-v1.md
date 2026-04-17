# Dema CLI Manifesto — v1 (amended per external convergence)

بسم الله الرحمن الرحيم

**Author:** Mumo (Muhammad Beshr), BIZRA Node0 principal
**v0 drafted:** 2026-04-17 Cycle-5 closure
**v1 drafted:** 2026-04-17 later same day
**Supersedes:** `docs/dema-cli-manifesto-v0.md` (preserved as founding state)
**Amendment record:** `docs/manifesto-amendments/v0-to-v1.md`
**Companion docs:** `docs/bizra-trust-compiler-thesis.md` · `docs/ftap-function-registry-rfc-seed.md`

---

## 1. One-line elevator (amended)

> **BIZRA is not building an assistant. BIZRA is building the operating law for assistants.**
> **`dema` is the principal's terminal face of that law: every action receipted, every claim verifiable, every mission lawful before it runs.**

The v0 elevator ("the only CLI where...") was accurate but centered on the CLI. The v1 elevator centers on the **category** — *operating law for assistants* — which is what the external convergence analysis (and BIZRA's own commit history) actually supports.

## 2. The paradigm upgrade (NEW in v1)

The AI industry is evolving through three product categories:

| Category | What the AI does | Example |
|---|---|---|
| **Generative** | produces text/images/audio | ChatGPT, Midjourney |
| **Agentic** | uses tools to take action | Claude Code, Manus, Devin |
| **Verificative** | evaluates admissibility, emits receipted proof, supports replay | **BIZRA** |

*Verificative AI* is not a refinement of Agentic — it is a different category. Agentic optimizes for capability breadth. Verificative optimizes for **proof that capability was used lawfully**. Dema CLI is the first shipping verificative interface.

## 3. What Dema CLI IS (unchanged from v0)

Six defining properties, all present today:

1. Cryptographic chain of every action (BLAKE3 domain-separated receipts)
2. Constitutional admissibility gates evaluated before any chain mutation
3. Independently-replayable receipts via decode round-trip
4. IHSAN_FLOOR refusal to canonicalize sub-quality work (0.95, no override)
5. NO_SHADOW_STATE: the chain is the source of truth; UI renders only what the chain holds
6. Fully local, zero-cloud sovereignty

## 4. What Dema CLI is NOT yet (unchanged from v0)

1. Not a complete agent framework (no tool execution yet)
2. Not an IDE assistant (Claude Code / Cursor / Aider serve that niche)
3. Not a cloud agent (Manus serves that niche)
4. Does not persist across restarts (InMemoryPayloadStore default)
5. Does not expose PAT-7 / SAT-5 internals (§8 rule)
6. Does not claim an impact receipt (intent receipts only)

## 5. Standing on whose shoulders (EXPANDED in v1)

### Intellectual ancestry (added in v1)

- **Ibn al-Haytham (Alhazen)** — *Kitāb al-Manāẓir*, 11th century. Verification over speculation, 600 years before Bacon. The admissibility gate is his scientific method applied to AI action.
- **Norbert Wiener** — *Cybernetics* (1948). State differential Δ = |ideal − current| is the error signal; the runtime is the governor; admissibility is the constraint on control actions. `FourStateModel.gap` is cybernetics in Rust.
- **Claude Shannon** — *A Mathematical Theory of Communication* (1948). The receipt chain is noise-resistant communication; domain-tagged BLAKE3 prevents semantic drift across receipt kinds.
- **Butler Lampson** — "Hints for Computer System Design" (1983). R1: the chain is truth, the graph is derived state.
- **البذرة / الرسالة** — founding covenant, Ramadan 1444. Constitutional anchors (IHSAN, RIBA_ZERO, SADAQAH, ZANN_ZERO) trace to these texts directly.

### Technical shoulders (unchanged from v0)

`tokio`, `axum`, `clap`, `blake3`, `sled` (future), `reqwest`, `serde`, MCP (Cycle-6+), Ollama (Cycle-7+), Rust ecosystem.

### Ecosystem shoulders (unchanged from v0)

Claude Code (IDE loop — we don't rebuild it, we wrap it), Manus (cloud agent — different category), OpenDevin/Aider (open agentic CLI — we route through admissibility), LangGraph/CrewAI/AutoGen (orchestration frameworks — mission-centric vs agent-centric).

## 6. What we uniquely add (unchanged from v0)

The five constitutional invariants enforced by code at `admissibility_freeze_v1.rs`:

1. **ZANN_ZERO** — no claim without evidence binding
2. **CLAIM_MUST_BIND** — every chain claim carries hash-addressed evidence
3. **RIBA_ZERO** — no extractive economic pattern
4. **NO_SHADOW_STATE** — UI can't render what chain doesn't hold
5. **IHSAN_FLOOR** — 0.95 quality floor, no override

## 7. Completion path — three arcs (unchanged from v0)

1. **Arc 1 (Cycle-6)** — tool execution with per-call admissibility via MCP
2. **Arc 2 (Cycle-7)** — LLM inference with IHSAN_FLOOR enforcement on completions
3. **Arc 3 (Cycle-6, parallel)** — persistence across restart via `sled-store` feature

When all three land, the verificative AI category is occupied in production, not just prototype.

## 8. Non-goals (unchanged from v0, one addition)

1. No global cloud state
2. No UI theatrics
3. No agent auto-spawn without admissibility
4. No silent retries or fallback fabrication
5. No competing with Claude Code on IDE ergonomics
6. **(NEW in v1)** No work on FTAP / function registry before the three completion arcs land. FTAP is Layer 2 — strategic north star, not immediate sprint scope. See `docs/ftap-function-registry-rfc-seed.md`.

## 9. Risk and honest weakness (unchanged from v0)

Market education cost, developer ergonomics gap, replay honesty ceiling (decode vs deterministic re-execution), supply-chain rigor, cross-language threshold drift.

## 10. Immediate next step (REORDERED in v1)

Per Mumo's Cycle-5 correction: the ordering v0 implied was right-in-principle but wrong-in-sequence. Immediate over long-range:

### Immediate next step (weeks, not months)

1. **G4** — Mumo's browser/CLI acceptance of the principal activation receipt
2. **Arc 3 — persistence** (Cycle-6 parallel): `sled-store` feature enabled + `rehydrate()` wired at gateway boot. `dema chain --since today` becomes answerable across restarts.
3. **Arc 1 — tool execution** (Cycle-6): MCP protocol wired into `submit_mission` as sub-mission pattern. First deliverable: `dema submit "organize my Downloads folder"` actually organizes files + produces per-file-move receipts.

### Long-range next step (cycles, not weeks)

4. **Arc 2 — LLM inference** (Cycle-7): Ollama + cloud fallback, IHSAN_FLOOR-gated completions
5. **FTAP-lite v0.1** (Cycle-8 or later): local function registry + principal-signed attestations — *strategic north star*, not sprint scope

v0 of this document said the "first domino" was FTAP. That was wrong. The first domino is G4 + persistence + tool execution in that order.

## 11. Version discipline (preserved from v0, strengthened)

Amendments require:
1. Explicit Mumo authorization with versioned commit message
2. Constitutional-filter audit (do all 5 invariants survive?)
3. Diff record in `docs/manifesto-amendments/`

v0 → v1 followed this protocol. See `docs/manifesto-amendments/v0-to-v1.md`.

---

## 12. Self-governance note

This manifesto is the only document in BIZRA governed by its own amendment discipline. That is deliberate: a constitutional-product's constitution must itself be constitutional. The fact that v0 → v1 happened within the same session is not a bug — it is the speed-of-doctrine that a live-built trust compiler requires. External convergence evidence (from a separate AI's analysis of unrelated transcripts) was the specific trigger; the amendment record documents what changed and why.

---

> **Close doctrine into contracts. Close contracts into runtime. Close runtime into proof. Close proof into reveal.**
>
> Dema CLI is where those four closures meet in one operator surface.
>
> BIZRA is the operating law under which they remain closed.
>
> الحمد لله.
