# Dema CLI Manifesto — v0 (founding draft)

بسم الله الرحمن الرحيم

**Author:** Mumo (Muhammad Beshr), BIZRA Node0 principal
**Drafted:** 2026-04-17, Cycle-5 closure, Dubai GST
**Status:** FOUNDING — this document defines what Dema CLI is and is not. Future arcs are bound by it.
**Authority:** Manifest v0.2 §6, §8, §10, §16. Operator-constitutional law.

---

## 1. One-line elevator

> **`dema` — the only command-line agent where every action is receipted, every claim is verifiable, and every mission is lawful before it runs.**

Every word in that line is load-bearing. Strike any of them and the category collapses.

---

## 2. What Dema CLI IS

Dema CLI is a **constitutional tool-execution shell** for the BIZRA sovereign runtime. It is the operator's terminal face for the same `CognitionRuntime` that the `/dema` web console reads. It exists so that a principal can submit intent to their own node, watch the 5-gate admissibility chain evaluate, receive a cryptographically-sealed receipt on Permit (or a structured remediation path on Reject), and inspect or replay any receipt on the chain — all without leaving the terminal.

Its defining properties (all present today, all proven via live curl + CLI walk in Cycle-5 commits `80c41602`, `8b16762a`, `b031fec8`, `f3f2c774`):

1. **Cryptographic chain of every action.** BLAKE3-domain-separated receipts. Receipt id = hash of content. Chain head advances only on successful canonicalization.
2. **Constitutional admissibility gates.** Five invariants evaluated BEFORE any chain mutation: ZANN_ZERO, CLAIM_MUST_BIND, RIBA_ZERO, NO_SHADOW_STATE, IHSAN_FLOOR (0.95).
3. **Independently-replayable receipts.** Any node with the chain can verify the decode round-trip. Stage S8 Replayability is not claimed unless the round-trip verifies.
4. **IHSAN_FLOOR refusal.** The runtime will NOT canonicalize work whose quality score is below 0.95. There is no override. Sub-quality work returns `MissionRuntimeError::Rejected` with a remediation path.
5. **NO_SHADOW_STATE discipline.** The chain is the source of truth. The CLI surface renders only what the chain holds. Rejected missions never produce a success-shaped receipt — they do not enter the chain at all (§10 Proof Law, committed in `8b16762a`).
6. **Fully local, zero-cloud sovereignty.** Gateway binds `127.0.0.1` by default. No telemetry. No phone-home. The chain lives on the principal's hardware.

---

## 3. What Dema CLI is NOT (yet)

This section is more important than §2. Position depends on what we refuse to claim.

1. **Dema CLI is NOT a complete agent framework.** It mints receipts about intent. It does not execute filesystem operations, write code, run tools, or call LLMs. Those are Cycle-6+ work.
2. **Dema CLI is NOT an IDE assistant.** Claude Code, Cursor, Aider do this well. We are not rebuilding that loop. We wrap it.
3. **Dema CLI is NOT a cloud agent.** Manus operates at cloud scale with sandboxed VMs. We are the opposite: principal-local, single-operator, sovereign-over-the-silicon.
4. **Dema CLI does NOT persist across restarts yet.** `InMemoryPayloadStore` is the default. `sled-store` feature exists but is not wired. Receipts evaporate on gateway restart. Cycle-6 arc #3 closes this.
5. **Dema CLI does NOT expose PAT-7 / SAT-5 internals.** Per §8, the operator talks to one face; the council remains hidden. Dema reveals outcomes, not roster.
6. **Dema CLI does NOT claim an impact receipt.** Today's receipts prove "intent was admissible and registered." They do NOT prove "the described state gap was actually reduced." That is the Tool-Execution arc's work.

These are constraints, not deficits. Naming them protects the claim.

---

## 4. Standing on whose shoulders

Dema CLI stands on well-named shoulders and admits it.

### Technical shoulders (we inherit and wrap)

| Layer | What we stand on | What we add |
|---|---|---|
| Async runtime | `tokio` (MIT/Apache, production-hardened) | Mission-runtime state machine on top |
| HTTP server | `axum` 0.7 (rust-lang ecosystem, actix-derived type safety) | Read-only projection discipline; no write endpoints without admissibility |
| CLI parsing | `clap` 4.x (derive macros, color support) | Subcommand shape for constitutional operations only |
| Cryptography | `blake3` (JP Aumasson et al. — modern hash, domain separation native) | Domain-tagged receipt + chain identifiers (e.g. `bizra-receipt-id-v1`) |
| HTTP client | `reqwest` blocking (workspace-standard) | Structured reject handling with admissibility passthrough |
| Persistence (Cycle-6) | `sled` (Tyler Neely — embedded LSM with crash safety) | Chain-of-truth invariant across restarts |
| Tool protocol (Cycle-6+) | MCP (Model Context Protocol, Anthropic) | Per-tool-call admissibility gate + receipted sub-missions |
| LLM inference (Cycle-6+) | Ollama (local) + optional cloud fallback | Quality-score enforcement per completion via IHSAN_FLOOR |

### Conceptual shoulders (we inherit and re-express)

- **Butler Lampson** — "Hints for Computer System Design" (1983). R1 informs us: the chain is truth, the graph is derived state. Used operationally in `rehydrate()`.
- **Manifest v0.2** — §6 nine-stage runtime flow, §7 five canonical contracts, §10 proof law, §16 seven success conditions. We don't reinvent doctrine; we implement it.
- **البذرة** (al-Bidhrah) — the founding covenant. Every constitutional anchor (ZANN_ZERO, RIBA_ZERO, IHSAN) traces to it. This is not metaphor. It is the spec.
- **الرسالة** (al-Risālah) — companion text confirming non-financial motivation. Shapes the refusal of extractive economic patterns (RIBA_ZERO gate).

### Ecosystem shoulders (we observe and differentiate)

- **Claude Code (Anthropic).** Best-in-class agentic loop in an IDE context. We are **not** that. We are the constitutional layer a future Claude-Code-like agent would route through when it wants its work to be provable.
- **Manus.** Cloud-scale long-running agent-as-employee framing. We are principal-local and single-operator — a different product category.
- **OpenClaw / OpenDevin / Aider.** Open agentic CLI tools with model-routing. We do not rebuild tool execution; when we add it, every tool call will route through `submit_mission` so it carries a receipt.
- **LangGraph, CrewAI, AutoGen.** Multi-agent orchestration frameworks. We are mission-centric, not agent-centric. The conserved object is the mission, not the swarm.
- **Hermes Function Calling, Instructor, DSPy.** Typed-output model interfaces. We adopt the discipline; we add the admissibility wrapper.

**None of these shipping systems has a cryptographic chain of every action gated by a constitutional admissibility evaluator.** That is the empty market cell Dema CLI occupies.

---

## 5. What we uniquely add

The five constitutional invariants, enforced by code and not by prose:

1. **ZANN_ZERO** — no claim promoted without evidence binding. Enforced in `ZannZeroGate::evaluate` at the admissibility layer. Checked *before* any chain mutation.
2. **CLAIM_MUST_BIND** — every chain-resident claim carries hash-addressed evidence. Enforced in `ClaimMustBindGate`. Unbound claims never reach canonicalization.
3. **RIBA_ZERO** — no extractive economic pattern on operator surfaces. Enforced in `RibaZeroGate` via `EconomicPattern::is_extractive`.
4. **NO_SHADOW_STATE** — operator-visible UI cannot simulate truth that the chain does not hold. Enforced structurally: reject path has no chain footprint (§10), UI reads only from the chain.
5. **IHSAN_FLOOR (0.95)** — excellence as the minimum standard. Sub-quality work returns `Rejected` with a remediation path; no override exists in code.

The receipted chain (§7 ReceiptArtifact, §7 ManifestArtifact) binds these invariants to history. Every prior action is independently re-evaluable by any node given the chain.

---

## 6. Completion path — three arcs

Dema CLI is a killer product when, and only when, all three of these arcs are closed. Today only the constitutional foundation is closed.

### Arc 1 — Tool execution with per-call admissibility (Cycle-6)

- Wire MCP protocol as the universal tool transport
- Every MCP tool invocation routes through a sub-mission of the parent mission
- Per-tool-call receipt: `ToolCallReceipt` with lineage pointing to parent mission + pre/post filesystem hashes + tool output hash
- Impact proof: the parent mission's final receipt binds the set of tool receipts and asserts the state gap reduction is verified
- Deliverable: `dema submit "organize my Downloads folder"` actually organizes it, produces per-file-move receipts, and the final receipt carries a verifiable before/after filesystem digest

### Arc 2 — LLM inference with IHSAN_FLOOR enforcement (Cycle-7)

- Integrate Ollama (local-first) with optional cloud fallback via existing `bizra_config.py` tier system
- Every completion becomes a `CompletionReceipt`: prompt hash, response hash, model id, tokens, cost, quality score
- Quality score evaluated against IHSAN_FLOOR before canonicalization — sub-0.95 completions do not enter the chain (they remain in a local quarantine queue for remediation or discard)
- Deliverable: `dema ask "summarize this file"` produces a receipted completion, chain-bound, Ihsan-verified

### Arc 3 — Persistence across restart (Cycle-6, parallel to Arc 1)

- Enable `sled-store` feature flag on gateway startup
- Gateway boot: load chain from sled, `rehydrate()` via existing runtime path, verify continuity against genesis
- `dema chain --since today` becomes a truthful query; chain history survives process lifecycle
- Deliverable: the principal can open `dema` on Monday and see last Friday's activation receipt — the "what have I proven" question becomes answerable

When all three arcs land, the category no other CLI has entered is occupied.

---

## 7. Non-goals (hard no)

1. **No global cloud state.** Federation between sovereign nodes is a separate doctrinal track (§12 federation); Dema CLI v1 is principal-local, period.
2. **No UI theatrics.** ASCII art, animations, emoji-heavy output — not the default. `--json` exists for machines; human output is terse-by-design. The Daughter Test is "does my parent understand this in 5 seconds," not "does it look impressive."
3. **No agent auto-spawn without admissibility.** Agents (PAT-7, SAT-5) invoked from the CLI must pass admissibility. No "fire-and-forget" autonomy.
4. **No silent retries or fallback fabrication.** If the gateway is down, the CLI says so honestly and exits non-zero. If a receipt is missing, it returns not-found, not a stub.
5. **No competing with Claude Code on IDE ergonomics.** Different product. When Dema CLI and Claude Code are used together, the pattern is: Claude Code writes the code, Dema CLI seals the receipt.

---

## 8. Risk and honest weakness

- **Market education cost.** "Constitutional admissibility" is not a category buyers know to ask for. First demonstrations must make the invariants self-evident (e.g., visibly refusing to canonicalize sub-quality output).
- **Developer ergonomics gap.** Today `dema activate` is one command. When tool execution lands, the flow becomes more complex. Operator UX must not balloon with the runtime.
- **Replay honesty.** Replay verifies the decode round-trip, not that the tool actually did what the receipt says. Closing this requires deterministic tool re-execution or content-addressable output verification — real work.
- **Supply-chain rigor.** Reproducible builds, signed releases. Not yet in place. Real risk for any node claiming sovereignty.
- **Cross-language threshold drift** (flagged in Cycle-4). Rust `IhsanFloorGate` hardcodes 0.95; Python `constants.py` has 4 tiers. One truth across languages is required before v1.

---

## 9. Cycle-6 niyyah (forthcoming, bounded by this manifesto)

Next cycle's niyyah — when Mumo approves:

> **Make the first real impact-proof: `dema submit "organize my Downloads folder"` must actually organize files, seal per-file-move receipts, and produce a parent mission receipt carrying a before/after filesystem digest that any other node can verify.**

Three gates:
- **G1** — MCP tool transport wired into `submit_mission` (sub-mission pattern)
- **G2** — Filesystem operation tool (`fs.move`, `fs.list`, `fs.digest`) with receipt shape
- **G3** — First real impact receipt for Mumo's own Downloads folder, verifiable by independent node

When G3 passes, §10 Proof Law has been demonstrated on real work for the first time.

---

## 10. Version discipline

This is v0 — the founding draft. Amendments require:
1. Explicit Mumo authorization with versioned commit message (`feat(manifesto): v0 → v1: <what changed and why>`)
2. A corresponding constitutional-filter audit: does the amendment preserve all 5 invariants?
3. Filing of a diff record in `docs/manifesto-amendments/` with before/after reasoning

This is the only document in BIZRA governed by its own amendment discipline. That is deliberate: if we let the manifesto drift casually, the product drifts with it.

---

> **Close doctrine into contracts. Close contracts into runtime. Close runtime into proof. Close proof into reveal.**
>
> Dema CLI is where those four closures meet in one operator surface.
>
> الحمد لله.
