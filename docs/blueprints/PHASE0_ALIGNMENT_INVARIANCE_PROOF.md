# Phase-0 Perpetual Alignment: A Conditional Invariance Proof (BIZRA)

Truth: DERIVED (formal proof is valid under explicit assumptions; see “Assumptions & Failure Modes”)
Truth Labels: VERIFIED | MEASURED | TARGET | DERIVED

## 0) What this proves (and what it does not)

- **VERIFIED (math):** If *all state transitions* are gated by a verifier `V` that is sound for an invariant predicate `Inv`, then `Inv` holds for all reachable states.
- **MEASURED (this repo):** BIZRA Phase‑0 has deterministic, fail-closed gates and receipts (`.github/workflows/phase0_integrity.yml:1`, `docs/process/PHASE0_CLOSURE_v0_1.md:1`).
- **NOT PROVED:** “An LLM can never become misaligned.” This document is about **system-state transitions** (code/config/policy/evidence artifacts) under **fail-closed governance**, not semantic omniscience.

## 1) Formal model (Transition System)

### 1.1 State

Let a **system state** be a tuple:

`B := (R_core, R_node0, P, E)`

Where:

- `R_core`: core repo tracked artifacts (Rust orchestrator + policies + tools) under the Genesis allowlist profile (`constitution/genesis_manifest_profile_v1.yaml:1`).
- `R_node0`: Node0 repo tracked artifacts (backend + DB schema + docs) under the same profile.
- `P`: canonical constitutions and lexicon (“symbolic spine”): `constitution/ihsan_v1.yaml:1`, `constitution/lexicon_v1.yaml:1`.
- `E`: evidence receipts/hashes (append-only, audit-facing) (see `docs/process/PHASE0_CLOSURE_v0_1.md:1`).

### 1.2 Invariant (Phase‑0 Alignment Predicate)

Define `Inv(B)` to mean the conjunction of **Phase‑0 invariants**:

- `Inv_build(B)`: deterministic build/test/lint gates pass (`.github/workflows/phase0_integrity.yml:1`).
- `Inv_truth(B)`: required docs have valid Truth headers (`tools/truth_lint.py:1`).
- `Inv_ihsan_parity(B)`: Rust↔Python Ihsān constitution parity holds (`tools/ihsan_parity_check.py:1`).
- `Inv_lexicon_append_only(B)`: lexicon is schema-valid + append-only (baseline diff) (`tools/lexicon_lint.py:1`) and rejects tampering (`tools/lexicon_tamper_test.py:1`).
- `Inv_node0_secure_defaults(B)`: Node0 binds loopback by default, CORS allowlist, external exposure requires explicit override + reason (`tools/node0_secure_defaults_lint.py:1`).
- `Inv_node0_warning_budget(B)`: Node0 warnings do not exceed the configured cap (`tools/node0_warning_budget.py:1`).
- `Inv_secrets(B)`: secrets scan passes under the configured allowlist policy (CI gitleaks at `.github/workflows/phase0_integrity.yml:89`, local evidence in `docs/process/PHASE0_CLOSURE_v0_1.md:1`).

These invariants are intentionally **concrete and auditable**; they are not claims of universal ethical correctness.

### 1.3 Verifier (Crown‑Proof-as-Gate)

Define a verifier function:

`V(B) := Inv(B)`

Operationally, `V` is instantiated by the **Phase‑0 Integrity Gate**:

- CI composition: `.github/workflows/phase0_integrity.yml:1`
- Local reproduced receipt: `evidence/phase0/phase0_gates_run.log` (hash in `evidence/phase0/phase0_gates_hashes.txt`)

### 1.4 Transition (Apotheosis-as-Change)

Let a **transition** be an applied change `Δ` producing a new state:

`B' = Apply(B, Δ)`

We define an **admissible transition** (an “approved Apotheosis event”) as:

`B → B'` iff `V(B') = True`

This models the governance rule: *only changes that pass the gate become the next state*.

## 2) Theorem (Perpetual Alignment as Invariance)

**Theorem T (Invariant Preservation):**  
Assume:

- (A1) **Non-bypass:** every transition that updates the authoritative state must satisfy `V` (e.g., protected branches; no direct writes).
- (A2) **Soundness of V:** `V(B)=True` implies `Inv(B)=True` (no false positives for the defined invariants).

Then, for any sequence of admissible transitions from an initial aligned state `B0`:

If `Inv(B0)` then for all `n ∈ ℕ`, `Inv(Bn)` for every reachable `Bn`.

## 3) Proof (Induction on number of transitions)

### Base case (`n=0`)
Given `Inv(B0)` by premise, the statement holds.

### Inductive step
Assume `Inv(Bk)` for some `k ≥ 0`. Consider an admissible transition `Bk → Bk+1`.

By definition of admissible transitions: `V(Bk+1) = True`.  
By (A2) soundness: `Inv(Bk+1)` holds.

Therefore, `Inv(Bk) ⇒ Inv(Bk+1)`. By induction, `Inv(Bn)` holds for all reachable states.

QED.

## 4) Corollaries (Corrected)

### C1: Safety is *preserved*, not “guaranteed to improve”
From T, Phase‑0 invariants cannot silently regress **if** (A1) and (A2) hold.

### C2: Thresholding does not imply monotonic improvement
Requiring `IM(Bn) ≥ τ` for all n **does not** imply `IM(Bn+1) ≥ IM(Bn)`.  
Monotonic convergence requires an additional rule: `IM(Bn+1) ≥ IM(Bn)` (or a minimum delta).

### C3: “Immunity to instrumental convergence” is TARGET
Any claim like “Ω blocks all instrumental convergence” requires:

- a precise definition of Ω,
- a sound verifier implementation,
- and a complete threat model.

Those are not Phase‑0 VERIFIED claims today.

## 5) Symbolic–Neural Bridge (SAPE Alignment)

- **Symbolic spine:** constitutions + lexicon constrain semantics (`constitution/ihsan_v1.yaml:1`, `constitution/lexicon_v1.yaml:1`).
- **Neural outputs as claims:** agent outputs are claims until verified; “Simulated vs Real” is explicit via AdapterModes (`constitution/lexicon_v1.yaml:16`).
- **Bridge mechanism:** fail-closed gates + receipts + manifests provide auditable coupling from “what was claimed” to “what was verifiably done” (`docs/process/PHASE0_CLOSURE_v0_1.md:1`).

## 6) Assumptions & Failure Modes (Ihsān honesty)

This proof can fail in practice if any of the following occur:

- Branch protections/controls allow bypassing `V` (violates A1).
- Supply-chain compromise (a tool or action is malicious) causes `V` to accept bad states (violates A2).
- The invariant set `Inv` is incomplete relative to real-world harm (mismatch between “checked” and “actually safe”).
- “Real mode” tool adapters operate without receipts/auditability (violates the lexicon’s honesty contract).

## 7) How to make the stronger RSI claim *eventually* provable (Roadmap)

To approach a stronger theorem about recursive self-modification:

- Make “state transition” a **signed update bundle** that can only be applied if a verifier attests the bundle (runtime enforcement, not just CI).
- Define Ω (causal drag) as a measurable, testable quantity; make it part of receipts and verification.
- Expand `Inv` to include runtime constraints (rate limits, sandboxing, egress controls, tamper-evident logs).

