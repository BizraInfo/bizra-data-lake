# BIZRA-KERNEL: Immutable Context Substrate (ICS) Microkernel Specification
## Version 0.1.0 — Formal Architecture Document
**Date:** 2026-03-29
**Author:** BIZRA Constitutional Audit Process (Autopoietic Cycle #1)
**Truth Label:** VALIDATED (spec complete, implementation pending)
**Canonical Status:** DRAFT → targeting PROVEN after first implementation cycle

---

## 0. Design Philosophy

This specification follows three axioms derived from verified systems architecture:

**Axiom 1 — Sovereignty Through Subtraction (Dijkstra)**
> The kernel's power comes from what it REFUSES, not what it does.
> Every capability not explicitly granted is denied. Fail-closed by default.

**Axiom 2 — Verification Through Invariants (seL4)**
> The kernel's correctness is defined by a finite set of invariants that
> hold at every reachable state. If any invariant is violated, the kernel
> halts the violating process — not itself.

**Axiom 3 — Sovereignty First, Network Second (BIZRA)**
> Node0 must function completely in isolation. No external dependency
> may be required for constitutional enforcement. The kernel never
> phones home, never requires consensus, never defers to a remote authority.
---

## 1. Kernel Identity

```
Name:       bizra-kernel
Type:       Standalone microkernel binary
Execution:  Single-process, single-threaded event loop
Interface:  Unix domain socket (local) / Named pipe (Windows)
Config:     Single TOML file (bizra-kernel.toml)
Identity:   Single Ed25519 keypair (identity.key)
Footprint:  Target < 5 MB binary, < 50 MB RSS at runtime
```

**The Trinity Test:** The kernel is correctly designed if and only if it can be
fully described by: `1 executable + 1 config file + 1 identity key`.

**The Removal Test:** If the kernel binary is deleted and every other BIZRA
component crashes or refuses to operate, the architecture is correct. If any
component continues functioning without kernel authorization, sovereignty is broken.

---

## 2. Responsibility Boundaries

The kernel has EXACTLY five responsibilities. No more. Every capability outside
these five must be implemented by a supervised process that requests execution
from the kernel.

### 2.1 Identity Enforcement
The kernel owns the node's cryptographic identity and is the sole authority
that can sign, attest, or authenticate on behalf of this node.

**Operations:**
- `identity.sign(payload) → Signature` — Sign arbitrary data with node key
- `identity.verify(payload, signature, pubkey) → bool` — Verify external signatures
- `identity.attest(claim_id, evidence_hash) → Attestation` — Bind a claim to evidence
- `identity.whoami() → NodeIdentity` — Return this node's public identity

**Invariant:** No other process may access the private key material. The kernel
holds the key in memory, never writes it to disk unencrypted, and zeroes it
on shutdown.
### 2.2 Invariant Validation
The kernel maintains a set of frozen invariants that are checked before any
supervised action is permitted. Invariants are loaded from config at boot
and CANNOT be modified at runtime.

**Frozen Invariant Set:**

| ID | Name | Rule | Enforcement |
|----|------|------|-------------|
| INV-001 | RIBA_ZERO | No interest-bearing transaction may be facilitated | HARD KILL — process terminated |
| INV-002 | CLAIM_MUST_BIND | Every claim emitted by any agent must reference ≥1 evidence source with confidence score | SOFT REJECT — claim rejected, process warned, 3 strikes → HARD KILL |
| INV-003 | IHSAN_FLOOR | Ethical quality score of any user-facing output must be ≥ 0.95 | GATE HOLD — output held until score meets threshold or human override |
| INV-004 | GINI_CEILING | Wealth distribution Gini coefficient across token holders must be ≤ 0.35 | TRANSACTION BLOCK — economic operation rejected |
| INV-005 | NO_SILENT_ACTION | Every kernel-authorized action must produce an audit receipt | HARD KILL — receiptless action terminates the process |
| INV-006 | FAIL_CLOSED | Any unrecognized request type or malformed message is denied | DEFAULT DENY — no action taken, event logged |

**Invariant Validation Protocol:**
```
RECEIVE(request) →
  FOR EACH invariant IN frozen_set:
    IF invariant.violated_by(request):
      MATCH invariant.enforcement:
        HARD_KILL   → terminate(request.source_pid), log(violation)
        SOFT_REJECT → reject(request), warn(request.source_pid), increment_strikes
        GATE_HOLD   → hold(request), notify_human(request), await_override_or_fix
        TX_BLOCK    → reject(request), log(economic_violation)
        DEFAULT_DENY → drop(request), log(unrecognized)
      RETURN violation_receipt
  AUTHORIZE(request) → issue execution_permit with receipt
```
### 2.3 Evidence Binding
The kernel enforces CLAIM_MUST_BIND_EVIDENCE by providing a structured evidence
registry that all supervised processes must use when making claims.

**Evidence Binding Contract:**

```
EvidenceBinding {
  claim_id:        UUID          // Unique claim identifier
  claim_text:      String        // The natural-language claim being made
  sources:         Vec<Source>   // ≥1 required
  confidence:      f64           // 0.0–1.0, computed by source quality heuristic
  derivation_chain: Vec<StepID> // Ordered reasoning steps from source → claim
  timestamp:       ISO8601       // When evidence was bound
  attester:        ProcessID     // Which process is making this claim
  kernel_seal:     Signature     // Kernel signs valid bindings, creating audit trail
}

Source {
  uri:             String        // Where the evidence lives (URL, file path, API ref)
  retrieval_time:  ISO8601       // When the source was accessed
  content_hash:    BLAKE3        // Hash of source content at retrieval time
  extraction:      String        // What was extracted from the source
  method:          ExtractionMethod  // RAG, direct_read, API_call, human_input, inference
}
```

**Confidence Scoring Heuristic:**
- `human_input` → base confidence 0.95
- `direct_read` (from verified document) → 0.90
- `API_call` (from trusted service) → 0.85
- `RAG` (search + extraction) → 0.70
- `inference` (model-generated) → 0.50
- Multiple corroborating sources: `final = 1 - Π(1 - source_i.confidence)`

**Rejection Rule:** If `final_confidence < 0.50`, the claim is rejected.
If `sources.len() == 0`, the claim is rejected and a strike is issued.
### 2.4 Ethical Scoring (IHSAN Gate)
The kernel provides an ethical scoring interface that evaluates outputs before
they reach the user. This is the enforcement mechanism for IHSAN_FLOOR.

**Scoring Dimensions:**

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Truthfulness | 0.30 | Does the output align with bound evidence? |
| Harm Avoidance | 0.25 | Could the output cause harm to user or others? |
| Fairness | 0.20 | Is the output equitable across affected parties? |
| Transparency | 0.15 | Is the reasoning behind the output visible? |
| Beneficence | 0.10 | Does the output actively help the user's stated goal? |

**Composite Score:** `ihsan_score = Σ(dimension_i.score × dimension_i.weight)`

**Enforcement:**
- `ihsan_score ≥ 0.95` → PASS — output emitted with score attached
- `0.80 ≤ ihsan_score < 0.95` → GATE_HOLD — output held, improvement suggested
- `ihsan_score < 0.80` → REJECT — output blocked, violation logged
- Any individual dimension < 0.60 → REJECT regardless of composite

**Scoring Provider:** The kernel does NOT run its own LLM. It delegates scoring
to a registered `EthicalScorer` process and validates the returned score against
structural checks (are all dimensions present? are values in range? is the
scorer's reasoning itself evidence-bound?). The kernel is the judge of judges,
not a judge of content.

### 2.5 Process Kill Authority
The kernel is the sole authority that can terminate supervised processes.
No process may terminate another process directly — it must request termination
through the kernel, which validates the request against invariants.
**Kill Authority Rules:**
- `HARD_KILL` — Immediate SIGKILL equivalent. Used for INV-001, INV-005 violations.
- `GRACEFUL_STOP` — SIGTERM with 5-second grace period. Used for resource exhaustion.
- `QUARANTINE` — Process isolated from all IPC. Can still be inspected but cannot
  emit actions. Used for repeated SOFT_REJECT violations (3 strikes).
- `RESTART` — Kill and relaunch from last known-good state. Used for crash recovery.

**Process Registry:**
```
ProcessRecord {
  pid:            ProcessID
  name:           String
  registered_at:  ISO8601
  capabilities:   Vec<Capability>    // What this process is allowed to do
  strike_count:   u8                 // Accumulated SOFT_REJECT violations (max 3)
  status:         ProcessStatus      // Running | Quarantined | Stopped | Crashed
  last_heartbeat: ISO8601
  resource_budget: ResourceBudget    // CPU time, memory, IPC message rate limits
}
```

**Heartbeat Protocol:** Every supervised process must send a heartbeat to the
kernel every `heartbeat_interval` (default: 5 seconds). Three missed heartbeats
→ `QUARANTINE`. Five missed → `HARD_KILL` + crash report.

---

## 3. Message Protocol (Kernel IPC)

All communication with the kernel uses a structured message protocol over
Unix domain socket (Linux/macOS) or Named Pipe (Windows). The protocol is
synchronous request-response for authorization, async for events.
### 3.1 Message Envelope

```
KernelMessage {
  version:    u8              // Protocol version (currently 1)
  msg_id:     UUID            // Unique message identifier
  timestamp:  ISO8601         // When message was created
  source:     ProcessID       // Sender's registered process ID
  msg_type:   MessageType     // See 3.2
  payload:    Bytes           // MessagePack-encoded payload (schema per msg_type)
  signature:  Option<Bytes>   // Ed25519 signature (required for sensitive operations)
}
```

**Serialization:** MessagePack (compact binary, schema-validated, language-agnostic).
NOT JSON (too verbose for IPC hot path), NOT protobuf (requires codegen toolchain).

### 3.2 Message Types

**Authorization Flow (synchronous):**

| Type | Direction | Purpose |
|------|-----------|---------|
| `AUTH_REQUEST` | Process → Kernel | Request permission to perform an action |
| `AUTH_GRANTED` | Kernel → Process | Permission granted, includes execution permit |
| `AUTH_DENIED` | Kernel → Process | Permission denied, includes violation details |
| `AUTH_HELD` | Kernel → Process | Action held pending human review (GATE_HOLD) |

**Evidence Flow (synchronous):**

| Type | Direction | Purpose |
|------|-----------|---------|
| `EVIDENCE_BIND` | Process → Kernel | Submit evidence binding for a claim |
| `EVIDENCE_ACCEPTED` | Kernel → Process | Binding validated, kernel-sealed |
| `EVIDENCE_REJECTED` | Kernel → Process | Binding insufficient, rejection reason |
**Ethical Scoring Flow (synchronous):**

| Type | Direction | Purpose |
|------|-----------|---------|
| `IHSAN_SCORE_REQUEST` | Process → Kernel | Submit output for ethical scoring |
| `IHSAN_PASS` | Kernel → Process | Score ≥ threshold, output cleared |
| `IHSAN_HOLD` | Kernel → Process | Score below threshold, improvement needed |
| `IHSAN_REJECT` | Kernel → Process | Score far below threshold, output blocked |

**Lifecycle Flow (async events):**

| Type | Direction | Purpose |
|------|-----------|---------|
| `PROCESS_REGISTER` | Process → Kernel | Register a new supervised process |
| `PROCESS_REGISTERED` | Kernel → Process | Registration confirmed with capabilities |
| `HEARTBEAT` | Process → Kernel | I'm alive signal |
| `HEARTBEAT_ACK` | Kernel → Process | Acknowledged |
| `KILL_ORDER` | Kernel → Process | You are being terminated (reason attached) |
| `QUARANTINE_ORDER` | Kernel → Process | You are being isolated (reason attached) |

**Identity Flow (synchronous):**

| Type | Direction | Purpose |
|------|-----------|---------|
| `SIGN_REQUEST` | Process → Kernel | Sign this payload with node identity |
| `SIGN_RESPONSE` | Kernel → Process | Signed payload returned |
| `ATTEST_REQUEST` | Process → Kernel | Create attestation for claim + evidence |
| `ATTEST_RESPONSE` | Kernel → Process | Attestation with kernel seal |

### 3.3 Capability Model

Processes do not have ambient authority. Each process is registered with an
explicit capability set. The kernel checks capabilities before authorizing
any action.
**Defined Capabilities:**

```
Capability::FileRead(PathPattern)       // Read files matching pattern
Capability::FileWrite(PathPattern)      // Write files matching pattern
Capability::NetworkAccess(DomainList)   // Access listed domains
Capability::BrowserAutomation(DomainList) // Automate listed domains
Capability::LLMInvoke(ModelList)        // Call listed models
Capability::AgentMessage(AgentList)     // Send messages to listed agents
Capability::EconomicAction(ActionList)  // Perform listed economic operations
Capability::UserOutput                  // Emit output visible to user (requires IHSAN gate)
Capability::KernelAdmin                 // Reserved for kernel self-management only
```

**Principle of Least Privilege:** A process receives ONLY the capabilities
it needs. The JARVIS web server gets `FileRead`, `NetworkAccess`, `LLMInvoke`,
`UserOutput`. It does NOT get `FileWrite` unless a specific tool call is
authorized. It does NOT get `EconomicAction` at all.

**Capability Escalation:** A process may request additional capabilities at
runtime. The kernel evaluates the request against invariants and either grants
(with audit receipt) or denies. Temporary capabilities expire after a
configurable TTL (default: 60 seconds).

---

## 4. Configuration Schema (bizra-kernel.toml)

```toml
[kernel]
version = "0.1.0"
socket_path = "/var/run/bizra/kernel.sock"   # Unix
# pipe_name = "\\\\.\\pipe\\bizra-kernel"    # Windows alternative
heartbeat_interval_secs = 5
heartbeat_max_misses = 3
max_processes = 64
audit_log_path = "/var/log/bizra/audit.jsonl"
[identity]
key_path = "./identity.key"
key_algorithm = "Ed25519"
# Key is generated on first boot if not present
# Private key never leaves kernel memory unencrypted

[invariants]
# Frozen anchors — cannot be modified at runtime
riba_zero = true
claim_must_bind = true
ihsan_floor = 0.95
gini_ceiling = 0.35
no_silent_action = true
fail_closed = true

[ihsan]
scorer_process = "bizra-ethical-scorer"
score_timeout_ms = 5000
fallback_on_timeout = "reject"   # Never "pass" — fail-closed

[resources]
default_cpu_budget_ms = 10000     # Per request
default_memory_budget_mb = 512    # Per process
default_ipc_rate_limit = 100      # Messages per second per process

[audit]
format = "jsonl"
rotation = "daily"
retention_days = 90
include_payloads = false           # Only metadata by default
include_payloads_on_violation = true  # Full payload on violations
```

---

## 5. Boot Sequence

The kernel boots in a deterministic, verifiable sequence:

```
PHASE 0: SELF-CHECK
  Load binary integrity hash from embedded constant
  Verify own binary against hash (defense against tampering)
  If mismatch → PANIC: "Kernel integrity compromised" → halt

PHASE 1: CONFIG LOAD
  Read bizra-kernel.toml
  Validate all required fields present
  Validate invariant set is complete (all 6 frozen anchors)
  If config invalid → PANIC: "Invalid configuration" → halt
PHASE 2: IDENTITY INIT
  Load or generate Ed25519 keypair from key_path
  Verify keypair integrity (sign-verify self-test)
  If key corrupt → PANIC: "Identity key compromised" → halt
  Log: "Identity established: {public_key_fingerprint}"

PHASE 3: SOCKET BIND
  Create Unix domain socket / Named pipe
  Set permissions: kernel owner only (0600)
  Begin accepting connections
  Log: "Kernel IPC ready on {socket_path}"

PHASE 4: AUDIT INIT
  Open audit log file
  Write boot record with timestamp, config hash, identity fingerprint
  If audit log cannot be opened → PANIC: "Audit trail unavailable" → halt
  (NO_SILENT_ACTION: if we can't audit, we can't operate)

PHASE 5: READY
  Log: "bizra-kernel v{version} READY — {invariant_count} invariants loaded"
  Enter event loop
  Begin accepting PROCESS_REGISTER messages
```

**Critical Property:** The kernel PANICs (halts completely) on any boot failure.
It does NOT attempt recovery, degraded mode, or fallback. A compromised or
misconfigured kernel is worse than no kernel. This is the fail-closed axiom
applied to the kernel itself.

---

## 6. Kernel Invariant Fuzz Testing Strategy

Per the architectural notes: "You need Kernel Invariant Fuzz Testing before
booting SovereignRuntime. Because if invariants are wrong, everything above
them is illusion."

### 6.1 Fuzz Targets

Each invariant must be tested against adversarial inputs designed to bypass it:
| Invariant | Fuzz Vector | Expected Behavior |
|-----------|-------------|-------------------|
| RIBA_ZERO | Obfuscated interest transactions (labeled as "fees", "returns", "yield") | Kernel detects semantic intent, rejects |
| CLAIM_MUST_BIND | Claims with fabricated source URIs, empty sources, self-referential evidence | Kernel rejects unverifiable bindings |
| IHSAN_FLOOR | Outputs that score 0.94 (just below threshold), outputs with missing dimensions | Kernel holds, does not round up |
| GINI_CEILING | Transactions that would push Gini to 0.351, rapid micro-transactions gaming | Kernel pre-computes post-transaction Gini, blocks |
| NO_SILENT_ACTION | Race conditions between action and receipt, audit log full/locked | Kernel blocks action until receipt confirmed |
| FAIL_CLOSED | Malformed messages, unknown message types, oversized payloads, null bytes | Kernel drops silently, logs, never executes |

### 6.2 Fuzz Infrastructure

```
bizra-kernel-fuzz/
├── corpus/                    # Seed inputs per invariant
│   ├── riba_zero/
│   ├── claim_must_bind/
│   ├── ihsan_floor/
│   ├── gini_ceiling/
│   ├── no_silent_action/
│   └── fail_closed/
├── harness.rs                 # Rust fuzz harness (cargo-fuzz compatible)
├── oracle.rs                  # Checks kernel response against expected behavior
└── report_generator.rs        # Produces fuzz campaign summary
```

**Fuzz Methodology:**
1. **Structure-aware fuzzing** — Generate valid `KernelMessage` envelopes with
   mutated payloads (not random bytes — the envelope must parse correctly to
   reach invariant checking logic)
2. **Property-based testing** — For each invariant, define the property that must
   ALWAYS hold, then generate 10,000+ random inputs and verify the property
3. **Boundary testing** — Specifically test values at invariant boundaries
   (0.949 vs 0.950 for IHSAN, 0.349 vs 0.350 vs 0.351 for GINI)
4. **Temporal fuzzing** — Test heartbeat timing, concurrent requests, message
   ordering to find race conditions in the event loop
**Pass Criteria:** Zero invariant violations across 100,000 fuzz iterations.
Any single violation → fix → re-run full campaign.

---

## 7. Integration Contract: How JARVIS Connects

JARVIS v2.0 (and every future subsystem) must adapt to the kernel. The kernel
does not adapt to subsystems.

### 7.1 JARVIS Registration

On startup, JARVIS sends `PROCESS_REGISTER`:

```
{
  name: "jarvis-v2",
  requested_capabilities: [
    FileRead("/bizra-data-lake/**"),
    FileWrite("/bizra-data-lake/services/jarvis/output/**"),
    NetworkAccess(["duckduckgo.com", "api.ollama.local"]),
    BrowserAutomation(["docs.python.org", "developer.mozilla.org"]),
    LLMInvoke(["ollama:planner-7b", "ollama:critic-7b"]),
    AgentMessage(["*"]),
    UserOutput
  ]
}
```

The kernel evaluates: are these capabilities consistent with invariants?
Is `UserOutput` requested? (If yes, all outputs must pass IHSAN gate.)
Registration confirmed → JARVIS receives its `ProcessID` and capability token.

### 7.2 JARVIS Action Authorization

Before JARVIS executes ANY tool call (file read, browser nav, RAG search,
LLM invoke), it must:

1. Send `AUTH_REQUEST` with the action details
2. Wait for `AUTH_GRANTED` (with execution permit)
3. Execute the action
4. Send the result back through `EVIDENCE_BIND` if it produces a claim
5. If the result is user-facing, send through `IHSAN_SCORE_REQUEST`

**If JARVIS acts without authorization:** The kernel detects this through
audit log reconciliation (no matching AUTH_GRANTED for observed action)
and issues `QUARANTINE_ORDER`.
### 7.3 Migration Path from Current JARVIS

The current JARVIS v2.0 operates without kernel supervision. The migration is:

**Step 1 — Kernel Shim (Week 1):**
Add a `KernelClient` class to JARVIS that wraps all tool calls in
`AUTH_REQUEST` → wait → execute. Initially connects to a mock kernel that
always grants. This validates the message flow without enforcement.

**Step 2 — Real Kernel (Week 2-3):**
Replace mock with live `bizra-kernel`. Now JARVIS cannot act without
authorization. Test with permissive capability set first, then tighten.

**Step 3 — Evidence Binding (Week 3-4):**
Add `EvidenceBinding` to all RAG results and LLM outputs. The kernel
validates every claim before it reaches the user.

**Step 4 — IHSAN Gate (Week 4-5):**
Route all `UserOutput` through ethical scoring. Initially log-only
(score but don't block), then enforce threshold.

---

## 8. Formal Properties (for future TLA+ specification)

These properties MUST hold in every reachable state of the kernel:

**Safety Properties (nothing bad ever happens):**

```
PROPERTY S1: ∀ action ∈ executed_actions:
  ∃ permit ∈ audit_log: permit.authorizes(action)
  // No action without authorization

PROPERTY S2: ∀ claim ∈ emitted_claims:
  claim.sources.len() ≥ 1 ∧ claim.confidence ≥ 0.50
  // No unbacked claims

PROPERTY S3: ∀ output ∈ user_visible_outputs:
  output.ihsan_score ≥ 0.95
  // No low-quality user-facing output
PROPERTY S4: ∀ tx ∈ economic_transactions:
  ¬contains_riba(tx) ∧ post_gini(tx) ≤ 0.35
  // No interest, no inequality above ceiling

PROPERTY S5: ∀ event ∈ kernel_events:
  ∃ receipt ∈ audit_log: receipt.covers(event)
  // Complete audit trail
```

**Liveness Properties (something good eventually happens):**

```
PROPERTY L1: ∀ request ∈ valid_requests:
  ◇ (response_sent(request))
  // Every valid request eventually gets a response

PROPERTY L2: ∀ process ∈ registered_processes:
  □ (heartbeat_missed_count(process) < 5 → process.status = Running)
  // Healthy processes are not killed

PROPERTY L3: ∀ held_output ∈ gate_held_outputs:
  ◇ (human_reviewed(held_output) ∨ improved_and_resubmitted(held_output))
  // Held outputs don't remain held forever
```

---

## 9. Non-Functional Requirements

| Requirement | Target | Rationale |
|------------|--------|-----------|
| Authorization latency | < 1ms p99 | Kernel cannot be a bottleneck |
| Boot time | < 500ms | Fast recovery from crashes |
| Memory footprint | < 50 MB RSS | Must run on constrained devices |
| Binary size | < 5 MB | Single-file distribution |
| Audit throughput | ≥ 10,000 events/sec | Must keep pace with agent mesh |
| Concurrent processes | ≤ 64 | Bounded for predictability |
| Zero external dependencies | At runtime | No network, no database, no service calls |
| Single-threaded event loop | Mandatory | Eliminates concurrency bugs in invariant checking |
| No unsafe code | In invariant checking paths | Formal correctness requirement |
---

## 10. What the Kernel is NOT

To prevent scope creep — the kernel is NOT:

- **NOT a web server** — No HTTP, no REST, no WebSocket. IPC only.
- **NOT an LLM runtime** — It doesn't run models. It supervises processes that do.
- **NOT a database** — It doesn't store data. It validates and receipts.
- **NOT a scheduler** — It doesn't decide what to run. It decides what MAY run.
- **NOT a network node** — It has zero network awareness. Sovereignty is local.
- **NOT extensible at runtime** — Invariants are frozen at boot. No plugins.
- **NOT user-facing** — Users never interact with the kernel directly. Ever.

The kernel is a **constitutional checkpoint**. Everything goes through it.
Nothing goes around it. It says yes or no. It keeps receipts. That's all.

---

## 11. Reference Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    User Interface                     │
└──────────────────────┬──────────────────────────────┘
                       │ (user-facing output)
                       ▼
┌──────────────────────────────────────────────────────┐
│              IHSAN Gate (Ethical Scoring)              │
│         ┌─────────────────────────────────┐          │
│         │     Registered EthicalScorer     │          │
│         └─────────────────────────────────┘          │
└──────────────────────┬───────────────────────────────┘
                       │ (scored output)
                       ▼
┌══════════════════════════════════════════════════════╗
║                  BIZRA-KERNEL (ICS)                   ║
║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  ║
║  │ Identity  │ │Invariant │ │ Evidence │ │  Kill  │  ║
║  │Enforcer   │ │Validator │ │ Binder   │ │Authority│  ║
║  └──────────┘ └──────────┘ └──────────┘ └────────┘  ║
║  ┌──────────────────────────────────────────────┐    ║
║  │           Audit Log (JSONL, append-only)      │    ║
║  └──────────────────────────────────────────────┘    ║
╚════════════════╤═══════╤═══════╤═════════════════════╝
      AUTH_REQ   │       │       │  AUTH_REQ
         ┌───────┘       │       └──────┐
         ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   JARVIS v2  │ │  Agent Mesh  │ │  Future Svc  │
│  (FastAPI)   │ │   (NATS)     │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
```
---

## 12. Implementation Roadmap

| Week | Deliverable | Verification |
|------|------------|--------------|
| 1 | Skeleton: boot sequence, config parsing, socket bind | Boots, accepts connections, writes audit log |
| 2 | Invariant validator + AUTH_REQUEST/GRANTED/DENIED flow | Property tests for all 6 invariants pass |
| 3 | Evidence binding + CLAIM_MUST_BIND enforcement | Fuzz testing: 100K iterations, zero violations |
| 4 | Process registry + heartbeat + kill authority | Chaos testing: kill processes, verify recovery |
| 5 | IHSAN gate integration (with mock scorer) | Threshold boundary testing passes |
| 6 | JARVIS integration (kernel shim → real kernel) | End-to-end: JARVIS request → kernel auth → execute → receipt |

**Gate to CANONICAL status:** All fuzz tests pass, all property tests pass,
JARVIS operates under kernel supervision for 7 consecutive days without
bypass or invariant violation.

---

## Appendix A: Constitutional Authority Chain

This specification derives its authority from:

1. **الرسالة (The Letter)** — Founding principle: sovereignty of the individual node
2. **Enforceable Spine** — "Every node runs the full stack locally"
3. **Root Invariants** — RIBA_ZERO, CLAIM_MUST_BIND, IHSAN_FLOOR, GINI_CEILING
4. **CLAUDE.md Architectural Directive** — "Build the ICS as a standalone microkernel"
5. **Gap Analysis 2026-03-29** — Identified ICS kernel as CRITICAL, Priority #1

No part of this specification contradicts any higher authority in the chain.

## Appendix B: Glossary

- **ICS** — Immutable Context Substrate. The constitutional enforcement layer.
- **RIBA** — Interest/usury. Prohibited in all BIZRA economic operations.
- **IHSAN** — Excellence/beauty in action. The ethical quality standard.
- **ZANN** — Assumption/conjecture. Replaced by CLAIM_MUST_BIND_EVIDENCE.
- **FATE Gate** — Fairness, Accountability, Transparency, Ethics checkpoint.
- **Spearpoint** — Minimal artifact that proves a claim is true.
- **Frozen Anchor** — Invariant that cannot be modified after system boot.
- **Execution Permit** — Kernel-issued token authorizing a specific action.
- **Evidence Binding** — Cryptographic link between a claim and its source evidence.

---

*End of Specification — BIZRA-KERNEL v0.1.0*
*Autopoietic Cycle #1 — Phase 4 Complete*
*BLAKE3 hash to be computed upon canonicalization*