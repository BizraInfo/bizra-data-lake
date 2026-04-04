# PRD: bizra-kernel — Immutable Context Substrate Microkernel
## Product Requirements Document v1.0
**Owner:** BIZRA Core Architecture
**Date:** 2026-03-29
**Status:** APPROVED FOR IMPLEMENTATION
**Spec Reference:** BIZRA_KERNEL_SPEC.md v0.1.0
**Truth Label:** VALIDATED

---

## 1. Problem Statement

Every BIZRA subsystem (JARVIS, agent parliament, economic engine) currently
operates without constitutional enforcement. Actions execute without ethical
checkpoints, claims are emitted without evidence binding, and no process
has authority to halt a violating component. This means the system's core
promise — sovereignty through verifiable ethical behavior — is architecturally
unenforceable.

**Who experiences this:** Every user interacting with BIZRA receives outputs
that have not passed through constitutional validation. Every developer
building on BIZRA has no guarantee that their subsystem respects frozen
anchors (RIBA_ZERO, CLAIM_MUST_BIND, IHSAN_FLOOR, GINI_CEILING).

**Frequency:** Every single request. 100% of system interactions are
currently unmediated by constitutional enforcement.

**Cost of not solving:** Without the kernel, BIZRA is an AI application
with aspirational ethics documentation. With the kernel, BIZRA is a
constitutionally enforced sovereign intelligence system. The difference
is the difference between a policy document and a law enforcement mechanism.
---

## 2. Goals

| # | Goal | Measurement | Target |
|---|------|-------------|--------|
| G1 | **Zero unauthorized actions** — Every action in the BIZRA system requires kernel authorization before execution | Audit log completeness: `authorized_actions / total_actions` | 100% (hard requirement) |
| G2 | **Evidence-bound claims** — Every claim emitted to users traces to ≥1 source | Evidence binding rate: `bound_claims / total_claims` | ≥ 95% within 30 days of kernel integration |
| G3 | **Ethical floor enforcement** — No user-facing output with IHSAN score < 0.95 | IHSAN gate pass rate at threshold | 100% enforcement (held outputs may be revised, never bypassed) |
| G4 | **Sub-millisecond authorization** — Kernel does not become a performance bottleneck | p99 AUTH_REQUEST → AUTH_GRANTED latency | < 1ms |
| G5 | **Sovereign isolation** — Kernel operates with zero external dependencies at runtime | External network calls during operation | 0 (hard requirement) |

---

## 3. Non-Goals

| # | Non-Goal | Why Out of Scope |
|---|----------|-----------------|
| NG1 | **Running LLM inference inside the kernel** | The kernel validates and gates — it does not reason. LLM inference belongs in supervised processes. Mixing them would violate the single-responsibility boundary. |
| NG2 | **Network-layer communication or node discovery** | Sovereignty first, network second. The kernel enforces local constitutional law. Network concerns are a separate subsystem (Phase 3+). |
| NG3 | **User-facing interface** | Users never interact with the kernel directly. They interact with supervised services (JARVIS) that operate under kernel authority. |
| NG4 | **Economic engine implementation** | Token logic (SEED/BLOOM) is Phase 2+. The kernel provides the GINI_CEILING and RIBA_ZERO invariants but does not implement token mechanics. |
| NG5 | **Runtime extensibility or plugin system** | Invariants are frozen at boot. This is a feature, not a limitation. Runtime modification of constitutional anchors would undermine sovereignty. |
---

## 4. User Stories

### Primary Persona: BIZRA Subsystem Developer
The developer building or maintaining a BIZRA service (JARVIS, agents, tools)
who needs constitutional enforcement guarantees.

**US-1 (P0):** As a subsystem developer, I want to register my process with the
kernel and declare my required capabilities so that the kernel knows what my
service is permitted to do and can enforce boundaries.

**US-2 (P0):** As a subsystem developer, I want to send an authorization request
before executing any tool call so that every action in my service is
constitutionally validated before it happens.

**US-3 (P0):** As a subsystem developer, I want to bind evidence to every claim
my service produces so that users receive only source-backed information and I
can prove my service operates within CLAIM_MUST_BIND.

**US-4 (P0):** As a subsystem developer, I want to submit user-facing outputs
to the IHSAN gate so that no output below the ethical quality threshold
reaches the user.

**US-5 (P1):** As a subsystem developer, I want to request temporary capability
escalation with a TTL so that my service can perform authorized one-time
operations without permanently expanding its permission set.

### Secondary Persona: BIZRA System Operator
The person deploying, monitoring, and maintaining a BIZRA node.

**US-6 (P0):** As a system operator, I want the kernel to produce a structured
audit log of every authorization decision so that I can verify constitutional
compliance and investigate incidents.

**US-7 (P0):** As a system operator, I want the kernel to automatically quarantine
processes that repeatedly violate invariants so that a misbehaving component
cannot compromise the system.

**US-8 (P1):** As a system operator, I want to review GATE_HOLD outputs and
either approve or reject them so that outputs the kernel is uncertain about
receive human judgment before reaching the user.

**US-9 (P1):** As a system operator, I want the kernel boot sequence to self-verify
binary integrity, config validity, and key integrity so that I can trust the
kernel is running unmodified code with correct configuration.
### Tertiary Persona: BIZRA End User
The person using BIZRA through a service like JARVIS. They do not interact with
the kernel directly but benefit from its enforcement.

**US-10 (P0):** As an end user, I want every response I receive to include a
confidence indicator and source references so that I can assess the reliability
of information presented to me.

**US-11 (P1):** As an end user, I want assurance that BIZRA will never facilitate
interest-bearing transactions on my behalf so that I can trust the system
aligns with my ethical requirements.

---

## 5. Requirements

### Must-Have (P0) — Kernel Cannot Ship Without These

**R-001: Process Registration and Capability Granting**
- Supervised processes register via PROCESS_REGISTER message
- Kernel validates requested capabilities against invariants
- Kernel issues ProcessID and capability token on success
- Unregistered processes cannot communicate with the kernel

*Acceptance Criteria:*
- Given a new process sends PROCESS_REGISTER with valid capabilities
- When the kernel receives the registration
- Then the kernel responds with PROCESS_REGISTERED containing ProcessID and token
- And the process appears in the kernel's process registry
---
- Given a process attempts to send AUTH_REQUEST without prior registration
- When the kernel receives the message
- Then the kernel drops the message and logs an unauthorized access attempt

**R-002: Authorization Gate (AUTH_REQUEST → AUTH_GRANTED/DENIED)**
- Every action from a supervised process requires AUTH_REQUEST
- Kernel checks action against process capabilities AND frozen invariants
- AUTH_GRANTED includes a time-limited execution permit
- AUTH_DENIED includes the specific invariant or capability violation

*Acceptance Criteria:*
- Given JARVIS is registered with FileRead("/bizra-data-lake/**") capability
- When JARVIS sends AUTH_REQUEST for FileRead("/bizra-data-lake/docs/readme.md")
- Then the kernel responds AUTH_GRANTED with an execution permit
---
- Given JARVIS is registered WITHOUT FileWrite capability
- When JARVIS sends AUTH_REQUEST for FileWrite("/bizra-data-lake/config.toml")
- Then the kernel responds AUTH_DENIED with reason "Capability not granted: FileWrite"
**R-003: Frozen Invariant Enforcement**
- All 6 frozen invariants (RIBA_ZERO, CLAIM_MUST_BIND, IHSAN_FLOOR,
  GINI_CEILING, NO_SILENT_ACTION, FAIL_CLOSED) are loaded from config at boot
- Invariants cannot be modified, disabled, or overridden at runtime
- Each invariant has a defined enforcement action (HARD_KILL, SOFT_REJECT,
  GATE_HOLD, TX_BLOCK, DEFAULT_DENY)

*Acceptance Criteria:*
- Given the kernel is running with all 6 invariants loaded
- When a process attempts an action that violates RIBA_ZERO
- Then the kernel terminates the process (HARD_KILL) and logs the violation
---
- Given the kernel config file is edited to remove IHSAN_FLOOR invariant
- When the kernel boots
- Then the kernel PANICs with "Invalid configuration: missing required invariant IHSAN_FLOOR"

**R-004: Evidence Binding Registry**
- Processes submit EvidenceBinding structs for every claim
- Kernel validates: sources.len() ≥ 1, confidence ≥ 0.50, all fields present
- Kernel signs valid bindings (kernel_seal) creating audit trail
- Invalid bindings are rejected with specific reason

*Acceptance Criteria:*
- Given a process submits EVIDENCE_BIND with 2 sources and confidence 0.82
- When the kernel validates the binding
- Then the kernel responds EVIDENCE_ACCEPTED with kernel_seal signature
---
- Given a process submits EVIDENCE_BIND with 0 sources
- When the kernel validates the binding
- Then the kernel responds EVIDENCE_REJECTED with reason "No sources provided"
- And the process receives a strike (SOFT_REJECT)

**R-005: IHSAN Ethical Scoring Gate**
- User-facing outputs must pass through IHSAN scoring before emission
- Kernel delegates scoring to registered EthicalScorer process
- Kernel validates returned score structure and enforces threshold
- Outputs scoring ≥ 0.95 pass; 0.80–0.94 are held; < 0.80 are rejected

*Acceptance Criteria:*
- Given a process submits IHSAN_SCORE_REQUEST for a user-facing output
- When the EthicalScorer returns composite score 0.97
- Then the kernel responds IHSAN_PASS and the output is cleared for emission
---
- Given the EthicalScorer returns composite score 0.91
- When the kernel evaluates the score
- Then the kernel responds IHSAN_HOLD with the score breakdown
- And a human review notification is generated
**R-006: Process Kill Authority**
- Kernel is sole authority for process termination
- Supports HARD_KILL, GRACEFUL_STOP, QUARANTINE, RESTART
- Strike system: 3 SOFT_REJECT violations → automatic QUARANTINE
- Heartbeat monitoring: 3 missed → QUARANTINE, 5 missed → HARD_KILL

*Acceptance Criteria:*
- Given a process has accumulated 3 strikes from CLAIM_MUST_BIND violations
- When the 3rd strike is recorded
- Then the kernel issues QUARANTINE_ORDER to the process
- And the process is isolated from all IPC except kernel communication
---
- Given a process has not sent a heartbeat for 15 seconds (3× 5s interval)
- When the kernel's heartbeat monitor fires
- Then the kernel issues QUARANTINE_ORDER and logs "heartbeat timeout"

**R-007: Audit Log (NO_SILENT_ACTION Enforcement)**
- Every kernel decision produces an append-only JSONL audit record
- If the audit log cannot be written, the kernel halts the action (not itself)
- Violations include full message payload; normal operations include metadata only
- Audit log rotates daily, retains 90 days

*Acceptance Criteria:*
- Given the kernel processes an AUTH_REQUEST
- When the decision is made (GRANTED or DENIED)
- Then a JSONL record is appended with: timestamp, msg_id, source_pid, decision, reason
---
- Given the audit log file is locked by another process
- When the kernel attempts to write a record
- Then the kernel blocks the pending action until the audit write succeeds
- And the action is never executed without its audit receipt

**R-008: Deterministic Boot Sequence**
- 5-phase boot: SELF-CHECK → CONFIG LOAD → IDENTITY INIT → SOCKET BIND → AUDIT INIT
- Any phase failure → PANIC (complete halt, no recovery attempt)
- Boot produces a startup audit record with config hash and identity fingerprint

*Acceptance Criteria:*
- Given the kernel binary has been tampered with (hash mismatch)
- When the kernel starts PHASE 0: SELF-CHECK
- Then the kernel PANICs with "Kernel integrity compromised" and exits code 1
---
- Given all 5 boot phases succeed
- When the kernel reaches PHASE 5: READY
- Then the kernel writes a boot record to audit log and begins accepting connections
### Nice-to-Have (P1) — Significant Improvement, Not Blocking v1

**R-009: Temporary Capability Escalation**
- Processes can request additional capabilities with configurable TTL
- Kernel evaluates against invariants before granting
- Escalated capabilities auto-expire, with audit record

**R-010: Human Override Interface for GATE_HOLD**
- Held outputs queue for human review via a simple CLI tool
- Operator can approve (release output), reject (block), or revise
- All decisions are audit-logged

**R-011: Process Crash Recovery (RESTART)**
- Kernel can restart a crashed process from last known-good state
- Requires process to have registered a restart command
- Restart count is tracked; excessive restarts → permanent QUARANTINE

**R-012: Resource Budget Enforcement**
- Per-process CPU time, memory, and IPC rate limits
- Kernel monitors resource usage via OS-level counters
- Budget exceeded → GRACEFUL_STOP with resource exhaustion reason

### Future Considerations (P2) — Design For, Don't Build Yet

**R-013: Multi-Node Attestation**
- When BIZRA network exists, kernels on different nodes will need to
  verify each other's attestations. Current identity system (Ed25519)
  supports this. Socket protocol does not — will need network transport.
- **Design implication:** Keep Attestation struct network-transportable.

**R-014: Invariant Hot-Reload for Non-Frozen Parameters**
- Frozen anchors never change. But operational parameters (heartbeat_interval,
  resource budgets) may need tuning without restart.
- **Design implication:** Separate frozen invariants from operational config
  in the TOML schema (already done in spec).

**R-015: TLA+ Formal Verification**
- 8 formal properties defined in kernel spec. Future cycle should produce
  actual TLA+ model and run model checker.
- **Design implication:** Keep state transitions explicit and enumerable.

---

## 6. Success Metrics
### Leading Indicators (Days to Weeks)

| Metric | Success Threshold | Stretch Target | Measurement |
|--------|------------------|----------------|-------------|
| Authorization coverage | 100% of JARVIS actions pass through kernel | — | `audit_log.count(AUTH_GRANTED + AUTH_DENIED) / jarvis_actions.count()` |
| Evidence binding rate | ≥ 95% of claims have valid bindings | 99% | `kernel.evidence_accepted / kernel.evidence_submitted` |
| IHSAN gate enforcement | 0 outputs emitted below 0.95 threshold | — | `audit_log.count(IHSAN_REJECT bypassed)` = 0 |
| Authorization latency p99 | < 1ms | < 0.5ms | Kernel internal timer on AUTH_REQUEST → response |
| Boot reliability | 100% successful boots on valid config | — | `successful_boots / boot_attempts` |
| Fuzz test pass rate | 0 invariant violations in 100K iterations | 0 in 1M iterations | cargo-fuzz campaign report |

### Lagging Indicators (Weeks to Months)

| Metric | Success Threshold | Evaluation Window | Measurement |
|--------|------------------|-------------------|-------------|
| Kernel uptime | 99.9% (< 8.7 hrs downtime/year) | 90 days post-launch | `kernel_uptime_seconds / total_seconds` |
| JARVIS under kernel supervision | 7 consecutive days, zero bypasses | First 30 days | Audit log reconciliation: no unmatched JARVIS actions |
| Developer adoption | All new BIZRA services register with kernel by default | 60 days | Process registry count vs. known service count |
| Invariant violation rate | < 0.1% of requests trigger any invariant | 90 days | `violation_count / total_auth_requests` |
| Constitutional trust score | Operator rates system trust ≥ 8/10 in monthly review | Monthly | Operator survey (N=1 initially, scaling with team) |

### Evaluation Schedule
- **Week 1 post-integration:** Authorization coverage, boot reliability, latency
- **Week 4:** Evidence binding rate, IHSAN enforcement, fuzz results
- **Week 12:** Uptime, violation rate, developer adoption, trust score

---

## 7. Open Questions

| # | Question | Owner | Blocking? |
|---|----------|-------|-----------|
| Q1 | Should the IHSAN scorer be a separate binary or a library linked into a scorer process? | Engineering | No — kernel only cares about the IPC interface |
| Q2 | What is the RIBA detection strategy for obfuscated interest transactions (labeled as "fees", "service charges")? | Architecture + Islamic Finance | Yes — needed before RIBA_ZERO fuzz testing |
| Q3 | How should GATE_HOLD outputs be presented to the operator? CLI? Web dashboard? Notification? | Product + Design | No — P1 requirement, can ship v1 with CLI only |
| Q4 | What is the key rotation strategy for the Ed25519 identity keypair? | Security | No — not needed for v1 single-node deployment |
| Q5 | Should the kernel support Windows Named Pipes in v1 or is Unix socket (via WSL) sufficient? | Engineering | Yes — determines dev environment requirements |
| Q6 | What is the maximum acceptable audit log size before rotation? | Ops | No — default 90-day rotation is reasonable for v1 |
---

## 8. Timeline and Phasing

### Phase A: Kernel Skeleton (Weeks 1–2)
**Deliverables:** Boot sequence, config parsing, socket bind, audit log,
process registration, heartbeat monitoring.
**Gate:** Kernel boots deterministically, accepts process registrations,
writes audit log. All boot-phase PANIC scenarios tested.

### Phase B: Authorization Engine (Weeks 2–3)
**Deliverables:** AUTH_REQUEST/GRANTED/DENIED flow, capability validation,
frozen invariant checking (all 6).
**Gate:** JARVIS can register and have actions authorized/denied. Property
tests pass for all 6 invariants.

### Phase C: Evidence Binding (Weeks 3–4)
**Deliverables:** EvidenceBinding struct, validation logic, kernel seal,
confidence scoring heuristic.
**Gate:** Claims with valid evidence are sealed. Claims without evidence are
rejected. Fuzz campaign: 100K iterations, zero violations.

### Phase D: IHSAN Gate + Kill Authority (Weeks 4–5)
**Deliverables:** Ethical scoring delegation, threshold enforcement, GATE_HOLD
flow, strike system, QUARANTINE logic.
**Gate:** Outputs below 0.95 are held. 3 strikes → quarantine. Heartbeat
timeout → quarantine.

### Phase E: JARVIS Integration (Weeks 5–6)
**Deliverables:** KernelClient shim in JARVIS, end-to-end flow (user request →
JARVIS → kernel auth → execute → evidence bind → IHSAN gate → user response).
**Gate:** JARVIS operates under full kernel supervision for 48+ hours with
zero bypass incidents.

### Hard Dependencies
- Rust toolchain (stable channel, no nightly features)
- tokio async runtime (for event loop)
- ed25519-dalek (for cryptographic identity)
- rmp-serde (for MessagePack serialization)
- blake3 (for content hashing)
- No external services required at runtime (by design)

---

## 9. Risks and Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Authorization latency exceeds 1ms p99 under load | HIGH | LOW | Single-threaded event loop eliminates lock contention. MessagePack parsing is sub-microsecond. Benchmark at each phase gate. |
| RIBA_ZERO false positives block legitimate transactions | MEDIUM | MEDIUM | Start with conservative semantic matching. Log false positives in first 30 days. Tune detection rules with Islamic finance advisor. |
| IHSAN scorer becomes a bottleneck (5s timeout) | MEDIUM | MEDIUM | Timeout → reject (fail-closed). Optimize scorer independently. Kernel latency is not affected by scorer latency. |
| Developer friction from authorization overhead | MEDIUM | HIGH | Provide KernelClient SDK with clean API. Make auth transparent for common patterns. Measure developer time-to-integration. |
| Audit log disk exhaustion | LOW | LOW | Daily rotation, 90-day retention, metadata-only by default. Alert at 80% disk usage. |
---

## 10. Definition of Done — Kernel v1.0

The kernel achieves v1.0 status when ALL of the following are true:

- [ ] All P0 requirements (R-001 through R-008) implemented and tested
- [ ] All 6 frozen invariants pass property-based testing (10,000+ inputs each)
- [ ] Fuzz campaign: 100,000 iterations across 6 targets, zero violations
- [ ] Authorization latency < 1ms p99 under simulated load (100 concurrent processes)
- [ ] Boot sequence passes all 5 PANIC scenarios
- [ ] Audit log captures 100% of kernel decisions (verified by reconciliation)
- [ ] JARVIS operates under kernel supervision for 7 consecutive days
- [ ] Binary size < 5 MB, RSS < 50 MB
- [ ] Zero external network calls during operation (verified by strace/dtrace)
- [ ] Operator can review GATE_HOLD outputs via CLI tool

**Canonical Gate:** After v1.0 DoD is met, BIZRA_KERNEL_SPEC.md transitions
from DRAFT → PROVEN in TOPOLOGY_CANON.md. After 30 days of stable operation,
it transitions from PROVEN → CANONICAL.

---

## Appendix: Traceability Matrix

| Requirement | User Story | Invariant | Spec Section | Test Strategy |
|-------------|-----------|-----------|--------------|---------------|
| R-001 | US-1 | FAIL_CLOSED | §2.5, §3.3 | Integration test: register/reject unregistered |
| R-002 | US-2 | All 6 | §2.2, §3.2 | Property test per invariant + integration |
| R-003 | US-11 | RIBA_ZERO, CLAIM_MUST_BIND, IHSAN_FLOOR, GINI_CEILING, NO_SILENT_ACTION, FAIL_CLOSED | §2.2 | Fuzz testing: 100K iterations per invariant |
| R-004 | US-3, US-10 | CLAIM_MUST_BIND | §2.3 | Unit test: valid/invalid bindings + fuzz |
| R-005 | US-4, US-10 | IHSAN_FLOOR | §2.4 | Boundary test: 0.94/0.95/0.96 scores |
| R-006 | US-7 | NO_SILENT_ACTION | §2.5 | Chaos test: kill processes, verify recovery |
| R-007 | US-6 | NO_SILENT_ACTION | §4, §5 | Fault injection: lock audit file, verify halt |
| R-008 | US-9 | All | §5 | Boot test: corrupt binary, bad config, bad key |

---

*PRD Complete — bizra-kernel v1.0*
*Constitutional authority: BIZRA_KERNEL_SPEC.md → SYSTEM_INSTRUCTION_CHAIN.md → DECLARATION.md*
*Next action: Phase A implementation (Weeks 1–2)*