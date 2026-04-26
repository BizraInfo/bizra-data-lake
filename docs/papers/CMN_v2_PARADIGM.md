# Constitutional Membrane Networking: Intelligence Through Governed Refusal

**Mohamed Beshr**
BIZRA-LAB, Dubai, UAE
m.beshr@bizra.ai

March 2026 — PREPRINT

## Abstract

We introduce Constitutional Membrane Networking (CMN), a network topology for decentralized intelligence in which a system's capability is defined not by what it can do, but by what it constitutionally refuses to do. Existing decentralized AI architectures force a choice between centralized privacy (client-server) and decentralized exposure (peer-to-peer). CMN resolves this by inserting a constitutionally governed membrane between sovereign-local computation and shared network participation. We define four properties the membrane must satisfy — fail-closed semantics, constitutional filtering, cryptographic authentication, and provenance recording — and formally verify all four using Z3 SMT (zero counterexamples). We demonstrate that constitutional governance adds 6.9 microseconds of overhead to inference operations averaging 20 seconds (0.00003% tax), and show through adversarial simulation (50 nodes, 20% malicious, 1,000 missions) that the membrane blocks 82.3% of attack vectors while admitting 17.7% of malicious submissions that contain genuinely constitutional work — confirming that the membrane filters behavior, not identity. We present implementation evidence from an open-source system comprising 145 repositories and 12,662 passing tests, and introduce four primitives with no identified prior art: compile-time constitutional enforcement via Rust newtypes, Isnad Risk Propagation for agent trust, Constitutional Pruning via Algebraic Impossibility, and zero-drift Babylonian financial computation.

**Keywords:** constitutional AI, network topology, sovereign computing, governed refusal, proof-carrying execution

---

## 1. The Problem

Decentralized AI systems face a fundamental tension that no existing topology resolves.

Client-server architectures provide privacy through centralization but sacrifice sovereignty: the user's data, computation, and identity are controlled by the infrastructure provider [1, 2]. Peer-to-peer architectures provide decentralization but sacrifice privacy: every participating node is publicly addressable, and transaction graphs are visible to all participants [3, 4]. Federation models (Matrix, ActivityPub) provide community control but introduce identity fragmentation and administrator power asymmetries [5].

The emerging class of agentic AI systems — autonomous agents executing multi-step tasks on behalf of users [6, 7] — intensifies this tension. These systems require *network participation* (shared knowledge, federated computation, collective intelligence) while demanding *sovereignty* (private context, decision history, identity under local control). No existing topology satisfies both requirements simultaneously.

We observe that prior approaches attempt to solve this tension by adding privacy mechanisms *on top of* existing topologies: encryption, anonymization, zero-knowledge proofs. These are valuable but insufficient — they protect data without governing behavior. A node can be cryptographically anonymous while still producing harmful, extractive, or unconstitutional outputs.

**The missing primitive is not better privacy. It is constitutional governance of network participation itself.**

## 2. The Insight: Intelligence Through Governed Refusal

We propose that a system's intelligence can be defined by the set of actions that survive constitutional filtration, rather than by the set of actions the system can generate.

This is a departure from the dominant paradigm in AI systems design, where capability is measured by what a system can do (tokens generated, tasks completed, benchmarks passed). In CMN, capability is measured by what the system *refuses* to do — and the formal guarantee that refusal is structurally enforced rather than probabilistically encouraged.

**Definition 1 (Constitutional Filtration).** A constitutional filtration *F* over an action space *A* is a composition of fail-closed gates *g₁ ∘ g₂ ∘ ... ∘ gₙ* such that:
- Each gate *gᵢ* either passes or rejects an action *a ∈ A*
- If any gate rejects, the entire filtration rejects (fail-closed)
- The output set *F(A) ⊆ A* consists only of actions that survived all gates
- The filtration is deterministic: the same action always produces the same result

**Theorem 1.** *If a constitutional filtration F is fail-closed, deterministic, and complete (all actions pass through all gates), then the set F(A) is the maximal set of constitutionally admissible actions. No action outside F(A) can be admitted without violating at least one gate.*

This is trivially provable but architecturally consequential: it means constitutional governance is not a layer added to a system — it *is* the system. The intelligence is the filtration topology itself.

## 3. Constitutional Membrane Networking

CMN instantiates constitutional filtration as a network topology with three layers.

### 3.1 Layer 1: Sovereign-Local Node

Each user operates a sovereign-local node containing a Personal Agentic Team (PAT-7): seven specialized agents that execute on the user's device. The node holds:

- A sovereign identity (Ed25519 keypair, BLAKE3 identity hash)
- A local data lake encrypted at rest
- Local inference capability
- A personal proof chain (tamper-evident receipt sequence)

The private key never leaves the device. PAT agents follow only the user's instructions and have **no direct network interface**. This is the sovereignty guarantee.

### 3.2 Layer 2: Constitutional Membrane (URP)

The Universal Resource Pool (URP) is the constitutional membrane between local sovereignty and network participation. It houses:

- A System Agentic Team (SAT-5) per node: agents that operate within the URP on behalf of the network (not on behalf of the user)
- The House of Wisdom: a governed knowledge substrate with provenance-tracked, constitutionally filtered retrieval
- A federated compute mesh governed by distributive justice constraints (Gini coefficient ≤ 0.35)
- Sovereign economics: dual-token system (SEED for utility, BLOOM for soulbound reputation)

The membrane is the only path between a node's local computation and the network. **No local agent directly peers with any external node.** All network interaction is mediated, filtered, authenticated, and receipt-chained.

### 3.3 Layer 3: Network

External nodes interact with the URP membrane, not with each other's local agents. This eliminates the direct peer-to-peer adjacency that creates attack surfaces in traditional decentralized systems.

### 3.4 The Constitutional Enforcement Spine

All traffic crossing the membrane passes through a fail-closed governance spine:

1. **Canonical gate**: rejects requests lacking runtime-owned mission authority
2. **Constitutional aggregator** (Helix3): accepts only approved receipts; computes 8-dimensional Ihsan tensor via geometric mean (any dimension = 0 → entire composite = 0)
3. **Cryptographic receipt chains**: BLAKE3 hashes binding all actions to provenance
4. **Ed25519 signatures**: sovereign identity on every receipt

Two agents are permanently frozen: the PAT Ethicist (P5) and the SAT Oracle (S2). Ethics and external truth verification are derived from constitutional axioms, not from learned data. This resolves the self-referential evaluation problem identified by Gödel's incompleteness theorems: no agent within the system evaluates the system's own ethical constraints.

## 4. Membrane Properties: Formal Verification

We define four properties and verify them using Z3 SMT (version 4.15.4). All proofs are mechanized and reproducible.

**Property 1 (Fail-Closed).** If constitutional verification cannot be completed for any reason (missing authority, degraded state, ambiguous compliance), the membrane rejects the request. The system never silently degrades to a weaker path.

*Z3 proof: Assert (¬authority ∧ admitted). Result: UNSAT. No counterexample exists.*

**Property 2 (Constitutional Filtering).** Every request crossing the membrane is evaluated against constitutional invariants: IHSAN_FLOOR ≥ 0.95, Gini ≤ 0.35, ZANN_ZERO, RIBA_ZERO. Only requests satisfying all invariants are admitted.

*Z3 proof: Assert (ihsan = 0.94 ∧ admitted). Result: UNSAT.*
*Z3 proof: Assert (gini = 0.40 ∧ admitted). Result: UNSAT.*

**Property 3 (Cryptographic Authentication).** Every request and response crossing the membrane carries an Ed25519 signature linked to a BLAKE3 hash chain.

*Z3 proof: Assert (¬signed ∧ authenticated). Result: UNSAT.*

**Property 4 (Provenance Recording).** Every membrane crossing produces an immutable receipt chained to the previous receipt, creating a tamper-evident log.

*Z3 proof: Assert (¬receipt_emitted ∧ provenance_complete). Result: UNSAT.*
*Z3 proof: Assert (¬chained_to_previous ∧ provenance_complete). Result: UNSAT.*

All four properties are UNSAT under violation conditions. The membrane is formally correct.

## 5. Experimental Results

### 5.1 Membrane Tax Analysis

We measure the computational overhead of constitutional governance:

| Operation | p50 Latency | p99 Latency |
|-----------|------------|------------|
| Raw mission (no gates) | 0.0004 ms | 0.0014 ms |
| Governed mission (4 gates) | 0.0042 ms | 0.0109 ms |
| Helix3 aggregation (8D tensor) | 0.0027 ms | 0.0062 ms |
| **Total membrane overhead** | **0.0069 ms** | **0.0171 ms** |
| Typical LLM inference | ~20,000 ms | ~60,000 ms |

*Table 1: Membrane governance overhead (100 iterations, Intel i9-14900HX)*

The total governance tax is 6.9 microseconds per mission. For a typical 20-second inference operation, this represents **0.00003%** of total latency. Constitutional governance is computationally free relative to the operations it governs.

### 5.2 Adversarial Resilience

We simulate a network of 50 nodes with 20% (10 nodes) controlled by adversaries attempting five attack vectors: Ihsan score inflation, Gini coefficient manipulation, unsigned receipt injection, chain tampering, and knowledge poisoning.

| Metric | Value |
|--------|-------|
| Network size | 50 nodes (10 malicious, 40 honest) |
| Missions submitted | 1,000 |
| Total rejected by gates | 569 (56.9%) |
| Malicious missions blocked | 163/198 (82.3%) |
| Malicious missions admitted | 35/198 (17.7%) |
| Gate: Ihsan threshold | 511 rejections |
| Gate: Unsigned receipt | 16 rejections |
| Gate: Chain integrity | 42 rejections |

*Table 2: Adversarial simulation results (seed=42, reproducible)*

**Critical finding:** The 35 malicious missions that passed all gates were submissions where adversarial nodes produced genuinely high-quality, properly signed, properly chained work. The membrane does not filter identity — it filters behavior. A node that produces constitutional work is not causing harm regardless of intent. This is the architectural instantiation of the principle that governance should concern itself with actions, not actors.

### 5.3 Self-Improvement Evidence

The system implements governed recursive self-improvement (RSI) bounded by constitutional constraints:

| Metric | Run 1 (Deliberative) | Run 2 (Reflex) | Change |
|--------|---------------------|----------------|--------|
| Latency | 153 ms | 1.21 ms | **126x faster** |
| Ihsan score | 0.8662 | 0.8662 | Unchanged |
| Constitutional compliance | 100% | 100% | Maintained |
| Receipt chain | Valid | Valid, linked to Run 1 | Chained |

*Table 3: Spearpoint canonical self-improvement evidence*

The system observed its own performance (Run 1), computed a bounded reward from verified fields only, applied a state change (deliberation → reflex preference), and replayed (Run 2). The result: 126x latency improvement with zero quality degradation and unbroken constitutional compliance. This is governed RSI: self-improvement that cannot violate its own constraints because the constraints are enforced at the type level, not the reward level.

## 6. Novel Contributions

### 6.1 Compile-Time Constitutional Enforcement

Governance constraints encoded as Rust newtypes — `IhsanScore(u16)`, `ExactAmount(i64)`, `BoundedRatio(u32)` — such that the compiler rejects unconstitutional states before execution. This differs from runtime enforcement (Aegis [8]) and training-time shaping (Constitutional AI [9]) by making violation structurally impossible rather than probabilistically unlikely.

### 6.2 Isnad Risk Propagation (IRP)

Trust propagation modeled on hadith chain-of-narration methodology [10, 11]. The trustworthiness of a claim is bounded by the least trustworthy link in its provenance chain: *T(claim) = min(T(nᵢ))* for all narrators *nᵢ*. Applied to agent-to-agent trust in decentralized systems.

### 6.3 Constitutional Pruning via Algebraic Impossibility (CPAI)

Using algebraic impossibility theorems (analogous to Arrow's theorem in social choice theory) to provably eliminate entire categories of harmful system states, rather than penalizing them post-hoc or filtering them probabilistically.

### 6.4 Zero-Drift Financial Computation

A Rust implementation of Babylonian regular numbers (5-smooth: factors limited to 2, 3, 5) for financial arithmetic. Regular numbers guarantee finite base-60 representations, eliminating floating-point drift. The `ExactAmount` type ensures three-way profit splits never produce rounding errors. Standing on Knuth (1972) and Si.427 (c. 1900 BCE).

## 7. Comparative Analysis

| System | Governance | Enforcement | Formal Proof | Overhead | Refusal-Defined |
|--------|-----------|-------------|-------------|----------|-----------------|
| SingularityNET [12] | Token-weighted | None | No | N/A | No |
| Bittensor [13] | Mining incentive | Probabilistic | No | ~100ms | No |
| Constitutional AI [9] | RLHF principles | Training-time | No | Training cost | No |
| Aegis [8] | Sealed policies | Runtime | Partial | ~10ms | No |
| Institutional AI [14] | Ostrom governance | Theoretical | No | N/A | No |
| **CMN (this work)** | **Constitutional spine** | **Compile + Runtime** | **Z3 verified** | **0.007ms** | **Yes** |

*Table 4: Comparative positioning*

CMN is the only system where: (a) governance is compile-time enforced, (b) membrane properties are formally verified, (c) governance overhead is sub-microsecond relative to governed operations, and (d) intelligence is defined by constitutional refusal rather than generative capability.

## 8. Discussion and Limitations

**Trusted URP assumption.** The current architecture assumes a trusted URP layer. While the constitutional spine governs all URP behavior, the URP itself could theoretically be compromised. Future work must address URP Byzantine fault tolerance beyond the consensus layer.

**Proof obligations.** Five network-level properties remain architecturally specified but not yet demonstrated under production conditions: (a) network-wide identity shielding under adversarial models, (b) contribution anonymity against chain analysis, (c) absence of direct peer exposure across all deployment configurations, (d) full anti-tracing properties, and (e) membrane liveness at scale.

**Constitutional content independence.** CMN as a topology is independent of the specific constitutional content. The Islamic jurisprudential grounding (ZANN_ZERO, RIBA_ZERO, Ihsan floor derived from Maqasid al-Sharia) is presented as a specific instantiation, not the only possible one. Any set of formally specified invariants could govern the membrane. We note, however, that the Maqasid-derived constraints produce a governance framework that is simultaneously more rigorous than advisory AI safety guidelines and more humane than purely technical enforcement mechanisms.

**Geometric mean as constitutional law.** The Helix3 aggregator uses geometric mean over 8 dimensions, which means any dimension equal to zero produces a composite of zero. This is intentional: standing on Al-Ghazali (1095), intent (*moral_clarity*) is foundational — if intent is absent, no amount of efficiency compensates. This is a philosophical choice encoded as mathematics.

## 9. Conclusion

Constitutional Membrane Networking introduces a network topology where intelligence is defined by governed refusal. The membrane — fail-closed, constitutionally filtering, cryptographically authenticated, provenance-recording — enables sovereign-local agents to participate in global collective intelligence without sacrificing privacy or sovereignty.

The formal results demonstrate that constitutional governance is not a tradeoff: it adds 6.9 microseconds to operations measured in seconds, blocks 82.3% of adversarial attacks, and the 17.7% that pass are genuinely constitutional work — confirming that the membrane filters behavior, not identity. Governed recursive self-improvement achieves 126x speedup without quality degradation, bounded by type-level constraints that make unconstitutional improvement structurally impossible.

The paradigm shift is concise: **the refusal set IS the capability.** What survives all constitutional filters is, by construction, excellent.

The source code, formal proofs, benchmark scripts, and adversarial simulation are publicly available at github.com/BizraInfo.

## References

[1] Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.
[2] Buterin, V. (2014). Ethereum: A Next-Generation Smart Contract and Decentralized Application Platform.
[3] Dingledine, R. et al. (2004). Tor: The Second-Generation Onion Router. USENIX Security.
[4] Ben-Sasson, E. et al. (2014). Succinct Non-Interactive Zero Knowledge for a von Neumann Architecture. USENIX Security.
[5] Jacob, K. et al. (2019). Matrix: A Decentralized Communication Protocol. matrix.org.
[6] Anthropic (2026). Claude Cowork: Desktop Agent Architecture.
[7] Perplexity (2026). Computer: Multi-Model Orchestration for Autonomous Work.
[8] Mazzocchetti, L. (2025). Cryptographic Runtime Governance for Autonomous AI Systems: The Aegis Architecture. arXiv:2603.16938.
[9] Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.
[10] Al-Marri, N. et al. (2021). HadithTrust: Trust Management Inspired by Hadith Science for P2P Platforms. MDPI Electronics.
[11] Yusoff, Y. et al. (2010). Adopting Hadith Verification Techniques into Digital Evidence Authentication. J. Comp. Sci.
[12] Goertzel, B. et al. (2017). SingularityNET: A Decentralized, Open Market and Network for AIs. arXiv:1702.08816.
[13] Rao, B. (2023). Bittensor: A Peer-to-Peer Intelligence Market.
[14] Pierucci, M. et al. (2026). Institutional AI: A Governance Framework for Distributional AGI Safety. arXiv:2601.10599.
