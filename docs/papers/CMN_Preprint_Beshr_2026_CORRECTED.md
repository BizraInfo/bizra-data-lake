# Constitutional Membrane Networking: A Governed Topology for Sovereign Decentralized Intelligence

**Mohamed Beshr**
BIZRA-LAB, Dubai, UAE
m.beshr@bizra.ai

March 2026 — PREPRINT (Not Peer-Reviewed)

## Abstract

We introduce Constitutional Membrane Networking (CMN), a network topology for decentralized intelligence systems in which sovereign-local agents do not participate through direct peer exposure, but through a constitutionally governed intermediary layer that canonicalizes missions, mediates execution, and binds outcomes to cryptographically chained receipts. CMN addresses a fundamental tension in distributed systems: decentralization requires network participation, but participation through direct peer adjacency creates identity exposure, transaction traceability, and expanded attack surfaces. Existing topologies force a choice between centralized privacy (client-server) and decentralized exposure (peer-to-peer). CMN introduces a third option: decentralized trust with constitutionally mediated exposure. We describe the architecture through three layers — sovereign-local nodes with personal agentic teams (PAT-7), a Universal Resource Pool (URP) housing shared system agents (SAT-5), and a constitutional enforcement spine with fail-closed semantics — and define four properties that the membrane must satisfy: it fails closed, filters constitutionally, authenticates cryptographically, and records with provenance. We present partial implementation evidence from the BIZRA system (145 repositories, 12,662 passing tests, BLAKE3 receipt chains, Ed25519 identity binding) and identify the proof obligations that remain for full network-level validation.

**Keywords:** constitutional AI, network topology, sovereign computing, decentralized intelligence, proof-carrying execution, Islamic computational governance

## 1. Introduction

The deployment of autonomous AI agents on distributed infrastructure has created a network topology problem that existing paradigms do not adequately address. Client-server architectures provide privacy through centralization but sacrifice sovereignty: the user's data, computation, and identity are controlled by the infrastructure provider. Peer-to-peer architectures provide decentralization but sacrifice privacy: every participating node is publicly addressable, and transaction graphs are visible to all participants [1, 2]. Federation models (Matrix, ActivityPub) provide community control but introduce identity fragmentation and administrator power asymmetries [3].

The emerging class of agentic AI systems — autonomous agents that execute multi-step tasks on behalf of users [4, 5] — intensifies this tension. These systems require network participation (to access shared knowledge, federated computation, and collective intelligence) while simultaneously demanding sovereignty (the user's private context, decision history, and identity must remain under local control). No existing topology satisfies both requirements.

We propose Constitutional Membrane Networking (CMN): a topology that resolves this tension by inserting a constitutionally governed intermediary — the membrane — between sovereign-local computation and shared network participation. The membrane ensures that no local agent directly peers with any external node. All network interaction is mediated, filtered, authenticated, and receipt-chained by the constitutional enforcement layer.

CMN is motivated by three principles drawn from Islamic jurisprudence (Maqasid al-Sharia): the elimination of unverified claims (zann, formalized as ZANN_ZERO), the prohibition of extractive economics (riba, formalized as RIBA_ZERO), and the requirement of excellence in all actions (ihsan, formalized as a minimum quality threshold). These principles, when implemented as compile-time type constraints and runtime governance gates, produce a network topology with properties that differ qualitatively from existing approaches.

## 2. Related Work

### 2.1 Decentralized AI Networks

SingularityNET [6] provides an AI service marketplace with token-weighted governance but does not enforce behavioral constraints on agents. Bittensor [7] incentivizes distributed inference through a mining protocol but exposes all validators to direct peer interaction. The ASI Alliance (Fetch.ai, Ocean Protocol) [8] provides data marketplace primitives without constitutional enforcement. None of these systems mediate peer interaction through a governance layer.

### 2.2 Constitutional AI

Anthropic's Constitutional AI [9] uses training-time behavioral shaping through RLHF/RLAIF with a set of principles. This is probabilistic and model-specific: the constraints can be circumvented by sufficiently capable models and apply to single models rather than multi-agent systems. The Aegis Architecture [10] proposes cryptographically sealed ethics policies for autonomous systems with runtime enforcement, but operates on centralized single-agent deployments. The Institutional AI framework [11] proposes governance-graphs based on Ostrom's institutional theory but remains theoretical.

### 2.3 Sovereign AI and Network Privacy

NVIDIA's sovereign AI initiative [12] focuses on hardware infrastructure sovereignty (who owns the GPUs, where data resides) but says nothing about how AI systems are governed or how network participation is mediated. Tor and onion routing [13] provide anonymity at the transport layer but do not govern the content or constitutionality of what passes through the network. Zero-knowledge proofs [14] enable verification without disclosure but do not provide a governance framework for agent behavior.

### 2.4 Hadith Science as Computational Trust

The isnad (chain of narration) methodology in hadith science has been applied to computational trust. HadithTrust [15] maps hadith authentication to P2P trust management. Yusoff et al. [16] proposed weakest-link isnad trust formulas for digital forensics. CMN extends this tradition by applying isnad-inspired provenance chains to all network interactions, not just trust evaluation.

## 3. The CMN Architecture

CMN consists of three layers connected by a constitutional enforcement spine. The key architectural invariant is: **no local agent directly peers with any external node.** All network interaction passes through the membrane.

### 3.1 Layer 1: Sovereign-Local Node

Each user operates a sovereign-local node containing a Personal Agentic Team (PAT-7): seven specialized agents (Planner, Researcher, Coder, Evaluator, Ethicist [frozen], Publisher, DEMA/Nexus) — planning, research, creation, evaluation, constitutional ethics, communication, and terminal persona respectively — that execute on the user's device. The node holds: (a) the user's sovereign identity (Ed25519 keypair, BLAKE3 identity hash, with SPHINCS+ post-quantum root key specified for future deployment), (b) a local data lake encrypted at rest, (c) local inference capability, and (d) the user's personal proof chain. The private key never leaves the device. The PAT-7 agents follow only the user's instructions and have no direct network interface.

### 3.2 Layer 2: Constitutional Membrane (URP)

The Universal Resource Pool (URP) is the constitutional membrane between local sovereignty and network participation. It houses: (a) a System Agentic Team (SAT-5) contributed by each node at genesis (5 agents per user: Coordinator, Oracle [frozen], Worker, Curator, Sentinel) — coordination, external truth (frozen), task execution, knowledge curation, and network security respectively, (b) the House of Wisdom — a governed knowledge substrate with provenance-tracked, constitutionally filtered retrieval, (c) a federated compute mesh governed by distributive justice constraints (Gini coefficient <= 0.35), and (d) the sovereign economics layer (dual-token: SEED for utility, BLOOM for soulbound reputation).

### 3.3 Layer 3: Network

The external network consists of other nodes' SAT agents operating within the URP, plus the distributed consensus mechanism (BFT-based). Critically, no node's PAT agents are visible on this layer. External nodes interact with the URP membrane, not with each other's local agents. This eliminates the direct peer-to-peer adjacency that creates attack surfaces in traditional decentralized systems.

### 3.4 The Constitutional Enforcement Spine

All traffic crossing the membrane passes through a fail-closed governance spine consisting of: (a) a canonical gate that rejects requests lacking runtime-owned mission authority, (b) a constitutional aggregator (Helix3) that accepts only approved receipts, (c) BLAKE3 cryptographic receipt chains binding all actions to provenance, and (d) Ed25519 signatures for sovereign identity on every receipt.

Two agents in the system are permanently frozen: the PAT Ethicist (P5) and the SAT Oracle (S2). Ethics and external truth verification are derived from constitutional axioms, not from learned data, resolving the self-referential evaluation problem identified by Godel's incompleteness theorems. No agent evaluates its own ethical constraints — the constraints come from outside the system.

## 4. Membrane Properties

We define four properties that the constitutional membrane must satisfy. A network topology qualifies as CMN if and only if all four hold.

**Property 1 (Fail-Closed).** If constitutional verification cannot be completed for any reason (missing authority, degraded state, ambiguous compliance), the membrane rejects the request. The system never silently degrades to a weaker, unverified path.

**Property 2 (Constitutional Filtering).** Every request crossing the membrane is evaluated against a set of constitutional invariants (in our implementation: ZANN_ZERO, RIBA_ZERO, IHSAN_FLOOR >= 0.95, Gini <= 0.35). Only requests satisfying all invariants are admitted to the network layer.

**Property 3 (Cryptographic Authentication).** Every request and response crossing the membrane carries a cryptographic identity binding (Ed25519 signature) and is linked to a hash chain (BLAKE3) that enables replay verification.

**Property 4 (Provenance Recording).** Every membrane crossing produces an immutable receipt that is chained to the previous receipt, creating a tamper-evident log of all network interactions without exposing the content or identity of the local node.

## 5. Comparative Topology Analysis

| Topology | Trust Model | Privacy | Sovereignty | Governance |
|----------|------------|---------|-------------|------------|
| Client-Server | Centralized | Provider-dependent | None | Provider policy |
| Peer-to-Peer | Distributed | Pseudonymous | Partial | Token-weighted |
| Federation | Community | Admin-dependent | Partial | Admin discretion |
| CMN (proposed) | Constitutional | Membrane-shielded | Full local | Fail-closed spine |

*Table 1: Comparative topology properties*

## 6. Implementation Evidence

CMN is partially implemented in the BIZRA system, a project comprising 145 repositories under the BizraInfo GitHub organization, with the primary codebase made publicly available at publication. The primary codebase (bizra-data-lake) contains approximately 18 MB of Python, 8 MB of Rust, and additional TypeScript, Shell, SMT, and WebAssembly code.

### 6.1 Verified Membrane Properties

| Property | Implementation | Status |
|----------|---------------|--------|
| Fail-closed | core/sovereign/api.py rejects without runtime-owned authority | Verified |
| Constitutional filtering | core/sovereign/helix3.py gates receipts by Ihsan threshold | Verified |
| Cryptographic auth | Ed25519 sign()/verify() in receipt.rs; BLAKE3 canonical.rs (219 LOC, 9 tests) | Verified |
| Provenance recording | Chained receipts with prev_receipt_hash; boot + breath receipts in heartbeat.py | Verified |

*Table 2: Membrane property implementation evidence*

### 6.2 Proof Obligations Not Yet Satisfied

The following properties are architecturally specified but not yet demonstrated under production conditions: (a) network-wide identity shielding under adversarial models, (b) contribution anonymity against chain analysis, (c) absence of direct peer exposure across all deployment configurations, (d) full anti-tracing properties, and (e) membrane liveness and resilience at scale. We identify these as explicit proof obligations for future work rather than implicit assumptions.

## 7. Novel Contributions

Beyond the CMN topology itself, this work introduces several algorithmic and architectural primitives with no identified prior art:

### 7.1 Compile-Time Constitutional Enforcement

Governance constraints are encoded as Rust newtypes (IhsanScore, ExactAmount, BoundedRatio) such that the compiler rejects unconstitutional states before execution. This differs from runtime enforcement (Aegis [10]) and training-time shaping (Constitutional AI [9]) by making violation structurally impossible rather than probabilistically unlikely.

### 7.2 Isnad Risk Propagation (IRP)

Trust propagation modeled on hadith chain-of-narration methodology [15, 16]. The trustworthiness of a claim is bounded by the least trustworthy link in its provenance chain, formalized as T(claim) = min(T(n_i)) for all narrators n_i in the chain. Applied to agent-to-agent trust in decentralized systems.

### 7.3 Constitutional Pruning via Algebraic Impossibility (CPAI)

Using algebraic impossibility theorems (analogous to Arrow's theorem in social choice theory) to provably eliminate entire categories of harmful system states, rather than penalizing them post-hoc or filtering them probabilistically.

### 7.4 Zero-Drift Financial Computation (bizra-sippar)

A Rust crate implementing Babylonian regular numbers (5-smooth numbers: factors limited to 2, 3, 5) for financial arithmetic. Regular numbers guarantee finite base-60 representations, eliminating floating-point drift in economic calculations. Implements ExactAmount, BoundedRatio, and RegularNumber types.

## 8. Discussion and Limitations

CMN is designed to reduce the direct-exposure costs of decentralized participation by replacing raw peer adjacency with constitutional mediation. We deliberately avoid the claim that CMN 'solves' the decentralization-privacy tension, as the full network-level proof remains incomplete. The current evidence demonstrates membrane governance behavior (fail-closed, filtering, authentication, recording) but does not yet prove the full external network theorem under adversarial conditions.

The system assumes a trusted URP layer. While the constitutional spine governs all URP behavior, the URP itself is a shared infrastructure that could theoretically be compromised. Future work must address URP Byzantine fault tolerance beyond the consensus layer, including constitutional spine integrity under partial network compromise.

The Islamic jurisprudential grounding (ZANN_ZERO, RIBA_ZERO, Ihsan floor) is presented as a specific instantiation of constitutional constraints, not as the only possible instantiation. CMN as a topology is independent of the specific constitutional content: any set of formally specified invariants could govern the membrane. However, we note that the specific constraints derived from Maqasid al-Sharia produce a governance framework that is simultaneously more rigorous than advisory AI safety guidelines and more humane than purely technical enforcement mechanisms.

## 9. Conclusion

Constitutional Membrane Networking introduces a fourth network topology option for decentralized intelligence: neither centralized (cloud), nor directly peer-exposed (blockchain), nor federated (community servers), but constitutionally mediated. The membrane — a fail-closed, constitutionally filtering, cryptographically authenticated, provenance-recording intermediary — enables sovereign-local agents to participate in global collective intelligence without sacrificing privacy or sovereignty.

The BIZRA implementation demonstrates that the four membrane properties can be realized in production code (Rust + Python, 145 repositories, 12,662 passing tests). The novel contributions — compile-time constitutional enforcement, Isnad Risk Propagation, Constitutional Pruning via Algebraic Impossibility, and zero-drift Babylonian financial computation — have no identified prior art in the computer science literature. The complete proof set for network-level CMN validation remains future work.

The source code is publicly available at github.com/BizraInfo.

## References

[1] Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.
[2] Buterin, V. (2014). Ethereum: A Next-Generation Smart Contract and Decentralized Application Platform.
[3] Jacob, K. et al. (2019). Matrix: A Decentralized Communication Protocol. matrix.org specification.
[4] Anthropic (2026). Claude Cowork: Desktop Agent Architecture. docs.anthropic.com.
[5] Perplexity (2026). Computer: Multi-Model Orchestration for Autonomous Work. perplexity.ai.
[6] Goertzel, B. et al. (2017). SingularityNET: A Decentralized, Open Market and Network for AIs. arXiv:1702.08816.
[7] Rao, B. (2023). Bittensor: A Peer-to-Peer Intelligence Market. bittensor.com/whitepaper.
[8] ASI Alliance (2024). Artificial Superintelligence Alliance: Merged Protocol Specification.
[9] Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.
[10] Mazzocchetti, L. (2025). Cryptographic Runtime Governance for Autonomous AI Systems: The Aegis Architecture. arXiv:2603.16938.
[11] Pierucci, M. et al. (2026). Institutional AI: A Governance Framework for Distributional AGI Safety. arXiv:2601.10599.
[12] NVIDIA (2024). Sovereign AI: National Transformation with AI Infrastructure. nvidia.com.
[13] Dingledine, R. et al. (2004). Tor: The Second-Generation Onion Router. USENIX Security.
[14] Ben-Sasson, E. et al. (2014). Succinct Non-Interactive Zero Knowledge for a von Neumann Architecture. USENIX Security.
[15] Al-Marri, N. et al. (2021). HadithTrust: Trust Management Approach Inspired by Hadith Science for P2P Platforms. MDPI Electronics.
[16] Yusoff, Y. et al. (2010). Adopting Hadith Verification Techniques into Digital Evidence Authentication. J. Comp. Sci.
