# BIZRA TOPOLOGY CANON

**Frozen:** 25 March 2026
**Author:** Mohamed Beshr, BIZRA Foundation
**Rule:** No AI session, no document, no diagram may contradict this file. If a conflict exists, this file wins.

---

## The one sentence

Each human node mints PAT-7 locally on their device and SAT-5 into one shared Universal Resource Pool. PAT serves the human. SAT serves the system. The membrane sits between them.

---

## What is local (per human)

**PAT-7 — Personal Agentic Team**

Seven agents minted on the human's own device at first activation:

- P1 Planner
- P2 Researcher
- P3 Coder
- P4 Evaluator
- P5 Ethicist (FROZEN — ethics from axioms, not data)
- P6 Publisher
- P7 DEMA / Nexus (the face — human talks to DEMA only)

PAT-7 is user-loyal. Their only purpose is to serve and empower their human. They live on the human's hardware. They are always on. They work locally first. The human never interacts with the network — only with PAT.

Also local: the human's devices, local data lake, local models, local FAISS index, local receipt chain (copy), local reflex cache.

---

## What is shared (one for the entire ecosystem)

**BIZRA Universal Resource Pool (URP) — one, singular, shared**

The URP is the system. Not a layer. Not middleware. Not per-user. One shared living organism for the entire BIZRA ecosystem.

Before any human joins, the URP is dormant — code with no power, no agents, no resources.

When the first human (Node0) activates:
- System mints PAT-7 on their device (local)
- System mints SAT-5 into the URP (shared)
- The URP wakes up with 5 employees and whatever resources Node0 contributes

Each subsequent node adds 5 more SAT agents to the shared URP, plus contributed resources.

**SAT-5 — System Agentic Team (per node, but lives in the URP)**

- S1 Validator — verifies receipts and proof integrity
- S2 Oracle (FROZEN — truth axioms, immutable)
- S3 Mediator — fair dispute resolution
- S4 Archivist — archives to House of Wisdom
- S5 Sentinel — threat detection and monitoring

SAT agents follow constitutional law only. No human designs their behavior.

**Also inside the URP:** Constitutional Spine, House of Wisdom, Proof Engine, SEED Treasury, Compute Pool, Storage Pool, Bandwidth Pool, Shared Reflex Registry, Receipt Log.

---

## The membrane

The constitutional membrane sits between every local node and the shared URP.

Four properties:
1. **Fail-closed:** Incomplete verification = reject
2. **Axiomatic filtering:** All constitutional invariants must hold
3. **Cryptographic provenance:** Every crossing produces a BLAKE3-chained, Ed25519-signed receipt
4. **Receipt completeness:** No gaps in the provenance log

**What NEVER crosses:** Human identity, raw private data, unverified claims, untagged information.

---

## The request flow

Human → DEMA (P7) → PAT handles locally if possible
                   → If help needed: PAT → Membrane → SAT (in URP)
                   → SAT manages → Network if needed
                   → Result: SAT → Membrane → PAT → DEMA → Human

The human NEVER touches the network.

---

## Common mistakes (do not repeat)

**WRONG:** "Each user has their own URP." **RIGHT:** There is ONE URP.
**WRONG:** "SAT-5 lives inside each user's local node." **RIGHT:** SAT-5 lives in the shared URP.
**WRONG:** "PAT connects directly to other nodes." **RIGHT:** PAT → Membrane → SAT. No peer-to-peer.
**WRONG:** "The URP is a server that nodes are clients of." **RIGHT:** The URP is a shared organism that grows with every node.

---

## Scaling

| Nodes | Local PAT | SAT in URP | Effect |
|---|---|---|---|
| 1 | 7 | 5 | System alive, flywheel starts |
| 1,000 | 7,000 | 5,000 | Serious governance capacity |
| 1,000,000 | 7M | 5M | Self-securing, self-evolving |
| 8,000,000,000 | 56B | 40B | Planetary intelligence |

---

*This file is the canonical source of truth for BIZRA's topology.*
*If any document contradicts it, this file wins.*
