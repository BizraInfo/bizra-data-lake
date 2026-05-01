# DEMA Identity Model v0.1

**Date:** 2026-05-01 GST
**Status:** SPECIFIED
**Scope:** Product/architecture doctrine for DEMA identity, inheritance, and
memory boundaries before Node0 Origin Memory Ingestion v0.1.
**Truth label:** DERIVED from implemented Dema profile/consent surfaces and
Node0 product doctrine; PLANNED for the next memory-ingestion implementation
slice.

---

## §1 SPARC specification

### Problem

Dema must be both personal and reusable without confusing the two. Node0's
first Dema instance carries origin context, but future Dema instances must not
inherit Mumu's private memory as if it were their own.

### Required model

| Name | Layer | Purpose | Privacy rule |
|---|---|---|---|
| **DEMA Core** | universal architecture | Shared archetype, proof rules, agent patterns, safety behavior, and base UX | Shared across nodes; contains no private human memory |
| **DEMA-0** | first personal instance | Genesis personal Dema rooted in Node0 | May hold Mumu-private memory locally |
| **Mumu-DEMA** | human-facing name | Relational name for DEMA-0 | Same privacy boundary as DEMA-0 |
| **Node0-DEMA** | runtime name | Technical name for DEMA-0 in Node0 systems | Same privacy boundary as DEMA-0 |
| **Genesis Dema** | historical name | Origin/reference implementation label | Historical/ceremonial, not a separate memory scope |
| **DEMA-1..n** | future personal instances | One personal Dema per future node | Fresh private memory for that node only |

### Non-negotiable rule

Future Dema instances inherit BIZRA DNA. They do **not** inherit Mumu's private
memory as their own.

---

## §2 SPARC pseudocode

Node0 Origin Memory Ingestion v0.1 must classify every candidate artifact
before promotion:

```text
for artifact in origin_import_batch:
    raw = read_only_scan(artifact)
    classification = classify(raw)

    if classification == MUMU_PRIVATE_MEMORY:
        store_only_under_node0_private_memory()
        require_explicit_consent_for_any_future_export()

    if classification == BIZRA_CANON:
        distill_to_non_private_doctrine()
        attach_source_receipt_and_truth_label()
        promote_to_bizra_dna_pack_candidate()

    if classification == URP_SHAREABLE_KNOWLEDGE:
        remove_or_scope_private_identifiers()
        transform_into_reusable_skill_or_workflow()
        attach_consent_receipt_access_policy_and_proof()
        promote_to_urp_seed_pack_candidate()

    if classification == UNVERIFIED_OR_AMBIGUOUS:
        keep_local_candidate_only()
        mark truth_label = UNKNOWN or PLANNED
        block sharing until review
```

No branch in this pseudocode may write raw private chat history, private files,
or private emotional context into shared URP storage.

---

## §3 SPARC architecture

### Three memory/knowledge layers

```text
1. DEMA Core DNA
   Shared architecture: constitution, proof discipline, reasoning protocols,
   agent behavior patterns, safety rules, UX behavior, and base capabilities.

2. Shared BIZRA URP Knowledge / Skill Layer
   Proof-checked reusable knowledge: public docs, non-private workflows,
   reusable coding patterns, verified skills, evaluation standards, and
   Proof-of-Impact contribution records.

3. Personal Node Memory
   Private profile, preferences, tasks, local files, emotional context, and
   human-specific memory for exactly one node.
```

### Node0 special case

Node0 is the origin seed, not the generic template for every person's private
life. DEMA-0 may learn:

- Mumu's three-year BIZRA journey.
- BIZRA origin documents and codebase patterns.
- Visual experiments, prompts, failures, breakthroughs, and GTM material.
- Agent patterns, proof discipline, and emotional intent.

But DEMA-0 must distill these into shareable outputs only through explicit
classification:

```text
Mumu Private Memory      -> local Node0 memory only
BIZRA Canon              -> non-private doctrine / architecture
BIZRA Skills             -> reusable workflows and tools
BIZRA Proof Patterns     -> proof/receipt/evaluation templates
URP Shareable Knowledge  -> consented, transformed, access-scoped seed pack
```

### Future-node inheritance

When Node1 or any later node joins, it may receive:

| Inherited from shared layers | Created fresh for the new node |
|---|---|
| DEMA Core behavior | New user profile |
| BIZRA constitution and proof protocols | New private memories |
| Safe action patterns and agent workflows | New local files and preferences |
| Verified reusable skills | New emotional context |
| Public knowledge packs | New private tasks |
| URP capabilities and evaluation standards | New consent rules |

This is the core sovereignty invariant: each future Dema starts stronger
because the forest has learned, but starts private because the human is new.

---

## §4 SPARC refinement gates

Node0 Origin Memory Ingestion v0.1 must not start until these gates exist:

1. **Classification gate** — every import candidate is labeled as
   `MUMU_PRIVATE_MEMORY`, `BIZRA_CANON`, `URP_SHAREABLE_KNOWLEDGE`, or
   `UNVERIFIED_OR_AMBIGUOUS`.
2. **Consent gate** — no private memory is shared, anonymized, transformed, or
   exported without an explicit consent receipt.
3. **Transformation gate** — shared URP candidates must be transformed into a
   reusable skill, workflow, doctrine, proof pattern, or knowledge pack.
4. **Redaction/scope gate** — private identifiers are removed or access-scoped
   before anything leaves Node0-private memory.
5. **Receipt gate** — every promotion candidate carries source, truth label,
   access policy, and approval status.
6. **Dashboard gate** — the operator can see what DEMA-0 knows about Mumu, what
   it knows about BIZRA, what is private, what is shareable, and what remains
   unverified.

---

## §5 SPARC completion checklist

Before claiming DEMA Identity Model v0.1 is implemented in code:

- [ ] `DemaProfile` or its successor records instance identity without storing
      secrets.
- [ ] Memory ingestion outputs separate directories or stores for private,
      canon, URP-shareable, and ambiguous candidates.
- [ ] Tests prove future-node bootstrap never imports Node0-private records by
      default.
- [ ] Tests prove URP promotion requires consent, transformation, truth label,
      and access policy.
- [ ] The Node0 UI exposes the memory dashboard categories without rendering
      raw private content in public surfaces.
- [ ] Receipts make promotion decisions replayable.

---

## §6 Naming convention

Use these names consistently:

```text
DEMA Core       = universal Dema architecture / species DNA
DEMA-0          = Node0 personal Dema instance
Mumu-DEMA       = human-facing name for DEMA-0
Node0-DEMA      = technical/runtime name for DEMA-0
Genesis Dema    = historical/origin name for DEMA-0
DEMA-1..DEMA-n  = future node-local personal Dema instances
```

Do not use "Dema" alone in architecture docs when the difference between Core,
DEMA-0, or a future personal instance materially affects privacy, consent, or
inheritance.

---

## §7 URP sharing doctrine

URP must not become a dumping ground for private memory.

Personal memory stays local by default. Shared knowledge enters URP only if:

1. the user consents,
2. private identifiers are removed or explicitly access-scoped,
3. value is transformed into reusable knowledge, skill, workflow, or proof
   pattern,
4. proof/receipt is attached,
5. access policy is clear.

Short form:

```text
Private Node Memory != Shared URP Knowledge.
Private memory personalizes DEMA.
URP strengthens the forest.
```

---

## §8 Bounds

This document does not implement memory ingestion, daemon start, Node1
federation, or public launch. It is a prerequisite doctrine and acceptance
contract for the next bounded memory slice.

If any clause conflicts with the BIZRA Topology Canon, Brand Canon, or
constitutional thresholds in `core/integration/constants.py`, those canonical
sources win and this document must be amended.
