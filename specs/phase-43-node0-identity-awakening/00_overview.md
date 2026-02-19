# Phase 43: Node0 Identity Awakening

Standing on Giants:
- Al-Ghazali (1095): Ihsan — excellence as non-negotiable self-knowledge
- Friston (2010): Active Inference — agents that model themselves
- Deming (1950): "If you can't describe what you're doing as a process, you don't know what you're doing"
- Shannon (1948): The identity IS the signal; everything else is noise

## Problem Statement

Node0 has infrastructure. It has SNR scoring (Phase 42), evidence chains (Phase 41),
GoT reasoning, and a 7-agent PAT team. But the PAT team operates **amnesiac** —
each mission starts with a 3-line system prompt that says nothing about WHO they
serve, WHAT assets exist, or WHY this work matters.

### Current State (the gap)

```
PAT Agent System Prompt (lines 775-777 of node0_activate.py):
  "You are the PAT Strategist. Your role is Strategic planning.
   Standing on Giants: Sun Tzu, John Boyd, Michael Porter.
   Be concise (2-3 paragraphs). Focus on actionable insights."

What's missing:
  - WHO is MoMo? What are his expertise, values, pain points?
  - WHAT assets does Node0 have? 1.3TB, 144 repos, 3451 papers, 84K vectors
  - WHY does this matter? The covenant from Ramadan 2023, the 2 seed files
  - WHAT are the current goals? Weekly targets, active focus
  - WHAT has been learned? Previous mission outcomes, evidence chain
```

The RAG retrieval partially compensates (top-3 chunks per mission), but it's
keyword-based — it doesn't inject IDENTITY. An agent might retrieve a random
chunk about "federation protocols" when what it needs is "MoMo is a solo
founder in Dubai with $12 budget who needs to prove one node can serve one human."

### What Momo Said

> "This info is not just info — this is all the work you were building on.
>  This is what I want my node, Node0, and my PAT team to be aware of:
>  every single moment of pain, of dreams, of goals.
>  This is what I seeded 3 years back and should be inside the DNA."

### Target State

Every PAT agent mission starts with a **founder context preamble** that:
1. Identifies WHO they serve (MoMo, Node0, the seed)
2. Declares WHAT assets are available (hardware, data, domains, repos)
3. Anchors WHY (the 2 genesis files, the covenant, the 15K hours)
4. States current GOALS (weekly targets from baseline)
5. Adapts based on MISSION TYPE (not every mission needs full context)

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Phase 43 Components                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐    ┌──────────────────────┐        │
│  │  Spec 01            │    │  Spec 02             │        │
│  │  FounderContext      │    │  ContextTier Router  │        │
│  │  Builder             │    │                      │        │
│  │                     │    │  full / standard /   │        │
│  │  Loads:             │    │  minimal             │        │
│  │  - user_profile     │    │                      │        │
│  │  - node0_baseline   │    │  Based on:           │        │
│  │  - genesis covenant │    │  - mission keywords  │        │
│  │                     │    │  - agent role         │        │
│  │  Produces:          │    │  - token budget       │        │
│  │  - context string   │    │                      │        │
│  └─────────┬───────────┘    └──────────┬───────────┘        │
│            │                           │                     │
│            ▼                           ▼                     │
│  ┌─────────────────────────────────────────────────┐        │
│  │  Spec 03: PAT System Prompt Enrichment           │        │
│  │                                                   │        │
│  │  system_prompt = role + giants + founder_context  │        │
│  │  user_message = mission + rag_context             │        │
│  └───────────────────────┬───────────────────────────┘        │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────┐        │
│  │  Spec 04: Genesis Covenant Loader                │        │
│  │                                                   │        │
│  │  Reads 00_GENESIS/LINEAGE_START.md               │        │
│  │  Extracts covenant themes                         │        │
│  │  Provides "DNA" preamble for Guardian agent       │        │
│  └───────────────────────────────────────────────────┘        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Success Criteria

1. PAT agents produce measurably more relevant output (SNR score comparison)
2. Guardian agent can cite the Three Invariants from genesis context
3. Strategist agent references current weekly goals without RAG retrieval
4. Context injection adds < 200 tokens overhead (minimal cost)
5. System prompt + founder context stays under 600 tokens total
6. All 15 smoke tests continue to pass
7. All 91 SNR maximizer tests continue to pass
8. All 24 protocol/adapter tests continue to pass

## Spec Index

| Spec | File | Description |
|------|------|-------------|
| 01 | `01_founder_context_builder.md` | Build FounderContext from sovereign_state/ |
| 02 | `02_context_tier_router.md` | Route context depth by mission type |
| 03 | `03_pat_prompt_enrichment.md` | Inject context into PAT system prompts |
| 04 | `04_genesis_covenant_loader.md` | Load genesis DNA for Guardian agent |
| 05 | `05_validation_plan.md` | Test plan and acceptance criteria |

## Non-Goals (Phase 43 boundary)

- Full SMA (Sovereign Memory Architecture) implementation — that's Phase 44+
- HMM behavioral prediction — that's Phase 45+
- Cross-node federation identity — that's post-MVP
- Embedding the actual PDF content of البذرة.pdf — the themes are already
  extracted in LINEAGE_START.md and genesis.json
