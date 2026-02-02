# UERS Overview — Universal Entropy Reduction Singularity

**Status:** PARTIAL (new module integrated; not yet wired into orchestrator runtime)

## What it is
UERS is a 5‑dimensional entropy reduction framework designed to quantify and drive system convergence:
- **Surface** (Shannon entropy)
- **Structural** (graph topology entropy)
- **Behavioral** (trace entropy)
- **Hypothetical** (path entropy)
- **Contextual** (semantic intent entropy)

## Core Modules
- `core/uers/__init__.py` — constants, manifold definition, PoI parameters
- `core/uers/entropy.py` — multi‑dimensional entropy calculators
- `core/uers/vectors.py` — vector manifold + probing
- `core/uers/convergence.py` — convergence loop
- `core/uers/impact.py` — Impact Oracle (PoI)
- `core/uers/sare_bridge.py` — bridge to SARE

## Integration Notes
- Currently **module‑level** integration only.
- Next step: wire UERS into `bizra_orchestrator.py` or a dedicated pipeline stage.

## Giants Protocol Mapping
- Shannon → Surface vector
- Lamport → Structural vector
- Vaswani → Contextual vector
- Besta → Hypothetical vector
- Anthropic → Contextual vector
