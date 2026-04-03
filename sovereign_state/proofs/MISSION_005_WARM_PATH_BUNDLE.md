# MISSION_005 — Warm-Path Proof Bundle
## GoT→LLM Bridge Fix + Honest Scoring Analysis
### Date: 2026-04-04 00:48 UTC | Phase 89 Continuation

---

## Mission Input

```
"What is the Ihsan principle and how does BIZRA enforce it constitutionally?"
```

## Bugs Found & Fixed

### Bug 1: GoT→LLM Bridge Never Initialized
- **File**: `scripts/node0_activate.py:674`
- **Root cause**: `InferenceGateway()` constructed but `await gateway.initialize()` never called
- **Effect**: GoT always fell back to template mode (capped SNR at ~0.47)
- **Fix**: Added `await gateway.initialize()` + null gateway on failure

### Bug 2: LM Studio Connects But Returns Empty
- **File**: `scripts/node0_activate.py:680`
- **Root cause**: LM Studio server reachable but model not loaded in VRAM → 0 tokens generated
- **Effect**: GoT received gateway with no working backend
- **Fix**: Added health-check inference after init, falls back to Ollama if empty

### Bug 3: Ollama Model Selection Ignored Env Var
- **File**: `core/inference/_backends.py:413`
- **Root cause**: Hardcoded preference list had `phi3` first, ignoring `BIZRA_OLLAMA_MODEL`
- **Effect**: GoT used phi3:mini (3.8B) instead of requested llama3.1:8b
- **Fix**: Respects `BIZRA_OLLAMA_MODEL` env var, reordered preferences by capability

## Mission Progression (5 runs)

| Mission | Model | GoT Mode | SNR | Ihsan | Status |
|---------|-------|----------|-----|-------|--------|
| 001 (cold) | LM Studio (template) | Template | 0.4754 | 47.54% | REVIEW |
| 002 | LM Studio (template) | Template | 0.4754 | 47.54% | REVIEW |
| 003 | Ollama qwen2.5:3b | **LLM** | **0.6150** | 61.50% | REVIEW |
| 004 | Ollama phi3:mini | **LLM** | 0.6063 | 60.63% | REVIEW |
| 005 | Ollama llama3.1:8b | **LLM** | 0.6043 | 60.43% | REVIEW |

**Key finding**: Template→LLM upgrade gave +30% SNR (0.47→0.61). Model size (3B→8B) had negligible effect.

## SNR Scoring Breakdown (Mission 005)

### V2 Engine (Shannon + Renyi-2): 0.7293
| Metric | Score | Assessment |
|--------|-------|------------|
| signal_strength | 0.60 | Moderate |
| diversity | 0.96 | Excellent |
| grounding | 0.65 | Missing code citations |
| semantic_relevance | 0.50 | Broad match, not precise |
| channel_efficiency | 0.43 | Low information density |
| entropy | 0.97 | Excellent |

### Text Engine (7-dim heuristic): 0.6105
| Signal Dimension | Score | Noise Dimension | Score |
|------------------|-------|-----------------|-------|
| relevance | 0.67 | redundancy | 0.00 |
| novelty | 0.87 | inconsistency | 0.00 |
| groundedness | 0.50 | ambiguity | 0.00 |
| coherence | 0.60 | irrelevance | 0.43 |
| actionability | 0.65 | hallucination | 0.00 |
| specificity | 0.70 | verbosity | 0.09 |
| | | bias | 0.00 |

### Ensemble (geometric mean): 0.6150
- **Recommendation**: "Add citations or evidence"

## Why PERMIT (0.85) Is Not Yet Reachable

The SNR scorer correctly identifies that LLM-generated prose, even when substantive, lacks:

1. **Grounded citations** (groundedness=0.50): No `core/path/file.py` references
2. **Precise relevance** (semantic_relevance=0.50): Essay prose vs. surgical answers
3. **Information density** (irrelevance=0.43): Connecting prose flagged as noise

**To reach 0.85 requires pipeline changes:**
- RAG-inject retrieved FAISS chunks as inline citations in the final output
- Structure GoT conclusion as evidence-backed assertions, not essay prose
- These are P1 roadmap items, not model swaps

## Evidence Chain

```json
{
  "entry_hash": "16f6e994a41030b6...",
  "prev_hash": "ff40b5b5ac10b3ad...",
  "receipt_id": "fa5f4e98f5b6cdaa...",
  "decision": "QUARANTINED",
  "reason_codes": ["SNR_BELOW_THRESHOLD"],
  "evidence_seq": 34
}
```

## Token Ledger (Missions 001-005, TX#49-108)

- 60 transactions across 5 missions
- 4 agents per mission: coordinator, executor, strategist, analyst
- SEED minted + zakat deducted per agent per mission
- IMPT (impact) tokens minted per agent

## Assessment

**Status: GoT→LLM bridge proven, warm-path operational, scoring honest.**

The constitutional spine correctly distinguishes between:
- Template prose (SNR ~0.47) — low signal, no reasoning
- LLM-synthesized prose (SNR ~0.61) — moderate signal, real reasoning
- Grounded evidence-backed output (SNR >0.85) — high signal, citations required

The 0.47→0.61 jump proves the bridge fix is real. The 0.61→0.85 gap is a pipeline architecture task (RAG citation injection), not a bug.

## Standing on Giants

- Shannon (information entropy scoring) — V2 engine
- Besta (Graph-of-Thoughts, 2024) — 3 LLM hypotheses + conclusion
- Deming (PDCA) — fix→measure→verify→document cycle
- Anthropic (Constitutional AI) — honest scoring, refuse to rubber-stamp

---

*Generated: 2026-04-04 | Node0 Genesis Block | Phase 89 | 5 missions, 3 bugs fixed*
