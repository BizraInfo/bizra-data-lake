# Phase 1: RAG Citation Injection into GoT Output
## Goal: Convert REVIEW → PERMIT (SNR ≥ 0.85, Ihsan ≥ 0.95)
### References: 00_master_spec.md §4 Phase 1

---

## 1. Problem Statement

GoT synthesis produces coherent prose but lacks grounded citations.
The SNR scorer penalizes:
- `groundedness = 0.50` (no file/code references)
- `irrelevance = 0.43` (connecting prose flagged as noise)
- `semantic_relevance = 0.50` (broad match, not precise)

Current ensemble: **0.61**. Target: **≥ 0.85**.

## 2. Root Cause Analysis

```
FAISS retrieves 102K vectors → chunks injected as agent context
Agents produce text using chunks as background knowledge
GoT synthesizes agent outputs into conclusion
SNR scorer evaluates the CONCLUSION text only
  └── conclusion has no inline citations from the retrieved chunks
  └── scorer sees "essay" not "evidence-backed assertion"
```

The knowledge retrieval results are consumed but not **surfaced** in the scored output.

## 3. Pseudocode: Citation-Enriched GoT Pipeline

```
FUNCTION synthesize_with_citations(mission, agent_results, retrieved_chunks):
    # Phase A: Extract citation-worthy chunks
    top_chunks = rank_by_relevance(retrieved_chunks, mission.query, k=5)
    citation_map = {}
    FOR i, chunk IN enumerate(top_chunks):
        citation_key = f"[{i+1}]"
        citation_map[citation_key] = {
            "source": chunk.source_file,
            "content": chunk.text[:200],
            "score": chunk.similarity_score
        }

    # Phase B: Inject citation context into GoT
    enriched_facts = []
    FOR result IN agent_results:
        IF result.success AND result.content:
            # Find which chunks this agent's output aligns with
            aligned = find_aligned_chunks(result.content, top_chunks, threshold=0.6)
            citations = [f"[{top_chunks.index(c)+1}]" FOR c IN aligned]
            enriched_facts.append(
                f"[{result.agent_name}]: {result.content} "
                f"(supported by: {', '.join(citations)})"
            )

    # Phase C: GoT with citation-aware prompt
    got = GraphOfThoughts(inference_gateway=gateway)
    got_prompt = TEMPLATE("""
        Synthesize the following agent analyses into a grounded conclusion.
        IMPORTANT: Reference specific sources using [N] citation markers.
        Include at least 3 inline citations from the provided evidence.

        Agent Analyses:
        {enriched_facts}

        Available Citations:
        {citation_map_formatted}

        Requirements:
        - Every major claim must have at least one [N] citation
        - Include specific file paths or module names where relevant
        - Minimize connecting prose; maximize evidence density
    """)

    reasoning_result = AWAIT got.reason(
        query=mission.query,
        context={"facts": enriched_facts, "citations": citation_map}
    )

    # Phase D: Post-process to ensure citation presence
    conclusion = reasoning_result.conclusion
    citation_count = count_pattern(r'\[\d+\]', conclusion)

    IF citation_count < 3:
        # Append citation block if LLM didn't inline them
        conclusion += "\n\nEvidence:\n"
        FOR key, cite IN citation_map.items():
            conclusion += f"  {key} {cite['source']}: {cite['content'][:100]}...\n"

    RETURN conclusion, citation_map
```

## 4. Scoring Impact Projection

```
Current (no citations):
  V2:   signal_strength=0.60, grounding=0.65, semantic_relevance=0.50
  Text: groundedness=0.50, irrelevance=0.43
  Ensemble: 0.61

Target (with citations):
  V2:   signal_strength=0.80+, grounding=0.85+, semantic_relevance=0.75+
  Text: groundedness=0.80+, irrelevance=0.15-
  Ensemble: 0.85+ (projected)
```

## 5. Implementation Touchpoints

| File | Change |
|------|--------|
| `scripts/node0_activate.py` | Pass `retrieved_chunks` to `_synthesize_with_got()` |
| `scripts/node0_activate.py:_synthesize_with_got()` | Add citation injection logic |
| `core/sovereign/graph_reasoning.py` | Accept citation context in `reason()` |
| `core/sovereign/graph_reasoning.py:_generate_hypotheses()` | Citation-aware prompt template |

## 6. TDD Anchors

```python
# tests/test_rag_citation_injection.py

def test_citation_injection_adds_references():
    """Citations from FAISS chunks appear in GoT output."""
    chunks = [make_chunk("core/pci/gates.py", "PCIGateKeeper enforces...")]
    result = synthesize_with_citations(mission, agent_results, chunks)
    assert "[1]" in result.conclusion
    assert "core/pci/gates.py" in result.conclusion

def test_citation_count_minimum():
    """At least 3 citations in output when 5+ chunks available."""
    chunks = [make_chunk(f"file_{i}.py", f"content {i}") for i in range(5)]
    result = synthesize_with_citations(mission, agent_results, chunks)
    assert count_citations(result.conclusion) >= 3

def test_snr_improvement_with_citations():
    """Grounded output scores higher than ungrounded."""
    ungrounded = "Ihsan is a meta-protocol for alignment."
    grounded = (
        "Ihsan is enforced via PCIGateKeeper [1] with threshold 0.95 "
        "defined in constants.py [2]. The FATE engine [3] validates..."
    )
    facade = build_snr_facade()
    score_bare = facade.calculate(text=ungrounded, query=QUERY)
    score_cited = facade.calculate(text=grounded, query=QUERY)
    assert score_cited.score > score_bare.score

def test_fallback_citation_block():
    """If LLM doesn't inline citations, append evidence block."""
    # Mock LLM that returns no [N] markers
    result = synthesize_with_citations(mission, results, chunks)
    assert "Evidence:" in result.conclusion

def test_empty_chunks_graceful():
    """No crash when FAISS returns 0 chunks."""
    result = synthesize_with_citations(mission, results, [])
    assert result.conclusion  # still produces output
```

## 7. Validation Gate

```
RUN mission with RAG citation injection
ASSERT SNR >= 0.85
ASSERT Ihsan >= 0.95
ASSERT verdict == "PERMIT"
ASSERT receipt.decision != "QUARANTINED"
ASSERT evidence_chain.sequence == prev + 1
ASSERT citation_count >= 3 in scored_output
```

## 8. Risk

- LLM may not follow citation instruction → fallback appends block
- FAISS chunks may be tangential → filter by similarity > 0.6
- Over-citation may reduce coherence → cap at 8 citations max
- Scorer may still penalize if citations are mechanical → test with real output

---

*Phase 1 is the critical path. All other phases can proceed in parallel,*
*but PERMIT cannot be achieved without grounded output.*
