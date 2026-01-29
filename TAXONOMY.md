# BIZRA Knowledge Taxonomy v1.0
## Discipline Enumeration & Classification System

**Document:** BIZRA-TAX-001
**Version:** 1.0.0
**Created:** 2026-01-22
**Status:** Active Reference

---

## 1. Overview

This taxonomy enumerates all knowledge disciplines, synergy types, compound pathways, and classification hierarchies within the BIZRA Data Lake ecosystem. It serves as the authoritative reference for cross-domain reasoning and pattern discovery.

---

## 2. Core Knowledge Domains

### 2.1 Primary Disciplines

| ID | Discipline | Description | Graph Node Type |
|----|------------|-------------|-----------------|
| D01 | **Computer Science** | Algorithms, data structures, systems | `cs:*` |
| D02 | **Artificial Intelligence** | ML, deep learning, NLP, vision | `ai:*` |
| D03 | **Mathematics** | Pure/applied math, statistics | `math:*` |
| D04 | **Philosophy** | Logic, ethics, epistemology | `phil:*` |
| D05 | **Physics** | Mechanics, quantum, thermodynamics | `phys:*` |
| D06 | **Engineering** | Software, systems, mechanical | `eng:*` |
| D07 | **Economics** | Markets, game theory, tokenomics | `econ:*` |
| D08 | **Psychology** | Cognitive, behavioral, social | `psych:*` |
| D09 | **Linguistics** | Syntax, semantics, pragmatics | `ling:*` |
| D10 | **Biology** | Systems biology, neuroscience | `bio:*` |
| D11 | **Information Theory** | Entropy, coding, compression | `info:*` |
| D12 | **Network Science** | Graphs, complexity, emergence | `net:*` |

### 2.2 Applied Domains

| ID | Domain | Description | Graph Node Type |
|----|--------|-------------|-----------------|
| A01 | **Product Development** | Design, UX, MVP iteration | `prod:*` |
| A02 | **Marketing** | Growth, positioning, storytelling | `mkt:*` |
| A03 | **Operations** | Processes, automation, scaling | `ops:*` |
| A04 | **Finance** | Modeling, valuation, treasury | `fin:*` |
| A05 | **Legal** | Compliance, IP, contracts | `legal:*` |
| A06 | **Research** | Methodology, experimentation | `research:*` |

---

## 3. Synergy Types (KEP Bridge)

The KEP Bridge identifies six fundamental synergy types that connect knowledge across domains.

### 3.1 Synergy Type Enumeration

```python
class SynergyType(Enum):
    """Cross-domain knowledge synergy classifications"""

    METHODOLOGICAL = "methodological"
    # Same method applied across domains
    # Example: Bayesian inference in AI + Economics

    STRUCTURAL = "structural"
    # Isomorphic structures across domains
    # Example: Network topology in Biology + Social Systems

    CONCEPTUAL = "conceptual"
    # Shared abstract concepts
    # Example: Emergence in Physics + Consciousness

    ANALOGICAL = "analogical"
    # Analogous patterns enabling transfer
    # Example: Evolutionary algorithms ← Natural selection

    CAUSAL = "causal"
    # Direct causal relationships
    # Example: Neuroscience → AI architectures

    EMERGENT = "emergent"
    # Novel insights from combination
    # Example: Quantum computing + Cryptography → Post-quantum security
```

### 3.2 Synergy Matrix

|  | CS | AI | Math | Phil | Phys | Eng |
|--|----|----|------|------|------|-----|
| **CS** | - | Causal | Structural | Conceptual | Analogical | Methodological |
| **AI** | Causal | - | Methodological | Conceptual | Structural | Causal |
| **Math** | Structural | Methodological | - | Structural | Causal | Methodological |
| **Phil** | Conceptual | Conceptual | Structural | - | Conceptual | Conceptual |
| **Phys** | Analogical | Structural | Causal | Conceptual | - | Causal |
| **Eng** | Methodological | Causal | Methodological | Conceptual | Causal | - |

---

## 4. Compound Types (Multi-hop Reasoning)

### 4.1 Compound Type Enumeration

```python
class CompoundType(Enum):
    """Multi-domain compound knowledge patterns"""

    SYNTHESIS = "synthesis"
    # Combining insights from multiple domains
    # Result: New unified understanding

    TRANSLATION = "translation"
    # Converting concepts between domain languages
    # Result: Cross-domain accessibility

    AMPLIFICATION = "amplification"
    # Domain knowledge strengthening another
    # Result: Enhanced insight depth

    VALIDATION = "validation"
    # Cross-domain evidence verification
    # Result: Increased confidence

    GENERATION = "generation"
    # Creating new knowledge at intersection
    # Result: Novel discoveries
```

### 4.2 Compound Pathway Count

Total pathways: **6 SynergyTypes × 5 CompoundTypes = 30 primary pathways**

Extended with domain pairs: **30 × C(12,2) = 30 × 66 = 1,980 potential pathways**

---

## 5. Graph-of-Thoughts Taxonomy

### 5.1 ThoughtType Hierarchy

```
ThoughtType
├── HYPOTHESIS
│   ├── Initial conjecture
│   ├── Derived hypothesis
│   └── Counter-hypothesis
│
├── EVIDENCE
│   ├── Direct evidence
│   ├── Circumstantial evidence
│   └── Counter-evidence
│
├── CONTRADICTION
│   ├── Logical contradiction
│   ├── Empirical contradiction
│   └── Temporal contradiction
│
├── SYNTHESIS
│   ├── Unifying synthesis
│   ├── Partial synthesis
│   └── Tentative synthesis
│
├── REFINEMENT
│   ├── Precision refinement
│   ├── Scope refinement
│   └── Confidence refinement
│
└── CONCLUSION
    ├── Strong conclusion (SNR ≥ 0.99)
    ├── Moderate conclusion (SNR ≥ 0.95)
    └── Weak conclusion (SNR < 0.95)
```

### 5.2 TensionType Classification

```
TensionType
├── GROUNDING_GAP
│   ├── Symbol-neural mismatch
│   ├── Abstraction level mismatch
│   └── Resolution: Additional grounding
│
├── SEMANTIC_DRIFT
│   ├── Meaning shift over reasoning
│   ├── Context loss
│   └── Resolution: Re-anchoring
│
├── COVERAGE_ASYMMETRY
│   ├── Unbalanced domain coverage
│   ├── Missing perspectives
│   └── Resolution: Targeted retrieval
│
├── CONTRADICTION
│   ├── Logical conflict
│   ├── Source disagreement
│   └── Resolution: Evidence weighting
│
└── COHERENT
    └── No tension detected (ideal state)
```

---

## 6. SNR Component Taxonomy

### 6.1 Signal Components

```python
SNR_COMPONENTS = {
    "signal_strength": {
        "weight": 0.35,
        "description": "Raw retrieval relevance score",
        "range": [0.0, 1.0],
        "calculation": "cosine_similarity(query, result)"
    },
    "information_density": {
        "weight": 0.25,
        "description": "Content richness and specificity",
        "range": [0.0, 1.0],
        "calculation": "unique_concepts / total_tokens"
    },
    "symbolic_grounding": {
        "weight": 0.25,
        "description": "Graph connectivity and knowledge anchoring",
        "range": [0.0, 1.0],
        "calculation": "connected_edges / max_possible_edges"
    },
    "coverage_balance": {
        "weight": 0.15,
        "description": "Query coverage completeness",
        "range": [0.0, 1.0],
        "calculation": "covered_aspects / total_aspects"
    }
}
```

### 6.2 SNR Quality Tiers

| Tier | SNR Range | Classification | Action |
|------|-----------|----------------|--------|
| **Ihsān** | ≥ 0.99 | Excellence | Accept |
| **High** | 0.95-0.99 | Acceptable | Optimize |
| **Medium** | 0.80-0.95 | Caution | Review |
| **Low** | < 0.80 | Reject | Retry/Expand |

---

## 7. DDAGI Consciousness Taxonomy

### 7.1 Consciousness Event Types

```python
class ConsciousnessEvent(Enum):
    """DDAGI consciousness event classifications"""

    REFLECTION = "reflection"
    # System self-assessment
    # Triggers: Periodic, post-query, error recovery

    LEARNING = "learning"
    # New knowledge integration
    # Triggers: Novel pattern detection

    ADAPTATION = "adaptation"
    # Behavioral adjustment
    # Triggers: Performance degradation

    SYNTHESIS = "synthesis"
    # Cross-domain insight generation
    # Triggers: Multi-hop reasoning completion

    ATTESTATION = "attestation"
    # Cryptographic knowledge verification
    # Triggers: POI checkpoint
```

### 7.2 Consciousness Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Reflection Frequency | Self-assessments per hour | ≥ 4 |
| Learning Rate | New patterns per session | ≥ 10 |
| Adaptation Speed | Time to behavioral change | < 60s |
| Synthesis Quality | SNR of generated insights | ≥ 0.95 |
| Attestation Validity | POI verification success | 100% |

---

## 8. POI (Proof-of-Impact) Taxonomy

### 8.1 Attestation Types

```python
class AttestationType(Enum):
    """POI attestation classifications"""

    GENESIS = "genesis"
    # Initial knowledge creation
    # Hash: SHA-256 of content + timestamp

    DERIVATION = "derivation"
    # Knowledge derived from existing
    # Hash: SHA-256 of (parent_hash + delta)

    SYNTHESIS = "synthesis"
    # Multi-source knowledge combination
    # Hash: Merkle root of source hashes

    VALIDATION = "validation"
    # External verification
    # Hash: Ed25519 signature

    EVOLUTION = "evolution"
    # Knowledge refinement
    # Hash: Chain of refinement hashes
```

### 8.2 POI Ledger Structure

```json
{
  "entry_id": "poi_001",
  "type": "synthesis",
  "timestamp": "2026-01-22T10:30:00Z",
  "content_hash": "sha256:abc123...",
  "parent_hashes": ["sha256:def456...", "sha256:ghi789..."],
  "merkle_root": "sha256:jkl012...",
  "attestation_hash": "ed25519:mno345...",
  "snr_at_creation": 0.97,
  "ihsan_compliant": true
}
```

---

## 9. Agent Taxonomy (PAT Engine)

### 9.1 Core Agents

| Agent | Role | ThinkingMode | Specialization |
|-------|------|--------------|----------------|
| **Strategist** | High-level planning | SYNTHESIS | Goal decomposition |
| **Researcher** | Information gathering | DEEP | Knowledge retrieval |
| **Analyst** | Data interpretation | CRITICAL | Pattern recognition |
| **Creator** | Content generation | CREATIVE | Synthesis output |
| **Guardian** | Quality assurance | CRITICAL | Ihsān enforcement |
| **Coordinator** | Orchestration | FAST | Workflow management |

### 9.2 ThinkingMode Taxonomy

```python
class ThinkingMode(Enum):
    """Agent cognitive mode classifications"""

    FAST = "fast"
    # Quick pattern matching
    # Use: Simple queries, coordination
    # Token budget: Low

    DEEP = "deep"
    # Thorough analysis
    # Use: Research, investigation
    # Token budget: High

    CREATIVE = "creative"
    # Divergent thinking
    # Use: Content generation, ideation
    # Token budget: Medium

    CRITICAL = "critical"
    # Rigorous evaluation
    # Use: Quality assurance, validation
    # Token budget: Medium

    SYNTHESIS = "synthesis"
    # Integration of multiple perspectives
    # Use: Strategic planning, conclusions
    # Token budget: High
```

---

## 10. Content Type Taxonomy

### 10.1 Document Types

| Type | Extensions | Processor | Embedding Model |
|------|------------|-----------|-----------------|
| Text | .txt, .md | Direct | MiniLM |
| Document | .pdf, .docx | Unstructured | MiniLM |
| Code | .py, .js, .ts | AST Parser | CodeBERT |
| Data | .csv, .json | Schema Extract | MiniLM |
| Image | .png, .jpg | CLIP | CLIP-ViT |
| Audio | .mp3, .wav | Whisper | Wav2Vec2 |

### 10.2 Chunk Types

| Type | Size | Overlap | Use Case |
|------|------|---------|----------|
| Sentence | ~50 tokens | 0 | Precise retrieval |
| Paragraph | ~200 tokens | 50 | Context-aware |
| Section | ~500 tokens | 100 | Deep reasoning |
| Document | Full | 0 | Holistic analysis |

---

## 11. Appendix: Complete Enumeration Counts

### Total Classifications

| Category | Count |
|----------|-------|
| Primary Disciplines | 12 |
| Applied Domains | 6 |
| Synergy Types | 6 |
| Compound Types | 5 |
| Primary Pathways | 30 |
| Extended Pathways | 1,980 |
| ThoughtTypes | 6 (18 subtypes) |
| TensionTypes | 5 (12 subtypes) |
| SNR Components | 4 |
| SNR Tiers | 4 |
| Consciousness Events | 5 |
| Attestation Types | 5 |
| Core Agents | 6 |
| ThinkingModes | 5 |
| Document Types | 6 |
| Chunk Types | 4 |

### Cross-Reference Index

```
Discipline → SynergyType → CompoundType → ThoughtType → Conclusion
     ↓            ↓              ↓             ↓
  Graph Node   Edge Type    Reasoning     SNR Score
     ↓            ↓              ↓             ↓
  Embedding    Weight       GoT Phase     Ihsān Gate
```

---

*BIZRA Knowledge Taxonomy v1.0*
*Total Enumerated Classifications: 2,090+ unique pathways*
*Ihsān Compliance: Required for all classifications*
