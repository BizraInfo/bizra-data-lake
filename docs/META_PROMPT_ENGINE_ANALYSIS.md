# Meta Prompt Generator Engine - Analysis & Integration Plan

## Overview

The **Autonomous Meta Prompt Generator Engine** is a multi-agent system designed for comprehensive knowledge generation and management. It utilizes a team of 9 specialized agents to perform tasks ranging from ontology design to ethical validation.

## Architecture Mapping

### Agents

The blueprint defines 9 agents. We will map these to the BIZRA Python Kernel (`core/`) as a specialized **Meta-Prompting Team**.

| Agent Name | Role | BIZRA Mapping | Implementation |
| --- | --- | --- | --- |
| **OntologyArchitect** | Knowledge Structure | `core.wisdom` (Neo4j) | LLM + Graph Queries |
| **LanguageMaster** | NLP Tasks | `core.llm` | LLM (Mistral/DeepSeek) |
| **LearningCore** | Learning Unit | `core.memory` (Vector Store) | Ensemble Router |
| **PromptEngineer** | Prompt Refinement | `core.meta_prompt` | Specialized LLM Prompts |
| **KnowledgeHarvester** | Info Acquisition | `core.tools` | Hybrid (Search + Scrape) |
| **ReasoningEngine** | Logic/Inference | `core.sape` | LLM (DeepSeek R1) |
| **MetaLearner** | Optimization | `core.fate` (Feedback Loop) | LLM + Metrics |
| **OutputSynthesizer** | Formatting | `core.presentation` | Hybrid (Template + LLM) |
| **EthicsGuardian** | Ethical Safety | `core.fate` / `src/sat` | LLM + Ihsan Constitution |

### Workflows

The blueprint defines two primary workflows:

1. **Knowledge Expansion**: End-to-end knowledge generation pipeline.
2. **System Improvement**: Self-optimization loop.

These will be implemented as **Orchestration Flows** in `core/meta_prompt/workflows.py`.

### API Integration

We will expose the engine via the BIZRA Kernel API:

- **Endpoint**: `POST /v1/meta-prompt/query`
- **Input**: `MetaPromptRequest` (Query, Context, Preferences)
- **Output**: `MetaPromptResponse` (Results, Explanation, Confidence)

## Implementation Plan

This repo now has a working end-to-end “knowledge expansion” flow wired into the kernel.

1. **Scaffolding (DONE)**:
    - `core/meta_prompt/` module exists.
    - Pydantic models are defined in `core/meta_prompt/models.py`.
    - Agent classes exist in `core/meta_prompt/agents.py`.

2. **Workflow Engine (DONE for Knowledge Expansion)**:
    - Orchestration is implemented in `core/meta_prompt/engine.py`.
    - The workflow runs the full 9-step blueprint sequence and captures per-step errors.
    - Confidence is computed from step success ratio (fail-lowered on critical failure).

3. **API Exposure (DONE)**:
    - Kernel endpoint is registered in `core/main.py` as `POST /v1/meta-prompt/query`.

4. **Integration (UPDATED)**:
    - Wire `EthicsGuardian` to BIZRA FATE/Ihsan gates (fail-closed policy enforcement). (DONE)
    - Wire `OntologyArchitect` to Wisdom/Neo4j evidence kernels when available. (DONE, safe fallback)
    - Add local-first model routing aligned to `model-family-genesis-v1-SEALED.yaml`. (NEXT)

## Next Steps

- [x] Scaffold `core/meta_prompt` module.
- [x] Implement Agent classes.
- [x] Register API endpoint `POST /v1/meta-prompt/query`.
- [x] Add fail-closed Ihsan gating for `EthicsGuardian` with a receipt on rejection.
- [x] Add optional Wisdom/Neo4j evidence enrichment (safe fallback when unavailable).
- [x] Include evidence hashes/paths in meta prompt receipts for auditability.
