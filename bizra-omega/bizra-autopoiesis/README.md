# bizra-autopoiesis

**BIZRA Autopoiesis** — Self-Creating Pattern Memory and Preference Tracking.

## Overview

Implements autopoietic (self-creating) systems for the BIZRA sovereign node:

- **Pattern Memory** — Stores, recalls, and evolves patterns using cosine similarity over embedding vectors
- **Preference Tracker** — Learns user preferences through reinforcement/decay cycles

## Key Types

| Type | Purpose |
|------|---------|
| `Pattern` | Embedding vector + metadata with confidence scoring |
| `PatternMemory` | Learn/recall interface backed by `PatternStore` |
| `Preference` | Reinforceable preference with strength convergence |
| `PreferenceTracker` | Observe, activate, and apply preferences to prompts |

## Constants

- `EMBEDDING_DIM = 384` — Embedding vector dimensionality
- `ELEVATION_THRESHOLD = 0.95` — Ihsān-grade quality bar

## Dependencies

- bizra-core

## License

MIT
