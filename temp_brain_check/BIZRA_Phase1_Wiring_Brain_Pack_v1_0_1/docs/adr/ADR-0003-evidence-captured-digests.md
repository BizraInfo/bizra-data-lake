# ADR-0003: Evidence-Captured Model Digests as Source of Truth

**Status:** PROPOSED  
**Date:** 2025-12-14  
**Scope:** Genesis Phase 0+

## Decision
All `pinned_artifacts.*.digest` values MUST be sourced from an evidence capture run (`evidence/audit-results-node0.json`),
not from `ollama list` short IDs nor human transcription.

## Rationale
- `ollama list` shows short IDs (12 hex) which are *not* a stable cryptographic digest.
- The Ollama REST endpoint `/api/tags` is the preferred source for full digests.
- Sealing requires deterministic provenance.

## Consequences
- Sealing automation must fail if a pinned digest is missing or not present in the evidence file.
- Any model update (pull/replace) requires a new evidence capture and reseal event.

## Verification
- CI step: `scripts/verify_digests.py` (to be added) checks manifest digests are present in evidence.
