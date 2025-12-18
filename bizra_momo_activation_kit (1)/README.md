# BIZRA Momo Architect Activation Kit

This kit is a **local-first** set of scripts + templates to help a solo developer transition to an **agentic team workflow**:
- **PAT** (Personal Agentic Team): user-directed builders
- **SAT** (System Agentic Team): autonomous validators / guards

## What you get
- `scripts/bizra_ops_bootstrap.py` – creates a lightweight ops workspace inside your repo and generates:
  - file inventory (json + csv)
  - duplicate detection (by hash)
  - fast secret-pattern scan (baseline)
  - evidence bundle per run (timestamped)

- `prompts/` – copy-paste ready agent prompt templates (PAT + SAT)

- `docs/MOMO_ARCHITECT_OS.md` – operating model + execution checklist

## Quick start (cross-platform)
From your repo root:

```bash
python scripts/bizra_ops_bootstrap.py --root . --fail-on-secrets
```

If you're on Windows PowerShell:

```powershell
python .\scripts\bizra_ops_bootstrap.py --root . --fail-on-secrets
```

Outputs are written to: `momo_ops/`

## What this DOES NOT do
This kit does not modify your production code automatically. It gives you:
- structure
- guardrails
- repeatable evidence

Then you decide what to change.

## Recommended workflow
1) Run bootstrap → fix secrets → rerun until clean
2) Feed inventory + findings to PAT agents (build)
3) SAT agents verify (tests, scans, policy checks)
4) Seal evidence and proceed phase-by-phase

