# Omnidirectional Hyper-dimensional Audit Engine (OHAE) v0.1

**Purpose:** Repeatable, evidence-based, read-only audit system for BIZRA across architecture, security, public claims, performance, documentation, scalability, error handling, dependencies, SWE practices, symbolic-neural coherence, Ihsan alignment, Node0 activation readiness, and golden-gem surfacing.

## Core doctrine (inherited from BIZRA)

- **Evidence before claim.** Every finding must cite a source.
- **Proof before public language.** Exact metrics that don't have a receipt get downgraded to directional claims.
- **Ihsan before scale.** Don't ship more surface than we can honestly back.
- **Observable reasoning only.** No private chain-of-thought is written into reports — only observable pattern extraction, evidence paths, and decision summaries.

## Design

- **Pure stdlib Python.** No third-party deps (no pydantic, no PyYAML). A tiny YAML loader handles the 2-level config.
- **Read-only.** Never writes to source files. Writes only under `--out-dir`.
- **Deterministic.** Same repo state + same config → same artifacts.
- **Bounded.** File-size cap per pattern scan; max-results cap per artifact.
- **Composable.** Each scanner is a module; `run_audit.py` orchestrates.

## Directory

```
tools/audit/omni_audit/
├── __init__.py
├── README.md                      ← this file
├── audit_config.yaml              ← scope, severity, runtime, output caps
├── schemas.py                     ← EvidenceItem, Claim, Finding, Risk, Mitigation, Kpi, Gate, GraphNode/Edge
├── run_audit.py                   ← orchestrator
├── evidence_index.py              ← doctrine + manifests + config inventory
├── claim_scanner.py               ← scans docs + website captures for claims
├── dependency_inventory.py        ← Rust / Python / Node deps + gap analysis
├── secret_pattern_scanner.py      ← redacted-preview secret scanner
├── code_risk_scanner.py           ← Python + Rust risk pattern scanner
├── website_claim_capture.py       ← bizra.ai / bizra.info fetch (offline-safe)
├── snr_classifier.py              ← Signal / Noise / Watchlist
├── hhmm_taxonomy.py               ← 4-level hierarchical hidden-state taxonomy
└── graph_export.py                ← audit evidence graph (JSON + Graphviz DOT)
```

## Usage

```bash
# From repo root. Writes all artifacts under --out-dir.
python3 -m tools.audit.omni_audit.run_audit \
  --repo-root . \
  --out-dir docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts \
  --no-network
```

### Flags

- `--repo-root PATH` — repo root (default: `.`)
- `--out-dir PATH` — output directory (required; created if missing)
- `--website URL` (repeatable) — override website targets
- `--no-network` — use the offline pre-check skeleton instead of live fetch
- `--strict` — currently a hint; reserved for future hard-fail on any PROHIBITED finding
- `--config PATH` — path to audit_config.yaml (default: sibling file)

## Outputs (artifacts)

- `evidence_index.{json,csv}` — scoped repo inventory with sha256
- `claims_register.{json,csv}` — A/B/C/D/E-classified public claims
- `secret_findings.json` — redacted-preview credential findings (values NEVER printed)
- `code_risks.json` — Rust + Python risk-pattern findings
- `dependencies.json` — full Rust/Python/Node inventory + gap list
- `website_claims.json` + `website_snapshot.txt` — captured (or pre-checked) live site content
- `snr_findings.json` — findings split into signal/noise/watchlist
- `hhmm_taxonomy.json` — 4-level taxonomy (domain → subsystem → mode → action)
- `audit_graph.{json,dot}` — nodes + typed edges
- `findings.json`, `gates.json`, `kpis.json`, `risks.json`, `mitigations.json` — structured registers
- `audit_summary.json` — one-shot summary for orchestrators and CI

## Safety / discipline

- **No mutation of source files, canon, runtime, or git state.**
- **No private chain-of-thought emitted.** Reports use observable reasoning traces, evidence paths, and decision summaries only.
- **Secrets scanner never prints the matched value.** Redacted preview only.
- **Exact metric claims** on public surfaces that can't be source-chained to a receipt are flagged `NEEDS_REWRITE` or `PROHIBITED`.
- **No publishing** of any artifact. All output lives under `--out-dir`.

## Re-running / CI

Safe to run in CI under a read-only mount. Add `--strict` to escalate once the strict rules are tightened (not yet implemented; stub accepted so invocations stay stable).
