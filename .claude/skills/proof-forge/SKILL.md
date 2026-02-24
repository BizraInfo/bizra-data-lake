---
name: proof-forge
description: >
  Converts any development step into audited, investor-grade proof automatically.
  Runs the BUILD → VERIFY → EVIDENCE loop: executes tests/benchmarks, generates
  SHA-256 hash-chained receipts, updates an evidence index, and produces a concise
  investor-readable Proof Summary. Use this skill whenever the user mentions evidence
  generation, proof packages, verification receipts, audit trails, investor proof,
  evidence chains, hash receipts, attestation, "prove this works," "show me the
  evidence," build verification, development evidence, MoneyShot evidence, spearpoint,
  proof-of-work documentation, or any request to convert development output into
  verifiable, traceable, investor-usable proof. Also trigger when the user says
  "forge," "receipt," "evidence index," "proof summary," or asks to demonstrate
  that something was built, tested, and verified. Even if the user simply says
  "I just finished X, make it provable" — this is the skill.
---

# Proof Forge

**The Sovereign Evidence Kernel — BUILD → VERIFY → EVIDENCE in one pass.**

Proof Forge converts any development action into a cryptographically chained,
investor-readable evidence artifact. It is the smallest system that makes every
other module provable, trustable, debuggable, and sellable.

## Why This Exists

Without verified evidence, every claim is just a claim. Proof Forge ensures that
every development step automatically produces:

1. A **verified artifact** — tests run, benchmarks captured, schemas validated
2. A **signed receipt** — SHA-256 hash chain linking this evidence to all prior evidence
3. An **updated evidence index** — the single source of truth for all proven work
4. A **Proof Summary** — concise, investor-readable, ready for a pitch deck or due diligence

## When to Use

- After completing any development milestone
- When preparing investor materials or due diligence packages
- When you need to prove a system works (not just claim it)
- Before any release, demo, or external presentation
- When onboarding auditors, partners, or new team members
- Any time someone says "prove it"

## Core Workflow

```
┌─────────────────────────────────────────────────┐
│                  PROOF FORGE                     │
│                                                  │
│  INPUT                                           │
│  ├── What was built/changed (description)        │
│  ├── Target directory or repo path               │
│  └── Optional: specific files to verify          │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  BUILD   │→ │  VERIFY  │→ │   EVIDENCE    │  │
│  │ Discover │  │ Run tests│  │ Hash receipt   │  │
│  │ artifacts│  │ Validate │  │ Update index   │  │
│  │ Collect  │  │ Benchmark│  │ Proof summary  │  │
│  │ metadata │  │ Check    │  │ Package        │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  OUTPUT                                          │
│  ├── .proof-forge/                               │
│  │   ├── receipts/YYYY-MM-DD_HHMMSS.json        │
│  │   ├── EVIDENCE_INDEX.json                     │
│  │   └── summaries/YYYY-MM-DD_HHMMSS.md         │
│  └── PROOF_SUMMARY.md (latest, top-level)        │
└─────────────────────────────────────────────────┘
```

## Execution Protocol

Follow these phases in order. Each phase feeds the next.

### Phase 1: BUILD — Discovery & Collection

Scan the target directory to understand what exists and what changed.

**Step 1.1 — Identify the target.**
Ask the user only if not already provided:
- What did you build or change? (natural language description)
- Where is it? (directory path)

If the user provides a git repo, use `git diff` and `git log` to identify changes.
If no git, scan for recently modified files.

**Step 1.2 — Collect artifact metadata.**
For every relevant file in the target:
- File path (relative to project root)
- File size in bytes
- SHA-256 hash of file contents
- Last modified timestamp
- File type classification: `code | test | config | doc | data | binary`

**Step 1.3 — Detect verification targets.**
Automatically discover what can be verified:
- Test files: `*test*`, `*spec*`, `test_*`, `*_test.*`
- Build configs: `Cargo.toml`, `package.json`, `Makefile`, `pyproject.toml`
- Benchmark files: `*bench*`, `*perf*`
- CI configs: `.github/workflows/*`, `.gitlab-ci.yml`
- Schema files: `*.schema.json`, `*.xsd`, `*.proto`

Store discovery results for Phase 2.

### Phase 2: VERIFY — Execution & Validation

Run every verification that can be run. Capture all output. Never silently skip.

**Step 2.1 — Run tests (if discovered).**

| Ecosystem | Command | Fallback |
|-----------|---------|----------|
| Rust | `cargo test 2>&1` | `cargo check 2>&1` |
| Node.js | `npm test 2>&1` | `npx jest 2>&1` or `npx vitest 2>&1` |
| Python | `python -m pytest 2>&1` | `python -m unittest discover 2>&1` |
| Go | `go test ./... 2>&1` | `go vet ./... 2>&1` |
| Make | `make test 2>&1` | `make check 2>&1` |

Capture: exit code, stdout, stderr, duration, pass/fail counts.

**Step 2.2 — Run benchmarks** if found. Capture numeric results.

**Step 2.3 — Validate schemas** if JSON schema files exist.

**Step 2.4 — Static checks** — linters or type checkers if configured.

**Step 2.5 — Compile verification report** as structured JSON.
See `references/evidence-schema.md` for the verification report schema.

If no automated checks exist, prompt for manual attestation:
"No automated tests found. Describe what you verified manually."
This still produces a valid receipt with verification type `manual_attestation`.

### Phase 3: EVIDENCE — Receipt, Index, Summary

The cryptographic backbone. Every receipt chains to the previous one.

**Step 3.1 — Generate hash receipt.**

```bash
python3 scripts/forge_evidence.py \
  --project-dir <target_directory> \
  --description "<what was built>" \
  --verification-report <path_to_verification_json>
```

The script:
- Loads previous receipt to get chain hash (or creates genesis)
- Computes composite evidence hash over all artifact hashes + verification results
- Creates receipt with `previous_hash` linking to the chain
- Writes receipt to `.proof-forge/receipts/`
- Updates `EVIDENCE_INDEX.json`

**Step 3.2 — Generate Proof Summary.**

```bash
python3 scripts/proof_summary.py \
  --receipt <path_to_new_receipt> \
  --project-dir <target_directory>
```

Produces investor-readable Markdown:
- **What was built** (1-2 sentences)
- **How it was verified** (test results, benchmarks, checks)
- **Evidence chain** (receipt hash, chain position, previous link)
- **Key metrics** (lines changed, test count, performance data)
- **Confidence level** (based on verification depth)

**Step 3.3 — Present results to user.**
Display: verification status, receipt hash, chain position, summary link, any warnings.

## Confidence Levels

| Level | Label | Criteria |
|-------|-------|----------|
| 5 | **Ironclad** | Tests + benchmarks + static analysis + schema validation + CI |
| 4 | **Strong** | Tests pass + at least one additional verification type |
| 3 | **Solid** | Tests pass OR build succeeds with static analysis |
| 2 | **Attested** | Manual attestation with description, no automated checks |
| 1 | **Logged** | Evidence collected but no verification was possible |

## Graceful Degradation

Proof Forge works at every level of project maturity:

- **Full CI/CD repo**: Runs everything → Ironclad confidence
- **Repo with tests**: Runs tests locally → Strong confidence
- **Repo with no tests**: Hashes artifacts, manual attestation → Attested
- **Single file**: Hashes file, records purpose → Logged+
- **Just a claim**: Records claim with timestamp → Logged

The skill never refuses to run. It always produces the best evidence possible
given what exists. The confidence level communicates what was actually verified.

## Chain Verification

To verify the integrity of an existing evidence chain:

```bash
python3 scripts/forge_evidence.py --verify --project-dir <target_directory>
```

Walks the chain from genesis to latest, recomputing each link hash
and reporting any breaks. Useful for auditors and due diligence.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/forge_evidence.py` | Receipt generation, chain management, verification |
| `scripts/proof_summary.py` | Investor-readable Proof Summary generator |
| `scripts/verify_artifacts.py` | Discovers and runs all available verification checks |

## Evidence Export

The `.proof-forge/` directory is fully portable. Share it with investors,
auditors, or partners. The chain is self-verifying — anyone can recompute
hashes to confirm integrity.
