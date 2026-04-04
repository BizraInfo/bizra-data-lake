# Canonical Loop Proof Artifact v1 — Execution Guide

**Priority #2 in the canonicalized stack. Uses commands that already exist.**

## What This Produces

One BLAKE3-sealed evidence bundle proving BIZRA's canonical mission loop end-to-end on NODE0. Ten stages, each captured as raw output + structured JSON, sealed with a single proof hash.

## Prerequisites (all already present on NODE0)

```powershell
# Verify CLI binary exists
C:\BIZRA-DATA-LAKE\bizra-omega\target\release\bizra.exe --help

# Verify b3sum is available (install if needed)
cargo install b3sum

# Verify git state
cd C:\BIZRA-DATA-LAKE
git log --oneline -3
# Should show 64f6a706 as HEAD
```

## Execution

### Option A: Full automated script (WSL)

```bash
cd /mnt/c/BIZRA-DATA-LAKE/bizra-omega
chmod +x canonical_loop_proof.sh
./canonical_loop_proof.sh
```

### Option B: Manual step-by-step (PowerShell)

Run each command, inspect output, proceed only if clean:

```powershell
cd C:\BIZRA-DATA-LAKE\bizra-omega
$BIZRA = "target\release\bizra.exe"
$BUNDLE = "proof_bundle_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $BUNDLE

# Stage 0: Genesis
& $BIZRA genesis | Tee-Object "$BUNDLE\00_GENESIS.txt"

# Stage 1: Agents
& $BIZRA agents | Tee-Object "$BUNDLE\01_AGENTS.txt"

# Stage 2: Node
& $BIZRA node | Tee-Object "$BUNDLE\02_NODE.txt"

# Stage 3: Mission
& $BIZRA mission "Analyze system health and recommend highest-priority improvement" | Tee-Object "$BUNDLE\03_MISSION.txt"

# Stage 4: Receipt verification
& $BIZRA receipt --verify | Tee-Object "$BUNDLE\04_RECEIPT.txt"

# Stage 5: Replay (use receipt ID prefix from Stage 3 output)
& $BIZRA replay <receipt-id-prefix> | Tee-Object "$BUNDLE\05_REPLAY.txt"

# Stage 6: Trust
& $BIZRA trust | Tee-Object "$BUNDLE\06_TRUST.txt"

# Stage 7: Manifest
& $BIZRA manifest | Tee-Object "$BUNDLE\07_MANIFEST.txt"

# Stage 8: Brief
& $BIZRA brief | Tee-Object "$BUNDLE\08_BRIEF.txt"

# Stage 9: Seal
b3sum "$BUNDLE\*" > "$BUNDLE\PROOF_SEAL.b3"
```

### Option C: Claude Code session on NODE0

Paste this into Claude Code:

```
Run the canonical loop proof: execute bizra genesis, agents, node, mission, 
receipt --verify, replay, trust, manifest, and brief in sequence. Capture 
each output. Bundle into proof_bundle/ with BLAKE3 seal. Commit and push.
```

## What Each Stage Proves

| Stage | Command | What It Proves | Canon Reference |
|-------|---------|----------------|-----------------|
| 0 | `bizra genesis` | Node identity + constitutional seal exist | CANON-005 Phase 1 |
| 1 | `bizra agents` | PAT-7 (user) + SAT-5 (system) = 12 agents live | CANON-002 |
| 2 | `bizra node` | Substrate awareness, model fleet, thresholds | CANON-006 |
| 3 | `bizra mission` | Governed execution through constitutional pipeline | CANON-001 |
| 4 | `bizra receipt` | Cross-process receipt verification (BLAKE3 + Ed25519) | CANON-005 Phase 3 |
| 5 | `bizra replay` | Deterministic replay from receipt evidence | CANON-005 Phase 4 |
| 6 | `bizra trust` | 13/13 constitutional checks → SOVEREIGN | CANON-006 |
| 7 | `bizra manifest` | Daily proof-of-life with chain seal | CANON-005 Phase 3 |
| 8 | `bizra brief` | Ghost proactive briefing from live backends | CANON-005 Phase 5 |
| 9 | Truth labels | Every claim bound to evidence taxonomy | CLAIM_MUST_BIND |

## Truth Label for This Artifact

**HARNESS-VERIFIED**

This bundle uses real CLI commands on real infrastructure on NODE0. It is explicitly labeled as a controlled canonical proof run. It does NOT claim hostile-environment battle-testing or production-grade scale. It DOES claim: one authoritative local loop, receipted, replayable, and constitutionally gated.

## After Execution

```bash
cd /mnt/c/BIZRA-DATA-LAKE
git add bizra-omega/proof_bundle_*/
git commit -m "evidence: Canonical Loop Proof Artifact v1 — HARNESS-VERIFIED"
git push
```

## Placement in Canon

```
B:\BIZRA-SOVEREIGN\03_EVIDENCE\canonical_loop_proof_v1\
C:\BIZRA-DATA-LAKE\bizra-omega\proof_bundle_YYYYMMDD_HHMMSS\
```

## Chain Position

```
Previous: RETROSPECTIVE_2026-04-04 (commit 64f6a706)
This:     CANONICAL_LOOP_PROOF_v1
Next:     Node1 Reproducibility Program
```
