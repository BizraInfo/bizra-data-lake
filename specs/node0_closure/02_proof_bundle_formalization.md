# Phase 2: Proof Bundle Formalization
## Goal: Canonical, self-contained, replayable proof packages
### References: 00_master_spec.md §4 Phase 2

---

## 1. Problem Statement

Current proof bundles (MISSION_001, MISSION_005) are manually assembled
markdown documents. They lack:
- Machine-parseable manifest schema
- Self-contained replay capability
- Cryptographic binding to evidence chain
- Automated generation from mission output

## 2. Canonical Proof Bundle Schema

```
ManifestArtifact:
  version: "1.0.0"
  mission_id: string           # "mission-001"
  timestamp: ISO8601
  node_id: string              # "node0-momo-genesis"

  # Core evidence
  receipt:
    receipt_id: string
    receipt_hash: BLAKE3
    prev_hash: BLAKE3
    decision: PERMIT | REVIEW | QUARANTINED
    reason_codes: [string]

  # Scoring
  gate_verdict:
    snr_score: float
    snr_engine: string
    ihsan_score: float
    ihsan_achieved: bool
    quality_tier: string

  # Token economy
  token_summary:
    total_transactions: int
    seed_minted: float
    zakat_deducted: float
    impt_minted: float
    agents_rewarded: [string]

  # Evidence chain position
  evidence:
    chain_file: string         # "sovereign_state/evidence.jsonl"
    sequence: int
    prev_hash: BLAKE3
    entry_hash: BLAKE3

  # Replay
  replay:
    command: string            # full shell command to reproduce
    env_vars: {string: string} # required environment
    timeout_s: int
    expected_artifacts: [string]

  # Memory
  memory:
    entries_loaded: int
    entries_persisted: int
    db_path: string

  # Pipeline trace
  pipeline:
    corpus_vectors: int
    corpus_chunks: int
    agents_assigned: [string]
    got_thoughts: int
    got_paths: int
    got_mode: "llm" | "template"
    inference_backend: string
    inference_model: string
    gpu_device: string
```

## 3. Pseudocode: Automated Bundle Generator

```
FUNCTION generate_proof_bundle(mission_result) -> ManifestArtifact:
    manifest = ManifestArtifact()
    manifest.version = "1.0.0"
    manifest.mission_id = mission_result.mission_id
    manifest.timestamp = utc_now()
    manifest.node_id = mission_result.node_id

    # Extract receipt from evidence chain
    evidence = load_evidence_chain(EVIDENCE_JSONL)
    latest = evidence.entries[-1]
    manifest.receipt = ReceiptSummary(
        receipt_id=latest.receipt.receipt_id,
        receipt_hash=latest.entry_hash,
        prev_hash=latest.prev_hash,
        decision=latest.receipt.decision,
        reason_codes=latest.receipt.reason_codes,
    )

    # Gate verdict
    manifest.gate_verdict = GateVerdict(
        snr_score=mission_result.snr_score,
        snr_engine=mission_result.snr_engine,
        ihsan_score=mission_result.ihsan_score,
        ihsan_achieved=mission_result.ihsan_score >= IHSAN_THRESHOLD,
        quality_tier=classify_tier(mission_result.snr_score),
    )

    # Token economy summary
    ledger = load_token_ledger(TOKEN_LEDGER_JSONL)
    mission_txs = filter_by_mission(ledger, mission_result.mission_id)
    manifest.token_summary = TokenSummary(
        total_transactions=len(mission_txs),
        seed_minted=sum(tx.amount FOR tx IN mission_txs IF tx.token == "SEED" AND tx.type == "mint"),
        zakat_deducted=sum(tx.amount FOR tx IN mission_txs IF tx.type == "zakat"),
        impt_minted=sum(tx.amount FOR tx IN mission_txs IF tx.token == "IMPT"),
        agents_rewarded=unique(tx.recipient FOR tx IN mission_txs),
    )

    # Evidence position
    manifest.evidence = EvidencePosition(
        chain_file="sovereign_state/evidence.jsonl",
        sequence=len(evidence.entries),
        prev_hash=latest.prev_hash,
        entry_hash=latest.entry_hash,
    )

    # Replay instructions
    manifest.replay = ReplayInstructions(
        command=build_replay_command(mission_result),
        env_vars=capture_env_vars(["BIZRA_ENABLE_OLLAMA_EXECUTE", "BIZRA_OLLAMA_MODEL"]),
        timeout_s=300,
        expected_artifacts=[EVIDENCE_JSONL, TOKEN_LEDGER_JSONL, MEMORY_DB],
    )

    # Serialize
    write_json(f"sovereign_state/proofs/{manifest.mission_id}_manifest.json", manifest)
    write_markdown(f"sovereign_state/proofs/{manifest.mission_id}_bundle.md", manifest)

    RETURN manifest
```

## 4. Pseudocode: Bundle Validator

```
FUNCTION validate_proof_bundle(manifest_path) -> ValidationResult:
    manifest = load_json(manifest_path)
    errors = []

    # 1. Schema validation
    IF NOT schema_valid(manifest, MANIFEST_SCHEMA_V1):
        errors.append("SCHEMA_INVALID")

    # 2. Receipt chain integrity
    evidence = load_evidence_chain(manifest.evidence.chain_file)
    entry = evidence.entries[manifest.evidence.sequence - 1]
    IF entry.entry_hash != manifest.evidence.entry_hash:
        errors.append("EVIDENCE_HASH_MISMATCH")
    IF entry.prev_hash != manifest.evidence.prev_hash:
        errors.append("PREV_HASH_MISMATCH")

    # 3. Token ledger consistency
    ledger = load_token_ledger(TOKEN_LEDGER_JSONL)
    mission_txs = filter_by_mission(ledger, manifest.mission_id)
    IF len(mission_txs) != manifest.token_summary.total_transactions:
        errors.append("TOKEN_COUNT_MISMATCH")

    # 4. Gate verdict consistency
    IF manifest.gate_verdict.ihsan_achieved:
        IF manifest.receipt.decision == "QUARANTINED":
            errors.append("IHSAN_DECISION_CONFLICT")

    # 5. Replay command parseable
    IF NOT command_parseable(manifest.replay.command):
        errors.append("REPLAY_UNPARSEABLE")

    RETURN ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        manifest=manifest,
    )
```

## 5. Implementation Touchpoints

| File | Change |
|------|--------|
| `scripts/node0_activate.py` | Call `generate_proof_bundle()` after mission |
| `core/proof_engine/manifest.py` | NEW: ManifestArtifact dataclass + generator |
| `core/proof_engine/bundle_validator.py` | NEW: validate_proof_bundle() |
| `sovereign_state/proofs/` | Output directory (gitignored, force-add releases) |

## 6. TDD Anchors

```python
# tests/core/proof_engine/test_manifest.py

def test_manifest_schema_valid():
    """Generated manifest passes JSON Schema validation."""
    manifest = generate_proof_bundle(mock_mission_result())
    assert schema_valid(manifest, MANIFEST_SCHEMA_V1)

def test_manifest_receipt_matches_evidence():
    """Receipt hash in manifest matches evidence chain entry."""
    manifest = generate_proof_bundle(mock_mission_result())
    evidence = load_evidence_chain(manifest.evidence.chain_file)
    assert evidence.entries[-1].entry_hash == manifest.evidence.entry_hash

def test_manifest_token_summary_correct():
    """Token counts in manifest match actual ledger."""
    manifest = generate_proof_bundle(mock_mission_result())
    ledger = load_token_ledger(TOKEN_LEDGER_JSONL)
    actual = sum(1 for tx in ledger if tx.mission_id == manifest.mission_id)
    assert actual == manifest.token_summary.total_transactions

def test_bundle_validator_catches_tampered_hash():
    """Validator rejects manifest with wrong evidence hash."""
    manifest = generate_proof_bundle(mock_mission_result())
    manifest.evidence.entry_hash = "deadbeef" * 8
    result = validate_proof_bundle(manifest)
    assert not result.valid
    assert "EVIDENCE_HASH_MISMATCH" in result.errors

def test_replay_command_executable():
    """Replay command string is syntactically valid shell."""
    manifest = generate_proof_bundle(mock_mission_result())
    assert "python scripts/node0_activate.py mission" in manifest.replay.command
```

## 7. Validation Gate

```
GENERATE proof bundle for latest mission
VALIDATE bundle passes schema + integrity checks
ASSERT receipt hash matches evidence chain
ASSERT token counts match ledger
ASSERT replay command is parseable
ASSERT bundle file exists at expected path
```

---

*Proof bundles are infrastructure, not documentation.*
*They are part of the trust substrate (Golden Gem #5).*
