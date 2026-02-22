# Phase 03 — Receipt & Verification Pipeline

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** Merkle (hash trees, 1988) · Lamport (timestamps, 1978) · Bernstein (Ed25519, 2011) · Shannon (SNR, 1948) · Al-Ghazali (Ihsan, 1095)
> **Reuses:** `core.pci.envelope`, `core.pci.crypto`, `core.integration.constants`

## 3.1 Functional Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| R-01 | Every non-trivial output produces a Receipt | Receipt generated for all PAT/TaskForce outputs |
| R-02 | Receipt contains: inputs hash, model/version, evidence links, SNR, Ihsan | All fields non-null |
| R-03 | Receipts form a hash chain (tamper-evident) | Each receipt includes prev_hash |
| R-04 | ZANN_ZERO: speculation labeled, unverifiable blocked | Zero untagged speculation in outputs |
| R-05 | IHSAN_FLOOR: below threshold degrades to silence | Output suppressed + evidence request |
| R-06 | Receipts stored locally, never transmitted without consent | Local encrypted store only |
| R-07 | Receipts are independently verifiable | Any party with public key can verify |

## 3.2 Receipt Data Structure

```pseudocode
MODULE ReceiptPipeline:
  """
  Proof-Carrying Inference receipt generation.
  Standing on Giants: Merkle (hash chain) · Lamport (timestamps) · Bernstein (Ed25519)
  Reuses: core.pci.envelope, core.pci.crypto
  """

  STRUCT Receipt:
    # ── Identity ──
    receipt_id:       UUID
    chain_index:      uint64              # Monotonic position in chain
    prev_hash:        bytes[32]           # SHA-256 of previous receipt (chain link)
    timestamp:        ISO8601             # UTC, Lamport-ordered

    # ── Provenance ──
    inputs_hash:      bytes[32]           # SHA-256 of canonical(input_data)
    model_id:         String              # e.g., "deepseek-r1-0528-qwen3-8b"
    model_version:    String              # Version or commit hash
    tool_used:        Optional<String>    # From TOOL_ALLOWLIST if applicable
    agent_id:         String              # PAT agent that produced the output

    # ── Evidence ──
    evidence_links:   List<EvidencePointer>  # Sources, files, URLs
    output_hash:      bytes[32]           # SHA-256 of canonical(output_data)
    output_size:      uint32              # Bytes

    # ── Quality Scores ──
    snr_score:        float               # 0.0 - 1.0 (Shannon)
    ihsan_score:      float               # 0.0 - 1.0 (Al-Ghazali)
    confidence:       float               # 0.0 - 1.0

    # ── Policy Decision ──
    policy_result:    PolicyDecision      # APPROVED / DEGRADED / BLOCKED
    zann_flags:       List<ZannFlag>      # Speculation markers

    # ── Cryptographic Seal ──
    node_id:          bytes[16]           # Issuing node fingerprint
    signature:        bytes[64]           # Ed25519(private_key, canonical(self))

  STRUCT EvidencePointer:
    type:     Enum(FILE, URL, RECEIPT, MEMORY)
    uri:      String                      # Path or URL
    hash:     Optional<bytes[32]>         # Content hash at time of reference
    label:    String                      # Human description

  ENUM PolicyDecision:
    APPROVED   # SNR >= threshold, Ihsan >= threshold, no ZANN violations
    DEGRADED   # Below threshold — output silenced, evidence requested
    BLOCKED    # ZANN_ZERO violation — speculation detected and blocked
```

## 3.3 Receipt Generation Flow

```pseudocode
  FUNCTION generate_receipt(
    task:           TaskNode,
    input_data:     Any,
    output_data:    Any,
    model_version:  String,
    evidence:       List<EvidencePointer>,
    snr_score:      float,
    ihsan_score:    float,
    agent_id:       String
  ) -> Receipt:

    # ── Step 1: Hash inputs and outputs ──
    inputs_hash  = sha256(canonical_json(input_data))    # core.pci.crypto
    output_hash  = sha256(canonical_json(output_data))

    # ── Step 2: ZANN_ZERO check ──
    zann_flags = scan_for_speculation(output_data)
    FOR flag IN zann_flags:
      IF flag.severity == UNVERIFIABLE:
        # Block the output entirely
        RETURN generate_blocked_receipt(
          reason="ZANN_ZERO: unverifiable claim detected",
          claim=flag.claim,
          inputs_hash=inputs_hash
        )
      IF flag.severity == SPECULATIVE:
        # Tag but allow (labeled speculation)
        output_data = tag_speculation(output_data, flag)

    # ── Step 3: IHSAN_FLOOR check ──
    IF ihsan_score < UNIFIED_IHSAN_THRESHOLD:           # 0.95
      RETURN generate_degraded_receipt(
        reason=f"Ihsan {ihsan_score:.3f} < {UNIFIED_IHSAN_THRESHOLD}",
        suggestion="Gather better evidence before answering",
        inputs_hash=inputs_hash
      )

    # ── Step 4: SNR floor check ──
    IF snr_score < UNIFIED_SNR_THRESHOLD:                # 0.85
      RETURN generate_degraded_receipt(
        reason=f"SNR {snr_score:.3f} < {UNIFIED_SNR_THRESHOLD}",
        suggestion="Output has too much noise — refine before presenting",
        inputs_hash=inputs_hash
      )

    # ── Step 5: Build receipt ──
    prev = get_last_receipt()
    receipt = Receipt(
      receipt_id     = uuid4(),
      chain_index    = prev.chain_index + 1 IF prev ELSE 0,
      prev_hash      = sha256(canonical_json(prev)) IF prev ELSE ZERO_HASH,
      timestamp      = utc_now(),
      inputs_hash    = inputs_hash,
      model_id       = get_active_model_id(),
      model_version  = model_version,
      tool_used      = task.tool_required,
      agent_id       = agent_id,
      evidence_links = evidence,
      output_hash    = output_hash,
      output_size    = len(canonical_json(output_data)),
      snr_score      = snr_score,
      ihsan_score    = ihsan_score,
      confidence     = min(snr_score, ihsan_score),
      policy_result  = PolicyDecision.APPROVED,
      zann_flags     = zann_flags,
      node_id        = get_node_id(),
      signature      = EMPTY  # Filled below
    )

    # ── Step 6: Sign ──
    receipt.signature = sign_message(               # core.pci.crypto
      private_key=load_private_key(),
      message=canonical_json(receipt, exclude=["signature"])
    )

    # ── Step 7: Persist to chain ──
    append_to_chain(receipt)

    RETURN receipt
```

## 3.4 ZANN_ZERO — Speculation Detection

```pseudocode
MODULE ZannZero:
  """
  Speculation firewall. No unverified claims pass without labels.
  Standing on Giants: Popper (falsifiability, 1934) · Shannon (noise detection)
  """

  ENUM ZannSeverity:
    FACTUAL        # Backed by evidence — no flag
    SPECULATIVE    # Plausible but unproven — labeled and passed
    UNVERIFIABLE   # Cannot be verified — BLOCKED

  STRUCT ZannFlag:
    claim:      String
    severity:   ZannSeverity
    reason:     String
    evidence_gap: String     # What evidence would resolve this

  FUNCTION scan_for_speculation(output: Any) -> List<ZannFlag>:
    flags = []
    statements = extract_claims(output)

    FOR stmt IN statements:
      evidence = find_supporting_evidence(stmt)
      IF evidence.score >= 0.9:
        CONTINUE  # Factual — no flag needed
      ELSE IF evidence.score >= 0.5:
        flags.append(ZannFlag(
          claim=stmt.text,
          severity=SPECULATIVE,
          reason="Partial evidence found",
          evidence_gap=evidence.missing_sources
        ))
      ELSE:
        flags.append(ZannFlag(
          claim=stmt.text,
          severity=UNVERIFIABLE,
          reason="No supporting evidence found",
          evidence_gap="Need primary source or empirical data"
        ))

    RETURN flags

  FUNCTION tag_speculation(output: Any, flag: ZannFlag) -> Any:
    """
    Wraps speculative claims with visible labels.
    User always knows what is proven vs. speculated.
    """
    RETURN output.replace(
      flag.claim,
      f"[SPECULATIVE: {flag.reason}] {flag.claim}"
    )
```

## 3.5 Chain Verification

```pseudocode
MODULE ChainVerification:
  """
  Independent verification of receipt chain integrity.
  Standing on Giants: Merkle (1988) · Bitcoin (hash chain, 2008)
  """

  FUNCTION verify_chain(chain: List<Receipt>, public_key: bytes[32]) -> ChainVerdict:
    errors = []

    FOR i, receipt IN enumerate(chain):
      # Check 1: Signature valid
      payload = canonical_json(receipt, exclude=["signature"])
      IF NOT verify_signature(public_key, payload, receipt.signature):
        errors.append(f"Receipt {i}: Invalid signature")

      # Check 2: Chain link valid
      IF i > 0:
        expected_prev = sha256(canonical_json(chain[i-1]))
        IF receipt.prev_hash != expected_prev:
          errors.append(f"Receipt {i}: Broken chain link")

      # Check 3: Monotonic index
      IF i > 0 AND receipt.chain_index != chain[i-1].chain_index + 1:
        errors.append(f"Receipt {i}: Non-monotonic index")

      # Check 4: Timestamp ordering
      IF i > 0 AND receipt.timestamp <= chain[i-1].timestamp:
        errors.append(f"Receipt {i}: Timestamp not strictly increasing")

    RETURN ChainVerdict(
      valid=len(errors) == 0,
      receipts_checked=len(chain),
      errors=errors
    )

  FUNCTION verify_single_receipt(receipt: Receipt, public_key: bytes[32]) -> bool:
    payload = canonical_json(receipt, exclude=["signature"])
    RETURN verify_signature(public_key, payload, receipt.signature)
```

## 3.6 TDD Anchors

```pseudocode
TEST "receipt_has_all_required_fields":
  receipt = generate_receipt(task, input, output, model, evidence, 0.92, 0.97, "master")
  ASSERT receipt.receipt_id IS NOT NONE
  ASSERT receipt.inputs_hash IS NOT NONE
  ASSERT receipt.output_hash IS NOT NONE
  ASSERT receipt.snr_score == 0.92
  ASSERT receipt.ihsan_score == 0.97
  ASSERT receipt.signature IS NOT NONE
  ASSERT len(receipt.signature) == 64

TEST "receipt_chain_links_valid":
  r1 = generate_receipt(...)
  r2 = generate_receipt(...)
  ASSERT r2.prev_hash == sha256(canonical_json(r1))
  ASSERT r2.chain_index == r1.chain_index + 1

TEST "ihsan_below_threshold_produces_degraded":
  receipt = generate_receipt(..., ihsan_score=0.80)
  ASSERT receipt.policy_result == PolicyDecision.DEGRADED
  ASSERT "Ihsan" IN receipt.reason

TEST "snr_below_threshold_produces_degraded":
  receipt = generate_receipt(..., snr_score=0.70, ihsan_score=0.99)
  ASSERT receipt.policy_result == PolicyDecision.DEGRADED
  ASSERT "SNR" IN receipt.reason

TEST "zann_zero_blocks_unverifiable":
  output = "The market will crash tomorrow by 50%"
  receipt = generate_receipt(..., output_data=output)
  ASSERT receipt.policy_result == PolicyDecision.BLOCKED
  ASSERT "ZANN_ZERO" IN receipt.reason

TEST "zann_zero_labels_speculative":
  output = "Based on partial data, this approach might work"
  receipt = generate_receipt(..., output_data=output)
  ASSERT any(f.severity == SPECULATIVE FOR f IN receipt.zann_flags)

TEST "chain_verification_detects_tampering":
  chain = generate_chain(5)
  # Tamper with receipt 3
  chain[2].output_hash = random_bytes(32)
  verdict = verify_chain(chain, public_key)
  ASSERT verdict.valid == false
  ASSERT "Receipt 3" IN verdict.errors[0]

TEST "receipt_signature_verifiable_by_any_party":
  receipt = generate_receipt(...)
  public_key = load_public_key()  # Separate from private key
  ASSERT verify_single_receipt(receipt, public_key) == true

TEST "degraded_receipt_suppresses_output":
  receipt = generate_receipt(..., ihsan_score=0.80)
  ASSERT receipt.policy_result == DEGRADED
  ASSERT receipt.output_hash == ZERO_HASH  # Output not stored
```
