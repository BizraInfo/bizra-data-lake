# Phase 3: Epistemic Calculus — Zann Zero (Z = 0)

**Spec:** CMN-003
**Status:** Formalization layer (existing: IMPLEMENTED)
**Formal Property:** Every knowledge entry epsilon = (Claim, DerivationChain, ValidatorSignature)
**Existing Code:** `core/proof_engine/evidence_ledger.py`, `core/urp/membrane.py`, `bizra-proofspace/receipt_chain.rs`

---

## 1. Objective

The Proof of Truth (PoT) algorithm ensures **Zero-Assumption Inference** — no claim enters
the shared knowledge registry without a verifiable BLAKE3 derivation chain and validator
signature. This spec formalizes the chain integrity theorem and adds gap tests.

---

## 2. Definitions

```
KnowledgeEntry epsilon := (
    claim:              str,                # the assertion
    derivation_chain:   list[BLAKE3Hash],   # ordered hashes of source materials
    chain_root:         BLAKE3Hash,         # hash of the entire chain
    validator_signature: Ed25519Signature,  # SAT validator who verified
    timestamp:          float,
    source_node:        NodeID              # anonymized after membrane crossing
)

INVARIANT (Zann Zero):
    For all epsilon in KnowledgeRegistry:
        verify_chain(epsilon.derivation_chain) == True
        AND verify_signature(epsilon.validator_signature, epsilon.chain_root) == True
        AND epsilon.claim is derivable from epsilon.derivation_chain

THEOREM (Tamper Evidence):
    Given a chain C = [h_0, h_1, ..., h_n] where h_i = BLAKE3(source_i || h_{i-1}):
    P(tamper undetected) <= 2^{-256}
```

---

## 3. Pseudocode

### 3.1 Proof of Truth Validator (new: `core/proof_engine/proof_of_truth.py`)

```python
class ProofOfTruth:
    """Validates that a knowledge entry satisfies Zann Zero."""

    def __init__(self, trust_anchors: list[Ed25519PublicKey]):
        self._trust_anchors = {k.node_id: k for k in trust_anchors}

    def validate_entry(self, entry: KnowledgeEntry) -> ValidationResult:
        """Full PoT validation: chain + signature + derivability."""
        chain_ok = self._verify_derivation_chain(entry.derivation_chain)
        sig_ok = self._verify_validator_signature(
            entry.validator_signature, entry.chain_root
        )
        derived = self._check_derivability(entry.claim, entry.derivation_chain)

        return ValidationResult(
            chain_integrity=chain_ok,
            signature_valid=sig_ok,
            claim_derivable=derived,
            zann_zero=chain_ok and sig_ok and derived,
        )

    def _verify_derivation_chain(self, chain: list[BLAKE3Hash]) -> bool:
        """Walk the chain, recompute each hash, verify linkage."""
        if not chain:
            return False
        prev = GENESIS_HASH
        for i, entry in enumerate(chain):
            expected = blake3(entry.source_bytes + prev.encode())
            if expected != entry.hash:
                return False
            prev = entry.hash
        return True

    def _verify_validator_signature(
        self, sig: Ed25519Signature, chain_root: BLAKE3Hash
    ) -> bool:
        """Signature must come from a known SAT validator."""
        validator = self._trust_anchors.get(sig.signer_id)
        if validator is None:
            return False  # unknown validator => reject
        return validator.verify(chain_root.encode(), sig.bytes)

    def _check_derivability(
        self, claim: str, chain: list[BLAKE3Hash]
    ) -> bool:
        """Claim must reference at least one source in the chain.
        Full semantic derivability is deferred to SAT validation."""
        return len(chain) > 0  # minimal: non-empty chain
```

### 3.2 Chain Fork Detection

```python
def detect_chain_fork(
    chain_a: list[BLAKE3Hash], chain_b: list[BLAKE3Hash]
) -> ForkResult:
    """If two chains diverge from a common prefix, detect the fork point."""
    common_length = 0
    for ha, hb in zip(chain_a, chain_b):
        if ha.hash == hb.hash:
            common_length += 1
        else:
            break

    if common_length == min(len(chain_a), len(chain_b)):
        return ForkResult(forked=False, common_prefix=common_length)

    return ForkResult(
        forked=True,
        fork_point=common_length,
        chain_a_divergent=chain_a[common_length:],
        chain_b_divergent=chain_b[common_length:],
    )
```

---

## 4. TDD Anchors

```python
# tests/core/test_zann_zero.py

def test_valid_chain_passes():
    """Well-formed BLAKE3 chain with valid signature => Zann Zero."""
    pot = ProofOfTruth(trust_anchors=[SAT_VALIDATOR_KEY])
    entry = build_valid_entry(sources=["paper.pdf", "dataset.csv"])
    result = pot.validate_entry(entry)
    assert result.zann_zero is True

def test_tampered_chain_detected():
    """Modify one hash in the chain => chain_integrity = False."""
    pot = ProofOfTruth(trust_anchors=[SAT_VALIDATOR_KEY])
    entry = build_valid_entry(sources=["paper.pdf"])
    entry.derivation_chain[0].hash = "deadbeef" * 8  # tamper
    result = pot.validate_entry(entry)
    assert result.chain_integrity is False
    assert result.zann_zero is False

def test_unknown_validator_rejected():
    """Signature from unknown validator => reject."""
    pot = ProofOfTruth(trust_anchors=[SAT_VALIDATOR_KEY])
    entry = build_valid_entry(sources=["paper.pdf"])
    entry.validator_signature.signer_id = "unknown_node"
    result = pot.validate_entry(entry)
    assert result.signature_valid is False

def test_empty_chain_rejected():
    """Claim with no derivation chain => not derivable."""
    pot = ProofOfTruth(trust_anchors=[SAT_VALIDATOR_KEY])
    entry = build_entry_no_sources()
    result = pot.validate_entry(entry)
    assert result.claim_derivable is False
    assert result.zann_zero is False

def test_chain_fork_detected():
    """Two chains diverging after block 3 => fork at index 3."""
    chain_a = build_chain(length=5, seed="alpha")
    chain_b = build_chain(length=5, seed="alpha", diverge_at=3)
    result = detect_chain_fork(chain_a, chain_b)
    assert result.forked is True
    assert result.fork_point == 3

def test_identical_chains_no_fork():
    """Same chain compared to itself => no fork."""
    chain = build_chain(length=5, seed="same")
    result = detect_chain_fork(chain, chain)
    assert result.forked is False
```

---

## 5. Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `core/proof_engine/evidence_ledger.py` | PoT wraps the existing append + verify_chain |
| `core/urp/membrane.py` | Membrane records are a special case of derivation chains |
| `bizra-proofspace/receipt_chain.rs` | Rust receipt chain is the hot path; Python PoT is the verifier |
| `core/urp/constitution.py` | `zann_zero: bool` flag is the constitutional switch |
| `bizra-mission/receipt.rs` | Mission receipts feed into PoT as evidence entries |
