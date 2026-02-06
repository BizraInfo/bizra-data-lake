# BIZRA V3 Secure Development Patterns

**Version:** 1.0.0
**Date:** 2026-02-03
**Author:** Security Architect (Agent #2)
**Classification:** DEVELOPER GUIDE

---

## Purpose

This document provides reusable secure coding patterns for the BIZRA ecosystem. All new code MUST follow these patterns to maintain the security posture established in the threat model.

---

## 1. Cryptographic Patterns

### 1.1 Key Generation

**REQUIRED:** Use the PCI crypto module for all key generation.

```python
# CORRECT: Use standard key generation
from core.pci import generate_keypair

private_key_hex, public_key_hex = generate_keypair()

# INCORRECT: Do not use custom key generation
# import secrets
# private_key = secrets.token_hex(32)  # WRONG - not Ed25519
```

**Rationale:** Consistent key format ensures interoperability across the federation.

### 1.2 Message Signing

**REQUIRED:** Use domain-separated digests for all signatures.

```python
# CORRECT: Domain-separated signing
from core.pci.crypto import (
    canonical_json,
    domain_separated_digest,
    sign_message,
    verify_signature,
)

# Prepare data
data = {"action": "vote", "proposal_id": "prop_123"}
canonical = canonical_json(data)
digest = domain_separated_digest(canonical)

# Sign
signature = sign_message(digest, private_key_hex)

# Verify
is_valid = verify_signature(digest, signature, public_key_hex)
```

**Rationale:** Domain separation prevents cross-protocol signature reuse attacks.

### 1.3 Signature Verification Order

**CRITICAL:** Always verify signatures BEFORE any caching or deduplication.

```python
# CORRECT: Signature verification first
async def handle_message(self, data: bytes):
    msg = deserialize(data)

    # Step 1: Verify signature FIRST
    if not verify_signature(msg.digest, msg.signature, msg.public_key):
        return None  # Silent rejection, do NOT cache

    # Step 2: Check for duplicates AFTER verification
    if self._is_duplicate(msg.message_id):
        return None

    # Step 3: Add to dedup cache
    self._seen_messages.add(msg.message_id)

    # Step 4: Process message
    return await self._process(msg)

# INCORRECT: This allows cache poisoning attacks
async def handle_message_insecure(self, data: bytes):
    msg = deserialize(data)

    # WRONG: Dedup check before signature verification
    if self._is_duplicate(msg.message_id):
        return None

    self._seen_messages.add(msg.message_id)  # Attacker can poison cache

    if not verify_signature(...):  # Too late!
        return None
```

**Rationale:** An attacker can send unsigned messages with valid IDs to poison the dedup cache, causing legitimate signed messages to be rejected.

### 1.4 Constant-Time Comparisons

**REQUIRED:** Use constant-time comparison for security-sensitive string comparisons.

```python
# CORRECT: Constant-time comparison
import hmac

def verify_policy(submitted_hash: str, expected_hash: str) -> bool:
    return hmac.compare_digest(submitted_hash, expected_hash)

# INCORRECT: Variable-time comparison enables timing attacks
def verify_policy_insecure(submitted_hash: str, expected_hash: str) -> bool:
    return submitted_hash == expected_hash  # VULNERABLE
```

**Rationale:** Timing attacks can reveal secret values through response time variations.

---

## 2. Input Validation Patterns

### 2.1 Public Key Validation

**REQUIRED:** Validate public key format before use.

```python
# CORRECT: Validate key before use
def register_peer(self, peer_id: str, public_key: str):
    if not public_key or len(public_key) < 64:
        raise ValueError(f"Invalid public key for peer {peer_id}")
    if not all(c in '0123456789abcdef' for c in public_key.lower()):
        raise ValueError("Public key must be hexadecimal")
    self._peer_keys[peer_id] = public_key

# INCORRECT: No validation
def register_peer_insecure(self, peer_id: str, public_key: str):
    self._peer_keys[peer_id] = public_key  # VULNERABLE
```

**Rationale:** Invalid keys can cause cryptographic failures or allow spoofing.

### 2.2 Threshold Validation

**REQUIRED:** Import thresholds from the authoritative constants module.

```python
# CORRECT: Use unified constants
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

def validate_pattern(pattern: ElevatedPattern) -> bool:
    if pattern.ihsan_score < UNIFIED_IHSAN_THRESHOLD:
        return False
    if pattern.snr_score < UNIFIED_SNR_THRESHOLD:
        return False
    return True

# INCORRECT: Hardcoded thresholds
IHSAN_THRESHOLD = 0.95  # WRONG - may drift from authoritative source
```

**Rationale:** Centralized threshold management prevents drift and ensures consistency.

### 2.3 JSON Canonicalization

**REQUIRED:** Use RFC 8785 canonicalization for all signed JSON.

```python
# CORRECT: RFC 8785 JCS
from core.pci.crypto import canonical_json

data = {"b": 2, "a": 1}
canonical = canonical_json(data)  # b'{"a":1,"b":2}'

# INCORRECT: Non-canonical JSON
import json
json.dumps(data)  # May produce '{"b": 2, "a": 1}' - wrong order
json.dumps(data, ensure_ascii=False)  # Unicode issues
```

**Rationale:** Signature verification requires identical byte sequences.

---

## 3. Cache Management Patterns

### 3.1 Bounded Cache with Eviction

**REQUIRED:** All caches MUST have size limits and eviction policies.

```python
# CORRECT: Bounded cache with eviction
class PatternCache:
    MAX_SIZE = 1000

    def __init__(self):
        self._cache: Dict[str, Pattern] = {}

    def add(self, pattern_id: str, pattern: Pattern):
        if len(self._cache) >= self.MAX_SIZE:
            self._evict_lowest_impact()
        self._cache[pattern_id] = pattern

    def _evict_lowest_impact(self):
        """Evict pattern with lowest impact score."""
        if not self._cache:
            return
        min_id = min(self._cache, key=lambda k: self._cache[k].impact_score)
        del self._cache[min_id]

# INCORRECT: Unbounded cache (DoS vulnerability)
class UnboundedCache:
    def __init__(self):
        self._cache = {}  # VULNERABLE - no limit

    def add(self, key, value):
        self._cache[key] = value  # Can exhaust memory
```

**Rationale:** Unbounded caches enable memory exhaustion DoS attacks.

### 3.2 TTL-Based Expiration

**REQUIRED:** Time-sensitive data MUST expire.

```python
# CORRECT: TTL-based nonce cache
class NonceCache:
    TTL_SECONDS = 300
    MAX_SIZE = 10000

    def __init__(self):
        self._nonces: Dict[str, float] = {}  # nonce -> timestamp
        self._last_prune = time.time()

    def check_and_add(self, nonce: str) -> bool:
        """Returns True if nonce is new, False if replay."""
        self._prune_if_needed()

        if nonce in self._nonces:
            return False  # Replay detected

        self._nonces[nonce] = time.time()
        return True

    def _prune_if_needed(self):
        """Remove expired nonces."""
        now = time.time()
        if now - self._last_prune < 60:
            return

        cutoff = now - self.TTL_SECONDS
        self._nonces = {n: t for n, t in self._nonces.items() if t > cutoff}

        # Emergency pruning if over limit
        if len(self._nonces) > self.MAX_SIZE:
            sorted_nonces = sorted(self._nonces.items(), key=lambda x: x[1])
            excess = len(self._nonces) - self.MAX_SIZE
            for nonce, _ in sorted_nonces[:excess]:
                del self._nonces[nonce]

        self._last_prune = now
```

**Rationale:** Expired nonces waste memory; TTL prevents indefinite accumulation.

---

## 4. Error Handling Patterns

### 4.1 Silent Rejection for Security

**REQUIRED:** Security failures should not leak information.

```python
# CORRECT: Silent rejection with logging
async def handle_message(self, data: bytes):
    try:
        msg = deserialize(data)
    except Exception:
        return None  # Silent rejection

    if not self._verify_signature(msg):
        # Log internally but don't reveal to attacker
        logger.warning(f"Invalid signature from {msg.sender_id}")
        return None  # No error response

    return await self._process(msg)

# INCORRECT: Detailed error responses
async def handle_message_verbose(self, data: bytes):
    msg = deserialize(data)
    if not self._verify_signature(msg):
        # WRONG: Leaks information about verification
        return {"error": "Signature verification failed", "expected_key": "..."}
```

**Rationale:** Detailed errors help attackers refine their attacks.

### 4.2 Specific Exception Handling

**REQUIRED:** Catch only expected exceptions in security code.

```python
# CORRECT: Specific exceptions
from cryptography.exceptions import InvalidSignature

def verify_signature(digest: str, signature: str, public_key: str) -> bool:
    try:
        pub_bytes = bytes.fromhex(public_key)
        sig_bytes = bytes.fromhex(signature)
        digest_bytes = bytes.fromhex(digest)

        key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        key.verify(sig_bytes, digest_bytes)
        return True
    except InvalidSignature:
        return False  # Expected: bad signature
    except ValueError:
        return False  # Expected: bad hex
    # Other exceptions (TypeError, MemoryError) propagate - don't hide bugs

# INCORRECT: Broad exception handling hides bugs
def verify_signature_insecure(digest, signature, public_key) -> bool:
    try:
        # ...
        return True
    except Exception:  # WRONG: hides programming errors
        return False
```

**Rationale:** Catching all exceptions hides bugs and makes debugging difficult.

---

## 5. Logging Patterns

### 5.1 Security-Sensitive Logging

**REQUIRED:** Never log secrets, but DO log security events.

```python
# CORRECT: Log security events without secrets
import logging
logger = logging.getLogger("SECURITY")

def authenticate_node(node_id: str, public_key: str) -> bool:
    # Log the attempt (not the key)
    logger.info(f"Authentication attempt from {node_id}")

    if not self._verify_key(public_key):
        # Log failure with details for investigation
        logger.warning(
            f"Authentication failed for {node_id}",
            extra={
                "key_prefix": public_key[:8] + "...",  # Only prefix
                "reason": "signature_mismatch",
            }
        )
        return False

    logger.info(f"Authentication successful for {node_id}")
    return True

# INCORRECT: Logging secrets
def authenticate_insecure(node_id: str, private_key: str):
    logger.debug(f"Using key: {private_key}")  # NEVER DO THIS
```

**Rationale:** Logs may be compromised; secrets in logs enable further attacks.

### 5.2 Structured Security Logs

**REQUIRED:** Use structured logging for security events.

```python
# CORRECT: Structured logging for SIEM integration
def log_security_event(event_type: str, details: dict):
    """Log security event with structured data."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "severity": details.get("severity", "INFO"),
        "node_id": self.node_id,
        **details,
    }
    logger.info(json.dumps(log_entry))

# Usage
log_security_event("SIGNATURE_VERIFICATION_FAILED", {
    "severity": "WARNING",
    "sender_id": msg.sender_id,
    "message_type": msg.msg_type,
    "ip_address": addr,
})
```

**Rationale:** Structured logs enable automated analysis and alerting.

---

## 6. Federation Patterns

### 6.1 Peer Registration

**REQUIRED:** All peers must be registered before participating in consensus.

```python
# CORRECT: Peer registration with validation
class ConsensusEngine:
    def __init__(self, node_id: str, private_key: str, public_key: str):
        self._peer_keys: Dict[str, str] = {}
        # Register self
        self._peer_keys[node_id] = public_key

    def register_peer(self, peer_id: str, public_key: str):
        """Register a peer's public key (required for voting)."""
        if not public_key or len(public_key) < 64:
            raise ValueError(f"Invalid public key for peer {peer_id}")
        self._peer_keys[peer_id] = public_key

    def receive_vote(self, vote: Vote) -> bool:
        # Verify peer is registered
        if vote.voter_id not in self._peer_keys:
            logger.error(f"Vote from unregistered peer: {vote.voter_id}")
            return False

        # Use REGISTERED key, not vote.public_key
        registered_key = self._peer_keys[vote.voter_id]
        if vote.public_key != registered_key:
            logger.error(f"Key mismatch for {vote.voter_id}")
            return False

        # Verify signature with registered key
        if not verify_signature(digest, vote.signature, registered_key):
            return False

        return True
```

**Rationale:** Prevents vote spoofing where attacker uses their own keypair with fake voter_id.

### 6.2 Quorum Calculation

**REQUIRED:** Use standard BFT quorum thresholds.

```python
# CORRECT: Standard BFT quorum
def calculate_quorum(node_count: int) -> int:
    """Calculate quorum for BFT consensus (2n/3 + 1)."""
    return (2 * node_count // 3) + 1

# Verification
assert calculate_quorum(3) == 3   # All nodes for n=3
assert calculate_quorum(4) == 3   # 3 of 4
assert calculate_quorum(7) == 5   # 5 of 7
assert calculate_quorum(10) == 7  # 7 of 10
```

**Rationale:** BFT requires 2f+1 votes where n >= 3f+1 to tolerate f Byzantine faults.

---

## 7. Constitutional AI Patterns

### 7.1 Multi-Dimensional Scoring

**REQUIRED:** Use geometric mean for FATE scoring to prevent single-dimension bypass.

```python
# CORRECT: Geometric mean prevents bypass
def compute_fate_score(dimensions: Dict[str, float]) -> float:
    """Compute FATE composite score using geometric mean."""
    fidelity = dimensions.get("fidelity", 1.0)
    accountability = dimensions.get("accountability", 1.0)
    transparency = dimensions.get("transparency", 1.0)
    ethics = dimensions.get("ethics", 1.0)

    # Geometric mean: if ANY dimension is 0, composite is 0
    composite = (fidelity * accountability * transparency * ethics) ** 0.25
    return composite

# INCORRECT: Arithmetic mean allows compensation
def compute_score_arithmetic(dimensions: Dict[str, float]) -> float:
    # WRONG: High scores in other dimensions can compensate for ethics=0
    return sum(dimensions.values()) / len(dimensions)
```

**Rationale:** Geometric mean ensures zero in any dimension results in zero composite.

### 7.2 Tool-Specific Thresholds

**REQUIRED:** Higher thresholds for risky tools.

```python
# CORRECT: Risk-based thresholds
TOOL_THRESHOLDS = {
    "Bash": 0.98,      # Command execution - highest risk
    "Write": 0.96,     # File modification - high risk
    "Edit": 0.96,      # Code changes - high risk
    "WebFetch": 0.95,  # External access - medium risk
    "Read": 0.90,      # Read-only - lower risk
}

def get_threshold(tool_name: str) -> float:
    return TOOL_THRESHOLDS.get(tool_name, 0.95)  # Default threshold
```

**Rationale:** More dangerous tools require higher confidence before execution.

---

## 8. Command Execution Patterns

### 8.1 Safe Subprocess Execution

**REQUIRED:** Use array-based arguments, never shell=True.

```python
# CORRECT: Array-based arguments
import subprocess

def run_nvidia_smi() -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ],
        capture_output=True,
        text=True,
        timeout=2,
        shell=False,  # Explicit for clarity
    )
    return parse_result(result.stdout)

# INCORRECT: Shell injection vulnerability
def run_command_insecure(user_input: str):
    # VULNERABLE: user_input could be "; rm -rf /"
    subprocess.run(f"echo {user_input}", shell=True)
```

**Rationale:** shell=True enables command injection via metacharacters.

### 8.2 Timeout and Resource Limits

**REQUIRED:** All subprocess calls must have timeouts.

```python
# CORRECT: With timeout
result = subprocess.run(
    ["nvidia-smi"],
    capture_output=True,
    timeout=2,  # 2-second timeout
)

# INCORRECT: No timeout (potential hang)
result = subprocess.run(["some-command"])  # VULNERABLE
```

**Rationale:** Subprocesses without timeouts can cause DoS via resource exhaustion.

---

## 9. Checklist for Security Review

### New Code Checklist

- [ ] Uses `generate_keypair()` from `core.pci`
- [ ] Uses `domain_separated_digest()` for all signatures
- [ ] Signature verification happens BEFORE caching
- [ ] Uses `hmac.compare_digest()` for security comparisons
- [ ] Public keys validated before use
- [ ] Thresholds imported from `core.integration.constants`
- [ ] Caches have MAX_SIZE limits
- [ ] Caches have TTL-based expiration
- [ ] Security exceptions are specific (not bare `except:`)
- [ ] Secrets are never logged
- [ ] Subprocess uses array args with shell=False
- [ ] Subprocess has timeout

### Security Review Criteria

| Criterion | Question | Required |
|-----------|----------|----------|
| Authentication | Is the caller verified? | YES |
| Authorization | Is the action permitted? | YES |
| Input Validation | Are all inputs validated? | YES |
| Output Encoding | Are outputs properly encoded? | YES |
| Cryptography | Are standard primitives used? | YES |
| Error Handling | Are errors handled securely? | YES |
| Logging | Are security events logged? | YES |
| Resource Limits | Are DoS protections in place? | YES |

---

## 10. References

- **RFC 8032:** Edwards-Curve Digital Signature Algorithm (EdDSA)
- **RFC 8785:** JSON Canonicalization Scheme (JCS)
- **BLAKE3 Specification:** https://github.com/BLAKE3-team/BLAKE3-specs
- **OWASP Secure Coding Practices:** https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/

---

*Document maintained by Security Architect Agent*
*Version: 1.0.0*
