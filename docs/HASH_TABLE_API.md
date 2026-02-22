# BIZRA Hash Table Infrastructure — API Reference

**Module:** `core.hashtable` | **Phase:** 44 | **Version:** 1.0.0
**Standing on Giants:** Bloom (1970) · Merkle (1979) · Kirsch & Mitzenmacher (2006) · Kahneman (2011) · Anderson (1982)

---

## Quick Start

```python
from core.hashtable import BloomFilter, MerkleTree, SkillCache

# Bloom filter — probabilistic membership for federation gossip
bf = BloomFilter(expected_items=10_000)
bf.add(b"node-alpha")
assert b"node-alpha" in bf

# Merkle tree — cryptographic inclusion proofs
tree = MerkleTree()
idx = tree.append(b"transaction-001")
proof = tree.prove(idx)
assert proof.verify()

# Skill cache — System 2→1 compression
cache = SkillCache()
key = cache.structural_hash([{"type": "observe"}, {"type": "hypothesize"}])
cache.put(key, {"answer": "42"}, snr_score=0.95)
result = cache.get(key)
```

---

## BloomFilter

**Import:** `from core.hashtable import BloomFilter, BloomFilterSaturatedError`
**Source:** `core/hashtable/bloom_filter.py`

A space-efficient probabilistic set membership filter. Uses BLAKE3 double hashing (Kirsch-Mitzenmacher optimization) for O(k) add/query operations.

**Key properties:**
- False positives possible (bounded by `false_positive_rate` parameter)
- False negatives impossible
- Merge via bitwise OR for federation gossip sharing
- Wire serialization for network transport

### Constructor

```python
BloomFilter(expected_items: int, false_positive_rate: float = 0.01)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_items` | `int` | required | Expected number of items to store |
| `false_positive_rate` | `float` | `0.01` | Target FPR (from `BLOOM_DEFAULT_FPR`) |

Raises `ValueError` if computed bit array exceeds `BLOOM_MAX_BITS` (10,000,000).

### Methods

#### `add(item: bytes) -> None`

Add an item to the filter.

```python
bf = BloomFilter(1000)
bf.add(b"hello")
bf.add(b"world")
```

Raises `BloomFilterSaturatedError` if `estimated_count() > expected_items * 2`. A saturated filter has meaningless membership results.

#### `__contains__(item: bytes) -> bool`

Test probable membership. May return false positive, never false negative.

```python
bf.add(b"hello")
assert b"hello" in bf      # Always True (no false negatives)
assert b"xyz" not in bf     # Probably True (small FPR chance of False)
```

#### `estimated_count() -> int`

Return the number of items added to the filter.

#### `false_positive_probability() -> float`

Estimated FPR given current fill level. Uses the formula `(1 - e^(-kn/m))^k`.

```python
bf = BloomFilter(1000, false_positive_rate=0.01)
for i in range(500):
    bf.add(str(i).encode())
print(bf.false_positive_probability())  # ~0.0003 (well below 0.01)
```

#### `merge(other: BloomFilter) -> BloomFilter`

Merge two Bloom filters via bitwise OR. Both filters must have identical parameters (same `expected_items` and `false_positive_rate`).

```python
# Node A's seen set
bf_a = BloomFilter(1000)
bf_a.add(b"node-alpha-data")

# Node B's seen set
bf_b = BloomFilter(1000)
bf_b.add(b"node-beta-data")

# Merged set — contains both
bf_merged = bf_a.merge(bf_b)
assert b"node-alpha-data" in bf_merged
assert b"node-beta-data" in bf_merged
```

Raises `ValueError` if parameters don't match.

#### `to_bytes() -> bytes`

Serialize to wire format with magic bytes, version, and parameters.

#### `from_bytes(data: bytes) -> BloomFilter` (classmethod)

Deserialize from wire format. Validates magic bytes and version.

```python
wire = bf.to_bytes()
bf_restored = BloomFilter.from_bytes(wire)
assert b"hello" in bf_restored
```

---

## MerkleTree

**Import:** `from core.hashtable import MerkleTree, MerkleProof`
**Source:** `core/hashtable/merkle_tree.py`

A cryptographic hash tree using BLAKE3 with RFC 6962 domain separation. Supports O(log n) incremental append via right-spine caching.

**Key properties:**
- Leaf prefix `\x00`, internal node prefix `\x01` (prevents second-preimage attacks)
- Cross-language compatible with Rust `merkle_root_0g` in bizra-omega
- O(log n) append, O(log n) proof generation

### Constructor

```python
MerkleTree(leaves: list[bytes] | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `leaves` | `list[bytes] \| None` | `None` | Initial leaf data. Each entry is hashed as a leaf. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `root` | `bytes` | 32-byte Merkle root hash |
| `root_hex` | `str` | 64-character hex string of root |
| `leaf_count` | `int` | Number of leaves in the tree |

### Methods

#### `append(data: bytes) -> int`

Append a leaf and update the right spine in O(log n). Returns the leaf index.

```python
tree = MerkleTree()
idx0 = tree.append(b"transaction-001")  # returns 0
idx1 = tree.append(b"transaction-002")  # returns 1
print(tree.root_hex)  # 64-char hex string
```

#### `prove(leaf_index: int) -> MerkleProof`

Generate an inclusion proof for the leaf at the given index.

```python
proof = tree.prove(0)
assert proof.verify()  # True — leaf is in the tree
```

Raises `IndexError` if `leaf_index` is out of range.

#### `verify_proof(proof: MerkleProof) -> bool` (static)

Verify a proof against its embedded root. Equivalent to `proof.verify()`.

### MerkleProof

Frozen dataclass with self-contained verification.

| Field | Type | Description |
|-------|------|-------------|
| `leaf_index` | `int` | Index of the proven leaf |
| `leaf_hash` | `bytes` | Hash of the leaf data |
| `siblings` | `tuple[tuple[bytes, bool], ...]` | Bottom-up sibling hashes with position flags |
| `root` | `bytes` | Expected Merkle root |

#### `verify() -> bool`

Walk the sibling path from leaf to root and compare against the embedded root.

```python
proof = tree.prove(0)

# Proof is self-contained — can be verified independently
assert proof.verify()

# Tamper detection — modifying any field causes verification failure
import dataclasses
tampered = dataclasses.replace(proof, leaf_index=999)
assert not tampered.verify()
```

---

## SkillCache

**Import:** `from core.hashtable import SkillCache, CachedSkillResult`
**Source:** `core/hashtable/skill_cache.py`

An LRU cache that compresses System 2 (deliberative) thought chains into System 1 (automatic) cached results via structural hashing. Thread-safe.

**Key properties:**
- Structural hash using `canonical_bytes()` + BLAKE3 (order-preserving)
- LRU eviction via `OrderedDict`
- TTL expiry (lazy eviction on `get()`)
- Ihsan floor: cached results below `UNIFIED_IHSAN_THRESHOLD` are auto-evicted
- Thread-safe via `threading.Lock`

### Constructor

```python
SkillCache(
    max_size: int = 256,
    default_ttl: int = 3600,
    ihsan_floor: float = 0.95
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_size` | `int` | `256` | Maximum cached entries (from `SKILL_CACHE_MAX_SIZE`) |
| `default_ttl` | `int` | `3600` | Time-to-live in seconds (from `SKILL_CACHE_DEFAULT_TTL`) |
| `ihsan_floor` | `float` | `0.95` | Minimum SNR score for cache retention (from `UNIFIED_IHSAN_THRESHOLD`) |

### Methods

#### `structural_hash(thought_chain: list[dict]) -> str`

Compute a deterministic structural hash of a thought chain. Two chains with the same structure and content produce the same hash. Order is preserved.

```python
cache = SkillCache()
key = cache.structural_hash([
    {"type": "observe", "target": "filesystem"},
    {"type": "hypothesize", "claim": "files are organized by date"},
    {"type": "test", "method": "list_directory"},
])
# Returns: 16-character hex string (e.g., "a3f8c1d2e5b74901")
```

#### `put(key: str, result: dict, snr_score: float) -> None`

Store a result in the cache. Evicts LRU entry if at capacity.

```python
cache.put(key, {"answer": "organized by date", "confidence": 0.97}, snr_score=0.96)
```

#### `get(key: str) -> CachedSkillResult | None`

Retrieve a cached result. Returns `None` if:
- Key not found
- Entry expired (TTL exceeded)
- Entry below Ihsan floor (`snr_score < ihsan_floor`)

```python
result = cache.get(key)
if result:
    print(result.result)       # {"answer": "organized by date", ...}
    print(result.snr_score)    # 0.96
    print(result.hit_count)    # Incremented on each get()
```

#### `invalidate(key: str) -> bool`

Remove a specific key. Returns `True` if it existed.

#### `stats() -> dict`

Return cache statistics.

```python
s = cache.stats()
# {
#     "size": 42,
#     "max_size": 256,
#     "hits": 150,
#     "misses": 30,
#     "evictions": 5,
#     "hit_rate": 0.833,
#     "fill_ratio": 0.164,
# }
```

### CachedSkillResult

Frozen dataclass returned by `get()`.

| Field | Type | Description |
|-------|------|-------------|
| `structural_hash` | `str` | 16-char hex hash of the thought chain |
| `query_pattern` | `str` | Human-readable pattern label |
| `result` | `dict` | The cached computation result |
| `snr_score` | `float` | SNR score at time of caching |
| `created_at` | `float` | Unix timestamp of creation |
| `ttl_seconds` | `int` | TTL in seconds |
| `hit_count` | `int` | Number of cache hits |
| `last_hit` | `float` | Unix timestamp of last hit |

---

## Constants

All constants are defined in `core/integration/constants.py`:

| Constant | Value | Used By |
|----------|-------|---------|
| `BLOOM_DEFAULT_FPR` | `0.01` | BloomFilter default FPR |
| `BLOOM_MAX_BITS` | `10,000,000` | BloomFilter max bit array (~1.2 MB) |
| `MERKLE_LEAF_PREFIX` | `b"\x00"` | MerkleTree leaf domain separation |
| `MERKLE_NODE_PREFIX` | `b"\x01"` | MerkleTree node domain separation |
| `SKILL_CACHE_MAX_SIZE` | `256` | SkillCache default capacity |
| `SKILL_CACHE_DEFAULT_TTL` | `3600` | SkillCache default TTL (1 hour) |

---

## Dependencies

Zero new pip dependencies. Uses only:

| Dependency | Source | Used For |
|-----------|--------|----------|
| `blake3` | `pyproject.toml` (existing) | All hashing |
| `threading` | stdlib | SkillCache thread safety |
| `collections.OrderedDict` | stdlib | SkillCache LRU |
| `math` | stdlib | Bloom filter sizing |
| `struct` | stdlib | Bloom filter serialization |
| `core.proof_engine.canonical` | existing module | `blake3_digest()`, `canonical_bytes()` |

---

## Test Coverage

92 tests across 3 test files:

```bash
pytest tests/core/hashtable/ -v
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_bloom_filter.py` | ~30 | FPR bounds, merge, serialization, saturation guard |
| `test_merkle_tree.py` | ~35 | Root determinism, proofs, tamper detection, domain separation |
| `test_skill_cache.py` | ~27 | LRU eviction, TTL, Ihsan floor, thread safety, stats |
