# Phase 6: Legacy System Adapters

> ADR-006 | Unified Memory Service — Backward Compatibility
> Standing on Giants: Gamma et al. (Adapter pattern, GoF 1994) · Lamport (state preservation)

## 6.1 — `core/memory/adapters/living_memory.py`

### Requirements
- Wrap existing `LivingMemoryCore` (core/living_memory/core.py)
- Map `MemoryEntry` to `MemoryRecord` bidirectionally
- Support both read and write through the adapter
- Do NOT modify `LivingMemoryCore` — wrap only

### Pseudocode

```
IMPORT LivingMemoryCore, MemoryEntry FROM core.living_memory.core
IMPORT MemoryRecord, MemoryType, MemoryState FROM core.memory.types

CLASS LivingMemoryAdapter:
    """Wraps LivingMemoryCore for AgentDB integration."""

    name = "living_memory"
    read_only = False

    FUNCTION __init__(living_core: LivingMemoryCore):
        self._core = living_core

    FUNCTION search(query: str, top_k: int = 10) -> List[MemoryRecord]:
        """Search via LivingMemoryCore.retrieve() and convert results."""
        entries = self._core.retrieve(query=query, top_k=top_k)
        RETURN [self._entry_to_record(entry) FOR entry IN entries]

    FUNCTION get(id: str) -> Optional[MemoryRecord]:
        """Get by ID from LivingMemoryCore's internal store."""
        entry = self._core._entries.get(id)   # Direct access to internal dict
        IF entry IS None:
            RETURN None
        RETURN self._entry_to_record(entry)

    FUNCTION store(record: MemoryRecord) -> None:
        """Write a MemoryRecord back into LivingMemoryCore."""
        entry = self._record_to_entry(record)
        self._core.store(entry)

    # ── Mapping ──

    FUNCTION _entry_to_record(entry: MemoryEntry) -> MemoryRecord:
        RETURN MemoryRecord(
            id=entry.id,
            content=entry.content,
            memory_type=MemoryType(entry.memory_type.value),
            embedding=entry.embedding,
            created_at=entry.created_at,
            last_accessed=entry.last_accessed,
            access_count=entry.access_count,
            ihsan_score=entry.ihsan_score,
            snr_score=entry.snr_score,
            confidence=entry.confidence,
            state=MemoryState(entry.state.value),
            source=entry.source,
            related_ids=entry.related_ids,
            parent_id=entry.parent_id,
            importance=entry.importance,
            emotional_weight=entry.emotional_weight,
            adapter_source="living_memory",
        )

    FUNCTION _record_to_entry(record: MemoryRecord) -> MemoryEntry:
        # Reverse mapping for writes
        RETURN MemoryEntry(
            id=record.id,
            content=record.content,
            memory_type=LM_MemoryType(record.memory_type.value),
            embedding=record.embedding,
            # ... map all fields ...
        )
```

---

## 6.2 — `core/memory/adapters/experience_ledger.py`

### Requirements
- Wrap `ExperienceLedger` (core/sovereign/experience_ledger.py)
- **STRICTLY READ-ONLY** — the hash-chain MUST NOT be modified
- No write method exists; `store()` raises NotImplementedError
- Search returns recent experiences by keyword match

### Pseudocode

```
CLASS ExperienceLedgerAdapter:
    """
    Read-only adapter for the hash-chained Experience Ledger.

    CONSTITUTIONAL INVARIANT: This adapter has NO write path.
    The SEL's hash-chain integrity must never be compromised.
    Attempting to write raises NotImplementedError unconditionally.
    """

    name = "experience_ledger"
    read_only = True  # LOCKED — cannot be changed

    FUNCTION __init__(ledger_path: Path):
        self._path = ledger_path

    FUNCTION search(query: str, top_k: int = 10) -> List[MemoryRecord]:
        """Search ledger entries by keyword match (no vector search)."""
        IF NOT self._path.exists():
            RETURN []

        matches = []
        FOR line IN self._path.read_text().strip().split("\n"):
            TRY:
                entry = json.loads(line)
                content = entry.get("content", "") OR entry.get("description", "")
                # Simple keyword match (no FTS5 for JSONL)
                IF query.lower() IN content.lower():
                    record = self._entry_to_record(entry)
                    matches.append(record)
            EXCEPT json.JSONDecodeError:
                CONTINUE

        # Sort by timestamp descending, return top_k
        matches.sort(key=LAMBDA r: r.created_at, reverse=True)
        RETURN matches[:top_k]

    FUNCTION get(id: str) -> Optional[MemoryRecord]:
        """Retrieve a specific ledger entry by ID."""
        FOR line IN self._path.read_text().strip().split("\n"):
            TRY:
                entry = json.loads(line)
                IF entry.get("id") == id OR entry.get("hash") == id:
                    RETURN self._entry_to_record(entry)
            EXCEPT:
                CONTINUE
        RETURN None

    FUNCTION store(*args, **kwargs):
        """BLOCKED — SEL is read-only. Hash-chain integrity is constitutional."""
        RAISE NotImplementedError(
            "ExperienceLedger is read-only. "
            "Hash-chain integrity is a constitutional invariant."
        )

    FUNCTION _entry_to_record(entry: Dict) -> MemoryRecord:
        RETURN MemoryRecord(
            id=entry.get("hash", entry.get("id", "")),
            content=entry.get("content", entry.get("description", "")),
            memory_type=MemoryType.EPISODIC,  # All SEL entries are episodic
            ihsan_score=entry.get("ihsan_score", 1.0),
            source=entry.get("source", "experience_ledger"),
            adapter_source="sel",
        )
```

---

## 6.3 — `core/memory/adapters/pattern_memory.py`

### Requirements
- Bridge to Rust autopoiesis PatternMemory via PyO3 (optional)
- Graceful fallback to no-op stub if `bizra_python` not available
- Read-only for now; Rust side owns mutation

### Pseudocode

```
TRY:
    IMPORT bizra_python   # PyO3 bindings from bizra-omega
    PATTERN_MEMORY_AVAILABLE = True
EXCEPT ImportError:
    PATTERN_MEMORY_AVAILABLE = False

CLASS PatternMemoryAdapter:
    """
    Adapter for Rust autopoiesis PatternMemory.

    Uses PyO3 bindings when available, falls back to no-op stub.
    Read-only from Python side — Rust owns pattern mutation.
    """

    name = "pattern_memory"
    read_only = True

    FUNCTION __init__():
        IF PATTERN_MEMORY_AVAILABLE:
            self._backend = bizra_python.PatternMemory()
        ELSE:
            self._backend = None
            logger.info("PatternMemory: PyO3 bindings not available, using no-op stub")

    FUNCTION search(query: str, top_k: int = 10) -> List[MemoryRecord]:
        IF self._backend IS None:
            RETURN []
        raw_results = self._backend.search(query, top_k)
        RETURN [self._to_record(r) FOR r IN raw_results]

    FUNCTION get(id: str) -> Optional[MemoryRecord]:
        IF self._backend IS None:
            RETURN None
        raw = self._backend.get(id)
        IF raw IS None:
            RETURN None
        RETURN self._to_record(raw)

    FUNCTION store(*args, **kwargs):
        RAISE NotImplementedError("PatternMemory is read-only from Python side")

    FUNCTION _to_record(raw: Dict) -> MemoryRecord:
        RETURN MemoryRecord(
            id=raw["id"],
            content=raw.get("pattern", ""),
            memory_type=MemoryType.PROCEDURAL,
            importance=raw.get("fitness", 0.5),
            adapter_source="pattern",
        )
```

## 6.4 — `core/memory/adapters/__init__.py`

```
FROM .living_memory IMPORT LivingMemoryAdapter
FROM .experience_ledger IMPORT ExperienceLedgerAdapter
FROM .pattern_memory IMPORT PatternMemoryAdapter

__all__ = [
    "LivingMemoryAdapter",
    "ExperienceLedgerAdapter",
    "PatternMemoryAdapter",
]
```

### TDD Anchors

```
TEST test_living_memory_adapter_roundtrip():
    core = LivingMemoryCore(max_entries=100)
    adapter = LivingMemoryAdapter(core)
    # Store through core, read through adapter
    entry = MemoryEntry(id="test", content="hello", memory_type=LM_MemoryType.SEMANTIC)
    core.store(entry)
    record = adapter.get("test")
    ASSERT record IS NOT None
    ASSERT record.content == "hello"
    ASSERT record.adapter_source == "living_memory"

TEST test_sel_adapter_read_only():
    adapter = ExperienceLedgerAdapter(ledger_path)
    WITH pytest.raises(NotImplementedError):
        adapter.store("anything")

TEST test_sel_adapter_search(tmp_path):
    ledger = tmp_path / "evidence.jsonl"
    ledger.write_text('{"id":"1","content":"quantum discovery","hash":"abc"}\n')
    adapter = ExperienceLedgerAdapter(ledger)
    results = adapter.search("quantum")
    ASSERT len(results) == 1
    ASSERT results[0].adapter_source == "sel"

TEST test_pattern_memory_fallback():
    # Without PyO3 bindings, should return empty results
    adapter = PatternMemoryAdapter()
    IF NOT PATTERN_MEMORY_AVAILABLE:
        ASSERT adapter.search("test") == []
        ASSERT adapter.get("any") IS None

TEST test_adapter_protocol_compliance():
    FOR AdapterCls IN [LivingMemoryAdapter, ExperienceLedgerAdapter, PatternMemoryAdapter]:
        ASSERT hasattr(AdapterCls, "name")
        ASSERT hasattr(AdapterCls, "read_only")
        ASSERT hasattr(AdapterCls, "search")
        ASSERT hasattr(AdapterCls, "get")
```

### Constitutional Invariant Check

```
TEST test_sel_has_no_write_path():
    """Verify that ExperienceLedgerAdapter has absolutely no write capability."""
    adapter = ExperienceLedgerAdapter(Path("/dev/null"))
    ASSERT adapter.read_only IS True
    # Verify NO method can write
    FOR method_name IN dir(adapter):
        IF method_name.startswith("_"):
            CONTINUE
        method = getattr(adapter, method_name)
        IF callable(method) AND method_name IN ["store", "save", "write", "append", "update", "delete"]:
            WITH pytest.raises((NotImplementedError, AttributeError)):
                method("test")
```
