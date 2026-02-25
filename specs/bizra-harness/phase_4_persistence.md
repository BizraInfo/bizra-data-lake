# Phase 4 — State Persistence and Regression Detection

> Standing on Giants: Lamport (append-only logs, 1978) · Deming (statistical
> process control, 1950) · Nakamoto (hash-chained history, 2008)

## Overview

Every harness run must be recorded. Every baseline must be sealed. Regression
detection compares the current run against the most recent sealed baseline.
The storage format is append-only JSONL — one line per event, hash-chained.

## File: `core/harness/persistence.py`

```pseudocode
IMPORTS:
    import hashlib, json
    from pathlib import Path
    from core.harness.types import HarnessResult, RegressionReport, PillarName

# ── Constants ──────────────────────────────────────────────────────

CONSTANT SPEARPOINT_DIR = Path(".spearpoint")
CONSTANT RUNS_FILE      = SPEARPOINT_DIR / "harness_runs.jsonl"
CONSTANT BASELINES_FILE = SPEARPOINT_DIR / "baselines.jsonl"
CONSTANT MAX_RUNS_KEPT  = 500   # rolling window for disk hygiene

# ── Run Entry (JSONL record) ──────────────────────────────────────

@dataclass
CLASS RunEntry:
    """A single recorded harness run."""
    run_id:        str
    timestamp:     str         # ISO 8601
    verdict:       str         # "pass" | "fail" | "inconclusive"
    snr_score:     float
    ihsan_score:   float
    tier:          str
    mode:          str
    claim:         str
    pillars:       dict[str, bool]   # {pillar_name: passed}
    duration_ms:   float
    prev_hash:     str               # SHA-256 of previous entry (chain)
    entry_hash:    str               # SHA-256 of this entry (chain)

    @staticmethod
    METHOD from_result(result: HarnessResult, prev_hash: str) -> 'RunEntry':
        """Convert HarnessResult to a persistable entry."""
        content = {
            "run_id":     result.run_id,
            "timestamp":  result.timestamp.isoformat(),
            "verdict":    result.verdict.value,
            "snr_score":  round(result.snr_score, 6),
            "ihsan_score": round(result.ihsan_score, 6),
            "tier":       result.tier,
            "mode":       result.config.mode.value,
            "claim":      result.config.claim,
            "pillars":    result.pillar_summary,
            "duration_ms": round(result.total_duration_ms, 1),
            "prev_hash":  prev_hash,
        }
        # Hash the content deterministically
        canonical = json.dumps(content, sort_keys=True, separators=(',', ':'))
        entry_hash = hashlib.sha256(canonical.encode()).hexdigest()
        RETURN RunEntry(**content, entry_hash=entry_hash)

    METHOD to_json_line(self) -> str:
        """Serialize to a single JSONL line."""
        d = dataclasses.asdict(self)
        RETURN json.dumps(d, sort_keys=True, separators=(',', ':'))

    @staticmethod
    METHOD from_json_line(line: str) -> 'RunEntry':
        """Deserialize from a JSONL line."""
        d = json.loads(line)
        RETURN RunEntry(**d)

# ── Baseline Entry ─────────────────────────────────────────────────

@dataclass
CLASS BaselineEntry:
    """A sealed baseline — a run declared as the reference point."""
    run_id:        str
    sealed_at:     str           # ISO 8601 when sealed
    snr_score:     float
    ihsan_score:   float
    pillars:       dict[str, bool]
    seal_hash:     str           # SHA-256 of the sealed content

    @staticmethod
    METHOD from_result(result: HarnessResult) -> 'BaselineEntry':
        content = {
            "run_id":      result.run_id,
            "sealed_at":   datetime.now(timezone.utc).isoformat(),
            "snr_score":   round(result.snr_score, 6),
            "ihsan_score": round(result.ihsan_score, 6),
            "pillars":     result.pillar_summary,
        }
        canonical = json.dumps(content, sort_keys=True, separators=(',', ':'))
        seal_hash = hashlib.sha256(canonical.encode()).hexdigest()
        RETURN BaselineEntry(**content, seal_hash=seal_hash)

# ── BaselineStore ──────────────────────────────────────────────────

CLASS BaselineStore:
    """Append-only store for harness runs and sealed baselines.

    Storage: .spearpoint/harness_runs.jsonl (all runs, hash-chained)
             .spearpoint/baselines.jsonl    (sealed baselines)

    Standing on Giants: Lamport — append-only log is the simplest
    structure that provides total ordering and tamper evidence.
    """

    METHOD __init__(self, base_dir: Optional[Path] = None):
        self._dir = base_dir or SPEARPOINT_DIR
        self._runs_path = self._dir / "harness_runs.jsonl"
        self._baselines_path = self._dir / "baselines.jsonl"

    @classmethod
    METHOD default(cls) -> 'BaselineStore':
        RETURN cls()

    METHOD _ensure_dir(self):
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- Run History -------------------------------------------------

    METHOD append_run(self, result: HarnessResult) -> RunEntry:
        """Append a run to the hash-chained log."""
        self._ensure_dir()

        # Get previous hash (tail of chain)
        prev_hash = self._last_run_hash()

        entry = RunEntry.from_result(result, prev_hash)
        WITH open(self._runs_path, "a") AS f:
            f.write(entry.to_json_line() + "\n")

        # Trim if over MAX_RUNS_KEPT
        self._trim_runs()

        RETURN entry

    METHOD _last_run_hash(self) -> str:
        """Get the hash of the last run entry (genesis if empty)."""
        IF NOT self._runs_path.exists():
            RETURN "genesis"
        TRY:
            # Read last line efficiently
            WITH open(self._runs_path, "rb") AS f:
                f.seek(0, 2)     # end
                pos = f.tell()
                IF pos == 0:
                    RETURN "genesis"
                # Scan backwards for newline
                WHILE pos > 0:
                    pos -= 1
                    f.seek(pos)
                    IF f.read(1) == b'\n' AND pos < f.seek(0, 2) - 1:
                        BREAK
                last_line = f.readline().decode().strip()
            IF NOT last_line:
                RETURN "genesis"
            entry = RunEntry.from_json_line(last_line)
            RETURN entry.entry_hash
        EXCEPT Exception:
            RETURN "genesis"

    METHOD _trim_runs(self):
        """Keep only the last MAX_RUNS_KEPT entries."""
        IF NOT self._runs_path.exists():
            RETURN
        lines = self._runs_path.read_text().strip().split("\n")
        IF len(lines) > MAX_RUNS_KEPT:
            # Keep last N lines
            trimmed = lines[-MAX_RUNS_KEPT:]
            self._runs_path.write_text("\n".join(trimmed) + "\n")

    METHOD get_run_history(self, last_n: int = 10) -> list[RunEntry]:
        """Read the last N runs from the log."""
        IF NOT self._runs_path.exists():
            RETURN []
        lines = self._runs_path.read_text().strip().split("\n")
        entries = []
        FOR line IN lines[-last_n:]:
            IF line.strip():
                TRY:
                    entries.append(RunEntry.from_json_line(line))
                EXCEPT Exception:
                    CONTINUE    # skip corrupt lines
        RETURN entries

    METHOD verify_chain(self) -> tuple[bool, int, Optional[str]]:
        """Verify the hash chain integrity.

        Returns: (valid, entries_checked, first_error)
        """
        IF NOT self._runs_path.exists():
            RETURN (True, 0, None)

        lines = self._runs_path.read_text().strip().split("\n")
        prev_hash = "genesis"
        checked = 0

        FOR line IN lines:
            IF NOT line.strip():
                CONTINUE
            TRY:
                entry = RunEntry.from_json_line(line)
            EXCEPT Exception as exc:
                RETURN (False, checked, f"Parse error at line {checked}: {exc}")

            IF entry.prev_hash != prev_hash:
                RETURN (False, checked,
                    f"Chain break at {entry.run_id}: "
                    f"expected prev_hash={prev_hash}, got {entry.prev_hash}")

            # Verify self-hash
            content = {k: v for k, v in dataclasses.asdict(entry).items()
                       if k != "entry_hash"}
            canonical = json.dumps(content, sort_keys=True, separators=(',', ':'))
            expected_hash = hashlib.sha256(canonical.encode()).hexdigest()
            IF entry.entry_hash != expected_hash:
                RETURN (False, checked,
                    f"Hash mismatch at {entry.run_id}: "
                    f"expected {expected_hash}, got {entry.entry_hash}")

            prev_hash = entry.entry_hash
            checked += 1

        RETURN (True, checked, None)

    # -- Baselines ---------------------------------------------------

    METHOD seal_baseline(self, result: HarnessResult) -> BaselineEntry:
        """Seal a run as the new reference baseline."""
        self._ensure_dir()
        entry = BaselineEntry.from_result(result)
        WITH open(self._baselines_path, "a") AS f:
            d = dataclasses.asdict(entry)
            f.write(json.dumps(d, sort_keys=True, separators=(',', ':')) + "\n")
        RETURN entry

    METHOD get_latest(self) -> Optional[BaselineEntry]:
        """Get the most recently sealed baseline."""
        IF NOT self._baselines_path.exists():
            RETURN None
        lines = self._baselines_path.read_text().strip().split("\n")
        IF NOT lines OR NOT lines[-1].strip():
            RETURN None
        TRY:
            d = json.loads(lines[-1])
            RETURN BaselineEntry(**d)
        EXCEPT Exception:
            RETURN None

    METHOD get_all_baselines(self) -> list[BaselineEntry]:
        """Load all sealed baselines."""
        IF NOT self._baselines_path.exists():
            RETURN []
        entries = []
        FOR line IN self._baselines_path.read_text().strip().split("\n"):
            IF line.strip():
                TRY:
                    entries.append(BaselineEntry(**json.loads(line)))
                EXCEPT Exception:
                    CONTINUE
        RETURN entries
```

## Hash Chain Diagram

```
harness_runs.jsonl:

  Entry 0: prev_hash="genesis"          hash=SHA256(entry0)
       │
  Entry 1: prev_hash=hash(entry0)       hash=SHA256(entry1)
       │
  Entry 2: prev_hash=hash(entry1)       hash=SHA256(entry2)
       │
      ...
```

If any entry is tampered with, `verify_chain()` detects the break.

## TDD Anchors

```python
# test_persistence.py — Phase 4 validation

def test_append_run_creates_file(tmp_path):
    store = BaselineStore(base_dir=tmp_path)
    result = _make_minimal_result(verdict=Verdict.PASS, snr=0.92)
    entry = store.append_run(result)
    assert (tmp_path / "harness_runs.jsonl").exists()
    assert entry.prev_hash == "genesis"   # first entry
    assert len(entry.entry_hash) == 64     # SHA-256 hex

def test_hash_chain_integrity(tmp_path):
    store = BaselineStore(base_dir=tmp_path)
    r1 = _make_minimal_result(run_id="r1", snr=0.90)
    r2 = _make_minimal_result(run_id="r2", snr=0.92)
    e1 = store.append_run(r1)
    e2 = store.append_run(r2)
    assert e2.prev_hash == e1.entry_hash   # chain linked
    valid, checked, err = store.verify_chain()
    assert valid is True
    assert checked == 2

def test_chain_detects_tamper(tmp_path):
    store = BaselineStore(base_dir=tmp_path)
    store.append_run(_make_minimal_result(run_id="r1"))
    store.append_run(_make_minimal_result(run_id="r2"))
    # Tamper with line 1
    path = tmp_path / "harness_runs.jsonl"
    lines = path.read_text().split("\n")
    tampered = json.loads(lines[0])
    tampered["snr_score"] = 0.999
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(',', ':'))
    path.write_text("\n".join(lines))
    valid, checked, err = store.verify_chain()
    assert valid is False
    assert "mismatch" in err.lower() or "break" in err.lower()

def test_seal_baseline(tmp_path):
    store = BaselineStore(base_dir=tmp_path)
    result = _make_minimal_result(snr=0.93, ihsan=0.97)
    entry = store.seal_baseline(result)
    assert len(entry.seal_hash) == 64
    latest = store.get_latest()
    assert latest is not None
    assert latest.snr_score == 0.93

def test_get_latest_returns_none_when_empty(tmp_path):
    store = BaselineStore(base_dir=tmp_path)
    assert store.get_latest() is None

def test_trim_runs(tmp_path):
    store = BaselineStore(base_dir=tmp_path)
    # Append more than MAX_RUNS_KEPT
    for i in range(MAX_RUNS_KEPT + 50):
        store.append_run(_make_minimal_result(run_id=f"r{i}"))
    lines = (tmp_path / "harness_runs.jsonl").read_text().strip().split("\n")
    assert len(lines) <= MAX_RUNS_KEPT

def test_get_run_history(tmp_path):
    store = BaselineStore(base_dir=tmp_path)
    for i in range(5):
        store.append_run(_make_minimal_result(run_id=f"r{i}"))
    history = store.get_run_history(last_n=3)
    assert len(history) == 3
    assert history[-1].run_id == "r4"   # most recent

def test_run_entry_from_result():
    result = _make_minimal_result(snr=0.91)
    entry = RunEntry.from_result(result, prev_hash="abc123")
    assert entry.prev_hash == "abc123"
    assert entry.snr_score == 0.91
    line = entry.to_json_line()
    roundtrip = RunEntry.from_json_line(line)
    assert roundtrip.run_id == entry.run_id
    assert roundtrip.entry_hash == entry.entry_hash
```

## Storage Layout

```
.spearpoint/
├── harness_runs.jsonl      # Append-only, hash-chained, trimmed at 500
├── baselines.jsonl         # Append-only, sealed reference points
└── scenarios.json          # Optional user-defined scenarios (Phase 3)
```
