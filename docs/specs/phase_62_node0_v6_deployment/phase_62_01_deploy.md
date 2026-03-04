# Phase 62 D1: Deploy v6 Modules into Workspace

## Scope

Copy 6 new Python modules, 5 new test files, poi.proto, verify_all.py, and
2 documentation files from `/tmp/bizra-node0-v6/` into `bizra-constitution/`.
Backport the `get_all_entries()` method to `evidence_receipt.py`.

## File Manifest

### New Modules (copy to `bizra-constitution/`)

```
identity_genesis.py      348 lines  Ed25519 keypair, HD agent keys, domain signing
ollama_provider.py       486 lines  Circuit breaker, model fallback, health metrics
production_pipeline.py   221 lines  Signed evidence, real identity + inference
node0_server.py          419 lines  FastAPI: /mission, /health, /evidence, /identity
node0_wire.py            352 lines  Adapter for MissionOrchestrator integration
verify_all.py            153 lines  Single-command constitution verification
poi.proto                193 lines  Protobuf: tensor PoI attestation + Pool messages
MIGRATION.md             226 lines  Integration guide (reference)
WIRE_GUIDE.md            196 lines  Wire adapter integration guide (reference)
```

### New Tests (copy to `bizra-constitution/tests/`)

```
test_identity_genesis.py     226 lines  35 tests (4 skip if no PyNaCl)
test_ollama_provider.py      261 lines  23 tests (mocked I/O)
test_production_pipeline.py  200 lines  20 tests
test_node0_server.py         158 lines  29 tests (FastAPI TestClient)
test_node0_wire.py           217 lines  29 tests
```

### Modified (backport diff to existing file)

```
evidence_receipt.py  +16 lines  Adds get_all_entries() method at end of class
```

## Pseudocode

```
PROCEDURE deploy_v6_modules:
    SOURCE := "/tmp/bizra-node0-v6"
    TARGET := "/mnt/c/BIZRA-DATA-LAKE/bizra-constitution"

    # 1. Copy new modules
    FOR EACH file IN [identity_genesis.py, ollama_provider.py,
                      production_pipeline.py, node0_server.py,
                      node0_wire.py, verify_all.py, poi.proto,
                      MIGRATION.md, WIRE_GUIDE.md]:
        COPY SOURCE/file → TARGET/file

    # 2. Copy new tests
    FOR EACH test IN [test_identity_genesis.py, test_ollama_provider.py,
                      test_production_pipeline.py, test_node0_server.py,
                      test_node0_wire.py]:
        COPY SOURCE/tests/test → TARGET/tests/test

    # 3. Backport evidence_receipt.py
    APPEND get_all_entries() method to TARGET/evidence_receipt.py
    # Method: load all JSONL lines, parse each as EvidenceReceipt

    # 4. Verify file count
    ASSERT count(TARGET/*.py) >= 12   # 6 v5 + 6 v6
    ASSERT count(TARGET/tests/*.py) >= 10  # 5 v5 + 5 v6
```

## TDD Anchors

```python
# test_d1_deploy_verification.py (post-deploy smoke)

def test_all_v6_modules_importable():
    """All 6 new modules import without error."""
    import identity_genesis
    import ollama_provider
    import production_pipeline
    import node0_server
    import node0_wire
    # verify_all is a script, not importable module

def test_evidence_receipt_has_get_all_entries():
    """Backported method exists."""
    from evidence_receipt import EvidenceLedger
    assert hasattr(EvidenceLedger, 'get_all_entries')

def test_poi_proto_exists():
    """Proto file is present."""
    from pathlib import Path
    assert (Path("bizra-constitution") / "poi.proto").exists()

def test_v6_file_count():
    """Expected file count after deployment."""
    import glob
    modules = glob.glob("bizra-constitution/*.py")
    tests = glob.glob("bizra-constitution/tests/test_*.py")
    assert len(modules) >= 12
    assert len(tests) >= 10
```

## Acceptance

- [ ] All 9 new files present in `bizra-constitution/`
- [ ] All 5 new test files present in `bizra-constitution/tests/`
- [ ] `evidence_receipt.py` contains `get_all_entries()` method
- [ ] `poi.proto` present at `bizra-constitution/poi.proto`
- [ ] No v5 files modified (except evidence_receipt.py)
