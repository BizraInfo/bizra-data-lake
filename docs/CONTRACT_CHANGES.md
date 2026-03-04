## Contract Change Log

### 71e5422 - SEC-001 BLAKE3 gate fix (evidence_ledger.py)
- Kept `# noqa: SEC-001` on same line as `hashlib.sha256()` after Black reformatting
- Assigned to intermediate variable to avoid Black splitting the noqa tag away
- 641 proof_engine tests GREEN after change

### 61f9f74 - Lint fixes (Ruff F401 + Clippy field_reassign_with_default)
- Removed unused imports in core/elite/ (time, uuid, HardwareAsset, Path, Literal)
- Fixed Clippy field_reassign_with_default in bizra-api/tests/api_tests.rs (3 sites)
- No behavioral changes — lint compliance only

### cf38159, d0805b8 - Codebase-wide formatting normalization
- Black + isort applied to 90 Python files (formatting only, no logic changes)
- cargo fmt applied to bizra-api (lib.rs, main.rs)
- CRLF to LF normalization for cross-platform consistency
- Phase 65 lifecycle protocol tests verified GREEN (17/17) after formatting

### fa9885c - Fixed Black formatting in evidence_ledger.py
- Wrapped long lines for Black compliance
- Maintained SEC-001 legacy compatibility
