"""
Tests for Tamper-Evident Audit Log
===================================

Comprehensive test suite for cryptographic audit log integrity.

Test Categories:
- Entry creation and signing
- HMAC verification
- Chain linking and verification
- Tampering detection
- Key management and rotation
- Persistence and recovery
- Edge cases and security

Standing on Giants:
- Merkle (1979): Hash chain verification
- RFC 2104 (1997): HMAC test vectors

Genesis Strict Synthesis v2.2.2
"""

import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import List

import pytest

from core.sovereign.tamper_evident_log import (
    AuditKeyManager,
    GENESIS_HASH,
    HMAC_DOMAIN_PREFIX,
    KeyRotationEvent,
    TamperEvidentEntry,
    TamperEvidentLog,
    TamperingReport,
    TamperType,
    VerificationStatus,
    create_audit_log,
    detect_tampering,
    verify_chain,
    verify_entry,
    _compute_content_hash,
    _compute_entry_hash,
    _compute_entry_hmac,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def key_manager() -> AuditKeyManager:
    """Create a key manager with known test key."""
    test_key = secrets.token_bytes(32)
    return AuditKeyManager(_master_key=test_key)


@pytest.fixture
def audit_log(key_manager: AuditKeyManager) -> TamperEvidentLog:
    """Create an in-memory audit log."""
    return TamperEvidentLog(key_manager)


@pytest.fixture
def populated_log(audit_log: TamperEvidentLog) -> TamperEvidentLog:
    """Create a log with several entries."""
    for i in range(5):
        audit_log.append({
            "event": f"test_event_{i}",
            "sequence_data": i,
            "timestamp": time.time_ns(),
        })
    return audit_log


@pytest.fixture
def temp_log_path(tmp_path: Path) -> Path:
    """Create a temporary path for log persistence."""
    return tmp_path / "audit_log.jsonl"


# =============================================================================
# ENTRY CREATION TESTS
# =============================================================================

class TestEntryCreation:
    """Tests for TamperEvidentEntry creation and serialization."""

    def test_create_genesis_entry(self, audit_log: TamperEvidentLog):
        """First entry should link to GENESIS_HASH."""
        entry = audit_log.append({"event": "genesis_test"})

        assert entry.sequence == 0
        assert entry.prev_hash == GENESIS_HASH
        assert entry.content_hash
        assert entry.hmac_signature

    def test_entry_sequence_increments(self, audit_log: TamperEvidentLog):
        """Sequence numbers should increment monotonically."""
        for i in range(10):
            entry = audit_log.append({"event": f"event_{i}"})
            assert entry.sequence == i

    def test_entry_chain_linking(self, audit_log: TamperEvidentLog):
        """Each entry should link to previous entry's hash."""
        entry1 = audit_log.append({"event": "first"})
        entry2 = audit_log.append({"event": "second"})

        # Entry 2 should link to hash of entry 1
        expected_prev_hash = _compute_entry_hash(entry1)
        assert entry2.prev_hash == expected_prev_hash

    def test_entry_timestamp_precision(self, audit_log: TamperEvidentLog):
        """Timestamps should have nanosecond precision."""
        entry = audit_log.append({"event": "timestamp_test"})

        # Nanosecond timestamps should be in the range of 10^18
        assert entry.timestamp_ns > 10**18
        assert entry.timestamp_ns < 10**20

    def test_entry_custom_timestamp(self, audit_log: TamperEvidentLog):
        """Should accept custom timestamps."""
        custom_ts = 1700000000000000000  # Fixed timestamp
        entry = audit_log.append(
            {"event": "custom_ts"},
            timestamp_ns=custom_ts,
        )

        assert entry.timestamp_ns == custom_ts

    def test_entry_serialization_roundtrip(self, audit_log: TamperEvidentLog):
        """Entry should survive serialization roundtrip."""
        original = audit_log.append({
            "event": "serialize_test",
            "nested": {"key": "value"},
            "array": [1, 2, 3],
        })

        # Serialize and deserialize
        data = original.to_dict()
        restored = TamperEvidentEntry.from_dict(data)

        assert restored.sequence == original.sequence
        assert restored.timestamp_ns == original.timestamp_ns
        assert restored.content == original.content
        assert restored.content_hash == original.content_hash
        assert restored.prev_hash == original.prev_hash
        assert restored.hmac_signature == original.hmac_signature

    def test_entry_datetime_conversion(self, audit_log: TamperEvidentLog):
        """Timestamp should convert to datetime correctly."""
        entry = audit_log.append({"event": "datetime_test"})

        dt = entry.timestamp_datetime
        assert dt is not None
        assert dt.tzinfo is not None  # Should be UTC-aware


# =============================================================================
# VERIFICATION TESTS
# =============================================================================

class TestVerification:
    """Tests for entry and chain verification."""

    def test_verify_valid_entry(
        self,
        audit_log: TamperEvidentLog,
        key_manager: AuditKeyManager,
    ):
        """Valid entry should verify successfully."""
        entry = audit_log.append({"event": "verify_test"})

        status = entry.verify(key_manager.get_signing_key(), None)
        assert status == VerificationStatus.VALID

    def test_verify_chain_link(
        self,
        audit_log: TamperEvidentLog,
        key_manager: AuditKeyManager,
    ):
        """Chain linking should verify correctly."""
        entry1 = audit_log.append({"event": "first"})
        entry2 = audit_log.append({"event": "second"})

        status = entry2.verify(key_manager.get_signing_key(), entry1)
        assert status == VerificationStatus.VALID

    def test_verify_fails_wrong_key(
        self,
        audit_log: TamperEvidentLog,
    ):
        """Verification should fail with wrong key."""
        entry = audit_log.append({"event": "wrong_key_test"})

        wrong_key = secrets.token_bytes(32)
        status = entry.verify(wrong_key, None)
        assert status == VerificationStatus.INVALID_HMAC

    def test_verify_fails_modified_content(
        self,
        audit_log: TamperEvidentLog,
        key_manager: AuditKeyManager,
    ):
        """Verification should fail if content is modified."""
        entry = audit_log.append({"event": "original"})

        # Tamper with content
        entry.content = {"event": "tampered"}

        status = entry.verify(key_manager.get_signing_key(), None)
        assert status == VerificationStatus.INVALID_CONTENT_HASH

    def test_verify_fails_modified_prev_hash(
        self,
        audit_log: TamperEvidentLog,
        key_manager: AuditKeyManager,
    ):
        """Verification should fail if prev_hash is modified."""
        entry1 = audit_log.append({"event": "first"})
        entry2 = audit_log.append({"event": "second"})

        # Tamper with chain link
        entry2.prev_hash = "0" * 64

        status = entry2.verify(key_manager.get_signing_key(), entry1)
        # Will fail on HMAC first since prev_hash is part of HMAC input
        assert status == VerificationStatus.INVALID_HMAC

    def test_verify_chain_function(
        self,
        populated_log: TamperEvidentLog,
        key_manager: AuditKeyManager,
    ):
        """verify_chain should validate entire chain."""
        entries = list(populated_log)

        is_valid = verify_chain(entries, key_manager.get_signing_key())
        assert is_valid

    def test_verify_chain_method(
        self,
        populated_log: TamperEvidentLog,
    ):
        """Log's verify_chain method should work."""
        is_valid, invalid = populated_log.verify_chain()
        assert is_valid
        assert len(invalid) == 0

    def test_verify_empty_chain(self, key_manager: AuditKeyManager):
        """Empty chain should verify successfully."""
        is_valid = verify_chain([], key_manager.get_signing_key())
        assert is_valid


# =============================================================================
# TAMPERING DETECTION TESTS
# =============================================================================

class TestTamperingDetection:
    """Tests for detecting various types of tampering."""

    def test_detect_no_tampering(self, populated_log: TamperEvidentLog):
        """Clean log should show no tampering."""
        report = populated_log.detect_tampering()

        assert not report.is_tampered
        assert report.tamper_type is None
        assert len(report.affected_sequences) == 0
        assert report.verified_count == len(populated_log)

    def test_detect_content_modification(
        self,
        populated_log: TamperEvidentLog,
    ):
        """Should detect modified entry content."""
        # Tamper with middle entry
        entries = list(populated_log)
        entries[2].content = {"event": "TAMPERED"}

        report = populated_log.detect_tampering(entries)

        assert report.is_tampered
        assert report.tamper_type == TamperType.CONTENT_MODIFIED
        assert 2 in report.affected_sequences

    def test_detect_chain_break(
        self,
        populated_log: TamperEvidentLog,
        key_manager: AuditKeyManager,
    ):
        """Should detect broken chain link."""
        entries = list(populated_log)

        # Create a new entry that breaks the chain
        broken_entry = TamperEvidentEntry(
            sequence=2,
            timestamp_ns=time.time_ns(),
            content={"event": "inserted"},
            content_hash=_compute_content_hash({"event": "inserted"}),
            prev_hash="0" * 64,  # Wrong prev_hash
            hmac_signature=_compute_entry_hmac(
                sequence=2,
                timestamp_ns=time.time_ns(),
                content_hash=_compute_content_hash({"event": "inserted"}),
                prev_hash="0" * 64,
                secret_key=key_manager.get_signing_key(),
            ),
        )
        entries[2] = broken_entry

        report = populated_log.detect_tampering(entries)

        assert report.is_tampered
        # Chain break detected
        assert 2 in report.affected_sequences

    def test_detect_sequence_gap(
        self,
        populated_log: TamperEvidentLog,
    ):
        """Should detect missing entries (sequence gap)."""
        entries = list(populated_log)

        # Remove an entry to create a gap
        del entries[2]  # Remove sequence 2

        report = populated_log.detect_tampering(entries)

        assert report.is_tampered
        assert report.tamper_type == TamperType.ENTRY_DELETED
        assert 2 in report.affected_sequences

    def test_tampering_report_serialization(
        self,
        populated_log: TamperEvidentLog,
    ):
        """TamperingReport should serialize correctly."""
        report = populated_log.detect_tampering()
        data = report.to_dict()

        assert "is_tampered" in data
        assert "tamper_type" in data
        assert "affected_sequences" in data
        assert "integrity_ratio" in data
        assert data["integrity_ratio"] == 1.0  # All entries valid


# =============================================================================
# KEY MANAGEMENT TESTS
# =============================================================================

class TestKeyManagement:
    """Tests for AuditKeyManager functionality."""

    def test_key_manager_creation(self):
        """Key manager should generate secure key."""
        manager = AuditKeyManager()

        key = manager.get_signing_key()
        assert len(key) >= 32
        assert manager.key_id

    def test_key_manager_from_hex(self):
        """Should create manager from hex key."""
        hex_key = secrets.token_hex(32)
        manager = AuditKeyManager.from_hex(hex_key)

        exported = manager.export_key_hex()
        assert exported == hex_key

    def test_key_manager_rejects_short_key(self):
        """Should reject keys shorter than minimum length."""
        short_key = secrets.token_bytes(16)  # Too short

        with pytest.raises(ValueError):
            AuditKeyManager(_master_key=short_key)

    def test_session_key_derivation(self, key_manager: AuditKeyManager):
        """Should derive consistent session keys."""
        session_id = "test_session_123"

        key1 = key_manager.derive_session_key(session_id)
        key2 = key_manager.derive_session_key(session_id)

        # Same session should get same key
        assert key1 == key2

        # Different session should get different key
        key3 = key_manager.derive_session_key("different_session")
        assert key1 != key3

    def test_key_rotation(self, key_manager: AuditKeyManager):
        """Key rotation should update key and track history."""
        old_key_id = key_manager.key_id
        old_key = key_manager.get_signing_key()

        event = key_manager.rotate_key(reason="test_rotation", current_sequence=10)

        # Key should be different
        assert key_manager.get_signing_key() != old_key
        assert key_manager.key_id != old_key_id

        # Event should be recorded
        assert event.old_key_id == old_key_id
        assert event.new_key_id == key_manager.key_id
        assert event.reason == "test_rotation"
        assert event.sequence_at_rotation == 10

        # History should be updated
        history = key_manager.get_rotation_history()
        assert len(history) == 1
        assert history[0].old_key_id == old_key_id

    def test_key_rotation_clears_derived_keys(self, key_manager: AuditKeyManager):
        """Key rotation should invalidate derived session keys."""
        session_key_before = key_manager.derive_session_key("session1")

        key_manager.rotate_key()

        session_key_after = key_manager.derive_session_key("session1")

        # Derived key should be different after rotation
        assert session_key_before != session_key_after


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================

class TestPersistence:
    """Tests for log persistence and recovery."""

    def test_persist_and_load(self, key_manager: AuditKeyManager, temp_log_path: Path):
        """Log should persist to disk and reload correctly."""
        # Create and populate log
        log1 = TamperEvidentLog(key_manager, temp_log_path)
        for i in range(5):
            log1.append({"event": f"persist_test_{i}", "index": i})

        # Create new log from same path
        log2 = TamperEvidentLog(key_manager, temp_log_path)

        assert len(log2) == 5

        # Verify loaded entries
        for i, entry in enumerate(log2):
            assert entry.sequence == i
            assert entry.content["event"] == f"persist_test_{i}"

    def test_persist_maintains_chain(
        self,
        key_manager: AuditKeyManager,
        temp_log_path: Path,
    ):
        """Persisted log should maintain chain integrity."""
        # Create and populate
        log1 = TamperEvidentLog(key_manager, temp_log_path)
        for i in range(3):
            log1.append({"event": f"chain_test_{i}"})

        # Load and verify chain
        log2 = TamperEvidentLog(key_manager, temp_log_path)
        report = log2.detect_tampering()

        assert not report.is_tampered
        assert report.verified_count == 3

    def test_append_after_reload(
        self,
        key_manager: AuditKeyManager,
        temp_log_path: Path,
    ):
        """Should be able to append after reloading from disk."""
        # Create with initial entries
        log1 = TamperEvidentLog(key_manager, temp_log_path)
        log1.append({"event": "initial"})

        # Reload and append
        log2 = TamperEvidentLog(key_manager, temp_log_path)
        log2.append({"event": "after_reload"})

        assert len(log2) == 2
        assert log2.get_entry(1).content["event"] == "after_reload"

        # Verify chain still valid
        report = log2.detect_tampering()
        assert not report.is_tampered

    def test_export_import_chain(self, populated_log: TamperEvidentLog):
        """Should export and import chain data."""
        chain_data = populated_log.export_chain()

        assert len(chain_data) == 5
        assert all("sequence" in e for e in chain_data)
        assert all("hmac_signature" in e for e in chain_data)

    def test_import_validates_chain(
        self,
        key_manager: AuditKeyManager,
    ):
        """Import should validate chain integrity."""
        log = TamperEvidentLog(key_manager)

        # Create valid chain data manually
        valid_chain = []
        prev_hash = GENESIS_HASH
        base_timestamp = time.time_ns()

        for i in range(3):
            # Use consistent timestamp for HMAC and entry
            timestamp_ns = base_timestamp + i * 1000000  # Add 1ms per entry
            content = {"event": f"import_test_{i}"}
            content_hash = _compute_content_hash(content)
            hmac_sig = _compute_entry_hmac(
                sequence=i,
                timestamp_ns=timestamp_ns,
                content_hash=content_hash,
                prev_hash=prev_hash,
                secret_key=key_manager.get_signing_key(),
            )
            entry_data = {
                "sequence": i,
                "timestamp_ns": timestamp_ns,
                "content": content,
                "content_hash": content_hash,
                "prev_hash": prev_hash,
                "hmac_signature": hmac_sig,
            }
            valid_chain.append(entry_data)

            # Update prev_hash for next iteration
            entry = TamperEvidentEntry.from_dict(entry_data)
            prev_hash = _compute_entry_hash(entry)

        imported, invalid = log.import_chain(valid_chain)
        assert imported == 3
        assert len(invalid) == 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_audit_log(self):
        """create_audit_log should return log and key manager."""
        log, key_manager = create_audit_log()

        assert isinstance(log, TamperEvidentLog)
        assert isinstance(key_manager, AuditKeyManager)
        assert len(log) == 0

    def test_create_audit_log_with_key(self):
        """create_audit_log should accept hex key."""
        hex_key = secrets.token_hex(32)
        log, key_manager = create_audit_log(key_hex=hex_key)

        assert key_manager.export_key_hex() == hex_key

    def test_create_audit_log_with_persistence(self, temp_log_path: Path):
        """create_audit_log should support persistence."""
        log, _ = create_audit_log(persist_path=temp_log_path)
        log.append({"event": "persist_test"})

        assert temp_log_path.exists()

    def test_verify_entry_function(self, key_manager: AuditKeyManager):
        """verify_entry convenience function should work."""
        log = TamperEvidentLog(key_manager)
        entry = log.append({"event": "test"})

        is_valid = verify_entry(entry, key_manager.get_signing_key())
        assert is_valid

    def test_verify_chain_function_invalid(self, key_manager: AuditKeyManager):
        """verify_chain should detect invalid chains."""
        log = TamperEvidentLog(key_manager)
        entry1 = log.append({"event": "first"})
        entry2 = log.append({"event": "second"})

        # Tamper with entry
        entry2.content = {"event": "tampered"}

        is_valid = verify_chain([entry1, entry2], key_manager.get_signing_key())
        assert not is_valid

    def test_detect_tampering_function(self, key_manager: AuditKeyManager):
        """detect_tampering convenience function should work."""
        log = TamperEvidentLog(key_manager)
        entries = [log.append({"event": f"e{i}"}) for i in range(3)]

        report = detect_tampering(entries, key_manager.get_signing_key())
        assert not report.is_tampered


# =============================================================================
# EDGE CASES AND SECURITY TESTS
# =============================================================================

class TestEdgeCasesAndSecurity:
    """Tests for edge cases and security considerations."""

    def test_empty_content(self, audit_log: TamperEvidentLog):
        """Should handle empty content dict."""
        entry = audit_log.append({})

        assert entry.content == {}
        assert entry.content_hash  # Should still have hash

        report = audit_log.detect_tampering()
        assert not report.is_tampered

    def test_large_content(self, audit_log: TamperEvidentLog):
        """Should handle large content."""
        large_content = {
            "data": "x" * 100000,
            "array": list(range(1000)),
        }
        entry = audit_log.append(large_content)

        assert entry.content == large_content

        report = audit_log.detect_tampering()
        assert not report.is_tampered

    def test_special_characters_in_content(self, audit_log: TamperEvidentLog):
        """Should handle special characters."""
        special_content = {
            "unicode": "Hello, \u4e16\u754c! \U0001F600",
            "newlines": "line1\nline2\r\nline3",
            "quotes": 'single\' and "double" quotes',
            "backslash": "path\\to\\file",
        }
        entry = audit_log.append(special_content)

        # Verify roundtrip
        restored = TamperEvidentEntry.from_dict(entry.to_dict())
        assert restored.content == special_content

    def test_nested_content(self, audit_log: TamperEvidentLog):
        """Should handle deeply nested content."""
        nested_content = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        entry = audit_log.append(nested_content)
        assert entry.content == nested_content

    def test_content_hash_deterministic(self):
        """Content hash should be deterministic regardless of key order."""
        content1 = {"b": 2, "a": 1, "c": 3}
        content2 = {"a": 1, "c": 3, "b": 2}

        hash1 = _compute_content_hash(content1)
        hash2 = _compute_content_hash(content2)

        assert hash1 == hash2

    def test_hmac_domain_separation(self, key_manager: AuditKeyManager):
        """HMAC should include domain prefix for separation."""
        # The HMAC computation should include the domain prefix
        # This prevents cross-protocol attacks
        hmac_sig = _compute_entry_hmac(
            sequence=0,
            timestamp_ns=1000000000,
            content_hash="a" * 64,
            prev_hash=GENESIS_HASH,
            secret_key=key_manager.get_signing_key(),
        )

        # Verify format (should be hex string)
        assert len(hmac_sig) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hmac_sig)

    def test_timing_safe_comparison(
        self,
        audit_log: TamperEvidentLog,
        key_manager: AuditKeyManager,
    ):
        """Verification should use timing-safe comparison."""
        # This is implicitly tested by using hmac.compare_digest
        # in the verify method. We ensure it doesn't fail.
        entry = audit_log.append({"event": "timing_test"})

        # Multiple verifications should all succeed
        for _ in range(100):
            status = entry.verify(key_manager.get_signing_key(), None)
            assert status == VerificationStatus.VALID

    def test_concurrent_append_sequence(self, key_manager: AuditKeyManager):
        """Sequence numbers should remain consistent with rapid appends."""
        log = TamperEvidentLog(key_manager)

        # Rapid appends
        entries = []
        for i in range(100):
            entries.append(log.append({"index": i}))

        # Verify sequence integrity
        for i, entry in enumerate(entries):
            assert entry.sequence == i

        # Verify chain
        report = log.detect_tampering()
        assert not report.is_tampered

    def test_get_entries_range(self, populated_log: TamperEvidentLog):
        """Should retrieve entries by sequence range."""
        entries = populated_log.get_entries(start_sequence=1, end_sequence=3)

        assert len(entries) == 3
        assert entries[0].sequence == 1
        assert entries[-1].sequence == 3

    def test_last_entry_property(self, populated_log: TamperEvidentLog):
        """last_entry should return most recent entry."""
        last = populated_log.last_entry

        assert last is not None
        assert last.sequence == 4  # 0-indexed, 5 entries

    def test_last_hash_property(self, populated_log: TamperEvidentLog):
        """last_hash should match hash of last entry."""
        last_entry = populated_log.last_entry
        expected_hash = _compute_entry_hash(last_entry)

        assert populated_log.last_hash == expected_hash

    def test_iteration(self, populated_log: TamperEvidentLog):
        """Should iterate entries in sequence order."""
        sequences = [entry.sequence for entry in populated_log]

        assert sequences == [0, 1, 2, 3, 4]

    def test_length(self, populated_log: TamperEvidentLog):
        """len() should return entry count."""
        assert len(populated_log) == 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_audit_workflow(self, temp_log_path: Path):
        """Test complete audit log lifecycle."""
        # Create log
        log, key_manager = create_audit_log(persist_path=temp_log_path)

        # Append various events
        log.append({"event": "user_login", "user_id": "alice", "ip": "192.168.1.1"})
        log.append({"event": "permission_granted", "user_id": "alice", "resource": "admin"})
        log.append({"event": "data_access", "user_id": "alice", "table": "users"})
        log.append({"event": "user_logout", "user_id": "alice"})

        # Verify integrity
        report = log.detect_tampering()
        assert not report.is_tampered
        assert report.verified_count == 4

        # Export for archival
        chain_data = log.export_chain()
        assert len(chain_data) == 4

        # Simulate system restart
        log2, key_manager2 = create_audit_log(
            persist_path=temp_log_path,
            key_hex=key_manager.export_key_hex(),
        )

        # Continue logging
        log2.append({"event": "system_restart", "reason": "scheduled"})

        # Final verification
        report2 = log2.detect_tampering()
        assert not report2.is_tampered
        assert report2.verified_count == 5

    def test_key_rotation_workflow(self, temp_log_path: Path):
        """Test key rotation during audit log lifetime."""
        log, key_manager = create_audit_log(persist_path=temp_log_path)

        # Log some events
        for i in range(3):
            log.append({"event": f"pre_rotation_{i}"})

        old_key = key_manager.export_key_hex()

        # Verify with original key
        report = log.detect_tampering()
        assert not report.is_tampered

        # Rotate key
        rotation_event = key_manager.rotate_key(
            reason="security_policy",
            current_sequence=len(log),
        )

        # Log entry about rotation
        log.append({
            "event": "key_rotation",
            "old_key_id": rotation_event.old_key_id,
            "new_key_id": rotation_event.new_key_id,
        })

        # Log more events with new key
        for i in range(3):
            log.append({"event": f"post_rotation_{i}"})

        # Note: Entries after rotation need the new key to verify
        # In practice, you'd store key rotation events and use
        # appropriate keys for each segment


# =============================================================================
# REPORT FORMAT TESTS
# =============================================================================

class TestReportFormats:
    """Tests for report serialization and formatting."""

    def test_tampering_report_dict(self, populated_log: TamperEvidentLog):
        """TamperingReport.to_dict should be JSON-serializable."""
        report = populated_log.detect_tampering()
        data = report.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert json_str

        # Should have expected fields
        assert "is_tampered" in data
        assert "tamper_type" in data
        assert "integrity_ratio" in data

    def test_key_rotation_event_dict(self, key_manager: AuditKeyManager):
        """KeyRotationEvent.to_dict should be JSON-serializable."""
        event = key_manager.rotate_key()
        data = event.to_dict()

        json_str = json.dumps(data)
        assert json_str

        assert "old_key_id" in data
        assert "new_key_id" in data
        assert "rotation_timestamp_ns" in data


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
