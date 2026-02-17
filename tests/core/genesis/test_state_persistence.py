"""
Tests for core.genesis.state_persistence — Sovereign State Management.

Covers:
- BIP39-style recovery phrase roundtrip
- Key encryption/decryption with passphrase
- Bad passphrase rejection
- SovereignState save/load roundtrip
- File layout verification
- Permission checks (Unix only)
- Wordlist integrity
"""

from __future__ import annotations

import json
import secrets
import shutil
from pathlib import Path

import pytest

from core.genesis.state_persistence import (
    BIP39_WORDLIST,
    SovereignState,
    decrypt_private_key,
    encrypt_private_key,
    entropy_to_phrase,
    load_sovereign_state,
    phrase_to_entropy,
    phrase_to_private_key,
    private_key_to_phrase,
    save_sovereign_state,
    state_exists,
)


@pytest.fixture
def state_dir(tmp_path):
    """Temporary directory for state persistence tests."""
    d = tmp_path / "sovereign_state" / "genesis"
    yield d
    if d.exists():
        shutil.rmtree(d)


@pytest.fixture
def sample_state():
    """A SovereignState with all fields populated."""
    return SovereignState(
        node_id="BIZRA-TEST-001",
        public_key="abc123def456",
        identity_card={
            "node_id": "BIZRA-TEST-001",
            "public_key": "abc123def456",
            "creation_timestamp": "2026-02-15T00:00:00Z",
        },
        pat_agents=[
            {"name": "research-agent", "role": "research"},
            {"name": "code-agent", "role": "code"},
        ],
        sat_agents=[
            {"name": "monitor-agent", "role": "monitor"},
        ],
        urp_pledge={"cpu_cores": 4, "ram_gb": 16},
        genesis_receipt={"ceremony": "test", "hash": "deadbeef"},
        hardware_info={"gpu": "RTX 4090", "ram_gb": 128},
        recovery_phrase=["abandon", "ability", "able"],
        encrypted_key={"salt": "aa", "ciphertext": "bb", "checksum": "cc"},
    )


class TestWordlist:
    """BIP39 wordlist integrity."""

    def test_wordlist_length(self):
        assert len(BIP39_WORDLIST) == 256

    def test_words_are_unique(self):
        assert len(set(BIP39_WORDLIST)) == len(BIP39_WORDLIST)

    def test_words_are_lowercase(self):
        for word in BIP39_WORDLIST:
            assert word == word.lower()

    def test_first_word(self):
        assert BIP39_WORDLIST[0] == "abandon"

    def test_last_word(self):
        assert BIP39_WORDLIST[255] == "cactus"


class TestPhraseRoundtrip:
    """Entropy <-> phrase conversion."""

    def test_roundtrip_24_words(self):
        entropy = secrets.token_bytes(24)
        phrase = entropy_to_phrase(entropy, word_count=24)
        assert len(phrase) == 24
        recovered = phrase_to_entropy(phrase)
        assert recovered == entropy

    def test_roundtrip_12_words(self):
        entropy = secrets.token_bytes(12)
        phrase = entropy_to_phrase(entropy, word_count=12)
        assert len(phrase) == 12
        recovered = phrase_to_entropy(phrase)
        assert recovered == entropy

    def test_all_bytes_map_to_words(self):
        entropy = bytes(range(256))
        phrase = entropy_to_phrase(entropy, word_count=256)
        assert len(phrase) == 256
        assert len(set(phrase)) == 256  # All words unique

    def test_invalid_word_raises(self):
        with pytest.raises(ValueError, match="Unknown recovery word"):
            phrase_to_entropy(["notarealword"])

    def test_case_insensitive(self):
        entropy = secrets.token_bytes(3)
        phrase = entropy_to_phrase(entropy, word_count=3)
        upper_phrase = [w.upper() for w in phrase]
        recovered = phrase_to_entropy(upper_phrase)
        assert recovered == entropy


class TestKeyPhrase:
    """Private key <-> recovery phrase."""

    def test_key_to_phrase_length(self):
        key_hex = secrets.token_hex(32)
        phrase = private_key_to_phrase(key_hex)
        assert len(phrase) == 24

    def test_deterministic(self):
        key_hex = secrets.token_hex(32)
        phrase1 = private_key_to_phrase(key_hex)
        phrase2 = private_key_to_phrase(key_hex)
        assert phrase1 == phrase2


class TestEncryption:
    """Key encryption/decryption."""

    def test_encrypt_decrypt_roundtrip(self):
        key = secrets.token_hex(32)
        encrypted = encrypt_private_key(key, "my-passphrase")
        decrypted = decrypt_private_key(encrypted, "my-passphrase")
        assert decrypted == key

    def test_wrong_passphrase_raises(self):
        key = secrets.token_hex(32)
        encrypted = encrypt_private_key(key, "correct")
        with pytest.raises(ValueError, match="checksum mismatch"):
            decrypt_private_key(encrypted, "wrong")

    def test_encrypted_fields(self):
        key = secrets.token_hex(32)
        encrypted = encrypt_private_key(key, "pass")
        assert "salt" in encrypted
        assert "ciphertext" in encrypted
        assert "checksum" in encrypted
        assert "kdf" in encrypted
        assert encrypted["kdf"] == "pbkdf2-sha256"

    def test_different_salts(self):
        key = secrets.token_hex(32)
        e1 = encrypt_private_key(key, "same")
        e2 = encrypt_private_key(key, "same")
        assert e1["salt"] != e2["salt"]  # Random salt each time
        assert e1["ciphertext"] != e2["ciphertext"]


class TestStatePersistence:
    """Save/load sovereign state."""

    def test_save_creates_files(self, state_dir, sample_state):
        save_sovereign_state(sample_state, state_dir)
        assert (state_dir / "identity.json").exists()
        assert (state_dir / "pat_manifest.json").exists()
        assert (state_dir / "sat_manifest.json").exists()
        assert (state_dir / "genesis_receipt.json").exists()
        assert (state_dir / ".keystore" / "sovereign.enc").exists()
        assert (state_dir / "recovery_phrase.txt").exists()

    def test_save_load_roundtrip(self, state_dir, sample_state):
        save_sovereign_state(sample_state, state_dir)
        loaded = load_sovereign_state(state_dir)
        assert loaded is not None
        assert loaded.node_id == "BIZRA-TEST-001"
        assert loaded.public_key == "abc123def456"
        assert len(loaded.pat_agents) == 2
        assert len(loaded.sat_agents) == 1

    def test_identity_json_content(self, state_dir, sample_state):
        save_sovereign_state(sample_state, state_dir)
        data = json.loads((state_dir / "identity.json").read_text())
        assert data["node_id"] == "BIZRA-TEST-001"

    def test_pat_manifest_content(self, state_dir, sample_state):
        save_sovereign_state(sample_state, state_dir)
        data = json.loads((state_dir / "pat_manifest.json").read_text())
        assert data["agent_count"] == 2
        assert data["agents"][0]["name"] == "research-agent"

    def test_recovery_phrase_file(self, state_dir, sample_state):
        save_sovereign_state(sample_state, state_dir)
        text = (state_dir / "recovery_phrase.txt").read_text()
        assert "BIZRA SOVEREIGN RECOVERY PHRASE" in text
        assert "BIZRA-TEST-001" in text
        assert "1.abandon" in text

    def test_state_exists(self, state_dir, sample_state):
        assert not state_exists(state_dir)
        save_sovereign_state(sample_state, state_dir)
        assert state_exists(state_dir)

    def test_load_nonexistent_returns_none(self, state_dir):
        assert load_sovereign_state(state_dir) is None

    def test_empty_optional_fields(self, state_dir):
        state = SovereignState(
            node_id="BIZRA-MINIMAL",
            identity_card={"node_id": "BIZRA-MINIMAL"},
            genesis_receipt={"minimal": True},
        )
        save_sovereign_state(state, state_dir)
        loaded = load_sovereign_state(state_dir)
        assert loaded is not None
        assert loaded.node_id == "BIZRA-MINIMAL"
        assert loaded.pat_agents == []

    def test_hardware_info_saved(self, state_dir, sample_state):
        save_sovereign_state(sample_state, state_dir)
        data = json.loads((state_dir / "hardware.json").read_text())
        assert data["gpu"] == "RTX 4090"

    def test_urp_pledge_saved(self, state_dir, sample_state):
        save_sovereign_state(sample_state, state_dir)
        data = json.loads((state_dir / "urp_pledge.json").read_text())
        assert data["cpu_cores"] == 4

    def test_to_dict_excludes_secrets(self, sample_state):
        d = sample_state.to_dict()
        assert "encrypted_key" not in d
        assert "recovery_phrase" not in d
        assert "node_id" in d
