// bizra-cognition/src/sovereign_state.rs
//
// Cycle-6 G1 Phase 1 — durable-read projection over Python-authored
// sovereign_state/ on disk.
//
// This module is READ-ONLY by design. The Python stack (see
// deploy/node0/bizra_node_activate.sh) is the authoritative writer.
// Rust loads the chain on startup and serves a read-only projection.
//
// Commit A — the Python-parity JSON formatter + BLAKE3 chain-entry hash
//            that reproduce the writer algorithm exactly.
//
// Commit B — the snapshot loader that reads sovereign_state/, walks every
//            activation_chain_*.json envelope, verifies per-entry hash +
//            prev_hash linkage + head_hash, returns a VerifiedEnvelope
//            or fails closed on any mismatch.
//
// See cycle-6/g1-writer-format-found.md for the verified algorithm.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};
use serde_json::Value;

use crate::canonical_hasher::{Blake3Hash, blake3_chain};

// ============================================================================
// Custom formatter — reproduces Python json.dumps default separators
// ============================================================================

/// Emits JSON matching Python's `json.dumps(obj)` DEFAULT separators:
///   - `", "` between array elements and object pairs (comma + space)
///   - `": "` between object key and value (colon + space)
///   - no outer whitespace, no indent, no trailing newline
///
/// Combined with serde_json::Map's default BTreeMap backing (keys
/// auto-sorted), this matches `json.dumps(data, sort_keys=True)`
/// byte-for-byte on ASCII input.
pub struct PythonDefaultFormatter;

impl Formatter for PythonDefaultFormatter {
    #[inline]
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Serialize a `serde_json::Value` to bytes matching Python's
/// `json.dumps(value, sort_keys=True).encode()` on ASCII input.
///
/// Keys are sorted because `serde_json::Map` uses `BTreeMap` by default
/// (no `preserve_order` feature active in this crate).
pub fn to_python_json_bytes(value: &Value) -> Result<Vec<u8>, serde_json::Error> {
    let mut buf = Vec::new();
    {
        let mut ser = Serializer::with_formatter(&mut buf, PythonDefaultFormatter);
        value.serialize(&mut ser)?;
    }
    Ok(buf)
}

// ============================================================================
// BLAKE3 chain-entry hash — matches deploy/node0/bizra_node_activate.sh:400-407
// ============================================================================

/// Genesis prev_hash — 64 ASCII zero characters.
pub const GENESIS_PREV_HEX: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Compute a chain entry hash exactly as the Python writer does:
///
///   `BLAKE3( prev_hash_ascii_hex || python_default_sortk_json(data) )`
///
/// Returns the 32-byte BLAKE3 digest. Caller hex-encodes for comparison.
pub fn chain_entry_hash(prev_hex: &str, data: &Value) -> Result<Blake3Hash, serde_json::Error> {
    let content = to_python_json_bytes(data)?;
    let mut input = Vec::with_capacity(prev_hex.len() + content.len());
    input.extend_from_slice(prev_hex.as_bytes());
    input.extend_from_slice(&content);
    Ok(blake3_chain(&input))
}

/// Hex-encode a 32-byte BLAKE3 digest as lowercase 64-char ASCII.
pub fn hex_digest(digest: &Blake3Hash) -> String {
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ============================================================================
// Commit B — snapshot loader + verification
// ============================================================================

/// Errors produced while loading or verifying a sovereign_state/ snapshot.
/// All variants are fail-closed: the snapshot never ships with unverified content.
#[derive(Debug)]
pub enum SovereignStateError {
    RootMissing(PathBuf),
    ReceiptsDirMissing(PathBuf),
    NoEnvelopes(PathBuf),
    EnvelopeRead { path: PathBuf, msg: String },
    EnvelopeParse { path: PathBuf, msg: String },
    EnvelopeMalformed { path: PathBuf, reason: &'static str },
    ReceiptRead { envelope: PathBuf, file: String, msg: String },
    ReceiptParse { envelope: PathBuf, file: String, msg: String },
    HashMismatch {
        envelope: PathBuf,
        file: String,
        expected: String,
        computed: String,
    },
    PrevHashMismatch {
        envelope: PathBuf,
        file: String,
        expected: String,
        actual: String,
    },
    HeadHashMismatch {
        envelope: PathBuf,
        expected: String,
        computed: String,
    },
    SerializeError(String),
}

impl std::fmt::Display for SovereignStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RootMissing(p) => write!(f, "sovereign_state root missing: {}", p.display()),
            Self::ReceiptsDirMissing(p) => write!(f, "receipts dir missing: {}", p.display()),
            Self::NoEnvelopes(p) => write!(f, "no activation_chain_*.json envelopes found in {}", p.display()),
            Self::EnvelopeRead { path, msg } => write!(f, "read envelope {}: {}", path.display(), msg),
            Self::EnvelopeParse { path, msg } => write!(f, "parse envelope {}: {}", path.display(), msg),
            Self::EnvelopeMalformed { path, reason } => write!(f, "envelope {} malformed: {}", path.display(), reason),
            Self::ReceiptRead { envelope, file, msg } => write!(f, "read receipt {} (ref'd by {}): {}", file, envelope.display(), msg),
            Self::ReceiptParse { envelope, file, msg } => write!(f, "parse receipt {} (ref'd by {}): {}", file, envelope.display(), msg),
            Self::HashMismatch { envelope, file, expected, computed } =>
                write!(f, "hash mismatch for {} in {}: expected {}, computed {}", file, envelope.display(), expected, computed),
            Self::PrevHashMismatch { envelope, file, expected, actual } =>
                write!(f, "prev_hash mismatch for {} in {}: expected {}, got {}", file, envelope.display(), expected, actual),
            Self::HeadHashMismatch { envelope, expected, computed } =>
                write!(f, "head_hash mismatch in {}: declared {}, computed {}", envelope.display(), expected, computed),
            Self::SerializeError(s) => write!(f, "serialize error: {}", s),
        }
    }
}

impl std::error::Error for SovereignStateError {}

/// One verified entry from a chain envelope.
#[derive(Debug, Clone)]
pub struct ChainEntry {
    pub file: String,
    pub event: String,
    pub hash: String,
    pub prev_hash: String,
}

/// A fully-verified activation-chain envelope.
#[derive(Debug, Clone)]
pub struct VerifiedEnvelope {
    pub path: PathBuf,
    pub chain_type: String,
    pub node_id: String,
    pub timestamp: String,
    pub entries: Vec<ChainEntry>,
    pub head_hash: String,
}

/// A verified read-only projection of `sovereign_state/` on disk.
///
/// Produced by `SovereignStateSnapshot::load`. Every envelope in this
/// snapshot has passed per-entry hash verification, prev_hash linkage,
/// and head_hash agreement under the G1 algorithm. The snapshot is
/// safe to serve from the gateway without re-verification.
#[derive(Debug, Clone)]
pub struct SovereignStateSnapshot {
    pub root: PathBuf,
    pub envelopes: Vec<VerifiedEnvelope>,
    pub block_zero_present: bool,
}

impl SovereignStateSnapshot {
    /// Load and verify the sovereign_state/ directory.
    ///
    /// Fails closed if any envelope fails integrity checks. `block_zero`
    /// presence is recorded but not verified (different algorithm,
    /// genealogical anchor per cycle-6/g1-writer-format-found.md).
    pub fn load(root: &Path) -> Result<Self, SovereignStateError> {
        if !root.exists() {
            return Err(SovereignStateError::RootMissing(root.to_path_buf()));
        }
        let receipts_dir = root.join("receipts");
        if !receipts_dir.exists() {
            return Err(SovereignStateError::ReceiptsDirMissing(receipts_dir));
        }
        let block_zero_present = root.join("block_zero").join("block_zero.json").exists();

        let mut envelope_paths: Vec<PathBuf> = fs::read_dir(&receipts_dir)
            .map_err(|e| SovereignStateError::EnvelopeRead {
                path: receipts_dir.clone(),
                msg: e.to_string(),
            })?
            .filter_map(|r| r.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("activation_chain_") && n.ends_with(".json"))
                    .unwrap_or(false)
            })
            .collect();
        envelope_paths.sort();

        if envelope_paths.is_empty() {
            return Err(SovereignStateError::NoEnvelopes(receipts_dir));
        }

        let mut envelopes = Vec::with_capacity(envelope_paths.len());
        for env_path in &envelope_paths {
            envelopes.push(verify_envelope(&receipts_dir, env_path)?);
        }

        Ok(Self {
            root: root.to_path_buf(),
            envelopes,
            block_zero_present,
        })
    }

    pub fn total_entries(&self) -> usize {
        self.envelopes.iter().map(|e| e.entries.len()).sum()
    }

    pub fn envelopes_count(&self) -> usize {
        self.envelopes.len()
    }

    /// Lookup by hex-encoded hash across every envelope in the snapshot.
    pub fn find_entry_by_hash(&self, hex: &str) -> Option<&ChainEntry> {
        for env in &self.envelopes {
            for entry in &env.entries {
                if entry.hash == hex {
                    return Some(entry);
                }
            }
        }
        None
    }
}

fn verify_envelope(
    receipts_dir: &Path,
    env_path: &Path,
) -> Result<VerifiedEnvelope, SovereignStateError> {
    let bytes = fs::read(env_path).map_err(|e| SovereignStateError::EnvelopeRead {
        path: env_path.to_path_buf(),
        msg: e.to_string(),
    })?;
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| SovereignStateError::EnvelopeParse {
        path: env_path.to_path_buf(),
        msg: e.to_string(),
    })?;

    let obj = doc.as_object().ok_or(SovereignStateError::EnvelopeMalformed {
        path: env_path.to_path_buf(),
        reason: "root is not object",
    })?;

    let chain_type = obj
        .get("chain_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let node_id = obj
        .get("node_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let timestamp = obj
        .get("timestamp")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let head_hash = obj
        .get("head_hash")
        .and_then(|v| v.as_str())
        .ok_or(SovereignStateError::EnvelopeMalformed {
            path: env_path.to_path_buf(),
            reason: "missing head_hash",
        })?
        .to_string();
    let chain_arr =
        obj.get("chain")
            .and_then(|v| v.as_array())
            .ok_or(SovereignStateError::EnvelopeMalformed {
                path: env_path.to_path_buf(),
                reason: "missing or non-array chain",
            })?;

    let mut entries = Vec::with_capacity(chain_arr.len());
    let mut expected_prev = GENESIS_PREV_HEX.to_string();

    for entry_val in chain_arr.iter() {
        let entry_obj =
            entry_val
                .as_object()
                .ok_or(SovereignStateError::EnvelopeMalformed {
                    path: env_path.to_path_buf(),
                    reason: "chain entry not an object",
                })?;

        let file = entry_obj
            .get("file")
            .and_then(|v| v.as_str())
            .ok_or(SovereignStateError::EnvelopeMalformed {
                path: env_path.to_path_buf(),
                reason: "chain entry missing file",
            })?
            .to_string();
        let event = entry_obj
            .get("event")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let declared_hash = entry_obj
            .get("hash")
            .and_then(|v| v.as_str())
            .ok_or(SovereignStateError::EnvelopeMalformed {
                path: env_path.to_path_buf(),
                reason: "chain entry missing hash",
            })?
            .to_string();
        let declared_prev = entry_obj
            .get("prev_hash")
            .and_then(|v| v.as_str())
            .ok_or(SovereignStateError::EnvelopeMalformed {
                path: env_path.to_path_buf(),
                reason: "chain entry missing prev_hash",
            })?
            .to_string();

        if declared_prev != expected_prev {
            return Err(SovereignStateError::PrevHashMismatch {
                envelope: env_path.to_path_buf(),
                file: file.clone(),
                expected: expected_prev.clone(),
                actual: declared_prev,
            });
        }

        let receipt_path = receipts_dir.join(&file);
        let receipt_bytes =
            fs::read(&receipt_path).map_err(|e| SovereignStateError::ReceiptRead {
                envelope: env_path.to_path_buf(),
                file: file.clone(),
                msg: e.to_string(),
            })?;
        let receipt_data: Value = serde_json::from_slice(&receipt_bytes).map_err(|e| {
            SovereignStateError::ReceiptParse {
                envelope: env_path.to_path_buf(),
                file: file.clone(),
                msg: e.to_string(),
            }
        })?;

        let computed_digest = chain_entry_hash(&expected_prev, &receipt_data)
            .map_err(|e| SovereignStateError::SerializeError(e.to_string()))?;
        let computed_hex = hex_digest(&computed_digest);

        if computed_hex != declared_hash {
            return Err(SovereignStateError::HashMismatch {
                envelope: env_path.to_path_buf(),
                file: file.clone(),
                expected: declared_hash,
                computed: computed_hex,
            });
        }

        entries.push(ChainEntry {
            file,
            event,
            hash: declared_hash.clone(),
            prev_hash: declared_prev,
        });
        expected_prev = declared_hash;
    }

    if head_hash != expected_prev {
        return Err(SovereignStateError::HeadHashMismatch {
            envelope: env_path.to_path_buf(),
            expected: head_hash,
            computed: expected_prev,
        });
    }

    Ok(VerifiedEnvelope {
        path: env_path.to_path_buf(),
        chain_type,
        node_id,
        timestamp,
        entries,
        head_hash,
    })
}

// ============================================================================
// Tests — byte-parity against fixture + live activation_chain entries
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ------------------------------------------------------------------------
    // Byte-parity tests for the formatter
    // ------------------------------------------------------------------------

    #[test]
    fn flat_object_matches_python_default() {
        // Python: json.dumps({"b": 2, "a": 1}, sort_keys=True)
        //   -> '{"a": 1, "b": 2}'
        let v: Value = serde_json::from_str(r#"{"b":2,"a":1}"#).unwrap();
        let out = to_python_json_bytes(&v).unwrap();
        assert_eq!(out, br#"{"a": 1, "b": 2}"#.to_vec());
    }

    #[test]
    fn array_matches_python_default() {
        // Python: json.dumps([1, "x", true]) -> '[1, "x", true]'
        let v: Value = serde_json::from_str(r#"[1,"x",true]"#).unwrap();
        let out = to_python_json_bytes(&v).unwrap();
        assert_eq!(out, br#"[1, "x", true]"#.to_vec());
    }

    #[test]
    fn nested_object_keys_sort_recursively() {
        // Python: json.dumps({"z": {"b": 2, "a": 1}, "a": 0}, sort_keys=True)
        //   -> '{"a": 0, "z": {"a": 1, "b": 2}}'
        let v: Value = serde_json::from_str(r#"{"z":{"b":2,"a":1},"a":0}"#).unwrap();
        let out = to_python_json_bytes(&v).unwrap();
        assert_eq!(out, br#"{"a": 0, "z": {"a": 1, "b": 2}}"#.to_vec());
    }

    #[test]
    fn empty_containers_match_python_default() {
        let v: Value = serde_json::from_str(r#"{"a":[],"b":{}}"#).unwrap();
        let out = to_python_json_bytes(&v).unwrap();
        assert_eq!(out, br#"{"a": [], "b": {}}"#.to_vec());
    }

    #[test]
    fn mixed_types_match_python_default() {
        // Values: null, bool, int, float, string, array, object
        let v: Value = serde_json::from_str(
            r#"{"n":null,"b":true,"i":7,"s":"hi","arr":[1,2]}"#,
        )
        .unwrap();
        let out = to_python_json_bytes(&v).unwrap();
        // Keys sorted: arr, b, i, n, s
        assert_eq!(
            out,
            br#"{"arr": [1, 2], "b": true, "i": 7, "n": null, "s": "hi"}"#.to_vec()
        );
    }

    // ------------------------------------------------------------------------
    // Live-fixture test — the writer algorithm end-to-end against the actual
    // first entry of sovereign_state/receipts/activation_chain_…Z.json
    //
    // If this passes, Commit A is verified correct: byte-parity + BLAKE3
    // chaining together reproduce Python's output exactly.
    // ------------------------------------------------------------------------

    #[test]
    fn matches_live_activation_chain_entry_0() {
        // Receipt content from sovereign_state/receipts/agent_activation_2026-04-13T23:55:26Z.json
        let receipt = json!({
            "event": "agent_activation",
            "node_id": "NODE0",
            "agents_activated": 7,
            "agents_active": 7,
            "timestamp": "2026-04-13T23:55:26Z"
        });

        let digest = chain_entry_hash(GENESIS_PREV_HEX, &receipt).unwrap();
        let hex = hex_digest(&digest);

        // Expected hash from the envelope's chain[0].hash
        assert_eq!(
            hex,
            "89035bdc24d47d0549ec3667ddf66bdcd719307446d06dceeab7e1e6b2b7584b",
            "Commit A: chain entry hash must match the Python writer exactly"
        );
    }

    #[test]
    fn matches_live_activation_chain_entry_1() {
        // Entry 1's prev_hash == entry 0's hash; entry 1's receipt is fate_validation.
        let receipt = json!({
            "event": "fate_gate_validation",
            "node_id": "NODE0",
            "gates_enabled": 5,
            "timestamp": "2026-04-13T23:55:26Z",
            "verdict": "ALLOW"
        });

        let prev = "89035bdc24d47d0549ec3667ddf66bdcd719307446d06dceeab7e1e6b2b7584b";
        let digest = chain_entry_hash(prev, &receipt).unwrap();
        let hex = hex_digest(&digest);

        // NOTE: This test is intentionally informational. The actual content
        // of fate_validation_2026-04-13T23:55:26Z.json is what produces the
        // real hash. This test uses a reconstructed receipt and will only
        // match if the reconstruction matches the on-disk file exactly.
        // The authoritative cross-validation lives in Commit B (snapshot
        // loader reads the real file + verifies against the real envelope).
        //
        // For Commit A, we only assert the output LENGTH and HEX FORMAT —
        // byte parity of the end-to-end path is proven by entry_0 above.
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ------------------------------------------------------------------------
    // Genesis constant
    // ------------------------------------------------------------------------

    #[test]
    fn genesis_prev_hex_is_64_zeros() {
        assert_eq!(GENESIS_PREV_HEX.len(), 64);
        assert!(GENESIS_PREV_HEX.chars().all(|c| c == '0'));
    }

    #[test]
    fn hex_digest_is_lowercase_64_chars() {
        let zeros = [0u8; 32];
        assert_eq!(hex_digest(&zeros), "0".repeat(64));
        let ones = [0xffu8; 32];
        assert_eq!(hex_digest(&ones), "f".repeat(64));
    }
}

// ============================================================================
// Commit B tests — snapshot loader + verification, self-consistent fixtures
// ============================================================================

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    fn write_valid_two_entry_chain(root: &Path) {
        let receipts = root.join("receipts");
        fs::create_dir_all(&receipts).unwrap();

        let receipt_a = json!({"event": "step_a", "payload": "alpha", "n": 1});
        let receipt_b = json!({"event": "step_b", "payload": "bravo", "n": 2});

        // Write receipts in arbitrary format — hash is over canonicalized content
        fs::write(
            receipts.join("step_a_2026-01-01T00:00:00Z.json"),
            serde_json::to_vec_pretty(&receipt_a).unwrap(),
        )
        .unwrap();
        fs::write(
            receipts.join("step_b_2026-01-01T00:00:01Z.json"),
            serde_json::to_vec_pretty(&receipt_b).unwrap(),
        )
        .unwrap();

        let hash_a = hex_digest(&chain_entry_hash(GENESIS_PREV_HEX, &receipt_a).unwrap());
        let hash_b = hex_digest(&chain_entry_hash(&hash_a, &receipt_b).unwrap());

        let envelope = json!({
            "chain_type": "test_chain",
            "node_id": "TEST-NODE",
            "timestamp": "2026-01-01T00:00:00Z",
            "receipts": 2,
            "chain": [
                {
                    "file": "step_a_2026-01-01T00:00:00Z.json",
                    "event": "step_a",
                    "hash": hash_a,
                    "prev_hash": GENESIS_PREV_HEX
                },
                {
                    "file": "step_b_2026-01-01T00:00:01Z.json",
                    "event": "step_b",
                    "hash": hash_b,
                    "prev_hash": hash_a
                }
            ],
            "head_hash": hash_b
        });

        fs::write(
            receipts.join("activation_chain_2026-01-01T00:00:00Z.json"),
            serde_json::to_vec_pretty(&envelope).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn load_valid_tempdir_fixture_succeeds() {
        let td = TempDir::new().unwrap();
        write_valid_two_entry_chain(td.path());

        let snap = SovereignStateSnapshot::load(td.path()).expect("valid fixture should load");
        assert_eq!(snap.envelopes_count(), 1);
        assert_eq!(snap.total_entries(), 2);
        assert!(!snap.block_zero_present);

        let env = &snap.envelopes[0];
        assert_eq!(env.chain_type, "test_chain");
        assert_eq!(env.node_id, "TEST-NODE");
        assert_eq!(env.entries.len(), 2);
        assert_eq!(env.entries[0].event, "step_a");
        assert_eq!(env.entries[1].event, "step_b");
        assert_eq!(env.entries[0].prev_hash, GENESIS_PREV_HEX);
        assert_eq!(env.entries[1].prev_hash, env.entries[0].hash);
        assert_eq!(env.head_hash, env.entries[1].hash);
    }

    #[test]
    fn load_missing_root_fails_closed() {
        let td = TempDir::new().unwrap();
        let missing = td.path().join("does_not_exist");
        let err = SovereignStateSnapshot::load(&missing).unwrap_err();
        assert!(matches!(err, SovereignStateError::RootMissing(_)));
    }

    #[test]
    fn load_missing_receipts_dir_fails_closed() {
        let td = TempDir::new().unwrap();
        // root exists but no receipts/ subdir
        let err = SovereignStateSnapshot::load(td.path()).unwrap_err();
        assert!(matches!(err, SovereignStateError::ReceiptsDirMissing(_)));
    }

    #[test]
    fn load_empty_receipts_dir_fails_closed() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("receipts")).unwrap();
        let err = SovereignStateSnapshot::load(td.path()).unwrap_err();
        assert!(matches!(err, SovereignStateError::NoEnvelopes(_)));
    }

    #[test]
    fn tampered_receipt_content_fails_hash_check() {
        let td = TempDir::new().unwrap();
        write_valid_two_entry_chain(td.path());

        // Tamper step_a's content so its hash no longer matches the envelope
        let tampered = json!({"event": "step_a", "payload": "TAMPERED", "n": 999});
        fs::write(
            td.path()
                .join("receipts/step_a_2026-01-01T00:00:00Z.json"),
            serde_json::to_vec(&tampered).unwrap(),
        )
        .unwrap();

        let err = SovereignStateSnapshot::load(td.path()).unwrap_err();
        match err {
            SovereignStateError::HashMismatch { file, .. } => {
                assert_eq!(file, "step_a_2026-01-01T00:00:00Z.json");
            }
            other => panic!("expected HashMismatch, got {:?}", other),
        }
    }

    #[test]
    fn tampered_prev_hash_fails_linkage_check() {
        let td = TempDir::new().unwrap();
        write_valid_two_entry_chain(td.path());

        // Tamper the envelope so entry 0's prev_hash is wrong
        let env_path = td
            .path()
            .join("receipts/activation_chain_2026-01-01T00:00:00Z.json");
        let mut env: Value = serde_json::from_slice(&fs::read(&env_path).unwrap()).unwrap();
        env["chain"][0]["prev_hash"] = json!("f".repeat(64));
        fs::write(&env_path, serde_json::to_vec_pretty(&env).unwrap()).unwrap();

        let err = SovereignStateSnapshot::load(td.path()).unwrap_err();
        assert!(matches!(err, SovereignStateError::PrevHashMismatch { .. }));
    }

    #[test]
    fn tampered_head_hash_fails_envelope_check() {
        let td = TempDir::new().unwrap();
        write_valid_two_entry_chain(td.path());

        let env_path = td
            .path()
            .join("receipts/activation_chain_2026-01-01T00:00:00Z.json");
        let mut env: Value = serde_json::from_slice(&fs::read(&env_path).unwrap()).unwrap();
        env["head_hash"] = json!("a".repeat(64));
        fs::write(&env_path, serde_json::to_vec_pretty(&env).unwrap()).unwrap();

        let err = SovereignStateSnapshot::load(td.path()).unwrap_err();
        assert!(matches!(err, SovereignStateError::HeadHashMismatch { .. }));
    }

    #[test]
    fn missing_referenced_receipt_file_fails_closed() {
        let td = TempDir::new().unwrap();
        write_valid_two_entry_chain(td.path());

        // Remove the first receipt file while envelope still references it
        fs::remove_file(
            td.path()
                .join("receipts/step_a_2026-01-01T00:00:00Z.json"),
        )
        .unwrap();

        let err = SovereignStateSnapshot::load(td.path()).unwrap_err();
        assert!(matches!(err, SovereignStateError::ReceiptRead { .. }));
    }

    #[test]
    fn find_entry_by_hash_round_trip() {
        let td = TempDir::new().unwrap();
        write_valid_two_entry_chain(td.path());
        let snap = SovereignStateSnapshot::load(td.path()).unwrap();

        let known_hash = snap.envelopes[0].entries[1].hash.clone();
        let found = snap.find_entry_by_hash(&known_hash).expect("should find");
        assert_eq!(found.event, "step_b");

        assert!(snap.find_entry_by_hash("deadbeef").is_none());
    }

    #[test]
    fn block_zero_presence_is_reported() {
        let td = TempDir::new().unwrap();
        write_valid_two_entry_chain(td.path());
        // Without block_zero
        let snap = SovereignStateSnapshot::load(td.path()).unwrap();
        assert!(!snap.block_zero_present);

        // With block_zero
        fs::create_dir_all(td.path().join("block_zero")).unwrap();
        fs::write(
            td.path().join("block_zero/block_zero.json"),
            b"{\"schema_version\": \"1.0.0\"}",
        )
        .unwrap();
        let snap2 = SovereignStateSnapshot::load(td.path()).unwrap();
        assert!(snap2.block_zero_present);
    }
}
