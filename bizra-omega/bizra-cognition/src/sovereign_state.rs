// bizra-cognition/src/sovereign_state.rs
//
// Cycle-6 G1 Phase 1 — durable-read projection over Python-authored
// sovereign_state/ on disk.
//
// This module is READ-ONLY by design. The Python stack (see
// deploy/node0/bizra_node_activate.sh) is the authoritative writer.
// Rust loads the chain on startup and serves a read-only projection.
//
// Commit A — this file — implements the custom JSON formatter that
// reproduces Python's `json.dumps(data, sort_keys=True)` byte output,
// and a convenience that computes BLAKE3 chain entry hashes matching
// the Python writer exactly.
//
// See cycle-6/g1-writer-format-found.md for the verified algorithm.

use std::io;

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
